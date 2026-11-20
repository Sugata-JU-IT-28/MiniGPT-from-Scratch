# MiniGPT design notes

## Byte-level BPE

The tokenizer splits text into pieces such as a word with its leading space, then encodes each piece as UTF-8 bytes. Training counts adjacent pairs on the training split, merges the most frequent pair within each piece, and updates counts only for affected pieces. During encoding, the earliest learned merge takes priority. Concatenating token bytes before decoding recovers the original text even when a character spans several tokens. [`pre_tokens`, `_learn`, `BPETokenizer._encode_piece` and `decode_bytes`](../minigpt/tokenizer/bpe.py)

- **How do you reproduce tied merge decisions?** Choose the smallest pair of token IDs among pairs with equal frequency. The ordered merge list determines all later IDs.
- **What happens with a word the tokenizer has never seen?** Its UTF-8 bytes already have IDs. Applicable learned merges shorten the sequence.

## Causal attention

A projection produces queries, keys and values for each head. Each query scores the keys using a dot product scaled by the inverse square root of the head dimension. A triangular mask gives future positions negative infinity before softmax, so their probability is zero. The head mixes values using the resulting weights, and the output projection mixes the heads. [`CausalSelfAttention.forward`](../minigpt/model.py)

- **Why scale the dot product?** With roughly independent unit-variance components, the variance of the sum grows with head width. Scaling keeps the logits in a range where softmax can distinguish alternatives without routinely saturating.
- **Which row explains a generated token?** The row for the last input position of the forward pass that predicted it. The sampled token joins the context on the next pass.
- **What does the mean-head view show?** An arithmetic average of the pre-dropout attention weights. A large weight describes routing within that layer. The values, output projection, other layers and residual stream also affect the prediction.

## Initialization

Linear layers and embeddings start with small zero-mean Gaussian weights, and linear biases start at zero. The attention output and MLP down projections feed additive residual branches. Their initial standard deviation decreases with the inverse square root of the number of those branches across the decoder, which limits the accumulated size of early residual updates. [`GPT._init_weights` and `GPT.__init__`](../minigpt/model.py)

- **Which matrices receive the depth-dependent scaling?** Each block's attention output projection and MLP down projection. Both write directly to the residual stream.
- **Does initialization constrain weights during training?** It determines their starting values. AdamW updates them from the first optimizer steps onward.

## Weight tying

When tying is enabled, the language-model head refers to the token embedding parameter itself. A row helps represent a token on input and scores that token against the final hidden state on output. Gradients from both uses accumulate into the shared parameter, saving the storage and optimizer state of a separate output matrix. [`GPT.__init__` and `GPT.count_params`](../minigpt/model.py)

- **Would copying the embedding values once achieve the same thing?** Only at initialization. Separate parameters can diverge on the next update.
- **How does the parameter count handle sharing?** The parameter iterator visits the shared object once. The non-embedding count subtracts the token table and any learned position table.

## AdamW parameter groups

The optimizer groups trainable parameters by their number of dimensions. Matrices, including embeddings, receive weight decay, while bias vectors and normalization scales have zero decay. AdamW applies shrinkage separately from its adaptive gradient update, so the decay strength avoids rescaling by a parameter's gradient-history denominator. [`build_optimizer`](../minigpt/train.py)

- **Does the tied head get decayed twice?** No. The optimizer sees a single parameter object.
- **Why exempt a norm scale?** Decaying it would continually shrink a learned feature gain. The dimension-based rule keeps gains and offsets free of that additional pressure.

## Learning-rate schedule

The trainer indexes the schedule by optimizer update, including when several microbatches form one update. With warmup enabled, the rate starts at zero and rises linearly toward the configured peak. The subsequent cosine segment decreases to the configured floor on the last planned update. Resuming restores the step counter and preserves the original token budget. [`learning_rate`, `Trainer.resume` and `Trainer.run`](../minigpt/train.py)

- **Why warm up?** Small early updates limit abrupt changes while the model and optimizer moments are still adapting.
- **What would happen if a resumed run reset the schedule?** The same training examples could see different update sizes, breaking replay even with identical weights and random states.

## Rotary position embeddings

RoPE rotates adjacent feature pairs in each head's queries and keys using position-dependent angles. Different pairs have different frequencies, so a query-key dot product can depend on their relative position. The implementation precomputes sine and cosine tables and uses the absolute position of each new token to index them. [`rotary_tables`, `apply_rotary_emb` and `CausalSelfAttention.forward`](../minigpt/model.py)

- **Why must the head dimension be even?** Each planar rotation consumes a pair of coordinates.
- **What gets cached after rotation?** The already-rotated keys and the unrotated values. Each new query uses the position after the cached prefix.

## RMSNorm

RMSNorm computes the mean square across the features of each token and scales by its reciprocal square root with an epsilon for stability. A learned per-feature gain restores flexibility after normalization. The calculation uses float32 internally and casts the result back to the activation dtype. [`RMSNorm.forward` and `_norm`](../minigpt/model.py)

- **How does this differ from LayerNorm?** LayerNorm subtracts the feature mean and scales by the standard deviation. RMSNorm retains the mean component and scales by root mean square.
- **Can one token change another token's normalization statistics?** No. Each token supplies its own feature vector, keeping this operation independent across time and batch.

## SwiGLU

The gated MLP projects the same token features through two learned matrices. It applies SiLU to the gate projection, multiplies it elementwise with the up projection, then projects the product back to the residual width. The gated hidden layer is narrower than the GELU hidden layer so the extra matrix has a comparable parameter budget after alignment. [`MLP.__init__` and `MLP.forward`](../minigpt/model.py)

- **What is the gate learning?** An input-dependent multiplier for each intermediate feature. SiLU allows negative values and is unbounded above.
- **Does the MLP mix information between positions?** No. It applies the same feature transformation separately at every position. Attention supplies the cross-position information.

## KV-cache

Prefill runs the prompt once and saves projected keys and values for each layer. On a subsequent draw the model processes the new suffix and attends to both cached and newly computed keys. The cache grows linearly with batch size, layer count, context length and embedding width. With exact sliding windows, the sampler discards a full cache and pre-fills the cropped context again. [`CausalSelfAttention.forward` and `GPT.forward`](../minigpt/model.py) · [`generate_stream`](../minigpt/generate.py)

- **Why omit old queries?** Future queries need the old keys for scores and the old values for the weighted sum. The old queries' outputs have already been used.
- **Why can dropping the oldest key and value change the answer?** Learned positions would be misaligned after a crop, and deeper-layer states can still encode the evicted prefix under either position scheme.
- **Does each cached draw have constant cost?** The new query still scores every cached position. This implementation also concatenates K and V tensors as the cache grows.

## Sampling

Raw probabilities come from softmax of the model's logits. The sampler applies any repetition penalty, scales for temperature, keeps the highest-scoring top-k entries, then chooses the smallest descending-probability prefix reaching the top-p threshold within the survivors. It renormalizes and draws from that distribution with a request-local generator. The inspector reports entropy in nats over each entire distribution and displays the remaining probability mass outside its candidate list. [`distributions`, `penalize_repetitions` and `next_token_distribution`](../minigpt/generate.py) · [`create_app`](../minigpt/api.py)

- **What happens at zero temperature?** The highest adjusted logit is selected deterministically. The final distribution places all probability on that token, so its entropy is zero.
- **Can the chosen token fall outside the displayed candidates?** Yes. The panel ranks by raw probability and shows the shortlist plus its omitted mass. The trace always records the chosen token separately.
- **Why keep raw and final probabilities?** Raw probabilities show what the model predicted. Final probabilities show how likely the sampler made each outcome, including zero for removed tokens.

## Bits per byte

Evaluation walks contiguous windows and scores each predictable target in the split exactly once, including the short tail. It weights each window's mean cross-entropy by its target count to obtain mean nats per predicted token. Dividing by the natural logarithm of two converts to bits per token, then dividing by the split's measured UTF-8 bytes per token gives the reported bits per byte. The first token has no predecessor, so the loss excludes that target while the byte-per-token denominator describes the whole split. [`evaluate`](../minigpt/metrics.py) · [`prepare_dataset`](../minigpt/data.py)

- **Why use this metric for different vocabularies?** A larger unit of text can naturally have higher loss per token. Bits per byte puts all models on the same source-text unit.
- **Why avoid averaging the batch losses equally?** The tail may contain fewer targets. An unweighted average would give each tail target more influence.
