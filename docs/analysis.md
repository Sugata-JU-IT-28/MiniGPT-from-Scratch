# What MiniGPT learned on TinyShakespeare

## Summary

Modern had the lowest mean held-out loss across three seeds, 4.3831 ± 0.0203 nats/token. RoPE alone accounted for most of its advantage over baseline. Model size behaved differently: the best baseline scaling result came from a middle-sized model. These are measurements of early learning. At vocabulary 1,024, the budget of 212,992 tokens divided by the 411,252-token training split is 0.5179 epoch equivalents. Random windows can overlap, so the fraction of unique text seen can be smaller. [Measured runs and split sizes](../results/experiment_aggregate.json)

All tables below report endpoints. Seed values are ordered from zero. Dispersion is the sample standard deviation with denominator n−1. I call a difference suggestive when the absolute difference of means exceeds the larger of the two seed SDs. Smaller differences are inconclusive. This descriptive rule supplies no significance level. [Tables, parameter counts and learning curves](results_tables.md)

## Learning-rate sweep

| Peak LR | Baseline, seed 0 | Modern, seed 0 |
|---:|---:|---:|
| 0.0003 | 5.1675 | 5.1055 |
| 0.001 | 4.7403 | 4.5721 |
| 0.003 | 4.6053 | 4.3598 |
| 0.006 | 4.7845 | 4.4950 |

The minima coincided at 0.003. Giving the presets equal weight yielded 4.4825 nats/token there, with runner-up 0.006 trailing by 0.1572. Both ends of the grid scored worse on this seed. Using one shared LR preserved each preset's measured optimum. With n=1 per cell, a seed SD is undefined and the ranking across seeds is inconclusive. [Selection and full-precision scores](../results/lr_selection.json)

## Architecture ablation

| Variant | Validation loss by seed | Mean ± SD, n=3 |
|---|---|---|
| baseline | 4.6053, 4.6234, 4.6402 | 4.6230 ± 0.0174 |
| rmsnorm | 4.6275, 4.6234, 4.6177 | 4.6229 ± 0.0049 |
| rope | 4.4108, 4.4173, 4.4155 | 4.4146 ± 0.0033 |
| swiglu | 4.7199, 4.7952, 4.7038 | 4.7396 ± 0.0487 |
| modern | 4.3598, 4.3976, 4.3921 | 4.3831 ± 0.0203 |

RoPE alone lowered the baseline mean by 0.2084 nats/token, well beyond the larger SD of 0.0174. Modern lowered it by 0.2398. Modern also beat RoPE alone on every seed, with a mean separation of about 0.0314 versus a larger SD of 0.0203. That is suggestive evidence for choosing modern for the final fit. RMSNorm alone was inconclusive. SwiGLU alone raised loss by 0.1166, a larger difference than its 0.0487 seed SD. The three single-toggle comparisons do not identify pairwise interactions within modern. [Architecture measurements](../results/experiment_aggregate.json)

## Shape ablation

| Variant | Validation loss by seed | Mean ± SD, n=3 |
|---|---|---|
| base | 4.6053, 4.6234, 4.6402 | 4.6230 ± 0.0174 |
| layers 2 | 4.7320, 4.5722, 4.7196 | 4.6746 ± 0.0888 |
| layers 6 | 4.7754, 4.6631, 4.6512 | 4.6965 ± 0.0685 |
| heads 2 | 4.5899, 4.7840, 4.6180 | 4.6640 ± 0.1049 |
| heads 8 | 4.5803, 4.6465, 4.6835 | 4.6368 ± 0.0523 |
| context 64 | 4.5594, 4.6059, 4.5532 | 4.5728 ± 0.0287 |
| context 256 | 4.7240, 4.6302, 4.9561 | 4.7701 ± 0.1677 |

The common control used four layers, four heads and a context of 128. At context 64 the mean was 0.0501 lower, exceeding the larger SD of 0.0287. The shorter context also puts more independent windows in each 512-token update. At context 256, the wide seed spread makes the difference inconclusive. Both head-count comparisons and the two-layer comparison were inconclusive. Six layers raised mean loss by 0.0735, only just beyond the larger SD of 0.0685. [Shape grid and contrasts](../results/experiment_aggregate.json)

## Scaling

| Layers and width | Validation loss by seed | Validation mean ± SD, n=2 | Validation minus train, mean ± SD |
|---|---|---|---|
| L1 D32 | 4.8353, 4.8115 | 4.8234 ± 0.0168 | -0.0897 ± 0.0029 |
| L2 D48 | 4.8356, 4.7007 | 4.7682 ± 0.0953 | -0.0835 ± 0.0053 |
| L3 D80 | 4.6249, 4.6180 | 4.6214 ± 0.0049 | -0.0616 ± 0.0110 |
| L4 D128 | 4.5933, 4.5940 | 4.5936 ± 0.0005 | -0.0561 ± 0.0119 |
| L5 D176 | 4.6971, 4.6709 | 4.6840 ± 0.0185 | -0.0749 ± 0.0153 |
| L6 D256 | 4.6977, 4.8288 | 4.7633 ± 0.0927 | -0.0705 ± 0.0098 |

The four-layer, width-128 baseline had the lowest mean at 940,800 total parameters. Loss fell again between the 325,840- and 940,800-parameter models, then rose by 0.0903 at 2,073,104 parameters, exceeding the larger SD of 0.0185. The last increase to 5,033,984 parameters was inconclusive. Both total and non-embedding parameter axes show the reversal. A single monotone power law cannot explain that shape across the grid, so I did not fit an exponent. [Scaling figure](img/scaling.png) · [Parameter counts and training losses](results_tables.md)

Every scaling seed reached its lowest recorded validation loss at the final update. Even the largest model had a negative mean validation-minus-train gap. Its training loss also exceeded that of the middle-sized model. The larger models performed poorly at this budget without a measured late validation upturn. Different split difficulty and optimization speed are plausible explanations. [Recorded evaluation curves](../results/experiment_aggregate.json)

## Tokenizer

| Vocabulary | Validation bits/byte by seed | Mean ± SD, n=2 |
|---|---|---|
| vocab 512 | 3.1194, 3.1004 | 3.1099 ± 0.0134 |
| vocab 1024 | 2.9432, 2.9548 | 2.9490 ± 0.0081 |
| vocab 2048 | 2.8222, 2.8407 | 2.8314 ± 0.0130 |

The held-out cost fell by 0.1608 bits/byte from 512 to 1,024 entries, then by 0.1175 bits/byte from 1,024 to 2,048. Both differences exceed the respective larger seed SD. All variants have 447,552 non-embedding parameters. The larger vocabulary also buys more source text per training token: the same budget amounts to 0.4121, 0.5179 and 0.6140 split equivalents, respectively. The end-to-end comparison therefore includes changes in vocabulary, text exposure and embedding size. [Tokenizer figure](img/tokenizer_bpb.png) · [Split-equivalent budgets](../results/experiment_aggregate.json)

## What did not behave as expected

The LR ranking changed with scale. On the 5,024,576-parameter modern model, matched seed-zero probes of 314,368 tokens reached 4.3784 nats/token at 0.003 and 4.2035 at 0.0015. The shared LR trained stably but scored worse on this seed. The final run used 0.0015. Replication across seeds would be needed to establish a general transfer advantage. [Final-shape LR check](../results/final_lr_check.json)

SwiGLU alone scored worse on every seed, yet the model combining it with RMSNorm and RoPE led the architecture suite. A shorter context also scored lower than the base context despite discarding potential history. Finally, the largest baseline showed worse training and validation loss than the middle-sized model while its validation curve continued downward. Treating that as overfitting would point to a diagnosis the curves do not support. [Training curves and endpoint gaps](../results/experiment_aggregate.json)

## Threats to validity

TinyShakespeare is a single small dataset. The contiguous validation split comes from the same collection of plays and has a different mix of speeches and characters. The same split informed LR selection, architecture ranking and checkpoint selection. LR tuning used one seed at the same short budget. The scaling suite kept that LR fixed at every size. The 0.5179-epoch base exposure favors variants that learn quickly early, and the tiny parameter scale limits extrapolation. Architecture toggles change parameter counts, especially when replacing learned positions. Equal token budgets also allow different amounts of computation. With two or three seeds the estimated SD can itself vary substantially. [Design and measurements](../results/experiment_aggregate.json)

## Final checkpoint and samples

At the same 3,145,728-token budget, the final modern checkpoint scored lower on the held-out split than the Phase 2 best. Both comparisons below use the 1,024-entry tokenizer. Each checkpoint represents one training seed, so the cross-seed reliability of the difference is inconclusive. The final fit changed width, depth, optimizer schedule and seed together. [Final checkpoint summary](../results/final_summary.json) · [Phase 2 selection](../results/phase2_summary.json)

| Checkpoint | Seed | Train loss | Validation loss | Validation bits/byte |
|---|---:|---:|---:|---:|
| Phase 2 best | 42 | 2.8019 | 3.3270 | 2.1262 |
| Phase 3 final | 0 | 2.5339 | 3.3017 | 2.1101 |

The six-layer, width-256 model reached its lowest measured validation loss at update 6,144, the last update. Its training loss fell much further below Phase 2 than its validation loss did. That wider gap is consistent with a closer fit to the training text. Reloading the saved weights reproduced validation perplexity 27.1611 and yielded identical cached and uncached greedy IDs across a 256-token continuation. [Training history and generation checks](../results/final_model.json)

At temperature 0.5, the Richard prompt opens with “Say, my lord, and I'll go with thee.” It produces recognizable speaker cues, then gives Queen Margaret two consecutive turns. At 1.0 the same prompt produces “silentouse green.” All three temperatures complete “What say you,” with “what?” before the responses diverge. The samples show learned dialogue formatting and local turns of phrase. They give little evidence of stable character identity or sustained meaning. [Nine unedited continuations](samples_final.md)

## What the results do NOT show

The experiments cannot establish a universal architecture ranking, an asymptotic scaling law, or the best vocabulary at equal raw-byte exposure. The trajectories leave open which variants would converge to the lowest loss after longer training and independent LR tuning. Held-out token prediction and sampled dialogue also leave factual accuracy, sustained narrative coherence and transfer to other authors unmeasured.
