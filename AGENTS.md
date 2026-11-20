# Project rules

1. Goal: a GPT-style LM built entirely by hand: byte-level BPE, manual causal attention, modern-architecture toggles (RMSNorm, RoPE, SwiGLU), KV-cache, sampling, rigorous experiments, and a "glass-box" web UI that shows the model's decisions.
2. Allowed runtime libraries: torch, numpy, matplotlib, fastapi, uvicorn, pydantic, pyyaml, tqdm, and the Python standard library. Dev tools: pytest, httpx, ruff.
3. FORBIDDEN: transformers, tiktoken, sentencepiece, tokenizers, nanoGPT or any copied model code, LangChain, or any LLM framework. Everything is implemented by hand.
4. Code standards: Python 3.10+, full type hints, short docstrings, ruff-clean (line length 100), every module covered by pytest. Mark slow tests with `@pytest.mark.slow`.
5. Reproducibility: every script takes `--seed` and seeds python/numpy/torch. Results are written to `results/` as JSON or CSV by scripts, never typed by hand.
6. Honesty: every number in docs or README must come from a file in `results/` produced by a script in this repo. Never invent, estimate, or round up results. Never claim a test or command passed unless you ran it. Never weaken or delete a test to make it pass: fix the code, or explain why the test itself was wrong. List anything unverified.
7. Workflow: at the start of each task read AGENTS.md and docs/PROGRESS.md. At the end, append a short dated summary (what was done, key numbers, open issues) to docs/PROGRESS.md, give me the same summary, then STOP. Do not start the next phase. Do not run git commit or git push; I commit between phases.
8. Long-running jobs: run them in the background with output to a log file and poll it. Experiment runners must be resumable. Auto-select the device (CUDA if available, else CPU). Never assume network access except the one-time dataset download.
