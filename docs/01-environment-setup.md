# Chapter 1 — Environment Setup

## What you'll do

Create an isolated, reproducible Python environment for training on Apple
Silicon — and understand the two gotchas that trip people up on a Mac:
**the Python version** and **the absence of CUDA**.

## Step 1 — Confirm your hardware

```bash
sysctl -n machdep.cpu.brand_string   # e.g. "Apple M2 Ultra"
sysctl -n hw.memsize | awk '{print $1/1073741824" GB"}'
```

Any Apple Silicon (M1/M2/M3/M4) works. RAM determines batch size: 16 GB is fine
for IBM Granite 3.1 1B with a small batch; 64 GB lets you be generous.

## Step 2 — Use Python 3.12 (not 3.14)

> **Gotcha #1.** PyTorch publishes wheels for stable Python versions. Brand-new
> releases like **3.14** often have *no* PyTorch wheel yet, so `import torch`
> fails. We pin **3.12**, which is fully supported.

`uv` reads `.python-version` (already set to `3.12` in this repo) and will fetch
that interpreter automatically — you don't need to install it system-wide.

```bash
uv python install 3.12   # only if uv doesn't already have it
```

## Step 3 — Install dependencies

From the project root:

```bash
uv sync
```

This reads `pyproject.toml`, creates `.venv/` with Python 3.12, and installs
`torch`, `transformers`, `datasets`, `accelerate`, and `huggingface_hub`. Run
everything afterward with `uv run …` so it uses this environment.

## Step 4 — Verify the MPS (Metal) backend

> **Gotcha #2.** Your Mac has **no NVIDIA GPU**, so there is **no CUDA**.
> PyTorch instead uses **MPS** (Metal Performance Shaders) to run on the Apple
> GPU. Any tutorial that says `bf16=True`, `device="cuda"`, or `bitsandbytes`
> assumes NVIDIA — we adapt those for MPS throughout.

```bash
uv run python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

Expect `MPS available: True`. If it prints `False`, training falls back to CPU
(slow but still works).

## Step 5 — Log in to Hugging Face

You'll push your model later, so authenticate now:

```bash
uv run huggingface-cli login
```

Paste a token from <https://huggingface.co/settings/tokens> (needs **write**
scope). Alternatively, export it for non-interactive use:

```bash
export HF_TOKEN=hf_xxx   # add to ~/.zshrc to persist
```

## Why it works

`uv` gives every project its own pinned interpreter and dependency set, so
"works on my machine" becomes "works everywhere." Pinning Python 3.12 sidesteps
the missing-wheel problem, and confirming MPS up front means no surprises when
training starts.

## Checkpoint

```bash
uv run python -c "import torch, transformers, datasets; \
print('torch', torch.__version__, '| transformers', transformers.__version__, \
'| mps', torch.backends.mps.is_available())"
```

If that prints versions and `mps True`, you're ready.

➡️ Next: [Chapter 2 — The Dataset](02-the-dataset.md)
