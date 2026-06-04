# Chapter 8 — Quantization for Apple Silicon

## What you'll do

Shrink your trained model so it loads fast and runs cheap locally — and pick the
**right** quantization method for Apple Silicon (Metal), because most of the
popular ones are CUDA-only.

## Step 1 — What quantization is

Weights are normally stored in 32-bit or 16-bit floats. **Quantization** stores
them in fewer bits (8-bit, 4-bit, even 2-bit integers), trading a little
accuracy for big wins in size and speed:

| Precision | 1B model size | Quality |
|---|---|---|
| fp32 | ~2.4 GB | reference |
| fp16 | ~1.2 GB | ~identical |
| Q8_0 (8-bit) | ~0.6 GB | near-identical |
| **Q4_K_M (4-bit)** | **~0.4 GB** | **slightly lower, great trade-off** |

For a model you serve locally and call constantly, 4-bit is usually the sweet
spot — it loads in well under a second and runs fast on the Metal GPU.

## Step 2 — Pick a method that supports Metal

This is the Apple-Silicon trap. Scanning the Transformers quantization table,
most methods (bitsandbytes, AWQ, GPTQ-on-CUDA, AQLM, HQQ…) target **CUDA**. The
ones that are 🟢 for **Metal (Apple Silicon)** are:

| Method | Apple Silicon | Best for |
|---|---|---|
| **GGUF / llama.cpp** | 🟢 | **our default** — runs in llama.cpp, Ollama, LM Studio |
| **MLX** | 🟢 | Apple's own framework, fastest on Mac for some workloads |
| Metal kernels / Quark / GPT-QModel | 🟢/🟡 | more specialized |

> **Decision for this workshop:** **GGUF, quantized to Q4_K_M.** It's the most
> portable (Ollama and LM Studio both consume it), Metal-accelerated, and
> exactly what Chapter 9 hosts and Chapter 10 pulls.

> **Note:** quantization here is for **inference/serving**, *after* training. We
> trained in fp32 (Chapter 4) precisely because low-precision *training* on MPS
> is unreliable — but low-precision *inference* is rock solid.

## Step 3 — Convert to GGUF and quantize

The script [`scripts/quantize_gguf.sh`](../scripts/quantize_gguf.sh) automates
the whole flow — clone llama.cpp, build it (Metal auto-enabled), convert your HF
model to GGUF, then quantize:

```bash
bash scripts/quantize_gguf.sh ./granite-ml-coder granite-ml-coder
```

It produces:

```
granite-ml-coder-f16.gguf      # full-precision GGUF (intermediate)
granite-ml-coder-Q4_K_M.gguf   # 4-bit — use this locally
```

Under the hood it runs:

```bash
python convert_hf_to_gguf.py ./granite-ml-coder --outfile granite-ml-coder-f16.gguf --outtype f16
llama-quantize granite-ml-coder-f16.gguf granite-ml-coder-Q4_K_M.gguf Q4_K_M
```

## Step 4 — Test the quantized model

```bash
.llama_cpp/build/bin/llama-cli -m granite-ml-coder-Q4_K_M.gguf \
  -p "Write a Keras autoencoder for network-traffic anomaly detection."
```

Sanity-check that quantization didn't degrade quality too much. Q4_K_M usually
keeps the model's behavior intact for a fine-tune like this.

## The MLX alternative (optional)

If you want Apple's native stack instead of GGUF:

```bash
uv pip install mlx-lm
uv run python -m mlx_lm.convert --hf-path ./granite-ml-coder -q   # 4-bit MLX
```

MLX can be faster on Apple Silicon for some models, but GGUF/Ollama is more
portable for sharing — which is why we lead with it.

## Why it works

Most of a model's quality lives in the *structure* of its weights, not their
last few bits of precision. Dropping to 4-bit (with smart per-block scaling like
Q4_K_M) preserves behavior while making the model small and fast enough to serve
on a laptop — the foundation for a *local* production app.

## Checkpoint

`granite-ml-coder-Q4_K_M.gguf` exists (~0.4 GB) and `llama-cli` generates a
sensible answer from it.

➡️ Next: [Chapter 9 — Hosting on Hugging Face & Ollama](09-hosting-hf-and-ollama.md)
