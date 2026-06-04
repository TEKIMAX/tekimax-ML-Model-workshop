# Chapter 0 — Introduction: Why a Local Model?

> *"The best AI for production isn't always the biggest one in the cloud —
> sometimes it's the one you own, running on your own hardware."*

## The purpose of this workshop

By the end of this book you will have done something many teams only talk about:
**built, owned, and deployed your own language model**, then used it to ship a
real machine-learning product. Concretely, you will:

1. **Fine-tune** a small open model (IBM Granite 3.1 1B) into a Python/ML coding
   assistant on your own Mac.
2. **Host** it two ways — on the **Hugging Face Hub** (shareable) and via
   **Ollama** (a local API you can `pull` and call from any app).
3. **Use it** as a private copilot inside a **Jupyter notebook** to build a
   **production neural network (DL / DNN)** — trained, evaluated, and accurate
   enough to ship.

The throughline: *local models can drive production applications.* You are not
renting intelligence by the token; you are deploying an asset you control.

## What is fine-tuning?

Fine-tuning continues training a large pretrained model on a smaller,
task-specific dataset. The model already learned language, code, and reasoning
during **pretraining** (on trillions of tokens). Fine-tuning nudges those
existing weights toward *your* domain — here, writing and explaining ML code.

It is identical to pretraining except:

- You **start from trained weights**, not random ones.
- It needs **far less data, compute, and time** — minutes-to-hours on a laptop,
  not months on a GPU cluster.

```
 Pretraining                         Fine-tuning
 ───────────                         ───────────
 random weights                      pretrained weights  ← we start here
   + trillions of tokens               + thousands of examples
   + thousands of GPUs                 + one Apple Silicon Mac
   = a general model                   = a specialized model
```

## Why local, and why small?

| Concern | Local small model | Cloud frontier API |
|---|---|---|
| Cost at scale | Fixed (your hardware) | Per-token, grows forever |
| Privacy | Data never leaves your machine | Sent to a third party |
| Latency | No network round-trip | Network-bound |
| Control | You own weights + versioning | Vendor can change/retire it |
| Raw capability | Lower | Higher |

A 1B model won't out-reason a frontier model. But for **structured,
repetitive, domain-shaped tasks** — scaffolding sklearn pipelines, explaining
gradient descent, drafting a Keras model — a fine-tuned small model is fast,
free to run, private, and *good enough to be useful in production loops*.

> **Honest expectations.** Throughout this book we treat the 1B model as a
> capable assistant, not an oracle. The exact same pipeline scales to 1.7B and
> 4B (with LoRA) when you need more — see Chapter 11.

## The tools

- **`uv`** — fast Python package/dep manager (Chapter 1).
- **`transformers` `Trainer`** — the training loop (Chapters 5–6).
- **PyTorch MPS** — runs training on the Mac's Metal GPU.
- **llama.cpp / Ollama** — serve the quantized model locally (Chapters 8–9).
- **Jupyter + Keras/PyTorch** — build the production anomaly detector (Chapter 10).

## How to read this book

Each chapter has the same shape:

- **What you'll do** — the concrete goal.
- **The steps** — commands and code, matching the scripts in `src/`.
- **Why it works** — the concept behind the step.
- **Checkpoint** — how to know it worked before moving on.

Read in order the first time. After that, each chapter stands alone as a reference.

➡️ Next: [Chapter 1 — Environment Setup](01-environment-setup.md)
