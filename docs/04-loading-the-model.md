# Chapter 4 — Loading the Model

## What you'll do

Load the pretrained IBM Granite 3.1 1B checkpoint correctly for **training on MPS**,
and understand the two settings that matter on a Mac: **dtype** and **the KV
cache**.

## Step 1 — Load the checkpoint

```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-3.1-1b-a400m-instruct", dtype=torch.float32)
```

`AutoModelForCausalLM` is the class for **causal language models** — models that
predict the next token given everything before it (GPT-style). That's exactly
the objective we fine-tune with.

## Step 2 — Choosing the dtype on Apple Silicon

The tutorial uses `dtype="auto"` to load weights in their saved precision (often
`bfloat16`), which halves memory. On **NVIDIA** that's great. On **MPS**, though,
half-precision *training* is still unreliable — you can get `NaN` losses. So we
make a deliberate choice:

> **On MPS, train in `float32`.** A 1B model in fp32 needs roughly:
> ~2.4 GB weights + ~2.4 GB gradients + ~4.8 GB optimizer state ≈ **10 GB**.
> That fits comfortably in 16 GB, easily in 64 GB.

```python
model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-3.1-1b-a400m-instruct", dtype=torch.float32)
```

(For *inference* you can later use lower precision or GGUF quantization — see
Chapters 8–9. Precision needs differ between training and serving.)

## Step 3 — Disable the KV cache during training

```python
model.config.use_cache = False
```

The **KV cache** speeds up *generation* by remembering past attention keys/values.
During training we process whole sequences at once and don't generate
token-by-token, so the cache wastes memory and conflicts with gradient
checkpointing. We turn it off for training and turn it back on before saving for
inference (`train.py` does both).

## Step 4 — Why this base model?

We use **IBM Granite 3.1 1B** (`ibm-granite/granite-3.1-1b-a400m-instruct`):

| Reason | Detail |
|---|---|
| Size | ~1.3B total / **400M active** (Mixture-of-Experts) — full fine-tune fits on a laptop |
| Modern | Recent Granite family, designed for code, RAG, and tool use |
| Chat-native | Ships a chat template (Chapter 3) so instruction tuning is clean |
| License | **Apache-2.0** — free to fine-tune, host, and distribute |

The pipeline is **model-agnostic** — swap `--model-name` for any chat model with
a chat template and everything else holds. A few easy drop-ins:

| Model | Notes |
|---|---|
| `ibm-granite/granite-3.3-2b-instruct` | bigger, standard transformer, stronger |
| `meta-llama/Llama-3.2-1B-Instruct` | gated — request access first |
| `microsoft/Phi-3.5-mini-instruct` | 3.8B, strong reasoning |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | small, fully open |

Everything downstream (data, Trainer, GGUF, Ollama) is identical regardless of
which base you pick — mind only the license and that it ships a chat template.

## Why it works

We're not changing the architecture — we're continuing to train existing weights.
Loading the pretrained checkpoint in a stable precision, with the cache disabled,
gives the Trainer a clean, memory-safe starting point on Metal.

## Checkpoint

```bash
uv run python -c "
import torch
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained('ibm-granite/granite-3.1-1b-a400m-instruct', dtype=torch.float32)
print('params:', sum(p.numel() for p in m.parameters())/1e6, 'M')"
```

Expect roughly **≈1.3B** parameters.

➡️ Next: [Chapter 5 — Training Configuration](05-training-configuration.md)
