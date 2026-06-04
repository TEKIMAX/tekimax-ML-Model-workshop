# Chapter 11 — Appendix & Troubleshooting

## The big one: MPS is unreliable for fine-tuning → use `--cpu`

On this stack (M2 Ultra · torch 2.12 · transformers 5.9 · Granite) the Metal/MPS
backend **could not train this model reliably.** We hit, in order:

1. **Memory runaway** — fp32 × batch-4 × 1024-token seqs → step time crept from
   5 s to 40 s as MPS memory pressure built.
2. **Masked-rows `nan`** — 3 rows whose prompt ≥ `max_length` left zero answer
   tokens → `nan` loss. (Filtered in `prepare_data.py`.)
3. **False out-of-memory** — the default watermark refused valid allocations at
   the cross-entropy step (`other allocations: 76 GiB` while 90 % RAM was free).
4. **Swap-stall** — `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` fixed #3 but let MPS
   over-allocate into swap; the process hung in uninterruptible state.
5. **Gradient-checkpoint leak** — ~340 MB/step growth → swap death.
6. **`mps.empty_cache()` deadlock** — clearing the cache mid-training froze the
   GPU queue.
7. **SDPA training `nan`** — even healthy runs exploded (`loss → 4460 → nan`)
   mid-epoch; eager attention is NaN-safe but **hangs** on MPS.

**Resolution:** train on **CPU**. Slower (~15 s/step) but it finishes with a
clean loss curve (`1.06 → 0.34`, zero `nan`):

```bash
uv run python src/train.py --cpu --epochs 3 \
  --batch-size 4 --grad-accum 4 --max-length 512
```

On an **NVIDIA GPU** none of this applies — drop `--cpu`, set `bf16=True`, done.
The MPS flags below (`--grad-checkpoint`, `--attn-eager`,
`PYTORCH_MPS_HIGH_WATERMARK_RATIO`) are kept for the record and for anyone who
wants to retry MPS on a newer torch, but CPU is the recommended path here.

## Common errors

### `import torch` fails / no wheel
You're probably on Python 3.13/3.14. Use 3.12 (Chapter 1). `uv` honors
`.python-version`; confirm with `uv run python --version`.

### `RuntimeError: MPS backend out of memory`
Lower `--batch-size` (try 2 or 1) and raise `--grad-accum` to keep the effective
batch size. Also lower `--max-length` (e.g. 512).

### Training loss is `nan`
Two causes we hit:
1. **Mixed precision on MPS** — keep `bf16=False` and `fp16=False` (defaults). fp32.
2. **A training example with no answer tokens** — if a prompt is ≥ `max_length`,
   truncation masks every label (`-100`) and the loss is a mean over zero valid
   tokens → `nan`. `prepare_data.py` filters these out (it dropped 3 of 2,341).

### `eval_loss` is `nan` but training loss is healthy
This is the **SDPA attention kernel NaN-ing on MPS** for some *padded eval
batches* — the weights are fine (training loss keeps falling), only the eval
metric is `nan`. Two options:
- **Recommended:** leave in-loop eval **off** (the default on MPS) and judge the
  model by generating samples (`src/chat.py`). Fast SDPA training is unaffected.
- Pass `--eval --attn-eager` to evaluate with the NaN-safe eager attention
  kernel — correct, but ~6× slower on MPS.

### Step time keeps growing / "will take hours"
MPS memory pressure from long sequences. Lower `--max-length`, lower
`--batch-size` (raise `--grad-accum` to keep the effective batch), add
`--grad-checkpoint`, and set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`.

### `PYTORCH_ENABLE_MPS_FALLBACK`
If a specific op isn't implemented on MPS yet:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```
This silently runs unsupported ops on CPU so training continues.

### Training uses CPU, not the GPU
Check `torch.backends.mps.is_available()` is `True`. If `False`, your PyTorch
build lacks MPS — reinstall with `uv sync`.

### `huggingface-cli: command not found`
Run it through the env: `uv run huggingface-cli login`.

### Push fails with 401 / permission
Your token needs **write** scope, and the repo namespace must be yours
(`Tekimax/...`). Re-login: `uv run huggingface-cli login`.

### Ollama: `connection refused` on :11434
Start the daemon: `ollama serve` (or open the Ollama app).

## Tuning cheat sheet

| Want | Change |
|---|---|
| Faster smoke test | `--max-samples 200 --epochs 1` |
| Less overfitting | fewer epochs, lower `--lr`, more data |
| Less underfitting | more epochs, higher `--lr`, bigger model |
| Fit a smaller Mac | `--batch-size 1 --grad-accum 16 --max-length 512` |
| Train on everything | `--all-rows` |

## Going bigger (when 1B isn't enough)

The whole pipeline is model-agnostic. To level up:

1. **Bigger base model.** Set `MODEL_NAME = "ibm-granite/granite-3.3-2b-instruct"` in
   `src/train.py`. Trains full on an M2 Ultra, just slower.

2. **4B with LoRA.** For `IBM Granite/granite-3.3-8b-instruct`, full fine-tuning gets tight. Use
   **LoRA** (PEFT) to train small adapter matrices instead of all weights:
   ```python
   from peft import LoraConfig, get_peft_model
   model = get_peft_model(model, LoraConfig(
       r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
       lora_dropout=0.05, task_type="CAUSAL_LM"))
   ```
   This cuts trainable params by >99% and fits comfortably. Everything else
   (data, Trainer, eval, GGUF export) stays the same.

3. **Better data.** The biggest quality lever isn't model size — it's the
   dataset. Curate or write higher-quality, more on-topic examples
   (Chapter 2's "bring your own data").

## Quantization reference (Apple Silicon)

| Method | Apple Silicon | Notes |
|---|---|---|
| **GGUF / llama.cpp** | 🟢 | our default; powers Ollama / LM Studio |
| **MLX** | 🟢 | Apple-native, often fastest on Mac |
| bitsandbytes / AWQ / GPTQ(CUDA) / AQLM / HQQ | 🔴/🟡 | CUDA-oriented — avoid on Mac |

GGUF quant levels, fast → smaller: `Q8_0` (best quality) → `Q5_K_M` →
**`Q4_K_M`** (recommended) → `Q3_K_M` → `Q2_K` (smallest, lowest quality).

## File map recap

| File | Role |
|---|---|
| `src/prepare_data.py` | load + filter + tokenize + mask |
| `src/train.py` | the Trainer loop (MPS-aware) |
| `src/chat.py` | inference / sanity check |
| `scripts/quantize_gguf.sh` | HF model → GGUF Q4_K_M |
| `Modelfile` | Ollama recipe |
| `notebooks/capstone_network_anomaly_detection.ipynb` | build the anomaly detector |

## Where to go next

- Hugging Face `Trainer` recipes — custom losses, callbacks, memory-efficient eval.
- PEFT/LoRA docs for adapter-based fine-tuning at larger scales.
- llama.cpp & Ollama docs for serving, GPU offload, and the OpenAI-compatible API.

⬅️ Back to [README](../README.md)
