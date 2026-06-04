# TEKIMAX ML Model Workshop

> **The goal:** stand up your *own* local model — fine-tuned from **IBM Granite
> 3.1 1B** (Apache-2.0), hosted on Hugging Face **or Ollama**, and pullable
> anywhere — then use it as a private coding copilot to build a **production
> neural network (DL / DNN)** in a Jupyter notebook with great accuracy.
> Explained as a step-by-step book.

This repository is two things at once:

1. **A working project** — runnable scripts in [`src/`](src/),
   [`scripts/`](scripts/), and a capstone notebook in [`notebooks/`](notebooks/)
   that take you from a public dataset to a published model to a production ML app.
2. **A book** — the [`docs/`](docs/) folder walks through *why* each step exists,
   not just *how*, so you can adapt it to your own data and goals.

---

## The workshop arc

```
  pick + fine-tune        host & pull            build the product
  ┌──────────────┐      ┌──────────────┐       ┌──────────────────────┐
  │ IBM Granite  │ push │ Hugging Face │ pull  │ DL anomaly detector   │
  │ 3.1 1B    ───┼─────▶│  + Ollama /  ├──────▶│ for network traffic — │
  │ → ML coder   │ GGUF │ HF Inference │       │ copilot on my machine │
  └──────────────┘      └──────────────┘       └──────────────────────┘
       ch 2–6                ch 8–9                    ch 10 (capstone)
```

You leave with: a private model you control, served locally (Ollama) or as a
cloud API (HF Inference), that you pull into a project to scaffold, explain, and
accelerate building real ML — ending in a deep-learning network-anomaly detector
that's accurate and production-ready.

## What you'll build

A fine-tuned **IBM Granite 3.1 1B** that writes Python ML code, explains pipeline steps,
reasons about overfitting and gradient descent, and suggests models for a task —
trained on a public Python instruction dataset filtered to ML/DS content,
quantized to **GGUF**, served via **Ollama or HF Inference**, and used in a
Jupyter notebook (Keras + scikit-learn) to build a **deep-learning anomaly
detector for network traffic** — fully local.

> **Expectation check.** A 1B model learns *style, format, and common
> patterns* extremely well, but it is not a frontier model. Treat it as a fast,
> private, ML-flavored autocomplete/explainer — not a replacement for Claude or
> GPT-class reasoning. The pipeline here scales unchanged to larger Granite
> sizes (+LoRA) when you want more capability.

## Limitations

- **It's a 1B model.** It learns style, format, and common patterns well, but is
  not a frontier model — it can produce incomplete or subtly wrong code, and it
  won't reliably "pick the best model" for hard problems. Treat its output as a
  fast first draft and verify before use.
- **English, Python-focused.** Strongest on classic ML (sklearn/pandas); weaker
  on large, novel, or multi-file tasks.
- **Inherits** any biases/limitations of the base model and the training dataset.

## Hardware this targets

| | |
|---|---|
| Machine | Apple M2 Ultra, 64 GB (any Apple Silicon works; less RAM → smaller batch) |
| Backend | PyTorch **MPS** (Metal) — *not* CUDA |
| Precision | fp32 during training (mixed precision on MPS is still unreliable) |
| Quantization | GGUF (llama.cpp) / MLX — the Apple-Silicon-friendly options |

## The book (read in order)

| Ch | File | Topic |
|---:|------|-------|
| 0 | [docs/00-introduction.md](docs/00-introduction.md) | What fine-tuning is, and the plan |
| 1 | [docs/01-environment-setup.md](docs/01-environment-setup.md) | `uv`, Python 3.12, and why not 3.14 |
| 2 | [docs/02-the-dataset.md](docs/02-the-dataset.md) | Choosing and filtering the dataset |
| 3 | [docs/03-tokenization-and-formatting.md](docs/03-tokenization-and-formatting.md) | Chat templates + prompt masking |
| 4 | [docs/04-loading-the-model.md](docs/04-loading-the-model.md) | Loading IBM Granite 3.1 1B on MPS |
| 5 | [docs/05-training-configuration.md](docs/05-training-configuration.md) | Every `TrainingArguments` choice |
| 6 | [docs/06-training.md](docs/06-training.md) | Running the Trainer + reading the loss |
| 7 | [docs/07-evaluation-and-testing.md](docs/07-evaluation-and-testing.md) | Overfitting, eval, and chatting |
| 8 | [docs/08-quantization.md](docs/08-quantization.md) | GGUF / MLX for Apple Silicon |
| 9 | [docs/09-hosting-hf-and-ollama.md](docs/09-hosting-hf-and-ollama.md) | Push to the Hub **and** serve via Ollama |
| 10 | [docs/10-capstone-network-anomaly-detection.md](docs/10-capstone-network-anomaly-detection.md) | Use your local model to build a DL network-anomaly detector in Jupyter |
| 11 | [docs/11-appendix-troubleshooting.md](docs/11-appendix-troubleshooting.md) | Errors, knobs, and going bigger |

## Quick start (TL;DR)

```bash
# 1. Install deps into an isolated env (Python 3.12)
uv sync

# 2. Authenticate with Hugging Face (needed to push later)
uv run huggingface-cli login

# 3. Train (CPU is the stable path on Apple Silicon). Tee to a log for live view:
PYTHONUNBUFFERED=1 uv run python src/train.py --cpu \
  --epochs 2 --batch-size 4 --grad-accum 4 --max-length 512 2>&1 | tee train.log

# 3b. In a SECOND terminal, watch progress in real time (bar, ETA, loss sparkline)
uv run python src/watch_training.py train.log

# 4. Talk to it
uv run python src/chat.py --model granite-ml-coder \
  --prompt "Write a scikit-learn pipeline to classify iris and explain how you avoid overfitting."

# 5. Full run + push to your Hub repo
uv run python src/train.py --push --hub-id Tekimax/granite-ml-coder

# 6. Quantize to GGUF for fast local use
bash scripts/quantize_gguf.sh ./granite-ml-coder granite-ml-coder

# 7. Serve locally with Ollama, then build the anomaly detector in the capstone notebook
ollama create granite-ml-coder -f Modelfile
uv run jupyter lab notebooks/capstone_network_anomaly_detection.ipynb
```

## Repository layout

```
tekimax-ML-Model-workshop/
├── README.md                 # you are here
├── pyproject.toml            # uv project + pinned deps
├── .python-version           # 3.12
├── src/
│   ├── prepare_data.py       # load + filter + tokenize + mask prompts
│   ├── train.py              # the Trainer loop (CPU/MPS, --model-name swaps US bases)
│   ├── chat.py               # inference / sanity check
│   └── watch_training.py     # real-time training monitor (bar, ETA, loss sparkline)
├── scripts/
│   └── quantize_gguf.sh      # HF model -> GGUF Q4_K_M
├── Modelfile                 # Ollama recipe for the GGUF
├── notebooks/
│   └── capstone_network_anomaly_detection.ipynb   # build the detector w/ copilot
└── docs/                     # the book (chapters 0–11)
```

## License & credits

**This workshop (cookbook, code, notebooks) is licensed under
[CC BY-NC 4.0](LICENSE)** — free to use, learn from, and adapt **with
attribution**, but **not for commercial use / resale** without permission.
© 2026 TEKIMAX / Christian Kaman.

The fine-tuned **model weights** on the Hub (`Tekimax/granite-ml-coder*`) are
**Apache-2.0** (inherited from IBM Granite), separate from the repo license.

Credits:
- Base model: [ibm-granite/granite-3.1-1b-a400m-instruct](https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-instruct) (Apache-2.0).
- Dataset: [iamtarun/python_code_instructions_18k_alpaca](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca).
- Built on Hugging Face `transformers` `Trainer`.
