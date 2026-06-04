# Chapter 9 — Hosting on Hugging Face & Ollama

## What you'll do

Publish your model two ways: to the **Hugging Face Hub** (so anyone — including
future-you — can download it) and to **Ollama** (a local server you `pull` and
call from any app). This is the "host it so we can pull it" step at the heart of
the workshop.

## Path A — Hugging Face Hub (share & version)

### A1. The full model

Push during training:

```bash
uv run python src/train.py --push --hub-id Tekimax/granite-ml-coder
```

Or push an already-trained directory:

```bash
uv run huggingface-cli upload Tekimax/granite-ml-coder ./granite-ml-coder .
```

`push_to_hub()` uploads the weights, generation config, tokenizer, and model
config. Your model now lives at `https://huggingface.co/Tekimax/granite-ml-coder`
and anyone can load it:

```python
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("Tekimax/granite-ml-coder")
```

### A2. The GGUF (for llama.cpp / Ollama users)

Upload the quantized file to a companion repo:

```bash
uv run huggingface-cli upload Tekimax/granite-ml-coder-GGUF granite-ml-coder-Q4_K_M.gguf
```

### A3. Write a model card

Add a `README.md` to the repo describing: base model, dataset, intended use,
limitations (it's 1B!), and an example prompt. A good card is what makes a
model *usable* by others — state plainly what it's good and bad at.

## Path B — Ollama (local API you can pull)

Ollama turns your GGUF into a local server with an OpenAI-compatible API —
perfect for building apps that call the model without any cloud.

### B1. Install Ollama

```bash
brew install ollama
ollama serve   # starts the local daemon (or run the menu-bar app)
```

### B2. Register your model from the GGUF

The repo ships a [`Modelfile`](../Modelfile) that points at your quantized model
and bakes in the system prompt:

```bash
ollama create granite-ml-coder -f Modelfile
```

### B3. Use it

From the CLI:

```bash
ollama run granite-ml-coder "Write a Keras autoencoder for network-traffic anomaly detection."
```

From Python (this is how the capstone notebook calls it):

```python
import requests
r = requests.post("http://localhost:11434/api/chat", json={
    "model": "granite-ml-coder",
    "messages": [{"role": "user", "content": "Write a sklearn preprocessing pipeline."}],
    "stream": False,
})
print(r.json()["message"]["content"])
```

Or via the OpenAI-compatible endpoint at `http://localhost:11434/v1`, so any
tool that speaks the OpenAI API works unchanged — just point it at localhost.

### B4. (Optional) Pull straight from the Hub

Ollama can pull GGUF models directly from a Hugging Face repo:

```bash
ollama run hf.co/Tekimax/granite-ml-coder-GGUF
```

That closes the loop: **fine-tune → push GGUF to HF → `ollama run` it anywhere.**

## Path C — Hugging Face Inference API (cloud endpoint)

Sometimes you want the model as a **remote API** (a teammate's laptop, a CI job,
a server with no GPU). Deploy your pushed model as an **Inference Endpoint**:

1. On your model page → **Deploy → Inference Endpoints** (uses your
   https://huggingface.co/storage + compute), or visit
   <https://endpoints.huggingface.co>.
2. You get an **OpenAI-compatible** URL:
   `https://<id>.endpoints.huggingface.cloud/v1/chat/completions`.
3. Call it with your `HF_TOKEN`:

```python
import os, requests
r = requests.post(os.environ["HF_ENDPOINT_URL"],
    headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"},
    json={"model": "Tekimax/granite-ml-coder",
          "messages": [{"role": "user", "content": "Write a sklearn pipeline."}],
          "stream": False})
print(r.json()["choices"][0]["message"]["content"])
```

The capstone notebook (Chapter 10) speaks **both** Ollama and this HF endpoint —
flip `COPILOT_BACKEND=hf` and the same model answers from the cloud.

## Which should I use?

| Use case | Pick |
|---|---|
| Share the model / load with `transformers` | Hugging Face Hub |
| Local app, OpenAI-compatible API, zero cloud | **Ollama** |
| Remote API, no local GPU, share with a team | **HF Inference Endpoint** |
| Both local + cloud (recommended) | push GGUF to HF; Ollama locally, Endpoint remotely |

## Why it works

A model is only useful if something can call it. The Hub makes it portable and
versioned; Ollama makes it a live, local, OpenAI-shaped endpoint. With both in
place you have a **production-ready, self-hosted inference stack** — no per-token
bill, no data leaving your machine.

## Checkpoint

`ollama run granite-ml-coder "hello"` returns a response, and your model page is
live on huggingface.co.

➡️ Next: [Chapter 10 — Capstone: Network-Traffic Anomaly Detection](10-capstone-network-anomaly-detection.md)
