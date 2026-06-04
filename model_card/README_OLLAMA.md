# granite-ml-coder — Ollama page description

Paste this into the description box at
https://ollama.com/tekimaxllc/granite-ml-coder (Edit), then Save.

---

A compact **Python / machine-learning coding assistant**, fine-tuned from
IBM Granite 3.1 1B. It writes runnable scikit-learn / pandas / NumPy / Keras code,
explains ML pipeline steps, and reasons about overfitting, cross-validation, and
gradient descent. Small enough to run fully **locally and offline** — a private
data-science copilot.

## Run

```bash
ollama run tekimaxllc/granite-ml-coder "Write a scikit-learn pipeline for the iris dataset and explain how you avoid overfitting"
```

OpenAI-compatible API (for apps / notebooks):

```bash
curl http://localhost:11434/v1/chat/completions -d '{
  "model": "tekimaxllc/granite-ml-coder",
  "messages": [{"role":"user","content":"Write a Keras autoencoder for network anomaly detection"}]
}'
```

## Details

- **Base:** ibm-granite/granite-3.1-1b-a400m-instruct (Apache-2.0)
- **Quantization:** Q4_K_M (~378 MB)
- **Built-in system prompt:** expert Python ML engineer (writes code, explains
  steps, watches overfitting / gradient descent, recommends models)
- **Context:** 4096 tokens · temperature 0.7 · top_p 0.9

## Limitations

It's a 1B model — great for fast, on-format ML code drafts and explanations,
but verify its output before use; it won't reliably solve hard, novel problems.

Full model card: https://huggingface.co/Tekimax/granite-ml-coder
