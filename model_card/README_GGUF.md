---
license: apache-2.0
base_model: Tekimax/granite-ml-coder
library_name: gguf
pipeline_tag: text-generation
tags:
  - code
  - machine-learning
  - data-science
  - gguf
  - llama.cpp
  - ollama
  - granite
language:
  - en
---

# granite-ml-coder-GGUF

GGUF builds of [`Tekimax/granite-ml-coder`](https://huggingface.co/Tekimax/granite-ml-coder)
— a compact **Python / machine-learning coding assistant** fine-tuned from
`ibm-granite/granite-3.1-1b-a400m-instruct`. These run on CPU and Apple Silicon (Metal) via
**llama.cpp**, **Ollama**, and **LM Studio**.

## Files

| File | Bits | Size | Notes |
|---|---|---|---|
| `granite-ml-coder-Q4_K_M.gguf` | 4-bit | ~378 MB | recommended — loads fast, near-full quality |

## Run it

### Ollama (from the public registry)
```bash
ollama run tekimaxllc/granite-ml-coder "Write a sklearn pipeline for the iris dataset"
```

### Ollama (from this GGUF directly)
```bash
ollama run hf.co/Tekimax/granite-ml-coder-GGUF
```

### llama.cpp
```bash
llama-cli -m granite-ml-coder-Q4_K_M.gguf \
  -p "Write a Keras autoencoder for network-traffic anomaly detection"
```

### LM Studio
Search for `Tekimax/granite-ml-coder-GGUF`, download the `Q4_K_M` file, and chat.

## Suggested system prompt
```
You are an expert Python machine-learning engineer. Write correct, runnable code,
explain each pipeline step, watch for overfitting and how gradient descent
converges, and recommend the best model or formula for the task.
```

## Intended use & limitations

Good for drafting Python ML code (scikit-learn, pandas, NumPy, Keras) and
explaining ML concepts, fully offline. **It's a 1B model** — treat output as a
fast first draft and verify before use. See the
[full model card](https://huggingface.co/Tekimax/granite-ml-coder) for training
details and limitations.

## License
Apache-2.0 (inherited from `ibm-granite/granite-3.1-1b-a400m-instruct`).
