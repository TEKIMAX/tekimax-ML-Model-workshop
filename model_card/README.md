---
license: apache-2.0
base_model: ibm-granite/granite-3.1-1b-a400m-instruct
library_name: transformers
pipeline_tag: text-generation
tags:
  - code
  - machine-learning
  - data-science
  - scikit-learn
  - fine-tuned
  - granite
  - mlx
  - gguf
datasets:
  - iamtarun/python_code_instructions_18k_alpaca
language:
  - en
---

# granite-ml-coder

A compact **Python / machine-learning coding assistant**, fine-tuned from
[`ibm-granite/granite-3.1-1b-a400m-instruct`](https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-instruct). It writes runnable
scikit-learn / pandas / NumPy code, explains ML pipeline steps, and reasons about
everyday concepts like overfitting, cross-validation, and gradient descent.

It is small enough to run **fully locally** — on a laptop CPU, via Ollama, or
quantized to GGUF — which makes it a good private copilot for data-science work
where your data shouldn't leave the machine.

> Built as the reference model for the **TEKIMAX ML Model Workshop**
> (fine-tune → host on HF/Ollama → build a production DL app).

## Quick start

### Transformers
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Tekimax/granite-ml-coder")
model = AutoModelForCausalLM.from_pretrained(
    "Tekimax/granite-ml-coder", dtype=torch.float32, attn_implementation="eager"
)

messages = [
    {"role": "system", "content": "You are an expert Python machine-learning engineer."},
    {"role": "user", "content": "Write a scikit-learn pipeline to classify the iris dataset and explain how you avoid overfitting."},
]
enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                             return_tensors="pt", return_dict=True, enable_thinking=False)
out = model.generate(**enc, max_new_tokens=400, do_sample=True, temperature=0.7, top_p=0.9)
print(tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
```

### Ollama (recommended for local use)
```bash
ollama run tekimaxllc/granite-ml-coder "Write a Keras autoencoder for network-traffic anomaly detection"
```

### GGUF / llama.cpp
A 4-bit `Q4_K_M` build (~378 MB) is available at
[`Tekimax/granite-ml-coder-GGUF`](https://huggingface.co/Tekimax/granite-ml-coder-GGUF):
```bash
llama-cli -m granite-ml-coder-Q4_K_M.gguf -p "Write a sklearn pipeline"
```

## Intended use

- Drafting Python ML code (scikit-learn, pandas, NumPy, Keras) inside notebooks/IDEs
- Explaining ML pipeline steps and concepts (overfitting, gradient descent, model choice)
- A private, offline coding copilot for data-science tasks

## Training

| | |
|---|---|
| Base model | `ibm-granite/granite-3.1-1b-a400m-instruct` (Apache-2.0, IBM) — ~1.3B total / 400M active MoE |
| Data | [`iamtarun/python_code_instructions_18k_alpaca`](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca), filtered to ML/DS rows (≈2,341 examples) |
| Method | Full fine-tune, instruction format with the Granite chat template; loss computed on the assistant answer only (prompt tokens masked) |
| Schedule | 2 epochs · effective batch size 16 · LR 2e-5 cosine · max_len 512 |
| Hardware | Apple M2 Ultra, **CPU** (the MPS/Metal backend was unstable for fine-tuning on torch 2.12 — see the workshop appendix) |
| Result | training loss decreasing steadily, no NaN |

## Limitations

- **It's a 1B model.** It learns *style, format, and common patterns* well, but
  is not a frontier model — it can produce incomplete or subtly wrong code, and
  it won't reliably "pick the best model" for hard problems. Treat its output as
  a fast first draft and verify before use.
- English, Python-focused. Strongest on classic ML (sklearn/pandas); weaker on
  large, novel, or multi-file tasks.
- Inherits any biases/limitations of the base model and the training dataset.

## License

Apache-2.0, inherited from the base model and dataset.

## Citation / credits

- Base model: [Granite](https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-instruct)
- Dataset: [iamtarun/python_code_instructions_18k_alpaca](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca)
- Built with Hugging Face `transformers` `Trainer`; quantized with `llama.cpp`.
