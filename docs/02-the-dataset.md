# Chapter 2 — The Dataset

## What you'll do

Choose a public dataset, understand its format, and **filter it to the ML/DS
content** that matches our goal — so the model learns to write the kind of code
you'll actually ask it for.

## The data decides the behavior

A fine-tuned model is a mirror of its training data. Want a model that writes
sklearn pipelines and explains overfitting? Train it on examples of exactly
that. The single most important choice in this whole workshop is **what's in the
dataset**.

## Step 1 — The dataset we use

[`iamtarun/python_code_instructions_18k_alpaca`](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca)
— 18.6K rows of Python coding tasks in **Alpaca format**:

| Column | Meaning |
|---|---|
| `instruction` | the task, e.g. *"Write a function to train a logistic regression model"* |
| `input` | optional extra context (a data sample, constraints) |
| `output` | the reference answer (the code) |
| `prompt` | a pre-rendered template (we ignore it and build our own) |

Inspect it without downloading the whole thing:

```bash
uv run python -c "from datasets import load_dataset; \
d=load_dataset('iamtarun/python_code_instructions_18k_alpaca', split='train'); \
print(d); print(d[0])"
```

## Step 2 — Filter to ML / DS / DNN / NN rows

There's no perfect public "ML-only" instruction set, so we **filter** the
general Python dataset down to rows that mention ML concepts and libraries
(`sklearn`, `pandas`, `numpy`, `torch`, `neural network`, `gradient descent`,
`overfitting`, …). This is implemented in
[`src/prepare_data.py`](../src/prepare_data.py):

```python
ML_KEYWORDS = ["sklearn", "pandas", "numpy", "tensorflow", "pytorch",
               "neural network", "gradient descent", "overfitting",
               "regression", "classifier", ...]

def is_ml_row(example):
    text = f"{example['instruction']}\n{example['output']}".lower()
    return any(kw in text for kw in ML_KEYWORDS)

ds = ds.filter(is_ml_row)
```

See how many rows survive the filter:

```bash
uv run python src/prepare_data.py --inspect 2
```

This prints the count plus a couple of fully-formatted examples (Chapter 3
explains the formatting). Filtering trades quantity for **relevance** — a
smaller, on-topic dataset usually beats a larger, noisy one for a focused model.

## Step 3 — Hold out an evaluation split

We later split off 10% as a **test set** (`train_test_split(test_size=0.1)`).
The model never trains on it, so its loss on that split tells us whether the
model is learning generalizable patterns or just memorizing — the core of the
overfitting discussion in Chapter 7.

## Why it works

Pretraining gave Granite broad competence. Fine-tuning on this filtered set
**concentrates** that competence on ML coding: the format of a good answer, the
libraries to reach for, the way to explain a step. We're not teaching it Python
from scratch — we're shaping its defaults.

## Bring your own data (optional)

To use your own corpus, produce rows with `instruction` / `input` / `output`
fields (JSONL is easiest) and point `load_dataset` at them:

```python
ds = load_dataset("json", data_files="data/my_ml_examples.jsonl", split="train")
```

Everything downstream stays the same.

## Checkpoint

`uv run python src/prepare_data.py --inspect 2` prints a non-zero ML row count
and two readable chat-formatted examples.

➡️ Next: [Chapter 3 — Tokenization & Formatting](03-tokenization-and-formatting.md)
