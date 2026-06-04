"""
prepare_data.py — Load the public alpaca-style Python dataset, filter it down to
machine-learning / data-science rows, and build a tokenized, instruction-tuned
dataset where the *prompt* tokens are masked so the model only learns to produce
the *answer*.

Used by train.py, but you can run it standalone to inspect what the model sees:

    uv run python src/prepare_data.py --inspect 3
"""

from __future__ import annotations

import argparse

from datasets import load_dataset

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
DATASET_ID = "iamtarun/python_code_instructions_18k_alpaca"

# We don't have a dedicated "ML only" public dataset, so we filter the general
# Python instruction set down to rows that mention ML / DS / DNN / NN concepts.
# This biases the fine-tune toward your goal: write ML code, explain pipeline
# steps, reason about overfitting / gradient descent / model selection.
ML_KEYWORDS = [
    # libraries
    "sklearn", "scikit-learn", "pandas", "numpy", "tensorflow", "keras",
    "pytorch", "torch", "xgboost", "lightgbm", "matplotlib", "seaborn",
    "scipy", "statsmodels",
    # concepts
    "machine learning", "deep learning", "neural network", "neural net",
    "gradient descent", "backpropagation", "overfitting", "underfitting",
    "regularization", "cross-validation", "cross validation", "train_test_split",
    "regression", "classifier", "classification", "clustering", "k-means",
    "kmeans", "random forest", "decision tree", "svm", "naive bayes",
    "logistic regression", "linear regression", "feature", "dataset",
    "model", "train", "predict", "accuracy", "precision", "recall", "f1",
    "loss function", "learning rate", "epoch", "activation", "convolution",
    "cnn", "rnn", "lstm", "transformer", "embedding", "dropout", "batch",
    "hyperparameter", "confusion matrix", "roc", "auc",
]


def is_ml_row(example: dict) -> bool:
    """True if the instruction or output looks ML / data-science related."""
    haystack = f"{example.get('instruction', '')}\n{example.get('output', '')}".lower()
    return any(kw in haystack for kw in ML_KEYWORDS)


def build_messages(example: dict) -> list[dict]:
    """Turn an alpaca row into chat messages.

    The alpaca schema is (instruction, input, output). When `input` is present
    it carries extra context (e.g. a data sample), so we append it to the user
    turn. A system prompt nudges the model toward the behaviour you asked for.
    """
    user = example["instruction"].strip()
    extra = (example.get("input") or "").strip()
    if extra:
        user = f"{user}\n\n{extra}"

    return [
        {
            "role": "system",
            "content": (
                "You are an expert Python machine-learning engineer. "
                "Write correct, runnable code, explain each pipeline step, "
                "watch for overfitting and how gradient descent converges, "
                "and recommend the best model or formula for the task."
            ),
        },
        {"role": "user", "content": user},
        {"role": "assistant", "content": example["output"].strip()},
    ]


def make_tokenize_fn(tokenizer, max_length: int):
    """Returns a map() function that produces input_ids + masked labels.

    Only the assistant's answer contributes to the loss. We do this by building
    the prompt (everything up to the assistant turn) and the full conversation
    separately, then setting the prompt portion of `labels` to -100 (ignored by
    the loss).
    """

    def tokenize(example: dict) -> dict:
        messages = build_messages(example)

        # Prompt = system + user, with the assistant generation prefix added.
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        # Full = prompt + assistant answer (+ end-of-turn token from the template).
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        full_ids = full_ids[:max_length]
        labels = list(full_ids)
        # Mask the prompt so loss is computed on the answer only.
        n_mask = min(len(prompt_ids), len(labels))
        for i in range(n_mask):
            labels[i] = -100

        # n_supervised = how many answer tokens survived truncation. If the
        # prompt alone is >= max_length, every label is -100 and this row would
        # produce a NaN loss (mean over zero valid tokens). We tag it so the
        # caller can drop it.
        n_supervised = sum(1 for l in labels if l != -100)

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
            "n_supervised": n_supervised,
        }

    return tokenize


def load_and_prepare(tokenizer, max_length: int = 1024, test_size: float = 0.1,
                     ml_only: bool = True, seed: int = 42):
    """Load, filter, tokenize, and split the dataset for the Trainer."""
    ds = load_dataset(DATASET_ID, split="train")
    print(f"Loaded {len(ds):,} rows from {DATASET_ID}")

    if ml_only:
        ds = ds.filter(is_ml_row)
        print(f"Filtered to {len(ds):,} ML/DS-related rows")

    tokenized = ds.map(
        make_tokenize_fn(tokenizer, max_length),
        remove_columns=ds.column_names,
        desc="Tokenizing",
    )

    # Drop rows whose answer was entirely truncated away (all labels == -100).
    # These produce NaN losses that poison training. Then remove the helper col.
    before = len(tokenized)
    tokenized = tokenized.filter(lambda ex: ex["n_supervised"] > 0)
    dropped = before - len(tokenized)
    if dropped:
        print(f"Dropped {dropped} rows with no answer tokens after truncation "
              f"(prompt >= max_length={max_length})")
    tokenized = tokenized.remove_columns(["n_supervised"])

    return tokenized.train_test_split(test_size=test_size, seed=seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the prepared dataset.")
    parser.add_argument("--inspect", type=int, default=2,
                        help="How many formatted examples to print.")
    parser.add_argument("--all", action="store_true",
                        help="Do NOT filter to ML rows (use the full dataset).")
    args = parser.parse_args()

    ds = load_dataset(DATASET_ID, split="train")
    if not args.all:
        ds = ds.filter(is_ml_row)
    print(f"\n{len(ds):,} rows after filtering (ml_only={not args.all})\n")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("ibm-granite/granite-3.1-1b-a400m-instruct")
    for row in ds.select(range(min(args.inspect, len(ds)))):
        print("=" * 80)
        print(tok.apply_chat_template(build_messages(row), tokenize=False,
                                      add_generation_prompt=False))
