# Chapter 7 — Evaluation & Testing

## What you'll do

Decide whether your model is actually good: read the eval curve, diagnose
**overfitting** vs **underfitting**, and talk to the model to judge quality by
hand. These are the same skills you'll apply to the DNN you build in Chapter 10.

> **MPS note.** On Apple Silicon we run with in-loop eval **off** (the SDPA
> attention kernel emits `nan` on padded eval batches — Chapter 11). So you
> won't see `eval_loss` per epoch by default; you monitor the **training loss**
> and judge generalization by **generating samples** (Step 4). The concepts
> below still matter — they apply with `--eval` (eager attn) or on a CUDA box,
> and to the DNN you build in Chapter 10.

## Step 1 — The two numbers that matter

- **Training loss** — error on data the model *sees*. Always drops with enough
  training; on its own it tells you almost nothing about quality.
- **Evaluation loss** — error on the **held-out** 10% the model *never trains
  on*. This is the real measure of generalization.

The gap between them is the whole story.

## Step 2 — Diagnose the curve

```
        loss
         │
         │＼                         ＼ train
         │  ＼____ train               ＼____
         │       ＼___                      ＼______ ← eval RISES
   eval  │＼___________ eval                          (overfitting)
         │  (healthy: both low,         (memorizing the
         │   small gap)                  training set)
         └────────────────── epochs
```

| Symptom | Diagnosis | Fix |
|---|---|---|
| Train ↓, eval ↓, small gap | **Healthy** | ship it |
| Train ↓↓, eval flat or ↑ | **Overfitting** | fewer epochs, more/》diverse data, lower LR, add regularization |
| Train high, eval high | **Underfitting** | train longer, higher LR, bigger model |

`load_best_model_at_end=True` already protects you somewhat: it keeps the
epoch with the lowest eval loss, not the over-trained last one.

### What is overfitting, really?

The model stops learning *generalizable patterns* and starts memorizing the
training examples (even their noise). It aces data it has seen and fails on
anything new. The held-out eval set is how we catch it — performance there is a
proxy for performance in production.

### What is gradient descent doing?

Each step moves the weights a small distance (`learning_rate`) in the direction
that most reduces the loss. Too-large steps overshoot and the loss diverges;
too-small steps crawl. Our cosine schedule starts modest, holds, then shrinks
steps near the end to settle into a good minimum — the practical art behind
the curve above.

## Step 3 — Re-run evaluation any time

```python
from transformers import Trainer
metrics = trainer.evaluate()
print(metrics)   # {'eval_loss': 1.27, ...}
```

Loss is convenient but abstract. For generation quality, nothing beats reading
outputs.

## Step 4 — Talk to the model (the real test)

```bash
uv run python src/chat.py --model granite-ml-coder \
  --prompt "Write a scikit-learn pipeline to classify the iris dataset, use cross-validation, and explain how you guard against overfitting."
```

Compare against the **base** model to see what fine-tuning bought you:

```bash
uv run python src/chat.py --model ibm-granite/granite-3.1-1b-a400m-instruct \
  --prompt "Write a scikit-learn pipeline to classify the iris dataset..."
```

Judge by hand:

- Does it produce **runnable** code (imports, no obvious errors)?
- Does it **explain** the steps as the system prompt asks?
- Does it mention **overfitting / cross-validation / model choice** unprompted?
- Is the **format** clean and consistent?

> **Reality check.** A 1B model will sometimes produce imperfect or incomplete
> code. That's expected. The fine-tune should make it *more reliably on-format
> and on-topic* than the base model — that's the win at this size. For
> higher-stakes code you'll pair it with a human or a bigger model (Chapter 11).

## Step 5 — A tiny qualitative test set

Keep a handful of prompts you re-run after every training change — a manual
regression test:

```text
1. Write a sklearn pipeline for iris with cross-validation.
2. Explain gradient descent to a beginner with a code example.
3. My model gets 99% train and 70% test accuracy — what's wrong and how do I fix it?
4. Build a Keras autoencoder to detect anomalies in network traffic.
```

If answers improve (or stop regressing) across runs, you're moving the right way.

## Why it works

Loss curves catch overfitting early and cheaply; hand-testing catches the things
loss can't see (does the code actually run? is the explanation right?). Together
they tell you when the model is production-ready — the same judgment you'll use
on the DNN in Chapter 10.

## Checkpoint

You can state your model's final `eval_loss`, confirm it's below the base
model's, and show one prompt where the fine-tuned model clearly beats the base.

➡️ Next: [Chapter 8 — Quantization](08-quantization.md)
