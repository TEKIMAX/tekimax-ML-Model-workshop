# Chapter 5 — Training Configuration

## What you'll do

Understand every `TrainingArguments` knob we set, why it has that value on Apple
Silicon, and which ones you'll tune for your own runs.

`TrainingArguments` is the single object that configures the whole run. Here's
ours (from [`src/train.py`](../src/train.py)), annotated.

```python
training_args = TrainingArguments(
    output_dir="granite-ml-coder",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    bf16=False,            # ← MPS: mixed precision OFF
    fp16=False,            # ← MPS: mixed precision OFF
    report_to="none",
)
```

## Duration & batch size

| Arg | Value | Meaning |
|---|---|---|
| `num_train_epochs` | `3` | how many passes over the data. 1 = quick, 3 = solid for a small set |
| `per_device_train_batch_size` | `4` | examples processed at once. Raise if RAM allows, lower if you OOM |
| `gradient_accumulation_steps` | `4` | accumulate gradients over 4 batches before updating |

**Effective batch size = batch_size × grad_accum = 4 × 4 = 16.** Gradient
accumulation lets a memory-limited Mac *simulate* a big batch (smoother training)
without holding it all in memory at once.

## The learning rate (the most important knob)

```python
learning_rate=2e-5, lr_scheduler_type="cosine", warmup_ratio=0.03
```

- `learning_rate=2e-5` — how big each weight update is. Fine-tuning uses **small**
  rates (1e-5–5e-5): the model is already good, so we nudge, not shove. Too high →
  it forgets pretraining ("catastrophic forgetting") and the loss diverges.
- `warmup_ratio=0.03` — ramp the LR up over the first 3% of steps so early,
  noisy gradients don't wreck the weights.
- `lr_scheduler_type="cosine"` — then smoothly decay the LR to near zero, which
  helps the model settle into a good minimum.

```
LR │      ___________
   │    /            \____
   │   /                  \___
   │  /(warmup)    (cosine decay) \___
   └─────────────────────────────────── steps
```

## Precision — the Apple Silicon adjustment

```python
bf16=False, fp16=False
```

> On **NVIDIA Ampere+** GPUs you'd set `bf16=True` for ~2× speed and half the
> memory. On **MPS**, mixed-precision training is still unreliable, so we keep
> both **off** and train in fp32 (Chapter 4). This is the single biggest
> difference from the stock tutorial.

`gradient_checkpointing` (trades compute for memory by recomputing activations)
is another NVIDIA-favorite — unnecessary for a 1B model in 64 GB, so we leave
it off for speed. Turn it on if you scale up and hit memory limits.

## Evaluation, saving, and best-model selection

```python
eval_strategy="epoch", save_strategy="epoch",
save_total_limit=2, load_best_model_at_end=True,
```

- `eval_strategy="epoch"` — run the held-out eval set after each epoch. This is
  how we **watch for overfitting** (Chapter 7).
- `save_strategy="epoch"` — checkpoint after each epoch so a crash isn't fatal.
- `save_total_limit=2` — keep only the 2 most recent checkpoints (disk hygiene).
- `load_best_model_at_end=True` — when training finishes, reload the checkpoint
  with the **lowest eval loss**, not necessarily the last one. Requires
  `eval_strategy` to be set.

## Logging

```python
logging_steps=10, report_to="none"
```

Print the training loss every 10 steps so you can watch it fall.
`report_to="none"` disables external loggers (Weights & Biases, etc.); set it to
`"wandb"` if you want dashboards.

## Why it works

These settings encode the fine-tuning philosophy: **small, warmed-up, decaying
updates** over a few epochs, **evaluated** each pass, **checkpointed** for
safety, with the **best** model kept — all within the memory and precision
constraints of Apple Silicon.

## Checkpoint

You don't run anything here — but you should now be able to answer: *What's my
effective batch size, and why is `bf16` off?* (Answers: 16, and because MPS
mixed precision is unreliable.)

➡️ Next: [Chapter 6 — Training](06-training.md)
