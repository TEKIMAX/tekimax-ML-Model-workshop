# Chapter 6 — Training

## What you'll do

Assemble the `Trainer`, run a quick smoke test, then launch the real training
run — and learn to read the loss numbers as they scroll by.

## Step 1 — Build the Trainer

The `Trainer` ties together everything from Chapters 2–5: model, args, data, and
collator. From [`src/train.py`](../src/train.py):

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
)
trainer.train()
```

Underneath, the `Trainer` handles the entire loop for you: batching, shuffling,
padding, the forward pass, loss computation, backpropagation, the optimizer
step, evaluation, and checkpointing. You supply the pieces; it runs the engine.

```
for each batch:
    logits = model(input_ids)        # forward pass
    loss   = cross_entropy(logits, labels)   # how wrong? (answer tokens only)
    loss.backward()                  # backprop: gradients
    optimizer.step()                 # nudge weights down the gradient
    optimizer.zero_grad()            # reset for next batch
```

That `optimizer.step()` *is* **gradient descent** — the model walks downhill on
the loss surface. (You'll explain this exact loop to your *own* model in the
capstone.)

## Step 2 — Smoke test first (always)

Never launch a multi-hour run without proving the pipeline end-to-end on a tiny
slice:

```bash
uv run python src/train.py --max-samples 200 --epochs 1
```

This trains on 200 examples for one epoch — a few minutes. If it completes and
saves to `./granite-ml-coder/`, every moving part works: data loads, tokenization
masks correctly, the model trains on MPS, checkpoints save. *Now* scale up.

## Step 3 — The full run (use CPU on Apple Silicon)

> **Hard-won lesson.** On this stack (M2 Ultra, torch 2.12, transformers 5.9,
> Granite) the **MPS/Metal backend is unreliable for fine-tuning**: SDPA attention
> produces `nan`, eager attention hangs the process, and the memory watermark
> has no safe setting (false-OOM vs. swap-stall). After many attempts, the
> backend that trains correctly and stably is the **CPU**. It's slower (~15s/step
> vs ~6s) but it *finishes* with a healthy loss curve. See Chapter 11 for the
> full debugging saga — it's the most useful part of this workshop.

```bash
uv run python src/train.py --cpu \
  --epochs 3 --batch-size 4 --grad-accum 4 --max-length 512
```

- `--cpu` — force CPU (sets `use_cpu=True`). The M2 Ultra's 24 cores handle a
  1B model fine; no Metal bugs.
- `--batch-size 4 --grad-accum 4` — effective batch size **16**.
- In-loop eval stays **off by default** (we monitor the training loss and judge
  quality by generating samples — Chapter 7).

Expect ~15s/step, ~105 min for 3 epochs on an M2 Ultra. A healthy run looks like
the loss falling `~1.06 → ~0.34` with **no `nan`**.

> **If you have an NVIDIA GPU**, drop `--cpu`, add `bf16=True` in
> `TrainingArguments`, and it trains in minutes — the MPS problems are
> Apple-Silicon-specific.

Other useful flags:

```bash
--epochs 2            # fewer/more passes
--lr 1e-5             # gentler learning rate
--all-rows            # train on the full (unfiltered) dataset
--max-samples 200     # tiny smoke test
```

## Step 4 — Reading the loss

You'll see lines like:

```
{'loss': 1.06, 'grad_norm': 5.1, 'learning_rate': 1.9e-05, 'epoch': 0.07}
{'loss': 0.90, 'grad_norm': 4.0, 'learning_rate': 2.0e-05, 'epoch': 0.14}
{'loss': 0.88, 'grad_norm': 4.3, 'learning_rate': 1.9e-05, 'epoch': 0.34}
```

(`eval_loss` lines only appear if you pass `--eval`; it's off by default on MPS.)

- **`loss`** — training error on the current batch. Should trend **down**.
- **`eval_loss`** — error on the held-out set, reported each epoch. This is the
  honest number — it measures generalization.
- **`learning_rate`** — watch it warm up then decay (the cosine schedule).

**What healthy looks like:** both losses fall, with `eval_loss` close behind
`loss`. **What overfitting looks like:** `loss` keeps falling but `eval_loss`
flattens or *rises* — the model is memorizing. Chapter 7 covers what to do.

## Step 5 — What gets saved

When training ends, `load_best_model_at_end` restores the lowest-`eval_loss`
checkpoint, then `train.py` re-enables the KV cache and writes the model +
tokenizer to `./granite-ml-coder/`:

```
granite-ml-coder/
├── config.json
├── model.safetensors          # your fine-tuned weights
├── tokenizer.json
├── tokenizer_config.json
└── ...
```

To push straight to the Hub in the same run, add `--push --hub-id Tekimax/granite-ml-coder`
(covered in Chapter 9).

## Why it works

The `Trainer` is a battle-tested implementation of the training loop. By feeding
it correctly-masked data and sane hyperparameters, gradient descent does the
rest: each step makes the model slightly better at producing ML answers, and
evaluation keeps us honest about whether it's *learning* or just *memorizing*.

## Checkpoint

`./granite-ml-coder/model.safetensors` exists, and your final `eval_loss` is lower
than the very first one logged.

➡️ Next: [Chapter 7 — Evaluation & Testing](07-evaluation-and-testing.md)
