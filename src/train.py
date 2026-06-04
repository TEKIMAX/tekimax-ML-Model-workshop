"""
train.py — Fine-tune IBM Granite 3.1 1B into a Python/ML coding assistant on Apple
Silicon (MPS backend), then optionally push to the Hugging Face Hub.

This adapts the Hugging Face Trainer tutorial for a Mac:
  * No CUDA → no bitsandbytes, no bf16/fp16 mixed precision (kept in fp32, which
    is the reliable path on MPS; a 1B model trains comfortably in 64 GB).
  * Instruction tuning with the Granite chat template and prompt-masked labels.

Examples
--------
Quick smoke test (tiny subset, 1 epoch, no push):
    uv run python src/train.py --max-samples 200 --epochs 1

Full run and push to your Hub repo:
    uv run python src/train.py --push --hub-id Tekimax/granite-ml-coder
"""

from __future__ import annotations

import argparse

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from prepare_data import load_and_prepare

MODEL_NAME = "ibm-granite/granite-3.1-1b-a400m-instruct"
# Note: do NOT call torch.mps.empty_cache() from a TrainerCallback during
# training — emptying the MPS cache while GPU work is queued can deadlock the
# process (it hangs in uninterruptible state). Keep peak memory low instead
# (small batch, no gradient checkpointing) so the cache never needs clearing.


def pick_device(force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a small LLM on ML code.")
    parser.add_argument("--model-name", default=MODEL_NAME,
                        help="Base model to fine-tune (HF id). Default: IBM Granite 3.1 1B. "
                             "Any chat model with a chat template works, e.g. "
                             "meta-llama/Llama-3.2-1B-Instruct.")
    parser.add_argument("--output-dir", default="granite-ml-coder")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Gradient clipping norm. Lower (e.g. 0.5) guards "
                             "against the loss-explosion-to-NaN seen on MPS.")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap training rows for a quick smoke test.")
    parser.add_argument("--all-rows", action="store_true",
                        help="Use the full dataset (skip the ML-only filter).")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="Enable gradient checkpointing (less memory, ~20-30%% slower).")
    parser.add_argument("--eval", action="store_true",
                        help="Run in-loop evaluation. OFF by default: SDPA on MPS "
                             "emits NaN eval loss on padded batches. Use with --attn-eager.")
    parser.add_argument("--attn-eager", action="store_true",
                        help="Use eager attention (NaN-safe on MPS but ~6x slower). "
                             "Pair with --eval if you need in-loop eval on MPS.")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU training. Slower than MPS but completely "
                             "stable — avoids the MPS NaN/hang/OOM issues entirely.")
    parser.add_argument("--push", action="store_true",
                        help="Push the final model to the Hub.")
    parser.add_argument("--hub-id", default=None,
                        help="Hub repo id, e.g. Tekimax/granite-ml-coder.")
    args = parser.parse_args()

    device = pick_device(force_cpu=args.cpu)
    print(f"Using device: {device}")

    # --- Tokenizer -----------------------------------------------------------
    print(f"Base model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Data ----------------------------------------------------------------
    dataset = load_and_prepare(
        tokenizer,
        max_length=args.max_length,
        ml_only=not args.all_rows,
    )
    if args.max_samples:
        dataset["train"] = dataset["train"].select(
            range(min(args.max_samples, len(dataset["train"])))
        )
    print(f"Train: {len(dataset['train']):,} | Eval: {len(dataset['test']):,}")

    # Pads input_ids/attention_mask and pads labels with -100.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, padding=True, label_pad_token_id=-100
    )

    # --- Model ---------------------------------------------------------------
    # dtype="auto" loads weights in their saved dtype. On MPS we keep the model
    # in fp32 for training stability (mixed precision on MPS is still flaky).
    #
    # We use the default SDPA attention (fast on MPS). Note: SDPA can emit NaNs
    # on MPS for some *padded* batches — this shows up in the in-loop eval loss
    # while training loss stays healthy. We therefore default eval OFF on MPS
    # (see --eval) and avoid the bug entirely. "eager" attention is NaN-safe but
    # ~6x slower on MPS, so it's not worth it here.
    attn = "eager" if args.attn_eager else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.float32, attn_implementation=attn
    )
    model.config.use_cache = False  # required when training; re-enable for inference

    # --- Training configuration ---------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_grad_norm=args.max_grad_norm,
        logging_steps=10,
        # Eval is opt-in: on MPS the SDPA kernel NaNs on padded eval batches.
        # With eval off we monitor the (healthy) training loss instead and judge
        # final quality by generating samples (src/chat.py).
        eval_strategy="epoch" if args.eval else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=args.eval,
        # Mixed precision OFF — MPS path. (On an NVIDIA box set bf16=True.)
        bf16=False,
        fp16=False,
        # Gradient checkpointing: recompute activations in backward to save
        # memory. Helps avoid MPS OOM on heavy (long-sequence) batches.
        gradient_checkpointing=args.grad_checkpoint,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.grad_checkpoint else None,
        report_to="none",
        push_to_hub=args.push,
        hub_model_id=args.hub_id,
        use_cpu=args.cpu,
        # Keep dataloading in the main process. On macOS, worker processes use
        # the 'spawn' start method and each re-imports + copies state, which
        # bloats RAM into swap and *slows* CPU training. 0 = no workers.
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # Re-enable the KV cache so the saved model generates fast at inference time.
    trainer.model.config.use_cache = True
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSaved model to ./{args.output_dir}")

    if args.push:
        trainer.push_to_hub()
        print(f"Pushed to https://huggingface.co/{args.hub_id}")


if __name__ == "__main__":
    main()
