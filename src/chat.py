"""
chat.py — Talk to your fine-tuned model (or the base model) to sanity-check it.

    uv run python src/chat.py --model granite-ml-coder \
        --prompt "Write a scikit-learn pipeline to classify the iris dataset and explain how you guard against overfitting."

Omit --prompt for an interactive REPL. Use --model ibm-granite/granite-3.1-1b-a400m-instruct to compare
against the un-tuned base model.
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def generate(model, tokenizer, device, user_prompt: str, max_new_tokens: int) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert Python machine-learning engineer. "
                "Write correct, runnable code, explain each pipeline step, "
                "watch for overfitting and how gradient descent converges, "
                "and recommend the best model or formula for the task."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]
    # transformers 5.x returns a dict (input_ids + attention_mask), so use
    # return_dict=True and pass it through with **inputs.
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,  # keep Granite out of <think> mode for clean code output
    ).to(device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the model.")
    parser.add_argument("--model", default="granite-ml-coder",
                        help="Local path or Hub id.")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    device = pick_device()
    print(f"Loading {args.model} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    ).to(device)
    model.eval()

    if args.prompt:
        print("\n" + generate(model, tokenizer, device, args.prompt, args.max_new_tokens))
        return

    print("Interactive mode — type a question (Ctrl-C to quit).\n")
    try:
        while True:
            q = input("you> ").strip()
            if not q:
                continue
            print("\nmodel> " + generate(model, tokenizer, device, q, args.max_new_tokens) + "\n")
    except (KeyboardInterrupt, EOFError):
        print("\nbye")


if __name__ == "__main__":
    main()
