# Chapter 3 — Tokenization & Formatting

## What you'll do

Turn human-readable rows into the integer tensors a model trains on — and do it
the *instruction-tuning* way, with a **chat template** and **prompt masking** so
the model learns to *answer*, not to *parrot the question*.

## Step 1 — Tokenization in one minute

Models don't see text; they see **token IDs**. The tokenizer splits text into
sub-word tokens and maps each to an integer.

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("ibm-granite/granite-3.1-1b-a400m-instruct")
tok("train a model")["input_ids"]   # e.g. [3983, 264, 1614]
```

The tokenizer produces two arrays the model consumes:

- `input_ids` — the token integers.
- `attention_mask` — `1` for real tokens, `0` for padding (so the model ignores pad).

## Step 2 — The chat template (this is the key idea)

Granite is a **chat** model. It was trained to expect conversations wrapped in
special control tokens that mark who is speaking. If you fine-tune with plain
text, you fight that structure. Instead, use the model's own template:

```python
messages = [
    {"role": "system",    "content": "You are an expert ML engineer..."},
    {"role": "user",      "content": "Write a sklearn pipeline for iris."},
    {"role": "assistant", "content": "<the reference code>"},
]
text = tok.apply_chat_template(messages, tokenize=False)
```

This renders something like:

```
<|im_start|>system
You are an expert ML engineer...<|im_end|>
<|im_start|>user
Write a sklearn pipeline for iris.<|im_end|>
<|im_start|>assistant
<the reference code><|im_end|>
```

Training in this exact shape means inference works the same way — no format
mismatch between how you train and how you call the model later.

We add a **system prompt** that encodes the behavior you asked for ("write
correct code, explain steps, watch overfitting, recommend the best model"). Every
training example reinforces that persona.

## Step 3 — Prompt masking (train on the answer only)

Here's the subtle, important part. If we compute loss over the *entire*
sequence, the model spends effort learning to reproduce the **question** — which
the user already typed. We only want it to learn the **answer**.

So we build two versions and mask:

```python
prompt_ids = tok(apply_chat_template(messages[:-1], add_generation_prompt=True))
full_ids   = tok(apply_chat_template(messages))           # prompt + answer

labels = list(full_ids)
for i in range(len(prompt_ids)):
    labels[i] = -100          # -100 == "ignore in the loss"
```

`-100` is PyTorch's "ignore index": those positions contribute zero to the loss.
The model still *reads* the prompt (it's in `input_ids`), but is only *graded* on
generating the answer. This is implemented in `make_tokenize_fn` in
[`src/prepare_data.py`](../src/prepare_data.py).

```
input_ids:  [ system + user prompt ........ | assistant answer ........ ]
labels:     [ -100  -100  -100 ... -100      | tok  tok  tok  ...  tok   ]
                    ↑ ignored (not graded)        ↑ this is what we learn
```

## Step 4 — Truncation

Long examples are truncated to `max_length` (default 1024 tokens) to bound
memory. Most code examples fit; the rare giant one is clipped.

## Step 5 — Dynamic padding with a data collator

Batches need equal-length rows. Rather than padding *every* example to 1024
(wasteful), a **data collator** pads each batch to its own longest row:

```python
from transformers import DataCollatorForSeq2Seq
collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)
```

We use `DataCollatorForSeq2Seq` (not `DataCollatorForLanguageModeling`) because
it pads our **precomputed `labels`** with `-100` too — preserving the masking
from Step 3. This saves compute by never processing unnecessary padding tokens.

## Why it works

The model learns a function: *given this conversation up to the assistant turn,
produce a good ML answer.* The chat template gives it the right scaffolding,
prompt masking points the loss at the right target, and the collator keeps it
efficient.

## Checkpoint

```bash
uv run python src/prepare_data.py --inspect 1
```

You should see one example wrapped in `<|im_start|>…<|im_end|>` blocks with a
system, user, and assistant turn.

➡️ Next: [Chapter 4 — Loading the Model](04-loading-the-model.md)
