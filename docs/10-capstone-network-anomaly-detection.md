# Chapter 10 — Capstone: Network-Traffic Anomaly Detection

## What you'll do

This is the payoff. You'll open a **Jupyter notebook**, use your fine-tuned
model as a coding **copilot** — running locally via **Ollama** *or* via the
**Hugging Face Inference API** — and together build a **deep-learning anomaly
detector for network traffic**: trained, evaluated, accurate, and deployable.

The companion notebook is
[`notebooks/capstone_network_anomaly_detection.ipynb`](../notebooks/capstone_network_anomaly_detection.ipynb).
This chapter is its narration. Stack: **Keras** (the DNN) + **scikit-learn**
(data prep + metrics).

## The scenario

Flag malicious / anomalous network connections using an **autoencoder**:

- Train it to **reconstruct *normal* traffic**.
- Normal connections reconstruct with low error; attacks reconstruct poorly.
- **Anomaly = high reconstruction error** — no attack labels needed at train time,
  so it catches **novel / zero-day** patterns.

We use the built-in **KDD Cup '99** intrusion dataset (`sklearn.datasets.fetch_kddcup99`)
so the notebook is fully reproducible with no manual downloads.

> **Why local matters here.** Network telemetry is sensitive. With a local
> copilot and local training, the data never leaves your machine — the whole
> point of owning your stack.

## Step 1 — Choose where the copilot runs

The **same fine-tuned model** can answer two ways. The notebook flips on one
env var:

```bash
# Option A — local on your Mac (default)
ollama serve
ollama run granite-ml-coder "ready?"

# Option B — Hugging Face Inference API
export COPILOT_BACKEND=hf
export HF_TOKEN=hf_xxx
export HF_ENDPOINT_URL=https://<your-endpoint>.endpoints.huggingface.cloud/v1/chat/completions
```

The `ask()` helper picks the backend automatically:

```python
BACKEND = os.environ.get("COPILOT_BACKEND", "ollama")   # 'ollama' or 'hf'
def ask(prompt):
    if BACKEND == "hf":   # OpenAI-compatible HF endpoint
        ...
    else:                 # local Ollama
        ...
```

So your model runs **on your Mac via Ollama** or **as an HF API** with zero code
changes downstream — exactly the portability you asked for.

## Step 2 — Use the model as a copilot

```python
print(ask("Write pandas + sklearn code to one-hot encode protocol_type/service/flag, "
          "make a binary normal-vs-attack target, and standardize features."))
```

You read its suggestion, adapt the good parts into a real cell, run it. **You
stay in control** — the model drafts, you verify. (At 1B it will sometimes be
wrong; treat it as a fast first draft.)

## Step 3 — Clean the data

Load KDD Cup '99, decode byte-strings, one-hot the three categorical columns,
build a binary target, and scale **using normal-train statistics only** (no
leakage). See cells 2–3 of the notebook.

## Step 4 — Build & train the autoencoder (Keras)

```python
model = keras.Sequential([
    keras.layers.Input((input_dim,)),
    keras.layers.Dense(64, activation="relu"), keras.layers.Dropout(0.1),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(16, activation="relu"),          # bottleneck
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(input_dim, activation="linear"), # reconstruction
])
model.compile(optimizer="adam", loss="mse")
model.fit(X_tr_norm, X_tr_norm, validation_data=(X_val_norm, X_val_norm),
          epochs=30, callbacks=[keras.callbacks.EarlyStopping(patience=3,
                                restore_best_weights=True)])
```

Note the through-line from earlier chapters: **gradient descent** (`adam`),
**overfitting** control (`Dropout` + `EarlyStopping`), and a held-out
**validation set** — the concepts you fine-tuned the LLM to explain, now applied
to your DNN.

## Step 5 — Threshold and measure accuracy (scikit-learn)

Set the anomaly threshold from normal-validation reconstruction errors (e.g. the
99th percentile), then evaluate on a mixed test set (held-out normal + all
attacks):

```python
y_pred = (recon_error(X_test) > threshold).astype(int)
print(roc_auc_score(y_test, recon_error(X_test)))   # honest separation metric
print(classification_report(y_test, y_pred, target_names=["normal", "attack"]))
```

ROC-AUC is the headline number — how well reconstruction error separates attacks
from normal. A good autoencoder scores high here.

## Step 6 — Improve with the copilot

```python
print(ask(f"My autoencoder anomaly detector gets ROC-AUC {auc:.3f}. Suggest 3 "
          "concrete changes to improve detection without too many false positives."))
```

Evaluate its ideas (bottleneck size, threshold percentile, feature engineering),
try the promising ones.

## Step 7 — Deploy in production

```python
model.save("netanomaly_autoencoder.keras")
joblib.dump(scaler, "netanomaly_scaler.joblib")
json.dump({"threshold": float(threshold), "columns": list(X.columns)},
          open("netanomaly_meta.json", "w"))

def score(flows):
    # load model + scaler + threshold, return anomaly flags + scores
    ...
```

That `score()` is your deployable surface — drop it behind a FastAPI route, a
stream processor, or a Cloudflare Worker. Saving **model + scaler + threshold**
together is what makes it reproducible and **governed** (Chapter 9): you can
version, audit, and roll back the whole detector.

## Why it works

You've assembled a fully local ML development loop: a private LLM copilot
(yours, fine-tuned, served by Ollama *or* HF) + Keras/sklearn + disciplined
evaluation. The LLM compresses the "how do I write this" friction; your judgment
and the held-out test set guarantee quality. Local, private, accurate,
deployable — the production application this workshop set out to build.

## Checkpoint

Your notebook trains the autoencoder, reports a strong ROC-AUC on the mixed test
set, saves the three artifacts, and `score()` flags attack rows — with the
copilot running on your Mac (or HF) the whole time.

➡️ Next: [Chapter 11 — Appendix & Troubleshooting](11-appendix-troubleshooting.md)
