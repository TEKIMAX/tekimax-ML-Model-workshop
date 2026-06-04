"""
watch_training.py — a tiny real-time dashboard for a training run.

It tails a training **log file**, parses the Trainer's tqdm progress line and the
periodic loss dicts, and redraws a live view: a progress bar, step/ETA/rate, the
latest metrics, and a loss sparkline.

Usage
-----
1) Run training while teeing its output to a log (unbuffered so loss shows live):
       PYTHONUNBUFFERED=1 uv run python src/train.py --cpu ... 2>&1 | tee train.log

2) In another terminal, watch it:
       uv run python src/watch_training.py train.log

You can also point it at any file that contains Trainer output. Ctrl-C to quit.
"""

from __future__ import annotations

import argparse
import re
import sys
import time

# Trainer's tqdm line, e.g. " 34%|███ | 100/294 [1:12:00<0:58:00, 43.47s/it]"
PROG_RE = re.compile(
    r"(\d+)/(\d+)\s*\[(\d+:\d{2}(?::\d{2})?)<(\d+:\d{2}(?::\d{2})?),\s*([\d.]+)(s/it|it/s)"
)
# Periodic log dict, e.g. {'loss': 0.41, 'grad_norm': 3.7, 'learning_rate': 1.2e-05, 'epoch': 1.36}
LOSS_RE = re.compile(
    r"'loss':\s*'?([\d.]+)'?.*?'grad_norm':\s*'?([\d.eE+-]+|nan)'?"
    r".*?'learning_rate':\s*'?([\d.eE+-]+)'?.*?'epoch':\s*'?([\d.]+)'?"
)

SPARK = "▁▂▃▄▅▆▇█"


def read(path: str) -> str:
    with open(path, "r", errors="ignore") as f:
        return f.read()


def last_progress(text: str):
    # tqdm overwrites with \r; treat \r as line breaks and scan for the last match.
    last = None
    for line in text.replace("\r", "\n").splitlines():
        m = PROG_RE.search(line)
        if m:
            last = m
    return last


def all_losses(text: str):
    out = []
    for m in LOSS_RE.finditer(text.replace("\r", "\n")):
        loss, gnorm, lr, epoch = m.groups()
        out.append((float(loss), gnorm, lr, float(epoch)))
    return out


def sparkline(values) -> str:
    vals = [v for v in values if v == v]  # drop NaN
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    return "".join(SPARK[min(len(SPARK) - 1, int((v - lo) / rng * (len(SPARK) - 1)))]
                    for v in vals[-48:])


def bar(frac: float, width: int = 34) -> str:
    fill = int(frac * width)
    return "█" * fill + "░" * (width - fill)


def render(path: str) -> str:
    text = read(path)
    p = last_progress(text)
    losses = all_losses(text)
    lines = []
    lines.append("\033[1mTEKIMAX — training monitor\033[0m")
    lines.append(f"log: {path}")
    lines.append("")
    if p:
        step, total, elapsed, remaining, rate, unit = p.groups()
        step_i, total_i = int(step), int(total)
        frac = step_i / total_i if total_i else 0
        lines.append(f"[{bar(frac)}] {frac*100:5.1f}%   step {step_i}/{total_i}")
        lines.append(f"elapsed {elapsed} · eta {remaining} · {rate} {unit}")
    else:
        lines.append("[" + bar(0) + "]   waiting for first step…")
        lines.append("(loading model / tokenizing dataset)")
    lines.append("")
    if losses:
        loss, gnorm, lr, epoch = losses[-1]
        loss_vals = [l[0] for l in losses]
        lines.append(f"loss {loss:.4f}   grad_norm {gnorm}   lr {lr}   epoch {epoch:.2f}")
        lines.append(f"loss {sparkline(loss_vals)}")
        finite = [v for v in loss_vals if v == v]
        if finite:
            lines.append(f"     min {min(finite):.3f}  max {max(finite):.3f}  "
                         f"first {finite[0]:.3f}  ({len(loss_vals)} logged)")
        if any(g == "nan" for *_, g in [(0, l[1]) for l in losses]):
            lines.append("\033[31m⚠ NaN grad_norm seen — training may have diverged\033[0m")
    else:
        lines.append("loss: (no log lines yet — run training with PYTHONUNBUFFERED=1)")
    if "Saved model" in text:
        lines.append("")
        lines.append("\033[32m✓ training complete — model saved\033[0m")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Real-time training monitor.")
    ap.add_argument("logfile", help="Path to the training log file to watch.")
    ap.add_argument("--interval", type=float, default=2.0, help="Refresh seconds.")
    args = ap.parse_args()

    try:
        while True:
            frame = render(args.logfile)
            sys.stdout.write("\033[2J\033[H" + frame + "\n")
            sys.stdout.flush()
            if "Saved model" in read(args.logfile):
                break
            time.sleep(args.interval)
    except FileNotFoundError:
        print(f"Log file not found: {args.logfile}")
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
