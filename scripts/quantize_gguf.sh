#!/usr/bin/env bash
# quantize_gguf.sh — Convert your fine-tuned model to GGUF and quantize to
# Q4_K_M so it runs fast on Apple Silicon (Metal) via llama.cpp / Ollama / LM Studio.
#
# Usage:
#   bash scripts/quantize_gguf.sh ./granite-ml-coder granite-ml-coder
#
# Args:
#   $1 = path to your trained model directory (default: ./granite-ml-coder)
#   $2 = output basename (default: granite-ml-coder)
set -euo pipefail

MODEL_DIR="${1:-./granite-ml-coder}"
OUT_NAME="${2:-granite-ml-coder}"
WORK="${WORK:-./.llama_cpp}"

echo "==> Model dir: $MODEL_DIR"

# 1. Get llama.cpp (conversion + quantization tooling).
if [ ! -d "$WORK" ]; then
  echo "==> Cloning llama.cpp into $WORK"
  git clone --depth 1 https://github.com/ggerganov/llama.cpp "$WORK"
fi

# 2. Build the quantize binary (Metal is auto-enabled on macOS).
echo "==> Building llama.cpp"
cmake -S "$WORK" -B "$WORK/build" >/dev/null
cmake --build "$WORK/build" --config Release -j --target llama-quantize >/dev/null

# 3. Install conversion deps in an isolated venv.
echo "==> Installing conversion requirements"
uv venv "$WORK/.venv" --python 3.12 >/dev/null
uv pip install --python "$WORK/.venv/bin/python" -r "$WORK/requirements.txt" >/dev/null

# 4. Convert HF model -> GGUF (fp16), then quantize -> Q4_K_M.
echo "==> Converting to fp16 GGUF"
"$WORK/.venv/bin/python" "$WORK/convert_hf_to_gguf.py" "$MODEL_DIR" \
  --outfile "${OUT_NAME}-f16.gguf" --outtype f16

echo "==> Quantizing to Q4_K_M"
"$WORK/build/bin/llama-quantize" "${OUT_NAME}-f16.gguf" "${OUT_NAME}-Q4_K_M.gguf" Q4_K_M

echo ""
echo "Done:"
echo "  ${OUT_NAME}-f16.gguf      (full precision GGUF)"
echo "  ${OUT_NAME}-Q4_K_M.gguf   (4-bit, recommended for local use)"
echo ""
echo "Run it:  $WORK/build/bin/llama-cli -m ${OUT_NAME}-Q4_K_M.gguf -p \"Write a sklearn pipeline\""
echo "Upload:  uv run huggingface-cli upload Tekimax/granite-ml-coder-GGUF ${OUT_NAME}-Q4_K_M.gguf"
