#!/usr/bin/env bash
# Download Qwen3-VL-4B-Instruct, the base VLM for the policy.
# Destination matches framework.qwenvl.base_vlm:
#   ./robomorpher/pretrained/vlm/Qwen3-VL-4B-Instruct
#
# Usage:
#   bash scripts/download_qwen3_vl_4b_instruct.sh
#   HF_ENDPOINT=https://hf-mirror.com bash scripts/download_qwen3_vl_4b_instruct.sh
#
# huggingface_hub is installed by install.sh. Auth is only needed if the repo is gated.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON="${TRAIN_PYTHON:-python3}"

REPO_ID="${HF_REPO_ID:-Qwen/Qwen3-VL-4B-Instruct}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/robomorpher/pretrained/vlm/Qwen3-VL-4B-Instruct}"

mkdir -p "$(dirname "${OUT_DIR}")"

echo "[download_qwen3_vl_4b_instruct] repo=${REPO_ID}"
echo "[download_qwen3_vl_4b_instruct] out=${OUT_DIR}"
if [[ -n "${HF_ENDPOINT:-}" ]]; then
  echo "[download_qwen3_vl_4b_instruct] HF_ENDPOINT=${HF_ENDPOINT}"
fi

if command -v hf >/dev/null 2>&1; then
  hf download "${REPO_ID}" --local-dir "${OUT_DIR}"
elif "$PYTHON" -c "import huggingface_hub" >/dev/null 2>&1; then
  "$PYTHON" - "$REPO_ID" "$OUT_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])
PY
else
  echo "[download_qwen3_vl_4b_instruct] ERROR: need the hf CLI or python package huggingface_hub" >&2
  echo "[download_qwen3_vl_4b_instruct]   bash install.sh" >&2
  exit 1
fi

if [[ ! -f "${OUT_DIR}/config.json" ]]; then
  echo "[download_qwen3_vl_4b_instruct] ERROR: ${OUT_DIR}/config.json missing after download." >&2
  exit 1
fi

echo "[download_qwen3_vl_4b_instruct] done -> ${OUT_DIR}"
