#!/usr/bin/env bash
# Install and check tokenizer and VLA dependencies with TRAIN_PYTHON.
set -euo pipefail
export USE_TF=0
export USE_FLAX=0
ADAPTER_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
MORPHGEN_DIR="$ADAPTER_DIR/robomorpher/third_party/MorphGen"
cd "$ADAPTER_DIR"
export PYTHONPATH="$ADAPTER_DIR${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${TRAIN_PYTHON:-python3}"
case "${1:-}" in
  ""|--install) echo "[install] Installing the RoboMorpher XPolicyLab adapter with $PYTHON." ;;
  --check) echo "[install] Checking the RoboMorpher XPolicyLab adapter with $PYTHON." ;;
  *) echo "[install] Usage: bash install.sh [--check|--install]" >&2; exit 2 ;;
esac

if [[ ! -e "$MORPHGEN_DIR" && ! -L "$MORPHGEN_DIR" && "${1:-}" != "--check" ]]; then
  echo "[install] Cloning robomorpher/MorphGen into $MORPHGEN_DIR."
  git clone https://github.com/robomorpher/MorphGen.git "$MORPHGEN_DIR"
fi
echo "[install] Checking the MorphGen checkout and tokenizer data generator files in $MORPHGEN_DIR."
if [[ ! -e "$MORPHGEN_DIR/.git" || ! -f "$MORPHGEN_DIR/utils/single_urdf_generator.py" ||
      ! -f "$MORPHGEN_DIR/utils/random_augment_without_collision.py" ]]; then
  echo "[install] Expected a MorphGen Git checkout at $MORPHGEN_DIR." >&2
  echo "[install] If the path is occupied, move it aside first. Then run bash install.sh to clone MorphGen." >&2
  exit 1
fi

if [[ "${1:-}" != "--check" ]]; then
  echo "[install] Checking for PyTorch."
  if "$PYTHON" -c "import torch" >/dev/null 2>&1; then
    echo "[install] Reusing the installed PyTorch build."
  else
    echo "[install] Installing torch==2.8.0+cu128 and torchvision==0.23.0+cu128."
    "$PYTHON" -m pip install "torch==2.8.0+cu128" "torchvision==0.23.0+cu128" \
      --index-url https://download.pytorch.org/whl/cu128
  fi
  echo "[install] Recording installed PyTorch, torchvision, torchaudio and Triton versions as pip constraints."
  REPORT="$(mktemp)"
  REQ="$(mktemp)"
  CONSTRAINT="$(mktemp)"
  trap 'rm -f "$REPORT" "$REQ" "$CONSTRAINT"' EXIT
  "$PYTHON" - "$CONSTRAINT" <<'PY'
import importlib.metadata as md
import sys
lines = []
for name in ("torch", "torchvision", "torchaudio", "triton"):
    try:
        lines.append(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        pass
open(sys.argv[1], "w").write("\n".join(lines) + ("\n" if lines else ""))
PY
  echo "[install] Preparing the core requirements. LeRobot and FlashAttention install separately."
  grep -v -E '^(lerobot==|flash-attn==|#|[[:space:]]*$)' robomorpher/requirements.txt > "$REQ"
  echo "[install] Resolving core dependencies with a pip dry run."
  "$PYTHON" -m pip install --dry-run --report "$REPORT" -c "$CONSTRAINT" -r "$REQ"
  echo "[install] Checking the pip plan for changes to the PyTorch and CUDA packages."
  "$PYTHON" - "$REPORT" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
protected = []
for item in plan.get("install", []):
    name = item["metadata"]["name"].lower().replace("_", "-")
    if name in {"torch", "torchvision", "torchaudio", "triton"} or name.startswith(("nvidia-", "cuda-")):
        protected.append(name)
if protected:
    raise SystemExit("[install] The dependency plan would change these PyTorch/CUDA packages: " + ", ".join(protected))
PY
  echo "[install] Installing core dependencies with the recorded constraints."
  "$PYTHON" -m pip install -c "$CONSTRAINT" -r "$REQ"
  echo "[install] Installing lerobot==0.4.4 with dependency resolution disabled."
  "$PYTHON" -m pip install --no-deps lerobot==0.4.4
  echo "[install] Installing flash-attn==2.7.4.post1 against the active PyTorch build."
  "$PYTHON" -m pip install flash-attn==2.7.4.post1 --no-build-isolation
fi

echo "[install] Checking shared tokenizer and VLA imports"
"$PYTHON" - <<'PY'
import importlib.metadata as md
from pathlib import Path
from robomorpher.model.tokenizer import build_tokenizer_from_config
from robomorpher.model.architectures.qwen_discrete_forcing import QwenDiscreteForcing
from robomorpher.model.architectures.qwen_pi_v3 import QwenPI_v3
from packaging.requirements import Requirement
print("[install] Checking installed project package versions", flush=True)
for line in Path("robomorpher/requirements.txt").read_text().splitlines():
    if line.strip() and not line.startswith("#"):
        requirement = Requirement(line)
        version = md.version(requirement.name)
        if version not in requirement.specifier:
            raise SystemExit(f"[install] {requirement}: installed {version}")
print("[install] Package versions:", {name:md.version(name) for name in ["torch","torchvision","transformers","diffusers","vector-quantize-pytorch"]})
PY
if [[ "${1:-}" == "--check" ]]; then
  echo "[install] Environment check complete"
else
  echo "[install] Installation and environment check complete"
fi
