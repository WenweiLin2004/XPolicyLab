#!/usr/bin/env bash
set -euo pipefail
export USE_TF=0
export USE_FLAX=0
# torchrun workers start new sessions and can lose their controlling terminal.
# Capture the launcher's actual stdin TTY before output redirection propagates.
if [[ -z "${EAT_CONSOLE_TTY:-}" ]]; then
  if training_console_tty="$(tty 2>/dev/null)"; then
    export EAT_CONSOLE_TTY="$training_console_tty"
  fi
fi
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
CONFIG="vla_robomorpher"
if [[ $# -ge 1 && $1 != *=* ]]; then CONFIG=$1; shift; fi
if [[ -f "$CONFIG" ]]; then CONFIG_PATH="$CONFIG"; else CONFIG_PATH="robomorpher/config/policy/${CONFIG%.yaml}.yaml"; fi
PYTHON="${TRAIN_PYTHON:-${POLICY_PYTHON:-python3}}"
NUM_GPUS="${NUM_GPUS:-8}"
MAIN_PORT="${MAIN_PORT:-29517}"
if [[ ! -f "$CONFIG_PATH" ]]; then echo "Missing config: $CONFIG_PATH" >&2; exit 1; fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then echo "Python not found: $PYTHON" >&2; exit 1; fi
if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then echo "NUM_GPUS must be a positive integer" >&2; exit 1; fi
if ! [[ "$MAIN_PORT" =~ ^[1-9][0-9]*$ ]] || ((MAIN_PORT > 65535)); then
  echo "MAIN_PORT must be an integer in [1, 65535]" >&2; exit 1
fi
echo "[INFO] [policy:start] config=$CONFIG_PATH gpus=$NUM_GPUS python=$PYTHON"
if ((NUM_GPUS > 1)); then
  exec "$PYTHON" -m torch.distributed.run --nproc_per_node="$NUM_GPUS" --master_port="$MAIN_PORT" -m robomorpher.utils.training.train_policy --config "$CONFIG_PATH" trainer.distributed=fsdp "$@"
else
  exec "$PYTHON" -m robomorpher.utils.training.train_policy --config "$CONFIG_PATH" trainer.distributed=single "$@"
fi
