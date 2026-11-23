#!/usr/bin/env bash
set -euo pipefail
export USE_TF=0
export USE_FLAX=0
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
# The integration workspace can share the standalone tokenizer corpus.
xpl_root=${XPOLICYLAB_ROOT:-$(dirname "$(dirname "$ROOT_DIR")")}
if [[ -z "${ROBOMORPHER_TOKENIZER_ROOT:-}" && -f "$xpl_root/setup_policy_server.py" && -d "$xpl_root/../data/tokenizer" ]]; then
  export ROBOMORPHER_TOKENIZER_ROOT="$(cd -P -- "$xpl_root/../data/tokenizer" && pwd -P)"
fi
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
CONFIG="robomorpher"
if [[ $# -ge 1 && $1 != -* && $1 != *=* ]]; then CONFIG=$1; shift; fi
if [[ -f "$CONFIG" ]]; then CONFIG_PATH="$CONFIG"; else CONFIG_PATH="robomorpher/config/tokenizer/${CONFIG%.yaml}.yaml"; fi
PYTHON="${TRAIN_PYTHON:-python3}"
NUM_GPUS="${NUM_GPUS:-8}"
MAIN_PORT="${MAIN_PORT:-29501}"
if [[ ! -f "$CONFIG_PATH" ]]; then echo "Missing config: $CONFIG_PATH" >&2; exit 1; fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then echo "Python not found: $PYTHON" >&2; exit 1; fi
if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then echo "NUM_GPUS must be a positive integer" >&2; exit 1; fi
LAUNCH_ARGS=(--num_processes "$NUM_GPUS")
if [[ "$NUM_GPUS" -gt 1 ]]; then LAUNCH_ARGS+=(--multi_gpu); fi
echo "[INFO] [tokenizer:start] config=$CONFIG_PATH gpus=$NUM_GPUS python=$PYTHON"
exec "$PYTHON" -m accelerate.commands.launch "${LAUNCH_ARGS[@]}" --main_process_port "$MAIN_PORT" --module robomorpher.utils.training.train_tokenizer --config "$CONFIG_PATH" "$@"
