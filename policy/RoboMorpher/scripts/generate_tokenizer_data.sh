#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
exec "${TRAIN_PYTHON:-python3}" -m robomorpher.utils.data.generate_tokenizer_data "$@"
