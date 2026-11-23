#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} == policy || ${1:-} == tokenizer ]]; then
  kind=$1
  shift
  exec bash "$root/scripts/train_$kind.sh" "$@"
fi

usage() {
  echo "Usage: bash train.sh <bench> [<ckpt_name> [<action_type> [<seed> [<gpu_ids>]]]] [key=value ...]" >&2
  echo "       bash train.sh <bench> <ckpt_name> <env_cfg> <action_type> [<seed> [<gpu_ids>]] [key=value ...]" >&2
  echo "       bash train.sh <tokenizer|policy> [<config>] [key=value ...]" >&2
  exit 2
}
case "${1:-}" in
  RoboTwin|robotwin) bench=RoboTwin ;;
  EmbodiBench|embodibench) bench=EmbodiBench ;;
  *) usage ;;
esac
shift
ckpt=vla_robomorpher
action=joint
seed=0
gpu_ids=0
if (( $# )) && [[ $1 != *=* ]]; then ckpt=$1; shift; fi
# In the XPolicyLab form, the environment precedes the action; all
# embodiments share a dataset and training run.
if (( $# >= 2 )) && [[ $2 == joint ]]; then
  [[ -n $1 && $1 != -* && $1 != *=* ]] || usage
  shift
fi
if (( $# )) && [[ $1 != *=* ]]; then action=$1; shift; fi
if (( $# )) && [[ $1 != *=* ]]; then seed=$1; shift; fi
if (( $# )) && [[ $1 != *=* ]]; then gpu_ids=$1; shift; fi
[[ $ckpt =~ ^[[:alnum:]_.-]+$ && $action == joint ]] || usage
[[ $seed =~ ^[0-9]+$ && $gpu_ids =~ ^[0-9]+(,[0-9]+)*$ ]] || usage
for arg in "$@"; do [[ $arg == *=* ]] || usage; done

setting="$bench-$ckpt-$action"
IFS=, read -ra gpus <<< "$gpu_ids"
export CUDA_VISIBLE_DEVICES="$gpu_ids" NUM_GPUS="${#gpus[@]}"
exec bash "$root/scripts/train_policy.sh" "${ROBOMORPHER_TRAIN_CONFIG:-vla_robomorpher}" \
  "data.root=${ROBOMORPHER_DATA_ROOT:-$root/data/$setting}" \
  "trainer.output_dir=$root/checkpoints/$setting-$seed" "trainer.seed=$seed" "data.seed=$seed" "$@"
