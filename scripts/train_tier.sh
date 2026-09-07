#!/usr/bin/env bash
# Run one ladder tier's stage on one GPU, detached, with its log and curve under results/runs/.
#
#   scripts/train_tier.sh <gpu> <stage> <tier>       e.g. scripts/train_tier.sh 0 pretrain 93m
#
# Idempotent: the stage configs set runtime.resume, which continues an existing latest.pt and
# starts fresh when there is none, so re-running after a kill picks up where it stopped and
# appends to the same curve.jsonl. (The CLI's --resume flag is the strict form and fails when no
# checkpoint exists yet, so it is deliberately not passed here.)
set -euo pipefail

gpu="${1:?gpu index}"
stage="${2:?stage: pretrain|midtrain|posttrain}"
tier="${3:?tier: 10m|45m|93m|700m}"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="$root/configs/$stage/la-$tier.yaml"
[ -f "$config" ] || { echo "no config: $config" >&2; exit 1; }

python="${LOCALAGENT_PYTHON:-$root/.venv/bin/python}"
[ -x "$python" ] || { echo "no interpreter: $python (set LOCALAGENT_PYTHON)" >&2; exit 1; }

log_dir="$root/explog"
mkdir -p "$log_dir"
log="$log_dir/$stage-la-$tier.log"

cd "$root"
CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$root/src" PYTHONUNBUFFERED=1 \
  nohup "$python" -m openlocalagent.cli train "$stage" "$config" >>"$log" 2>&1 &

echo "started $stage la-$tier on gpu $gpu (pid $!)"
echo "  log   $log"
echo "  curve $root/runs/$stage-la-$tier/curve.jsonl"
