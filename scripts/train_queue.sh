#!/usr/bin/env bash
# Run several ladder jobs back to back on one GPU, detached.
#
#   scripts/train_queue.sh <gpu> <stage:tier> [<stage:tier> ...]
#   scripts/train_queue.sh 1 pretrain:45m midtrain:45m posttrain:45m evalsuite:45m
#
# One GPU runs one job at a time; the next starts only when the previous exits cleanly, so a
# stage never trains from a checkpoint the stage before it failed to finish. Each job resumes its
# own latest.pt (runtime.resume), which makes re-running the whole queue safe.
#
# `evalsuite:<tier>` is not a training stage - it scores the tier's posttrain checkpoint on the
# ten public benchmarks and writes results/evalsuite-full/la-<tier>.json. Putting it in the same
# queue is what keeps a finished spine from sitting unscored.
set -euo pipefail

gpu="${1:?gpu index}"
shift
[ "$#" -gt 0 ] || { echo "give at least one <stage>:<tier>" >&2; exit 1; }

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${LOCALAGENT_PYTHON:-$root/.venv/bin/python}"
mkdir -p "$root/explog"
queue_log="$root/explog/queue-gpu$gpu.log"

# Two queues that overlap on a tier write the same run directory and corrupt each other's
# checkpoint and curve. This happened: a chain armed twice fired both copies in the same second.
lock_dir="$root/runs/.queue-lock"
mkdir -p "$root/runs"

run_queue() {
  for job in "$@"; do
    stage="${job%%:*}"
    tier="${job##*:}"
    lock="$lock_dir-$stage-la-$tier"
    if ! mkdir "$lock" 2>/dev/null; then
      echo "[$(date +%H:%M:%S)] ALREADY RUNNING $stage la-$tier (lock $lock) - stopping queue" >&2
      return 1
    fi
    # shellcheck disable=SC2064
    trap "rmdir '$lock' 2>/dev/null" RETURN
    echo "[$(date +%H:%M:%S)] START $stage la-$tier"

    if [ "$stage" = "evalsuite" ]; then
      checkpoint="$root/runs/posttrain-la-$tier/latest.pt"
      if [ ! -f "$checkpoint" ]; then
        echo "[$(date +%H:%M:%S)] NO CHECKPOINT $checkpoint - stopping queue" >&2
        return 1
      fi
      mkdir -p "$root/results/evalsuite-full"
      receipt="$root/results/evalsuite-full/la-$tier.json"
      # Idempotent, like the training stages: a receipt newer than the checkpoint it scores is
      # the finished job. Without this every relaunch re-scored every earlier tier on its way
      # to the first real job - two hours of eval per walk, three walks in one night.
      if [ -s "$receipt" ] && [ "$receipt" -nt "$checkpoint" ]; then
        echo "[$(date +%H:%M:%S)] DONE evalsuite la-$tier (receipt newer than checkpoint)"
        rmdir "$lock" 2>/dev/null
        continue
      fi
      if ! CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$root/src" PYTHONUNBUFFERED=1 \
          "$python" -m openlocalagent.eval.suite \
            --model "catalog:$checkpoint" \
            --out "$root/results/evalsuite-full/la-$tier.json" \
            --rows 999999 --device cuda --batch-size 32 \
          >>"$root/explog/evalsuite-la-$tier.log" 2>&1; then
        echo "[$(date +%H:%M:%S)] FAILED evalsuite la-$tier - stopping queue" >&2
        return 1
      fi
      echo "[$(date +%H:%M:%S)] DONE evalsuite la-$tier"
      rmdir "$lock" 2>/dev/null
      continue
    fi

    config="$root/configs/$stage/la-$tier.yaml"
    if [ ! -f "$config" ]; then
      echo "[$(date +%H:%M:%S)] MISSING CONFIG $config - stopping queue" >&2
      return 1
    fi
    if ! CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$root/src" PYTHONUNBUFFERED=1 \
        "$python" -m openlocalagent.cli train "$stage" "$config" \
        >>"$root/explog/$stage-la-$tier.log" 2>&1; then
      echo "[$(date +%H:%M:%S)] FAILED $stage la-$tier - stopping queue" >&2
      return 1
    fi
    echo "[$(date +%H:%M:%S)] DONE $stage la-$tier"
    rmdir "$lock" 2>/dev/null
  done
  echo "[$(date +%H:%M:%S)] QUEUE COMPLETE"
}

cd "$root"
# The queue body runs in a detached shell, so the paths it reads have to travel with it.
export root python gpu lock_dir
export -f run_queue
nohup bash -c "run_queue $*" >>"$queue_log" 2>&1 &
echo "queue on gpu $gpu: $*"
echo "  log $queue_log"
