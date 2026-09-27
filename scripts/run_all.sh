#!/bin/bash
# Master orchestrator: drive the full improvement roadmap to completion, SEQUENTIALLY (no overlap,
# so the box never OOMs). Waits for the in-flight dispatch run, then chains scenario+extended
# training and arg audits. Each stage logs to this script's stdout (/tmp/run_all.log).
set -u
cd /home/user/LocalAgent
say(){ echo "=== [orchestrator $(date +%H:%M:%S)] $* ==="; }

say "waiting for dispatch fine-tune to finish (runs/tiny-30m-dispatch-long.pt + LONG_DONE)"
until [ -f runs/tiny-30m-dispatch-long.pt ] && grep -q LONG_DONE /tmp/long_resume.log 2>/dev/null; do
  sleep 30
done
say "dispatch done. baseline argument audit:"
python -u scripts/audit_args.py --ckpt runs/tiny-30m-dispatch-long.pt 2>&1

say "scenario + extended fine-tune (3000 steps, folds clarify/abstain/parallel + episodes)"
python -u scripts/train_scenarios.py --init runs/tiny-30m-dispatch-long.pt --steps 3000 --seg 400 2>&1

say "final argument audit on the scenario model:"
python -u scripts/audit_args.py --ckpt runs/tiny-30m-scenarios.pt 2>&1

say "ALL TRAINING STAGES COMPLETE"
echo "ALL_DONE"
