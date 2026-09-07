#!/usr/bin/env bash
# Retire a run and its benchmark receipt together, under one label.
#
#   scripts/archive_run.sh <label> <stage:tier> [<stage:tier> ...]
#   scripts/archive_run.sh recipe-v1 midtrain:45m posttrain:45m
#
# A published number needs the weights that produced it. Renaming the receipt and deleting the run
# leaves a score nobody can re-score - which is exactly what happened to la-45m's 25.61: the
# receipt survived, the checkpoint did not, so the AndroidControl clean-subset number for it can
# only be recovered by retraining from the config in git history.
#
# This moves runs/<stage>-la-<tier> to results/runs/archive/<label>/ and renames any matching
# results/evalsuite-full/la-<tier>.json to la-<tier>-<label>.json, so the two stay together.
set -euo pipefail

label="${1:?label, e.g. recipe-v1}"
shift
[ "$#" -gt 0 ] || { echo "give at least one <stage>:<tier>" >&2; exit 1; }

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
archive="$root/runs/archive/$label"
mkdir -p "$archive"

for job in "$@"; do
  stage="${job%%:*}"
  tier="${job##*:}"
  run="$root/runs/$stage-la-$tier"
  if [ -d "$run" ]; then
    mv "$run" "$archive/$stage-la-$tier"
    echo "archived $stage-la-$tier -> results/runs/archive/$label/"
  else
    echo "no run directory for $stage la-$tier - skipping" >&2
  fi

  receipt="$root/results/evalsuite-full/la-$tier.json"
  if [ -f "$receipt" ]; then
    mv "$receipt" "$root/results/evalsuite-full/la-$tier-$label.json"
    echo "archived receipt   -> la-$tier-$label.json"
  fi
done

echo "archive: $archive"
du -sh "$archive" 2>/dev/null || true
