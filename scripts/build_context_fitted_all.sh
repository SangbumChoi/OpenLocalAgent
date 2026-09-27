#!/usr/bin/env bash
# Refit every midtrain and posttrain conversation source to the context budget under the CURRENT
# tokenizer, writing tokenizer-tagged outputs so a refit never overwrites the set another
# tokenizer's models were trained on.
#
# Why this exists: `openai_full_catalog_v1` refuses to truncate, so one oversized row aborts a
# stage. The fit2048-* files were fitted against the tokenizer that was later overwritten; the same
# conversations tokenize longer under the replacement (2,150 > 2,048 on the first midtrain row),
# and the first spine of the new generation died on exactly that. Fitting is a property of a
# (corpus, tokenizer) pair, so the output name carries the tokenizer's sha256 prefix.
#
# Usage:  bash scripts/build_context_fitted_all.sh            # all sources
#         ONLY="toucan,xlam" bash scripts/build_context_fitted_all.sh
# Env:    TOKENIZER (default data/tokenizer-h100-16k.json), MAX_TOKENS (2048), DATA_ROOT
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${DATA_ROOT:-$ROOT/data}"
TOKENIZER="${TOKENIZER:-$DATA/tokenizer-h100-16k.json}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
PY="$ROOT/.venv/bin/python"
TAG="$(shasum -a 256 "$TOKENIZER" 2>/dev/null || sha256sum "$TOKENIZER")"; TAG="${TAG:0:8}"
cd "$ROOT"

# name=original. Names are the ones the configs reference; originals are what the previous
# fit2048-* files were built from (verified by byte size on the box, 2026-09-02).
SOURCES=(
  toucan=public/toucan-train.jsonl
  r0b0t_tools=public/r0b0t-tools-train.jsonl
  device=public/device-train.jsonl
  distill_reasoning=distill2/train-clean.jsonl
  free_episodes=agentic-free/train.jsonl
  merged-v2=merged-v2/train.jsonl
  androidcontrol=public/androidcontrol-train.jsonl
  agentnet=public/agentnet-train-catalog.jsonl
  xlam=public/xlam-train.jsonl
  mobileactions=public/mobileactions-train.jsonl
  toolace=public/toolace-train.jsonl
  toolbench=public/toolbench-train.jsonl
  mind2web=public/mind2web-train.jsonl
  toolsandbox=public/toolsb-synth-train.jsonl
  # Open agent midtrain sets (scripts/normalize_midtrain_open.py); not benchmark-derived.
  orca_tool_use=public/orca_tool_use.jsonl
  orca_webagent_flow=public/orca_webagent_flow.jsonl
  orca_code=public/orca_code.jsonl
  orca_follow_up=public/orca_follow_up.jsonl
  orca_task=public/orca_task.jsonl
  codeact=public/codeact.jsonl
)

echo "[$(date +%H:%M:%S)] tokenizer $TOKENIZER (sha256 $TAG...) budget $MAX_TOKENS"
for entry in "${SOURCES[@]}"; do
  name="${entry%%=*}"; rel="${entry#*=}"
  if [ -n "${ONLY:-}" ] && ! grep -qE "(^|,)$name(,|$)" <<<"$ONLY"; then continue; fi
  src="$DATA/$rel"; out="$DATA/public/fit${MAX_TOKENS}-$TAG-$name.jsonl"
  # A missing original must stop the run: a config pointing at an output that was never written
  # fails a stage later with a far less informative error.
  [ -s "$src" ] || { echo "MISSING original for $name: $src" >&2; exit 1; }
  if [ -s "$out" ]; then echo "[$(date +%H:%M:%S)] have $(basename "$out")"; continue; fi
  echo "[$(date +%H:%M:%S)] fitting $name"
  PYTHONPATH="$ROOT/src" "$PY" scripts/build_context_fitted.py \
    --in "$src" --out "$out.part" --max-tokens "$MAX_TOKENS" --tokenizer "$TOKENIZER"
  mv "$out.part" "$out"
done
echo "[$(date +%H:%M:%S)] CONTEXT_FITTED_DONE tag=$TAG"
