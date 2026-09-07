#!/usr/bin/env bash
# Fetch the upstream sources for the two suites that had no checked-in rebuild path.
#
# mcpatlas and mobileactions existed only as JSONL on the training box - no producer, in this tree
# or at the pre-refactor tag. A suite you cannot rebuild is a suite you cannot check, and
# mobileactions is one of la-93m's two strongest scores, so "it works on the box" was not good
# enough. Both upstreams are CC-BY-4.0 and ungated.
#
# The rebuild is verified, not assumed: the normalizers reproduce the exact row counts (495, 961)
# AND identical gold-answer multisets to the files that produced every published number.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
out="${1:-${EVAL_SUITE_RAW:-/tmp/suites}}"
mkdir -p "$out"

curl -sSL -m 300 -o "$out/mcpatlas.parquet" \
  "https://huggingface.co/datasets/ScaleAI/MCP-Atlas/resolve/8c563b55/MCP-Atlas.parquet"
curl -sSL -m 300 -o "$out/mobileactions.jsonl" \
  "https://huggingface.co/datasets/google/mobile-actions/resolve/e920309b/dataset.jsonl"

python scripts/normalize_mcpatlas.py \
  --in "$out/mcpatlas.parquet" --out data/public/mcpatlas-eval.jsonl
python scripts/normalize_mobileactions.py \
  --in "$out/mobileactions.jsonl" \
  --out-eval data/public/mobileactions-eval.jsonl \
  --out-train data/public/mobileactions-train.jsonl
