#!/usr/bin/env bash
# Build a ~4B-token open pretraining mixture, replacing the single-source 460.7M pt-big.
#
# pt-big is one web source plus two small ones, and every tier from la-300m up repeats it 6.8-9.1
# times. This mixture is what the corpus-scale experiment needs: more unique tokens AND more
# domains, so knowledge density and structured text stop being whatever FineWeb happened to carry.
#
# Reachability was measured against the Nexus HuggingFace mirror, not assumed. Serving (206):
# HuggingFaceFW/fineweb-edu, wikimedia/wikipedia, allenai/c4, HuggingFaceTB/{finemath,smollm-corpus}.
# Refusing (404, gated upstream): bigcode/the-stack-*, bigcode/starcoderdata, codeparrot/github-code,
# tiiuae/falcon-refinedweb, manu/project_gutenberg, allenai/dolma. Do not re-add those without
# re-probing - a 404 here is an access decision upstream, not a typo.
#
# Whole-file GET on this mirror runs at ~302 MB/s; a range read of the same file runs at ~61 KB/s.
# Never range-read it. Its /api/** is 400, so filenames come from the release pattern.
#
# --reuse-tokenizer is not optional and its absence is silent. Without it build_corpus.py RETRAINS
# the BPE and OVERWRITES --tokenizer-path, which is the shared data/tokenizer-h100-16k.json every
# existing checkpoint was trained against. That happened: packing this corpus rewrote the file at
# 00:41 mid-run, the sha moved from 236945b5 to 2f49a644, and the next stage to chain off any
# earlier checkpoint died on "init_from checkpoint tokenizer lineage does not match configured
# tokenizer" - the guard working exactly as intended, on damage already done. A new corpus must
# reuse the ladder's tokenizer anyway, or its shards are not comparable with anything already run.
#
# Usage:  bash scripts/fetch_pretrain_4b.sh
# Env:    PRETRAIN_RAW (staging dir), OUT_SHARDS (shard dir), FW/WIKI/C4/MATH/COSMO file counts
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${HF_MIRROR:-http://nexus.tossbank.bz/repository/huggingface-proxy}"
RAW="${PRETRAIN_RAW:-$ROOT/data/raw/pt-4b}"
OUT="${OUT_SHARDS:-$ROOT/data/shards/pt-4b}"
PY="$ROOT/.venv/bin/python"

# Roughly 4B tokens. Per-file token yields are release constants, measured once and written down
# here so the mixture is auditable rather than emergent.
FW="${FW:-3}"        # fineweb-edu-dedup   ~0.94B tok/file  -> 2.8B  web (educational)
C4N="${C4N:-4}"      # c4 en               ~0.15B tok/file  -> 0.6B  web (different filter)
WIKI="${WIKI:-4}"    # wikipedia 20231101  ~0.11B tok/file  -> 0.4B  knowledge density
MATH="${MATH:-1}"    # finemath-3plus      ~0.27B tok/file  -> 0.3B  reasoning
COSMO="${COSMO:-1}"  # cosmopedia-v2       ~0.27B tok/file  -> 0.3B  synthetic textbook

SMOL_REV=3ba9d605774198c5868892d7a8deda78031a781f

mkdir -p "$RAW"
cd "$ROOT"

# One download+convert step, resumable at file granularity: a non-empty JSONL is never refetched.
# The box stops every 3-4 hours, so anything not resumable here never finishes.
fetch_parquet() {
  local tag="$1" repo="$2" rev="$3" path="$4" jsonl="$RAW/$5" col="${6:-text}"
  if [ -s "$jsonl" ]; then echo "[$(date +%H:%M:%S)] have $(basename "$jsonl")"; return 0; fi
  local pq="$RAW/.staging-$tag.parquet"
  echo "[$(date +%H:%M:%S)] GET $tag $path"
  if ! curl -fsS -o "$pq" -m 3600 "$BASE/datasets/$repo/resolve/$rev/$path"; then
    echo "[$(date +%H:%M:%S)] MISS $tag $path - skipping" >&2
    rm -f "$pq"; return 0
  fi
  "$PY" - "$pq" "$jsonl" "$col" <<'PY'
import json, sys
import pyarrow.parquet as pq
source, target, column = sys.argv[1], sys.argv[2], sys.argv[3]
written = 0
# Write to a temp name and rename, so an interrupted convert never leaves a truncated JSONL that
# the `-s` check above would then treat as complete.
with open(target + ".part", "w", encoding="utf-8") as out:
    for batch in pq.ParquetFile(source).iter_batches(batch_size=2000, columns=[column]):
        for text in batch.column(column).to_pylist():
            if text:
                out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                written += 1
import os
os.replace(target + ".part", target)
print(f"rows={written}", flush=True)
PY
  rm -f "$pq"
}

for i in $(seq 0 $((FW - 1))); do
  n=$(printf "%05d" "$i")
  fetch_parquet "fw$n" HuggingFaceTB/smollm-corpus "$SMOL_REV" \
    "fineweb-edu-dedup/train-$n-of-00234.parquet" "fineweb-$n.jsonl"
done

for i in $(seq 0 $((COSMO - 1))); do
  n=$(printf "%05d" "$i")
  fetch_parquet "cos$n" HuggingFaceTB/smollm-corpus "$SMOL_REV" \
    "cosmopedia-v2/train-$n-of-00104.parquet" "cosmopedia-$n.jsonl"
done

for i in $(seq 0 $((WIKI - 1))); do
  n=$(printf "%05d" "$i")
  fetch_parquet "wiki$n" wikimedia/wikipedia main \
    "20231101.en/train-$n-of-00041.parquet" "wikipedia-$n.jsonl"
done

for i in $(seq 0 $((MATH - 1))); do
  n=$(printf "%05d" "$i")
  fetch_parquet "math$n" HuggingFaceTB/finemath main \
    "finemath-3plus/train-$n-of-00128.parquet" "finemath-$n.jsonl"
done

# C4 ships gzipped JSON lines rather than parquet, so it takes its own path.
for i in $(seq 0 $((C4N - 1))); do
  n=$(printf "%05d" "$i")
  jsonl="$RAW/c4-$n.jsonl"
  if [ -s "$jsonl" ]; then echo "[$(date +%H:%M:%S)] have c4-$n.jsonl"; continue; fi
  echo "[$(date +%H:%M:%S)] GET c4 $n"
  if curl -fsS -o "$RAW/.staging-c4.json.gz" -m 3600 \
      "$BASE/datasets/allenai/c4/resolve/main/en/c4-train.$n-of-01024.json.gz"; then
    gzip -dc "$RAW/.staging-c4.json.gz" \
      | "$PY" -c 'import sys,json
for line in sys.stdin:
    if line.strip():
        sys.stdout.write(json.dumps({"text": json.loads(line)["text"]}, ensure_ascii=False) + "\n")' \
      > "$jsonl.part" && mv "$jsonl.part" "$jsonl"
    rm -f "$RAW/.staging-c4.json.gz"
  else
    echo "[$(date +%H:%M:%S)] MISS c4 $n - skipping" >&2
  fi
done

echo "[$(date +%H:%M:%S)] staged:"
du -sh "$RAW"/*.jsonl 2>/dev/null | sed 's/^/    /'

# The staging SQLite must not live on the NFS home. Packing this corpus there ran for ten hours,
# reached a 30.5 GB staging file, and then sat in uninterruptible I/O wait at 9% CPU with the file
# not growing at all - SQLite's random writes over NFS do not scale, and this is the same wall the
# 7.5B pt-xl attempt hit. The pod has a local overlay with terabytes free; stage there.
# Trade-off worth knowing: local overlay is ephemeral, so the staging DB does not survive a pod
# restart. The shards do, and they are what training reads - only a derived-corpus build would
# need the staging DB back.
STAGING="${CORPUS_STAGING:-/var/tmp/ola-corpus-staging}"
mkdir -p "$STAGING"
echo "[$(date +%H:%M:%S)] staging db -> $STAGING (free: $(df -h "$STAGING" | awk 'NR==2{print $4}'))"

echo "[$(date +%H:%M:%S)] packing shards -> $OUT"
# PYTHONPATH is not optional: the venv's editable install resolves `openlocalagent` to an OLDER
# checkout at ~/openlocalagent that predates the data/ package split, so build_corpus.py dies on
# `No module named openlocalagent.data.decontam` while the trainers (which set this) run fine.
PYTHONPATH="$ROOT/src" "$PY" scripts/build_corpus.py "$RAW" \
  --out "$OUT" \
  --reuse-tokenizer \
  --staging-db "$STAGING/pt-4b.sqlite3" \
  --seq-len 2048 --rows-per-shard 4096 --val-fraction 0.01 \
  --tokenizer bpe --tokenizer-path data/tokenizer-h100-16k.json --vocab-size 16384 \
  --no-near-dedup
echo "[$(date +%H:%M:%S)] PT_4B_DONE"
