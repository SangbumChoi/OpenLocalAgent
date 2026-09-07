#!/usr/bin/env bash
# Extend the open pretraining corpus so la-700m can be trained at a sane tokens-per-parameter.
#
# pt-big holds 460.7M tokens; a 700M model wants ~15B. FineWeb-Edu-dedup ships 234 parquet files
# of ~0.94B tokens each, and the Nexus mirror serves a whole file at ~300 MB/s (a range read is
# ~61 KB/s, so never range-read here). Eight files is ~7.5B tokens, which is two epochs to a
# Chinchilla-matched budget.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE=http://nexus.tossbank.bz/repository/huggingface-proxy
REV=3ba9d605774198c5868892d7a8deda78031a781f
RAW="${FINEWEB_RAW:-$ROOT/data/raw/fineweb-xl}"
FILES="${FILES:-8}"

mkdir -p "$RAW"
cd "$ROOT"

for i in $(seq 0 $((FILES - 1))); do
  n=$(printf "%05d" "$i")
  jsonl="$RAW/fineweb-$n.jsonl"
  if [ -s "$jsonl" ]; then echo "[$(date +%H:%M:%S)] have $jsonl"; continue; fi
  pq="$RAW/train-$n.parquet"
  echo "[$(date +%H:%M:%S)] GET train-$n-of-00234.parquet"
  curl -sS -o "$pq" -m 3600 \
    "$BASE/datasets/HuggingFaceTB/smollm-corpus/resolve/$REV/fineweb-edu-dedup/train-$n-of-00234.parquet"
  echo "[$(date +%H:%M:%S)] convert $n -> jsonl"
  "$ROOT/.venv/bin/python" - "$pq" "$jsonl" <<'PY'
import json, sys
import pyarrow.parquet as pq
source, target = sys.argv[1], sys.argv[2]
written = 0
with open(target, "w", encoding="utf-8") as out:
    parquet = pq.ParquetFile(source)
    for batch in parquet.iter_batches(batch_size=2000, columns=["text"]):
        for text in batch.column("text").to_pylist():
            if text:
                out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                written += 1
print(f"rows={written}", flush=True)
PY
  rm -f "$pq"                       # the JSONL is what the packer reads; parquet is 2.4GB each
  echo "[$(date +%H:%M:%S)] done $n"
done

echo "[$(date +%H:%M:%S)] packing shards"
"$ROOT/.venv/bin/python" scripts/build_corpus.py "$RAW" \
  --out data/shards/pt-xl \
  --reuse-tokenizer \
  --seq-len 2048 --rows-per-shard 4096 --val-fraction 0.01 \
  --tokenizer bpe --tokenizer-path data/tokenizer-h100-16k.json --vocab-size 16384 \
  --no-near-dedup
echo "[$(date +%H:%M:%S)] PT_XL_DONE"
