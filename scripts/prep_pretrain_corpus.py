#!/usr/bin/env python
"""Turn an open dataset into plain pretraining text, and account for what it actually costs.

A pretraining-corpus comparison is only meaningful with the budget written down, so this records
what the corpus contains (documents, characters, tokens) as well as what a run would consume. The
pretraining stage itself is held to a fixed step count elsewhere, which fixes tokens consumed and
leaves the corpus content as the only variable; the numbers here say how many epochs that implies.

  python scripts/prep_pretrain_corpus.py --source data/hf-campaign/ultrachat_200k \
      --out data/pretrain/ultrachat.txt
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

# Field names carrying the text, in the order we prefer them.
TEXT_FIELDS = ("text", "content", "completion", "response", "output", "answer", "body")
TURN_FIELDS = ("messages", "conversations", "conversation", "chosen", "rejected", "turns")


def text_from(record) -> list[str]:
    """Every natural-language span in a record, whatever schema it uses."""
    if isinstance(record, str):
        return [record]
    if isinstance(record, list):
        return [span for item in record for span in text_from(item)]
    if not isinstance(record, dict):
        return []
    spans: list[str] = []
    for field in TURN_FIELDS:
        if field in record:
            spans += text_from(record[field])
    for field in TEXT_FIELDS:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            spans.append(value.strip())
    if not spans:
        # Unknown schema: take every reasonably long string value rather than silently yielding
        # nothing, which would look like an empty corpus instead of an unparsed one.
        spans = [value.strip() for value in record.values()
                 if isinstance(value, str) and len(value.strip()) > 40]
    return spans


def read_records(path: Path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
            yield from batch.to_pylist()
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield line


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="directory of parquet/jsonl(.gz) files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="data/tokenizer-h100-16k.json")
    ap.add_argument("--max-documents", type=int, default=0)
    ap.add_argument("--min-chars", type=int, default=40)
    args = ap.parse_args()

    from localagent.model.tokenizer import load_tokenizer

    tok = load_tokenizer("bpe", args.tokenizer)
    files = sorted(path for path in Path(args.source).rglob("*")
                   if path.suffix in (".parquet", ".jsonl", ".gz", ".json")
                   and ".cache" not in path.parts)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    documents = characters = tokens = 0
    with out.open("w", encoding="utf-8") as handle:
        for path in files:
            for record in read_records(path):
                for span in text_from(record):
                    if len(span) < args.min_chars:
                        continue
                    handle.write(span.replace("\n", " ").strip() + "\n")
                    documents += 1
                    characters += len(span)
                    tokens += len(tok.encode(span))
                if args.max_documents and documents >= args.max_documents:
                    break
            if args.max_documents and documents >= args.max_documents:
                break

    manifest = {
        "source": args.source, "files_read": len(files), "documents": documents,
        "characters": characters, "tokens": tokens,
        "chars_per_token": characters / max(tokens, 1),
        "extraction_seconds": round(time.time() - started, 1),
        "output": str(out), "output_bytes": out.stat().st_size,
    }
    Path(str(out) + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print("CORPUS_PREP_DONE " + str(out), flush=True)


if __name__ == "__main__":
    main()
