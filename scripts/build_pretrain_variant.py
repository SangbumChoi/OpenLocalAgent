#!/usr/bin/env python
"""Build a pretraining shard directory from one prepared corpus, mirroring the reference layout.

The comparison only means something if the corpus is the single thing that changes, so this copies
the reference shard directory's structure and schema and swaps in the corpus text, rather than
inventing a format. The record schema is read from the reference's own first line.

  python scripts/build_pretrain_variant.py --corpus data/pretrain/ultrachat.txt \
      --reference data/shards/h100-mix --out data/shards/pt-ultrachat
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


def text_field(reference: Path) -> str:
    """The field the reference corpus keeps its text in."""
    with (reference / "filtered.jsonl").open(encoding="utf-8") as handle:
        record = json.loads(handle.readline())
    for name in ("text", "content", "document", "body"):
        if isinstance(record.get(name), str):
            return name
    longest = max(record.items(), key=lambda kv: len(kv[1]) if isinstance(kv[1], str) else 0)
    return longest[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="plain text, one document per line")
    ap.add_argument("--reference", default="data/shards/h100-mix")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-documents", type=int, default=0)
    args = ap.parse_args()

    reference, out = Path(args.reference), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    field = text_field(reference)

    started = time.time()
    written = characters = 0
    with (out / "filtered.jsonl").open("w", encoding="utf-8") as sink, \
            Path(args.corpus).open(encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if not line:
                continue
            sink.write(json.dumps({field: line}) + "\n")
            written += 1
            characters += len(line)
            if args.max_documents and written >= args.max_documents:
                break

    manifest = json.loads((reference / "manifest.json").read_text())
    manifest["corpus_variant"] = {
        "corpus": args.corpus, "reference": str(reference), "text_field": field,
        "documents": written, "characters": characters,
        "build_seconds": round(time.time() - started, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for extra in ("generations",):
        source_path = reference / extra
        if source_path.is_dir() and not (out / extra).exists():
            shutil.copytree(source_path, out / extra)
    print(json.dumps(manifest["corpus_variant"], indent=2))
    print("VARIANT_DONE " + str(out), flush=True)


if __name__ == "__main__":
    main()
