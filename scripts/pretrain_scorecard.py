#!/usr/bin/env python
"""Evaluate a lineage-matched pretrain checkpoint on immutable raw held-out documents.

Example:
  PYTHONPATH=src python scripts/pretrain_scorecard.py \
    --checkpoint runs/pretrain/latest.pt \
    --manifest data/shards/paper-all/manifest.json \
    --tokenizer-path data/tokenizer-paper-16k.json \
    --group general=mixture:fineweb_edu_dedup,mixture:cosmopedia_v2 \
    --group code=mixture:permissive_python \
    --group structured=mixture:structured_html \
    --output runs/pretrain/scorecard.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from localagent.eval.pretrain_scorecard import (
    evaluate_pretrain_checkpoint,
    parse_source_groups,
    write_scorecard,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="pretrain checkpoint")
    parser.add_argument("--manifest", required=True, help="matching packed corpus manifest")
    parser.add_argument(
        "--corpus-db",
        help="manifest-bound staging database; inferred from the manifest when available",
    )
    parser.add_argument("--tokenizer-kind", choices=("byte", "bpe"))
    parser.add_argument("--tokenizer-path", help="BPE tokenizer artifact")
    parser.add_argument(
        "--group",
        action="append",
        required=True,
        metavar="NAME=SOURCE_FAMILY[,SOURCE_FAMILY...]",
        help="requested source grouping; repeat for multiple groups",
    )
    parser.add_argument("--output", required=True, help="JSON scorecard output path")
    parser.add_argument(
        "--document-sidecar",
        help="optional compact per-document JSONL metrics for paired comparison",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--chunk-length",
        type=int,
        help="context length per deterministic non-overlapping chunk; defaults to model maximum",
    )
    args = parser.parse_args()

    try:
        if (
            args.document_sidecar is not None
            and Path(args.document_sidecar).resolve() == Path(args.output).resolve()
        ):
            raise ValueError("--document-sidecar and --output must be different files")
        groups = parse_source_groups(args.group)
        report = evaluate_pretrain_checkpoint(
            args.checkpoint,
            args.manifest,
            groups,
            corpus_db_path=args.corpus_db,
            tokenizer_kind=args.tokenizer_kind,
            tokenizer_path=args.tokenizer_path,
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
            chunk_length=args.chunk_length,
            document_sidecar_path=args.document_sidecar,
        )
        write_scorecard(report, args.output)
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
