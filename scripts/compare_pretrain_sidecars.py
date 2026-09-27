#!/usr/bin/env python
"""Paired document-bootstrap comparison of attention and hybrid pretrain sidecars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from localagent.eval.pretrain_compare import compare_pretrain_sidecars
from localagent.eval.pretrain_scorecard import write_scorecard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention", required=True, help="attention per-document JSONL sidecar")
    parser.add_argument("--hybrid", required=True, help="hybrid per-document JSONL sidecar")
    parser.add_argument("--output", required=True, help="paired comparison JSON output")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    try:
        output = Path(args.output).resolve()
        if output in {Path(args.attention).resolve(), Path(args.hybrid).resolve()}:
            raise ValueError("--output must differ from both input sidecars")
        report = compare_pretrain_sidecars(
            args.attention,
            args.hybrid,
            seed=args.seed,
            resamples=args.resamples,
            confidence=args.confidence,
        )
        write_scorecard(report, args.output)
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
