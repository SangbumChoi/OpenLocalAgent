#!/usr/bin/env python3
"""Aggregate trained cached-decode browser payloads into a tracked summary."""

from __future__ import annotations

import argparse
from pathlib import Path

from localagent.eval.webgpu_decode_summary import (
    build_trained_decode_summary,
    write_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path, nargs="+", help="independent browser result payloads")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--quality-summary", type=Path, required=True)
    parser.add_argument("--paired-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = build_trained_decode_summary(
        args.raw,
        repository_root=args.repository_root,
        quality_summary_path=args.quality_summary,
        paired_comparison_path=args.paired_comparison,
    )
    write_summary(args.output, summary)


if __name__ == "__main__":
    main()
