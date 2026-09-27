#!/usr/bin/env python3
"""Validate and aggregate three WebGPU single-step DOM pilot result payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from localagent.eval.webgpu_dom_summary import (
    build_webgpu_dom_summary,
    write_webgpu_dom_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "raw",
        type=Path,
        nargs=3,
        help="three independent raw DOM browser results in chronological order",
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        summary = build_webgpu_dom_summary(
            args.raw,
            repository_root=args.repository_root,
        )
        write_webgpu_dom_summary(summary, args.output)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
