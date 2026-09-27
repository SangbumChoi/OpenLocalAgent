#!/usr/bin/env python
"""Validate existing scorecard/preflight evidence and report RL readiness."""

from __future__ import annotations

import argparse
import json

from localagent.eval.rl_readiness import (
    reproduce_historical_rl_readiness_v1,
    summarize_rl_readiness,
    write_rl_readiness_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Versioned RL-readiness YAML config")
    parser.add_argument("--output", help="Optional canonical JSON summary destination")
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit with status 2 when valid evidence does not pass every promotion gate",
    )
    parser.add_argument(
        "--historical-verify-v1",
        action="store_true",
        help=(
            "Reproduce a sealed schema-v1 summary for historical verification only; "
            "cannot be combined with --require-ready"
        ),
    )
    args = parser.parse_args()
    if args.historical_verify_v1 and args.require_ready:
        parser.error("--historical-verify-v1 cannot be combined with --require-ready")

    try:
        summary = (
            reproduce_historical_rl_readiness_v1(args.config)
            if args.historical_verify_v1
            else summarize_rl_readiness(args.config)
        )
        if args.output:
            write_rl_readiness_summary(summary, args.output)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    print(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True))
    if args.require_ready and not summary["decision"]["promotion_allowed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
