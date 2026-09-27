#!/usr/bin/env python3
"""Create or verify a canonical no-model post-training budget plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from localagent.train.stage_budget import (
    build_stage_budget_plan,
    verify_stage_budget_plan,
    write_stage_budget_plan,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        help="midtrain/SFT/RL YAML config; omit with --verify",
    )
    parser.add_argument("--out", help="canonical JSON output path; otherwise print to stdout")
    parser.add_argument(
        "--max-steps",
        type=int,
        help="fixed scheduling horizon (defaults to schedule.total_steps)",
    )
    parser.add_argument(
        "--min-supervised-tokens",
        type=int,
        help="select the smallest fixed-horizon prefix reaching this many loss tokens",
    )
    parser.add_argument(
        "--max-supervised-tokens",
        type=int,
        help="fail if the selected prefix exceeds this many loss tokens",
    )
    parser.add_argument(
        "--verify",
        metavar="PLAN",
        help="verify canonical encoding, self-hash, and current artifact replay",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.verify:
        if args.config or args.out or any(
            value is not None
            for value in (
                args.max_steps,
                args.min_supervised_tokens,
                args.max_supervised_tokens,
            )
        ):
            raise SystemExit("--verify cannot be combined with planning arguments")
        plan = verify_stage_budget_plan(args.verify)
        print(
            json.dumps(
                {
                    "verified": str(args.verify),
                    "stage": plan["stage"],
                    "plan_self_sha256": plan["plan_self_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    if not args.config:
        raise SystemExit("CONFIG is required unless --verify is used")
    plan = build_stage_budget_plan(
        Path(args.config),
        max_steps=args.max_steps,
        min_supervised_tokens=args.min_supervised_tokens,
        max_supervised_tokens=args.max_supervised_tokens,
    )
    if args.out:
        write_stage_budget_plan(args.out, plan)
        print(
            json.dumps(
                {
                    "out": str(args.out),
                    "stage": plan["stage"],
                    "plan_self_sha256": plan["plan_self_sha256"],
                },
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(plan, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
