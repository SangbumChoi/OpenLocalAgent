#!/usr/bin/env python3
"""Run production RL only from a current, ready, self-hashed schema-v2 summary."""

from __future__ import annotations

import argparse
import json

from localagent.eval.rl_readiness import run_ready_rl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "summary",
        help="sealed schema-v2 RL-readiness JSON summary",
    )
    args = parser.parse_args(argv)
    try:
        result = run_ready_rl(args.summary)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
