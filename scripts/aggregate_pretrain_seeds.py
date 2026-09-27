#!/usr/bin/env python
"""Aggregate three config-bound paired pretraining comparisons by training seed."""

from __future__ import annotations

import argparse
import json

from localagent.eval.pretrain_scorecard import write_scorecard
from localagent.eval.pretrain_seed_aggregate import (
    SeedComparisonSpec,
    aggregate_pretrain_seeds,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        nargs=4,
        required=True,
        metavar=("SEED", "ATTENTION_CONFIG", "HYBRID_CONFIG", "COMPARISON"),
        help="repeat exactly three times",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        specifications = [
            SeedComparisonSpec(
                seed=int(seed),
                attention_config=attention,
                hybrid_config=hybrid,
                comparison=comparison,
            )
            for seed, attention, hybrid, comparison in args.run
        ]
        report = aggregate_pretrain_seeds(specifications)
        write_scorecard(report, args.output)
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
