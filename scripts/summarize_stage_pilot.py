#!/usr/bin/env python
"""Validate and summarize one bounded midtrain -> SFT -> RL pilot."""

from __future__ import annotations

import argparse
import json

from localagent.eval.stage_pilot_summary import (
    StagePilotInput,
    summarize_stage_pilot,
    write_stage_pilot_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for stage in ("midtrain", "sft", "rl"):
        parser.add_argument(f"--{stage}-config", required=True)
        parser.add_argument(f"--{stage}-metrics", required=True)
        parser.add_argument(f"--{stage}-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    inputs = [
        StagePilotInput(
            stage=stage,
            metrics_path=getattr(args, f"{stage}_metrics"),
            checkpoint_path=getattr(args, f"{stage}_checkpoint"),
            config_path=getattr(args, f"{stage}_config"),
        )
        for stage in ("midtrain", "sft", "rl")
    ]
    try:
        summary = summarize_stage_pilot(inputs)
        write_stage_pilot_summary(summary, args.output)
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
