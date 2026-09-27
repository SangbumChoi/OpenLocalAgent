#!/usr/bin/env python
"""Evaluate sealed SFT checkpoints on their frozen teacher-forced held-out set."""

from __future__ import annotations

import argparse
import json

from localagent.eval.sft_checkpoint_sweep import (
    run_sft_checkpoint_sweep,
    write_sft_checkpoint_sweep_result,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to the checkpoint-sweep YAML config")
    parser.add_argument("--output", required=True, help="Canonical result JSON path")
    args = parser.parse_args()
    try:
        result = run_sft_checkpoint_sweep(args.config)
        write_sft_checkpoint_sweep_result(result, args.output)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result["summary"], allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
