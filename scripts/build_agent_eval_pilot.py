#!/usr/bin/env python
"""Derive the frozen 65-row capability-pilot evaluation artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from localagent.data.agent_eval_subset import derive_agent_eval_pilot_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the contract-pinned 65-row pilot subset from the frozen agent eval "
            "artifact, audit it against frozen agent training data, and write a manifest."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/synth/agent_eval.jsonl"),
        help="frozen evaluation JSONL",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/synth/agent_eval.jsonl.manifest.json"),
        help="manifest for the frozen evaluation JSONL",
    )
    parser.add_argument(
        "--train-source",
        type=Path,
        default=Path("data/synth/agent_sft.jsonl"),
        help="frozen training JSONL used for the leakage audit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synth/agent_eval_pilot65.jsonl"),
        help="derived subset JSONL",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/synth/agent_eval_pilot65.jsonl.manifest.json"),
        help="derived subset manifest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = derive_agent_eval_pilot_subset(
        args.source,
        args.output,
        args.manifest,
        source_manifest_path=args.source_manifest,
        train_source_path=args.train_source,
    )
    print(json.dumps(manifest["output"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
