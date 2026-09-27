#!/usr/bin/env python3
"""Build a provenance-bound continuation-SFT format curriculum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from localagent.data.format_bootstrap import build_format_bootstrap


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select, seal, and verify a short-to-hard curriculum from a verified train artifact"
        )
    )
    parser.add_argument("config", type=Path, help="format-bootstrap YAML config")
    args = parser.parse_args()

    receipt = build_format_bootstrap(args.config)
    selection = receipt["selection"]
    summary = {
        "audit_sha256": selection["audit_sha256"],
        "format_bootstrap_rows": receipt["output"]["rows"],
        "manifest_sha256": receipt["output"]["manifest"]["sha256"],
        "output_sha256": receipt["output"]["jsonl"]["sha256"],
        "overlap": {
            "rendered_prompts": receipt["overlap_audit"]["rendered_prompt_overlap"],
            "semantic_rows": receipt["overlap_audit"]["semantic_overlap"],
        },
        "phase_rows": {
            phase: audit["selected_rows"]
            for phase, audit in selection["phases"].items()
        },
        "receipt_self_sha256": receipt["receipt_self_sha256"],
        "target_tokens_including_eos": selection["output"]["tokens"][
            "target_tokens_including_eos"
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
