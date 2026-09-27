#!/usr/bin/env python
"""Normalize pinned public agent datasets into provenance-bound Conversation JSONL."""

from __future__ import annotations

import argparse
import json

from localagent.data.public_agent import build_public_agent_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic LocalAgent Conversation splits from already-downloaded, "
            "byte-pinned public source snapshots."
        )
    )
    parser.add_argument("config", help="public-agent ingestion YAML")
    args = parser.parse_args()
    result = build_public_agent_dataset(args.config)
    print(
        json.dumps(
            {
                "manifest": str(result.manifest_path),
                "manifest_self_sha256": result.manifest["manifest_self_sha256"],
                "outputs": {
                    split: {
                        "path": str(result.outputs[split]),
                        "rows": len(rows),
                    }
                    for split, rows in sorted(result.conversations.items())
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
