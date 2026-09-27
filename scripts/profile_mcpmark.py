#!/usr/bin/env python
"""Profile MCPMark metadata without retaining task descriptions or verifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from localagent.data.mcpmark import MCPMARK_REVISION, profile_mcpmark


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="checked-out MCPMark root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default=MCPMARK_REVISION)
    args = parser.parse_args()
    profile = profile_mcpmark(args.root, revision=args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(profile, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
