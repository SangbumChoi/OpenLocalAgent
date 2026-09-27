#!/usr/bin/env python
"""Profile Toolathlon-GYM task configs without reading benchmark prompts or verifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from localagent.data.toolathlon_gym import TOOLATHLON_GYM_REVISION, profile_toolathlon_gym


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="checked-out Toolathlon-GYM root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default=TOOLATHLON_GYM_REVISION)
    args = parser.parse_args()
    profile = profile_toolathlon_gym(args.root, revision=args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(profile, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
