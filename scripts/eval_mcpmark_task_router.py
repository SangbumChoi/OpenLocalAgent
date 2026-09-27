#!/usr/bin/env python
"""Evaluate a checkpoint's service/tool-family routing on public MCPMark descriptions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from localagent.eval.mcpmark_router import evaluate_mcpmark_router


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True, help="pinned MCPMark checkout")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--suite", choices=("standard", "easy"), default="standard")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = evaluate_mcpmark_router(
        args.checkout,
        args.checkpoint,
        suite=args.suite,
        device=args.device,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("dataset", "source", "overall")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
