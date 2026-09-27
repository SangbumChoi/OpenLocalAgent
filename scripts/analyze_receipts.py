#!/usr/bin/env python
"""Report evaluation receipts by category, not by one mean.

The ten-suite mean averaged sound tool calling with a floor-bound suite, a leaked one, a wall
and seventeen rows of noise (docs/HARNESS.md). This prints, for each receipt, the headline
sound-tool-calling mean, the projected-GUI mean, every other suite with its floor, and the legacy
ten-suite mean labelled as such. Receipts written before the summary block get it computed here.

  python scripts/analyze_receipts.py results/evalsuite-full/la-*-matrix.json
  python scripts/analyze_receipts.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from openlocalagent.eval.interaction import SPECS, Category, summarize  # noqa: E402

RECEIPTS = Path("results/evalsuite-full")


def pct(value: float | None) -> str:
    return "   -  " if value is None else f"{100 * value:6.1f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", nargs="*", type=Path)
    parser.add_argument("--all", action="store_true", help=f"every receipt under {RECEIPTS}")
    args = parser.parse_args()
    paths = sorted(RECEIPTS.glob("*.json")) if args.all else args.receipts
    if not paths:
        parser.error("give receipts or --all")

    singles = [n for n, s in SPECS.items() if not s.category.averaged and s.category is not Category.BLOCKED]
    print(f"{'receipt':<34}{'sound':>7}{'proj':>7}" + "".join(f"{n[:9]:>10}" for n in singles)
          + f"{'legacy10':>10}")
    for path in paths:
        receipt = json.loads(path.read_text())
        summary = receipt.get("summary") or summarize(receipt.get("suites", {}))
        cats = summary["by_category"]
        row = f"{path.stem:<34}{pct(summary['headline']['value']):>7}"
        row += f"{pct(cats.get(Category.PROJECTED.value, {}).get('mean')):>7}"
        for name in singles:
            block = cats.get(SPECS[name].category.value, {}).get("suites", {}).get(name)
            row += f"{pct(block['step_success_rate'] if block else None):>10}"
        row += f"{pct(summary['legacy_ten_suite_mean']):>10}"
        print(row)
    print("sound = mean step_success over xlam/toolace/bfcl; proj = androidcontrol/mobileactions; "
          "singles are reported, not averaged; legacy10 kept for continuity only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
