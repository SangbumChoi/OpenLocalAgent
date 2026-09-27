#!/usr/bin/env python
"""Fill the ladder demo template with live run curves and benchmark receipts."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MAX_CURVE_POINTS = 200


@dataclass(frozen=True)
class RunCurve:
    """One run's downsampled loss curve, as the page consumes it."""

    n: int
    points: list[list[float]]
    last: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkRow:
    """One model's ten-benchmark result, as the leaderboard and scatter consume it."""

    name: str
    params: int | None
    kind: str | None
    mean: float
    suites: dict[str, float] = field(default_factory=dict)


def read_curves(runs_dir: Path) -> dict[str, RunCurve]:
    curves: dict[str, RunCurve] = {}
    for run in sorted(runs_dir.glob("*-la-*")):
        path = run / "curve.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            continue
        stride = max(1, len(rows) // MAX_CURVE_POINTS)
        kept = rows[::stride]
        if kept[-1] is not rows[-1]:
            kept.append(rows[-1])
        curves[run.name] = RunCurve(
            n=len(rows),
            points=[[row["step"], round(row["loss"], 4)] for row in kept],
            # source_draws is a per-source counter dict; it would dominate the payload
            last={k: v for k, v in rows[-1].items() if k != "source_draws"},
        )
    return curves


def read_results(results_dir: Path) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            report = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        scores = {
            name: block["step_success_rate"]
            for name, block in report.get("suites", {}).items()
            if block.get("step_success_rate") is not None
        }
        if len(scores) < 10:      # a partial receipt is not a comparable row
            continue
        rows.append(BenchmarkRow(
            name=path.stem,
            params=report.get("parameters"),
            kind=report.get("kind"),
            mean=round(statistics.mean(scores.values()) * 100, 2),
            suites={name: round(value * 100, 2) for name, value in scores.items()},
        ))
    rows.sort(key=lambda row: -row.mean)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--results", default="results/evalsuite-full")
    parser.add_argument("--template", default="demos/ladder/page.template.html")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    curves = read_curves(Path(args.runs))
    bench = read_results(Path(args.results))
    payload = {
        "curves": {name: asdict(curve) for name, curve in curves.items()},
        "bench": [asdict(row) for row in bench],
    }
    template = Path(args.template).read_text()
    if "__DATA__" not in template:
        raise SystemExit(f"{args.template} has no __DATA__ placeholder")
    Path(args.out).write_text(template.replace("__DATA__", json.dumps(payload, separators=(",", ":"))))
    print(f"wrote {args.out}: {len(curves)} runs, {len(bench)} result rows")


if __name__ == "__main__":
    main()
