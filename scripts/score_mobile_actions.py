#!/usr/bin/env python
"""Score predictions against normalized AndroidControl/AITW mobile rows.

Both files are JSONL.  The expected file contains ``localagent_v1`` rows.  Each prediction line is
either a list of ``{tool,args}``/``{name,arguments}`` calls or an object with an ``actions`` list.
This command is an offline action diagnostic and never launches an emulator.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from localagent.eval.mobile import score_mobile_row


def _jsonl(path: Path) -> Iterator[Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from error
            yield value


def _prediction_actions(value: object, *, label: str) -> tuple[Sequence[object], object | None]:
    record_id = None
    if isinstance(value, list):
        raw = value
    elif isinstance(value, Mapping):
        record_id = value.get("record_id")
        raw = value.get("actions")
    else:
        raise ValueError(f"{label} must be a list or object with actions")
    if raw is None:
        raise ValueError(f"{label} must contain an actions list")
    if not isinstance(raw, list):
        raise ValueError(f"{label}.actions must be a list")
    return raw, record_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", required=True, type=Path, help="normalized mobile JSONL")
    parser.add_argument("--predictions", required=True, type=Path, help="prediction JSONL")
    parser.add_argument("--output", required=True, type=Path, help="score receipt JSON")
    args = parser.parse_args()

    expected = iter(_jsonl(args.expected))
    predictions = iter(_jsonl(args.predictions))
    scores: list[dict[str, Any]] = []
    source_families: Counter[str] = Counter()
    sentinel = object()
    row_number = 0
    while True:
        raw_row = next(expected, sentinel)
        prediction = next(predictions, sentinel)
        if raw_row is sentinel and prediction is sentinel:
            break
        row_number += 1
        if raw_row is sentinel or prediction is sentinel:
            raise ValueError(f"expected/prediction row count mismatch at row {row_number}")
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"expected:{row_number} must contain a normalized row object")
        prediction_actions, prediction_id = _prediction_actions(
            prediction, label=f"predictions:{row_number}"
        )
        if prediction_id is not None and prediction_id != raw_row.get("record_id"):
            raise ValueError(f"row {row_number} prediction record_id does not match expected row")
        score = score_mobile_row(
            raw_row,
            prediction_actions,
        )
        scores.append(score)
        source_families[str(score.get("source_family", "unknown"))] += 1

    if not scores:
        raise ValueError("at least one expected/prediction row is required")

    summary = {
        "rows": len(scores),
        "trajectory_exact": sum(score["trajectory_exact"] is True for score in scores) / len(scores),
        "tool_accuracy": sum(float(score["tool_accuracy"]) for score in scores) / len(scores),
        "action_exact_accuracy": sum(
            float(score["action_exact_accuracy"]) for score in scores
        )
        / len(scores),
        "source_families": dict(sorted(source_families.items())),
    }
    receipt = {
        "kind": "localagent_mobile_action_score",
        "schema_version": 1,
        "expected_path": str(args.expected),
        "predictions_path": str(args.predictions),
        "summary": summary,
        "scores": scores,
        "claim_scope": (
            "offline normalized AndroidControl/AITW action diagnostic; not an official device, "
            "AndroidWorld, AndroidControl, or AITW benchmark score"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
