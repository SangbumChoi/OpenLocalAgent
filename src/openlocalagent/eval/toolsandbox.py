"""Fail-closed aggregation for Apple ToolSandbox result summaries.

ToolSandbox evaluates stateful, conversational tool trajectories with milestone and minefield
similarities. The upstream CLI writes a JSON result summary containing one record per scenario.
This module consumes that summary without importing ToolSandbox, executing tools, or recomputing
state from a trajectory. It is therefore suitable for a local WebGPU result receipt, but it does
not claim an official leaderboard score.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_AUGMENTED_CATEGORIES = frozenset(
    {
        "TOOL_NAME_SCRAMBLED",
        "TOOL_DESCRIPTION_SCRAMBLED",
        "ARG_DESCRIPTION_SCRAMBLED",
        "ARG_TYPE_SCRAMBLED",
        "ARG_NAME_SCRAMBLED",
    }
)


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def _finite(value: object, *, label: str, minimum: float = 0.0, maximum: float = 1.0) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(numeric) or not minimum <= numeric <= maximum:
        raise ValueError(f"{label} must be finite and in [{minimum}, {maximum}]")
    return numeric


def _nonnegative(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return numeric


def _expected_names(expected_scenarios: Sequence[str] | None) -> tuple[str, ...] | None:
    if expected_scenarios is None:
        return None
    if isinstance(expected_scenarios, (str, bytes)) or not expected_scenarios:
        raise ValueError("expected_scenarios must be a non-empty sequence")
    names = tuple(sorted(_text(name, label="expected scenario") for name in expected_scenarios))
    if len(set(names)) != len(names):
        raise ValueError("expected_scenarios contains duplicates")
    return names


def _load_results(path: Path) -> list[Mapping[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid ToolSandbox result summary: {path}") from error
    if isinstance(payload, list):
        raw_results = payload
    elif isinstance(payload, Mapping):
        raw_results = payload.get("scenario_results", payload.get("results"))
        if raw_results is None:
            raise ValueError(
                "ToolSandbox summary object must contain scenario_results or results"
            )
    else:
        raise ValueError("ToolSandbox result summary must be a list or object")
    if not isinstance(raw_results, list) or not raw_results:
        raise ValueError("ToolSandbox result summary must contain a non-empty result list")
    results: list[Mapping[str, Any]] = []
    for index, row in enumerate(raw_results):
        if not isinstance(row, Mapping):
            raise ValueError(f"ToolSandbox result {index} must be an object")
        results.append(row)
    return results


def _validate_mapping(row: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    name = _text(row.get("name"), label=f"{label}.name")
    categories = row.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError(f"{label}.categories must be a non-empty list")
    category_names = tuple(
        sorted({_text(category, label=f"{label}.categories[]") for category in categories})
    )
    similarity = _finite(row.get("similarity"), label=f"{label}.similarity")
    milestone_similarity = _finite(
        row.get("milestone_similarity", similarity),
        label=f"{label}.milestone_similarity",
    )
    minefield_similarity = _finite(
        row.get("minefield_similarity", similarity),
        label=f"{label}.minefield_similarity",
    )
    turn_count = _nonnegative(row.get("turn_count"), label=f"{label}.turn_count")
    traceback = row.get("traceback")
    if traceback is not None and not isinstance(traceback, str):
        raise ValueError(f"{label}.traceback must be text or null")
    mapping = row.get("milestone_mapping", {})
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{label}.milestone_mapping must be an object")
    normalized_mapping: dict[str, list[float]] = {}
    for milestone, value in mapping.items():
        milestone_name = _text(str(milestone), label=f"{label}.milestone_mapping key")
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{label}.milestone_mapping[{milestone_name!r}] must have two values")
        turn = _nonnegative(value[0], label=f"{label}.milestone_mapping[{milestone_name!r}].turn")
        if not turn.is_integer():
            raise ValueError(f"{label}.milestone_mapping[{milestone_name!r}].turn must be integral")
        score = _finite(
            value[1],
            label=f"{label}.milestone_mapping[{milestone_name!r}].similarity",
        )
        normalized_mapping[milestone_name] = [int(turn), score]
    return {
        "name": name,
        "categories": list(category_names),
        "similarity": similarity,
        "milestone_similarity": milestone_similarity,
        "minefield_similarity": minefield_similarity,
        "turn_count": turn_count,
        "traceback": traceback,
        "milestone_mapping": normalized_mapping,
    }


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows) if rows else 0.0


def aggregate_toolsandbox_results(
    result_summary: str | Path,
    *,
    expected_scenarios: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Aggregate one official ToolSandbox JSON result summary.

    An exact scenario list is required for `completeness.verified`. Without it the observed
    summary remains useful for debugging, but the receipt is deliberately marked incomplete.
    """

    path = Path(result_summary)
    expected = _expected_names(expected_scenarios)
    raw_results = _load_results(path)
    rows = [_validate_mapping(row, label=f"results[{index}]") for index, row in enumerate(raw_results)]
    names = [row["name"] for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("ToolSandbox result summary contains duplicate scenario names")
    discovered = tuple(sorted(names))
    expected_set = set(expected or ())
    discovered_set = set(discovered)
    missing = sorted(expected_set - discovered_set) if expected is not None else []
    unexpected = sorted(discovered_set - expected_set) if expected is not None else []
    complete = bool(expected is not None and not missing and not unexpected)

    category_rows: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        categories = set(row["categories"])
        for category in categories:
            if (
                category == "THREE_DISTRACTION_TOOLS"
                and categories & _AUGMENTED_CATEGORIES
            ):
                continue
            category_rows.setdefault(category, []).append(row)
        category_rows.setdefault("ALL_CATEGORIES", []).append(row)
    by_category = {
        category: {
            "scenarios": len(category_values),
            "mean_similarity": _mean(category_values, "similarity"),
            "mean_milestone_similarity": _mean(category_values, "milestone_similarity"),
            "mean_minefield_similarity": _mean(category_values, "minefield_similarity"),
            "mean_turn_count": _mean(category_values, "turn_count"),
            "exact_similarity_rate": sum(
                float(row["similarity"]) == 1.0 for row in category_values
            )
            / len(category_values),
        }
        for category, category_values in sorted(category_rows.items())
    }
    source_bytes = path.read_bytes()
    exceptions = sum(row["traceback"] is not None for row in rows)
    return {
        "kind": "localagent_toolsandbox_result_aggregate",
        "schema_version": 1,
        "result_summary_path": str(path),
        "result_summary_bytes": len(source_bytes),
        "result_summary_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "scenarios": len(rows),
        "overall": {
            "mean_similarity": _mean(rows, "similarity"),
            "mean_milestone_similarity": _mean(rows, "milestone_similarity"),
            "mean_minefield_similarity": _mean(rows, "minefield_similarity"),
            "mean_turn_count": _mean(rows, "turn_count"),
            "exact_similarity_rate": sum(float(row["similarity"]) == 1.0 for row in rows)
            / len(rows),
            "exception_scenarios": exceptions,
        },
        "by_category": by_category,
        "scenarios_detail": rows,
        "completeness": {
            "verified": complete,
            "expected_scenarios": list(expected) if expected is not None else None,
            "discovered_scenarios": list(discovered),
            "missing_scenarios": missing,
            "unexpected_scenarios": unexpected,
        },
        "status": "complete" if complete else "incomplete",
        "claim_scope": (
            "Local aggregation of an upstream ToolSandbox result_summary.json; exact scenario "
            "coverage is required for status=complete. This is not an official leaderboard score "
            "and does not independently recompute ToolSandbox milestone DAGs from trajectories."
        ),
    }


__all__ = ["aggregate_toolsandbox_results"]
