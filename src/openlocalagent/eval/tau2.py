"""Fail-closed aggregation for public tau2-bench trajectory results.

The upstream tau2-bench runner writes either one monolithic results.json object or a directory
containing results.json plus simulations/*.json. This module reads those JSON artifacts without
importing tau2-bench, running a user simulator, or launching external tools. Its Pass^k values
mirror the upstream combinatorial estimator; they are a local receipt, not an official
leaderboard submission.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_INFRASTRUCTURE_TERMINATIONS = frozenset({"infrastructure_error"})
_SUCCESS_EPSILON = 1e-6


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


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
        raise ValueError(f"{label} must be finite and non-negative")
    return numeric


def _expected_cases(expected_cases: Sequence[str] | None) -> tuple[str, ...] | None:
    if expected_cases is None:
        return None
    if isinstance(expected_cases, (str, bytes)) or not expected_cases:
        raise ValueError("expected_cases must be a non-empty sequence")
    cases = tuple(sorted(_text(case, label="expected case") for case in expected_cases))
    if len(set(cases)) != len(cases):
        raise ValueError("expected_cases contains duplicates")
    return cases


def _json_object(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be an object: {path}")
    return payload


def _load_result_payload(
    path: Path,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]], list[dict[str, Any]]]:
    """Load either official monolithic or directory-based tau2 output."""

    if path.is_dir():
        metadata_path = path / "results.json"
        simulations_dir = path / "simulations"
    else:
        metadata_path = path
        simulations_dir = path.parent / "simulations"
    metadata = _json_object(metadata_path, label="tau2 results metadata")
    metadata_payload = metadata_path.read_bytes()
    raw_simulations: list[Mapping[str, Any]] = []
    file_records = [
        {
            "path": metadata_path.relative_to(path if path.is_dir() else metadata_path.parent).as_posix(),
            "bytes": len(metadata_payload),
            "sha256": hashlib.sha256(metadata_payload).hexdigest(),
        }
    ]
    if simulations_dir.is_dir():
        simulation_paths = sorted(simulations_dir.glob("*.json"))
        if not simulation_paths:
            raise ValueError(f"tau2 simulations directory is empty: {simulations_dir}")
        for simulation_path in simulation_paths:
            raw_simulations.append(_json_object(simulation_path, label="tau2 simulation"))
            payload = simulation_path.read_bytes()
            file_records.append(
                {
                    "path": simulation_path.relative_to(metadata_path.parent).as_posix(),
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    else:
        candidate = metadata.get("simulations")
        if not isinstance(candidate, list) or not candidate:
            raise ValueError(
                "tau2 results must contain simulations[] or a sibling simulations/*.json directory"
            )
        for index, simulation in enumerate(candidate):
            if not isinstance(simulation, Mapping):
                raise ValueError(f"tau2 simulations[{index}] must be an object")
            raw_simulations.append(simulation)
    return metadata, raw_simulations, file_records


def _message_count(simulation: Mapping[str, Any], *, label: str) -> int:
    messages = simulation.get("messages")
    ticks = simulation.get("ticks")
    if messages is not None:
        if not isinstance(messages, list):
            raise ValueError(f"{label}.messages must be a list or null")
        return len(messages)
    if ticks is not None:
        if not isinstance(ticks, list):
            raise ValueError(f"{label}.ticks must be a list or null")
        return len(ticks)
    return 0


def _validate_simulation(
    simulation: Mapping[str, Any],
    *,
    domain: str,
    label: str,
) -> dict[str, Any]:
    simulation_id = _text(simulation.get("id"), label=f"{label}.id")
    raw_task_id = simulation.get("task_id")
    if raw_task_id is None or isinstance(raw_task_id, bool):
        raise ValueError(f"{label}.task_id must be text or an integer")
    if not isinstance(raw_task_id, (str, int)):
        raise ValueError(f"{label}.task_id must be text or an integer")
    task_id = _text(str(raw_task_id), label=f"{label}.task_id")
    trial = _positive_int(simulation.get("trial"), label=f"{label}.trial")
    termination = _text(
        simulation.get("termination_reason"),
        label=f"{label}.termination_reason",
    ).lower()
    reward_info = simulation.get("reward_info")
    if reward_info is not None and not isinstance(reward_info, Mapping):
        raise ValueError(f"{label}.reward_info must be an object or null")
    reward_value = (
        reward_info.get("reward")
        if isinstance(reward_info, Mapping)
        else simulation.get("reward")
    )
    if reward_value is None and termination in _INFRASTRUCTURE_TERMINATIONS:
        reward: float | None = None
    else:
        reward = _finite(reward_value, label=f"{label}.reward")
    duration = _nonnegative(simulation.get("duration", 0.0), label=f"{label}.duration")
    agent_cost = simulation.get("agent_cost")
    if agent_cost is not None:
        agent_cost = _nonnegative(agent_cost, label=f"{label}.agent_cost")
    case = f"{domain}/{task_id}@{trial}"
    return {
        "id": simulation_id,
        "domain": domain,
        "task_id": task_id,
        "trial": trial,
        "case": case,
        "reward": reward,
        "success": reward is not None and reward >= 1.0 - _SUCCESS_EPSILON,
        "termination_reason": termination,
        "infrastructure_error": termination in _INFRASTRUCTURE_TERMINATIONS,
        "duration_s": duration,
        "agent_cost": agent_cost,
        "messages": _message_count(simulation, label=label),
    }


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows if row[field] is not None]
    return sum(values) / len(values) if values else 0.0


def _group_by_task(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(f"{row['domain']}/{row['task_id']}", []).append(row)
    return grouped


def _pass_hat(rows: Sequence[Mapping[str, Any]], k: int) -> float:
    eligible = [row for row in rows if not row["infrastructure_error"]]
    if len(eligible) < k:
        return 0.0
    successes = sum(bool(row["success"]) for row in eligible)
    return math.comb(successes, k) / math.comb(len(eligible), k)


def _metric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if not row["infrastructure_error"]]
    task_groups = _group_by_task(eligible)
    max_k = min((len(group) for group in task_groups.values()), default=0)
    pass_hat = {
        str(k): (
            sum(_pass_hat(task_rows, k) for task_rows in task_groups.values())
            / len(task_groups)
            if task_groups
            else 0.0
        )
        for k in range(1, max_k + 1)
    }
    return {
        "simulations": len(rows),
        "eligible_simulations": len(eligible),
        "tasks": len(task_groups),
        "average_reward": _mean(eligible, "reward"),
        "pass_hat_k": pass_hat,
        "mean_messages": _mean(eligible, "messages"),
        "mean_duration_s": _mean(eligible, "duration_s"),
        "mean_agent_cost": _mean(eligible, "agent_cost"),
        "infrastructure_errors": sum(row["infrastructure_error"] for row in rows),
        "successes": sum(row["success"] for row in eligible),
        "termination_reasons": dict(
            sorted(Counter(row["termination_reason"] for row in rows).items())
        ),
    }


def aggregate_tau2_results(
    result_path: str | Path,
    *,
    expected_cases: Sequence[str] | None = None,
    expected_trials: int | None = None,
    source_revision: str | None = None,
    runtime_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Aggregate tau2 JSON output and verify exact task/trial coverage.

    expected_cases uses the stable form domain/task_id@trial. A complete receipt also requires
    every observed task to have the requested number of non-infrastructure trials and rejects
    infrastructure failures. Without an expected case list the result is deliberately marked
    incomplete, even though diagnostic metrics are still returned.
    """

    path = Path(result_path)
    if not path.exists():
        raise ValueError(f"tau2 result path is missing: {path}")
    expected = _expected_cases(expected_cases)
    metadata, raw_simulations, file_records = _load_result_payload(path)
    info = metadata.get("info")
    if not isinstance(info, Mapping):
        raise ValueError("tau2 results must contain an info object")
    environment = info.get("environment_info")
    if not isinstance(environment, Mapping):
        raise ValueError("tau2 info must contain environment_info")
    domain = _text(environment.get("domain_name"), label="tau2 domain")
    declared_trials = info.get("num_trials")
    if declared_trials is not None:
        declared_trials = _positive_int(declared_trials, label="tau2 info.num_trials")
    requested_trials = (
        _positive_int(expected_trials, label="expected_trials")
        if expected_trials is not None
        else declared_trials
    )
    rows = [
        _validate_simulation(simulation, domain=domain, label=f"simulations[{index}]")
        for index, simulation in enumerate(raw_simulations)
    ]
    cases = [row["case"] for row in rows]
    if len(set(cases)) != len(cases):
        raise ValueError("tau2 results contain duplicate simulation task/trial cases")
    discovered = tuple(sorted(cases))
    expected_set = set(expected or ())
    discovered_set = set(discovered)
    missing = sorted(expected_set - discovered_set) if expected is not None else []
    unexpected = sorted(discovered_set - expected_set) if expected is not None else []
    groups = _group_by_task(rows)
    eligible_groups = _group_by_task([row for row in rows if not row["infrastructure_error"]])
    count_mismatches = (
        {
            task: len(task_rows)
            for task, task_rows in sorted(eligible_groups.items())
            if requested_trials is not None and len(task_rows) != requested_trials
        }
        if requested_trials is not None
        else {}
    )
    infra_cases = sorted(row["case"] for row in rows if row["infrastructure_error"])
    complete = bool(
        expected is not None
        and not missing
        and not unexpected
        and not count_mismatches
        and not infra_cases
        and (
            declared_trials is None
            or requested_trials is None
            or declared_trials == requested_trials
        )
    )
    by_task = {
        task: {
            "simulations": len(task_rows),
            "eligible_simulations": sum(
                not row["infrastructure_error"] for row in task_rows
            ),
            "successes": sum(
                row["success"] for row in task_rows if not row["infrastructure_error"]
            ),
            "average_reward": _mean(
                [row for row in task_rows if not row["infrastructure_error"]], "reward"
            ),
            "pass_hat_k": {
                str(k): _pass_hat(task_rows, k)
                for k in range(
                    1,
                    len(
                        [row for row in task_rows if not row["infrastructure_error"]]
                    )
                    + 1,
                )
            },
        }
        for task, task_rows in sorted(groups.items())
    }
    by_domain = {domain: _metric_summary(rows)}
    source_bytes = sum(record["bytes"] for record in file_records)
    source_hash = hashlib.sha256()
    for record in file_records:
        source_hash.update(record["sha256"].encode("ascii"))
    overall = _metric_summary(rows)
    overall["domains"] = len(by_domain)
    return {
        "kind": "localagent_tau2_result_aggregate",
        "schema_version": 1,
        "result_path": str(path),
        "source_bytes": source_bytes,
        "source_sha256": source_hash.hexdigest(),
        "source_files": file_records,
        "source_revision": source_revision,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "domain": domain,
        "declared_trials": declared_trials,
        "requested_trials": requested_trials,
        "simulations": len(rows),
        "tasks": len(groups),
        "overall": overall,
        "by_domain": by_domain,
        "by_task": by_task,
        "completeness": {
            "verified": complete,
            "expected_cases": list(expected) if expected is not None else None,
            "discovered_cases": list(discovered),
            "missing_cases": missing,
            "unexpected_cases": unexpected,
            "count_mismatches": count_mismatches,
            "infrastructure_cases": infra_cases,
        },
        "status": "complete" if complete else "incomplete",
        "claim_scope": (
            "Local aggregation of tau2-bench Results JSON. Pass^k is the upstream combinatorial "
            "estimator over successful reward approximately 1 trials; exact base-split coverage and "
            "the upstream validator are still required for an official leaderboard submission. "
            "This receipt does not run tau2-bench, user simulation, or external tools."
        ),
    }


__all__ = ["aggregate_tau2_results"]
