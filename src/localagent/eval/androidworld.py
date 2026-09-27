"""Fail-closed aggregation for AndroidWorld emulator checkpointer runs.

AndroidWorld's ``IncrementalCheckpointer`` writes one gzip-compressed pickle per task instance.
This module consumes those files after an operator has run the upstream emulator suite.  It does
not launch ``adb`` or AndroidWorld, and it never calls the result an official leaderboard score.

Pickle is executable by design.  The default loader accepts only builtin Python containers and
scalars, which is useful for small exported metadata fixtures.  Real AndroidWorld checkpoints
usually contain screenshots and other non-builtin episode objects, so the CLI requires an explicit
``--allow-unsafe-pickle`` acknowledgement to load a trusted local checkpoint produced by the
operator's own run.
"""

from __future__ import annotations

import gzip
import hashlib
import math
import pickle
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_PICKLE_SUFFIX = ".pkl.gz"
_BUILTIN_GLOBALS = {
    ("builtins", name): getattr(__import__("builtins"), name)
    for name in (
        "bool",
        "bytes",
        "complex",
        "dict",
        "float",
        "frozenset",
        "int",
        "list",
        "set",
        "str",
        "tuple",
    )
}


class _BuiltinUnpickler(pickle.Unpickler):
    """Unpickler that refuses imports and therefore cannot reconstruct arbitrary classes."""

    def find_class(self, module: str, name: str) -> Any:  # noqa: ANN401
        try:
            return _BUILTIN_GLOBALS[(module, name)]
        except KeyError as error:
            raise pickle.UnpicklingError(
                f"safe AndroidWorld loader rejected global {module}.{name}"
            ) from error


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _sort_key(path: Path) -> tuple[str, int | str]:
    """Match AndroidWorld's task-group ordering (task name, then instance number)."""

    name = path.name[: -len(_PICKLE_SUFFIX)]
    if "_" not in name:
        return name, 0
    task, instance = name.rsplit("_", maxsplit=1)
    try:
        return task, int(instance)
    except ValueError:
        return name, 0


def _regular_file(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"AndroidWorld result must not be a symlink: {path}")
    try:
        mode = path.stat().st_mode
    except OSError as error:
        raise ValueError(f"cannot stat AndroidWorld result: {path}") from error
    if not stat.S_ISREG(mode):
        raise ValueError(f"AndroidWorld result must be a regular file: {path}")


def _load_group(path: Path, *, allow_unsafe_pickle: bool) -> object:
    _regular_file(path)
    try:
        with gzip.open(path, "rb") as handle:
            if allow_unsafe_pickle:
                return pickle.load(handle)
            return _BuiltinUnpickler(handle).load()
    except Exception as error:  # noqa: BLE001 - convert malformed/unavailable upstream objects
        mode = "unsafe" if allow_unsafe_pickle else "safe"
        raise ValueError(f"unable to read {mode} AndroidWorld pickle: {path}") from error


def _finite_number(value: object, *, label: str, allow_missing: bool = True) -> float | None:
    if value is None and allow_missing:
        return None
    if isinstance(value, bool):
        numeric = float(value)
    else:
        try:
            numeric = float(value)  # Supports numpy scalar values in trusted checkpoints.
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(numeric):
        if allow_missing:
            return None
        raise ValueError(f"{label} must be finite")
    return numeric


def _success(episode: Mapping[str, Any], *, label: str) -> float:
    value = _finite_number(episode.get("is_successful"), label=f"{label}.is_successful")
    if value is None:
        # AndroidWorld's failed-result helper stores NaN and a traceback.  The upstream tally
        # excludes that episode from the correct count; counting it as an observed failure keeps
        # the denominator explicit while still rejecting silent malformed records.
        if episode.get("exception_info") is not None:
            return 0.0
        raise ValueError(f"{label}.is_successful is missing or non-finite")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{label}.is_successful must lie in [0, 1]")
    return 1.0 if value > 0.5 else 0.0


def _expected_names(expected_tasks: Sequence[str] | None) -> tuple[str, ...] | None:
    if expected_tasks is None:
        return None
    if isinstance(expected_tasks, (str, bytes)) or not expected_tasks:
        raise ValueError("expected_tasks must be a non-empty sequence")
    names = tuple(sorted(_text(name, label="expected task") for name in expected_tasks))
    if len(set(names)) != len(names):
        raise ValueError("expected_tasks contains duplicates")
    return names


def _discover_files(root: Path) -> tuple[Path, ...]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"AndroidWorld run directory is missing or not a directory: {root}")
    files = tuple(sorted(root.glob(f"*{_PICKLE_SUFFIX}"), key=_sort_key))
    if not files:
        raise ValueError(f"AndroidWorld run contains no {_PICKLE_SUFFIX} files: {root}")
    return files


def aggregate_androidworld_run(
    run_dir: str | Path,
    *,
    expected_tasks: Sequence[str] | None = None,
    n_task_combinations: int | None = None,
    source_revision: str | None = None,
    agent_name: str | None = None,
    allow_unsafe_pickle: bool = False,
) -> dict[str, Any]:
    """Aggregate one AndroidWorld ``run_...`` directory.

    ``expected_tasks`` and ``n_task_combinations`` are required for a complete, publication-ready
    coverage check.  Without both, observed metrics are returned for debugging but
    ``completeness.verified`` is false and ``status`` is ``incomplete``.
    """

    root = Path(run_dir)
    names = _expected_names(expected_tasks)
    combinations = (
        _positive_int(n_task_combinations, label="n_task_combinations")
        if n_task_combinations is not None
        else None
    )
    files = _discover_files(root)
    rows: list[dict[str, Any]] = []
    seen_instances: set[tuple[str, int]] = set()
    file_records: list[dict[str, Any]] = []
    for path in files:
        payload = _load_group(path, allow_unsafe_pickle=allow_unsafe_pickle)
        if not isinstance(payload, list) or not payload:
            raise ValueError(f"AndroidWorld task group must be a non-empty list: {path}")
        file_bytes = path.read_bytes()
        file_records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": len(file_bytes),
                "sha256": hashlib.sha256(file_bytes).hexdigest(),
                "episodes": len(payload),
            }
        )
        for index, episode in enumerate(payload):
            label = f"{path.name}[{index}]"
            if not isinstance(episode, Mapping):
                raise ValueError(f"{label} must be an object")
            task = _text(episode.get("task_template"), label=f"{label}.task_template")
            instance = episode.get("instance_id")
            if isinstance(instance, bool) or not isinstance(instance, int) or instance < 0:
                raise ValueError(f"{label}.instance_id must be a non-negative integer")
            key = (task, instance)
            if key in seen_instances:
                raise ValueError(f"duplicate AndroidWorld episode instance: {task}_{instance}")
            seen_instances.add(key)
            runtime = _finite_number(episode.get("run_time"), label=f"{label}.run_time")
            length = _finite_number(
                episode.get("episode_length"), label=f"{label}.episode_length"
            )
            if runtime is not None and runtime < 0:
                raise ValueError(f"{label}.run_time must be non-negative")
            if length is not None and (length < 0 or not length.is_integer()):
                raise ValueError(f"{label}.episode_length must be a non-negative integer")
            rows.append(
                {
                    "task_template": task,
                    "instance_id": instance,
                    "success": _success(episode, label=label),
                    "run_time_s": runtime,
                    "episode_length": int(length) if length is not None else None,
                    "exception": episode.get("exception_info") is not None,
                }
            )

    discovered = tuple(sorted({row["task_template"] for row in rows}))
    expected_set = set(names or ())
    discovered_set = set(discovered)
    missing = sorted(expected_set - discovered_set) if names is not None else []
    unexpected = sorted(discovered_set - expected_set) if names is not None else []
    counts = {task: sum(row["task_template"] == task for row in rows) for task in discovered}
    count_mismatches = (
        {
            task: count
            for task, count in counts.items()
            if combinations is not None and count != combinations
        }
        if combinations is not None
        else {}
    )
    verified = bool(
        names is not None
        and combinations is not None
        and not missing
        and not unexpected
        and not count_mismatches
    )

    by_task: dict[str, dict[str, Any]] = {}
    for task in discovered:
        task_rows = [row for row in rows if row["task_template"] == task]
        times = [row["run_time_s"] for row in task_rows if row["run_time_s"] is not None]
        lengths = [row["episode_length"] for row in task_rows if row["episode_length"] is not None]
        by_task[task] = {
            "episodes": len(task_rows),
            "successes": int(sum(row["success"] for row in task_rows)),
            "success_rate": sum(row["success"] for row in task_rows) / len(task_rows),
            "exception_episodes": sum(row["exception"] for row in task_rows),
            "mean_run_time_s": sum(times) / len(times) if times else None,
            "mean_episode_length": sum(lengths) / len(lengths) if lengths else None,
        }
    successes = int(sum(row["success"] for row in rows))
    times = [row["run_time_s"] for row in rows if row["run_time_s"] is not None]
    lengths = [row["episode_length"] for row in rows if row["episode_length"] is not None]
    return {
        "kind": "localagent_androidworld_result_aggregate",
        "schema_version": 1,
        "run_dir": str(root),
        "source_revision": source_revision,
        "agent_name": agent_name,
        "loader": "unsafe_pickle" if allow_unsafe_pickle else "builtin_only_pickle",
        "files": file_records,
        "tasks": len(discovered),
        "episodes": len(rows),
        "overall": {
            "successes": successes,
            "success_rate": successes / len(rows),
            "exception_episodes": sum(row["exception"] for row in rows),
            "mean_run_time_s": sum(times) / len(times) if times else None,
            "mean_episode_length": sum(lengths) / len(lengths) if lengths else None,
        },
        "by_task": by_task,
        "completeness": {
            "verified": verified,
            "expected_tasks": list(names) if names is not None else None,
            "expected_task_count": len(names) if names is not None else None,
            "n_task_combinations": combinations,
            "discovered_tasks": list(discovered),
            "observed_task_counts": counts,
            "missing_tasks": missing,
            "unexpected_tasks": unexpected,
            "count_mismatches": count_mismatches,
        },
        "status": "complete" if verified else "incomplete",
        "claim_scope": (
            "Complete local AndroidWorld checkpointer aggregation only when completeness.verified "
            "is true; not an official AndroidWorld leaderboard score or a replacement for the "
            "upstream live-emulator reward protocol."
        ),
    }


__all__ = ["aggregate_androidworld_run"]
