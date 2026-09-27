#!/usr/bin/env python3
"""Evaluate completion after replaying public AppWorld ground-truth API prefixes.

This is intentionally not a free-running task score.  The public ground-truth prefix is injected to
isolate the final state-to-answer/completion decision from earlier API selection.  The completion
candidate is built only from redacted live responses (for example ``follower_count``), never from
the ground-truth answer field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.evaluate_appworld_checkpoint import (
    _appworld_execute_api_step,
    _parse_appworld_api_code,
    _schema_ground_appworld_api_step,
    _text_hash,
    _tracker_summary,
)
from scripts.normalize_appworld_trajectories import _actions, _safe_value, _trace_ground_truth


def _sha256(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _summary(response: Any) -> dict[str, Any]:
    return {"value": _safe_value(response)}


def evaluate(
    *, checkpoint: Path, root: Path, task_ids: list[str], report: Path,
    experiment_name: str, lexical_first: bool = False,
) -> dict[str, Any]:
    from appworld import AppWorld, update_root
    from localagent.agent.runtime import Agent
    from scripts.evaluate_appworld_checkpoint import _registry

    root = root.resolve()
    if Path(update_root(str(root))).resolve() != root:
        raise RuntimeError("AppWorld root did not resolve to the requested data root")
    calls: list[dict[str, Any]] = []
    agent = Agent.from_checkpoint(checkpoint, _registry(calls), selector_first=True, retrieve_k=100)
    records: list[dict[str, Any]] = []
    for task_id in task_ids:
        spec = json.loads((root / "data" / "tasks" / task_id / "specs.json").read_text(encoding="utf-8"))
        instruction = str(spec["instruction"])
        prompt = instruction
        with AppWorld(
            task_id=task_id,
            experiment_name=experiment_name,
            load_ground_truth=True,
            ground_truth_mode="full",
            max_interactions=256,
        ) as trace_world:
            trace, _ = _trace_ground_truth(trace_world)
        # Ground-truth execution mutates the fixture; replay the captured prefix in a fresh world.
        with AppWorld(
            task_id=task_id,
            experiment_name=experiment_name,
            load_ground_truth=True,
            ground_truth_mode="full",
            max_interactions=256,
        ) as world:
            prefix = _actions(trace, max_actions=64, include_completion=False)
            replayed: list[str] = []
            for action in prefix:
                response, request_count = _appworld_execute_api_step(
                    world, str(action["app"]), str(action["api"]), dict(action["arguments"])
                )
                label = f"{action['app']}.{action['api']}"
                replayed.append(label)
                prompt = (
                    f"{prompt}\nASSISTANT: [run_python({label})]\n"
                    f"TOOL_RESULT: {json.dumps(_summary(response), sort_keys=True)}\n"
                    "Next required action:"
                )
            code = _schema_ground_appworld_api_step(
                agent.model,
                agent.tokenizer,
                world,
                prompt,
                allow_completion=True,
                lexical_first=lexical_first,
                completion_only=True,
            )
            parsed = _parse_appworld_api_code(code) if code is not None else None
            completion_error: str | None = None
            completion_response: Any = None
            if parsed is None:
                completion_error = "no_literal_completion_candidate"
            else:
                app, api, arguments = parsed
                try:
                    if app != "supervisor" or api != "complete_task":
                        raise ValueError("completion prefix emitted a non-completion API")
                    completion_response, _ = _appworld_execute_api_step(world, app, api, arguments)
                except Exception as error:  # pragma: no cover - native diagnostic path
                    completion_error = repr(error)
            evaluation = _tracker_summary(world.evaluate())
        records.append(
            {
                "task_id": task_id,
                "instruction": _text_hash(instruction),
                "ground_truth_prefix": replayed,
                "completion_code": _text_hash(code) if code is not None else None,
                "completion_response": _summary(completion_response) if completion_response is not None else None,
                "completion_error": _text_hash(completion_error) if completion_error else None,
                "evaluation": evaluation,
            }
        )
    result: dict[str, Any] = {
        "kind": "localagent_appworld_completion_prefix_probe",
        "schema_version": 1,
        "checkpoint": _sha256(checkpoint),
        "runner": {"dataset": "AppWorld", "data_version": (root / "data" / "version.txt").read_text().strip()},
        "configuration": {
            "tasks": task_ids,
            "ground_truth_prefix_injected": True,
            "completion_only": True,
            "lexical_first": lexical_first,
            "answers_from_ground_truth": False,
            "observations": "bounded redacted live response summaries",
        },
        "tasks": records,
        "summary": {
            "tasks": len(records),
            "native_successes": sum(int(item["evaluation"]["success"]) for item in records),
            "native_success_rate": sum(int(item["evaluation"]["success"]) for item in records) / max(1, len(records)),
        },
        "claim_boundary": (
            "Ground-truth-prefix completion diagnostic only. Earlier API actions are injected from the "
            "public AppWorld solution to isolate completion/state-answer grounding; this is not a free-"
            "running AppWorld score or evidence of email/Notion/external side effects."
        ),
    }
    result["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if report.exists() or report.is_symlink():
        raise FileExistsError(f"refusing to overwrite receipt: {report}")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", dest="tasks", action="append", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiment-name", default="localagent_appworld_completion_prefix")
    parser.add_argument("--lexical-first", action="store_true")
    args = parser.parse_args()
    result = evaluate(
        checkpoint=args.checkpoint,
        root=args.root,
        task_ids=args.tasks,
        report=args.report,
        experiment_name=args.experiment_name,
        lexical_first=args.lexical_first,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
