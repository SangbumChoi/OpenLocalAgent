#!/usr/bin/env python3
"""Run a bounded free-running AppWorld API trajectory probe.

The model chooses a LocalAgent tool on every step; a strict schema adapter then selects one literal
``apis.<app>.<api>(...)`` candidate and executes it in a resettable AppWorld fixture.  The adapter
does not inject a ground-truth action, answer, or completion call.  This is a native closed-loop
diagnostic, not an AppWorld leaderboard implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from scripts.evaluate_appworld_checkpoint import (
    _appworld_execute_api_step,
    _parse_appworld_api_code,
    _registry,
    _schema_ground_appworld_api_step,
    _text_hash,
    _tracker_summary,
)
from scripts.normalize_appworld_trajectories import _safe_value


def _sha256(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _response_summary(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return {
            "type": "object",
            "keys": sorted(str(key) for key in response)[:24],
            "value": _safe_value(response),
        }
    if isinstance(response, list):
        return {"type": "array", "length": len(response), "value": _safe_value(response)}
    return {"type": type(response).__name__, "value": _safe_value(response)}


def evaluate(
    *, checkpoint: Path, root: Path, task_ids: list[str], report: Path,
    max_steps: int, retrieve_k: int, experiment_name: str,
    appworld_api_head: Path | None = None, allow_completion: bool = False,
    lexical_first: bool = False,
) -> dict[str, Any]:
    if max_steps < 1 or retrieve_k < 1:
        raise ValueError("max_steps and retrieve_k must be positive")
    from appworld import AppWorld, update_root
    from localagent.agent.runtime import Agent
    from localagent.eval.appworld_api_head import load_appworld_api_head

    root = root.resolve()
    if Path(update_root(str(root))).resolve() != root:
        raise RuntimeError("AppWorld root did not resolve to the requested data root")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    for task_id in task_ids:
        if not (root / "data" / "tasks" / task_id / "specs.json").is_file():
            raise FileNotFoundError(f"AppWorld task specs not found for {task_id!r}")
    calls: list[dict[str, Any]] = []
    agent = Agent.from_checkpoint(
        checkpoint,
        _registry(calls),
        selector_first=True,
        retrieve_k=retrieve_k,
    )
    api_head = (
        load_appworld_api_head(appworld_api_head, d_model=agent.model.cfg.d_model)
        if appworld_api_head is not None else None
    )
    records: list[dict[str, Any]] = []
    for task_id in task_ids:
        spec = json.loads((root / "data" / "tasks" / task_id / "specs.json").read_text(encoding="utf-8"))
        instruction = str(spec["instruction"])
        prompt = instruction
        seen_codes: set[str] = set()
        steps: list[dict[str, Any]] = []
        replay_error: str | None = None
        with AppWorld(
            task_id=task_id,
            experiment_name=experiment_name,
            load_ground_truth=True,
            ground_truth_mode="full",
            max_interactions=max_steps * 4,
        ) as world:
            for step_index in range(max_steps):
                calls.clear()
                model_output = agent.chat(prompt, max_tool_hops=1)
                selected_tool = calls[0]["name"] if calls else None
                code = _schema_ground_appworld_api_step(
                    agent.model,
                    agent.tokenizer,
                    world,
                    prompt,
                    api_head=api_head,
                    allow_completion=allow_completion,
                    lexical_first=lexical_first,
                )
                parsed = _parse_appworld_api_code(code) if code is not None else None
                step_record: dict[str, Any] = {
                    "step": step_index,
                    "model_output": _text_hash(model_output),
                    "selected_tool": selected_tool,
                    "translated_code": _text_hash(code) if code is not None else None,
                    "action_replayed": False,
                    "native_api_calls": 0,
                }
                if parsed is None:
                    step_record["stop_reason"] = "no_literal_api_candidate"
                    steps.append(step_record)
                    break
                app, api, arguments = parsed
                canonical = json.dumps(
                    {"app": app, "api": api, "arguments": arguments},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if canonical in seen_codes:
                    step_record["stop_reason"] = "repeated_api_candidate"
                    steps.append(step_record)
                    break
                seen_codes.add(canonical)
                try:
                    response, request_count = _appworld_execute_api_step(world, app, api, arguments)
                    step_record.update(
                        {
                            "action_replayed": True,
                            "api": f"{app}.{api}",
                            "native_api_calls": request_count,
                            "response": _response_summary(response),
                        }
                    )
                    prompt = (
                        f"{prompt}\nASSISTANT: [run_python({canonical})]\n"
                        f"TOOL_RESULT: {json.dumps(_response_summary(response), sort_keys=True)}\n"
                        "Next required action:"
                    )
                except Exception as error:  # fail closed but preserve the step diagnosis
                    replay_error = repr(error)
                    step_record["stop_reason"] = "native_api_error"
                    step_record["error"] = _text_hash(replay_error)
                    steps.append(step_record)
                    break
                steps.append(step_record)
            tracker = world.evaluate()
            evaluation = _tracker_summary(tracker)
        records.append(
            {
                "task_id": task_id,
                "instruction": _text_hash(instruction),
                "steps": steps,
                "action_replayed": sum(int(item["action_replayed"]) for item in steps),
                "native_api_calls": sum(int(item["native_api_calls"]) for item in steps),
                "evaluation": evaluation,
                "replay_error": _text_hash(replay_error) if replay_error else None,
            }
        )
    result: dict[str, Any] = {
        "kind": "localagent_appworld_checkpoint_free_running_trajectory_probe",
        "schema_version": 1,
        "checkpoint": _sha256(checkpoint),
        "runner": {
            "dataset": "AppWorld",
            "data_version": (root / "data" / "version.txt").read_text(encoding="utf-8").strip(),
            "root": str(root),
            "task_count": len(task_ids),
        },
        "configuration": {
            "tasks": task_ids,
            "max_steps": max_steps,
            "retrieve_k": retrieve_k,
            "selector_first": True,
            "observations": "redacted response type/keys summaries",
            "schema_adapter": "strict one literal API call per step",
            "appworld_api_head": str(appworld_api_head) if appworld_api_head else None,
            "allow_completion": allow_completion,
            "lexical_first": lexical_first,
        },
        "environment": {
            "native_runtime_executed": True,
            "environment_reset_per_task": True,
            "external_accounts": False,
            "protected_test_used": False,
        },
        "tasks": records,
        "summary": {
            "tasks": len(records),
            "native_successes": sum(int(item["evaluation"]["success"]) for item in records),
            "native_success_rate": sum(int(item["evaluation"]["success"]) for item in records) / max(1, len(records)),
            "action_replayed": sum(item["action_replayed"] for item in records),
            "native_api_calls": sum(item["native_api_calls"] for item in records),
            "steps_attempted": sum(len(item["steps"]) for item in records),
        },
        "claim_boundary": (
            "Native resettable AppWorld free-running trajectory probe using a strict schema adapter and "
            "redacted observations. The model is not given ground-truth actions or task answers; this "
            "is not an official leaderboard score and does not claim live email/Notion side effects. "
            "When completion is enabled, only a literal status=success supervisor candidate is added; "
            "question answers are not injected."
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
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("APPWORLD_ROOT", ".")))
    parser.add_argument("--task", dest="tasks", action="append", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--retrieve-k", type=int, default=100)
    parser.add_argument("--experiment-name", default="localagent_appworld_free_running_trajectory")
    parser.add_argument("--appworld-api-head", type=Path)
    parser.add_argument(
        "--allow-completion",
        action="store_true",
        help="add the strict supervisor.complete_task(status='success') candidate after an API step",
    )
    parser.add_argument(
        "--lexical-first",
        action="store_true",
        help="use the top schema-overlap candidate as a diagnostic grounding control",
    )
    args = parser.parse_args()
    result = evaluate(
        checkpoint=args.checkpoint,
        root=args.root,
        task_ids=args.tasks,
        report=args.report,
        max_steps=args.max_steps,
        retrieve_k=args.retrieve_k,
        experiment_name=args.experiment_name,
        appworld_api_head=args.appworld_api_head,
        allow_completion=args.allow_completion,
        lexical_first=args.lexical_first,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
