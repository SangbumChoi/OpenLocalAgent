"""Seal the matched free-running AppWorld trajectory probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def assemble(
    warm_training_path: Path,
    random_training_path: Path,
    warm_native_path: Path,
    random_native_path: Path,
    output: Path,
) -> dict[str, Any]:
    warm_training = _load(warm_training_path)
    random_training = _load(random_training_path)
    warm_native = _load(warm_native_path)
    random_native = _load(random_native_path)
    for label, report in (("warm", warm_native), ("random", random_native)):
        if report.get("kind") != "localagent_appworld_checkpoint_free_running_trajectory_probe":
            raise ValueError(f"{label} native report kind mismatch")
        if report.get("configuration", {}).get("max_steps") != 8:
            raise ValueError(f"{label} max-step mismatch")
        if report.get("configuration", {}).get("retrieve_k") != 100:
            raise ValueError(f"{label} retrieve-k mismatch")
        if report.get("summary", {}).get("tasks") != 6:
            raise ValueError(f"{label} native task count mismatch")
    warm_tasks = warm_native["configuration"]["tasks"]
    if warm_tasks != random_native["configuration"]["tasks"]:
        raise ValueError("warm/random native task order differs")
    payload: dict[str, Any] = {
        "kind": "localagent_m652_appworld_trajectory_native_receipt",
        "schema_version": 1,
        "benchmark": {
            "dataset": "AppWorld",
            "data_version": "0.2.0",
            "source_url": "https://github.com/StonyBrookNLP/appworld",
            "native_tasks": 6,
            "native_split": "public dev subset",
            "protected_test_used": False,
            "max_steps": 8,
            "retrieve_k": 100,
        },
        "teacher_forced_context": {
            "warm_report": _identity(warm_training_path),
            "random_report": _identity(random_training_path),
            "warm_eval_token_accuracy": warm_training["after"]["eval"]["assistant_token_accuracy"],
            "random_eval_token_accuracy": random_training["after"]["eval"]["assistant_token_accuracy"],
        },
        "native_replay": {
            "warm": {
                "report": _identity(warm_native_path),
                "checkpoint": warm_native["checkpoint"],
                "summary": warm_native["summary"],
            },
            "random": {
                "report": _identity(random_native_path),
                "checkpoint": random_native["checkpoint"],
                "summary": random_native["summary"],
            },
            "paired_task_ids": warm_tasks,
        },
        "decision": {
            "promote_to_native_appworld_success": False,
            "reason": (
                "The matched free-running adapter executes 3 warm and 15 random API actions across "
                "six resettable public dev tasks, but both arms score 0/6 native task success. The "
                "warm child repeats show_song_queue or fails required-argument grounding; teacher-forced "
                "trajectory gains do not transfer to a stateful free-run policy."
            ),
        },
        "claim_boundary": (
            "Native AppWorld resettable free-running trajectory probe with strict one-call schema "
            "translation, redacted observations, and matched warm/random checkpoints. This is not an "
            "official leaderboard score, complete AppWorld task result, external-account side effect, "
            "or WebGPU email/Notion deployment claim."
        ),
        "inputs": {
            "warm_training": _identity(warm_training_path),
            "random_training": _identity(random_training_path),
            "warm_native": _identity(warm_native_path),
            "random_native": _identity(random_native_path),
        },
    }
    payload["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-training", type=Path, required=True)
    parser.add_argument("--random-training", type=Path, required=True)
    parser.add_argument("--warm-native", type=Path, required=True)
    parser.add_argument("--random-native", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(
        args.warm_training, args.random_training, args.warm_native, args.random_native, args.out
    )["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
