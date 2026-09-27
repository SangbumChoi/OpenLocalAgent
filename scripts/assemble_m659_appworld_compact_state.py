"""Seal compact state-sketch AppWorld continuation and native replay."""

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
    warm_report_path: Path, random_report_path: Path, warm_weight_path: Path,
    random_weight_path: Path, warm_native_path: Path, random_native_path: Path,
    train_manifest_path: Path, eval_manifest_path: Path, output: Path,
) -> dict[str, Any]:
    warm, random = _load(warm_report_path), _load(random_report_path)
    warm_weight, random_weight = _load(warm_weight_path), _load(random_weight_path)
    warm_native, random_native = _load(warm_native_path), _load(random_native_path)
    for report, parent in ((warm, "9c760ff270d12c797db937c94366ef33e801cac1f9fbee2c0015184f7837111e"), (random, "f39869027d7822f454fcd181d24cc139b4286e7837d349856c497633675559cd")):
        if report.get("kind") != "localagent_public_agent_continuation_report":
            raise ValueError("continuation kind mismatch")
        if report.get("parent", {}).get("sha256") != parent:
            raise ValueError("continuation parent mismatch")
        if report.get("source", {}).get("dataset") != "appworld_ground_truth_api_trajectory_compact":
            raise ValueError("compact dataset mismatch")
        if report.get("rows") != {"train": 64, "eval": 18}:
            raise ValueError("split mismatch")
    for report in (warm_native, random_native):
        if report.get("kind") != "localagent_appworld_checkpoint_free_running_trajectory_probe":
            raise ValueError("native kind mismatch")
        if report.get("summary", {}).get("tasks") != 6:
            raise ValueError("native task count mismatch")
    warm_after = warm["after"]["eval"]["assistant_token_accuracy"]
    random_after = random["after"]["eval"]["assistant_token_accuracy"]
    payload: dict[str, Any] = {
        "kind": "localagent_m659_appworld_compact_state_receipt",
        "schema_version": 1,
        "benchmark": {
            "dataset": "AppWorld", "data_version": "0.2.0",
            "source_url": "https://github.com/StonyBrookNLP/appworld",
            "train_rows": 64, "eval_rows": 18, "max_actions_per_row": 3,
            "native_tasks": 6, "native_max_steps": 8, "retrieve_k": 100,
            "protected_test_used": False,
            "train_tokens_over_window": 1,
            "eval_tokens_over_window": 3,
        },
        "metrics": {
            "warm_before": warm["before"]["eval"], "warm_after": warm["after"]["eval"],
            "random_before": random["before"]["eval"], "random_after": random["after"]["eval"],
            "warm_delta_pp": 100 * (warm_after - warm["before"]["eval"]["assistant_token_accuracy"]),
            "random_delta_pp": 100 * (random_after - random["before"]["eval"]["assistant_token_accuracy"]),
            "warm_after_minus_random_after_pp": 100 * (warm_after - random_after),
            "warm_exact_sequence_accuracy": warm["after"]["eval"]["assistant_sequence_accuracy"],
            "random_exact_sequence_accuracy": random["after"]["eval"]["assistant_sequence_accuracy"],
        },
        "weight_movement": {"warm": warm_weight["groups"], "random": random_weight["groups"]},
        "native_replay": {
            "warm": {"report": _identity(warm_native_path), "summary": warm_native["summary"]},
            "random": {"report": _identity(random_native_path), "summary": random_native["summary"]},
        },
        "decision": {
            "promote_to_native_success": False,
            "retain_compact_state_sketch": True,
            "reason": (
                "Compact state sketches fit the 2,048-token context for 35/36 rows and increase warm "
                "native action replay from 3 to 5, but verified success remains 0/6; exact sequence "
                "accuracy is zero and completion/state planning is still missing."
            ),
        },
        "claim_boundary": (
            "Public AppWorld train/dev compact response-state continuation with credentials and other "
            "sensitive fields redacted, plus a six-task native diagnostic. This is not an official "
            "leaderboard score, complete task success, or WebGPU email/Notion promotion."
        ),
        "inputs": {
            "warm_report": _identity(warm_report_path), "random_report": _identity(random_report_path),
            "warm_weight": _identity(warm_weight_path), "random_weight": _identity(random_weight_path),
            "warm_native": _identity(warm_native_path), "random_native": _identity(random_native_path),
            "train_manifest": _identity(train_manifest_path), "eval_manifest": _identity(eval_manifest_path),
        },
    }
    payload["receipt_self_sha256"] = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("warm-report", "random-report", "warm-weight", "random-weight", "warm-native", "random-native", "train-manifest", "eval-manifest"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.warm_report, args.random_report, args.warm_weight, args.random_weight, args.warm_native, args.random_native, args.train_manifest, args.eval_manifest, args.out)["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
