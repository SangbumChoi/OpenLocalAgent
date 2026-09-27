"""Seal a current-backbone AppWorld API-schema head and native one-step replay."""

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


def assemble(api_report_path: Path, native_path: Path, output: Path) -> dict[str, Any]:
    api = _load(api_report_path)
    native = _load(native_path)
    if api.get("kind") != "localagent_appworld_api_head_training_report":
        raise ValueError("API-head report kind mismatch")
    if native.get("kind") != "localagent_appworld_checkpoint_native_probe":
        raise ValueError("native report kind mismatch")
    if api.get("metrics", {}).get("eval", {}).get("rows") != 15:
        raise ValueError("API-head eval slice must contain 15 seen-label rows")
    if native.get("summary", {}).get("tasks") != 6:
        raise ValueError("native task count mismatch")
    payload: dict[str, Any] = {
        "kind": "localagent_m650_appworld_api_head_receipt",
        "schema_version": 1,
        "benchmark": {
            "dataset": "AppWorld",
            "data_version": "0.2.0",
            "source_url": "https://github.com/StonyBrookNLP/appworld",
            "api_head_train_rows": api["source"]["train_rows"],
            "api_head_eval_rows": api["metrics"]["eval"]["rows"],
            "native_tasks": native["summary"]["tasks"],
            "protected_test_used": False,
            "unseen_eval_labels_excluded": True,
        },
        "api_head": {
            "report": _identity(api_report_path),
            "child": api["child"],
            "classes": api["classes"],
            "metrics": api["metrics"],
        },
        "native_replay": {
            "report": _identity(native_path),
            "checkpoint": native["checkpoint"],
            "summary": native["summary"],
            "configuration": native["configuration"],
        },
        "decision": {
            "retain_api_head_as_diagnostic": True,
            "promote_to_native_success": False,
            "reason": (
                "A frozen API head reaches 60% on a disjoint seen-label public dev slice and "
                "restricts all six replayed actions to the observed Spotify schema, but one-step "
                "native success remains 0/6 because these tasks require multi-step stateful plans."
            ),
        },
        "claim_boundary": (
            "Current m649 warm-body AppWorld API-schema head trained on public train rows and measured "
            "on a disjoint seen-label dev slice, plus six resettable one-step native replays. This is "
            "not an official leaderboard score, complete task success, external-account result, or "
            "WebGPU deployment evidence."
        ),
        "inputs": {"api_report": _identity(api_report_path), "native_report": _identity(native_path)},
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
    parser.add_argument("--api-report", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.api_report, args.native, args.out)["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
