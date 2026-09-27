"""Seal the AppWorld runtime-pool correction and bounded native replay."""

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


def assemble(m648_path: Path, m649_path: Path, output: Path) -> dict[str, Any]:
    m648 = _load(m648_path)
    m649 = _load(m649_path)
    if m648.get("kind") != "localagent_m648_appworld_head_native_receipt":
        raise ValueError("m648 receipt kind mismatch")
    if m649.get("kind") != "localagent_appworld_checkpoint_native_probe":
        raise ValueError("m649 native report kind mismatch")
    if m649.get("configuration", {}).get("retrieve_k") != 100:
        raise ValueError("m649 must use the full standard tool pool")
    if m649.get("summary", {}).get("tasks") != 6:
        raise ValueError("m649 task count mismatch")
    if m649.get("summary", {}).get("action_replayed") != 6:
        raise ValueError("m649 did not dispatch all six actions")
    payload: dict[str, Any] = {
        "kind": "localagent_m649_appworld_runtime_pool_receipt",
        "schema_version": 1,
        "benchmark": {
            "dataset": "AppWorld",
            "data_version": "0.2.0",
            "source_url": "https://github.com/StonyBrookNLP/appworld",
            "native_tasks": 6,
            "native_split": "public dev subset",
            "protected_test_used": False,
        },
        "diagnostic": {
            "m648_default_retrieval": m648["native_replay"]["head_adapted"]["summary"],
            "m649_full_pool": m649["summary"],
            "default_retrieve_k": 10,
            "matched_retrieve_k": 100,
            "finding": (
                "The m648 head ranked run_python correctly in direct feature probes, but the default "
                "retriever could remove run_python before the dense selector. Full-pool dispatch "
                "replayed six run_python actions; native task success remained zero because this "
                "bounded adapter executes only one API step per task."
            ),
        },
        "native_replay": {
            "report": _identity(m649_path),
            "checkpoint": m649["checkpoint"],
            "configuration": m649["configuration"],
            "summary": m649["summary"],
            "contract_verification": m649["runner"]["contract_verification"],
        },
        "decision": {
            "retain_full_pool_for_head_dispatch": True,
            "promote_to_appworld_success": False,
            "reason": (
                "Pool correction fixes the tool-dispatch interface (6/6 actions replayed) but not "
                "the multi-step AppWorld objective (0/6 native successes); keep the full-pool setting "
                "and extend the evaluator to a measured multi-step policy before promotion."
            ),
        },
        "claim_boundary": (
            "AppWorld 0.2.0 resettable native replay with a full STANDARD_TOOLS candidate pool and "
            "strict one-step schema translation. This is not an official leaderboard score, complete "
            "task success, external-account side effect, or WebGPU deployment result."
        ),
        "inputs": {"m648": _identity(m648_path), "m649_native": _identity(m649_path)},
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
    parser.add_argument("--m648", type=Path, required=True)
    parser.add_argument("--m649-native", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.m648, args.m649_native, args.out)["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
