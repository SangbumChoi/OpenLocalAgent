#!/usr/bin/env python3
"""Compact the matched random-backbone ToolSandbox control subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


RANDOM_CHECKPOINT_SHA256 = "390f1414260e118cd621af735fe6e87b01e8641b1cff650d594585e39b212e45"
SOURCE_REVISION = "165848b9a78cead7ca7fe7c89c688b58e6501219"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def assemble(raw_path: Path, warm_path: Path, output: Path) -> dict[str, Any]:
    raw = _load(raw_path)
    warm = _load(warm_path)
    if raw.get("kind") != "localagent_toolsandbox_native_matched_random_control":
        raise ValueError("unexpected random-control receipt kind")
    if raw.get("source_revision") != SOURCE_REVISION:
        raise ValueError("ToolSandbox source revision mismatch")
    if raw.get("checkpoint", {}).get("sha256") != RANDOM_CHECKPOINT_SHA256:
        raise ValueError("random control is not bound to the matched m626 random checkpoint")
    if raw.get("task_count") != 25:
        raise ValueError("expected 25 matched control scenarios")
    if warm.get("success_count") != 27 or warm.get("task_count") != 129:
        raise ValueError("unexpected current warm ToolSandbox base receipt")
    warm_by = {row["scenario"]: row for row in warm["scenarios"]}
    rows = raw["scenarios"]
    if any(row["scenario"] not in warm_by for row in rows):
        raise ValueError("random control contains a scenario absent from warm base matrix")
    categories: dict[str, dict[str, int]] = defaultdict(lambda: {"tasks": 0, "random_exact": 0, "warm_exact": 0})
    for row in rows:
        random_exact = int(float(row.get("random_similarity", 0.0)) == 1.0)
        warm_exact = int(float(row.get("warm_similarity", 0.0)) == 1.0)
        for category in row.get("categories", []):
            counts = categories[str(category)]
            counts["tasks"] += 1
            counts["random_exact"] += random_exact
            counts["warm_exact"] += warm_exact
    payload: dict[str, Any] = {
        "kind": "localagent_m636_toolsandbox_matched_random_control_receipt",
        "schema_version": 1,
        "benchmark_id": "toolsandbox",
        "checkpoint": raw["checkpoint"],
        "source": {"url": raw["source_url"], "revision": raw["source_revision"]},
        "protocol": {
            "name": raw["protocol"],
            "task_count": raw["task_count"],
            "official_split_verified": raw["official_split_verified"],
            "user_simulator_executed": raw["user_simulator_executed"],
            "verifier_executed": raw["verifier_executed"],
            "external_api_called": raw["external_api_called"],
        },
        "random_control": {
            "task_count": raw["task_count"],
            "success_count": raw["success_count"],
            "success_rate": raw["success_rate"],
        },
        "matched_warm": {
            "task_count": raw["task_count"],
            "success_count": raw["matched_warm_success_count"],
            "success_rate": raw["matched_warm_success_rate"],
        },
        "category_summary": {
            category: {
                **counts,
                "random_exact_rate": counts["random_exact"] / counts["tasks"],
                "warm_exact_rate": counts["warm_exact"] / counts["tasks"],
            }
            for category, counts in sorted(categories.items())
        },
        "scenarios": rows,
        "random_raw_report": {"path": str(raw_path), "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest()},
        "warm_receipt": {"path": str(warm_path), "sha256": hashlib.sha256(warm_path.read_bytes()).hexdigest()},
        "claim_boundary": raw["claim_boundary"],
    }
    payload["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--warm", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.raw, args.warm, args.out), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
