#!/usr/bin/env python3
"""Bind the current m626 workshop gate to the m645 native grounding diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CURRENT_CHECKPOINT_SHA256 = "d6b5df5ff0ba53cb797501b21fd788ca904b2773cf7ea56269864d0350670e3c"
M645_KIND = "localagent_m645_mind2web_browsergym_grounding_canary_receipt"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def assemble(gate_path: Path, grounding_path: Path, output: Path) -> dict[str, Any]:
    gate = _load(gate_path)
    grounding = _load(grounding_path)
    if gate.get("kind") != "localagent_workshop_publication_gate" or gate.get("ready") is not False:
        raise ValueError("gate must remain fail-closed")
    if gate.get("current_checkpoint", {}).get("sha256") != CURRENT_CHECKPOINT_SHA256:
        raise ValueError("gate is not bound to m626")
    if grounding.get("kind") != M645_KIND:
        raise ValueError("grounding receipt kind mismatch")
    if grounding.get("benchmark", {}).get("official_split_verified") is not False:
        raise ValueError("m645 must remain non-official")
    if grounding.get("paired_checkpoint_identity", {}).get("baseline_sha256") != CURRENT_CHECKPOINT_SHA256:
        raise ValueError("m645 baseline checkpoint mismatch")
    checks = {item["requirement"]: item for item in gate.get("checks", [])}
    payload: dict[str, Any] = {
        "kind": "localagent_m646_workshop_gate_current_m626_grounding",
        "schema_version": 1,
        "current_checkpoint": gate["current_checkpoint"],
        "gate": gate,
        "gate_input": _identity(gate_path),
        "grounding_diagnostic": _identity(grounding_path),
        "grounding_summary": {
            "semantic_success_rate_baseline": grounding["semantic_fallback"]["baseline"]["success_rate"],
            "semantic_success_rate_child": grounding["semantic_fallback"]["mind2web_child"]["success_rate"],
            "coordinate_success_rate_baseline": grounding["coordinate_fallback"]["baseline"]["success_rate"],
            "coordinate_success_rate_child": grounding["coordinate_fallback"]["mind2web_child"]["success_rate"],
            "semantic_child_success_delta": grounding["semantic_fallback"]["paired_comparison"]["success_delta"],
            "coordinate_child_success_delta": grounding["coordinate_fallback"]["paired_comparison"]["success_delta"],
        },
        "decision": {
            "ready": False,
            "passed_checks": sorted(
                requirement for requirement, check in checks.items() if check.get("status") == "pass"
            ),
            "blocking_requirements": gate["blocking_requirements"],
            "grounding_diagnostic_changes_readiness": False,
        },
        "claim_boundary": (
            "Current m626 fail-closed publication gate with the m645 paired native BrowserGym/MiniWoB "
            "grounding diagnostic attached. Semantic accessibility grounding remains 4/16 for both "
            "arms; the optional DOM-coordinate bridge reaches 8/16 for both arms, so neither is an "
            "official benchmark result or a Mind2Web transfer gain. Missing native suites, the "
            "ToolSandbox official-split blocker, and an authenticated public model/demo manifest "
            "continue to block publication."
        ),
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
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = assemble(args.gate, args.grounding, args.out)
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
