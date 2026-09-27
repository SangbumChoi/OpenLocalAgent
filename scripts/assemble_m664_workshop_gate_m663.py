"""Seal the current-checkpoint workshop gate alongside the m663 AppWorld audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from localagent.eval.workshop_gate import build_workshop_gate


def _identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def assemble(
    *,
    output: Path,
    appworld_receipt: Path,
    current_checkpoint: Path,
    manifest: Path,
    weight_reports: list[Path],
    rl_receipt: Path,
    mobilegym: Path,
    browsergym: Path,
    toolsandbox: Path,
    webgpu: Path,
) -> dict[str, Any]:
    appworld = _load(appworld_receipt)
    if appworld.get("kind") != "localagent_m663_appworld_grounding_receipt":
        raise ValueError("m663 receipt kind mismatch")
    gate = build_workshop_gate(
        "configs/data/realistic-agent-eval.catalog.yaml",
        repo_root=".",
        native_receipts={
            "mobilegym": mobilegym,
            "browsergym_miniwob": browsergym,
            "toolsandbox": toolsandbox,
        },
        webgpu_receipt=webgpu,
        weight_reports=weight_reports,
        public_artifact_manifest=manifest,
        rl_preflight_receipt=rl_receipt,
        current_checkpoint=current_checkpoint,
    )
    payload: dict[str, Any] = {
        "kind": "localagent_m664_workshop_gate_m663_receipt",
        "schema_version": 1,
        "gate": gate,
        "current_checkpoint": _identity(current_checkpoint),
        "m663_appworld": {
            "receipt": _identity(appworld_receipt),
            "decision": appworld["decision"],
            "metrics": appworld["metrics"],
            "claim_boundary": appworld["claim_boundary"],
        },
        "inputs": {
            "manifest": _identity(manifest),
            "weight_reports": [_identity(path) for path in weight_reports],
            "rl_receipt": _identity(rl_receipt),
            "mobilegym": _identity(mobilegym),
            "browsergym": _identity(browsergym),
            "toolsandbox": _identity(toolsandbox),
            "webgpu": _identity(webgpu),
        },
        "decision": {
            "ready": gate["ready"],
            "promote_to_public_workshop": False,
            "reason": (
                "m663 improves AppWorld observation grounding and proves persisted completion-prefix "
                "control, but fully free-running AppWorld success remains 0/6. The supplied native "
                "MobileGym/BrowserGym/RL receipts are bound to an older checkpoint, the m663 movement "
                "reports are not a canonical transfer/no-transfer ablation, and the public manifest is "
                "not bound to the m663 child. The fail-closed gate therefore remains blocked."
            ),
        },
        "claim_boundary": (
            "This receipt is a current-checkpoint publication decision, not a new benchmark score. "
            "Ground-truth-prefix AppWorld completion is explicitly diagnostic and cannot satisfy the "
            "native free-running or public-artifact requirements."
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
    parser.add_argument("--appworld-receipt", type=Path, required=True)
    parser.add_argument("--current-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weight-report", type=Path, action="append", required=True)
    parser.add_argument("--rl-receipt", type=Path, required=True)
    parser.add_argument("--mobilegym", type=Path, required=True)
    parser.add_argument("--browsergym", type=Path, required=True)
    parser.add_argument("--toolsandbox", type=Path, required=True)
    parser.add_argument("--webgpu", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = assemble(
        output=args.out,
        appworld_receipt=args.appworld_receipt,
        current_checkpoint=args.current_checkpoint,
        manifest=args.manifest,
        weight_reports=args.weight_report,
        rl_receipt=args.rl_receipt,
        mobilegym=args.mobilegym,
        browsergym=args.browsergym,
        toolsandbox=args.toolsandbox,
        webgpu=args.webgpu,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
