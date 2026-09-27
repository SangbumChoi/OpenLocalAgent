"""Seal the fail-closed workshop gate for the m666 AppWorld child."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from localagent.eval.workshop_gate import build_workshop_gate


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def assemble(
    *,
    appworld_receipt: Path,
    current_checkpoint: Path,
    manifest: Path,
    weight_ablation: Path,
    warm_weight: Path,
    random_weight: Path,
    rl_receipt: Path,
    mobilegym: Path,
    browsergym: Path,
    toolsandbox: Path,
    webgpu: Path,
    output: Path,
) -> dict[str, Any]:
    appworld = _load(appworld_receipt)
    if appworld.get("kind") != "localagent_m666_appworld_public_full_receipt":
        raise ValueError("m666 receipt kind mismatch")
    gate = build_workshop_gate(
        "configs/data/realistic-agent-eval.catalog.yaml",
        repo_root=".",
        native_receipts={
            "mobilegym": mobilegym,
            "browsergym_miniwob": browsergym,
            "toolsandbox": toolsandbox,
        },
        webgpu_receipt=webgpu,
        weight_reports=[weight_ablation],
        public_artifact_manifest=manifest,
        rl_preflight_receipt=rl_receipt,
        current_checkpoint=current_checkpoint,
    )
    payload: dict[str, Any] = {
        "kind": "localagent_m667_workshop_gate_m666_receipt",
        "schema_version": 1,
        "gate": gate,
        "current_checkpoint": _identity(current_checkpoint),
        "m666_appworld": {
            "receipt": _identity(appworld_receipt),
            "metrics": appworld["metrics"],
            "decision": appworld["decision"],
            "claim_boundary": appworld["claim_boundary"],
        },
        "inputs": {
            "manifest": _identity(manifest),
            "weight_ablation": _identity(weight_ablation),
            "warm_weight": _identity(warm_weight),
            "random_weight": _identity(random_weight),
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
                "The m666 warm child improves public AppWorld teacher-forced continuation, but its "
                "fully free-running native result remains 0/6. The 6/6 schema-planner control is "
                "explicitly not a model score. Current MobileGym and BrowserGym receipts now bind "
                "this checkpoint, but ToolSandbox/RL evidence is stale while the public manifest is "
                "not bound to this child. "
                "The canonical m668 warm/random movement envelope now passes its structural gate, "
                "but the fail-closed workshop decision remains blocked."
            ),
        },
        "claim_boundary": (
            "Current-checkpoint workshop/publication decision with m666 AppWorld diagnostics. The "
            "schema-planner control cannot satisfy native learned-policy or public-artifact gates."
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
    parser.add_argument("--weight-ablation", type=Path, required=True)
    parser.add_argument("--warm-weight", type=Path, required=True)
    parser.add_argument("--random-weight", type=Path, required=True)
    parser.add_argument("--rl-receipt", type=Path, required=True)
    parser.add_argument("--mobilegym", type=Path, required=True)
    parser.add_argument("--browsergym", type=Path, required=True)
    parser.add_argument("--toolsandbox", type=Path, required=True)
    parser.add_argument("--webgpu", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = assemble(
        appworld_receipt=args.appworld_receipt,
        current_checkpoint=args.current_checkpoint,
        manifest=args.manifest,
        weight_ablation=args.weight_ablation,
        warm_weight=args.warm_weight,
        random_weight=args.random_weight,
        rl_receipt=args.rl_receipt,
        mobilegym=args.mobilegym,
        browsergym=args.browsergym,
        toolsandbox=args.toolsandbox,
        webgpu=args.webgpu,
        output=args.out,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
