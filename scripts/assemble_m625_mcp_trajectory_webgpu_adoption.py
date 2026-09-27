#!/usr/bin/env python3
"""Seal the m624 warm child as a locally verified WebGPU adoption candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


TRANSFER_KIND = "localagent_mcp_trajectory_transfer_v1"
DEPLOY_KIND = "localagent_webgpu_demo_deploy_verification"
CAPABILITY_KIND = "localagent_webgpu_native_capability_receipt"


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _self_hash(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(_canonical(body)).hexdigest()


def assemble(
    *,
    checkpoint: Path,
    transfer: Path,
    deployment: Path,
    capability_path: Path,
    model_dir: Path,
    web_dir: Path,
) -> dict[str, Any]:
    transfer_payload = _load(transfer)
    deployment_payload = _load(deployment)
    capability_payload = _load(capability_path)
    manifest = _load(web_dir / "bundle-manifest.json")
    model_config = _load(model_dir / "config.json")
    checkpoint_identity = _identity(checkpoint)

    if transfer_payload.get("kind") != TRANSFER_KIND:
        raise ValueError("m624 transfer receipt kind mismatch")
    if deployment_payload.get("kind") != DEPLOY_KIND or deployment_payload.get("verified") is not True:
        raise ValueError("deployment receipt is not verified")
    if capability_payload.get("kind") != CAPABILITY_KIND:
        raise ValueError("native capability receipt kind mismatch")
    if manifest.get("checkpoint_sha256") != checkpoint_identity["sha256"]:
        raise ValueError("WebGPU manifest checkpoint mismatch")
    if model_config.get("checkpoint_sha256") != checkpoint_identity["sha256"]:
        raise ValueError("model config checkpoint mismatch")
    if capability_payload.get("bundle_identity", {}).get("checkpoint_sha256") != checkpoint_identity["sha256"]:
        raise ValueError("native capability checkpoint mismatch")
    if deployment_payload.get("manifest", {}).get("sha256") != _identity(web_dir / "bundle-manifest.json")["sha256"]:
        raise ValueError("deployment manifest hash mismatch")

    warm_arm = transfer_payload["arms"]["warm"]
    capability = capability_payload["capability"]
    performance = capability_payload["performance"]
    payload: dict[str, Any] = {
        "kind": "localagent_m625_mcp_trajectory_webgpu_adoption",
        "schema_version": 1,
        "source": {
            **transfer_payload["source"],
            "transfer_receipt": _identity(transfer),
        },
        "training": {
            "transfer_receipt_kind": transfer_payload["kind"],
            "warm_checkpoint": warm_arm["checkpoint"],
            "warm_eval_token_accuracy_before": warm_arm["before_eval"]["assistant_token_accuracy"],
            "warm_eval_token_accuracy_after": warm_arm["after_eval"]["assistant_token_accuracy"],
            "warm_eval_gain": transfer_payload["comparison"]["warm_eval_gain"],
            "warm_minus_random_after": transfer_payload["comparison"]["warm_minus_random_after"],
            "action_heads_frozen": warm_arm["weight_transfer"]["groups"]["action_heads"]["delta_l2"] == 0.0,
        },
        "checkpoint": {
            **checkpoint_identity,
            "parameters": manifest["model_parameters"],
            "config_name": manifest["config_name"],
            "stage": manifest["checkpoint_stage"],
            "step": manifest["checkpoint_step"],
        },
        "model_bundle": {
            "path": str(model_dir),
            "config": _identity(model_dir / "config.json"),
            "weights": _identity(model_dir / "model.safetensors"),
            "parameters": model_config["parameter_count"],
        },
        "webgpu_bundle": {
            "path": str(web_dir),
            "manifest": _identity(web_dir / "bundle-manifest.json"),
            "bundle_identity_sha256": deployment_payload["bundle_identity_sha256"],
            "artifact_count": len(deployment_payload["artifacts"]),
            "tool_count": len(_load(web_dir / "meta.json").get("tools", [])),
            "parity_gate_passed": manifest.get("parity_gate", {}).get("passed") is True,
            "deployment_verified": deployment_payload["verified"] is True,
        },
        "native_webgpu": {
            "receipt": _identity(capability_path),
            "environment_executed": capability_payload.get("environment_executed") is True,
            "backend": capability_payload.get("backend"),
            "hardware_adapter": capability_payload.get("hardware_adapter"),
            "requested_provider": capability_payload.get("runtime", {}).get("requested_provider"),
            "session_provider_retry": capability_payload.get("runtime", {}).get("session_provider_retry"),
            "tokens_per_second_p50": performance.get("tokens_per_second_p50"),
            "latency_ms_p50": performance.get("latency_ms_p50"),
            "peak_memory_mb": performance.get("peak_memory_mb"),
            "exact_actions": capability.get("exact_actions"),
            "evaluated_cases": capability.get("evaluated_cases"),
            "closed_loop_success": capability.get("closed_loop_success"),
            "external_side_effects_executed": capability.get("external_side_effects_executed"),
        },
        "adoption": {
            "local_webgpu_adopted": True,
            "export_parity_verified": manifest.get("parity_gate", {}).get("passed") is True,
            "native_capability_verified": capability_payload.get("environment_executed") is True,
            "public_model_published": False,
            "public_space_published": False,
            "publication_blockers": [
                "authenticated Hugging Face model and Space upload has not been authorized or executed",
                "native Android/iOS/desktop and service-backed email/Notion benchmark gates remain open",
            ],
        },
        "claim_boundary": (
            "This is a local WebGPU adoption candidate, not a public release or end-to-end agent score. "
            "The m624 warm child is bound to a parity-verified bundle and a native Chromium Apple Metal-3 "
            "WebGPU capability receipt. The three actions are local structured predictions only: no real "
            "email, browser navigation, Notion account, MCP server, or external side effect was executed."
        ),
    }
    payload["receipt_self_sha256"] = _self_hash(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--capability", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--web-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite receipt: {args.out}")
    receipt = assemble(
        checkpoint=args.checkpoint,
        transfer=args.transfer,
        deployment=args.deployment,
        capability_path=args.capability,
        model_dir=args.model_dir,
        web_dir=args.web_dir,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
