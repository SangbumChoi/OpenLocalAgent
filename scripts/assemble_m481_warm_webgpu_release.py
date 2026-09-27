#!/usr/bin/env python3
"""Seal a parity-verified warm-child HF/WebGPU release candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _self_hash(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def assemble(
    *,
    checkpoint: Path,
    model_dir: Path,
    web_dir: Path,
    verify: Path,
    capability: Path,
    native_replay: Path,
    output: Path,
) -> dict[str, Any]:
    verification = _load(verify)
    manifest = _load(web_dir / "bundle-manifest.json")
    config = _load(model_dir / "config.json")
    capability_payload = _load(capability)
    native_payload = _load(native_replay)
    checkpoint_identity = _identity(checkpoint)
    if verification.get("kind") != "localagent_webgpu_demo_deploy_verification":
        raise ValueError("deployment verification kind mismatch")
    if verification.get("verified") is not True:
        raise ValueError("deployment bundle is not verified")
    if manifest.get("checkpoint_sha256") != checkpoint_identity["sha256"]:
        raise ValueError("WebGPU manifest checkpoint mismatch")
    if config.get("checkpoint_sha256") != checkpoint_identity["sha256"]:
        raise ValueError("HF config checkpoint mismatch")
    if capability_payload.get("kind") != "localagent_webgpu_native_capability_receipt":
        raise ValueError("capability receipt kind mismatch")
    if capability_payload.get("bundle_identity", {}).get("checkpoint_sha256") != checkpoint_identity["sha256"]:
        raise ValueError("capability receipt checkpoint mismatch")
    if native_payload.get("kind") != "localagent_mcpmark_native_playwright_replay_receipt":
        raise ValueError("native replay kind mismatch")
    payload: dict[str, Any] = {
        "kind": "localagent_warm_child_hf_webgpu_release_candidate",
        "schema_version": 1,
        "checkpoint": {
            **checkpoint_identity,
            "parameters": config["parameter_count"],
            "stage": manifest.get("checkpoint_stage"),
            "step": manifest.get("checkpoint_step"),
        },
        "model_bundle": {
            "path": str(model_dir),
            "config": _identity(model_dir / "config.json"),
            "files": sorted(path.name for path in model_dir.iterdir() if path.is_file()),
        },
        "webgpu_bundle": {
            "path": str(web_dir),
            "manifest": _identity(web_dir / "bundle-manifest.json"),
            "bundle_identity_sha256": verification["bundle_identity_sha256"],
            "parity_gate_passed": manifest.get("parity_gate", {}).get("passed"),
            "files": sorted(path.name for path in web_dir.iterdir() if path.is_file()),
        },
        "native_webgpu": {
            "receipt": _identity(capability),
            "environment_executed": capability_payload.get("environment_executed"),
            "backend": capability_payload.get("backend"),
            "hardware_adapter": capability_payload.get("hardware_adapter"),
            "tokens_per_second_p50": capability_payload.get("performance", {}).get(
                "tokens_per_second_p50"
            ),
            "latency_ms_p50": capability_payload.get("performance", {}).get("latency_ms_p50"),
            "peak_memory_mb": capability_payload.get("performance", {}).get("peak_memory_mb"),
            "exact_actions": capability_payload.get("capability", {}).get("exact_actions"),
            "evaluated_cases": capability_payload.get("capability", {}).get("evaluated_cases"),
            "closed_loop_success": capability_payload.get("capability", {}).get(
                "closed_loop_success"
            ),
        },
        "native_mcpmark": {
            "receipt": _identity(native_replay),
            "warm_verifier_pass": native_payload.get("comparison", {}).get("warm_verifier_pass"),
            "official_split_verified": native_payload.get("source", {}).get(
                "official_split_verified"
            ),
        },
        "publication": {
            "published": False,
            "hf_authenticated": False,
            "model_url": None,
            "space_url": None,
            "blocker": "HF_TOKEN_or_hf_auth_login_required",
        },
        "claim_boundary": (
            "The warm child has a local HF-format model bundle, parity-verified WebGPU bundle, and "
            "native Apple WebGPU capability receipt. No Hugging Face upload, public URL, official "
            "MCPMark score, real email/Notion account, or closed-loop productivity success is claimed."
        ),
    }
    payload["receipt_self_sha256"] = _self_hash(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--web-dir", type=Path, required=True)
    parser.add_argument("--verify", type=Path, required=True)
    parser.add_argument("--capability", type=Path, required=True)
    parser.add_argument("--native-replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    payload = assemble(
        checkpoint=args.checkpoint,
        model_dir=args.model_dir,
        web_dir=args.web_dir,
        verify=args.verify,
        capability=args.capability,
        native_replay=args.native_replay,
        output=args.output,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
