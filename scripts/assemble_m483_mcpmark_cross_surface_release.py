#!/usr/bin/env python3
"""Seal the m482 cross-surface child, native MCP replay, and WebGPU release evidence."""

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
        raise ValueError(f"expected JSON object: {path}")
    return value


def _self_hash(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def assemble(
    *,
    transfer: Path,
    native: Path,
    verify: Path,
    capability: Path,
    checkpoint: Path,
    model_dir: Path,
    web_dir: Path,
) -> dict[str, Any]:
    transfer_payload = _load(transfer)
    native_payload = _load(native)
    verify_payload = _load(verify)
    capability_payload = _load(capability)
    manifest = _load(web_dir / "bundle-manifest.json")
    config = _load(model_dir / "config.json")
    checkpoint_identity = _identity(checkpoint)
    if transfer_payload.get("kind") != "localagent_mcpmark_cross_surface_transfer_receipt":
        raise ValueError("transfer receipt kind mismatch")
    if native_payload.get("kind") != "localagent_mcpmark_native_playwright_current_checkpoint":
        raise ValueError("native receipt kind mismatch")
    if verify_payload.get("kind") != "localagent_webgpu_demo_deploy_verification":
        raise ValueError("deployment verification kind mismatch")
    if verify_payload.get("verified") is not True:
        raise ValueError("WebGPU deployment bundle is not verified")
    if capability_payload.get("kind") != "localagent_webgpu_native_capability_receipt":
        raise ValueError("capability receipt kind mismatch")
    if transfer_payload["parent"]["sha256"] != "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45":
        raise ValueError("unexpected transfer parent")
    if native_payload["model"]["sha256"] != checkpoint_identity["sha256"]:
        raise ValueError("native receipt does not bind the supplied transfer warm child")
    if manifest.get("checkpoint_sha256") != checkpoint_identity["sha256"]:
        raise ValueError("bundle checkpoint mismatch")
    if config.get("checkpoint_sha256") != checkpoint_identity["sha256"]:
        raise ValueError("HF config checkpoint mismatch")
    if capability_payload.get("bundle_identity", {}).get("checkpoint_sha256") != checkpoint_identity["sha256"]:
        raise ValueError("capability checkpoint mismatch")
    payload: dict[str, Any] = {
        "kind": "localagent_mcpmark_cross_surface_webgpu_evidence_receipt",
        "schema_version": 1,
        "benchmark_id": "mcpmark",
        "checkpoint": {**checkpoint_identity, "parameters": config["parameter_count"]},
        "transfer": {
            "receipt": _identity(transfer),
            "aggregate": transfer_payload["aggregate"],
            "surfaces": transfer_payload["surfaces"],
            "rows": transfer_payload["training"]["rows"],
            "decision": transfer_payload["transfer_decision"],
            "weight_groups": transfer_payload["weight_groups"],
        },
        "native_mcpmark": {
            "receipt": _identity(native),
            "runtime": native_payload["environment"],
            "summary": native_payload["summary"],
            "task": native_payload["results"][0],
            "official_split_verified": native_payload["dataset"].get("official_split_verified", False),
        },
        "webgpu": {
            "verification": _identity(verify),
            "bundle_identity_sha256": verify_payload["bundle_identity_sha256"],
            "manifest": _identity(web_dir / "bundle-manifest.json"),
            "parity_gate_passed": manifest.get("parity_gate", {}).get("passed"),
            "capability": _identity(capability),
            "backend": capability_payload.get("backend"),
            "hardware_adapter": capability_payload.get("hardware_adapter"),
            "tokens_per_second_p50": capability_payload.get("performance", {}).get("tokens_per_second_p50"),
            "latency_ms_p50": capability_payload.get("performance", {}).get("latency_ms_p50"),
            "peak_memory_mb": capability_payload.get("performance", {}).get("peak_memory_mb"),
        },
        "publication": {
            "published": False,
            "hf_authenticated": False,
            "model_url": None,
            "space_url": None,
            "blocker": "HF_TOKEN_or_hf_auth_login_required",
        },
        "claim_boundary": (
            "The larger public MCPMark cross-surface child has a matched transfer audit, native "
            "Playwright MCP/verifier replay, and parity-verified Apple WebGPU bundle. Native task "
            "success is zero, the official split is unverified, no real productivity side effect is "
            "claimed, and the model/Space remain unpublished without HF authentication."
        ),
    }
    payload["receipt_self_sha256"] = _self_hash(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--verify", type=Path, required=True)
    parser.add_argument("--capability", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--web-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    payload = assemble(
        transfer=args.transfer,
        native=args.native,
        verify=args.verify,
        capability=args.capability,
        checkpoint=args.checkpoint,
        model_dir=args.model_dir,
        web_dir=args.web_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
