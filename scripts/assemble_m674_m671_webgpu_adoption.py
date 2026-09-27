#!/usr/bin/env python3
"""Bind native WebGPU runtime evidence to a local m671 release preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _identity(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"expected regular file: {path}")
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def assemble(
    *,
    preparation: Path,
    capability: Path,
    checkpoint: Path,
    adoption_kind: str = "localagent_m674_m671_webgpu_adoption",
    child_label: str = "m671",
) -> dict[str, Any]:
    prep = _load(preparation)
    runtime = _load(capability)
    checkpoint_identity = _identity(checkpoint)
    checkpoint_sha = checkpoint_identity["sha256"]
    preparation_kind = str(prep.get("kind", ""))
    if not preparation_kind.endswith("_hf_space_preparation"):
        raise ValueError("release preparation kind mismatch")
    if prep.get("checkpoint", {}).get("sha256") != checkpoint_sha:
        raise ValueError("release preparation checkpoint mismatch")
    if runtime.get("kind") != "localagent_webgpu_native_capability_receipt":
        raise ValueError("WebGPU capability kind mismatch")
    if runtime.get("checkpoint", {}).get("sha256") != checkpoint_sha:
        raise ValueError("WebGPU capability checkpoint mismatch")
    if runtime.get("environment_executed") is not True or runtime.get("backend") != "webgpu":
        raise ValueError("native WebGPU runtime was not verified")
    capability_metrics = runtime.get("capability", {})
    if capability_metrics.get("evaluated_cases") != 3 or capability_metrics.get("exact_actions") != 3:
        raise ValueError("native WebGPU capability cases are incomplete")
    if capability_metrics.get("external_side_effects_executed") is not False:
        raise ValueError("WebGPU side-effect safety is not explicit")

    payload: dict[str, Any] = {
        "kind": adoption_kind,
        "schema_version": 1,
        "checkpoint": checkpoint_identity,
        "release_preparation": {
            "receipt": _identity(preparation),
            "model_repo": prep["intended_repositories"]["model"],
            "space_repo": prep["intended_repositories"]["space"],
            "prepared": prep["prepared"],
            "published": prep["published"],
            "bundle_identity_sha256": prep["webgpu_bundle"]["bundle_identity_sha256"],
        },
        "native_webgpu": {
            "receipt": _identity(capability),
            "environment_executed": True,
            "adapter": runtime.get("hardware_adapter"),
            "tokens_per_second_p50": runtime["performance"]["tokens_per_second_p50"],
            "latency_ms_p50": runtime["performance"]["latency_ms_p50"],
            "peak_memory_mb": runtime["performance"]["peak_memory_mb"],
            "evaluated_cases": capability_metrics["evaluated_cases"],
            "exact_actions": capability_metrics["exact_actions"],
            "closed_loop_success": capability_metrics["closed_loop_success"],
            "external_side_effects_executed": capability_metrics["external_side_effects_executed"],
        },
        "adoption": {
            "local_webgpu_adopted": True,
            "public_model_published": False,
            "public_space_published": False,
            "publication_blockers": [
                "authenticated Hugging Face upload has not been executed",
                "native visual/service-backed task gates remain open",
            ],
        },
        "claim_boundary": (
            f"The {child_label} child is locally export-, parity-, and native-WebGPU verified. The three "
            "actions are structured local predictions only; no real email, browser navigation, "
            "Notion account, MCP server, or external side effect ran."
        ),
    }
    payload["receipt_self_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--capability", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adoption-kind", default="localagent_m674_m671_webgpu_adoption")
    parser.add_argument("--child-label", default="m671")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        raise SystemExit(f"refusing to overwrite receipt: {args.out}")
    payload = assemble(
        preparation=args.preparation,
        capability=args.capability,
        checkpoint=args.checkpoint,
        adoption_kind=args.adoption_kind,
        child_label=args.child_label,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
