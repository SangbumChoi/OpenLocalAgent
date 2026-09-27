#!/usr/bin/env python3
"""Seal a checkpoint-bound local Hugging Face/WebGPU release preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from localagent.eval.demo_deploy import _bundle_identity


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
    checkpoint: Path,
    model_dir: Path,
    web_dir: Path,
    space_dir: Path,
    model_repo: str,
    space_repo: str,
    receipt_kind: str = "localagent_m673_m671_hf_space_preparation",
) -> dict[str, Any]:
    checkpoint_identity = _identity(checkpoint)
    model_config = _load(model_dir / "config.json")
    web_manifest = _load(web_dir / "bundle-manifest.json")
    space_manifest = _load(space_dir / "bundle-manifest.json")
    if model_config.get("checkpoint_sha256") != checkpoint_identity["sha256"]:
        raise ValueError("model config is not bound to checkpoint")
    if web_manifest.get("checkpoint_sha256") != checkpoint_identity["sha256"]:
        raise ValueError("WebGPU bundle is not bound to checkpoint")
    if space_manifest != web_manifest:
        raise ValueError("Space bundle manifest differs from verified WebGPU bundle")
    if web_manifest.get("parity_gate", {}).get("passed") is not True:
        raise ValueError("WebGPU parity gate did not pass")

    model_files = {
        path.name: _identity(path)
        for path in sorted(model_dir.iterdir())
        if path.is_file() and not path.is_symlink()
    }
    artifact_names = web_manifest.get("artifacts", {})
    if not isinstance(artifact_names, dict) or not artifact_names:
        raise ValueError("WebGPU manifest has no artifacts")
    web_files: dict[str, dict[str, Any]] = {}
    for name in sorted(artifact_names):
        web_path = web_dir / name
        space_path = space_dir / name
        web_identity = _identity(web_path)
        if _identity(space_path)["sha256"] != web_identity["sha256"]:
            raise ValueError(f"Space artifact differs from WebGPU artifact: {name}")
        web_files[name] = web_identity
    bundle_identity = _bundle_identity(
        [{"file": name, "bytes": item["bytes"], "sha256": item["sha256"]} for name, item in web_files.items()]
    )

    payload: dict[str, Any] = {
        "kind": receipt_kind,
        "schema_version": 1,
        "checkpoint": {
            **checkpoint_identity,
            "parameters": model_config.get("parameter_count"),
            "stage": web_manifest.get("checkpoint_stage"),
            "step": web_manifest.get("checkpoint_step"),
        },
        "intended_repositories": {"model": model_repo, "space": space_repo},
        "prepared": True,
        "published": False,
        "publication_blocker": (
            "HF authentication is absent; no Hub model or Space upload was attempted."
        ),
        "model_bundle": {"path": str(model_dir), "artifacts": model_files},
        "webgpu_bundle": {
            "path": str(web_dir),
            "bundle_manifest": _identity(web_dir / "bundle-manifest.json"),
            "bundle_identity_sha256": bundle_identity,
            "parity_passed": True,
            "artifacts": web_files,
            "tokenizer_sha256": web_manifest.get("artifacts", {})
            .get("tokenizer.json", {})
            .get("sha256"),
        },
        "space_staging": {
            "path": str(space_dir),
            "bundle_manifest_checkpoint_match": True,
            "static_files_copied": True,
            "bundle_artifact_count": len(web_files),
        },
        "claim_boundary": (
            "Local-only release preparation with checkpoint-bound model artifacts, static Space "
            "staging, and ONNX/PyTorch parity. No public Hub URL, native service success, or "
            "Gmail/Notion side effect is implied until authenticated upload and anonymous audit."
        ),
    }
    payload["receipt_self_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--web-dir", type=Path, required=True)
    parser.add_argument("--space-dir", type=Path, required=True)
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--space-repo", required=True)
    parser.add_argument(
        "--receipt-kind",
        default="localagent_m673_m671_hf_space_preparation",
        help="receipt kind for the sealed preparation (default: m673 m671 preparation)",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        raise SystemExit(f"refusing to overwrite receipt: {args.out}")
    payload = assemble(
        checkpoint=args.checkpoint,
        model_dir=args.model_dir,
        web_dir=args.web_dir,
        space_dir=args.space_dir,
        model_repo=args.model_repo,
        space_repo=args.space_repo,
        receipt_kind=args.receipt_kind,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
