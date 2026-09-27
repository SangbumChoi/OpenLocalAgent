#!/usr/bin/env python3
"""Record a parity-verified local HF/WebGPU release without claiming publication."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CHECKPOINT_SHA256 = "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"


def identity(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--web-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify = json.loads(args.verify.read_text(encoding="utf-8"))
    if verify.get("kind") != "localagent_webgpu_demo_deploy_verification":
        raise ValueError("unexpected demo verification kind")
    if verify.get("verified") is not True:
        raise ValueError("cannot record an unverified local release")
    manifest = json.loads((args.web_dir / "bundle-manifest.json").read_text(encoding="utf-8"))
    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    if manifest.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("WebGPU manifest is not bound to the current checkpoint")
    if config.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("HF config is not bound to the current checkpoint")
    payload = {
        "kind": "localagent_hf_webgpu_local_release_receipt",
        "schema_version": 1,
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": CHECKPOINT_SHA256,
            "bytes": args.checkpoint.stat().st_size,
            "parameters": config["parameter_count"],
        },
        "model_bundle": {
            "path": str(args.model_dir),
            "config": identity(args.model_dir / "config.json"),
            "files": sorted(path.name for path in args.model_dir.iterdir() if path.is_file()),
        },
        "webgpu_bundle": {
            "path": str(args.web_dir),
            "manifest": identity(args.web_dir / "bundle-manifest.json"),
            "bundle_identity_sha256": verify["bundle_identity_sha256"],
            "parity_gate_passed": manifest.get("parity_gate", {}).get("passed"),
            "files": sorted(path.name for path in args.web_dir.iterdir() if path.is_file()),
        },
        "publication": {
            "published": False,
            "hf_authenticated": False,
            "model_url": None,
            "space_url": None,
            "blocker": "HF_TOKEN_or_hf_auth_login_required",
        },
        "claim_boundary": (
            "The exact current checkpoint has a locally verified HF-format model bundle and static "
            "WebGPU bundle with ONNX parity. No Hugging Face upload, anonymous re-fetch, public "
            "URL, hardware WebGPU task score, browser account, email, or Notion claim is made."
        ),
        "source_verification": identity(args.verify),
    }
    payload["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["publication"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
