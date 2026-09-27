#!/usr/bin/env python3
"""Bind exported-child WebGPU trajectory evidence without overstating capability.

The browser payload is a local synthetic state machine.  This receipt binds the exact child
checkpoint, exporter manifest, and in-app-browser JSON so a future rerun can distinguish graph
parity from real email/Notion/browser task success.
"""

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


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--browser-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite receipt: {args.output}")

    manifest_path = args.bundle_dir / "bundle-manifest.json"
    manifest = _load(manifest_path)
    browser = _load(args.browser_output)
    meta = _load(args.bundle_dir / "meta.json")
    artifact_names = sorted(
        name for name in manifest.get("artifacts", {}) if name != "bundle-manifest.json"
    )
    artifacts = {
        name: _identity(args.bundle_dir / name)
        for name in artifact_names
        if (args.bundle_dir / name).is_file()
    }
    receipt: dict[str, Any] = {
        "kind": "localagent_webgpu_stateful_current_export_trajectory_receipt",
        "schema_version": 1,
        "suite": browser.get("suite"),
        "checkpoint": _identity(args.checkpoint),
        "bundle": {
            "directory": str(args.bundle_dir),
            "manifest": _identity(manifest_path),
            "manifest_sha256": _identity(manifest_path)["sha256"],
            "checkpoint_sha256": manifest.get("checkpoint_sha256"),
            "model_parameters": meta.get("model_parameters"),
            "artifacts": artifacts,
            "parity_gate": manifest.get("parity_gate"),
        },
        "browser": {
            "surface": "codex_in_app_browser",
            "provider_requested": "webgpu",
            "observed_backend": "WEBGPU",
            "network": "local_http_only",
            "external_accounts": False,
            "trusted_os_input": False,
            "browser_output": _identity(args.browser_output),
            "summary": browser,
        },
        "structured_action_parity": {
            "passed": False,
            "scope": "offline CPU full-stack diagnostic on the reused 20-case suite",
            "failure": (
                "fp16 pointer-score numerical gate exceeded 1.0 (maximum 1.1609694053373119); "
                "all fp16 route/tool/grounded-action exact gates still matched"
            ),
            "claim_boundary": (
                "This is not a quality failure of the browser trajectory suite, and it is not "
                "a reason to relax the hard parity threshold without an explicit review."
            ),
        },
        "decision": "diagnostic_only",
        "claim_boundary": (
            "Current-checkpoint ONNX/WebGPU export and an in-app WebGPU replay of three local "
            "synthetic email, Notion, and browser workflows. It is not AndroidWorld, BrowserGym, "
            "OSWorld, MCPMark, EnterpriseOps-Gym, real email or Notion access, screenshot grounding, "
            "trusted computer control, native device reward, or a public Hugging Face deployment."
        ),
    }
    receipt["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
