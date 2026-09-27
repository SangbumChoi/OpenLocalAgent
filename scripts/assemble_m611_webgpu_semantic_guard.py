#!/usr/bin/env python3
"""Assemble the browser-observed semantic text safety-guard receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from localagent.eval.demo_deploy import verify_demo_deploy


KIND = "localagent_m611_webgpu_semantic_guard_probe"


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def assemble(demo_dir: Path, observations_path: Path) -> dict[str, Any]:
    deployment = verify_demo_deploy(demo_dir, expected_tool_count=63)
    observations = json.loads(observations_path.read_text(encoding="utf-8"))
    app = demo_dir / "app.js"
    index = demo_dir / "index.html"
    app_text = app.read_text(encoding="utf-8")
    if "semantic_text_safety_guard" not in app_text:
        raise ValueError("semantic guard marker is absent from app.js")
    if observations.get("backend") != "webgpu" or observations.get("model_ready") is not True:
        raise ValueError("probe did not observe a WebGPU model-ready session")
    if observations.get("side_effects_executed") is not False:
        raise ValueError("probe must have zero side effects")
    if observations.get("semantic", {}).get("abstained") is not True:
        raise ValueError("semantic request was not stopped")
    if observations.get("email", {}).get("confirmation_required") is not True:
        raise ValueError("email confirmation boundary missing")
    if observations.get("plan", {}).get("notion_confirmation_required") is not True:
        raise ValueError("Notion confirmation boundary missing")
    body: dict[str, Any] = {
        "kind": KIND,
        "schema_version": 1,
        "deployment": deployment,
        "static_files": {"app.js": _identity(app), "index.html": _identity(index)},
        "observation_input": _identity(observations_path),
        "observations": observations,
        "decision": {
            "semantic_guard_verified": True,
            "email_confirmation_verified": True,
            "notion_confirmation_verified": True,
            "external_side_effects_verified_absent": True,
            "publish_ready": False,
            "official_benchmark_score": False,
        },
        "claim_boundary": (
            "Local static-bundle browser observation only. The explicit lexical semantic guard is a "
            "fail-closed deployment policy, not learned semantic competence. No account, API, MCP "
            "service, public Hub upload, native benchmark task, or external side effect ran."
        ),
    }
    body["receipt_self_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-dir", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite receipt: {args.out}")
    receipt = assemble(args.demo_dir, args.observations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
