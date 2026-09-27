#!/usr/bin/env python3
"""Bind a browser-observed WebGPU demo probe to the verified local bundle.

This receipt deliberately records both deployment behavior and a quality negative control.  A
browser loading the ONNX artifacts and preserving confirmation boundaries is not the same as
successful email, Notion, or general question answering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from localagent.eval.demo_deploy import verify_demo_deploy


KIND = "localagent_m609_webgpu_local_demo_probe"


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def assemble(demo_dir: Path, observations_path: Path, *, expected_tool_count: int = 63) -> dict[str, Any]:
    deployment = verify_demo_deploy(
        demo_dir,
        expected_tool_count=expected_tool_count,
    )
    observations = json.loads(observations_path.read_text(encoding="utf-8"))
    if not isinstance(observations, dict):
        raise ValueError("observations must be a JSON object")
    if observations.get("backend") != "webgpu":
        raise ValueError("probe must be observed with the WebGPU backend")
    if observations.get("model_ready") is not True:
        raise ValueError("probe did not reach model-ready state")
    if observations.get("side_effects_executed") is not False:
        raise ValueError("probe must not execute external side effects")
    if observations.get("plan", {}).get("step_count") != 2:
        raise ValueError("expected the two-step search-to-Notion plan")
    if observations.get("plan", {}).get("confirmation_required") is not True:
        raise ValueError("Notion write was not held behind confirmation")
    if observations.get("email", {}).get("confirmation_required") is not True:
        raise ValueError("email write was not held behind confirmation")
    if observations.get("quality_negative_control", {}).get("passed") is not False:
        raise ValueError("quality negative control must remain a failed semantic probe")

    body: dict[str, Any] = {
        "kind": KIND,
        "schema_version": 1,
        "deployment": deployment,
        "observations": observations,
        "observation_input": _identity(observations_path),
        "decision": {
            "local_bundle_runtime_verified": bool(deployment["verified"]),
            "confirmation_boundary_verified": True,
            "external_side_effects_verified_absent": True,
            "public_hf_or_space_deployment": False,
            "publish_ready": False,
            "quality_gate": False,
        },
        "claim_boundary": (
            "Local browser observation only. The exact static bundle loaded through WebGPU and the "
            "demo held email/Notion writes behind confirmation, but no account, API, browser-side "
            "external effect, public Hub upload, or official benchmark task ran. The semantic "
            "negative control misrouted a definition question to a click action; this is deployment "
            "evidence and a quality failure, not a task-success score."
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
