#!/usr/bin/env python3
"""Export one trained structured visual sidecar and bind CPU ONNX parity evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from localagent.inference.export.visual_action import export_visual_action_onnx


def identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.sidecar.is_file() or not args.pilot.is_file():
        raise SystemExit("sidecar and pilot receipt must exist")
    pilot = json.loads(args.pilot.read_text(encoding="utf-8"))
    if pilot.get("kind") != "localagent_m714_androidcontrol_structured_visual_pilot":
        raise ValueError("unexpected pilot receipt kind")
    parity = export_visual_action_onnx(args.sidecar, args.onnx, sequence_length=128, check=True)
    payload = {
        "kind": "localagent_m716_structured_visual_onnx_parity",
        "schema_version": 1,
        "pilot": identity(args.pilot),
        "sidecar": identity(args.sidecar),
        "onnx": parity["artifact"],
        "abi": {
            "inputs": parity["inputs"],
            "outputs": parity["outputs"],
            "action_names": parity["metadata"]["action_names"],
            "image_range": "float32 RGB [0,1]",
        },
        "parity": parity["parity"],
        "deployment_boundary": {
            "pytorch_reference": True,
            "onnx_cpu_runtime": True,
            "onnx_webgpu_runtime": False,
            "browser_demo_bound": False,
            "native_mobile_verifier": False,
        },
        "claim_boundary": (
            "This receipt proves CPU ONNX parity for a trained structured visual sidecar and its "
            "explicit image/action tensor ABI. It does not claim onnxruntime-web WebGPU execution, "
            "browser deployment, Android emulator success, or an official AndroidControl score."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["parity"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
