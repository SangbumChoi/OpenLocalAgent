#!/usr/bin/env python
"""Verify one single-checkpoint WebGPU acceptance result and emit a canonical receipt."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Sequence

from localagent.eval.webgpu_decode_receipt import (
    build_webgpu_decode_receipt,
    read_stable_webgpu_evidence_file,
    write_webgpu_decode_receipt,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="raw JSON downloaded from decode-benchmark.html")
    parser.add_argument("--output", type=Path, required=True, help="canonical receipt output path")
    parser.add_argument(
        "--expected-checkpoint-sha256",
        required=True,
        help="externally supplied accepted checkpoint digest that the result must match",
    )
    parser.add_argument(
        "--expected-wrapper-manifest-sha256",
        required=True,
        help=(
            "externally supplied single-decode.json SHA-256 root; the browser result and "
            "embedded manifest must both match it"
        ),
    )
    parser.add_argument("--expected-run-challenge", required=True)
    parser.add_argument("--expected-machine-condition-sha256", required=True)
    parser.add_argument("--expected-harness-html-sha256", required=True)
    parser.add_argument("--expected-harness-javascript-sha256", required=True)
    parser.add_argument("--expected-ort-javascript-sha256", required=True)
    parser.add_argument("--expected-ort-wasm-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite receipt output: {args.output}")
    payload = read_stable_webgpu_evidence_file(
        args.result,
        label="WebGPU decode result",
    )
    receipt = build_webgpu_decode_receipt(
        payload,
        expected_wrapper_manifest_sha256=args.expected_wrapper_manifest_sha256,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_run_challenge=args.expected_run_challenge,
        expected_machine_condition_sha256=args.expected_machine_condition_sha256,
        expected_harness_html_sha256=args.expected_harness_html_sha256,
        expected_harness_javascript_sha256=args.expected_harness_javascript_sha256,
        expected_ort_javascript_sha256=args.expected_ort_javascript_sha256,
        expected_ort_wasm_sha256=args.expected_ort_wasm_sha256,
    )
    write_webgpu_decode_receipt(args.output, receipt)
    receipt_payload = read_stable_webgpu_evidence_file(
        args.output,
        label="published WebGPU decode receipt",
    )
    print(f"verified result: {hashlib.sha256(payload).hexdigest()}")
    print(f"external wrapper root: {args.expected_wrapper_manifest_sha256}")
    print(f"canonical receipt: {args.output}")
    print(f"receipt file SHA-256: {hashlib.sha256(receipt_payload).hexdigest()}")
    print(f"receipt self SHA-256: {receipt['receipt_self_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
