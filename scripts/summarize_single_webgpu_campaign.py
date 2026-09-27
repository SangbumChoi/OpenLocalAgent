#!/usr/bin/env python
"""Gate three raw WebGPU results plus exact receipts and publish one campaign."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Sequence

from localagent.eval.webgpu_decode_campaign import (
    build_webgpu_decode_campaign,
    write_webgpu_decode_campaign,
)
from localagent.eval.webgpu_decode_receipt import read_stable_webgpu_evidence_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        type=Path,
        metavar=("RAW_RESULT", "RECEIPT"),
        required=True,
        help="repeat exactly three times in predetermined challenge/chronological order",
    )
    parser.add_argument("--output", type=Path, required=True, help="new campaign output path")
    parser.add_argument(
        "--expected-checkpoint-sha256",
        required=True,
        help="externally supplied checkpoint SHA-256 accepted by the campaign",
    )
    parser.add_argument(
        "--expected-wrapper-manifest-sha256",
        required=True,
        help="externally supplied single-decode.json SHA-256 accepted by the campaign",
    )
    parser.add_argument(
        "--run-challenge",
        action="append",
        required=True,
        help="repeat exactly three times in the same order as --run",
    )
    parser.add_argument("--expected-machine-condition-sha256", required=True)
    parser.add_argument("--expected-harness-html-sha256", required=True)
    parser.add_argument("--expected-harness-javascript-sha256", required=True)
    parser.add_argument("--expected-ort-javascript-sha256", required=True)
    parser.add_argument("--expected-ort-wasm-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite campaign output: {args.output}")
    if len(args.run) != 3 or len(args.run_challenge) != 3:
        raise ValueError("campaign requires exactly three --run pairs and three --run-challenge")
    campaign = build_webgpu_decode_campaign(
        [(raw, receipt) for raw, receipt in args.run],
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_wrapper_manifest_sha256=args.expected_wrapper_manifest_sha256,
        expected_run_challenges=args.run_challenge,
        expected_machine_condition_sha256=args.expected_machine_condition_sha256,
        expected_harness_html_sha256=args.expected_harness_html_sha256,
        expected_harness_javascript_sha256=args.expected_harness_javascript_sha256,
        expected_ort_javascript_sha256=args.expected_ort_javascript_sha256,
        expected_ort_wasm_sha256=args.expected_ort_wasm_sha256,
    )
    run_pairs = [(raw, receipt) for raw, receipt in args.run]
    write_webgpu_decode_campaign(
        args.output,
        campaign,
        run_pairs,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_wrapper_manifest_sha256=args.expected_wrapper_manifest_sha256,
        expected_run_challenges=args.run_challenge,
        expected_machine_condition_sha256=args.expected_machine_condition_sha256,
        expected_harness_html_sha256=args.expected_harness_html_sha256,
        expected_harness_javascript_sha256=args.expected_harness_javascript_sha256,
        expected_ort_javascript_sha256=args.expected_ort_javascript_sha256,
        expected_ort_wasm_sha256=args.expected_ort_wasm_sha256,
    )
    payload = read_stable_webgpu_evidence_file(
        args.output,
        label="published WebGPU decode campaign",
    )
    counts = campaign["counts"]
    print(f"verified campaign: {args.output}")
    print(f"campaign file SHA-256: {hashlib.sha256(payload).hexdigest()}")
    print(f"campaign self SHA-256: {campaign['campaign_self_sha256']}")
    print(
        "recomputed counts: "
        f"{counts['warmup_records']} warmups, "
        f"{counts['measured_records']} measurements, "
        f"{counts['graph_calls']} graph calls"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
