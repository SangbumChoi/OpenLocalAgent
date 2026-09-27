#!/usr/bin/env python3
"""Hard-gate the complete corrected browser structured-action export path."""

from __future__ import annotations

import argparse

from localagent.inference.export.structured_action_parity import (
    build_structured_action_parity,
    write_structured_action_parity,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="runs/sft-webgpu-proxy-pilot-hybrid-seed2027/latest.pt",
    )
    parser.add_argument(
        "--bundle-dir",
        default="runs/sft-webgpu-proxy-pilot-hybrid-seed2027/web",
    )
    parser.add_argument(
        "--action-suite",
        default="spaces/localagent-webgpu/benchmark-cases.json",
    )
    parser.add_argument("--app-js", default="spaces/localagent-webgpu/app.js")
    parser.add_argument(
        "--benchmark-js",
        default="spaces/localagent-webgpu/benchmark.js",
    )
    parser.add_argument("--target-input-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--node", default="node")
    parser.add_argument(
        "--output",
        default=(
            "docs/paper/results/"
            "sft-structured-export-parity-seed2027.summary.json"
        ),
    )
    args = parser.parse_args()
    payload = build_structured_action_parity(
        checkpoint_path=args.checkpoint,
        bundle_dir=args.bundle_dir,
        action_suite_path=args.action_suite,
        app_js_path=args.app_js,
        benchmark_js_path=args.benchmark_js,
        target_input_tokens=args.target_input_tokens,
        batch_size=args.batch_size,
        node_executable=args.node,
    )
    write_structured_action_parity(payload, args.output)
    aggregate = payload["aggregate"]
    print(
        f"wrote {args.output} "
        f"(passed={payload['passed']}, cases={aggregate['eligible_cases']}, "
        f"summary_sha256={payload['summary_sha256']})"
    )


if __name__ == "__main__":
    main()
