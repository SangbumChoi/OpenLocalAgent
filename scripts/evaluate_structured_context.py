#!/usr/bin/env python3
"""Audit frozen SFT route/selector heads at natural and browser-fixed context lengths."""

from __future__ import annotations

import argparse

from localagent.eval.structured_context import build_context_audit, write_context_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="runs/sft-webgpu-proxy-pilot-hybrid-seed2027/latest.pt",
    )
    parser.add_argument(
        "--tokenizer",
        default="data/tokenizer-webgpu-proxy-16k.json",
    )
    parser.add_argument(
        "--agent-eval",
        default="data/synth/agent_eval_pilot65.jsonl",
    )
    parser.add_argument(
        "--action-suite",
        default="spaces/localagent-webgpu/benchmark-cases.json",
    )
    parser.add_argument(
        "--fixed-context",
        action="append",
        type=int,
        dest="fixed_contexts",
        help="Exact tokenizer length to audit; repeat for multiple lengths (default: 128, 512).",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        default=(
            "docs/paper/results/"
            "sft-structured-context-robustness-seed2027.summary.json"
        ),
    )
    args = parser.parse_args()
    payload = build_context_audit(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        agent_eval_path=args.agent_eval,
        action_suite_path=args.action_suite,
        fixed_contexts=args.fixed_contexts or (128, 512),
        batch_size=args.batch_size,
        device=args.device,
    )
    write_context_audit(payload, args.output)
    print(
        f"wrote {args.output} "
        f"(summary_sha256={payload['summary_sha256']})"
    )


if __name__ == "__main__":
    main()
