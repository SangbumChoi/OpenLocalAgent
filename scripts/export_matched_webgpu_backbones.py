#!/usr/bin/env python
"""Export deterministic random hybrid/attention backbones for latency measurement only.

These graphs are intentionally untrained and expose hidden states only. They cannot generate
language or actions and must never be used as evidence of model capability or quality.

Example:
  python scripts/export_matched_webgpu_backbones.py \
    --out runs/webgpu/random-backbone-latency
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from localagent.inference.export.to_onnx import export_matched_random_backbones

DEFAULT_HYBRID_CONFIG = "configs/model/webgpu-35m-hybrid.yaml"
DEFAULT_ATTENTION_CONFIG = "configs/model/webgpu-35m-attn.yaml"
DEFAULT_SEED = 20260728
WARNING = "UNTRAINED RANDOM WEIGHTS — LATENCY ONLY; NOT A CAPABILITY OR QUALITY ARTIFACT."


def _fixture_lengths(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from exc
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a matched, deterministic random-weight ONNX backbone pair for local "
            "WebGPU/WASM latency timing. No capability claims are valid for these artifacts."
        )
    )
    parser.add_argument("--hybrid-config", default=DEFAULT_HYBRID_CONFIG)
    parser.add_argument("--attention-config", default=DEFAULT_ATTENTION_CONFIG)
    parser.add_argument("--out", default="runs/webgpu/random-backbone-latency")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--fixture-lengths",
        type=_fixture_lengths,
        default=(1, 8, 31),
        help="comma-separated sequence lengths used for mandatory ONNX/PyTorch parity",
    )
    parser.add_argument(
        "--fp32-only",
        action="store_true",
        help="omit the fp16 WebGPU timing graph (fp32 parity remains mandatory)",
    )
    parser.add_argument(
        "--parameter-tolerance",
        type=float,
        default=0.01,
        help="maximum relative parameter-count mismatch for the pair (default: 0.01)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(WARNING)
    result = export_matched_random_backbones(
        args.hybrid_config,
        args.attention_config,
        args.out,
        seed=args.seed,
        fp16=not args.fp32_only,
        opset=args.opset,
        fixture_lengths=args.fixture_lengths,
        parameter_tolerance=args.parameter_tolerance,
    )
    for role in ("hybrid", "attention"):
        exported = result[role]
        print(
            f"{role}: {exported['model_parameters']:,} parameters, "
            f"state={exported['state_dict_sha256']}"
        )
        for name, parity in exported["parity"].items():
            graph = Path(exported["fp16_path"] if "fp16" in name else exported["fp32_path"])
            print(
                f"  {graph}: {graph.stat().st_size / 1e6:.2f} MB, "
                f"max|Δhidden|={parity['max_abs_diff']:.3e} <= {parity['atol']:.3e}"
            )
    print(f"pair manifest: {result['manifest_path']}")
    print(WARNING)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
