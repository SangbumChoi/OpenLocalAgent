#!/usr/bin/env python
"""Export matched prefill/decode ONNX graphs for WebGPU inference.

Without checkpoint arguments, the legacy deterministic random-weight latency workflow is
preserved. Supplying both checkpoints strict-loads trained pretraining weights; export parity is
still a hard gate, while model quality remains a separate evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from localagent.inference.export.to_onnx import export_matched_cached_decode

DEFAULT_HYBRID_CONFIG = "configs/model/webgpu-35m-hybrid.yaml"
DEFAULT_ATTENTION_CONFIG = "configs/model/webgpu-35m-attn.yaml"
DEFAULT_SEED = 20260728
WARNING = "UNTRAINED RANDOM WEIGHTS — LATENCY ONLY; NOT A CAPABILITY OR QUALITY ARTIFACT."
TRAINED_WARNING = (
    "TRAINED CHECKPOINT WEIGHTS — EXPORT PARITY ONLY; QUALITY MUST BE EVALUATED SEPARATELY."
)


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
            "Export a matched ONNX pair with separate prompt prefill and one-token "
            "cache-bearing decode graphs, using deterministic random weights by default "
            "or strict pretrain checkpoints when both are supplied."
        )
    )
    parser.add_argument("--hybrid-config", default=DEFAULT_HYBRID_CONFIG)
    parser.add_argument("--attention-config", default=DEFAULT_ATTENTION_CONFIG)
    parser.add_argument("--out", default="runs/webgpu/random-cached-decode-latency")
    parser.add_argument(
        "--hybrid-checkpoint",
        help="trained pretrain checkpoint for the hybrid arm (requires --attention-checkpoint)",
    )
    parser.add_argument(
        "--attention-checkpoint",
        help="trained pretrain checkpoint for the attention arm (requires --hybrid-checkpoint)",
    )
    parser.add_argument(
        "--tokenizer",
        help=(
            "tokenizer artifact used by both checkpoints; overrides a recorded checkpoint path "
            "and must match its SHA-256 and model vocabulary"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--fixture-lengths",
        type=_fixture_lengths,
        default=(1, 8, 31),
        help="comma-separated prompt lengths used for mandatory trajectory parity",
    )
    parser.add_argument(
        "--decode-steps",
        type=int,
        default=4,
        help="iterative one-token decode steps per parity fixture (minimum: 3)",
    )
    parser.add_argument(
        "--fp32-only",
        action="store_true",
        help="omit genuine-fp16 cache-I/O graphs",
    )
    parser.add_argument(
        "--parameter-tolerance",
        type=float,
        default=0.01,
        help="maximum relative parameter-count mismatch for the pair",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    trained = args.hybrid_checkpoint is not None or args.attention_checkpoint is not None
    warning = TRAINED_WARNING if trained else WARNING
    print(warning)
    result = export_matched_cached_decode(
        args.hybrid_config,
        args.attention_config,
        args.out,
        seed=args.seed,
        hybrid_checkpoint_path=args.hybrid_checkpoint,
        attention_checkpoint_path=args.attention_checkpoint,
        tokenizer_path=args.tokenizer,
        fp16=not args.fp32_only,
        opset=args.opset,
        fixture_lengths=args.fixture_lengths,
        decode_steps=args.decode_steps,
        parameter_tolerance=args.parameter_tolerance,
    )
    for role in ("hybrid", "attention"):
        exported = result[role]
        print(
            f"{role}: {exported['model_parameters']:,} parameters, "
            f"state={exported['state_dict_sha256']}"
        )
        weights = exported["provenance"]["weights"]
        if weights["checkpoint"] is not None:
            print(
                f"  checkpoint={weights['checkpoint']} "
                f"sha256={weights['checkpoint_sha256']} "
                f"step={weights['checkpoint_step']} "
                f"tokens={weights['tokens_seen']:,}"
            )
        for precision, parity in exported["parity"].items():
            graph_paths = exported["graph_paths"][precision]
            total_bytes = sum(Path(path).stat().st_size for path in graph_paths.values())
            print(
                f"  {precision}: {total_bytes / 1e6:.2f} MB total, "
                f"max|Δcache|={parity['max_cache_abs_diff']:.3e} "
                f"<= {parity['cache_atol']:.3e}, "
                f"{parity['decode_steps']} decode steps/fixture"
            )
    print(f"pair manifest: {result['manifest_path']}")
    print(warning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
