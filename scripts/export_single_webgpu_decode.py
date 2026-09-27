#!/usr/bin/env python
"""Export one accepted checkpoint as a parity-gated cached-decode WebGPU bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from localagent.eval.webgpu_decode_receipt import (
    RECEIPT_KIND,
    verify_webgpu_decode_receipt_bytes,
)
from localagent.inference.export.to_onnx import export_cached_decode


DEFAULT_SEED = 20260728


def _fixture_lengths(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated positive integers"
        ) from error
    if len(values) < 2 or any(value < 1 for value in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(
            "expected at least two distinct comma-separated positive integers"
        )
    return values


def _training_artifact(raw: str) -> dict[str, object]:
    try:
        path = Path(raw).expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise argparse.ArgumentTypeError(f"training artifact does not exist: {raw}") from error
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"training artifact is not a regular file: {raw}")
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    stable_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise argparse.ArgumentTypeError(f"training artifact changed while reading: {raw}")
    artifact_kind = "opaque_file"
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        value = None
    if isinstance(value, dict):
        declared_kind = value.get("kind") or value.get("artifact_type")
        if isinstance(declared_kind, str) and declared_kind:
            artifact_kind = declared_kind
        if declared_kind == RECEIPT_KIND:
            try:
                verify_webgpu_decode_receipt_bytes(payload)
            except ValueError as error:
                raise argparse.ArgumentTypeError(
                    f"invalid WebGPU acceptance receipt {raw}: {error}"
                ) from error
        elif "receipt_self_sha256" in value:
            raise argparse.ArgumentTypeError(
                f"unsupported self-hashed training receipt kind in {raw}"
            )
    return {
        "artifact_kind": artifact_kind,
        "bytes": len(payload),
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="exact model YAML used by the checkpoint")
    parser.add_argument("--checkpoint", required=True, help="accepted checkpoint path")
    parser.add_argument("--out", required=True, help="new or empty output directory")
    parser.add_argument(
        "--tokenizer",
        help="verified tokenizer override; otherwise use the checkpoint-recorded tokenizer",
    )
    parser.add_argument(
        "--pair-role",
        default="accepted_checkpoint",
        help="stable provenance role recorded in the single-model bundle",
    )
    parser.add_argument(
        "--training-artifact",
        action="append",
        default=[],
        type=_training_artifact,
        metavar="PATH",
        help=(
            "actual training input/receipt file; repeat for the exact accepted set. "
            "Files are read, semantically checked when supported, hashed, and rechecked by "
            "the exporter. At least one is mandatory for midtrain/SFT/RL checkpoints."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--fixture-lengths",
        type=_fixture_lengths,
        default=(1, 8, 31),
        help="distinct prompt lengths for mandatory trajectory parity",
    )
    parser.add_argument(
        "--decode-steps",
        type=int,
        default=4,
        help="one-token decode steps per parity fixture (minimum: 3)",
    )
    parser.add_argument(
        "--fp32-only",
        action="store_true",
        help="omit the genuine-fp16 cache-I/O graphs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seed < 0:
        raise SystemExit("--seed must be non-negative")
    if args.decode_steps < 3:
        raise SystemExit("--decode-steps must be at least three")
    training_artifacts = args.training_artifact
    artifact_paths = [identity["path"] for identity in training_artifacts]
    artifact_hashes = [identity["sha256"] for identity in training_artifacts]
    if len(set(artifact_paths)) != len(artifact_paths):
        raise SystemExit("--training-artifact paths must be unique")
    if len(set(artifact_hashes)) != len(artifact_hashes):
        raise SystemExit("--training-artifact file hashes must be unique")
    result = export_cached_decode(
        args.config,
        args.out,
        seed=args.seed,
        pair_role=args.pair_role,
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        fp16=not args.fp32_only,
        opset=args.opset,
        fixture_lengths=args.fixture_lengths,
        decode_steps=args.decode_steps,
        training_artifact_sha256=artifact_hashes or None,
        training_artifact_identities=training_artifacts or None,
        require_posttraining_training_artifacts=True,
    )
    provenance = result["provenance"]
    weights = provenance["weights"]
    print(f"bundle: {Path(args.out)}")
    print(f"single manifest: {result['single_manifest_path']}")
    print(f"checkpoint SHA-256: {weights['checkpoint_sha256']}")
    print(f"checkpoint stage: {weights['checkpoint_stage']}")
    print(f"tokenizer SHA-256: {provenance['tokenizer']['sha256']}")
    print(f"trajectory parity: {', '.join(sorted(result['parity']))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
