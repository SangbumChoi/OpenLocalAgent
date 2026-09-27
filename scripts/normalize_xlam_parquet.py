#!/usr/bin/env python3
"""Normalize public xLAM-derived Parquet shards into Conversation JSONL.

The Salesforce xLAM repository is gated in some environments.  The public
``product-science/xlam-function-calling-60k-raw`` derivative exposes the same
records as Apache-2.0 Parquet shards with explicit train/test directories.  This
converter keeps that source identity in every Conversation and writes a small
hash-bound manifest; it never downloads data and never treats a test shard as
training input.

The output is compatible with the existing cross-surface continuation trainer,
but it is deliberately not labeled as the official Salesforce split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from localagent.data.public_agent import PublicSourceSnapshot, _conversation, _xlam_record


DERIVED_DATASET = "product-science/xlam-function-calling-60k-raw"
DERIVED_REVISION = "dfbd3c669354c27f2727870d39a4d86c32381448"
DERIVED_URL = "https://huggingface.co/datasets/product-science/xlam-function-calling-60k-raw"
DERIVED_LICENSE = "apache-2.0"
DERIVED_LICENSE_URL = "https://www.apache.org/licenses/LICENSE-2.0"


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": size, "sha256": digest.hexdigest()}


def _source(path: Path, *, split: str, identity: dict[str, Any]) -> PublicSourceSnapshot:
    return PublicSourceSnapshot(
        source_id=f"xlam-derived-{split}-{path.stem}",
        dataset=DERIVED_DATASET,
        subset=split,
        revision=DERIVED_REVISION,
        url=DERIVED_URL,
        license=DERIVED_LICENSE,
        license_url=DERIVED_LICENSE_URL,
        adapter="xlam_v1",
        split=split,  # type: ignore[arg-type]
        path=path.resolve(),
        declared_path=str(path),
        bytes=int(identity["bytes"]),
        sha256=str(identity["sha256"]),
    )


def normalize(
    inputs: list[Path],
    output: Path,
    *,
    split: str,
    max_records: int = 0,
    skip_invalid: bool = False,
) -> dict[str, Any]:
    """Convert one or more Parquet shards and return a reproducibility summary."""

    if split not in {"train", "eval"}:
        raise ValueError("split must be train or eval")
    if not inputs:
        raise ValueError("at least one Parquet input is required")
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:  # pragma: no cover - environment-dependent dependency
        raise RuntimeError("pyarrow is required; install it in an isolated environment") from error

    output = output.resolve()
    if any(path.resolve() == output for path in inputs):
        raise ValueError("output must differ from every input")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_files: list[dict[str, Any]] = []
    records = 0
    invalid_rows = 0
    invalid_examples: list[dict[str, Any]] = []
    with output.open("w", encoding="utf-8", newline="\n") as destination:
        for path in inputs:
            identity = _identity(path)
            source_files.append(identity)
            source = _source(path, split=split, identity=identity)
            for source_line, raw in enumerate(parquet.read_table(path).to_pylist(), start=1):
                try:
                    record = _xlam_record(raw, source_line=source_line)
                    conversation = _conversation(
                        record,
                        source,
                        enrichment_level=0,
                        derivation="public_derivative_source",
                    )
                except (TypeError, ValueError) as error:
                    if not skip_invalid:
                        raise
                    invalid_rows += 1
                    if len(invalid_examples) < 10:
                        invalid_examples.append(
                            {
                                "file": str(path),
                                "source_line": source_line,
                                "record_id": str(raw.get("id", source_line - 1)),
                                "reason": str(error),
                            }
                        )
                    continue
                destination.write(conversation.to_json() + "\n")
                records += 1
                if max_records and records >= max_records:
                    break
            if max_records and records >= max_records:
                break
    output_identity = _identity(output)
    return {
        "kind": "localagent_xlam_derived_parquet_normalization",
        "schema_version": 1,
        "source": {
            "dataset": DERIVED_DATASET,
            "revision": DERIVED_REVISION,
            "url": DERIVED_URL,
            "license": DERIVED_LICENSE,
            "license_url": DERIVED_LICENSE_URL,
            "split": split,
            "official_salesforce_split": False,
        },
        "source_files": source_files,
        "records": records,
        "invalid_rows": invalid_rows,
        "invalid_examples": invalid_examples,
        "invalid_policy": "skip_and_record_first_10" if skip_invalid else "fail_closed",
        "output": output_identity,
        "claim_boundary": (
            "Public derivative source normalization only; not an official gated Salesforce xLAM "
            "split, benchmark score, multi-call evaluation, or live API execution."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "eval"), required=True)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="skip rows whose tool schemas/arguments violate the canonical Conversation contract",
    )
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.max_records < 0:
        raise SystemExit("--max-records must be non-negative")
    summary = normalize(
        args.input,
        args.output,
        split=args.split,
        max_records=args.max_records,
        skip_invalid=args.skip_invalid,
    )
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
