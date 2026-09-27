#!/usr/bin/env python
"""Export a bounded, split-audited AndroidControl shard slice as localagent_v1 JSONL.

The input must be an operator-acquired GZIP shard or a bounded HTTP range of one.  This command
never downloads data and never treats a range as the complete upstream dataset.  The manifest
records both the official object identity and the local range identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from localagent.data.androidcontrol import _example_features, _int_list, example_to_intermediate
from localagent.data.androidcontrol import iter_gzip_tfrecords


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_splits(path: Path) -> dict[str, set[int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("split file must be an object")
    result: dict[str, set[int]] = {}
    for split, values in raw.items():
        if not isinstance(split, str) or not isinstance(values, list):
            raise ValueError("split file values must be lists")
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            raise ValueError(f"split {split!r} contains a non-integer episode id")
        result[split] = set(values)
    return result


def _episode_id(payload: bytes) -> int:
    values = _int_list(_example_features(payload), "episode_id")
    if not values:
        raise ValueError("AndroidControl record has no episode_id")
    return int(values[0])


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--source-shard", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-full-bytes", required=True, type=int)
    parser.add_argument("--source-full-md5", required=True)
    parser.add_argument("--range-start", type=int, default=0)
    parser.add_argument("--range-end", type=int, default=None, help="inclusive local byte offset")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--max-records", type=int, default=32)
    parser.add_argument("--allow-truncated", action="store_true")
    parser.add_argument("--no-crc", action="store_true")
    args = parser.parse_args()
    if args.source_full_bytes < 1 or args.range_start < 0:
        raise SystemExit("source-full-bytes and range-start must be non-negative/positive")
    if args.range_end is not None and args.range_end < args.range_start:
        raise SystemExit("range-end must be >= range-start")
    if args.max_records < 1:
        raise SystemExit("max-records must be positive")
    if args.output.resolve() == args.input.resolve() or args.manifest.resolve() == args.input.resolve():
        raise SystemExit("output and manifest must differ from input")

    splits = _load_splits(args.splits)
    allowed_ids = splits.get(args.split)
    if allowed_ids is None:
        raise SystemExit(f"split file has no {args.split!r} split")
    input_bytes = args.input.stat().st_size
    range_end = args.range_end if args.range_end is not None else args.range_start + input_bytes - 1
    if range_end - args.range_start + 1 != input_bytes:
        raise SystemExit("range-start/range-end do not match the local input byte size")
    split_sha256 = _sha256(args.splits)
    range_sha256 = _sha256(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    output_digest = hashlib.sha256()
    selected: list[int] = []
    records_seen = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as destination:
        for payload in iter_gzip_tfrecords(
            args.input,
            allow_truncated=args.allow_truncated,
            verify_crc=not args.no_crc,
        ):
            records_seen += 1
            episode_id = _episode_id(payload)
            if episode_id not in allowed_ids:
                continue
            row = example_to_intermediate(
                payload,
                source_revision=args.source_revision,
                source_record_index=records_seen - 1,
                split=args.split,
            )
            row["quality"].update(
                {
                    "source_shard": args.source_shard,
                    "source_full_bytes": args.source_full_bytes,
                    "source_full_md5": args.source_full_md5,
                    "source_range_start": args.range_start,
                    "source_range_end": range_end,
                    "source_range_sha256": range_sha256,
                    "source_split_file_sha256": split_sha256,
                }
            )
            encoded = _canonical_bytes(row) + b"\n"
            destination.write(encoded.decode("utf-8"))
            output_digest.update(encoded)
            selected.append(episode_id)
            if len(selected) >= args.max_records:
                break

    manifest = {
        "kind": "localagent_androidcontrol_acquisition_manifest",
        "schema_version": 1,
        "dataset": "AndroidControl",
        "dataset_url": "https://github.com/google-research/google-research/tree/master/android_control",
        "dataset_license": "Apache-2.0",
        "source_object": {
            "bucket": "gresearch",
            "name": f"android_control/{args.source_shard}",
            "revision": args.source_revision,
            "bytes": args.source_full_bytes,
            "md5": args.source_full_md5,
        },
        "local_range": {
            "path": str(args.input),
            "start": args.range_start,
            "end_inclusive": range_end,
            "bytes": input_bytes,
            "sha256": range_sha256,
            "truncated_gzip_allowed": bool(args.allow_truncated),
        },
        "split_file": {
            "path": str(args.splits),
            "sha256": split_sha256,
            "counts": {name: len(values) for name, values in sorted(splits.items())},
            "selected_split": args.split,
        },
        "normalization": {
            "module": "localagent.data.androidcontrol",
            "version": 1,
            "interchange": "localagent_v1",
            "text_projection": "android_env_proto_text_v1",
        },
        "records": {
            "tfrecords_seen": records_seen,
            "selected": len(selected),
            "episode_ids": selected,
            "output_path": str(args.output),
            "output_sha256": output_digest.hexdigest(),
        },
    }
    manifest_without_hash = dict(manifest)
    manifest["manifest_sha256"] = hashlib.sha256(_canonical_bytes(manifest_without_hash)).hexdigest()
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
