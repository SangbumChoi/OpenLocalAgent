#!/usr/bin/env python
"""Normalize a bounded, train-only slice of official Android-in-the-Wild TFRecords."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from localagent.data.aitw import episode_to_intermediate, iter_aitw_episodes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--source-subset", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-full-bytes", type=int, required=True)
    parser.add_argument("--source-full-md5", required=True)
    parser.add_argument("--range-start", type=int, default=0)
    parser.add_argument("--range-end", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--allow-truncated", action="store_true")
    parser.add_argument("--no-crc", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.source_full_bytes < 1 or args.range_start < 0 or args.range_end < args.range_start:
        raise ValueError("invalid source/range identity")
    if args.max_records < 0:
        raise ValueError("max-records must be non-negative")
    split_payload = args.splits.read_bytes()
    split_data = json.loads(split_payload.decode("utf-8"))
    if not isinstance(split_data, dict) or not isinstance(split_data.get(args.split), list):
        raise ValueError("AITW split manifest must contain train/validation/test arrays")
    selected_ids = {int(value) for value in split_data[args.split]}
    rows: list[dict[str, Any]] = []
    seen_episodes = 0
    for episode in iter_aitw_episodes(
        args.input,
        allow_truncated=args.allow_truncated,
        verify_crc=not args.no_crc,
    ):
        if int(episode["episode_id"]) not in selected_ids:
            continue
        rows.append(
            episode_to_intermediate(
                episode,
                source_revision=args.source_revision,
                split=args.split,
            )
        )
        seen_episodes += 1
        if args.max_records and seen_episodes >= args.max_records:
            break
    if not rows:
        raise ValueError(f"no complete {args.split} episodes found in {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    local_bytes = args.input.stat().st_size
    local_sha = _sha256(args.input)
    output_sha = _sha256(args.output)
    manifest: dict[str, Any] = {
        "kind": "localagent_aitw_acquisition_manifest",
        "schema_version": 1,
        "dataset": "Android in the Wild",
        "dataset_url": "https://github.com/google-research/google-research/tree/master/android_in_the_wild",
        "dataset_license": "CC-BY-4.0",
        "source_object": {
            "bucket": "gresearch",
            "name": f"android-in-the-wild/{args.source_subset}",
            "bytes": args.source_full_bytes,
            "md5": args.source_full_md5,
            "revision": args.source_revision,
        },
        "local_range": {
            "path": str(args.input),
            "bytes": local_bytes,
            "sha256": local_sha,
            "start": args.range_start,
            "end_inclusive": args.range_end,
            "truncated_gzip_allowed": bool(args.allow_truncated),
        },
        "split_file": {
            "path": str(args.splits),
            "sha256": _sha256(args.splits),
            "selected_split": args.split,
            "selected_ids": len(selected_ids),
            "counts": {key: len(value) for key, value in sorted(split_data.items()) if isinstance(value, list)},
        },
        "normalization": {
            "interchange": "localagent_v1",
            "module": "localagent.data.aitw",
            "screen_projection": "aitw_annotations_text_v1",
            "version": 1,
        },
        "records": {
            "selected": len(rows),
            "episode_ids": [int(row["quality"]["source_episode_id"]) for row in rows],
            "output_path": str(args.output),
            "output_sha256": output_sha,
        },
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
