#!/usr/bin/env python3
"""Create a deterministic whole-episode split for Android-Control rows.

The public Android-Control mirror used by this project is a bounded JSON export.  It
contains one text-only action projection per screenshot, and does not include a
train/eval split or an explicit parent id.  Episode ids are recoverable from the
source image names, so this utility adds the parent id and assigns complete episodes
to train or eval using a stable SHA-256 bucket.  Keeping the split construction in a
script makes the no-leakage contract reproducible without committing the dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


EPISODE_RE = re.compile(r"(?:^|/)out_episode_(?P<episode>\d+)_step_")
SOURCE_REVISION = "hf:OfficerChul/Android-Control-84k@train4096"
SOURCE_DATASET = "OfficerChul/Android-Control-84k"
SOURCE_URL = "https://github.com/google-research/google-research/tree/master/android_control"


def _episode_id(row: dict[str, Any]) -> str:
    refs = row.get("meta", {}).get("image_references", [])
    if not isinstance(refs, list):
        raise ValueError("meta.image_references must be a list")
    for ref in refs:
        match = EPISODE_RE.search(str(ref))
        if match:
            return match.group("episode")
    raise ValueError("could not recover Android-Control episode id from image_references")


def _bucket(episode_id: str) -> int:
    """Return a stable 0..99 bucket independent of Python hash randomization."""

    digest = hashlib.sha256(f"androidcontrol:{episode_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % 100


def _identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return {"rows": count, **_identity(path)}


def split_rows(rows: list[dict[str, Any]], *, eval_percent: int = 20) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not 1 <= eval_percent <= 49:
        raise ValueError("eval_percent must be between 1 and 49")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_episode_id(row)].append(row)

    train: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    eval_ids: list[str] = []
    train_ids: list[str] = []
    for episode_id in sorted(grouped, key=lambda value: int(value)):
        split = "eval" if _bucket(episode_id) < eval_percent else "train"
        target = evaluation if split == "eval" else train
        ids = eval_ids if split == "eval" else train_ids
        ids.append(episode_id)
        for row in sorted(grouped[episode_id], key=lambda item: item.get("meta", {}).get("source_row_index", 0)):
            copy = dict(row)
            meta = dict(copy.get("meta", {}))
            meta["parent_record_id"] = f"androidcontrol:episode:{episode_id}"
            meta["split"] = split
            meta["split_contract"] = "whole_episode_sha256_bucket_v1"
            copy["meta"] = meta
            target.append(copy)

    counts = Counter(_episode_id(row) for row in rows)
    manifest = {
        "schema_version": 1,
        "source": {
            "dataset": SOURCE_DATASET,
            "revision": SOURCE_REVISION,
            "url": SOURCE_URL,
            "rows": len(rows),
            "episodes": len(grouped),
        },
        "split_contract": {
            "mode": "whole_episode_sha256_bucket",
            "hash_key": "androidcontrol:<episode_id>",
            "eval_percent": eval_percent,
            "no_episode_overlap": True,
            "episode_id_pattern": EPISODE_RE.pattern,
        },
        "train": {"rows": len(train), "episodes": len(train_ids), "episode_ids": train_ids},
        "eval": {"rows": len(evaluation), "episodes": len(eval_ids), "episode_ids": eval_ids},
        "row_count_histogram": dict(sorted(Counter(counts.values()).items())),
    }
    return train, evaluation, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--eval-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--eval-percent", type=int, default=20)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    train, evaluation, manifest = split_rows(rows, eval_percent=args.eval_percent)
    manifest["source"]["input"] = _identity(args.input)
    manifest["train"].update(_write_rows(args.train_output, train))
    manifest["eval"].update(_write_rows(args.eval_output, evaluation))
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"train": manifest["train"]["rows"], "eval": manifest["eval"]["rows"], "episodes": manifest["source"]["episodes"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
