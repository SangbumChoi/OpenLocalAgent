#!/usr/bin/env python
"""Normalize the public AndroidControl JSON mirror into Conversation JSONL.

This command never downloads data.  It accepts the `and_ctrl_train.json`/`and_ctrl_test.json`
files from the public mirror, records their local SHA-256, and writes a manifest that makes the
text-only/screenshot-omitted limitation explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from localagent.data.androidcontrol_json import canonical_action_from_conversation, json_row_to_conversation


def _sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()
    if args.max_records < 0:
        raise SystemExit("--max-records must be non-negative")
    if args.input.resolve() in {args.output.resolve(), args.manifest.resolve()}:
        raise SystemExit("output and manifest must differ from input")
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit("input must be a JSON array")
    selected = raw if not args.max_records else raw[: args.max_records]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    action_counts: Counter[str] = Counter()
    output_digest = hashlib.sha256()
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for index, row in enumerate(selected):
            conversation = json_row_to_conversation(
                row,
                source_revision=args.source_revision,
                split=args.split,
                row_index=index,
            )
            name, _ = canonical_action_from_conversation(conversation)
            action_counts[name] += 1
            encoded = conversation.to_json() + "\n"
            handle.write(encoded)
            output_digest.update(encoded.encode("utf-8"))
    input_bytes, input_sha = _sha256(args.input)
    manifest = {
        "kind": "localagent_androidcontrol_json_manifest",
        "schema_version": 1,
        "dataset": "OfficerChul/Android-Control-84k",
        "dataset_url": "https://huggingface.co/datasets/OfficerChul/Android-Control-84k",
        "original_dataset_url": "https://github.com/google-research/google-research/tree/master/android_control",
        "license": "Apache-2.0 (reformatted mirror; verify upstream terms before redistribution)",
        "source": {
            "path": str(args.input),
            "bytes": input_bytes,
            "sha256": input_sha,
            "revision": args.source_revision,
            "split": args.split,
            "rows_available": len(raw),
            "rows_selected": len(selected),
        },
        "output": {
            "path": str(args.output),
            "sha256": output_digest.hexdigest(),
            "rows": len(selected),
            "action_counts": dict(sorted(action_counts.items())),
        },
        "input_contract": {
            "screenshot_bytes_loaded": False,
            "visual_input_omitted": True,
            "grounding_evaluable": False,
            "train_test_boundary": "input split is preserved; do not train on test output",
        },
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
