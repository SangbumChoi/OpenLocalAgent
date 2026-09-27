#!/usr/bin/env python3
"""Normalize the public Agent-Diff JSONL task split into Conversations.

Agent-Diff evaluates API agents by deterministic state-diff assertions.  This adapter keeps the
natural-language task as the user message and the canonical assertion contract as the assistant
target.  It does not execute a service or copy seeded environment state into the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from localagent.data.schema import Conversation, Message, Role


DATASET = "hubertmarek/agent-diff-bench"
DATASET_URL = "https://huggingface.co/datasets/hubertmarek/agent-diff-bench"
SOURCE_URL = "https://github.com/agent-diff-bench/agent-diff"
REVISION = "4a96ea93a8d074daba93ded109f340da7fae2f70"
LICENSE = "MIT"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: str) -> str:
    parsed = json.loads(value)
    return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _conversation(raw: dict[str, Any], source: Path, line_number: int, split: str) -> Conversation:
    question = raw.get("question")
    answer = raw.get("answer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Agent-Diff question must be non-empty text")
    if not isinstance(answer, str):
        raise ValueError("Agent-Diff answer must be a JSON string")
    target = _canonical_json(answer)
    service = str(raw.get("service", "unknown"))
    return Conversation(
        messages=[Message(Role.user, question), Message(Role.assistant, target)],
        meta={
            "dataset": DATASET,
            "dataset_url": DATASET_URL,
            "source_url": SOURCE_URL,
            "source_revision": REVISION,
            "license": LICENSE,
            "split": split,
            "train_policy": "train" if split == "train" else "eval_only",
            "service": service,
            "test_id": str(raw.get("test_id", "")),
            "test_name": str(raw.get("test_name", "")),
            "task_horizon": int(raw.get("task_horizon", 0)),
            "operation_type": str(raw.get("operation_type", "")),
            "entity_scope": str(raw.get("entity_scope", "")),
            "information_availability": str(raw.get("information_availability", "")),
            "prompt_ambiguity": str(raw.get("prompt_ambiguity", "")),
            "environment_info": str(raw.get("info", "")),
            "source_record": f"{source.name}:{line_number}",
            "observation_policy": "task_and_state_diff_assertion_text_only;service_not_executed",
        },
    )


def normalize(source: Path, *, split: str) -> tuple[list[Conversation], dict[str, Any]]:
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    rows: list[Conversation] = []
    services: Counter[str] = Counter()
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = _conversation(json.loads(line), source, line_number, split)
            rows.append(row)
            services[str(row.meta["service"])] += 1
    return rows, {
        "source": {"path": str(source), "bytes": source.stat().st_size, "sha256": _sha256(source)},
        "rows": len(rows),
        "services": dict(sorted(services.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise SystemExit("refusing to overwrite output or manifest")
    rows, selection = normalize(args.input, split=args.split)
    digest = hashlib.sha256()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            encoded = (row.to_json() + "\n").encode("utf-8")
            handle.write(encoded.decode("utf-8"))
            digest.update(encoded)
    manifest: dict[str, Any] = {
        "kind": "localagent_agentdiff_normalization_manifest",
        "schema_version": 1,
        "dataset": DATASET,
        "dataset_url": DATASET_URL,
        "source_url": SOURCE_URL,
        "source_revision": REVISION,
        "license": LICENSE,
        "split": args.split,
        "train_policy": "train" if args.split == "train" else "eval_only",
        "normalization": {
            "interchange": "localagent.data.schema.Conversation",
            "user_target": "question_to_state_diff_assertion",
            "assertions": "canonical_sorted_compact_json",
            "execution": "no service execution; no external side effects",
        },
        "selection": selection,
        "records": {"selected": len(rows), "output_path": str(args.output), "output_sha256": digest.hexdigest()},
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
