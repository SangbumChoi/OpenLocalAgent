#!/usr/bin/env python3
"""Normalize Qwen AgentWorldBench test turns into eval-only Conversations.

AgentWorldBench is a world-model benchmark: each row predicts the environment observation after
an action, not a tool call.  This adapter preserves the prior trajectory turns as context and the
last observation as the assistant target.  It never marks the public test rows as trainable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from localagent.data.schema import Conversation, Message, Role


DATASET = "Qwen/AgentWorldBench"
DATASET_URL = "https://huggingface.co/datasets/Qwen/AgentWorldBench"
SOURCE_URL = "https://github.com/QwenLM/Qwen-AgentWorld"
REVISION = "6b8d28437042434dcdd168434227ca0de408c5ba"
LICENSE = "Apache-2.0"
ACTION_RE = re.compile(r"\*\*Action:\*\*\s*```(?:json|text)?\s*(.*?)```", re.DOTALL)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _action_signature(prompt: str) -> str:
    match = ACTION_RE.search(prompt)
    return (match.group(1).strip() if match else "").replace("\n", " ")[:500]


def _conversation(raw: dict[str, Any], source: Path, line_number: int) -> Conversation:
    prompts = raw.get("prompt")
    responses = raw.get("response")
    if not isinstance(prompts, list) or not isinstance(responses, list) or not prompts:
        raise ValueError("prompt/response must be non-empty lists")
    if len(prompts) != len(responses):
        raise ValueError("prompt and response history lengths differ")
    system = str(raw.get("system_str", ""))
    messages: list[Message] = []
    if system:
        messages.append(Message(Role.system, system))
    for prompt, response in zip(prompts, responses):
        messages.append(Message(Role.user, str(prompt)))
        messages.append(Message(Role.assistant, str(response[0] if isinstance(response, list) else response)))
    domain = str(raw.get("task", "unknown"))
    trajectory_id = str(raw.get("id", ""))
    turn_index = int(raw.get("turn_idx", len(prompts)))
    messages[-1].content = str(responses[-1][0] if isinstance(responses[-1], list) else responses[-1])
    return Conversation(
        messages=messages,
        meta={
            "dataset": DATASET,
            "dataset_url": DATASET_URL,
            "source_url": SOURCE_URL,
            "source_revision": REVISION,
            "license": LICENSE,
            "split": "test",
            "train_policy": "eval_only",
            "domain": domain,
            "trajectory_id": trajectory_id,
            "turn_idx": turn_index,
            "total_turns": int(raw.get("total_turns", turn_index)),
            "action_signature": _action_signature(str(prompts[-1])),
            "source_record": f"{source.name}:{line_number}",
            "observation_policy": "text_only_reference_observation; screenshots_not_embedded",
        },
    )


def normalize(inputs: list[Path], *, max_per_domain: int) -> tuple[list[Conversation], dict[str, Any]]:
    if max_per_domain < 1:
        raise ValueError("max_per_domain must be positive")
    rows: list[Conversation] = []
    counts: Counter[str] = Counter()
    source_identities: list[dict[str, Any]] = []
    for source in inputs:
        source_rows = 0
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                domain = str(raw.get("task", "unknown"))
                if counts[domain] >= max_per_domain:
                    continue
                rows.append(_conversation(raw, source, line_number))
                counts[domain] += 1
                source_rows += 1
        source_identities.append({"path": str(source), "bytes": source.stat().st_size, "sha256": _sha256(source), "selected": source_rows})
    return rows, {"sources": source_identities, "domain_counts": dict(sorted(counts.items()))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-per-domain", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise SystemExit("refusing to overwrite output or manifest")
    rows, selection = normalize(args.input, max_per_domain=args.max_per_domain)
    digest = hashlib.sha256()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            encoded = (row.to_json() + "\n").encode("utf-8")
            handle.write(encoded.decode("utf-8"))
            digest.update(encoded)
    manifest: dict[str, Any] = {
        "kind": "localagent_agentworldbench_eval_manifest",
        "schema_version": 1,
        "dataset": DATASET,
        "dataset_url": DATASET_URL,
        "source_url": SOURCE_URL,
        "source_revision": REVISION,
        "license": LICENSE,
        "split": "test",
        "train_policy": "eval_only",
        "normalization": {
            "module": "scripts.normalize_agentworldbench",
            "interchange": "localagent.data.schema.Conversation",
            "history": "prior_prompt_response_pairs_preserved;last_response_is_target",
            "screenshots": "not_embedded",
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
