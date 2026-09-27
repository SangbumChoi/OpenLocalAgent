#!/usr/bin/env python3
"""Enrich normalized Mind2Web Conversations with compact grounded DOM observations.

Mind2Web's action labels include positive/negative DOM candidates and backend node IDs.  The
original LocalAgent adapter intentionally emitted text-first rows, which made a ``target_id``
impossible to copy because the identifier was absent from the prompt.  This adapter keeps the
source action sequence but adds a bounded, deterministic candidate snapshot before every action.
It never downloads data or claims official Mind2Web test scoring.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import random
import re
from pathlib import Path
from typing import Any

from localagent.data.schema import Conversation, Role


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _node_text(cleaned_html: str, backend_node_id: str) -> str:
    """Extract a short visible-text hint without parsing the full untrusted HTML tree."""

    match = re.search(
        rf'backend_node_id="{re.escape(backend_node_id)}"[^>]*>(.*?)</',
        cleaned_html,
        flags=re.DOTALL,
    )
    if not match:
        return ""
    text = html.unescape(re.sub(r"<[^>]+>", " ", match.group(1)))
    return re.sub(r"\s+", " ", text).strip()[:96]


def _candidate_line(candidate: dict[str, Any], cleaned_html: str) -> str:
    attrs = candidate.get("attributes", {})
    if isinstance(attrs, str):
        attrs = json.loads(attrs)
    backend_node_id = str(candidate.get("backend_node_id") or attrs.get("backend_node_id", ""))
    fields = [f"target_id={backend_node_id}"]
    for key in ("tag", "role", "id", "title", "aria_label", "placeholder", "value"):
        value = attrs.get(key)
        if value not in (None, ""):
            fields.append(f"{key}={str(value)[:96]}")
    text = _node_text(cleaned_html, backend_node_id)
    if text:
        fields.append(f"text={text}")
    return " ".join(fields)


def _snapshot(action: dict[str, Any], *, max_candidates: int) -> str:
    candidates = list(action.get("pos_candidates") or []) + list(action.get("neg_candidates") or [])
    if not candidates:
        raise ValueError("Mind2Web action has no DOM candidates")
    # The source stores positives separately from negatives.  Shuffle the bounded view so a
    # model cannot solve the task by always copying candidate zero; the seed is source-stable.
    seed = str(action.get("action_uid", ""))
    random.Random(seed).shuffle(candidates)
    selected = candidates[:max_candidates]
    positive_ids = {
        str(item.get("backend_node_id"))
        for item in action.get("pos_candidates") or []
        if item.get("backend_node_id") is not None
    }
    if not any(str(item.get("backend_node_id")) in positive_ids for item in selected):
        selected[-1] = (action.get("pos_candidates") or [])[0]
    lines = [_candidate_line(item, action.get("cleaned_html", "")) for item in selected]
    operation = action.get("operation") or {}
    op = operation.get("op", operation.get("original_op", ""))
    value = operation.get("value", "")
    suffix = f" operation={op}" if op else ""
    if value != "":
        suffix += f" value={str(value)[:160]}"
    return "Browser DOM candidates:" + suffix + " " + " | ".join(lines)


def enrich_conversation(
    conversation: Conversation,
    source_row: dict[str, Any],
    *,
    max_candidates: int = 12,
) -> Conversation:
    actions = list(source_row.get("actions") or [])
    if not actions:
        return conversation
    result = copy.deepcopy(conversation)
    action_index = 0
    for message_index, message in enumerate(result.messages):
        if message.role == Role.user and action_index == 0:
            message.content = f"{message.content}\n{_snapshot(actions[0], max_candidates=max_candidates)}"
        if message.role == Role.tool and action_index < len(actions):
            message.tool_response = (
                f"{message.tool_response or ''}\n"
                f"{_snapshot(actions[action_index], max_candidates=max_candidates)}"
            ).strip()
        if message.role == Role.assistant and message.tool_calls:
            action_index += 1
    result.meta = {
        **result.meta,
        "derivation": "mind2web_grounded_dom_v1",
        "enrichment_level": 1,
        "dom_candidate_cap": max_candidates,
        "source_action_count": len(actions),
    }
    return result


def _load_conversations(path: Path) -> list[Conversation]:
    return [
        Conversation.from_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write(path: Path, rows: list[Conversation]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            encoded = row.to_json()
            handle.write(encoded + "\n")
            digest.update((encoded + "\n").encode("utf-8"))
    return {"path": str(path), "rows": len(rows), "sha256": digest.hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="raw Mind2Web JSON array")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, action="append", required=True)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    if len(args.input) != len(args.output):
        raise SystemExit("--input and --output must be supplied in matching counts")
    source_rows = json.loads(args.source.read_text(encoding="utf-8"))
    by_id = {str(row["annotation_id"]): row for row in source_rows}
    outputs = []
    for input_path, output_path in zip(args.input, args.output, strict=True):
        rows = _load_conversations(input_path)
        enriched = []
        for row in rows:
            parent_id = str(row.meta.get("parent_record_id", ""))
            if parent_id not in by_id:
                raise SystemExit(f"missing raw Mind2Web annotation_id {parent_id}")
            enriched.append(enrich_conversation(row, by_id[parent_id], max_candidates=args.max_candidates))
        outputs.append(_write(output_path, enriched))
    print(json.dumps({
        "source": _identity(args.source),
        "revision": args.revision,
        "outputs": outputs,
        "max_candidates": args.max_candidates,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
