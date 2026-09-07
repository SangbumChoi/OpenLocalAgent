#!/usr/bin/env python
"""google/mobile-actions -> Conversation rows for the mobileactions suite.

Upstream: google/mobile-actions (CC-BY-4.0), one JSONL whose `metadata` field carries the split.
Rows are already OpenAI-shaped: `tools` as `{"function": {...}}` wrappers and `messages` with a
developer turn, a user turn, and an assistant turn holding the gold call.

**Claim boundary.** This is the dataset authors' own eval split, scored on the first gold tool call
per row. The developer turn carries date/time context the model would otherwise never see, so it
is folded into the user turn rather than dropped - a model that cannot read it would fail rows for
the wrong reason.

The train split is a legitimate posttrain source and is emitted separately; the eval split is
scored and never trained on.

  python scripts/normalize_mobileactions.py --in dataset.jsonl \
      --out-eval data/public/mobileactions-eval.jsonl \
      --out-train data/public/mobileactions-train.jsonl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from openlocalagent.data.schema import ToolSpec

EVAL_CLAIM_BOUNDARY = (
    "the dataset authors' own eval split, first gold tool call per row; "
    "the developer turn (date/time context) is folded into the user turn"
)
#: Upstream writes JSON Schema types in upper case ("OBJECT", "STRING").
_TYPE_KEY = "type"


def _normalise_schema(schema: object) -> dict:
    """Lower-case the JSON Schema type keywords upstream writes in upper case."""
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    out: dict = {}
    for key, value in schema.items():
        if key == _TYPE_KEY and isinstance(value, str):
            out[key] = value.lower()
        elif isinstance(value, dict):
            out[key] = _normalise_schema(value)
        elif isinstance(value, list):
            out[key] = [_normalise_schema(v) if isinstance(v, dict) else v for v in value]
        else:
            out[key] = value
    return out


def _tools(raw: object) -> list[ToolSpec]:
    specs: list[ToolSpec] = []
    for entry in raw or []:
        function = (entry or {}).get("function") or entry or {}
        name = function.get("name")
        if not name:
            continue
        specs.append(ToolSpec(
            name=name,
            description=function.get("description") or name,
            parameters=_normalise_schema(function.get("parameters")),
        ))
    return specs


def _user_content(messages: list[dict]) -> str:
    """The user's request, prefixed with the developer turn's context when there is one."""
    developer = next((m.get("content") for m in messages if m.get("role") == "developer"), "")
    user = next((m.get("content") for m in messages if m.get("role") == "user"), "")
    parts = [p.strip() for p in (developer, user) if isinstance(p, str) and p.strip()]
    return "\n\n".join(parts)


def _first_call(messages: list[dict]) -> dict | None:
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or call
            name = function.get("name")
            if not name:
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            return {"name": name, "arguments": arguments if isinstance(arguments, dict) else {}}
    return None


def build_rows(source: Path) -> dict[str, list[dict]]:
    by_split: dict[str, list[dict]] = {"train": [], "eval": []}
    with open(source, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            split = str(record.get("metadata") or "").strip().lower()
            if split not in by_split:
                continue
            messages = record.get("messages") or []
            tools = _tools(record.get("tools"))
            call = _first_call(messages)
            content = _user_content(messages)
            if not (tools and call and content):
                continue
            meta = {"source_family": f"mobile_actions_{split}", "source_id": f"{split}-{index}"}
            if split == "eval":
                meta["claim_boundary"] = EVAL_CLAIM_BOUNDARY
            by_split[split].append({
                "messages": [
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": "", "tool_calls": [call]},
                ],
                "tools": [asdict(tool) for tool in tools],
                "meta": meta,
            })
    return by_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="source", required=True)
    parser.add_argument("--out-eval", required=True)
    parser.add_argument("--out-train", required=True)
    args = parser.parse_args()

    by_split = build_rows(Path(args.source))
    for split, target in (("eval", args.out_eval), ("train", args.out_train)):
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            for row in by_split[split]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{split:6s} {len(by_split[split]):>7,} rows -> {target}")


if __name__ == "__main__":
    main()
