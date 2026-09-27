#!/usr/bin/env python3
"""Profile one public MCPMark trajectory log without retaining its text or outputs.

The Hugging Face trajectory-log dataset contains public multi-turn MCP traces, including tool
outputs that can be very large and may contain third-party documents.  This profiler validates the
event envelope and writes only counts, names, byte hashes, and an explicit no-training boundary.
It is therefore safe to use as an acquisition/provenance preflight before a separately audited
normalizer decides whether any content is eligible for training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


DATASET = "Jakumetsu/mcpmark-trajectory-log"
DATASET_URL = "https://huggingface.co/datasets/Jakumetsu/mcpmark-trajectory-log"
DEFAULT_REVISION = "e50578f0ab904d8e6a7c576c387c1e76ae482c89"


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": size, "sha256": digest.hexdigest()}


def _json_arguments(value: Any, *, index: int) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"event {index} function_call.arguments is not JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"event {index} function_call.arguments must be an object")
    return value


def _content_chars(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if not isinstance(value, list):
        return 0
    return sum(
        len(item.get("text", ""))
        for item in value
        if isinstance(item, dict) and isinstance(item.get("text"), str)
    )


def profile(path: Path, *, revision: str = DEFAULT_REVISION) -> dict[str, Any]:
    """Validate a trajectory event list and return a metadata-only provenance receipt."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read MCPMark trajectory: {path}") from error
    if not isinstance(raw, list) or not raw:
        raise ValueError("MCPMark trajectory must be a non-empty JSON list")

    calls: dict[str, str] = {}
    outputs: set[str] = set()
    event_types: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    tool_names: Counter[str] = Counter()
    user_chars = assistant_chars = output_chars = 0
    for index, event in enumerate(raw):
        if not isinstance(event, dict):
            raise ValueError(f"event {index} must be an object")
        event_type = event.get("type")
        # MCPMark's first user event in older logs omits ``type`` but is otherwise a normal
        # message.  Normalize that historical envelope without changing the retained source.
        if event_type is None and event.get("role") in {"user", "assistant"}:
            event_type = "message"
        if not isinstance(event_type, str):
            raise ValueError(f"event {index} is missing type")
        event_types[event_type] += 1
        role = event.get("role")
        if isinstance(role, str):
            roles[role] += 1
        if event_type == "function_call":
            name = event.get("name")
            call_id = event.get("call_id")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"event {index} function_call.name must be non-empty")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError(f"event {index} function_call.call_id is missing")
            if call_id in calls:
                raise ValueError(f"duplicate function call id: {call_id}")
            _json_arguments(event.get("arguments", {}), index=index)
            calls[call_id] = name
            tool_names[name] += 1
        elif event_type == "function_call_output":
            call_id = event.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError(f"event {index} function_call_output.call_id is missing")
            if call_id in outputs:
                raise ValueError(f"duplicate function output id: {call_id}")
            outputs.add(call_id)
            output = event.get("output")
            if isinstance(output, str):
                output_chars += len(output)
        elif event_type == "message":
            if role == "user":
                user_chars += _content_chars(event.get("content"))
            elif role == "assistant":
                assistant_chars += _content_chars(event.get("content"))

    missing_outputs = sorted(set(calls) - outputs)
    orphan_outputs = sorted(outputs - set(calls))
    if missing_outputs or orphan_outputs:
        raise ValueError(
            f"trajectory call/output pairing is incomplete: missing={missing_outputs[:3]} "
            f"orphan={orphan_outputs[:3]}"
        )
    if roles["user"] < 1 or not calls:
        raise ValueError("trajectory must contain a user message and at least one tool call")

    payload: dict[str, Any] = {
        "kind": "localagent_mcpmark_trajectory_metadata_receipt",
        "schema_version": 1,
        "dataset": {
            "name": DATASET,
            "url": DATASET_URL,
            "revision": revision,
            "license": "MIT",
            "split": "test",
            "source_file_url": (
                f"{DATASET_URL}/resolve/{revision}/mcpmark-v1-0905/"
                "gpt-4-1-mini__filesystem/run-4/papers__find_math_paper/messages.json"
            ),
        },
        "source": {
            "file": _identity(path),
            "event_rows": len(raw),
            "event_types": dict(sorted(event_types.items())),
            "roles": dict(sorted(roles.items())),
            "tool_calls": len(calls),
            "paired_tool_outputs": len(outputs),
            "unique_tools": sorted(tool_names),
            "tool_call_counts": dict(sorted(tool_names.items())),
            "user_content_chars": user_chars,
            "assistant_content_chars": assistant_chars,
            "tool_output_chars": output_chars,
            "metadata_only": True,
            "raw_text_retained": False,
            "training_used": False,
            "model_replayed": False,
            "tools_replayed": False,
        },
        "claim_boundary": (
            "Metadata-only profiling of one public MCPMark trajectory file. Tool call names and "
            "event counts are retained, but prompts, assistant text, arguments, and tool outputs "
            "are not copied into the receipt. No model score, verifier result, live MCP execution, "
            "or training use is claimed. A separate content audit is required before normalization."
        ),
    }
    payload["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    receipt = profile(args.input, revision=args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
