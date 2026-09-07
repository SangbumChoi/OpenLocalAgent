#!/usr/bin/env python
"""Normalize ToolBench (ToolLLM, OpenBMB) into the repository's Conversation JSONL.

ToolBench is the largest open tool-use benchmark and it is evaluation-only here, by the same corpus
policy that keeps BFCL out of training. Its own metrics — ToolEval pass rate and win rate — need
live RapidAPI calls and a GPT-4 judge, neither of which this harness has, so what is scored is the
benchmark's *first action*: given the query and the task's own API catalog, which API does the
model call. That is a strictly narrower claim than a ToolBench score and is recorded as such on
every row.

The reference is the first action of the released DFS trajectory. It is one valid opening move
rather than the only one, so exact match here is a lower bound; the same caveat does not apply to
the function name, which the trajectory commits to.

  python scripts/normalize_toolbench.py --source data/public/toolbench_G123_dfs_eval.json \
      --out data/public/toolbench-eval.jsonl
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path

from openlocalagent.data.prompt_contract import OPENAI_FULL_CATALOG_V1
from openlocalagent.data.render import render_conversation_rows
from openlocalagent.data.schema import Conversation, Message, Role, ToolCall, ToolSpec
from openlocalagent.model.tokenizer import load_tokenizer

# The ReAct trajectory writes its call as two lines; the input is JSON with occasional trailing
# prose, so the brace span is taken rather than the rest of the string.
ACTION = re.compile(r"Action:\s*(\S+)\s*\nAction Input:\s*(\{.*)", re.S)
# ToolBench's loop terminator. It is control flow in their agent, not an API, and a single-call
# harness cannot score it, so it is dropped from the catalog rather than offered as a candidate.
CONTROL_FUNCTIONS = {"Finish"}
TYPE_MAP = {"dict": "object", "float": "number", "double": "number", "int": "integer",
            "integer": "integer", "long": "integer", "tuple": "array", "list": "array",
            "array": "array", "string": "string", "str": "string", "bool": "boolean",
            "boolean": "boolean", "object": "object", "number": "number", "any": "string"}


def normalise_parameters(schema: dict) -> dict:
    """Coerce a ToolBench schema into the JSON-Schema subset the prompt contract accepts."""
    if not isinstance(schema, dict):
        return {"type": "string"}
    out: dict = {}
    for key, value in schema.items():
        if key == "type":
            declared = value if isinstance(value, str) else "string"
            out["type"] = TYPE_MAP.get(declared.lower(), "string")
        elif key == "properties" and isinstance(value, dict):
            out["properties"] = {name: normalise_parameters(entry)
                                 for name, entry in value.items()}
        elif key == "items":
            out["items"] = normalise_parameters(value) if isinstance(value, dict) \
                else {"type": "string"}
        elif key in ("description", "required", "enum"):
            out[key] = value
    out.setdefault("type", "object" if "properties" in out else "string")
    if out["type"] == "object":
        out.setdefault("properties", {})
        out["required"] = sorted(name for name in out.get("required", [])
                                 if name in out["properties"])
    else:
        out.pop("required", None)
        out.pop("properties", None)
    return out


def catalog_from_system(text: str) -> list[dict] | None:
    """The task's API list, which the system turn carries as a Python literal."""
    start = text.rfind("Specifically, you have access to the following APIs:")
    if start < 0:
        start = text.find("[{'name'")
    if start < 0:
        return None
    bracket = text.find("[", start)
    if bracket < 0:
        return None
    try:
        specs = ast.literal_eval(text[bracket:].strip())
    except (ValueError, SyntaxError):
        return None
    return specs if isinstance(specs, list) else None


def first_action(turns: list[dict]) -> tuple[str, dict] | None:
    for turn in turns:
        if turn.get("from") != "assistant":
            continue
        found = ACTION.search(turn.get("value", ""))
        if not found:
            continue
        name, payload = found.group(1).strip(), found.group(2)
        depth, end = 0, None
        for index, character in enumerate(payload):
            depth += (character == "{") - (character == "}")
            if depth == 0:
                end = index + 1
                break
        try:
            arguments = json.loads(payload[:end]) if end else {}
        except json.JSONDecodeError:
            arguments = {}
        return name, (arguments if isinstance(arguments, dict) else {})
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/public/toolbench_G123_dfs_eval.json")
    ap.add_argument("--out", default="data/public/toolbench-eval.jsonl")
    ap.add_argument("--manifest", default="data/public/toolbench-eval.manifest.json")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--tokenizer", default="data/tokenizer-h100-16k.json")
    args = ap.parse_args()

    tok = load_tokenizer("bpe", args.tokenizer)

    def renderable(conversation: Conversation) -> bool:
        """Keep only rows every model can be scored on, as the BFCL adapter does."""
        try:
            render_conversation_rows(conversation, tok, prompt_contract=OPENAI_FULL_CATALOG_V1,
                                     max_seq_len=args.max_seq_len)
        except (ValueError, KeyError):
            return False
        return True

    records = json.loads(Path(args.source).read_text())
    rows, dropped, catalog_sizes = [], 0, []
    for record in records:
        if len(rows) >= args.limit:
            break
        turns = record.get("conversations") or []
        system = next((turn["value"] for turn in turns if turn.get("from") == "system"), "")
        specs = catalog_from_system(system)
        query = next((turn["value"] for turn in turns if turn.get("from") == "user"), "").strip()
        action = first_action(turns)
        if not specs or not query or action is None:
            continue
        name, arguments = action
        tools = [ToolSpec(name=spec["name"],
                          description=spec.get("description", "") or spec["name"],
                          parameters=normalise_parameters(spec.get("parameters", {})))
                 for spec in specs
                 if spec.get("name") and spec["name"] not in CONTROL_FUNCTIONS]
        if not tools or name not in {tool.name for tool in tools}:
            continue
        conversation = Conversation(
            messages=(Message(role=Role.user, content=query),
                      Message(role=Role.assistant, content="",
                              tool_calls=(ToolCall(name=name, arguments=arguments),))),
            tools=tuple(tools),
            meta={"source_family": "toolbench_g123_dfs", "source_id": str(record.get("id", ""))[:80],
                  "claim_boundary": "first action of the released DFS trajectory, catalog "
                                    "excluding the Finish control function, contract-renderable "
                                    "rows only; this is not a ToolEval pass rate or win rate, "
                                    "which require live RapidAPI calls and a GPT-4 judge"})
        if not renderable(conversation):
            dropped += 1
            continue
        rows.append(conversation)
        catalog_sizes.append(len(tools))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), default=str, sort_keys=True) + "\n")
    manifest = {
        "dataset": "OpenBMB/ToolBench (ToolLLM), G1+G2+G3 DFS evaluation split",
        "source_file": args.source, "records_available": len(records), "rows": len(rows),
        "dropped_unrenderable": dropped,
        "catalog_per_task_mean": round(sum(catalog_sizes) / max(len(catalog_sizes), 1), 2),
        "catalog_per_task_max": max(catalog_sizes) if catalog_sizes else 0,
        "path": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "policy": "evaluation only; never merged into a training split",
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print("TOOLBENCH_DONE " + str(out), flush=True)


if __name__ == "__main__":
    main()
