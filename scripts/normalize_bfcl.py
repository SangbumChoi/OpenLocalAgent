#!/usr/bin/env python
"""Normalize the Berkeley Function Calling Leaderboard into the repository's Conversation JSONL.

BFCL is the canonical open benchmark for tool calling, and it is evaluation-only here: the
repository's corpus policy rejects it as training material, so it is normalised for scoring and
never merged into a training split.

Only the single-call, AST-checkable categories are taken — a call is one function with one set of
arguments, which is what this harness scores. Ground truth lists several acceptable values per
argument; the first is used as the reference and the rest are ignored, so argument-level exact
match here is stricter than BFCL's own AST checker and should not be quoted as a BFCL score.

  python scripts/normalize_bfcl.py --out data/public/bfcl-eval.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from openlocalagent.data.prompt_contract import OPENAI_FULL_CATALOG_V1
from openlocalagent.data.render import render_conversation_rows
from openlocalagent.data.schema import Conversation, Message, Role, ToolCall, ToolSpec
from openlocalagent.model.tokenizer import load_tokenizer

CATEGORIES = ("simple", "multiple", "live_simple", "live_multiple")
ROOT = Path("data/hf-campaign/Berkeley-Function-Calling-Leaderboard")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def first_value(value):
    """Ground truth gives a list of acceptable values per argument; take the first."""
    if isinstance(value, list):
        return first_value(value[0]) if value else ""
    return value


# BFCL writes Python-flavoured type names; the prompt contract wants JSON Schema and rejects the
# rest outright, which would silently drop half the suite for catalog-conditioned models while
# leaving it intact for models that build their own catalog.
TYPE_MAP = {"dict": "object", "float": "number", "double": "number", "int": "integer",
            "integer": "integer", "long": "integer", "tuple": "array", "list": "array",
            "array": "array", "string": "string", "str": "string", "bool": "boolean",
            "boolean": "boolean", "object": "object", "number": "number", "any": "string"}


def normalise_parameters(schema: dict) -> dict:
    """Coerce a BFCL schema into the JSON-Schema subset the contract accepts."""
    if not isinstance(schema, dict):
        return {"type": "string"}
    out = {}
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
        elif key in ("description", "required", "enum", "default"):
            out[key] = value
    out.setdefault("type", "object" if "properties" in out else "string")
    if out["type"] == "object":
        out.setdefault("properties", {})
        required = [name for name in out.get("required", []) if name in out["properties"]]
        out["required"] = sorted(required)
    else:
        out.pop("required", None)
        out.pop("properties", None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/public/bfcl-eval.jsonl")
    ap.add_argument("--per-category", type=int, default=120)
    ap.add_argument("--manifest", default="data/public/bfcl-eval.manifest.json")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--tokenizer", default="data/tokenizer-h100-16k.json")
    args = ap.parse_args()

    tok = load_tokenizer("bpe", args.tokenizer)

    def renderable(conversation) -> bool:
        """Keep only rows every model can be scored on.

        A catalog-conditioned model reads the catalog through the prompt contract, which refuses
        rows whose arguments do not satisfy their own schema; a model that builds its own catalog
        never sees that check. Dropping them here keeps one row set for everyone instead of
        silently scoring the two groups on different data.
        """
        try:
            render_conversation_rows(conversation, tok, prompt_contract=OPENAI_FULL_CATALOG_V1,
                                     max_seq_len=args.max_seq_len)
        except (ValueError, KeyError):
            return False
        return True

    rows, counts, dropped = [], {}, 0
    for category in CATEGORIES:
        prompts = read_jsonl(ROOT / f"BFCL_v3_{category}.json")
        answers = {row["id"]: row for row in read_jsonl(ROOT / "possible_answer"
                                                        / f"BFCL_v3_{category}.json")}
        kept = 0
        for row in prompts:
            if kept >= args.per_category:
                break
            truth = answers.get(row["id"], {}).get("ground_truth")
            if not truth or len(truth) != 1 or not isinstance(truth[0], dict):
                continue
            name, arguments = next(iter(truth[0].items()))
            turns = row.get("question") or []
            text = " ".join(turn.get("content", "") for block in turns for turn in block
                            if turn.get("role") == "user")
            tools = [ToolSpec(name=spec["name"],
                              description=spec.get("description", "") or spec["name"],
                              parameters=normalise_parameters(spec.get("parameters", {})))
                     for spec in row.get("function", []) if spec.get("name")]
            if not text or not tools or name not in {tool.name for tool in tools}:
                continue
            call = ToolCall(name=name,
                            arguments={key: first_value(value)
                                       for key, value in (arguments or {}).items()})
            conversation = Conversation(
                messages=(Message(role=Role.user, content=text),
                          Message(role=Role.assistant, content="", tool_calls=(call,))),
                tools=tuple(tools),
                meta={"source_family": "bfcl_v3", "category": category, "source_id": row["id"],
                      "claim_boundary": "single-call AST subset, first acceptable value per "
                                        "argument, contract-renderable rows only; not an "
                                        "official BFCL score"})
            if not renderable(conversation):
                dropped += 1
                continue
            rows.append(conversation)
            kept += 1
        counts[category] = kept
        print(f"{category:16s} kept={kept}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), default=str, sort_keys=True) + "\n")
    payload = out.read_bytes()
    manifest = {"dataset": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
                "categories": counts, "rows": len(rows),
                "dropped_unrenderable": dropped, "path": str(out),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "policy": "evaluation only; never merged into a training split"}
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print("BFCL_DONE " + str(out), flush=True)


if __name__ == "__main__":
    main()
