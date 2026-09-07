#!/usr/bin/env python
"""Give AgentNet's *training* rows the desktop action space they actually call.

AgentNet ships `tools: []` in every row, so `openai_full_catalog_v1` rejects it - a row whose gold
answer calls a tool absent from its own catalog cannot be trained on under that contract. Dropping
it from midtrain was measurably expensive: AgentNet's own suite fell from 28.95 to 0.02 step
success, roughly 2.9 points of the ten-benchmark mean, plus knock-on losses elsewhere.

The action space is small, fixed and fully observable from the corpus - eight actions with stable
argument shapes - so the catalog is recoverable rather than invented. This writes a repaired copy
with that catalog attached to every row.

Only the train split is repaired. The eval split keeps its empty catalog, because that is what the
open baselines in results/evalsuite-full were scored against, and changing it would inflate our
numbers against theirs.

  python scripts/build_agentnet_catalog.py \
      --in data/public/agentnet-unused-train.jsonl \
      --out data/public/agentnet-train-catalog.jsonl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from openlocalagent.data.schema import ToolSpec

def _one_string(name: str, description: str) -> dict:
    """A JSON Schema taking a single required string argument."""
    return {
        "type": "object",
        "properties": {name: {"type": "string", "description": description}},
        "required": [name],
    }


#: The AgentNet desktop action space, as observed in the corpus. Coordinates are packed into a
#: single string exactly as the rows encode them, so the catalog describes the data rather than a
#: tidier scheme the model would then never see.
ACTION_SPACE: list[ToolSpec] = [
    ToolSpec(
        name="click",
        description="Click a point on the screen, optionally with a named mouse button.",
        parameters=_one_string(
            "target", "Packed click target, e.g. 'button=left;x=0.021000;y=0.951000'."),
    ),
    ToolSpec(
        name="double_click",
        description="Double-click a point on the screen.",
        parameters=_one_string(
            "target", "Packed target, e.g. 'clicks=2;x=0.213000;y=0.309000'."),
    ),
    ToolSpec(
        name="type_text",
        description="Type a literal string into the focused element.",
        parameters=_one_string("text", "The text to type."),
    ),
    ToolSpec(
        name="key_press",
        description="Press a named key or key combination.",
        parameters=_one_string("key", "Key name, e.g. 'pagedown', 'ctrl+s'."),
    ),
    ToolSpec(
        name="drag",
        description="Drag from one point to another.",
        parameters={
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Packed start point, 'x=..;y=..'."},
                "dest": {"type": "string", "description": "Packed end point, 'x=..;y=..'."},
            },
            "required": ["source", "dest"],
        },
    ),
    ToolSpec(
        name="scroll",
        description="Scroll the active surface in a direction.",
        parameters=_one_string("direction", "'up', 'down', 'left' or 'right'."),
    ),
    ToolSpec(
        name="move_cursor",
        description="Move the cursor to a point without clicking.",
        parameters=_one_string("target", "Packed point, 'x=..;y=..'."),
    ),
    ToolSpec(
        name="wait",
        description="Wait for the interface to settle.",
        parameters={
            "type": "object",
            "properties": {"seconds": {"type": "number", "description": "Seconds to wait."}},
            "required": ["seconds"],
        },
    ),
]

KNOWN_ACTIONS = {action.name for action in ACTION_SPACE}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="source", required=True)
    parser.add_argument("--out", dest="target", required=True)
    args = parser.parse_args()

    written = skipped = 0
    unknown: set[str] = set()
    with open(args.source, encoding="utf-8") as handle, \
         open(args.target, "w", encoding="utf-8") as out:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            called = {
                call.get("name")
                for message in row.get("messages", [])
                for call in (message.get("tool_calls") or [])
            }
            # A row calling something outside the observed space would still fail the contract, so
            # drop it rather than ship a catalog that does not cover its own answer.
            if not called <= KNOWN_ACTIONS:
                unknown |= called - KNOWN_ACTIONS
                skipped += 1
                continue
            row["tools"] = [asdict(action) for action in ACTION_SPACE]
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"wrote {written:,} rows to {args.target}")
    if skipped:
        print(f"skipped {skipped:,} rows calling actions outside the space: {sorted(unknown)}")
    print(f"catalog: {len(ACTION_SPACE)} actions - {', '.join(sorted(KNOWN_ACTIONS))}")


if __name__ == "__main__":
    main()
