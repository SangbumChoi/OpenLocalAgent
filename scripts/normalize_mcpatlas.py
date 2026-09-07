#!/usr/bin/env python
"""MCP-Atlas -> Conversation rows for the mcpatlas eval suite.

Upstream: ScaleAI/MCP-Atlas (CC-BY-4.0), one parquet of 500 tasks. Each carries the user PROMPT,
the ENABLED_TOOLS the task exposes, and a reference TRAJECTORY of assistant turns with tool calls.

**Claim boundary.** This scores the first action of the reference trajectory against the task's
enabled tools. MCP-Atlas ships no parameter schemas - ENABLED_TOOLS is a list of names - so the
catalog renders name-only and the model is choosing from names and descriptions it does not have.
That is a harder prompt than a real MCP client would give it, and the number is not an MCP-Atlas
claim-verification score.

Evaluation-only: mcpatlas has no training split here and enters no mixture.

  python scripts/normalize_mcpatlas.py --in MCP-Atlas.parquet --out data/public/mcpatlas-eval.jsonl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from openlocalagent.data.schema import ToolSpec

CLAIM_BOUNDARY = (
    "first action of the reference trajectory, catalog = the task's enabled MCP tools by name "
    "(MCP-Atlas ships no parameter schemas, so catalogs render name-only); "
    "not an MCP-Atlas claim-verification score"
)
#: MCP-Atlas gives names only, so every tool takes a free-form object.
_OPEN_SCHEMA = {"type": "object", "properties": {}}


def _loads(value: object) -> object:
    """Fields arrive as JSON-encoded strings."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def _first_tool_call(trajectory: object) -> dict | None:
    if not isinstance(trajectory, list):
        return None
    for message in trajectory:
        for call in (message or {}).get("tool_calls") or []:
            function = call.get("function") or {}
            name = function.get("name")
            if not name:
                continue
            arguments = _loads(function.get("arguments")) or {}
            return {"name": name, "arguments": arguments if isinstance(arguments, dict) else {}}
    return None


def build_rows(source: Path) -> list[dict]:
    import pyarrow.parquet as pq

    rows: list[dict] = []
    for batch in pq.ParquetFile(source).iter_batches(batch_size=256):
        for record in batch.to_pylist():
            prompt = record.get("PROMPT")
            names = _loads(record.get("ENABLED_TOOLS"))
            call = _first_tool_call(_loads(record.get("TRAJECTORY")))
            if not (isinstance(prompt, str) and prompt.strip() and isinstance(names, list) and call):
                continue
            if call["name"] not in names:
                continue          # a call outside the task's own catalog is unanswerable
            tools = [ToolSpec(name=n, description=n, parameters=dict(_OPEN_SCHEMA))
                     for n in names if isinstance(n, str)]
            rows.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "", "tool_calls": [call]},
                ],
                "tools": [asdict(tool) for tool in tools],
                "meta": {"claim_boundary": CLAIM_BOUNDARY, "source_family": "mcp_atlas_eval",
                         "source_id": record.get("TASK")},
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="source", required=True)
    parser.add_argument("--out", dest="target", required=True)
    args = parser.parse_args()

    rows = build_rows(Path(args.source))
    Path(args.target).parent.mkdir(parents=True, exist_ok=True)
    with open(args.target, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows):,} rows to {args.target}")


if __name__ == "__main__":
    main()
