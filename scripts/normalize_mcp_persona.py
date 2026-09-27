#!/usr/bin/env python3
"""Project MCP-Persona's English eval release to tool-chain text without executing tools."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from localagent.data.schema import Conversation, Message, Role


DATASET = "MCP-Persona"
SOURCE_URL = "https://github.com/wwh0411/MCP-Persona"
REVISION = "b510f5a5371c4524a58aeeb679c1ace845603e95"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize(source: Path) -> tuple[list[Conversation], dict[str, Any]]:
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("MCP-Persona source must be a non-empty task list")
    rows: list[Conversation] = []
    query_types: Counter[str] = Counter()
    for row in raw:
        instruction = row.get("instruction")
        chains = row.get("chains")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("MCP-Persona instruction must be non-empty text")
        if not isinstance(chains, list) or not chains or not all(isinstance(tool, str) for tool in chains):
            raise ValueError("MCP-Persona chains must be a non-empty list of tool names")
        query_type = str(row.get("query_type", "unknown"))
        query_types[query_type] += 1
        rows.append(
            Conversation(
                messages=[
                    Message(Role.user, instruction),
                    Message(Role.assistant, _canonical({"tool_chain": chains})),
                ],
                meta={
                    "dataset": DATASET,
                    "source_url": SOURCE_URL,
                    "source_revision": REVISION,
                    "split": "test",
                    "train_policy": "eval_only",
                    "task_id": int(row["id"]),
                    "query_type": query_type,
                    "chain_length": len(chains),
                    "first_server": chains[0].split(":", 1)[0],
                    "observation_policy": "instruction_to_tool_chain_text_only;no_context_or_tool_execution",
                },
            )
        )
    return rows, {
        "source": {"path": str(source), "bytes": source.stat().st_size, "sha256": _sha256(source)},
        "rows": len(rows),
        "query_types": dict(sorted(query_types.items())),
        "target": "canonical_compact_json_tool_chain",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise SystemExit("refusing to overwrite output or manifest")
    rows, selection = normalize(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            encoded = (row.to_json() + "\n").encode("utf-8")
            handle.write(encoded.decode("utf-8"))
            digest.update(encoded)
    manifest: dict[str, Any] = {
        "kind": "localagent_mcp_persona_projection_manifest",
        "schema_version": 1,
        "dataset": DATASET,
        "source_url": SOURCE_URL,
        "source_revision": REVISION,
        "split": "test",
        "train_policy": "eval_only",
        "selection": selection,
        "records": {"selected": len(rows), "output_path": str(args.output), "output_sha256": digest.hexdigest()},
        "claim_boundary": (
            "Tool-chain text projection only. Persona context, ground-truth checkpoints, tool outputs, "
            "simulators, and external services are excluded; this projection is not native MCP-Persona success."
        ),
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
