#!/usr/bin/env python
"""Normalize the open agent midtrain sets into the Conversation schema.

Two sources survive the midtrain line (does it correspond to a suite the harness scores?):

  microsoft/orca-agentinstruct-1M-v1  CDLA-Permissive-2.0  tool_use, webagent_flow, code_,
                                                            follow_up, fs_cot_flow, struct2text_flow,
                                                            rag - synthetic, not benchmark-derived
  xingyaoww/code-act                  Apache-2.0           python control in an execute loop

Two that were fetched beside them do not, measured 2026-09-03 on the box:

  nvidia/When2Call            2,243 of its 3,215 tool names are in the xlam eval catalog (95%)
  interstellarninja/hermes_reasoning_tool_use   built from When2Call, ToolAce and xLAM

so they are refused here by name, not merely omitted.

Rows keep their assistant answers as text. No tool_calls are parsed: what midtrain is meant to
expose is the distribution of structured interaction, and the eval contract is only enforced in
posttrain.

  python scripts/normalize_midtrain_open.py --raw data/raw/midtrain-open --out data/public
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from openlocalagent.data.schema import Conversation, Message, Role  # noqa: E402

REFUSED = {
    "when2call-sft.jsonl": "95% tool-name overlap with the xlam eval catalog",
    "hermes-tool-use.parquet": "derived from When2Call, ToolAce and xLAM, all scored suites",
}

#: output name -> (raw files, licence). Orca splits that are not agentic (mcq, fermi, creative,
#: brand guideline) stay out: they would be general instruction data, which this stage is not.
SOURCES = {
    "orca_tool_use": (["orca-tool_use.parquet"], "CDLA-Permissive-2.0"),
    "orca_webagent_flow": (["orca-webagent_flow.parquet"], "CDLA-Permissive-2.0"),
    "orca_code": (["orca-code-0.parquet", "orca-code-1.parquet"], "CDLA-Permissive-2.0"),
    "orca_follow_up": (["orca-follow_up-0.parquet", "orca-follow_up-1.parquet"], "CDLA-Permissive-2.0"),
    "orca_task": (["orca-fs_cot_flow.parquet", "orca-struct2text_flow.parquet", "orca-rag.parquet"],
                  "CDLA-Permissive-2.0"),
    "codeact": (["codeact.parquet"], "Apache-2.0"),
}

ROLES = {"system": Role.system, "user": Role.user, "human": Role.user,
         "assistant": Role.assistant, "gpt": Role.assistant}


def orca_rows(path: Path):
    import pyarrow.parquet as pq

    for raw in pq.read_table(path, columns=["messages"]).column("messages").to_pylist():
        yield json.loads(raw)


def codeact_rows(path: Path):
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["conversations"])
    for turns in table.column("conversations").to_pylist():
        yield [{"role": t.get("role") or t.get("from"), "content": t.get("content") or t.get("value")}
               for t in turns]


def to_conversation(turns: list[dict], source: str) -> Conversation | None:
    messages = []
    for turn in turns:
        role = ROLES.get(str(turn.get("role", "")).lower())
        content = (turn.get("content") or "").strip()
        if role is None:
            return None
        if role is Role.system and not content:
            continue
        messages.append(Message(role=role, content=content))
    roles = [m.role for m in messages]
    if Role.user not in roles or roles[-1] is not Role.assistant or not messages[-1].content:
        return None
    return Conversation(messages=messages, meta={"source": source})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw/midtrain-open", type=Path)
    parser.add_argument("--out", default="data/public", type=Path)
    parser.add_argument("--only", help="comma-separated subset of source names")
    args = parser.parse_args()
    wanted = set(args.only.split(",")) if args.only else set(SOURCES)
    manifest: dict[str, dict] = {"refused": REFUSED}
    for name, (files, licence) in SOURCES.items():
        if name not in wanted:
            continue
        out = args.out / f"{name}.jsonl"
        kept = dropped = 0
        digest = hashlib.sha256()
        with out.open("w", encoding="utf-8") as handle:
            for file in files:
                path = args.raw / file
                rows = codeact_rows(path) if name == "codeact" else orca_rows(path)
                for turns in rows:
                    conv = to_conversation(turns, name)
                    if conv is None:
                        dropped += 1
                        continue
                    line = conv.to_json()
                    handle.write(line + "\n")
                    digest.update(line.encode("utf-8"))
                    kept += 1
        manifest[name] = {"path": str(out), "rows": kept, "dropped": dropped, "license": licence,
                          "files": files, "sha256": digest.hexdigest()}
        print(f"{name:<20} kept={kept:>7} dropped={dropped:>5}  {licence}", flush=True)
    (args.out / "midtrain-open.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
