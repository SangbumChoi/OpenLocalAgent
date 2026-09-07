#!/usr/bin/env python
"""tau2-bench tasks -> Conversation rows, as a first-action projection.

tau2-bench (sierra-research/tau2-bench, MIT) is dual-control and stateful: a simulated user talks
to the agent over many turns, tools mutate a database, and the official score is Pass^k over final
DB state plus LLM-judged natural-language assertions. None of that is reachable from a single-step
harness, and a sub-100M model cannot hold the conversation it presupposes.

**Claim boundary.** What this produces is the *first AGENT action of the reference solution*,
scored by name and arguments - the same projection this repository already applies to ToolBench,
and explicitly NOT a tau2-bench Pass^k. A number from it says the model picks the right opening
move in an unfamiliar domain; it says nothing about completing the episode.

"Agent action" is load-bearing. tau2 is dual-control, so a reference solution interleaves the
agent's tool calls with the *user's* - in telecom the first action is a device toggle the simulated
user performs 2,245 times out of 2,285. Scoring a model that plays the agent on the user's move
would be measuring the wrong actor, so the projection takes the first action whose name appears in
the agent's own catalog and skips tasks that have none.

The tool catalog is real, not inferred: it is parsed from each domain's `tools.py`, where every
tool carries an `@is_tool` decorator, typed parameters and a docstring. Only the argument *types*
are mapped onto JSON Schema; names and descriptions come from the source.

Evaluation-only. tau2 corresponds to no training split here and enters no mixture, so its score is
pure transfer - which is the point, since the airline, retail and telecom policies appear nowhere
in training.

  python scripts/normalize_tau2.py --src /tmp/tau2 --out data/public/tau2-eval.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict
from pathlib import Path

from openlocalagent.data.schema import ToolSpec

DOMAINS = ("airline", "retail", "telecom", "banking_knowledge")

#: Python annotation -> JSON Schema type. Anything unrecognised is described as a string and keeps
#: its original annotation in the description, rather than being silently dropped.
_SCALARS = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}


def _json_type(annotation: str | None) -> tuple[str, str]:
    """(json schema type, note to append to the description)."""
    if annotation is None:
        return "string", ""
    text = annotation.strip()
    if text in _SCALARS:
        return _SCALARS[text], ""
    if text.startswith(("List[", "list[")):
        return "array", ""
    if text.startswith("Optional[") or "| None" in text:
        inner = text.removeprefix("Optional[").removesuffix("]").replace("| None", "").strip()
        return _json_type(inner or None)[0], ""
    return "string", f" (source type: {text})"


def parse_tools(tools_py: Path) -> list[ToolSpec]:
    """Every `@is_tool`-decorated method in a domain's tools.py, as a ToolSpec."""
    tree = ast.parse(tools_py.read_text(encoding="utf-8"))
    specs: list[ToolSpec] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for fn in node.body:
            if not isinstance(fn, ast.FunctionDef) or fn.name.startswith("_"):
                continue
            if not any("is_tool" in ast.unparse(d) for d in fn.decorator_list):
                continue
            doc = (ast.get_docstring(fn) or fn.name).strip().split("\n")[0]
            properties: dict[str, dict] = {}
            required: list[str] = []
            defaults = len(fn.args.args) - len(fn.args.defaults)
            for index, arg in enumerate(fn.args.args):
                if arg.arg == "self":
                    continue
                annotation = ast.unparse(arg.annotation) if arg.annotation else None
                json_type, note = _json_type(annotation)
                properties[arg.arg] = {"type": json_type, "description": f"{arg.arg}{note}"}
                if index >= defaults:
                    continue
                required.append(arg.arg)
            specs.append(ToolSpec(
                name=fn.name,
                description=doc,
                parameters={"type": "object", "properties": properties, "required": required},
            ))
    return specs


def _instruction(task: dict) -> str:
    """The user's brief. Domains disagree on the shape: airline/retail/telecom give a dict of
    fields, banking_knowledge gives one prose string."""
    scenario = task.get("user_scenario") or {}
    instructions = scenario.get("instructions")
    if isinstance(instructions, str):
        parts = [instructions]
    elif isinstance(instructions, dict):
        parts = [
            instructions.get("reason_for_call"),
            instructions.get("task_instructions"),
            instructions.get("known_info"),
        ]
    else:
        parts = []
    parts.append((task.get("description") or {}).get("purpose"))
    return "\n".join(part.strip() for part in parts if isinstance(part, str) and part.strip())


def _stratified(rows: list[dict], cap: int) -> list[dict]:
    """Deterministically thin a domain to `cap` rows, spread evenly across its task order.

    Telecom contributes 1,720 usable tasks against airline's 43, so an uncapped suite is 87%
    telecom and its score is a telecom score wearing a tau2 label. Taking every Nth task keeps the
    domain's own spread and is reproducible without a seed.
    """
    if cap <= 0 or len(rows) <= cap:
        return rows
    stride = len(rows) / cap
    return [rows[int(index * stride)] for index in range(cap)]


def build_rows(src: Path, domain: str, cap: int = 0) -> list[dict]:
    tools = parse_tools(src / f"{domain}_tools.py")
    tasks = json.loads((src / f"{domain}_tasks.json").read_text(encoding="utf-8"))
    known = {tool.name for tool in tools}
    rows = []
    for task in tasks:
        actions = (task.get("evaluation_criteria") or {}).get("actions") or []
        if not actions:
            continue                     # refusal / communication-only tasks have no gold call
        # tau2 is dual-control: take the first action the AGENT owns, not the user's device moves.
        first = next((a for a in actions if a.get("name") in known), None)
        if first is None:
            continue
        instruction = _instruction(task)
        if not instruction:
            continue
        rows.append({
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": "",
                 "tool_calls": [{"name": first["name"], "arguments": first.get("arguments") or {}}]},
            ],
            "tools": [asdict(tool) for tool in tools],
            "meta": {"source": "tau2-bench", "domain": domain, "task_id": task.get("id"),
                     "projection": "first_agent_action_of_reference_solution"},
        })
    return _stratified(rows, cap)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="directory holding <domain>_tasks.json and <domain>_tools.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--domains", default=",".join(DOMAINS))
    parser.add_argument("--max-per-domain", type=int, default=150,
                        help="cap per domain so one does not dominate the suite; 0 disables")
    args = parser.parse_args()

    src = Path(args.src)
    everything: list[dict] = []
    for domain in args.domains.split(","):
        rows = build_rows(src, domain, args.max_per_domain)
        everything.extend(rows)
        print(f"{domain:20s} {len(rows):>6,} rows")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for row in everything:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(everything):,} rows to {args.out}")


if __name__ == "__main__":
    main()
