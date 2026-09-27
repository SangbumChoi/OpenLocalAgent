#!/usr/bin/env python3
"""ToolSandbox -> eval_suite rows, statically.

Single-turn projection of a stateful benchmark: each single-tool-call scenario's user turn is
scored against the full AGENT-visible tool inventory, gold = the scenario's single allow-listed
tool. Milestones verify database state rather than a call, so arguments are withheld and only
type match carries a claim. Parsed by AST so the package's execution stack is never imported.
"""

import ast
import json
import re
import sys
from pathlib import Path

TS = Path(sys.argv[1] if len(sys.argv) > 1 else "ToolSandbox")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "toolsandbox-eval.jsonl")

PY_TO_JSON = {"bool": "boolean", "int": "integer", "float": "number", "str": "string",
              "list": "array", "dict": "object"}


def annotation_type(node) -> str:
    if isinstance(node, ast.Name):
        return PY_TO_JSON.get(node.id, "string")
    if isinstance(node, ast.Subscript):  # Optional[str], list[str], Union[...]
        if isinstance(node.value, ast.Name) and node.value.id in ("Optional", "Union"):
            inner = node.slice.elts[0] if isinstance(node.slice, ast.Tuple) else node.slice
            return annotation_type(inner)
        if isinstance(node.value, ast.Name):
            return PY_TO_JSON.get(node.value.id.lower(), "string")
    return "string"


def docstring_arg_lines(doc: str) -> dict[str, str]:
    """The Args: block of a Google-style docstring, name -> first description line."""
    out, in_args, current = {}, False, None
    for line in (doc or "").splitlines():
        stripped = line.strip()
        if stripped == "Args:":
            in_args = True
            continue
        if in_args and stripped.endswith(":") and stripped[:-1] in ("Returns", "Raises", "Yields"):
            break
        if in_args:
            match = re.match(r"(\w+):\s*(.*)", stripped)
            if match:
                current = match.group(1)
                out[current] = match.group(2)
            elif current and stripped:
                out[current] += " " + stripped
    return out


def collect_tools() -> list[dict]:
    tools = []
    for path in sorted((TS / "tool_sandbox/tools").glob("*.py")):
        if path.name in ("__init__.py", "user_tools.py"):  # user tools are not agent-visible
            continue
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            decorated = any(
                (isinstance(d, ast.Call) and getattr(d.func, "id", "") == "register_as_tool")
                for d in node.decorator_list)
            if not decorated:
                continue
            doc = ast.get_docstring(node) or ""
            arg_docs = docstring_arg_lines(doc)
            properties, required = {}, []
            defaults_start = len(node.args.args) - len(node.args.defaults)
            for i, arg in enumerate(node.args.args):
                if arg.arg in ("self",):
                    continue
                properties[arg.arg] = {"type": annotation_type(arg.annotation),
                                       "description": arg_docs.get(arg.arg, "")}
                if i < defaults_start:
                    required.append(arg.arg)
            tools.append({
                "name": node.name,
                "description": doc.split("\n\n")[0].strip(),
                "parameters": {"type": "object", "properties": properties,
                               "required": required},
            })
    return tools


def literal(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return None


def user_message(messages_node) -> str | None:
    """The USER -> AGENT turn's content, skipping simulator instructions."""
    if not isinstance(messages_node, (ast.List, ast.Tuple)):
        return None
    for element in messages_node.elts:
        if not isinstance(element, ast.Dict):
            continue
        fields = {}
        for key, value in zip(element.keys, element.values):
            key_name = literal(key)
            if key_name in ("sender", "recipient"):
                fields[key_name] = ast.unparse(value)
            elif key_name == "content":
                fields["content"] = value
        if fields.get("sender", "").endswith("USER") and fields.get("recipient", "").endswith("AGENT"):
            content = literal(fields.get("content"))
            if isinstance(content, str):
                return content
    return None


def collect_rows(tools: list[dict]) -> list[dict]:
    rows, seen = [], set()
    scenario_file = TS / "tool_sandbox/scenarios/single_tool_call_scenarios.py"
    tree = ast.parse(scenario_file.read_text())
    tool_names = {t["name"] for t in tools}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "ScenarioExtension"):
            continue
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        allow = literal(kwargs.get("tool_allow_list"))
        request = user_message(kwargs.get("messages"))
        name = literal(kwargs.get("name")) or ""
        if not (isinstance(allow, list) and len(allow) == 1 and request):
            continue
        gold = allow[0]
        if gold not in tool_names or (name, request) in seen:
            continue
        seen.add((name, request))
        rows.append({
            "messages": [
                {"content": request, "role": "user", "tool_calls": [], "tool_response": None},
                {"content": "", "role": "assistant",
                 "tool_calls": [{"arguments": {}, "name": gold}], "tool_response": None},
            ],
            "meta": {
                "claim_boundary": (
                    "single-turn projection of a stateful benchmark: the scenario's user turn "
                    "against the full agent-visible tool inventory, gold = the scenario's single "
                    "allow-listed tool; milestones verify database state, so arguments are "
                    "withheld and this is not a ToolSandbox milestone score, which requires "
                    "stateful execution and a user simulator"),
                "source_family": "toolsandbox_single_tool",
                "source_id": name,
            },
            "tools": tools,
        })
    return rows


def main() -> None:
    tools = collect_tools()
    rows = collect_rows(tools)
    with OUT.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"tools={len(tools)} rows={len(rows)} -> {OUT}")


if __name__ == "__main__":
    main()
