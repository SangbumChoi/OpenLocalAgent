"""Tool registry (Phase 7).

`@tool` registers a Python callable + its JSON schema; the registry renders ToolSpecs for the
prompt and dispatches ToolCalls. A couple of safe built-ins are provided; network/file tools
should go through a sandbox boundary (TODO).
"""

from __future__ import annotations

from typing import Any, Callable

from localagent.data.schema import ToolSpec


class ToolRegistry:
    def __init__(self):
        self._fns: dict[str, Callable] = {}
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec, fn: Callable) -> None:
        self._fns[spec.name] = fn
        self._specs[spec.name] = spec

    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def dispatch(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in self._fns:
            return {"error": f"unknown tool: {name}"}
        try:
            return self._fns[name](**arguments)
        except Exception as e:  # tools must never crash the agent loop
            return {"error": str(e)}


# Module-level default registry + decorator sugar.
default_registry = ToolRegistry()


def tool(name: str, description: str, parameters: dict[str, Any]):
    def deco(fn: Callable) -> Callable:
        default_registry.register(ToolSpec(name, description, parameters), fn)
        return fn

    return deco


@tool(
    name="calculator",
    description="Evaluate an arithmetic expression.",
    parameters={
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    },
)
def _calculator(expression: str):
    # Minimal safe arithmetic eval (no names/builtins). Expand cautiously.
    import ast
    import operator as op

    ops = {
        ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
        ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg, ast.Mod: op.mod,
    }

    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp):
            return ops[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")

    return {"result": ev(ast.parse(expression, mode="eval").body)}
