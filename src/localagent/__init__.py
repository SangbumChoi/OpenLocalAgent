"""LocalAgent — a <100M-param LLM that works as a local tool-calling agent.

Minimal, hackable, pure-PyTorch pipeline. See docs/ARCHITECTURE.md for the map.

Quickstart (reliable tool calling on any JSON-schema tools, no model required):

    from localagent import ToolCaller
    caller = ToolCaller(my_tools)        # list of ToolSpec
    caller.call("Move src/app.py to backup/app.py.")   # -> ToolCall | None (abstains)
"""

__version__ = "0.1.0"


def __getattr__(name):  # lazy export so `import localagent` stays light
    if name == "ToolCaller":
        from localagent.agent.caller import ToolCaller
        return ToolCaller
    raise AttributeError(name)
