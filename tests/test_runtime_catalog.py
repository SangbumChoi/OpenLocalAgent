from openlocalagent.agent.runtime import Agent
from openlocalagent.agent.tools import ToolRegistry
from openlocalagent.data.tool_catalog import build_catalog


def _agent(n=80):
    tools = build_catalog(n, seed=0)
    reg = ToolRegistry()
    for t in tools:
        reg.register(t, lambda **kw: {"ok": True, **kw})
    return Agent(reg, catalog=tools), tools  # retriever auto-built from catalog


def test_agent_retrieves_grounds_dispatches():
    agent, tools = _agent()
    t = tools[3]
    verb, noun = t.name.split("_", 1)
    out = agent.chat(f"Please {verb} the {noun} 'demo value'.")   # literal -> top-1 should hit
    assert out.startswith(f"[{t.name}(") and "demo value" in out


def test_agent_scales_selection_to_topk():
    agent, _ = _agent(80)
    # retriever restricts candidates to top-k, not the whole catalog
    specs = agent._select_specs("open the report 'x'")
    assert len(specs) == agent.retrieve_k


def test_selector_path_keeps_retrieved_catalog_bounded(monkeypatch):
    agent, tools = _agent(80)
    seen: dict[str, object] = {}

    class DummySelector:
        def rank(self, _feat, allowed_names=None):
            return list(allowed_names or ())

    def fake_decode(_model, _tokenizer, _prompt, candidates, **_kwargs):
        seen["candidate_names"] = [tool.name for tool in candidates]
        return f'<tool_call>{{"name":"{candidates[0].name}","arguments":{{}}}}</tool_call>'

    monkeypatch.setattr("openlocalagent.agent.constrained.hybrid_decode", fake_decode)
    agent.model = object()
    agent.tokenizer = object()
    agent.selector = DummySelector()
    out = agent.chat("open the report 'x'")
    assert out.startswith("[")
    assert len(seen["candidate_names"]) == agent.retrieve_k
    assert set(seen["candidate_names"]).issubset({tool.name for tool in tools})


def test_agent_runs_bounded_multi_hop_loop_and_exposes_tool_results(monkeypatch):
    agent, _ = _agent(80)
    calls = {"count": 0, "names": []}

    def fake_decode(_model, _tokenizer, _prompt, candidates, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            name = candidates[0].name
            calls["names"].append(name)
            return f'<tool_call>{{"name":"{name}","arguments":{{"value":"first"}}}}</tool_call>'
        if calls["count"] == 2:
            name = candidates[1].name
            calls["names"].append(name)
            return f'<tool_call>{{"name":"{name}","arguments":{{"value":"second"}}}}</tool_call>'
        return "done"

    monkeypatch.setattr("openlocalagent.agent.constrained.hybrid_decode", fake_decode)
    agent.model = object()
    agent.tokenizer = object()
    agent.selector = object()
    out = agent.chat("open the report", max_tool_hops=3)
    assert calls["count"] == 3
    assert out.startswith(f"[{calls['names'][0]}(")
    assert f"[{calls['names'][1]}(" in out
    assert out.endswith("done")


def test_agent_rejects_nonpositive_hop_budget():
    agent, _ = _agent()
    import pytest

    with pytest.raises(ValueError, match="max_tool_hops"):
        agent.chat("open the report", max_tool_hops=0)


def test_agent_stops_repeated_identical_call_before_second_dispatch(monkeypatch):
    agent, _ = _agent(80)
    dispatches = {"count": 0}

    def fake_decode(_model, _tokenizer, _prompt, candidates, **_kwargs):
        name = candidates[0].name
        return f'<tool_call>{{"name":"{name}","arguments":{{"value":"same"}}}}</tool_call>'

    def dispatch(_name, _arguments):
        dispatches["count"] += 1
        return {"ok": True}

    monkeypatch.setattr("openlocalagent.agent.constrained.hybrid_decode", fake_decode)
    monkeypatch.setattr(agent.tools, "dispatch", dispatch)
    agent.model = object()
    agent.tokenizer = object()
    agent.selector = object()
    out = agent.chat("open the report", max_tool_hops=4)
    assert dispatches["count"] == 1
    assert "loop_stopped: repeated" in out
