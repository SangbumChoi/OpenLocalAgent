from openlocalagent.agent.parser import extract_tool_calls, strip_tool_calls
from openlocalagent.agent.tools import default_registry
from openlocalagent.eval.tool_eval import irrelevance_correct, match_calls


def test_parser_extracts_tool_call():
    text = 'sure <tool_call>{"name": "calculator", "arguments": {"expression": "2+2"}}</tool_call>'
    calls = extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "calculator"
    assert strip_tool_calls(text) == "sure"


def test_parser_plain_text_is_abstention():
    calls = extract_tool_calls("just a normal answer, no tools")
    assert calls == []
    assert irrelevance_correct(calls) is True


def test_builtin_calculator_dispatch():
    out = default_registry.dispatch("calculator", {"expression": "2 * (3 + 4)"})
    assert out == {"result": 14}


def test_unknown_tool_is_safe():
    out = default_registry.dispatch("does_not_exist", {})
    assert "error" in out


def test_ast_match_is_order_insensitive():
    from openlocalagent.data.schema import ToolCall

    pred = [ToolCall("a", {"x": 1}), ToolCall("b", {"y": 2})]
    ref = [ToolCall("b", {"y": 2}), ToolCall("a", {"x": 1})]
    assert match_calls(pred, ref)
