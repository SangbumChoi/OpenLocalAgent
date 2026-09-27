from openlocalagent.agent.routes import route_of
from openlocalagent.agent.toolset import REALISTIC_BROWSER_TOOLS, STANDARD_TOOLS


def test_realistic_browser_tools_extend_without_mutating_legacy_catalog():
    assert len(STANDARD_TOOLS) == 50
    names = [tool.name for tool in REALISTIC_BROWSER_TOOLS]
    assert names[: len(STANDARD_TOOLS)] == [tool.name for tool in STANDARD_TOOLS]
    assert names[-3:] == ["web_click", "web_type", "web_select"]
    assert len(names) == len(set(names))


def test_realistic_browser_tools_have_grounding_schemas_and_computer_route():
    by_name = {tool.name: tool for tool in REALISTIC_BROWSER_TOOLS[-3:]}
    assert set(by_name["web_click"].parameters["required"]) == {"target_id"}
    assert set(by_name["web_type"].parameters["required"]) == {"target_id", "text"}
    assert set(by_name["web_select"].parameters["required"]) == {"target_id", "value"}
    assert all(route_of(name) == "computer_use" for name in by_name)
