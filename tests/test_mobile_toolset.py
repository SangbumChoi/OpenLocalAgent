from openlocalagent.agent.mobile_toolset import mobile_tools, realistic_productivity_tools
from openlocalagent.agent.routes import route_of


def test_mobile_toolset_is_additive_and_schema_complete():
    tools = mobile_tools()
    assert len(tools) == 11
    assert all(tool.name.startswith("mobile_") for tool in tools)
    assert {tool.name for tool in tools} == {
        "mobile_click",
        "mobile_long_press",
        "mobile_scroll",
        "mobile_swipe",
        "mobile_open_app",
        "mobile_input_text",
        "mobile_navigate_home",
        "mobile_navigate_back",
        "mobile_press_enter",
        "mobile_wait",
        "mobile_submit_answer",
    }
    for tool in tools:
        schema = tool.parameters
        assert schema["type"] == "object"
        assert set(schema) >= {"properties", "required"}
        assert route_of(tool.name) == "computer_use"

    click = next(tool for tool in tools if tool.name == "mobile_click")
    assert click.parameters["required"] == ["x", "y"]
    scroll = next(tool for tool in tools if tool.name == "mobile_scroll")
    assert scroll.parameters["properties"]["direction"]["enum"] == [
        "up",
        "down",
        "left",
        "right",
    ]
    enter = next(tool for tool in tools if tool.name == "mobile_press_enter")
    assert enter.parameters["properties"]["key"]["enum"] == ["ENTER"]


def test_mobile_toolset_returns_detached_specs():
    first = mobile_tools()
    second = mobile_tools()
    first[0].parameters["properties"]["x"]["type"] = "integer"
    assert second[0].parameters["properties"]["x"]["type"] == "number"


def test_realistic_productivity_tools_have_full_field_contracts():
    tools = realistic_productivity_tools()
    assert [tool.name for tool in tools] == ["email_send", "notion_create_page"]
    assert all(route_of(tool.name) == "app_action" for tool in tools)
    email = tools[0].parameters
    assert email["required"] == ["to", "subject", "body"]
    notion = tools[1].parameters
    assert notion["required"] == ["title", "content"]
