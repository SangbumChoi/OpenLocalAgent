from openlocalagent.agent.caller import ToolCaller
from openlocalagent.agent.schema_decode import fill_tool, validate
from openlocalagent.data.schema import ToolSpec


def _T(name, desc, props, required=None):
    return ToolSpec(name, desc, {"type": "object", "properties": props,
                                 "required": required if required is not None else list(props)})


WEATHER = _T("get_weather", "weather for a city", {"city": {"type": "string"}})
MOVE = _T("move_file", "move or rename a file",
          {"source": {"type": "string", "format": "path"},
           "dest": {"type": "string", "format": "path"}})
THERMO = _T("set_thermostat", "set the thermostat temperature",
            {"temperature": {"type": "integer"}, "unit": {"type": "string", "enum": ["c", "f"]}})
CONVERT = _T("convert_currency", "convert money",
             {"amount": {"type": "number"}, "to": {"type": "string", "enum": ["USD", "EUR"]}})


def test_multiarg_paths_in_order():
    assert fill_tool("Move src/a.py to backup/a.py.", MOVE) == {"source": "src/a.py", "dest": "backup/a.py"}


def test_typed_args():
    assert fill_tool("Set the thermostat to 72 f.", THERMO) == {"temperature": 72, "unit": "f"}
    assert isinstance(fill_tool("Set the thermostat to 72 f.", THERMO)["temperature"], int)
    assert fill_tool("Convert 100 to EUR.", CONVERT) == {"amount": 100.0, "to": "EUR"}


def test_required_unfillable_returns_none():
    assert fill_tool("Move something somewhere.", MOVE) is None     # no paths -> can't fill


def test_validate_rejects_bad_types_and_enums():
    assert validate({"temperature": 72, "unit": "f"}, THERMO.parameters)
    assert not validate({"temperature": "hot", "unit": "f"}, THERMO.parameters)
    assert not validate({"temperature": True, "unit": "f"}, THERMO.parameters)
    assert not validate({"amount": False, "to": "EUR"}, CONVERT.parameters)
    assert not validate({"temperature": 72, "unit": "kelvin"}, THERMO.parameters)
    assert not validate({"unit": "f"}, THERMO.parameters)          # missing required


def test_caller_selects_relevant_tool():
    c = ToolCaller([WEATHER, MOVE, THERMO, CONVERT])
    assert c.call("What's the weather in Cusco?").name == "get_weather"
    assert c.call("Convert 50 to EUR.").name == "convert_currency"
    assert c.call("Move x/a.py to y/a.py.").arguments == {"source": "x/a.py", "dest": "y/a.py"}


def test_caller_abstains_below_threshold():
    c = ToolCaller([WEATHER, MOVE, THERMO, CONVERT], min_score=0.15)
    assert c.call("Tell me a joke.") is None


def test_caller_scales_with_distractors():
    from openlocalagent.data.tool_catalog import build_catalog
    c = ToolCaller([WEATHER] + build_catalog(500, seed=1))
    assert c.call("What's the weather in Oslo?").name == "get_weather"


def test_plan_decomposes_multistep():
    c = ToolCaller([WEATHER, MOVE, THERMO, CONVERT])
    plan = c.plan("Move x/a.py to y/a.py then convert 50 to EUR.")
    assert [tc.name for tc in plan] == ["move_file", "convert_currency"]
    assert plan[0].arguments == {"source": "x/a.py", "dest": "y/a.py"}
    assert plan[1].arguments == {"amount": 50.0, "to": "EUR"}


def test_plan_single_step_is_singleton():
    c = ToolCaller([WEATHER, MOVE, THERMO, CONVERT])
    assert [tc.name for tc in c.plan("What's the weather in Cusco?")] == ["get_weather"]


def test_plan_drops_ungroundable_steps():
    c = ToolCaller([WEATHER, MOVE, THERMO, CONVERT], min_score=0.15)
    # only the first clause grounds; "tell me a joke" abstains and is dropped
    plan = c.plan("Convert 50 to EUR and tell me a joke.")
    assert [tc.name for tc in plan] == ["convert_currency"]
