from openlocalagent.agent.caller import ToolCaller
from openlocalagent.eval import scenarios_bench as sb


def test_modalities_present():
    assert set(sb.MODALITY.values()) == {"MCP", "REST", "CLI", "SDK"}
    assert len(sb.build_tools()) == len(sb.SCENARIOS)


def test_mcp_namespaced_tool_grounds():
    c = ToolCaller(sb.build_tools(), examples=sb.examples())
    r = c.call("Open a GitHub issue 'broken link'.")
    assert r.name == "mcp__github__create_issue"
    assert r.arguments == {"title": "broken link"}


def test_cli_typed_args():
    c = ToolCaller(sb.build_tools(), examples=sb.examples())
    r = c.call("Run the postgres container on port 3000.")
    assert r.name == "docker_run"
    assert r.arguments.get("port") == 3000 and isinstance(r.arguments["port"], int)


def test_sdk_enum_and_number():
    c = ToolCaller(sb.build_tools(), examples=sb.examples())
    r = c.call("Charge the card 250 gbp.")
    assert r.name == "stripe_charge"
    assert r.arguments == {"amount": 250, "currency": "gbp"}


def test_train_eval_disjoint():
    for _mod, _name, _desc, args, *_ in sb.SCENARIOS:
        for _, sch, tr, ev in args:
            if "enum" not in sch:                 # enums are a fixed set; overlap is expected
                assert not (set(tr) & set(ev))
