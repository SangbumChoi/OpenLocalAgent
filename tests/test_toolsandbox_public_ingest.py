import json
from pathlib import Path

from scripts.ingest_toolsandbox_public import _extract_file, _tool_specs, build


_SCENARIO = '''
import json
from tool_sandbox.common.execution_context import RoleType, ScenarioCategories

ScenarioExtension(
    name="wifi_off",
    messages=[
        {"sender": RoleType.SYSTEM, "recipient": RoleType.USER, "content": "context"},
        {"sender": RoleType.USER, "recipient": RoleType.AGENT, "content": "Turn off wifi"},
    ],
    tool_allow_list=["set_wifi_status"],
    categories=[ScenarioCategories.STATE_DEPENDENCY],
    milestones=[
        {"trace": json.dumps({"tool_name": "set_wifi_status", "arguments": {}})}
    ],
)
'''


def _checkout(tmp_path: Path) -> Path:
    root = tmp_path / "ToolSandbox"
    scenario_root = root / "tool_sandbox" / "scenarios"
    scenario_root.mkdir(parents=True)
    for name in (
        "single_tool_call_scenarios.py",
        "multiple_tool_call_scenarios.py",
        "multiple_user_turn_scenarios.py",
        "insufficient_information_scenarios.py",
    ):
        (scenario_root / name).write_text(_SCENARIO, encoding="utf-8")
    return root


def test_ast_projection_extracts_user_tool_and_category(tmp_path: Path) -> None:
    path = tmp_path / "single_tool_call_scenarios.py"
    path.write_text(_SCENARIO, encoding="utf-8")
    rows = _extract_file(path, split="train", holdout_modulo=2)
    if not rows:
        rows = _extract_file(path, split="eval", holdout_modulo=2)
    assert len(rows) == 1
    row = rows[0]
    assert row.messages[0].content == "Turn off wifi"
    assert row.messages[1].tool_calls[0].name == "set_wifi_status"
    assert row.meta["categories"] == ["STATE_DEPENDENCY"]
    assert row.meta["provenance"]["revision"]
    assert row.meta["environment_executed"] is False


def test_build_writes_disjoint_canonical_jsonl_and_manifest(tmp_path: Path) -> None:
    root = _checkout(tmp_path)
    output = tmp_path / "out"
    manifest = build(root, output, holdout_modulo=2)
    assert manifest["rows"]["train"] > 0
    assert manifest["rows"]["eval"] > 0
    assert manifest["source_policy"]["ast_only"] is True
    assert manifest["source_policy"]["tools_executed"] is False
    train_ids = {
        json.loads(line)["meta"]["parent_record_id"]
        for line in (output / "toolsandbox-train.jsonl").read_text().splitlines()
    }
    eval_ids = {
        json.loads(line)["meta"]["parent_record_id"]
        for line in (output / "toolsandbox-eval.jsonl").read_text().splitlines()
    }
    assert train_ids.isdisjoint(eval_ids)
    assert (output / "toolsandbox-manifest.json").is_file()


def test_ast_tool_schema_projection_keeps_signature_and_docstring(tmp_path: Path) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "messaging.py").write_text(
        '''
def send_message(phone_number: str, content: str, urgent: bool = False) -> str:
    """Send a message to a phone recipient."""
    raise RuntimeError("fixture is never executed")
''',
        encoding="utf-8",
    )
    specs, identities = _tool_specs(tools)
    spec = specs["send_message"]
    assert identities[0]["path"] == "messaging.py"
    assert spec.description == "Send a message to a phone recipient."
    assert spec.parameters["required"] == ["phone_number", "content"]
    assert spec.parameters["properties"]["phone_number"] == {"type": "string"}
    assert spec.parameters["properties"]["urgent"] == {"type": "boolean"}


def test_published_toolsandbox_transfer_receipt_is_fail_closed() -> None:
    receipt = json.loads(
        Path("docs/paper/results/raw/m59-toolsandbox-public-projection-transfer-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["dataset"]["name"] == "apple/ToolSandbox"
    assert receipt["projection_policy"]["tools_executed"] is False
    assert receipt["continuation"]["after"]["eval_token_accuracy"] > receipt["continuation"]["before"]["eval_token_accuracy"]
    assert receipt["selector_probe"]["arms"]["retrained_pretrained_backbone"]["top1"] == 0.75
    assert receipt["selector_probe"]["adoption"] == "do_not_adopt_as_representation_evidence"


def test_published_toolsandbox_schema_transfer_receipt_binds_schema_policy() -> None:
    receipt = json.loads(
        Path("docs/paper/results/raw/m61-toolsandbox-schema-conditioned-transfer-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["projection_policy"]["tool_schema_ast_only"] is True
    assert receipt["dataset"]["tool_schema_functions"] == 38
    assert receipt["selector_probe"]["arms"]["retrained_pretrained_backbone"]["top3"] == 1.0
    assert receipt["selector_probe"]["arms"]["retrained_matched_random_backbone"]["top1"] == 0.75
    assert receipt["selector_probe"]["adoption"] == "do_not_adopt_as_representation_evidence"
    assert receipt["mcpmark_untouched_description_proxy"]["standard"]["selector_top1"] == 0.023668639053254437
    assert receipt["mcpmark_untouched_description_proxy"]["mcp_servers_executed"] is False
