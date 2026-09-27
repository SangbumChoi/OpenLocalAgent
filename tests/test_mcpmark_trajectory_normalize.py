import json
from pathlib import Path

from localagent.data.schema import Conversation, Role
from scripts.normalize_mcpmark_trajectory import normalize


def test_m118_receipt_binds_redacted_sft_and_weight_analysis() -> None:
    receipt = json.loads(
        Path(
            "docs/paper/results/raw/m118-mcpmark-redacted-trajectory-sft-transfer-v1.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["dataset"]["source_rows"] == 3
    assert receipt["dataset"]["source_tool_calls"] == 67
    assert receipt["normalized_data"]["redacted_tool_outputs"] is True
    assert receipt["normalized_data"]["redacted_absolute_paths"] is True
    assert receipt["training"]["teacher_forced"]["warm_child"]["eval_loss"] < receipt[
        "training"
    ]["teacher_forced"]["warm_parent"]["eval_loss"]
    assert receipt["weight_analysis"]["warm_random_parent_backbone_state_exact"] is True
    assert receipt["weight_analysis"]["warm_random_child_backbone_state_exact"] is True
    assert receipt["first_unseen_playwright_action"]["warm_child"]["exact"] is False
    assert "not an official MCPMark score" in receipt["claim_boundary"]


def test_normalize_redacts_paths_outputs_and_keeps_zero_argument_tools(tmp_path: Path) -> None:
    source = tmp_path / "trajectory.json"
    source.write_text(
        json.dumps(
            [
                {"role": "user", "content": "Use the tools."},
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "list_allowed_directories",
                    "arguments": "",
                },
                {"type": "function_call_output", "call_id": "c1", "output": "secret"},
                {
                    "type": "function_call",
                    "call_id": "c2",
                    "name": "read_multiple_files",
                    "arguments": '{"paths":["/home/private/a.html"]}',
                },
                {"type": "function_call_output", "call_id": "c2", "output": "secret"},
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "normalized.jsonl"
    metadata = tmp_path / "metadata.json"
    report = normalize([source], output, metadata, revision="test-revision")
    row = Conversation.from_json(output.read_text(encoding="utf-8").strip())
    assert {tool.name for tool in row.tools} == {
        "list_allowed_directories",
        "read_multiple_files",
    }
    calls = [message.tool_calls[0] for message in row.messages if message.tool_calls]
    assert calls[0].arguments == {}
    assert calls[1].arguments["paths"] == ["<workspace>/home/private/a.html"]
    assert all(message.tool_response == "[MCP tool output redacted for content audit]" for message in row.messages if message.role == Role.tool)
    assert report["training_used"] is True
    serialized = output.read_text(encoding="utf-8")
    assert '"/home/private' not in serialized
    assert "<workspace>/home/private/a.html" in serialized
