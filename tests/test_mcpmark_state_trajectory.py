import json
from pathlib import Path

from scripts.normalize_mcpmark_state_trajectory import state_summary


def test_state_summary_keeps_shape_without_result_content() -> None:
    output = json.dumps(
        {"content": [{"type": "text", "text": "Page URL: https://secret.example/a\nTitle: Hidden"}]}
    )
    marker = json.loads(state_summary(output, call_id="call-1", index=4))
    assert marker["state"] == "mcp_result_v1"
    assert marker["status"] == "ok"
    assert marker["result_shape"] == "object"
    assert marker["content_types"] == ["text"]
    assert marker["has_page_state"] is True
    assert "secret.example" not in state_summary(output, call_id="call-1", index=4)
    assert "Hidden" not in state_summary(output, call_id="call-1", index=4)


def test_state_summary_marks_errors_and_is_deterministic() -> None:
    output = "Error: Access denied - path outside allowed directories"
    first = state_summary(output, call_id="call-2", index=7)
    second = state_summary(output, call_id="call-2", index=7)
    assert first == second
    marker = json.loads(first)
    assert marker["status"] == "error"
    assert marker["result_shape"] == "text"


def test_normalizer_contract_declares_public_split_and_redaction_boundary() -> None:
    source = Path("scripts/normalize_mcpmark_state_trajectory.py").read_text(encoding="utf-8")
    assert "state_shape_status_digest_only" in source
    assert "not an official MCPMark" in source
    assert "source_split" in source
