from __future__ import annotations

from scripts.train_mcp_service_probe import SERVICE_PROMPTS, SERVICE_TARGETS, _synthetic_rows


def test_mcp_service_probe_is_contract_only_and_balanced():
    rows = _synthetic_rows()
    assert len(rows) == 28
    assert set(SERVICE_PROMPTS) == {
        "filesystem",
        "github",
        "notion",
        "playwright",
        "postgres",
    }
    assert all(len(SERVICE_PROMPTS[name]) == len(SERVICE_TARGETS[name]) for name in SERVICE_PROMPTS)
    assert all(row.meta["synthetic_probe"] == "mcp_service_contract_v1" for row in rows)
    assert all("MCPMark" not in message.content for row in rows for message in row.messages)


def test_mcp_service_probe_has_tool_calls_for_every_contract_row():
    rows = _synthetic_rows()
    calls = [message.tool_calls[0] for row in rows for message in row.messages if message.tool_calls]
    assert len(calls) == 28
    assert {call.name for call in calls} >= {
        "read_file",
        "git_status",
        "notion_create_page",
        "open_url",
        "sql_query",
    }
