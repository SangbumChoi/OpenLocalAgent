import importlib.util
from pathlib import Path

from localagent.data.schema import Conversation, Message, Role, ToolCall


_SPEC = importlib.util.spec_from_file_location(
    "export_mind2web_grounded_rows",
    Path(__file__).parents[1] / "scripts/export_mind2web_grounded_rows.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
enrich_conversation = _MODULE.enrich_conversation


def test_mind2web_dom_enrichment_keeps_target_id_in_each_action_context():
    conversation = Conversation(
        messages=[
            Message(role=Role.user, content="Open the result."),
            Message(
                role=Role.assistant,
                tool_calls=[ToolCall(name="web_click", arguments={"target_id": "42"})],
            ),
            Message(role=Role.tool, tool_response="Observed source action: CLICK"),
            Message(
                role=Role.assistant,
                tool_calls=[ToolCall(name="web_click", arguments={"target_id": "84"})],
            ),
        ],
    )
    source = {
        "actions": [
            {
                "action_uid": "a1",
                "cleaned_html": '<a backend_node_id="42"><text>Result</text></a>',
                "operation": {"op": "CLICK", "value": ""},
                "pos_candidates": [
                    {
                        "backend_node_id": "42",
                        "attributes": '{"backend_node_id":"42","tag":"a","id":"result"}',
                    }
                ],
                "neg_candidates": [],
            },
            {
                "action_uid": "a2",
                "cleaned_html": '<button backend_node_id="84"><text>Next</text></button>',
                "operation": {"op": "CLICK", "value": ""},
                "pos_candidates": [
                    {
                        "backend_node_id": "84",
                        "attributes": '{"backend_node_id":"84","tag":"button"}',
                    }
                ],
                "neg_candidates": [],
            },
        ]
    }
    enriched = enrich_conversation(conversation, source, max_candidates=2)
    assert "target_id=42" in enriched.messages[0].content
    assert "target_id=84" in (enriched.messages[2].tool_response or "")
    assert enriched.meta["derivation"] == "mind2web_grounded_dom_v1"
