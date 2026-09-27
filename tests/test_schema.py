from openlocalagent.data.schema import Conversation, Message, Role, ToolCall, ToolSpec


def test_conversation_roundtrip():
    conv = Conversation(
        tools=[ToolSpec("get_weather", "weather", {"type": "object", "properties": {}})],
        messages=[
            Message(role=Role.user, content="weather in Paris?"),
            Message(role=Role.assistant,
                    tool_calls=[ToolCall("get_weather", {"city": "Paris"})]),
            Message(role=Role.tool, tool_response='{"temp_c": 19}'),
            Message(role=Role.assistant, content="19C in Paris."),
        ],
    )
    back = Conversation.from_json(conv.to_json())
    assert len(back.messages) == 4
    assert back.messages[1].tool_calls[0].name == "get_weather"
    assert back.tools[0].name == "get_weather"


def test_generate_weighted_oversamples():
    from collections import Counter

    from openlocalagent.data.synth.agent_synth import Generator
    s = Generator(2, 1, "train").generate_weighted(300, {"weather": 12.0})
    c = Counter(x.category for x in s)
    assert c["weather"] > 50  # heavy weight dominates


def test_toolcall_normalized_is_order_independent():
    a = ToolCall("f", {"x": 1, "y": 2})
    b = ToolCall("f", {"y": 2, "x": 1})
    assert a.normalized() == b.normalized()
