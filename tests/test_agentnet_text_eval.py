from localagent.data.schema import Conversation, Message, Role, ToolCall
from scripts.evaluate_agentnet_text import _ground_truth, _prediction, _source_action


def _row(name: str, arguments: dict, code: str) -> Conversation:
    return Conversation(
        messages=[
            Message(role=Role.user, content="Task: wait\nObservation: desktop"),
            Message(role=Role.assistant, tool_calls=[ToolCall(name=name, arguments=arguments)]),
        ],
        meta={"action_code": code, "parent_record_id": "task-1", "step_index": 0},
    )


def test_agentnet_text_eval_preserves_wait_and_hotkey_projection() -> None:
    wait = _row("wait", {"seconds": 1}, "computer.wait()")
    assert _source_action(wait) == "wait"
    assert _ground_truth(wait) == {"type": "wait", "params": {"seconds": 1}}
    assert _prediction("agentnet_wait", {"seconds": 1}) == {
        "type": "wait",
        "params": {"seconds": 1},
    }

    hotkey = _row("key_press", {"key": "ctrl+o"}, "pyautogui.hotkey(['ctrl', 'o'])")
    assert _source_action(hotkey) == "hotkey"
    assert _ground_truth(hotkey) == {"type": "hotkey", "params": {"keys": ["ctrl", "o"]}}


def test_agentnet_text_eval_accepts_legacy_list_wrapped_arguments() -> None:
    row = _row("wait", {"seconds": 1}, "computer.wait()")
    row.messages[1].tool_calls[0].arguments = ["wait", {"seconds": 1}]
    assert _ground_truth(row) == {"type": "wait", "params": {"seconds": 1}}
