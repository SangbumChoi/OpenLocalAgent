"""Generic mobile/action lexical guard regressions for the constrained local runtime."""

from localagent.agent.constrained import _action_tail, _mobile_lexical_tool
from localagent.data.stateful_productivity import apply_action, build_tasks, state_prompt, tool_specs


def test_mobile_guard_resolves_disjoint_email_ui_actions() -> None:
    task = next(task for task in build_tasks("eval") if task.family == "email")
    state = task.initial_state.copy()
    expected = [
        "mobile_open_app",
        "mobile_click",
        "mobile_input_text",
        "mobile_input_text",
        "mobile_input_text",
        "email_send",
    ]
    for index, action in enumerate(task.actions):
        assert _mobile_lexical_tool(state_prompt(task, index, state), tool_specs()) == expected[index]
        state = apply_action(task, index, state, action.tool, action.arguments).state


def test_mobile_guard_does_not_capture_browser_focused_field() -> None:
    task = next(task for task in build_tasks("eval") if task.family == "browser")
    state = task.initial_state.copy()
    for index, action in enumerate(task.actions):
        hint = _mobile_lexical_tool(state_prompt(task, index, state), tool_specs())
        assert hint is None
        state = apply_action(task, index, state, action.tool, action.arguments).state


def test_mobile_guard_uses_current_action_and_compose_focus_state() -> None:
    prompt = (
        'Goal: Open Gmail, compose an email, fill its fields, and send it. '
        'Current state JSON: {"screen":"compose","focus":"to"} '
        "Next required action: Type 'alice@example.com' into the focused recipient field."
    )
    assert _action_tail(prompt) == "Type 'alice@example.com' into the focused recipient field."
    assert _mobile_lexical_tool(prompt, tool_specs()) == "mobile_input_text"
