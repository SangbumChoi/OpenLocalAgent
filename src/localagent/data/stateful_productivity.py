"""Deterministic stateful email/Notion/browser tasks for local agent training and evaluation.

This module is a local protocol projection, not a copy of any public benchmark.  It provides the
small state machine needed to rehearse closed-loop behavior before AndroidWorld, BrowserGym,
OSWorld, or MCP services are installed.  Train and evaluation slots and prompt skeletons are
disjoint; public benchmark task text is never embedded here.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Mapping

from localagent.agent.mobile_toolset import mobile_tools, realistic_productivity_tools
from localagent.agent.toolset import STANDARD_TOOLS
from localagent.data.schema import Conversation, Message, Role, ToolCall


SUITE_ID = "localagent-stateful-productivity-v1"


@dataclass(frozen=True)
class StatefulAction:
    """One expected action in a stateful task."""

    instruction: str
    tool: str | None
    arguments: dict[str, Any]

    @property
    def expected(self) -> dict[str, Any] | None:
        if self.tool is None:
            return None
        return {"tool": self.tool, "args": copy.deepcopy(self.arguments)}


@dataclass(frozen=True)
class StatefulTask:
    """A deterministic task with an ordered action contract and a final-state target."""

    task_id: str
    family: str
    split: str
    goal: str
    initial_state: dict[str, Any]
    actions: tuple[StatefulAction, ...]
    final_state: dict[str, Any]
    recovery: bool = False


@dataclass(frozen=True)
class ActionResult:
    """Result of applying one candidate action to a task state."""

    state: dict[str, Any]
    schema_valid: bool
    exact_tool: bool
    exact_args: bool
    state_transition: bool
    error: str | None = None

    @property
    def exact_action(self) -> bool:
        return self.exact_tool and self.exact_args

    @property
    def closed_loop_success(self) -> bool:
        return self.schema_valid and self.exact_action and self.state_transition


class StatefulRuntime:
    """Resettable local runtime that executes one task with retry semantics.

    ``apply_action`` is intentionally a pure scorer.  This wrapper adds the bit of environment
    state that a deployed agent needs: the current action index, an error observation after a
    rejected call, and an append-only event log.  A failed call does not advance the task, so a
    caller can retry against the same state.  The runtime is deterministic and in-memory; it is
    not an Android emulator, browser session, or external email/Notion account.
    """

    def __init__(self, task: StatefulTask) -> None:
        self.task = task
        self.reset()

    def reset(self, task: StatefulTask | None = None) -> None:
        """Reset the runtime to a detached initial state for ``task``."""

        if task is not None:
            self.task = task
        self._state = copy.deepcopy(self.task.initial_state)
        self._step_index = 0
        self._last_result: ActionResult | None = None
        self._events: list[dict[str, Any]] = []

    @property
    def state(self) -> dict[str, Any]:
        """Return a detached observation of the current state."""

        return copy.deepcopy(self._state)

    @property
    def step_index(self) -> int:
        """Return the next expected action index."""

        return self._step_index

    @property
    def done(self) -> bool:
        """Whether all actions have been accepted and the final projection holds."""

        return self._step_index >= len(self.task.actions) and task_complete(self.task, self._state)

    @property
    def events(self) -> list[dict[str, Any]]:
        """Return a detached execution log suitable for a reproducible receipt."""

        return copy.deepcopy(self._events)

    def prompt(self) -> str:
        """Render the current state and the last rejection as an agent observation."""

        if self.done:
            return f"Goal complete. Current state JSON: {canonical_json(self._state)}"
        prompt = state_prompt(self.task, self._step_index, self._state)
        if self._last_result is not None and self._last_result.error:
            prompt += f" Last tool result: error={self._last_result.error}"
        return prompt

    def execute(self, tool: str | None, arguments: Mapping[str, Any] | None = None) -> ActionResult:
        """Execute a candidate call, advancing only after an exact successful transition."""

        if self.done:
            raise RuntimeError(f"task {self.task.task_id} is already complete")
        expected_step = self._step_index
        result = apply_action(self.task, expected_step, self._state, tool, arguments)
        self._last_result = result
        accepted = result.closed_loop_success
        if accepted:
            self._state = result.state
            self._step_index += 1
        self._events.append(
            {
                "step_index": expected_step,
                "tool": tool,
                "arguments": copy.deepcopy(dict(arguments or {})),
                "schema_valid": result.schema_valid,
                "exact_tool": result.exact_tool,
                "exact_args": result.exact_args,
                "state_transition": result.state_transition,
                "accepted": accepted,
                "error": result.error,
            }
        )
        return result


def stateful_reward(result: ActionResult, *, terminal: bool = False) -> float:
    """Return a bounded deterministic reward suitable for an offline RL simulation.

    The components expose useful signal before a complete workflow succeeds, while the terminal
    bonus prevents a sequence of locally plausible actions from being confused with task success.
    No learned judge or external environment is involved.
    """

    reward = (
        0.10 * result.schema_valid
        + 0.25 * result.exact_tool
        + 0.25 * result.exact_args
        + 0.25 * result.state_transition
    )
    if terminal:
        reward += 0.15
    return float(reward)


def stateful_reward_spec() -> dict[str, float]:
    """Expose the frozen shaped-reward weights for receipts and RL preflight."""

    return {
        "schema_valid": 0.10,
        "exact_tool": 0.25,
        "exact_args": 0.25,
        "state_transition": 0.25,
        "terminal": 0.15,
    }


def canonical_json(value: Any) -> str:
    """Canonical compact JSON used in state prompts and receipt hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def initial_state() -> dict[str, Any]:
    """Return a detached initial state for one local task episode."""

    return {
        "app": "home",
        "screen": "home",
        "focus": None,
        "browser": {"page": None, "query": None, "results_loaded": False, "last_error": None},
        "email": {"to": None, "subject": None, "body": None, "sent": False},
        "notion": None,
    }


def tool_specs():
    """Return the detached local catalog used by the stateful protocol."""

    return [*STANDARD_TOOLS, *mobile_tools(), *realistic_productivity_tools()]


def _slots(split: str) -> dict[str, str]:
    if split == "train":
        return {
            "recipient": "maya@example.com",
            "subject": "Weekly rollout",
            "body": "Build is green",
            "notion_title": "Sprint notes",
            "notion_content": "Review action traces",
            "browser_url": "https://local.test/dashboard",
            "browser_query": "release risk",
            "recovery_bad_url": "https://local.test/missing",
            "recovery_good_url": "https://local.test/metrics",
        }
    if split == "eval":
        return {
            "recipient": "zoe@example.com",
            "subject": "Incident review",
            "body": "Queue is clear",
            "notion_title": "Launch log",
            "notion_content": "Verify the browser gate",
            "browser_url": "https://local.test/inbox",
            "browser_query": "unread invoices",
            "recovery_bad_url": "https://local.test/retired",
            "recovery_good_url": "https://local.test/reports",
        }
    raise ValueError(f"split must be 'train' or 'eval', got {split!r}")


def _instructions(split: str) -> dict[str, tuple[str, ...]]:
    if split == "train":
        return {
            "open_gmail": ("Launch Gmail on the Android phone.",),
            "compose": ("Tap Compose at x=120 y=220 on the phone.",),
            "recipient": ("Type the recipient into the focused field.",),
            "subject": ("Enter the email subject in the focused field.",),
            "body": ("Type the message body into the active field.",),
            "send": ("Send the drafted email.",),
            "open_notion": ("Open the Notion app on the phone.",),
            "create_page": ("Create the requested Notion page.",),
            "open_browser": ("Open the dashboard URL in the browser.",),
            "search_click": ("Click the Search box on the dashboard.",),
            "search_type": ("Type the requested search phrase.",),
            "search_submit": ("Press Enter to submit the dashboard search.",),
            "open_bad": ("Open the retired browser link.",),
            "open_good": ("Open the replacement browser link from the error.",),
            "refresh": ("Click the Refresh control.",),
            "abstain": ("Acknowledge that no action is needed.",),
        }
    if split == "eval":
        return {
            "open_gmail": ("Bring up Gmail on the handset.",),
            "compose": ("Select Compose at x=120 y=220 on Android.",),
            "recipient": ("Fill the focused address field with the recipient.",),
            "subject": ("Put the subject into the active subject field.",),
            "body": ("Fill the focused message field with the body.",),
            "send": ("Submit the completed message.",),
            "open_notion": ("Start the Notion application on the handset.",),
            "create_page": ("Save the requested page in Notion.",),
            "open_browser": ("Navigate the browser to the inbox URL.",),
            "search_click": ("Select the mail search field.",),
            "search_type": ("Enter the inbox search phrase in the focused field.",),
            "search_submit": ("Submit the inbox search with Enter.",),
            "open_bad": ("Visit the retired link in the browser.",),
            "open_good": ("Follow the replacement URL from the error message.",),
            "refresh": ("Select the Refresh control on the report page.",),
            "abstain": ("Reply that the requested work is already complete.",),
        }
    raise ValueError(f"unsupported split {split!r}")


def _action(instructions: Mapping[str, tuple[str, ...]], key: str, tool: str | None, args: dict[str, Any]) -> StatefulAction:
    return StatefulAction(instruction=instructions[key][0], tool=tool, arguments=args)


def build_tasks(split: str = "train") -> list[StatefulTask]:
    """Build the disjoint local task suite for ``split``."""

    slots = _slots(split)
    text = _instructions(split)
    # Recovery URLs are part of the visible browser observation in a deployed agent.  Keep them
    # in the current-step instruction so the first navigation is grounded rather than requiring
    # the policy to guess a hidden fixture argument.
    text = dict(text)
    text["open_bad"] = (f"Open {slots['recovery_bad_url']} in the browser.",)
    text["open_good"] = (f"Open {slots['recovery_good_url']} after the error.",)
    email = StatefulTask(
        task_id=f"{split}-email-send",
        family="email",
        split=split,
        goal=(
            f"Send an email to {slots['recipient']} with subject {slots['subject']!r} "
            f"and body {slots['body']!r}."
        ),
        initial_state=initial_state(),
        actions=(
            _action(text, "open_gmail", "mobile_open_app", {"app_name": "Gmail"}),
            _action(text, "compose", "mobile_click", {"x": 120, "y": 220}),
            _action(text, "recipient", "mobile_input_text", {"text": slots["recipient"]}),
            _action(text, "subject", "mobile_input_text", {"text": slots["subject"]}),
            _action(text, "body", "mobile_input_text", {"text": slots["body"]}),
            _action(
                text,
                "send",
                "email_send",
                {"to": slots["recipient"], "subject": slots["subject"], "body": slots["body"]},
            ),
        ),
        final_state={
            "email": {
                "to": slots["recipient"],
                "subject": slots["subject"],
                "body": slots["body"],
                "sent": True,
            }
        },
    )
    notion = StatefulTask(
        task_id=f"{split}-notion-create",
        family="notion",
        split=split,
        goal=(
            f"Create a Notion page titled {slots['notion_title']!r} with content "
            f"{slots['notion_content']!r}."
        ),
        initial_state=initial_state(),
        actions=(
            _action(text, "open_notion", "mobile_open_app", {"app_name": "Notion"}),
            _action(
                text,
                "create_page",
                "notion_create_page",
                {"title": slots["notion_title"], "content": slots["notion_content"]},
            ),
        ),
        final_state={
            "notion": {"title": slots["notion_title"], "content": slots["notion_content"]}
        },
    )
    browser = StatefulTask(
        task_id=f"{split}-browser-search",
        family="browser",
        split=split,
        goal=f"Open {slots['browser_url']} and search for {slots['browser_query']!r}.",
        initial_state=initial_state(),
        actions=(
            _action(text, "open_browser", "open_url", {"url": slots["browser_url"]}),
            _action(text, "search_click", "click", {"target": "the Search box"}),
            _action(text, "search_type", "type_text", {"text": slots["browser_query"]}),
            _action(text, "search_submit", "key_press", {"key": "Enter"}),
        ),
        final_state={
            "browser": {
                "page": slots["browser_url"],
                "query": slots["browser_query"],
                "results_loaded": True,
            }
        },
    )
    recovery = StatefulTask(
        task_id=f"{split}-browser-recovery",
        family="recovery",
        split=split,
        goal=f"Open the report at {slots['recovery_good_url']} if the first link is unavailable.",
        initial_state=initial_state(),
        recovery=True,
        actions=(
            _action(text, "open_bad", "open_url", {"url": slots["recovery_bad_url"]}),
            _action(text, "open_good", "open_url", {"url": slots["recovery_good_url"]}),
            _action(text, "refresh", "click", {"target": "the Refresh control"}),
        ),
        final_state={
            "browser": {
                "page": slots["recovery_good_url"],
                "results_loaded": True,
            }
        },
    )
    abstain = StatefulTask(
        task_id=f"{split}-abstain",
        family="abstention",
        split=split,
        goal=(
            "The user says the work is already complete; acknowledge it without invoking a tool."
        ),
        initial_state=initial_state(),
        actions=(_action(text, "abstain", None, {}),),
        final_state=initial_state(),
    )
    return [email, notion, browser, recovery, abstain]


def state_prompt(task: StatefulTask, step_index: int, state: Mapping[str, Any]) -> str:
    """Render a state-conditioned next-action prompt without embedding the gold tool name."""

    action = task.actions[step_index]
    return (
        f"Goal: {task.goal} Current state JSON: {canonical_json(state)} "
        f"Next required action: {action.instruction}"
    )


def conversation_for_task(task: StatefulTask) -> Conversation:
    """Render a task as the canonical user/assistant/tool trajectory used by SFT and RL."""

    messages = [Message(role=Role.user, content=task.goal)]
    state = copy.deepcopy(task.initial_state)
    for index, action in enumerate(task.actions):
        if action.tool is None:
            messages.append(Message(role=Role.assistant, content="I won't invoke a tool."))
            continue
        messages.append(
            Message(
                role=Role.assistant,
                tool_calls=[ToolCall(name=action.tool, arguments=copy.deepcopy(action.arguments))],
            )
        )
        result = apply_action(task, index, state, action.tool, action.arguments)
        state = result.state
        messages.append(
            Message(
                role=Role.tool,
                tool_response=canonical_json(
                    {"ok": result.closed_loop_success, "error": result.error, "state": state}
                ),
            )
        )
    return Conversation(
        messages=messages,
        tools=tool_specs(),
        meta={
            "kind": "stateful_productivity_task",
            "suite": SUITE_ID,
            "task_id": task.task_id,
            "family": task.family,
            "split": task.split,
            "recovery": task.recovery,
        },
    )


def _schema_valid(tool: str | None, arguments: Mapping[str, Any]) -> bool:
    if tool is None:
        return not arguments
    spec = next((item for item in tool_specs() if item.name == tool), None)
    if spec is None or not isinstance(arguments, Mapping):
        return False
    schema = spec.parameters or {}
    required = set(schema.get("required", []))
    properties = schema.get("properties", {})
    if not required.issubset(arguments):
        return False
    for name, value in arguments.items():
        prop = properties.get(name)
        if not isinstance(prop, Mapping):
            return False
        if "enum" in prop and value not in prop["enum"]:
            return False
        if prop.get("type") == "string" and not isinstance(value, str):
            return False
        if prop.get("type") in {"number", "integer"} and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            return False
    return True


def _same_args(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    def normalize_number(value: Any) -> Any:
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, Mapping):
            return {key: normalize_number(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize_number(item) for item in value]
        return value

    def normalize(value: Mapping[str, Any]) -> dict[str, Any]:
        result = normalize_number(value)
        target = result.get("target")
        if isinstance(target, str):
            low = target.lower().removeprefix("the ")
            # Accessibility labels vary between "Search box" and "mail search field" while
            # referring to the same control in this resettable fixture.  Keep the canonical
            # target in the final state so exact-action metrics do not reward arbitrary text.
            if "search" in low and ("box" in low or "field" in low):
                result["target"] = "the Search box"
            elif "refresh" in low:
                result["target"] = "the Refresh control"
        return result

    return canonical_json(normalize(left)) == canonical_json(normalize(right))


def _transition(state: dict[str, Any], tool: str, arguments: Mapping[str, Any]) -> tuple[bool, str | None]:
    """Apply a generic local transition; return ``(transitioned, error)``."""

    if tool == "mobile_open_app":
        app = arguments.get("app_name")
        state["app"] = app
        state["screen"] = "inbox" if app == "Gmail" else "home"
        state["focus"] = None
        return True, None
    if tool == "mobile_click":
        if state.get("app") != "Gmail" or state.get("screen") != "inbox":
            return False, "compose_click_requires_gmail_inbox"
        if arguments != {"x": 120, "y": 220}:
            return False, "unknown_mobile_target"
        state["screen"], state["focus"] = "compose", "to"
        return True, None
    if tool == "mobile_input_text":
        focus = state.get("focus")
        if focus not in {"to", "subject", "body"}:
            return False, "input_requires_focused_email_field"
        state["email"][focus] = arguments.get("text")
        state["focus"] = {"to": "subject", "subject": "body", "body": None}[focus]
        return True, None
    if tool == "email_send":
        email = state["email"]
        if state.get("app") != "Gmail" or state.get("screen") != "compose":
            return False, "send_requires_gmail_compose"
        if any(email.get(key) is None for key in ("to", "subject", "body")):
            return False, "send_requires_complete_draft"
        if not _same_args(
            {"to": email["to"], "subject": email["subject"], "body": email["body"]},
            arguments,
        ):
            return False, "send_args_do_not_match_draft"
        email["sent"] = True
        return True, None
    if tool == "notion_create_page":
        if state.get("app") != "Notion":
            return False, "notion_page_requires_notion_app"
        state["notion"] = {
            "title": arguments.get("title"),
            "content": arguments.get("content"),
        }
        return True, None
    if tool == "open_url":
        url = arguments.get("url")
        browser = state["browser"]
        browser["last_error"] = None
        browser["results_loaded"] = False
        if str(url).endswith(("/missing", "/retired")):
            browser["page"] = None
            browser["last_error"] = f"404: {url}"
            return True, "http_404_recoverable"
        browser["page"] = url
        browser["query"] = None
        state["focus"] = None
        return True, None
    if tool == "click":
        browser = state["browser"]
        if browser.get("page") is None:
            return False, "browser_click_requires_page"
        target = arguments.get("target")
        if isinstance(target, str):
            low = target.lower().removeprefix("the ")
            if "search" in low and ("box" in low or "field" in low):
                target = "the Search box"
            elif "refresh" in low:
                target = "the Refresh control"
        if target not in {"the Search box", "the Refresh control"}:
            return False, "unknown_browser_target"
        state["focus"] = "search" if target == "the Search box" else "refresh"
        if target == "the Refresh control":
            state["browser"]["results_loaded"] = True
        return True, None
    if tool == "type_text":
        if state.get("focus") != "search":
            return False, "type_requires_search_focus"
        state["browser"]["query"] = arguments.get("text")
        return True, None
    if tool == "key_press":
        if arguments.get("key") != "Enter" or state.get("focus") != "search":
            return False, "enter_requires_search_focus"
        state["browser"]["results_loaded"] = state["browser"].get("query") is not None
        return state["browser"]["results_loaded"], None
    return False, "unsupported_stateful_tool"


def apply_action(
    task: StatefulTask,
    step_index: int,
    state: Mapping[str, Any],
    tool: str | None,
    arguments: Mapping[str, Any] | None,
) -> ActionResult:
    """Score and apply one candidate action without mutating the caller's state."""

    if not 0 <= step_index < len(task.actions):
        raise IndexError(f"step_index {step_index} is outside task {task.task_id}")
    expected = task.actions[step_index]
    args = dict(arguments or {})
    before = copy.deepcopy(dict(state))
    schema_valid = _schema_valid(tool, args)
    exact_tool = tool == expected.tool
    exact_args = expected.tool is None and not args or (
        expected.tool is not None and _same_args(args, expected.arguments)
    )
    if not schema_valid or not exact_tool or not exact_args:
        return ActionResult(before, schema_valid, exact_tool, exact_args, False, "action_mismatch")
    if expected.tool is None:
        return ActionResult(before, True, True, True, True, None)
    transitioned, error = _transition(before, expected.tool, args)
    return ActionResult(before, True, True, True, transitioned, error)


def task_complete(task: StatefulTask, state: Mapping[str, Any]) -> bool:
    """Check the task's final-state projection, allowing unrelated state fields to differ."""

    def contains(expected: Any, actual: Any) -> bool:
        if isinstance(expected, Mapping):
            return isinstance(actual, Mapping) and all(
                key in actual and contains(value, actual[key]) for key, value in expected.items()
            )
        return expected == actual

    return contains(task.final_state, state)


def task_prompts(task: StatefulTask) -> list[str]:
    """Return standalone state-conditioned prompts for selector/pointer training."""

    state = copy.deepcopy(task.initial_state)
    prompts: list[str] = []
    for index, action in enumerate(task.actions):
        prompts.append(state_prompt(task, index, state))
        if action.tool is not None:
            result = apply_action(task, index, state, action.tool, action.arguments)
            state = result.state
    return prompts


def suite_inventory(split: str) -> dict[str, Any]:
    """Summarize task counts and slot hashes without returning task text."""

    tasks = build_tasks(split)
    return {
        "suite": SUITE_ID,
        "split": split,
        "tasks": len(tasks),
        "families": {family: sum(task.family == family for task in tasks) for family in sorted({task.family for task in tasks})},
        "steps": sum(len(task.actions) for task in tasks),
        "recovery_tasks": sum(task.recovery for task in tasks),
        "abstention_tasks": sum(task.family == "abstention" for task in tasks),
        "conversation_rows": len([conversation_for_task(task) for task in tasks]),
    }
