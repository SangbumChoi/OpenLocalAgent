from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
WEB_APP = ROOT / "demos" / "webgpu" / "app.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_browser_typed_schema_grounding_matches_action_contracts():
    cases = [
        {
            "prompt": "Drag 'the notification bell' onto 'the Done button'.",
            "schema": {
                "properties": {
                    "source": {"type": "string", "format": "quoted"},
                    "dest": {"type": "string", "format": "quoted"},
                },
                "required": ["source", "dest"],
            },
            "expected": {"source": "the notification bell", "dest": "the Done button"},
        },
        {
            "prompt": "Send an Escape keypress.",
            "schema": {
                "properties": {
                    "key": {"type": "string", "enum": ["Enter", "Tab", "Escape"]},
                },
                "required": ["key"],
            },
            "expected": {"key": "Escape"},
        },
        {
            "prompt": "Give it 12 seconds.",
            "schema": {
                "properties": {"seconds": {"type": "integer"}},
                "required": ["seconds"],
            },
            "expected": {"seconds": 12},
        },
        {
            "prompt": "Navigate to huggingface.co in the browser.",
            "schema": {
                "properties": {"url": {"type": "string", "format": "url"}},
                "required": ["url"],
            },
            "expected": {"url": "huggingface.co"},
        },
        {
            "prompt": "Show me the contents of api/routes.go.",
            "schema": {
                "properties": {"path": {"type": "string", "format": "path"}},
                "required": ["path"],
            },
            "expected": {"path": "api/routes.go"},
        },
        {
            "prompt": "Turn caching off.",
            "schema": {
                "properties": {"enabled": {"type": "boolean"}},
                "required": ["enabled"],
            },
            "expected": {"enabled": False},
        },
    ]
    script = """
global.window = { __localAgentSkipInit: true, location: { search: "" } };
const { groundFromSchema } = require(process.argv[1]);
const cases = JSON.parse(process.argv[2]);
process.stdout.write(JSON.stringify(
  cases.map((item) => groundFromSchema(item.prompt, item.schema))
));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(WEB_APP), json.dumps(cases)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == [case["expected"] for case in cases]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_browser_grounding_rejects_missing_required_value():
    script = """
global.window = { __localAgentSkipInit: true, location: { search: "" } };
const { groundFromSchema } = require(process.argv[1]);
const schema = {
  properties: { direction: { type: "string", enum: ["up", "down"] } },
  required: ["direction"],
};
process.stdout.write(JSON.stringify(groundFromSchema("Please move somehow.", schema)));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(WEB_APP)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) is None


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_stateful_grounding_ignores_goal_and_observation_quotes():
    script = """
global.window = { __localAgentSkipInit: true, location: { search: "" } };
const { groundFromSchema, compactDispatchQuery } = require(process.argv[1]);
const prompt = 'Goal: compose and fill an email. Current state JSON: {"app":"home"}. ' +
  'Next required action: Open the Gmail app on the Android phone. Return exactly one structured action';
const app = groundFromSchema(prompt, {
  properties: {app_name: {type: "string", format: "quoted"}}, required: ["app_name"]
});
const urlPrompt = 'Goal: search mail. Current state JSON: {"page":null}. ' +
  'Next required action: Open https://example.local/mail in the browser.';
const url = groundFromSchema(urlPrompt, {
  properties: {url: {type: "string", format: "url"}}, required: ["url"]
});
const compact = compactDispatchQuery(urlPrompt);
process.stdout.write(JSON.stringify({app, url, compact}));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(WEB_APP)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {
        "app": {"app_name": "Gmail"},
        "url": {"url": "https://example.local/mail"},
        "compact": "Open https://example.local/mail in the browser.",
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_url_grounding_strips_serialized_user_marker_from_pointer_span():
    script = """
global.window = { __localAgentSkipInit: true, location: { search: "" } };
const { groundFromSchema } = require(process.argv[1]);
const schema = { properties: { url: {type: "string", format: "url"} }, required: ["url"] };
process.stdout.write(JSON.stringify(groundFromSchema(
  "Open https://example.com",
  schema,
  { url: "<|user|>Open https://example.com" }
)));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(WEB_APP)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {"url": "https://example.com"}


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_email_and_notion_grounding_prefer_explicit_contract_cues_over_pointer_span():
    script = """
global.window = { __localAgentSkipInit: true, location: { search: "" } };
const { groundFromSchema } = require(process.argv[1]);
const email = groundFromSchema(
  "Send an email to alice@example.com with subject WebGPU test and body Local bundle verified.",
  { properties: {recipient: {type: "string"}}, required: ["recipient"] },
  { recipient: "GPU test and body Local bundle verified" }
);
const notion = groundFromSchema(
  "Create a Notion page titled Launch log with content Verify the browser gate.",
  { properties: {content: {type: "string"}}, required: ["content"] },
  { content: "content Verify the browser gate" }
);
process.stdout.write(JSON.stringify({email, notion}));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(WEB_APP)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {
        "email": {"recipient": "alice@example.com"},
        "notion": {"content": "Verify the browser gate"},
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_notion_save_grounding_does_not_copy_the_destination_name():
    script = """
global.window = { __localAgentSkipInit: true, location: { search: "" } };
const { groundFromSchema } = require(process.argv[1]);
process.stdout.write(JSON.stringify(groundFromSchema(
  "Save the search result to Notion.",
  { properties: {content: {type: "string", format: "quoted"}}, required: ["content"] }
)));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(WEB_APP)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {"content": "search result"}


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_dispatch_safety_guards_are_explicit_and_bounded():
    script = """
global.window = { __localAgentSkipInit: true, location: { search: "" } };
const { dispatchSelect } = require(process.argv[1]);
const dispatch = {dense_selector: {tool_names: ["open_url", "web_search", "click"]}};
const url = dispatchSelect(null, 0, "Open https://example.com", dispatch);
const compound = dispatchSelect(null, 0, "Search the web for AI news, then save it to Notion.", dispatch);
process.stdout.write(JSON.stringify({url, compound}));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(WEB_APP)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {
        "url": {
            "name": "open_url",
            "route": "web_search",
            "conf": 1,
            "isStop": False,
            "selection_policy": "explicit_url_safety_guard",
        },
        "compound": {
            "name": "web_search",
            "route": "web_search",
            "conf": 1,
            "isStop": False,
            "selection_policy": "compound_search_first_step_guard",
        },
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_productivity_intent_guards_prevent_cross_surface_side_effect_misrouting():
    script = """
global.window = { __localAgentSkipInit: true, location: { search: "" } };
const { dispatchSelect } = require(process.argv[1]);
const dispatch = {
  dense_selector: {
    tool_names: ["set_timer", "open_url", "send_email", "notion_write", "web_search"],
  },
};
const email = dispatchSelect(null, 0, "Email Dana the quarterly report", dispatch);
const notion = dispatchSelect(null, 0, "Save the search result to Notion.", dispatch);
const compound = dispatchSelect(
  null, 0, "Search the web for AI news, then save it to Notion.", dispatch
);
process.stdout.write(JSON.stringify({ email, notion, compound }));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(WEB_APP)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["email"] == {
        "name": "send_email",
        "route": "app_action",
        "conf": 1,
        "isStop": False,
        "selection_policy": "productivity_email_intent_guard",
    }
    assert payload["notion"] == {
        "name": "notion_write",
        "route": "app_action",
        "conf": 1,
        "isStop": False,
        "selection_policy": "productivity_notion_intent_guard",
    }
    assert payload["compound"]["selection_policy"] == "compound_search_first_step_guard"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_productivity_guard_can_be_disabled_for_learned_selector_control():
    script = """
global.window = { __localAgentSkipInit: true, location: { search: "?productivity_guard=0" } };
const { dispatchSelect } = require(process.argv[1]);
const dispatch = {
  route_head: {
    weight: [[0], [1], [0], [0], [0]], bias: [0, 0, 0, 0, 0],
    routes: ["text", "app_action", "web_search", "computer_use", "filesystem"], stop_index: 0,
  },
  dense_selector: {
    q_proj_weight: [[1]], q_proj_bias: [0], proj: 1,
    tool_matrix: [[1], [0]], tool_names: ["set_timer", "send_email"], normalize_query: true,
  },
};
const selected = dispatchSelect(
  { data: new Float32Array([1]), dims: [1, 1, 1] },
  1,
  "Email Dana the quarterly report",
  dispatch,
);
process.stdout.write(JSON.stringify(selected));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(WEB_APP)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {
        "name": "set_timer",
        "route": "app_action",
        "conf": 1,
        "isStop": False,
        "selection_policy": "dense_selector",
        "candidate_count": 2,
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_stateful_dispatch_uses_latest_action_and_serialized_focus_state():
    script = """
global.window = { __localAgentSkipInit: true, location: { search: "" } };
const { dispatchSelect } = require(process.argv[1]);
const dispatch = {
  dense_selector: {
    tool_names: [
      "mobile_click", "mobile_input_text", "email_send", "send_email",
      "notion_create_page", "open_url", "click", "type_text", "key_press",
    ],
  },
};
const prefix = 'Goal: Open Gmail, compose an email, fill its fields, and send it. ' +
  'Current state JSON: {"screen":"compose","focus":"to"} ';
const fill = dispatchSelect(
  null, 0, prefix + "Next required action: Type 'alice@example.com' into the focused recipient field.", dispatch
);
const send = dispatchSelect(
  null, 0, prefix + "Next required action: Send the composed email now.", dispatch
);
const notion = dispatchSelect(
  null, 0,
  `Goal: Open Notion and create a page. Current state JSON: {"app":"Notion"} ` +
  `Next required action: Create a Notion page titled 'Mobile pilot'.`,
  dispatch
);
const browser = dispatchSelect(
  null, 0,
  'Goal: Open the local mail page. Current state JSON: {"page":null} ' +
  'Next required action: Open https://example.local/mail in the browser.',
  dispatch
);
process.stdout.write(JSON.stringify({fill, send, notion, browser}));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(WEB_APP)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["fill"]["name"] == "mobile_input_text"
    assert payload["fill"]["selection_policy"] == "mobile_lexical_guard"
    assert payload["send"]["name"] == "email_send"
    assert payload["send"]["selection_policy"] == "mobile_lexical_guard"
    assert payload["notion"]["name"] == "notion_create_page"
    assert payload["notion"]["selection_policy"] == "productivity_notion_intent_guard"
    assert payload["browser"]["name"] == "open_url"
    assert payload["browser"]["selection_policy"] == "explicit_url_safety_guard"
