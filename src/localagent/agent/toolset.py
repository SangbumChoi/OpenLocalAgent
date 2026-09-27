"""Standard agent tool schemas (JSON-schema parameters) used by eval + the grounded decoder.

These carry real `parameters` so the grounded decoder can be **schema-driven** (read arg names,
types, enums, required) instead of hardcoding per-tool extraction. Mirrors configs/data/tool_pool.json.
"""

from __future__ import annotations

from localagent.data.schema import ToolSpec

STANDARD_TOOLS = [
    ToolSpec(
        name="get_weather",
        description="Get the current weather for a city.",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["c", "f"]},
            },
            "required": ["city"],
        },
    ),
    ToolSpec(
        name="calculator",
        description="Evaluate an arithmetic expression.",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string", "format": "arithmetic"}},
            "required": ["expression"],
        },
    ),
    ToolSpec(
        name="web_search",
        description="Search the web.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer"},
            },
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="planner",
        description="Make a plan to achieve a goal.",
        parameters={
            "type": "object",
            "properties": {"goal": {"type": "string"}},
            "required": ["goal"],
        },
    ),
    ToolSpec(
        name="define",
        description="Define a term.",
        parameters={"type": "object", "properties": {"term": {"type": "string"}},
                    "required": ["term"]},
    ),
    ToolSpec(
        name="play_music",
        description="Play a song.",
        parameters={"type": "object", "properties": {"song": {"type": "string"}},
                    "required": ["song"]},
    ),
    ToolSpec(
        name="get_news",
        description="Get news on a topic.",
        parameters={"type": "object", "properties": {"topic": {"type": "string"}},
                    "required": ["topic"]},
    ),
    # --- coding-agent tools (Claude Code / Codex-style) ---
    ToolSpec(
        name="read_file", description="Read a file.",
        parameters={"type": "object", "properties": {"path": {"type": "string", "format": "path"}},
                    "required": ["path"]},
    ),
    ToolSpec(
        name="write_file", description="Create or write a file.",
        parameters={"type": "object", "properties": {"path": {"type": "string", "format": "path"}},
                    "required": ["path"]},
    ),
    ToolSpec(
        name="grep_search", description="Search the codebase for a pattern.",
        parameters={"type": "object",
                    "properties": {"pattern": {"type": "string", "format": "quoted"}},
                    "required": ["pattern"]},
    ),
    ToolSpec(
        name="run_command", description="Run a shell command.",
        parameters={"type": "object",
                    "properties": {"command": {"type": "string", "format": "quoted"}},
                    "required": ["command"]},
    ),
    ToolSpec(
        name="git_commit", description="Make a git commit.",
        parameters={"type": "object",
                    "properties": {"message": {"type": "string", "format": "quoted"}},
                    "required": ["message"]},
    ),
    ToolSpec(
        name="run_tests", description="Run the test suite.",
        parameters={"type": "object", "properties": {}, "required": []},
    ),
    # --- popular everyday tools ---
    ToolSpec(
        name="set_reminder", description="Set a reminder for a task.",
        parameters={"type": "object", "properties": {"task": {"type": "string"}},
                    "required": ["task"]},
    ),
    ToolSpec(
        name="set_timer", description="Set a timer for a duration.",
        parameters={"type": "object", "properties": {"duration": {"type": "string"}},
                    "required": ["duration"]},
    ),
    # --- computer-use / productivity tools ---
    ToolSpec(
        name="calendar_event", description="Create a Google Calendar event.",
        parameters={"type": "object",
                    "properties": {"title": {"type": "string", "format": "quoted"}},
                    "required": ["title"]},
    ),
    ToolSpec(
        name="send_email", description="Send an email to someone.",
        parameters={"type": "object", "properties": {"recipient": {"type": "string"}},
                    "required": ["recipient"]},
    ),
    ToolSpec(
        name="open_url", description="Open a URL in the web browser.",
        parameters={"type": "object",
                    "properties": {"url": {"type": "string", "format": "url"}},
                    "required": ["url"]},
    ),
    ToolSpec(
        name="notion_write", description="Write a note in Notion.",
        parameters={"type": "object",
                    "properties": {"content": {"type": "string", "format": "quoted"}},
                    "required": ["content"]},
    ),
    ToolSpec(
        name="slack_send", description="Send a Slack message.",
        parameters={"type": "object",
                    "properties": {"message": {"type": "string", "format": "quoted"}},
                    "required": ["message"]},
    ),
    ToolSpec(
        name="jira_issue", description="Create a Jira issue.",
        parameters={"type": "object",
                    "properties": {"summary": {"type": "string", "format": "quoted"}},
                    "required": ["summary"]},
    ),
    # --- computer-use family (text-grounded; all string args are literal prompt substrings) ---
    # The model is byte-level text-only (no vision), so targets are SEMANTIC element descriptions
    # copied from the request (quoted spans), never pixel coordinates. Enums/ints ground via the
    # schema extractors (key_press.key, scroll.direction, wait.seconds).
    ToolSpec(
        name="screenshot", description="Take a screenshot of the screen.",
        parameters={"type": "object", "properties": {}, "required": []},
    ),
    ToolSpec(
        name="click", description="Click a UI element described in text.",
        parameters={"type": "object",
                    "properties": {"target": {"type": "string", "format": "quoted"}},
                    "required": ["target"]},
    ),
    ToolSpec(
        name="double_click", description="Double-click a UI element described in text.",
        parameters={"type": "object",
                    "properties": {"target": {"type": "string", "format": "quoted"}},
                    "required": ["target"]},
    ),
    ToolSpec(
        name="type_text", description="Type text into the focused field.",
        parameters={"type": "object",
                    "properties": {"text": {"type": "string", "format": "quoted"}},
                    "required": ["text"]},
    ),
    ToolSpec(
        name="key_press", description="Press a keyboard key.",
        parameters={"type": "object",
                    "properties": {"key": {"type": "string",
                                           "enum": ["Enter", "Tab", "Escape", "Backspace", "Space",
                                                    "Delete", "ArrowUp", "ArrowDown", "ArrowLeft",
                                                    "ArrowRight"]}},
                    "required": ["key"]},
    ),
    ToolSpec(
        name="scroll", description="Scroll the view in a direction.",
        parameters={"type": "object",
                    "properties": {"direction": {"type": "string",
                                                 "enum": ["up", "down", "left", "right"]}},
                    "required": ["direction"]},
    ),
    ToolSpec(
        name="drag", description="Drag from a source element to a destination element.",
        parameters={"type": "object",
                    "properties": {"source": {"type": "string", "format": "quoted"},
                                   "dest": {"type": "string", "format": "quoted"}},
                    "required": ["source", "dest"]},
    ),
    ToolSpec(
        name="wait", description="Wait for a number of seconds.",
        parameters={"type": "object",
                    "properties": {"seconds": {"type": "integer"}},
                    "required": ["seconds"]},
    ),
    ToolSpec(
        name="move_cursor", description="Move the cursor to a UI element described in text.",
        parameters={"type": "object",
                    "properties": {"target": {"type": "string", "format": "quoted"}},
                    "required": ["target"]},
    ),
    ToolSpec(
        name="open_app", description="Open a desktop application by name.",
        parameters={"type": "object",
                    "properties": {"name": {"type": "string", "format": "quoted"}},
                    "required": ["name"]},
    ),
    # --- modern dev / agentic tools (2025-era). Args ground as path/url/quoted/enum substrings. ---
    ToolSpec(
        name="run_python", description="Run a snippet of Python code.",
        parameters={"type": "object",
                    "properties": {"code": {"type": "string", "format": "quoted"}},
                    "required": ["code"]},
    ),
    ToolSpec(
        name="edit_file", description="Edit a file at a path.",
        parameters={"type": "object",
                    "properties": {"path": {"type": "string", "format": "path"}},
                    "required": ["path"]},
    ),
    ToolSpec(
        name="apply_patch", description="Apply a patch to a file at a path.",
        parameters={"type": "object",
                    "properties": {"path": {"type": "string", "format": "path"}},
                    "required": ["path"]},
    ),
    ToolSpec(
        name="http_request", description="Make an HTTP request to a URL.",
        parameters={"type": "object",
                    "properties": {"url": {"type": "string", "format": "url"},
                                   "method": {"type": "string",
                                              "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"]}},
                    "required": ["url"]},
    ),
    ToolSpec(
        name="sql_query", description="Run a SQL query.",
        parameters={"type": "object",
                    "properties": {"query": {"type": "string", "format": "quoted"}},
                    "required": ["query"]},
    ),
    ToolSpec(
        name="list_dir", description="List the contents of a directory.",
        parameters={"type": "object",
                    "properties": {"path": {"type": "string", "format": "path"}},
                    "required": ["path"]},
    ),
    ToolSpec(
        name="find_files", description="Find files matching a glob pattern.",
        parameters={"type": "object",
                    "properties": {"pattern": {"type": "string", "format": "quoted"}},
                    "required": ["pattern"]},
    ),
    ToolSpec(
        name="git_diff", description="Show the git diff.",
        parameters={"type": "object", "properties": {}, "required": []},
    ),
    ToolSpec(
        name="git_status", description="Show the git status.",
        parameters={"type": "object", "properties": {}, "required": []},
    ),
    ToolSpec(
        name="install_package", description="Install a package by name.",
        parameters={"type": "object",
                    "properties": {"name": {"type": "string", "format": "quoted"}},
                    "required": ["name"]},
    ),
    ToolSpec(
        name="kill_process", description="Kill a running process by name.",
        parameters={"type": "object",
                    "properties": {"name": {"type": "string", "format": "quoted"}},
                    "required": ["name"]},
    ),
    ToolSpec(
        name="read_clipboard", description="Read the system clipboard.",
        parameters={"type": "object", "properties": {}, "required": []},
    ),
    ToolSpec(
        name="write_clipboard", description="Write text to the system clipboard.",
        parameters={"type": "object",
                    "properties": {"text": {"type": "string", "format": "quoted"}},
                    "required": ["text"]},
    ),
    ToolSpec(
        name="download_file", description="Download a file from a URL.",
        parameters={"type": "object",
                    "properties": {"url": {"type": "string", "format": "url"}},
                    "required": ["url"]},
    ),
    ToolSpec(
        name="unzip", description="Unzip an archive at a path.",
        parameters={"type": "object",
                    "properties": {"path": {"type": "string", "format": "path"}},
                    "required": ["path"]},
    ),
    ToolSpec(
        name="env_get", description="Read an environment variable by name.",
        parameters={"type": "object",
                    "properties": {"name": {"type": "string", "format": "quoted"}},
                    "required": ["name"]},
    ),
    ToolSpec(
        name="make_dir", description="Create a directory at a path.",
        parameters={"type": "object",
                    "properties": {"path": {"type": "string", "format": "path"}},
                    "required": ["path"]},
    ),
    ToolSpec(
        name="list_processes", description="List running processes.",
        parameters={"type": "object", "properties": {}, "required": []},
    ),
    ToolSpec(
        name="docker_run", description="Run a Docker container from an image.",
        parameters={"type": "object",
                    "properties": {"image": {"type": "string", "format": "quoted"}},
                    "required": ["image"]},
    ),
]

# Public browser datasets use grounded backend-node IDs and distinguish click, type, and select
# actions.  Keep these tools in an explicit extension pool instead of changing STANDARD_TOOLS:
# existing 50-tool checkpoints and their legacy fixed head remain loadable, while the dense
# two-tower selector can score the extra tools without a tensor reshape.
REALISTIC_BROWSER_TOOLS = [
    *STANDARD_TOOLS,
    ToolSpec(
        name="web_click",
        description="Click a grounded web-page element by backend node id.",
        parameters={
            "type": "object",
            "properties": {"target_id": {"type": "string"}},
            "required": ["target_id"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        name="web_type",
        description="Type text into a grounded web-page element by backend node id.",
        parameters={
            "type": "object",
            "properties": {
                "target_id": {"type": "string"},
                "text": {"type": "string"},
            },
            "required": ["target_id", "text"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        name="web_select",
        description="Select an option on a grounded web-page element by backend node id.",
        parameters={
            "type": "object",
            "properties": {
                "target_id": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["target_id", "value"],
            "additionalProperties": False,
        },
    ),
]
