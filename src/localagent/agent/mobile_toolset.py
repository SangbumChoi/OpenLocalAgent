"""Text-first Android/handset action schemas for the optional WebGPU mobile bundle.

The base demo keeps its standard 50-tool catalog stable. These schemas are an additive pool used
by the realistic mobile pilot and can be passed to the dense selector/exporter without reshaping
the legacy fixed tool head. Coordinates require a text/semantic screen observation; this module
does not claim pixel or screenshot grounding.
"""

from __future__ import annotations

from copy import deepcopy

from localagent.data.schema import ToolSpec


MOBILE_TOOLS = [
    ToolSpec(
        name="mobile_click",
        description="Tap a mobile UI target at screen coordinates.",
        parameters={
            "type": "object",
            "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
            "required": ["x", "y"],
        },
    ),
    ToolSpec(
        name="mobile_long_press",
        description="Long-press a mobile UI target at screen coordinates.",
        parameters={
            "type": "object",
            "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
            "required": ["x", "y"],
        },
    ),
    ToolSpec(
        name="mobile_scroll",
        description="Scroll a mobile screen in a direction.",
        parameters={
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["up", "down", "left", "right"]}
            },
            "required": ["direction"],
        },
    ),
    ToolSpec(
        name="mobile_swipe",
        description="Swipe on a mobile screen from one coordinate to another.",
        parameters={
            "type": "object",
            "properties": {
                "start_x": {"type": "number"},
                "start_y": {"type": "number"},
                "end_x": {"type": "number"},
                "end_y": {"type": "number"},
            },
            "required": ["start_x", "start_y", "end_x", "end_y"],
        },
    ),
    ToolSpec(
        name="mobile_open_app",
        description="Open a mobile application by name.",
        parameters={
            "type": "object",
            "properties": {"app_name": {"type": "string", "format": "quoted"}},
            "required": ["app_name"],
        },
    ),
    ToolSpec(
        name="mobile_input_text",
        description="Type text into the focused mobile field.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string", "format": "quoted"}},
            "required": ["text"],
        },
    ),
    ToolSpec(
        name="mobile_navigate_home",
        description="Navigate to the mobile home screen.",
        parameters={"type": "object", "properties": {}, "required": []},
    ),
    ToolSpec(
        name="mobile_navigate_back",
        description="Navigate back on the mobile device.",
        parameters={"type": "object", "properties": {}, "required": []},
    ),
    ToolSpec(
        name="mobile_press_enter",
        description="Press the Enter key on the mobile keyboard.",
        parameters={
            "type": "object",
            "properties": {"key": {"type": "string", "enum": ["ENTER"]}},
            "required": ["key"],
        },
    ),
    ToolSpec(
        name="mobile_wait",
        description="Wait for a mobile UI update.",
        parameters={
            "type": "object",
            "properties": {"seconds": {"type": "number"}},
            "required": [],
        },
    ),
    ToolSpec(
        name="mobile_submit_answer",
        description="Submit a text answer for the current mobile task.",
        parameters={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    ),
]


# Optional productivity contracts for the realistic pilot. The legacy ``send_email`` and
# ``notion_write`` tools intentionally remain unchanged for standard-bundle compatibility; these
# additive schemas expose the fields a real browser/MCP integration needs.
PRODUCTIVITY_TOOLS = [
    ToolSpec(
        name="email_send",
        description="Send an email with recipient, subject, and body.",
        parameters={
            "type": "object",
            "properties": {
                "to": {"type": "string", "format": "quoted"},
                "subject": {"type": "string", "format": "quoted"},
                "body": {"type": "string", "format": "quoted"},
            },
            "required": ["to", "subject", "body"],
        },
    ),
    ToolSpec(
        name="notion_create_page",
        description="Create a Notion page with a title and body content.",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "format": "quoted"},
                "content": {"type": "string", "format": "quoted"},
            },
            "required": ["title", "content"],
        },
    ),
]


def mobile_tools() -> list[ToolSpec]:
    """Return a detached copy so callers can safely append or reorder tool pools."""

    return [
        ToolSpec(tool.name, tool.description, deepcopy(tool.parameters))
        for tool in MOBILE_TOOLS
    ]


def realistic_productivity_tools() -> list[ToolSpec]:
    """Return detached full-field email/Notion contracts for the realistic pilot."""

    return [
        ToolSpec(tool.name, tool.description, deepcopy(tool.parameters))
        for tool in PRODUCTIVITY_TOOLS
    ]
