"""Curated computer-use tool catalog for the demo (incl. the newly requested actions).

These are demonstrated via *retrieval* selection (not the fixed trained head), so new tools work
with **zero retraining** — that's the scaling property from scripts/tool_scale_analysis.py. Each
entry carries a description + example phrasings (the retriever indexes those) + an argument
format the grounded decoder extracts (`quoted` / `path` / `url` / `string`).
"""

from __future__ import annotations

from localagent.data.schema import ToolSpec

# (name, description, arg, format, [example phrasings with {v}])
CURATED = [
    ("write_file", "write or create a file", "path", "path",
     ["Write to {v}.", "Create the file {v}.", "Save the file {v}."]),
    ("move_file", "move or rename a file", "path", "path",
     ["Move the file {v}.", "Relocate {v}.", "Move {v} to the archive."]),
    ("read_file", "read a file", "path", "path",
     ["Read the file {v}.", "Open {v}.", "Show me the contents of {v}."]),
    ("google_search", "search Google for something", "query", "string",
     ["Google {v}.", "Search Google for {v}.", "Look up {v} on Google."]),
    ("click_link", "click a link / open a website in the browser", "url", "url",
     ["Click the link {v}.", "Open the website {v}.", "Click through to {v}."]),
    ("summarize", "summarize some text or a document", "content", "quoted",
     ["Summarize '{v}'.", "Give me a summary of '{v}'.", "TL;DR of '{v}'."]),
    ("send_email", "send an email to someone", "recipient", "string",
     ["Send an email to {v}.", "Email {v}.", "Compose an email to {v}."]),
    ("calendar_event", "create a calendar event", "title", "quoted",
     ["Add a calendar event '{v}'.", "Schedule '{v}'.", "Put '{v}' on my calendar."]),
    ("slack_send", "send a Slack message", "message", "quoted",
     ["Send a Slack message '{v}'.", "Slack the team '{v}'.", "Post '{v}' to Slack."]),
    ("jira_issue", "create a Jira issue", "summary", "quoted",
     ["Create a Jira ticket '{v}'.", "File a Jira bug for '{v}'.", "Open a Jira issue '{v}'."]),
    ("get_weather", "get the weather for a city", "city", "string",
     ["What's the weather in {v}?", "Weather for {v}.", "How's the weather in {v}?"]),
    ("run_command", "run a shell command", "command", "quoted",
     ["Run the command '{v}'.", "Execute '{v}'.", "Run '{v}' in the shell."]),
    ("git_commit", "make a git commit", "message", "quoted",
     ["Commit with message '{v}'.", "Git commit '{v}'.", "Commit the changes '{v}'."]),
    ("grep_search", "search the codebase for a pattern", "pattern", "quoted",
     ["Grep for '{v}'.", "Search the code for '{v}'.", "Find '{v}' in the repo."]),
    ("play_music", "play a song", "song", "string",
     ["Play {v}.", "Put on {v}.", "Start playing {v}."]),
    ("set_timer", "set a timer", "duration", "string",
     ["Set a timer for {v}.", "Timer for {v}.", "Wake me in {v}."]),
]


def curated_specs() -> list[ToolSpec]:
    return [ToolSpec(name=n, description=d,
                     parameters={"type": "object",
                                 "properties": {a: {"type": "string", "format": f}
                                                if f in ("quoted", "path", "url") else {"type": "string"}},
                                 "required": [a]})
            for n, d, a, f, _ in CURATED]


def curated_examples() -> dict:
    """{tool_name: [example query strings]} for example-augmented retrieval."""
    return {n: [p.format(v="…") for p in ex] for n, d, a, f, ex in CURATED}
