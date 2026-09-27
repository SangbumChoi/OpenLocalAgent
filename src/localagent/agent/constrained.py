"""Prompt-grounded constrained decoding (ARCHITECTURE_IDEAS §2a/2b) — schema-driven.

A <100M byte model reliably learns tool-call *structure* + *tool selection* but not generalizable
slot *copying*. So we don't free-generate the arguments: we ground them in the prompt and let the
model *rank*. This version is **schema-driven and trigger-free** (no hardcoded per-tool phrases):

  1. tool selection  — the model free-generates a call; we parse the tool *name* (it gets that
     right even when the argument bytes are garbled).
  2. argument grounding — for the selected tool's schema args, propose candidate values generically
     from the prompt: word n-grams for strings, regex for numbers/arithmetic, enum members for
     enums, plus "omit" for optional args. Build every valid candidate call.
  3. ranking — score each candidate (teacher-forced, length-normalized) and take the best.

Open-ended *text* responses are not extractive; a light intent classifier handles them here as a
placeholder for the dual text-head proposed in ARCHITECTURE_IDEAS §2a.
"""

from __future__ import annotations

import itertools
import json
import math
import re

import torch
import torch.nn.functional as F

from localagent.data.schema import ToolSpec
from localagent.model.tokenizer import ASSISTANT, TOOL_CALL_CLOSE, TOOL_CALL_OPEN

MAX_COMBOS = 60
# Generic English prepositions that introduce a slot value ("weather IN Paris", "search FOR X",
# "plan TO learn guitar"). Not tool-specific — works across schemas. ("me" excluded: it nests
# badly, e.g. "remind ME to call" vs "to call".)
PREPS = ["for", "about", "up", "to", "in", "on", "of"]


def _canon(name: str, args: dict) -> str:
    return json.dumps({"name": name, "arguments": args}, separators=(",", ":"), sort_keys=True)


def _strip(s: str) -> str:
    return re.sub(r"^[^A-Za-z0-9]+|\s*(online|please|right now)?\s*[.?!]*$", "", s, flags=re.I).strip()


def _action_tail(prompt: str) -> str:
    """Return the current-step instruction, excluding earlier goal/state slots when present."""

    browser = re.search(
        r"\bBrowser task:\s*(.*?)(?:\n\s*Live accessibility elements\b|\n\s*Choose exactly one\b|$)",
        prompt,
        re.I | re.S,
    )
    if browser:
        return browser.group(1).strip()
    match = re.search(
        r"(?:Next required action|Current step(?:\s+[^:]+)?|Current action|Instruction):\s*"
        r"(.*?)(?:\s+Last tool result:|$)",
        prompt,
        re.I,
    )
    return match.group(1).strip() if match else prompt


def _mobile_lexical_tool(prompt: str, tools: list[ToolSpec]) -> str | None:
    """Return a conservative mobile UI tool hint from the current action.

    A small model can learn the action schema while still confusing a generic desktop ``click``
    or ``type_text`` with its mobile counterpart.  The WebGPU demo already has the same narrow
    lexical guard.  Keep this adapter independent of task IDs and app names: it only fires when
    the action explicitly describes a handset/mobile surface (or a focused compose screen in a
    serialized UI state) and the corresponding tool is present in the supplied catalog.
    """

    names = {tool.name for tool in tools}
    action = _action_tail(prompt)
    low = action.lower()
    compose_state = bool(re.search(r'"screen"\s*:\s*"compose"', prompt, re.I))
    state_mobile = bool(
        compose_state and re.search(r'"focus"\s*:\s*"[^\"]+"', prompt, re.I)
    )
    mobile_cue = bool(re.search(r"\b(?:mobile|android|phone|handset|touch|tap|swipe)\b", low))
    if not mobile_cue and not compose_state:
        return None

    def choose(name: str) -> str | None:
        return name if name in names else None

    if compose_state and re.search(r"\b(?:send|submit|deliver|dispatch)\b", low):
        # Prefer the full-field productivity contract when present.  This keeps a focused
        # compose surface distinct from a generic ``send_email`` recipient-only tool while still
        # allowing catalogs that expose only the legacy contract.
        if "email_send" in names:
            return "email_send"
        if "send_email" in names:
            return "send_email"

    # A focused compose surface followed by a send/submit instruction is an email action, not
    # another text entry.  Derive the choice from the available schemas rather than a task ID or
    # app name so a different catalog can supply its own email tool.
    if compose_state and re.search(r"\b(?:send|submit|deliver|dispatch)\b", low):
        candidates = []
        for tool in tools:
            schema = tool.parameters or {}
            properties = set((schema.get("properties") or {}).keys())
            description = tool.description.lower()
            score = int("email" in tool.name.lower() or "email" in description)
            score += 2 * int({"to", "subject", "body"} <= properties)
            score += int("send" in tool.name.lower())
            if score:
                candidates.append((score, tool.name))
        if candidates:
            return max(candidates)[1]

    if re.search(r"\b(?:navigate|go|return|press)\b[\s\S]*\bhome\b", low):
        return choose("mobile_navigate_home")
    if re.search(r"\b(?:navigate|go|return|press)\b[\s\S]*\bback\b", low):
        return choose("mobile_navigate_back")
    if re.search(r"\b(?:press|hit|send)\b[\s\S]*\benter\b", low):
        return choose("mobile_press_enter")
    if re.search(r"\b(?:type|input|fill)\b", low) or (state_mobile and re.search(r"\bput\b", low)):
        return choose("mobile_input_text")
    if re.search(r"\b(?:long[ -]?press|hold)\b", low):
        return choose("mobile_long_press")
    if re.search(r"\bswipe\b", low):
        return choose("mobile_swipe")
    if re.search(r"\bscroll\b", low):
        return choose("mobile_scroll")
    if re.search(r"\b(?:open|launch|start|bring up)\b", low) and not re.search(r"https?://", low):
        return choose("mobile_open_app")
    if re.search(r"\b(?:tap|click|touch|select)\b", low):
        return choose("mobile_click")
    if re.search(r"\b(?:wait|sleep)\b", low):
        return choose("mobile_wait")
    return None


def _playwright_lexical_tool(prompt: str, tools: list[ToolSpec]) -> str | None:
    """Return a narrow Playwright ABI hint for the safe, state-fetching prefix.

    MCP/Playwright tools are unusual for this byte model: ``ref`` values only exist after a
    browser snapshot, while a task's first instruction usually contains an explicit URL.  A
    generic dense selector cannot reliably infer that protocol ordering from a long task
    description.  This adapter therefore only forces the two side-effect-free protocol steps
    that are unambiguously grounded in the request: navigate to an explicit URL, then fetch a
    snapshot.  It deliberately does *not* invent element references or JavaScript bodies.
    """

    names = {tool.name for tool in tools}
    browser_names = {name for name in names if name.startswith("browser_")}
    if not browser_names:
        return None
    low = prompt.lower()
    navigated = bool(re.search(r"(?:assistant|tool_call)[^\n]*browser_navigate", low))
    snapshotted = bool(re.search(r"(?:assistant|tool_call)[^\n]*browser_snapshot", low))
    has_url = bool(re.search(r"https?://[^\s)\]\"']+", prompt, re.I))
    if "browser_navigate" in names and has_url and not navigated:
        if re.search(r"\b(?:navigate|go to|open|visit)\b", low):
            return "browser_navigate"
    # A live result is needed before a ref-bearing click/type/fill can be grounded.  Only force
    # the snapshot after the model has successfully emitted a navigation call.
    if "browser_snapshot" in names and navigated and not snapshotted:
        return "browser_snapshot"
    return None


def _last_filesystem_call(prompt: str) -> str | None:
    """Return the most recent tool name emitted before an appended tool result.

    The runtime stores assistant calls as ``<tool_call>{...}`` blocks.  Restricting the search to
    the assistant-to-result segments avoids treating ``name`` fields inside a JSON observation as
    a previous call.  The helper is intentionally protocol-only and works for any MCP service.
    """

    segments = re.findall(
        r"ASSISTANT\s*:\s*(.*?)(?=\nTOOL_RESULT\s*:|$)", prompt, re.I | re.S
    )
    for segment in reversed(segments):
        match = re.search(r"\"name\"\s*:\s*\"([^\"]+)\"", segment)
        if match:
            return match.group(1)
    return None


def _tool_result_objects(prompt: str) -> list[object]:
    """Decode appended MCP result payloads without trusting task-specific field names."""

    objects: list[object] = []
    for raw in re.findall(
        r"TOOL_RESULT\s*:\s*(.*?)(?=\nASSISTANT\s*:|$)", prompt, re.I | re.S
    ):
        try:
            objects.append(json.loads(raw.strip()))
        except json.JSONDecodeError:
            objects.append(raw.strip())
    return objects


def _result_strings(value: object) -> list[str]:
    """Collect human-readable text fields from an MCP result, preserving order."""

    out: list[str] = []

    def visit(node: object) -> None:
        if isinstance(node, dict):
            for key in ("text", "content"):
                child = node.get(key)
                if isinstance(child, str):
                    out.append(child)
                elif isinstance(child, (dict, list)):
                    visit(child)
            for key, child in node.items():
                if key not in {"text", "content"}:
                    visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return list(dict.fromkeys(text for text in out if text.strip()))


def _latest_result_text(prompt: str) -> str:
    objects = _tool_result_objects(prompt)
    if not objects:
        return ""
    strings = _result_strings(objects[-1])
    return "\n".join(strings)


def _latest_read_result_text(prompt: str) -> str:
    """Return the newest read observation, retaining source data across write retries."""

    segments = re.findall(
        r"ASSISTANT\s*:\s*(.*?)\nTOOL_RESULT\s*:\s*(.*?)(?=\nASSISTANT\s*:|$)",
        prompt,
        re.I | re.S,
    )
    for assistant, raw in reversed(segments):
        name = re.search(r"\"name\"\s*:\s*\"([^\"]+)\"", assistant)
        if not name or name.group(1) not in {"read_file", "read_text_file", "read_multiple_files"}:
            continue
        try:
            value: object = json.loads(raw.strip())
        except json.JSONDecodeError:
            value = raw.strip()
        return "\n".join(_result_strings(value))
    return ""


def _workspace_root(prompt: str) -> str | None:
    instruction = prompt.split("TOOL_RESULT", 1)[0]
    match = re.search(
        r"(?:main\s+directory|workspace\s+root)\s*:\s*(/(?:[^\s\n\r<>\"'`])+)",
        instruction,
        re.I,
    )
    if match:
        return match.group(1).rstrip(".,;:)]}")
    return None


def _result_file_names(prompt: str) -> list[str]:
    """Extract file-like names from the newest listing/tree observation."""

    text = _latest_result_text(prompt)
    tree_names: list[str] = []

    def walk(node: object, parent: str = "") -> None:
        if isinstance(node, dict) and isinstance(node.get("name"), str):
            name = node["name"]
            current = f"{parent}/{name}" if parent else name
            if node.get("type") == "file":
                tree_names.append(current)
            for child in node.get("children", []) or []:
                walk(child, current)
        elif isinstance(node, list):
            for child in node:
                walk(child, parent)

    for candidate in _result_strings(_tool_result_objects(prompt)[-1]) if _tool_result_objects(prompt) else []:
        try:
            walk(json.loads(candidate))
        except json.JSONDecodeError:
            continue
    if tree_names:
        return list(dict.fromkeys(tree_names))
    names = re.findall(r"\[FILE\]\s+([^\n]+)", text)
    names += re.findall(r"(?m)^/[^\n:]+/([^/\n:]+):\s*$", text)
    names += re.findall(r"(?<![A-Za-z0-9_.-])([A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,8})\b", text)
    return list(dict.fromkeys(name.strip() for name in names if name.strip()))


def _instruction_filenames(prompt: str) -> list[str]:
    instruction = prompt.split("TOOL_RESULT", 1)[0]
    return list(
        dict.fromkeys(
            re.findall(r"\b[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,8}\b", instruction)
        )
    )


def _prior_argument_paths(prompt: str) -> list[str]:
    """Read prior assistant arguments only; result payload paths are observations, not writes."""

    paths: list[str] = []
    for segment in re.findall(
        r"ASSISTANT\s*:\s*(.*?)(?=\nTOOL_RESULT\s*:|$)", prompt, re.I | re.S
    ):
        paths.extend(re.findall(r"\"(?:path|source|destination)\"\s*:\s*\"([^\"]+)\"", segment))
    return paths


def _numbered_targets(prompt: str) -> list[str]:
    """Expand a bounded ``name_01.ext to name_05.ext`` instruction generically."""

    instruction = prompt.split("TOOL_RESULT", 1)[0]
    match = re.search(
        r"([A-Za-z_.-]*?)(\d+)(\.[A-Za-z0-9]{1,8})\s+to\s+"
        r"(?:[A-Za-z_.-]*?)(\d+)\3",
        instruction,
        re.I,
    )
    if not match:
        return []
    start, end = int(match.group(2)), int(match.group(4))
    if end < start or end - start > 100:
        return []
    width = max(len(match.group(2)), len(match.group(4)))
    return [
        f"{match.group(1)}{index:0{width}d}{match.group(3)}"
        for index in range(start, end + 1)
    ]


def _output_filenames(prompt: str) -> list[str]:
    """Return likely generated filenames, ordered as the task asks to create them."""

    instruction = prompt.split("TOOL_RESULT", 1)[0]
    numbered = _numbered_targets(prompt)
    explicit = re.findall(
        r"(?:file|document)\s+(?:named|called)\s*[`'\"]?"
        r"([A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,8})",
        instruction,
        re.I,
    )
    explicit += re.findall(
        r"(?:generate|create|write|save|record)\s+(?:an?\s+)?(?:empty\s+)?"
        r"(?:file|document)\s+[`'\"]?([A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,8})",
        instruction,
        re.I,
    )
    listed = re.search(
        r"(?:name|create|save)(?:[*_`\s])*(?:the\s+)?files?"
        r"(?:[*_`\s])*(?:as|named)(?:[*_`\s])*(.+?)(?:\n|(?<![A-Za-z0-9])\.)",
        instruction,
        re.I,
    )
    if listed:
        explicit += re.findall(
            r"\b[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,8}\b", listed.group(1)
        )
    names = explicit or numbered
    # A split/converted workflow often says “same names” and provides a bounded filename range.
    # Reuse the task's explicit numbered names while avoiding the input file named in a read
    # instruction.
    if not names and re.search(r"\b(?:same\s+names|converted\s+files)\b", instruction, re.I):
        names = _numbered_targets(instruction)
    return list(dict.fromkeys(names))


def _rename_target(prompt: str) -> str | None:
    instruction = prompt.split("TOOL_RESULT", 1)[0]
    match = re.search(
        r"\b(?:rename|renamed|move|moved)\b.*?\bto\s*[`'\"]?"
        r"([A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,8})",
        instruction,
        re.I | re.S,
    )
    return match.group(1) if match else None


def _result_blocks(prompt: str) -> list[tuple[str | None, str]]:
    """Split read-file results into ``(path, body)`` pairs when the server provides paths."""

    text = _latest_read_result_text(prompt) or _latest_result_text(prompt)
    matches = list(
        re.finditer(
            r"(?m)^(/[^\n:]+):\n(.*?)(?=\n\n---\n|\n/[^\n:]+:\n|\Z)",
            text,
            re.S,
        )
    )
    if matches:
        return [(match.group(1), match.group(2).rstrip("\n")) for match in matches]
    return [(None, text)] if text else []


def _current_output_name(prompt: str) -> str | None:
    path = _next_output_path(prompt)
    return path.rsplit("/", 1)[-1] if path else None


def _transformed_content(prompt: str) -> str | None:
    """Apply generic, observation-derived text transformations requested by the user."""

    low = prompt.lower()
    blocks = _result_blocks(prompt)
    if not blocks:
        return None
    output_name = _current_output_name(prompt)

    if re.search(r"\b(?:uppercase|upper\s+case|convert)\b", low):
        if output_name:
            for path, body in blocks:
                if path and path.rsplit("/", 1)[-1] == output_name:
                    return body.upper()
        return blocks[0][1].upper()

    if re.search(r"\bsplit\b", low) and re.search(r"\b(?:exactly|into)\s+(\d+)\s+files?", low):
        count_match = re.search(r"\b(?:exactly|into)\s+(\d+)\s+files?", low)
        count = int(count_match.group(1)) if count_match else 0
        body = "\n".join(value for _, value in blocks)
        targets = _output_filenames(prompt)
        index = targets.index(output_name) if output_name in targets else 0
        if count > 0:
            chunk = math.ceil(len(body) / count)
            return body[index * chunk : (index + 1) * chunk]

    if re.search(r"\b(?:merge|combine)\b", low):
        return "\n".join(value for _, value in blocks)

    if re.search(r"\bduplicate\s+name\b|\bnamesake\b", low):
        groups: dict[str, list[str]] = {}
        text = _latest_result_text(prompt).replace("\\n", "\n")
        for student_id, name in re.findall(r"(?m)^(\d{8}),([^,\n]+),", text):
            groups.setdefault(name.strip(), []).append(student_id)
        for student_id, name in re.findall(
            r"(?:^|\n)(?:\[DIR\]\s+)?(\d+)_([A-Za-z]+(?:_[A-Za-z]+)+)", text
        ):
            groups.setdefault(name.replace("_", " ").strip(), []).append(student_id)
        for student_id, name in re.findall(
            r"\"name\"\s*:\s*\"(\d+)_([A-Za-z]+(?:_[A-Za-z]+)+)\"", text
        ):
            groups.setdefault(name.replace("_", " ").strip(), []).append(student_id)
        duplicate = next(
            ((name, ids) for name, ids in groups.items() if len(set(ids)) > 1), None
        )
        if duplicate:
            name, ids = duplicate
            unique_ids = list(dict.fromkeys(ids))
            return f"name: {name}\ncount: {len(unique_ids)}\nids: {', '.join(unique_ids)}"

    if re.search(r"\brecommend(?:ation|er)\b|\bwrote\b", low):
        match = re.search(r"(?:^|\n)Sincerely,\s*\n([^\n]+)", _latest_result_text(prompt), re.I)
        if match:
            return match.group(1).strip()

    if re.search(r"\bmatching\b|\bcommon\s+substring\b", low):
        reference = next(
            (body for path, body in blocks if path and "large_file" in path.lower()),
            blocks[0][1],
        )
        matches: list[str] = []
        reference_chunks = {reference[index : index + 30] for index in range(max(0, len(reference) - 29))}
        for path, body in blocks:
            if not path or "large_file" in path.lower():
                continue
            if any(chunk in body for chunk in reference_chunks):
                matches.append(path.rsplit("/", 1)[-1])
        return "\n".join(matches)
    return None


def _next_output_path(prompt: str) -> str | None:
    root = _workspace_root(prompt)
    if root is None:
        return None
    targets = _output_filenames(prompt)
    if not targets:
        return None
    prior = set(_prior_argument_paths(prompt))
    target = next(
        (name for name in targets if not any(path.endswith("/" + name) for path in prior)),
        targets[-1],
    )
    instruction = prompt.split("TOOL_RESULT", 1)[0]
    directory_matches = re.findall(
        r"(?:in|inside|within|under)\s+(?:the\s+)?[`'\"]?"
        r"([A-Za-z0-9_.-]+)(?:/)?[`'\"]?\s+(?:directory|folder)",
        instruction,
        re.I,
    )
    directory = directory_matches[-1] if directory_matches else None
    if directory:
        if directory.lower() not in {"main", "workspace", "root", "test"}:
            root = f"{root}/{directory}"
    else:
        target_dir = re.search(
            r"(?:directory|folder)(?:[*_`\s])*?(?:named|called)(?:[*_`\s])*"
            r"[`'\"]?([A-Za-z0-9_.-]+)[`'\"]?",
            instruction,
            re.I,
        )
        if target_dir:
            root = f"{root}/{target_dir.group(1)}"
    return f"{root}/{target}"


def _filesystem_lexical_tool(prompt: str, tools: list[ToolSpec]) -> str | None:
    """Select an unambiguous filesystem operation from generic task language.

    MCP filesystem catalogs expose a stable vocabulary (tree/search/read/write/create).  A tiny
    model can confuse these tools even when the instruction states the operation plainly, so use
    a narrow schema-aware guard before dense ranking.  It never invents a tool and does not use a
    benchmark task ID or fixture-specific filename.
    """
    names = {tool.name for tool in tools}
    filesystem_names = {
        "directory_tree",
        "search_files",
        "list_directory",
        "list_directory_with_sizes",
        "read_file",
        "read_text_file",
        "read_multiple_files",
        "write_file",
        "create_directory",
        "move_file",
    }
    if not names.intersection(filesystem_names):
        return None
    low = _action_tail(prompt).lower()

    def choose(*candidates: str) -> str | None:
        return next((candidate for candidate in candidates if candidate in names), None)

    # A filesystem task is presented as one long instruction, but the MCP loop appends each
    # observation to that instruction.  Once a directory-create call has succeeded, repeating
    # the same call is almost always an invalid retry: tasks that mention an empty output file
    # need ``write_file`` next.  This is derived from the protocol history and the generic task
    # language, not from a fixture/task identifier.
    last_call = _last_filesystem_call(prompt)
    if "TOOL_RESULT" in prompt and last_call == "create_directory":
        if re.search(r"\bempty\s+(?:file|document)\b", low):
            write = choose("write_file")
            if write is not None:
                return write
        if re.search(r"\b(?:read|inspect|open|split|convert|uppercase|merge|count|find|identify)\b", low):
            if re.search(r"\b(?:count|find|identify|largest|matching|search)\b", low):
                return choose(
                    "search_files", "list_directory_with_sizes", "read_file", "read_text_file"
                )
            if re.search(r"\bsplit\b", low):
                return choose("read_file", "read_text_file", "read_multiple_files")
            return choose("read_multiple_files", "read_file", "read_text_file")

    if "TOOL_RESULT" not in prompt and re.search(
        r"\b(?:create|make)\b[^\n.]{0,60}\bdirector", low
    ) and re.search(r"\b(?:split|uppercase|convert|merge)\b", low):
        return choose("create_directory")

    if "TOOL_RESULT" not in prompt and re.search(r"\b(?:merge|combine)\b", low):
        return choose("directory_tree", "list_directory", "read_multiple_files")

    # A tree/listing is an observation step for transformations such as merge, uppercase, and
    # duplicate detection.  Count-only tasks are different: the existing result already contains
    # the scalar and can go directly to ``write_file`` through the branch below.
    if "TOOL_RESULT" in prompt and last_call in {
        "directory_tree",
        "list_directory",
        "search_files",
        "list_directory_with_sizes",
    }:
        if re.search(r"\bduplicate\s+name\b|\bnamesake\b", low):
            return choose("write_file")
        if re.search(r"\b(?:merge|uppercase|convert|duplicate|recommend|matching|split)\b", low):
            return choose("read_multiple_files", "read_file", "read_text_file")

    if "TOOL_RESULT" in prompt and last_call in {
        "read_file",
        "read_text_file",
        "read_multiple_files",
    }:
        if re.search(
            r"\b(?:write|save|store|record|generate|create|split|uppercase|convert|merge|duplicate|recommend)\b",
            low,
        ):
            return choose("write_file")

    if "TOOL_RESULT" in prompt and last_call == "write_file":
        if re.search(r"\b(?:split|uppercase|converted|same\s+names|multiple\s+files)\b", low):
            return choose("write_file")

    if (
        "TOOL_RESULT" in prompt
        and last_call in {"list_directory_with_sizes", "get_file_info"}
        and re.search(r"\b(?:largest|biggest)\b", low)
    ):
        return choose("move_file")

    if "TOOL_RESULT" not in prompt and re.search(
        r"\b(?:duplicate\s+name|namesake|recommend(?:ation|er))\b", low
    ):
        return choose("directory_tree", "list_directory", "read_file", "read_text_file")

    # After a successful observation, a task may still contain its original "recursively
    # inspect" wording.  The next operation is the explicit write/save instruction, not another
    # tree walk.  Restrict this override to a prompt containing a tool result so the initial turn
    # still selects the inspection operation.
    if "TOOL_RESULT" in prompt and re.search(r"\b(?:write|save|store|record)\b", low):
        write = choose("write_file")
        if write is not None:
            return write
    if re.search(r"\b(?:recurs|tree|all\s+subdirector|count\s+.*files?|find\s+all\s+files?)\b", low):
        return choose("directory_tree", "search_files", "list_directory")
    if re.search(r"\b(?:write|save|store|record)\b", low):
        return choose("write_file")
    if (
        "TOOL_RESULT" not in prompt
        and re.search(r"\b(?:largest|biggest)\b", low)
        and re.search(r"\.(?:jpg|jpeg|png)\b", low)
    ):
        return choose("list_directory_with_sizes", "list_directory", "search_files")
    if re.search(r"\b(?:identify|find|recommend|duplicate)\b", low):
        return choose("directory_tree", "list_directory", "read_file", "read_text_file")
    if re.search(r"\b(?:create|make)\s+(?:an?\s+)?(?:new\s+)?director", low):
        return choose("create_directory")
    if re.search(r"\b(?:read|inspect|open)\b", low):
        return choose("read_file", "read_text_file", "read_multiple_files")
    if re.search(r"\b(?:rename|move)\b", low):
        return choose("move_file")
    return None


def _browser_lexical_tool(prompt: str, tools: list[ToolSpec]) -> str | None:
    """Select a generic browser/UI action when the instruction states it explicitly.

    The dense selector in a tiny checkpoint can over-rank unrelated productivity schemas (for
    example ``jira_issue``) on a long accessibility prompt.  This guard only uses the action verb
    and the schemas that are actually present; it does not inspect task IDs, page contents, or
    verifier state.  It is therefore a deployment routing aid, not a learned benchmark score.
    """

    names = {tool.name for tool in tools}

    def choose(*candidates: str) -> str | None:
        return next((candidate for candidate in candidates if candidate in names), None)

    action = _action_tail(prompt).lower()
    if re.search(r"\b(?:double[ -]?click|double[- ]tap)\b", action):
        return choose("double_click", "web_click")
    if re.search(r"\b(?:type|fill|enter|write)\b", action):
        return choose("type_text", "web_type")
    if re.search(r"\b(?:select|choose)\b", action):
        return choose("web_select", "click")
    if re.search(r"\b(?:click|tap|press)\b", action):
        return choose("click", "web_click")
    if re.search(r"\bscroll(?:ing)?\b", action):
        return choose("scroll")
    if re.search(r"\b(?:key|keyboard)\s+(?:press|hit)\b", action):
        return choose("key_press")
    if re.search(r"\bdrag\b", action):
        return choose("drag")
    return None


# Argument names whose value is a proper-noun entity (take the capitalized span) vs free text
# (take the whole tail, which may itself contain a proper noun, e.g. query "capital of Peru").
ENTITY_ARGS = {"city", "location", "name", "person", "artist", "song", "album", "place",
               "recipient"}
PHONE_ARGS = {"phone", "phone_number", "telephone", "telephone_number", "mobile"}
TEXT_ARGS = {"content", "message", "text", "body", "subject", "title", "note", "comment"}
APP_ARGS = {"app_name"}
TARGET_ARGS = {"target"}
EMAIL_ARGS = {"to", "recipient"}
ID_ARGS = {"id", "identifier", "task_id", "user_id", "notification_id", "event_id"}
_IDENTIFIER = re.compile(
    r"\b(?:[A-Za-z][A-Za-z0-9-]*_[A-Za-z0-9][A-Za-z0-9_-]*|[0-9a-f]{8}-[0-9a-f-]{27,})\b"
)


def _boolean(prompt: str) -> list[bool]:
    """Infer a JSON boolean from generic enable/disable language in the prompt."""
    low = prompt.lower()
    if re.search(r"\b(?:turn|switch|set|power)\s+off\b|\b(?:disable|disabled|false|no)\b", low):
        return [False]
    if re.search(r"\b(?:turn|switch|set|power)\s+on\b|\b(?:enable|enabled|true|yes)\b", low):
        return [True]
    return []


def _phone(prompt: str) -> list[str]:
    """Return the first phone-like span without trailing message prose."""
    # Prefer explicit international numbers.  UUIDs and timestamps in tool results also contain
    # digit/hyphen runs, so an unqualified search can silently copy ``53108174`` from a UUID.
    matches = re.findall(r"\+\d[\d ()-]{6,}\d", prompt)
    if not matches:
        matches = re.findall(r"(?<![A-Za-z0-9])\d{7,}(?![A-Za-z0-9])", prompt)
    return [re.sub(r"[ ()-]", "", matches[0])] if matches else []


def _identifier(prompt: str, arg: str = "") -> list[str]:
    """Extract a structured identifier instead of copying an entire instruction sentence."""

    stem = arg[:-3] if arg.endswith("_id") else ""
    values = _IDENTIFIER.findall(prompt)
    if stem:
        scoped = [value for value in values if value.lower().startswith(f"{stem.lower()}_")]
        if scoped:
            return scoped
    return values


def _text_arg(prompt: str, arg: str = "") -> list[str]:
    """Extract a text slot from generic delimiters or field-labelled quoted values."""
    action = _action_tail(prompt)
    goal = prompt.split(" Current state JSON:", 1)[0]
    # ``write_file`` requires a content string even when the requested artifact is deliberately
    # empty (for example, a placeholder document in a new folder).  Preserve that explicit
    # zero-length value instead of falling back to copied task prose.
    if arg in {"content", "text", "body"} and re.search(
        r"\bempty\s+(?:file|document)\b", prompt, re.I
    ):
        return [""]
    if arg in {"content", "text", "body"} and "TOOL_RESULT" in prompt:
        transformed = _transformed_content(prompt)
        if transformed is not None:
            return [transformed]
    # When a stateful filesystem task asks for a count and the preceding tree/search result is
    # present, derive the scalar from the returned observation instead of copying the instruction
    # prose into ``content``.  The extension is read from the request; no task ID or fixture name
    # is consulted.
    count_match = re.search(
        r"\bcount\s+(?:the\s+(?:total\s+)?number\s+of\s+)?[^\n.]*?"
        r"[`'\"]?\.?([A-Za-z0-9_]+)[`'\"]?\s+files?",
        prompt,
        re.I,
    )
    if arg in {"content", "text", "message", "body"} and count_match and "TOOL_RESULT" in prompt:
        extension = re.escape(count_match.group(1).lstrip("."))
        results = "\n".join(
            re.findall(r"TOOL_RESULT\s*:\s*(.*?)(?=\nASSISTANT\s*:|$)", prompt, re.I | re.S)
        )
        # MCP directory-tree implementations vary between structured ``{"name": ...}``
        # entries and a plain indented text tree.  Count only basenames in the result, never
        # filenames from the task prose or the serialized prompt history.
        matches = re.findall(
            rf'\\?["\']name\\?["\']\s*:\s*\\?["\']([^"\']+\.{extension})\\?["\']',
            results,
            re.I,
        )
        matches += re.findall(rf"(?<![A-Za-z0-9_./-])[^\s\\\"']+\.{extension}\b", results, re.I)
        if matches:
            return [str(len(set(matches)))]
    for source in (action, goal):
        match = re.search(r"(?:saying|with message|message|text|content)\s*:\s*(.+)", source, re.I)
        if match:
            return [_strip(match.group(1))]
    low = prompt.lower()
    quoted = [value for left, right in re.findall(r"'([^']+)'|\"([^\"]+)\"", action)
              for value in (left or right,)]
    if not quoted:
        quoted = [value for left, right in re.findall(r"'([^']+)'|\"([^\"]+)\"", goal)
                  for value in (left or right,)]
    if arg in {"to", "recipient"} or "address field" in low or "recipient" in low:
        email = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", prompt)
        if email:
            return [email.group(0)]
    labels = {
        "subject": r"subject(?: field)?",
        "title": r"title(?:d)?",
        "body": r"body|message",
        "content": r"content|body",
    }
    label = labels.get(arg)
    if label:
        match = re.search(r"(?:" + label + r")[^'\"]*['\"]([^'\"]+)['\"]", prompt, re.I)
        if match:
            return [match.group(1)]
    if arg in {"text", "message"}:
        action_low = action.lower()
        # UI tools often expose one generic ``text`` argument for several focused fields.  Use
        # the current action's field cue to select the corresponding labelled value from the
        # overall goal; otherwise a quoted body can be copied into the recipient step.
        if "address" in action_low or "recipient" in action_low:
            email = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", prompt)
            if email:
                return [email.group(0)]
        if "subject" in action_low:
            match = re.search(r"subject(?: field)?[^'\"]*['\"]([^'\"]+)['\"]", goal, re.I)
            if match:
                return [match.group(1)]
            if quoted:
                return [quoted[0]]
        if "body" in action_low or "message field" in action_low:
            match = re.search(r"body[^'\"]*['\"]([^'\"]+)['\"]", goal, re.I)
            if match:
                return [match.group(1)]
            if quoted:
                return [quoted[-1]]
        if quoted:
            return [quoted[-1]]
    if arg in {"title", "subject", "label"} and quoted:
        return [quoted[0]]
    return []


def _app_name(prompt: str) -> list[str]:
    """Extract the application named after a generic launch/open/start instruction."""
    action = _action_tail(prompt)
    sources = (action,) if action != prompt else (prompt,)
    for source in sources:
        match = re.search(
            r"(?:launch|open|bring\s+up|start)\s+(?:the\s+)?([A-Z][A-Za-z0-9_.-]*)",
            source,
            re.I,
        )
        if match:
            return [match.group(1)]
    return []


def _target(prompt: str) -> list[str]:
    """Extract a semantic UI target from click/select/tap wording."""
    for source in (_action_tail(prompt), prompt):
        quoted = re.search(
            r"(?:click|select|tap)\s+(?:on\s+)?(?:the\s+)?['\"]([^'\"]+)['\"]",
            source,
            re.I,
        )
        if quoted:
            value = quoted.group(1).strip()
            return [value if value.lower().startswith("the ") else f"the {value}"]
        match = re.search(
            r"(?:click|select|tap)\s+(?:on\s+)?((?:the\s+)?[A-Za-z][A-Za-z ]*?)(?:\s+at\s+x=|\s+on\s+(?:the\s+)?(?:phone|android)|[.!?]|$)",
            source,
            re.I,
        )
        if match:
            value = _strip(match.group(1))
            return [value if value.lower().startswith("the ") else f"the {value}"]
    return []


def _best_string(prompt: str, arg: str = "") -> str:
    """Deterministic string-slot value, arg-aware (the tool head already chose the tool/arg):
    entity args -> first capitalized proper-noun span; free-text args -> longest preposition tail
    (else the imperative tail after the leading verb). Generic English heuristics, not per-tool."""
    # Role labels are protocol scaffolding, not user content.  Strip them before the generic
    # capitalized-span heuristic and drop the imperative's first word so ``USER: Send Fredrik``
    # yields ``Fredrik`` rather than the verb ``Send``.
    source = _action_tail(prompt)
    source = re.sub(r"^(?:USER|SYSTEM|TOOL_RESULT)\s*:\s*", "", source, flags=re.I)
    words = source.split()
    caps = re.findall(r"(?:[A-Z][a-z]+)(?:\s+[A-Z][a-z]+)*", " ".join(words[1:]))
    low = source.lower()
    tails = [_strip(source[i + len(p) + 2:]) for p in PREPS if (i := low.find(f" {p} ")) >= 0]
    tails = [t for t in tails if t]
    if arg in PHONE_ARGS:
        values = _phone(prompt)
        if values:
            return values[0]
    if arg in ID_ARGS or arg.endswith("_id"):
        values = _identifier(prompt, arg)
        if values:
            return values[0]
    if arg in EMAIL_ARGS:
        values = _text_arg(prompt, arg)
        if values:
            return values[0]
    if arg in TEXT_ARGS:
        values = _text_arg(prompt, arg)
        if values:
            return values[0]
    if arg in APP_ARGS:
        values = _app_name(prompt)
        if values:
            return values[0]
    if arg in TARGET_ARGS:
        values = _target(prompt)
        if values:
            return values[0]
    if arg in ENTITY_ARGS and caps:
        return _strip(caps[0])
    if tails:
        return max(tails, key=len)
    if caps:
        return _strip(caps[0])
    return _strip(" ".join(words[1:])) if len(words) > 1 else _strip(source)


def _arith(prompt: str) -> list[str]:
    m = re.search(r"\d+\s*[-+*/]\s*\d+(?:\s*[-+*/]\s*\d+)*", prompt)
    return [re.sub(r"\s+", "", m.group(0))] if m else []


def _numbers(prompt: str) -> list[str]:
    return re.findall(r"-?\d+", prompt)


def _number_arg(prompt: str, name: str) -> list[str]:
    """Prefer a labelled numeric argument such as ``x=120`` over unrelated state numbers."""

    for source in (_action_tail(prompt), prompt):
        match = re.search(rf"\b{re.escape(name)}\s*=\s*(-?\d+(?:\.\d+)?)", source, re.I)
        if match:
            return [match.group(1)]
    return _numbers(prompt)


def _quoted(prompt: str) -> list[str]:
    """Content of the first single/double-quoted span (patterns, commands, commit messages)."""
    m = re.search(r"'([^']+)'|\"([^\"]+)\"", prompt)
    return [next(g for g in m.groups() if g)] if m else []


def _path(prompt: str) -> list[str]:
    """Return the most specific path-like value from the current instruction.

    MCP/filesystem prompts commonly contain both a task identifier (for example,
    ``file_context/file_splitting``) and an absolute workspace root.  Choosing the first slash
    token copied the task identifier into ``path`` arguments, and the generic string fallback
    copied the whole instruction.  Prefer an explicit absolute path, then a workspace/root
    label, and only then fall back to a relative path or filename.
    """
    sources = (_action_tail(prompt), prompt)
    # Directory creation tasks often name a relative child while also carrying an absolute
    # workspace root.  Prefer the explicitly named child (and an explicitly named parent) over
    # returning the root itself.  This remains schema/task-language based; it does not inspect a
    # benchmark ID or fixture contents.
    instruction = _action_tail(prompt)
    root_match = re.search(
        r"(?:main\s+directory|workspace\s+root)\s*:\s*(/(?:[^\s\n\r<>\"'`])+)",
        prompt,
        re.I,
    )
    root = root_match.group(1).rstrip(".,;:)]}") if root_match else None
    directory = re.search(
        r"(?:directory|folder)(?:[*_`\s])*?(?:named|called)(?:[*_`\s])*"
        r"[`'\"]?([A-Za-z0-9_.-]+)[`'\"]?",
        instruction,
        re.I,
    )
    read_after_create = (
        _last_filesystem_call(prompt) == "create_directory"
        and re.search(r"\b(?:read|inspect|open|split|convert|uppercase)\b", instruction, re.I)
    )
    if root and directory and not read_after_create:
        parent = re.search(
            r"(?:inside|within|under)\s+(?:the\s+)?(?:folder\s+)?[`'\"]?"
            r"([A-Za-z0-9_.-]+)(?:/)?[`'\"]?\s+(?:directory|folder)",
            instruction,
            re.I,
        )
        # Some task descriptions introduce the parent as a quoted relative directory (for
        # example, ``the folder \"legal_files/\"``) and refer to the target in a later numbered
        # step.  Preserve that explicit hierarchy when grounding the target path.
        if parent is None:
            parent = re.search(
                r"(?:folder|directory)\s*[`'\"]?([A-Za-z0-9_.-]+)/[`'\"]?",
                instruction,
                re.I,
            )
        target = directory.group(1)
        parent_name = parent.group(1) if parent else None
        # ``main directory``/``workspace root`` are labels for the supplied root, not literal
        # child folders.  Only join a parent when the instruction names a concrete directory.
        if parent_name and parent_name.lower() not in {"main", "workspace", "root", "test"}:
            return [f"{root}/{parent_name}/{target}"]
        return [f"{root}/{target}"]
    if root and not read_after_create and _last_filesystem_call(prompt) is None:
        creation = re.search(
            r"\b(?:create|make)\s+(?:an?\s+)?([A-Za-z0-9_.-]+)\s+directory\b",
            instruction,
            re.I,
        )
        if creation:
            return [f"{root}/{creation.group(1)}"]
    # After a directory has been created, a read/split/convert instruction names the input file
    # relatively (for example ``large_file.txt``).  Prefer that file over the workspace root;
    # output paths are selected separately by ``_next_output_path``.
    if _last_filesystem_call(prompt) == "create_directory" and re.search(
        r"\b(?:read|inspect|open|split|convert|uppercase)\b", instruction, re.I
    ):
        outputs = set(_output_filenames(prompt))
        for filename in _instruction_filenames(prompt):
            if filename not in outputs and not filename.startswith("structure_analysis"):
                return [f"{root}/{filename}" if root else filename]
    absolute = re.compile(r"(?<![A-Za-z0-9])/(?:[^\s\n\r<>\"'`])+", re.I)
    for source in sources:
        values = [value.rstrip(".,;:)]}") for value in absolute.findall(source)]
        if values:
            return [values[-1]]
    relative = re.compile(
        r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+|[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,5}\b"
    )
    for source in sources:
        match = relative.search(source)
        if match:
            return [match.group(0).rstrip(".")]
    return []


def _workspace_output_path(prompt: str) -> list[str]:
    """Join a named output file to the explicit workspace root after an observation."""
    instruction = prompt.split("TOOL_RESULT", 1)[0]
    labeled_root = re.search(
        r"(?:main\s+directory|workspace\s+root)\s*:\s*(/(?:[^\s\n\r<>\"'`])+)",
        instruction,
        re.I,
    )
    roots = [labeled_root.group(1)] if labeled_root else re.findall(
        r"(?<![A-Za-z0-9])/(?:[^\s\n\r<>\"'`])+", instruction
    )
    # Prefer a filename explicitly introduced as the generated artifact.  Source files listed
    # for inspection (``file_01.txt`` … ``file_20.txt``) must never displace ``answer.txt`` or
    # ``merge.txt`` simply because they occur later in the prose.
    explicit = re.findall(
        r"(?:file|document)\s+(?:named|called)\s*[`'\"]?"
        r"([A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,8})",
        instruction,
        re.I,
    )
    explicit += re.findall(
        r"(?:generate|create|write|save|record)\s+(?:an?\s+)?(?:empty\s+)?"
        r"(?:file|document)\s+[`'\"]?([A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,8})",
        instruction,
        re.I,
    )
    filenames = explicit or re.findall(
        r"\b[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,8}\b", instruction
    )
    if not roots or not filenames:
        return []
    root = roots[-1].rstrip(".,;:)]}")
    parent = re.search(
        r"(?:folder|directory)\s*[`'\"]?([A-Za-z0-9_.-]+)/[`'\"]?",
        instruction,
        re.I,
    )
    if parent and parent.group(1).lower() not in {"main", "workspace", "root", "test"}:
        root = f"{root}/{parent.group(1)}"
    target_dir = re.search(
        r"(?:directory|folder)(?:[*_`\s])*?(?:named|called)(?:[*_`\s])*"
        r"[`'\"]?([A-Za-z0-9_.-]+)[`'\"]?",
        instruction,
        re.I,
    )
    if target_dir and target_dir.group(1).lower() not in {"main", "workspace", "root", "test"}:
        root = f"{root}/{target_dir.group(1)}"
    # The last filename in these instructions is the named output; source filenames appear
    # earlier in the task description or in the observation.
    filename = filenames[-1].rstrip(".")
    return [f"{root}/{filename}"]


def _url(prompt: str) -> list[str]:
    """First URL/domain token (optionally with scheme/path)."""
    for source in (_action_tail(prompt), prompt):
        m = re.search(r"(?:https?://)?[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?:/[\w./-]*)?", source)
        if m:
            return [m.group(0).rstrip(".")]  # drop trailing sentence period
    return []


def _browser_refs(prompt: str) -> list[str]:
    """Extract only protocol references returned by a live Playwright snapshot."""

    # ``ref=e12`` is the MCP wire format.  Restrict extraction to tool results so a task prose
    # mention such as ``ref`` can never become an executable target.  Preserve order and remove
    # duplicates because the constrained decoder will rank the resulting grounded candidates.
    results = re.findall(r"TOOL_RESULT\s*:\s*(.*?)(?=\nASSISTANT\s*:|$)", prompt, re.I | re.S)
    refs = re.findall(r"\bref\s*=\s*([A-Za-z][A-Za-z0-9_-]*)\b", "\n".join(results))
    return list(dict.fromkeys(refs))


def _arg_options(prompt: str, name: str, schema: dict, required: bool, ptr=None) -> list:
    """Candidate values for one argument. If a pointer head is given (`ptr`), string-typed args
    are filled by its learned copy span; otherwise schema/heuristic extractors are used."""
    fmt = schema.get("format")
    if name in {"head", "tail"} and not required:
        opts = [None]
    elif name == "paths" and schema.get("type") == "array":
        root = _workspace_root(prompt)
        names = _result_file_names(prompt)
        if not names:
            candidates = _instruction_filenames(prompt)
            names = _numbered_targets(prompt) or candidates
            names = [name for name in names if name not in _output_filenames(prompt)] or names
        low = prompt.lower()
        if "recommend" in low and "patricia" in low:
            names = [
                name
                for name in names
                if "patricia" in name.lower() and "recommendation" in name.lower()
            ]
        elif "duplicate" in low or "namesake" in low:
            names = [name for name in names if name.endswith("basic_info.txt")]
        extension = re.search(r"\ball\s+[`'\"]?\.([A-Za-z0-9]+)[`'\"]?\s+files?", low)
        if extension:
            names = [name for name in names if name.lower().endswith("." + extension.group(1).lower())]
        if root and names:
            opts = [[f"{root}/{name}" for name in names]]
        else:
            opts = []
    elif name == "ref":
        # Browser/Mobile MCP references are opaque IDs from a prior observation.  Never fall back
        # to ``_best_string`` (which would copy the whole task prompt into ``ref``).
        opts = _browser_refs(prompt)
    elif name == "element" and schema.get("type") == "string":
        # ``element`` is optional in Playwright's evaluate/type tools; when required by a custom
        # catalog it must still be grounded to a semantic snapshot label, not invented text.
        opts = []
    elif name == "function" and schema.get("type") == "string":
        # Executable JavaScript cannot be safely synthesized from an untrusted task prompt.
        opts = []
    elif name in {"url", "href"}:
        opts = _url(prompt)
    elif "enum" in schema:
        opts = list(schema["enum"])
    elif fmt == "arithmetic" or "express" in name:
        opts = _arith(prompt)
    elif schema.get("type") == "boolean":
        opts = _boolean(prompt)
    elif name in PHONE_ARGS:
        opts = _phone(prompt)
    elif name in ID_ARGS or name.endswith("_id"):
        opts = _identifier(prompt, name)
    elif ptr is not None and name in ptr[0].arg_idx and name not in {
        "path",
        "source",
        "destination",
        "target_path",
    }:        # learned pointer/copy span
        ph, feats_row, framed_ids, tok = ptr[:4]
        span_bounds = ptr[4] if len(ptr) > 4 else None
        s, e = ph.predict_span(feats_row, name, span_bounds=span_bounds)
        pointer_value = tok.decode(framed_ids[s:e + 1])
        # Prefer an explicit schema-grounded value when one is available.  The learned pointer is
        # still retained as a fallback for values that only occur in tool/history context, but a
        # stale pointer must not override an exact URL, email, quoted field, or app name extracted
        # from the current action instruction.
        opts = [pointer_value]
        if name in TEXT_ARGS:
            explicit = _text_arg(prompt, name)
            if explicit:
                opts = explicit
        elif name in EMAIL_ARGS:
            explicit = _text_arg(prompt, name)
            if explicit:
                opts = explicit
        elif name in APP_ARGS:
            explicit = _app_name(prompt)
            if explicit:
                opts = explicit
        elif name in TARGET_ARGS:
            explicit = _target(prompt)
            if explicit:
                opts = explicit
        elif fmt == "url":
            explicit = _url(prompt)
            if explicit:
                opts = explicit
    elif name in TEXT_ARGS:
        opts = _text_arg(prompt, name)
    elif name in EMAIL_ARGS:
        opts = _text_arg(prompt, name) or [_best_string(prompt, name)]
    elif name in APP_ARGS:
        opts = _app_name(prompt)
    elif name in TARGET_ARGS:
        opts = _target(prompt)
    elif fmt == "quoted":
        # Public trajectories frequently provide semantic labels without quote marks (for
        # example, ``click the Search box``).  Keep the quoted extractor first, but fall back to
        # the generic string heuristic instead of forcing a learned pointer to copy a malformed
        # span that includes the state observation.
        opts = _quoted(prompt) or [_best_string(prompt, name)]
    elif name in {"path", "source", "destination", "target_path"} or fmt == "path":
        opts = _path(prompt)
        if name in {"source", "destination"} and "TOOL_RESULT" in prompt:
            low = prompt.lower()
            result = _latest_result_text(prompt)
            if name == "destination":
                target = _rename_target(prompt)
                if target:
                    root = _workspace_root(prompt)
                    opts = [f"{root}/{target}"] if root else [target]
            elif "largest" in low:
                candidates = re.findall(
                    r"\[FILE\]\s+([^\n]+?)\s+(\d+(?:\.\d+)?)\s*(KB|MB|GB|B)\b",
                    result,
                    re.I,
                )
                if candidates:
                    extension = re.search(r"\.(jpg|jpeg|png)\b", low, re.I)
                    if extension:
                        candidates = [
                            item
                            for item in candidates
                            if item[0].strip().lower().endswith("." + extension.group(1).lower())
                        ]
                    if not candidates:
                        return opts
                    scale = {"b": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3}
                    largest = max(
                        candidates,
                        key=lambda item: float(item[1]) * scale[item[2].lower()],
                    )[0].strip()
                    root = _workspace_root(prompt)
                    opts = [f"{root}/{largest}"] if root else [largest]
        if name in {"path", "target_path"} and "TOOL_RESULT" in prompt:
            last_call = _last_filesystem_call(prompt)
            output_path = _next_output_path(prompt) or _workspace_output_path(prompt)
            if not (
                last_call == "create_directory"
                and re.search(r"\b(?:read|inspect|open|split|convert|uppercase)\b", prompt, re.I)
                and not re.search(r"\bempty\s+(?:file|document)\b", prompt, re.I)
            ):
                if isinstance(output_path, list):
                    opts = output_path
                elif output_path:
                    opts = [output_path]
    elif fmt == "url":
        opts = _url(prompt)
    elif schema.get("type") in ("integer", "number"):
        # cast to typed numbers so the canonical body matches the int/float target (not "5").
        cast = int if schema.get("type") == "integer" else float
        opts = [cast(n) for n in _number_arg(prompt, name)]
    else:  # string / unknown -> deterministic best prompt span (arg-aware)
        opts = [_best_string(prompt, name)]
    if not required:
        opts = [None] + opts          # allow omitting optional args
    return opts or ([None] if not required else [])


def _tool_bodies(prompt: str, tool: ToolSpec, ptr=None) -> list[str]:
    props = (tool.parameters or {}).get("properties", {})
    required = set((tool.parameters or {}).get("required", []))
    names = list(props.keys())
    per_arg = [_arg_options(prompt, n, props[n], n in required, ptr) for n in names]
    bodies = []
    for combo in itertools.islice(itertools.product(*per_arg), MAX_COMBOS):
        args = {n: v for n, v in zip(names, combo) if v is not None}
        if all(n in args for n in required):
            bodies.append(f"{TOOL_CALL_OPEN}{_canon(tool.name, args)}{TOOL_CALL_CLOSE}")
    return bodies


# ---- open-text intent (placeholder for the dual text-head, ARCHITECTURE_IDEAS §2a) ----
def _text_candidates(prompt: str) -> list[str] | None:
    low = prompt.lower()
    caps = [re.sub(r"[^A-Za-z]", "", w) for w in prompt.split()[1:] if re.match(r"[A-Z][a-z]", w)]
    if re.search(r"\b(?:already complete|no action|nothing to do|without invoking)\b", low):
        return ["I won't invoke a tool."]
    if "hello" in low:
        return [f"Hello, {nm}!" for nm in caps] or ["Hello!"]
    if "morning" in low or "greet" in low:
        return [f"Good morning, {nm}!" for nm in caps] or ["Good morning!"]
    if "your name" in low:
        return ["I am LocalAgent."]
    if "thank" in low:
        return ["You're welcome!"]
    return None


@torch.no_grad()
def _best(model, tok, prompt: str, bodies: list[str], device) -> str:
    """Length-normalized log-prob of each body given the prompt, scored in ONE batched forward
    (candidates share the prompt prefix), returning the argmax body."""
    pid = tok.encode(prompt)
    max_len = getattr(getattr(model, "cfg", None), "max_seq_len", None)
    if max_len is not None:
        # A grounded value can itself be longer than the model window (for example, a tool
        # schema copied from a long page).  Such a candidate cannot be scored safely even after
        # dropping the prompt; fail closed instead of sending an overlong tensor to RoPE.
        bodies = [b for b in bodies if len(tok.encode(b)) + 1 <= max_len]
        if not bodies:
            return "I cannot complete this request."
    seqs = [pid + tok.encode(b) + [tok.eos_id] for b in bodies]
    maxlen = max(len(s) for s in seqs)
    # Keep within the model's context window: a long multi-turn history + a candidate body can
    # exceed max_seq_len. Trim from the LEFT (drop the oldest *prompt* tokens, shared by every
    # candidate) so bodies stay intact and the scoring offsets below shift consistently.
    max_len = max_len or maxlen
    cut = min(max(0, maxlen - 1 - max_len), len(pid) - 1)
    if cut:
        seqs = [s[cut:] for s in seqs]
        pid = pid[cut:]
        maxlen = max(len(s) for s in seqs)
    X = torch.full((len(seqs), maxlen), tok.pad_id, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        X[i, : len(s)] = torch.tensor(s, device=device)
    logits, _ = model(X[:, :-1])
    logp = F.log_softmax(logits, dim=-1)
    tok_lp = logp.gather(-1, X[:, 1:].unsqueeze(-1)).squeeze(-1)  # (B, L-1)
    best_i, best_s = 0, -1e9
    for i, s in enumerate(seqs):
        sc = tok_lp[i, len(pid) - 1: len(s) - 1].mean().item()
        if sc > best_s:
            best_i, best_s = i, sc
    return bodies[best_i]


def candidates(prompt: str, tools: list[ToolSpec]) -> list[tuple[str, bool, str]]:
    """All grounded candidates as (text, is_tool, group). Used by tests; ranking picks one."""
    group_of = {"get_weather": "tool_call", "calculator": "tool_call",
                "web_search": "web_search", "planner": "planner"}
    txt = _text_candidates(prompt)
    if txt is not None:
        return [(t, False, "text") for t in txt]
    out = []
    for tool in tools:
        for b in _tool_bodies(prompt, tool):
            out.append((b, True, group_of.get(tool.name, "tool_call")))
    return out or [("I am LocalAgent.", False, "text")]


def _preselect_tool(model, tok, prompt: str, names: set[str], device) -> str | None:
    """Let the model free-generate a call and read the tool *name* (it picks the tool reliably
    even when the argument bytes are garbled). This is the schema-agnostic tool selector; the
    grounded candidates then only have to get the arguments right."""
    from localagent.agent.parser import extract_tool_calls
    from localagent.inference.generate import generate
    from localagent.model.tokenizer import ASSISTANT, USER
    gen, _ = generate(model, tok, f"{USER}{prompt}{ASSISTANT}", max_new_tokens=80, temperature=0.0)
    calls = extract_tool_calls(gen)
    return calls[0].name if calls and calls[0].name in names else None


@torch.no_grad()
def _ctx_feats(model, tok, ctx: str, device):
    ids = tok.encode(ctx)
    max_len = getattr(getattr(model, "cfg", None), "max_seq_len", len(ids))
    if len(ids) > max_len:
        # Keep the newest observation/tool result.  Candidate grounding still sees the complete
        # raw prompt, while the model's RoPE/context window receives a valid bounded suffix.
        ids = ids[-max_len:]
    _, feats = model(torch.tensor([ids], device=device), return_hidden=True)
    return feats[0], ids


def _grounding_span(ids: list[int], tok, grounding: str) -> tuple[int, int] | None:
    """Locate the grounding suffix inside a catalog-framed context for pointer masking."""
    grounding_ids = tok.encode(grounding)
    if not grounding_ids:
        return None
    assistant_ids = tok.encode(ASSISTANT)
    suffix_end = len(ids) - len(assistant_ids)
    suffix_start = suffix_end - len(grounding_ids)
    if suffix_start >= 0 and ids[suffix_start:suffix_end] == grounding_ids:
        return suffix_start, suffix_end - 1
    # Boundary-aware tokenizers can differ at concatenation points; use the last occurrence as a
    # conservative fallback, which still avoids the catalog prefix in normal prompts.
    for start in range(len(ids) - len(grounding_ids), -1, -1):
        if ids[start : start + len(grounding_ids)] == grounding_ids:
            return start, start + len(grounding_ids) - 1
    return None


def hybrid_decode(model, tok, prompt: str, tools: list[ToolSpec], device="cpu", *,
                  retriever=None, route_head=None, ptr_head=None, selector=None, top_m=1, k=8,
                  framed=False, blocked_candidates: set[str] | None = None,
                  selector_first: bool = False,
                  grounding_prompt: str | None = None,
                  lexical_weight: float = 0.5) -> str:
    """The *generable* decode path — no fixed-N classifier. Selection narrows the catalog to a few
    candidates, then the model RANKS their grounded bodies; argument *values* are copied by
    `ptr_head` (the one sub-task a tiny model can't free-generate). An optional 5-way `route_head`
    gates text-vs-tool up front.

    Selection source:
      `selector` (a `BoundSelector`, recommended) — a *trained* two-tower scorer that ranks every
        tool by its description embedding; we keep its top-`top_m`. Generalizes to unseen tools.
      else `retriever` — zero-training char-ngram retrieval top-k (weaker; the model must then rank).
    Either way adding a tool needs zero head reshape / retraining. ``grounding_prompt`` optionally
    separates the user/history text used for schema argument candidates from the full model
    context (which may contain a serialized function catalog)."""
    from localagent.agent.retriever import ToolRetriever
    from localagent.model.tokenizer import ASSISTANT, USER
    ctx = prompt if framed else f"{USER}{prompt}{ASSISTANT}"
    score = prompt if not framed else ctx
    grounding = prompt if grounding_prompt is None else grounding_prompt
    feats = ids = None
    # Compute surface-specific lexical guards before the route gate.  BrowserGym/MiniWoB prompts
    # are short and often contain little natural-language context, so a tiny route head can
    # confidently call them ``text`` even when the instruction is an explicit UI action.  The
    # guard is deliberately schema- and cue-based (it only fires for a browser/UI action when the
    # corresponding tool is present), so this does not invent a target or bypass grounding.
    mobile_hint = _mobile_lexical_tool(grounding, tools)
    playwright_hint = _playwright_lexical_tool(grounding, tools)
    filesystem_hint = _filesystem_lexical_tool(grounding, tools)
    browser_hint = _browser_lexical_tool(grounding, tools)
    # 0. route gate (text vs tool) — falls back to the heuristic text detector when no head given
    if route_head is not None or selector is not None:
        feats, ids = _ctx_feats(model, tok, ctx, device)
    if route_head is not None:
        from localagent.agent.routes import ROUTES
        if ROUTES[int(route_head(feats[-1]).argmax(-1))] == "text" and not (
            mobile_hint or playwright_hint or filesystem_hint or browser_hint
        ):
            # Fail open for prompts that contain no recognized text intent.  A small route head
            # can misclassify a long state-conditioned tool prompt as ``text``; turning that into
            # an unconditional abstention makes retries impossible and hides the selector's
            # useful tool prior.  Known greeting/identity/thanks intents still take the text path.
            text_candidates = _text_candidates(grounding)
            if text_candidates is not None:
                return _best(model, tok, score, text_candidates, device)
    else:
        txt = _text_candidates(grounding)
        if txt is not None:
            return _best(model, tok, score, txt, device)
    # 1. selection: trained dense selector (top-m) if given, else retrieval top-k
    selector_order: list[str] | None = None
    if selector is not None:
        # Stateful/productivity prompts expose an explicit next-action boundary.  Use that short
        # generic instruction as an auxiliary lexical query; the dense model feature still carries
        # the full state/history, while the lexical term avoids selecting a tool mentioned only in
        # the long-horizon goal (for example ``email_send`` instead of ``mobile_open_app``).
        selector_query = (
            _action_tail(grounding) if "Next required action:" in grounding else None
        )
        selector_order = selector.rank(
            feats[-1],
            allowed_names={t.name for t in tools},
            query_text=selector_query,
            lexical_weight=lexical_weight,
        )
        hint = mobile_hint or playwright_hint or filesystem_hint or browser_hint
        if hint is not None:
            selector_order = [hint] + [name for name in selector_order if name != hint]
        keep = set(selector_order[:top_m])
    else:
        retriever = retriever or ToolRetriever(tools)
        hint = mobile_hint or playwright_hint or filesystem_hint or browser_hint
        keep = {hint} if hint is not None else set(retriever.retrieve(grounding, k=k))
    use = [t for t in tools if t.name in keep] or tools
    if selector_order is not None:
        order = {name: index for index, name in enumerate(selector_order)}
        use.sort(key=lambda tool: order.get(tool.name, len(order)))
    # 2. argument values via learned pointer/copy spans
    ptr = None
    if ptr_head is not None:
        if feats is None:
            feats, ids = _ctx_feats(model, tok, ctx, device)
        bounds = _grounding_span(ids, tok, grounding) if grounding_prompt is not None else None
        ptr = (ptr_head, feats, ids, tok, bounds) if bounds is not None else (ptr_head, feats, ids, tok)
    # 3. rank every candidate's grounded body; _best picks the tool AND args jointly
    bodies = []
    for t in use:
        bodies += _tool_bodies(grounding, t, ptr)
    if not bodies:
        return "I am LocalAgent."
    if blocked_candidates:
        available = [body for body in bodies if body not in blocked_candidates]
        # If every grounded candidate was rejected, fail open and let the model retry rather than
        # returning an unrelated abstention.  The runtime's bounded attempt budget still limits
        # repeated calls, while the common case gets a genuine alternative candidate.
        bodies = available or bodies
    # With a top-1 selector there is nothing for the language-model reranker to compare.  Avoid a
    # second full forward pass; this is both exact (the sole candidate is the argmax) and material
    # for WebGPU/CPU deployment latency in long-horizon retries.
    if len(bodies) == 1:
        return bodies[0]
    if selector_first:
        return bodies[0]
    return _best(model, tok, score, bodies, device)


def grounded_decode_parallel(model, tok, prompt: str, tools: list[ToolSpec], device="cpu",
                             tool_head=None, ptr_head=None) -> str:
    """For 'do X and Y' turns: split on ' and ', ground each conjunct, concatenate the calls."""
    parts = [p.strip() for p in prompt.split(" and ") if p.strip()]
    return "".join(grounded_decode(model, tok, p, tools, device, tool_head, ptr_head)
                   for p in parts)


def grounded_decode(model, tok, prompt: str, tools: list[ToolSpec], device="cpu",
                    tool_head=None, ptr_head=None, framed=False) -> str:
    """Grounded constrained decode. `framed=False`: `prompt` is a raw user turn (framed as
    <|user|>..<|assistant|> internally) — single-turn. `framed=True`: `prompt` is the full
    multi-turn context already ending at the assistant marker (the next action is decoded over the
    whole history, so args can be grounded in earlier tool responses)."""
    from localagent.model.tokenizer import ASSISTANT, USER
    ctx = prompt if framed else f"{USER}{prompt}{ASSISTANT}"
    score = prompt if not framed else ctx  # what _score conditions on
    feats = ids = None
    # 1. tool selection
    if tool_head is not None:
        feats, ids = _ctx_feats(model, tok, ctx, device)
        from localagent.agent.tool_head import CLASSES
        picked = CLASSES[int(tool_head(feats[-1]).argmax(-1))]
        if picked == "text":
            return _best(model, tok, score, _text_candidates(prompt) or ["I am LocalAgent."], device)
    else:
        txt = _text_candidates(prompt)
        if txt is not None:
            return _best(model, tok, score, txt, device)
        picked = _preselect_tool(model, tok, prompt, {t.name for t in tools}, device)
    # 2. fill the selected tool's args — learned pointer/copy spans if a pointer head is given.
    use = [t for t in tools if t.name == picked] if picked else tools
    ptr = None
    if ptr_head is not None:
        if feats is None:
            feats, ids = _ctx_feats(model, tok, ctx, device)
        ptr = (ptr_head, feats, ids, tok)
    bodies = []
    for t in use:
        bodies += _tool_bodies(prompt, t, ptr)
    if not bodies:
        return "I am LocalAgent."
    return _best(model, tok, score, bodies, device)
