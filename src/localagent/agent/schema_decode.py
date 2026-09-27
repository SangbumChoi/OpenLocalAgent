"""Schema-guided constrained tool-call decoding (the reliable core).

Given a user turn and a tool's JSON schema, we **construct** a call from the schema rather than
free-generate JSON — so the output is *always* schema-valid. Each property is filled by a typed,
arg-aware slot-filler that grounds values in the user's text:

  - ``enum``                 -> the enum member mentioned (else unfilled)
  - ``integer``/``number``   -> a number from the text
  - ``boolean``              -> inferred from yes/no/enable/disable cues
  - string ``format: path``  -> a file path; ``url`` -> a URL; ``arithmetic`` -> an expression
  - string ``format: quoted``-> a quoted span
  - string (entity-ish name) -> a capitalised proper-noun span
  - string (free text)       -> a quoted span, else the descriptive tail

**Multi-argument** tools work because same-typed values are pulled from a shared pool *in schema
order* (``move_file(source, dest)`` over two paths → first path = source, second = dest).

If a required argument can't be grounded, the tool doesn't fill (the caller can abstain or try the
next candidate). This makes a tiny model *reliable* on arbitrary schemas: no invalid JSON, ever.
"""

from __future__ import annotations

import re

# argument-name hints (used alongside type/format) — generic, not per-tool
PATH_HINTS = {"path", "file", "filepath", "filename", "source", "src", "dest", "destination", "target"}
URL_HINTS = {"url", "link", "website", "site", "href", "address", "endpoint"}
ENTITY_HINTS = {"name", "recipient", "person", "city", "location", "user", "author", "artist",
                "assignee", "owner", "contact", "to", "sender"}
QUOTED_HINTS = {"message", "subject", "title", "content", "body", "text", "note", "summary",
                "caption", "comment", "description", "label"}
_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_TRUE = {"on", "enable", "yes", "true", "turn on", "activate"}
_FALSE = {"off", "disable", "no", "false", "turn off", "deactivate"}
PREPS = ["for", "about", "to", "in", "on", "of", "with", "from"]


def _strip(s: str) -> str:
    return re.sub(r"^[^A-Za-z0-9'\"]+|\s*(online|please)?\s*[.?!]*$", "", s, flags=re.I).strip()


def extract_pools(prompt: str) -> dict:
    """Typed candidate value pools mined once from the prompt (lists, consumed in schema order)."""
    body = " ".join(prompt.split()[1:])  # drop the leading verb for proper-noun detection
    return {
        "quoted": [m.group(1) or m.group(2)
                   for m in re.finditer(r"'([^']+)'|\"([^\"]+)\"", prompt)],
        "path": [m.group(0).rstrip(".") for m in re.finditer(
            r"[A-Za-z0-9_.\-/]+/[A-Za-z0-9_.\-/]*|[A-Za-z0-9_.\-/]+\.[A-Za-z0-9]{1,5}\b", prompt)],
        "url": [m.group(0).rstrip(".") for m in re.finditer(
            r"(?:https?://)?[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?:/[\w./-]*)?", prompt)],
        "caps": [m.group(0) for m in re.finditer(r"(?:[A-Z][a-z]+)(?:\s+[A-Z][a-z]+)*", body)],
        "number": _NUM.findall(prompt),
        "arith": ([re.sub(r"\s+", "", m.group(0))]
                  if (m := re.search(r"\d+\s*[-+*/]\s*\d+(?:\s*[-+*/]\s*\d+)*", prompt)) else []),
    }


def _pop(pool: list):
    return pool.pop(0) if pool else None


def _free_text(prompt: str) -> str | None:
    low = prompt.lower()
    tails = [_strip(prompt[i + len(p) + 2:]) for p in PREPS if (i := low.find(f" {p} ")) >= 0]
    tails = [t for t in tails if t]
    if tails:
        return max(tails, key=len)
    words = prompt.split()
    return _strip(" ".join(words[1:])) if len(words) > 1 else None


def _fill_one(prompt: str, name: str, schema: dict, pools: dict, required: bool):
    fmt, typ = schema.get("format"), schema.get("type", "string")
    tail = (lambda: _free_text(prompt)) if required else (lambda: None)  # greedy only if required
    if "enum" in schema:
        for e in schema["enum"]:
            if re.search(rf"\b{re.escape(str(e))}\b", prompt, re.I):
                return e
        return None
    if typ == "boolean":
        low = prompt.lower()
        if any(t in low for t in _TRUE):
            return True
        if any(t in low for t in _FALSE):
            return False
        return None
    if fmt == "arithmetic" or "express" in name:
        return _pop(pools["arith"])
    if typ in ("integer", "number"):
        v = _pop(pools["number"])
        if v is None:
            return None
        return int(float(v)) if typ == "integer" else float(v)
    # explicit `format` wins over name-based hints (e.g. `target` is a PATH_HINT but a
    # computer-use click target declares `format: quoted` and must ground as a quoted span).
    if fmt == "quoted":
        return _pop(pools["quoted"]) or tail()
    if fmt == "path":
        return _pop(pools["path"])
    if fmt == "url":
        return _pop(pools["url"])
    if name in PATH_HINTS:
        return _pop(pools["path"])
    if name in URL_HINTS:
        return _pop(pools["url"])
    if name in QUOTED_HINTS:
        return _pop(pools["quoted"]) or tail()
    if name in ENTITY_HINTS:
        return _pop(pools["caps"]) or tail()
    return _pop(pools["quoted"]) or tail()                # generic free-text string


def fill_tool(prompt: str, tool) -> dict | None:
    """Ground a schema-valid argument dict for `tool`, or None if a required arg can't be filled."""
    params = tool.parameters or {}
    props = params.get("properties", {})
    required = set(params.get("required", []))
    pools = extract_pools(prompt)
    args = {}
    for name, schema in props.items():
        val = _fill_one(prompt, name, schema, pools, name in required)
        if val is not None and val != "":
            args[name] = val
        elif name in required:
            return None
    return args if validate(args, params) else None


def validate(args: dict, params: dict) -> bool:
    """Check args against the JSON schema (required present, types match, enums respected)."""
    props = params.get("properties", {})
    for r in params.get("required", []):
        if r not in args:
            return False
    for k, v in args.items():
        sch = props.get(k)
        if not sch:
            return False
        t = sch.get("type", "string")
        if "enum" in sch and v not in sch["enum"]:
            return False
        if t == "integer" and (isinstance(v, bool) or not isinstance(v, int)):
            return False
        if t == "number" and (isinstance(v, bool) or not isinstance(v, (int, float))):
            return False
        if t == "boolean" and not isinstance(v, bool):
            return False
        if t == "string" and not isinstance(v, str):
            return False
    return True
