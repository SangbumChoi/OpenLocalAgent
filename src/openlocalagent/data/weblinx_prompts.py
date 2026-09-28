"""Private, prompt-only WebLINX decontamination export.

The adapter reads only caller-verified local JSON/JSON.GZ snapshots.  It treats HTML as inert
text, parses action history with a restricted AST (never ``eval``), excludes an entire
demonstration when bounded sensitive-pattern rules fire, and emits no current-step action or
score.  It is not an implementation of the official WebLINX evaluator.
"""

from __future__ import annotations

import ast
import gzip
import hashlib
import json
import os
import re
import sqlite3
import stat
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

WEBLINX_PROMPT_ADAPTER = "weblinx-private-prompt-rows-v1"
WEBLINX_PROMPT_ADAPTER_VERSION = WEBLINX_PROMPT_ADAPTER
WEBLINX_AUDIT_KIND = "localagent_weblinx_prompt_adapter_audit"
WEBLINX_AUDIT_SCHEMA_VERSION = 1
WEBLINX_PRIVACY_FILTER_VERSION = "localagent_weblinx_whole_demo_privacy_v1"
PRODUCTION_WEBLINX_REVISION = "be2e19d624febb57173e98772c1312d041a6d3b1"
PRODUCTION_WEBLINX_SPLIT = "test_web"
PRODUCTION_WEBLINX_CHAT_BYTES = 2_187_263
PRODUCTION_WEBLINX_CHAT_SHA256 = (
    "10d780712da997da9ff2d15d642aa199410ebe5d30d2ea3f9ba56fb044a745db"
)
PRODUCTION_WEBLINX_SPLITS_BYTES = 38_210
PRODUCTION_WEBLINX_SPLITS_SHA256 = (
    "db6fd50e6b1ba053817ede3f2a8ec61a292ad2710dd7f4e300cf685f70d843e6"
)
PRODUCTION_WEBLINX_SPLIT_DEMOS = 211

DEFAULT_MAX_CHAT_SOURCE_BYTES = 128 * 1024 * 1024
DEFAULT_MAX_SPLITS_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_DECOMPRESSED_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_RECORD_CHARS = 16 * 1024 * 1024
# Keep default output compatible with the generic prompt-suite freezer. Oversized observations
# fail closed; any later context ranker must be frozen independently of the evaluation labels.
DEFAULT_MAX_PROMPT_BYTES = 512 * 1024
DEFAULT_MAX_OUTPUT_BYTES = 128 * 1024 * 1024
DEFAULT_MAX_ROWS = 100_000

_READ_CHUNK_BYTES = 1024 * 1024
_JSON_CHUNK_BYTES = 64 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
_CHAT_KEYS = frozenset(
    {
        "demo",
        "turn",
        "action",
        "action_history",
        "utterances",
        "candidates",
        "clean_html",
        "viewport",
    }
)
_SPLIT_KEYS = frozenset(
    {
        "train",
        "valid",
        "test_iid",
        "test_web",
        "test_cat",
        "test_geo",
        "test_vis",
        "iid_all",
    }
)
_ACTION_ARGUMENTS: dict[str, tuple[tuple[str, type], ...]] = {
    "change": (("value", str), ("uid", str)),
    "click": (("uid", str),),
    "load": (("url", str),),
    "say": (("speaker", str), ("utterance", str)),
    "scroll": (("x", int), ("y", int)),
    "submit": (("uid", str),),
    "text_input": (("text", str), ("uid", str)),
}
_HISTORY_MARKUP = (
    "</s><s>[INST]",
    "</s>",
    "<s>",
    "[/INST]",
    "[INST]",
)
_EMAIL_RE = re.compile(
    r"(?i)(?<![A-Z0-9._%+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}"
    r"(?![A-Z0-9.-])"
)
_LABELED_SECRET_RE = re.compile(
    r"(?i)\b(?:password|passwd|pwd|api[ _-]?key|access[ _-]?token|auth[ _-]?token|"
    r"secret)\b\s*(?:is\s+|[:=]\s*)[\"']?[^\s<>{}\"']{4,}"
)
_SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_LABELED_PHONE_RE = re.compile(
    r"(?i)\b(?:phone|mobile|telephone|tel)\b\s*(?:is\s+|[:=]\s*)"
    r"\+?\d[\d ().-]{7,}\d"
)
_PAYMENT_NUMBER_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")


@dataclass(frozen=True)
class WebLINXSource:
    """One immutable local WebLINX artifact identity supplied by the caller."""

    path: Path
    bytes: int
    sha256: str


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_READ_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is forbidden")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _positive_int(value: int, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _validate_revision(revision: str) -> str:
    if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
        raise ValueError("revision must be a lowercase 40-character immutable commit SHA")
    return revision


def _validated_source_path(
    source: WebLINXSource,
    *,
    max_source_bytes: int,
    label: str,
) -> Path:
    path = Path(source.path)
    if not _SHA256_RE.fullmatch(source.sha256):
        raise ValueError(f"{label} has an invalid expected SHA-256")
    if (
        isinstance(source.bytes, bool)
        or not isinstance(source.bytes, int)
        or source.bytes < 0
    ):
        raise ValueError(f"{label} has an invalid expected byte size")
    if source.bytes > max_source_bytes:
        raise ValueError(f"{label} exceeds the {max_source_bytes}-byte cap: {path}")
    return path


def _snapshot_verified_source(
    source: WebLINXSource,
    snapshot_path: Path,
    *,
    max_source_bytes: int,
    label: str,
) -> tuple[Path, dict[str, int | str]]:
    """Copy and hash one source once, then return the private verified parse snapshot."""

    path = _validated_source_path(
        source,
        max_source_bytes=max_source_bytes,
        label=label,
    )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} must be a regular non-symlink file: {path}") from error

    observed_bytes = 0
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "rb") as source_handle, snapshot_path.open("xb") as snapshot:
            source_stat = os.fstat(source_handle.fileno())
            if not stat.S_ISREG(source_stat.st_mode):
                raise ValueError(f"{label} must be a regular non-symlink file: {path}")
            if source_stat.st_size > max_source_bytes:
                raise ValueError(
                    f"{label} exceeds the {max_source_bytes}-byte cap: {path}"
                )
            if source_stat.st_size != source.bytes:
                raise ValueError(
                    f"{label} byte-size mismatch: "
                    f"expected {source.bytes}, got {source_stat.st_size}"
                )
            for chunk in iter(lambda: source_handle.read(_READ_CHUNK_BYTES), b""):
                observed_bytes += len(chunk)
                if observed_bytes > max_source_bytes:
                    raise ValueError(
                        f"{label} exceeds the {max_source_bytes}-byte cap: {path}"
                    )
                snapshot.write(chunk)
                digest.update(chunk)
            snapshot.flush()
            os.fsync(snapshot.fileno())
    except Exception:
        snapshot_path.unlink(missing_ok=True)
        raise

    observed_sha256 = digest.hexdigest()
    if observed_bytes != source.bytes:
        snapshot_path.unlink(missing_ok=True)
        raise ValueError(
            f"{label} byte-size mismatch: expected {source.bytes}, got {observed_bytes}"
        )
    if observed_sha256 != source.sha256:
        snapshot_path.unlink(missing_ok=True)
        raise ValueError(f"{label} SHA-256 mismatch")
    snapshot_path.chmod(0o400)
    return (
        snapshot_path,
        {"bytes": observed_bytes, "name": path.name, "sha256": observed_sha256},
    )


class _Utf8ChunkReader:
    """Incrementally decode UTF-8 while enforcing a decompression ceiling."""

    def __init__(self, handle: BinaryIO, *, max_bytes: int) -> None:
        import codecs

        self._handle = handle
        self._decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self._max_bytes = max_bytes
        self._bytes = 0
        self._finished = False

    def read(self) -> str:
        if self._finished:
            return ""
        payload = self._handle.read(_JSON_CHUNK_BYTES)
        self._bytes += len(payload)
        if self._bytes > self._max_bytes:
            raise ValueError(
                f"decoded chat JSON exceeds max_decompressed_bytes={self._max_bytes}"
            )
        if payload:
            try:
                return self._decoder.decode(payload, final=False)
            except UnicodeDecodeError as error:
                raise ValueError("chat source is not valid UTF-8") from error
        self._finished = True
        try:
            return self._decoder.decode(b"", final=True)
        except UnicodeDecodeError as error:
            raise ValueError("chat source is not valid UTF-8") from error

    @property
    def finished(self) -> bool:
        return self._finished


def _open_chat(path: Path) -> tuple[BinaryIO, str]:
    with path.open("rb") as probe:
        magic = probe.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rb"), "gzip"
    return path.open("rb"), "plain"


def _iter_json_records(
    path: Path,
    *,
    max_decompressed_bytes: int,
    max_record_chars: int,
) -> tuple[Iterator[Any], str]:
    """Return a streaming iterator for a JSON array or whitespace-delimited JSON objects."""

    handle, compression = _open_chat(path)

    def generate() -> Iterator[Any]:
        decoder = json.JSONDecoder(
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_constant,
        )
        reader = _Utf8ChunkReader(handle, max_bytes=max_decompressed_bytes)
        buffer = ""
        position = 0
        mode: str | None = None
        expect_value = True
        after_comma = False

        def fill() -> bool:
            nonlocal buffer, position
            if position:
                buffer = buffer[position:]
                position = 0
            chunk = reader.read()
            buffer += chunk
            return bool(chunk)

        try:
            while True:
                while position >= len(buffer) and not reader.finished:
                    fill()
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if mode is None:
                    if position >= len(buffer):
                        raise ValueError(f"{path} is empty")
                    if buffer[position] == "[":
                        position += 1
                        mode = "array"
                        continue
                    if buffer[position] == "{":
                        mode = "sequence"
                    else:
                        raise ValueError(
                            f"{path} must contain a JSON array or JSON object stream"
                        )

                while position >= len(buffer) and not reader.finished:
                    fill()
                    while position < len(buffer) and buffer[position].isspace():
                        position += 1

                if mode == "array":
                    if position < len(buffer) and buffer[position] == "]":
                        if after_comma:
                            raise ValueError(f"{path} JSON array has a trailing comma")
                        position += 1
                        while not reader.finished:
                            fill()
                        if buffer[position:].strip():
                            raise ValueError(
                                f"{path} has trailing content after its JSON array"
                            )
                        return
                    if not expect_value:
                        if position >= len(buffer):
                            raise ValueError(f"{path} has an unterminated JSON array")
                        if buffer[position] != ",":
                            raise ValueError(
                                f"{path} JSON array entries must be comma-separated"
                            )
                        position += 1
                        expect_value = True
                        after_comma = True
                        continue
                elif position >= len(buffer) and reader.finished:
                    return

                record_start = position
                while True:
                    try:
                        value, end = decoder.raw_decode(buffer, position)
                    except json.JSONDecodeError as error:
                        if reader.finished:
                            raise ValueError(f"{path} contains invalid JSON") from error
                        if len(buffer) - record_start > max_record_chars:
                            raise ValueError(
                                f"{path} contains a record exceeding "
                                f"{max_record_chars} characters"
                            ) from error
                        fill()
                        record_start = 0
                        continue
                    if end - record_start > max_record_chars:
                        raise ValueError(
                            f"{path} contains a record exceeding "
                            f"{max_record_chars} characters"
                        )
                    position = end
                    if mode == "array":
                        expect_value = False
                        after_comma = False
                    yield value
                    break
        finally:
            handle.close()

    return generate(), compression


def _load_splits(path: Path, *, max_bytes: int) -> dict[str, tuple[str, ...]]:
    if path.stat().st_size > max_bytes:
        raise ValueError(f"splits JSON exceeds the {max_bytes}-byte cap")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("splits source is not valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise TypeError("splits source must be a JSON object")
    observed_keys = frozenset(payload)
    if not observed_keys or not observed_keys.issubset(_SPLIT_KEYS):
        raise ValueError(
            "splits schema drift: "
            f"unsupported keys={sorted(observed_keys - _SPLIT_KEYS)}"
        )
    splits: dict[str, tuple[str, ...]] = {}
    for name, raw_ids in payload.items():
        if not isinstance(raw_ids, list):
            raise TypeError(f"splits[{name!r}] must be an array")
        ids = tuple(
            _nonempty_string(value, label=f"splits[{name!r}] demo")
            for value in raw_ids
        )
        if len(ids) != len(set(ids)):
            raise ValueError(f"splits[{name!r}] contains duplicate demo IDs")
        splits[name] = ids
    return splits


def _assert_finite_json(value: Any, *, label: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain only finite JSON values") from error


def _ast_scalar(node: ast.AST, *, expected: type, label: str) -> str | int:
    if isinstance(node, ast.Constant):
        value = node.value
    elif (
        expected is int
        and isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
        and not isinstance(node.operand.value, bool)
    ):
        value = node.operand.value
        if isinstance(node.op, ast.USub):
            value = -value
    else:
        raise ValueError(f"{label} must be a literal {expected.__name__}")
    if expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{label} must be a literal int")
    elif not isinstance(value, expected):
        raise ValueError(f"{label} must be a literal {expected.__name__}")
    return value


def _parse_history_call(expression: str) -> str:
    try:
        parsed = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError) as error:
        raise ValueError("action history contains invalid call syntax") from error
    call = parsed.body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        raise TypeError("action history entries must be direct function calls")
    name = call.func.id
    schema = _ACTION_ARGUMENTS.get(name)
    if schema is None:
        raise ValueError(f"action history contains unsupported action {name!r}")
    if call.args:
        raise ValueError("action history calls must use keyword arguments")
    expected = {argument: value_type for argument, value_type in schema}
    observed: dict[str, str | int] = {}
    for keyword in call.keywords:
        if keyword.arg is None or keyword.arg not in expected:
            raise ValueError(f"action history {name!r} has an unsupported argument")
        if keyword.arg in observed:
            raise ValueError(
                f"action history {name!r} repeats argument {keyword.arg!r}"
            )
        observed[keyword.arg] = _ast_scalar(
            keyword.value,
            expected=expected[keyword.arg],
            label=f"{name}.{keyword.arg}",
        )
    if set(observed) != set(expected):
        raise ValueError(
            f"action history {name!r} arguments differ from "
            f"{sorted(expected)}"
        )
    arguments = ", ".join(
        f"{argument}={json.dumps(observed[argument], ensure_ascii=False)}"
        for argument, _ in schema
    )
    return f"{name}({arguments})"


def parse_weblinx_action_history(
    action_history: str | None,
    *,
    max_calls: int = 64,
) -> tuple[str, ...]:
    """Parse and canonicalize WebLINX history calls without executing source text."""

    _positive_int(max_calls, label="max_calls")
    if action_history is None or not action_history.strip():
        return ()
    if action_history.strip().casefold() in {"<none>", "none", "null"}:
        return ()

    text = action_history
    calls: list[str] = []
    position = 0
    while position < len(text):
        progressed = True
        while progressed:
            progressed = False
            while position < len(text) and (
                text[position].isspace() or text[position] == ";"
            ):
                position += 1
                progressed = True
            for marker in _HISTORY_MARKUP:
                if text.startswith(marker, position):
                    position += len(marker)
                    progressed = True
                    break
        if position >= len(text):
            break
        name_match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", text[position:])
        if name_match is None:
            raise ValueError(
                f"action history has unsupported content at character {position}"
            )
        cursor = position + len(name_match.group(0))
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text) or text[cursor] != "(":
            raise ValueError("action history entries must be function calls")
        depth = 0
        quote: str | None = None
        escaped = False
        end = cursor
        while end < len(text):
            character = text[end]
            if quote is not None:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == quote:
                    quote = None
            elif character in {"'", '"'}:
                quote = character
            elif character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    end += 1
                    break
                if depth < 0:
                    raise ValueError("action history has unbalanced parentheses")
            end += 1
        if depth != 0 or quote is not None:
            raise ValueError("action history has an unterminated call")
        calls.append(_parse_history_call(text[position:end]))
        if len(calls) > max_calls:
            raise ValueError(f"action history exceeds max_calls={max_calls}")
        position = end
    return tuple(calls)


def _context_string(value: Any, *, label: str) -> str:
    if value is None:
        return "<none>"
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string or null")
    return value if value else "<none>"


def render_weblinx_step_prompt(
    *,
    clean_html: str | None,
    utterances: str | None,
    viewport: str | None,
    candidates: str | None,
    action_history: str | None,
    max_prompt_bytes: int = DEFAULT_MAX_PROMPT_BYTES,
) -> str:
    """Render a versioned prompt from the official model-visible context fields."""

    _positive_int(max_prompt_bytes, label="max_prompt_bytes")
    html = _context_string(clean_html, label="clean_html")
    dialogue = _context_string(utterances, label="utterances")
    viewport_text = _context_string(viewport, label="viewport")
    candidate_text = _context_string(candidates, label="candidates")
    # The pinned compact release contains natural-language strings whose embedded quotes are not
    # valid Python call syntax. History is model-visible context, not executable code: preserve the
    # immutable source text verbatim and never send it through eval or a target-aware normalizer.
    history = _context_string(action_history, label="action_history")
    prompt = (
        f"[{WEBLINX_PROMPT_ADAPTER_VERSION}]\n"
        f"{html}\n"
        "The HTML elements above are available for the current webpage.\n"
        "Predict the next action for the user's request using one of:\n"
        "change(value=[str], uid=[str]); click(uid=[str]); load(url=[str]); "
        'say(speaker="navigator", utterance=[str]); scroll(x=[int], y=[int]); '
        "submit(uid=[str]); text_input(text=[str], uid=[str]).\n"
        f"<utterances>\n{dialogue}\n</utterances>\n"
        f"<viewport>\n{viewport_text}\n</viewport>\n"
        f"<candidates>\n{candidate_text}\n</candidates>\n"
        f"<action_history>\n{history}\n</action_history>\n"
        "Return only the next action in the specified format."
    )
    prompt_bytes = len(prompt.encode("utf-8"))
    if prompt_bytes > max_prompt_bytes:
        raise ValueError(
            f"rendered WebLINX prompt is {prompt_bytes} bytes, exceeding "
            f"max_prompt_bytes={max_prompt_bytes}"
        )
    return prompt


def _walk_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _walk_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_strings(child)


def _luhn_valid(number: str) -> bool:
    digits = [int(character) for character in number if character.isdigit()]
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    total = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def detect_weblinx_privacy_reasons(value: Any) -> frozenset[str]:
    """Return versioned reason codes for the filter's documented sensitive-pattern subset."""

    reasons: set[str] = set()
    for text in _walk_strings(value):
        if _EMAIL_RE.search(text):
            reasons.add("email")
        if _LABELED_SECRET_RE.search(text):
            reasons.add("labeled_secret")
        if _SSN_RE.search(text):
            reasons.add("ssn")
        if _LABELED_PHONE_RE.search(text):
            reasons.add("labeled_phone")
        if any(_luhn_valid(match.group(0)) for match in _PAYMENT_NUMBER_RE.finditer(text)):
            reasons.add("payment_card")
    return frozenset(reasons)


def _case_id(revision: str, split: str, demo: str, turn: int) -> str:
    identity = {
        "demo": demo,
        "revision": revision,
        "split": split,
        "turn": turn,
    }
    return f"weblinx-step-v1:{_sha256(_canonical_json_bytes(identity))}"


def _private_demo_hash(revision: str, demo: str) -> str:
    return _sha256(
        _canonical_json_bytes(
            {
                "benchmark": "WebLINX",
                "demo": demo,
                "revision": revision,
            }
        )
    )


def _canonical_split_demo_id(demo: str, split_ids: frozenset[str]) -> str:
    """Map compact-chat prefixed IDs to the canonical bare IDs in ``splits.json``."""

    if demo in split_ids:
        return demo
    suffix_matches = [
        split_id for split_id in split_ids if demo.endswith(f"_{split_id}")
    ]
    if len(suffix_matches) != 1:
        raise ValueError(
            "chat demo does not map uniquely to one canonical declared-split ID"
        )
    return suffix_matches[0]


def _stage_chat_row(
    connection: sqlite3.Connection,
    raw_row: Any,
    *,
    row_index: int,
    split_ids: frozenset[str],
) -> tuple[str, int, frozenset[str]]:
    label = f"chat row {row_index}"
    if not isinstance(raw_row, dict):
        raise TypeError(f"{label} must be an object")
    observed = frozenset(raw_row)
    if observed != _CHAT_KEYS:
        raise ValueError(
            f"{label} schema drift: missing={sorted(_CHAT_KEYS - observed)}, "
            f"extra={sorted(observed - _CHAT_KEYS)}"
        )
    _assert_finite_json(raw_row, label=label)
    raw_demo = _nonempty_string(raw_row["demo"], label=f"{label}.demo")
    try:
        demo = _canonical_split_demo_id(raw_demo, split_ids)
    except ValueError as error:
        raise ValueError(f"{label} demo is not a member of the declared split") from error
    turn = raw_row["turn"]
    if isinstance(turn, bool) or not isinstance(turn, int) or turn < 0:
        raise ValueError(f"{label}.turn must be a nonnegative integer")
    _nonempty_string(raw_row["action"], label=f"{label}.action")
    contexts = {
        field: _context_string(raw_row[field], label=f"{label}.{field}")
        for field in (
            "action_history",
            "utterances",
            "candidates",
            "clean_html",
            "viewport",
        )
    }
    try:
        connection.execute(
            """
            INSERT INTO chat_rows(
                demo, turn, action_history, utterances, candidates, clean_html, viewport
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                demo,
                turn,
                contexts["action_history"],
                contexts["utterances"],
                contexts["candidates"],
                contexts["clean_html"],
                contexts["viewport"],
            ),
        )
    except sqlite3.IntegrityError as error:
        raise ValueError(f"duplicate WebLINX (demo, turn) at {label}") from error
    return demo, turn, detect_weblinx_privacy_reasons(raw_row)


def _write_staged_rows(
    connection: sqlite3.Connection,
    staged_path: Path,
    *,
    revision: str,
    split: str,
    excluded_demos: frozenset[str],
    max_prompt_bytes: int,
    max_output_bytes: int,
) -> tuple[int, int, str, str]:
    output_digest = hashlib.sha256()
    case_digest = hashlib.sha256()
    output_bytes = 0
    rows = 0
    first_case = True
    with staged_path.open("wb") as handle:
        for (
            demo,
            turn,
            action_history,
            utterances,
            candidates,
            clean_html,
            viewport,
        ) in connection.execute(
            """
            SELECT demo, turn, action_history, utterances, candidates, clean_html, viewport
            FROM chat_rows
            ORDER BY demo, turn
            """
        ):
            if demo in excluded_demos:
                continue
            try:
                prompt = render_weblinx_step_prompt(
                    clean_html=clean_html,
                    utterances=utterances,
                    viewport=viewport,
                    candidates=candidates,
                    action_history=action_history,
                    max_prompt_bytes=max_prompt_bytes,
                )
            except (TypeError, ValueError) as error:
                private_id = _private_demo_hash(revision, demo)
                raise ValueError(
                    f"WebLINX history/context validation failed for private demo "
                    f"{private_id}, turn {turn}: {error}"
                ) from error
            source_case_id = _case_id(revision, split, demo, turn)
            row = {"prompt": prompt, "source_case_id": source_case_id}
            if set(row) != {"source_case_id", "prompt"}:
                raise AssertionError("prompt-only row gained an unexpected field")
            payload = _canonical_json_bytes(row) + b"\n"
            output_bytes += len(payload)
            if output_bytes > max_output_bytes:
                raise ValueError(
                    f"prompt output exceeds max_output_bytes={max_output_bytes}"
                )
            handle.write(payload)
            output_digest.update(payload)
            if not first_case:
                case_digest.update(b"\n")
            case_digest.update(source_case_id.encode("ascii"))
            first_case = False
            rows += 1
        handle.flush()
        os.fsync(handle.fileno())
    return rows, output_bytes, output_digest.hexdigest(), case_digest.hexdigest()


def _existing_matches(path: Path, *, expected_bytes: int, expected_sha256: str) -> bool:
    return (
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == expected_bytes
        and _hash_file(path) == expected_sha256
    )


def _preflight_output(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    label: str,
) -> None:
    if path.exists() and not _existing_matches(
        path,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
    ):
        raise RuntimeError(f"refusing to overwrite drifted {label}: {path}")


def _publish_staged(
    staged_path: Path,
    destination: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    label: str,
) -> None:
    if not destination.exists():
        try:
            os.link(staged_path, destination)
        except FileExistsError:
            pass
    if not _existing_matches(
        destination,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
    ):
        raise RuntimeError(f"concurrently published {label} does not match: {destination}")


def export_weblinx_prompt_rows(
    chat_source: WebLINXSource,
    splits_source: WebLINXSource,
    output_path: str | Path,
    *,
    revision: str,
    split: str,
    audit_path: str | Path | None = None,
    max_chat_source_bytes: int = DEFAULT_MAX_CHAT_SOURCE_BYTES,
    max_splits_bytes: int = DEFAULT_MAX_SPLITS_BYTES,
    max_decompressed_bytes: int = DEFAULT_MAX_DECOMPRESSED_BYTES,
    max_record_chars: int = DEFAULT_MAX_RECORD_CHARS,
    max_prompt_bytes: int = DEFAULT_MAX_PROMPT_BYTES,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> dict[str, Any]:
    """Validate one compact split and atomically export canonical prompt-only rows."""

    revision = _validate_revision(revision)
    split = _nonempty_string(split, label="split")
    mode = (
        "production"
        if revision == PRODUCTION_WEBLINX_REVISION
        and split == PRODUCTION_WEBLINX_SPLIT
        else "fixture"
    )
    for value, label in (
        (max_chat_source_bytes, "max_chat_source_bytes"),
        (max_splits_bytes, "max_splits_bytes"),
        (max_decompressed_bytes, "max_decompressed_bytes"),
        (max_record_chars, "max_record_chars"),
        (max_prompt_bytes, "max_prompt_bytes"),
        (max_output_bytes, "max_output_bytes"),
        (max_rows, "max_rows"),
    ):
        _positive_int(value, label=label)
    if mode == "production":
        expected_chat_identity = (
            PRODUCTION_WEBLINX_CHAT_BYTES,
            PRODUCTION_WEBLINX_CHAT_SHA256,
        )
        if (chat_source.bytes, chat_source.sha256) != expected_chat_identity:
            raise ValueError(
                "production WebLINX chat identity mismatch: "
                f"expected bytes={expected_chat_identity[0]}, "
                f"sha256={expected_chat_identity[1]}"
            )
        expected_splits_identity = (
            PRODUCTION_WEBLINX_SPLITS_BYTES,
            PRODUCTION_WEBLINX_SPLITS_SHA256,
        )
        if (splits_source.bytes, splits_source.sha256) != expected_splits_identity:
            raise ValueError(
                "production WebLINX splits identity mismatch: "
                f"expected bytes={expected_splits_identity[0]}, "
                f"sha256={expected_splits_identity[1]}"
            )

    chat_path = _validated_source_path(
        chat_source,
        max_source_bytes=max_chat_source_bytes,
        label="chat source",
    )
    splits_path = _validated_source_path(
        splits_source,
        max_source_bytes=max_splits_bytes,
        label="splits source",
    )
    if chat_path.resolve() == splits_path.resolve():
        raise ValueError("chat and splits sources must be different files")

    output = Path(output_path)
    audit_output = Path(audit_path) if audit_path is not None else None
    output.parent.mkdir(parents=True, exist_ok=True)
    if audit_output is not None:
        audit_output.parent.mkdir(parents=True, exist_ok=True)
        if output.resolve() == audit_output.resolve():
            raise ValueError("output_path and audit_path must be different files")
    for source_path in (chat_path, splits_path):
        if source_path.resolve() == output.resolve() or (
            audit_output is not None and source_path.resolve() == audit_output.resolve()
        ):
            raise ValueError("source and output paths must be distinct")

    staged_output_name: str | None = None
    staged_audit_name: str | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="localagent-weblinx-prompts-") as temporary:
            temporary_path = Path(temporary)
            snapshot_directory = temporary_path / "sources"
            snapshot_directory.mkdir(mode=0o700)
            chat_snapshot, chat_identity = _snapshot_verified_source(
                chat_source,
                snapshot_directory / f"chat-{chat_path.name}",
                max_source_bytes=max_chat_source_bytes,
                label="chat source",
            )
            splits_snapshot, splits_identity = _snapshot_verified_source(
                splits_source,
                snapshot_directory / f"splits-{splits_path.name}",
                max_source_bytes=max_splits_bytes,
                label="splits source",
            )
            splits = _load_splits(splits_snapshot, max_bytes=max_splits_bytes)
            if split not in splits:
                raise ValueError(f"declared split {split!r} is absent from splits source")
            split_ids = frozenset(splits[split])
            if not split_ids:
                raise ValueError(f"declared split {split!r} is empty")
            if mode == "production" and len(split_ids) != PRODUCTION_WEBLINX_SPLIT_DEMOS:
                raise ValueError(
                    "production WebLINX split demo-count mismatch: "
                    f"expected {PRODUCTION_WEBLINX_SPLIT_DEMOS}, observed {len(split_ids)}"
                )

            database_path = temporary_path / "rows.sqlite3"
            connection = sqlite3.connect(database_path)
            try:
                connection.execute("PRAGMA journal_mode=OFF")
                connection.execute("PRAGMA synchronous=OFF")
                connection.execute(
                    """
                    CREATE TABLE chat_rows (
                        demo TEXT NOT NULL,
                        turn INTEGER NOT NULL,
                        action_history TEXT NOT NULL,
                        utterances TEXT NOT NULL,
                        candidates TEXT NOT NULL,
                        clean_html TEXT NOT NULL,
                        viewport TEXT NOT NULL,
                        PRIMARY KEY (demo, turn)
                    ) WITHOUT ROWID
                    """
                )
                records, compression = _iter_json_records(
                    chat_snapshot,
                    max_decompressed_bytes=max_decompressed_bytes,
                    max_record_chars=max_record_chars,
                )
                observed_demos: set[str] = set()
                demo_reasons: dict[str, set[str]] = {}
                source_rows = 0
                for row_index, raw_row in enumerate(records):
                    demo, _, reasons = _stage_chat_row(
                        connection,
                        raw_row,
                        row_index=row_index,
                        split_ids=split_ids,
                    )
                    observed_demos.add(demo)
                    if reasons:
                        demo_reasons.setdefault(demo, set()).update(reasons)
                    source_rows += 1
                    if source_rows > max_rows:
                        raise ValueError(f"WebLINX source exceeds max_rows={max_rows}")
                if source_rows == 0:
                    raise ValueError("WebLINX chat source contains no rows")
                if observed_demos != split_ids:
                    missing_hashes = sorted(
                        _private_demo_hash(revision, demo)
                        for demo in split_ids - observed_demos
                    )
                    raise ValueError(
                        "chat source does not exactly cover the declared split; "
                        f"missing demo hashes={missing_hashes}"
                    )
                connection.commit()

                excluded_demos = frozenset(demo_reasons)
                excluded_rows = connection.execute(
                    "SELECT COUNT(*) FROM chat_rows WHERE demo IN "
                    f"({','.join('?' for _ in excluded_demos)})",
                    tuple(sorted(excluded_demos)),
                ).fetchone()[0] if excluded_demos else 0

                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=output.parent,
                    prefix=f".{output.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as staged_handle:
                    staged_output_name = staged_handle.name
                (
                    written_rows,
                    output_bytes,
                    output_sha256,
                    case_ids_sha256,
                ) = _write_staged_rows(
                    connection,
                    Path(staged_output_name),
                    revision=revision,
                    split=split,
                    excluded_demos=excluded_demos,
                    max_prompt_bytes=max_prompt_bytes,
                    max_output_bytes=max_output_bytes,
                )
                if written_rows != source_rows - excluded_rows:
                    raise AssertionError("staged WebLINX row count changed")
            finally:
                connection.close()

        reason_counts: dict[str, int] = {}
        for reasons in demo_reasons.values():
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        excluded_demo_hashes = sorted(
            _private_demo_hash(revision, demo) for demo in excluded_demos
        )
        audit_without_hash: dict[str, Any] = {
            "adapter": WEBLINX_PROMPT_ADAPTER,
            "adapter_version": WEBLINX_PROMPT_ADAPTER_VERSION,
            "benchmark": "weblinx-chat-v1.0",
            "freeze_binding": {
                "adapter": WEBLINX_PROMPT_ADAPTER,
                "benchmark": "weblinx-chat-v1.0",
                "mode": mode,
                "revision": revision,
                "split": split,
                "prompt_only": True,
                "contains_current_step_labels": False,
                "output": {
                    "bytes": output_bytes,
                    "sha256": output_sha256,
                    "records": written_rows,
                },
            },
            "kind": WEBLINX_AUDIT_KIND,
            "label_isolation": {
                "current_action_emitted": False,
                "expected_calls_emitted": False,
                "labels_emitted": False,
                "scores_emitted": False,
            },
            "limits": {
                "max_chat_source_bytes": max_chat_source_bytes,
                "max_decompressed_bytes": max_decompressed_bytes,
                "max_output_bytes": max_output_bytes,
                "max_prompt_bytes": max_prompt_bytes,
                "max_record_chars": max_record_chars,
                "max_rows": max_rows,
                "max_splits_bytes": max_splits_bytes,
            },
            "ordering": "demo_id_then_integer_turn",
            "mode": mode,
            "output": {
                "bytes": output_bytes,
                "path": output.name,
                "rows": written_rows,
                "sha256": output_sha256,
                "source_case_ids_sha256": case_ids_sha256,
            },
            "privacy": {
                "accepted_demos": len(observed_demos - excluded_demos),
                "contains_private_heldout_prompts": True,
                "excluded_demo_id_sha256": excluded_demo_hashes,
                "excluded_demo_ids_sha256": _sha256(
                    _canonical_json_bytes(excluded_demo_hashes)
                ),
                "excluded_demos": len(excluded_demos),
                "excluded_rows": excluded_rows,
                "filter_version": WEBLINX_PRIVACY_FILTER_VERSION,
                "reason_counts": dict(sorted(reason_counts.items())),
                "redistribution_authorized": False,
                "scanned_demos": len(observed_demos),
            },
            "purpose": "prompt_only_pretraining_decontamination_not_official_scoring",
            "revision": revision,
            "schema_version": WEBLINX_AUDIT_SCHEMA_VERSION,
            "source_rows": source_rows,
            "sources": {
                "chat": {**chat_identity, "compression": compression},
                "splits": splits_identity,
            },
            "split": split,
            "split_demos": len(split_ids),
        }
        audit = {
            **audit_without_hash,
            "audit_self_sha256": _sha256(_canonical_json_bytes(audit_without_hash)),
        }
        audit_payload = _canonical_json_bytes(audit) + b"\n"
        audit_sha256 = _sha256(audit_payload)

        _preflight_output(
            output,
            expected_bytes=output_bytes,
            expected_sha256=output_sha256,
            label="WebLINX prompt export",
        )
        if audit_output is not None:
            _preflight_output(
                audit_output,
                expected_bytes=len(audit_payload),
                expected_sha256=audit_sha256,
                label="WebLINX adapter audit",
            )
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=audit_output.parent,
                prefix=f".{audit_output.name}.",
                suffix=".tmp",
                delete=False,
            ) as audit_handle:
                staged_audit_name = audit_handle.name
                audit_handle.write(audit_payload)
                audit_handle.flush()
                os.fsync(audit_handle.fileno())

        _publish_staged(
            Path(staged_output_name),
            output,
            expected_bytes=output_bytes,
            expected_sha256=output_sha256,
            label="WebLINX prompt export",
        )
        if audit_output is not None and staged_audit_name is not None:
            _publish_staged(
                Path(staged_audit_name),
                audit_output,
                expected_bytes=len(audit_payload),
                expected_sha256=audit_sha256,
                label="WebLINX adapter audit",
            )
        return audit
    finally:
        if staged_output_name is not None:
            Path(staged_output_name).unlink(missing_ok=True)
        if staged_audit_name is not None:
            Path(staged_audit_name).unlink(missing_ok=True)
