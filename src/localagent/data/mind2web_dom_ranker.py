"""Deterministic, gold-independent DOM ranking for private Mind2Web prompts.

The ranker is intentionally tokenizer-free and accepts only the natural-language task plus the
current cleaned HTML.  Benchmark actions, candidate pools, and labels are outside this module's
type boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import stat
import sys
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from types import MappingProxyType
from typing import Any

MIND2WEB_DOM_RANKER_VERSION = "mind2web-dom-lexical-v1"
MIND2WEB_RANKED_PROMPT_ADAPTER_VERSION = "mind2web-private-prompt-rows-v2"
MIND2WEB_DOM_RANKER_CONFIG_KIND = "localagent_mind2web_dom_ranker_config"
MIND2WEB_DOM_RANKER_CONFIG_SCHEMA_VERSION = 1
MIND2WEB_DOM_RANKING_AUDIT_KIND = "localagent_mind2web_dom_ranking_audit"
MIND2WEB_DOM_RANKING_AUDIT_SCHEMA_VERSION = 1

_MAX_CONFIG_BYTES = 64 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TOKEN_RE = re.compile(r"[\w]+")
_CONFIG_KEYS = frozenset(
    {
        "adapter_version",
        "budget",
        "config_self_sha256",
        "features",
        "implementation",
        "input_projection",
        "kind",
        "normalization",
        "parser",
        "ranker_version",
        "ranking",
        "runtime",
        "schema_version",
    }
)
_BUDGET_KEYS = frozenset(
    {
        "assistant_marker",
        "generation_reserve_tokens_including_eos",
        "max_framed_prompt_bytes",
        "max_unframed_prompt_bytes",
        "minimum_dom_bytes",
        "model_max_seq_len",
        "user_marker",
    }
)
_FEATURE_KEYS = frozenset(
    {
        "ancestor_depth",
        "attribute_priority",
        "interactive_roles",
        "interactive_tags",
        "max_ancestor_text_bytes",
        "max_attribute_count",
        "max_attribute_value_bytes",
        "max_node_line_bytes",
        "max_stable_identifier_bytes",
        "max_text_bytes",
        "stable_identifier_attributes",
    }
)
_IMPLEMENTATION_KEYS = frozenset({"module", "path", "sha256"})
_INPUT_PROJECTION_KEYS = frozenset({"allowed", "forbidden"})
_NORMALIZATION_KEYS = frozenset(
    {"casefold", "form", "token_pattern", "unicode_version", "whitespace"}
)
_PARSER_KEYS = frozenset(
    {
        "convert_charrefs",
        "excluded_content_tags",
        "implementation",
        "max_depth",
        "max_html_bytes",
        "max_nodes",
        "void_tags",
    }
)
_RANKING_KEYS = frozenset({"score", "selection", "serialization", "tie_break"})
_RUNTIME_KEYS = frozenset({"python_implementation", "python_minor"})

_EXPECTED_ALLOWED_FIELDS = ("confirmed_task", "cleaned_html")
_EXPECTED_FORBIDDEN_FIELDS = (
    "action_reprs",
    "action_uid",
    "neg_candidates",
    "operation",
    "pos_candidates",
    "raw_html",
)
_EXPECTED_SCORE = (
    "exact_normalized_task_phrase",
    "own_distinct_task_token_overlap",
    "ancestor_distinct_task_token_overlap",
    "interactive_tag_or_role",
)
_EXPECTED_SELECTION = "ranked_greedy_skip_nonfitting_then_document_order"
_EXPECTED_SERIALIZATION = "canonical_compact_json_node_lines_v1"
_EXPECTED_TIE_BREAK = "parser_preorder_ascending"
_EXPECTED_IMPLEMENTATION_MODULE = "localagent.data.mind2web_dom_ranker"
_EXPECTED_IMPLEMENTATION_PATH = "src/localagent/data/mind2web_dom_ranker.py"
_EXPECTED_PARSER_IMPLEMENTATION = "stdlib.html.parser.HTMLParser"
_EXPECTED_WHITESPACE = "unicode_split_join"
_EXPECTED_USER_MARKER = "<|user|>"
_EXPECTED_ASSISTANT_MARKER = "<|assistant|>"
_EXPECTED_EXCLUDED_CONTENT_TAGS = ("noscript", "script", "style")
_EXPECTED_VOID_TAGS = (
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
)
_EXPECTED_ATTRIBUTE_PRIORITY = (
    "backend_node_id",
    "id",
    "aria-label",
    "name",
    "role",
    "type",
    "placeholder",
    "title",
    "alt",
    "value",
    "href",
    "class",
)
_EXPECTED_STABLE_IDENTIFIER_ATTRIBUTES = ("backend_node_id", "id")
_EXPECTED_INTERACTIVE_TAGS = ("a", "button", "input", "option", "select", "textarea")
_EXPECTED_INTERACTIVE_ROLES = (
    "button",
    "checkbox",
    "combobox",
    "link",
    "menuitem",
    "option",
    "radio",
    "searchbox",
    "slider",
    "spinbutton",
    "switch",
    "tab",
    "textbox",
    "treeitem",
)
_FORBIDDEN_PROMPT_MARKERS = (
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
    "<|end|>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<task>",
    "</task>",
    "<ranked_dom>",
    "</ranked_dom>",
)


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


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is forbidden")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], *, label: str) -> None:
    observed = frozenset(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{label} schema drift: missing={missing}, extra={extra}")


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _string_tuple(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty array")
    values = tuple(_string(item, label=f"{label}[{index}]") for index, item in enumerate(value))
    if len(values) != len(set(values)):
        raise ValueError(f"{label} entries must be unique")
    return values


def _read_verified_config(path: Path) -> bytes:
    if path.is_symlink():
        raise ValueError(f"ranker config must be a regular non-symlink file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"ranker config must be a regular non-symlink file: {path}") from error
    with os.fdopen(descriptor, "rb") as handle:
        source_stat = os.fstat(handle.fileno())
        if not stat.S_ISREG(source_stat.st_mode) or source_stat.st_size > _MAX_CONFIG_BYTES:
            raise ValueError(
                f"ranker config must be a regular file no larger than {_MAX_CONFIG_BYTES} bytes"
            )
        payload = handle.read(_MAX_CONFIG_BYTES + 1)
    if not payload or len(payload) > _MAX_CONFIG_BYTES:
        raise ValueError(
            f"ranker config must be non-empty and no larger than {_MAX_CONFIG_BYTES} bytes"
        )
    return payload


def runtime_identity() -> dict[str, str]:
    """Return the runtime fields that must match when ranked contexts are replayed."""

    return {
        "html_parser": _EXPECTED_PARSER_IMPLEMENTATION,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "unicode_version": unicodedata.unidata_version,
    }


def implementation_identity() -> dict[str, int | str]:
    """Return the exact implementation-file identity used for this ranker invocation."""

    path = Path(__file__)
    payload = path.read_bytes()
    return {
        "bytes": len(payload),
        "module": _EXPECTED_IMPLEMENTATION_MODULE,
        "path": _EXPECTED_IMPLEMENTATION_PATH,
        "sha256": _sha256(payload),
    }


@dataclass(frozen=True)
class Mind2WebDomRankerConfig:
    """Strictly validated immutable ranker configuration and artifact identity."""

    path: Path
    bytes: int
    sha256: str
    config_self_sha256: str
    user_marker: str
    assistant_marker: str
    model_max_seq_len: int
    generation_reserve_tokens_including_eos: int
    max_framed_prompt_bytes: int
    max_unframed_prompt_bytes: int
    minimum_dom_bytes: int
    max_html_bytes: int
    max_nodes: int
    max_depth: int
    excluded_content_tags: frozenset[str]
    void_tags: frozenset[str]
    ancestor_depth: int
    attribute_priority: tuple[str, ...]
    stable_identifier_attributes: frozenset[str]
    interactive_tags: frozenset[str]
    interactive_roles: frozenset[str]
    max_attribute_count: int
    max_attribute_value_bytes: int
    max_stable_identifier_bytes: int
    max_text_bytes: int
    max_ancestor_text_bytes: int
    max_node_line_bytes: int
    implementation: Mapping[str, int | str]
    runtime: Mapping[str, str]

    def audit_identity(self) -> dict[str, Any]:
        """Return the non-secret identity embedded in production adapter audits."""

        return {
            "artifact": {
                "bytes": self.bytes,
                "config_self_sha256": self.config_self_sha256,
                "name": self.path.name,
                "sha256": self.sha256,
            },
            "implementation": dict(self.implementation),
            "input_projection": {
                "allowed": list(_EXPECTED_ALLOWED_FIELDS),
                "forbidden": list(_EXPECTED_FORBIDDEN_FIELDS),
            },
            "ranker_version": MIND2WEB_DOM_RANKER_VERSION,
            "runtime": dict(self.runtime),
        }


def load_mind2web_dom_ranker_config(
    config_path: str | Path,
) -> Mind2WebDomRankerConfig:
    """Load, self-hash, and validate the one supported immutable ranker config."""

    path = Path(config_path)
    payload = _read_verified_config(path)
    try:
        raw = json.loads(
            payload,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("ranker config must be strict UTF-8 JSON") from error
    if not isinstance(raw, dict):
        raise ValueError("ranker config must be a JSON object")
    _exact_keys(raw, _CONFIG_KEYS, label="ranker config")
    if payload != _canonical_json_bytes(raw) + b"\n":
        raise ValueError("ranker config must use canonical JSON with one trailing newline")
    if raw["kind"] != MIND2WEB_DOM_RANKER_CONFIG_KIND:
        raise ValueError("ranker config kind is unsupported")
    if (
        raw["schema_version"] != MIND2WEB_DOM_RANKER_CONFIG_SCHEMA_VERSION
        or isinstance(raw["schema_version"], bool)
    ):
        raise ValueError("ranker config schema_version is unsupported")
    if raw["ranker_version"] != MIND2WEB_DOM_RANKER_VERSION:
        raise ValueError("ranker config version is unsupported")
    if raw["adapter_version"] != MIND2WEB_RANKED_PROMPT_ADAPTER_VERSION:
        raise ValueError("ranker config adapter_version is unsupported")
    self_hash = raw["config_self_sha256"]
    if not isinstance(self_hash, str) or _SHA256_RE.fullmatch(self_hash) is None:
        raise ValueError("ranker config config_self_sha256 is invalid")
    without_hash = dict(raw)
    without_hash.pop("config_self_sha256")
    if _sha256(_canonical_json_bytes(without_hash)) != self_hash:
        raise ValueError("ranker config self-hash mismatch")

    budget = raw["budget"]
    features = raw["features"]
    implementation = raw["implementation"]
    input_projection = raw["input_projection"]
    normalization = raw["normalization"]
    parser = raw["parser"]
    ranking = raw["ranking"]
    runtime = raw["runtime"]
    for value, expected, label in (
        (budget, _BUDGET_KEYS, "ranker config budget"),
        (features, _FEATURE_KEYS, "ranker config features"),
        (implementation, _IMPLEMENTATION_KEYS, "ranker config implementation"),
        (input_projection, _INPUT_PROJECTION_KEYS, "ranker config input_projection"),
        (normalization, _NORMALIZATION_KEYS, "ranker config normalization"),
        (parser, _PARSER_KEYS, "ranker config parser"),
        (ranking, _RANKING_KEYS, "ranker config ranking"),
        (runtime, _RUNTIME_KEYS, "ranker config runtime"),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be an object")
        _exact_keys(value, expected, label=label)

    allowed = _string_tuple(input_projection["allowed"], label="input_projection.allowed")
    forbidden = _string_tuple(
        input_projection["forbidden"], label="input_projection.forbidden"
    )
    if allowed != _EXPECTED_ALLOWED_FIELDS or forbidden != _EXPECTED_FORBIDDEN_FIELDS:
        raise ValueError("ranker config input projection is unsupported")
    if normalization != {
        "casefold": True,
        "form": "NFKC",
        "token_pattern": r"[\w]+",
        "unicode_version": unicodedata.unidata_version,
        "whitespace": _EXPECTED_WHITESPACE,
    }:
        raise ValueError("ranker config normalization/runtime is unsupported")
    python_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    if runtime != {
        "python_implementation": platform.python_implementation(),
        "python_minor": python_minor,
    }:
        raise ValueError("ranker config Python runtime is unsupported")
    if parser["implementation"] != _EXPECTED_PARSER_IMPLEMENTATION:
        raise ValueError("ranker config parser implementation is unsupported")
    if parser["convert_charrefs"] is not True:
        raise ValueError("ranker config must enable deterministic character-reference conversion")
    if ranking != {
        "score": list(_EXPECTED_SCORE),
        "selection": _EXPECTED_SELECTION,
        "serialization": _EXPECTED_SERIALIZATION,
        "tie_break": _EXPECTED_TIE_BREAK,
    }:
        raise ValueError("ranker config ranking algorithm is unsupported")

    if implementation["module"] != _EXPECTED_IMPLEMENTATION_MODULE:
        raise ValueError("ranker config implementation module is unsupported")
    if implementation["path"] != _EXPECTED_IMPLEMENTATION_PATH:
        raise ValueError("ranker config implementation path is unsupported")
    configured_implementation_sha256 = implementation["sha256"]
    if (
        not isinstance(configured_implementation_sha256, str)
        or _SHA256_RE.fullmatch(configured_implementation_sha256) is None
    ):
        raise ValueError("ranker config implementation SHA-256 is invalid")
    observed_implementation = implementation_identity()
    if configured_implementation_sha256 != observed_implementation["sha256"]:
        raise ValueError("ranker implementation SHA-256 disagrees with its frozen config")

    model_max_seq_len = _positive_int(
        budget["model_max_seq_len"], label="budget.model_max_seq_len"
    )
    generation_reserve = _positive_int(
        budget["generation_reserve_tokens_including_eos"],
        label="budget.generation_reserve_tokens_including_eos",
    )
    max_framed_prompt_bytes = _positive_int(
        budget["max_framed_prompt_bytes"], label="budget.max_framed_prompt_bytes"
    )
    max_unframed_prompt_bytes = _positive_int(
        budget["max_unframed_prompt_bytes"],
        label="budget.max_unframed_prompt_bytes",
    )
    minimum_dom_bytes = _positive_int(
        budget["minimum_dom_bytes"], label="budget.minimum_dom_bytes"
    )
    user_marker = _string(budget["user_marker"], label="budget.user_marker")
    assistant_marker = _string(
        budget["assistant_marker"], label="budget.assistant_marker"
    )
    if (
        model_max_seq_len != 2048
        or generation_reserve != 256
        or max_framed_prompt_bytes != model_max_seq_len - generation_reserve
        or max_unframed_prompt_bytes != 1771
        or minimum_dom_bytes != 768
        or user_marker != _EXPECTED_USER_MARKER
        or assistant_marker != _EXPECTED_ASSISTANT_MARKER
    ):
        raise ValueError("ranker config framing/model/generation budget is unsupported")
    framing_bytes = len((user_marker + assistant_marker).encode("utf-8"))
    if max_unframed_prompt_bytes != max_framed_prompt_bytes - framing_bytes:
        raise ValueError("ranker config framed and unframed byte budgets disagree")
    if minimum_dom_bytes > max_unframed_prompt_bytes:
        raise ValueError("ranker config minimum DOM budget exceeds the prompt budget")

    attribute_priority = _string_tuple(
        features["attribute_priority"], label="features.attribute_priority"
    )
    stable_identifier_attributes = _string_tuple(
        features["stable_identifier_attributes"],
        label="features.stable_identifier_attributes",
    )
    if not set(stable_identifier_attributes).issubset(attribute_priority):
        raise ValueError("stable identifier attributes must be in attribute_priority")
    max_attribute_count = _positive_int(
        features["max_attribute_count"], label="features.max_attribute_count"
    )
    if max_attribute_count > len(attribute_priority):
        raise ValueError("features.max_attribute_count exceeds attribute_priority")
    excluded_content_tags = _string_tuple(
        parser["excluded_content_tags"], label="parser.excluded_content_tags"
    )
    void_tags = _string_tuple(parser["void_tags"], label="parser.void_tags")
    interactive_tags = _string_tuple(
        features["interactive_tags"], label="features.interactive_tags"
    )
    interactive_roles = _string_tuple(
        features["interactive_roles"], label="features.interactive_roles"
    )
    max_html_bytes = _positive_int(
        parser["max_html_bytes"], label="parser.max_html_bytes"
    )
    max_nodes = _positive_int(parser["max_nodes"], label="parser.max_nodes")
    max_depth = _positive_int(parser["max_depth"], label="parser.max_depth")
    ancestor_depth = _positive_int(
        features["ancestor_depth"], label="features.ancestor_depth"
    )
    max_attribute_value_bytes = _positive_int(
        features["max_attribute_value_bytes"],
        label="features.max_attribute_value_bytes",
    )
    max_stable_identifier_bytes = _positive_int(
        features["max_stable_identifier_bytes"],
        label="features.max_stable_identifier_bytes",
    )
    max_text_bytes = _positive_int(
        features["max_text_bytes"], label="features.max_text_bytes"
    )
    max_ancestor_text_bytes = _positive_int(
        features["max_ancestor_text_bytes"],
        label="features.max_ancestor_text_bytes",
    )
    max_node_line_bytes = _positive_int(
        features["max_node_line_bytes"], label="features.max_node_line_bytes"
    )
    if (
        excluded_content_tags != _EXPECTED_EXCLUDED_CONTENT_TAGS
        or void_tags != _EXPECTED_VOID_TAGS
        or max_html_bytes != 8 * 1024 * 1024
        or max_nodes != 100_000
        or max_depth != 256
        or ancestor_depth != 2
        or attribute_priority != _EXPECTED_ATTRIBUTE_PRIORITY
        or stable_identifier_attributes != _EXPECTED_STABLE_IDENTIFIER_ATTRIBUTES
        or interactive_tags != _EXPECTED_INTERACTIVE_TAGS
        or interactive_roles != _EXPECTED_INTERACTIVE_ROLES
        or max_attribute_count != 6
        or max_attribute_value_bytes != 48
        or max_stable_identifier_bytes != 96
        or max_text_bytes != 128
        or max_ancestor_text_bytes != 48
        or max_node_line_bytes != 768
    ):
        raise ValueError("ranker config parser/feature profile is unsupported")

    return Mind2WebDomRankerConfig(
        path=path,
        bytes=len(payload),
        sha256=_sha256(payload),
        config_self_sha256=self_hash,
        user_marker=user_marker,
        assistant_marker=assistant_marker,
        model_max_seq_len=model_max_seq_len,
        generation_reserve_tokens_including_eos=generation_reserve,
        max_framed_prompt_bytes=max_framed_prompt_bytes,
        max_unframed_prompt_bytes=max_unframed_prompt_bytes,
        minimum_dom_bytes=minimum_dom_bytes,
        max_html_bytes=max_html_bytes,
        max_nodes=max_nodes,
        max_depth=max_depth,
        excluded_content_tags=frozenset(excluded_content_tags),
        void_tags=frozenset(void_tags),
        ancestor_depth=ancestor_depth,
        attribute_priority=attribute_priority,
        stable_identifier_attributes=frozenset(stable_identifier_attributes),
        interactive_tags=frozenset(interactive_tags),
        interactive_roles=frozenset(interactive_roles),
        max_attribute_count=max_attribute_count,
        max_attribute_value_bytes=max_attribute_value_bytes,
        max_stable_identifier_bytes=max_stable_identifier_bytes,
        max_text_bytes=max_text_bytes,
        max_ancestor_text_bytes=max_ancestor_text_bytes,
        max_node_line_bytes=max_node_line_bytes,
        implementation=MappingProxyType(observed_implementation),
        runtime=MappingProxyType(runtime_identity()),
    )


def _collapse(value: str) -> str:
    return " ".join(value.split())


def _truncate_utf8(value: str, max_bytes: int) -> str:
    payload = value.encode("utf-8")
    if len(payload) <= max_bytes:
        return value
    return payload[:max_bytes].decode("utf-8", errors="ignore")


def _normalized(value: str) -> str:
    return _collapse(unicodedata.normalize("NFKC", value).casefold())


def _validate_inert_prompt_text(value: str, *, label: str) -> None:
    if "\x00" in value:
        raise ValueError(f"{label} contains a forbidden NUL byte")
    for marker in _FORBIDDEN_PROMPT_MARKERS:
        if marker in value:
            raise ValueError(f"{label} contains forbidden prompt-control marker {marker!r}")


@dataclass
class _Node:
    ordinal: int
    depth: int
    tag: str
    parent: int | None
    attrs: tuple[tuple[str, str], ...]
    children: list[int] = field(default_factory=list)
    direct_text: str = ""
    descendant_text: str = ""


class _BoundedDomParser(HTMLParser):
    def __init__(self, config: Mind2WebDomRankerConfig) -> None:
        super().__init__(convert_charrefs=True)
        self.config = config
        self.nodes: list[_Node] = []
        self.stack: list[int] = []
        self.suppressed_tags: list[str] = []

    def _filtered_attrs(
        self,
        attrs: list[tuple[str, str | None]],
    ) -> tuple[tuple[str, str], ...]:
        observed: dict[str, str] = {}
        for raw_name, raw_value in attrs:
            name = raw_name.casefold()
            if name in observed:
                raise ValueError(f"cleaned HTML contains duplicate attribute {name!r}")
            observed[name] = "" if raw_value is None else raw_value
        selected: list[tuple[str, str]] = []
        for name in self.config.attribute_priority:
            if name not in observed or len(selected) >= self.config.max_attribute_count:
                continue
            value = _collapse(observed[name])
            if not value:
                continue
            if name in self.config.stable_identifier_attributes:
                if len(value.encode("utf-8")) > self.config.max_stable_identifier_bytes:
                    raise ValueError(
                        f"stable DOM identifier {name!r} exceeds "
                        f"{self.config.max_stable_identifier_bytes} bytes"
                    )
            else:
                value = _truncate_utf8(value, self.config.max_attribute_value_bytes)
            selected.append((name, value))
        return tuple(selected)

    def _start(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
        *,
        self_closing: bool,
    ) -> None:
        normalized_tag = tag.casefold()
        if self.suppressed_tags:
            if normalized_tag in self.config.excluded_content_tags:
                if not self_closing:
                    self.suppressed_tags.append(normalized_tag)
            return
        if normalized_tag in self.config.excluded_content_tags:
            if not self_closing:
                self.suppressed_tags.append(normalized_tag)
            return
        if len(self.nodes) >= self.config.max_nodes:
            raise ValueError(f"cleaned HTML exceeds max_nodes={self.config.max_nodes}")
        depth = len(self.stack)
        if depth > self.config.max_depth:
            raise ValueError(f"cleaned HTML exceeds max_depth={self.config.max_depth}")
        parent = self.stack[-1] if self.stack else None
        ordinal = len(self.nodes)
        node = _Node(
            ordinal=ordinal,
            depth=depth,
            tag=normalized_tag,
            parent=parent,
            attrs=self._filtered_attrs(attrs),
        )
        self.nodes.append(node)
        if parent is not None:
            self.nodes[parent].children.append(ordinal)
        if not self_closing and normalized_tag not in self.config.void_tags:
            self.stack.append(ordinal)

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        self._start(tag, attrs, self_closing=False)

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        self._start(tag, attrs, self_closing=True)

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.casefold()
        if self.suppressed_tags:
            if normalized_tag == self.suppressed_tags[-1]:
                self.suppressed_tags.pop()
            return
        for offset in range(len(self.stack) - 1, -1, -1):
            if self.nodes[self.stack[offset]].tag == normalized_tag:
                del self.stack[offset:]
                return

    def handle_data(self, data: str) -> None:
        if not self.stack or self.suppressed_tags:
            return
        node = self.nodes[self.stack[-1]]
        combined = _collapse(f"{node.direct_text} {data}")
        node.direct_text = _truncate_utf8(combined, self.config.max_text_bytes)


@dataclass(frozen=True)
class _Candidate:
    ordinal: int
    score: tuple[int, int, int, int]
    line: str


@dataclass(frozen=True)
class RankedDom:
    """One bounded ranked context plus private-safe per-row accounting."""

    context: str
    full_html_bytes: int
    full_html_sha256: str
    parsed_nodes: int
    eligible_nodes: int
    selected_nodes: int
    selected_bytes: int
    ranked_dom_sha256: str


def _ancestor_summaries(
    node: _Node,
    nodes: list[_Node],
    config: Mind2WebDomRankerConfig,
) -> list[list[str]]:
    summaries: list[list[str]] = []
    parent = node.parent
    while parent is not None and len(summaries) < config.ancestor_depth:
        ancestor = nodes[parent]
        attr_text = " ".join(value for _, value in ancestor.attrs)
        label = _truncate_utf8(
            _collapse(f"{attr_text} {ancestor.descendant_text}"),
            config.max_ancestor_text_bytes,
        )
        summaries.append([ancestor.tag, label])
        parent = ancestor.parent
    return summaries


def _serialized_candidate(
    node: _Node,
    ancestors: list[list[str]],
    config: Mind2WebDomRankerConfig,
) -> str:
    attrs = [list(item) for item in node.attrs]

    def render(text: str, ancestor_values: list[list[str]], values: list[list[str]]) -> str:
        return "node " + json.dumps(
            [node.ordinal, node.depth, node.tag, values, text, ancestor_values],
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )

    line = render(node.descendant_text, ancestors, attrs)
    if len(line.encode("utf-8")) <= config.max_node_line_bytes:
        return line
    ancestors_without_text = [[tag, ""] for tag, _ in ancestors]
    line = render("", ancestors_without_text, attrs)
    while len(line.encode("utf-8")) > config.max_node_line_bytes:
        removable = next(
            (
                index
                for index in range(len(attrs) - 1, -1, -1)
                if attrs[index][0] not in config.stable_identifier_attributes
            ),
            None,
        )
        if removable is None:
            break
        del attrs[removable]
        line = render("", ancestors_without_text, attrs)
    if len(line.encode("utf-8")) > config.max_node_line_bytes:
        raise ValueError(
            "canonical DOM node exceeds max_node_line_bytes after removing "
            "all truncatable context"
        )
    return line


def rank_mind2web_dom(
    confirmed_task: str,
    cleaned_html: str,
    *,
    context_byte_budget: int,
    config: Mind2WebDomRankerConfig,
) -> RankedDom:
    """Rank one current DOM using only the task text and cleaned HTML."""

    if not isinstance(confirmed_task, str) or not confirmed_task.strip():
        raise ValueError("confirmed_task must be a non-empty string")
    if not isinstance(cleaned_html, str) or not cleaned_html.strip():
        raise ValueError("cleaned_html must be a non-empty string")
    _validate_inert_prompt_text(confirmed_task, label="confirmed_task")
    _validate_inert_prompt_text(cleaned_html, label="cleaned_html")
    context_byte_budget = _positive_int(
        context_byte_budget, label="context_byte_budget"
    )
    if context_byte_budget < config.minimum_dom_bytes:
        raise ValueError(
            f"ranked DOM residual budget is {context_byte_budget} bytes, below "
            f"minimum_dom_bytes={config.minimum_dom_bytes}"
        )
    html_payload = cleaned_html.encode("utf-8")
    if len(html_payload) > config.max_html_bytes:
        raise ValueError(
            f"cleaned_html exceeds max_html_bytes={config.max_html_bytes}"
        )
    parser = _BoundedDomParser(config)
    try:
        parser.feed(cleaned_html)
        parser.close()
    except (RecursionError, UnicodeError) as error:
        raise ValueError("cleaned_html could not be parsed deterministically") from error
    nodes = parser.nodes
    if not nodes:
        raise ValueError("cleaned_html contains no eligible DOM nodes")

    for node in reversed(nodes):
        descendant_parts = [node.direct_text]
        descendant_parts.extend(nodes[child].descendant_text for child in node.children)
        node.descendant_text = _truncate_utf8(
            _collapse(" ".join(descendant_parts)),
            config.max_text_bytes,
        )

    normalized_task = _normalized(confirmed_task)
    query_tokens = frozenset(_TOKEN_RE.findall(normalized_task))
    if not query_tokens:
        raise ValueError("confirmed_task contains no rankable Unicode word tokens")
    candidates: list[_Candidate] = []
    for node in nodes:
        ancestors = _ancestor_summaries(node, nodes, config)
        own_text = _normalized(
            " ".join([*(value for _, value in node.attrs), node.descendant_text])
        )
        ancestor_text = _normalized(" ".join(label for _, label in ancestors))
        own_tokens = frozenset(_TOKEN_RE.findall(own_text))
        ancestor_tokens = frozenset(_TOKEN_RE.findall(ancestor_text))
        role_values = {
            _normalized(value)
            for name, value in node.attrs
            if name == "role"
        }
        score = (
            int(bool(normalized_task) and normalized_task in own_text),
            len(query_tokens & own_tokens),
            len(query_tokens & ancestor_tokens),
            int(
                node.tag in config.interactive_tags
                or bool(role_values & config.interactive_roles)
            ),
        )
        candidates.append(
            _Candidate(
                ordinal=node.ordinal,
                score=score,
                line=_serialized_candidate(node, ancestors, config),
            )
        )

    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -candidate.score[0],
            -candidate.score[1],
            -candidate.score[2],
            -candidate.score[3],
            candidate.ordinal,
        ),
    )
    selected: list[_Candidate] = []
    selected_bytes = 0
    for candidate in ranked:
        candidate_bytes = len(candidate.line.encode("utf-8"))
        separator_bytes = int(bool(selected))
        if selected_bytes + separator_bytes + candidate_bytes > context_byte_budget:
            continue
        selected.append(candidate)
        selected_bytes += separator_bytes + candidate_bytes
    if not selected:
        raise ValueError("no canonical DOM node fits the ranked context byte budget")
    selected.sort(key=lambda candidate: candidate.ordinal)
    context = "\n".join(candidate.line for candidate in selected)
    observed_context_bytes = len(context.encode("utf-8"))
    if observed_context_bytes != selected_bytes or selected_bytes > context_byte_budget:
        raise AssertionError("ranked DOM byte accounting drifted")
    return RankedDom(
        context=context,
        full_html_bytes=len(html_payload),
        full_html_sha256=_sha256(html_payload),
        parsed_nodes=len(nodes),
        eligible_nodes=len(candidates),
        selected_nodes=len(selected),
        selected_bytes=selected_bytes,
        ranked_dom_sha256=_sha256(context.encode("utf-8")),
    )
