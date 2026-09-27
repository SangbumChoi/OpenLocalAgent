"""Strict verification and overlap auditing for synthetic Conversation JSONL artifacts.

The generator publishes a canonical, self-hashed sidecar next to each JSONL file.  Consumers use
``load_verified_conversation_artifact`` rather than trusting the JSONL pathname alone, so the
generator config, declared verification policy, bytes actually parsed, and lineage identities stay
bound together.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal

from localagent.data.prompt_contract import (
    LEGACY_CONVERSATION_PROMPT_CONTRACT,
    OPENAI_FULL_CATALOG_V1,
    FunctionCatalogCache,
    assistant_training_examples,
    resolve_conversation_prompt_contract,
)
from localagent.data.schema import Conversation, Message, Role, ToolCall, ToolSpec
from localagent.model.tokenizer import (
    ASSISTANT,
    TOOL,
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    TOOL_RESPONSE_CLOSE,
    TOOL_RESPONSE_OPEN,
    USER,
)

MANIFEST_KIND = "localagent_synthetic_conversation_artifact"
MANIFEST_SCHEMA_VERSION = 1
CONVERSATION_FORMAT = "localagent.data.schema.Conversation"
CONVERSATION_SERIALIZATION = "schema_roundtrip_jsonl_utf8_lf_v1"

EnvironmentPolicy = Literal["forbid", "allow", "require"]

__all__ = [
    "CONVERSATION_FORMAT",
    "CONVERSATION_SERIALIZATION",
    "MANIFEST_KIND",
    "MANIFEST_SCHEMA_VERSION",
    "ConversationArtifactIdentity",
    "ConversationOverlapAudit",
    "FileIdentity",
    "VerifiedConversationArtifact",
    "assert_no_conversation_overlap",
    "audit_conversation_overlap",
    "canonical_json_bytes",
    "conversation_semantic_sha256",
    "load_verified_conversation_artifact",
    "rendered_assistant_prompts",
    "rendered_prompt_sha256",
    "self_hashed_manifest",
]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_CONFIG_BYTES = 4 * 1024 * 1024
_MAX_ROW_BYTES = 16 * 1024 * 1024
_MANIFEST_KEYS = frozenset(
    {
        "argument_schema_coverage",
        "argument_value_counts",
        "behavior_counts",
        "behavior_definitions",
        "complexity_contract",
        "conversation_serialization",
        "coverage_contract",
        "environment_executed",
        "exact_prompt_holdouts",
        "format",
        "generator_config",
        "irrelevance",
        "kind",
        "level",
        "manifest_self_sha256",
        "model_verified",
        "multi_turn",
        "output_bytes",
        "output_sha256",
        "plan_length_counts",
        "rows",
        "rule_verification_scope",
        "rule_verified",
        "schema_version",
        "seed",
        "single_turn",
        "split",
        "split_contract",
        "structural_counts",
        "verification_claim",
    }
)


class _ReadOnlyList(list[Any]):
    """A JSON-compatible list that cannot mutate after verified parsing."""

    __slots__ = ()
    _localagent_verified_read_only = True

    @staticmethod
    def _immutable(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("verified conversation snapshots are read-only")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable

    def __copy__(self) -> _ReadOnlyList:
        return self

    def __deepcopy__(self, _memo: dict[int, Any]) -> _ReadOnlyList:
        return self


class _ReadOnlyDict(dict[str, Any]):
    """A JSON-compatible dict that cannot mutate after verified parsing."""

    __slots__ = ()

    @staticmethod
    def _immutable(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("verified conversation snapshots are read-only")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __copy__(self) -> _ReadOnlyDict:
        return self

    def __deepcopy__(self, _memo: dict[int, Any]) -> _ReadOnlyDict:
        return self


class _ReadOnlyToolSpec(ToolSpec):
    """A ToolSpec-compatible immutable value safe to share between artifact rows."""

    __slots__ = ()

    def __init__(self, name: str, description: str, parameters: Any) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "parameters", _freeze_json(parameters))

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise TypeError("verified conversation snapshots are read-only")

    def __delattr__(self, _name: str) -> None:
        raise TypeError("verified conversation snapshots are read-only")

    def __copy__(self) -> _ReadOnlyToolSpec:
        return self

    def __deepcopy__(self, _memo: dict[int, Any]) -> _ReadOnlyToolSpec:
        return self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolSpec):
            return NotImplemented
        return (
            self.name == other.name
            and self.description == other.description
            and self.parameters == other.parameters
        )


class _ReadOnlyToolCall(ToolCall):
    """A ToolCall-compatible immutable value for a verified snapshot."""

    __slots__ = ()

    def __init__(self, name: str, arguments: Any) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "arguments", _freeze_json(arguments))

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise TypeError("verified conversation snapshots are read-only")

    def __delattr__(self, _name: str) -> None:
        raise TypeError("verified conversation snapshots are read-only")

    def __copy__(self) -> _ReadOnlyToolCall:
        return self

    def __deepcopy__(self, _memo: dict[int, Any]) -> _ReadOnlyToolCall:
        return self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolCall):
            return NotImplemented
        return self.name == other.name and self.arguments == other.arguments


class _ReadOnlyMessage(Message):
    """A Message-compatible immutable value for a verified snapshot."""

    __slots__ = ()

    def __init__(
        self,
        role: Role,
        content: str = "",
        tool_calls: Sequence[ToolCall] | None = None,
        tool_response: str | None = None,
    ) -> None:
        frozen_calls = _ReadOnlyList(
            call
            if isinstance(call, _ReadOnlyToolCall)
            else _ReadOnlyToolCall(name=call.name, arguments=call.arguments)
            for call in (tool_calls or [])
        )
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "tool_calls", frozen_calls)
        object.__setattr__(self, "tool_response", tool_response)

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise TypeError("verified conversation snapshots are read-only")

    def __delattr__(self, _name: str) -> None:
        raise TypeError("verified conversation snapshots are read-only")

    def __copy__(self) -> _ReadOnlyMessage:
        return self

    def __deepcopy__(self, _memo: dict[int, Any]) -> _ReadOnlyMessage:
        return self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Message):
            return NotImplemented
        return (
            self.role == other.role
            and self.content == other.content
            and self.tool_calls == other.tool_calls
            and self.tool_response == other.tool_response
        )


class _ReadOnlyConversation(Conversation):
    """A Conversation-compatible immutable graph returned only by verified loading."""

    __slots__ = ()

    def __init__(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        frozen_messages = _ReadOnlyList(
            message
            if isinstance(message, _ReadOnlyMessage)
            else _ReadOnlyMessage(
                role=message.role,
                content=message.content,
                tool_calls=message.tool_calls,
                tool_response=message.tool_response,
            )
            for message in messages
        )
        if isinstance(tools, _ReadOnlyList):
            frozen_tools = tools
        else:
            frozen_tools = _ReadOnlyList(
                tool
                if isinstance(tool, _ReadOnlyToolSpec)
                else _ReadOnlyToolSpec(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters,
                )
                for tool in (tools or [])
            )
        object.__setattr__(self, "messages", frozen_messages)
        object.__setattr__(self, "tools", frozen_tools)
        object.__setattr__(self, "meta", _freeze_json(dict(meta or {})))

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise TypeError("verified conversation snapshots are read-only")

    def __delattr__(self, _name: str) -> None:
        raise TypeError("verified conversation snapshots are read-only")

    def __copy__(self) -> _ReadOnlyConversation:
        return self

    def __deepcopy__(self, _memo: dict[int, Any]) -> _ReadOnlyConversation:
        return self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Conversation):
            return NotImplemented
        return (
            self.messages == other.messages
            and self.tools == other.tools
            and self.meta == other.meta
        )


@dataclass(frozen=True)
class _RegularFileState:
    """Stable fields used to detect ordinary descriptor and pathname drift."""

    device: int
    inode: int
    mode: int
    links: int
    size: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> _RegularFileState:
        return cls(
            device=value.st_dev,
            inode=value.st_ino,
            mode=value.st_mode,
            links=value.st_nlink,
            size=value.st_size,
            modified_ns=value.st_mtime_ns,
            changed_ns=value.st_ctime_ns,
        )


@dataclass
class _CatalogCacheEntry:
    serialized: bytes
    tools: _ReadOnlyList


_CatalogCache = dict[bytes, list[_CatalogCacheEntry]]


@dataclass(frozen=True)
class FileIdentity:
    """Content identity independent of an artifact's filesystem location."""

    bytes: int
    sha256: str

    @classmethod
    def from_bytes(cls, payload: bytes) -> FileIdentity:
        return cls(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())

    def as_dict(self) -> dict[str, int | str]:
        return {"bytes": self.bytes, "sha256": self.sha256}


@dataclass(frozen=True)
class ConversationArtifactIdentity:
    """The complete byte lineage for one verified synthetic dataset."""

    jsonl: FileIdentity
    sidecar: FileIdentity
    generator_config: FileIdentity
    manifest_self_sha256: str
    split: str

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": MANIFEST_KIND,
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "split": self.split,
            "jsonl": self.jsonl.as_dict(),
            "sidecar": {
                **self.sidecar.as_dict(),
                "manifest_self_sha256": self.manifest_self_sha256,
            },
            "generator_config": self.generator_config.as_dict(),
        }


@dataclass(frozen=True)
class VerifiedConversationArtifact:
    """Parsed conversations plus the identities and policy that were verified."""

    conversations: tuple[Conversation, ...]
    manifest: Mapping[str, Any]
    identity: ConversationArtifactIdentity
    data_path: Path
    manifest_path: Path
    config_path: Path
    rule_verified: bool
    environment_executed: bool

    def lineage_identity(self) -> dict[str, object]:
        """Return the location-independent identity suitable for stage lineage hashing."""

        return self.identity.as_dict()


@dataclass(frozen=True)
class ConversationOverlapAudit:
    """Semantic-row and exact rendered-assistant-prompt overlap evidence."""

    left_rows: int
    right_rows: int
    left_rendered_prompts: int
    right_rendered_prompts: int
    left_semantic_set_sha256: str
    right_semantic_set_sha256: str
    left_rendered_prompt_set_sha256: str
    right_rendered_prompt_set_sha256: str
    semantic_overlap_sha256: tuple[str, ...]
    rendered_prompt_overlap_sha256: tuple[str, ...]
    conversation_prompt_contract: str = LEGACY_CONVERSATION_PROMPT_CONTRACT

    @property
    def clean(self) -> bool:
        return not self.semantic_overlap_sha256 and not self.rendered_prompt_overlap_sha256

    def as_dict(self) -> dict[str, object]:
        if self.conversation_prompt_contract == OPENAI_FULL_CATALOG_V1:
            rendered_prompt_contract = (
                "sha256(UTF-8 exact full-catalog assistant decode prompt: canonical catalog+EOS, "
                "system/role history, prior assistant EOS, current assistant marker)"
            )
        else:
            rendered_prompt_contract = (
                "sha256(UTF-8 legacy history through the current assistant marker; "
                "system and catalog excluded, prior assistant EOS excluded)"
            )
        return {
            "fingerprint_contract": {
                "conversation_prompt_contract": self.conversation_prompt_contract,
                "semantic_row": (
                    "sha256(canonical compact sorted-key JSON of messages+tools; meta excluded)"
                ),
                "rendered_prompt": rendered_prompt_contract,
                "set_aggregation": "sha256(sorted unique fingerprints joined by LF)",
            },
            "left_rows": self.left_rows,
            "right_rows": self.right_rows,
            "left_rendered_prompts": self.left_rendered_prompts,
            "right_rendered_prompts": self.right_rendered_prompts,
            "left_semantic_set_sha256": self.left_semantic_set_sha256,
            "right_semantic_set_sha256": self.right_semantic_set_sha256,
            "left_rendered_prompt_set_sha256": self.left_rendered_prompt_set_sha256,
            "right_rendered_prompt_set_sha256": self.right_rendered_prompt_set_sha256,
            "semantic_overlap": len(self.semantic_overlap_sha256),
            "rendered_prompt_overlap": len(self.rendered_prompt_overlap_sha256),
            "semantic_overlap_sha256": list(self.semantic_overlap_sha256),
            "rendered_prompt_overlap_sha256": list(self.rendered_prompt_overlap_sha256),
        }


def canonical_json_bytes(value: Any) -> bytes:
    """Return the project's canonical compact, sorted-key JSON bytes with one trailing LF."""

    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def self_hashed_manifest(
    manifest_without_hash: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    """Attach a canonical self-hash and return the manifest plus publishable bytes."""

    if "manifest_self_sha256" in manifest_without_hash:
        raise ValueError("unsigned manifest must not contain manifest_self_sha256")
    core = dict(manifest_without_hash)
    self_sha256 = hashlib.sha256(canonical_json_bytes(core)).hexdigest()
    manifest = {**core, "manifest_self_sha256": self_sha256}
    payload = canonical_json_bytes(manifest)
    if len(payload) > _MAX_MANIFEST_BYTES:
        raise ValueError(
            f"canonical synthetic conversation sidecar exceeds {_MAX_MANIFEST_BYTES} bytes"
        )
    return manifest, payload


def _strict_json(payload: bytes, *, label: str) -> Any:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not valid UTF-8") from exc

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-finite JSON number {value!r}")

    try:
        return json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc


def _same_open_file(left: _RegularFileState, right: _RegularFileState) -> bool:
    return left.device == right.device and left.inode == right.inode


def _path_file_state(path: Path, *, label: str) -> _RegularFileState:
    try:
        value = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"{label} changed while it was being verified: {path}") from exc
    state = _RegularFileState.from_stat(value)
    if not stat.S_ISREG(state.mode):
        raise ValueError(f"{label} changed while it was being verified: {path}")
    return state


@contextmanager
def _open_regular_file(
    path: Path,
    *,
    label: str,
) -> Iterator[tuple[BinaryIO, _RegularFileState]]:
    """Open one non-symlink regular-file descriptor and verify it stays path-bound."""

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{label} is missing or not a regular non-symlink file: {path}") from exc

    handle: BinaryIO | None = None
    try:
        descriptor_state = _RegularFileState.from_stat(os.fstat(descriptor))
        path_state = _path_file_state(path, label=label)
        if (
            not stat.S_ISREG(descriptor_state.mode)
            or not _same_open_file(descriptor_state, path_state)
            or descriptor_state != path_state
        ):
            raise ValueError(f"{label} is missing or not a regular non-symlink file: {path}")

        handle = os.fdopen(descriptor, "rb")
        descriptor = -1
        try:
            yield handle, descriptor_state
        finally:
            try:
                final_descriptor_state = _RegularFileState.from_stat(os.fstat(handle.fileno()))
                final_path_state = _path_file_state(path, label=label)
                if (
                    final_descriptor_state != descriptor_state
                    or final_path_state != descriptor_state
                    or not _same_open_file(final_descriptor_state, final_path_state)
                ):
                    raise ValueError(f"{label} changed while it was being verified: {path}")
            finally:
                handle.close()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_regular_file(path: Path, *, label: str, max_bytes: int | None) -> bytes:
    with _open_regular_file(path, label=label) as (handle, state):
        if max_bytes is not None and state.size > max_bytes:
            raise ValueError(f"{label} exceeds hard byte cap {max_bytes}: {path}")
        payload = handle.read() if max_bytes is None else handle.read(max_bytes + 1)
    if max_bytes is not None and len(payload) > max_bytes:
        raise ValueError(f"{label} exceeds hard byte cap {max_bytes}: {path}")
    return payload


def _identity_object(value: Any, *, label: str) -> FileIdentity:
    if not isinstance(value, Mapping) or set(value) != {"bytes", "sha256"}:
        raise ValueError(f"{label} must contain exactly bytes and sha256")
    byte_count = value.get("bytes")
    sha256 = value.get("sha256")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise ValueError(f"{label}.bytes must be a non-negative integer")
    if not isinstance(sha256, str) or _SHA256.fullmatch(sha256) is None:
        raise ValueError(f"{label}.sha256 must be a lowercase SHA-256")
    return FileIdentity(bytes=byte_count, sha256=sha256)


def _manifest(
    manifest_path: Path,
    *,
    expected_identity: FileIdentity | None,
) -> tuple[dict[str, Any], bytes, FileIdentity]:
    payload = _read_regular_file(
        manifest_path,
        label="synthetic conversation sidecar",
        max_bytes=_MAX_MANIFEST_BYTES,
    )
    identity = FileIdentity.from_bytes(payload)
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("synthetic conversation sidecar byte identity mismatch")
    value = _strict_json(payload, label="synthetic conversation sidecar")
    if not isinstance(value, dict):
        raise TypeError("synthetic conversation sidecar must be a JSON object")
    missing = sorted(_MANIFEST_KEYS - set(value))
    extra = sorted(set(value) - _MANIFEST_KEYS)
    if missing or extra:
        raise ValueError(
            f"synthetic conversation sidecar keys mismatch: missing={missing}, extra={extra}"
        )
    if (
        value.get("kind") != MANIFEST_KIND
        or isinstance(value.get("schema_version"), bool)
        or value.get("schema_version") != MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError(
            f"synthetic conversation sidecar must be {MANIFEST_KIND!r} "
            f"schema_version {MANIFEST_SCHEMA_VERSION}"
        )
    if value.get("format") != CONVERSATION_FORMAT:
        raise ValueError(f"synthetic conversation sidecar format must be {CONVERSATION_FORMAT!r}")
    if value.get("conversation_serialization") != CONVERSATION_SERIALIZATION:
        raise ValueError("unsupported synthetic conversation serialization")
    self_sha256 = value.get("manifest_self_sha256")
    if not isinstance(self_sha256, str) or _SHA256.fullmatch(self_sha256) is None:
        raise ValueError("manifest_self_sha256 must be a lowercase SHA-256")
    unsigned = dict(value)
    unsigned.pop("manifest_self_sha256")
    if hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != self_sha256:
        raise ValueError("synthetic conversation sidecar manifest_self_sha256 mismatch")
    if payload != canonical_json_bytes(value):
        raise ValueError("synthetic conversation sidecar must use canonical JSON bytes")
    return value, payload, identity


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        if isinstance(value, _ReadOnlyDict):
            return value
        return _ReadOnlyDict((key, _freeze_json(item)) for key, item in value.items())
    if isinstance(value, list):
        if isinstance(value, _ReadOnlyList):
            return value
        return _ReadOnlyList(_freeze_json(item) for item in value)
    return value


def _catalog_digest(serialized: bytes) -> bytes:
    """Return the cache bucket key; exact bytes are still compared inside the bucket."""

    return hashlib.sha256(serialized).digest()


def _intern_tool_catalog(raw_tools: Any, cache: _CatalogCache) -> _ReadOnlyList:
    serialized = json.dumps(raw_tools, allow_nan=False).encode("utf-8")
    digest = _catalog_digest(serialized)
    bucket = cache.setdefault(digest, [])
    for entry in bucket:
        if entry.serialized == serialized:
            return entry.tools

    frozen_tools = []
    for tool in raw_tools:
        frozen_tool = dict(tool)
        frozen_tool["parameters"] = _freeze_json(tool["parameters"])
        frozen_tools.append(_ReadOnlyToolSpec(**frozen_tool))
    tools = _ReadOnlyList(frozen_tools)
    bucket.append(_CatalogCacheEntry(serialized=serialized, tools=tools))
    return tools


def _conversation_from_raw(raw: dict[str, Any], cache: _CatalogCache) -> Conversation:
    tools = _intern_tool_catalog(raw.get("tools", []), cache)
    messages = []
    for message in raw["messages"]:
        calls = [_ReadOnlyToolCall(**call) for call in message.get("tool_calls", [])]
        messages.append(
            _ReadOnlyMessage(
                role=Role(message["role"]),
                content=message.get("content", ""),
                tool_calls=calls,
                tool_response=message.get("tool_response"),
            )
        )
    return _ReadOnlyConversation(messages=messages, tools=tools, meta=raw.get("meta", {}))


def _parse_conversation_line(
    line: bytes,
    *,
    line_number: int,
    cache: _CatalogCache,
) -> Conversation:
    label = f"synthetic conversation JSONL line {line_number}"
    if not line.endswith(b"\n") or line.endswith(b"\r\n"):
        raise ValueError(f"{label} must end in exactly one LF")
    body = line[:-1]
    if not body:
        raise ValueError(f"{label} must not be empty")
    raw = _strict_json(body, label=label)
    if not isinstance(raw, dict):
        raise TypeError(f"{label} must be a Conversation JSON object")
    try:
        conversation = _conversation_from_raw(raw, cache)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} does not match the Conversation schema") from exc
    canonical = (conversation.to_json() + "\n").encode("utf-8")
    if line != canonical:
        raise ValueError(f"{label} is not the canonical Conversation schema serialization")
    return conversation


def _stream_canonical_conversations(
    data_path: Path,
    *,
    declared_identity: FileIdentity,
    max_jsonl_bytes: int | None,
) -> tuple[tuple[Conversation, ...], FileIdentity]:
    """Parse one descriptor in row-sized buffers while binding the exact streamed bytes."""

    label = "synthetic conversation JSONL"
    with _open_regular_file(data_path, label=label) as (handle, state):
        if max_jsonl_bytes is not None and state.size > max_jsonl_bytes:
            raise ValueError(f"{label} exceeds hard byte cap {max_jsonl_bytes}: {data_path}")
        if state.size != declared_identity.bytes:
            raise ValueError(f"{label} byte identity mismatch")

        observed_bytes = 0
        digest = hashlib.sha256()
        conversations: list[Conversation] = []
        catalog_cache: _CatalogCache = {}
        first_row_error: TypeError | ValueError | None = None
        line_number = 0

        while True:
            line = handle.readline(_MAX_ROW_BYTES + 1)
            if not line:
                break
            line_number += 1
            observed_bytes += len(line)
            digest.update(line)
            if max_jsonl_bytes is not None and observed_bytes > max_jsonl_bytes:
                raise ValueError(f"{label} exceeds hard byte cap {max_jsonl_bytes}: {data_path}")
            if observed_bytes > declared_identity.bytes:
                raise ValueError(f"{label} byte identity mismatch")

            if len(line) > _MAX_ROW_BYTES:
                if first_row_error is None:
                    first_row_error = ValueError(
                        f"{label} line {line_number} exceeds hard byte cap {_MAX_ROW_BYTES}"
                    )
                while line and not line.endswith(b"\n"):
                    line = handle.readline(_MAX_ROW_BYTES + 1)
                    observed_bytes += len(line)
                    digest.update(line)
                    if max_jsonl_bytes is not None and observed_bytes > max_jsonl_bytes:
                        raise ValueError(
                            f"{label} exceeds hard byte cap {max_jsonl_bytes}: {data_path}"
                        )
                    if observed_bytes > declared_identity.bytes:
                        raise ValueError(f"{label} byte identity mismatch")
                continue

            if first_row_error is not None:
                continue
            try:
                conversation = _parse_conversation_line(
                    line,
                    line_number=line_number,
                    cache=catalog_cache,
                )
            except (TypeError, ValueError) as exc:
                first_row_error = exc
            else:
                conversations.append(conversation)

        observed_identity = FileIdentity(bytes=observed_bytes, sha256=digest.hexdigest())
        if observed_identity != declared_identity:
            raise ValueError(f"{label} byte identity mismatch")
        if first_row_error is not None:
            raise first_row_error
        if not conversations:
            raise ValueError(f"{label} must not be empty")
        return tuple(conversations), observed_identity


def load_verified_conversation_artifact(
    data_path: str | Path,
    *,
    config_path: str | Path,
    expected_split: str,
    manifest_path: str | Path | None = None,
    expected_rule_verified: bool | None = True,
    environment_policy: EnvironmentPolicy = "forbid",
    expected_manifest_identity: FileIdentity | None = None,
    max_jsonl_bytes: int | None = None,
) -> VerifiedConversationArtifact:
    """Verify and parse a provenance-bound synthetic Conversation JSONL artifact.

    ``environment_policy`` is fail-closed: ``forbid`` accepts only unexecuted template data,
    ``require`` accepts only environment-executed data, and ``allow`` accepts either state while
    still requiring every row to agree with the manifest.
    """

    if not isinstance(expected_split, str) or not expected_split:
        raise ValueError("expected_split must be a non-empty string")
    if expected_rule_verified is not None and not isinstance(expected_rule_verified, bool):
        raise ValueError("expected_rule_verified must be bool or None")
    if environment_policy not in {"forbid", "allow", "require"}:
        raise ValueError("environment_policy must be 'forbid', 'allow', or 'require'")
    if max_jsonl_bytes is not None and (
        isinstance(max_jsonl_bytes, bool)
        or not isinstance(max_jsonl_bytes, int)
        or max_jsonl_bytes < 0
    ):
        raise ValueError("max_jsonl_bytes must be a non-negative integer or None")

    data = Path(data_path)
    config = Path(config_path)
    sidecar = (
        Path(manifest_path)
        if manifest_path is not None
        else data.with_suffix(data.suffix + ".manifest.json")
    )
    manifest, _manifest_payload, manifest_identity = _manifest(
        sidecar,
        expected_identity=expected_manifest_identity,
    )

    config_payload = _read_regular_file(
        config,
        label="synthetic conversation generator config",
        max_bytes=_MAX_CONFIG_BYTES,
    )
    config_identity = FileIdentity.from_bytes(config_payload)
    declared_config_identity = _identity_object(
        manifest.get("generator_config"),
        label="generator_config",
    )
    if config_identity != declared_config_identity:
        raise ValueError("synthetic conversation generator config byte identity mismatch")

    declared_data_identity = _identity_object(
        {
            "bytes": manifest.get("output_bytes"),
            "sha256": manifest.get("output_sha256"),
        },
        label="synthetic conversation output",
    )
    conversations, data_identity = _stream_canonical_conversations(
        data,
        declared_identity=declared_data_identity,
        max_jsonl_bytes=max_jsonl_bytes,
    )
    rows = manifest.get("rows")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
        raise ValueError("synthetic conversation sidecar rows must be a positive integer")
    if len(conversations) != rows:
        raise ValueError(
            "synthetic conversation JSONL row count mismatch: "
            f"declared={rows}, observed={len(conversations)}"
        )

    split = manifest.get("split")
    if split != expected_split:
        raise ValueError(
            f"synthetic conversation split mismatch: expected={expected_split!r}, "
            f"observed={split!r}"
        )
    rule_verified = manifest.get("rule_verified")
    environment_executed = manifest.get("environment_executed")
    if not isinstance(rule_verified, bool):
        raise TypeError("synthetic conversation sidecar rule_verified must be boolean")
    if not isinstance(environment_executed, bool):
        raise TypeError("synthetic conversation sidecar environment_executed must be boolean")
    if expected_rule_verified is not None and rule_verified is not expected_rule_verified:
        raise ValueError(
            "synthetic conversation rule-verification state mismatch: "
            f"expected={expected_rule_verified}, observed={rule_verified}"
        )
    if environment_policy == "forbid" and environment_executed:
        raise ValueError("environment-executed synthetic conversations are forbidden")
    if environment_policy == "require" and not environment_executed:
        raise ValueError("environment-executed synthetic conversations are required")

    for row_number, conversation in enumerate(conversations, start=1):
        if conversation.meta.get("split") != split:
            raise ValueError(
                f"synthetic conversation row {row_number} split disagrees with sidecar"
            )
        if conversation.meta.get("rule_verified") is not rule_verified:
            raise ValueError(
                f"synthetic conversation row {row_number} rule_verified disagrees with sidecar"
            )
        if conversation.meta.get("environment_executed") is not environment_executed:
            raise ValueError(
                f"synthetic conversation row {row_number} "
                "environment_executed disagrees with sidecar"
            )

    identity = ConversationArtifactIdentity(
        jsonl=data_identity,
        sidecar=manifest_identity,
        generator_config=config_identity,
        manifest_self_sha256=str(manifest["manifest_self_sha256"]),
        split=expected_split,
    )
    return VerifiedConversationArtifact(
        conversations=conversations,
        manifest=_freeze_json(manifest),
        identity=identity,
        data_path=data,
        manifest_path=sidecar,
        config_path=config,
        rule_verified=rule_verified,
        environment_executed=environment_executed,
    )


def conversation_semantic_sha256(conversation: Conversation) -> str:
    """Hash messages and tools while intentionally excluding mutable provenance metadata."""

    payload = asdict(conversation)
    payload.pop("meta", None)
    return hashlib.sha256(canonical_json_bytes(payload)[:-1]).hexdigest()


def _tool_call_text(message: Message) -> str:
    return "".join(
        TOOL_CALL_OPEN
        + json.dumps(
            {"name": call.name, "arguments": call.arguments},
            separators=(",", ":"),
            sort_keys=True,
        )
        + TOOL_CALL_CLOSE
        for call in message.tool_calls
    )


def rendered_assistant_prompts(
    conversation: Conversation,
    *,
    conversation_prompt_contract: str | None = None,
    catalog_cache: FunctionCatalogCache | None = None,
) -> tuple[str, ...]:
    """Return each exact assistant decode prompt under the selected prompt contract."""

    contract = resolve_conversation_prompt_contract(conversation_prompt_contract)
    if contract == OPENAI_FULL_CATALOG_V1:
        return tuple(
            example.prompt
            for example in assistant_training_examples(
                conversation,
                catalog_cache=catalog_cache,
            )
        )
    history: list[str] = []
    prompts: list[str] = []
    for message in conversation.messages:
        if message.role == Role.user:
            history.append(USER + message.content)
        elif message.role == Role.tool:
            history.append(
                TOOL + TOOL_RESPONSE_OPEN + (message.tool_response or "") + TOOL_RESPONSE_CLOSE
            )
        elif message.role == Role.assistant:
            prompts.append("".join(history) + ASSISTANT)
            body = _tool_call_text(message) if message.tool_calls else message.content
            history.append(ASSISTANT + body)
    return tuple(prompts)


def rendered_prompt_sha256(prompt: str) -> str:
    """Hash one exact UTF-8 rendered assistant decode prefix."""

    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _fingerprint_set_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(set(values))).encode("ascii")).hexdigest()


def audit_conversation_overlap(
    left: Sequence[Conversation],
    right: Sequence[Conversation],
    *,
    conversation_prompt_contract: str | None = None,
) -> ConversationOverlapAudit:
    """Audit semantic rows and all rendered assistant prefixes across two splits."""

    contract = resolve_conversation_prompt_contract(conversation_prompt_contract)
    catalog_cache = FunctionCatalogCache()
    left_semantic = [conversation_semantic_sha256(row) for row in left]
    right_semantic = [conversation_semantic_sha256(row) for row in right]
    left_prompts = [
        rendered_prompt_sha256(prompt)
        for conversation in left
        for prompt in rendered_assistant_prompts(
            conversation,
            conversation_prompt_contract=contract,
            catalog_cache=catalog_cache,
        )
    ]
    right_prompts = [
        rendered_prompt_sha256(prompt)
        for conversation in right
        for prompt in rendered_assistant_prompts(
            conversation,
            conversation_prompt_contract=contract,
            catalog_cache=catalog_cache,
        )
    ]
    return ConversationOverlapAudit(
        left_rows=len(left),
        right_rows=len(right),
        left_rendered_prompts=len(left_prompts),
        right_rendered_prompts=len(right_prompts),
        left_semantic_set_sha256=_fingerprint_set_sha256(left_semantic),
        right_semantic_set_sha256=_fingerprint_set_sha256(right_semantic),
        left_rendered_prompt_set_sha256=_fingerprint_set_sha256(left_prompts),
        right_rendered_prompt_set_sha256=_fingerprint_set_sha256(right_prompts),
        semantic_overlap_sha256=tuple(sorted(set(left_semantic) & set(right_semantic))),
        rendered_prompt_overlap_sha256=tuple(sorted(set(left_prompts) & set(right_prompts))),
        conversation_prompt_contract=contract,
    )


def assert_no_conversation_overlap(
    left: Sequence[Conversation],
    right: Sequence[Conversation],
    *,
    left_label: str = "train",
    right_label: str = "eval",
    conversation_prompt_contract: str | None = None,
) -> ConversationOverlapAudit:
    """Return clean audit evidence or reject semantic/rendered-prompt contamination."""

    audit = audit_conversation_overlap(
        left,
        right,
        conversation_prompt_contract=conversation_prompt_contract,
    )
    if not audit.clean:
        raise ValueError(
            f"{left_label}/{right_label} conversation contamination: "
            f"semantic_rows={len(audit.semantic_overlap_sha256)}, "
            f"rendered_prompts={len(audit.rendered_prompt_overlap_sha256)}"
        )
    return audit
