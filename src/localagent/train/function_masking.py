"""Deterministic schema-preserving tool-name masking for SFT.

ToolSandbox-style augmentations deliberately make the function name less informative while
leaving the callable schema and arguments intact.  This module creates a reproducible augmented
conversation stream: every selected source row is retained, followed by one or more copies whose
tool names are replaced by collision-free opaque names.  The held-out stream is never transformed.

The transform is intentionally limited to tool names.  Descriptions, JSON-schema parameters,
argument values, user text, tool responses, and message order are preserved byte-for-byte.  This
keeps the operation useful for measuring schema/function-name robustness without pretending that
it teaches visual grounding or a native environment protocol.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from localagent.data.conversation_artifact import conversation_semantic_sha256
from localagent.data.schema import Conversation, Message, Role, ToolCall, ToolSpec

FUNCTION_MASKING_KIND = "schema_preserving_tool_name_mask_v1"
FUNCTION_MASKING_ALGORITHM = "sha256_row_seeded_selection_and_aliases_v1"
_PREFIX_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")
_CONFIG_KEYS = frozenset({"enabled", "mask_fraction", "variants", "name_prefix"})


def _disabled_audit(*, seed: int, source_rows: int) -> dict[str, Any]:
    return {
        "algorithm": FUNCTION_MASKING_ALGORITHM,
        "enabled": False,
        "kind": FUNCTION_MASKING_KIND,
        "mask_fraction": 0.0,
        "masked_rows": 0,
        "mapping_sha256": hashlib.sha256(b"").hexdigest(),
        "name_prefix": "fn_mask",
        "seed": seed,
        "source_rows": source_rows,
        "variants": 0,
        "output_rows": source_rows,
    }


def _resolve_config(config: object, *, seed: int) -> dict[str, Any]:
    if config is None or config is False:
        return {
            "enabled": False,
            "mask_fraction": 0.0,
            "variants": 0,
            "name_prefix": "fn_mask",
            "seed": seed,
        }
    if config is True:
        raw: Mapping[str, Any] = {}
    elif isinstance(config, Mapping):
        unknown = sorted(set(config) - _CONFIG_KEYS)
        if unknown:
            raise ValueError(f"function_masking has unknown keys: {unknown}")
        raw = config
    else:
        raise TypeError("data.function_masking must be boolean or a mapping")

    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise TypeError("function_masking.enabled must be boolean")
    fraction = raw.get("mask_fraction", 0.5 if enabled else 0.0)
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise TypeError("function_masking.mask_fraction must be a number")
    fraction = float(fraction)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("function_masking.mask_fraction must be finite in [0, 1]")
    variants = raw.get("variants", 1 if enabled and fraction > 0 else 0)
    if isinstance(variants, bool) or not isinstance(variants, int) or not 0 <= variants <= 4:
        raise ValueError("function_masking.variants must be an integer in [0, 4]")
    prefix = raw.get("name_prefix", "fn_mask")
    if not isinstance(prefix, str) or _PREFIX_RE.fullmatch(prefix) is None:
        raise ValueError(
            "function_masking.name_prefix must match ^[A-Za-z][A-Za-z0-9_]{0,31}$"
        )
    if not enabled:
        fraction = 0.0
        variants = 0
    return {
        "enabled": enabled,
        "mask_fraction": fraction,
        "variants": variants,
        "name_prefix": prefix,
        "seed": seed,
    }


def _row_selected(*, seed: int, index: int, conversation: Conversation, fraction: float) -> bool:
    if fraction <= 0.0:
        return False
    digest = hashlib.sha256(
        f"{seed}:{index}:{conversation_semantic_sha256(conversation)}".encode("ascii")
    ).digest()
    threshold = int(fraction * (1 << 256))
    return int.from_bytes(digest, "big") < threshold


def _alias_mapping(
    conversation: Conversation,
    *,
    seed: int,
    row_index: int,
    variant: int,
    prefix: str,
) -> dict[str, str]:
    names = [tool.name for tool in conversation.tools]
    if len(set(names)) != len(names):
        raise ValueError(f"conversation {row_index} has duplicate tool names")
    if not names:
        return {}
    digest = hashlib.sha256(
        f"{seed}:{row_index}:{variant}:{conversation_semantic_sha256(conversation)}".encode(
            "ascii"
        )
    ).hexdigest()[:12]
    return {
        name: f"{prefix}_{digest}_{position:03d}"
        for position, name in enumerate(sorted(names))
    }


def _masked_copy(
    conversation: Conversation,
    mapping: Mapping[str, str],
    *,
    row_index: int,
    variant: int,
) -> Conversation:
    tool_names = set(mapping)
    tools = [
        ToolSpec(
            name=mapping[tool.name],
            description=tool.description,
            parameters=copy.deepcopy(tool.parameters),
        )
        for tool in conversation.tools
    ]
    messages: list[Message] = []
    for message in conversation.messages:
        calls = []
        for call in message.tool_calls:
            if call.name not in tool_names:
                raise ValueError(
                    f"conversation {row_index} has a tool call without a catalog entry: "
                    f"{call.name!r}"
                )
            calls.append(
                ToolCall(name=mapping[call.name], arguments=copy.deepcopy(call.arguments))
            )
        messages.append(
            Message(
                role=Role(message.role),
                content=message.content,
                tool_calls=calls,
                tool_response=message.tool_response,
            )
        )
    meta = dict(conversation.meta)
    meta["function_masking"] = {
        "kind": FUNCTION_MASKING_KIND,
        "row_index": row_index,
        "variant": variant,
        "mapping_sha256": hashlib.sha256(
            json.dumps(dict(sorted(mapping.items())), sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
    }
    return Conversation(messages=messages, tools=tools, meta=meta)


def augment_conversations(
    conversations: Sequence[Conversation],
    config: object = False,
    *,
    seed: int = 0,
) -> tuple[list[Conversation], list[int], dict[str, Any]]:
    """Return originals plus deterministic masked variants and source-index metadata.

    The returned source indices align one-for-one with the returned conversations.  They let SFT
    preserve the original configured artifact path even when an augmented row is inserted.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("function masking seed must be an integer")
    resolved = _resolve_config(config, seed=seed)
    if not resolved["enabled"] or resolved["variants"] == 0:
        return list(conversations), list(range(len(conversations))), _disabled_audit(
            seed=seed,
            source_rows=len(conversations),
        )

    output: list[Conversation] = []
    source_indices: list[int] = []
    mapping_records: list[dict[str, str]] = []
    masked_rows = 0
    for index, conversation in enumerate(conversations):
        output.append(conversation)
        source_indices.append(index)
        if not conversation.tools or not _row_selected(
            seed=seed,
            index=index,
            conversation=conversation,
            fraction=resolved["mask_fraction"],
        ):
            continue
        masked_rows += 1
        for variant in range(1, resolved["variants"] + 1):
            mapping = _alias_mapping(
                conversation,
                seed=seed,
                row_index=index,
                variant=variant,
                prefix=resolved["name_prefix"],
            )
            output.append(_masked_copy(conversation, mapping, row_index=index, variant=variant))
            source_indices.append(index)
            mapping_records.append(dict(sorted(mapping.items())))
    mapping_sha = hashlib.sha256(
        json.dumps(mapping_records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    audit = {
        "algorithm": FUNCTION_MASKING_ALGORITHM,
        "enabled": True,
        "kind": FUNCTION_MASKING_KIND,
        "mask_fraction": resolved["mask_fraction"],
        "masked_rows": masked_rows,
        "mapping_sha256": mapping_sha,
        "name_prefix": resolved["name_prefix"],
        "seed": seed,
        "source_rows": len(conversations),
        "variants": resolved["variants"],
        "output_rows": len(output),
    }
    return output, source_indices, audit


__all__ = [
    "FUNCTION_MASKING_ALGORITHM",
    "FUNCTION_MASKING_KIND",
    "augment_conversations",
]
