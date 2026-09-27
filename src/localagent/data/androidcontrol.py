"""Dependency-light AndroidControl TFRecord and accessibility-tree decoder.

The official AndroidControl payload is a set of GZIP-compressed ``tf.train.Example`` TFRecord
shards.  TensorFlow is intentionally not a model dependency in LocalAgent, so this module parses
the small protobuf wire subset used by the published records and emits the intermediate mobile
rows consumed by :func:`localagent.data.realistic_adapters.normalize_mobile_row`.

The decoder is streamable and can operate on a bounded HTTP range.  A range is never treated as a
complete source snapshot: callers must record the full object size/MD5, byte range, range SHA-256,
and ``allow_truncated=True`` in their acquisition manifest.
"""

from __future__ import annotations

import json
import struct
import zlib
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from localagent.data.realistic_adapters import normalize_mobile_row

_CRC32C_POLY = 0x82F63B78
_CRC32C_TABLE = []
for _index in range(256):
    _value = _index
    for _ in range(8):
        _value = (_value >> 1) ^ _CRC32C_POLY if _value & 1 else _value >> 1
    _CRC32C_TABLE.append(_value)


def _crc32c(payload: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in payload:
        crc = _CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def _masked_crc32c(payload: bytes) -> int:
    crc = _crc32c(payload)
    return ((crc >> 15) | ((crc << 17) & 0xFFFFFFFF)) + 0xA282EAD8 & 0xFFFFFFFF


def _read_varint(payload: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(payload):
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
        if shift > 70:
            raise ValueError("protobuf varint is too long")
    raise ValueError("truncated protobuf varint")


def _signed_int32(value: int) -> int:
    return value - (1 << 32) if value >= (1 << 31) else value


def _fields(payload: bytes) -> Iterator[tuple[int, int, int | bytes]]:
    """Yield ``(field_number, wire_type, value)`` for a protobuf message."""

    offset = 0
    while offset < len(payload):
        key, offset = _read_varint(payload, offset)
        field_number, wire_type = key >> 3, key & 0x07
        if field_number < 1:
            raise ValueError("protobuf field number must be positive")
        if wire_type == 0:
            value, offset = _read_varint(payload, offset)
            yield field_number, wire_type, value
        elif wire_type == 1:
            end = offset + 8
            if end > len(payload):
                raise ValueError("truncated fixed64 protobuf field")
            yield field_number, wire_type, payload[offset:end]
            offset = end
        elif wire_type == 2:
            size, offset = _read_varint(payload, offset)
            end = offset + size
            if end > len(payload):
                raise ValueError("truncated length-delimited protobuf field")
            yield field_number, wire_type, payload[offset:end]
            offset = end
        elif wire_type == 5:
            end = offset + 4
            if end > len(payload):
                raise ValueError("truncated fixed32 protobuf field")
            yield field_number, wire_type, payload[offset:end]
            offset = end
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type}")


def _packed_varints(payload: bytes) -> list[int]:
    values: list[int] = []
    offset = 0
    while offset < len(payload):
        value, offset = _read_varint(payload, offset)
        values.append(value)
    return values


def _feature_values(feature: bytes) -> tuple[list[bytes], list[int], list[float]]:
    byte_values: list[bytes] = []
    int_values: list[int] = []
    float_values: list[float] = []
    for field, wire_type, value in _fields(feature):
        if wire_type != 2 or not isinstance(value, bytes):
            continue
        if field == 1:  # BytesList.value, repeated bytes.
            byte_values.extend(
                nested_value
                for nested_field, nested_wire, nested_value in _fields(value)
                if nested_field == 1 and nested_wire == 2 and isinstance(nested_value, bytes)
            )
        elif field == 3:  # Int64List.value, packed or unpacked int64.
            for nested_field, nested_wire, nested_value in _fields(value):
                if nested_field != 1:
                    continue
                if nested_wire == 0 and isinstance(nested_value, int):
                    int_values.append(nested_value)
                elif nested_wire == 2 and isinstance(nested_value, bytes):
                    int_values.extend(_packed_varints(nested_value))
        elif field == 2:  # FloatList.value, packed or unpacked float32.
            for nested_field, nested_wire, nested_value in _fields(value):
                if nested_field != 1 or not isinstance(nested_value, bytes):
                    continue
                if nested_wire == 5 and len(nested_value) == 4:
                    float_values.append(struct.unpack("<f", nested_value)[0])
                elif nested_wire == 2 and len(nested_value) % 4 == 0:
                    float_values.extend(
                        struct.unpack("<" + "f" * (len(nested_value) // 4), nested_value)
                    )
    return byte_values, int_values, float_values


def _example_features(payload: bytes) -> dict[str, tuple[list[bytes], list[int], list[float]]]:
    result: dict[str, tuple[list[bytes], list[int], list[float]]] = {}
    for field, wire_type, value in _fields(payload):
        if field != 1 or wire_type != 2 or not isinstance(value, bytes):
            continue
        for map_field, map_wire, map_value in _fields(value):
            if map_field != 1 or map_wire != 2 or not isinstance(map_value, bytes):
                continue
            key: str | None = None
            feature: bytes | None = None
            for entry_field, entry_wire, entry_value in _fields(map_value):
                if entry_field == 1 and entry_wire == 2 and isinstance(entry_value, bytes):
                    key = entry_value.decode("utf-8", errors="strict")
                elif entry_field == 2 and entry_wire == 2 and isinstance(entry_value, bytes):
                    feature = entry_value
            if key is not None and feature is not None:
                result[key] = _feature_values(feature)
    return result


def _first_bytes(
    features: Mapping[str, tuple[list[bytes], list[int], list[float]]], key: str
) -> bytes:
    values = features.get(key, ([], [], []))[0]
    if not values:
        raise ValueError(f"AndroidControl example is missing bytes feature {key!r}")
    return values[0]


def _bytes_list(
    features: Mapping[str, tuple[list[bytes], list[int], list[float]]], key: str
) -> list[bytes]:
    return list(features.get(key, ([], [], []))[0])


def _int_list(
    features: Mapping[str, tuple[list[bytes], list[int], list[float]]], key: str
) -> list[int]:
    return list(features.get(key, ([], [], []))[1])


def _float_list(
    features: Mapping[str, tuple[list[bytes], list[int], list[float]]], key: str
) -> list[float]:
    return list(features.get(key, ([], [], []))[2])


def _decode_json_bytes(value: bytes, *, label: str) -> Any:
    try:
        return json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error


def _rect(payload: bytes) -> tuple[int, int, int, int] | None:
    values: dict[int, int] = {}
    for field, wire_type, value in _fields(payload):
        if wire_type == 0 and isinstance(value, int) and field in {1, 2, 3, 4}:
            values[field] = _signed_int32(value)
    if len(values) != 4:
        return None
    return tuple(values[index] for index in range(1, 5))  # type: ignore[return-value]


def _node(payload: bytes) -> dict[str, Any]:
    node: dict[str, Any] = {"flags": []}
    for field, wire_type, value in _fields(payload):
        if field == 1 and wire_type == 0 and isinstance(value, int):
            node["id"] = _signed_int32(value)
        elif field == 2 and wire_type == 2 and isinstance(value, bytes):
            node["bounds"] = _rect(value)
        elif field in {3, 4, 5, 6, 7, 10, 31} and wire_type == 2 and isinstance(value, bytes):
            node[{3: "class", 4: "content", 5: "hint", 6: "package", 7: "text", 10: "view_id", 31: "tooltip"}[field]] = value.decode("utf-8", errors="replace")
        elif field in {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23} and wire_type == 0 and value:
            node["flags"].append(
                {
                    12: "checkable",
                    13: "checked",
                    14: "clickable",
                    15: "editable",
                    16: "enabled",
                    17: "focusable",
                    18: "focused",
                    19: "long_clickable",
                    20: "password",
                    21: "scrollable",
                    22: "selected",
                    23: "visible",
                }[field]
            )
        elif field == 25:
            if wire_type == 2 and isinstance(value, bytes):
                node["children"] = [_signed_int32(item) for item in _packed_varints(value)]
            elif wire_type == 0 and isinstance(value, int):
                node.setdefault("children", []).append(_signed_int32(value))
        elif field == 27 and wire_type == 0 and isinstance(value, int):
            node["depth"] = _signed_int32(value)
    return node


def compact_accessibility_tree(payload: bytes, *, max_nodes: int = 160, max_chars: int = 12_000) -> str:
    """Decode node labels/roles/bounds into a deterministic text observation."""

    windows: list[dict[str, Any]] = []
    for field, wire_type, value in _fields(payload):
        if field != 1 or wire_type != 2 or not isinstance(value, bytes):
            continue
        window: dict[str, Any] = {"nodes": []}
        for window_field, window_wire, window_value in _fields(value):
            if window_field == 5 and window_wire == 2 and isinstance(window_value, bytes):
                window["title"] = window_value.decode("utf-8", errors="replace")
            elif window_field == 11 and window_wire == 2 and isinstance(window_value, bytes):
                for tree_field, tree_wire, tree_value in _fields(window_value):
                    if tree_field == 1 and tree_wire == 2 and isinstance(tree_value, bytes):
                        window["nodes"].append(_node(tree_value))
        windows.append(window)

    lines: list[str] = []
    for window_index, window in enumerate(windows):
        title = str(window.get("title", ""))
        lines.append(f"WINDOW[{window_index}] title={title!r}")
        nodes = sorted(
            window["nodes"],
            key=lambda item: (int(item.get("depth", 0)), int(item.get("id", 0))),
        )
        for node in nodes[:max_nodes]:
            labels = [
                str(node[key]).strip()
                for key in ("text", "content", "hint", "view_id")
                if node.get(key)
            ]
            flags = ",".join(sorted(set(node.get("flags", []))))
            if not labels and not flags:
                continue
            bounds = node.get("bounds")
            bound_text = f" bounds={bounds}" if bounds else ""
            lines.append(
                f"NODE depth={node.get('depth', 0)} id={node.get('id', '?')} "
                f"class={node.get('class', '')!r} labels={labels!r} flags={flags!r}{bound_text}"
            )
    text = "\n".join(lines)
    return text[:max_chars]


def iter_gzip_tfrecords(
    path: str | Path,
    *,
    allow_truncated: bool = False,
    verify_crc: bool = True,
    chunk_size: int = 8 * 1024 * 1024,
) -> Iterator[bytes]:
    """Yield complete TFRecord payloads from a GZIP shard or bounded gzip range."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    buffer = b""
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            buffer += decompressor.decompress(chunk)
            while len(buffer) >= 16:
                record_length = struct.unpack_from("<Q", buffer, 0)[0]
                total = 16 + record_length
                if total > len(buffer):
                    break
                if verify_crc:
                    expected_length_crc = struct.unpack_from("<I", buffer, 8)[0]
                    expected_payload_crc = struct.unpack_from("<I", buffer, 12 + record_length)[0]
                    if _masked_crc32c(buffer[:8]) != expected_length_crc:
                        raise ValueError("TFRecord length CRC mismatch")
                    if _masked_crc32c(buffer[12 : 12 + record_length]) != expected_payload_crc:
                        raise ValueError("TFRecord payload CRC mismatch")
                yield buffer[12 : 12 + record_length]
                buffer = buffer[total:]
    if not allow_truncated and not decompressor.eof:
        raise ValueError("gzip stream ended before its footer; pass allow_truncated for a range")
    if buffer and not allow_truncated:
        raise ValueError("gzip stream contains an incomplete TFRecord")


def example_to_intermediate(
    payload: bytes,
    *,
    source_revision: str,
    source_record_index: int,
    split: str,
) -> dict[str, Any]:
    """Convert one official Example payload into a text-first localagent_v1 row."""

    features = _example_features(payload)
    episode_values = _int_list(features, "episode_id")
    if not episode_values:
        raise ValueError("AndroidControl example has no episode_id")
    episode_id = int(episode_values[0])
    goal = _first_bytes(features, "goal").decode("utf-8", errors="strict")
    actions_raw = _bytes_list(features, "actions")
    instructions_raw = _bytes_list(features, "step_instructions")
    trees_raw = _bytes_list(features, "accessibility_trees")
    if len(actions_raw) != len(trees_raw) - 1:
        raise ValueError(
            f"episode {episode_id} action/tree mismatch: {len(actions_raw)} actions vs "
            f"{len(trees_raw)} observations"
        )
    steps: list[dict[str, Any]] = []
    for index, action_bytes in enumerate(actions_raw):
        action = _decode_json_bytes(action_bytes, label=f"episode {episode_id} action {index}")
        if not isinstance(action, Mapping):
            raise ValueError(f"episode {episode_id} action {index} must be an object")
        instruction = (
            instructions_raw[index].decode("utf-8", errors="strict")
            if index < len(instructions_raw)
            else goal
        )
        steps.append(
            {
                "instruction": instruction,
                "accessibility_tree": compact_accessibility_tree(trees_raw[index]),
                "next_observation": compact_accessibility_tree(trees_raw[index + 1]),
                "action": dict(action),
            }
        )
    row = normalize_mobile_row(
        {"record_id": f"androidcontrol:{episode_id}", "goal": goal, "steps": steps},
        family="androidcontrol",
        source_revision=source_revision,
    )
    row["quality"].update(
        {
            "source_record_index": source_record_index,
            "source_episode_id": episode_id,
            "source_split": split,
            "accessibility_projection": "android_env_proto_text_v1",
        }
    )
    return row
