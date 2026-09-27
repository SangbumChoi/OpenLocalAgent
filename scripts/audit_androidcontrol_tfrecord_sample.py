#!/usr/bin/env python3
"""Audit one bounded official AndroidControl TFRecord episode without retaining screenshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import urllib.request
import zlib
from pathlib import Path
from typing import Any, Iterable

OBJECT_SIZE = 2_492_987_829
OBJECT_MD5_BASE64 = "QuhP+X5aiP+iD5amlNM6tA=="
OBJECT_URL = "https://storage.googleapis.com/gresearch/android_control/android_control-00000-of-00020"
SPLITS_URL = "https://storage.googleapis.com/gresearch/android_control/splits.json"
ORIGINAL_REPO = "https://github.com/google-research/google-research/tree/master/android_control"
GCS_BROWSER = "https://console.cloud.google.com/storage/browser/gresearch/android_control"


def _varint(data: bytes, index: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if index >= len(data):
            raise ValueError("truncated protobuf varint")
        byte = data[index]
        index += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, index
        shift += 7
        if shift > 70:
            raise ValueError("protobuf varint is too long")


def _fields(data: bytes) -> Iterable[tuple[int, int, int | bytes]]:
    index = 0
    while index < len(data):
        tag, index = _varint(data, index)
        number, wire = tag >> 3, tag & 7
        if wire == 0:
            value, index = _varint(data, index)
            yield number, wire, value
        elif wire == 2:
            length, index = _varint(data, index)
            end = index + length
            if end > len(data):
                raise ValueError("truncated protobuf bytes field")
            yield number, wire, data[index:end]
            index = end
        elif wire == 1:
            end = index + 8
            yield number, wire, data[index:end]
            index = end
        elif wire == 5:
            end = index + 4
            yield number, wire, data[index:end]
            index = end
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")


def _first_tfrecord(gzip_prefix: bytes) -> bytes:
    decompressed = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(gzip_prefix)
    if len(decompressed) < 12:
        raise ValueError("gzip prefix did not contain a TFRecord header")
    length = struct.unpack("<Q", decompressed[:8])[0]
    end = 12 + length
    if end > len(decompressed):
        raise ValueError("bounded gzip prefix ended before the first TFRecord")
    return decompressed[12 : 12 + length]


def _bytes_list(feature: bytes) -> list[bytes]:
    values: list[bytes] = []
    for number, wire, value in _fields(feature):
        if number != 1 or wire != 2:
            continue
        assert isinstance(value, bytes)
        values.extend(
            item
            for item_number, item_wire, item in _fields(value)
            if item_number == 1 and item_wire == 2 and isinstance(item, bytes)
        )
    return values


def _int64_list(feature: bytes) -> list[int]:
    values: list[int] = []
    for number, wire, value in _fields(feature):
        if number != 3 or wire != 2:
            continue
        assert isinstance(value, bytes)
        for item_number, item_wire, item in _fields(value):
            if item_number != 1:
                continue
            if item_wire == 0:
                values.append(int(item))
            elif item_wire == 2 and isinstance(item, bytes):
                index = 0
                while index < len(item):
                    parsed, index = _varint(item, index)
                    values.append(parsed)
    return values


def _feature_map(example: bytes) -> dict[str, bytes]:
    features = next(value for number, wire, value in _fields(example) if number == 1 and wire == 2)
    result: dict[str, bytes] = {}
    assert isinstance(features, bytes)
    for number, wire, entry in _fields(features):
        if number != 1 or wire != 2:
            continue
        assert isinstance(entry, bytes)
        key: str | None = None
        value: bytes | None = None
        for entry_number, entry_wire, entry_value in _fields(entry):
            if entry_number == 1 and entry_wire == 2 and isinstance(entry_value, bytes):
                key = entry_value.decode("utf-8")
            elif entry_number == 2 and entry_wire == 2 and isinstance(entry_value, bytes):
                value = entry_value
        if key is not None and value is not None:
            result[key] = value
    return result


def _download(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def audit(*, prefix_bytes: int = 10 * 1024 * 1024) -> dict[str, Any]:
    if prefix_bytes < 1:
        raise ValueError("prefix_bytes must be positive")
    raw_prefix = _download(OBJECT_URL, 0, prefix_bytes - 1)
    record = _first_tfrecord(raw_prefix)
    features = _feature_map(record)
    screenshots = _bytes_list(features["screenshots"])
    trees = _bytes_list(features["accessibility_trees"])
    instructions = _bytes_list(features["step_instructions"])
    actions = _bytes_list(features["actions"])
    episode_id_values = _int64_list(features["episode_id"])
    widths = _int64_list(features["screenshot_widths"])
    heights = _int64_list(features["screenshot_heights"])
    split_manifest = json.loads(urllib.request.urlopen(SPLITS_URL, timeout=30).read())
    episode_id = episode_id_values[0] if episode_id_values else None
    split = "unknown"
    if episode_id is not None:
        split = next((name for name, ids in split_manifest.items() if episode_id in ids), "unknown")
    action_types = []
    for action in actions:
        try:
            action_types.append(json.loads(action.decode("utf-8")).get("action_type"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            action_types.append("<unparseable>")
    return {
        "kind": "localagent_androidcontrol_official_tfrecord_visual_sample",
        "schema_version": 1,
        "source": {
            "dataset": "AndroidControl",
            "original_repository": ORIGINAL_REPO,
            "gcs_browser": GCS_BROWSER,
            "object_url": OBJECT_URL,
            "object_size_bytes": OBJECT_SIZE,
            "object_md5_base64": OBJECT_MD5_BASE64,
            "split_manifest_url": SPLITS_URL,
            "shard": "android_control-00000-of-00020",
            "compression": "gzip",
            "record_format": "TFRecord tf.train.Example",
        },
        "bounded_fetch": {
            "range_start": 0,
            "range_end_exclusive": len(raw_prefix),
            "prefix_bytes": len(raw_prefix),
            "prefix_sha256": hashlib.sha256(raw_prefix).hexdigest(),
            "first_record_bytes": len(record),
            "first_record_sha256": hashlib.sha256(record).hexdigest(),
        },
        "episode": {
            "episode_id": episode_id,
            "split": split,
            "goal": _bytes_list(features["goal"])[0].decode("utf-8", errors="replace"),
            "screenshot_count": len(screenshots),
            "screenshot_total_bytes": sum(len(item) for item in screenshots),
            "screenshot_bytes_sha256": hashlib.sha256(b"".join(screenshots)).hexdigest(),
            "screenshot_widths": widths,
            "screenshot_heights": heights,
            "accessibility_tree_count": len(trees),
            "accessibility_tree_total_bytes": sum(len(item) for item in trees),
            "step_instruction_count": len(instructions),
            "action_count": len(actions),
            "action_types": action_types,
        },
        "pipeline_boundary": {
            "official_visual_bytes_present": True,
            "current_localagent_projection_consumes_screenshots": False,
            "current_androidcontrol_receipts_visual_input_omitted": True,
            "training_admission": "provenance_only_until_a_vision_encoder_and_visual_eval_are_bound",
        },
        "claim_boundary": (
            "This receipt proves that the original public AndroidControl TFRecord source contains "
            "screenshot PNG bytes, accessibility trees, instructions, and actions for a train-split "
            "episode. It does not train or evaluate a vision model, and it does not claim Android "
            "emulator, AndroidWorld, MobileGym, or WebGPU screenshot success."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-bytes", type=int, default=10 * 1024 * 1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(prefix_bytes=args.prefix_bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["episode"], sort_keys=True))


if __name__ == "__main__":
    main()
