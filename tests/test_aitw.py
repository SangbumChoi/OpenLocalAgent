import gzip
import struct

from localagent.data.aitw import episode_to_intermediate, iter_aitw_episodes
from localagent.data.androidcontrol import _masked_crc32c


def _varint(value: int) -> bytes:
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _field(number: int, wire: int, value: bytes | int) -> bytes:
    prefix = _varint((number << 3) | wire)
    if wire == 0:
        return prefix + _varint(int(value))
    payload = bytes(value)
    return prefix + _varint(len(payload)) + payload


def _feature_bytes(value: bytes) -> bytes:
    return _field(1, 2, _field(1, 2, value))


def _feature_int(value: int) -> bytes:
    return _field(3, 2, _field(1, 0, value))


def _feature_float(values: list[float]) -> bytes:
    packed = b"".join(struct.pack("<f", value) for value in values)
    return _field(2, 2, _field(1, 2, packed))


def _example(step: int, action_type: int) -> bytes:
    features = {
        "episode_id": _feature_bytes(b"123"),
        "step_id": _feature_int(step),
        "episode_length": _feature_int(2),
        "goal_info": _feature_bytes(b"Open the browser"),
        "current_activity": _feature_bytes(b"Chrome"),
        "image/width": _feature_int(540),
        "image/height": _feature_int(1080),
        "image/ui_annotations_text": _feature_bytes(b"Search"),
        "image/ui_annotations_ui_types": _feature_bytes(b"TEXT"),
        "image/ui_annotations_positions": _feature_float([0.1, 0.2, 0.1, 0.4]),
        "results/action_type": _feature_int(action_type),
    }
    if action_type == 4:
        features["results/yx_touch"] = _feature_float([0.15, 0.3])
        features["results/yx_lift"] = _feature_float([0.15, 0.3])
    features_message = b"".join(
        _field(
            1,
            2,
            _field(1, 2, key.encode("utf-8")) + _field(2, 2, value),
        )
        for key, value in features.items()
    )
    return _field(1, 2, features_message)


def _write_tfrecord(path, payloads: list[bytes]) -> None:
    record_bytes = bytearray()
    for payload in payloads:
        length = struct.pack("<Q", len(payload))
        record_bytes.extend(length)
        record_bytes.extend(struct.pack("<I", _masked_crc32c(length)))
        record_bytes.extend(payload)
        record_bytes.extend(struct.pack("<I", _masked_crc32c(payload)))
    path.write_bytes(gzip.compress(bytes(record_bytes)))


def test_aitw_reconstructs_step_records_and_maps_dual_point(tmp_path) -> None:
    path = tmp_path / "aitw.tfrecord.gz"
    _write_tfrecord(path, [_example(0, 4), _example(1, 10)])
    episodes = list(iter_aitw_episodes(path))
    assert len(episodes) == 1
    row = episode_to_intermediate(episodes[0], source_revision="rev", split="train")
    assert row["quality"]["source_executable_steps"] == 1
    call = row["messages"][2]["tool_calls"][0]
    assert call["name"] == "mobile_click"
    assert call["arguments"] == {"x": 162.0, "y": 162.0}
