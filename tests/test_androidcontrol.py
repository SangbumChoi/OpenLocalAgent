import gzip
import struct

from localagent.data.androidcontrol import (
    _masked_crc32c,
    compact_accessibility_tree,
    iter_gzip_tfrecords,
)


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


def test_compact_accessibility_tree_is_deterministic() -> None:
    rect = b"".join(_field(number, 0, value) for number, value in enumerate((1, 2, 100, 200), 1))
    node = b"".join(
        (
            _field(1, 0, 7),
            _field(2, 2, rect),
            _field(3, 2, b"android.widget.Button"),
            _field(7, 2, b"Send"),
            _field(16, 0, 1),
            _field(23, 0, 1),
            _field(27, 0, 2),
        )
    )
    tree = _field(1, 2, node)
    forest = _field(1, 2, _field(5, 2, b"Mail") + _field(11, 2, tree))
    text = compact_accessibility_tree(forest)
    assert "WINDOW[0] title='Mail'" in text
    assert "class='android.widget.Button'" in text
    assert "labels=['Send']" in text
    assert "enabled,visible" in text
    assert "bounds=(1, 2, 100, 200)" in text


def test_iter_gzip_tfrecords_checks_crc(tmp_path) -> None:
    payload = b"androidcontrol-test"
    header = struct.pack("<Q", len(payload)) + struct.pack("<I", _masked_crc32c(struct.pack("<Q", len(payload))))
    record = header + payload + struct.pack("<I", _masked_crc32c(payload))
    path = tmp_path / "sample.tfrecord.gz"
    path.write_bytes(gzip.compress(record))
    assert list(iter_gzip_tfrecords(path)) == [payload]
