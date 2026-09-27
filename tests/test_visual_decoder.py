import struct
import zlib

import pytest
import torch

from localagent.data.visual import decode_png_rgb


def _png(width: int, height: int, channels: int, rows: list[bytes]) -> bytes:
    color_type = 2 if channels == 3 else 6
    raw = b"".join(b"\x00" + row for row in rows)

    def chunk(kind: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + kind
            + body
            + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def test_decode_png_rgb_and_drop_alpha() -> None:
    payload = _png(2, 1, 4, [bytes([255, 0, 0, 255, 0, 128, 255, 64])])
    image = decode_png_rgb(payload)
    assert image.shape == (3, 1, 2)
    assert image.dtype == torch.float32
    assert torch.allclose(image[:, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(image[:, 0, 1], torch.tensor([0.0, 128 / 255, 1.0]))


def test_decode_png_rejects_unsupported_format() -> None:
    with pytest.raises(ValueError, match="signature"):
        decode_png_rgb(b"not-a-png")
