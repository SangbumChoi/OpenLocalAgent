"""Dependency-free PNG decoding for bounded public screenshot experiments.

This intentionally supports the common 8-bit RGB/RGBA, non-interlaced PNGs emitted by the
AndroidControl and AgentNet archives.  It returns a float tensor suitable for the optional visual
patch bridge and rejects formats that would otherwise be silently misinterpreted.
"""

from __future__ import annotations

import struct
import zlib

import torch


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _paeth(a: int, b: int, c: int) -> int:
    prediction = a + b - c
    pa, pb, pc = abs(prediction - a), abs(prediction - b), abs(prediction - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def decode_png_rgb(payload: bytes) -> torch.Tensor:
    """Decode an 8-bit non-interlaced RGB/RGBA PNG into ``[3, H, W]`` float32 pixels."""

    if not payload.startswith(_PNG_SIGNATURE):
        raise ValueError("PNG signature is missing")
    offset = len(_PNG_SIGNATURE)
    width = height = bit_depth = color_type = interlace = None
    compressed = bytearray()
    while offset + 12 <= len(payload):
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        chunk_start = offset + 8
        chunk_end = chunk_start + length
        if chunk_end + 4 > len(payload):
            raise ValueError("truncated PNG chunk")
        chunk_type = payload[offset + 4 : offset + 8]
        chunk = payload[chunk_start:chunk_end]
        offset = chunk_end + 4  # CRC is validated by the archive/source, not decoded here.
        if chunk_type == b"IHDR":
            if len(chunk) != 13:
                raise ValueError("invalid PNG IHDR")
            width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", chunk
            )
            if compression != 0 or filtering != 0 or interlace != 0:
                raise ValueError("only standard non-interlaced PNGs are supported")
            if bit_depth != 8 or color_type not in {2, 6}:
                raise ValueError("only 8-bit RGB/RGBA PNGs are supported")
        elif chunk_type == b"IDAT":
            compressed.extend(chunk)
        elif chunk_type == b"IEND":
            break
    if width is None or height is None or not compressed:
        raise ValueError("PNG is missing IHDR or IDAT")
    channels = 3 if color_type == 2 else 4
    row_bytes = width * channels
    decoded = zlib.decompress(bytes(compressed))
    expected = height * (row_bytes + 1)
    if len(decoded) != expected:
        raise ValueError("PNG scanline payload length mismatch")
    rows = bytearray(height * row_bytes)
    cursor = 0
    for row in range(height):
        filter_type = decoded[cursor]
        cursor += 1
        source = decoded[cursor : cursor + row_bytes]
        cursor += row_bytes
        destination = memoryview(rows)[row * row_bytes : (row + 1) * row_bytes]
        previous = memoryview(rows)[(row - 1) * row_bytes : row * row_bytes] if row else None
        for index, value in enumerate(source):
            left = destination[index - channels] if index >= channels else 0
            up = previous[index] if previous is not None else 0
            upper_left = previous[index - channels] if previous is not None and index >= channels else 0
            if filter_type == 0:
                result = value
            elif filter_type == 1:
                result = value + left
            elif filter_type == 2:
                result = value + up
            elif filter_type == 3:
                result = value + ((left + up) // 2)
            elif filter_type == 4:
                result = value + _paeth(left, up, upper_left)
            else:
                raise ValueError(f"unsupported PNG filter type {filter_type}")
            destination[index] = result & 0xFF
    values = torch.frombuffer(rows, dtype=torch.uint8).reshape(height, width, channels)
    return values[..., :3].permute(2, 0, 1).contiguous().float().div_(255.0)


def load_png_rgb(path: str) -> torch.Tensor:
    """Read and decode one PNG file without adding a Pillow/torchvision dependency."""

    with open(path, "rb") as handle:
        return decode_png_rgb(handle.read())
