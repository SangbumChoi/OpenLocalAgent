"""FineMath text preparation for direct online pretraining batches."""

from __future__ import annotations

from collections.abc import Mapping


def preprocess(text: str, row: Mapping[str, object]) -> str:
    """Keep equations and line breaks; normalize transport whitespace only."""
    del row
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "").strip()
