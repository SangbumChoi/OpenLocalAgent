"""Dataset-specific text adapters for streaming pretraining sources.

Adapters normalize only source-format noise. They deliberately do not rewrite content, strip code
structure, or infer licenses; admission and provenance remain the corpus pipeline's responsibility.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping


def _plain(text: str, row: Mapping[str, object]) -> str:
    del row
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _prose(text: str, row: Mapping[str, object]) -> str:
    del row
    return re.sub(r"[\t\f\v ]+", " ", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def _code(text: str, row: Mapping[str, object]) -> str:
    del row
    # Preserve indentation and blank lines; only normalize line endings and trailing file noise.
    return text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")


def _html(text: str, row: Mapping[str, object]) -> str:
    del row
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


_ADAPTERS: dict[str, Callable[[str, Mapping[str, object]], str]] = {
    "plain_text": _plain,
    "web_prose": _prose,
    "math_prose": _prose,
    "python_code": _code,
    "html_text": _html,
}


def adapter_names() -> tuple[str, ...]:
    """Return registered adapter names in stable order."""

    return tuple(_ADAPTERS)


def adapt_source_text(adapter: str, text: str, row: Mapping[str, object]) -> str:
    """Normalize one source row, rejecting misspelled/unregistered adapters."""

    try:
        transform = _ADAPTERS[adapter]
    except KeyError as exc:
        raise ValueError(f"unknown pretraining source adapter {adapter!r}") from exc
    return transform(text, row)
