"""On-demand, bounded-memory batches from revision-pinned Hugging Face streams."""

from __future__ import annotations

import importlib
import gzip
import json
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from typing import Any
from urllib.parse import quote

import torch

from openlocalagent.data.hf_corpus import (
    _load_stream,
    _normalize_license,
    _selected_raw_files,
)
from openlocalagent.data.source_adapters import adapt_source_text


def _preprocessor(path: str | None) -> Callable[[str, Mapping[str, Any]], str] | None:
    if path is None:
        return None
    module_name, separator, function_name = path.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("preprocessor must be a module:function path")
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"preprocessor {path!r} is not callable")
    return function


def _online_stream(source: Mapping[str, Any], seed: int) -> Iterable[Mapping[str, Any]]:
    """Avoid Hub whole-shard caching for legacy CodeParrot gzip exports.

    This stream reads one deterministically selected public shard by HTTP as rows are consumed.
    The full-shard SHA cannot be verified until EOF, so this path is exploratory only; the
    audited corpus builder remains the path for a publishable, checksum-verified run.
    """
    raw = source.get("raw_stream")
    if not isinstance(raw, Mapping) or raw.get("backend") != "hf-jsonl-gzip-v1":
        return _load_stream(source, seed)
    selected = _selected_raw_files(source, seed)
    if not selected:
        raise RuntimeError(f"source {source['name']!r} selected no gzip shard")
    item = selected[0]

    def rows() -> Iterable[Mapping[str, Any]]:
        import requests

        url = (
            f"https://huggingface.co/datasets/{source['dataset']}/resolve/"
            f"{source['revision']}/{quote(item['path'], safe='/')}?download=1"
        )
        with requests.get(url, stream=True, timeout=(15, 90)) as response:
            response.raise_for_status()
            response.raw.decode_content = False
            with gzip.GzipFile(fileobj=response.raw) as handle:
                for line_number, payload in enumerate(handle, 1):
                    if len(payload) > 64 * 1024 * 1024:
                        raise RuntimeError(
                            f"{item['path']}:{line_number}: raw JSONL row exceeds 64 MiB"
                        )
                    try:
                        row = json.loads(payload)
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise RuntimeError(
                            f"{item['path']}:{line_number}: invalid JSONL row"
                        ) from exc
                    if not isinstance(row, dict):
                        raise RuntimeError(
                            f"{item['path']}:{line_number}: expected JSON object"
                        )
                    yield row

    return rows()


class OnlineSourceDataset:
    """Read rows only as sampled; retain at most one document plus one micro-batch.

    This intentionally has no exact-resume claim: Hugging Face iterator position is not part of
    the model checkpoint. Every new invocation starts a fresh pinned-revision stream.
    """

    def __init__(
        self,
        source: Mapping[str, Any],
        tokenizer: Any,
        *,
        seq_len: int,
        seed: int,
        shuffle_buffer: int = 128,
        preprocessor: str | None = None,
        stream_factory: Callable[[Mapping[str, Any], int], Iterable[Mapping[str, Any]]] = _online_stream,
    ) -> None:
        if seq_len < 1 or shuffle_buffer < 1:
            raise ValueError("seq_len and shuffle_buffer must be positive")
        self.source = dict(source)
        self.source["shuffle_buffer"] = shuffle_buffer
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.seed = seed
        self.stream_factory = stream_factory
        self.transform = _preprocessor(preprocessor)
        self._rows: Any = None
        self._iterator: Any = None
        self._tokens: deque[int] = deque()
        self.rows_seen = 0
        self.rows_used = 0
        self.rows_license_rejected = 0
        self.tokens_read = 0

    def _fill(self, minimum: int) -> None:
        if self._iterator is None:
            self._rows = self.stream_factory(self.source, self.seed)
            self._iterator = iter(self._rows)
        while len(self._tokens) < minimum:
            try:
                row = next(self._iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    f"online source {self.source['name']!r} exhausted before a complete batch"
                ) from exc
            self.rows_seen += 1
            license_field = self.source.get("license_field")
            if license_field:
                allowed = set(self.source.get("allowed_licenses", ()))
                label = _normalize_license(row.get(license_field))
                if not allowed or label not in allowed:
                    self.rows_license_rejected += 1
                    continue
            raw = row.get(self.source["text_field"])
            if not isinstance(raw, str):
                continue
            if self.transform is None:
                text = adapt_source_text(self.source["adapter"], raw, row)
            else:
                text = self.transform(raw, row)
            if not isinstance(text, str):
                raise TypeError("source preprocessor must return text")
            if len(text) < int(self.source.get("min_document_chars", 200)):
                continue
            max_chars = int(self.source.get("max_document_chars", 1_000_000))
            if len(text) > max_chars:
                continue
            tokens = self.tokenizer.encode(text, add_eos=True)
            if not tokens:
                continue
            self._tokens.extend(tokens)
            self.tokens_read += len(tokens)
            self.rows_used += 1

    def sample_batch(self, batch_size: int, rng: Any, device: Any):
        """Return sequential online windows in the trainer's ``sample_batch`` contract."""
        del rng
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        width = self.seq_len + 1
        self._fill(batch_size * width)
        rows = [[self._tokens.popleft() for _ in range(width)] for _ in range(batch_size)]
        batch = torch.tensor(rows, dtype=torch.long, device=device)
        return batch[:, :-1], batch[:, 1:]

    def stats(self) -> dict[str, int]:
        """Return acquisition counters without retaining raw text or credentials."""
        return {
            "rows_seen": self.rows_seen,
            "rows_used": self.rows_used,
            "rows_license_rejected": self.rows_license_rejected,
            "tokens_read": self.tokens_read,
            "buffered_tokens": len(self._tokens),
        }

    def close(self) -> None:
        """Release an underlying streaming iterator if it exposes ``close``."""
        resources = (self._iterator, self._rows)
        for index, resource in enumerate(resources):
            if resource is None or (index == 1 and resource is resources[0]):
                continue
            close = getattr(resource, "close", None)
            if callable(close):
                close()
        self._iterator = None
        self._rows = None
