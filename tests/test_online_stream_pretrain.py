"""Online pretraining consumes fresh rows and enforces per-row source policy."""

from __future__ import annotations

import gzip
import io

import pytest

from openlocalagent.data.corpus.online_stream import OnlineSourceDataset, _online_stream
from openlocalagent.model.tokenizer import ByteTokenizer


def _source() -> dict[str, object]:
    return {
        "name": "permissive_python",
        "text_field": "content",
        "adapter": "python_code",
        "license_field": "license",
        "allowed_licenses": ["mit"],
        "min_document_chars": 1,
        "max_document_chars": 1000,
    }


def test_stream_consumes_more_rows_as_batches_are_drawn() -> None:
    opened = []

    def factory(source, seed):
        opened.append((source["shuffle_buffer"], seed))
        return iter(
            [
                {"content": "rejected", "license": "gpl-3.0"},
                {"content": "abcd", "license": "MIT"},
                {"content": "efgh", "license": ["mit"]},
                {"content": "ijkl", "license": "mit"},
            ]
        )

    dataset = OnlineSourceDataset(
        _source(), ByteTokenizer(), seq_len=3, seed=7, shuffle_buffer=2,
        stream_factory=factory,
    )
    first, _ = dataset.sample_batch(1, None, "cpu")
    assert first.tolist() == [[97, 98, 99]]
    assert dataset.stats()["rows_used"] == 1
    second, _ = dataset.sample_batch(1, None, "cpu")
    assert second.tolist() == [[0, 101, 102]]
    assert dataset.stats()["rows_used"] == 2
    assert dataset.stats()["rows_license_rejected"] == 1
    assert opened == [(2, 7)]


def test_stream_never_silently_recycles_exhausted_rows() -> None:
    dataset = OnlineSourceDataset(
        _source(), ByteTokenizer(), seq_len=3, seed=1,
        stream_factory=lambda source, seed: iter([{"content": "abcd", "license": "mit"}]),
    )
    dataset.sample_batch(1, None, "cpu")
    with pytest.raises(RuntimeError, match="exhausted"):
        dataset.sample_batch(1, None, "cpu")


def test_invalid_preprocessor_is_rejected() -> None:
    with pytest.raises(ValueError, match="module:function"):
        OnlineSourceDataset(_source(), ByteTokenizer(), seq_len=2, seed=1,
                            preprocessor="not_a_path")


def test_source_specific_python_preprocessor() -> None:
    source = _source()
    source["license_field"] = None
    source["allowed_licenses"] = []
    dataset = OnlineSourceDataset(
        source, ByteTokenizer(), seq_len=2, seed=1,
        preprocessor="openlocalagent.data.adapters.pretrain_finemath:preprocess",
        stream_factory=lambda source, seed: iter([{"content": "a\r\nb\x00c"}]),
    )
    inputs, _ = dataset.sample_batch(1, None, "cpu")
    assert inputs.tolist() == [[97, 10]]


def test_codeparrot_online_transport_does_not_download_a_shard(monkeypatch) -> None:
    payload = gzip.compress(b'{"content":"abcd","license":"mit"}\n')
    seen_urls = []

    class Response:
        raw = io.BytesIO(payload)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def raise_for_status(self):
            return None

    def get(url, *, stream, timeout):
        seen_urls.append((url, stream, timeout))
        return Response()

    import requests

    monkeypatch.setattr(requests, "get", get)
    source = {
        **_source(),
        "dataset": "codeparrot/codeparrot-clean",
        "revision": "a" * 40,
        "raw_stream": {
            "backend": "hf-jsonl-gzip-v1",
            "interleave_files": 1,
            "file_inventory": {"files": [{"path": "data/file-000000000001.json.gz"}]},
        },
    }
    assert list(_online_stream(source, 7)) == [{"content": "abcd", "license": "mit"}]
    assert seen_urls[0][1] is True
    assert "/resolve/" in seen_urls[0][0]


def test_close_releases_active_iterator() -> None:
    closed = []

    def rows():
        try:
            yield {"content": "abcd", "license": "mit"}
            yield {"content": "efgh", "license": "mit"}
        finally:
            closed.append(True)

    dataset = OnlineSourceDataset(
        _source(), ByteTokenizer(), seq_len=2, seed=1,
        stream_factory=lambda source, seed: rows(),
    )
    dataset.sample_batch(1, None, "cpu")
    dataset.close()
    assert closed == [True]
