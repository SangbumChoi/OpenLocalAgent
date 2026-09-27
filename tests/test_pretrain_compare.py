from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from localagent.eval.pretrain_compare import compare_pretrain_sidecars
from localagent.eval.pretrain_scorecard import (
    DOCUMENT_SIDECAR_KIND,
    DOCUMENT_SIDECAR_SCHEMA_VERSION,
    write_document_sidecar,
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fingerprint(values: list[str]) -> str:
    return _sha256("\n".join(sorted(values)))


def _documents(*, attention: bool) -> list[dict]:
    nll = [1.0, 2.0, 6.0, 7.0] if attention else [2.0, 3.0, 5.0, 6.0]
    correct = [8, 16, 20, 25] if attention else [7, 15, 22, 28]
    tokens = [10, 20, 30, 40]
    utf8_bytes = [15, 25, 35, 45]
    rows = []
    for index in range(4):
        group = "general" if index < 2 else "code"
        rows.append(
            {
                "document_identity_sha256": _sha256(f"document-{index}"),
                "document_content_sha256": _sha256(f"content-{index}"),
                "source_family": f"mixture:{group}",
                "source_group": group,
                "utf8_bytes": utf8_bytes[index],
                "tokens": tokens[index],
                "nll_nats": nll[index],
                "correct_tokens": correct[index],
            }
        )
    return rows


def _header(documents: list[dict], *, attention: bool, tokenizer_sha256: str) -> dict:
    identities = [document["document_identity_sha256"] for document in documents]
    content_bindings = [
        (
            f"{document['document_identity_sha256']}:"
            f"{document['document_content_sha256']}"
        )
        for document in documents
    ]
    document_set_sha256 = _fingerprint(identities)
    document_content_sha256 = _fingerprint(content_bindings)
    tokens = sum(document["tokens"] for document in documents)
    utf8_bytes = sum(document["utf8_bytes"] for document in documents)
    correct_tokens = sum(document["correct_tokens"] for document in documents)
    nll_nats = sum(document["nll_nats"] for document in documents)
    return {
        "kind": DOCUMENT_SIDECAR_KIND,
        "schema_version": DOCUMENT_SIDECAR_SCHEMA_VERSION,
        "bindings": {
            "checkpoint_sha256": _sha256(
                "attention-checkpoint" if attention else "hybrid-checkpoint"
            ),
            "model_config_sha256": _sha256(
                "attention-config" if attention else "hybrid-config"
            ),
            "pretrain_config_sha256": _sha256(
                "attention-pretrain-config" if attention else "hybrid-pretrain-config"
            ),
            "tokenizer_sha256": tokenizer_sha256,
            "manifest_sha256": _sha256("manifest-file"),
            "manifest_canonical_sha256": _sha256("manifest-canonical"),
            "staging_database_sha256": _sha256("staging-database"),
            "split_assignment_sha256": _sha256("split-assignment"),
            "validation_document_set_sha256": document_set_sha256,
            "validation_document_content_sha256": document_content_sha256,
        },
        "checkpoint_step": 100,
        "checkpoint_training_seed": 77,
        "checkpoint_token_accounting": {
            "source": "checkpoint.token_accounting",
            "input_tokens": 52_756_480,
            "loss_tokens": 52_700_000,
        },
        "groups": {
            "general": ["mixture:general"],
            "code": ["mixture:code"],
        },
        "evaluation": {
            "device": "cpu",
            "dtype": "float32",
            "batch_size": 1,
            "chunk_length": 128,
            "boundary_policy": "fixed-test-policy",
        },
        "validation": {
            "documents": len(documents),
            "utf8_bytes": utf8_bytes,
            "tokens": tokens,
            "correct_tokens": correct_tokens,
            "nll_nats": nll_nats,
            "cross_entropy_nats_per_token": nll_nats / tokens,
            "bits_per_byte": nll_nats / (math.log(2.0) * utf8_bytes),
            "top1_accuracy": correct_tokens / tokens,
            "document_set_sha256": document_set_sha256,
            "document_content_sha256": document_content_sha256,
        },
    }


def _write_pair(tmp_path: Path) -> tuple[Path, Path]:
    tokenizer_sha256 = _sha256("tokenizer")
    attention_documents = _documents(attention=True)
    hybrid_documents = _documents(attention=False)
    attention_path = tmp_path / "attention.jsonl"
    hybrid_path = tmp_path / "hybrid.jsonl"
    write_document_sidecar(
        _header(
            attention_documents,
            attention=True,
            tokenizer_sha256=tokenizer_sha256,
        ),
        attention_documents,
        attention_path,
    )
    write_document_sidecar(
        _header(
            hybrid_documents,
            attention=False,
            tokenizer_sha256=tokenizer_sha256,
        ),
        hybrid_documents,
        hybrid_path,
    )
    return attention_path, hybrid_path


def test_paired_document_bootstrap_is_deterministic_and_reports_orientation(
    tmp_path: Path,
) -> None:
    attention_path, hybrid_path = _write_pair(tmp_path)

    first = compare_pretrain_sidecars(
        attention_path,
        hybrid_path,
        seed=77,
        resamples=10_000,
    )
    second = compare_pretrain_sidecars(
        attention_path,
        hybrid_path,
        seed=77,
        resamples=10_000,
    )

    assert first == second
    assert first["bootstrap"]["difference"] == "attention_minus_hybrid"
    assert first["bootstrap"]["resamples"] == 10_000
    assert first["inputs"]["attention"]["token_accounting"]["input_tokens"] == 52_756_480
    assert (
        first["inputs"]["attention"]["pretrain_config_sha256"]
        == _sha256("attention-pretrain-config")
    )
    assert first["inputs"]["attention"]["training_seed"] == 77
    assert first["inputs"]["hybrid"]["training_seed"] == 77
    assert first["overall"]["documents"] == 4
    assert set(first["groups"]) == {"general", "code"}
    overall_difference = first["overall"]["difference_attention_minus_hybrid"]
    assert overall_difference["cross_entropy_nats_per_token"]["estimate"] == pytest.approx(0.0)
    assert overall_difference["bits_per_byte"]["estimate"] == pytest.approx(0.0)
    assert overall_difference["top1_accuracy"]["estimate"] == pytest.approx(-3 / 100)
    assert first["groups"]["general"]["difference_attention_minus_hybrid"][
        "cross_entropy_nats_per_token"
    ]["estimate"] == pytest.approx(-2 / 30)
    assert first["groups"]["code"]["difference_attention_minus_hybrid"][
        "cross_entropy_nats_per_token"
    ]["estimate"] == pytest.approx(2 / 70)
    assert first["overall"]["attention"]["bits_per_byte"] == pytest.approx(
        16.0 / (math.log(2.0) * 120)
    )
    for subset in [first["overall"], *first["groups"].values()]:
        for metric in subset["difference_attention_minus_hybrid"].values():
            assert metric["percentile_ci"]["lower"] <= metric["percentile_ci"]["upper"]
            assert metric["attention_win_fraction"] + metric["hybrid_win_fraction"] + metric[
                "tie_fraction"
            ] == pytest.approx(1.0)
    assert first["inputs"]["attention"]["sidecar"]["sha256"] == hashlib.sha256(
        attention_path.read_bytes()
    ).hexdigest()
    assert len(first["comparison_sha256"]) == 64


def test_paired_comparison_rejects_shared_binding_mismatch(tmp_path: Path) -> None:
    attention_path, hybrid_path = _write_pair(tmp_path)
    hybrid_documents = _documents(attention=False)
    write_document_sidecar(
        _header(
            hybrid_documents,
            attention=False,
            tokenizer_sha256=_sha256("different-tokenizer"),
        ),
        hybrid_documents,
        hybrid_path,
    )

    with pytest.raises(ValueError, match="tokenizer_sha256"):
        compare_pretrain_sidecars(attention_path, hybrid_path)


def test_paired_comparison_requires_at_least_10000_resamples(tmp_path: Path) -> None:
    attention_path, hybrid_path = _write_pair(tmp_path)

    with pytest.raises(ValueError, match="at least 10000"):
        compare_pretrain_sidecars(
            attention_path,
            hybrid_path,
            resamples=9_999,
        )
