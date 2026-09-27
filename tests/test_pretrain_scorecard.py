from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from localagent.data.pretrain_corpus import (
    CorpusDocument,
    _source_family,
    build_disk_backed_corpus,
    pack_disk_backed_shards,
)
from localagent.eval.pretrain_scorecard import (
    evaluate_pretrain_checkpoint,
    parse_source_groups,
    write_scorecard,
)
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import ByteTokenizer, train_bpe
from localagent.train.stage_data import canonical_sha256, tokenizer_identity


def _build_scorecard_artifacts(
    tmp_path: Path,
    *,
    tokenizer_kind: str = "byte",
) -> dict[str, object]:
    documents = [
        CorpusDocument(
            text=(
                f"Document {index} contains deterministic held-out language and symbols. "
                f"token-{index} value-{index * 7}."
            ),
            source=f"test-source-{index}",
            doc_id=f"document-{index}",
            license="MIT",
            meta={"mixture_source": f"family-{index}"},
        )
        for index in range(8)
    ]
    shards_dir = tmp_path / "shards"
    database_path = shards_dir / "corpus-staging.sqlite3"
    corpus = build_disk_backed_corpus(
        documents,
        database_path,
        min_chars=1,
        max_repetition_ratio=1.0,
        near_dedup=False,
        val_fraction=0.5,
        seed=17,
    )
    tokenizer_path = None
    if tokenizer_kind == "byte":
        tokenizer = ByteTokenizer()
        tokenizer_training = {"kind": "byte", "trained": False, "split": None}
    elif tokenizer_kind == "bpe":
        tokenizer_path = tmp_path / "tokenizer.json"
        tokenizer = train_bpe(
            (document.text for document in corpus.iter_documents("train")),
            tokenizer_path,
            vocab_size=300,
            min_frequency=1,
        )
        bpe_identity = tokenizer_identity(
            "bpe",
            vocab_size=tokenizer.vocab_size,
            path=tokenizer_path,
        )
        tokenizer_training = {
            "kind": "bpe",
            "trained": True,
            "split": "train",
            "artifact": bpe_identity["artifact"],
        }
    else:
        raise AssertionError(tokenizer_kind)
    manifest = pack_disk_backed_shards(
        corpus,
        tokenizer,
        seq_len=8,
        shards_dir=str(shards_dir),
        rows_per_shard=4,
        tokenizer_training=tokenizer_training,
    )

    config = ModelConfig(
        name="pretrain-scorecard-test",
        vocab_size=tokenizer.vocab_size,
        d_model=8,
        embed_dim=8,
        n_layers=1,
        n_loops=1,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden=16,
        max_seq_len=8,
        dropout=0.0,
    )
    torch.manual_seed(23)
    model = LocalAgentLM(config)
    tokenizer_lineage = tokenizer_identity(
        tokenizer_kind,
        vocab_size=tokenizer.vocab_size,
        path=tokenizer_path,
    )
    manifest_sha256 = canonical_sha256(manifest)
    lineage = {
        "version": 1,
        "stage": "pretrain",
        "config_sha256": canonical_sha256({"test": "pretrain-config"}),
        "model_config_sha256": canonical_sha256(config.__dict__),
        "data_sha256": canonical_sha256(
            {
                "kind": "packed_shards",
                "manifest_sha256": manifest_sha256,
                "split": "train",
            }
        ),
        "tokenizer_sha256": tokenizer_lineage["sha256"],
    }
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "cfg": config.__dict__,
            "state_dict": model.state_dict(),
            "stage": "pretrain",
            "step": 3,
            "training_seed": 29,
            "token_accounting": {
                "input_tokens": 12_345,
                "loss_tokens": 12_000,
            },
            "lineage": lineage,
            "tokenizer": {
                "kind": tokenizer_kind,
                "path": str(tokenizer_path) if tokenizer_path is not None else None,
                "sha256": tokenizer_lineage["sha256"],
            },
            "data": {
                "kind": "packed_shards",
                "path": str(shards_dir),
                "split": "train",
                "manifest_sha256": manifest_sha256,
            },
        },
        checkpoint_path,
    )
    validation_documents = list(corpus.iter_documents("val"))
    selector = _source_family(
        validation_documents[0].source,
        validation_documents[0].meta,
    )
    return {
        "checkpoint": checkpoint_path,
        "manifest": shards_dir / "manifest.json",
        "database": database_path,
        "validation_documents": validation_documents,
        "selector": selector,
        "tokenizer": tokenizer_path,
    }


def test_scorecard_reports_correct_denominators_hashes_and_is_deterministic(
    tmp_path: Path,
) -> None:
    artifacts = _build_scorecard_artifacts(tmp_path)
    kwargs = {
        "checkpoint_path": artifacts["checkpoint"],
        "manifest_path": artifacts["manifest"],
        "source_groups": {"focus": [artifacts["selector"]]},
        "corpus_db_path": artifacts["database"],
        "device": "cpu",
        "dtype": "fp32",
        "batch_size": 2,
        "chunk_length": 5,
    }

    first = evaluate_pretrain_checkpoint(**kwargs)
    second = evaluate_pretrain_checkpoint(**kwargs)

    assert first == second
    assert "document_sidecar" not in first
    validation_documents = artifacts["validation_documents"]
    expected_bytes = sum(len(document.text.encode("utf-8")) for document in validation_documents)
    assert first["aggregate"]["documents"] == len(validation_documents)
    assert first["aggregate"]["utf8_bytes"] == expected_bytes
    assert first["aggregate"]["tokens"] == expected_bytes
    assert first["aggregate"]["bits_per_byte"] == pytest.approx(
        first["aggregate"]["nll_nats"] / (math.log(2.0) * expected_bytes)
    )
    assert first["aggregate"]["cross_entropy_nats_per_token"] == pytest.approx(
        first["aggregate"]["nll_nats"] / first["aggregate"]["tokens"]
    )
    assert first["aggregate"]["top1_accuracy"] == pytest.approx(
        first["aggregate"]["correct_tokens"] / first["aggregate"]["tokens"]
    )
    assert first["groups"]["focus"]["documents"] == 1
    assert first["dataset"]["selection"] == "all_verified_validation_documents"
    assert first["checkpoint"]["token_accounting"] == {
        "source": "checkpoint.token_accounting",
        "input_tokens": 12_345,
        "loss_tokens": 12_000,
    }
    assert "non-overlapping target chunks" in first["evaluation"]["boundary_policy"]
    assert "overlap the preceding source token" in first["evaluation"]["boundary_policy"]
    assert first["evaluation"]["boundary_policy"].endswith(
        "exclude closing EOS from CE and BPB"
    )
    for value in (
        first["checkpoint"]["sha256"],
        first["tokenizer"]["sha256"],
        first["dataset"]["manifest"]["sha256"],
        first["dataset"]["manifest"]["canonical_sha256"],
        first["dataset"]["staging_database"]["sha256"],
        first["aggregate"]["document_set_sha256"],
        first["aggregate"]["document_content_sha256"],
    ):
        assert len(value) == 64

    output = tmp_path / "scorecard.json"
    write_scorecard(first, output)
    assert json.loads(output.read_text(encoding="utf-8")) == first


def test_scorecard_writes_hash_bound_text_free_document_sidecar(tmp_path: Path) -> None:
    artifacts = _build_scorecard_artifacts(tmp_path)
    validation_documents = artifacts["validation_documents"]
    selectors = sorted(
        {
            _source_family(document.source, document.meta)
            for document in validation_documents
        }
    )
    sidecar_path = tmp_path / "document-metrics.jsonl"

    report = evaluate_pretrain_checkpoint(
        artifacts["checkpoint"],
        artifacts["manifest"],
        {"heldout": selectors},
        corpus_db_path=artifacts["database"],
        device="cpu",
        batch_size=2,
        document_sidecar_path=sidecar_path,
    )

    raw_lines = sidecar_path.read_text(encoding="utf-8").splitlines()
    header = json.loads(raw_lines[0])
    documents = [json.loads(line) for line in raw_lines[1:]]
    assert len(documents) == len(validation_documents)
    assert header["record_type"] == "header"
    assert header["checkpoint_token_accounting"]["input_tokens"] == 12_345
    assert header["checkpoint_training_seed"] == 29
    assert report["checkpoint"]["training_seed"] == 29
    assert header["bindings"]["checkpoint_sha256"] == report["checkpoint"]["sha256"]
    assert (
        header["bindings"]["pretrain_config_sha256"]
        == report["checkpoint"]["pretrain_config_sha256"]
    )
    assert header["bindings"]["tokenizer_sha256"] == report["tokenizer"]["sha256"]
    assert (
        header["bindings"]["manifest_canonical_sha256"]
        == report["dataset"]["manifest"]["canonical_sha256"]
    )
    assert (
        header["bindings"]["validation_document_content_sha256"]
        == report["aggregate"]["document_content_sha256"]
    )
    assert sum(document["tokens"] for document in documents) == report["aggregate"]["tokens"]
    assert sum(document["utf8_bytes"] for document in documents) == report["aggregate"][
        "utf8_bytes"
    ]
    assert sum(document["correct_tokens"] for document in documents) == report["aggregate"][
        "correct_tokens"
    ]
    for document in documents:
        assert set(document) == {
            "record_type",
            "document_identity_sha256",
            "document_content_sha256",
            "source_family",
            "source_group",
            "utf8_bytes",
            "tokens",
            "nll_nats",
            "correct_tokens",
        }
        assert "text" not in document
    assert report["document_sidecar"]["documents"] == len(documents)
    assert report["document_sidecar"]["sha256"] == hashlib.sha256(
        sidecar_path.read_bytes()
    ).hexdigest()


def test_scorecard_fails_closed_on_checkpoint_dataset_lineage_mismatch(
    tmp_path: Path,
) -> None:
    artifacts = _build_scorecard_artifacts(tmp_path)
    checkpoint = torch.load(artifacts["checkpoint"], map_location="cpu", weights_only=False)
    checkpoint["data"]["manifest_sha256"] = "0" * 64
    bad_checkpoint = tmp_path / "wrong-data.pt"
    torch.save(checkpoint, bad_checkpoint)

    with pytest.raises(ValueError, match="manifest lineage mismatch"):
        evaluate_pretrain_checkpoint(
            bad_checkpoint,
            artifacts["manifest"],
            {"focus": [artifacts["selector"]]},
            corpus_db_path=artifacts["database"],
            device="cpu",
        )


def test_scorecard_loads_manifest_bound_bpe_tokenizer(tmp_path: Path) -> None:
    artifacts = _build_scorecard_artifacts(tmp_path, tokenizer_kind="bpe")

    report = evaluate_pretrain_checkpoint(
        artifacts["checkpoint"],
        artifacts["manifest"],
        {"focus": [artifacts["selector"]]},
        corpus_db_path=artifacts["database"],
        device="cpu",
    )

    assert report["tokenizer"]["kind"] == "bpe"
    assert report["tokenizer"]["vocab_size"] > 256
    assert report["tokenizer"]["artifact"]["path"] == str(artifacts["tokenizer"])
    assert report["tokenizer"]["artifact"]["sha256"] == hashlib.sha256(
        Path(artifacts["tokenizer"]).read_bytes()
    ).hexdigest()


def test_scorecard_fails_closed_on_tokenizer_lineage_mismatch(tmp_path: Path) -> None:
    artifacts = _build_scorecard_artifacts(tmp_path)
    checkpoint = torch.load(artifacts["checkpoint"], map_location="cpu", weights_only=False)
    checkpoint["tokenizer"]["sha256"] = "f" * 64
    bad_checkpoint = tmp_path / "wrong-tokenizer.pt"
    torch.save(checkpoint, bad_checkpoint)

    with pytest.raises(ValueError, match="tokenizer artifact lineage mismatch"):
        evaluate_pretrain_checkpoint(
            bad_checkpoint,
            artifacts["manifest"],
            {"focus": [artifacts["selector"]]},
            corpus_db_path=artifacts["database"],
            device="cpu",
        )


def test_scorecard_rejects_empty_and_overlapping_source_groups(tmp_path: Path) -> None:
    artifacts = _build_scorecard_artifacts(tmp_path)
    with pytest.raises(ValueError, match="requested source group.*empty"):
        evaluate_pretrain_checkpoint(
            artifacts["checkpoint"],
            artifacts["manifest"],
            {"missing": ["mixture:not-in-validation"]},
            corpus_db_path=artifacts["database"],
            device="cpu",
        )

    with pytest.raises(ValueError, match="belongs to both"):
        parse_source_groups(
            [
                "general=mixture:shared",
                "code=mixture:shared",
            ]
        )
