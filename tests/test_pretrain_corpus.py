import importlib.util
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from openlocalagent.data.corpus import pretrain_corpus
from openlocalagent.data.corpus.pretrain_corpus import (
    CorpusDocument,
    DiskBackedCorpus,
    PackedShardDataset,
    build_disk_backed_corpus,
    iter_documents,
    load_frozen_split_assignment_manifest,
    near_deduplicate,
    pack_disk_backed_shards,
    pack_shards,
    quality_filter,
    read_evaluation_denylist,
    screen_evaluation_contamination,
    split_documents,
    suggested_training_tokens,
)
from openlocalagent.model.tokenizer import ByteTokenizer


def test_quality_filter_deduplicates_and_preserves_provenance():
    text = "A useful document about agents and tools. " * 12
    docs = [
        CorpusDocument(text, source="a", doc_id="one", license="MIT"),
        CorpusDocument(text, source="b", doc_id="two", license="Apache-2.0"),
        CorpusDocument("short", source="c"),
    ]
    accepted = quality_filter(docs)
    assert len(accepted) == 1
    assert accepted[0].source == "a"
    assert accepted[0].license == "MIT"
    assert quality_filter(reversed(docs))[0].doc_id == "one"


def test_iter_documents_reads_text_and_jsonl(tmp_path):
    (tmp_path / "one.txt").write_text("plain text")
    (tmp_path / "two.jsonl").write_text(json.dumps({"text": "json text", "license": "MIT"}) + "\n")
    docs = list(iter_documents(tmp_path))
    assert {doc.text for doc in docs} == {"plain text", "json text"}
    assert all(doc.doc_id for doc in docs)


def test_jsonl_missing_ids_use_content_identity_independent_of_line_order(tmp_path):
    rows = [
        {"text": "first content-identified document", "license": "MIT"},
        {"text": "second content-identified document", "license": "Apache-2.0"},
        {"text": "explicit identity stays explicit", "doc_id": "upstream-id"},
    ]
    forward_path = tmp_path / "forward.jsonl"
    reverse_path = tmp_path / "reverse.jsonl"
    forward_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    reverse_path.write_text(
        "".join(json.dumps(row) + "\n" for row in reversed(rows)),
        encoding="utf-8",
    )

    forward = list(iter_documents(forward_path))
    reverse = list(iter_documents(reverse_path))
    forward_ids = {document.text: document.doc_id for document in forward}
    reverse_ids = {document.text: document.doc_id for document in reverse}
    assert forward_ids == reverse_ids
    assert forward_ids["explicit identity stays explicit"] == "upstream-id"
    assert len(forward_ids["first content-identified document"]) == 64

    forward_splits = split_documents(forward, val_fraction=0.34, seed=19)
    reverse_splits = split_documents(reverse, val_fraction=0.34, seed=19)
    for split in ("train", "val"):
        assert {document.doc_id for document in forward_splits[split]} == {
            document.doc_id for document in reverse_splits[split]
        }


def test_iter_documents_enforces_streaming_byte_guards(tmp_path):
    oversized_jsonl = tmp_path / "oversized.jsonl"
    oversized_jsonl.write_text(
        json.dumps({"text": "x" * 256}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="JSONL record exceeds 64 bytes"):
        list(iter_documents(oversized_jsonl, max_document_bytes=64))

    oversized_text = tmp_path / "oversized.txt"
    oversized_text.write_bytes(b"x" * 65)
    with pytest.raises(ValueError, match="document exceeds 64 bytes"):
        list(iter_documents(oversized_text, max_document_bytes=64))

    valid_jsonl = tmp_path / "valid.jsonl"
    valid_jsonl.write_text(json.dumps({"text": "bounded"}) + "\n", encoding="utf-8")
    assert [document.text for document in iter_documents(valid_jsonl, max_document_bytes=64)] == [
        "bounded"
    ]
    with pytest.raises(ValueError, match="max_document_bytes must be positive"):
        list(iter_documents(valid_jsonl, max_document_bytes=0))


def test_pack_shards_masks_padding_and_separates_documents(tmp_path):
    tok = ByteTokenizer()
    docs = [
        CorpusDocument(("alpha beta gamma " * 30) + str(i), doc_id=f"doc-{i}") for i in range(10)
    ]
    manifest = pack_shards(
        docs,
        tok,
        seq_len=32,
        shards_dir=str(tmp_path),
        rows_per_shard=3,
        val_fraction=0.2,
        seed=7,
    )
    assert manifest["splits"]["train"]["documents"] == 8
    assert manifest["splits"]["val"]["documents"] == 2
    assert len(manifest["split_assignment_sha256"]) == 64
    assert len(manifest["splits"]["train"]["document_set_sha256"]) == 64
    assert len(manifest["generation"]) == 32
    assert sum(manifest["source_token_counts"].values()) == manifest["total_tokens"]
    for split in ("train", "val"):
        assert (
            sum(manifest["splits"][split]["source_token_counts"].values())
            == manifest["splits"][split]["tokens"]
        )
        for entry in manifest["splits"][split]["shards"]:
            assert Path(entry["tokens"]).parts[:2] == (
                "generations",
                manifest["generation"],
            )
            assert entry["bytes"] > 0 and entry["lengths_bytes"] > 0
    train = PackedShardDataset(tmp_path, "train")
    val = PackedShardDataset(tmp_path, "val")
    assert len(train) and len(val)
    x, y = train.row(0)
    assert x.shape == y.shape == (32,)
    assert x.dtype == y.dtype == np.int64
    bx, by = train.sample_batch(2, __import__("random").Random(0), "cpu")
    assert bx.shape == by.shape == (2, 32)
    materialized_rng = random.Random(23)
    counted_rng = random.Random(23)
    bx, by = train.sample_batch(7, materialized_rng, "cpu")
    expected_loss_tokens = int((by != -100).sum())
    expected_input_tokens = int(np.minimum((by != -100).sum(axis=1).numpy() + 1, bx.shape[1]).sum())
    assert train.sample_batch_token_counts(7, counted_rng) == (
        expected_input_tokens,
        expected_loss_tokens,
    )
    assert counted_rng.getstate() == materialized_rng.getstate()
    pad_dir = tmp_path / "padding-case"
    pack_shards(
        [CorpusDocument("short row", doc_id="short")],
        tok,
        seq_len=32,
        shards_dir=str(pad_dir),
        val_fraction=0,
    )
    assert (PackedShardDataset(pad_dir, "train").row(0)[1] == -100).any()


def test_packed_shard_loader_rejects_same_size_tampering(tmp_path):
    manifest = pack_shards(
        [CorpusDocument("integrity checked document " * 12, doc_id="integrity")],
        ByteTokenizer(),
        seq_len=32,
        shards_dir=str(tmp_path),
        val_fraction=0,
    )
    entry = manifest["splits"]["train"]["shards"][0]
    token_path = tmp_path / entry["tokens"]
    with token_path.open("r+b") as handle:
        handle.seek(-1, 2)
        original = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([original[0] ^ 0x01]))
    assert token_path.stat().st_size == entry["bytes"]

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        PackedShardDataset(tmp_path, "train")


def test_manifest_rejects_shards_from_a_stale_generation(tmp_path):
    first = pack_shards(
        [CorpusDocument("first immutable generation " * 12, doc_id="first")],
        ByteTokenizer(),
        seq_len=32,
        shards_dir=str(tmp_path),
        val_fraction=0,
    )
    second = pack_shards(
        [CorpusDocument("second immutable generation " * 12, doc_id="second")],
        ByteTokenizer(),
        seq_len=32,
        shards_dir=str(tmp_path),
        val_fraction=0,
    )
    assert first["generation"] != second["generation"]
    assert (tmp_path / "generations" / first["generation"]).is_dir()
    assert (tmp_path / "generations" / second["generation"]).is_dir()

    stale_entry = first["splits"]["train"]["shards"][0]
    current_entry = second["splits"]["train"]["shards"][0]
    current_entry.update(
        {
            key: stale_entry[key]
            for key in (
                "tokens",
                "lengths",
                "bytes",
                "lengths_bytes",
                "sha256",
                "lengths_sha256",
            )
        }
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(second, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not part of manifest generation"):
        PackedShardDataset(tmp_path, "train")


def test_document_split_is_disjoint_and_independent_of_input_order():
    docs = [CorpusDocument(f"document body {index}", doc_id=f"doc-{index}") for index in range(20)]
    forward = split_documents(docs, val_fraction=0.2, seed=17)
    reversed_ = split_documents(list(reversed(docs)), val_fraction=0.2, seed=17)

    train_ids = {document.doc_id for document in forward["train"]}
    val_ids = {document.doc_id for document in forward["val"]}
    assert not train_ids & val_ids
    assert train_ids | val_ids == {document.doc_id for document in docs}
    assert train_ids == {document.doc_id for document in reversed_["train"]}
    assert val_ids == {document.doc_id for document in reversed_["val"]}


def test_near_dedup_is_order_stable_and_keeps_unrelated_documents():
    base = " ".join(f"agent_token_{index}" for index in range(80))
    docs = [
        CorpusDocument(base, source="a", doc_id="base"),
        CorpusDocument(base + " trailing_unique_token", source="b", doc_id="near"),
        CorpusDocument(
            " ".join(f"unrelated_topic_{index}" for index in range(80)),
            source="c",
            doc_id="unrelated",
        ),
    ]
    forward, forward_audit = near_deduplicate(docs)
    reversed_, reversed_audit = near_deduplicate(reversed(docs))

    assert {document.doc_id for document in forward} == {"near", "unrelated"}
    assert {document.doc_id for document in reversed_} == {"near", "unrelated"}
    assert forward_audit["removed_documents"] == 1
    assert reversed_audit["removal_pairs_sha256"] == forward_audit["removal_pairs_sha256"]
    assert forward_audit["exhaustive"] is False


def test_near_dedup_preserves_case_sensitive_code_variants(tmp_path):
    upper = "\n".join(
        f"def transform_{index}(UserID): return UserID + {index}" for index in range(32)
    )
    lower = upper.replace("UserID", "userId")
    docs = [
        CorpusDocument(upper, source="src/upper.py", doc_id="upper", license="MIT"),
        CorpusDocument(lower, source="src/lower.py", doc_id="lower", license="MIT"),
    ]

    retained, audit = near_deduplicate(docs)
    assert {document.doc_id for document in retained} == {"upper", "lower"}
    assert audit["code_documents_bypassed"] == 2
    assert audit["code_policy"] == "exact_content_deduplication_only"
    assert audit["case_sensitive_shingles"] is True

    corpus = build_disk_backed_corpus(
        docs,
        tmp_path / "code.sqlite3",
        min_chars=1,
        near_dedup=True,
        val_fraction=0,
    )
    assert {document.doc_id for document in corpus.iter_documents()} == {"upper", "lower"}
    disk_audit = corpus.corpus_audit["near_deduplication"]
    assert disk_audit["code_documents_bypassed"] == 2
    assert disk_audit["removed_documents"] == 0


def test_disk_backed_document_iterator_allows_sequential_thread_handoff(tmp_path):
    corpus = build_disk_backed_corpus(
        [
            CorpusDocument("alpha document text", doc_id="alpha"),
            CorpusDocument("beta document text", doc_id="beta"),
        ],
        tmp_path / "thread-handoff.sqlite3",
        min_chars=1,
        near_dedup=False,
        val_fraction=0,
    )
    documents = corpus.iter_documents("train")
    first = next(documents)

    with ThreadPoolExecutor(max_workers=1) as executor:
        remainder = executor.submit(list, documents).result()

    assert {first.doc_id, *(document.doc_id for document in remainder)} == {
        "alpha",
        "beta",
    }


def test_eval_denylist_reads_text_and_jsonl_and_excludes_contamination(tmp_path):
    prompt = (
        "Open the account settings page and change the notification frequency "
        "to weekly without modifying any privacy controls."
    )
    text_path = tmp_path / "eval.txt"
    text_path.write_text(prompt + "\n", encoding="utf-8")
    jsonl_path = tmp_path / "eval.jsonl"
    jsonl_path.write_text(
        json.dumps({"messages": [{"role": "user", "content": "Delete the draft report."}]}) + "\n",
        encoding="utf-8",
    )
    denylist = read_evaluation_denylist([text_path, jsonl_path])
    assert set(denylist) == {prompt, "Delete the draft report."}

    docs = [
        CorpusDocument(
            f"Browser benchmark task: {prompt} Completion should be verified.",
            doc_id="contaminated",
        ),
        CorpusDocument(
            "A technical chapter about sparse attention, cache layouts, and quantization.",
            doc_id="unrelated",
        ),
    ]
    retained, audit = screen_evaluation_contamination(docs, denylist)
    assert [document.doc_id for document in retained] == ["unrelated"]
    assert audit["removed_documents"] == 1
    assert audit["matched_denylist_entries"] == 1
    assert audit["denylist_sha256"]
    assert audit["exhaustive"] is False


def test_eval_denylist_reads_versioned_json_benchmark_suites(tmp_path):
    suite_path = tmp_path / "benchmark-cases.json"
    suite_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "held-out-browser-actions",
                "cases": [
                    {"id": "query-case", "query": "Select the Confirm button."},
                    {"id": "prompt-case", "prompt": "Type customer feedback."},
                ],
            }
        ),
        encoding="utf-8",
    )
    assert read_evaluation_denylist(suite_path) == [
        "Select the Confirm button.",
        "Type customer feedback.",
    ]

    invalid_path = tmp_path / "unversioned.json"
    invalid_path.write_text(
        json.dumps({"cases": [{"query": "This suite is missing its version."}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="needs a schema_version/version"):
        read_evaluation_denylist(invalid_path)


def test_eval_denylist_enforces_total_record_and_entry_bounds(tmp_path):
    first = tmp_path / "first.jsonl"
    first.write_text(
        json.dumps({"prompt": "A bounded evaluation prompt."}) + "\n",
        encoding="utf-8",
    )
    second = tmp_path / "second.txt"
    second.write_text("Another bounded evaluation prompt.\n", encoding="utf-8")

    with pytest.raises(ValueError, match="max_total_bytes must be a positive integer"):
        read_evaluation_denylist(first, max_total_bytes=0)
    with pytest.raises(ValueError, match="denylist row exceeds"):
        read_evaluation_denylist(first, max_record_bytes=16)
    with pytest.raises(ValueError, match="total bytes"):
        read_evaluation_denylist(
            [first, second],
            max_total_bytes=first.stat().st_size,
        )
    with pytest.raises(ValueError, match="prompt entries"):
        read_evaluation_denylist([first, second], max_entries=1)


def test_eval_denylist_rejects_invalid_utf8_without_replacement(tmp_path):
    jsonl_path = tmp_path / "invalid.jsonl"
    jsonl_path.write_bytes(b'{"prompt":"bad \\xff"}\n')
    with pytest.raises(ValueError, match="invalid UTF-8 JSON"):
        read_evaluation_denylist(jsonl_path)

    text_path = tmp_path / "invalid.txt"
    text_path.write_bytes(b"bad \xff\n")
    with pytest.raises(ValueError, match="not valid UTF-8"):
        read_evaluation_denylist(text_path)


def test_prepare_corpus_trains_bpe_on_exact_train_split(tmp_path, monkeypatch, capsys):
    script_path = Path(__file__).parents[1] / "scripts" / "build_corpus.py"
    spec = importlib.util.spec_from_file_location("prepare_corpus_script", script_path)
    assert spec is not None and spec.loader is not None
    prepare_corpus = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare_corpus)

    raw_path = tmp_path / "raw.jsonl"
    raw_documents = [
        CorpusDocument(
            text=f"Document {index}: " + ("agent tool pretraining text " * 12),
            source="unit-test",
            doc_id=f"doc-{index}",
            license="MIT",
        )
        for index in range(10)
    ]
    raw_path.write_text(
        "".join(json.dumps(document.__dict__) + "\n" for document in raw_documents),
        encoding="utf-8",
    )
    accepted = quality_filter(raw_documents)
    initial_splits = split_documents(accepted, val_fraction=0.2, seed=9)
    denied_document = initial_splits["train"][0]
    denylist_path = tmp_path / "eval-denylist.txt"
    denylist_path.write_text(denied_document.text + "\n", encoding="utf-8")
    decontaminated, _ = screen_evaluation_contamination(
        accepted,
        read_evaluation_denylist(denylist_path),
    )
    expected_splits = split_documents(decontaminated, val_fraction=0.2, seed=9)
    tokenizer_documents: list[str] = []

    def fake_train_bpe(documents, path, *, vocab_size):
        del path, vocab_size
        tokenizer_documents.extend(documents)
        return ByteTokenizer()

    output_dir = tmp_path / "shards"
    monkeypatch.setattr(prepare_corpus, "train_bpe", fake_train_bpe)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_corpus.py",
            str(raw_path),
            "--out",
            str(output_dir),
            "--seq-len",
            "32",
            "--val-fraction",
            "0.2",
            "--seed",
            "9",
            "--tokenizer",
            "bpe",
            "--vocab-size",
            "320",
            "--eval-denylist",
            str(denylist_path),
            "--no-near-dedup",
        ],
    )
    prepare_corpus.main()
    capsys.readouterr()

    assert tokenizer_documents == [document.text for document in expected_splits["train"]]
    assert denied_document.text not in tokenizer_documents
    assert not set(tokenizer_documents) & {document.text for document in expected_splits["val"]}
    manifest = json.loads((output_dir / "manifest.json").read_text())
    tokenizer_audit = manifest["tokenizer_training"]
    assert tokenizer_audit["split"] == "train"
    assert tokenizer_audit["documents"] == len(expected_splits["train"])
    assert tokenizer_audit["excluded_documents"] == len(expected_splits["val"])
    assert manifest["total_documents"] == len(decontaminated)
    assert manifest["corpus_audit"]["evaluation_decontamination"]["removed_documents"] == 1
    assert (
        tokenizer_audit["document_set_sha256"] == manifest["splits"]["train"]["document_set_sha256"]
    )
    preparation = manifest["preparation"]
    assert preparation["mode"] == "sqlite_disk_backed"
    assert preparation["staging_database"]["sha256"]
    assert preparation["provenance"]["filtered_jsonl"]["sha256"]


def test_disk_backed_hygiene_is_order_stable_and_packs_canonical_splits(tmp_path):
    prompt = (
        "Open account settings and set browser notifications to weekly while "
        "leaving every privacy control unchanged."
    )
    near_base = " ".join(f"browser_agent_token_{index}" for index in range(80))
    exact_text = "An exact duplicate document about deterministic corpus preparation. " * 4
    docs = [
        CorpusDocument(exact_text, source="z", doc_id="z-exact", license="MIT"),
        CorpusDocument(exact_text, source="a", doc_id="a-exact", license="MIT"),
        CorpusDocument(near_base, source="near", doc_id="near-short", license="MIT"),
        CorpusDocument(
            near_base + " trailing_unique_token",
            source="near",
            doc_id="near-long",
            license="MIT",
        ),
        CorpusDocument(
            f"Evaluation task copied into training: {prompt}",
            source="contaminated",
            doc_id="contaminated",
            license="MIT",
        ),
        *[
            CorpusDocument(
                " ".join(f"independent_topic_{index}_{token}" for token in range(40)),
                source="independent",
                doc_id=f"independent-{index}",
                license="Apache-2.0",
            )
            for index in range(8)
        ],
    ]
    expected, _ = screen_evaluation_contamination(
        quality_filter(docs, min_chars=1),
        [prompt],
    )
    expected, expected_near_audit = near_deduplicate(expected)
    expected_splits = split_documents(expected, val_fraction=0.25, seed=73)

    forward = build_disk_backed_corpus(
        iter(docs),
        tmp_path / "forward.sqlite3",
        min_chars=1,
        denylist=[prompt],
        val_fraction=0.25,
        seed=73,
    )
    reversed_ = build_disk_backed_corpus(
        reversed(docs),
        tmp_path / "reversed.sqlite3",
        min_chars=1,
        denylist=[prompt],
        val_fraction=0.25,
        seed=73,
    )

    assert isinstance(forward, DiskBackedCorpus)
    expected_ids = [document.doc_id for document in expected]
    assert [document.doc_id for document in forward.iter_documents()] == expected_ids
    assert [document.doc_id for document in reversed_.iter_documents()] == expected_ids
    assert "contaminated" not in expected_ids
    assert "a-exact" in expected_ids and "z-exact" not in expected_ids
    assert (
        forward.corpus_audit["near_deduplication"]["removed_documents"]
        == (expected_near_audit["removed_documents"])
    )
    for split in ("train", "val"):
        split_ids = [document.doc_id for document in expected_splits[split]]
        assert [document.doc_id for document in forward.iter_documents(split)] == split_ids
        assert [document.doc_id for document in reversed_.iter_documents(split)] == split_ids

    forward_manifest = pack_disk_backed_shards(
        forward,
        ByteTokenizer(),
        seq_len=64,
        shards_dir=str(tmp_path / "forward-shards"),
        rows_per_shard=3,
        tokenizer_training={"kind": "byte", "trained": False, "split": None},
    )
    reversed_manifest = pack_disk_backed_shards(
        reversed_,
        ByteTokenizer(),
        seq_len=64,
        shards_dir=str(tmp_path / "reversed-shards"),
        rows_per_shard=3,
        tokenizer_training={"kind": "byte", "trained": False, "split": None},
    )
    assert (
        forward_manifest["split_assignment_sha256"]
        == (reversed_manifest["split_assignment_sha256"])
    )
    for split in ("train", "val"):
        assert (
            forward_manifest["splits"][split]["document_set_sha256"]
            == (reversed_manifest["splits"][split]["document_set_sha256"])
        )
    assert forward_manifest["preparation"]["mode"] == "sqlite_disk_backed"
    assert forward_manifest["total_documents"] == len(expected_ids)


def test_frozen_split_manifest_preserves_base_validation_membership_for_subset(tmp_path):
    docs = [
        CorpusDocument(
            f"Frozen split document {index} has distinct auditable content. " * 4,
            source="unit-test",
            doc_id=f"frozen-{index}",
            license="MIT",
        )
        for index in range(8)
    ]
    base = build_disk_backed_corpus(
        docs,
        tmp_path / "base.sqlite3",
        min_chars=1,
        near_dedup=False,
        val_fraction=0.25,
        seed=31,
    )
    held_out = next(base.iter_documents("val"))
    assert split_documents([held_out], val_fraction=0.25, seed=31)["train"] == [held_out]

    base_shards = tmp_path / "base-shards"
    base_manifest = pack_disk_backed_shards(
        base,
        ByteTokenizer(),
        seq_len=32,
        shards_dir=str(base_shards),
    )
    reference = load_frozen_split_assignment_manifest(base_shards / "manifest.json")
    assert reference.assignment_sha256 == base_manifest["split_assignment_sha256"]
    assert reference.sha256 == base_manifest["split_assignment"]["sha256"]

    derived = build_disk_backed_corpus(
        [held_out],
        tmp_path / "derived.sqlite3",
        min_chars=1,
        near_dedup=False,
        val_fraction=0.25,
        seed=31,
        frozen_split_assignment=reference,
    )
    assert derived.count("train") == 0
    assert [document.doc_id for document in derived.iter_documents("val")] == [held_out.doc_id]
    audit = derived.corpus_audit["split_assignment"]
    assert audit["mode"] == "frozen"
    assert audit["missing_documents"] == 0
    assert audit["source_assignment_sha256"] == base_manifest["split_assignment_sha256"]


def test_frozen_split_manifest_fails_on_missing_or_changed_document_binding(tmp_path):
    base_document = CorpusDocument(
        "The original frozen document content is stable and auditable. " * 4,
        doc_id="stable-upstream-id",
    )
    base = build_disk_backed_corpus(
        [base_document],
        tmp_path / "base.sqlite3",
        min_chars=1,
        near_dedup=False,
        val_fraction=0,
        seed=7,
    )
    base_shards = tmp_path / "base-shards"
    pack_disk_backed_shards(
        base,
        ByteTokenizer(),
        seq_len=32,
        shards_dir=str(base_shards),
    )
    reference = load_frozen_split_assignment_manifest(base_shards / "manifest.json")

    missing = CorpusDocument(
        "A newly introduced document cannot inherit an unknown split. " * 4,
        doc_id="new-upstream-id",
    )
    changed = CorpusDocument(
        "The upstream content changed while retaining the same document id. " * 4,
        doc_id=base_document.doc_id,
    )
    for index, document in enumerate((missing, changed)):
        with pytest.raises(ValueError, match="missing 1 retained document/content binding"):
            build_disk_backed_corpus(
                [document],
                tmp_path / f"invalid-{index}.sqlite3",
                min_chars=1,
                near_dedup=False,
                val_fraction=0,
                seed=7,
                frozen_split_assignment=reference,
            )


def test_frozen_split_manifest_rejects_tampered_assignment_artifact(tmp_path):
    corpus = build_disk_backed_corpus(
        [CorpusDocument("Immutable assignment artifact text. " * 8, doc_id="immutable")],
        tmp_path / "base.sqlite3",
        min_chars=1,
        near_dedup=False,
        val_fraction=0,
    )
    shards = tmp_path / "shards"
    manifest = pack_disk_backed_shards(
        corpus,
        ByteTokenizer(),
        seq_len=32,
        shards_dir=str(shards),
    )
    assignment_path = shards / manifest["split_assignment"]["path"]
    payload = bytearray(assignment_path.read_bytes())
    payload[-2] ^= 1
    assignment_path.write_bytes(payload)

    with pytest.raises(ValueError, match="split assignment SHA-256 mismatch"):
        load_frozen_split_assignment_manifest(shards / "manifest.json")


def test_disk_backed_staging_replaces_existing_database_only_after_success(tmp_path):
    database = tmp_path / "corpus.sqlite3"
    database.write_bytes(b"previous-complete-artifact")

    def interrupted_documents():
        yield CorpusDocument(
            "A valid document that reaches the staging transaction. " * 8,
            doc_id="valid",
        )
        raise RuntimeError("simulated interrupted input")

    with pytest.raises(RuntimeError, match="simulated interrupted input"):
        build_disk_backed_corpus(
            interrupted_documents(),
            database,
            min_chars=1,
        )

    assert database.read_bytes() == b"previous-complete-artifact"
    assert not database.with_suffix(database.suffix + ".tmp").exists()


def test_disk_exact_dedup_retains_alias_provenance_and_fingerprint(tmp_path):
    text = "A normalized-content duplicate with auditable source aliases. " * 4
    canonical = CorpusDocument(
        text,
        source="hf://datasets/example/corpus/canonical",
        doc_id="a-canonical",
        license="MIT",
        meta={"repository": "canonical"},
    )
    alias = CorpusDocument(
        text,
        source="hf://datasets/example/corpus/alias",
        doc_id="z-alias",
        license="Apache-2.0",
        meta={"repository": "alias"},
    )
    docs = [alias, canonical, alias]
    corpus = build_disk_backed_corpus(
        docs,
        tmp_path / "aliases.sqlite3",
        min_chars=1,
        near_dedup=False,
        val_fraction=0,
    )

    audit = corpus.corpus_audit["quality_and_exact_deduplication"]
    provenance_audit = audit["exact_dedup_provenance"]
    assert audit["retained_documents"] == 1
    assert provenance_audit["alias_records"] == 1
    assert provenance_audit["alias_occurrences"] == 2
    assert provenance_audit["storage"] == "sqlite:exact_dedup_aliases"
    assert provenance_audit["constant_python_memory"] is True
    assert len(provenance_audit["alias_fingerprint_sha256"]) == 64

    records = list(corpus.iter_exact_dedup_aliases())
    assert {record["doc_id"] for record in records} == {"z-alias"}
    assert sum(record["occurrences"] for record in records) == 2
    assert not records[0]["matches_retained_provenance"]
    assert all("text" not in record for record in records)

    reversed_corpus = build_disk_backed_corpus(
        reversed(docs),
        tmp_path / "aliases-reversed.sqlite3",
        min_chars=1,
        near_dedup=False,
        val_fraction=0,
    )
    reversed_audit = reversed_corpus.corpus_audit["quality_and_exact_deduplication"][
        "exact_dedup_provenance"
    ]
    assert (
        reversed_audit["alias_fingerprint_sha256"] == provenance_audit["alias_fingerprint_sha256"]
    )


def test_disk_packing_interruption_keeps_previous_manifest_and_generation(tmp_path):
    docs = [
        CorpusDocument(
            f"transactional shard publication document {index} " * 8,
            source="family",
            doc_id=f"doc-{index}",
        )
        for index in range(3)
    ]
    corpus = build_disk_backed_corpus(
        docs,
        tmp_path / "corpus.sqlite3",
        min_chars=1,
        near_dedup=False,
        val_fraction=0,
    )
    shards = tmp_path / "shards"
    first = pack_disk_backed_shards(
        corpus,
        ByteTokenizer(),
        seq_len=32,
        shards_dir=str(shards),
        rows_per_shard=1,
    )
    committed_manifest = (shards / "manifest.json").read_bytes()

    class InterruptingTokenizer(ByteTokenizer):
        def __init__(self):
            self.calls = 0

        def encode(self, text: str, add_eos: bool = False) -> list[int]:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated packing interruption")
            return super().encode(text, add_eos=add_eos)

    with pytest.raises(RuntimeError, match="simulated packing interruption"):
        pack_disk_backed_shards(
            corpus,
            InterruptingTokenizer(),
            seq_len=32,
            shards_dir=str(shards),
            rows_per_shard=1,
        )

    assert (shards / "manifest.json").read_bytes() == committed_manifest
    assert (shards / "generations" / first["generation"]).is_dir()
    assert not list((shards / "generations").glob(".*.tmp"))
    assert len(PackedShardDataset(shards, "train")) == first["splits"]["train"]["rows"]


def test_manifest_commit_interruption_leaves_old_generation_readable(tmp_path, monkeypatch):
    corpus = build_disk_backed_corpus(
        [
            CorpusDocument("manifest commit document alpha " * 8, doc_id="alpha"),
            CorpusDocument("manifest commit document beta " * 8, doc_id="beta"),
        ],
        tmp_path / "commit.sqlite3",
        min_chars=1,
        near_dedup=False,
        val_fraction=0,
    )
    shards = tmp_path / "commit-shards"
    committed = pack_disk_backed_shards(
        corpus,
        ByteTokenizer(),
        seq_len=32,
        shards_dir=str(shards),
    )
    committed_manifest = (shards / "manifest.json").read_bytes()
    original_replace = Path.replace

    def interrupt_manifest_replace(path: Path, target: Path):
        if path.name.startswith(".manifest.") and Path(target).name == "manifest.json":
            raise RuntimeError("simulated manifest commit interruption")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", interrupt_manifest_replace)
    with pytest.raises(RuntimeError, match="simulated manifest commit interruption"):
        pack_disk_backed_shards(
            corpus,
            ByteTokenizer(),
            seq_len=32,
            shards_dir=str(shards),
        )

    assert (shards / "manifest.json").read_bytes() == committed_manifest
    # A crash in this narrow window may orphan a complete immutable generation. It is harmless:
    # the old manifest still points only at the old committed generation.
    generation_dirs = [path for path in (shards / "generations").iterdir() if path.is_dir()]
    assert len(generation_dirs) == 2
    assert len(PackedShardDataset(shards, "train")) == committed["splits"]["train"]["rows"]


def test_disk_manifest_groups_sources_and_licenses_with_bounded_memory(tmp_path):
    docs = [
        CorpusDocument(
            f"bounded provenance document {index} " * 4,
            source=f"hf://datasets/example/repository-{index}/row-{index}",
            doc_id=f"bounded-{index}",
            license=f"license-{index}",
            meta={"mixture_source": f"mixture-{index}"},
        )
        for index in range(pretrain_corpus.MAX_MANIFEST_GROUPS + 6)
    ]
    docs.extend(
        [
            CorpusDocument(
                "shared source family document alpha " * 4,
                source="hf://datasets/example/shared/row-alpha",
                doc_id="shared-alpha",
                license="MIT",
                meta={"mixture_source": "aaa-shared-family"},
            ),
            CorpusDocument(
                "shared source family document beta " * 4,
                source="hf://datasets/example/shared/row-beta",
                doc_id="shared-beta",
                license="MIT",
                meta={"mixture_source": "aaa-shared-family"},
            ),
        ]
    )
    corpus = build_disk_backed_corpus(
        docs,
        tmp_path / "bounded.sqlite3",
        min_chars=1,
        near_dedup=False,
        val_fraction=0.2,
        seed=29,
    )
    manifest = pack_disk_backed_shards(
        corpus,
        ByteTokenizer(),
        seq_len=32,
        shards_dir=str(tmp_path / "bounded-shards"),
        rows_per_shard=5,
    )

    assert len(manifest["source_counts"]) <= pretrain_corpus.MAX_MANIFEST_GROUPS + 1
    assert len(manifest["license_counts"]) <= pretrain_corpus.MAX_MANIFEST_GROUPS + 1
    assert sum(manifest["source_counts"].values()) == len(docs)
    assert sum(manifest["license_counts"].values()) == len(docs)
    assert manifest["source_counts"]["mixture:aaa-shared-family"] == 2
    assert manifest["provenance_summary"]["source_count_aggregation"]["overflow_documents"] > 0
    assert manifest["provenance_summary"]["license_count_aggregation"]["overflow_documents"] > 0
    fingerprint = manifest["provenance_summary"]["retained_document_provenance"]
    assert fingerprint["records"] == len(docs)
    assert len(fingerprint["sha256"]) == 64
    assert set(manifest["source_token_counts"]) == set(manifest["source_counts"])
    assert sum(manifest["source_token_counts"].values()) == manifest["total_tokens"]

    reversed_corpus = build_disk_backed_corpus(
        reversed(docs),
        tmp_path / "bounded-reversed.sqlite3",
        min_chars=1,
        near_dedup=False,
        val_fraction=0.2,
        seed=29,
    )
    reversed_manifest = pack_disk_backed_shards(
        reversed_corpus,
        ByteTokenizer(),
        seq_len=32,
        shards_dir=str(tmp_path / "bounded-reversed-shards"),
        rows_per_shard=5,
    )
    assert reversed_manifest["source_counts"] == manifest["source_counts"]
    assert reversed_manifest["license_counts"] == manifest["license_counts"]
    assert reversed_manifest["source_token_counts"] == manifest["source_token_counts"]
    assert reversed_manifest["provenance_summary"]["retained_document_provenance"] == fingerprint


def test_training_token_suggestion():
    assert suggested_training_tokens(1_000_000) == 20_000_000
