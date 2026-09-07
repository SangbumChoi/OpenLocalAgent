from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import openlocalagent.data.corpus.derived_corpus as derived
from openlocalagent.data.corpus.pretrain_corpus import (
    CorpusDocument,
    PackedShardDataset,
    build_disk_backed_corpus,
    load_frozen_split_assignment_manifest,
    pack_disk_backed_shards,
)
from openlocalagent.model.tokenizer import ByteTokenizer


def _identity(path: Path) -> dict[str, int | str]:
    payload = path.read_bytes()
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


@dataclass
class _Fixture:
    root: Path
    freeze_path: Path
    spec_path: Path
    filtered_path: Path
    manifest_path: Path
    tokenizer_path: Path
    freeze: dict[str, Any]
    documents: list[CorpusDocument]
    groups: dict[str, tuple[str, ...]]

    def outputs(self) -> list[Path]:
        return [self.root / output for output in self.groups]


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Fixture:
    sources = (
        "fineweb_edu_dedup",
        "cosmopedia_v2",
        "permissive_python",
        "structured_html",
    )
    documents = [
        CorpusDocument(
            text=(
                f"Retained parent document {index} from {source}. "
                "It has deterministic text for a lightweight derived corpus test."
            ),
            source=f"hf://datasets/fixture/{source}",
            doc_id=f"{source}:document-{index}",
            license="mit",
            meta={"mixture_source": source, "dataset": f"fixture/{source}"},
        )
        for index, source in enumerate((*sources, *sources))
    ]
    tokenizer_path = tmp_path / "tokenizer.byte"
    tokenizer_path.write_text("openlocalagent byte tokenizer identity\n", encoding="utf-8")
    tokenizer_identity = _identity(tokenizer_path)
    parent_dir = tmp_path / "parent"
    corpus = build_disk_backed_corpus(
        documents,
        parent_dir / "parent-staging.sqlite3",
        min_chars=1,
        max_chars=10_000,
        near_dedup=False,
        val_fraction=0.25,
        seed=17,
    )
    filtered_path = parent_dir / "filtered.jsonl"
    filtered_artifact = corpus.write_filtered_jsonl(filtered_path)
    documents = list(corpus.iter_documents())
    pack_disk_backed_shards(
        corpus,
        ByteTokenizer(),
        16,
        str(parent_dir),
        rows_per_shard=2,
        tokenizer_training={
            "kind": "byte",
            "vocab_size": 256,
            "artifact": tokenizer_identity,
            "trained": True,
            "split": "train",
        },
        preparation_provenance={"filtered_jsonl": filtered_artifact},
    )
    manifest_path = parent_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec_path = tmp_path / "freeze-spec.yaml"
    spec_path.write_text("kind: fixture-freeze-spec\n", encoding="utf-8")
    assignment = load_frozen_split_assignment_manifest(manifest_path)
    freeze = {
        "format": "localagent_packed_corpus_freeze",
        "schema_version": 1,
        "spec": _identity(spec_path),
        "contract": {
            "seq_len": 16,
            "vocab_size": 256,
            "tokenizer_training_split": "train",
        },
        "packed_corpus": {
            "manifest": {
                **_identity(manifest_path),
                "canonical_sha256": _canonical_sha256(manifest),
            },
            "generation": manifest["generation"],
            "seq_len": 16,
            "vocab_size": 256,
            "total_documents": len(documents),
            "source_counts": manifest["source_counts"],
        },
        "tokenizer": {
            "kind": "byte",
            "vocab_size": 256,
            "artifact": tokenizer_identity,
        },
        "split_assignment": {
            "artifact": {
                "bytes": assignment.bytes,
                "sha256": assignment.sha256,
            },
            "assignment_sha256": assignment.assignment_sha256,
            "records": assignment.records,
            "seed": assignment.seed,
            "val_fraction": assignment.val_fraction,
        },
        "decontamination": {
            "required_suites": [{"name": "fixture-heldout", "sha256": "a" * 64}],
            "retained_documents": len(documents),
        },
        "quality_and_exact_deduplication": corpus.corpus_audit["quality_and_exact_deduplication"],
        "provenance": {
            "filtered_jsonl": _identity(filtered_path),
            "staging_database": manifest["preparation"]["staging_database"],
        },
    }
    freeze["freeze_sha256"] = _canonical_sha256(freeze)
    freeze_path = parent_dir / "freeze.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(derived, "verify_corpus_freeze", lambda *args, **kwargs: freeze)
    return _Fixture(
        root=tmp_path,
        freeze_path=freeze_path,
        spec_path=spec_path,
        filtered_path=filtered_path,
        manifest_path=manifest_path,
        tokenizer_path=tokenizer_path,
        freeze=freeze,
        documents=documents,
        groups={
            "children/general": ("fineweb_edu_dedup", "cosmopedia_v2"),
            "children/code": ("permissive_python",),
            "children/structured": ("structured_html",),
        },
    )


def _prepare(fixture: _Fixture, **overrides: Any) -> dict[str, dict[str, Any]]:
    arguments: dict[str, Any] = {
        "freeze_path": fixture.freeze_path,
        "spec_path": fixture.spec_path,
        "parent_filtered_jsonl": fixture.filtered_path,
        "parent_manifest": fixture.manifest_path,
        "tokenizer_path": fixture.tokenizer_path,
        "groups": fixture.groups,
        "project_root": fixture.root,
        "rows_per_shard": 2,
    }
    arguments.update(overrides)
    return derived.build_derived_corpora(**arguments)


def _rewrite_mock_freeze(fixture: _Fixture) -> None:
    unsigned = dict(fixture.freeze)
    unsigned.pop("freeze_sha256", None)
    fixture.freeze["freeze_sha256"] = _canonical_sha256(unsigned)
    fixture.freeze_path.write_text(
        json.dumps(fixture.freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _assignment_rows(path: Path) -> set[tuple[str, str, str]]:
    assignment = load_frozen_split_assignment_manifest(path / "manifest.json")
    rows: set[tuple[str, str, str]] = set()
    with assignment.path.open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            row = json.loads(line)
            rows.add((row["document_id"], row["document_sha256"], row["split"]))
    return rows


def test_parent_freeze_fanout_is_one_pass_disjoint_and_split_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    traversals = 0
    parent_assignment_loads = 0
    original_iterator = derived._iter_canonical_documents
    original_assignment_loader = derived.load_frozen_split_assignment_manifest

    def counted_iterator(*args: Any, **kwargs: Any):
        nonlocal traversals
        traversals += 1
        yield from original_iterator(*args, **kwargs)

    def counted_assignment_loader(path: str | Path):
        nonlocal parent_assignment_loads
        if Path(path) == fixture.manifest_path:
            parent_assignment_loads += 1
        return original_assignment_loader(path)

    monkeypatch.setattr(derived, "_iter_canonical_documents", counted_iterator)
    monkeypatch.setattr(
        derived,
        "load_frozen_split_assignment_manifest",
        counted_assignment_loader,
    )
    import openlocalagent.data.decontam.evaluation_denylist_suite as denylist_suite

    def unexpected_suite_call(*args: Any, **kwargs: Any) -> None:
        pytest.fail("derived fan-out invoked an external denylist suite operation")

    monkeypatch.setattr(
        denylist_suite,
        "freeze_evaluation_denylist_suite",
        unexpected_suite_call,
    )
    monkeypatch.setattr(
        denylist_suite,
        "verify_evaluation_denylist_suite",
        unexpected_suite_call,
    )

    manifests = _prepare(fixture)

    assert traversals == 1
    assert parent_assignment_loads == 1
    assert sum(manifest["total_documents"] for manifest in manifests.values()) == len(
        fixture.documents
    )
    parent_rows = _assignment_rows(fixture.manifest_path.parent)
    child_rows: set[tuple[str, str, str]] = set()
    for relative_output, sources in fixture.groups.items():
        output = fixture.root / relative_output
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        assert str(fixture.root) not in json.dumps(manifest, sort_keys=True)
        expected_ids = {
            document.doc_id
            for document in fixture.documents
            if document.meta["mixture_source"] in sources
        }
        rows = _assignment_rows(output)
        assert {row[0] for row in rows} == expected_ids
        assert not child_rows & rows
        child_rows.update(rows)
        assert manifest["derived_corpus"]["no_rescreen"]["performed"] is False
        assert (
            manifest["derived_corpus"]["inherited_decontamination_audit"]
            == fixture.freeze["decontamination"]
        )
        assert manifest["derived_corpus"]["fanout"]["parent_filtered_decodes"] == 1
        publication = manifest["derived_corpus"]["publication"]
        assert publication["coordination"] == "cooperative_exclusive_lock_files"
        assert publication["commit"] == "no_replace_manifests_last_replayable"
        assert publication["crash_atomic_multi_group"] is False
        assert "never delete public destination names" in publication["failure_state"]
        assert "never replace or remove" in publication["lock_contract"]
        assert "manually remove stale lock files" in publication["recovery"]
        assert manifest["tokenizer_training"]["artifact"] == _identity(fixture.tokenizer_path)
        PackedShardDataset(output, "train")
        if manifest["splits"]["val"]["documents"]:
            PackedShardDataset(output, "val")
    assert child_rows == parent_rows


@pytest.mark.parametrize(
    "failure",
    ["tamper", "relabel", "truncate", "tokenizer", "manifest", "split"],
)
def test_parent_artifact_drift_and_family_relabel_fail_without_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    if failure == "tamper":
        fixture.filtered_path.write_bytes(fixture.filtered_path.read_bytes() + b"\n")
    elif failure == "relabel":
        rows = [
            json.loads(line)
            for line in fixture.filtered_path.read_text(encoding="utf-8").splitlines()
        ]
        rows[0]["meta"]["mixture_source"] = "permissive_python"
        fixture.filtered_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        fixture.freeze["provenance"]["filtered_jsonl"] = _identity(fixture.filtered_path)
        _rewrite_mock_freeze(fixture)
    elif failure == "truncate":
        lines = fixture.filtered_path.read_text(encoding="utf-8").splitlines(keepends=True)
        fixture.filtered_path.write_text("".join(lines[:-1]), encoding="utf-8")
        fixture.freeze["provenance"]["filtered_jsonl"] = _identity(fixture.filtered_path)
        _rewrite_mock_freeze(fixture)
    elif failure == "tokenizer":
        fixture.tokenizer_path.write_text("tampered tokenizer\n", encoding="utf-8")
    elif failure == "manifest":
        fixture.manifest_path.write_bytes(fixture.manifest_path.read_bytes() + b" ")
    else:
        manifest = json.loads(fixture.manifest_path.read_text(encoding="utf-8"))
        assignment_path = fixture.manifest_path.parent / manifest["split_assignment"]["path"]
        assignment_path.write_bytes(assignment_path.read_bytes() + b" ")

    with pytest.raises(ValueError):
        _prepare(fixture)
    assert all(not output.exists() for output in fixture.outputs())


def test_freeze_failure_is_fail_closed_before_any_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        derived,
        "verify_corpus_freeze",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("freeze drift")),
    )
    with pytest.raises(ValueError, match="freeze drift"):
        _prepare(fixture)
    assert all(not output.exists() for output in fixture.outputs())


def test_split_artifact_swap_after_verification_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    original_loader = derived.load_frozen_split_assignment_manifest
    swapped = False

    def swapping_loader(path: str | Path):
        nonlocal swapped
        assignment = original_loader(path)
        if Path(path) == fixture.manifest_path and not swapped:
            swapped = True
            assignment.path.write_bytes(assignment.path.read_bytes() + b" ")
        return assignment

    monkeypatch.setattr(
        derived,
        "load_frozen_split_assignment_manifest",
        swapping_loader,
    )
    with pytest.raises(ValueError, match="split assignment"):
        _prepare(fixture)
    assert swapped is True
    assert all(not output.exists() for output in fixture.outputs())


def test_tokenizer_loader_only_consumes_the_verified_private_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    original_bytes = fixture.tokenizer_path.read_bytes()
    original_identity = _identity(fixture.tokenizer_path)
    original_loader = derived.load_tokenizer
    loaded_paths: list[Path] = []

    def swapping_loader(kind: str, path: str | Path):
        snapshot = Path(path)
        loaded_paths.append(snapshot)
        fixture.tokenizer_path.write_bytes(b"unverified same-vocabulary tokenizer bytes\n")
        assert snapshot != fixture.tokenizer_path
        assert snapshot.read_bytes() == original_bytes
        return original_loader(kind, snapshot)

    monkeypatch.setattr(derived, "load_tokenizer", swapping_loader)
    manifests = _prepare(fixture)

    assert len(loaded_paths) == 1
    assert fixture.tokenizer_path.read_bytes() != original_bytes
    assert all(
        manifest["tokenizer"]["sha256"] == original_identity["sha256"]
        and manifest["tokenizer"]["bytes"] == original_identity["bytes"]
        for manifest in manifests.values()
    )


def test_equal_count_family_swap_is_rejected_by_frozen_staging_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    rows = [
        json.loads(line) for line in fixture.filtered_path.read_text(encoding="utf-8").splitlines()
    ]
    first = next(
        index
        for index, row in enumerate(rows)
        if row["meta"]["mixture_source"] == "fineweb_edu_dedup"
    )
    second = next(
        index for index, row in enumerate(rows) if row["meta"]["mixture_source"] == "cosmopedia_v2"
    )
    rows[first]["meta"]["mixture_source"], rows[second]["meta"]["mixture_source"] = (
        rows[second]["meta"]["mixture_source"],
        rows[first]["meta"]["mixture_source"],
    )
    fixture.filtered_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    fixture.freeze["provenance"]["filtered_jsonl"] = _identity(fixture.filtered_path)
    _rewrite_mock_freeze(fixture)

    with pytest.raises(ValueError, match="relabeled"):
        _prepare(fixture)
    assert all(not output.exists() for output in fixture.outputs())


def test_complete_parent_default_and_explicit_partial_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    partial_groups = {
        "children/general": ("fineweb_edu_dedup", "cosmopedia_v2"),
    }
    with pytest.raises(ValueError, match="do not assign every parent"):
        _prepare(fixture, groups=partial_groups)
    assert not (tmp_path / "children/general").exists()

    manifests = _prepare(
        fixture,
        groups=partial_groups,
        require_complete_parent=False,
    )
    only_manifest = next(iter(manifests.values()))
    assert only_manifest["total_documents"] == 4
    assert (
        only_manifest["derived_corpus"]["fanout"]["complete_parent_required_by_invocation"] is False
    )


def test_duplicate_group_ownership_and_unknown_source_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="multiple derived groups"):
        _prepare(
            fixture,
            groups={
                "children/a": ("fineweb_edu_dedup",),
                "children/b": ("fineweb_edu_dedup",),
            },
        )
    with pytest.raises(ValueError, match="absent from the parent freeze"):
        _prepare(
            fixture,
            groups={
                **fixture.groups,
                "children/unknown": ("not_a_parent_source",),
            },
        )


def test_output_paths_are_canonicalized_and_symlink_ancestors_fail_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="contain the project root"):
        derived.normalize_group_definitions(
            {"children/..": ("fineweb_edu_dedup",)},
            project_root=tmp_path,
        )
    with pytest.raises(ValueError, match="output is repeated"):
        derived.normalize_group_definitions(
            {
                "children/a/../b": ("fineweb_edu_dedup",),
                "children/b": ("cosmopedia_v2",),
            },
            project_root=tmp_path,
        )

    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic-link ancestor"):
        derived.normalize_group_definitions(
            {"linked/output": ("fineweb_edu_dedup",)},
            project_root=tmp_path,
        )


def test_direct_group_mapping_accepts_cli_delimiter_characters_in_source_names(
    tmp_path: Path,
) -> None:
    definitions = derived.normalize_group_definitions(
        {"children/direct": ("family+plus", "family=equals")},
        project_root=tmp_path,
    )
    assert definitions[0].mixture_sources == ("family+plus", "family=equals")


@pytest.mark.parametrize("failure", ["connect", "schema"])
def test_group_stage_setup_failure_removes_its_private_temp_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    definition = derived.DerivedCorpusGroup(
        output_dir=tmp_path / "children" / "group",
        mixture_sources=("source",),
        logical_name="children/group",
    )
    original_connect = derived.sqlite3.connect

    if failure == "connect":

        def failing_connect(*args: Any, **kwargs: Any):
            raise sqlite3.OperationalError("injected connect failure")

        monkeypatch.setattr(derived.sqlite3, "connect", failing_connect)
    else:

        class FailingSchemaConnection:
            def __init__(self, connection: sqlite3.Connection) -> None:
                self.connection = connection

            @property
            def row_factory(self) -> Any:
                return self.connection.row_factory

            @row_factory.setter
            def row_factory(self, value: Any) -> None:
                self.connection.row_factory = value

            def execute(self, *args: Any, **kwargs: Any) -> Any:
                return self.connection.execute(*args, **kwargs)

            def executescript(self, script: str) -> None:
                raise sqlite3.OperationalError("injected schema failure")

            def close(self) -> None:
                self.connection.close()

        def failing_schema_connect(*args: Any, **kwargs: Any) -> FailingSchemaConnection:
            return FailingSchemaConnection(original_connect(*args, **kwargs))

        monkeypatch.setattr(derived.sqlite3, "connect", failing_schema_connect)

    with pytest.raises(sqlite3.OperationalError, match=failure):
        derived._create_group_stage(definition, expected_documents=1)
    assert not list(definition.output_dir.parent.glob(".group.derived.*"))


def test_read_only_sqlite_uri_handles_reserved_filename_characters(tmp_path: Path) -> None:
    database = tmp_path / "parent?#%.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE witness (value TEXT NOT NULL)")
    connection.execute("INSERT INTO witness VALUES ('verified')")
    connection.commit()
    connection.close()

    read_only = derived._open_sqlite_read_only(database)
    try:
        assert read_only.execute("SELECT value FROM witness").fetchone() == ("verified",)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            read_only.execute("INSERT INTO witness VALUES ('forbidden')")
    finally:
        read_only.close()


def test_tree_identity_tracks_empty_directories_and_rejects_special_nodes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    baseline = derived._tree_identity(root)
    (root / "empty").mkdir()
    assert derived._tree_identity(root) != baseline

    symlink = root / "link"
    symlink.symlink_to(root / "empty", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink or special node"):
        derived._tree_identity(root)
    symlink.unlink()

    fifo = root / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="symlink or special node"):
        derived._tree_identity(root)


def test_pack_failure_publishes_no_partial_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    original_pack = derived.pack_disk_backed_shards
    calls = 0

    def failing_pack(*args: Any, **kwargs: Any):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("injected pack failure")
        return original_pack(*args, **kwargs)

    monkeypatch.setattr(derived, "pack_disk_backed_shards", failing_pack)
    with pytest.raises(ValueError, match="injected pack failure"):
        _prepare(fixture)
    assert calls == 2
    assert all(not output.exists() for output in fixture.outputs())


def test_publication_lock_fails_closed_without_touching_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    children = tmp_path / "children"
    children.mkdir()
    lock = children / ".code.derived-corpus.lock"
    lock.write_text("another publisher\n", encoding="utf-8")

    with pytest.raises(ValueError, match="locked by another publisher"):
        _prepare(fixture)
    assert all(not output.exists() for output in fixture.outputs())
    assert lock.read_text(encoding="utf-8") == "another publisher\n"


@pytest.mark.parametrize("failure", ["write", "fsync"])
def test_publication_lock_setup_failure_removes_only_its_own_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    original_write = derived.os.write
    original_fsync = derived.os.fsync
    lock_descriptors: set[int] = set()

    def tracked_write(descriptor: int, data: Any) -> int:
        if isinstance(data, bytes) and data.startswith(b"pid="):
            lock_descriptors.add(descriptor)
            if failure == "write":
                raise OSError("injected lock write failure")
        return original_write(descriptor, data)

    def tracked_fsync(descriptor: int) -> None:
        if failure == "fsync" and descriptor in lock_descriptors:
            raise OSError("injected lock fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(derived.os, "write", tracked_write)
    monkeypatch.setattr(derived.os, "fsync", tracked_fsync)
    with pytest.raises(OSError, match=f"lock {failure} failure"):
        _prepare(fixture)

    assert all(not output.exists() for output in fixture.outputs())
    assert not list((tmp_path / "children").glob(".*.derived-corpus.lock"))


def test_lock_cleanup_preserves_a_replacement_racing_the_retirement_move(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    original_rename = derived.os.rename
    foreign = b"foreign lock replacement\n"
    replaced_lock: Path | None = None

    def racing_rename(
        source: str | Path,
        destination: str | Path,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        nonlocal replaced_lock
        source_path = Path(source)
        if replaced_lock is None and source_path.name.endswith(".derived-corpus.lock"):
            owned_backup = source_path.parent / f"{source_path.name}.owned-backup"
            original_rename(source, owned_backup)
            source_path.write_bytes(foreign)
            replaced_lock = source_path
        original_rename(source, destination, *args, **kwargs)

    monkeypatch.setattr(derived.os, "rename", racing_rename)
    _prepare(fixture)

    assert replaced_lock is not None
    assert replaced_lock.read_bytes() == foreign
    quarantined = [
        path
        for path in replaced_lock.parent.rglob("candidate")
        if path.parent.name.startswith(f".{replaced_lock.name}.cleanup.")
    ]
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == foreign


def test_manifest_publication_failure_leaves_only_replayable_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    original_link = derived.os.link
    manifest_links = 0

    def failing_link(
        source: str | Path,
        destination: str | Path,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        nonlocal manifest_links
        if str(destination) == "manifest.json":
            manifest_links += 1
            if manifest_links == 2:
                raise OSError("injected manifest link failure")
        original_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(derived.os, "link", failing_link)
    with pytest.raises(OSError, match="injected manifest link failure"):
        _prepare(fixture)
    assert manifest_links == 2
    assert all(output.is_dir() for output in fixture.outputs())
    assert sum((output / "manifest.json").is_file() for output in fixture.outputs()) == 1
    assert not list((tmp_path / "children").glob(".*.derived-corpus.lock"))

    _prepare(fixture)
    assert all((output / "manifest.json").is_file() for output in fixture.outputs())


def test_failure_abandonment_does_not_delete_a_last_moment_foreign_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    original_link = derived.os.link
    original_abandon = derived._abandon_publication
    foreign = b"foreign replacement installed as failure begins\n"
    replacement: Path | None = None

    def failing_link(
        source: str | Path,
        destination: str | Path,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if str(destination) == "manifest.json":
            raise OSError("injected manifest link failure")
        original_link(source, destination, *args, **kwargs)

    def replacing_abandonment(state: derived._PublicationState) -> None:
        nonlocal replacement
        if replacement is None:
            replacement = state.stage.definition.output_dir / "derived-staging.sqlite3"
            replacement.unlink()
            replacement.write_bytes(foreign)
        original_abandon(state)

    monkeypatch.setattr(derived.os, "link", failing_link)
    monkeypatch.setattr(derived, "_abandon_publication", replacing_abandonment)
    with pytest.raises(OSError, match="injected manifest link failure"):
        _prepare(fixture)

    assert replacement is not None
    assert replacement.read_bytes() == foreign
    assert not (replacement.parent / "manifest.json").exists()


def test_publication_never_overwrites_or_removes_a_racing_foreign_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    original_link = derived.os.link
    foreign = b"foreign writer owns these bytes\n"
    injected = False

    def racing_link(
        source: str | Path,
        destination: str | Path,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        nonlocal injected
        if not injected and str(destination) == "derived-staging.sqlite3":
            injected = True
            descriptor = os.open(
                str(destination),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
                dir_fd=kwargs["dst_dir_fd"],
            )
            try:
                os.write(descriptor, foreign)
            finally:
                os.close(descriptor)
        original_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(derived.os, "link", racing_link)
    with pytest.raises(ValueError, match="differs or was swapped"):
        _prepare(fixture)

    target = tmp_path / "children" / "code" / "derived-staging.sqlite3"
    assert injected is True
    assert target.read_bytes() == foreign
    assert not (target.parent / "manifest.json").exists()


def test_publication_never_follows_or_removes_a_racing_foreign_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    original_link = derived.os.link
    foreign = tmp_path / "foreign-token.bin"
    foreign.write_bytes(b"foreign symlink target\n")
    injected = False

    def racing_link(
        source: str | Path,
        destination: str | Path,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        nonlocal injected
        if not injected and str(destination) == "derived-staging.sqlite3":
            injected = True
            os.symlink(foreign, str(destination), dir_fd=kwargs["dst_dir_fd"])
        original_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(derived.os, "link", racing_link)
    with pytest.raises(ValueError, match="symbolic link"):
        _prepare(fixture)

    target = tmp_path / "children" / "code" / "derived-staging.sqlite3"
    assert injected is True
    assert target.is_symlink()
    assert foreign.read_bytes() == b"foreign symlink target\n"
    assert not (target.parent / "manifest.json").exists()


def test_publication_directory_swap_preserves_the_foreign_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    original_populate = derived._populate_publication_directory
    foreign = b"foreign replacement directory\n"
    orphan: Path | None = None

    def swapping_populate(
        state: derived._PublicationState,
        source: Path,
        target: derived._PinnedDirectory,
        *,
        skip_manifest: bool,
    ) -> None:
        nonlocal orphan
        if orphan is None:
            output = state.stage.definition.output_dir
            orphan = output.parent / f".{output.name}.publisher-orphan"
            output.rename(orphan)
            output.mkdir()
            (output / "foreign.bin").write_bytes(foreign)
        original_populate(state, source, target, skip_manifest=skip_manifest)

    monkeypatch.setattr(derived, "_populate_publication_directory", swapping_populate)
    with pytest.raises(ValueError, match="identity mismatch"):
        _prepare(fixture)

    replacement = tmp_path / "children" / "code"
    assert (replacement / "foreign.bin").read_bytes() == foreign
    assert not (replacement / "manifest.json").exists()
    assert orphan is not None and orphan.is_dir()


def test_child_split_membership_mismatch_is_rejected_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    original_pack = derived.pack_disk_backed_shards

    def mismatched_pack(*args: Any, **kwargs: Any) -> dict[str, Any]:
        manifest = original_pack(*args, **kwargs)
        manifest["splits"]["train"]["documents"] += 1
        return manifest

    monkeypatch.setattr(derived, "pack_disk_backed_shards", mismatched_pack)
    with pytest.raises(ValueError, match="membership count"):
        _prepare(fixture)
    assert all(not output.exists() for output in fixture.outputs())


def test_manifest_replay_is_deterministic_and_existing_drift_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    first = _prepare(fixture)
    first_bytes = {output: (Path(output) / "manifest.json").read_bytes() for output in first}
    second = _prepare(fixture)
    assert set(second) == set(first)
    assert {
        output: (Path(output) / "manifest.json").read_bytes() for output in second
    } == first_bytes

    drifted = fixture.root / "children/code/manifest.json"
    drifted.write_bytes(drifted.read_bytes() + b" ")
    unaffected = fixture.root / "children/general/manifest.json"
    unaffected_before = unaffected.read_bytes()
    with pytest.raises(ValueError, match="existing derived output differs"):
        _prepare(fixture)
    assert unaffected.read_bytes() == unaffected_before


def test_manifestless_exact_partial_group_is_repaired_by_deterministic_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    _prepare(fixture)
    output = fixture.root / "children/code"
    expected_tree = derived._tree_identity(output)
    manifest = output / "manifest.json"
    manifest_bytes = manifest.read_bytes()
    manifest.unlink()

    assert not manifest.exists()
    _prepare(fixture)

    assert manifest.read_bytes() == manifest_bytes
    assert derived._tree_identity(output) == expected_tree


def test_replay_rejects_an_extra_empty_directory_without_removing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    _prepare(fixture)
    extra = fixture.root / "children/code/foreign-empty"
    extra.mkdir()

    with pytest.raises(ValueError, match="existing derived output differs"):
        _prepare(fixture)
    assert extra.is_dir()
