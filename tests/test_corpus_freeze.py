from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from localagent.data.corpus_freeze import (
    FREEZE_FORMAT,
    build_corpus_freeze,
    verify_corpus_freeze,
    write_corpus_freeze,
)
from localagent.data.hf_corpus import (
    acquisition_runtime_identity,
    build_mixture_plan,
    normalize_evaluation_decontamination,
)
from localagent.data.pretrain_corpus import (
    build_disk_backed_corpus,
    iter_documents,
    pack_disk_backed_shards,
    read_evaluation_denylist,
)
from localagent.model.tokenizer import ByteTokenizer, train_bpe
from localagent.train.pretrain import _verify_configured_corpus_freeze


def _identity(path: Path) -> dict[str, int | str]:
    return {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_yaml(path: Path, value: object) -> None:
    path.write_text(yaml.safe_dump(value), encoding="utf-8")


def _build_fixture(root: Path, monkeypatch, *, tokenizer_kind: str = "bpe") -> Path:
    monkeypatch.chdir(root)
    suite_path = Path("eval/local-eval.jsonl")
    suite_path.parent.mkdir(parents=True)
    suite_path.write_text(
        json.dumps({"prompt": "A frozen evaluation phrase absent from training."}) + "\n",
        encoding="utf-8",
    )
    suite_identity = _identity(suite_path)
    corpus_config = {
        "version": 1,
        "seed": 17,
        "target_chars": 20_000,
        "min_document_chars": 1,
        "max_document_chars": 1_000_000,
        "evaluation_decontamination": {
            "manifest_kind": "localagent_evaluation_denylist_manifest",
            "manifest_schema_version": 1,
            "required_suites": [
                {
                    "name": "local-eval",
                    **suite_identity,
                }
            ],
        },
        "sources": [
            {
                "name": "general",
                "dataset": "example/general",
                "revision": "a" * 40,
                "split": "train",
                "text_field": "text",
                "license": "MIT",
                "weight": 1.0,
            }
        ],
    }
    corpus_config_path = Path("configs/data/test-paper.yaml")
    corpus_config_path.parent.mkdir(parents=True)
    corpus_config_path.write_text(yaml.safe_dump(corpus_config), encoding="utf-8")

    raw_path = Path("data/raw/paper/mixture.jsonl")
    raw_path.parent.mkdir(parents=True)
    documents = [
        {
            "text": (
                f"Document {index} explains deterministic local model training. "
                f"Its distinct checksum marker is item-{index:02d}. "
                "The corpus sentence remains readable, licensed, and suitable for a unit test."
            ),
            "source": "hf://datasets/example/general",
            "doc_id": f"general:document-{index:02d}",
            "license": "mit",
            "meta": {
                "mixture_source": "general",
                "dataset": "example/general",
                "subset": None,
                "revision": "a" * 40,
            },
        }
        for index in range(16)
    ]
    raw_path.write_text(
        "".join(json.dumps(document, ensure_ascii=False) + "\n" for document in documents),
        encoding="utf-8",
    )
    raw_identity = _identity(raw_path)
    config_identity = _identity(corpus_config_path)
    acquisition_plan = build_mixture_plan(corpus_config_path)
    runtime = acquisition_runtime_identity()
    planned_source = acquisition_plan["sources"][0]
    source_stats = {
        "accepted_chars": sum(len(document["text"]) for document in documents),
        "accepted_documents": len(documents),
        "dataset": "example/general",
        "license_evidence": None,
        "normalized_weight": planned_source["normalized_weight"],
        "requested_chars": planned_source["requested_chars"],
        "revision": "a" * 40,
        "skipped": {},
        "stream_exhausted_before_budget": True,
        "subset": None,
        "weight": 1.0,
    }
    source_spool_path = Path("data/raw/paper/download_state/000-general.jsonl")
    source_spool_path.parent.mkdir(parents=True)
    source_spool_path.write_bytes(raw_path.read_bytes())
    source_spool_identity = _identity(source_spool_path)
    source_state = {
        "data_bytes": source_spool_identity["bytes"],
        "data_sha256": source_spool_identity["sha256"],
        "kind": "localagent_hf_mixture_source_state",
        "license_counts": {"mit": len(documents)},
        "plan_sha256": acquisition_plan["plan_sha256"],
        "runtime_sha256": runtime["runtime_sha256"],
        "source_index": 0,
        "source_plan_sha256": _canonical_sha256(planned_source),
        "stats": source_stats,
        "version": 1,
    }
    source_state["state_sha256"] = _canonical_sha256(source_state)
    source_state_path = Path("data/raw/paper/download_state/000-general.manifest.json")
    _write_json(source_state_path, source_state)
    source_state_identity = _identity(source_state_path)
    download_manifest = {
        "accepted_chars": source_stats["accepted_chars"],
        "accepted_documents": source_stats["accepted_documents"],
        "acquisition_plan": acquisition_plan,
        "config": str(corpus_config_path),
        "config_bytes": config_identity["bytes"],
        "config_sha256": config_identity["sha256"],
        "kind": "localagent_hf_mixture_download_manifest",
        "license_evidence": [],
        "plan_sha256": acquisition_plan["plan_sha256"],
        "raw_jsonl": str(raw_path),
        "raw_jsonl_bytes": raw_identity["bytes"],
        "raw_jsonl_sha256": raw_identity["sha256"],
        "requested_chars": 20_000,
        "runtime": runtime,
        "seed": 17,
        "license_counts": {"mit": len(documents)},
        "source_artifacts": [
            {
                "data_jsonl": {
                    "path": str(source_spool_path),
                    **source_spool_identity,
                },
                "name": "general",
                "source_index": 0,
                "state_manifest": {
                    "path": str(source_state_path),
                    **source_state_identity,
                    "state_sha256": source_state["state_sha256"],
                },
            }
        ],
        "sources": {"general": source_stats},
        "version": 2,
        "evaluation_decontamination": normalize_evaluation_decontamination(corpus_config),
    }
    download_manifest["manifest_sha256"] = _canonical_sha256(download_manifest)
    download_path = Path("data/raw/paper/download_manifest.json")
    _write_json(download_path, download_manifest)
    download_identity = _identity(download_path)

    shards = Path("data/shards/paper-all")
    staging = shards / "corpus-staging.sqlite3"
    corpus = build_disk_backed_corpus(
        iter_documents(raw_path),
        staging,
        min_chars=1,
        max_chars=1_000_000,
        denylist=read_evaluation_denylist(suite_path),
        near_dedup=False,
        val_fraction=0.25,
        seed=17,
    )
    filtered_path = shards / "filtered.jsonl"
    filtered_identity = corpus.write_filtered_jsonl(filtered_path)
    tokenizer_path: Path | None
    if tokenizer_kind == "byte":
        tokenizer_path = None
        tokenizer = ByteTokenizer()
        tokenizer_training = {
            "kind": "byte",
            "trained": False,
            "split": None,
        }
    else:
        assert tokenizer_kind == "bpe"
        tokenizer_path = Path("data/tokenizer-paper.json")
        tokenizer = train_bpe(
            (document.text for document in corpus.iter_documents("train")),
            tokenizer_path,
            vocab_size=320,
            min_frequency=1,
        )
        tokenizer_training = {
            "kind": "bpe",
            "trained": True,
            "split": "train",
            "path": str(tokenizer_path),
            "requested_vocab_size": 320,
            "vocab_size": tokenizer.vocab_size,
            "artifact": _identity(tokenizer_path),
        }
    canonical_inputs = [{"name": "local-eval", **suite_identity}]
    preparation_provenance = {
        "input_paths": [str(raw_path)],
        "included_mixture_sources": [],
        "source_manifests": [
            {
                "path": str(download_path),
                **download_identity,
            }
        ],
        "evaluation_denylist_paths": [f"local-eval={suite_path}"],
        "evaluation_denylist": {
            "inputs": [
                {
                    "name": "local-eval",
                    "path": str(suite_path),
                    **suite_identity,
                    "source": "cli",
                }
            ],
            "input_count": 1,
            "inputs_sha256": _canonical_sha256(canonical_inputs),
            "list_manifests": [],
            "required_suites": [{"name": "local-eval", **suite_identity}],
            "required_suite_policy_sources": [
                {
                    "path": str(download_path),
                    **download_identity,
                }
            ],
            "normalized_entries": 1,
        },
        "filtered_jsonl": filtered_identity,
    }
    manifest = pack_disk_backed_shards(
        corpus,
        tokenizer,
        32,
        str(shards),
        rows_per_shard=4,
        tokenizer_training=tokenizer_training,
        preparation_provenance=preparation_provenance,
    )
    assert manifest["splits"]["train"]["tokens"] >= 32

    model_config_path = Path("configs/model/test-paper.yaml")
    model_config_path.parent.mkdir(parents=True)
    model_config_path.write_text(
        yaml.safe_dump(
            {
                "name": "test-paper",
                "vocab_size": tokenizer.vocab_size,
                "max_seq_len": 32,
            }
        ),
        encoding="utf-8",
    )
    training_config_path = Path("configs/train/test-paper.yaml")
    training_config_path.parent.mkdir(parents=True)
    configured_tokenizer = {"kind": tokenizer_kind}
    if tokenizer_path is not None:
        configured_tokenizer["path"] = str(tokenizer_path)
    training_config_path.write_text(
        yaml.safe_dump(
            {
                "stage": "pretrain",
                "model_config": str(model_config_path),
                "data": {
                    "shards_dir": str(shards),
                    "min_train_tokens": 32,
                    "corpus_freeze": {
                        "spec": "configs/data/test-paper-freeze.yaml",
                        "path": "data/shards/paper-all/freeze.json",
                    },
                    "tokenizer": configured_tokenizer,
                },
                "schedule": {"total_steps": 1},
                "batch": {"micro_batch_size": 1, "grad_accum_steps": 1},
                "runtime": {"seed": 17},
            }
        ),
        encoding="utf-8",
    )
    spec_path = Path("configs/data/test-paper-freeze.yaml")
    tokenizer_spec = {
        "kind": tokenizer_kind,
        "vocab_size": tokenizer.vocab_size,
    }
    if tokenizer_path is not None:
        tokenizer_spec["path"] = str(tokenizer_path)
    spec_path.write_text(
        yaml.safe_dump(
            {
                "kind": "localagent_packed_corpus_freeze_spec",
                "schema_version": 1,
                "corpus_config": str(corpus_config_path),
                "shards_dir": str(shards),
                "freeze_path": "data/shards/paper-all/freeze.json",
                "tokenizer": tokenizer_spec,
                "expected": {
                    "seq_len": 32,
                    "min_train_tokens": 32,
                    "tokenizer_training_split": (None if tokenizer_kind == "byte" else "train"),
                },
                "training_configs": [str(training_config_path)],
            }
        ),
        encoding="utf-8",
    )
    return spec_path


def test_corpus_freeze_is_deterministic_and_verifiable(tmp_path, monkeypatch) -> None:
    spec_path = _build_fixture(tmp_path, monkeypatch)
    first = build_corpus_freeze(spec_path, project_root=".")
    second = build_corpus_freeze(spec_path, project_root=".")

    assert first == second
    assert str(tmp_path) not in json.dumps(first, sort_keys=True)
    assert first["format"] == FREEZE_FORMAT
    assert first["contract"]["minimum_train_tokens"] == 32
    assert first["contract"]["available_train_tokens"] >= 32
    assert first["split_assignment"]["overlap"] == {
        "identity_fingerprints": 0,
        "content_fingerprints": 0,
    }
    assert first["tokenizer"]["training_split"] == "train"
    assert [row["name"] for row in first["decontamination"]["required_suites"]] == ["local-eval"]

    freeze_path = Path("data/shards/paper-all/freeze.json")
    write_corpus_freeze(first, freeze_path)
    assert verify_corpus_freeze(freeze_path, spec_path, project_root=".") == first
    configured = {
        "corpus_freeze": {
            "spec": str(spec_path),
            "path": str(freeze_path),
        }
    }
    assert _verify_configured_corpus_freeze(configured, project_root=".") == {
        "path": str(freeze_path),
        "spec": str(spec_path),
        "sha256": first["freeze_sha256"],
    }


def test_byte_corpus_freeze_uses_intrinsic_tokenizer_contract(tmp_path, monkeypatch) -> None:
    spec_path = _build_fixture(tmp_path, monkeypatch, tokenizer_kind="byte")

    first = build_corpus_freeze(spec_path, project_root=".")
    second = build_corpus_freeze(spec_path, project_root=".")

    assert first == second
    assert first["contract"]["vocab_size"] == 256
    assert first["contract"]["tokenizer_training_split"] is None
    assert first["tokenizer"] == {
        "kind": "byte",
        "vocab_size": 256,
        "trained": False,
        "training_split": None,
        "training_documents": 0,
    }
    freeze_path = Path("data/shards/paper-all/freeze.json")
    write_corpus_freeze(first, freeze_path)
    assert verify_corpus_freeze(freeze_path, spec_path, project_root=".") == first


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("path", "data/invented-byte-tokenizer.json"),
        ("artifact", {"bytes": 0, "sha256": "0" * 64}),
    ],
)
def test_byte_freeze_spec_rejects_artifact_metadata(tmp_path, monkeypatch, field, value) -> None:
    spec_path = _build_fixture(tmp_path, monkeypatch, tokenizer_kind="byte")
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    spec["tokenizer"][field] = value
    _write_yaml(spec_path, spec)

    with pytest.raises(ValueError, match="intrinsic byte tokenizer must not declare"):
        build_corpus_freeze(spec_path, project_root=".")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("path", "data/invented-byte-tokenizer.json"),
        ("artifact", {"bytes": 0, "sha256": "0" * 64}),
    ],
)
def test_byte_manifest_rejects_artifact_metadata(tmp_path, monkeypatch, field, value) -> None:
    spec_path = _build_fixture(tmp_path, monkeypatch, tokenizer_kind="byte")
    manifest_path = Path("data/shards/paper-all/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tokenizer_training"][field] = value
    _write_json(manifest_path, manifest)

    with pytest.raises(
        ValueError,
        match="intrinsic byte tokenizer lineage must contain only",
    ):
        build_corpus_freeze(spec_path, project_root=".")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("spec_vocab", "vocabulary must be exactly 256"),
        ("manifest_vocab", "packed corpus vocabulary"),
        ("manifest_kind", "packed tokenizer kind"),
        ("manifest_trained", "must record trained=false"),
        ("manifest_split", "must record split=null"),
        ("manifest_documents", "must record zero training documents"),
        ("consumer_kind", "tokenizer kind is inconsistent"),
        ("consumer_path", "intrinsic byte tokenizer must not declare"),
        ("training_split", "byte tokenizer_training_split must be null"),
        ("missing_training_split", "must declare tokenizer_training_split"),
    ],
)
def test_byte_freeze_rejects_wrong_or_ambiguous_contracts(
    tmp_path, monkeypatch, mutation, match
) -> None:
    spec_path = _build_fixture(tmp_path, monkeypatch, tokenizer_kind="byte")
    manifest_path = Path("data/shards/paper-all/manifest.json")
    training_config_path = Path("configs/train/test-paper.yaml")
    if mutation == "spec_vocab":
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        spec["tokenizer"]["vocab_size"] = 255
        _write_yaml(spec_path, spec)
    elif mutation == "manifest_vocab":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["vocab_size"] = 255
        _write_json(manifest_path, manifest)
    elif mutation == "manifest_kind":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["tokenizer_training"]["kind"] = "bpe"
        _write_json(manifest_path, manifest)
    elif mutation == "manifest_trained":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["tokenizer_training"]["trained"] = True
        _write_json(manifest_path, manifest)
    elif mutation == "manifest_split":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["tokenizer_training"]["split"] = "train"
        _write_json(manifest_path, manifest)
    elif mutation == "manifest_documents":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["tokenizer_training"]["documents"] = 1
        _write_json(manifest_path, manifest)
    elif mutation == "consumer_kind":
        config = yaml.safe_load(training_config_path.read_text(encoding="utf-8"))
        config["data"]["tokenizer"]["kind"] = "bpe"
        _write_yaml(training_config_path, config)
    elif mutation == "consumer_path":
        config = yaml.safe_load(training_config_path.read_text(encoding="utf-8"))
        config["data"]["tokenizer"]["path"] = "data/invented-byte-tokenizer.json"
        _write_yaml(training_config_path, config)
    elif mutation == "training_split":
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        spec["expected"]["tokenizer_training_split"] = "train"
        _write_yaml(spec_path, spec)
    else:
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        del spec["expected"]["tokenizer_training_split"]
        _write_yaml(spec_path, spec)

    with pytest.raises(ValueError, match=match):
        build_corpus_freeze(spec_path, project_root=".")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_spec_path", "freeze tokenizer path must be a non-empty path string"),
        ("missing_manifest_artifact", "no content-addressed artifact"),
        ("not_locally_trained", "must record local split-only training"),
    ],
)
def test_bpe_freeze_keeps_artifact_and_split_training_strict(
    tmp_path, monkeypatch, mutation, match
) -> None:
    spec_path = _build_fixture(tmp_path, monkeypatch)
    manifest_path = Path("data/shards/paper-all/manifest.json")
    if mutation == "missing_spec_path":
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        del spec["tokenizer"]["path"]
        _write_yaml(spec_path, spec)
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if mutation == "missing_manifest_artifact":
            del manifest["tokenizer_training"]["artifact"]
        else:
            manifest["tokenizer_training"]["trained"] = False
        _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match=match):
        build_corpus_freeze(spec_path, project_root=".")


def test_configured_corpus_freeze_fails_closed_on_missing_or_tampered_record(
    tmp_path, monkeypatch
) -> None:
    spec_path = _build_fixture(tmp_path, monkeypatch)
    configured = {
        "corpus_freeze": {
            "spec": str(spec_path),
            "path": "data/shards/paper-all/freeze.json",
        }
    }
    with pytest.raises(ValueError, match="missing"):
        _verify_configured_corpus_freeze(configured, project_root=".")

    freeze_path = Path("data/shards/paper-all/freeze.json")
    write_corpus_freeze(build_corpus_freeze(spec_path, project_root="."), freeze_path)
    recorded = json.loads(freeze_path.read_text(encoding="utf-8"))
    recorded["contract"]["available_train_tokens"] += 1
    _write_json(freeze_path, recorded)
    with pytest.raises(ValueError, match="invalid self-hash"):
        _verify_configured_corpus_freeze(configured, project_root=".")


def test_corpus_freeze_rejects_cross_split_content_overlap(tmp_path, monkeypatch) -> None:
    spec_path = _build_fixture(tmp_path, monkeypatch)
    manifest_path = Path("data/shards/paper-all/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assignment_path = Path("data/shards/paper-all") / manifest["split_assignment"]["path"]
    lines = assignment_path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines[1:]]
    train_content = next(row["document_sha256"] for row in rows if row["split"] == "train")
    val_row = next(row for row in rows if row["split"] == "val")
    val_row["document_sha256"] = train_content
    assignment_path.write_text(
        json.dumps(json.loads(lines[0]), sort_keys=True, separators=(",", ":"))
        + "\n"
        + "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    semantic = hashlib.sha256(
        "\n".join(
            f"{row['identity_sha256']}:{row['document_sha256']}:{row['split']}" for row in rows
        ).encode("ascii")
    ).hexdigest()
    assignment_identity = _identity(assignment_path)
    manifest["split_assignment"]["bytes"] = assignment_identity["bytes"]
    manifest["split_assignment"]["sha256"] = assignment_identity["sha256"]
    manifest["split_assignment"]["assignment_sha256"] = semantic
    manifest["split_assignment_sha256"] = semantic
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="content fingerprint"):
        build_corpus_freeze(spec_path, project_root=".")


def test_paper_freeze_spec_matches_all_pretraining_consumers() -> None:
    root = Path(__file__).parents[1]
    spec = yaml.safe_load(
        (root / "configs/data/pretrain-paper-freeze.yaml").read_text(encoding="utf-8")
    )

    assert spec["expected"] == {
        "seq_len": 2048,
        "min_train_tokens": 171_409_408,
        "tokenizer_training_split": "train",
    }
    assert spec["freeze_path"] == "data/shards/paper-all/freeze.json"
    assert len(spec["training_configs"]) == 8
    for relative in spec["training_configs"]:
        config = yaml.safe_load((root / relative).read_text(encoding="utf-8"))
        scheduled = (
            config["schedule"]["total_steps"]
            * config["batch"]["micro_batch_size"]
            * config["batch"]["grad_accum_steps"]
            * spec["expected"]["seq_len"]
        )
        assert config["data"]["shards_dir"] == spec["shards_dir"]
        assert config["data"]["corpus_freeze"] == {
            "spec": "configs/data/pretrain-paper-freeze.yaml",
            "path": spec["freeze_path"],
        }
        assert config["data"]["tokenizer"] == {
            "kind": spec["tokenizer"]["kind"],
            "path": spec["tokenizer"]["path"],
        }
        assert config["data"]["min_train_tokens"] == 171_409_408
        if "5tpp" in relative:
            assert scheduled == 171_409_408
        else:
            assert scheduled > 171_409_408
