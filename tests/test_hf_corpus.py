import gzip
import hashlib
import json
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

import yaml

import openlocalagent.data.corpus.hf_corpus as hf_corpus
from openlocalagent.data.corpus.hf_corpus import (
    _load_raw_parquet_text,
    _load_raw_jsonl_gzip,
    _selected_raw_files,
    audit_mixture_readiness,
    build_mixture_plan,
    normalize_evaluation_decontamination,
    stream_mixture,
)


def test_stream_mixture_enforces_license_and_records_provenance(tmp_path):
    config = {
        "seed": 7,
        "target_chars": 400,
        "min_document_chars": 20,
        "sources": [
            {
                "name": "text",
                "dataset": "example/text",
                "revision": "a" * 40,
                "text_field": "text",
                "license": "ODC-By-1.0",
                "weight": 1,
                "source_fields": ["id"],
            },
            {
                "name": "code",
                "dataset": "example/code",
                "revision": "b" * 40,
                "text_field": "content",
                "license_field": "license",
                "allowed_licenses": ["mit"],
                "weight": 1,
                "source_fields": ["repo", "path"],
            },
        ],
    }
    config_path = tmp_path / "mixture.yaml"
    config_path.write_text(yaml.safe_dump(config))
    rows = {
        "text": [{"id": "a", "text": "educational language " * 20}],
        "code": [
            {
                "repo": "bad",
                "path": "x.py",
                "content": "print('copyleft') " * 20,
                "license": "gpl-3.0",
            },
            {
                "repo": "good",
                "path": "y.py",
                "content": "def useful_function(): return 42\n" * 20,
                "license": "MIT",
            },
        ],
    }

    def loader(source, seed):
        assert seed in (7, 8)
        assert len(source["revision"]) == 40
        return rows[source["name"]]

    manifest = stream_mixture(config_path, tmp_path / "out", loader=loader)
    documents = [
        json.loads(line)
        for line in (tmp_path / "out" / "mixture.jsonl").read_text().splitlines()
    ]

    assert manifest["accepted_documents"] == 2
    assert manifest["license_counts"] == {"mit": 1, "odc-by-1.0": 1}
    assert manifest["sources"]["code"]["skipped"]["license"] == 1
    assert manifest["sources"]["text"]["revision"] == "a" * 40
    raw_path = tmp_path / "out" / "mixture.jsonl"
    assert manifest["raw_jsonl_bytes"] == raw_path.stat().st_size
    assert manifest["raw_jsonl_sha256"] == hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert documents[1]["source"].endswith("example/code/good/y.py")
    assert documents[1]["meta"]["mixture_source"] == "code"
    assert documents[1]["meta"]["revision"] == "b" * 40


def test_stream_mixture_carries_config_owned_decontamination_policy(tmp_path):
    config = {
        "seed": 3,
        "target_chars": 40,
        "min_document_chars": 1,
        "evaluation_decontamination": {
            "manifest_kind": "localagent_evaluation_denylist_manifest",
            "manifest_schema_version": 1,
            "required_suites": [
                {"name": "bfcl"},
                {
                    "name": "local-agent-eval",
                    "bytes": 123,
                    "sha256": "a" * 64,
                },
            ],
        },
        "sources": [
            {
                "name": "text",
                "dataset": "example/text",
                "revision": "a" * 40,
                "text_field": "text",
                "license": "MIT",
                "weight": 1,
            }
        ],
    }
    config_path = tmp_path / "mixture.yaml"
    config_path.write_text(yaml.safe_dump(config))
    manifest = stream_mixture(
        config_path,
        tmp_path / "out",
        loader=lambda *_: [{"text": "auditable corpus row " * 3}],
    )
    assert manifest["evaluation_decontamination"] == (
        normalize_evaluation_decontamination(config)
    )
    assert manifest["config_bytes"] == config_path.stat().st_size


def test_paper_policy_requires_frozen_agent_and_tracked_local_suites():
    root = Path(__file__).parents[1]
    config = yaml.safe_load(
        (root / "configs/data/pretrain-paper.yaml").read_text()
    )
    policy = normalize_evaluation_decontamination(config)
    assert policy is not None
    suites = {suite["name"]: suite for suite in policy["required_suites"]}
    assert set(suites) == {
        "bfcl",
        "browsergym",
        "local-agent-eval",
        "local-browser-tasks",
        "local-realtime-actions",
        "mind2web",
        "weblinx",
    }
    local_paths = {
        "local-browser-tasks": (
            root / "demos/webgpu/browser-task-cases.json"
        ),
        "local-realtime-actions": (
            root / "demos/webgpu/benchmark-cases.json"
        ),
    }
    for name, path in local_paths.items():
        assert suites[name]["bytes"] == path.stat().st_size
        assert suites[name]["sha256"] == hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    assert suites["local-agent-eval"]["bytes"] > 0
    assert len(suites["local-agent-eval"]["sha256"]) == 64

    plan = build_mixture_plan(root / "configs/data/pretrain-paper.yaml")
    assert [source["requested_chars"] for source in plan["sources"]] == [
        1_100_000_000,
        330_000_000,
        550_000_000,
        220_000_000,
    ]
    assert sum(source["requested_chars"] for source in plan["sources"]) == 2_200_000_000
    assert {item["id"] for item in plan["license_evidence"]} == {
        "codeparrot-card",
        "smollm-card",
        "websight-card",
    }
    assert plan["require_full_source_budgets"] is True
    assert plan["storage"] == {
        "max_raw_jsonl_bytes": 13_200_000_000,
        "minimum_free_bytes": 60_000_000_000,
    }
    code_source = plan["sources"][2]
    assert code_source["raw_stream"]["backend"] == "hf-jsonl-gzip-v1"
    assert len(code_source["raw_stream"]["file_inventory"]["files"]) == 48
    assert code_source["raw_stream"]["file_inventory"]["total_compressed_bytes"] == 11_501_025_766
    assert [
        item["path"] for item in _selected_raw_files(code_source, plan["seed"] + 2)
    ] == [
        "file-000000000016.json.gz",
        "file-000000000002.json.gz",
        "file-000000000014.json.gz",
        "file-000000000034.json.gz",
    ]
    html_source = plan["sources"][3]
    assert html_source["subset"] == "v0.2"
    assert html_source["raw_stream"]["backend"] == "hf-parquet-text-v1"
    assert html_source["raw_stream"]["reader_runtime"] == {
        "library": "pyarrow",
        "version": "25.0.0",
    }
    inventory = html_source["raw_stream"]["file_inventory"]
    assert inventory["bytes"] == 141_431
    assert inventory["sha256"] == (
        "16c6db1cd43843a1d5f852d2e676b984d2b651fedab5db9fc08606641f99412b"
    )
    assert inventory["manifest_sha256"] == (
        "cf331bfe2fb13628487b9dd078aaf2adc2dd2fbae1ea8138e8214627ffca85c5"
    )
    assert inventory["shard_count"] == 738
    assert inventory["total_artifact_bytes"] == 285_785_923_573
    selected_html = _selected_raw_files(html_source, plan["seed"] + 3)
    assert len(selected_html) == 64
    assert sum(item["bytes"] for item in selected_html) == 25_243_071_432
    assert [item["path"] for item in selected_html[:8]] == [
        "v0.2/train-00109-of-00738-b8b861dc04695181.parquet",
        "v0.2/train-00082-of-00738-69973ef14928e568.parquet",
        "v0.2/train-00400-of-00738-dc2f03d7d0db093f.parquet",
        "v0.2/train-00275-of-00738-b7d629db0646f03f.parquet",
        "v0.2/train-00591-of-00738-6777aeee7499115a.parquet",
        "v0.2/train-00202-of-00738-f2e985be28875937.parquet",
        "v0.2/train-00130-of-00738-07859e5316ba3d6b.parquet",
        "v0.2/train-00542-of-00738-f7949ba50dd19e17.parquet",
    ]


def test_stream_mixture_rejects_unpinned_source(tmp_path):
    config_path = tmp_path / "mixture.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "target_chars": 100,
                "sources": [
                    {
                        "name": "floating",
                        "dataset": "example/floating",
                        "text_field": "text",
                        "weight": 1,
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="full 40-character"):
        stream_mixture(config_path, tmp_path / "out", loader=lambda *_: [])


def test_plan_apportions_every_character_deterministically(tmp_path):
    config = {
        "target_chars": 10,
        "min_document_chars": 1,
        "sources": [
            {
                "name": name,
                "dataset": f"example/{name}",
                "revision": str(index + 1) * 40,
                "license": "MIT",
                "weight": 1,
            }
            for index, name in enumerate(("alpha", "beta", "gamma"))
        ],
    }
    config_path = tmp_path / "mixture.yaml"
    config_path.write_text(yaml.safe_dump(config))

    first = build_mixture_plan(config_path)
    second = build_mixture_plan(config_path)

    assert first == second
    assert [source["requested_chars"] for source in first["sources"]] == [4, 3, 3]
    assert sum(source["requested_chars"] for source in first["sources"]) == 10
    assert first["plan_sha256"] == second["plan_sha256"]


@pytest.mark.parametrize(
    ("source_updates", "message"),
    [
        ({"license": "MIT", "license_field": "license"}, "exactly one"),
        ({"license": None, "license_field": "license"}, "allowed_licenses"),
        ({"license": "unknown"}, "must not be unknown"),
    ],
)
def test_plan_rejects_ambiguous_or_unbounded_license_policy(
    tmp_path,
    source_updates,
    message,
):
    source = {
        "name": "source",
        "dataset": "example/source",
        "revision": "a" * 40,
        "license": "MIT",
        "weight": 1,
    }
    source.update(source_updates)
    config_path = tmp_path / "mixture.yaml"
    config_path.write_text(
        yaml.safe_dump({"target_chars": 10, "sources": [source]})
    )

    with pytest.raises(ValueError, match=message):
        build_mixture_plan(config_path)


def _write_raw_stream_fixture(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    files: dict[str, Path] = {}
    inventory = []
    for file_index in range(1, 4):
        name = f"file-{file_index:012d}.json.gz"
        path = tmp_path / name
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for row_index in range(2):
                row = {
                    "content": f"file {file_index} row {row_index}",
                    "license": "mit",
                    "path": f"{file_index}/{row_index}.py",
                    "repo_name": f"repo-{file_index}",
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        files[name] = path
        payload = path.read_bytes()
        inventory.append(
            {
                "bytes": len(payload),
                "path": name,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )

    revision = "a" * 40
    unsigned_manifest = {
        "api_url": (
            "https://huggingface.co/api/datasets/example/raw/tree/"
            f"{revision}?recursive=true&expand=true"
        ),
        "dataset": "example/raw",
        "files": inventory,
        "kind": "localagent_hf_raw_jsonl_gzip_file_manifest",
        "revision": revision,
        "version": 1,
    }
    manifest = {
        **unsigned_manifest,
        "manifest_sha256": hashlib.sha256(
            json.dumps(
                unsigned_manifest,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
    }
    manifest_path = tmp_path / "raw-files.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path, files


def test_pinned_raw_jsonl_gzip_loader_verifies_selects_and_interleaves(
    tmp_path,
    monkeypatch,
):
    manifest_path, files = _write_raw_stream_fixture(tmp_path)
    manifest_payload = manifest_path.read_bytes()
    config = {
        "target_chars": 10,
        "min_document_chars": 1,
        "sources": [
            {
                "dataset": "example/raw",
                "license_field": "license",
                "allowed_licenses": ["mit"],
                "name": "raw",
                "raw_stream": {
                    "backend": "hf-jsonl-gzip-v1",
                    "files_manifest": manifest_path.name,
                    "files_manifest_bytes": len(manifest_payload),
                    "files_manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
                    "interleave_files": 2,
                    "selection": "sha256-seed-path-v1",
                },
                "revision": "a" * 40,
                "shuffle_buffer": 1,
                "text_field": "content",
                "weight": 1,
            }
        ],
    }
    config_path = tmp_path / "mixture.yaml"
    config_path.write_text(yaml.safe_dump(config))
    source = build_mixture_plan(config_path)["sources"][0]
    selected = _selected_raw_files(source, 19)
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(files[kwargs["filename"]])

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    rows = list(_load_raw_jsonl_gzip(source, 19))

    assert [call["filename"] for call in calls] == [
        item["path"] for item in selected
    ]
    assert all(call["revision"] == "a" * 40 for call in calls)
    assert all(call["repo_type"] == "dataset" for call in calls)
    expected_contents = []
    for row_index in range(2):
        for item in selected:
            file_index = int(item["path"].split("-")[1].split(".")[0])
            expected_contents.append(f"file {file_index} row {row_index}")
    assert [row["content"] for row in rows] == expected_contents


def test_raw_stream_manifest_is_bound_by_bytes_and_self_hash(tmp_path):
    manifest_path, _ = _write_raw_stream_fixture(tmp_path)
    payload = manifest_path.read_bytes()
    base_source = {
        "dataset": "example/raw",
        "license": "MIT",
        "name": "raw",
        "raw_stream": {
            "backend": "hf-jsonl-gzip-v1",
            "files_manifest": manifest_path.name,
            "files_manifest_bytes": len(payload),
            "files_manifest_sha256": hashlib.sha256(payload).hexdigest(),
            "interleave_files": 2,
            "selection": "sha256-seed-path-v1",
        },
        "revision": "a" * 40,
        "weight": 1,
    }
    config_path = tmp_path / "mixture.yaml"
    config_path.write_text(
        yaml.safe_dump({"target_chars": 10, "sources": [base_source]})
    )
    assert build_mixture_plan(config_path)["sources"][0]["raw_stream"] is not None

    tampered = json.loads(manifest_path.read_text())
    tampered["files"][0]["bytes"] += 1
    manifest_path.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n")
    tampered_payload = manifest_path.read_bytes()
    base_source["raw_stream"]["files_manifest_bytes"] = len(tampered_payload)
    base_source["raw_stream"]["files_manifest_sha256"] = hashlib.sha256(
        tampered_payload
    ).hexdigest()
    config_path.write_text(
        yaml.safe_dump({"target_chars": 10, "sources": [base_source]})
    )
    with pytest.raises(ValueError, match="self-hash mismatch"):
        build_mixture_plan(config_path)


def _parquet_schema_contract() -> dict:
    return {
        "fields": [
            {
                "name": "image",
                "nullable": True,
                "type": {
                    "fields": [
                        {"name": "bytes", "nullable": True, "type": "binary"},
                        {"name": "path", "nullable": True, "type": "string"},
                    ],
                    "kind": "struct",
                },
            },
            {"name": "text", "nullable": True, "type": "string"},
            {"name": "llm_generated_idea", "nullable": True, "type": "string"},
        ],
        "text_field": "text",
    }


def _write_self_hashed_parquet_manifest(path: Path, manifest: dict) -> bytes:
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    payload = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    path.write_bytes(payload)
    return payload


def _pin_manifest_in_config(config_path: Path, manifest_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text())
    payload = manifest_path.read_bytes()
    raw_stream = config["sources"][0]["raw_stream"]
    raw_stream["files_manifest_bytes"] = len(payload)
    raw_stream["files_manifest_sha256"] = hashlib.sha256(payload).hexdigest()
    config_path.write_text(yaml.safe_dump(config))


def _write_parquet_stream_fixture(
    tmp_path: Path,
    *,
    schema_drift: bool = False,
) -> tuple[Path, Path, dict[str, Path]]:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    subset = "v0.2"
    split = "train"
    revision = "a" * 40
    shard_count = 3
    shard_dir = tmp_path / subset
    shard_dir.mkdir()
    files: dict[str, Path] = {}
    inventory = []
    image_type = pa.struct(
        [
            pa.field("bytes", pa.binary(), nullable=True),
            pa.field("path", pa.string(), nullable=True),
        ]
    )
    for file_index in range(shard_count):
        relative = (
            f"{subset}/{split}-{file_index:05d}-of-{shard_count:05d}-"
            f"{file_index:016x}.parquet"
        )
        path = tmp_path / relative
        text_name = "html" if schema_drift else "text"
        schema = pa.schema(
            [
                pa.field("image", image_type, nullable=True),
                pa.field(text_name, pa.string(), nullable=True),
                pa.field("llm_generated_idea", pa.string(), nullable=True),
            ]
        )
        table = pa.Table.from_arrays(
            [
                pa.array(
                    [
                        {
                            "bytes": f"image-poison-{file_index}-{row_index}".encode(),
                            "path": f"{file_index}/{row_index}.png",
                        }
                        for row_index in range(2)
                    ],
                    type=image_type,
                ),
                pa.array(
                    [
                        f"file {file_index} row {row_index} <html>auditable text</html>"
                        for row_index in range(2)
                    ]
                ),
                pa.array(
                    [
                        f"idea-poison-{file_index}-{row_index}"
                        for row_index in range(2)
                    ]
                ),
            ],
            schema=schema,
        )
        pq.write_table(table, path, row_group_size=1, use_dictionary=False)
        files[relative] = path
        payload = path.read_bytes()
        inventory.append(
            {
                "bytes": len(payload),
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )

    manifest = {
        "api_url": (
            "https://huggingface.co/api/datasets/example/raw/tree/"
            f"{revision}/{subset}?recursive=true&expand=true"
        ),
        "dataset": "example/raw",
        "files": inventory,
        "kind": "localagent_hf_raw_parquet_file_manifest",
        "parquet_schema": _parquet_schema_contract(),
        "reader_runtime": {
            "library": "pyarrow",
            "version": importlib_metadata.version("pyarrow"),
        },
        "revision": revision,
        "shard_count": shard_count,
        "split": split,
        "subset": subset,
        "total_bytes": sum(item["bytes"] for item in inventory),
        "version": 1,
    }
    manifest_path = tmp_path / "parquet-files.json"
    manifest_payload = _write_self_hashed_parquet_manifest(manifest_path, manifest)
    config = {
        "target_chars": 10,
        "min_document_chars": 1,
        "sources": [
            {
                "dataset": "example/raw",
                "license": "MIT",
                "name": "raw",
                "raw_stream": {
                    "backend": "hf-parquet-text-v1",
                    "files_manifest": manifest_path.name,
                    "files_manifest_bytes": len(manifest_payload),
                    "files_manifest_sha256": hashlib.sha256(
                        manifest_payload
                    ).hexdigest(),
                    "interleave_files": 2,
                    "selection": "sha256-seed-path-v1",
                },
                "revision": revision,
                "shuffle_buffer": 1,
                "split": split,
                "subset": subset,
                "text_field": "text",
                "weight": 1,
            }
        ],
    }
    config_path = tmp_path / "mixture.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path, manifest_path, files


def test_parquet_manifest_is_canonical_self_hashed_and_strict(tmp_path):
    config_path, manifest_path, _ = _write_parquet_stream_fixture(tmp_path)
    original = json.loads(manifest_path.read_text())
    assert build_mixture_plan(config_path)["sources"][0]["raw_stream"] is not None

    stale_self_hash = json.loads(json.dumps(original))
    stale_self_hash["files"][0]["bytes"] += 1
    manifest_path.write_text(json.dumps(stale_self_hash, indent=2, sort_keys=True) + "\n")
    _pin_manifest_in_config(config_path, manifest_path)
    with pytest.raises(ValueError, match="self-hash mismatch"):
        build_mixture_plan(config_path)

    compact = json.loads(json.dumps(original))
    compact_payload = (
        json.dumps(compact, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    manifest_path.write_bytes(compact_payload)
    _pin_manifest_in_config(config_path, manifest_path)
    with pytest.raises(ValueError, match="canonical pretty JSON"):
        build_mixture_plan(config_path)

    traversal = json.loads(json.dumps(original))
    traversal["files"][0]["path"] = "../outside.parquet"
    _write_self_hashed_parquet_manifest(manifest_path, traversal)
    _pin_manifest_in_config(config_path, manifest_path)
    with pytest.raises(ValueError, match="invalid path"):
        build_mixture_plan(config_path)

    duplicate_hash = json.loads(json.dumps(original))
    duplicate_hash["files"][1]["sha256"] = duplicate_hash["files"][0]["sha256"]
    _write_self_hashed_parquet_manifest(manifest_path, duplicate_hash)
    _pin_manifest_in_config(config_path, manifest_path)
    with pytest.raises(ValueError, match="hashes must be unique"):
        build_mixture_plan(config_path)

    schema_drift = json.loads(json.dumps(original))
    schema_drift["parquet_schema"]["fields"][1]["name"] = "html"
    _write_self_hashed_parquet_manifest(manifest_path, schema_drift)
    _pin_manifest_in_config(config_path, manifest_path)
    with pytest.raises(ValueError, match="schema contract mismatch"):
        build_mixture_plan(config_path)


def test_pinned_parquet_loader_selects_interleaves_and_projects_only_text(
    tmp_path,
    monkeypatch,
):
    config_path, _, files = _write_parquet_stream_fixture(tmp_path)
    source = build_mixture_plan(config_path)["sources"][0]
    selected = _selected_raw_files(source, 19)
    download_calls = []

    def fake_download(**kwargs):
        download_calls.append(kwargs)
        return str(files[kwargs["filename"]])

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    pq = pytest.importorskip("pyarrow.parquet")
    parquet_file = pq.ParquetFile
    projections = []

    class TrackingParquetFile:
        def __init__(self, *args, **kwargs):
            self._inner = parquet_file(*args, **kwargs)

        @property
        def schema_arrow(self):
            return self._inner.schema_arrow

        def iter_batches(self, **kwargs):
            projections.append(kwargs)
            return self._inner.iter_batches(**kwargs)

    monkeypatch.setattr(pq, "ParquetFile", TrackingParquetFile)
    rows = list(_load_raw_parquet_text(source, 19))

    assert [call["filename"] for call in download_calls] == [
        item["path"] for item in selected
    ]
    assert all(call["revision"] == "a" * 40 for call in download_calls)
    assert all(call["repo_type"] == "dataset" for call in download_calls)
    expected_texts = []
    for row_index in range(2):
        for item in selected:
            file_index = int(item["path"].split("train-")[1].split("-")[0])
            expected_texts.append(
                f"file {file_index} row {row_index} <html>auditable text</html>"
            )
    assert rows == [{"text": text} for text in expected_texts]
    assert projections
    assert all(call["columns"] == ["text"] for call in projections)
    assert all(call["use_threads"] is False for call in projections)
    assert all(set(row) == {"text"} for row in rows)
    assert not any("poison" in row["text"] for row in rows)

    manifest = stream_mixture(config_path, tmp_path / "out")
    stats = manifest["sources"]["raw"]["raw_stream"]
    selected_for_plan = _selected_raw_files(source, 42)
    assert stats["selected_file_count"] == 2
    assert stats["selected_total_bytes"] == sum(
        item["bytes"] for item in selected_for_plan
    )
    assert stats["selected_files"] == selected_for_plan
    assert stats["projection"] == {
        "columns": ["text"],
        "materializes_only_configured_text": True,
    }


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("bytes", "byte-size mismatch"),
        ("sha256", "SHA-256 mismatch"),
    ],
)
def test_pinned_parquet_loader_rejects_artifact_drift(
    tmp_path,
    monkeypatch,
    corruption,
    message,
):
    config_path, _, files = _write_parquet_stream_fixture(tmp_path)
    source = build_mixture_plan(config_path)["sources"][0]
    selected = _selected_raw_files(source, 19)
    selected_path = files[selected[0]["path"]]
    payload = bytearray(selected_path.read_bytes())
    if corruption == "bytes":
        payload.append(0)
    else:
        payload[0] ^= 1
    selected_path.write_bytes(payload)

    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda **kwargs: str(files[kwargs["filename"]]),
    )
    with pytest.raises(RuntimeError, match=message):
        list(_load_raw_parquet_text(source, 19))


def test_pinned_parquet_loader_rejects_runtime_and_schema_drift(tmp_path, monkeypatch):
    config_path, _, files = _write_parquet_stream_fixture(tmp_path, schema_drift=True)
    source = build_mixture_plan(config_path)["sources"][0]
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda **kwargs: str(files[kwargs["filename"]]),
    )
    with pytest.raises(RuntimeError, match="schema mismatch"):
        list(_load_raw_parquet_text(source, 19))

    actual_version = importlib_metadata.version

    def drifted_version(name):
        return "0.0.0" if name == "pyarrow" else actual_version(name)

    monkeypatch.setattr(hf_corpus.importlib_metadata, "version", drifted_version)
    with pytest.raises(RuntimeError, match="requires pyarrow"):
        list(_load_raw_parquet_text(source, 19))


def test_paper_readiness_checks_pinned_parquet_runtime_and_disk_floor(
    tmp_path,
    monkeypatch,
):
    root = Path(__file__).parents[1]
    plan = build_mixture_plan(root / "configs/data/pretrain-paper.yaml")
    evidence = {}
    fixture_evidence = {}
    for item in plan["license_evidence"]:
        evidence_id = item["id"]
        payload = f"fixture license evidence for {evidence_id}\n".encode()
        evidence_path = tmp_path / f"{evidence_id}.md"
        evidence_path.write_bytes(payload)
        evidence[evidence_id] = evidence_path
        fixture_evidence[evidence_id] = {
            **item,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    plan["license_evidence"] = [
        fixture_evidence[item["id"]] for item in plan["license_evidence"]
    ]
    for source in plan["sources"]:
        if source["license_evidence"] is not None:
            source["license_evidence"] = fixture_evidence[
                source["license_evidence"]["id"]
            ]
    plan["plan_sha256"] = hf_corpus._canonical_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    runtime = {
        "packages": {
            "PyYAML": "6.0.2",
            "datasets": "5.0.0",
            "fsspec": "2026.5.0",
            "huggingface_hub": "1.25.1",
            "pyarrow": "25.0.0",
        },
        "runtime_sha256": "a" * 64,
    }
    monkeypatch.setattr(hf_corpus, "acquisition_runtime_identity", lambda: runtime)
    monkeypatch.setattr(
        hf_corpus.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(
            total=100_000_000_000,
            used=40_000_000_000,
            free=60_000_000_000,
        ),
    )
    ready = audit_mixture_readiness(
        plan,
        tmp_path / "not-created",
        license_evidence=evidence,
        require_stream_runtime=True,
    )
    assert ready["ready"] is True
    assert ready["disk"]["minimum_free_bytes"] == 60_000_000_000

    runtime["packages"]["pyarrow"] = "24.0.0"
    drifted = audit_mixture_readiness(
        plan,
        tmp_path / "not-created",
        license_evidence=evidence,
        require_stream_runtime=True,
    )
    assert drifted["ready"] is False
    assert "requires pyarrow 25.0.0, found 24.0.0" in drifted["blockers"][0]


def test_readiness_verifies_pinned_license_evidence_without_streaming(tmp_path):
    evidence = b"license: mit\nrevision-bound evidence\n"
    evidence_path = tmp_path / "card.md"
    evidence_path.write_bytes(evidence)
    revision = "a" * 40
    config = {
        "target_chars": 10,
        "min_document_chars": 1,
        "require_license_evidence": True,
        "sources": [
            {
                "name": "source",
                "dataset": "example/source",
                "revision": revision,
                "license": "MIT",
                "license_evidence": {
                    "id": "source-card",
                    "url": (
                        "https://huggingface.co/datasets/example/source/resolve/"
                        f"{revision}/README.md"
                    ),
                    "bytes": len(evidence),
                    "sha256": hashlib.sha256(evidence).hexdigest(),
                    "scope": "dataset-distribution",
                },
                "weight": 1,
            }
        ],
    }
    config_path = tmp_path / "mixture.yaml"
    config_path.write_text(yaml.safe_dump(config))
    plan = build_mixture_plan(config_path)

    missing = audit_mixture_readiness(plan, tmp_path / "not-created")
    verified = audit_mixture_readiness(
        plan,
        tmp_path / "not-created",
        license_evidence={"source-card": evidence_path},
    )

    assert missing["ready"] is False
    assert missing["blockers"] == ["missing license evidence 'source-card'"]
    assert verified["ready"] is True
    assert verified["license_evidence"][0]["status"] == "verified"
    assert not (tmp_path / "not-created").exists()


def test_required_source_budget_fails_without_publishing_partial_data(tmp_path):
    config = {
        "target_chars": 100,
        "min_document_chars": 1,
        "require_full_source_budgets": True,
        "sources": [
            {
                "name": "short",
                "dataset": "example/short",
                "revision": "a" * 40,
                "license": "MIT",
                "weight": 1,
            }
        ],
    }
    config_path = tmp_path / "mixture.yaml"
    config_path.write_text(yaml.safe_dump(config))
    out = tmp_path / "out"

    with pytest.raises(RuntimeError, match="exhausted 90 characters"):
        stream_mixture(
            config_path,
            out,
            loader=lambda *_: [{"text": "0123456789"}],
        )

    assert not (out / "mixture.jsonl").exists()
    assert not (out / "download_manifest.json").exists()
    assert (out / "download_state/000-short.jsonl.tmp").is_file()


def test_resume_reuses_verified_completed_sources_and_replays_partial_source(tmp_path):
    config = {
        "target_chars": 80,
        "min_document_chars": 1,
        "require_full_source_budgets": True,
        "sources": [
            {
                "name": name,
                "dataset": f"example/{name}",
                "revision": revision * 40,
                "license": "MIT",
                "weight": 1,
            }
            for name, revision in (("alpha", "a"), ("beta", "b"))
        ],
    }
    config_path = tmp_path / "mixture.yaml"
    config_path.write_text(yaml.safe_dump(config))
    out = tmp_path / "out"

    def interrupted_loader(source, _seed):
        if source["name"] == "alpha":
            return [{"text": "alpha " * 10}]
        raise OSError("simulated interrupted stream")

    with pytest.raises(OSError, match="simulated"):
        stream_mixture(config_path, out, loader=interrupted_loader)

    resumed_calls = []

    def resumed_loader(source, _seed):
        resumed_calls.append(source["name"])
        assert source["name"] == "beta"
        return [{"text": "beta " * 10}]

    manifest = stream_mixture(
        config_path,
        out,
        resume=True,
        loader=resumed_loader,
    )

    assert resumed_calls == ["beta"]
    assert manifest["accepted_documents"] == 2
    assert manifest["sources"]["alpha"]["accepted_chars"] == 59
    assert manifest["sources"]["beta"]["accepted_chars"] == 49
    assert manifest["raw_jsonl_sha256"] == hashlib.sha256(
        (out / "mixture.jsonl").read_bytes()
    ).hexdigest()
    assert len(manifest["manifest_sha256"]) == 64
    assert [artifact["name"] for artifact in manifest["source_artifacts"]] == [
        "alpha",
        "beta",
    ]
    for artifact in manifest["source_artifacts"]:
        for key in ("data_jsonl", "state_manifest"):
            path = Path(artifact[key]["path"])
            assert artifact[key]["bytes"] == path.stat().st_size
            assert artifact[key]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="already exists"):
        stream_mixture(config_path, out, loader=lambda *_: [])
    assert stream_mixture(
        config_path,
        out,
        resume=True,
        loader=lambda *_: pytest.fail("completed download should be reused"),
    ) == manifest

    manifest_path = out / "download_manifest.json"
    original_manifest_payload = manifest_path.read_bytes()
    missing_admission = json.loads(original_manifest_payload)
    missing_admission.pop("storage_admission")
    missing_admission.pop("manifest_sha256")
    missing_admission["manifest_sha256"] = hf_corpus._canonical_sha256(
        missing_admission
    )
    manifest_path.write_text(json.dumps(missing_admission))
    with pytest.raises(RuntimeError, match="storage-admission identity"):
        stream_mixture(config_path, out, resume=True, loader=lambda *_: [])

    manifest_path.write_bytes(original_manifest_payload)
    tampered = json.loads(manifest_path.read_text())
    tampered["accepted_documents"] = 999
    manifest_path.write_text(json.dumps(tampered))
    with pytest.raises(RuntimeError, match="self-hash mismatch"):
        stream_mixture(config_path, out, resume=True, loader=lambda *_: [])


def test_resume_uses_plan_bound_storage_admission_after_initial_headroom_is_consumed(
    tmp_path,
    monkeypatch,
):
    config = {
        "target_chars": 80,
        "min_document_chars": 1,
        "require_full_source_budgets": True,
        "storage": {
            "max_raw_jsonl_bytes": 1000,
            "minimum_free_bytes": 2000,
        },
        "sources": [
            {
                "name": name,
                "dataset": f"example/{name}",
                "revision": revision * 40,
                "license": "MIT",
                "weight": 1,
            }
            for name, revision in (("alpha", "a"), ("beta", "b"))
        ],
    }
    config_path = tmp_path / "mixture.yaml"
    config_path.write_text(yaml.safe_dump(config))
    out = tmp_path / "out"
    free_bytes = 2000
    monkeypatch.setattr(
        hf_corpus.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(
            total=2000,
            used=2000 - free_bytes,
            free=free_bytes,
        ),
    )

    def interrupted_loader(source, _seed):
        if source["name"] == "alpha":
            return [{"text": "alpha " * 10}]
        raise OSError("simulated interrupted stream")

    with pytest.raises(OSError, match="simulated"):
        stream_mixture(config_path, out, loader=interrupted_loader)

    admission_path = out / "storage_admission.json"
    admission = json.loads(admission_path.read_text())
    assert admission["available_free_bytes_at_admission"] == 2000
    assert admission["minimum_free_bytes"] == 2000
    assert admission["filesystem_device"] == out.stat().st_dev

    plan = build_mixture_plan(config_path)
    original_admission_payload = admission_path.read_bytes()
    other_device = dict(admission)
    other_device["filesystem_device"] += 1
    other_device.pop("admission_sha256")
    other_device["admission_sha256"] = hf_corpus._canonical_sha256(other_device)
    admission_path.write_text(json.dumps(other_device))
    wrong_filesystem = audit_mixture_readiness(plan, out, resume=True)
    assert wrong_filesystem["ready"] is False
    assert "filesystem device disagrees" in wrong_filesystem["blockers"][0]
    admission_path.write_bytes(original_admission_payload)

    free_bytes = 0
    blocked = audit_mixture_readiness(plan, out, resume=True)
    assert blocked["ready"] is False
    assert blocked["disk"]["minimum_free_bytes"] > 0
    assert any("insufficient free disk" in item for item in blocked["blockers"])

    free_bytes = 1900
    readiness = audit_mixture_readiness(plan, out, resume=True)
    assert readiness["ready"] is True
    assert readiness["disk"]["configured_minimum_free_bytes"] == 2000
    assert 0 < readiness["disk"]["minimum_free_bytes"] < 2000
    assert readiness["disk"]["mode"] == "resume_admission"
    assert readiness["disk"]["remaining_capacity"]["completed_source_count"] == 1

    manifest = stream_mixture(
        config_path,
        out,
        resume=True,
        loader=lambda source, _seed: [{"text": f"{source['name']} " * 10}],
    )
    assert manifest["storage_admission"]["admission_sha256"] == (
        admission["admission_sha256"]
    )
    completed = audit_mixture_readiness(plan, out, resume=True)
    assert completed["ready"] is True
    assert completed["disk"]["mode"] == "completed_verification"
    assert stream_mixture(
        config_path,
        out,
        resume=True,
        loader=lambda *_: pytest.fail("completed download should be reused"),
    ) == manifest


def test_opened_raw_artifact_is_the_verified_regular_file(tmp_path):
    expected = tmp_path / "selected.parquet"
    expected.write_bytes(b"verified bytes")
    item = {
        "bytes": expected.stat().st_size,
        "path": "v0.2/train-00000-of-00001-0000000000000000.parquet",
        "sha256": hashlib.sha256(expected.read_bytes()).hexdigest(),
    }
    with hf_corpus._open_verified_raw_artifact(expected, item) as handle:
        assert handle.read() == b"verified bytes"

    blob = tmp_path / "blob.parquet"
    blob.write_bytes(b"verified bytes")
    expected.unlink()
    expected.symlink_to(blob)
    with hf_corpus._open_verified_raw_artifact(expected, item) as handle:
        assert handle.read() == b"verified bytes"

    replacement = tmp_path / "replacement.parquet"
    replacement.write_bytes(b"tampered bytes")
    expected.unlink()
    expected.symlink_to(replacement)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        hf_corpus._open_verified_raw_artifact(expected, item)


def test_raw_byte_cap_aborts_before_publishing(tmp_path):
    config = {
        "target_chars": 10,
        "min_document_chars": 1,
        "storage": {"max_raw_jsonl_bytes": 10},
        "sources": [
            {
                "name": "source",
                "dataset": "example/source",
                "revision": "a" * 40,
                "license": "MIT",
                "weight": 1,
            }
        ],
    }
    config_path = tmp_path / "mixture.yaml"
    config_path.write_text(yaml.safe_dump(config))
    out = tmp_path / "out"

    with pytest.raises(RuntimeError, match="max_raw_jsonl_bytes"):
        stream_mixture(
            config_path,
            out,
            loader=lambda *_: [{"text": "0123456789"}],
        )
    assert not (out / "mixture.jsonl").exists()
    assert not (out / "download_manifest.json").exists()
