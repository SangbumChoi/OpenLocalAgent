import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from localagent.eval.external_action_contract import (
    HARDENED_SCHEMA_VERSION,
    TRAINING_LINEAGE_KIND,
    action_template_sha256,
    freeze_external_action_slice,
    normalize_prompt,
    validate_frozen_slice,
)
from localagent.eval.realtime import paired_clustered_exact_action_delta_ci


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_bytes(value))


def _identity(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _tool() -> dict:
    return {
        "name": "archive_file",
        "description": "Archive one file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "format": "path"}},
            "required": ["path"],
        },
    }


def _conversation(prompt: str, path: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": prompt, "tool_calls": [], "tool_response": None},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": "archive_file", "arguments": {"path": path}}],
                "tool_response": None,
            },
        ],
        "tools": [_tool()],
        "meta": {"split": "train"},
    }


def _source(case_count: int = 8) -> dict:
    cases = []
    phrasings = (
        "Archive {path}.",
        "Put {path} in the archive.",
        "Store {path} safely.",
        "Move {path} into archival storage.",
    )
    for index in range(case_count):
        path = f"external/task-{index}.txt"
        template_index = index % len(phrasings)
        cases.append(
            {
                "source_case_id": f"upstream-{index}",
                "task_cluster_id": f"cluster-{index // 2}",
                "template_id": f"template-{template_index}",
                "family": "filesystem",
                "prompt": phrasings[template_index].format(path=path),
                "tools": [_tool()],
                "expected_calls": [{"name": "archive_file", "arguments": {"path": path}}],
                "metadata": {"upstream_partition": index % 2},
            }
        )
    return {
        "kind": "localagent_external_action_export",
        "schema_version": 1,
        "benchmark": "external-fixture",
        "revision": "deadbeef",
        "split": "heldout",
        "cases": cases,
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    source_path = tmp_path / "source.json"
    pretrain_path = tmp_path / "pretrain.jsonl"
    midtrain_path = tmp_path / "midtrain.jsonl"
    sft_path = tmp_path / "sft.jsonl"
    contract_path = tmp_path / "contract.json"
    _write_json(source_path, _source())
    pretrain_path.write_text(
        json.dumps(
            {
                "text": "A general document about astronomy and careful experimental design.",
                "doc_id": "general-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    midtrain_path.write_text(
        json.dumps(_conversation("Safeguard local/midtrain.py.", "local/midtrain.py")) + "\n",
        encoding="utf-8",
    )
    sft_path.write_text(
        json.dumps(_conversation("Preserve local/sft.py for later.", "local/sft.py")) + "\n",
        encoding="utf-8",
    )
    training = []
    for stage, name, path, artifact_format in (
        ("pretrain", "pretrain-filtered", pretrain_path, "corpus_jsonl"),
        ("midtrain", "midtrain-agent", midtrain_path, "conversation_jsonl"),
        ("sft", "sft-agent", sft_path, "conversation_jsonl"),
    ):
        training.append(
            {
                **_identity(path),
                "stage": stage,
                "name": name,
                "format": artifact_format,
                "records": 1,
            }
        )
    contract = {
        "kind": "localagent_fresh_external_action_eval_contract",
        "schema_version": 1,
        "source": {
            **_identity(source_path),
            "benchmark": "external-fixture",
            "revision": "deadbeef",
            "split": "heldout",
        },
        "limits": {
            "max_artifact_bytes": 1_000_000,
            "max_source_bytes": 500_000,
            "max_record_bytes": 100_000,
            "max_source_cases": 100,
        },
        "selection": {
            "seed": "paper-freeze-v1",
            "min_cases": 6,
            "max_cases": 6,
            "min_task_clusters": 3,
            "max_cases_per_task_cluster": 2,
            "max_cases_per_template": 2,
        },
        "decontamination": {
            "shingle_size": 5,
            "min_shingles": 8,
            "min_coverage": 0.9,
            "anchors_per_entry": 8,
            "max_denylist_shingles": 2048,
        },
        "training_artifacts": training,
        "analysis": {
            "bootstrap_resamples": 1_000,
            "bootstrap_seed": 31,
            "exact_action_noninferiority_margin": -0.02,
        },
    }
    _write_json(contract_path, contract)
    return contract_path, source_path, pretrain_path, midtrain_path, sft_path


def _lineage_export(
    stage: str,
    artifact_sha256: list[str],
    *,
    marker: str,
    parent_checkpoint_sha256: str | None = None,
) -> dict[str, object]:
    lineage = {
        "version": 1,
        "stage": stage,
        "config_sha256": hashlib.sha256(f"{stage}-config".encode()).hexdigest(),
        "model_config_sha256": hashlib.sha256(f"{stage}-model".encode()).hexdigest(),
        "data_sha256": hashlib.sha256(f"{stage}-data".encode()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(b"tokenizer").hexdigest(),
        "git": {
            "commit": "a" * 40,
            "repository_sha256": hashlib.sha256(b"repository").hexdigest(),
            "dirty": False,
            "worktree_sha256": hashlib.sha256(b"tree").hexdigest(),
        },
    }
    if parent_checkpoint_sha256 is not None:
        lineage["parent_checkpoint_sha256"] = parent_checkpoint_sha256
    return {
        "kind": TRAINING_LINEAGE_KIND,
        "schema_version": 1,
        "stage": stage,
        "checkpoint_sha256": marker * 64,
        "lineage": lineage,
        "training_artifact_sha256": artifact_sha256,
        "conversation_prompt_contract": (None if stage == "pretrain" else "openai_full_catalog_v1"),
    }


def _fixture_v2(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, Path], dict[str, Path]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    contract_path, source_path, pretrain_path, midtrain_path, sft_path = _fixture(tmp_path)
    source = _source()
    source["schema_version"] = HARDENED_SCHEMA_VERSION
    _write_json(source_path, source)
    rl_path = tmp_path / "rl.jsonl"
    rl_path.write_text(
        json.dumps(_conversation("Retain local/rl.py.", "local/rl.py")) + "\n",
        encoding="utf-8",
    )
    artifacts = {
        "pretrain": pretrain_path,
        "midtrain": midtrain_path,
        "sft": sft_path,
        "rl": rl_path,
    }
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["schema_version"] = HARDENED_SCHEMA_VERSION
    contract["source"] = {
        **_identity(source_path),
        "benchmark": "external-fixture",
        "revision": "deadbeef",
        "split": "heldout",
    }
    training = []
    for stage, path in artifacts.items():
        artifact_format = "corpus_jsonl" if stage == "pretrain" else "conversation_jsonl"
        training.append(
            {
                **_identity(path),
                "stage": stage,
                "name": f"{stage}-data",
                "format": artifact_format,
                "records": 1,
                **(
                    {"conversation_prompt_contract": "openai_full_catalog_v1"}
                    if artifact_format == "conversation_jsonl"
                    else {}
                ),
            }
        )
    contract["training_artifacts"] = training
    lineage_paths: dict[str, Path] = {}
    lineage_declarations = []
    parent_checkpoint_sha256: str | None = None
    for index, (stage, path) in enumerate(artifacts.items()):
        lineage_path = tmp_path / f"{stage}-lineage.json"
        marker = "abcdef0123456789"[index]
        _write_json(
            lineage_path,
            _lineage_export(
                stage,
                [_identity(path)["sha256"]],
                marker=marker,
                parent_checkpoint_sha256=parent_checkpoint_sha256,
            ),
        )
        parent_checkpoint_sha256 = marker * 64
        lineage_paths[stage] = lineage_path
        lineage_declarations.append(
            {
                **_identity(lineage_path),
                "stage": stage,
                "name": f"{stage}-lineage",
            }
        )
    contract["lineage_artifacts"] = lineage_declarations
    _write_json(contract_path, contract)
    return contract_path, source_path, artifacts, lineage_paths


def _refresh_v2_training_and_lineage(
    contract_path: Path,
    *,
    stage: str,
    artifact_path: Path,
    lineage_path: Path,
) -> None:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    declaration = next(item for item in contract["training_artifacts"] if item["stage"] == stage)
    declaration.update(_identity(artifact_path))
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    lineage["training_artifact_sha256"] = [_identity(artifact_path)["sha256"]]
    _write_json(lineage_path, lineage)
    lineage_declaration = next(
        item for item in contract["lineage_artifacts"] if item["stage"] == stage
    )
    lineage_declaration.update(_identity(lineage_path))
    _write_json(contract_path, contract)


def _freeze_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    return (
        tmp_path / "frozen-slice.json",
        tmp_path / "prompt-denylist.json",
        tmp_path / "freeze-manifest.json",
    )


def test_prompt_normalization_and_action_template_ignore_gold_scalar_values():
    first = action_template_sha256(
        "Archive reports/Q1.txt.",
        [{"name": "archive_file", "arguments": {"path": "reports/Q1.txt"}}],
    )
    second = action_template_sha256(
        "Archive private/notes.md.",
        [{"name": "archive_file", "arguments": {"path": "private/notes.md"}}],
    )
    different = action_template_sha256(
        "Please archive private/notes.md now.",
        [{"name": "archive_file", "arguments": {"path": "private/notes.md"}}],
    )
    assert first == second
    assert first != different
    parallel = [
        {"name": "copy_file", "arguments": {"path": "reports/Q1.txt"}},
        {"name": "archive_file", "arguments": {"path": "private/notes.md"}},
    ]
    assert action_template_sha256(
        "Copy reports/Q1.txt and archive private/notes.md.",
        parallel,
    ) == action_template_sha256(
        "Copy reports/Q1.txt and archive private/notes.md.",
        list(reversed(parallel)),
    )
    assert normalize_prompt("Ｆｉｌｅ\u00a0Name") == "file name"


def test_freeze_is_deterministic_idempotent_and_prompt_only_denylist_is_separate(
    tmp_path: Path,
):
    contract_path, *_ = _fixture(tmp_path)
    slice_path, denylist_path, manifest_path = _freeze_paths(tmp_path)
    first = freeze_external_action_slice(
        contract_path,
        slice_path=slice_path,
        denylist_path=denylist_path,
        manifest_path=manifest_path,
    )
    first_bytes = (slice_path.read_bytes(), denylist_path.read_bytes(), manifest_path.read_bytes())
    second = freeze_external_action_slice(
        contract_path,
        slice_path=slice_path,
        denylist_path=denylist_path,
        manifest_path=manifest_path,
    )
    assert second == first
    assert (
        slice_path.read_bytes(),
        denylist_path.read_bytes(),
        manifest_path.read_bytes(),
    ) == first_bytes
    assert first["selection"]["selected_cases"] == 6
    assert first["selection"]["selected_task_clusters"] >= 3
    assert first["decontamination_audit"]["prompt_or_shingle_overlap_units"] == 0
    assert first["decontamination_audit"]["derived_action_template_overlap_units"] == 0
    frozen = json.loads(slice_path.read_text(encoding="utf-8"))
    denylist = json.loads(denylist_path.read_text(encoding="utf-8"))
    assert len({case["case_id"] for case in frozen["cases"]}) == 6
    assert all(case["case_id"].startswith("extcase-") for case in frozen["cases"])
    assert all("expected_calls" in case for case in frozen["cases"])
    assert all("expected_calls" not in case for case in denylist["cases"])
    assert [case["case_id"] for case in frozen["cases"]] == sorted(
        case["case_id"] for case in frozen["cases"]
    )
    without_self_hash = dict(first)
    self_hash = without_self_hash.pop("manifest_self_sha256")
    assert hashlib.sha256(_canonical_bytes(without_self_hash)).hexdigest() == self_hash

    denylist_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing to overwrite drifted"):
        freeze_external_action_slice(
            contract_path,
            slice_path=slice_path,
            denylist_path=denylist_path,
            manifest_path=manifest_path,
        )


def test_freeze_manifest_is_independent_of_absolute_bundle_location(tmp_path: Path):
    manifests = []
    for directory_name in ("first", "second"):
        directory = tmp_path / directory_name
        directory.mkdir()
        contract_path, *_ = _fixture(directory)
        slice_path, denylist_path, manifest_path = _freeze_paths(directory)
        freeze_external_action_slice(
            contract_path,
            slice_path=slice_path,
            denylist_path=denylist_path,
            manifest_path=manifest_path,
        )
        manifests.append(manifest_path.read_bytes())
    assert manifests[0] == manifests[1]


def test_freeze_rejects_exact_or_shingle_training_overlap(tmp_path: Path):
    contract_path, _, pretrain_path, *_ = _fixture(tmp_path)
    source_case = _source()["cases"][0]
    pretrain_path.write_text(
        json.dumps(
            {"text": ("An accidentally copied benchmark prompt follows. " + source_case["prompt"])}
        )
        + "\n",
        encoding="utf-8",
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["training_artifacts"][0].update(_identity(pretrain_path))
    _write_json(contract_path, contract)
    with pytest.raises(ValueError, match="prompt_or_shingle_units=1"):
        freeze_external_action_slice(
            contract_path,
            slice_path=tmp_path / "slice.json",
            denylist_path=tmp_path / "denylist.json",
            manifest_path=tmp_path / "manifest.json",
        )
    assert not (tmp_path / "slice.json").exists()


def test_freeze_rejects_ambiguous_corpus_text_fields_instead_of_skipping_one(
    tmp_path: Path,
):
    contract_path, _, pretrain_path, *_ = _fixture(tmp_path)
    pretrain_path.write_text(
        json.dumps(
            {
                "text": "A benign first field.",
                "content": _source()["cases"][0]["prompt"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["training_artifacts"][0].update(_identity(pretrain_path))
    _write_json(contract_path, contract)
    with pytest.raises(ValueError, match="ambiguous text-bearing fields"):
        freeze_external_action_slice(
            contract_path,
            slice_path=tmp_path / "slice.json",
            denylist_path=tmp_path / "denylist.json",
            manifest_path=tmp_path / "manifest.json",
        )


def test_freeze_screens_canonical_conversation_argument_keys_and_scalars(
    tmp_path: Path,
):
    contract_path, _, _, _, sft_path = _fixture(tmp_path)
    training_row = _conversation("Perform an unrelated local action.", "local/safe.txt")
    training_row["messages"][1]["tool_calls"][0]["arguments"] = {
        case["prompt"]: index for index, case in enumerate(_source()["cases"])
    }
    sft_path.write_text(json.dumps(training_row) + "\n", encoding="utf-8")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["training_artifacts"][2].update(_identity(sft_path))
    _write_json(contract_path, contract)

    with pytest.raises(ValueError, match="prompt_or_shingle_units=1"):
        freeze_external_action_slice(
            contract_path,
            slice_path=tmp_path / "slice.json",
            denylist_path=tmp_path / "denylist.json",
            manifest_path=tmp_path / "manifest.json",
        )


@pytest.mark.parametrize(
    ("limit_name", "limit_value"),
    [
        ("max_artifact_bytes", 8 * 1024 * 1024 * 1024 + 1),
        ("max_source_bytes", 256 * 1024 * 1024 + 1),
        ("max_record_bytes", 16 * 1024 * 1024 + 1),
        ("max_source_cases", 50_001),
    ],
)
def test_freeze_rejects_limits_above_hard_memory_caps(
    tmp_path: Path,
    limit_name: str,
    limit_value: int,
):
    contract_path, *_ = _fixture(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["limits"]["max_artifact_bytes"] = 8 * 1024 * 1024 * 1024
    contract["limits"][limit_name] = limit_value
    _write_json(contract_path, contract)

    with pytest.raises(ValueError, match="hard maximum"):
        freeze_external_action_slice(
            contract_path,
            slice_path=tmp_path / "slice.json",
            denylist_path=tmp_path / "denylist.json",
            manifest_path=tmp_path / "manifest.json",
        )


def test_freeze_rejects_labeled_action_template_overlap_with_different_value(
    tmp_path: Path,
):
    contract_path, _, _, _, sft_path = _fixture(tmp_path)
    # Same derived "Archive <arg> ." skeleton as one external template, but no exact prompt.
    sft_path.write_text(
        json.dumps(_conversation("Archive private/unseen.md.", "private/unseen.md")) + "\n",
        encoding="utf-8",
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["training_artifacts"][2].update(_identity(sft_path))
    _write_json(contract_path, contract)
    with pytest.raises(ValueError, match="derived_action_template_units=1"):
        freeze_external_action_slice(
            contract_path,
            slice_path=tmp_path / "slice.json",
            denylist_path=tmp_path / "denylist.json",
            manifest_path=tmp_path / "manifest.json",
        )


def test_hardened_freeze_binds_rl_lineage_and_exact_full_catalog_training_text(
    tmp_path: Path,
):
    contract_path, _, _, _ = _fixture_v2(tmp_path)
    slice_path, denylist_path, manifest_path = _freeze_paths(tmp_path)
    manifest = freeze_external_action_slice(
        contract_path,
        slice_path=slice_path,
        denylist_path=denylist_path,
        manifest_path=manifest_path,
    )
    assert manifest["schema_version"] == HARDENED_SCHEMA_VERSION
    assert {item["stage"] for item in manifest["training_artifacts"]} == {
        "pretrain",
        "midtrain",
        "sft",
        "rl",
    }
    assert {item["stage"] for item in manifest["lineage_artifacts"]} == {
        "pretrain",
        "midtrain",
        "sft",
        "rl",
    }
    conversation_audits = [
        item
        for item in manifest["decontamination_audit"]["artifacts"]
        if item["format"] == "conversation_jsonl"
    ]
    assert conversation_audits
    assert all(
        item["rendered_training_text_contract"].startswith("openai_full_catalog_v1")
        for item in conversation_audits
    )
    assert all(
        item["conversation_prompt_contract"] == "openai_full_catalog_v1"
        for item in conversation_audits
    )
    assert (
        validate_frozen_slice(json.loads(slice_path.read_text(encoding="utf-8")))["schema_version"]
        == HARDENED_SCHEMA_VERSION
    )


def test_hardened_freeze_screens_catalog_text_and_rejects_legacy_prompt_boundary(
    tmp_path: Path,
):
    contract_path, _, artifacts, lineage_paths = _fixture_v2(tmp_path)
    row = _conversation("Perform an unrelated local action.", "local/safe.txt")
    row["tools"][0]["description"] = "Catalog canary: " + _source()["cases"][0]["prompt"]
    artifacts["sft"].write_text(json.dumps(row) + "\n", encoding="utf-8")
    _refresh_v2_training_and_lineage(
        contract_path,
        stage="sft",
        artifact_path=artifacts["sft"],
        lineage_path=lineage_paths["sft"],
    )
    with pytest.raises(ValueError, match="prompt_or_shingle_units=1"):
        freeze_external_action_slice(
            contract_path,
            slice_path=tmp_path / "slice.json",
            denylist_path=tmp_path / "denylist.json",
            manifest_path=tmp_path / "manifest.json",
        )

    contract_path, _, _, _ = _fixture_v2(tmp_path / "legacy")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["training_artifacts"][1]["conversation_prompt_contract"] = "legacy"
    _write_json(contract_path, contract)
    with pytest.raises(ValueError, match="supports only conversation_prompt_contract"):
        freeze_external_action_slice(
            contract_path,
            slice_path=tmp_path / "legacy-slice.json",
            denylist_path=tmp_path / "legacy-denylist.json",
            manifest_path=tmp_path / "legacy-manifest.json",
        )


def test_hardened_freeze_rejects_missing_rl_lineage_and_recursive_gold_schema_failure(
    tmp_path: Path,
):
    contract_path, source_path, _, _ = _fixture_v2(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["lineage_artifacts"] = [
        item for item in contract["lineage_artifacts"] if item["stage"] != "rl"
    ]
    _write_json(contract_path, contract)
    with pytest.raises(ValueError, match="must cover pretrain, midtrain, sft, and rl"):
        freeze_external_action_slice(
            contract_path,
            slice_path=tmp_path / "slice.json",
            denylist_path=tmp_path / "denylist.json",
            manifest_path=tmp_path / "manifest.json",
        )

    broken = tmp_path / "broken-lineage"
    contract_path, _, _, lineage_paths = _fixture_v2(broken)
    lineage = json.loads(lineage_paths["rl"].read_text(encoding="utf-8"))
    lineage["lineage"]["parent_checkpoint_sha256"] = "f" * 64
    _write_json(lineage_paths["rl"], lineage)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    declaration = next(item for item in contract["lineage_artifacts"] if item["stage"] == "rl")
    declaration.update(_identity(lineage_paths["rl"]))
    _write_json(contract_path, contract)
    with pytest.raises(ValueError, match="absent from frozen sft lineage"):
        freeze_external_action_slice(
            contract_path,
            slice_path=broken / "slice.json",
            denylist_path=broken / "denylist.json",
            manifest_path=broken / "manifest.json",
        )

    second = tmp_path / "recursive"
    contract_path, source_path, _, _ = _fixture_v2(second)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    nested = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    source["cases"][0]["tools"][0]["parameters"]["properties"]["options"] = nested
    source["cases"][0]["tools"][0]["parameters"]["required"].append("options")
    source["cases"][0]["expected_calls"][0]["arguments"]["options"] = {
        "value": "safe",
        "injected": True,
    }
    _write_json(source_path, source)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["source"].update(_identity(source_path))
    _write_json(contract_path, contract)
    with pytest.raises(ValueError, match="arguments fail the declared tool schema"):
        freeze_external_action_slice(
            contract_path,
            slice_path=second / "slice.json",
            denylist_path=second / "denylist.json",
            manifest_path=second / "manifest.json",
        )


def test_freeze_rejects_duplicate_and_unknown_json_keys(tmp_path: Path):
    contract_path, source_path, *_ = _fixture(tmp_path)
    source_text = source_path.read_text(encoding="utf-8")
    source_path.write_text(
        source_text.replace(
            '"benchmark":"external-fixture"',
            '"benchmark":"external-fixture","benchmark":"shadow"',
            1,
        ),
        encoding="utf-8",
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["source"].update(_identity(source_path))
    _write_json(contract_path, contract)
    with pytest.raises(ValueError, match="duplicate JSON key 'benchmark'"):
        freeze_external_action_slice(
            contract_path,
            slice_path=tmp_path / "slice.json",
            denylist_path=tmp_path / "denylist.json",
            manifest_path=tmp_path / "manifest.json",
        )

    unknown = tmp_path / "unknown"
    unknown.mkdir()
    contract_path, source_path, *_ = _fixture(unknown)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["cases"][0]["unscored_shadow_gold"] = []
    _write_json(source_path, source)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["source"].update(_identity(source_path))
    _write_json(contract_path, contract)
    with pytest.raises(ValueError, match=r"extra=\['unscored_shadow_gold'\]"):
        freeze_external_action_slice(
            contract_path,
            slice_path=unknown / "slice.json",
            denylist_path=unknown / "denylist.json",
            manifest_path=unknown / "manifest.json",
        )


def _quality_rows(
    outcomes: dict[str, list[bool]],
    clusters: dict[str, str],
) -> list[dict[str, object]]:
    return [
        {
            "case_id": case_id,
            "task_cluster_id": clusters[case_id],
            "repetition": repetition,
            "success": success,
        }
        for case_id, values in outcomes.items()
        for repetition, success in enumerate(values)
    ]


def _system(name: str, marker: str) -> dict[str, str]:
    return {
        "name": name,
        "checkpoint_sha256": marker * 64,
        "bundle_sha256": hashlib.sha256(f"{name}-bundle".encode()).hexdigest(),
    }


def test_paired_cluster_bootstrap_averages_repeats_and_resamples_task_clusters():
    clusters = {
        "case-a": "cluster-1",
        "case-b": "cluster-1",
        "case-c": "cluster-2",
        "case-d": "cluster-3",
    }
    baseline = _quality_rows(
        {
            "case-a": [False, False],
            "case-b": [True, True],
            "case-c": [False, False],
            "case-d": [True, True],
        },
        clusters,
    )
    candidate = _quality_rows(
        {
            "case-a": [True, True],
            "case-b": [True, True],
            "case-c": [False, False],
            "case-d": [False, False],
        },
        clusters,
    )
    got = paired_clustered_exact_action_delta_ci(
        baseline,
        candidate,
        resamples=2_000,
        seed=19,
        noninferiority_margin=-0.02,
    )
    assert got == paired_clustered_exact_action_delta_ci(
        list(reversed(baseline)),
        list(reversed(candidate)),
        resamples=2_000,
        seed=19,
        noninferiority_margin=-0.02,
    )
    assert got["estimate"] == 0.0
    assert got["baseline_estimate"] == 0.5
    assert got["candidate_estimate"] == 0.5
    assert got["case_count"] == 4
    assert got["cluster_count"] == 3
    assert got["observations_per_system"] == 8
    assert got["method"] == "paired_percentile_task_cluster_bootstrap"
    assert got["latency_gate_included"] is False
    assert got["passes_noninferiority"] is False
    assert got["lower"] <= 0.0 <= got["upper"]

    with pytest.raises(ValueError, match="different task clusters"):
        changed = [dict(row) for row in candidate]
        for row in changed[:2]:
            row["task_cluster_id"] = "other"
        paired_clustered_exact_action_delta_ci(
            baseline,
            changed,
            resamples=10,
        )
    with pytest.raises(ValueError, match="different repetition IDs"):
        paired_clustered_exact_action_delta_ci(
            baseline,
            candidate[:-1],
            resamples=10,
        )
    with pytest.raises(ValueError, match="at least two task clusters"):
        one_cluster = {case_id: "only-cluster" for case_id in clusters}
        paired_clustered_exact_action_delta_ci(
            _quality_rows(
                {
                    "case-a": [False],
                    "case-b": [True],
                    "case-c": [False],
                    "case-d": [True],
                },
                one_cluster,
            ),
            _quality_rows(
                {
                    "case-a": [True],
                    "case-b": [True],
                    "case-c": [False],
                    "case-d": [False],
                },
                one_cluster,
            ),
            resamples=10,
        )


def test_compare_cli_recomputes_ast_success_and_refuses_stale_reported_score(
    tmp_path: Path,
):
    contract_path, *_ = _fixture(tmp_path)
    slice_path, denylist_path, manifest_path = _freeze_paths(tmp_path)
    freeze_external_action_slice(
        contract_path,
        slice_path=slice_path,
        denylist_path=denylist_path,
        manifest_path=manifest_path,
    )
    frozen = json.loads(slice_path.read_text(encoding="utf-8"))
    baseline_records = []
    candidate_records = []
    for case in frozen["cases"]:
        common = {
            "case_id": case["case_id"],
            "task_cluster_id": case["task_cluster_id"],
            "repetition": 0,
        }
        baseline_records.append({**common, "predicted_calls": [], "success": False})
        # Reverse keys to prove scoring is argument-order insensitive.
        call = case["expected_calls"][0]
        candidate_records.append(
            {
                **common,
                "predicted_calls": [
                    {
                        "arguments": dict(reversed(list(call["arguments"].items()))),
                        "name": call["name"],
                    }
                ],
                "success": True,
            }
        )
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    _write_json(
        baseline_path,
        {"system": _system("baseline", "a"), "records": baseline_records},
    )
    _write_json(
        candidate_path,
        {"system": _system("candidate", "b"), "records": candidate_records},
    )
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(root / "scripts" / "fresh_action_eval.py"),
        "compare",
        "--manifest",
        str(manifest_path),
        "--slice",
        str(slice_path),
        "--baseline",
        str(baseline_path),
        "--candidate",
        str(candidate_path),
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["kind"] == "localagent_fresh_external_action_comparison"
    assert "evaluation_input_contract" not in summary
    assert summary["exact_action_comparison"]["estimate"] == 1.0
    assert summary["exact_action_comparison"]["passes_noninferiority"] is True
    assert summary["promotion_decision"]["promote"] is False
    assert summary["promotion_decision"]["latency_gate_evaluated"] is False

    same_file = subprocess.run(
        [
            *command[:-1],
            str(baseline_path),
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        check=False,
        text=True,
    )
    assert same_file.returncode != 0
    assert "must be distinct files" in same_file.stderr

    candidate_records[0]["success"] = False
    _write_json(
        candidate_path,
        {"system": _system("candidate", "b"), "records": candidate_records},
    )
    rejected = subprocess.run(
        command,
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        check=False,
        text=True,
    )
    assert rejected.returncode != 0
    assert "disagrees with independent AST scoring" in rejected.stderr


def _tool_output(call: dict[str, object]) -> str:
    return (
        "<tool_call>"
        + json.dumps(call, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "</tool_call>"
    )


def _raw_system(
    name: str,
    *,
    checkpoint: Path,
    bundle: Path,
) -> dict[str, object]:
    return {
        "name": name,
        "checkpoint": _identity(checkpoint),
        "bundle": _identity(bundle),
    }


def _compare_command(
    *,
    manifest_path: Path,
    slice_path: Path,
    baseline_path: Path,
    candidate_path: Path,
) -> tuple[list[str], Path]:
    root = Path(__file__).resolve().parents[1]
    return [
        sys.executable,
        str(root / "scripts" / "fresh_action_eval.py"),
        "compare",
        "--manifest",
        str(manifest_path),
        "--slice",
        str(slice_path),
        "--baseline",
        str(baseline_path),
        "--candidate",
        str(candidate_path),
    ], root


def test_raw_output_result_strictly_parses_and_verifies_actual_model_artifacts(
    tmp_path: Path,
):
    contract_path, _, _, lineage_paths = _fixture_v2(tmp_path)
    checkpoint_path = tmp_path / "model.pt"
    baseline_bundle = tmp_path / "baseline.bundle"
    candidate_bundle = tmp_path / "candidate.bundle"
    checkpoint_path.write_bytes(b"checkpoint-used-for-both-policies")
    baseline_bundle.write_bytes(b"autoregressive-bundle")
    candidate_bundle.write_bytes(b"structured-bundle")
    rl_lineage = json.loads(lineage_paths["rl"].read_text(encoding="utf-8"))
    rl_lineage["checkpoint_sha256"] = _identity(checkpoint_path)["sha256"]
    _write_json(lineage_paths["rl"], rl_lineage)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    rl_declaration = next(item for item in contract["lineage_artifacts"] if item["stage"] == "rl")
    rl_declaration.update(_identity(lineage_paths["rl"]))
    _write_json(contract_path, contract)
    slice_path, denylist_path, manifest_path = _freeze_paths(tmp_path)
    freeze_external_action_slice(
        contract_path,
        slice_path=slice_path,
        denylist_path=denylist_path,
        manifest_path=manifest_path,
    )
    frozen = json.loads(slice_path.read_text(encoding="utf-8"))

    baseline_records = []
    candidate_records = []
    for case in frozen["cases"]:
        common = {
            "case_id": case["case_id"],
            "task_cluster_id": case["task_cluster_id"],
            "repetition": 0,
        }
        baseline_records.append(
            {
                **common,
                "raw_output": "",
                "finish_reason": "eos",
                "success": False,
            }
        )
        candidate_records.append(
            {
                **common,
                "raw_output": "".join(_tool_output(call) for call in case["expected_calls"]),
                "finish_reason": "eos",
                "success": True,
            }
        )
    baseline_path = tmp_path / "baseline-raw.json"
    candidate_path = tmp_path / "candidate-raw.json"
    _write_json(
        baseline_path,
        {
            "kind": "localagent_external_raw_model_output_result",
            "schema_version": 1,
            "system": _raw_system(
                "baseline",
                checkpoint=checkpoint_path,
                bundle=baseline_bundle,
            ),
            "records": baseline_records,
        },
    )
    _write_json(
        candidate_path,
        {
            "kind": "localagent_external_raw_model_output_result",
            "schema_version": 1,
            "system": _raw_system(
                "candidate",
                checkpoint=checkpoint_path,
                bundle=candidate_bundle,
            ),
            "records": candidate_records,
        },
    )
    command, root = _compare_command(
        manifest_path=manifest_path,
        slice_path=slice_path,
        baseline_path=baseline_path,
        candidate_path=candidate_path,
    )
    completed = subprocess.run(
        command,
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["kind"] == "localagent_fresh_external_raw_output_action_comparison"
    assert summary["evaluation_input_contract"] == {
        "actual_checkpoint_and_bundle_identity_verified": True,
        "evaluation_semantics": "raw_whole_output_strict_parse_tool_output_v1",
        "raw_model_output_observed": True,
        "strict_parse_tool_output": True,
    }
    assert summary["candidate"]["parse_outcomes"] == {
        "format_invalid": 0,
        "incomplete_finish": 0,
        "schema_invalid": 0,
    }
    assert summary["exact_action_comparison"]["estimate"] == 1.0

    candidate_bundle.write_bytes(b"tampered-after-result-capture")
    tampered = subprocess.run(
        command,
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        check=False,
        text=True,
    )
    assert tampered.returncode != 0
    assert "byte-size mismatch" in tampered.stderr or "SHA-256 mismatch" in tampered.stderr

    candidate_bundle.write_bytes(b"structured-bundle")
    alternate_checkpoint = tmp_path / "unfrozen-model.pt"
    alternate_checkpoint.write_bytes(b"not-present-in-frozen-rl-lineage")
    candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate_payload["system"]["checkpoint"] = _identity(alternate_checkpoint)
    _write_json(candidate_path, candidate_payload)
    unbound = subprocess.run(
        command,
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        check=False,
        text=True,
    )
    assert unbound.returncode != 0
    assert "absent from the frozen RL lineage artifacts" in unbound.stderr


def test_raw_output_rejects_duplicate_call_keys_incomplete_finish_and_mixed_lanes(
    tmp_path: Path,
):
    contract_path, *_ = _fixture(tmp_path)
    slice_path, denylist_path, manifest_path = _freeze_paths(tmp_path)
    freeze_external_action_slice(
        contract_path,
        slice_path=slice_path,
        denylist_path=denylist_path,
        manifest_path=manifest_path,
    )
    frozen = json.loads(slice_path.read_text(encoding="utf-8"))
    checkpoint_path = tmp_path / "model.pt"
    baseline_bundle = tmp_path / "baseline.bundle"
    candidate_bundle = tmp_path / "candidate.bundle"
    checkpoint_path.write_bytes(b"checkpoint")
    baseline_bundle.write_bytes(b"baseline")
    candidate_bundle.write_bytes(b"candidate")
    baseline_records = []
    candidate_records = []
    normalized_records = []
    normalized_baseline_records = []
    for index, case in enumerate(frozen["cases"]):
        common = {
            "case_id": case["case_id"],
            "task_cluster_id": case["task_cluster_id"],
            "repetition": 0,
        }
        expected = case["expected_calls"][0]
        baseline_records.append(
            {**common, "raw_output": "", "finish_reason": "eos", "success": False}
        )
        candidate_records.append(
            {
                **common,
                "raw_output": (
                    '<tool_call>{"name":"archive_file","name":"shadow","arguments":{}}</tool_call>'
                    if index == 0
                    else _tool_output(expected)
                ),
                "finish_reason": "length" if index == 1 else "eos",
                "success": index not in {0, 1},
            }
        )
        normalized_records.append(
            {
                **common,
                "predicted_calls": case["expected_calls"],
                "success": True,
            }
        )
        normalized_baseline_records.append(
            {
                **common,
                "predicted_calls": [],
                "success": False,
            }
        )
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    normalized_path = tmp_path / "normalized.json"
    normalized_baseline_path = tmp_path / "normalized-baseline.json"
    _write_json(
        baseline_path,
        {
            "kind": "localagent_external_raw_model_output_result",
            "schema_version": 1,
            "system": _raw_system(
                "baseline",
                checkpoint=checkpoint_path,
                bundle=baseline_bundle,
            ),
            "records": baseline_records,
        },
    )
    _write_json(
        candidate_path,
        {
            "kind": "localagent_external_raw_model_output_result",
            "schema_version": 1,
            "system": _raw_system(
                "candidate",
                checkpoint=checkpoint_path,
                bundle=candidate_bundle,
            ),
            "records": candidate_records,
        },
    )
    _write_json(
        normalized_path,
        {
            "kind": "localagent_external_normalized_call_result",
            "schema_version": 1,
            "system": _system("normalized", "c"),
            "records": normalized_records,
        },
    )
    _write_json(
        normalized_baseline_path,
        {
            "kind": "localagent_external_normalized_call_result",
            "schema_version": 1,
            "system": _system("normalized-baseline", "d"),
            "records": normalized_baseline_records,
        },
    )
    command, root = _compare_command(
        manifest_path=manifest_path,
        slice_path=slice_path,
        baseline_path=baseline_path,
        candidate_path=candidate_path,
    )
    completed = subprocess.run(
        command,
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["candidate"]["parse_outcomes"]["format_invalid"] == 1
    assert summary["candidate"]["parse_outcomes"]["incomplete_finish"] == 1

    normalized_command, _ = _compare_command(
        manifest_path=manifest_path,
        slice_path=slice_path,
        baseline_path=normalized_baseline_path,
        candidate_path=normalized_path,
    )
    normalized = subprocess.run(
        normalized_command,
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        check=False,
        text=True,
    )
    assert normalized.returncode == 0, normalized.stderr
    normalized_summary = json.loads(normalized.stdout)
    assert normalized_summary["kind"] == "localagent_fresh_external_normalized_call_comparison"
    assert normalized_summary["evaluation_input_contract"] == {
        "actual_checkpoint_and_bundle_identity_verified": False,
        "evaluation_semantics": "adapter_supplied_normalized_calls_v1",
        "raw_model_output_observed": False,
        "strict_parse_tool_output": False,
    }

    mixed_command, _ = _compare_command(
        manifest_path=manifest_path,
        slice_path=slice_path,
        baseline_path=normalized_path,
        candidate_path=candidate_path,
    )
    mixed = subprocess.run(
        mixed_command,
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        check=False,
        text=True,
    )
    assert mixed.returncode != 0
    assert "refusing to mix raw whole-output evaluation" in mixed.stderr

    unknown = json.loads(candidate_path.read_text(encoding="utf-8"))
    unknown["records"][0]["predicted_calls"] = []
    _write_json(candidate_path, unknown)
    rejected = subprocess.run(
        command,
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        check=False,
        text=True,
    )
    assert rejected.returncode != 0
    assert "extra=['predicted_calls']" in rejected.stderr
