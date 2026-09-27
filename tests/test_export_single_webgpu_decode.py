from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from localagent.inference.export import to_onnx


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "export_single_webgpu_decode.py"
SHA256 = "a" * 64


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("export_single_webgpu_decode_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_export_cli_forwards_acceptance_lineage_guard(
    monkeypatch,
    capsys,
    tmp_path: Path,
):
    module = _load_script()
    observed = {}
    training_artifact = tmp_path / "sft-conversations.json"
    training_artifact.write_text(
        json.dumps(
            {
                "kind": "localagent_sft_conversation_artifact",
                "examples": [],
            }
        )
    )
    training_payload = training_artifact.read_bytes()
    training_sha256 = hashlib.sha256(training_payload).hexdigest()

    def fake_export(config: str, output: str, **kwargs):
        observed.update({"config": config, "output": output, **kwargs})
        return {
            "single_manifest_path": f"{output}/single-decode.json",
            "parity": {"fp16": {}, "fp32": {}},
            "provenance": {
                "weights": {
                    "checkpoint_sha256": "b" * 64,
                    "checkpoint_stage": "sft",
                },
                "tokenizer": {"sha256": "c" * 64},
            },
        }

    monkeypatch.setattr(module, "export_cached_decode", fake_export)
    result = module.main(
        [
            "--config",
            "model.yaml",
            "--checkpoint",
            "accepted.pt",
            "--out",
            "bundle",
            "--training-artifact",
            str(training_artifact),
        ]
    )

    assert result == 0
    assert observed["config"] == "model.yaml"
    assert observed["checkpoint_path"] == "accepted.pt"
    assert observed["training_artifact_sha256"] == [training_sha256]
    assert observed["training_artifact_identities"] == [
        {
            "artifact_kind": "localagent_sft_conversation_artifact",
            "bytes": len(training_payload),
            "path": str(training_artifact.resolve()),
            "sha256": training_sha256,
        }
    ]
    assert observed["require_posttraining_training_artifacts"] is True
    assert observed["pair_role"] == "accepted_checkpoint"
    assert "single manifest: bundle/single-decode.json" in capsys.readouterr().out


def test_single_export_cli_rejects_duplicate_training_hashes(tmp_path: Path):
    module = _load_script()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text('{"kind":"fixture"}')
    second.write_bytes(first.read_bytes())
    with pytest.raises(SystemExit, match="must be unique"):
        module.main(
            [
                "--config",
                "model.yaml",
                "--checkpoint",
                "accepted.pt",
                "--out",
                "bundle",
                "--training-artifact",
                str(first),
                "--training-artifact",
                str(second),
            ]
        )


def test_exporter_guard_rejects_posttraining_without_artifact_hashes_before_writing(
    monkeypatch,
    tmp_path: Path,
):
    config_path = ROOT / "configs" / "model" / "ultra-tiny-1m.yaml"
    output = tmp_path / "bundle"

    monkeypatch.setattr(
        to_onnx,
        "_load_cached_decode_checkpoint",
        lambda *args, **kwargs: (object(), {"stage": "sft"}),
    )
    with pytest.raises(
        ValueError,
        match="requires at least one training artifact file identity",
    ):
        to_onnx.export_cached_decode(
            str(config_path),
            str(output),
            pair_role="accepted_checkpoint",
            checkpoint_path="accepted.pt",
            fp16=False,
            require_posttraining_training_artifacts=True,
        )

    assert not output.exists()


def test_exporter_guard_rejects_naked_arbitrary_hash_even_when_supplied(
    monkeypatch,
    tmp_path: Path,
):
    config_path = ROOT / "configs" / "model" / "ultra-tiny-1m.yaml"
    output = tmp_path / "bundle"
    monkeypatch.setattr(
        to_onnx,
        "_load_cached_decode_checkpoint",
        lambda *args, **kwargs: (object(), {"stage": "sft"}),
    )

    with pytest.raises(ValueError, match="training artifact file identity"):
        to_onnx.export_cached_decode(
            str(config_path),
            str(output),
            pair_role="accepted_checkpoint",
            checkpoint_path="accepted.pt",
            fp16=False,
            training_artifact_sha256=[SHA256],
            require_posttraining_training_artifacts=True,
        )

    assert not output.exists()


def test_training_artifact_parser_rejects_missing_and_semantically_invalid_receipt(
    tmp_path: Path,
):
    module = _load_script()
    with pytest.raises(argparse.ArgumentTypeError, match="does not exist"):
        module._training_artifact(str(tmp_path / "missing.json"))

    invalid_receipt = tmp_path / "invalid-receipt.json"
    invalid_receipt.write_text(
        json.dumps(
            {
                "kind": "localagent_webgpu_cached_decode_acceptance_receipt",
                "schema_version": 1,
                "verified": True,
                "receipt_self_sha256": "0" * 64,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    with pytest.raises(
        argparse.ArgumentTypeError,
        match="invalid WebGPU acceptance receipt",
    ):
        module._training_artifact(str(invalid_receipt))


def test_training_lineage_rehash_rejects_artifact_drift(tmp_path: Path):
    artifact = tmp_path / "training.json"
    artifact.write_bytes(b'{"kind":"fixture","value":1}')
    payload = artifact.read_bytes()
    identity = {
        "artifact_kind": "fixture",
        "bytes": len(payload),
        "path": str(artifact.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    artifact.write_bytes(b'{"kind":"fixture","value":2}')
    checkpoint_info = {
        "stage": "sft",
        "checkpoint_sha256": "b" * 64,
        "conversation_prompt_contract": "openai_full_catalog_v1",
        "lineage": {
            "version": 1,
            "stage": "sft",
            "config_sha256": "c" * 64,
            "model_config_sha256": "d" * 64,
            "data_sha256": "e" * 64,
            "tokenizer_sha256": "f" * 64,
            "git": {
                "commit": "1" * 40,
                "repository_sha256": "2" * 64,
                "dirty": False,
                "worktree_sha256": "3" * 64,
            },
        },
    }
    with pytest.raises(RuntimeError, match="changed before export"):
        to_onnx._training_lineage_export(
            checkpoint_info,
            [identity["sha256"]],
            [identity],
        )


def test_removed_raw_hash_cli_option_is_rejected():
    module = _load_script()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "--config",
                "model.yaml",
                "--checkpoint",
                "accepted.pt",
                "--out",
                "bundle",
                "--training-artifact-sha256",
                SHA256,
            ]
        )


@pytest.mark.parametrize(
    ("stage", "require_guard"),
    [("pretrain", True), ("sft", False)],
)
def test_exporter_guard_preserves_pretrain_and_legacy_paths(
    monkeypatch,
    tmp_path: Path,
    stage: str,
    require_guard: bool,
):
    class ReachedExport(RuntimeError):
        pass

    def stop_after_guard(_model):
        raise ReachedExport

    config_path = ROOT / "configs" / "model" / "ultra-tiny-1m.yaml"
    monkeypatch.setattr(
        to_onnx,
        "_load_cached_decode_checkpoint",
        lambda *args, **kwargs: (
            object(),
            {"stage": stage, "tokenizer_source_path": None},
        ),
    )
    monkeypatch.setattr(
        to_onnx,
        "_state_dict_sha256",
        stop_after_guard,
    )

    with pytest.raises(ReachedExport):
        to_onnx.export_cached_decode(
            str(config_path),
            str(tmp_path / f"{stage}-{require_guard}"),
            pair_role="compatibility_fixture",
            checkpoint_path="fixture.pt",
            fp16=False,
            require_posttraining_training_artifacts=require_guard,
        )
