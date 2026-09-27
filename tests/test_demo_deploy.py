import hashlib
import json
from pathlib import Path

import pytest

from localagent.eval.demo_deploy import BUNDLE_FILES, STATIC_FILES, sync_demo_bundle, verify_demo_deploy


def _write_valid_bundle(root: Path) -> None:
    for name in (*STATIC_FILES, *BUNDLE_FILES):
        (root / name).write_bytes(name.encode("utf-8"))
    (root / "app.js").write_text("fetch('bundle-manifest.json')", encoding="utf-8")
    artifacts = {
        name: {
            "file": name,
            "bytes": (root / name).stat().st_size,
            "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
        }
        for name in BUNDLE_FILES
    }
    (root / "meta.json").write_text(
        json.dumps({"action_model_file": "action_model.fp16.onnx", "tools": ["tool"]}),
        encoding="utf-8",
    )
    # meta.json is itself an artifact, so update its identity after writing it.
    artifacts["meta.json"] = {
        "file": "meta.json",
        "bytes": (root / "meta.json").stat().st_size,
        "sha256": hashlib.sha256((root / "meta.json").read_bytes()).hexdigest(),
    }
    (root / "bundle-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "checkpoint_sha256": "checkpoint",
                "artifacts": artifacts,
                "parity_gate": {"hard_gate": True, "passed": True},
            }
        ),
        encoding="utf-8",
    )


def test_verify_valid_bundle(tmp_path: Path) -> None:
    _write_valid_bundle(tmp_path)
    report = verify_demo_deploy(tmp_path)
    assert report["verified"] is True
    assert report["blockers"] == []
    assert len(report["bundle_identity_sha256"]) == 64


def test_missing_manifest_is_explicit_failure(tmp_path: Path) -> None:
    for name in STATIC_FILES:
        (tmp_path / name).write_text("ok", encoding="utf-8")
    report = verify_demo_deploy(tmp_path)
    assert report["verified"] is False
    assert "manifest:missing_or_non_regular_file" in report["blockers"]


def test_hash_mismatch_is_explicit_failure(tmp_path: Path) -> None:
    _write_valid_bundle(tmp_path)
    (tmp_path / "heads.json").write_text("tampered", encoding="utf-8")
    report = verify_demo_deploy(tmp_path)
    assert "artifact:heads.json:identity_mismatch" in report["blockers"]


def test_expected_checkpoint_and_tool_pool_bindings_are_fail_closed(tmp_path: Path) -> None:
    _write_valid_bundle(tmp_path)
    assert verify_demo_deploy(
        tmp_path,
        expected_checkpoint_sha256="checkpoint",
        expected_tool_count=1,
    )["verified"] is True
    checkpoint_mismatch = verify_demo_deploy(tmp_path, expected_checkpoint_sha256="other")
    assert "manifest:checkpoint_sha256_mismatch" in checkpoint_mismatch["blockers"]
    tool_mismatch = verify_demo_deploy(tmp_path, expected_tool_count=63)
    assert "meta:tool_count_mismatch:expected=63:actual=1" in tool_mismatch["blockers"]


def test_separate_source_bundle_requires_target_staging(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_valid_bundle(source)
    target = tmp_path / "target"
    target.mkdir()
    for name in STATIC_FILES:
        (target / name).write_bytes((source / name).read_bytes())
    report = verify_demo_deploy(target, bundle_dir=source)
    assert report["verified"] is False
    assert report["deployment_bundle_present"] is False
    assert "deployment_target:bundle_not_staged" in report["blockers"]


def test_sync_requires_verified_source_and_copies_bundle(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    _write_valid_bundle(source)
    for name in STATIC_FILES:
        (target / name).write_bytes((source / name).read_bytes())
    report = sync_demo_bundle(source, target)
    assert report["verified"] is True
    assert (target / "bundle-manifest.json").is_file()
    assert (target / "model.onnx").read_bytes() == b"model.onnx"


def test_sync_rejects_invalid_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unverified bundle"):
        sync_demo_bundle(tmp_path, tmp_path / "target")
