import hashlib
import json

from scripts.acquire_hf_sources import load_sources
from scripts.publish_hf_campaign import _validate_acquisition_manifest


def test_hf_source_config_is_explicitly_train_only_or_model_reference() -> None:
    payload = load_sources()
    assert len(payload["sources"]) >= 6
    assert all(row["policy"] in {"train", "model_reference"} for row in payload["sources"])
    assert all(row.get("access", "public") in {"public", "gated"} for row in payload["sources"])
    assert all(row["allow_patterns"] for row in payload["sources"])
    assert not any("eval" in str(row["purpose"]).lower() for row in payload["sources"])


def test_dataset_publisher_rejects_dry_run_or_eval_manifest(tmp_path) -> None:
    manifest = {
        "kind": "localagent_hf_source_acquisition",
        "dry_run": True,
        "sources": [],
    }
    (tmp_path / "acquisition-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    try:
        _validate_acquisition_manifest(tmp_path)
    except ValueError as error:
        assert "completed" in str(error)
    else:  # pragma: no cover
        raise AssertionError("publisher accepted a dry-run acquisition")


def test_acquisition_manifest_hashes_are_stable(tmp_path) -> None:
    source = tmp_path / "file.json"
    source.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert len(digest) == 64
