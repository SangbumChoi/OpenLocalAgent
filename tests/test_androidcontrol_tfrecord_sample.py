import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m707-androidcontrol-official-tfrecord-visual-sample-v1.json")


def test_m707_binds_official_visual_source_and_pipeline_gap() -> None:
    payload = json.loads(RECEIPT.read_text())
    assert payload["kind"] == "localagent_androidcontrol_official_tfrecord_visual_sample"
    assert payload["episode"]["episode_id"] == 0
    assert payload["episode"]["split"] == "train"
    assert payload["episode"]["screenshot_count"] == 4
    assert payload["episode"]["screenshot_total_bytes"] > 500_000
    assert payload["episode"]["accessibility_tree_count"] == 4
    assert payload["episode"]["action_count"] == 3
    assert payload["pipeline_boundary"]["official_visual_bytes_present"] is True
    assert payload["pipeline_boundary"]["current_localagent_projection_consumes_screenshots"] is False
    assert payload["pipeline_boundary"]["training_admission"].startswith("provenance_only")
    assert payload["source"]["original_repository"].startswith("https://github.com/google-research")


def test_m707_has_stable_receipt_fields() -> None:
    payload = json.loads(RECEIPT.read_text())
    assert len(payload["bounded_fetch"]["prefix_sha256"]) == 64
    assert len(payload["episode"]["screenshot_bytes_sha256"]) == 64
    assert payload["episode"]["action_types"] == ["open_app", "wait", "click"]
