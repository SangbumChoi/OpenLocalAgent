import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m708-agentnet-visual-source-audit-v1.json")


def test_m708_binds_pinned_agentnet_visual_archive_and_projection_gap() -> None:
    payload = json.loads(RECEIPT.read_text())
    assert payload["kind"] == "localagent_agentnet_visual_source_audit"
    assert payload["source"]["dataset"] == "xlangai/AgentNet"
    assert len(payload["source"]["revision"]) == 40
    assert payload["trajectory_sample"]["rows"] == 3
    assert payload["trajectory_sample"]["image_reference_count"] > 0
    assert payload["pipeline_boundary"]["visual_archive_present"] is True
    assert payload["pipeline_boundary"]["local_projection_consumes_images"] is False
    assert payload["pipeline_boundary"]["training_admission"].startswith("provenance_only")


def test_m708_records_desktop_action_and_screen_metadata() -> None:
    payload = json.loads(RECEIPT.read_text())
    assert payload["trajectory_sample"]["action_family_counts"]
    assert payload["trajectory_sample"]["metadata_sample_rows"]
