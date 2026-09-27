import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m709-m679-agentnet-selector-transfer-v1.json")


def test_m709_binds_matched_agentnet_selector_transfer() -> None:
    payload = json.loads(RECEIPT.read_text())
    assert payload["kind"] == "localagent_m709_m679_agentnet_selector_transfer"
    assert len(payload["parent"]["sha256"]) == 64
    assert payload["source"]["revision"] == "d76ee50a63fad81cfdbe576416757d7c2091ed50"
    assert payload["source"]["official_split_verified"] is False
    assert payload["training"]["backbone_frozen"] is True
    assert payload["evaluation"]["projected_actions"] == 257
    assert payload["evaluation"]["warm"]["completeness_verified"] is True
    assert payload["evaluation"]["random"]["completeness_verified"] is True


def test_m709_rejects_selector_adoption_and_visual_claims() -> None:
    payload = json.loads(RECEIPT.read_text())
    assert payload["weight_analysis"]["adoption_decision"] == "reject_agentnet_selector_for_webgpu"
    assert payload["source"]["screenshots_consumed"] is False
    assert payload["source"]["desktop_runtime_executed"] is False
