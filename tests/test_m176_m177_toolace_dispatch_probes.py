import json
from pathlib import Path


SELECTOR = Path("docs/paper/results/raw/m176-current-child-toolace-action-history-selector-transfer-v1.json")
SELECTOR_RUN = Path("docs/paper/results/raw/m176-current-child-toolace-action-history-selector-free-run-v1.json")
POINTER = Path("docs/paper/results/raw/m177-current-child-toolace-action-history-pointer-transfer-v1.json")
POINTER_RUN = Path("docs/paper/results/raw/m177-current-child-toolace-action-history-pointer-free-run-v1.json")


def test_m176_selector_control_blocks_representation_adoption() -> None:
    payload = json.loads(SELECTOR.read_text(encoding="utf-8"))
    assert payload["kind"] == "localagent_toolace_action_history_selector_transfer_probe"
    assert payload["source"]["dataset"] == "Team-ACE/ToolACE"
    assert payload["source"]["revision"] == "6bda777c88d21e5a204703c1ee45597a8fa4f734"
    assert payload["arms"]["retrained_pretrained_backbone"]["metrics"]["selector_top1"] == 0.2920353982300885
    assert payload["arms"]["retrained_matched_random_backbone"]["metrics"]["selector_top1"] == 0.3274336283185841
    assert payload["decision"]["transfer_beats_random_top1"] is False


def test_m176_free_run_receipt_is_bounded() -> None:
    payload = json.loads(SELECTOR_RUN.read_text(encoding="utf-8"))
    assert payload["rows_requested"] == payload["rows_evaluated"] == 16
    assert payload["metrics"]["steps"] == 30
    assert payload["metrics"]["tool_exact_rate"] == 0.16666666666666666
    assert "no tool dispatch" in payload["claim_boundary"]


def test_m177_pointer_control_blocks_adoption() -> None:
    payload = json.loads(POINTER.read_text(encoding="utf-8"))
    assert payload["kind"] == "localagent_toolace_action_history_pointer_transfer_probe"
    assert payload["arms"]["retrained_pretrained_backbone"]["metrics"]["coverage"] == 0.49206349206349204
    assert payload["arms"]["retrained_pretrained_backbone"]["metrics"]["decoded_value_exact"] == 0.0967741935483871
    assert payload["arms"]["retrained_matched_random_backbone"]["metrics"]["decoded_value_exact"] == 0.1935483870967742
    assert payload["decision"]["transfer_beats_random_decoded_value"] is False


def test_m177_free_run_receipt_records_pointer_probe_boundary() -> None:
    payload = json.loads(POINTER_RUN.read_text(encoding="utf-8"))
    assert payload["metrics"]["tool_exact_rate"] == 0.13333333333333333
    assert payload["metrics"]["argument_exact_rate"] == 0.03333333333333333
    assert "no tool dispatch" in payload["claim_boundary"]
