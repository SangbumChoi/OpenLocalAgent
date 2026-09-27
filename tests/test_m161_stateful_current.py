import json
from pathlib import Path


def test_m161_current_stateful_transfer_receipt_is_hash_bound_and_non_native() -> None:
    path = Path(__file__).parents[1] / "docs/paper/results/raw/m161-stateful-productivity-current-transfer-v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["kind"] == "localagent_stateful_productivity_current_transfer_receipt"
    assert payload["decision"] == "diagnostic_only"
    assert payload["source"]["native_runtime_executed"] is False
    assert payload["source"]["tools_executed"] is False
    assert payload["arms"]["pretrained_frozen_backbone"]["closed_loop"]["steps"] == 16
    assert payload["arms"]["pretrained_lowrate_unfrozen_backbone"]["closed_loop"][
        "closed_loop_success_rate"
    ] == 0.3125
    assert payload["arms"]["matched_random_backbone"]["closed_loop"]["closed_loop_success_rate"] == 0.0625
    assert "real email" in payload["claim_boundary"]
