from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m156-cua-gym-surface-probe-v1.json"


def _load() -> dict:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def test_m156_receipt_self_hash_and_source_boundary() -> None:
    payload = _load()
    claimed = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert claimed == expected
    assert payload["dataset"] == "xlangai/CUA-Gym"
    assert payload["source_revision"] == "3c021d0"
    assert "not CUA-Gym task success" in payload["claim_boundary"]


def test_m156_is_task_disjoint_frozen_probe_with_matched_control() -> None:
    payload = _load()
    assert payload["selection"]["all_metadata_rows"] == 10910
    assert payload["selection"]["selected_rows"] == 3005
    assert payload["selection"]["task_id_disjoint"] is True
    warm = payload["arms"]["warm"]
    random_arm = payload["arms"]["random"]
    assert warm["backbone_updated"] is False
    assert random_arm["backbone_updated"] is False
    assert warm["backbone_relative_l2"] == 0.0
    assert random_arm["backbone_relative_l2"] == 0.0
    assert payload["comparison"]["eval_accuracy_delta_warm_minus_random"] > 0.0
    assert payload["decision"] == "diagnostic_only"
