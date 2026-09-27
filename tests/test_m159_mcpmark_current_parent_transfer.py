from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m159-mcpmark-current-parent-transfer-v1.json"


def _load() -> dict:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def test_m159_receipt_self_hash_and_public_boundary() -> None:
    payload = _load()
    claimed = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert claimed == expected
    assert payload["dataset"] == "Jakumetsu/mcpmark-trajectory-log"
    assert payload["source_revision"] == "e50578f0ab904d8e6a7c576c387c1e76ae482c89"
    assert payload["decision"] == "diagnostic_only"
    assert "not an official MCPMark score" in payload["claim_boundary"]


def test_m159_split_is_redacted_and_disjoint() -> None:
    payload = _load()
    split = payload["split_audit"]
    assert split["train_rows"] == 8
    assert split["eval_rows"] == 2
    assert split["train_parent_records"] == 8
    assert split["eval_parent_records"] == 2
    assert split["parent_record_disjoint"] is True
    assert split["tool_outputs_redacted"] is True
    assert split["assistant_text_redacted"] is True
    assert split["visual_input_omitted"] is True


def test_m159_warm_current_parent_beats_random_but_sequence_exact_stays_zero() -> None:
    payload = _load()
    comparison = payload["comparison"]
    assert comparison["warm_start_better_after"] is True
    assert comparison["warm_minus_random_after_pp"] > 40.0
    assert comparison["warm_after_sequence_accuracy"] == 0.0
    assert comparison["random_after_sequence_accuracy"] == 0.0
    warm_groups = payload["training"]["warm"]["weight_transfer"]["groups"]
    assert warm_groups["embedding"]["relative_delta_l2"] < 0.01
    assert warm_groups["attention_or_mixer"]["relative_delta_l2"] < 0.01
    assert warm_groups["ffn"]["relative_delta_l2"] < 0.01
