from __future__ import annotations

import hashlib
import json
from pathlib import Path


RECEIPT = Path(__file__).resolve().parents[1] / "docs/paper/results/raw/m185-mcpmark-current-m180-transfer-v1.json"


def _load() -> dict:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def test_m185_receipt_is_hash_bound_and_source_pinned() -> None:
    payload = _load()
    claimed = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert claimed == expected
    assert payload["dataset"] == "Jakumetsu/mcpmark-trajectory-log"
    assert payload["source_revision"] == "e50578f0ab904d8e6a7c576c387c1e76ae482c89"
    assert payload["inputs"]["parent_record_disjoint"] is True
    assert payload["inputs"]["tool_outputs_redacted"] is True
    assert payload["decision"] == "diagnostic_only"


def test_m185_warm_control_and_weight_lineage_are_explicit() -> None:
    payload = _load()
    warm = payload["training"]["warm"]
    random = payload["training"]["random"]
    assert warm["after_eval"]["assistant_token_accuracy"] == 0.42726580350342724
    assert random["after_eval"]["assistant_token_accuracy"] == 0.04722010662604722
    assert payload["comparison"]["warm_minus_random_after_pp"] == 38.004569687738005
    assert warm["after_eval"]["sequence_accuracy"] == 0.0
    assert random["after_eval"]["sequence_accuracy"] == 0.0
    assert warm["relative_delta_l2"]["embedding"] < 0.01
    assert warm["relative_delta_l2"]["attention_or_mixer"] < 0.01
    assert random["relative_delta_l2"]["embedding"] > 1.0
    assert random["relative_delta_l2"]["attention_or_mixer"] > 0.5
    assert "not an official MCPMark score" in payload["claim_boundary"]
