from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m219-toolace-public-source-transfer-audit-v1.json"


def test_m219_toolace_receipt_is_hash_bound_and_split_safe() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    claimed = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert claimed == expected
    assert payload["dataset"]["url"] == "https://huggingface.co/datasets/Team-ACE/ToolACE"
    assert payload["dataset"]["license"] == "Apache-2.0"
    assert payload["projection"]["train_eval_parent_overlap"] == 0
    assert payload["projection"]["train_eval_prompt_overlap"] == 0
    assert payload["projection"]["accepted_rows"] == 8993


def test_m219_transfer_evidence_supports_backbone_reuse_not_capability_promotion() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    evidence = payload["transfer_evidence"]
    assert [item["projection"] for item in evidence] == [
        "first_action",
        "multiturn",
        "action_history",
    ]
    assert all(item["warm_minus_random_after_pp"] > 0 for item in evidence)
    assert all(item["warm_sequence_exact"] == 0.0 for item in evidence)
    assert payload["weight_adoption"]["compatible_shared_tensors"] == 51
    assert "retain_pretrained_backbone" in payload["weight_adoption"]["recommendation"]
    assert "do_not_promote_tool_heads" in payload["weight_adoption"]["recommendation"]
