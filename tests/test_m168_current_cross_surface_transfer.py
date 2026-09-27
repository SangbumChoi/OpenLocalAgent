from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m168-current-cross-surface-transfer-v1.json"


def _load() -> dict:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def test_m168_receipt_is_hash_bound_and_source_linked() -> None:
    payload = _load()
    claimed = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert claimed == expected
    assert payload["kind"] == "localagent_current_child_cross_surface_transfer_receipt"
    assert payload["rows"] == {"train": 4637, "eval": 1043}
    assert [source["label"] for source in payload["train_sources"]] == [
        "androidcontrol",
        "agentnet",
        "mind2web",
        "mcpmark",
    ]
    references = {
        source["label"]: source["public_reference"] for source in payload["train_sources"]
    }
    assert references["androidcontrol"]["dataset"] == "OfficerChul/Android-Control-84k"
    assert references["agentnet"]["dataset"] == "xlangai/AgentNet"
    assert references["mind2web"]["dataset"] == "osunlp/Mind2Web"
    assert references["mcpmark"]["dataset"] == "Jakumetsu/mcpmark-trajectory-log"
    assert payload["decision"] == "diagnostic_only"
    assert "not an official benchmark score" in payload["claim_boundary"]


def test_m168_split_audit_and_non_visual_boundary_are_explicit() -> None:
    payload = _load()
    audit = payload["split_audit"]
    assert audit["parent_record_disjoint"] is True
    assert audit["train_eval_parent_record_overlap"] == {}
    assert audit["cross_source_train_parent_collisions"] == []
    assert audit["per_source"]["agentnet"] == {
        "train_rows": 513,
        "eval_rows": 133,
        "train_parent_records": 32,
        "eval_parent_records": 8,
    }
    assert audit["per_source"]["mind2web"] == {
        "train_rows": 20,
        "eval_rows": 4,
        "train_parent_records": 5,
        "eval_parent_records": 1,
    }
    assert audit["visual_input_omitted_rows"]["androidcontrol"] == 904


def test_m168_warm_start_dominates_random_on_each_surface() -> None:
    payload = _load()
    comparison = payload["comparison"]
    assert comparison["decision"] == "warm_start_dominates_matched_random_on_all_surfaces"
    aggregate = comparison["aggregate"]
    assert aggregate["warm_start_better_after"] is True
    assert aggregate["warm_minus_random_after_pp"] == 39.30989449084098
    assert set(comparison["surfaces"]) == {"androidcontrol", "agentnet", "mind2web", "mcpmark"}
    assert all(item["warm_start_better_after"] for item in comparison["surfaces"].values())
    assert all(
        item["warm_start"]["after_token_accuracy"] > item["random_backbone"]["after_token_accuracy"]
        for item in comparison["surfaces"].values()
    )
    assert payload["training"]["warm_parent_backbone"]["after"]["eval"][
        "assistant_sequence_accuracy"
    ] == 0.0
    assert payload["training"]["random_backbone_control"]["after"]["eval"][
        "assistant_sequence_accuracy"
    ] == 0.0


def test_m168_weight_transfer_supports_low_rate_body_reuse() -> None:
    payload = _load()
    warm = payload["training"]["warm_parent_backbone"]["weight_transfer"]
    random = payload["training"]["random_backbone_control"]["weight_transfer"]
    expected_compatibility = {
        "added_tensors": [],
        "config_mismatches": {},
        "removed_tensors": [],
        "shape_mismatches": {},
        "shared_tensor_count": 51,
        "tokenizer_sha256_equal": True,
    }
    assert warm["compatibility"] == expected_compatibility
    assert random["compatibility"] == expected_compatibility
    assert warm["groups"]["action_heads"]["relative_delta_l2"] == 0.0
    assert warm["groups"]["embedding"]["relative_delta_l2"] < 0.01
    assert warm["groups"]["attention_or_mixer"]["relative_delta_l2"] < 0.01
    assert warm["groups"]["ffn"]["relative_delta_l2"] < 0.01
    assert random["groups"]["embedding"]["relative_delta_l2"] > 1.0
    assert random["groups"]["attention_or_mixer"]["relative_delta_l2"] > 0.5
    assert random["groups"]["ffn"]["relative_delta_l2"] > 0.5
