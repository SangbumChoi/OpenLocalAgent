from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m372-all-public-candidate-transfer-32step-v1.json"
LABELS = {"androidcontrol", "aitw", "agentnet", "mind2web", "toolace", "xlam"}


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def test_m372_receipt_is_hashed_and_binds_the_matched_public_protocol() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    expected = receipt.pop("receipt_self_sha256")
    assert hashlib.sha256(_canonical(receipt)).hexdigest() == expected
    assert receipt["benchmark_id"] == "cross_surface_all_public_train_candidates_transfer_32step_v1"
    assert receipt["comparison"]["rows"] == {"train": 86, "eval": 80}
    assert receipt["protocol"]["steps"] == 32
    assert receipt["protocol"]["max_seq_len"] == 512
    assert set(receipt["dataset_provenance"]["source_manifest"]) == LABELS
    assert receipt["dataset_provenance"]["official_split_verified"] is False
    assert receipt["dataset_provenance"]["native_execution"] is False


def test_m372_warm_parent_dominates_random_without_sequence_success() -> None:
    comparison = json.loads(RECEIPT.read_text(encoding="utf-8"))["comparison"]
    assert comparison["aggregate"]["warm_start_better_after"] is True
    assert comparison["aggregate"]["warm_after_token_accuracy"] == 0.5535307517084282
    assert comparison["aggregate"]["random_after_token_accuracy"] == 0.07431662870159453
    assert comparison["aggregate"]["warm_minus_random_after_pp"] > 47.9
    assert all(item["warm_start_better_after"] for item in comparison["surfaces"].values())
    assert "not an official benchmark score" in comparison["claim_boundary"]


def test_m372_weight_lineage_supports_initialization_only() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    warm = receipt["weight_transfer_analysis"]["warm"]
    random = receipt["weight_transfer_analysis"]["random"]
    assert warm["compatibility"]["shared_tensor_count"] == 51
    assert warm["compatibility"]["tokenizer_sha256_equal"] is True
    for group in ("embedding", "attention_or_mixer", "ffn", "normalization"):
        assert warm["groups"][group]["relative_delta_l2"] < 0.005
        assert random["groups"][group]["relative_delta_l2"] > 0.05
    assert warm["groups"]["action_heads"]["relative_delta_l2"] == 0.0
    assert "less than 0.5%" in receipt["weight_transfer_analysis"]["interpretation"]
    assert receipt["decision"]["export_child_to_webgpu"] is False
