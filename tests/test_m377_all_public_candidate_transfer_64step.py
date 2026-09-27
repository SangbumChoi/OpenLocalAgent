from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m377-all-public-candidate-transfer-64step-v1.json"
LABELS = {"androidcontrol", "aitw", "agentnet", "mind2web", "toolace", "xlam"}


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def test_m377_receipt_binds_source_local_split_and_current_parent() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    expected = receipt.pop("receipt_self_sha256")
    assert hashlib.sha256(_canonical(receipt)).hexdigest() == expected
    assert receipt["benchmark_id"] == "cross_surface_all_public_train_candidates_transfer_64step_v1"
    assert receipt["parent_checkpoint"]["sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )
    assert receipt["comparison"]["rows"] == {"train": 86, "eval": 80}
    assert receipt["protocol"]["steps"] == 64
    assert receipt["protocol"]["max_seq_len"] == 512
    assert receipt["protocol"]["split_contract"]["mode"] == (
        "source_local_parent_and_slot_disjoint"
    )
    assert set(receipt["protocol"]["split_contract"]["labels_checked"]) == LABELS
    assert receipt["dataset_provenance"]["official_split_verified"] is False
    assert receipt["dataset_provenance"]["native_execution"] is False


def test_m377_warm_transfer_wins_all_surfaces_without_sequence_success() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    comparison = receipt["comparison"]
    assert comparison["aggregate"]["warm_after_token_accuracy"] == 0.5617881548974943
    assert comparison["aggregate"]["random_after_token_accuracy"] == 0.31264236902050113
    assert comparison["aggregate"]["warm_minus_random_after_pp"] > 24.9
    assert comparison["aggregate"]["warm_start_better_after"] is True
    assert all(item["warm_start_better_after"] for item in comparison["surfaces"].values())
    assert receipt["decision"]["export_child_to_webgpu"] is False
    assert receipt["decision"]["native_promotion"] is False


def test_m377_weight_lineage_separates_warm_and_random_movement() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    warm = receipt["weight_transfer_analysis"]["warm"]
    random = receipt["weight_transfer_analysis"]["random"]
    assert warm["compatibility"]["shared_tensor_count"] == 51
    assert warm["compatibility"]["tokenizer_sha256_equal"] is True
    assert warm["groups"]["embedding"]["relative_delta_l2"] == 0.008371738373755204
    assert warm["groups"]["action_heads"]["relative_delta_l2"] == 0.0
    assert random["groups"]["attention_or_mixer"]["relative_delta_l2"] > 0.7
    assert random["groups"]["embedding"]["relative_delta_l2"] > 1.1
    assert "approximately 0.837%" in receipt["weight_transfer_analysis"]["interpretation"]
