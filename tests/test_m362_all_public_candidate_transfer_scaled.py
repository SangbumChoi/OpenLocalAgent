from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m362-all-public-candidate-transfer-scaled-v1.json"
LABELS = {"androidcontrol", "aitw", "agentnet", "mind2web", "toolace", "xlam"}


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def test_m362_scaled_receipt_is_hashed_and_has_expanded_public_rows() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    expected = receipt.pop("receipt_self_sha256")
    assert hashlib.sha256(_canonical(receipt)).hexdigest() == expected
    assert receipt["benchmark_id"] == "cross_surface_all_public_train_candidates_transfer_scaled_v1"
    assert receipt["comparison"]["rows"] == {"train": 86, "eval": 80}
    assert set(receipt["dataset_provenance"]["source_manifest"]) == LABELS
    assert receipt["dataset_provenance"]["official_split_verified"] is False
    assert receipt["dataset_provenance"]["native_execution"] is False


def test_m362_warm_start_wins_on_all_surfaces_without_sequence_success() -> None:
    comparison = json.loads(RECEIPT.read_text(encoding="utf-8"))["comparison"]
    assert comparison["aggregate"]["warm_start_better_after"] is True
    assert comparison["aggregate"]["warm_minus_random_after_pp"] > 50.0
    assert all(item["warm_start_better_after"] for item in comparison["surfaces"].values())
    assert comparison["aggregate"]["warm_after_token_accuracy"] == 0.5259111617312073
    assert all(
        item["warm_start"]["after_token_accuracy"] >= 0.0
        and item["random_backbone"]["after_token_accuracy"] == 0.0
        for item in comparison["surfaces"].values()
    )


def test_m362_weight_lineage_remains_compatible_and_low_rate() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    warm = receipt["weight_transfer_analysis"]["warm"]
    random = receipt["weight_transfer_analysis"]["random"]
    assert warm["compatibility"]["shared_tensor_count"] == 51
    assert warm["compatibility"]["tokenizer_sha256_equal"] is True
    for group in ("embedding", "attention_or_mixer", "ffn", "normalization"):
        assert warm["groups"][group]["relative_delta_l2"] < 0.01
        assert random["groups"][group]["relative_delta_l2"] > 0.05
    assert receipt["decision"]["export_child_to_webgpu"] is False
