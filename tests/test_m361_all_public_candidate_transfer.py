from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m361-all-public-candidate-transfer-v1.json"
LABELS = {"androidcontrol", "aitw", "agentnet", "mind2web", "toolace", "xlam"}


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def test_m361_receipt_is_self_hashed_and_source_bound() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    expected = receipt.pop("receipt_self_sha256")
    assert hashlib.sha256(_canonical(receipt)).hexdigest() == expected
    assert receipt["benchmark_id"] == "cross_surface_all_public_train_candidates_transfer"
    assert receipt["parent_checkpoint"]["sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )

    manifest = receipt["dataset_provenance"]["source_manifest"]
    assert set(manifest) == LABELS
    assert all(item["train"]["rows"] == 4 and item["eval"]["rows"] == 4 for item in manifest.values())
    assert receipt["dataset_provenance"]["official_split_verified"] is False
    assert receipt["dataset_provenance"]["native_execution"] is False
    assert "not the official AITW test split" in receipt["dataset_provenance"]["projection_boundaries"]["aitw"]


def test_m361_warm_start_beats_random_on_each_bounded_surface() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    comparison = receipt["comparison"]
    assert comparison["rows"] == {"train": 24, "eval": 24}
    assert comparison["aggregate"]["warm_start_better_after"] is True
    assert comparison["aggregate"]["warm_minus_random_after_pp"] > 50.0
    assert all(item["warm_start_better_after"] for item in comparison["surfaces"].values())
    assert comparison["surfaces"]["aitw"]["warm_start"]["after_token_accuracy"] == 1 / 9


def test_m361_weight_transfer_recommends_lineage_without_promotion() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    warm_groups = receipt["weight_transfer_analysis"]["warm"]["groups"]
    random_groups = receipt["weight_transfer_analysis"]["random"]["groups"]
    for group in ("embedding", "attention_or_mixer", "ffn", "normalization"):
        assert warm_groups[group]["relative_delta_l2"] < 0.01
        assert random_groups[group]["relative_delta_l2"] > 0.05
    assert receipt["decision"]["export_child_to_webgpu"] is False
    assert receipt["decision"]["native_promotion"] is False
