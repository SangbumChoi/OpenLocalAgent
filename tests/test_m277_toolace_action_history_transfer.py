"""Integrity checks for the current ToolACE action-history transfer control."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRANSFER = ROOT / "docs/paper/results/raw/m277-toolace-action-history-transfer-v1.json"
WARM_FREE_RUN = ROOT / "docs/paper/results/raw/m277-toolace-action-history-warm-v1.json"
RANDOM_FREE_RUN = ROOT / "docs/paper/results/raw/m277-toolace-action-history-random-v1.json"


def _canonical_hash(payload: dict) -> str:
    return hashlib.sha256(
        (
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode()
    ).hexdigest()


def test_m277_transfer_is_source_disjoint_and_weight_compatible() -> None:
    payload = json.loads(TRANSFER.read_text(encoding="utf-8"))
    assert payload["kind"] == "localagent_cross_surface_transfer_ablation_report"
    assert payload["rows"] == {"train": 256, "eval": 64}
    assert payload["train_sources"][0]["unique_parent_records"] == 256
    assert payload["eval_sources"][0]["unique_parent_records"] == 64
    assert payload["train_sources"][0]["public_reference"]["dataset"] == "Team-ACE/ToolACE"
    assert payload["eval_sources"][0]["input"]["sha256"] != payload["train_sources"][0]["input"]["sha256"]
    assert payload["warm_weight_groups"]["action_heads"]["relative_delta_l2"] == 0.0
    assert payload["random_weight_groups"]["action_heads"]["relative_delta_l2"] == 0.0
    assert payload["aggregate"]["warm_start_better_after"] is True
    assert payload["aggregate"]["warm_minus_random_after_pp"] > 47.0


def test_m277_free_run_receipts_are_hashed_and_keep_the_adoption_boundary() -> None:
    warm = json.loads(WARM_FREE_RUN.read_text(encoding="utf-8"))
    random = json.loads(RANDOM_FREE_RUN.read_text(encoding="utf-8"))
    for payload in (warm, random):
        expected = payload.pop("receipt_self_sha256")
        assert _canonical_hash(payload) == expected
        assert payload["kind"] == "localagent_toolace_action_history_free_run_probe"
        assert payload["source"]["training_used"] is False
        assert payload["source"]["dataset"] == "Team-ACE/ToolACE"
        assert payload["rows_evaluated"] == 16
        assert payload["metrics"]["episode_exact_rate"] == 0.0
    assert warm["metrics"]["tool_exact_rate"] == 0.2
    assert warm["metrics"]["argument_exact_rate"] == 0.0
    assert random["metrics"]["tool_exact_rate"] == 1 / 6
    assert random["metrics"]["step_exact_rate"] == 1 / 30
    assert random["metrics"]["step_exact_rate"] > warm["metrics"]["step_exact_rate"]
