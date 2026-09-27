from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m375-toolace-parent-warm-free-run-v1.json"


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def test_m375_receipt_is_self_hashed_and_dataset_bound() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    expected = receipt.pop("receipt_self_sha256")
    assert hashlib.sha256(_canonical(receipt)).hexdigest() == expected
    assert receipt["kind"] == "localagent_toolace_free_run_parent_child_transfer_receipt"
    assert receipt["dataset"]["dataset"] == "Team-ACE/ToolACE"
    assert receipt["dataset"]["revision"] == "6bda777c88d21e5a204703c1ee45597a8fa4f734"
    assert receipt["dataset"]["rows_evaluated"] == 16
    assert receipt["parent"]["checkpoint"]["sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )
    assert receipt["child"]["checkpoint"]["sha256"] == (
        "7443272ffd8da3de75d6e5eadc41dfc5b15f4d38877b0d6476cff1ca557a34c0"
    )


def test_m375_rejects_policy_promotion_after_free_run_regression() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["parent"]["metrics"]["steps"] == 30
    assert receipt["child"]["metrics"]["steps"] == 30
    assert receipt["parent"]["metrics"]["tool_exact_rate"] == 8 / 30
    assert receipt["child"]["metrics"]["tool_exact_rate"] == 7 / 30
    assert abs(receipt["comparison"]["tool_exact_rate_delta_pp"] + (100 / 30)) < 1e-9
    assert receipt["comparison"]["argument_exact_rate_delta_pp"] == 0.0
    assert receipt["comparison"]["step_exact_rate_delta_pp"] == 0.0
    assert receipt["comparison"]["episode_exact_rate_delta_pp"] == 0.0
    assert receipt["decision"]["adoption"] == "reject_full_policy_promotion"
    assert receipt["decision"]["free_run_tool_exact_improves"] is False
    assert "tool dispatch" in receipt["claim_boundary"].lower()
