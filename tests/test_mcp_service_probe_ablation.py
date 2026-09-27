from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.train_mcp_service_probe_ablation import (
    MCPMARK_REVISION,
    _combined_mcpmark_metrics,
    _receipt_sha256,
)


def test_ablation_binds_published_transfer_receipt() -> None:
    path = Path(__file__).parents[1] / "docs/paper/results/raw/m38-mcp-service-contract-probe-v1.json"
    assert MCPMARK_REVISION == json.loads(path.read_text(encoding="utf-8"))["dataset"]["revision"]
    assert len(_receipt_sha256(path)) == 64


def test_published_no_transfer_receipt_is_self_hashed_and_matched() -> None:
    path = Path(__file__).parents[1] / (
        "docs/paper/results/raw/m53-mcp-service-contract-no-transfer-v1.json"
    )
    receipt = json.loads(path.read_text(encoding="utf-8"))
    expected = receipt.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert actual == expected
    assert receipt["comparison"]["rows"] == 239
    assert receipt["comparison"]["random_minus_transfer"]["selector_top1"] < -0.1


def test_mind2web_mcp_probe_records_a_matched_cross_domain_negative() -> None:
    root = Path(__file__).parents[1] / "docs/paper/results/raw"
    transfer = json.loads(
        (root / "m93-mind2web-mcp-service-contract-v1.json").read_text(encoding="utf-8")
    )
    random = json.loads(
        (root / "m94-mind2web-mcp-service-random-matched-v1.json").read_text(encoding="utf-8")
    )
    assert transfer["weight_delta"]["backbone"] == 0.0
    assert transfer["parent"]["sha256"] == random["parent"]["sha256"]
    assert random["transfer_reference"]["receipt"].endswith(
        "m93-mind2web-mcp-service-contract-v1.json"
    )
    transfer_metrics = _combined_mcpmark_metrics(transfer)
    assert transfer_metrics["route_accuracy"] == random["transfer_reference"][
        "combined_route_accuracy"
    ]

    def combined(receipt: dict) -> tuple[float, float, float]:
        rows = [receipt["mcpmark"][name]["overall"] for name in ("standard", "easy")]
        total = sum(item["rows"] for item in rows)
        return tuple(
            sum(item[key] * item["rows"] for item in rows) / total
            for key in ("route_accuracy", "selector_top1", "selector_top3")
        )

    transfer_scores = combined(transfer)
    random_scores = combined(random)
    assert random_scores[0] - transfer_scores[0] > 0.20
    assert transfer_scores[1] - random_scores[1] > 0.05
    assert transfer_scores[2] == random_scores[2]


def test_lowrate_mind2web_mcp_probe_binds_all_three_arms_and_weight_movement() -> None:
    path = Path(__file__).parents[1] / (
        "docs/paper/results/raw/m96-mind2web-mcp-service-lowrate-matched-v1.json"
    )
    receipt = json.loads(path.read_text(encoding="utf-8"))
    expected = receipt.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert actual == expected
    assert receipt["comparison"]["rows"] == 239
    assert receipt["comparison"]["lowrate_minus_random"]["route_accuracy"] > 0.0
    assert receipt["comparison"]["lowrate_minus_random"]["selector_top1"] < 0.0
    assert receipt["weight_delta"]["groups"]["backbone"] == pytest.approx(
        0.002970523722023013
    )
