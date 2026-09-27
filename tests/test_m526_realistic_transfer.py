import json
from pathlib import Path


def test_m526_public_transfer_receipt_is_current_and_fail_closed() -> None:
    raw = Path("docs/paper/results/raw")
    receipt = json.loads(
        (raw / "m526-realistic-cross-surface-warm-random-8step-combined-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["parent_checkpoint"]["sha256"] == (
        "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
    )
    assert {source["label"] for source in receipt["sources"]} == {
        "androidcontrol",
        "agentnet",
        "mind2web",
        "mcpmark",
    }
    assert receipt["comparison"]["aggregate"]["warm_start_better_after"] is True
    assert receipt["comparison"]["aggregate"]["random_after_token_accuracy"] == 0.0
    assert receipt["arms"]["warm"]["after"]["assistant_sequence_accuracy"] == 0.0
    assert "native emulator/browser/desktop/MCP execution" in receipt["interpretation"]["not_claimed"]


def test_m528_gate_keeps_publication_blocked() -> None:
    gate = json.loads(
        Path("docs/paper/results/raw/m528-workshop-gate-current-m526-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert gate["ready"] is False
    weight = next(
        check for check in gate["checks"] if check["requirement"] == "weights:transfer_and_no_transfer_ablation"
    )
    assert weight["status"] == "pass"
    assert any(item["requirement"] == "artifacts:public_model_demo_manifest" for item in gate["blocking_requirements"])


def test_m531_corrected_gate_accepts_official_browsergym_and_larger_ablation() -> None:
    raw = Path("docs/paper/results/raw")
    receipt = json.loads(
        (raw / "m530-realistic-cross-surface-warm-random-64step-combined-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["comparison"]["aggregate"]["warm_after_token_accuracy"] > 0.55
    assert receipt["comparison"]["aggregate"]["random_after_token_accuracy"] > 0.24
    gate = json.loads(
        (raw / "m531-workshop-gate-current-m530-v1.json").read_text(encoding="utf-8")
    )
    assert gate["ready"] is False
    checks = {check["requirement"]: check for check in gate["checks"]}
    assert checks["native:browsergym_miniwob"]["status"] == "pass"
    assert checks["native:mobilegym"]["status"] == "pass"
    assert checks["weights:transfer_and_no_transfer_ablation"]["status"] == "pass"
