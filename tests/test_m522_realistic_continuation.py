import json
from pathlib import Path


_RAW = Path("docs/paper/results/raw")


def _load(name: str) -> dict:
    return json.loads((_RAW / name).read_text(encoding="utf-8"))


def test_m522_comparison_binds_warm_random_arms_to_current_checkpoint() -> None:
    receipt = _load("m522-realistic-cross-surface-warm-random-2step-comparison-v1.json")
    assert receipt["benchmark_id"] == "realistic_agent_cross_surface_warm_random_2step"
    assert receipt["environment_executed"] is False
    assert receipt["official_split_verified"] is False
    assert receipt["parent"]["sha256"] == (
        "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
    )
    assert {source["label"] for source in receipt["sources"]} == {
        "androidcontrol",
        "agentnet",
        "mind2web",
        "mcpmark_fs",
    }
    warm = receipt["arms"]["warm"]
    random = receipt["arms"]["random"]
    assert warm["after"]["assistant_token_accuracy"] > warm["before"]["assistant_token_accuracy"]
    assert warm["after"]["assistant_sequence_accuracy"] == 0.0
    assert random["after"]["assistant_token_accuracy"] == 0.0
    assert warm["weight_movement_relative_l2"]["embedding"] < 0.001
    assert random["weight_movement_relative_l2"]["embedding"] > 1.0
    assert "not an official" in receipt["interpretation"]["not_claimed"]


def test_m523_gate_accepts_weight_ablation_but_stays_fail_closed() -> None:
    receipt = _load("m523-workshop-gate-current-m522-v2.json")
    assert receipt["kind"] == "localagent_workshop_publication_gate"
    assert receipt["ready"] is False
    weight = next(
        check
        for check in receipt["checks"]
        if check["requirement"] == "weights:transfer_and_no_transfer_ablation"
    )
    assert weight["status"] == "pass"
    assert {item["requirement"] for item in receipt["blocking_requirements"]} >= {
        "native:androidworld",
        "native:osworld",
        "artifacts:public_model_demo_manifest",
    }
