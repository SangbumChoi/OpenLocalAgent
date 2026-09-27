import json
from pathlib import Path


def _load(name: str) -> dict:
    return json.loads(Path("docs/paper/results/raw", name).read_text(encoding="utf-8"))


def test_m148_browser_adapter_offline_gain_does_not_hide_native_canary_failure() -> None:
    adapter = _load("m148-browser-context-adapter-m142-v1.json")
    assert adapter["checkpoint"]["sha256"] == (
        "8cc3ee42ed38b830b9b04935e156f80d166abeece2dd0c37184ee4d692de7eb1"
    )
    assert adapter["train"]["rows"] == 589
    assert adapter["eval"]["rows"] == 10
    assert adapter["before"]["route_accuracy"] == 0.7
    assert adapter["before"]["tool_accuracy"] == 0.1
    assert adapter["before"]["argument_accuracy"] == 0.1
    assert adapter["after"]["route_accuracy"] == 1.0
    assert adapter["after"]["tool_accuracy"] == 0.7
    assert adapter["after"]["argument_accuracy"] == 0.2
    assert "not a BrowserGym score" in adapter["claim_boundary"]

    canary = _load("m148-browsergym-native-adapter-canary-v1.json")
    assert canary["benchmark_id"] == "browsergym_miniwob"
    assert canary["environment_executed"] is True
    assert canary["official_split_verified"] is False
    assert canary["task_count"] == 10
    assert canary["success_rate"] == 0.0
    assert canary["checkpoint"]["sha256"] == adapter["output"]["sha256"]
    steps = [step for case in canary["cases"] for step in case["steps"]]
    assert len(steps) == 100
    assert sum(bool(step["grounded"]) for step in steps) == 0
    assert sum(step["action"] == "noop(0)" for step in steps) == 100


def test_m148_browser_adapter_weight_audit_is_compatible_but_not_a_quality_pass() -> None:
    report = _load("m148-browser-context-m142-weight-transfer-v1.json")
    compatibility = report["compatibility"]
    assert compatibility["config_mismatches"] == {}
    assert compatibility["shape_mismatches"] == {}
    assert compatibility["shared_tensor_count"] == 51
    assert compatibility["tokenizer_sha256_equal"] is True
    groups = report["groups"]
    assert groups["embedding"]["relative_delta_l2"] > 0.20
    assert groups["action_heads"]["relative_delta_l2"] > 0.70
    assert "not that transfer is optimal" in report["recommendation"]["claim_boundary"]
