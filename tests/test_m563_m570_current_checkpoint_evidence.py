import hashlib
import json
from pathlib import Path


ROOT = Path("docs/paper/results/raw")
CURRENT_SHA = "43b32b15899bcd97d7a822ef8441b04521d863b30c23c17080dbfaf4f0a14d7c"


def _load_verified(name: str) -> dict:
    payload = json.loads((ROOT / name).read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    return payload


def test_m563_lineage_chain_is_checkpoint_bound() -> None:
    payload = _load_verified("m563-m553-lineage-training-v1.json")
    assert payload["chain"]["grounded_child"]
    assert payload["chain"]["multisurface_child"] == payload["candidate_checkpoint"]["sha256"]
    assert payload["candidate_checkpoint"]["lineage_stage"] == "sft"


def test_m564_rl_preflight_has_gate_visible_lineage() -> None:
    payload = _load_verified("m564-m553-stateful-rl-preflight-v1.json")
    assert payload["status"] == "passed"
    assert payload["metrics"]["lineage"]["parent_checkpoint_sha256"] == CURRENT_SHA
    assert payload["measurement"]["realized_optimizer_updates"] == 2


def test_m565_transfer_ablation_has_matched_random_control() -> None:
    payload = _load_verified("m565-m553-transfer-ablation-v1.json")
    assert payload["parent_checkpoint"]["sha256"] == CURRENT_SHA
    comparison = payload["comparison"]
    assert comparison["aggregate"]["warm_start_better_after"] is True
    assert comparison["surfaces"].keys() == {"androidcontrol", "agentnet"}
    assert payload["weight_transfer_analysis"]["warm"]["compatibility"]["config_mismatches"] == {}


def test_m566_webgpu_capability_is_native_and_checkpoint_bound() -> None:
    payload = _load_verified("m566-m553-webgpu-capability-v1.json")
    assert payload["checkpoint"]["sha256"] == CURRENT_SHA
    assert payload["backend"] == "webgpu"
    assert payload["environment_executed"] is True
    assert payload["capability"]["exact_actions"] == 3
    assert payload["performance"]["tokens_per_second_p50"] >= 100


def test_m567_m568_native_receipts_are_official_split_negative_controls() -> None:
    mobile = _load_verified("m567-m553-mobilegym-native-full-v1.json")
    browser = _load_verified("m568-m553-browsergym-native-full-v1.json")
    assert mobile["checkpoint_sha256"] == CURRENT_SHA
    assert browser["checkpoint_sha256"] == CURRENT_SHA
    assert mobile["environment"]["official_split_verified"] is True
    assert browser["environment"]["official_split_verified"] is True
    assert mobile["result"]["success_rate"] == 1 / 256
    assert browser["result"]["success_rate"] == 5 / 240


def test_m570_workshop_gate_is_fail_closed_with_passed_current_checks() -> None:
    payload = _load_verified("m570-workshop-gate-current-m553-v1.json")
    assert payload["ready"] is False
    assert payload["current_checkpoint"]["sha256"] == CURRENT_SHA
    assert payload["checks"]["native_mobilegym"] == "pass"
    assert payload["checks"]["native_browsergym_miniwob"] == "pass"
    assert payload["checks"]["webgpu_native_capability_and_latency"] == "pass"
    assert payload["checks"]["weights_transfer_and_no_transfer_ablation"] == "pass"
    assert payload["checks"]["training_rl_preflight"] == "pass"
    assert "native:androidworld" in payload["blocking_requirements"]
    assert "artifacts:public_model_demo_manifest" in payload["blocking_requirements"]


def test_m571_public_matrix_audit_preserves_train_eval_boundary() -> None:
    payload = _load_verified("m571-public-realistic-eval-matrix-audit-v1.json")
    assert payload["matrix"]["entries"] == 28
    assert payload["coverage"]["families"] == {
        "browser": 4,
        "computer": 4,
        "mobile": 9,
        "terminal": 1,
        "tool_api": 10,
    }
    assert payload["coverage"]["train_eligible"] == [
        "androidcontrol",
        "android_in_the_wild",
        "mind2web",
        "agentnet",
        "xlam_function_calling",
        "toolace",
    ]
    assert len(payload["upstream"]) == 10
    assert all(row["revision"] for row in payload["upstream"])


def test_m573_gate_binds_refreshed_catalog_without_promoting_runtime_rows() -> None:
    payload = _load_verified("m573-workshop-gate-current-m553-matrix-refresh-v1.json")
    assert payload["ready"] is False
    assert payload["catalog"]["entries"] == 42
    assert payload["matrix_audit"]["entries"] == 28
    assert payload["checks"]["catalog_realistic_family_coverage"] == "pass"
    assert payload["checks"]["training_rl_preflight"] == "pass"
    assert "native:androidworld" in payload["blocking_requirements"]


def test_m574_xlam_derivative_transfer_is_explicitly_non_official() -> None:
    payload = _load_verified("m574-xlam-derived-warm-random-transfer-v1.json")
    assert payload["parent_checkpoint"]["sha256"] == CURRENT_SHA
    assert payload["source"]["original_dataset"] == "Salesforce/xlam-function-calling-60k"
    assert payload["source"]["original_access"] == "gated_in_current_environment"
    assert payload["source"]["normalization"]["train_records"] == 256
    assert payload["source"]["normalization"]["eval_records"] == 128
    assert payload["metrics"]["warm_start_better_after"] is True
    assert payload["metrics"]["warm_minus_random_after_pp"] == 31.32127955493741
    assert payload["metrics"]["sequence_exact_warm"] == 0.0


def test_m575_gate_accepts_xlam_transfer_without_claiming_public_readiness() -> None:
    payload = _load_verified("m575-workshop-gate-current-m553-xlam-v1.json")
    assert payload["ready"] is False
    assert payload["checks"]["weights_transfer_and_no_transfer_ablation"] == "pass"
    assert (
        "docs/paper/results/raw/m574-xlam-derived-warm-random-transfer-v1.json"
        in payload["weight_reports"]
    )
    assert "artifacts_public_model_demo_manifest" in payload["checks"]


def test_m576_public_release_audit_rejects_legacy_url_as_current() -> None:
    payload = _load_verified("m576-public-release-audit-v1.json")
    assert payload["public"] is True
    assert payload["current_checkpoint"]["sha256"] == CURRENT_SHA
    assert payload["public_existing_release"]["current_checkpoint_match"] is False
    assert payload["local_candidate"]["published"] is False
    assert payload["local_candidate"]["webgpu_bundle_verified"] is True
    assert payload["local_candidate"]["onnx_parity_gate"] is True
    assert payload["current_checkpoint_sha256"] is None


def test_m577_gate_distinguishes_public_legacy_artifact_from_current_candidate() -> None:
    payload = _load_verified("m577-workshop-gate-current-m553-public-audit-v1.json")
    assert payload["ready"] is False
    assert payload["checks"]["artifacts_public_model_demo_manifest"] == "blocked"
    assert payload["checks"]["public_artifact_blocker"] == "current_checkpoint_not_bound"
    assert payload["evidence"]["local_candidate_checkpoint_match"] is True
    assert payload["evidence"]["existing_public_release_checkpoint_match"] is False
    assert "native:androidworld" in payload["blocking_requirements"]


def test_m578_toolsandbox_smoke_is_current_but_not_official_split() -> None:
    payload = _load_verified("m578-toolsandbox-native-m553-v1.json")
    assert payload["checkpoint_sha256"] == CURRENT_SHA
    assert payload["environment_executed"] is True
    assert payload["verifier_executed"] is True
    assert payload["official_split_verified"] is False
    assert payload["success_rate"] == 1.0
    assert payload["task_count"] == 3


def test_m579_gate_records_toolsandbox_official_split_blocker() -> None:
    payload = _load_verified("m579-workshop-gate-current-m553-toolsandbox-v1.json")
    assert payload["ready"] is False
    assert payload["checks"]["native_toolsandbox"] == "blocked"
    assert payload["checks"]["native_toolsandbox_blocker"] == "official_split_not_verified"
    assert payload["evidence"]["toolsandbox_current_checkpoint_bound"] is True
    assert "native:toolsandbox" in payload["blocking_requirements"]


def test_m580_toolsandbox_interactive_stress_is_a_negative_control() -> None:
    payload = _load_verified("m580-toolsandbox-m553-interactive-stress-v1.json")
    assert payload["checkpoint_sha256"] == CURRENT_SHA
    assert payload["protocol"] == "bounded_multi_step_scripted_user"
    assert payload["official_split_verified"] is False
    assert payload["task_count"] == 3
    assert payload["success_count"] == 0
    assert payload["success_rate"] == 0.0
    assert max(item["turn_count"] for item in payload["scenarios"]) == 13


def test_m583_stateful_head_transfer_is_negative_native_bridge() -> None:
    payload = _load_verified("m583-stateful-head-transfer-toolsandbox-v1.json")
    assert payload["parent_checkpoint"]["sha256"] == CURRENT_SHA
    assert payload["local_probe"]["warm"]["selector_top1"] > payload["local_probe"]["matched_random"]["selector_top1"]
    assert payload["local_probe"]["warm"]["closed_loop_success_rate"] == payload["local_probe"]["matched_random"]["closed_loop_success_rate"]
    assert payload["native_toolsandbox_bridge"]["parent_success_rate"] == 0.0
    assert payload["native_toolsandbox_bridge"]["child_success_rate"] == 0.0
    assert payload["native_toolsandbox_bridge"]["single_step_child_success_rate"] == 1.0
