import json
from pathlib import Path

from localagent.eval.workshop_gate import build_workshop_gate, write_workshop_gate


ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "configs/data/realistic-agent-eval.catalog.yaml"


def test_default_gate_is_not_ready_and_lists_native_blockers() -> None:
    report = build_workshop_gate(CATALOG, repo_root=ROOT)
    assert report["ready"] is False
    requirements = {item["requirement"] for item in report["blocking_requirements"]}
    assert "native:androidworld" in requirements
    assert "native:mobilegym" in requirements
    assert "native:mobile_safety_bench" in requirements
    assert "native:iosworld" in requirements
    assert "native:agentnet" in requirements
    assert "native:osworld_v2" in requirements
    assert "webgpu:native_capability_and_latency" in requirements
    assert "artifacts:public_model_demo_manifest" in requirements
    assert "training:rl_preflight" in requirements


def test_rl_preflight_receipt_must_pass_and_bind_current_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"current-checkpoint")
    receipt = tmp_path / "rl-preflight.json"
    receipt.write_text(
        json.dumps(
            {
                "kind": "localagent_one_update_training_preflight",
                "schema_version": 1,
                "status": "failed",
                "metrics": {
                    "lineage": {
                        "parent_checkpoint_sha256": "0" * 64,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    report = build_workshop_gate(
        CATALOG,
        repo_root=ROOT,
        rl_preflight_receipt=receipt,
        current_checkpoint=checkpoint,
    )
    check = next(item for item in report["checks"] if item["requirement"] == "training:rl_preflight")
    assert check["status"] == "blocked"
    assert check["blockers"] == ["current_checkpoint_sha256_mismatch", "preflight_status_not_passed"]

    payload = json.loads(receipt.read_text(encoding="utf-8"))
    import hashlib

    payload["status"] = "passed"
    payload["metrics"]["lineage"]["parent_checkpoint_sha256"] = hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    report = build_workshop_gate(
        CATALOG,
        repo_root=ROOT,
        rl_preflight_receipt=receipt,
        current_checkpoint=checkpoint,
    )
    check = next(item for item in report["checks"] if item["requirement"] == "training:rl_preflight")
    assert check["status"] == "pass"


def test_native_receipt_contract_requires_execution_and_split(tmp_path: Path) -> None:
    receipt = tmp_path / "androidworld.json"
    receipt.write_text(
        json.dumps(
            {
                "benchmark_id": "androidworld",
                "environment_executed": True,
                "official_split_verified": True,
                "task_count": 1,
                "success_rate": 1.0,
            }
        ),
        encoding="utf-8",
    )
    report = build_workshop_gate(
        CATALOG,
        repo_root=ROOT,
        native_receipts={"androidworld": receipt},
    )
    check = next(item for item in report["checks"] if item["requirement"] == "native:androidworld")
    assert check["status"] == "pass"


def test_native_receipt_must_bind_current_checkpoint_when_requested(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"current-checkpoint")
    receipt = tmp_path / "androidworld.json"
    receipt.write_text(
        json.dumps(
            {
                "benchmark_id": "androidworld",
                "environment_executed": True,
                "official_split_verified": True,
                "task_count": 1,
                "success_rate": 1.0,
            }
        ),
        encoding="utf-8",
    )
    report = build_workshop_gate(
        CATALOG,
        repo_root=ROOT,
        native_receipts={"androidworld": receipt},
        current_checkpoint=checkpoint,
    )
    check = next(item for item in report["checks"] if item["requirement"] == "native:androidworld")
    assert check["status"] == "blocked"
    assert check["blockers"] == ["current_checkpoint_not_bound"]

    import hashlib

    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    report = build_workshop_gate(
        CATALOG,
        repo_root=ROOT,
        native_receipts={"androidworld": receipt},
        current_checkpoint=checkpoint,
    )
    check = next(item for item in report["checks"] if item["requirement"] == "native:androidworld")
    assert check["status"] == "pass"


def test_native_browsergym_probe_is_recorded_but_not_official_score() -> None:
    receipt = ROOT / "docs/paper/results/raw/m29-browsergym-native-model-eval-v1.json"
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["environment_executed"] is True
    assert payload["task_count"] == 240
    assert payload["success_rate"] == 0.0
    report = build_workshop_gate(
        CATALOG,
        repo_root=ROOT,
        native_receipts={"browsergym_miniwob": receipt},
    )
    check = next(item for item in report["checks"] if item["requirement"] == "native:browsergym_miniwob")
    assert check["status"] == "blocked"
    assert check["blockers"] == ["official_split_not_verified"]


def test_native_browsergym_full_pinned_split_passes_receipt_contract() -> None:
    receipt = ROOT / "docs/paper/results/raw/m41-browsergym-native-full-model-eval-v1.json"
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["environment_executed"] is True
    assert payload["official_split_verified"] is True
    assert payload["task_count"] == 240
    assert payload["success_rate"] == 0.0
    assert len(payload["cases"]) == 240
    assert all(case["abstentions"] == 10 for case in payload["cases"])
    report = build_workshop_gate(
        CATALOG,
        repo_root=ROOT,
        native_receipts={"browsergym_miniwob": receipt},
    )
    check = next(item for item in report["checks"] if item["requirement"] == "native:browsergym_miniwob")
    assert check["status"] == "pass"


def test_native_browsergym_adapter_ablation_passes_receipt_contract() -> None:
    receipt = ROOT / "docs/paper/results/raw/m43-browsergym-native-adapter-full-model-eval-v1.json"
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["official_split_verified"] is True
    assert payload["task_count"] == 240
    assert payload["success_rate"] == 5 / 240
    successes = [case for case in payload["cases"] if case["success"]]
    assert len(successes) == 5
    report = build_workshop_gate(
        CATALOG,
        repo_root=ROOT,
        native_receipts={"browsergym_miniwob": receipt},
    )
    check = next(item for item in report["checks"] if item["requirement"] == "native:browsergym_miniwob")
    assert check["status"] == "pass"


def test_native_mobilegym_full_official_text_eval_passes_receipt_contract() -> None:
    receipt = ROOT / "docs/paper/results/raw/m133-mobilegym-native-text-eval-v1.json"
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["environment_executed"] is True
    assert payload["official_split_verified"] is True
    assert payload["native_receipt_eligible"] is True
    assert payload["task_count"] == 256
    assert payload["success_rate"] == 13 / 256
    report = build_workshop_gate(
        CATALOG,
        repo_root=ROOT,
        native_receipts={"mobilegym": receipt},
    )
    check = next(item for item in report["checks"] if item["requirement"] == "native:mobilegym")
    assert check["status"] == "pass"


def test_native_webgpu_capability_receipt_passes_hardware_contract() -> None:
    receipt = ROOT / "docs/paper/results/raw/m39-webgpu-native-capability-v1.json"
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["hardware_adapter"] == "vendor=apple; architecture=metal-3"
    assert payload["capability"]["closed_loop_success"] == 0
    assert payload["capability"]["cases"][1]["last_result"]["args"]["url"] == "https://example.com"
    report = build_workshop_gate(
        CATALOG,
        repo_root=ROOT,
        webgpu_receipt=receipt,
    )
    check = next(item for item in report["checks"] if item["requirement"] == "webgpu:native_capability_and_latency")
    assert check["status"] == "pass"


def test_native_webgpu_capability_receipt_covers_notion_dispatch() -> None:
    receipt = ROOT / "docs/paper/results/raw/m40-webgpu-native-capability-notion-v1.json"
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["capability"]["evaluated_cases"] == 3
    notion = next(
        case
        for case in payload["capability"]["cases"]
        if case["expected_tool"] == "notion_write"
    )
    assert notion["exact_action_rate"] == 1.0
    assert notion["last_result"]["args"]["content"] == "WebGPU state loop passed"
    report = build_workshop_gate(CATALOG, repo_root=ROOT, webgpu_receipt=receipt)
    check = next(
        item
        for item in report["checks"]
        if item["requirement"] == "webgpu:native_capability_and_latency"
    )
    assert check["status"] == "pass"


def test_weight_gate_requires_two_compatible_labeled_reports(tmp_path: Path) -> None:
    payload = {
        "compatibility": {
            "config_mismatches": {},
            "shape_mismatches": {},
            "tokenizer_sha256_equal": True,
        },
        "ablation": "parent_heads",
        "held_out": {"selector": 0.5},
    }
    first = tmp_path / "parent.json"
    second = tmp_path / "random.json"
    first.write_text(json.dumps(payload), encoding="utf-8")
    payload["ablation"] = "random"
    second.write_text(json.dumps(payload), encoding="utf-8")
    report = build_workshop_gate(
        CATALOG,
        repo_root=ROOT,
        weight_reports=[first, second],
    )
    check = next(item for item in report["checks"] if item["requirement"] == "weights:transfer_and_no_transfer_ablation")
    assert check["status"] == "pass"


def test_weight_gate_accepts_combined_parent_vs_random_receipt() -> None:
    report = build_workshop_gate(
        CATALOG,
        repo_root=ROOT,
        weight_reports=["docs/paper/results/raw/m25-weight-transfer-ablation-v1.json"],
    )
    check = next(item for item in report["checks"] if item["requirement"] == "weights:transfer_and_no_transfer_ablation")
    assert check["status"] == "pass"


def test_weight_gate_accepts_current_unified_cross_surface_receipt() -> None:
    report = build_workshop_gate(
        CATALOG,
        repo_root=ROOT,
        weight_reports=["docs/paper/results/raw/m377-all-public-candidate-transfer-64step-v1.json"],
        current_checkpoint="runs/sft-webgpu-browser-context-adapter-20260802/latest.pt",
    )
    check = next(
        item
        for item in report["checks"]
        if item["requirement"] == "weights:transfer_and_no_transfer_ablation"
    )
    assert check["status"] == "pass"


def test_weight_gate_rejects_unified_cross_surface_checkpoint_mismatch(tmp_path: Path) -> None:
    source = ROOT / "docs/paper/results/raw/m377-all-public-candidate-transfer-64step-v1.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["parent_checkpoint"]["sha256"] = "0" * 64
    receipt = tmp_path / "m377-mismatch.json"
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"different-current-checkpoint")
    report = build_workshop_gate(
        CATALOG,
        repo_root=ROOT,
        weight_reports=[receipt],
        current_checkpoint=checkpoint,
    )
    check = next(
        item
        for item in report["checks"]
        if item["requirement"] == "weights:transfer_and_no_transfer_ablation"
    )
    assert check["status"] == "blocked"
    assert any("unified_current_checkpoint_sha256_mismatch" in blocker for blocker in check["blockers"])


def test_write_gate_refuses_overwrite(tmp_path: Path) -> None:
    report = build_workshop_gate(CATALOG, repo_root=ROOT)
    output = tmp_path / "gate.json"
    write_workshop_gate(report, output)
    assert json.loads(output.read_text(encoding="utf-8"))["ready"] is False
    try:
        write_workshop_gate(report, output)
    except FileExistsError:
        pass
    else:
        raise AssertionError("write_workshop_gate must refuse overwrite")


def test_public_manifest_must_bind_the_current_checkpoint_when_requested(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"current-checkpoint")
    manifest = tmp_path / "public.json"
    manifest.write_text(
        json.dumps(
            {
                "public": True,
                "model_url": "https://huggingface.co/example/model",
                "demo_url": "https://huggingface.co/spaces/example/demo",
                "artifact_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    report = build_workshop_gate(
        CATALOG,
        repo_root=ROOT,
        public_artifact_manifest=manifest,
        current_checkpoint=checkpoint,
    )
    check = next(
        item for item in report["checks"] if item["requirement"] == "artifacts:public_model_demo_manifest"
    )
    assert check["status"] == "blocked"
    assert check["blockers"] == ["current_checkpoint_not_bound"]

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    import hashlib

    payload["current_checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    report = build_workshop_gate(
        CATALOG,
        repo_root=ROOT,
        public_artifact_manifest=manifest,
        current_checkpoint=checkpoint,
    )
    check = next(
        item for item in report["checks"] if item["requirement"] == "artifacts:public_model_demo_manifest"
    )
    assert check["status"] == "pass"
