"""Current unified gate admits nested official native receipts without weakening checks."""

import json
from pathlib import Path

from localagent.eval.workshop_gate import build_workshop_gate


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "configs/data/realistic-agent-eval.catalog.yaml"
CHECKPOINT = ROOT / "runs/sft-mind2web-public-continuation-20260805/latest.pt"
GATE = ROOT / "docs/paper/results/raw/m460-workshop-gate-current-native-nested-v1.json"


def test_current_nested_mobile_and_browser_receipts_are_admitted() -> None:
    report = build_workshop_gate(
        CATALOG,
        repo_root=ROOT,
        native_receipts={
            "mobilegym": ROOT / "docs/paper/results/raw/m428-mobilegym-native-child-full-v1.json",
            "browsergym_miniwob": ROOT
            / "docs/paper/results/raw/m431-browsergym-native-child-full-v1.json",
        },
        current_checkpoint=CHECKPOINT,
    )
    checks = {item["requirement"]: item for item in report["checks"]}
    assert checks["native:mobilegym"]["status"] == "pass"
    assert checks["native:browsergym_miniwob"]["status"] == "pass"
    assert checks["native:mobilegym"]["blockers"] == []
    assert checks["native:browsergym_miniwob"]["blockers"] == []


def test_nested_receipts_still_fail_when_checkpoint_binding_is_wrong(tmp_path: Path) -> None:
    receipt = tmp_path / "nested.json"
    receipt.write_text(
        json.dumps(
            {
                "benchmark_id": "androidworld",
                "checkpoint": {"sha256": "0" * 64},
                "environment": {
                    "environment_executed": True,
                    "official_split_verified": True,
                    "task_count": 1,
                },
                "result": {"success_rate": 1.0},
            }
        ),
        encoding="utf-8",
    )
    report = build_workshop_gate(
        CATALOG,
        repo_root=ROOT,
        native_receipts={"androidworld": receipt},
        current_checkpoint=CHECKPOINT,
    )
    check = next(item for item in report["checks"] if item["requirement"] == "native:androidworld")
    assert check["status"] == "blocked"
    assert check["blockers"] == ["current_checkpoint_sha256_mismatch"]


def test_m460_current_gate_admits_two_official_native_splits() -> None:
    payload = json.loads(GATE.read_text(encoding="utf-8"))
    checks = {item["requirement"]: item for item in payload["checks"]}
    assert payload["ready"] is False
    assert checks["native:mobilegym"]["status"] == "pass"
    assert checks["native:browsergym_miniwob"]["status"] == "pass"
    assert checks["native:toolsandbox"]["blockers"] == ["official_split_not_verified"]
    assert checks["native:mcpmark"]["blockers"] == ["official_split_not_verified"]
    assert checks["training:rl_preflight"]["status"] == "pass"
