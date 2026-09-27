import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert claimed == actual
    return payload


def _load_audit(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("audit_sha256")
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert claimed == actual
    return payload


def test_m671_current_mcp_transfer_is_source_and_weight_bound() -> None:
    receipt = _load(ROOT / "docs/paper/results/raw/m671-m666-mcp-current-transfer-v1.json")
    assert receipt["source"]["dataset"] == "obaydata/mcp-agent-trajectory-benchmark"
    assert receipt["source"]["revision"] == "f4f449d65271abc1e4ccd5157d121a59a1dd38c4"
    assert receipt["protocol"]["official_test_split"] is False
    assert receipt["protocol"]["train_rows"] == 86
    assert receipt["protocol"]["eval_rows"] == 21
    assert receipt["weight_adoption"]["config_compatible"] is True
    assert receipt["weight_adoption"]["tokenizer_compatible"] is True
    assert receipt["weight_adoption"]["action_heads_frozen"] is True
    comparison = receipt["comparison"]
    assert comparison["warm_start_better_after"] is True
    assert comparison["warm_minus_random_after_pp"] > 15.0
    assert comparison["exact_sequence_accuracy"] == {"warm": 0.0, "random": 0.0}


def test_m671_public_snapshot_audit_keeps_original_links() -> None:
    audit = _load_audit(ROOT / "docs/paper/results/raw/m671-public-dataset-snapshot-audit-v1.json")
    assert audit["kind"] == "localagent_public_dataset_snapshot_audit"
    assert len(audit["datasets"]) == 9
    assert any(row["id"] == "mcp_trajectory_benchmark" for row in audit["datasets"])
    assert all(row["original_url"].startswith("https://") for row in audit["datasets"])
    assert all(row["revision"] for row in audit["datasets"])
