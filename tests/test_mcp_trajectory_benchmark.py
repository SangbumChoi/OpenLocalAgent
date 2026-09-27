import hashlib
import json
from pathlib import Path

from localagent.data.realistic_catalog import load_catalog


def test_mcp_trajectory_catalog_is_pinned() -> None:
    catalog, _ = load_catalog("configs/data/realistic-agent-eval.mcp-trajectory.yaml")
    row = catalog["entries"][0]
    assert row["id"] == "mcp_trajectory_benchmark"
    assert row["source_revision"] == "f4f449d65271abc1e4ccd5157d121a59a1dd38c4"
    assert row["scale"]["internal_train_agents"] == 30
    assert row["scale"]["internal_eval_agents"] == 8


def test_mcp_trajectory_transfer_receipt_is_self_consistent() -> None:
    path = Path("docs/paper/results/raw/m624-mcp-trajectory-transfer-v1.json")
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["protocol"]["official_test_split"] is False
    assert payload["protocol"]["train_rows"] == 86
    assert payload["protocol"]["eval_rows"] == 21
