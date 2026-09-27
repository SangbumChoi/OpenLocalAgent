import hashlib
import json
from pathlib import Path


def test_agentdiff_projection_receipt_is_self_consistent_and_eval_only() -> None:
    path = Path("docs/paper/results/raw/m618-agentdiff-transfer-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["dataset"]["name"] == "hubertmarek/agent-diff-bench"
    assert payload["dataset"]["train_policy"] == "train_split_only;test_split_eval_only"
    assert payload["dataset"]["test_rows"] == 45
    assert payload["dataset"]["services"] == ["box", "calendar", "linear", "slack"]
    assert "not deterministic state-diff success" in payload["claim_boundary"]
