import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m728-public-realistic-eval-matrix-audit-v1.json"


def test_m728_matrix_audit_is_current_and_split_explicit() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["kind"] == "localagent_public_realistic_eval_matrix_audit"
    assert payload["matrix"]["entries"] == 28
    assert payload["counts"]["families"] == {
        "browser": 4,
        "computer": 4,
        "mobile": 9,
        "terminal": 1,
        "tool_api": 10,
    }
    assert payload["counts"]["train_policy"]["train"] == 6
    assert len(payload["train_eligible"]) == 6
    assert all(row["source_url"].startswith("https://") for row in payload["sources"])
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    assert payload["receipt_self_sha256"] == hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
