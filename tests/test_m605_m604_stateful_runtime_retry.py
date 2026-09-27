import hashlib
import json
from pathlib import Path


def test_m605_m604_runtime_retry_is_checkpoint_bound_and_local_only() -> None:
    path = Path("docs/paper/results/raw/m605-m604-stateful-runtime-retry-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["checkpoint"]["sha256"].startswith("ff7f50ad")
    assert payload["runtime"]["public_benchmark"] is False
    assert payload["runtime"]["external_accounts_used"] is False
    assert payload["oracle"]["task_complete_rate"] == 1.0
    assert payload["model"]["task_complete_rate"] == 1.0
    assert payload["model"]["accepted_steps"] == payload["model"]["expected_steps"] == 16
    assert payload["model"]["attempts"] == 28
    assert payload["model"]["attempt_success_rate"] == 0.5714285714285714
    assert "Neither pass is an AndroidWorld" in payload["claim_boundary"]
