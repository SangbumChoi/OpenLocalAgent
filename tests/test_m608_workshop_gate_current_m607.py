import hashlib
import json
from pathlib import Path


def test_m608_gate_is_current_bound_and_still_fail_closed() -> None:
    path = Path("docs/paper/results/raw/m608-workshop-gate-current-m607-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    gate = payload["gate"]
    assert gate["ready"] is False
    checks = {row["requirement"]: row for row in gate["checks"]}
    assert checks["weights:transfer_and_no_transfer_ablation"]["status"] == "pass"
    assert checks["webgpu:native_capability_and_latency"]["status"] == "pass"
    assert len(gate["blocking_requirements"]) == 10
    assert payload["transfer_receipt"]["parent_checkpoint"]["sha256"].startswith("6553dc2b")
