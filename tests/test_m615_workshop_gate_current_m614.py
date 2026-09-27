import hashlib
import json
from pathlib import Path


def test_m615_gate_is_self_consistent_and_still_fail_closed() -> None:
    path = Path("docs/paper/results/raw/m615-workshop-gate-current-m614-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    gate = payload["gate"]
    assert gate["ready"] is False
    assert gate["current_checkpoint"]["sha256"] == "6553dc2b161c03a916379fb77f174866143da6ef87173be07a12b57c4417b1ff"
    blocked = {row["requirement"]: row for row in gate["blocking_requirements"]}
    assert blocked["native:toolsandbox"]["blockers"] == ["official_split_not_verified"]
    assert blocked["artifacts:public_model_demo_manifest"]["blockers"]
    assert len(gate["blocking_requirements"]) == 10
