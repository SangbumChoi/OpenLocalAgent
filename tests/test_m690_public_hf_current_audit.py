import hashlib
import json
from pathlib import Path


def test_m690_public_audit_is_reachable_but_not_current_bound() -> None:
    path = Path(__file__).parents[1] / "docs/paper/results/raw/m690-public-hf-current-audit-v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("receipt_self_sha256")
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert expected == actual
    assert payload["public"] is True
    assert payload["verification"]["public_model_http_status"] == 200
    assert payload["verification"]["public_demo_http_status"] == 200
    assert payload["verification"]["current_checkpoint_match"] is False
    assert payload["current_checkpoint_sha256"] == "dbf45ea710e1b88cbc6631813b2abc7cfd6b454ee0052b0d0d4881c85d932533"
