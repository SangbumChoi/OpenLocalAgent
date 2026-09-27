import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_m680_enterprise_email_control_is_current_and_fail_closed() -> None:
    path = ROOT / "docs/paper/results/raw/m680-m679-enterprise-email-control-v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert claimed == actual
    assert payload["source"]["dataset"] == "ServiceNow-AI/EnterpriseOps-Gym"
    assert payload["source"]["revision"] == "c8e538eae8a6205294f0a86675fefdc1fac408f6"
    assert payload["protocol"]["records"] == 67
    assert payload["protocol"]["official_native_score"] is False
    assert payload["weight_adoption"]["reuse_warm_backbone_for_email"] is False
    assert payload["comparison"]["warm_minus_random_hit_at_3_pp"] < -25.0
