import hashlib
import json
from pathlib import Path


def test_toolsandbox_protocol_audit_is_hash_bound_and_does_not_invent_a_split() -> None:
    path = Path("docs/paper/results/raw/m613-toolsandbox-protocol-audit-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    protocol = payload["official_protocol"]
    assert protocol["official_split"] is None
    assert protocol["official_split_verified"] is False
    assert protocol["scenario_count"] == 1032
    assert protocol["base_scenario_count"] == 129
    assert protocol["augmentation_factor"] == 8
    assert protocol["requires_user_simulator"] is True
    assert payload["interpretation"]["training_policy"] == "eval_only"
    assert payload["interpretation"]["native_smoke_is_official_score"] is False
