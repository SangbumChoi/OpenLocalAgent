import hashlib
import json
from pathlib import Path


def test_m610_route_calibration_is_measured_but_not_promoted() -> None:
    path = Path("docs/paper/results/raw/m610-m585-route-abstention-calibration-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["parent"]["sha256"].startswith("6553dc2b")
    assert payload["evaluation"]["warm"]["rows"] == 103
    assert sum(row["route_exact"] for row in payload["evaluation"]["warm_semantic"]) == 6
    assert payload["decision"]["adopt_warm_route_head"] is False
    assert payload["decision"]["export_to_webgpu"] is False
    assert payload["weight_movement"]["backbone_relative_l2"] == 0.0
