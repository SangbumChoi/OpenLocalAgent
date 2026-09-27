"""Checks that the refreshed catalog remains fail-closed at the workshop gate."""

import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m478-workshop-gate-current-catalog-refresh-v1.json")
CATALOG_SHA256 = "04bdd515e080c4e7fedab935ae9a31a01a77ca73268645719e5c9129d2eb3c61"


def test_m478_gate_binds_catalog_and_preserves_native_blockers() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["ready"] is False
    assert payload["catalog"]["entries"] == 42
    assert payload["catalog"]["sha256"] == CATALOG_SHA256
    blocked = {item["requirement"]: item["blockers"] for item in payload["blocking_requirements"]}
    assert blocked["native:mobile_safety_bench"] == ["receipt_not_supplied"]
    assert blocked["native:iosworld"] == ["receipt_not_supplied"]
    assert blocked["native:osworld"] == ["receipt_not_supplied"]
    assert blocked["native:osworld_v2"] == ["receipt_not_supplied"]
    assert blocked["native:agentnet"] == ["receipt_not_supplied"]
    assert blocked["artifacts:public_model_demo_manifest"] == ["manifest_not_supplied"]
