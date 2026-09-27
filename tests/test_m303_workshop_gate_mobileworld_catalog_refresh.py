import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m303-workshop-gate-mobileworld-catalog-refresh-v1.json")
CATALOG = Path("configs/data/realistic-agent-eval.catalog.yaml")


def test_m303_gate_records_mobileworld_catalog_refresh_without_native_promotion() -> None:
    gate = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert gate["ready"] is False
    assert gate["catalog"]["entries"] == 40
    assert gate["catalog"]["path"] == str(CATALOG)
    assert "mobileworld" in gate["catalog"]["blocked_ids"]
    assert any(
        item["requirement"] == "native:toolsandbox"
        and item["blockers"] == ["official_split_not_verified"]
        for item in gate["blocking_requirements"]
    )
    assert not any(
        item["requirement"] == "native:mobileworld" for item in gate["blocking_requirements"]
    )
