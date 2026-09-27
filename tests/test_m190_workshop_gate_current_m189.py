import json
from pathlib import Path


def test_m190_gate_binds_current_m189_mobilegym_and_stays_fail_closed() -> None:
    path = Path("docs/paper/results/raw/m190-workshop-gate-current-m189-v1.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["kind"] == "localagent_workshop_publication_gate"
    assert report["ready"] is False
    mobilegym = next(
        item for item in report["checks"] if item["requirement"] == "native:mobilegym"
    )
    assert mobilegym["status"] == "pass"
    assert any(
        item["requirement"] == "native:androidworld"
        and item["status"] == "blocked"
        for item in report["checks"]
    )
    assert any(
        item["requirement"] == "native:mcpmark"
        and item["status"] == "blocked"
        for item in report["checks"]
    )
