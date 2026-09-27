import json
from pathlib import Path


def test_m298_gate_keeps_tool_sandbox_non_official_receipt_blocked() -> None:
    path = Path("docs/paper/results/raw/m298-workshop-gate-current-toolsandbox-v1.json")
    gate = json.loads(path.read_text(encoding="utf-8"))
    assert gate["ready"] is False
    toolsandbox = next(
        check for check in gate["checks"] if check["requirement"] == "native:toolsandbox"
    )
    assert toolsandbox["status"] == "blocked"
    assert toolsandbox["blockers"] == ["official_split_not_verified"]
    assert "native:androidworld" in {item["requirement"] for item in gate["blocking_requirements"]}
