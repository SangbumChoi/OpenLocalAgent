import json
from pathlib import Path


def test_m301_gate_joins_current_base_replay_but_stays_fail_closed() -> None:
    path = Path("docs/paper/results/raw/m301-workshop-gate-current-toolsandbox-base-v1.json")
    gate = json.loads(path.read_text(encoding="utf-8"))
    assert gate["ready"] is False
    toolsandbox = next(
        check for check in gate["checks"] if check["requirement"] == "native:toolsandbox"
    )
    assert toolsandbox["status"] == "blocked"
    assert toolsandbox["blockers"] == ["official_split_not_verified"]
    assert any(
        item["requirement"] == "native:browsergym_miniwob" and item["status"] == "pass"
        for item in gate["checks"]
    )
