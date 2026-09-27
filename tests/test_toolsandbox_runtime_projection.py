import json
from pathlib import Path


def test_runtime_projection_script_declares_state_history_boundary() -> None:
    source = Path("scripts/ingest_toolsandbox_runtime_projection.py").read_text(encoding="utf-8")
    assert "get_available_tools" in source
    assert "runtime_state_history_available" in source
    assert "official split were not executed" in source


def test_runtime_projection_contract_is_not_official_score() -> None:
    path = Path("/private/tmp/m211-toolsandbox-runtime-projection/toolsandbox-train.manifest.json")
    if not path.exists():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["environment_executed"] is False
    assert manifest["verifier_executed"] is False
    assert manifest["state_history_available"] is False
