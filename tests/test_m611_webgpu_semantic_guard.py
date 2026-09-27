import hashlib
import json
from pathlib import Path


def test_m611_semantic_guard_receipt_is_self_consistent_and_fail_closed() -> None:
    path = Path("docs/paper/results/raw/m611-webgpu-semantic-guard-probe-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["deployment"]["verified"] is True
    assert payload["observations"]["backend"] == "webgpu"
    assert payload["observations"]["semantic"]["abstained"] is True
    assert payload["observations"]["email"]["confirmation_required"] is True
    assert payload["observations"]["plan"]["notion_confirmation_required"] is True
    assert payload["decision"]["publish_ready"] is False


def test_m611_adapter_contains_only_the_narrow_semantic_guard_marker() -> None:
    app = Path("spaces/localagent-webgpu/app.js").read_text(encoding="utf-8")
    assert "function semanticTextLexicalSelect" in app
    assert 'selection_policy: "semantic_text_safety_guard"' in app
    assert "semanticTextLexicalSelect(query)" in app
