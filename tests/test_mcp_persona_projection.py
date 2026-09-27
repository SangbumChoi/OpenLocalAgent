import hashlib
import json
from pathlib import Path

from scripts.normalize_mcp_persona import normalize


def test_mcp_persona_normalization_preserves_eval_only_tool_chain() -> None:
    source = Path("/private/tmp/m622-mcp-persona.QyoOVP/data/tasks/en_release_data.json")
    if not source.exists():
        return
    rows, selection = normalize(source)
    assert len(rows) == 173
    assert selection["target"] == "canonical_compact_json_tool_chain"
    assert rows[0].meta["train_policy"] == "eval_only"
    assert rows[0].messages[1].content.startswith('{"tool_chain":')


def test_mcp_persona_projection_receipt_is_self_consistent() -> None:
    path = Path("docs/paper/results/raw/m623-mcp-persona-tool-chain-projection-v1.json")
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["dataset"]["split"] == "test"
    assert payload["dataset"]["train_policy"] == "eval_only"
