import hashlib
import json
from pathlib import Path

from localagent.data.realistic_catalog import load_catalog


def test_mcp_persona_catalog_is_pinned_eval_only() -> None:
    catalog, _ = load_catalog("configs/data/realistic-agent-eval.mcp-persona.yaml")
    row = catalog["entries"][0]
    assert row["id"] == "mcp_persona"
    assert row["source_revision"] == "b510f5a5371c4524a58aeeb679c1ace845603e95"
    assert row["train_policy"] == "eval_only"
    assert row["scale"]["tasks"] == 173
    assert row["scale"]["unique_tools"] == 139


def test_mcp_persona_source_audit_is_self_consistent() -> None:
    path = Path("docs/paper/results/raw/m622-mcp-persona-source-audit-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["releases"]["english"]["tasks"] == 173
    assert payload["releases"]["english"]["unique_tools"] == 139
    assert payload["cross_release"]["english_chinese_ids_match"] is True
    assert payload["evaluation_boundary"]["official_train_test_split"] is False
    assert payload["source"]["license_evidence"]["license_file_present"] is False
