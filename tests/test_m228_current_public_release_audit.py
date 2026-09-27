"""Integrity checks for the current authoritative public-release audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m228-current-public-release-audit-v1.json"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_m228_audit_is_hash_pinned_and_never_training_data() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    sources = {source["id"]: source for source in payload["sources"]}
    assert set(sources) == {
        "bfcl_v4_agentic",
        "iosworld",
        "mobile_safety_bench",
        "osworld_v2",
        "mobile_bench",
    }
    assert all(source["training_rows_admitted"] == 0 for source in sources.values())
    assert all(source["runtime_executed"] is False for source in sources.values())

    bfcl = sources["bfcl_v4_agentic"]
    assert "agentic_web_search" in bfcl["evaluation_contract"]["categories"]
    assert "agentic_memory" in bfcl["evaluation_contract"]["categories"]

    ios = sources["iosworld"]
    assert ios["evaluation_contract"]["apps"] == 26
    assert ios["evaluation_contract"]["tasks"] == 133
    assert ios["evaluation_contract"]["tool_use_mode"] == "optional_MCP_server"

    safety = sources["mobile_safety_bench"]
    assert safety["published_contracts"]["project_page_total_tasks"] == 250
    assert safety["published_contracts"]["pinned_paper_repository_suite"] == 100
    assert "must not be combined" in safety["reconciliation"]

    osworld = sources["osworld_v2"]
    assert osworld["release"] == "osworld-v2-2026.06.24"
    assert "gated" in osworld["access_status"]

    assert "androidcontrol" in payload["localagent_adaptation"]["recommended_training_sources"]
    assert "mobile_safety_bench" in payload["localagent_adaptation"]["evaluation_only_sources"]
    assert "Train only" in payload["localagent_adaptation"]["reason"]
