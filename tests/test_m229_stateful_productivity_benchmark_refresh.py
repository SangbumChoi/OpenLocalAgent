"""Integrity checks for the stateful productivity benchmark refresh."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m229-stateful-productivity-benchmark-refresh-v1.json"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_m229_productivity_contracts_are_hash_pinned_and_eval_only() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    sources = {source["id"]: source for source in payload["sources"]}
    assert set(sources) == {"mcpmark", "enterpriseopsgym", "appworld_ul", "tau3_bench", "tua_bench"}
    assert all(source["training_rows_admitted"] == 0 for source in sources.values())
    assert all(source["runtime_executed"] is False for source in sources.values())

    mcpmark = sources["mcpmark"]
    assert mcpmark["task_contract"]["task_files"] == ["meta.json", "description.md", "verify.py"]
    assert "Notion" in mcpmark["services"]
    assert "Playwright" in mcpmark["services"]

    enterprise = sources["enterpriseopsgym"]
    assert enterprise["task_contract"]["rows"] == 649
    assert enterprise["task_contract"]["email_rows"] == 67
    assert enterprise["task_contract"]["verification"] == "SQL/database-state verifiers"

    appworld_ul = sources["appworld_ul"]
    assert appworld_ul["task_contract"]["tasks"] == 516
    assert set(appworld_ul["task_contract"]["interaction_types"]) == {
        "clarification",
        "confirmation",
        "infeasible_request_handling",
    }
    assert appworld_ul["code_url"] == "https://github.com/StonyBrookNLP/appworld"
    assert appworld_ul["code_contract"]["linked_from_official_ul_page"] is True
    assert appworld_ul["code_contract"]["license"] == "Apache-2.0"
    assert "UL-specific task assets" in appworld_ul["code_contract"]["release_caveat"]

    tau3 = sources["tau3_bench"]
    assert tau3["task_contract"]["knowledge_base_documents"] == 698
    assert "full_duplex_voice" in tau3["task_contract"]["evaluation_modes"]

    tua = sources["tua_bench"]
    assert tua["task_contract"]["tasks"] == 120
    assert "email_management" in tua["task_contract"]["realistic_productivity_areas"]

    decision = payload["adoption_and_training_decision"]
    assert "androidcontrol" in decision["trainable_sources"]
    assert "mcpmark" in decision["evaluation_only_sources"]
    assert "matched random" in decision["weight_adoption_rule"]
    assert "confirmation-required" in decision["deployment_rule"]
