"""Integrity checks for the current xLAM constrained-decoder canary."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m464-xlam-current-free-run-row-canary-v1.json")
PARENT_SHA256 = "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
SOURCE_SHA256 = "43db9250b50f44d96d2be31983690e101bd083eefea2a4a327e13a3ed8caeee1"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m464_binds_current_parent_source_and_canary_policy() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["kind"] == "localagent_xlam_current_free_run_row_canary"
    assert payload["source"]["source_file"]["sha256"] == SOURCE_SHA256
    assert payload["parent"]["checkpoint"]["sha256"] == PARENT_SHA256
    assert payload["protocol"]["candidate_mode"] == "row_retriever"
    assert payload["protocol"]["rows"] == 8


def test_m464_canary_does_not_promote_warm_child() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    comparison = payload["comparison"]
    assert comparison["parent_first_tool_exact_rate"] == 0.5
    assert comparison["warm_first_tool_exact_rate"] == 0.5
    assert comparison["parent_first_arguments_exact_rate"] == 0.0
    assert comparison["warm_first_arguments_exact_rate"] == 0.0
    assert comparison["parent_schema_valid_rate"] == 1.0
    assert comparison["warm_schema_valid_rate"] == 1.0
    assert payload["decision"]["webgpu_export_allowed"] is False
