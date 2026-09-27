"""Integrity checks for the authoritative realistic-source refresh."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path(
    "docs/paper/results/raw/m404-authoritative-realistic-source-refresh-v1.json"
)


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_m404_is_self_hashed_and_binds_current_source_contracts() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    sources = {source["id"]: source for source in payload["sources"]}
    assert sources["androidworld"]["contract"]["tasks"] == 116
    assert sources["androidworld"]["contract"]["apps"] == 20
    assert sources["iosworld"]["contract"]["tasks"] == 133
    assert sources["iosworld"]["contract"]["apps"] == 26
    assert sources["mobileworld"]["contract"]["tasks"] == 201
    assert sources["osworld2"]["contract"]["workflows"] == 108
    assert sources["agentnet"]["revision"] == "d76ee50a63fad81cfdbe576416757d7c2091ed50"
    assert "DatasetGenerationCastError" in sources["agentnet"]["contract"]["viewer_status"]
    assert payload["policy"]["original_source_required"] is True
    assert payload["policy"]["native_runner_required_for_task_success"] is True
