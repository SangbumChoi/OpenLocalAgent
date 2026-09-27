"""Integrity checks for the live public Space black-box receipt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m224-public-space-black-box-realistic-prompts-v1.json"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_m224_receipt_is_self_hashed_and_binds_public_space() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    deployment = payload["deployment"]
    assert deployment["space_url"] == "https://huggingface.co/spaces/danelcsb/localagent-webgpu"
    assert deployment["http_status"] == 200
    assert deployment["runtime_status"] == "Running"
    assert deployment["backend_label"] == "WEBGPU"
    assert deployment["source_bytes"] == 10986
    assert deployment["bundle_manifest_present"] is False


def test_m224_preserves_realistic_failure_boundary() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in payload["cases"]}
    assert cases["open_url"]["observed_tool"] == "open_url"
    assert cases["email_send"]["observed_tool"] == "set_reminder"
    assert cases["email_send"]["outcome"].startswith("wrong_tool")
    assert cases["search_then_notion"]["observed_tool"] == "notion_write"
    assert "closed loop" in cases["search_then_notion"]["outcome"]
    assert payload["decision"]["publication_ready"] is False
    assert payload["evaluator"]["side_effects"] is False
    assert payload["evaluator"]["official_benchmark"] is False
