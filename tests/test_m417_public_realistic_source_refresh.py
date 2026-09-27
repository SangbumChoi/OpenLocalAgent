"""Integrity checks for the current public realistic-source refresh."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m417-public-realistic-source-refresh-v1.json")


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m417_receipt_is_self_hashed_and_keeps_native_boundary() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    sources = {source["id"]: source for source in payload["sources"]}
    assert sources["androidworld"]["release_contract"] == {
        "tasks": 116,
        "apps": 20,
        "dynamic_variations": True,
        "runtime": "Android emulator with task setup and reward verifier",
    }
    assert sources["mobileworld"]["release_contract"]["tasks"] == 201
    assert sources["agentnet"]["release_contract"]["human_annotated_tasks"] == 22600
    assert sources["cua_lite_agentnet"]["release_contract"]["validation_trajectories"] == 92
    assert sources["mcpmark"]["release_contract"]["total_tasks"] == 177
    assert payload["host_preflight"]["adb"] is False
    assert payload["host_preflight"]["docker"] is False
    assert payload["host_preflight"]["checkpoint_training_rows_admitted"] == 0
    assert payload["training_admission"]["new_source_rows_admitted"] == 0
    assert payload["policy"]["native_runner_required_for_task_success"] is True


def test_m417_source_families_cover_mobile_browser_computer_and_mcp() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    families = {source["family"] for source in payload["sources"]}
    assert families == {"mobile", "browser", "computer", "tool_api"}
    browser = next(source for source in payload["sources"] if source["id"] == "browsergym")
    assert "WebArenaVerified" in browser["release_contract"]["benchmark_families"]
    assert "VisualWebArena" in browser["release_contract"]["benchmark_families"]
