"""Integrity checks for the AppWorld API-schema adapter controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from localagent.eval.appworld_api_head import api_label_from_code


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "docs/paper/results/raw"
LINEAR = RAW / "m281-appworld-api-head-training-v1.json"
RETRIEVER = RAW / "m281-appworld-api-retriever-v1.json"
NATIVE = RAW / "m281-appworld-retriever-native-v1.json"
EXACTNESS = RAW / "m281-appworld-first-action-exactness-v1.json"


def _canonical_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _receipt(path: Path) -> dict:
    payload = _load(path)
    expected = payload.pop("receipt_self_sha256")
    assert _canonical_hash(payload) == expected
    return payload


def test_appworld_api_code_parser_is_literal_and_canonical() -> None:
    assert api_label_from_code("apis.spotify.show_song_library(page_index=0)") == (
        "spotify.show_song_library"
    )
    assert api_label_from_code("apis.phone.search_contacts(query='Kristin')") == (
        "phone.search_contacts"
    )


def test_m281_learned_head_and_lexical_control_are_reported_separately() -> None:
    linear = _load(LINEAR)
    retriever = _load(RETRIEVER)
    assert linear["kind"] == "localagent_appworld_api_head_training_report"
    assert linear["metrics"]["eval"] == {"accuracy": 0.5, "exact": 6, "rows": 12}
    assert retriever["kind"] == "localagent_appworld_api_retriever_report"
    assert retriever["metrics"] == {"accuracy": 1.0, "exact": 12, "eval_rows": 12}
    assert retriever["learned_weights"] is False
    assert retriever["source"]["train_rows"] == 24
    assert retriever["source"]["eval_rows"] == 12


def test_m281_native_retriever_replays_exact_first_actions_without_full_task_success() -> None:
    native = _receipt(NATIVE)
    assert native["configuration"]["appworld_api_head"].endswith(
        "m281-appworld-api-retriever-v2.pt"
    )
    assert native["configuration"]["selector_first"] is True
    assert native["environment"]["native_runtime_executed"] is True
    assert native["environment"]["environment_reset_per_task"] is True
    assert native["summary"] == {
        "action_replayed": 12,
        "native_action_api_calls": 12,
        "native_api_calls": 48,
        "native_bootstrap_api_calls": 36,
        "native_success_rate": 0.0,
        "native_successes": 0,
        "tasks": 12,
    }
    assert all(task["appworld_api_head_applied"] for task in native["tasks"])
    assert all(task["action_replayed"] for task in native["tasks"])
    assert all(not task["evaluation"]["success"] for task in native["tasks"])


def test_m281_first_action_hashes_are_exact_and_claim_bounded() -> None:
    exactness = _receipt(EXACTNESS)
    assert exactness["metrics"] == {
        "exact_code": 12,
        "exact_code_rate": 1.0,
        "predicted_code_available": 12,
        "rows": 12,
    }
    assert all(task["exact"] for task in exactness["tasks"])
    assert "first-action exactness only" in exactness["claim_boundary"]
