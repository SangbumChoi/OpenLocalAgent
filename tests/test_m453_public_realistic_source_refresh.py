"""Integrity checks for the newly discovered public realistic-agent sources."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m453-public-realistic-source-refresh-v1.json")


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m453_is_self_hashed_and_keeps_new_sources_out_of_training() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["catalog_update"]["canonical_gate_changed"] is False
    assert payload["catalog_update"]["sft_inputs_changed"] is False
    assert {row["id"] for row in payload["sources"]} == {
        "markov_computer_use",
        "agentworldbench",
        "scalecua_data",
        "gui_world",
    }
    markov = next(row for row in payload["sources"] if row["id"] == "markov_computer_use")
    assert markov["reported_scale"]["trajectories"] == 160
    assert "thunderbird_email" in markov["realistic_surfaces"]
    world = next(row for row in payload["sources"] if row["id"] == "agentworldbench")
    assert world["reported_scale"]["mcp"] == 286
    scale = next(row for row in payload["sources"] if row["id"] == "scalecua_data")
    assert scale["reported_scale"]["platforms"] == 6
    gui = next(row for row in payload["sources"] if row["id"] == "gui_world")
    assert gui["reported_scale"]["videos"] == 12379
    assert "No task payload" in payload["claim_boundary"]
