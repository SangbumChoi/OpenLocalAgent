"""Integrity checks for the m590 matched desktop action-prior control."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m590-computer-agent-arena-m585-warm-parent-control-v1.json"


def test_m590_receipt_binds_public_eval_and_matched_checkpoints() -> None:
    data = json.loads(RECEIPT.read_text())
    assert data["dataset"] == "xlangai/computer-agent-arena"
    assert data["source_revision"] == "897b9f45287c516a44f9e79879b14bc3c1bc5b0a"
    assert data["source"]["bytes"] == 50609777
    assert data["source"]["selection"]["rows"] == 256
    assert data["prompt_contract"]["uses_screenshot"] is False
    assert data["parent"]["overall"] == data["warm_child"]["overall"]
    assert data["transfer_delta"] == {
        "tool_exact_pp": 0.0,
        "family_exact_pp": 0.0,
        "route_accuracy_pp": 0.0,
        "abstention_pp": 0.0,
        "by_family_tool_exact_pp": {
            "keyboard": 0.0,
            "observation": 0.0,
            "pointer": 0.0,
            "scroll": 0.0,
            "type": 0.0,
            "wait": 0.0,
        },
        "interpretation": data["transfer_delta"]["interpretation"],
    }
    assert "not a Computer Agent Arena" in data["claim_boundary"]


def test_m590_receipt_self_hash() -> None:
    data = json.loads(RECEIPT.read_text())
    declared = data.pop("receipt_self_sha256")
    assert declared
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == declared
