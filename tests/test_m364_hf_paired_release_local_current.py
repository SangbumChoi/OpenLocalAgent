from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m364-hf-paired-release-local-current-v1.json"
CHECKPOINT_SHA256 = "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
TOOL_POOL_SHA256 = "fd3c5d30a3c7fcb2c010c04e79e1b47cd88e3897712e98f5a144d09699f25694"


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def test_m364_local_release_is_self_hashed_and_checkpoint_bound() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    expected = receipt.pop("receipt_self_sha256")
    assert hashlib.sha256(_canonical(receipt)).hexdigest() == expected
    assert receipt["checkpoint"]["sha256"] == CHECKPOINT_SHA256
    assert receipt["checkpoint"]["parameters"] == 10_524_544
    assert receipt["model"]["tool_pool_count"] == 63
    assert receipt["model"]["tool_pool_sha256"] == TOOL_POOL_SHA256
    assert receipt["webgpu"]["tool_pool_sha256"] == TOOL_POOL_SHA256
    assert receipt["webgpu"]["parity_gate"]["passed"] is True
    assert receipt["webgpu"]["parity_gate"]["hard_gate"] is True
    assert receipt["space"]["verified"] is True
    assert receipt["publication"] == {
        "authenticated": False,
        "published": False,
        "reason": "hf auth whoami reports Not logged in",
        "uploaded": False,
    }
