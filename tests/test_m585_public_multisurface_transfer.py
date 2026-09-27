from __future__ import annotations

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m585-public-multisurface-transfer-webgpu-v1.json")


def _load() -> dict:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert recorded == expected
    return payload


def test_m585_public_transfer_is_source_bound_and_warm_dominates_random() -> None:
    payload = _load()
    assert payload["warm_child"]["parameters"] < 100_000_000
    assert payload["public_sources"]["train_rows"] == 202
    assert payload["public_sources"]["eval_rows"] == 53
    metrics = payload["transfer_metrics"]
    assert metrics["warm_eval_token_accuracy"] > metrics["random_eval_token_accuracy"]
    assert metrics["warm_minus_random_eval_pp"] > 20.0
    assert metrics["warm_sequence_exact"] == metrics["random_sequence_exact"] == 0.0
    assert payload["weight_transfer"]["compatibility"]["tokenizer_sha256_equal"] is True


def test_m585_webgpu_and_local_stateful_bridge_are_checkpoint_bound() -> None:
    payload = _load()
    checkpoint = payload["warm_child"]["sha256"]
    assert payload["webgpu"]["backend"] == "webgpu"
    assert payload["webgpu"]["exact_cases"] == payload["webgpu"]["exact_actions"] == 3
    assert payload["webgpu"]["tokens_per_second_p50"] > 100.0
    assert payload["local_stateful_bridge"]["pass_at_1"] == 1.0
    assert payload["local_stateful_bridge"]["steps"] == 13
    assert "external_side_effects" in payload["local_stateful_bridge"]
    assert checkpoint == "6553dc2b161c03a916379fb77f174866143da6ef87173be07a12b57c4417b1ff"
