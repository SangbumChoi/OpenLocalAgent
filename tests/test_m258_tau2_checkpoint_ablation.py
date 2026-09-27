from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m258-tau2-checkpoint-ablation-v1.json"


def _canonical(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def test_tau2_checkpoint_ablation_is_hash_bound_and_bounded() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    expected = receipt.pop("receipt_self_sha256")
    assert hashlib.sha256(_canonical(receipt)).hexdigest() == expected
    assert receipt["source"] == {
        "dataset": "tau2-bench",
        "domain": "mock",
        "split": "base",
        "revision": "363133ada1936491fb5bcec33cd62c3518a99f65",
        "task_count": 10,
        "task_text_retained": False,
        "tool_outputs_retained": False,
    }
    arms = {item["label"]: item for item in receipt["arms"]}
    assert arms["checkpoint_m46"]["summary"]["bounded_native_successes"] == 0
    assert arms["checkpoint_browser"]["summary"]["bounded_native_successes"] == 2
    assert arms["retriever_m46_k1"]["summary"]["bounded_native_successes"] == 3
    assert arms["retriever_m46_k1"]["configuration"]["selector_mode"] == "retriever"
    compatibility = receipt["weight_transfer"]["compatibility"]
    assert compatibility["config_mismatches"] == {}
    assert compatibility["shape_mismatches"] == {}
    assert compatibility["tokenizer_sha256_equal"] is True
    assert compatibility["shared_tensor_count"] == 51
    assert "complete tau2 base split" in receipt["claim_boundary"]
