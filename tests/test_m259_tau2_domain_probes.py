from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = {
    "airline": ROOT / "docs/paper/results/raw/m259-tau2-airline-m46-retriever-k1-v1.json",
    "retail": ROOT / "docs/paper/results/raw/m259-tau2-retail-m46-retriever-k1-v1.json",
    "telecom": ROOT / "docs/paper/results/raw/m259-tau2-telecom-m46-retriever-k1-v1.json",
}


def _canonical(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def test_m259_domain_receipts_are_native_bounded_and_hash_only() -> None:
    expected_counts = {"airline": 50, "retail": 114, "telecom": 114}
    for domain, path in RECEIPTS.items():
        receipt = json.loads(path.read_text(encoding="utf-8"))
        expected = receipt.pop("receipt_self_sha256")
        assert hashlib.sha256(_canonical(receipt)).hexdigest() == expected
        assert receipt["source"]["domain"] == domain
        assert receipt["source"]["task_count"] == expected_counts[domain]
        assert receipt["configuration"]["tasks"]
        assert receipt["configuration"]["selector_mode"] == "retriever"
        assert receipt["configuration"]["retrieve_k"] == 1
        assert receipt["environment"]["native_runtime_executed"] is True
        assert receipt["environment"]["reset_per_task"] is True
        assert receipt["summary"]["tasks"] == 8
        assert receipt["summary"]["bounded_native_successes"] == 0
        for task in receipt["tasks"]:
            assert set(task["instruction"]) == {"bytes", "sha256"}
            assert set(task["model_output"]) == {"bytes", "sha256"}

    assert json.loads(RECEIPTS["airline"].read_text())["contract_verification"]["passed"] is True
    assert json.loads(RECEIPTS["retail"].read_text())["contract_verification"]["passed"] is True
    assert json.loads(RECEIPTS["telecom"].read_text())["contract_verification"]["passed"] is True
