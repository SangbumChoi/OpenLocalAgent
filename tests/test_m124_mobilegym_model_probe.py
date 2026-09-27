import hashlib
import json
from pathlib import Path


def _canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_m124_mobilegym_model_probe_is_native_but_non_gating_and_redacted() -> None:
    receipt = json.loads(
        Path("docs/paper/results/raw/m124-mobilegym-model-probe-v1.json").read_text(
            encoding="utf-8"
        )
    )
    self_hash = receipt.pop("receipt_self_sha256")
    assert self_hash == _canonical_sha256(receipt)
    assert receipt["native_benchmark_id"] == "mobilegym"
    assert receipt["native_receipt_eligible"] is False
    assert receipt["environment_executed"] is True
    assert receipt["official_split"] == "test"
    assert receipt["official_split_verified"] is True
    assert receipt["task_count"] == 1
    assert receipt["success_rate"] == 0.0
    assert receipt["model_invocations"] == 2
    assert receipt["model_calls"] == 2
    assert receipt["observation_mode"] == "text_projection"
    assert receipt["vision_used"] is False
    assert receipt["judge"]["passed"] is False
    assert receipt["judge"]["issue_count"] == 3
    assert "expected" not in receipt["judge"]
    assert "actual" not in receipt["judge"]
    for event in receipt["trace"]:
        assert set(event) == {
            "step",
            "tool",
            "argument_keys",
            "arguments_sha256",
            "model_output_sha256",
            "translated",
        }
