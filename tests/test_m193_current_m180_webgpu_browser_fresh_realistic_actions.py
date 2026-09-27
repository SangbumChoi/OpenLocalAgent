import hashlib
import json
from pathlib import Path


RECEIPT = Path(
    "docs/paper/results/raw/m193-current-m180-webgpu-browser-fresh-realistic-actions-v1.json"
)


def test_m193_binds_the_m180_bundle_and_preserves_its_negative_quality_result() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    assert expected == actual
    assert payload["checkpoint"]["sha256"] == (
        "10827649f07182a08f8c11104d4713e76b8acaddc73020c5a4c77950de7b23a0"
    )
    assert payload["checkpoint"]["parameters"] < 100_000_000
    assert payload["bundle"]["parity_gate_passed"] is True
    assert payload["browser"]["model_ready"] is True
    assert payload["browser"]["backend_badge"] == "WEBGPU"
    assert payload["browser"]["external_side_effects"] is False
    assert payload["single_step"]["unambiguous_rows"] == 9
    assert payload["single_step"]["exact_tool_count"] == 1
    assert payload["single_step"]["exact_tool_rate"] == 1 / 9
    assert payload["planner"]["exact_trajectory_count"] == 0
    assert payload["planner"]["trajectory_count"] == 2
    assert "not an official benchmark split" in payload["source"]["claim"]
