import hashlib
import json
from pathlib import Path


RECEIPT = Path(
    "docs/paper/results/raw/m191-current-m180-webgpu-browser-fresh-realistic-actions-v1.json"
)


def test_m191_fresh_browser_smoke_is_webgpu_but_not_a_public_benchmark_score() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    assert expected == actual
    assert payload["source"]["public_benchmark_rows_used"] == 0
    assert payload["browser"]["model_ready"] is True
    assert payload["browser"]["backend_badge"] == "WEBGPU"
    assert payload["browser"]["external_side_effects"] is False
    assert payload["bundle"]["model_parameters"] < 100_000_000
    assert payload["bundle"]["checkpoint_sha256"] == (
        "9bba544ae37190fdc24ea51ac9f3f8d80350cd905c01e13186156c19f7e0f9ee"
    )
    assert payload["bundle"]["parity_gate_passed"] is True
    assert payload["single_step"]["unambiguous_rows"] == 9
    assert payload["single_step"]["exact_tool_count"] == 4
    assert payload["single_step"]["exact_tool_rate"] == 4 / 9
    assert payload["planner"]["exact_trajectory_count"] == 0
    assert payload["planner"]["trajectory_count"] == 2
    assert "not an official benchmark split" in payload["source"]["claim"]
    assert "does not establish learned quality" in payload["claim_boundary"]
