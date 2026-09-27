import hashlib
import json
from pathlib import Path


def test_m332_historical_receipt_is_source_pinned_and_not_an_official_score() -> None:
    path = Path("docs/paper/results/raw/m332-mobilesafety-text-policy-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert expected == actual
    assert payload["source"]["dataset"] == "MobileSafetyBench"
    assert payload["source"]["revision"] == "bc5e0579626a280c4f551261abcb721442ff92ea"
    assert payload["source"]["task_rows"] == 90
    assert payload["source"]["qa_rows"] == 3
    assert payload["policy"]["native_execution"] is False
    assert payload["policy"]["external_side_effects"] is False
    assert payload["policy"]["app_sha256"] == (
        "a2d8fed113f3a6716798a0d2ac656c47d17e759dc45b2899b50f65bc03160ca6"
    )
    assert payload["policy"]["app_sha256"] != hashlib.sha256(
        Path("spaces/localagent-webgpu/app.js").read_bytes()
    ).hexdigest()
    assert payload["policy"]["risk_version"] == "text_harm_block_v1"
    assert payload["notable_findings"]["task_text_committed"] is False
    assert payload["summary"]["policy_status_counts"] == {
        "allowed": 23,
        "blocked": 22,
        "confirmation_required": 45,
    }
    assert payload["qa_summary"]["policy_status_counts"] == {
        "blocked": 1,
        "confirmation_required": 2,
    }
    assert payload["notable_findings"]["helpful_blocked_rows"] == 0
    assert "official safety score" in payload["claim_boundary"]


def test_m332_receipt_does_not_embed_public_task_text() -> None:
    payload = json.loads(
        Path("docs/paper/results/raw/m332-mobilesafety-text-policy-v1.json").read_text(
            encoding="utf-8"
        )
    )
    def contains_instruction(value: object) -> bool:
        if isinstance(value, dict):
            return any(contains_instruction(item) for item in value.values())
        if isinstance(value, list):
            return any(contains_instruction(item) for item in value)
        return isinstance(value, str) and "instruction" in value

    assert not contains_instruction(payload["summary"])
    assert not contains_instruction(payload["qa_summary"])
