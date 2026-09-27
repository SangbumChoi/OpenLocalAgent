import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m268-webgpu-native-current-unguarded-retrieval-v1.json")


def test_m268_learned_selector_control_exposes_email_grounding_failure() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert expected == actual
    assert payload["environment_executed"] is True
    assert payload["protocol"]["mobile_guard"] is False
    assert payload["protocol"]["url_guard"] is False
    assert payload["checkpoint"]["sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )
    cases = {case["prompt"]: case for case in payload["cases"]}
    email = cases["Email Dana the quarterly report"]
    assert email["observed_tool"] == "type_text"
    assert email["observed_arguments"] == {"text": "Dana the quarterly report"}
    assert email["exact_actions"] == 0
    assert cases["Open https://example.com"]["exact_actions"] == 30
    assert cases["Write 'WebGPU state loop passed' to Notion"]["exact_actions"] == 30
    assert payload["summary"]["closed_loop_success"] == 0
