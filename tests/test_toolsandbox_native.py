import json
from pathlib import Path
from types import SimpleNamespace

from scripts.run_toolsandbox_native import _render_prompt, _success_message


def test_toolsandbox_prompt_omits_system_policy_from_generic_slot_grounding() -> None:
    prompt = _render_prompt(
        [
            SimpleNamespace(sender="RoleType.SYSTEM", content="Don't guess values."),
            SimpleNamespace(sender="RoleType.USER", content="Message Fredrik Thordendal."),
            SimpleNamespace(sender="RoleType.EXECUTION_ENVIRONMENT", content="[]"),
        ]
    )
    assert "SYSTEM" not in prompt
    assert "USER: Message Fredrik Thordendal." in prompt
    assert "TOOL_RESULT: []" in prompt


def test_toolsandbox_native_completion_templates_are_argument_grounded() -> None:
    assert _success_message("set_wifi_status", {"on": False}) == "Wifi is turned off"
    assert _success_message(
        "send_message_with_phone_number",
        {"phone_number": "+12453344098", "content": "hello"},
    ) == "Your message to +12453344098 has been sent saying: hello"


def test_toolsandbox_native_receipt_marks_smoke_boundary() -> None:
    receipt = json.loads(
        Path("docs/paper/results/raw/m64-toolsandbox-native-smoke-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["environment_executed"] is True
    assert receipt["verifier_executed"] is True
    assert receipt["official_split_verified"] is False
    assert receipt["post_tool_response_policy"] == (
        "deterministic_function_name_and_argument_template"
    )
    assert receipt["success_rate"] == 1.0


def test_toolsandbox_weight_receipt_keeps_transfer_claim_bounded() -> None:
    receipt = json.loads(
        Path("docs/paper/results/raw/m65-toolsandbox-weight-transfer-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["compatibility"]["config_mismatches"] == {}
    assert receipt["compatibility"]["shape_mismatches"] == {}
    assert receipt["compatibility"]["tokenizer_sha256_equal"] is True
    assert receipt["groups"]["embedding"]["relative_delta_l2"] < 0.01
    assert "not that transfer is optimal" in receipt["recommendation"]["claim_boundary"]
