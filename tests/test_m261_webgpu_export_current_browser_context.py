import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m261-webgpu-export-current-browser-context-v1.json")


def test_m261_current_browser_context_webgpu_export_is_parity_gated_not_public() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert expected == actual
    assert payload["checkpoint"]["parameters"] == 10524544
    assert payload["bundle"]["artifact_count"] == 9
    assert payload["bundle"]["parity_gate"] == {
        "hard_gate": True,
        "passed": True,
        "model_onnx": {
            "fp32_logits_max_abs_diff": 7.152557373046875e-06,
            "fp32_hidden_max_abs_diff": 4.351139068603516e-06,
            "minimum_logits_argmax_agreement": 1.0,
        },
        "model_fp16_onnx": {
            "fp16_logits_max_abs_diff": 0.006051778793334961,
            "fp16_hidden_max_abs_diff": 0.004587888717651367,
            "minimum_logits_argmax_agreement": 1.0,
        },
        "action_model_onnx": {
            "fp32_hidden_max_abs_diff": 4.351139068603516e-06,
        },
        "action_model_fp16_onnx": {
            "fp16_hidden_max_abs_diff": 0.004587888717651367,
        },
    }
    assert payload["deployment"] == {
        "static_bundle_ready": True,
        "native_webgpu_provider_verified": False,
        "public_space_uploaded": False,
        "reason": "export and CPU parity are verified locally; no current-checkpoint Space upload or browser-hardware receipt exists",
    }
