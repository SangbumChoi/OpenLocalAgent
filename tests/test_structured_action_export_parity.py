from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from localagent.inference.export.structured_action_parity import (
    _browser_ground_actions,
    canonical_sha256,
    materialize_trailing_compute,
)
from localagent.model.tokenizer import ASSISTANT, USER, ByteTokenizer


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "docs"
    / "paper"
    / "results"
    / "sft-structured-export-parity-seed2027.summary.json"
)
M32_RESULT = (
    ROOT
    / "docs"
    / "paper"
    / "results"
    / "raw"
    / "m32-webgpu-realistic-browser-tool-pool-v1.json"
)


def test_trailing_compute_materialization_preserves_natural_decision_boundary() -> None:
    tokenizer = ByteTokenizer()
    query = "Select 'Confirm'."
    natural = tokenizer.encode(f"{USER}{query}{ASSISTANT}")
    materialized = materialize_trailing_compute(
        tokenizer,
        query,
        target_input_tokens=len(natural) + 9,
        max_seq_len=256,
    )

    assert materialized["ids"][: len(natural)] == natural
    assert materialized["ids"][len(natural) :] == tokenizer.encode(" ") * 9
    assert materialized["input_tokens"] == len(natural) + 9
    assert materialized["natural_input_tokens"] == len(natural)
    assert materialized["decision_feature_index"] == len(natural) - 1
    assert materialized["pointer_domain"] == [0, len(natural)]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_grounding_bridge_executes_exact_browser_pointer_and_normalization_contract() -> None:
    grounded, node_version = _browser_ground_actions(
        app_js_path=ROOT / "spaces/localagent-webgpu/app.js",
        benchmark_js_path=ROOT / "spaces/localagent-webgpu/benchmark.js",
        rows=[
            {
                "key": "tool",
                "is_stop": False,
                "prompt": "Show api/routes.go.",
                "tool": "read_file",
                "schema": {
                    "properties": {
                        "path": {"type": "string", "format": "path"},
                    },
                    "required": ["path"],
                },
                "pointer_values": {"path": "learned/copied.py"},
            },
            {
                "key": "stop",
                "is_stop": True,
                "prompt": "Explain the sky.",
                "tool": None,
                "schema": {},
                "pointer_values": {},
            },
        ],
        node_executable=str(shutil.which("node")),
    )

    assert node_version.startswith("v")
    assert grounded == [
        {
            "key": "tool",
            "grounded_args": {"path": "learned/copied.py"},
            "schema_valid": True,
            "normalized_action": {
                "tool": "read_file",
                "args": {"path": "learned/copied.py"},
            },
        },
        {
            "key": "stop",
            "grounded_args": None,
            "schema_valid": True,
            "normalized_action": {"abstain": True},
        },
    ]


def test_tracked_seed2027_full_stack_export_parity_is_self_consistent() -> None:
    payload = json.loads(RESULT.read_text())
    expected_hash = canonical_sha256(
        {key: value for key, value in payload.items() if key != "summary_sha256"}
    )

    assert payload["passed"] is True
    assert payload["summary_sha256"] == expected_hash
    assert payload["diagnostic_status"]["suite_role"] == "diagnostic_reuse"
    assert payload["diagnostic_status"]["independent_capability_estimate"] is False
    assert payload["checkpoint"]["sha256"] == (
        "79387105de75d332413262e8d8ddb847b6cc13bc03f5e4df3c81663d9897aef1"
    )
    identities = payload["identities"]["bundle_artifacts"]
    assert identities["action_model.onnx"]["sha256"] == (
        "5bf3817b9f147e528056d237a5cab964a90ed4f3db7a2cc398523d7a391bfcba"
    )
    assert identities["action_model.fp16.onnx"]["sha256"] == (
        "b91e7f84077155640a5e288a7c58c2245c298859ddd86bd7268e71039e65c49a"
    )
    assert identities["tokenizer.json"]["sha256"] == (
        "8365405524329487aea3b087cc999db887d8276115e67e88ebfcf7901b15617c"
    )
    assert payload["head_serialization"]["passed"] is True

    aggregate = payload["aggregate"]
    assert aggregate["configured_cases"] == aggregate["eligible_cases"] == 20
    assert aggregate["tool_required_cases"] == 19
    assert aggregate["all_inputs_exact_target_length"] is True
    assert aggregate["all_decision_indices_natural"] is True
    assert aggregate["all_pointer_domains_natural"] is True
    for pair in (
        "onnx_fp32_vs_native_fp32",
        "onnx_fp16_vs_native_fp32",
    ):
        assert aggregate["pair_gates"][pair]["passed"] is True
        assert aggregate["comparisons"][pair]["route_exact"] == 20
        assert aggregate["comparisons"][pair]["selected_tool_exact"] == 20
        assert aggregate["comparisons"][pair]["pointer_span_exact"] == 11
        assert aggregate["comparisons"][pair]["grounded_args_exact"] == 20
        assert aggregate["comparisons"][pair]["final_normalized_action_exact"] == 20

    diagnostics = aggregate["runtime_diagnostics"]
    assert {row["exact_action"] for row in diagnostics.values()} == {16}
    assert {row["schema_valid"] for row in diagnostics.values()} == {20}


def test_tracked_m32_realistic_browser_bundle_receipt_is_self_consistent() -> None:
    payload = json.loads(M32_RESULT.read_text())
    expected_hash = canonical_sha256(
        {key: value for key, value in payload.items() if key != "summary_sha256"}
    )

    assert payload["summary_sha256"] == expected_hash
    assert payload["status"] == "passed_export_and_structured_parity"
    assert payload["tool_pool"]["count"] == 53
    assert payload["tool_pool"]["names"][-3:] == ["web_click", "web_type", "web_select"]
    assert payload["export"]["hard_parity"]["passed"] is True
    parity = payload["structured_action_parity"]
    assert parity["passed"] is True
    assert parity["configured_cases"] == 20
    assert parity["native_vs_onnx_fp32"]["final_normalized_action_exact"] == 20
    assert parity["native_vs_onnx_fp16"]["final_normalized_action_exact"] == 20
    assert parity["diagnostic_model_quality"]["tool_required_exact_action_accuracy"] == 0.0
