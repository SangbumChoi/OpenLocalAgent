from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT
    / "docs"
    / "paper"
    / "results"
    / "webgpu-proxy-pilot-seed2027.corrected-browser.protocol.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_corrected_browser_protocol_is_frozen_against_tracked_inputs() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())

    assert protocol["schema_version"] == 2
    assert protocol["frozen_before_corrected_browser_run"] is True
    assert protocol["status"] == "superseded_before_external_timestamp_and_browser_runs"
    assert protocol["supersession"] == {
        "date": "2026-07-29",
        "reason": (
            "The live browser runner was intentionally upgraded to a cache-bearing "
            "autoregressive prefill/decode ABI before this local-only protocol received an "
            "external timestamp or collected any corrected-condition runs."
        ),
        "frozen_hashes_relabelled": False,
        "corrected_condition_results_collected": False,
        "replacement_protocol_status": (
            "must be frozen from the final trained bundle and current runner before any new "
            "latency or capability collection"
        ),
    }
    assert protocol["local_freeze"]["external_timestamp_status"] == (
        "required_before_browser_collection"
    )
    assert protocol["feature_contract"] == {
        "policy": "structured_one_forward",
        "materialization": (
            "append single-token spaces after the complete natural prompt until "
            "input_tokens equals 512"
        ),
        "decision_feature": "hidden[natural_input_tokens - 1]",
        "pointer_domain": "token and hidden positions [0, natural_input_tokens)",
        "claim": "natural-context quality with 512-token graph compute",
        "not_claimed": "quality after conditioning on 512 meaningful context tokens",
    }

    assert {
        name: artifact["sha256"]
        for name, artifact in protocol["frozen_runner_sources"].items()
    } == {
        "app_js": "1f4fcbe9cb3a519b30a553b5b6b54c539b6ca95a725a4b5d0eea428828599caf",
        "action_js": "0ece65fe99315dfc1f861931800f34920624e81c315ceedcea4c263dc563d0e4",
        "action_html": "1358b624163448c3b521a7210543fe1cfc32535cfe67b5b43b6065eb9c1f6459",
        "dom_js": "fc075ba08d6c2be1bda42c5af9d7a256a88441195f2feca6932779096d3ea99c",
        "dom_html": "b4e543c971db3bf91ed4014372c6ee75c3d8c8ac1247d00900c7c10a941596ce",
    }
    drifted_sources = {
        name
        for name, artifact in protocol["frozen_runner_sources"].items()
        if _sha256(ROOT / artifact["path"]) != artifact["sha256"]
    }
    assert drifted_sources == {"app_js", "action_js", "action_html"}

    for suite in (
        protocol["browser_conditions"]["action"]["suite"],
        protocol["browser_conditions"]["dom"]["suite"],
    ):
        assert _sha256(ROOT / suite["path"]) == suite["sha256"]

    conditions = protocol["browser_conditions"]
    assert conditions["action"]["runner_version"] == "rtab-0.4"
    assert conditions["action"]["case_order_seed"] == "slmw2026-v2-trailing"
    assert conditions["dom"]["runner_version"] == "rtab-dom-0.4"
    assert conditions["dom"]["case_order_seed"] == "dom-loop-v2-trailing"
    assert conditions["action_timeout_ms"] == 10_000
    assert "aborts the entire page collection" in conditions["timeout_contract"]


def test_corrected_browser_protocol_binds_prior_and_offline_evidence() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())

    for prior in (
        protocol["prior_failed_condition"]["action_summary"],
        protocol["prior_failed_condition"]["dom_summary"],
    ):
        summary = json.loads((ROOT / prior["path"]).read_text())
        assert summary["summary_sha256"] == prior["summary_sha256"]

    audit = protocol["offline_packaging_invariance_audit"]
    audit_path = ROOT / audit["path"]
    summary = json.loads(audit_path.read_text())
    assert _sha256(audit_path) == audit["file_sha256"]
    assert summary["summary_sha256"] == audit["summary_sha256"]

    parity = protocol["full_stack_structured_export_parity"]
    parity_path = ROOT / parity["path"]
    summary = json.loads(parity_path.read_text())
    assert parity["status"] == summary["status"] == "passed"
    assert parity["suite_role"] == summary["diagnostic_status"]["suite_role"]
    assert _sha256(parity_path) == parity["file_sha256"]
    assert summary["summary_sha256"] == parity["summary_sha256"]
