"""Integrity checks for the corrected MCPMark closed-loop diagnostic."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m439-mcpmark-faithful-dispatch-grounding-v1.json")


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m439_binds_current_child_and_corrected_closed_loop_protocol() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["benchmark_id"] == "mcpmark"
    assert payload["dataset"]["revision"] == "cd45b7f57923b9b3985467f5139927575f83141c"
    assert payload["checkpoint"]["parent_sha256"] == (
        "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
    )
    protocol = payload["faithful_native_holdout"]["protocol"]
    assert protocol["continues_after_write_or_move"] is True
    assert protocol["tool_results_returned"] is True
    assert protocol["independent_verifiers"] is True


def test_m439_dispatch_and_grounding_repairs_do_not_claim_native_success() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    heads = payload["dispatch_head_transfer"]
    native = payload["faithful_native_holdout"]
    pointer = payload["pointer_transfer"]
    assert heads["held_out_route_accuracy_after"] > heads["held_out_route_accuracy_before"]
    assert heads["held_out_selector_top1_after"] == 0.0
    assert pointer["span_exact_before"] == pointer["span_exact_after"] == 0.0
    assert native["verifier_passes"] == 0
    assert native["verifier_failures"] == native["protocol"]["tasks"] == 5
    assert native["changed_workspaces"] == 0
    assert payload["top3_candidate_probe"]["selected_tool"] == "write_file"
    assert payload["top3_candidate_probe"]["verifier_pass"] is False
