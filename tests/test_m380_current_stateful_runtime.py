from __future__ import annotations

import json
from pathlib import Path

from localagent.train.stage_data import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m380-current-stateful-runtime-v1.json"


def test_m380_current_stateful_runtime_binds_parent_and_keeps_boundary_honest() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    assert expected == canonical_sha256(body)
    assert payload["kind"] == "localagent_stateful_runtime_evaluation"
    assert payload["checkpoint"]["sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )
    assert payload["runtime"]["public_benchmark"] is False
    assert payload["runtime"]["tool_side_effects"] == "in_memory_only"
    assert payload["oracle"]["task_complete_rate"] == 1.0
    assert payload["model"]["task_complete_rate"] == 0.0
    assert payload["model"]["accepted_steps"] == 0
    assert payload["model"]["tasks"] == 5
    assert "Neither pass is an AndroidWorld" in payload["claim_boundary"]
