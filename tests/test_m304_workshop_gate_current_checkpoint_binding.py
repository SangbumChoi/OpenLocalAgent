import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m304-workshop-gate-current-checkpoint-binding-v1.json")


def test_m304_gate_rejects_legacy_public_artifact_for_current_checkpoint() -> None:
    gate = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert gate["ready"] is False
    assert gate["current_checkpoint"]["sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )
    public = next(
        item
        for item in gate["checks"]
        if item["requirement"] == "artifacts:public_model_demo_manifest"
    )
    assert public["status"] == "blocked"
    assert public["blockers"] == ["current_checkpoint_not_bound"]
