import json
from pathlib import Path

from localagent.train.stage_data import canonical_sha256


ROOT = Path(__file__).parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m327-computer-agent-arena-current-v1.json"


def test_m327_current_checkpoint_desktop_probe_is_hash_bound_and_honest() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    assert expected == canonical_sha256(body)
    assert payload["dataset"] == "xlangai/computer-agent-arena"
    assert payload["source_revision"] == "897b9f45287c516a44f9e79879b14bc3c1bc5b0a"
    assert payload["source"]["split_policy"] == "evaluation_only"
    assert payload["checkpoint"]["sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )
    assert payload["selection"]["evaluated_rows"] == 256
    assert payload["selection"]["unique_parent_tasks"] == 256
    assert payload["overall"]["route_accuracy"] == 1.0
    assert payload["overall"]["tool_exact_rate"] == 0.0390625
    assert payload["by_gold_family"]["pointer"]["tool_exact_rate"] == 0.0
    assert "not a Computer Agent Arena score" in payload["claim_boundary"]
