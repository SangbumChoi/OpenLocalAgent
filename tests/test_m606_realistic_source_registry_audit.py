import hashlib
import json
from pathlib import Path


def test_m606_registry_audit_reconciles_canonical_and_supplemental_sources() -> None:
    path = Path("docs/paper/results/raw/m606-realistic-source-registry-audit-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    counts = payload["counts"]
    assert counts["canonical_entries"] == 28
    assert counts["supplemental_entries"] == 30
    assert counts["unique_source_ids"] == 49
    assert counts["overlapping_ids"] == 9
    assert counts["metadata_conflicts"] == 5
    assert counts["canonical_train_rows"] == 6
    assert counts["supplemental_catalog_only_rows"] == 30
    assert {row["id"] for row in payload["train_eligible"]} == {
        "androidcontrol",
        "android_in_the_wild",
        "mind2web",
        "agentnet",
        "xlam_function_calling",
        "toolace",
    }
    assert "No task prompts" in payload["claim_boundary"]
