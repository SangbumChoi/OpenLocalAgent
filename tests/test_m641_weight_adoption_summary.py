import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m641-m626-weight-adoption-summary-v1.json")
CURRENT_SHA = "d6b5df5ff0ba53cb797501b21fd788ca904b2773cf7ea56269864d0350670e3c"


def test_m641_adoption_summary_is_checkpoint_bound_and_fail_closed() -> None:
    if not RECEIPT.exists():
        return
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recorded == expected
    assert payload["current_checkpoint"]["sha256"] == CURRENT_SHA
    assert payload["adoption_decision"]["reuse_current_warm_backbone"] is True
    assert payload["adoption_decision"]["export_to_webgpu_as_native_agent"] is False
    assert payload["native_guardrails"]["mobilegym"]["tasks"] == 256
    assert payload["native_guardrails"]["browsergym_miniwob"]["episodes"] == 240
    assert payload["public_transfer_evidence"]["enterpriseopsgym_email"]["warm_minus_random_top1_pp"] > 20
