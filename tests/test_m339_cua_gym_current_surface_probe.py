import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m339-cua-gym-current-surface-probe-v1.json"


def _self_hash(payload: dict) -> str:
    body = dict(payload)
    body.pop("receipt_self_sha256")
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def test_m339_current_checkpoint_and_split_are_bound() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["arms"]["warm"]["checkpoint_sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )
    assert payload["source"]["revision"] == "3c021d0"
    assert payload["selection"]["task_id_disjoint"] is True
    assert payload["selection"]["selected_rows"] == 3005
    assert payload["arms"]["warm"]["eval"]["rows"] == 628


def test_m339_rejects_warm_transfer_and_native_claims() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["decision"] == "reject_transfer_for_deployment"
    assert payload["comparison"]["eval_accuracy_delta_warm_minus_random"] < 0
    assert payload["comparison"]["eval_balanced_accuracy_delta_warm_minus_random"] < 0
    assert "not CUA-Gym task success" in payload["claim_boundary"]
    assert "native desktop/browser control" in payload["claim_boundary"]
