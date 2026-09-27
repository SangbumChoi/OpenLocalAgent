import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m349-toolsandbox-weight-transfer-native-control-v1.json"


def _self_hash(payload: dict) -> str:
    body = dict(payload)
    body.pop("receipt_self_sha256")
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def test_m349_binds_source_disjoint_transfer_and_native_arms() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["checkpoint"]["parent_sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )
    assert payload["provenance"]["train_rows"] == 107
    assert payload["provenance"]["eval_rows"] == 20
    assert payload["provenance"]["eval_parent_ids"] == 20
    assert payload["ablation"]["arms"] == ["warm", "random"]


def test_m349_rejects_transfer_when_native_outcome_is_unchanged() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    held_out = payload["metrics"]["held_out"]
    native = payload["metrics"]["native_single_step"]
    assert held_out["warm"]["selector_top1"] == 0.85
    assert held_out["random"]["selector_top1"] == 0.8
    assert native["warm"]["success_rate"] == native["random"]["success_rate"] == 2 / 3
    assert native["warm"]["message_similarity"] == native["random"]["message_similarity"]
    assert payload["decision"]["native_quality_lift"] is False
    assert payload["decision"]["adoption"] == "reject_transfer_for_deployment"
