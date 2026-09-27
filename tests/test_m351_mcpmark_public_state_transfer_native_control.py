import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m351-mcpmark-public-state-transfer-native-control-v1.json"


def _self_hash(payload: dict) -> str:
    body = dict(payload)
    body.pop("receipt_self_sha256")
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def test_m351_binds_source_disjoint_current_parent_and_children() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["benchmark_id"] == "mcpmark"
    assert payload["source"]["revision"] == "e50578f0ab904d8e6a7c576c387c1e76ae482c89"
    assert payload["source"]["train_rows"] == 8
    assert payload["source"]["eval_rows"] == 2
    assert payload["source"]["source_disjoint"] is True
    assert payload["checkpoint"]["parent_sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )
    assert payload["checkpoint_sha256"] == payload["checkpoint"]["parent_sha256"]


def test_m351_warm_token_gain_does_not_promote_native_transfer() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    warm = payload["teacher_forced"]["warm"]
    random = payload["teacher_forced"]["random"]
    assert warm["eval_token_accuracy_after"] > random["eval_token_accuracy_after"]
    assert warm["sequence_accuracy_after"] == random["sequence_accuracy_after"] == 0.0
    assert payload["native"]["warm"]["success_rate"] == payload["native"]["random"]["success_rate"] == 0.0
    assert payload["native"]["warm"]["verifier_exit_code"] == payload["native"]["random"]["verifier_exit_code"] == 1
    assert payload["decision"]["native_quality_lift"] is False
    assert payload["decision"]["adoption"] == "reject_mcpmark_transfer_for_deployment"
    assert payload["official_split_verified"] is False
    assert "neither child is adopted" in payload["claim_boundary"]
