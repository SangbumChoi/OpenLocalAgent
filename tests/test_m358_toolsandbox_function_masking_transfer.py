import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m358-toolsandbox-function-masking-transfer-v1.json"


def _self_hash(payload: dict) -> str:
    body = dict(payload)
    body.pop("receipt_self_sha256")
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def test_m358_receipt_is_self_bound_and_data_bound() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["benchmark_id"] == "toolsandbox_function_masking_transfer_pilot"
    assert payload["source"]["source_url"] == "https://github.com/apple/ToolSandbox"
    assert payload["source"]["revision"] == "165848b9a78cead7ca7fe7c89c688b58e6501219"
    assert payload["data"]["schema_valid_train_rows"] == 78
    assert payload["data"]["schema_invalid_train_rows"] == 29
    assert payload["data"]["schema_valid_eval_rows"] == 15
    assert payload["data"]["schema_invalid_eval_rows"] == 5
    assert payload["data"]["masked_train_rows"] == 156
    assert payload["data"]["masked_eval_rows"] == 30
    assert payload["data"]["prompt_contract"] == "openai_full_catalog_v1"


def test_m358_reports_warm_random_movement_and_keeps_official_gate_closed() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    warm = payload["arms"]["warm_start"]
    random_arm = payload["arms"]["matched_random"]
    assert warm["weight_movement_relative_l2"]["backbone"] < 0.01
    assert random_arm["weight_movement_relative_l2"]["backbone"] > warm[
        "weight_movement_relative_l2"
    ]["backbone"]
    assert warm["after"]["eval_canonical"]["assistant_token_accuracy"] > warm[
        "before"
    ]["eval_canonical"]["assistant_token_accuracy"]
    assert "not the official ToolSandbox split" in payload["claim_boundary"]
    assert "No ToolSandbox simulator" in payload["claim_boundary"]
