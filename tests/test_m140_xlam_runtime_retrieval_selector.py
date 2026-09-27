import hashlib
import json
from pathlib import Path


def _canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_m140_receipt_binds_runtime_retrieval_and_keeps_checkpoint_unpromoted() -> None:
    receipt = json.loads(
        Path("docs/paper/results/raw/m140-xlam-runtime-retrieval-selector-v1.json").read_text(
            encoding="utf-8"
        )
    )
    self_hash = receipt.pop("receipt_self_sha256")
    assert self_hash == _canonical_sha256(receipt)
    assert receipt["source"]["official_original_split_verified"] is False
    assert receipt["candidate_policy"]["retrieve_k"] == 10
    runtime = receipt["modes"]["runtime_retriever_selector"]
    global_selector = receipt["modes"]["global_selector"]
    assert runtime["first_tool_exact_rate"] > global_selector["first_tool_exact_rate"]
    assert runtime["schema_valid_rate"] > global_selector["schema_valid_rate"]
    assert runtime["first_arguments_exact_rate"] == 0.0
    assert receipt["decision"]["checkpoint_promoted"] is False
    assert receipt["decision"]["workshop_gate_eligible"] is False
