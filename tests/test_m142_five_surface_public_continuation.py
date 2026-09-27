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


def test_m142_receipt_binds_five_public_surfaces_and_rejects_promotion() -> None:
    path = Path("docs/paper/results/raw/m142-five-surface-public-continuation-v1.json")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    self_hash = receipt.pop("receipt_self_sha256")
    assert self_hash == _canonical_sha256(receipt)
    assert receipt["rows"] == {"train": 4746, "eval": 1064, "source_disjoint": True}
    assert {source["label"] for source in receipt["sources"]} == {
        "android",
        "agentnet",
        "mind2web",
        "toolsandbox",
        "mcpmark",
    }
    assert all(source["url"].startswith("https://") for source in receipt["sources"])
    assert receipt["metrics"]["overall"]["after"]["assistant_token_accuracy"] > receipt[
        "metrics"
    ]["overall"]["before"]["assistant_token_accuracy"]
    assert receipt["metrics"]["exact_sequence_accuracy"] == 0.0
    assert receipt["weight_transfer"]["compatibility"]["tokenizer_sha256_equal"] is True
    assert receipt["decision"]["checkpoint_promoted"] is False
    assert receipt["xlam_diagnostic"]["runtime_retriever_selector"]["first_tool_exact_rate"] == 0.0703125
    assert receipt["xlam_diagnostic"]["global_selector"]["first_tool_exact_rate"] < receipt[
        "xlam_diagnostic"
    ]["runtime_retriever_selector"]["first_tool_exact_rate"]
    assert receipt["xlam_diagnostic"]["runtime_retriever_selector"]["first_arguments_exact_rate"] == 0.0
    assert receipt["random_control"]["same_schedule"] is True
    assert receipt["random_control"]["warm_minus_random_token_accuracy"] > 0.25
    assert receipt["metrics"]["overall"]["after"]["assistant_token_accuracy"] > receipt[
        "random_control"
    ]["after"]["assistant_token_accuracy"]
