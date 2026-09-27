import hashlib
import json
from pathlib import Path


def _canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_m200_toolsandbox_full_base_comparison_is_exact_and_bounded() -> None:
    path = Path("docs/paper/results/raw/m200-toolsandbox-native-current-base-v1.json")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    self_hash = receipt.pop("receipt_self_sha256")
    assert self_hash == _canonical_sha256(receipt)
    assert receipt["source_url"] == "https://github.com/apple/ToolSandbox"
    assert receipt["source_revision"] == "165848b9a78cead7ca7fe7c89c688b58e6501219"
    assert receipt["protocol"]["scenario_selection"].startswith("all 129")
    assert receipt["metrics"]["task_count"] == 129
    assert receipt["metrics"]["current_success_count"] == 28
    assert receipt["metrics"]["baseline_success_count"] == 28
    assert receipt["comparison"] == {
        "mean_similarity_delta": 0.0,
        "same_scenario_count": 129,
        "warm_wins": 0,
        "warm_losses": 0,
        "warm_minus_baseline_success_rate_pp": 0.0,
    }
    assert receipt["official_split_verified"] is False
    assert receipt["decision"]["checkpoint_promoted"] is False
