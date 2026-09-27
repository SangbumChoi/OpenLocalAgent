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


def test_m198_toolsandbox_receipt_is_hash_bound_and_not_official() -> None:
    path = Path("docs/paper/results/raw/m198-toolsandbox-native-current-v1.json")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    self_hash = receipt.pop("receipt_self_sha256")
    assert self_hash == _canonical_sha256(receipt)
    assert receipt["source_url"] == "https://github.com/apple/ToolSandbox"
    assert receipt["source_revision"] == "165848b9a78cead7ca7fe7c89c688b58e6501219"
    assert receipt["checkpoint"]["sha256"] == (
        "3b3fd817c6f52e8b922f84c9bb36e7d55e0e243a0f8ee6d0f20c85962a4eeba7"
    )
    assert receipt["environment_executed"] is True
    assert receipt["verifier_executed"] is True
    assert receipt["official_split_verified"] is False
    assert receipt["results"]["success_count"] == 3
    assert receipt["results"]["task_count"] == 3
    assert receipt["results"]["success_rate"] == 1.0
    assert receipt["decision"] == "diagnostic_only"
