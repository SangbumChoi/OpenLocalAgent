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


def test_m208_source_audit_is_explicit_about_training_and_native_boundaries() -> None:
    path = Path("docs/paper/results/raw/m208-realistic-evaluation-source-audit-v1.json")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    self_hash = receipt.pop("receipt_self_sha256")
    assert self_hash == _canonical_sha256(receipt)
    assert receipt["catalogs"]["canonical"]["entries"] == 40
    assert receipt["catalogs"]["supplemental"]["entries"] == 21
    assert len(receipt["sources"]) == 15
    assert receipt["admission_summary"]["train_eligible_currently"] == [
        "androidcontrol",
        "android_in_the_wild",
        "xlam_function_calling",
        "mind2web_train",
    ]
    assert "official_split_verified" in receipt["admission_summary"]["native_receipt_requirements"]
    assert "Do not admit evaluation task text" in receipt["admission_summary"]["decision"]
