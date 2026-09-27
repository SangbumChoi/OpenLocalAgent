import hashlib
import json
from pathlib import Path


def _canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_m212_runtime_heads_require_native_gain_before_adoption() -> None:
    path = Path("docs/paper/results/raw/m212-toolsandbox-runtime-heads-native-bridge-v1.json")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    claimed_hash = receipt.pop("receipt_self_sha256")
    assert claimed_hash == _canonical_sha256(receipt)
    assert receipt["source"]["train"]["rows"] == 105
    assert receipt["source"]["train"]["dropped_static_rows"] == 2
    assert receipt["source"]["state_history_available"] is False
    assert receipt["training"]["warm"]["after_eval"]["selector_top1"] == 0.7
    assert receipt["training"]["random"]["after_eval"]["selector_top1"] == 0.8
    assert receipt["native_toolsandbox_interactive"]["native_success_parity"] is True
    assert receipt["decision"]["export_child"] is False
    assert receipt["decision"]["adopt_warm_child"] is False


def test_runtime_head_scripts_keep_official_split_boundary() -> None:
    for name in ("scripts/train_toolsandbox_runtime_heads.py", "scripts/run_toolsandbox_native.py"):
        source = Path(name).read_text(encoding="utf-8")
        assert "official" in source.lower()
        assert "user simulator" in source.lower()
