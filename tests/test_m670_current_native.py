import hashlib
import json
from pathlib import Path

from localagent.eval.workshop_gate import build_workshop_gate


ROOT = Path(__file__).parents[1]


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert claimed == actual
    return payload


def test_m670_current_native_receipts_bind_the_m666_checkpoint() -> None:
    mobile_path = ROOT / "docs/paper/results/raw/m670-m666-mobilegym-native-v1.json"
    browser_path = ROOT / "docs/paper/results/raw/m670-m666-browsergym-native-v1.json"
    mobile = _load(mobile_path)
    browser = _load(browser_path)
    assert mobile["benchmark_id"] == "mobilegym"
    assert mobile["official_split_verified"] is True
    assert mobile["task_count"] == 256
    assert browser["benchmark_id"] == "browsergym_miniwob"
    assert browser["official_split_verified"] is True
    assert browser["task_count"] == 240
    checkpoint = mobile["checkpoint"]["sha256"]
    assert browser["checkpoint"]["sha256"] == checkpoint
    report = build_workshop_gate(
        ROOT / "configs/data/realistic-agent-eval.catalog.yaml",
        repo_root=ROOT,
        native_receipts={"mobilegym": mobile_path, "browsergym_miniwob": browser_path},
        current_checkpoint=Path("/private/tmp/m665-appworld-full/max64/warm-child.pt"),
    )
    checks = {item["requirement"]: item for item in report["checks"]}
    assert checks["native:mobilegym"]["status"] == "pass"
    assert checks["native:browsergym_miniwob"]["status"] == "pass"
