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


def test_m672_current_native_receipts_bind_the_m671_child() -> None:
    mobile_path = ROOT / "docs/paper/results/raw/m672-m671-mobilegym-native-v1.json"
    browser_path = ROOT / "docs/paper/results/raw/m672-m671-browsergym-native-v1.json"
    mobile = _load(mobile_path)
    browser = _load(browser_path)
    assert mobile["kind"] == "localagent_m672_m671_mobilegym_native_receipt"
    assert mobile["official_split_verified"] is True
    assert mobile["task_count"] == 256
    assert mobile["result"]["passed_tasks"] == 1
    assert browser["kind"] == "localagent_m672_m671_browsergym_native_receipt"
    assert browser["official_split_verified"] is True
    assert browser["task_count"] == 240
    assert browser["result"]["passed_tasks"] == 5
    checkpoint = mobile["checkpoint"]["sha256"]
    assert browser["checkpoint"]["sha256"] == checkpoint
    report = build_workshop_gate(
        ROOT / "configs/data/realistic-agent-eval.catalog.yaml",
        repo_root=ROOT,
        native_receipts={"mobilegym": mobile_path, "browsergym_miniwob": browser_path},
        current_checkpoint=Path("/private/tmp/m671-mcp-current-v2/warm-child.pt"),
    )
    checks = {item["requirement"]: item for item in report["checks"]}
    assert checks["native:mobilegym"]["status"] == "pass"
    assert checks["native:browsergym_miniwob"]["status"] == "pass"
