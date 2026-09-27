import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m335-mcpmark-current-source-audit-v1.json"


def _self_hash(payload: dict) -> str:
    body = dict(payload)
    body.pop("receipt_self_sha256")
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def test_m335_receipt_is_hash_bound_to_current_public_tree() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    source = payload["source"]
    assert source["revision"] == "cd45b7f57923b9b3985467f5139927575f83141c"
    assert source["task_count"] == 177
    assert source["standard_tasks"] == 127
    assert source["easy_tasks"] == 50
    assert source["metadata_description_files"] == 354
    assert source["task_payload_retained"] is False
    assert payload["runtime"]["executed"] is False
    assert payload["runtime"]["training_rows_admitted"] == 0


def test_m335_distinguishes_five_services_from_six_task_roots() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    source = payload["source"]
    assert len(source["mcp_services"]) == 5
    assert set(source["service_task_roots"]) == {
        "filesystem",
        "github",
        "notion",
        "playwright",
        "playwright_webarena",
        "postgres",
    }
    assert sum(source["service_task_roots"].values()) == source["task_count"]
