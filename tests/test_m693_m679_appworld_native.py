"""Integrity checks for the current native AppWorld execution probe."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m693-m679-appworld-native-v1.json")


def _payload() -> dict:
    value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_m693_is_self_hashed_and_contract_bound() -> None:
    payload = _payload()
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    expected = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert payload["receipt_self_sha256"] == expected
    assert payload["runner_contract"] == {"verified": True, "oracle_tasks": 1, "all_reports_contract_passed": True}
    assert payload["protocol"]["native_runtime"].startswith("AppWorld resettable")


def test_m693_records_matched_zero_success_controls() -> None:
    payload = _payload()
    assert payload["source"]["tasks"] == ["6bdbc26_1", "6bdbc26_2", "6bdbc26_3", "396c5a2_1", "396c5a2_2", "396c5a2_3"]
    assert payload["comparison"]["native_task_success"] == {name: 0 for name in ("m679_warm", "m679_random", "m692_warm", "m692_random")}
    assert payload["comparison"]["native_task_success_rate"] == {name: 0.0 for name in ("m679_warm", "m679_random", "m692_warm", "m692_random")}
    assert all(arm["checkpoint"]["sha256"] for arm in payload["arms"].values())
    assert payload["parent_checkpoint"]["sha256"] == "dbf45ea710e1b88cbc6631813b2abc7cfd6b454ee0052b0d0d4881c85d932533"
