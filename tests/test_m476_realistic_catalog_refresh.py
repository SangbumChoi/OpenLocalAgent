"""Integrity checks for the source-linked realistic-agent catalog refresh."""

import hashlib
import json
from pathlib import Path

from localagent.data.realistic_catalog import load_catalog


ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "configs/data/realistic-agent-eval.catalog.yaml"
RECEIPT = ROOT / "docs/paper/results/raw/m476-realistic-agent-catalog-refresh-v1.json"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_catalog_refresh_binds_two_new_mobile_benchmarks() -> None:
    catalog, fingerprint = load_catalog(CATALOG)
    rows = {entry["id"]: entry for entry in catalog["entries"]}
    assert len(catalog["entries"]) == 42
    assert rows["iosworld"]["train_policy"] == "eval_only"
    assert rows["iosworld"]["scale"]["tasks"] == 133
    assert rows["iosworld"]["scale"]["apps"] == 26
    assert rows["mobile_safety_bench"]["train_policy"] == "eval_only"
    assert rows["mobile_safety_bench"]["scale"]["tasks"] == 250
    assert rows["mobile_safety_bench"]["scale"]["prompt_injection_tasks"] == 50
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["receipt_self_sha256"] == _self_hash(receipt)
    assert receipt["catalog"]["sha256"] == fingerprint
    assert receipt["catalog"]["entries"] == 42
    assert receipt["decision"]["training_admission"] is False


def test_catalog_refresh_keeps_exactly_four_train_sources() -> None:
    catalog, _ = load_catalog(CATALOG)
    assert [entry["id"] for entry in catalog["entries"] if entry["train_policy"] == "train"] == [
        "androidcontrol",
        "android_in_the_wild",
        "xlam_function_calling",
        "mind2web_train",
    ]
