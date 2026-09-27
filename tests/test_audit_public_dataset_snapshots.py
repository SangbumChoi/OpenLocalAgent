from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.audit_public_dataset_snapshots import snapshot_dataset


def test_snapshot_dataset_records_immutable_revision_and_bounded_file_sample() -> None:
    info = SimpleNamespace(
        sha="abc123",
        created_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
        last_modified=datetime(2026, 3, 4, tzinfo=timezone.utc),
        private=False,
        card_data=SimpleNamespace(license="mit"),
        siblings=[
            SimpleNamespace(rfilename=name, size=index)
            for index, name in enumerate(("z.jsonl", "a.jsonl", "m.jsonl", "b.jsonl"))
        ],
    )
    api = SimpleNamespace(dataset_info=lambda dataset: info)
    row = snapshot_dataset(
        api,
        {
            "id": "demo",
            "dataset": "org/demo",
            "original_url": "https://example.com/original",
            "policy": "evaluation_only",
        },
        sample_files=1,
    )
    assert row["revision"] == "abc123"
    assert row["license"] == "mit"
    assert row["file_count"] == 4
    assert [item["rfilename"] for item in row["file_sample"]] == ["a.jsonl", "z.jsonl"]


def test_m584_receipt_is_hash_bound_and_preserves_train_eval_policy() -> None:
    path = Path("docs/paper/results/raw/m584-public-dataset-snapshot-audit-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert recorded == expected
    assert len(payload["datasets"]) == 8
    by_id = {row["id"]: row for row in payload["datasets"]}
    assert by_id["agentnet"]["policy"].startswith("train_only")
    assert by_id["mind2web"]["policy"].startswith("train_only")
    assert by_id["osworld2_trajectory"]["policy"].startswith("evaluation_only")
    assert by_id["enterpriseopsgym"]["policy"].startswith("evaluation_only")
