import hashlib
import json
from pathlib import Path


def _load(name: str) -> dict:
    return json.loads(Path("docs/paper/results/raw", name).read_text(encoding="utf-8"))


def _self_hash(payload: dict) -> str:
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert expected == actual
    return actual


def test_m154_cua_gym_metadata_receipt_is_complete_but_not_training_data() -> None:
    receipt = _load("m154-cua-gym-metadata-v1.json")
    _self_hash(receipt)
    assert receipt["dataset"] == "xlangai/CUA-Gym"
    assert receipt["source_revision"] == "3c021d0"
    assert receipt["license"] == "CC-BY-4.0"
    assert receipt["split"]["rows"] == 10910
    assert receipt["split"]["official_eval_split_present"] is False
    assert receipt["coverage"]["unique_task_ids"] == 10910
    assert receipt["coverage"]["app_types"] == 327
    assert receipt["coverage"]["ground_truth_counts"] == {"False": 10244, "True": 666}
    assert receipt["source"]["instruction_text_retained"] is False
    assert receipt["source"]["task_artifacts_downloaded"] is False
    assert receipt["source"]["reward_code_executed"] is False


def test_m155_osworld_trajectory_audit_keeps_archives_out_of_sft() -> None:
    receipt = _load("m155-osworld-public-trajectory-source-audit-v1.json")
    assert len(receipt["sources"]) == 2
    osworld2 = next(row for row in receipt["sources"] if row["id"] == "osworld2.0-trajectory")
    assert osworld2["content_consumed"] is False
    assert osworld2["runtime_executed"] is False
    assert osworld2["inventory"]["listed_siblings"] == 99146
    verified = next(row for row in receipt["sources"] if row["id"] == "ubuntu_osworld_verified_trajs")
    assert verified["license"] == "MIT"
    assert verified["inventory"]["root_zip_archives"] == 46
    assert verified["policy"].startswith("evaluation_only")
