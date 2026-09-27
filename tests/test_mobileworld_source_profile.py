import json
from pathlib import Path

from localagent.data.public_eval_matrix import load_matrix
from scripts.profile_mobileworld_source import profile


RECEIPT = Path("docs/paper/results/raw/m302-mobileworld-source-runtime-audit-v1.json")


def test_mobileworld_profile_counts_public_tasks_and_keeps_native_score_fail_closed() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["dataset"] == "Tongyi-MAI/MobileWorld"
    assert payload["benchmark_contract"]["tasks"] == 201
    assert sum(payload["benchmark_contract"]["domains"].values()) == 201
    assert payload["execution"]["official_runner_executed"] is False
    assert payload["execution"]["native_environment_executed"] is False
    assert payload["execution"]["score"] is None
    assert "no native score" in payload["claim_boundary"].lower()


def test_mobileworld_matrix_row_is_bound_to_the_profile_and_remains_eval_only() -> None:
    matrix = load_matrix(Path("configs/data/realistic-agent-public-eval-matrix.v1.json"))
    row = next(item for item in matrix["entries"] if item["id"] == "mobileworld")
    assert row["source_revision"] == json.loads(RECEIPT.read_text(encoding="utf-8"))["source_revision"]
    assert row["train_policy"] == "eval_only"
    assert row["local_status"] == "metadata_only"


def test_mobileworld_profile_is_reproducible_on_pinned_checkout() -> None:
    # This check verifies the public profiler remains callable without MobileWorld's optional
    # dependency stack.  The checkout is temporary and intentionally not vendored into Git.
    payload = profile(
        Path("/private/tmp/MobileWorld-0dcd0980eac64d76f498f93568a1ec0594b743c4"),
        revision="0dcd0980eac64d76f498f93568a1ec0594b743c4",
    )
    assert payload["benchmark_contract"]["tasks"] == 201
