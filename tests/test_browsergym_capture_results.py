from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from localagent.data.browsergym_prompts import (
    PRODUCTION_BROWSERGYM_REVISION,
    PRODUCTION_CAPTURE_BYTES,
    PRODUCTION_CAPTURE_FILE,
    PRODUCTION_CAPTURE_RECEIPT_IDENTITY,
    PRODUCTION_CAPTURE_SHA256,
    PRODUCTION_EPISODES,
    PRODUCTION_FIXED_SEEDS,
    PRODUCTION_LOCAL_POLICY_EXCLUSIONS,
    PRODUCTION_MINIWOB_REVISION,
    PRODUCTION_RUNTIME_MANIFEST_IDENTITY,
    PRODUCTION_SIMILARITY_GROUPS,
    PRODUCTION_TASK_VARIANTS,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = (
    ROOT
    / "docs"
    / "paper"
    / "results"
    / "browsergym-miniwob-reset-capture-20260728.reproducibility.json"
)
PLAN_PATH = ROOT / "configs" / "data" / "evaluation-benchmarks-paper.yaml"


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


def _canonical_bytes(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        return f"{payload}\n".encode()
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def test_browsergym_reset_capture_reproducibility_result_is_pinned() -> None:
    raw = RESULT_PATH.read_bytes()
    report = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    assert raw == _canonical_bytes(report, pretty=True)
    without_hash = dict(report)
    report_self_sha256 = without_hash.pop("report_self_sha256")
    assert hashlib.sha256(_canonical_bytes(without_hash)).hexdigest() == (
        report_self_sha256
    )

    plan = yaml.safe_load(PLAN_PATH.read_text(encoding="utf-8"))
    browsergym_plan = plan["suites"]["browsergym"]
    assert browsergym_plan["prompt_capture"] == {
        "status": "frozen_controlled_acquisition",
        "file": PRODUCTION_CAPTURE_FILE,
        "bytes": PRODUCTION_CAPTURE_BYTES,
        "sha256": PRODUCTION_CAPTURE_SHA256,
        "requirement": "freeze_before_tokenizer_fit",
    }
    assert browsergym_plan["capture_receipt"] == dict(
        PRODUCTION_CAPTURE_RECEIPT_IDENTITY
    )
    assert report["capture"] == {
        "bytes": PRODUCTION_CAPTURE_BYTES,
        "rows": PRODUCTION_EPISODES,
        "sha256": PRODUCTION_CAPTURE_SHA256,
    }
    receipt_policy = PRODUCTION_CAPTURE_RECEIPT_IDENTITY
    assert report["producer_receipt"] == {
        "bytes": receipt_policy["bytes"],
        "kind": receipt_policy["kind"],
        "producer": receipt_policy["producer"],
        "receipt_self_sha256": receipt_policy["receipt_self_sha256"],
        "schema_version": receipt_policy["schema_version"],
        "sha256": receipt_policy["sha256"],
    }

    assert browsergym_plan["runtime_pins"]["environment_manifest"] == dict(
        PRODUCTION_RUNTIME_MANIFEST_IDENTITY
    )
    runtime_policy = PRODUCTION_RUNTIME_MANIFEST_IDENTITY
    assert report["runtime_manifest"] == {
        "bytes": runtime_policy["bytes"],
        "distributions": runtime_policy["distributions"],
        "file": f"configs/data/{runtime_policy['file']}",
        "kind": runtime_policy["kind"],
        "playwright_driver_sha256": runtime_policy[
            "playwright_driver_sha256"
        ],
        "schema_version": runtime_policy["schema_version"],
        "self_sha256": runtime_policy["self_sha256"],
        "sha256": runtime_policy["sha256"],
    }
    runtime_path = ROOT / report["runtime_manifest"]["file"]
    runtime_payload = runtime_path.read_bytes()
    assert len(runtime_payload) == report["runtime_manifest"]["bytes"]
    assert hashlib.sha256(runtime_payload).hexdigest() == (
        report["runtime_manifest"]["sha256"]
    )

    assert report["source_revisions"] == {
        "browsergym": PRODUCTION_BROWSERGYM_REVISION,
        "miniwob_plusplus": PRODUCTION_MINIWOB_REVISION,
    }
    assert browsergym_plan["revision"] == PRODUCTION_BROWSERGYM_REVISION
    assert browsergym_plan["miniwob_revision"] == PRODUCTION_MINIWOB_REVISION
    assert report["controlled_plan"] == {
        "fixed_seeds": list(PRODUCTION_FIXED_SEEDS),
        "localagent_policy_exclusions": list(
            PRODUCTION_LOCAL_POLICY_EXCLUSIONS
        ),
        "similarity_groups": PRODUCTION_SIMILARITY_GROUPS,
        "task_variants": PRODUCTION_TASK_VARIANTS,
    }
    assert report["goal_statistics"] == {
        "exact_unique_goals": 163,
        "goal_utf8_bytes_max": 100,
        "goal_utf8_bytes_mean": 57.479166666666664,
        "goal_utf8_bytes_min": 12,
        "goal_utf8_bytes_sum": 13_795,
        "normalized_duplicate_rows_removed": 77,
        "normalized_unique_denylist_prompts": 163,
        "seed_static_tasks": 15,
        "seed_varying_tasks": 45,
    }
    assert (
        report["goal_statistics"]["normalized_unique_denylist_prompts"]
        + report["goal_statistics"]["normalized_duplicate_rows_removed"]
        == PRODUCTION_EPISODES
    )
    assert (
        report["goal_statistics"]["seed_static_tasks"]
        + report["goal_statistics"]["seed_varying_tasks"]
        == PRODUCTION_TASK_VARIANTS
    )
    assert report["derived_decontamination_suite"] == {
        "adapter_audit": {
            "audit_self_sha256": (
                "037f41f1eb11b0d7f1df51d619bb0e7aa1b7e1a0114554f023d7b0073de77175"
            ),
            "bytes": 6_841,
            "sha256": (
                "5fcef66fe418e2643f5306850363848e60d619f447937b712f0bbeb94e83e03e"
            ),
        },
        "benchmark_plan": {
            "bytes": 10_113,
            "sha256": (
                "09584d59c133345eb3fe4507b65a1b7514f04378d6f9b7cae1372afda4a108db"
            ),
        },
        "denylist": {
            "bytes": 26_402,
            "records": 163,
            "sha256": (
                "808b4c5206e5d6ebcd0704b88ba87b7c5c9c7e1866ccfe96ab4201228f436120"
            ),
        },
        "freeze_contract": {
            "bytes": 2_340,
            "sha256": (
                "c244a7ba584dbaa1353aa1b4e56f0c1b933896afd13cb7d3e64785ad08a4f557"
            ),
        },
        "independent_replay": {
            "denylist_byte_identical": True,
            "provenance_byte_identical": True,
            "public_verifier_passed": True,
        },
        "license_evidence": {
            "browsergym": {
                "bytes": 579,
                "sha256": (
                    "b192c58991e8ff585cc574615d40e74185404d4b96c1109d423071ab1367344b"
                ),
            },
            "miniwob_plusplus": {
                "bytes": 1_222,
                "sha256": (
                    "fd6529fb8f648d4130fd969e3d7fc740c9992a0bd94a1998f52555a873311511"
                ),
            },
        },
        "prompt_export": {
            "bytes": 30_773,
            "rows": 240,
            "sha256": (
                "383513a042f0421f9a40da24fb42fae5b45ece69b53bdf762dc0cf6086d0f477"
            ),
        },
        "provenance": {
            "bytes": 4_082,
            "file_sha256": (
                "b9387948eb98837baf5191415acf0458c08d7fb899f70236b56d9f02ced98922"
            ),
            "manifest_self_sha256": (
                "63c8b48a8fdbc58c3086d41537594a2335d7f573360390a9ecf612561372e95f"
            ),
        },
    }

    assert report["comparison"] == {
        "capture_byte_identical": True,
        "receipt_byte_identical": True,
        "separate_python_and_chromium_processes": True,
    }
    assert report["runs"] == [
        {
            "capture_file": (
                "data/private/browsergym/browsergym-miniwob-reset-goals-a.jsonl"
            ),
            "name": "a",
            "receipt_file": (
                "data/private/browsergym/"
                "browsergym-miniwob-reset-goals-a.receipt.json"
            ),
        },
        {
            "capture_file": (
                "data/private/browsergym/browsergym-miniwob-reset-goals-b.jsonl"
            ),
            "name": "b",
            "receipt_file": (
                "data/private/browsergym/"
                "browsergym-miniwob-reset-goals-b.receipt.json"
            ),
        },
    ]
    assert report["kind"] == (
        "localagent_browsergym_miniwob_capture_reproducibility"
    )
    assert report["schema_version"] == 1
    assert report["status"] == (
        "verified_reproducible_controlled_acquisition_and_prompt_freeze"
    )
    assert "not agent evaluation or task-success evidence" in report["claim_scope"]
