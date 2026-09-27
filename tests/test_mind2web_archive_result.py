from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from localagent.data.mind2web_prompts import (
    PRODUCTION_MIND2WEB_ARCHIVE_BYTES,
    PRODUCTION_MIND2WEB_ARCHIVE_SHA256,
    PRODUCTION_MIND2WEB_MEMBERS,
    PRODUCTION_MIND2WEB_REVISION,
    PRODUCTION_MIND2WEB_TOTAL_UNCOMPRESSED_BYTES,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = (
    ROOT
    / "docs"
    / "paper"
    / "results"
    / "mind2web-protected-archive-20260728.reproducibility.json"
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


def test_mind2web_protected_archive_result_is_pinned_and_prompt_free() -> None:
    raw = RESULT_PATH.read_bytes()
    report = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    assert raw == _canonical_bytes(report, pretty=True)
    without_hash = dict(report)
    report_self_sha256 = without_hash.pop("report_self_sha256")
    assert report_self_sha256 == (
        "954c7f509b4aa0aeb709efecea42402170ca109373ecc6ed025f7cd5c4d4dddd"
    )
    assert hashlib.sha256(_canonical_bytes(without_hash)).hexdigest() == (
        report_self_sha256
    )

    plan_raw = PLAN_PATH.read_bytes()
    plan = yaml.safe_load(plan_raw.decode("utf-8"))
    mind2web_plan = plan["suites"]["mind2web"]
    archive_plan = mind2web_plan["protected_test_archive"]
    assert report["dataset_revision"] == PRODUCTION_MIND2WEB_REVISION
    assert report["dataset_revision"] == mind2web_plan["revision"]
    assert report["code_revision"] == mind2web_plan["code_revision"]
    assert report["archive"] == {
        "bytes": PRODUCTION_MIND2WEB_ARCHIVE_BYTES,
        "encryption": archive_plan["encryption"],
        "members": archive_plan["members"],
        "sha256": PRODUCTION_MIND2WEB_ARCHIVE_SHA256,
        "total_uncompressed_bytes": PRODUCTION_MIND2WEB_TOTAL_UNCOMPRESSED_BYTES,
    }

    members = report["members"]
    expected_members = {
        member
        for split_members in PRODUCTION_MIND2WEB_MEMBERS.values()
        for member in split_members
    }
    assert {member["member"] for member in members} == expected_members
    assert len(members) == archive_plan["members"] == 15
    assert sum(member["bytes"] for member in members) == (
        PRODUCTION_MIND2WEB_TOTAL_UNCOMPRESSED_BYTES
    )
    assert all(len(member["sha256"]) == 64 for member in members)
    assert all(len(member["crc32"]) == 8 for member in members)
    assert hashlib.sha256(_canonical_bytes(members)).hexdigest() == (
        report["plaintext_member_identities_sha256"]
    )
    assert report["plaintext_member_identities_sha256"] == (
        "aac613c15140d08cd1760c0935fa7d3fc3f6a87065ca6b2d62b0c024796555c8"
    )

    attempt = report["historical_v1_adapter_attempt"]
    assert attempt == {
        "adapter": "mind2web-private-prompt-rows-v1",
        "audit_published": False,
        "failure": "rendered_prompt_exceeds_max_prompt_bytes",
        "max_output_bytes": 128 * 1024 * 1024,
        "max_prompt_bytes": 512 * 1024,
        "observed_prompt_bytes": 858_832,
        "prompt_output_published": False,
    }
    assert attempt["observed_prompt_bytes"] > attempt["max_prompt_bytes"]

    prompt_export = report["production_v2_prompt_export"]
    assert prompt_export == {
        "adapter": "mind2web-private-prompt-rows-v2",
        "adapter_audit": {
            "audit_self_sha256": (
                "0700ed62ca7fbd1e1e24b5e65c0da01b46367ff0981257eb0fe09136e44cedf0"
            ),
            "bytes": 17_278,
            "file_sha256": (
                "1c0a0399b2f463b354a477e3f27cd470615c8e7474a74358e3a106358424d51b"
            ),
            "schema_version": 3,
        },
        "adapter_implementation": {
            "bytes": 73_241,
            "sha256": (
                "9083ad20cc10f95cdd03ddbaa034faea668e7227c19eda7f5de219c75e22267d"
            ),
        },
        "label_isolation": {
            "current_action_emitted": False,
            "expected_calls_emitted": False,
            "negative_candidates_emitted": False,
            "positive_candidates_emitted": False,
            "prior_action_representations_emitted": False,
            "scores_emitted": False,
        },
        "max_prompt_bytes": 1_771,
        "mode": "production",
        "prompt_export": {
            "bytes": 18_424_572,
            "rows": 9_378,
            "sha256": (
                "4d754431f42eb3a5dbf5cc9dd33783d2a2b8da38a104ac1d08ec0b3bb7a71e2d"
            ),
        },
        "ranker": {
            "config": {
                "bytes": 2_273,
                "config_self_sha256": (
                    "ea258f0eee464f69f18baa97e0631a259c06051d2a9b2ae8454ce34b3244b8f3"
                ),
                "sha256": (
                    "cf9c6e75c465827a97121601b937e77ce992cda25316620dbae73b32b90a3f46"
                ),
            },
            "implementation": {
                "bytes": 33_592,
                "sha256": (
                    "7b6ed35015e8829481c083ce8634277517991e570bbb9485e0ec9b4354b0ddc8"
                ),
            },
            "recall_ceiling_measured": False,
            "scores_emitted": False,
            "version": "mind2web-dom-lexical-v1",
        },
        "split": "cross_domain+cross_task+cross_website",
        "tasks": 1_341,
    }
    assert prompt_export["adapter"] == mind2web_plan["adapter"]
    assert prompt_export["split"] == mind2web_plan["prompt_freeze_split"]
    ranker_plan = mind2web_plan["prompt_ranker"]
    assert prompt_export["adapter_implementation"] == {
        "bytes": ranker_plan["adapter_implementation"]["bytes"],
        "sha256": ranker_plan["adapter_implementation"]["sha256"],
    }
    assert prompt_export["max_prompt_bytes"] == (
        ranker_plan["budget"]["max_unframed_prompt_bytes"]
    )
    assert prompt_export["ranker"]["config"] == {
        "bytes": ranker_plan["config"]["bytes"],
        "config_self_sha256": ranker_plan["config"]["config_self_sha256"],
        "sha256": ranker_plan["config"]["sha256"],
    }
    assert prompt_export["ranker"]["implementation"] == {
        "bytes": ranker_plan["implementation"]["bytes"],
        "sha256": ranker_plan["implementation"]["sha256"],
    }

    raw_chain = report["raw_chain_v3_freeze"]
    assert raw_chain == {
        "benchmark_plan": {
            "bytes": 10_113,
            "sha256": (
                "09584d59c133345eb3fe4507b65a1b7514f04378d6f9b7cae1372afda4a108db"
            ),
        },
        "deduplication": {
            "input_rows": 9_378,
            "normalized_prompt_duplicates_removed": 0,
            "unique_normalized_prompts": 9_378,
        },
        "denylist": {
            "bytes": 18_330_792,
            "records": 9_378,
            "sha256": (
                "759a19c0135ecd5da3d657c2ca43d2c047148bbba243e34d389070756a113f27"
            ),
        },
        "freeze_contract": {
            "bytes": 2_505,
            "sha256": (
                "298837f37991646d322844d9d20ccb062ff064974cf6f68394793b8ac5db0cc6"
            ),
        },
        "independent_replay": {
            "denylist_byte_identical": True,
            "provenance_byte_identical": True,
            "public_verifier_passed": True,
            "raw_prompt_reexport_byte_identical": True,
        },
        "isolation": {
            "benchmark_score_evidence": False,
            "contains_labels_or_expected_outputs": False,
            "fresh_labeled_evaluation_evidence": False,
            "prompt_only": True,
            "purpose": "pretraining_corpus_decontamination_only",
        },
        "provenance": {
            "bytes": 4_175,
            "file_sha256": (
                "8024bb59e77c36b01b874e46fe62e764bbbec1896cf2cc37fb5542da418c8cef"
            ),
            "manifest_self_sha256": (
                "1c9d5e91feaee6a10b1567ae1a76a62f1f9518b350288121fa0675a6f236e6df"
            ),
        },
    }
    assert raw_chain["benchmark_plan"] == {
        "bytes": len(plan_raw),
        "sha256": hashlib.sha256(plan_raw).hexdigest(),
    }
    assert raw_chain["deduplication"]["input_rows"] == (
        prompt_export["prompt_export"]["rows"]
    )
    assert raw_chain["deduplication"]["unique_normalized_prompts"] == (
        raw_chain["denylist"]["records"]
    )

    assert report["canary_checks"]["paper_pretraining_corpus_scan"] == (
        "pending_until_corpus_freeze"
    )
    assert report["status"] == (
        "verified_archive_members_v2_prompt_export_and_v3_raw_chain_freeze"
    )
    assert report["kind"] == (
        "localagent_mind2web_protected_archive_reproducibility"
    )
    assert report["schema_version"] == 1
    assert "not benchmark-score or chronological-freshness evidence" in (
        report["claim_scope"]
    )
    assert "No protected prompt strings, HTML, labels, actions, or source IDs" in (
        report["claim_scope"]
    )
    for protected_content_key in (
        "prompt",
        "html",
        "raw_html",
        "cleaned_html",
        "source_case_id",
        "source_case_id_sha256",
    ):
        assert f'"{protected_content_key}":' not in raw.decode("utf-8")
