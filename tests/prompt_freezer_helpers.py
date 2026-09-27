from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml

from openlocalagent.data.decontam.evaluation_denylist_suite import (
    CONTRACT_KIND,
    freeze_evaluation_denylist_suite,
    verify_evaluation_denylist_suite,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "data" / "evaluation-benchmarks-paper.yaml"


def _identity(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _relative_identity(path: Path, *, relative_to: Path) -> dict[str, object]:
    return {
        **_identity(path),
        "path": Path(os.path.relpath(path, start=relative_to)).as_posix(),
    }


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def freeze_production_adapter_output(
    directory: Path,
    *,
    suite_name: str,
    prompt_path: Path,
    audit_path: Path,
    plan_attestation_path: Path | None = None,
    ranker_config_path: Path | None = None,
) -> dict[str, object]:
    """Exercise the real adapter-audit shape through the generic suite verifier."""

    plan = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
    suite = plan["suites"][suite_name]
    plan_copy = directory / f"{suite_name}-benchmark-plan.yaml"
    audit = json.loads(
        (plan_attestation_path or audit_path).read_text(encoding="utf-8")
    )
    source_attestation = audit.get("source_attestation")
    plan_mutated = False
    if suite_name == "bfcl" and isinstance(audit.get("sources"), list):
        suite["categories"] = audit["selection"]["caller_declared_categories"]
        suite["expected_input_rows"] = audit["selection"]["source_rows"]
        suite["pinned_prompt_sources"] = {
            source["category"]: {
                "bytes": source["bytes"],
                "file": source["file"],
                "rows": source["rows"],
                "sha256": source["sha256"],
            }
            for source in audit["sources"]
        }
        plan_mutated = True
    elif suite_name == "browsergym" and isinstance(audit.get("capture"), dict):
        suite["prompt_capture"].update(
            {
                "bytes": audit["capture"]["bytes"],
                "sha256": audit["capture"]["sha256"],
                "status": "frozen_controlled_acquisition",
            }
        )
        plan_mutated = True
    if suite_name == "mind2web" and isinstance(source_attestation, dict):
        archive = source_attestation["archive"]
        archive_format = source_attestation["archive_format"]
        protected_archive = suite["protected_test_archive"]
        suite["heldout_splits"] = source_attestation["tasks_by_split"]
        protected_archive.update(
            {
                "bytes": archive["bytes"],
                "compression": archive_format["compression"],
                "encryption": archive_format["encryption"],
                "members": archive_format["members"],
                "member_splits": {
                    member["member"]: member["split"]
                    for member in source_attestation["members"]
                },
                "sha256": archive["sha256"],
                "uncompressed_bytes": source_attestation["total_uncompressed_bytes"],
            }
        )
        plan_mutated = True
    elif suite_name == "weblinx" and isinstance(audit.get("privacy"), dict):
        suite["expected_demonstrations"] = audit["split_demos"]
        suite["expected_source_rows"] = audit["source_rows"]
        suite["pinned_prompt_sources"] = {
            "chat": {
                "bytes": audit["sources"]["chat"]["bytes"],
                "file": audit["sources"]["chat"]["name"],
                "sha256": audit["sources"]["chat"]["sha256"],
            },
            "splits": {
                "bytes": audit["sources"]["splits"]["bytes"],
                "file": audit["sources"]["splits"]["name"],
                "sha256": audit["sources"]["splits"]["sha256"],
            },
        }
        privacy = audit["privacy"]
        suite["privacy_filter_receipt"] = {
            "accepted_demos": privacy["accepted_demos"],
            "excluded_demos": privacy["excluded_demos"],
            "excluded_rows": privacy["excluded_rows"],
            "filter_version": privacy["filter_version"],
            "reason_counts": privacy["reason_counts"],
            "retained_rows": audit["output"]["rows"],
            "scanned_demos": privacy["scanned_demos"],
        }
        plan_mutated = True
    if plan_mutated:
        plan_copy.write_text(yaml.safe_dump(plan, sort_keys=True), encoding="utf-8")
    else:
        plan_copy.write_bytes(PLAN.read_bytes())
    license_evidence = directory / f"{suite_name}-license.txt"
    license_evidence.write_text(
        f"Fixture-only license evidence for {suite_name} integration testing.\n",
        encoding="utf-8",
    )
    records = sum(1 for line in prompt_path.read_bytes().splitlines() if line.strip())
    contract_path = directory / f"{suite_name}-freeze-contract.json"
    frozen_output = directory / f"{suite_name}-frozen-prompts.jsonl"
    provenance_path = directory / f"{suite_name}-frozen-prompts.provenance.json"
    contract = {
        "kind": CONTRACT_KIND,
        "schema_version": 1,
        "suite": {
            "name": suite_name,
            "benchmark": suite["benchmark"],
            "revision": suite["revision"],
            "split": suite["prompt_freeze_split"],
            "adapter": {
                "name": suite["adapter"],
                "version": suite["adapter"],
            },
        },
        "benchmark_plan": {
            **_identity(plan_copy),
            "name": "paper-benchmark-plan",
        },
        "sources": [
            {
                **_identity(prompt_path),
                "name": "adapter-prompt-output",
                "records": records,
            }
        ],
        "adapter_provenance": [
            {
                **_identity(audit_path),
                "name": "source-adapter-audit",
            }
        ],
        "license_evidence": [
            {
                **_identity(license_evidence),
                "name": "benchmark-license",
            }
        ],
        "limits": {
            "max_source_bytes": 128 * 1024 * 1024,
            "max_benchmark_plan_bytes": 1024 * 1024,
            "max_adapter_provenance_bytes": 16 * 1024 * 1024,
            "max_license_evidence_bytes": 16 * 1024 * 1024,
            "max_rows": 250_000,
            "max_record_bytes": 4 * 1024 * 1024,
        },
    }
    if suite_name == "bfcl":
        raw_root = (plan_attestation_path or audit_path).parent
        source_manifest = raw_root / audit["source_manifest"]["file"]
        contract["raw_artifacts"] = [
            {
                **_relative_identity(source_manifest, relative_to=directory),
                "name": "bfcl-source-manifest",
                "role": "bfcl_source_manifest",
            },
            *[
                {
                    **_relative_identity(
                        raw_root / source["file"],
                        relative_to=directory,
                    ),
                    "name": f"bfcl-source-{source['category']}",
                    "role": f"bfcl_source_{source['category']}",
                }
                for source in audit["sources"]
            ],
        ]
    elif suite_name == "mind2web":
        raw_root = (plan_attestation_path or audit_path).parent
        archive_name = (
            audit["source_attestation"]["archive"]["name"]
            if isinstance(audit.get("source_attestation"), dict)
            else "test.zip"
        )
        archive = raw_root / archive_name
        ranker_config = ranker_config_path or (
            ROOT / "configs/data/mind2web-dom-lexical-v1.json"
        )
        contract["raw_artifacts"] = [
            {
                **_relative_identity(archive, relative_to=directory),
                "name": "mind2web-protected-test-archive",
                "role": "mind2web_archive_source",
            },
            {
                **_relative_identity(ranker_config, relative_to=directory),
                "name": "mind2web-dom-ranker-config",
                "role": "mind2web_ranker_config",
            },
        ]
    elif suite_name == "weblinx":
        raw_root = (plan_attestation_path or audit_path).parent
        contract["raw_artifacts"] = [
            {
                **_relative_identity(
                    raw_root / audit["sources"]["chat"]["name"],
                    relative_to=directory,
                ),
                "name": "weblinx-compact-chat-source",
                "role": "weblinx_chat_source",
            },
            {
                **_relative_identity(
                    raw_root / audit["sources"]["splits"]["name"],
                    relative_to=directory,
                ),
                "name": "weblinx-splits-source",
                "role": "weblinx_splits_source",
            },
        ]
    contract_path.write_bytes(_canonical_bytes(contract))
    frozen = freeze_evaluation_denylist_suite(
        contract_path,
        output_path=frozen_output,
        manifest_path=provenance_path,
    )
    assert verify_evaluation_denylist_suite(provenance_path) == frozen
    return frozen
