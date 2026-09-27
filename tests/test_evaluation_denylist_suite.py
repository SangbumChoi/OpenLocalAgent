import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import localagent.data.bfcl_prompts as bfcl_prompts
import localagent.data.browsergym_capture as browsergym_capture
import localagent.data.browsergym_prompts as browsergym_prompts
import localagent.data.evaluation_denylist_suite as denylist_suite
from localagent.data.bfcl_prompts import (
    BFCL_SOURCE_MANIFEST_KIND,
    PRODUCTION_BFCL_REVISION,
    export_bfcl_prompt_rows,
)
from localagent.data.evaluation_denylist_suite import (
    CONTRACT_KIND,
    MANIFEST_KIND,
    freeze_evaluation_denylist_suite,
    verify_evaluation_denylist_suite,
)
from localagent.data.browsergym_prompts import (
    BROWSERGYM_PROMPT_ADAPTER,
    BROWSERGYM_PROMPT_AUDIT_KIND,
    BROWSERGYM_PROMPT_AUDIT_SCHEMA_VERSION,
    PRODUCTION_BROWSERGYM_REVISION,
    PRODUCTION_BROWSERGYM_VERSION,
    PRODUCTION_CAPTURE_RECEIPT_IDENTITY,
    PRODUCTION_FIXED_SEEDS,
    PRODUCTION_LOCAL_POLICY_EXCLUSIONS,
    PRODUCTION_MINIWOB_REVISION,
    PRODUCTION_RUNTIME_MANIFEST_IDENTITY,
    PRODUCTION_TASK_GROUPS,
)
from localagent.data.pretrain_corpus import read_evaluation_denylist


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_bytes(value))


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(b"".join(_canonical_bytes(row) for row in rows))


def _identity(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


_BFCL_TEST_CATEGORIES = (
    "simple_python",
    "multiple",
    "parallel",
    "parallel_multiple",
)


def _bfcl_raw_chain_fixture(
    directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    source_paths: dict[str, Path] = {}
    source_identities: dict[str, dict[str, object]] = {}
    for category in _BFCL_TEST_CATEGORIES:
        source_path = directory / f"BFCL_v4_{category}.json"
        source_path.write_bytes(
            _canonical_bytes(
                {
                    "id": f"{category}_00",
                    "question": [
                        [
                            {
                                "role": "user",
                                "content": f"Prompt for {category}.",
                            }
                        ]
                    ],
                    "function": [
                        {
                            "name": f"lookup_{category}",
                            "description": f"Look up {category}.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "value": {
                                        "description": "A value.",
                                        "type": "string",
                                    }
                                },
                                "required": ["value"],
                            },
                        }
                    ],
                }
            ).removesuffix(b"\n")
        )
        source_paths[category] = source_path
        source_identities[category] = _identity(source_path)

    monkeypatch.setattr(
        bfcl_prompts,
        "PRODUCTION_BFCL_SOURCE_IDENTITIES",
        {
            category: (
                int(source_identities[category]["bytes"]),
                1,
                str(source_identities[category]["sha256"]),
            )
            for category in _BFCL_TEST_CATEGORIES
        },
    )
    source_manifest_path = directory / "source-manifest.json"
    _write_json(
        source_manifest_path,
        {
            "kind": BFCL_SOURCE_MANIFEST_KIND,
            "schema_version": 1,
            "benchmark": "bfcl-v4",
            "revision": PRODUCTION_BFCL_REVISION,
            "sources": [
                {
                    "category": category,
                    "path": source_paths[category].name,
                    "bytes": source_identities[category]["bytes"],
                    "sha256": source_identities[category]["sha256"],
                }
                for category in _BFCL_TEST_CATEGORIES
            ],
        },
    )

    prompt_path = directory / "adapter-prompts.jsonl"
    audit_path = directory / "adapter-audit.json"
    audit = export_bfcl_prompt_rows(
        source_manifest_path,
        prompt_path,
        audit_path,
    )

    root = Path(__file__).resolve().parents[1]
    plan = yaml.safe_load(
        (root / "configs" / "data" / "evaluation-benchmarks-paper.yaml").read_text(
            encoding="utf-8"
        )
    )
    suite_plan = plan["suites"]["bfcl"]
    suite_plan["categories"] = list(_BFCL_TEST_CATEGORIES)
    suite_plan["expected_input_rows"] = len(_BFCL_TEST_CATEGORIES)
    suite_plan["pinned_prompt_sources"] = {
        category: {
            "file": source_paths[category].name,
            "bytes": source_identities[category]["bytes"],
            "rows": 1,
            "sha256": source_identities[category]["sha256"],
        }
        for category in _BFCL_TEST_CATEGORIES
    }
    plan_path = directory / "benchmark-plan.yaml"
    plan_path.write_text(
        yaml.safe_dump(plan, sort_keys=True),
        encoding="utf-8",
    )
    license_path = directory / "LICENSE"
    license_path.write_text(
        "Fixture-only Apache-2.0 evidence for BFCL raw-chain tests.\n",
        encoding="utf-8",
    )

    contract_path = directory / "freeze-contract.json"
    frozen_output_path = directory / "frozen-prompts.jsonl"
    provenance_path = directory / "frozen-prompts.provenance.json"
    prompt_identity = _identity(prompt_path)
    contract = {
        "kind": CONTRACT_KIND,
        "schema_version": 1,
        "suite": {
            "name": "bfcl",
            "benchmark": "bfcl-v4",
            "revision": PRODUCTION_BFCL_REVISION,
            "split": "multiple+parallel+parallel_multiple+simple_python",
            "adapter": {
                "name": "bfcl-v4-prompt-rows-v1",
                "version": "bfcl-v4-prompt-rows-v1",
            },
        },
        "benchmark_plan": {
            **_identity(plan_path),
            "name": "paper-benchmark-plan",
        },
        "sources": [
            {
                **prompt_identity,
                "name": "adapter-prompt-output",
                "records": audit["output"]["rows"],
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
                **_identity(license_path),
                "name": "benchmark-license",
            }
        ],
        "raw_artifacts": [
            {
                **_identity(source_manifest_path),
                "name": "bfcl-source-manifest",
                "role": "bfcl_source_manifest",
            },
            *[
                {
                    **source_identities[category],
                    "name": f"bfcl-source-{category}",
                    "role": f"bfcl_source_{category}",
                }
                for category in _BFCL_TEST_CATEGORIES
            ],
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
    _write_json(contract_path, contract)
    return {
        "contract": contract_path,
        "prompt": prompt_path,
        "audit": audit_path,
        "frozen_output": frozen_output_path,
        "provenance": provenance_path,
        "source_manifest": source_manifest_path,
        **{
            f"source_{category}": source_path
            for category, source_path in source_paths.items()
        },
    }


def _rows() -> list[dict[str, object]]:
    return [
        {
            "source_case_id": "case-type-ascii",
            "prompt": "Type the quarterly report.",
        },
        {
            "source_case_id": "case-open",
            "prompt": "Open account settings.",
        },
        {
            "source_case_id": "case-type-fullwidth",
            "prompt": "Ｔｙｐｅ the quarterly report.",
        },
    ]


def _fixture(
    directory: Path,
    *,
    rows: list[dict[str, object]] | None = None,
) -> tuple[Path, Path, Path, Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    source_path = directory / "adapter-prompts.jsonl"
    adapter_audit_path = directory / "adapter-audit.json"
    benchmark_plan_path = directory / "benchmark-plan.yaml"
    evidence_path = directory / "benchmark-license.txt"
    contract_path = directory / "denylist-contract.json"
    output_path = directory / "benchmark-prompts.jsonl"
    manifest_path = directory / "benchmark-prompts.provenance.json"
    source_rows = rows if rows is not None else _rows()
    _write_jsonl(source_path, source_rows)
    source_identity = _identity(source_path)
    benchmark_plan_path.write_text(
        yaml.safe_dump(
            {
                "kind": "localagent_external_benchmark_plan",
                "schema_version": 1,
                "purpose": "pretraining_prompt_only_decontamination",
                "forbid_gold_in_prompt_exports": True,
                "prompt_freeze": {
                    "suite_contract_kind": CONTRACT_KIND,
                    "suite_provenance_kind": MANIFEST_KIND,
                    "list_manifest_kind": "localagent_evaluation_denylist_manifest",
                    "schema_version": 1,
                    "require_adapter_audit_binding": True,
                    "require_license_evidence_binding": True,
                    "require_prompt_only_isolation": True,
                    "external_manifest_suites": ["browser-actions-heldout"],
                    "config_hash_pinned_direct_suites": [],
                },
                "suites": {
                    "browser-actions-heldout": {
                        "benchmark": "BrowserActions",
                        "revision": "release-2026.1",
                        "adapter": "browser-actions-jsonl",
                        "prompt_freeze_split": "heldout",
                        "v1_evaluation_scope": "derived",
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    adapter_audit = {
        "adapter": "browser-actions-jsonl",
        "freeze_binding": {
            "adapter": "browser-actions-jsonl",
            "benchmark": "BrowserActions",
            "mode": "production",
            "revision": "release-2026.1",
            "split": "heldout",
            "prompt_only": True,
            "contains_current_step_labels": False,
            "output": {
                "bytes": source_identity["bytes"],
                "sha256": source_identity["sha256"],
                "records": len(source_rows),
            },
        },
        "kind": "localagent_browser_actions_prompt_export_audit",
        "output": {
            "bytes": source_identity["bytes"],
            "sha256": source_identity["sha256"],
        },
        "purpose": "prompt_only_corpus_decontamination",
        "revision": "release-2026.1",
        "schema_version": 1,
    }
    _write_json(
        adapter_audit_path,
        {
            **adapter_audit,
            "audit_self_sha256": hashlib.sha256(
                _canonical_bytes(adapter_audit)[:-1]
            ).hexdigest(),
        },
    )
    evidence_path.write_text(
        "Benchmark license evidence pinned at revision release-2026.1.\n",
        encoding="utf-8",
    )
    contract = {
        "kind": CONTRACT_KIND,
        "schema_version": 1,
        "suite": {
            "name": "browser-actions-heldout",
            "benchmark": "BrowserActions",
            "revision": "release-2026.1",
            "split": "heldout",
            "adapter": {
                "name": "browser-actions-jsonl",
                "version": "browser-actions-jsonl",
            },
        },
        "benchmark_plan": {
            **_identity(benchmark_plan_path),
            "name": "paper-benchmark-plan",
        },
        "sources": [
            {
                **source_identity,
                "name": "heldout-prompts",
                "records": len(source_rows),
            }
        ],
        "adapter_provenance": [
            {
                **_identity(adapter_audit_path),
                "name": "adapter-audit",
            }
        ],
        "license_evidence": [
            {
                **_identity(evidence_path),
                "name": "benchmark-license",
            }
        ],
        "limits": {
            "max_source_bytes": 1_000_000,
            "max_benchmark_plan_bytes": 100_000,
            "max_adapter_provenance_bytes": 100_000,
            "max_license_evidence_bytes": 100_000,
            "max_rows": 100,
            "max_record_bytes": 100_000,
        },
    }
    _write_json(contract_path, contract)
    return contract_path, source_path, evidence_path, output_path, manifest_path


def _browsergym_fixture(
    directory: Path,
) -> tuple[Path, Path, Path, Path, Path, dict[str, object]]:
    directory.mkdir(parents=True, exist_ok=True)
    source_path = directory / "browsergym-prompts.jsonl"
    audit_path = directory / "browsergym-audit.json"
    plan_path = directory / "browsergym-plan.yaml"
    evidence_path = directory / "browsergym-license.txt"
    contract_path = directory / "browsergym-contract.json"
    output_path = directory / "browsergym-frozen.jsonl"
    manifest_path = directory / "browsergym-frozen.provenance.json"
    capture_path = directory / "browsergym-capture.jsonl"
    receipt_path = directory / "browsergym-capture.receipt.json"
    source_pins = {
        "browsergym_revision": PRODUCTION_BROWSERGYM_REVISION,
        "browsergym_version": PRODUCTION_BROWSERGYM_VERSION,
        "miniwob_revision": PRODUCTION_MINIWOB_REVISION,
    }
    runtime_pins = {
        "action_set": "highlevel-default-unused-reset-only",
        "architecture": "arm64",
        "browser_executable": {"bytes": 111, "sha256": "d" * 64},
        "browser_installation": {"bytes": 222, "sha256": "e" * 64},
        "chromium_revision": "1117",
        "chromium_version": "125.0.6422.26",
        "device_scale_factor": 1.0,
        "environment_manifest": dict(PRODUCTION_RUNTIME_MANIFEST_IDENTITY),
        "headless": True,
        "locale": "en-US",
        "max_steps": 10,
        "observation_mode": "processed-dom-axtree-screenshot",
        "os": "darwin",
        "playwright_operation_timeout_seconds": 30.0,
        "playwright_version": "1.44.0",
        "python_version": "3.12.4",
        "timezone_id": "UTC",
        "viewport": {"height": 720, "width": 1280},
    }
    capture_rows = [
        {
            "goal": f"BrowserGym held-out goal {task_name} seed {seed}.",
            "runtime_pins": runtime_pins,
            "seed": seed,
            "similarity_group": group,
            "source_pins": source_pins,
            "split": "test",
            "task_name": task_name,
        }
        for task_name, group in sorted(PRODUCTION_TASK_GROUPS.items())
        for seed in PRODUCTION_FIXED_SEEDS
    ]
    _write_jsonl(capture_path, capture_rows)
    rows = [
        {
            "source_case_id": f"browsergym:{row['task_name']}:{row['seed']}",
            "prompt": row["goal"],
        }
        for row in capture_rows
    ]
    _write_jsonl(source_path, rows)
    source_identity = _identity(source_path)

    capture_identity = {
        "bytes": capture_path.stat().st_size,
        "file": capture_path.name,
        "rows": len(rows),
        "sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(),
    }
    receipt_without_hash = {
        "capture": {
            "bytes": capture_identity["bytes"],
            "sha256": capture_identity["sha256"],
        },
        "kind": PRODUCTION_CAPTURE_RECEIPT_IDENTITY["kind"],
        "producer": PRODUCTION_CAPTURE_RECEIPT_IDENTITY["producer"],
        "schema_version": PRODUCTION_CAPTURE_RECEIPT_IDENTITY["schema_version"],
    }
    receipt = {
        **receipt_without_hash,
        "receipt_self_sha256": hashlib.sha256(
            _canonical_bytes(receipt_without_hash)[:-1]
        ).hexdigest(),
    }
    _write_json(receipt_path, receipt)
    receipt_file_identity = _identity(receipt_path)
    receipt_identity: dict[str, object] = {
        "bytes": receipt_file_identity["bytes"],
        "file": receipt_path.name,
        "kind": receipt["kind"],
        "producer": receipt["producer"],
        "receipt_self_sha256": receipt["receipt_self_sha256"],
        "schema_version": receipt["schema_version"],
        "sha256": receipt_file_identity["sha256"],
    }
    plan = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "data"
            / "evaluation-benchmarks-paper.yaml"
        ).read_text(encoding="utf-8")
    )
    browsergym_plan = plan["suites"]["browsergym"]
    browsergym_plan["prompt_capture"].update(
        {
            "bytes": capture_identity["bytes"],
            "file": capture_identity["file"],
            "sha256": capture_identity["sha256"],
            "status": "frozen_controlled_acquisition",
        }
    )
    browsergym_plan["capture_receipt"] = {
        **receipt_identity,
        "status": "frozen_controlled_acquisition",
    }
    plan_path.write_text(yaml.safe_dump(plan, sort_keys=True), encoding="utf-8")

    task_groups = dict(sorted(PRODUCTION_TASK_GROUPS.items()))
    grouping: dict[str, list[str]] = {}
    for task, group in task_groups.items():
        grouping.setdefault(str(group), []).append(task)
    output_identity = {
        "bytes": source_identity["bytes"],
        "file": source_path.name,
        "rows": len(rows),
        "sha256": source_identity["sha256"],
        "row_keys": ["prompt", "source_case_id"],
    }
    audit_without_hash = {
        "adapter": BROWSERGYM_PROMPT_ADAPTER,
        "benchmark": "browsergym-miniwob",
        "boundary": "fixture trust-chain audit",
        "capture": capture_identity,
        "capture_receipt": receipt_identity,
        "freeze_binding": {
            "adapter": BROWSERGYM_PROMPT_ADAPTER,
            "benchmark": "browsergym-miniwob",
            "mode": "production",
            "revision": PRODUCTION_BROWSERGYM_REVISION,
            "split": "test",
            "prompt_only": True,
            "contains_current_step_labels": False,
            "output": {
                "bytes": source_identity["bytes"],
                "sha256": source_identity["sha256"],
                "records": len(rows),
            },
        },
        "kind": BROWSERGYM_PROMPT_AUDIT_KIND,
        "limits": {},
        "mode": "production",
        "output": output_identity,
        "plan": {
            "episode_rows": len(rows),
            "fixed_seeds": list(PRODUCTION_FIXED_SEEDS),
            "grouping_sha256": hashlib.sha256(
                _canonical_bytes(task_groups)[:-1]
            ).hexdigest(),
            "localagent_policy_exclusions": list(
                PRODUCTION_LOCAL_POLICY_EXCLUSIONS
            ),
            "similarity_group_count": len(set(task_groups.values())),
            "similarity_groups": grouping,
            "splits": ["test"],
            "task_groups": task_groups,
            "task_variants": len(task_groups),
        },
        "purpose": "prompt_only_corpus_decontamination",
        "revision": PRODUCTION_BROWSERGYM_REVISION,
        "runtime_pins": runtime_pins,
        "schema_version": BROWSERGYM_PROMPT_AUDIT_SCHEMA_VERSION,
        "source_pins": source_pins,
        "split": "test",
    }
    audit = {
        **audit_without_hash,
        "audit_self_sha256": hashlib.sha256(
            _canonical_bytes(audit_without_hash)[:-1]
        ).hexdigest(),
    }
    _write_json(audit_path, audit)
    evidence_path.write_text("BrowserGym licenses pinned for fixture.\n", encoding="utf-8")
    contract = {
        "kind": CONTRACT_KIND,
        "schema_version": 1,
        "suite": {
            "name": "browsergym",
            "benchmark": "browsergym-miniwob",
            "revision": PRODUCTION_BROWSERGYM_REVISION,
            "split": "test",
            "adapter": {
                "name": BROWSERGYM_PROMPT_ADAPTER,
                "version": BROWSERGYM_PROMPT_ADAPTER,
            },
        },
        "benchmark_plan": {
            **_identity(plan_path),
            "name": "paper-benchmark-plan",
        },
        "sources": [
            {
                **source_identity,
                "name": "browsergym-prompt-export",
                "records": len(rows),
            }
        ],
        "adapter_provenance": [
            {
                **_identity(audit_path),
                "name": "browsergym-adapter-audit",
            }
        ],
        "license_evidence": [
            {
                **_identity(evidence_path),
                "name": "browsergym-license",
            }
        ],
        "raw_artifacts": [
            {
                **_identity(capture_path),
                "name": "browsergym-raw-capture",
                "role": "browsergym_capture",
            },
            {
                **_identity(receipt_path),
                "name": "browsergym-raw-receipt",
                "role": "browsergym_receipt",
            },
        ],
        "limits": {
            "max_source_bytes": 1_000_000,
            "max_benchmark_plan_bytes": 1_000_000,
            "max_adapter_provenance_bytes": 1_000_000,
            "max_license_evidence_bytes": 100_000,
            "max_rows": 1_000,
            "max_record_bytes": 100_000,
        },
    }
    _write_json(contract_path, contract)
    return (
        contract_path,
        audit_path,
        plan_path,
        output_path,
        manifest_path,
        receipt_identity,
    )


def _rebind_browsergym_audit(contract_path: Path, audit_path: Path) -> None:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit.pop("audit_self_sha256", None)
    audit["audit_self_sha256"] = hashlib.sha256(
        _canonical_bytes(audit)[:-1]
    ).hexdigest()
    _write_json(audit_path, audit)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["adapter_provenance"][0].update(_identity(audit_path))
    _write_json(contract_path, contract)


def _patch_browsergym_code_pins(
    monkeypatch: pytest.MonkeyPatch,
    *,
    audit_path: Path,
    receipt_identity: dict[str, object],
) -> None:
    _patch_browsergym_raw_verifier(monkeypatch)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        browsergym_prompts,
        "PRODUCTION_CAPTURE_FILE",
        audit["capture"]["file"],
    )
    monkeypatch.setattr(
        browsergym_prompts,
        "PRODUCTION_CAPTURE_BYTES",
        audit["capture"]["bytes"],
    )
    monkeypatch.setattr(
        browsergym_prompts,
        "PRODUCTION_CAPTURE_SHA256",
        audit["capture"]["sha256"],
    )
    monkeypatch.setattr(
        browsergym_prompts,
        "PRODUCTION_CAPTURE_RECEIPT_IDENTITY",
        {
            **receipt_identity,
            "status": "frozen_controlled_acquisition",
        },
    )


def _patch_browsergym_raw_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    def verify(_capture_path: Path, receipt_path: Path) -> dict[str, object]:
        return json.loads(Path(receipt_path).read_text(encoding="utf-8"))

    monkeypatch.setattr(
        browsergym_capture,
        "verify_browsergym_capture_receipt",
        verify,
    )


def test_freeze_is_deterministic_prompt_only_and_reader_compatible(tmp_path: Path):
    contract_path, _, _, output_path, manifest_path = _fixture(tmp_path)
    first = freeze_evaluation_denylist_suite(
        contract_path,
        output_path=output_path,
        manifest_path=manifest_path,
    )
    first_bytes = output_path.read_bytes(), manifest_path.read_bytes()
    second = freeze_evaluation_denylist_suite(
        contract_path,
        output_path=output_path,
        manifest_path=manifest_path,
    )
    assert second == first
    assert verify_evaluation_denylist_suite(manifest_path) == first
    assert (output_path.read_bytes(), manifest_path.read_bytes()) == first_bytes

    output_rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert all(
        set(row) == {"prompt", "source_case_id_sha256"} for row in output_rows
    )
    assert all(
        len(row["source_case_id_sha256"]) == 64 for row in output_rows
    )
    assert "case-type-ascii" not in output_path.read_text(encoding="utf-8")
    assert read_evaluation_denylist(output_path) == [
        "Open account settings.",
        "Type the quarterly report.",
    ]
    assert first["kind"] == MANIFEST_KIND
    assert first["deduplication_audit"] == {
        **first["deduplication_audit"],
        "input_rows": 3,
        "unique_normalized_prompts": 2,
        "normalized_prompt_duplicates_removed": 1,
    }
    assert first["isolation"]["prompt_only"] is True
    assert first["isolation"]["fresh_labeled_evaluation_evidence"] is False
    assert first["isolation"]["benchmark_score_evidence"] is False
    adapter_provenance = first["adapter_provenance"]
    assert len(adapter_provenance) == 1
    assert adapter_provenance[0]["name"] == "adapter-audit"
    assert adapter_provenance[0]["adapter"] == "browser-actions-jsonl"
    assert adapter_provenance[0]["audit_kind"] == (
        "localagent_browser_actions_prompt_export_audit"
    )
    frozen_source_identity = _identity(tmp_path / "adapter-prompts.jsonl")
    assert adapter_provenance[0]["bound_prompt_source"] == {
        "bytes": frozen_source_identity["bytes"],
        "records": 3,
        "sha256": frozen_source_identity["sha256"],
    }
    assert frozen_source_identity["bytes"] > 0
    without_hash = dict(first)
    self_hash = without_hash.pop("manifest_self_sha256")
    assert hashlib.sha256(_canonical_bytes(without_hash)).hexdigest() == self_hash


def test_freeze_manifest_and_output_are_portable_across_bundle_locations(
    tmp_path: Path,
):
    artifacts: list[tuple[bytes, bytes]] = []
    for name in ("first", "second"):
        contract_path, _, _, output_path, manifest_path = _fixture(tmp_path / name)
        freeze_evaluation_denylist_suite(
            contract_path,
            output_path=output_path,
            manifest_path=manifest_path,
        )
        artifacts.append((output_path.read_bytes(), manifest_path.read_bytes()))
    assert artifacts[0] == artifacts[1]


def test_manifest_paths_are_portable_relative_to_the_manifest(tmp_path: Path):
    contract_path, _, _, _, _ = _fixture(tmp_path)
    output_path = tmp_path / "bundle" / "data" / "prompts.jsonl"
    manifest_path = tmp_path / "bundle" / "provenance" / "suite.json"
    manifest = freeze_evaluation_denylist_suite(
        contract_path,
        output_path=output_path,
        manifest_path=manifest_path,
    )
    assert manifest["output"]["path"] == "../data/prompts.jsonl"
    assert (manifest_path.parent / manifest["output"]["path"]).resolve() == (
        output_path.resolve()
    )
    assert (manifest_path.parent / manifest["contract"]["path"]).resolve() == (
        contract_path.resolve()
    )


@pytest.mark.parametrize(
    "extra",
    [
        {"label": True},
        {"expected_calls": [{"name": "click", "arguments": {}}]},
        {"metadata": {"gold": {"answer": "secret"}}},
        {"trace": [{"toolCalls": [], "output": "secret"}]},
    ],
)
def test_freeze_rejects_label_and_action_fields_recursively(
    tmp_path: Path,
    extra: dict[str, object],
):
    row = {
        "source_case_id": "case-1",
        "prompt": "Open account settings.",
        **extra,
    }
    contract_path, _, _, output_path, manifest_path = _fixture(
        tmp_path,
        rows=[row],
    )
    with pytest.raises(ValueError, match="forbidden label/action field"):
        freeze_evaluation_denylist_suite(
            contract_path,
            output_path=output_path,
            manifest_path=manifest_path,
        )
    assert not output_path.exists()
    assert not manifest_path.exists()


def test_freeze_rejects_unallowlisted_metadata_even_when_prompt_only(tmp_path: Path):
    row = {
        "source_case_id": "case-1",
        "prompt": "Open account settings.",
        "metadata": {"family": "navigation"},
    }
    contract_path, _, _, output_path, manifest_path = _fixture(
        tmp_path,
        rows=[row],
    )
    with pytest.raises(ValueError, match="keys mismatch"):
        freeze_evaluation_denylist_suite(
            contract_path,
            output_path=output_path,
            manifest_path=manifest_path,
        )


@pytest.mark.parametrize("artifact", ["source", "plan", "adapter", "evidence"])
def test_freeze_rejects_source_adapter_or_license_evidence_drift(
    tmp_path: Path,
    artifact: str,
):
    contract_path, source_path, evidence_path, output_path, manifest_path = _fixture(
        tmp_path
    )
    drifted = {
        "source": source_path,
        "plan": tmp_path / "benchmark-plan.yaml",
        "adapter": tmp_path / "adapter-audit.json",
        "evidence": evidence_path,
    }[artifact]
    drifted.write_bytes(drifted.read_bytes() + b"drift\n")
    with pytest.raises(ValueError, match="byte identity disagrees"):
        freeze_evaluation_denylist_suite(
            contract_path,
            output_path=output_path,
            manifest_path=manifest_path,
        )
    assert not output_path.exists()
    assert not manifest_path.exists()


def test_freeze_rejects_adapter_audit_not_bound_to_prompt_source(tmp_path: Path):
    contract_path, _, _, output_path, manifest_path = _fixture(tmp_path)
    adapter_audit_path = tmp_path / "adapter-audit.json"
    audit = json.loads(adapter_audit_path.read_text(encoding="utf-8"))
    audit.pop("audit_self_sha256")
    audit["output"]["sha256"] = "b" * 64
    audit["audit_self_sha256"] = hashlib.sha256(
        _canonical_bytes(audit)[:-1]
    ).hexdigest()
    _write_json(adapter_audit_path, audit)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["adapter_provenance"][0].update(_identity(adapter_audit_path))
    _write_json(contract_path, contract)

    with pytest.raises(ValueError, match="not a declared prompt source"):
        freeze_evaluation_denylist_suite(
            contract_path,
            output_path=output_path,
            manifest_path=manifest_path,
        )


def test_freeze_rejects_declared_prompt_source_without_adapter_binding(
    tmp_path: Path,
):
    contract_path, _, _, output_path, manifest_path = _fixture(tmp_path)
    unaudited_path = tmp_path / "unaudited-prompts.jsonl"
    _write_jsonl(
        unaudited_path,
        [{"source_case_id": "unaudited-case", "prompt": "Unaudited prompt."}],
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["sources"].append(
        {
            **_identity(unaudited_path),
            "name": "unaudited-prompt-export",
            "records": 1,
        }
    )
    _write_json(contract_path, contract)

    with pytest.raises(ValueError, match="bind every declared prompt source exactly once"):
        freeze_evaluation_denylist_suite(
            contract_path,
            output_path=output_path,
            manifest_path=manifest_path,
        )
    assert not output_path.exists()
    assert not manifest_path.exists()


def test_freeze_rejects_benchmark_plan_semantic_mismatch(tmp_path: Path):
    contract_path, _, _, output_path, manifest_path = _fixture(tmp_path)
    benchmark_plan_path = tmp_path / "benchmark-plan.yaml"
    plan = yaml.safe_load(benchmark_plan_path.read_text(encoding="utf-8"))
    plan["suites"]["browser-actions-heldout"]["adapter"] = "wrong-adapter-v1"
    benchmark_plan_path.write_text(
        yaml.safe_dump(plan, sort_keys=True),
        encoding="utf-8",
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["benchmark_plan"].update(_identity(benchmark_plan_path))
    _write_json(contract_path, contract)

    with pytest.raises(ValueError, match="disagrees with the freeze contract"):
        freeze_evaluation_denylist_suite(
            contract_path,
            output_path=output_path,
            manifest_path=manifest_path,
        )


def test_freeze_rejects_partial_mind2web_identity_bypass(tmp_path: Path) -> None:
    contract_path, _, _, output_path, manifest_path = _fixture(tmp_path)
    benchmark_plan_path = tmp_path / "benchmark-plan.yaml"
    plan = yaml.safe_load(benchmark_plan_path.read_text(encoding="utf-8"))
    suite_plan = plan["suites"].pop("browser-actions-heldout")
    plan["suites"]["mind2web"] = suite_plan
    plan["prompt_freeze"]["external_manifest_suites"] = ["mind2web"]
    benchmark_plan_path.write_text(
        yaml.safe_dump(plan, sort_keys=True),
        encoding="utf-8",
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["suite"]["name"] = "mind2web"
    contract["benchmark_plan"].update(_identity(benchmark_plan_path))
    _write_json(contract_path, contract)

    with pytest.raises(ValueError, match="partial mind2web production identity"):
        freeze_evaluation_denylist_suite(
            contract_path,
            output_path=output_path,
            manifest_path=manifest_path,
        )


@pytest.mark.parametrize(
    ("name", "benchmark", "revision", "split", "adapter"),
    [
        (
            "bfcl",
            "bfcl-v4",
            "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8",
            "multiple+parallel+parallel_multiple+simple_python",
            "bfcl-v4-prompt-rows-v1",
        ),
        (
            "browsergym",
            "browsergym-miniwob",
            "9e779f087de9a65668b6974d11f9ce9816026e96",
            "test",
            "browsergym-miniwob-reset-capture-prompt-rows-v1",
        ),
        (
            "mind2web",
            "mind2web",
            "17ece8eb89862368edc0cc806acee6fca5163474",
            "cross_domain+cross_task+cross_website",
            "mind2web-private-prompt-rows-v2",
        ),
        (
            "weblinx",
            "weblinx-chat-v1.0",
            "be2e19d624febb57173e98772c1312d041a6d3b1",
            "test_web",
            "weblinx-private-prompt-rows-v1",
        ),
    ],
)
def test_freeze_rejects_minimal_fabricated_known_adapter_audit(
    tmp_path: Path,
    name: str,
    benchmark: str,
    revision: str,
    split: str,
    adapter: str,
) -> None:
    contract_path, source_path, _, output_path, manifest_path = _fixture(tmp_path)
    benchmark_plan_path = tmp_path / "benchmark-plan.yaml"
    adapter_audit_path = tmp_path / "adapter-audit.json"
    plan = yaml.safe_load(benchmark_plan_path.read_text(encoding="utf-8"))
    suite_plan = plan["suites"].pop("browser-actions-heldout")
    suite_plan.update(
        {
            "adapter": adapter,
            "benchmark": benchmark,
            "prompt_freeze_split": split,
            "revision": revision,
        }
    )
    plan["suites"][name] = suite_plan
    plan["prompt_freeze"]["external_manifest_suites"] = [name]
    benchmark_plan_path.write_text(
        yaml.safe_dump(plan, sort_keys=True),
        encoding="utf-8",
    )
    source_identity = _identity(source_path)
    _write_json(
        adapter_audit_path,
        {
            "adapter": adapter,
            "freeze_binding": {
                "adapter": adapter,
                "benchmark": benchmark,
                "mode": "production",
                "revision": revision,
                "split": split,
                "prompt_only": True,
                "contains_current_step_labels": False,
                "output": {
                    "bytes": source_identity["bytes"],
                    "sha256": source_identity["sha256"],
                    "records": 3,
                },
            },
            "kind": "fabricated_prompt_audit",
            "output": {
                "bytes": source_identity["bytes"],
                "sha256": source_identity["sha256"],
            },
            "purpose": "prompt_only_corpus_decontamination",
            "revision": revision,
            "schema_version": 999,
        },
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["suite"] = {
        "name": name,
        "benchmark": benchmark,
        "revision": revision,
        "split": split,
        "adapter": {"name": adapter, "version": adapter},
    }
    contract["benchmark_plan"].update(_identity(benchmark_plan_path))
    contract["adapter_provenance"][0].update(_identity(adapter_audit_path))
    _write_json(contract_path, contract)

    if name == "browsergym":
        expected_error = "raw capture and one raw receipt"
    elif name == "bfcl":
        expected_error = "source manifest and four pinned sources"
    elif name == "mind2web":
        expected_error = "protected archive and one ranker config"
    elif name == "weblinx":
        expected_error = "compact chat source and one splits source"
    else:
        expected_error = "kind/schema disagrees"
    with pytest.raises(ValueError, match=expected_error):
        freeze_evaluation_denylist_suite(
            contract_path,
            output_path=output_path,
            manifest_path=manifest_path,
        )


def test_bfcl_freezer_requires_and_reexports_exact_raw_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _bfcl_raw_chain_fixture(tmp_path, monkeypatch)

    frozen = freeze_evaluation_denylist_suite(
        paths["contract"],
        output_path=paths["frozen_output"],
        manifest_path=paths["provenance"],
    )

    assert frozen["suite"]["name"] == "bfcl"
    assert {artifact["role"] for artifact in frozen["raw_artifacts"]} == {
        "bfcl_source_manifest",
        "bfcl_source_simple_python",
        "bfcl_source_multiple",
        "bfcl_source_parallel",
        "bfcl_source_parallel_multiple",
    }
    assert verify_evaluation_denylist_suite(paths["provenance"]) == frozen


def test_bfcl_freezer_rejects_fully_recomputed_prompt_forgery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _bfcl_raw_chain_fixture(tmp_path, monkeypatch)
    forged_rows = [
        {
            **row,
            "prompt": f"Caller-authored replacement {index}.",
        }
        for index, row in enumerate(
            [
                json.loads(line)
                for line in paths["prompt"].read_text(encoding="utf-8").splitlines()
            ]
        )
    ]
    _write_jsonl(paths["prompt"], forged_rows)
    forged_identity = _identity(paths["prompt"])

    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    audit["output"].update(
        {
            "bytes": forged_identity["bytes"],
            "rows": len(forged_rows),
            "sha256": forged_identity["sha256"],
        }
    )
    audit["freeze_binding"]["output"].update(
        {
            "bytes": forged_identity["bytes"],
            "records": len(forged_rows),
            "sha256": forged_identity["sha256"],
        }
    )
    _write_json(paths["audit"], audit)

    contract = json.loads(paths["contract"].read_text(encoding="utf-8"))
    contract["sources"][0].update(forged_identity)
    contract["sources"][0]["records"] = len(forged_rows)
    contract["adapter_provenance"][0].update(_identity(paths["audit"]))
    _write_json(paths["contract"], contract)

    with pytest.raises(ValueError):
        freeze_evaluation_denylist_suite(
            paths["contract"],
            output_path=paths["frozen_output"],
            manifest_path=paths["provenance"],
        )
    assert not paths["frozen_output"].exists()
    assert not paths["provenance"].exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_manifest",
        "missing_simple_python",
        "swapped_category_roles",
    ],
)
def test_bfcl_freezer_rejects_missing_or_swapped_raw_roles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    paths = _bfcl_raw_chain_fixture(tmp_path, monkeypatch)
    contract = json.loads(paths["contract"].read_text(encoding="utf-8"))
    raw_artifacts = contract["raw_artifacts"]
    if mutation == "missing_manifest":
        contract["raw_artifacts"] = [
            artifact
            for artifact in raw_artifacts
            if artifact["role"] != "bfcl_source_manifest"
        ]
    elif mutation == "missing_simple_python":
        contract["raw_artifacts"] = [
            artifact
            for artifact in raw_artifacts
            if artifact["role"] != "bfcl_source_simple_python"
        ]
    else:
        simple = next(
            artifact
            for artifact in raw_artifacts
            if artifact["role"] == "bfcl_source_simple_python"
        )
        multiple = next(
            artifact
            for artifact in raw_artifacts
            if artifact["role"] == "bfcl_source_multiple"
        )
        simple["role"], multiple["role"] = multiple["role"], simple["role"]
    _write_json(paths["contract"], contract)

    with pytest.raises(ValueError):
        freeze_evaluation_denylist_suite(
            paths["contract"],
            output_path=paths["frozen_output"],
            manifest_path=paths["provenance"],
        )


def test_bfcl_freezer_rejects_raw_source_tampering_when_contract_is_rebound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _bfcl_raw_chain_fixture(tmp_path, monkeypatch)
    source_path = paths["source_simple_python"]
    source_row = json.loads(source_path.read_text(encoding="utf-8"))
    source_row["question"][0][0]["content"] = "Tampered but contract-bound raw prompt."
    source_path.write_bytes(_canonical_bytes(source_row).removesuffix(b"\n"))

    contract = json.loads(paths["contract"].read_text(encoding="utf-8"))
    raw_source = next(
        artifact
        for artifact in contract["raw_artifacts"]
        if artifact["role"] == "bfcl_source_simple_python"
    )
    raw_source.update(_identity(source_path))
    _write_json(paths["contract"], contract)

    with pytest.raises(ValueError):
        freeze_evaluation_denylist_suite(
            paths["contract"],
            output_path=paths["frozen_output"],
            manifest_path=paths["provenance"],
        )


def test_browsergym_freezer_requires_exact_frozen_receipt_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        contract_path,
        audit_path,
        _,
        output_path,
        manifest_path,
        receipt_identity,
    ) = _browsergym_fixture(tmp_path)
    _patch_browsergym_code_pins(
        monkeypatch,
        audit_path=audit_path,
        receipt_identity=receipt_identity,
    )

    frozen = freeze_evaluation_denylist_suite(
        contract_path,
        output_path=output_path,
        manifest_path=manifest_path,
    )

    assert frozen["suite"]["name"] == "browsergym"
    assert frozen["adapter_provenance"][0]["audit_schema_version"] == (
        BROWSERGYM_PROMPT_AUDIT_SCHEMA_VERSION
    )
    assert {artifact["role"] for artifact in frozen["raw_artifacts"]} == {
        "browsergym_capture",
        "browsergym_receipt",
    }
    assert receipt_identity["schema_version"] == (
        PRODUCTION_CAPTURE_RECEIPT_IDENTITY["schema_version"]
    )


def test_browsergym_freezer_rejects_missing_receipt_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        contract_path,
        audit_path,
        _,
        output_path,
        manifest_path,
        receipt_identity,
    ) = _browsergym_fixture(tmp_path)
    _patch_browsergym_code_pins(
        monkeypatch,
        audit_path=audit_path,
        receipt_identity=receipt_identity,
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit.pop("capture_receipt")
    _write_json(audit_path, audit)
    _rebind_browsergym_audit(contract_path, audit_path)

    with pytest.raises(
        ValueError,
        match=r"capture_receipt disagrees with the receipt-verified raw capture",
    ):
        freeze_evaluation_denylist_suite(
            contract_path,
            output_path=output_path,
            manifest_path=manifest_path,
        )


def test_browsergym_freezer_rejects_receipt_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        contract_path,
        audit_path,
        _,
        output_path,
        manifest_path,
        receipt_identity,
    ) = _browsergym_fixture(tmp_path)
    _patch_browsergym_code_pins(
        monkeypatch,
        audit_path=audit_path,
        receipt_identity=receipt_identity,
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["capture_receipt"]["sha256"] = "f" * 64
    _write_json(audit_path, audit)
    _rebind_browsergym_audit(contract_path, audit_path)

    with pytest.raises(ValueError, match="capture_receipt disagrees"):
        freeze_evaluation_denylist_suite(
            contract_path,
            output_path=output_path,
            manifest_path=manifest_path,
        )


def test_browsergym_freezer_rejects_recomputed_forged_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        contract_path,
        audit_path,
        plan_path,
        output_path,
        manifest_path,
        receipt_identity,
    ) = _browsergym_fixture(tmp_path)
    _patch_browsergym_code_pins(
        monkeypatch,
        audit_path=audit_path,
        receipt_identity=receipt_identity,
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["capture_receipt"].update(
        {
            "producer": "browsergym-miniwob-controlled-reset-goals-forged",
            "receipt_self_sha256": "d" * 64,
            "sha256": "e" * 64,
        }
    )
    _write_json(audit_path, audit)
    # Recompute the outer adapter-audit hash, rewrite the copied benchmark plan to agree with the
    # forgery, and rebind both artifacts in the contract. The immutable production constants,
    # rather than an incidental stale hash or one mutable copy, must reject this forged chain.
    _rebind_browsergym_audit(contract_path, audit_path)
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    plan["suites"]["browsergym"]["capture_receipt"] = {
        **audit["capture_receipt"],
        "status": "frozen_controlled_acquisition",
    }
    plan_path.write_text(yaml.safe_dump(plan, sort_keys=True), encoding="utf-8")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["benchmark_plan"].update(_identity(plan_path))
    _write_json(contract_path, contract)

    with pytest.raises(ValueError, match="receipt-verified raw capture"):
        freeze_evaluation_denylist_suite(
            contract_path,
            output_path=output_path,
            manifest_path=manifest_path,
        )


def test_browsergym_freezer_rejects_rewritten_prompt_with_all_hashes_recomputed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        contract_path,
        audit_path,
        _,
        output_path,
        manifest_path,
        receipt_identity,
    ) = _browsergym_fixture(tmp_path)
    _patch_browsergym_code_pins(
        monkeypatch,
        audit_path=audit_path,
        receipt_identity=receipt_identity,
    )
    source_path = tmp_path / "browsergym-prompts.jsonl"
    rows = [
        json.loads(line)
        for line in source_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["prompt"] = "Attacker-rewritten held-out goal."
    _write_jsonl(source_path, rows)
    source_identity = _identity(source_path)

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["output"].update(
        {
            "bytes": source_identity["bytes"],
            "sha256": source_identity["sha256"],
        }
    )
    audit["freeze_binding"]["output"].update(
        {
            "bytes": source_identity["bytes"],
            "sha256": source_identity["sha256"],
        }
    )
    _write_json(audit_path, audit)
    _rebind_browsergym_audit(contract_path, audit_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["sources"][0].update(source_identity)
    _write_json(contract_path, contract)

    with pytest.raises(
        ValueError,
        match="prompt source differs from the receipt-verified capture",
    ):
        freeze_evaluation_denylist_suite(
            contract_path,
            output_path=output_path,
            manifest_path=manifest_path,
        )


@pytest.mark.parametrize(
    "runtime_key",
    [
        "device_scale_factor",
        "playwright_operation_timeout_seconds",
    ],
)
def test_browsergym_freezer_rejects_bool_numeric_pins_with_hashes_recomputed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_key: str,
) -> None:
    (
        contract_path,
        audit_path,
        plan_path,
        output_path,
        manifest_path,
        receipt_identity,
    ) = _browsergym_fixture(tmp_path)
    _patch_browsergym_code_pins(
        monkeypatch,
        audit_path=audit_path,
        receipt_identity=receipt_identity,
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["runtime_pins"][runtime_key] = True
    _write_json(audit_path, audit)
    _rebind_browsergym_audit(contract_path, audit_path)

    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    plan["suites"]["browsergym"]["runtime_pins"][runtime_key] = True
    plan_path.write_text(yaml.safe_dump(plan, sort_keys=True), encoding="utf-8")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["benchmark_plan"].update(_identity(plan_path))
    _write_json(contract_path, contract)

    with pytest.raises(ValueError, match="runtime_pins disagrees"):
        freeze_evaluation_denylist_suite(
            contract_path,
            output_path=output_path,
            manifest_path=manifest_path,
        )


def test_browsergym_freezer_fails_closed_while_receipt_policy_is_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        contract_path,
        audit_path,
        plan_path,
        output_path,
        manifest_path,
        receipt_identity,
    ) = _browsergym_fixture(tmp_path)
    _patch_browsergym_code_pins(
        monkeypatch,
        audit_path=audit_path,
        receipt_identity=receipt_identity,
    )
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    plan["suites"]["browsergym"]["capture_receipt"].update(
        {
            "status": "pending_controlled_acquisition",
            "file": None,
            "bytes": None,
            "sha256": None,
            "receipt_self_sha256": None,
        }
    )
    plan_path.write_text(yaml.safe_dump(plan, sort_keys=True), encoding="utf-8")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["benchmark_plan"].update(_identity(plan_path))
    _write_json(contract_path, contract)

    with pytest.raises(ValueError, match="receipt remains pending"):
        freeze_evaluation_denylist_suite(
            contract_path,
            output_path=output_path,
            manifest_path=manifest_path,
        )


def test_freeze_rejects_adapter_audit_self_hash_mismatch(tmp_path: Path):
    contract_path, _, _, output_path, manifest_path = _fixture(tmp_path)
    adapter_audit_path = tmp_path / "adapter-audit.json"
    audit = json.loads(adapter_audit_path.read_text(encoding="utf-8"))
    audit["audit_self_sha256"] = "c" * 64
    _write_json(adapter_audit_path, audit)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["adapter_provenance"][0].update(_identity(adapter_audit_path))
    _write_json(contract_path, contract)

    with pytest.raises(ValueError, match="self-hash mismatch"):
        freeze_evaluation_denylist_suite(
            contract_path,
            output_path=output_path,
            manifest_path=manifest_path,
        )


def test_freeze_rejects_adapter_audit_replaced_after_identity_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    contract_path, _, _, output_path, manifest_path = _fixture(tmp_path)
    adapter_audit_path = tmp_path / "adapter-audit.json"
    original_verified_artifact = denylist_suite._verified_artifact
    replaced = False

    def replace_after_identity(
        artifact,
        *,
        max_bytes: int,
        artifact_kind: str,
    ):
        nonlocal replaced
        identity = original_verified_artifact(
            artifact,
            max_bytes=max_bytes,
            artifact_kind=artifact_kind,
        )
        if artifact_kind == "adapter provenance" and not replaced:
            replaced = True
            audit = json.loads(adapter_audit_path.read_text(encoding="utf-8"))
            audit["freeze_binding"]["mode"] = "fixture"
            _write_json(adapter_audit_path, audit)
        return identity

    monkeypatch.setattr(
        denylist_suite,
        "_verified_artifact",
        replace_after_identity,
    )
    with pytest.raises(ValueError, match="changed while it was being verified"):
        freeze_evaluation_denylist_suite(
            contract_path,
            output_path=output_path,
            manifest_path=manifest_path,
        )
    assert replaced is True
    assert not output_path.exists()
    assert not manifest_path.exists()


def test_verify_rechecks_bound_adapter_and_license_artifacts(tmp_path: Path):
    contract_path, _, evidence_path, output_path, manifest_path = _fixture(tmp_path)
    freeze_evaluation_denylist_suite(
        contract_path,
        output_path=output_path,
        manifest_path=manifest_path,
    )
    evidence_path.write_bytes(evidence_path.read_bytes() + b"drift\n")
    with pytest.raises(ValueError, match="byte identity disagrees"):
        verify_evaluation_denylist_suite(manifest_path)


@pytest.mark.parametrize(
    ("limit_name", "limit_value", "message"),
    [
        ("max_rows", 2, "declared source records exceed"),
        ("max_source_bytes", 100, "declared source exceeds"),
        (
            "max_benchmark_plan_bytes",
            10,
            "declared benchmark plan exceeds",
        ),
        (
            "max_license_evidence_bytes",
            10,
            "declared license evidence exceeds",
        ),
        (
            "max_adapter_provenance_bytes",
            10,
            "declared adapter provenance exceeds",
        ),
        ("max_record_bytes", 40, "exceeds max_record_bytes"),
    ],
)
def test_freeze_enforces_contract_file_row_and_record_caps(
    tmp_path: Path,
    limit_name: str,
    limit_value: int,
    message: str,
):
    contract_path, _, _, output_path, manifest_path = _fixture(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["limits"][limit_name] = limit_value
    if limit_name == "max_source_bytes":
        contract["limits"]["max_record_bytes"] = limit_value
    _write_json(contract_path, contract)
    with pytest.raises(ValueError, match=message):
        freeze_evaluation_denylist_suite(
            contract_path,
            output_path=output_path,
            manifest_path=manifest_path,
        )


def test_freeze_rejects_limits_above_hard_caps(tmp_path: Path):
    contract_path, _, _, output_path, manifest_path = _fixture(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["limits"]["max_rows"] = 250_001
    _write_json(contract_path, contract)
    with pytest.raises(ValueError, match="exceeds hard maximum"):
        freeze_evaluation_denylist_suite(
            contract_path,
            output_path=output_path,
            manifest_path=manifest_path,
        )


def test_freeze_rejects_oversized_contract_before_parsing(tmp_path: Path):
    contract_path = tmp_path / "oversized-contract.json"
    contract_path.write_bytes(b" " * (1024 * 1024 + 1))

    with pytest.raises(ValueError, match="contract exceeds 1048576 bytes"):
        freeze_evaluation_denylist_suite(
            contract_path,
            output_path=tmp_path / "output.jsonl",
            manifest_path=tmp_path / "manifest.json",
        )


def test_freeze_accepts_bounded_json_escape_expansion_above_one_mib(
    tmp_path: Path,
):
    escaped_prompt = "\x01" * 300_000
    contract_path, _, _, output_path, manifest_path = _fixture(
        tmp_path,
        rows=[{"source_case_id": "escape-heavy", "prompt": escaped_prompt}],
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["limits"]["max_source_bytes"] = 2_500_000
    contract["limits"]["max_record_bytes"] = 2_500_000
    _write_json(contract_path, contract)

    manifest = freeze_evaluation_denylist_suite(
        contract_path,
        output_path=output_path,
        manifest_path=manifest_path,
    )

    assert output_path.stat().st_size > 1024 * 1024
    assert manifest["output"]["records"] == 1
    assert verify_evaluation_denylist_suite(manifest_path) == manifest


def test_freeze_rejects_duplicate_source_case_ids(tmp_path: Path):
    rows = [
        {"source_case_id": "duplicate", "prompt": "First prompt."},
        {"source_case_id": "duplicate", "prompt": "Second prompt."},
    ]
    contract_path, _, _, output_path, manifest_path = _fixture(
        tmp_path,
        rows=rows,
    )
    with pytest.raises(ValueError, match="duplicate source_case_id"):
        freeze_evaluation_denylist_suite(
            contract_path,
            output_path=output_path,
            manifest_path=manifest_path,
        )


def test_freeze_requires_jsonl_source_and_output_suffixes(tmp_path: Path):
    contract_path, source_path, _, output_path, manifest_path = _fixture(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    renamed_source = source_path.with_suffix(".txt")
    source_path.rename(renamed_source)
    contract["sources"][0]["path"] = renamed_source.name
    _write_json(contract_path, contract)
    with pytest.raises(ValueError, match="JSONL or NDJSON"):
        freeze_evaluation_denylist_suite(
            contract_path,
            output_path=output_path,
            manifest_path=manifest_path,
        )

    contract_path, _, _, output_path, manifest_path = _fixture(tmp_path / "output")
    with pytest.raises(ValueError, match=r"\.jsonl or \.ndjson"):
        freeze_evaluation_denylist_suite(
            contract_path,
            output_path=output_path.with_suffix(".txt"),
            manifest_path=manifest_path,
        )


def test_freeze_never_clobbers_drifted_outputs(tmp_path: Path):
    contract_path, _, _, output_path, manifest_path = _fixture(tmp_path)
    output_path.write_text("user-owned output\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing to overwrite drifted"):
        freeze_evaluation_denylist_suite(
            contract_path,
            output_path=output_path,
            manifest_path=manifest_path,
        )
    assert output_path.read_text(encoding="utf-8") == "user-owned output\n"
    assert not manifest_path.exists()

    output_path.unlink()
    freeze_evaluation_denylist_suite(
        contract_path,
        output_path=output_path,
        manifest_path=manifest_path,
    )
    original_output = output_path.read_bytes()
    manifest_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing to overwrite drifted"):
        freeze_evaluation_denylist_suite(
            contract_path,
            output_path=output_path,
            manifest_path=manifest_path,
        )
    assert output_path.read_bytes() == original_output
    assert manifest_path.read_text(encoding="utf-8") == "{}\n"


def test_cli_freezes_the_contract_and_prints_the_manifest(tmp_path: Path):
    contract_path, _, _, output_path, manifest_path = _fixture(tmp_path)
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "freeze_evaluation_denylist_suite.py"),
            str(contract_path),
            "--output",
            str(output_path),
            "--manifest",
            str(manifest_path),
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
