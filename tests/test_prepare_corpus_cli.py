import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from openlocalagent.data.decontam.evaluation_denylist_manifest import (
    build_evaluation_denylist_manifest,
)
from openlocalagent.data.decontam.evaluation_denylist_suite import (
    CONTRACT_KIND,
    freeze_evaluation_denylist_suite,
)
from openlocalagent.data.corpus.pretrain_corpus import CorpusDocument


_SCRIPT = Path(__file__).parents[1] / "scripts" / "build_corpus.py"
_SPEC = importlib.util.spec_from_file_location("prepare_corpus_cli", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_select_mixture_sources = _MODULE._select_mixture_sources
_evaluation_denylist_inputs = _MODULE._evaluation_denylist_inputs
_merge_required_suites = _MODULE._merge_required_suites
_source_manifest_inputs = _MODULE._source_manifest_inputs


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


def _identity(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _freeze_denylist_suite(
    directory: Path,
    *,
    name: str,
    prompt: str,
    benchmark_plan_suite_names: tuple[str, ...] | None = None,
) -> tuple[Path, Path]:
    directory.mkdir()
    source = directory / "source.jsonl"
    adapter_audit = directory / "adapter-audit.json"
    benchmark_plan = directory / "benchmark-plan.yaml"
    evidence = directory / "license.txt"
    contract = directory / "contract.json"
    output = directory / "prompts.jsonl"
    provenance = directory / "prompts.provenance.json"
    source.write_bytes(
        _canonical_bytes(
            {
                "source_case_id": f"{name}-case",
                "prompt": prompt,
            }
        )
    )
    source_identity = _identity(source)
    plan_suite_names = (
        (name,)
        if benchmark_plan_suite_names is None
        else tuple(sorted(benchmark_plan_suite_names))
    )
    if name not in plan_suite_names:
        raise ValueError("fixture suite must be present in its benchmark plan")
    benchmark_plan.write_text(
        yaml.safe_dump(
            {
                "kind": "localagent_external_benchmark_plan",
                "schema_version": 1,
                "purpose": "pretraining_prompt_only_decontamination",
                "forbid_gold_in_prompt_exports": True,
                "prompt_freeze": {
                    "suite_contract_kind": CONTRACT_KIND,
                    "suite_provenance_kind": (
                        "localagent_evaluation_denylist_suite_provenance"
                    ),
                    "list_manifest_kind": (
                        "localagent_evaluation_denylist_manifest"
                    ),
                    "schema_version": 1,
                    "require_adapter_audit_binding": True,
                    "require_license_evidence_binding": True,
                    "require_prompt_only_isolation": True,
                    "external_manifest_suites": list(plan_suite_names),
                    "config_hash_pinned_direct_suites": [],
                },
                "suites": {
                    suite_name: {
                        "benchmark": f"Fixture-{suite_name}",
                        "revision": "fixture-release-2026.1",
                        "adapter": "1.0.0",
                        "prompt_freeze_split": "heldout",
                        "v1_evaluation_scope": "derived",
                    }
                    for suite_name in plan_suite_names
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    adapter_audit.write_bytes(
        _canonical_bytes(
            {
                "adapter": "1.0.0",
                "freeze_binding": {
                    "adapter": "1.0.0",
                    "benchmark": f"Fixture-{name}",
                    "mode": "production",
                    "revision": "fixture-release-2026.1",
                    "split": "heldout",
                    "prompt_only": True,
                    "contains_current_step_labels": False,
                    "output": {
                        "bytes": source_identity["bytes"],
                        "sha256": source_identity["sha256"],
                        "records": 1,
                    },
                },
                "kind": "localagent_prepare_corpus_fixture_adapter_audit",
                "output": {
                    "bytes": source_identity["bytes"],
                    "sha256": source_identity["sha256"],
                },
                "purpose": "prompt_only_corpus_decontamination",
                "revision": "fixture-release-2026.1",
                "schema_version": 1,
            }
        )
    )
    evidence.write_text("Fixture license evidence.\n", encoding="utf-8")
    contract.write_bytes(
        _canonical_bytes(
            {
                "kind": CONTRACT_KIND,
                "schema_version": 1,
                "suite": {
                    "name": name,
                    "benchmark": f"Fixture-{name}",
                    "revision": "fixture-release-2026.1",
                    "split": "heldout",
                    "adapter": {
                        "name": "prepare-corpus-fixture-adapter",
                        "version": "1.0.0",
                    },
                },
                "benchmark_plan": {
                    **_identity(benchmark_plan),
                    "name": "benchmark-plan",
                },
                "sources": [
                    {
                        **source_identity,
                        "name": "prompt-export",
                        "records": 1,
                    }
                ],
                "adapter_provenance": [
                    {
                        **_identity(adapter_audit),
                        "name": "adapter-audit",
                    }
                ],
                "license_evidence": [
                    {
                        **_identity(evidence),
                        "name": "license-evidence",
                    }
                ],
                "limits": {
                    "max_source_bytes": 100_000,
                    "max_benchmark_plan_bytes": 100_000,
                    "max_adapter_provenance_bytes": 100_000,
                    "max_license_evidence_bytes": 100_000,
                    "max_rows": 100,
                    "max_record_bytes": 100_000,
                },
            }
        )
    )
    freeze_evaluation_denylist_suite(
        contract,
        output_path=output,
        manifest_path=provenance,
    )
    return output, provenance


def test_select_mixture_sources_filters_only_declared_families():
    documents = [
        CorpusDocument("general", meta={"mixture_source": "general"}),
        CorpusDocument("code", meta={"mixture_source": "code"}),
        CorpusDocument("structured", meta={"mixture_source": "structured"}),
    ]
    selected = list(_select_mixture_sources(iter(documents), {"general", "structured"}))
    assert [document.text for document in selected] == ["general", "structured"]


def test_select_mixture_sources_is_inert_without_filter():
    documents = [CorpusDocument("one"), CorpusDocument("two")]
    assert list(_select_mixture_sources(iter(documents), set())) == documents


def test_evaluation_denylist_manifest_verifies_named_suite_artifacts(tmp_path):
    plan_suites = ("fixture-bfcl", "local-browser")
    bfcl, bfcl_provenance = _freeze_denylist_suite(
        tmp_path / "bfcl-suite",
        name="fixture-bfcl",
        prompt="Call the weather tool for Seoul.",
        benchmark_plan_suite_names=plan_suites,
    )
    local, local_provenance = _freeze_denylist_suite(
        tmp_path / "local-suite",
        name="local-browser",
        prompt="Select the submit button.",
        benchmark_plan_suite_names=plan_suites,
    )
    list_manifest = tmp_path / "denylist-manifest.json"
    build_evaluation_denylist_manifest(
        [local_provenance, bfcl_provenance],
        output_path=list_manifest,
    )

    paths, audit = _evaluation_denylist_inputs([], [str(list_manifest)])
    assert paths == [bfcl, local]
    assert [entry["name"] for entry in audit["inputs"]] == [
        "fixture-bfcl",
        "local-browser",
    ]
    assert audit["list_manifests"][0]["sha256"]
    assert audit["list_manifests"][0]["manifest_self_sha256"]
    assert all("provenance" in entry for entry in audit["inputs"])
    assert audit["inputs_sha256"]

    bfcl.write_text(json.dumps({"prompt": "Tampered benchmark prompt."}) + "\n")
    with pytest.raises(ValueError, match="suite 'fixture-bfcl'.*byte-size disagrees"):
        _evaluation_denylist_inputs([], [str(list_manifest)])


def test_source_corpus_policy_cannot_be_weakened_by_denylist_self_declaration(tmp_path):
    _bfcl, bfcl_provenance = _freeze_denylist_suite(
        tmp_path / "bfcl-suite",
        name="fixture-bfcl",
        prompt="Call the weather tool.",
    )
    list_manifest = tmp_path / "denylist-manifest.json"
    build_evaluation_denylist_manifest(
        [bfcl_provenance],
        output_path=list_manifest,
    )
    corpus_config = tmp_path / "pretrain-paper.yaml"
    corpus_config.write_text(
        "evaluation_decontamination:\n"
        "  manifest_kind: localagent_evaluation_denylist_manifest\n"
        "  manifest_schema_version: 1\n"
        "  required_suites:\n"
        "    - name: fixture-bfcl\n"
        "    - name: local-agent-eval\n"
        "      bytes: 123\n"
        f"      sha256: {'a' * 64}\n"
    )
    config_artifact = _MODULE._file_artifact(corpus_config)
    source_manifest = tmp_path / "download_manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "config": str(corpus_config),
                "config_bytes": config_artifact["bytes"],
                "config_sha256": config_artifact["sha256"],
            }
        )
    )

    _artifacts, policies = _source_manifest_inputs([str(source_manifest)])
    required, sources = _merge_required_suites(policies)
    with pytest.raises(ValueError, match="local-agent-eval"):
        _evaluation_denylist_inputs(
            [],
            [str(list_manifest)],
            required,
            sources,
        )


def test_corpus_policy_pinned_suite_hash_is_enforced_for_direct_input(tmp_path):
    suite = tmp_path / "agent-eval.jsonl"
    suite.write_text(json.dumps({"prompt": "Frozen agent evaluation prompt."}) + "\n")
    required = {
        "local-agent-eval": {
            "bytes": suite.stat().st_size,
            "sha256": "0" * 64,
        }
    }
    with pytest.raises(ValueError, match="policy SHA-256 mismatch"):
        _evaluation_denylist_inputs(
            [f"local-agent-eval={suite}"],
            [],
            required,
        )


def test_unpinned_required_suite_rejects_direct_cli_input(tmp_path):
    suite = tmp_path / "bfcl.jsonl"
    suite.write_text(json.dumps({"prompt": "Frozen BFCL prompt."}) + "\n")
    with pytest.raises(ValueError, match="eval-denylist-manifest"):
        _evaluation_denylist_inputs(
            [f"bfcl={suite}"],
            [],
            {"bfcl": {}},
        )


def test_prepare_cli_reuses_frozen_base_split_for_single_family(tmp_path, monkeypatch, capsys):
    raw_path = tmp_path / "mixture.jsonl"
    documents = [
        CorpusDocument(
            text=f"Family-bound frozen split document {index}. " * 6,
            source="unit-test",
            doc_id=f"family-doc-{index}",
            license="MIT",
            meta={"mixture_source": f"family-{index}"},
        )
        for index in range(8)
    ]
    raw_path.write_text(
        "".join(json.dumps(document.__dict__) + "\n" for document in documents),
        encoding="utf-8",
    )
    base_out = tmp_path / "base"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_corpus.py",
            str(raw_path),
            "--out",
            str(base_out),
            "--seq-len",
            "32",
            "--min-chars",
            "1",
            "--val-fraction",
            "0.25",
            "--seed",
            "47",
            "--no-near-dedup",
        ],
    )
    _MODULE.main()
    capsys.readouterr()
    base_manifest = json.loads((base_out / "manifest.json").read_text())
    assignment_rows = [
        json.loads(line)
        for line in (base_out / base_manifest["split_assignment"]["path"])
        .read_text()
        .splitlines()[1:]
    ]
    held_out_id = next(row["document_id"] for row in assignment_rows if row["split"] == "val")
    held_out_index = int(held_out_id.rsplit("-", 1)[1])

    derived_out = tmp_path / "derived"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_corpus.py",
            str(raw_path),
            "--out",
            str(derived_out),
            "--seq-len",
            "32",
            "--min-chars",
            "1",
            "--include-mixture-source",
            f"family-{held_out_index}",
            "--frozen-split-manifest",
            str(base_out / "manifest.json"),
            "--no-near-dedup",
        ],
    )
    _MODULE.main()
    capsys.readouterr()

    derived_manifest = json.loads((derived_out / "manifest.json").read_text())
    assert derived_manifest["splits"]["train"]["documents"] == 0
    assert derived_manifest["splits"]["val"]["documents"] == 1
    assert derived_manifest["corpus_audit"]["split_assignment"]["mode"] == "frozen"
    assert derived_manifest["seed"] == 47
    assert derived_manifest["val_fraction"] == 0.25
