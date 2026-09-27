import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from localagent.data.evaluation_denylist_manifest import (
    MANIFEST_KIND,
    build_evaluation_denylist_manifest,
    verify_evaluation_denylist_manifest,
)
from localagent.data.evaluation_denylist_suite import (
    CONTRACT_KIND,
    freeze_evaluation_denylist_suite,
)


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


def _identity(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _freeze_suite(
    directory: Path,
    *,
    name: str,
    prompt: str,
    benchmark_plan_suite_names: tuple[str, ...] | None = None,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True)
    source_path = directory / "adapter-prompts.jsonl"
    adapter_audit_path = directory / "adapter-audit.json"
    benchmark_plan_path = directory / "benchmark-plan.yaml"
    evidence_path = directory / "license.txt"
    contract_path = directory / "contract.json"
    output_path = directory / "prompts.jsonl"
    provenance_path = directory / "suite.provenance.json"

    rows = [{"source_case_id": f"{name}-case", "prompt": prompt}]
    source_path.write_bytes(b"".join(_canonical_bytes(row) for row in rows))
    source_identity = _identity(source_path)
    plan_suite_names = (
        (name,)
        if benchmark_plan_suite_names is None
        else tuple(sorted(benchmark_plan_suite_names))
    )
    if name not in plan_suite_names:
        raise ValueError("fixture suite must be present in its benchmark plan")
    benchmark_plan_path.write_text(
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
                    "list_manifest_kind": MANIFEST_KIND,
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
    adapter_audit = {
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
                "records": len(rows),
            },
        },
        "kind": "localagent_fixture_prompt_adapter_audit",
        "output": {
            "bytes": source_identity["bytes"],
            "sha256": source_identity["sha256"],
        },
        "purpose": "prompt_only_corpus_decontamination",
        "revision": "fixture-release-2026.1",
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
        "Fixture license evidence pinned to fixture-release-2026.1.\n",
        encoding="utf-8",
    )
    _write_json(
        contract_path,
        {
            "kind": CONTRACT_KIND,
            "schema_version": 1,
            "suite": {
                "name": name,
                "benchmark": f"Fixture-{name}",
                "revision": "fixture-release-2026.1",
                "split": "heldout",
                "adapter": {
                    "name": "fixture-prompt-adapter",
                    "version": "1.0.0",
                },
            },
            "benchmark_plan": {
                **_identity(benchmark_plan_path),
                "name": "benchmark-plan",
            },
            "sources": [
                {
                    **source_identity,
                    "name": "prompt-export",
                    "records": len(rows),
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
        },
    )
    freeze_evaluation_denylist_suite(
        contract_path,
        output_path=output_path,
        manifest_path=provenance_path,
    )
    return output_path, provenance_path


def _rewrite_list_self_hash(path: Path, manifest: dict[str, object]) -> None:
    without_hash = dict(manifest)
    without_hash.pop("manifest_self_sha256", None)
    manifest["manifest_self_sha256"] = hashlib.sha256(
        _canonical_bytes(without_hash)
    ).hexdigest()
    path.write_bytes(_canonical_bytes(manifest))


def _rewrite_suite_self_hash(path: Path, manifest: dict[str, object]) -> None:
    without_hash = dict(manifest)
    without_hash.pop("manifest_self_sha256", None)
    manifest["manifest_self_sha256"] = hashlib.sha256(
        _canonical_bytes(without_hash)
    ).hexdigest()
    path.write_bytes(_canonical_bytes(manifest))


def test_build_is_deterministic_portable_and_fully_reverifiable(tmp_path: Path):
    alpha_output, alpha_provenance = _freeze_suite(
        tmp_path / "private" / "alpha",
        name="alpha",
        prompt="Open the alpha settings page.",
        benchmark_plan_suite_names=("alpha", "zeta"),
    )
    zeta_output, zeta_provenance = _freeze_suite(
        tmp_path / "private" / "zeta",
        name="zeta",
        prompt="Call the zeta weather tool.",
        benchmark_plan_suite_names=("alpha", "zeta"),
    )
    output_path = tmp_path / "bundle" / "manifests" / "denylist.json"

    first = build_evaluation_denylist_manifest(
        [zeta_provenance, alpha_provenance],
        output_path=output_path,
    )
    first_bytes = output_path.read_bytes()
    second = build_evaluation_denylist_manifest(
        [alpha_provenance, zeta_provenance],
        output_path=output_path,
    )

    assert second == first
    assert output_path.read_bytes() == first_bytes
    assert first["kind"] == MANIFEST_KIND
    assert first["schema_version"] == 1
    assert first["required_suites"] == ["alpha", "zeta"]
    assert [row["name"] for row in first["suites"]] == ["alpha", "zeta"]
    expected = {
        "alpha": (alpha_output.resolve(), alpha_provenance.resolve()),
        "zeta": (zeta_output.resolve(), zeta_provenance.resolve()),
    }
    for row in first["suites"]:
        assert not Path(row["path"]).is_absolute()
        assert not Path(row["provenance"]["path"]).is_absolute()
        assert (output_path.parent / row["path"]).resolve() == expected[row["name"]][0]
        assert (
            output_path.parent / row["provenance"]["path"]
        ).resolve() == expected[row["name"]][1]

    without_hash = dict(first)
    self_hash = without_hash.pop("manifest_self_sha256")
    assert hashlib.sha256(_canonical_bytes(without_hash)).hexdigest() == self_hash
    artifacts, identity = verify_evaluation_denylist_manifest(output_path)
    assert [artifact["name"] for artifact in artifacts] == ["alpha", "zeta"]
    assert all("provenance" in artifact for artifact in artifacts)
    assert identity["manifest_self_sha256"] == self_hash
    assert json.loads(alpha_provenance.read_bytes())["raw_artifacts"] == []


def test_build_requires_the_canonical_raw_artifacts_field(tmp_path: Path):
    _, provenance = _freeze_suite(
        tmp_path / "suite",
        name="raw-field",
        prompt="Require the raw artifact field.",
    )
    suite_manifest = json.loads(provenance.read_bytes())
    suite_manifest.pop("raw_artifacts")
    _rewrite_suite_self_hash(provenance, suite_manifest)

    with pytest.raises(ValueError, match=r"missing=\['raw_artifacts'\]"):
        build_evaluation_denylist_manifest(
            [provenance],
            output_path=tmp_path / "list.json",
        )


@pytest.mark.parametrize(
    ("raw_artifacts", "message"),
    [
        (
            [
                {
                    "bytes": 1,
                    "name": "raw-a",
                    "role": "role-a",
                    "sha256": "0" * 64,
                    "unexpected": True,
                }
            ],
            "keys mismatch",
        ),
        (
            [
                {
                    "bytes": 1,
                    "name": "raw-a",
                    "role": "role-a",
                    "sha256": "0" * 64,
                },
                {
                    "bytes": 1,
                    "name": "raw-b",
                    "role": "role-a",
                    "sha256": "1" * 64,
                },
            ],
            "duplicate artifact roles",
        ),
        (
            [
                {
                    "bytes": 1,
                    "name": "raw-a",
                    "role": "role-b",
                    "sha256": "0" * 64,
                },
                {
                    "bytes": 1,
                    "name": "raw-b",
                    "role": "role-a",
                    "sha256": "1" * 64,
                },
            ],
            "sorted by role",
        ),
        (
            [
                {
                    "bytes": 1,
                    "name": "raw-a",
                    "role": "role-a",
                    "sha256": "0" * 64,
                },
                {
                    "bytes": 1,
                    "name": "raw-a",
                    "role": "role-b",
                    "sha256": "1" * 64,
                },
            ],
            "duplicate artifact names",
        ),
    ],
)
def test_build_rejects_malformed_raw_artifact_identities(
    tmp_path: Path,
    raw_artifacts: list[dict[str, object]],
    message: str,
):
    _, provenance = _freeze_suite(
        tmp_path / "suite",
        name="raw-schema",
        prompt="Validate raw artifact identities.",
    )
    suite_manifest = json.loads(provenance.read_bytes())
    suite_manifest["raw_artifacts"] = raw_artifacts
    _rewrite_suite_self_hash(provenance, suite_manifest)

    with pytest.raises(ValueError, match=message):
        build_evaluation_denylist_manifest(
            [provenance],
            output_path=tmp_path / "list.json",
        )


def test_build_reverifies_structurally_valid_raw_artifacts_transitively(
    tmp_path: Path,
):
    _, provenance = _freeze_suite(
        tmp_path / "suite",
        name="raw-replay",
        prompt="Reverify raw artifact semantics.",
    )
    suite_manifest = json.loads(provenance.read_bytes())
    suite_manifest["raw_artifacts"] = [
        {
            "bytes": 1,
            "name": "raw-a",
            "role": "role-a",
            "sha256": "0" * 64,
        }
    ]
    _rewrite_suite_self_hash(provenance, suite_manifest)

    with pytest.raises(RuntimeError, match="drifted frozen artifact"):
        build_evaluation_denylist_manifest(
            [provenance],
            output_path=tmp_path / "list.json",
        )


@pytest.mark.parametrize("duplicate", ["path", "name"])
def test_build_rejects_duplicate_suite_paths_or_provenance_names(
    tmp_path: Path,
    duplicate: str,
):
    _, first_provenance = _freeze_suite(
        tmp_path / "first",
        name="same-name",
        prompt="First prompt.",
    )
    if duplicate == "path":
        inputs = [first_provenance, first_provenance]
        message = "same suite provenance path"
    else:
        _, second_provenance = _freeze_suite(
            tmp_path / "second",
            name="same-name",
            prompt="Second prompt.",
        )
        inputs = [first_provenance, second_provenance]
        message = "duplicate suite names"
    with pytest.raises(ValueError, match=message):
        build_evaluation_denylist_manifest(
            inputs,
            output_path=tmp_path / "list.json",
        )


def test_build_rejects_suite_provenance_or_output_drift(tmp_path: Path):
    output, provenance = _freeze_suite(
        tmp_path / "suite",
        name="drift-check",
        prompt="Original frozen prompt.",
    )
    original_provenance = provenance.read_bytes()
    suite_manifest = json.loads(original_provenance)
    suite_manifest["isolation"]["prompt_only"] = False
    provenance.write_bytes(_canonical_bytes(suite_manifest))
    with pytest.raises(ValueError, match="manifest_self_sha256 mismatch"):
        build_evaluation_denylist_manifest(
            [provenance],
            output_path=tmp_path / "list.json",
        )

    provenance.write_bytes(original_provenance)
    output.write_text(
        json.dumps(
            {
                "prompt": "Tampered prompt.",
                "source_case_id_sha256": "0" * 64,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="byte-size disagrees"):
        build_evaluation_denylist_manifest(
            [provenance],
            output_path=tmp_path / "list.json",
        )


def test_build_rejects_suites_bound_to_different_benchmark_plans(tmp_path: Path):
    _, alpha_provenance = _freeze_suite(
        tmp_path / "alpha",
        name="alpha",
        prompt="Alpha prompt.",
    )
    _, zeta_provenance = _freeze_suite(
        tmp_path / "zeta",
        name="zeta",
        prompt="Zeta prompt.",
    )

    with pytest.raises(ValueError, match="same benchmark plan identity"):
        build_evaluation_denylist_manifest(
            [alpha_provenance, zeta_provenance],
            output_path=tmp_path / "list.json",
        )


def test_verify_rejects_list_self_hash_and_nested_provenance_drift(tmp_path: Path):
    _, provenance = _freeze_suite(
        tmp_path / "suite",
        name="bound-suite",
        prompt="One bound prompt.",
    )
    output_path = tmp_path / "list.json"
    build_evaluation_denylist_manifest(
        [provenance],
        output_path=output_path,
    )

    original = output_path.read_bytes()
    manifest = json.loads(original)
    manifest["required_suites"] = []
    output_path.write_bytes(_canonical_bytes(manifest))
    with pytest.raises(ValueError, match="manifest_self_sha256 mismatch"):
        verify_evaluation_denylist_manifest(output_path)

    output_path.write_bytes(original)
    manifest = json.loads(original)
    manifest["suites"][0]["provenance"]["sha256"] = "0" * 64
    _rewrite_list_self_hash(output_path, manifest)
    with pytest.raises(ValueError, match="provenance SHA-256 mismatch"):
        verify_evaluation_denylist_manifest(output_path)


def test_verify_rejects_extra_keys_even_with_a_valid_list_self_hash(tmp_path: Path):
    _, provenance = _freeze_suite(
        tmp_path / "suite",
        name="strict-suite",
        prompt="Strict prompt.",
    )
    output_path = tmp_path / "list.json"
    build_evaluation_denylist_manifest(
        [provenance],
        output_path=output_path,
    )
    manifest = json.loads(output_path.read_bytes())
    manifest["unexpected"] = True
    _rewrite_list_self_hash(output_path, manifest)
    with pytest.raises(ValueError, match="keys mismatch"):
        verify_evaluation_denylist_manifest(output_path)


def test_verify_requires_provenance_and_required_suites_exactly_all_inputs(
    tmp_path: Path,
):
    _, provenance = _freeze_suite(
        tmp_path / "suite",
        name="required-suite",
        prompt="Required prompt.",
    )
    output_path = tmp_path / "list.json"
    build_evaluation_denylist_manifest(
        [provenance],
        output_path=output_path,
    )
    original = output_path.read_bytes()

    manifest = json.loads(original)
    manifest["suites"][0].pop("provenance")
    _rewrite_list_self_hash(output_path, manifest)
    with pytest.raises(ValueError, match="keys mismatch"):
        verify_evaluation_denylist_manifest(output_path)

    manifest = json.loads(original)
    manifest["required_suites"] = []
    _rewrite_list_self_hash(output_path, manifest)
    with pytest.raises(ValueError, match="must exactly equal"):
        verify_evaluation_denylist_manifest(output_path)


def test_build_never_clobbers_a_drifted_destination(tmp_path: Path):
    _, provenance = _freeze_suite(
        tmp_path / "suite",
        name="no-clobber",
        prompt="Do not clobber this output.",
    )
    output_path = tmp_path / "list.json"
    output_path.write_text("user-owned bytes\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing to overwrite drifted"):
        build_evaluation_denylist_manifest(
            [provenance],
            output_path=output_path,
        )
    assert output_path.read_text(encoding="utf-8") == "user-owned bytes\n"


def test_build_enforces_a_hard_suite_count_before_reading_inputs(tmp_path: Path):
    with pytest.raises(ValueError, match="hard cap 128"):
        build_evaluation_denylist_manifest(
            [tmp_path / f"missing-{index}.json" for index in range(129)],
            output_path=tmp_path / "list.json",
        )


def test_build_rejects_an_unportable_generated_relative_path(tmp_path: Path):
    _, provenance = _freeze_suite(
        tmp_path / "has\\backslash",
        name="portable-suite",
        prompt="Portable prompt.",
    )
    output_path = tmp_path / "list.json"
    with pytest.raises(ValueError, match="portable relative POSIX path"):
        build_evaluation_denylist_manifest(
            [provenance],
            output_path=output_path,
        )
    assert not output_path.exists()


def test_cli_builds_and_prints_the_manifest(tmp_path: Path):
    _, provenance = _freeze_suite(
        tmp_path / "suite",
        name="cli-suite",
        prompt="CLI prompt.",
    )
    output_path = tmp_path / "list.json"
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "build_evaluation_denylist_manifest.py"),
            "--suite-provenance",
            str(provenance),
            "--out",
            str(output_path),
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == json.loads(output_path.read_bytes())
