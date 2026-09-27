from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import openlocalagent.data.adapters.bfcl_prompts as bfcl_prompts
from openlocalagent.data.adapters.bfcl_prompts import (
    BFCL_PROMPT_ADAPTER,
    BFCL_SOURCE_MANIFEST_KIND,
    BFCLPromptLimits,
    PRODUCTION_BFCL_CATEGORIES,
    PRODUCTION_BFCL_REVISION,
    export_bfcl_prompt_rows,
)
from prompt_freezer_helpers import freeze_production_adapter_output


REVISION = "a" * 40


def _canonical_line(value: object) -> bytes:
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


def _function(name: str) -> dict:
    return {
        "name": name,
        "description": f"Call {name}.",
        "parameters": {
            "type": "dict",
            "properties": {"value": {"description": "A value.", "type": "string"}},
            "required": ["value"],
        },
    }


def _case(case_id: str, prompt: str, *, function_name: str = "lookup") -> dict:
    return {
        "id": case_id,
        "question": [[{"role": "user", "content": prompt}]],
        "function": [_function(function_name)],
    }


def _write_source(
    path: Path,
    records: list[dict],
    *,
    terminal_newline: bool = True,
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(_canonical_line(record) for record in records)
    if not terminal_newline:
        payload = payload.removesuffix(b"\n")
    path.write_bytes(payload)
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_manifest(
    path: Path,
    sources: list[tuple[str, Path, dict[str, object]]],
    *,
    revision: str = REVISION,
) -> Path:
    value = {
        "kind": BFCL_SOURCE_MANIFEST_KIND,
        "schema_version": 1,
        "benchmark": "bfcl-v4",
        "revision": revision,
        "sources": [
            {
                "category": category,
                "path": str(source.relative_to(path.parent)),
                **identity,
            }
            for category, source, identity in sources
        ],
    }
    path.write_bytes(_canonical_line(value))
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_bfcl_export_is_deterministic_canonical_and_prompt_only(tmp_path: Path) -> None:
    alpha = tmp_path / "BFCL_v4_alpha.json"
    beta = tmp_path / "BFCL_v4_beta.json"
    alpha_identity = _write_source(
        alpha,
        [
            _case("alpha_10", "Later alpha prompt.", function_name="zeta"),
            {
                "id": "alpha_02",
                "question": [
                    [
                        {"role": "system", "content": "System-visible instruction."},
                        {"role": "user", "content": "Earlier alpha prompt."},
                    ]
                ],
                "function": [_function("alpha")],
            },
        ],
        terminal_newline=False,
    )
    beta_identity = _write_source(
        beta,
        [_case("beta_00", "Beta prompt.", function_name="beta")],
    )
    manifest = _write_manifest(
        tmp_path / "bfcl-sources.json",
        [
            ("beta", beta, beta_identity),
            ("alpha", alpha, alpha_identity),
        ],
    )
    output = tmp_path / "prompts.jsonl"
    audit_path = tmp_path / "audit.json"

    audit = export_bfcl_prompt_rows(manifest, output, audit_path)
    rows = _read_jsonl(output)

    assert [row["source_case_id"] for row in rows] == [
        "bfcl:alpha:alpha_02:question:0000:0000",
        "bfcl:alpha:alpha_02:question:0000:0001",
        "bfcl:alpha:alpha_02:function:0000",
        "bfcl:alpha:alpha_10:question:0000:0000",
        "bfcl:alpha:alpha_10:function:0000",
        "bfcl:beta:beta_00:question:0000:0000",
        "bfcl:beta:beta_00:function:0000",
    ]
    assert all(set(row) == {"source_case_id", "prompt"} for row in rows)
    assert rows[2]["prompt"] == json.dumps(
        _function("alpha"),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert audit["adapter"] == BFCL_PROMPT_ADAPTER
    assert audit["selection"]["caller_declared_categories"] == ["alpha", "beta"]
    assert audit["selection"]["question_prompt_rows"] == 4
    assert audit["selection"]["function_spec_prompt_rows"] == 3
    assert audit["output"]["rows"] == 7

    original_output = output.read_bytes()
    original_audit = audit_path.read_bytes()
    assert export_bfcl_prompt_rows(manifest, output, audit_path) == audit
    assert output.read_bytes() == original_output
    assert audit_path.read_bytes() == original_audit

    output.write_bytes(b'{"prompt":"drift","source_case_id":"drift"}\n')
    with pytest.raises(RuntimeError, match="drifted derived artifact"):
        export_bfcl_prompt_rows(manifest, output, audit_path)


def test_bfcl_production_audit_freezes_through_generic_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources: list[tuple[str, Path, dict[str, object]]] = []
    for category in sorted(PRODUCTION_BFCL_CATEGORIES):
        source = tmp_path / f"BFCL_v4_{category}.json"
        identity = _write_source(
            source,
            [_case(f"{category}_00", f"Prompt for {category}.")],
            terminal_newline=False,
        )
        sources.append((category, source, identity))
    monkeypatch.setattr(
        bfcl_prompts,
        "PRODUCTION_BFCL_SOURCE_IDENTITIES",
        {
            category: (
                int(identity["bytes"]),
                1,
                str(identity["sha256"]),
            )
            for category, _, identity in sources
        },
    )
    manifest = _write_manifest(
        tmp_path / "bfcl-production-sources.json",
        list(reversed(sources)),
        revision=PRODUCTION_BFCL_REVISION,
    )
    output = tmp_path / "bfcl-production-prompts.jsonl"
    audit_path = tmp_path / "bfcl-production-audit.json"

    audit = export_bfcl_prompt_rows(manifest, output, audit_path)
    assert audit["mode"] == "production"
    assert audit["split"] == (
        "multiple+parallel+parallel_multiple+simple_python"
    )

    frozen = freeze_production_adapter_output(
        tmp_path,
        suite_name="bfcl",
        prompt_path=output,
        audit_path=audit_path,
    )
    assert frozen["suite"]["name"] == "bfcl"
    assert frozen["sources"][0]["records"] == audit["output"]["rows"]


def test_bfcl_generic_freezer_rejects_raw_source_attestation_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources: list[tuple[str, Path, dict[str, object]]] = []
    for category in sorted(PRODUCTION_BFCL_CATEGORIES):
        source = tmp_path / f"BFCL_v4_{category}.json"
        identity = _write_source(
            source,
            [_case(f"{category}_00", f"Prompt for {category}.")],
            terminal_newline=False,
        )
        sources.append((category, source, identity))
    monkeypatch.setattr(
        bfcl_prompts,
        "PRODUCTION_BFCL_SOURCE_IDENTITIES",
        {
            category: (
                int(identity["bytes"]),
                1,
                str(identity["sha256"]),
            )
            for category, _, identity in sources
        },
    )
    manifest = _write_manifest(
        tmp_path / "bfcl-production-sources.json",
        sources,
        revision=PRODUCTION_BFCL_REVISION,
    )
    output = tmp_path / "bfcl-production-prompts.jsonl"
    original_audit = tmp_path / "bfcl-production-audit.json"
    export_bfcl_prompt_rows(manifest, output, original_audit)

    tampered_dir = tmp_path / "tampered"
    tampered_dir.mkdir()
    tampered_output = tampered_dir / output.name
    tampered_output.write_bytes(output.read_bytes())
    tampered_audit = tampered_dir / original_audit.name
    audit = json.loads(original_audit.read_text(encoding="utf-8"))
    audit["sources"][0]["sha256"] = "0" * 64
    tampered_audit.write_bytes(_canonical_line(audit))

    with pytest.raises(ValueError, match="raw-source reexport"):
        freeze_production_adapter_output(
            tampered_dir,
            suite_name="bfcl",
            prompt_path=tampered_output,
            audit_path=tampered_audit,
            plan_attestation_path=original_audit,
        )


def test_bfcl_production_rejects_non_authoritative_source_identities(
    tmp_path: Path,
) -> None:
    sources: list[tuple[str, Path, dict[str, object]]] = []
    for category in sorted(PRODUCTION_BFCL_CATEGORIES):
        source = tmp_path / f"BFCL_v4_{category}.json"
        identity = _write_source(
            source,
            [_case(f"{category}_00", f"Fabricated prompt for {category}.")],
            terminal_newline=False,
        )
        sources.append((category, source, identity))
    manifest = _write_manifest(
        tmp_path / "bfcl-untrusted-sources.json",
        sources,
        revision=PRODUCTION_BFCL_REVISION,
    )

    with pytest.raises(ValueError, match="authoritative pinned"):
        export_bfcl_prompt_rows(
            manifest,
            tmp_path / "bfcl-untrusted-prompts.jsonl",
            tmp_path / "bfcl-untrusted-audit.json",
        )


def test_bfcl_export_rejects_gold_paths_and_gold_schema_fields(tmp_path: Path) -> None:
    gold_path = tmp_path / "possible_answer" / "BFCL_v4_alpha.json"
    gold_identity = _write_source(
        gold_path,
        [
            {
                "id": "alpha_00",
                "ground_truth": [{"lookup": {"value": ["secret"]}}],
            }
        ],
    )
    gold_manifest = _write_manifest(
        tmp_path / "gold-sources.json",
        [("alpha", gold_path, gold_identity)],
    )
    with pytest.raises(ValueError, match="possible_answer/gold"):
        export_bfcl_prompt_rows(
            gold_manifest,
            tmp_path / "gold-output.jsonl",
            tmp_path / "gold-audit.json",
        )

    drifted_path = tmp_path / "BFCL_v4_alpha.json"
    record = _case("alpha_00", "Prompt.")
    record["ground_truth"] = [{"lookup": {"value": ["secret"]}}]
    drifted_identity = _write_source(drifted_path, [record])
    drifted_manifest = _write_manifest(
        tmp_path / "drifted-sources.json",
        [("alpha", drifted_path, drifted_identity)],
    )
    with pytest.raises(ValueError, match="schema drift.*ground_truth"):
        export_bfcl_prompt_rows(
            drifted_manifest,
            tmp_path / "drifted-output.jsonl",
            tmp_path / "drifted-audit.json",
        )


def test_bfcl_export_rejects_duplicate_ids_identity_drift_and_caps(tmp_path: Path) -> None:
    alpha = tmp_path / "BFCL_v4_alpha.json"
    beta = tmp_path / "BFCL_v4_beta.json"
    alpha_identity = _write_source(
        alpha,
        [_case("duplicate_00", "One."), _case("alpha_01", "Two.")],
    )
    beta_identity = _write_source(beta, [_case("duplicate_00", "Three.")])
    manifest = _write_manifest(
        tmp_path / "duplicate-sources.json",
        [("alpha", alpha, alpha_identity), ("beta", beta, beta_identity)],
    )
    with pytest.raises(ValueError, match="duplicate BFCL id"):
        export_bfcl_prompt_rows(
            manifest,
            tmp_path / "duplicate-output.jsonl",
            tmp_path / "duplicate-audit.json",
        )

    single_manifest = _write_manifest(
        tmp_path / "single-sources.json",
        [("alpha", alpha, alpha_identity)],
    )
    with pytest.raises(ValueError, match="source limit"):
        export_bfcl_prompt_rows(
            single_manifest,
            tmp_path / "capped-output.jsonl",
            tmp_path / "capped-audit.json",
            limits=BFCLPromptLimits(max_source_bytes=alpha.stat().st_size - 1),
        )
    with pytest.raises(ValueError, match="row limit"):
        export_bfcl_prompt_rows(
            single_manifest,
            tmp_path / "row-output.jsonl",
            tmp_path / "row-audit.json",
            limits=BFCLPromptLimits(max_source_rows=1),
        )

    identity_manifest_value = json.loads(single_manifest.read_text(encoding="utf-8"))
    identity_manifest_value["sources"][0]["sha256"] = "0" * 64
    single_manifest.write_bytes(_canonical_line(identity_manifest_value))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        export_bfcl_prompt_rows(
            single_manifest,
            tmp_path / "identity-output.jsonl",
            tmp_path / "identity-audit.json",
        )


def test_bfcl_export_fails_closed_on_nested_schema_drift(tmp_path: Path) -> None:
    source = tmp_path / "BFCL_v4_alpha.json"
    record = _case("alpha_00", "Prompt.")
    record["question"][0][0]["metadata"] = {"unexpected": True}
    identity = _write_source(source, [record])
    manifest = _write_manifest(
        tmp_path / "sources.json",
        [("alpha", source, identity)],
    )

    with pytest.raises(ValueError, match="question.*schema drift"):
        export_bfcl_prompt_rows(
            manifest,
            tmp_path / "output.jsonl",
            tmp_path / "audit.json",
        )
    assert not (tmp_path / "output.jsonl").exists()
    assert not (tmp_path / "audit.json").exists()
