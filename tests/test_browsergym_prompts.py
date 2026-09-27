from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

import openlocalagent.data.adapters.browsergym_capture as browsergym_capture
import openlocalagent.data.adapters.browsergym_prompts as browsergym_prompts
import prompt_freezer_helpers
from openlocalagent.data.adapters.browsergym_prompts import (
    BROWSERGYM_PROMPT_ADAPTER,
    BROWSERGYM_PROMPT_AUDIT_SCHEMA_VERSION,
    PRODUCTION_BROWSERGYM_REVISION,
    PRODUCTION_BROWSERGYM_VERSION,
    PRODUCTION_CAPTURE_RECEIPT_IDENTITY,
    PRODUCTION_CHROMIUM_REVISION,
    PRODUCTION_CHROMIUM_VERSION,
    PRODUCTION_EPISODES,
    PRODUCTION_FIXED_SEEDS,
    PRODUCTION_LOCAL_POLICY_EXCLUSIONS,
    PRODUCTION_MAX_STEPS,
    PRODUCTION_MINIWOB_REVISION,
    PRODUCTION_PLAYWRIGHT_VERSION,
    PRODUCTION_RUNTIME_MANIFEST_IDENTITY,
    PRODUCTION_SIMILARITY_GROUPS,
    PRODUCTION_TASK_GROUPS,
    PRODUCTION_TASK_VARIANTS,
    BrowserGymPromptLimits,
    export_browsergym_prompt_rows,
)
from prompt_freezer_helpers import freeze_production_adapter_output


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


def _source_pins() -> dict:
    return {
        "browsergym_revision": PRODUCTION_BROWSERGYM_REVISION,
        "browsergym_version": PRODUCTION_BROWSERGYM_VERSION,
        "miniwob_revision": PRODUCTION_MINIWOB_REVISION,
    }


def _runtime_pins() -> dict:
    return {
        "playwright_version": PRODUCTION_PLAYWRIGHT_VERSION,
        "chromium_revision": PRODUCTION_CHROMIUM_REVISION,
        "chromium_version": PRODUCTION_CHROMIUM_VERSION,
        "python_version": "3.11.9",
        "os": "linux",
        "architecture": "x86_64",
        "locale": "en-US",
        "timezone_id": "UTC",
        "headless": True,
        "viewport": {"width": 1280, "height": 720},
        "device_scale_factor": 1.0,
        "action_set": "highlevel-default-unused-reset-only",
        "observation_mode": "processed-dom-axtree-screenshot",
        "max_steps": PRODUCTION_MAX_STEPS,
        "playwright_operation_timeout_seconds": 30.0,
        "browser_executable": {"bytes": 123_456, "sha256": "c" * 64},
        "browser_installation": {"bytes": 789_012, "sha256": "d" * 64},
        "environment_manifest": dict(PRODUCTION_RUNTIME_MANIFEST_IDENTITY),
    }


def _row(task_name: str, seed: int, group: int, goal: str) -> dict:
    return {
        "task_name": task_name,
        "seed": seed,
        "goal": goal,
        "similarity_group": group,
        "split": "test",
        "source_pins": _source_pins(),
        "runtime_pins": _runtime_pins(),
    }


def _write_capture(path: Path, rows: list[dict]) -> tuple[int, str]:
    payload = b"".join(_canonical_line(row) for row in rows)
    path.write_bytes(payload)
    return len(payload), hashlib.sha256(payload).hexdigest()


def _export(
    capture: Path,
    output: Path,
    audit: Path,
    *,
    production: bool,
    receipt: Path | None = None,
    limits: BrowserGymPromptLimits = BrowserGymPromptLimits(),
) -> dict:
    size, sha256 = len(capture.read_bytes()), hashlib.sha256(capture.read_bytes()).hexdigest()
    return export_browsergym_prompt_rows(
        capture,
        output,
        audit,
        expected_capture_bytes=size,
        expected_capture_sha256=sha256,
        receipt_path=receipt,
        production=production,
        limits=limits,
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_receipt_stub(capture: Path, receipt_path: Path) -> dict:
    capture_payload = capture.read_bytes()
    without_hash = {
        "kind": PRODUCTION_CAPTURE_RECEIPT_IDENTITY["kind"],
        "schema_version": PRODUCTION_CAPTURE_RECEIPT_IDENTITY["schema_version"],
        "producer": PRODUCTION_CAPTURE_RECEIPT_IDENTITY["producer"],
        "capture": {
            "bytes": len(capture_payload),
            "sha256": hashlib.sha256(capture_payload).hexdigest(),
        },
    }
    receipt = {
        **without_hash,
        "receipt_self_sha256": hashlib.sha256(
            _canonical_line(without_hash)[:-1]
        ).hexdigest(),
    }
    receipt_path.write_bytes(_canonical_line(receipt))
    receipt_payload = receipt_path.read_bytes()
    return {
        "bytes": len(receipt_payload),
        "file": receipt_path.name,
        "kind": receipt["kind"],
        "producer": receipt["producer"],
        "receipt_self_sha256": receipt["receipt_self_sha256"],
        "schema_version": receipt["schema_version"],
        "sha256": hashlib.sha256(receipt_payload).hexdigest(),
    }


def _patch_receipt_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    def verify(_capture: Path, receipt: Path) -> dict:
        return json.loads(Path(receipt).read_text(encoding="utf-8"))

    monkeypatch.setattr(
        browsergym_capture,
        "verify_browsergym_capture_receipt",
        verify,
    )


def _freeze_browsergym_pins(
    monkeypatch: pytest.MonkeyPatch,
    *,
    capture: Path,
    receipt_identity: dict,
) -> None:
    payload = capture.read_bytes()
    monkeypatch.setattr(
        browsergym_prompts,
        "PRODUCTION_CAPTURE_FILE",
        capture.name,
    )
    monkeypatch.setattr(browsergym_prompts, "PRODUCTION_CAPTURE_BYTES", len(payload))
    monkeypatch.setattr(
        browsergym_prompts,
        "PRODUCTION_CAPTURE_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    monkeypatch.setattr(
        browsergym_prompts,
        "PRODUCTION_CAPTURE_RECEIPT_IDENTITY",
        {
            **receipt_identity,
            "status": "frozen_controlled_acquisition",
        },
    )


def test_browsergym_fixture_export_is_deterministic_and_strips_metadata(
    tmp_path: Path,
) -> None:
    capture = tmp_path / "capture.jsonl"
    _write_capture(
        capture,
        [
            _row("miniwob.z-task", 9, 2, "Goal Z9."),
            _row("miniwob.a-task", 7, 1, "Goal A7."),
            _row("miniwob.a-task", 3, 1, "Goal A3."),
        ],
    )
    output = tmp_path / "prompts.jsonl"
    audit_path = tmp_path / "audit.json"

    audit = _export(capture, output, audit_path, production=False)
    rows = _read_jsonl(output)

    assert rows == [
        {"source_case_id": "browsergym:miniwob.a-task:3", "prompt": "Goal A3."},
        {"source_case_id": "browsergym:miniwob.a-task:7", "prompt": "Goal A7."},
        {"source_case_id": "browsergym:miniwob.z-task:9", "prompt": "Goal Z9."},
    ]
    assert all(set(row) == {"source_case_id", "prompt"} for row in rows)
    assert not any(
        forbidden in output.read_text(encoding="utf-8")
        for forbidden in ("similarity_group", "runtime_pins", "source_pins", "task_name")
    )
    assert audit["adapter"] == BROWSERGYM_PROMPT_ADAPTER
    assert audit["schema_version"] == BROWSERGYM_PROMPT_AUDIT_SCHEMA_VERSION
    assert audit["mode"] == "fixture"
    assert audit["plan"]["task_variants"] == 2
    assert audit["plan"]["similarity_group_count"] == 2
    without_hash = dict(audit)
    audit_self_sha256 = without_hash.pop("audit_self_sha256")
    assert hashlib.sha256(_canonical_line(without_hash)[:-1]).hexdigest() == (
        audit_self_sha256
    )

    output_payload = output.read_bytes()
    audit_payload = audit_path.read_bytes()
    assert _export(capture, output, audit_path, production=False) == audit
    assert output.read_bytes() == output_payload
    assert audit_path.read_bytes() == audit_payload

    output.write_bytes(b'{"prompt":"drift","source_case_id":"drift"}\n')
    with pytest.raises(RuntimeError, match="drifted derived artifact"):
        _export(capture, output, audit_path, production=False)


def test_browsergym_export_rejects_schema_drift_duplicates_and_caps(tmp_path: Path) -> None:
    drifted = _row("miniwob.a-task", 3, 1, "Goal.")
    drifted["unexpected"] = "metadata"
    capture = tmp_path / "drifted.jsonl"
    _write_capture(capture, [drifted])
    with pytest.raises(ValueError, match="schema drift.*unexpected"):
        _export(
            capture,
            tmp_path / "drifted-output.jsonl",
            tmp_path / "drifted-audit.json",
            production=False,
        )

    duplicate = tmp_path / "duplicate.jsonl"
    repeated = _row("miniwob.a-task", 3, 1, "Goal.")
    _write_capture(duplicate, [repeated, repeated])
    with pytest.raises(ValueError, match="duplicate BrowserGym"):
        _export(
            duplicate,
            tmp_path / "duplicate-output.jsonl",
            tmp_path / "duplicate-audit.json",
            production=False,
        )

    capped = tmp_path / "capped.jsonl"
    _write_capture(capped, [_row("miniwob.a-task", 3, 1, "Long goal.")])
    with pytest.raises(ValueError, match="prompt limit"):
        _export(
            capped,
            tmp_path / "capped-output.jsonl",
            tmp_path / "capped-audit.json",
            production=False,
            limits=BrowserGymPromptLimits(max_prompt_bytes=4),
        )
    with pytest.raises(ValueError, match="capture limit"):
        _export(
            capped,
            tmp_path / "bytes-output.jsonl",
            tmp_path / "bytes-audit.json",
            production=False,
            limits=BrowserGymPromptLimits(max_capture_bytes=capped.stat().st_size - 1),
        )


def test_browsergym_production_requires_a_readable_verified_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = tmp_path / "capture.jsonl"
    size, sha256 = _write_capture(
        capture,
        [_row("miniwob.a-task", 3, 1, "Goal.")],
    )
    arguments = {
        "expected_capture_bytes": size,
        "expected_capture_sha256": sha256,
        "production": True,
    }
    with pytest.raises(ValueError, match="requires receipt_path"):
        export_browsergym_prompt_rows(
            capture,
            tmp_path / "missing-path-output.jsonl",
            tmp_path / "missing-path-audit.json",
            **arguments,
        )
    with pytest.raises(ValueError, match="receipt.*readable"):
        export_browsergym_prompt_rows(
            capture,
            tmp_path / "missing-file-output.jsonl",
            tmp_path / "missing-file-audit.json",
            receipt_path=tmp_path / "missing.receipt.json",
            **arguments,
        )

    receipt = tmp_path / "forged.receipt.json"
    receipt.write_bytes(b'{"forged":true}\n')
    called = False

    def reject_forgery(_capture: Path, _receipt: Path) -> dict:
        nonlocal called
        called = True
        raise ValueError("BrowserGym receipt self-hash mismatch")

    monkeypatch.setattr(
        browsergym_capture,
        "verify_browsergym_capture_receipt",
        reject_forgery,
    )
    with pytest.raises(ValueError, match="receipt self-hash mismatch"):
        export_browsergym_prompt_rows(
            capture,
            tmp_path / "forged-output.jsonl",
            tmp_path / "forged-audit.json",
            receipt_path=receipt,
            **arguments,
        )
    assert called is True


def test_browsergym_production_plan_requires_all_tasks_groups_and_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert len(PRODUCTION_TASK_GROUPS) == PRODUCTION_TASK_VARIANTS == 60
    assert len(set(PRODUCTION_TASK_GROUPS.values())) == PRODUCTION_SIMILARITY_GROUPS == 41
    assert len(PRODUCTION_TASK_GROUPS) * len(PRODUCTION_FIXED_SEEDS) == PRODUCTION_EPISODES

    rows = [
        _row(
            task_name,
            seed,
            group,
            f"Complete {task_name} deterministically for seed {seed}.",
        )
        for task_name, group in PRODUCTION_TASK_GROUPS.items()
        for seed in PRODUCTION_FIXED_SEEDS
    ]
    rows.reverse()
    capture = tmp_path / "production.jsonl"
    _write_capture(capture, rows)
    receipt_path = tmp_path / "production.receipt.json"
    receipt_identity = _write_receipt_stub(capture, receipt_path)
    _patch_receipt_verifier(monkeypatch)
    output = tmp_path / "production-prompts.jsonl"
    audit_path = tmp_path / "production-audit.json"

    with pytest.raises(ValueError, match="requires receipt_path"):
        _export(capture, output, audit_path, production=True)
    monkeypatch.setattr(browsergym_prompts, "PRODUCTION_CAPTURE_BYTES", None)
    monkeypatch.setattr(browsergym_prompts, "PRODUCTION_CAPTURE_SHA256", None)
    with pytest.raises(ValueError, match="capture origin is not frozen"):
        _export(
            capture,
            output,
            audit_path,
            production=True,
            receipt=receipt_path,
        )
    _freeze_browsergym_pins(
        monkeypatch,
        capture=capture,
        receipt_identity=receipt_identity,
    )
    audit = _export(
        capture,
        output,
        audit_path,
        production=True,
        receipt=receipt_path,
    )

    assert audit["mode"] == "production"
    assert audit["capture_receipt"] == receipt_identity
    assert audit["plan"]["episode_rows"] == 240
    assert audit["plan"]["task_variants"] == 60
    assert audit["plan"]["similarity_group_count"] == 41
    assert audit["plan"]["fixed_seeds"] == list(PRODUCTION_FIXED_SEEDS)
    assert audit["plan"]["localagent_policy_exclusions"] == list(
        PRODUCTION_LOCAL_POLICY_EXCLUSIONS
    )
    assert len(_read_jsonl(output)) == 240

    original_receipt_payload = receipt_path.read_bytes()
    forged_receipt = json.loads(original_receipt_payload)
    forged_receipt["producer"] = f"{forged_receipt['producer']}-forged"
    forged_without_hash = dict(forged_receipt)
    forged_without_hash.pop("receipt_self_sha256")
    forged_receipt["receipt_self_sha256"] = hashlib.sha256(
        _canonical_line(forged_without_hash)[:-1]
    ).hexdigest()
    receipt_path.write_bytes(_canonical_line(forged_receipt))
    with pytest.raises(ValueError, match="receipt identity does not match"):
        _export(
            capture,
            output,
            audit_path,
            production=True,
            receipt=receipt_path,
        )
    receipt_path.write_bytes(original_receipt_payload)

    plan = yaml.safe_load(
        prompt_freezer_helpers.PLAN.read_text(encoding="utf-8")
    )
    plan["suites"]["browsergym"]["capture_receipt"] = {
        **receipt_identity,
        "status": "frozen_controlled_acquisition",
    }
    frozen_plan = tmp_path / "frozen-browsergym-plan.yaml"
    frozen_plan.write_text(yaml.safe_dump(plan, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(prompt_freezer_helpers, "PLAN", frozen_plan)
    with pytest.raises(ValueError, match="raw capture and one raw receipt"):
        freeze_production_adapter_output(
            tmp_path,
            suite_name="browsergym",
            prompt_path=output,
            audit_path=audit_path,
        )

    tampered_dir = tmp_path / "tampered-freeze"
    tampered_dir.mkdir()
    tampered_output = tampered_dir / output.name
    tampered_output.write_bytes(output.read_bytes())
    tampered_audit_path = tampered_dir / audit_path.name
    tampered_audit = deepcopy(audit)
    tampered_audit["capture"]["sha256"] = "0" * 64
    tampered_audit_path.write_bytes(_canonical_line(tampered_audit))
    with pytest.raises(ValueError, match="raw capture and one raw receipt"):
        freeze_production_adapter_output(
            tampered_dir,
            suite_name="browsergym",
            prompt_path=tampered_output,
            audit_path=tampered_audit_path,
            plan_attestation_path=audit_path,
        )

    missing_capture = tmp_path / "production-missing.jsonl"
    _write_capture(missing_capture, rows[:-1])
    missing_receipt = tmp_path / "production-missing.receipt.json"
    _write_receipt_stub(missing_capture, missing_receipt)
    with pytest.raises(ValueError, match="episode-count mismatch"):
        _export(
            missing_capture,
            tmp_path / "missing-output.jsonl",
            tmp_path / "missing-audit.json",
            production=True,
            receipt=missing_receipt,
        )

    wrong_group_rows = deepcopy(rows)
    wrong_group_task = wrong_group_rows[0]["task_name"]
    for row in wrong_group_rows:
        if row["task_name"] == wrong_group_task:
            row["similarity_group"] = 999
    wrong_group_capture = tmp_path / "production-wrong-group.jsonl"
    _write_capture(wrong_group_capture, wrong_group_rows)
    wrong_group_receipt = tmp_path / "production-wrong-group.receipt.json"
    _write_receipt_stub(wrong_group_capture, wrong_group_receipt)
    with pytest.raises(ValueError, match="task/group plan mismatch"):
        _export(
            wrong_group_capture,
            tmp_path / "group-output.jsonl",
            tmp_path / "group-audit.json",
            production=True,
            receipt=wrong_group_receipt,
        )


def test_browsergym_production_rejects_pin_and_source_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_receipt_verifier(monkeypatch)
    rows = [
        _row(task_name, seed, group, f"{task_name} seed {seed}")
        for task_name, group in PRODUCTION_TASK_GROUPS.items()
        for seed in PRODUCTION_FIXED_SEEDS
    ]
    for row in rows:
        row["source_pins"]["browsergym_revision"] = "b" * 40
    capture = tmp_path / "wrong-pins.jsonl"
    size, sha256 = _write_capture(capture, rows)
    receipt = tmp_path / "wrong-pins.receipt.json"
    _write_receipt_stub(capture, receipt)
    with pytest.raises(ValueError, match="source pins mismatch"):
        export_browsergym_prompt_rows(
            capture,
            tmp_path / "pin-output.jsonl",
            tmp_path / "pin-audit.json",
            expected_capture_bytes=size,
            expected_capture_sha256=sha256,
            receipt_path=receipt,
        )

    max_steps_rows = [
        _row(task_name, seed, group, f"{task_name} seed {seed}")
        for task_name, group in PRODUCTION_TASK_GROUPS.items()
        for seed in PRODUCTION_FIXED_SEEDS
    ]
    for row in max_steps_rows:
        row["runtime_pins"]["max_steps"] = PRODUCTION_MAX_STEPS + 1
    max_steps_capture = tmp_path / "wrong-max-steps.jsonl"
    max_steps_size, max_steps_sha256 = _write_capture(
        max_steps_capture,
        max_steps_rows,
    )
    max_steps_receipt = tmp_path / "wrong-max-steps.receipt.json"
    _write_receipt_stub(max_steps_capture, max_steps_receipt)
    with pytest.raises(ValueError, match="max_steps mismatch"):
        export_browsergym_prompt_rows(
            max_steps_capture,
            tmp_path / "steps-output.jsonl",
            tmp_path / "steps-audit.json",
            expected_capture_bytes=max_steps_size,
            expected_capture_sha256=max_steps_sha256,
            receipt_path=max_steps_receipt,
        )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        export_browsergym_prompt_rows(
            capture,
            tmp_path / "identity-output.jsonl",
            tmp_path / "identity-audit.json",
            expected_capture_bytes=size,
            expected_capture_sha256="0" * 64,
            production=False,
        )
