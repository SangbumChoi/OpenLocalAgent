from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import localagent.data.browsergym_capture as browsergym_capture
from localagent.data.browsergym_capture import (
    BROWSERGYM_CAPTURE_PRODUCER,
    BrowserGymCaptureSettings,
    attest_git_checkout,
    attest_runtime,
    capture_browsergym_goals,
    make_browsergym_environment,
    production_capture_plan,
    verify_browsergym_capture_receipt,
)
from localagent.data.browsergym_prompts import (
    PRODUCTION_BROWSERGYM_REVISION,
    PRODUCTION_BROWSERGYM_VERSION,
    PRODUCTION_CHROMIUM_REVISION,
    PRODUCTION_CHROMIUM_VERSION,
    PRODUCTION_EPISODES,
    PRODUCTION_FIXED_SEEDS,
    PRODUCTION_MAX_STEPS,
    PRODUCTION_MINIWOB_REVISION,
    PRODUCTION_PLAYWRIGHT_VERSION,
    PRODUCTION_RUNTIME_MANIFEST_IDENTITY,
    PRODUCTION_SIMILARITY_GROUPS,
    PRODUCTION_TASK_GROUPS,
    PRODUCTION_TASK_VARIANTS,
    export_browsergym_prompt_rows,
)


def _canonical_bytes(value: Any, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def _source_evidence(*, marker: str = "a") -> dict[str, Any]:
    return {
        "source_pins": {
            "browsergym_revision": PRODUCTION_BROWSERGYM_REVISION,
            "browsergym_version": PRODUCTION_BROWSERGYM_VERSION,
            "miniwob_revision": PRODUCTION_MINIWOB_REVISION,
        },
        "repositories": {
            "browsergym": {
                "revision": PRODUCTION_BROWSERGYM_REVISION,
                "git_tree_sha1": marker * 40,
                "tracked_tree": {
                    "bytes": 123,
                    "records": 2,
                    "sha256": marker * 64,
                    "verified_worktree_sha256": marker * 64,
                },
                "worktree": "verified_against_head_no_extra_files_or_index_flags",
            },
            "miniwob": {
                "revision": PRODUCTION_MINIWOB_REVISION,
                "git_tree_sha1": marker * 40,
                "tracked_tree": {
                    "bytes": 456,
                    "records": 3,
                    "sha256": marker * 64,
                    "verified_worktree_sha256": marker * 64,
                },
                "worktree": "verified_against_head_no_extra_files_or_index_flags",
            },
        },
    }


def _runtime_evidence(*, marker: str = "c") -> dict[str, Any]:
    return {
        "runtime_pins": {
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
            "browser_executable": {"bytes": 100, "sha256": marker * 64},
            "browser_installation": {"bytes": 200, "sha256": marker * 64},
            "environment_manifest": dict(PRODUCTION_RUNTIME_MANIFEST_IDENTITY),
        },
        "attestation": {
            "installed_distributions": {
                "browsergym-core": PRODUCTION_BROWSERGYM_VERSION,
                "browsergym-miniwob": PRODUCTION_BROWSERGYM_VERSION,
                "gymnasium": "1.1.1",
                "playwright": PRODUCTION_PLAYWRIGHT_VERSION,
            },
            "python_implementation": "CPython",
            "browser": {
                "reported_version": PRODUCTION_CHROMIUM_VERSION,
                "installation_entries": 2,
                "installation_files": 2,
                "installation_symlinks": 0,
                "executable_scope": "inside_attested_installation",
            },
        },
    }


def _patch_constant_evidence(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source: dict[str, Any] | None = None,
    runtime: dict[str, Any] | None = None,
) -> None:
    source = _source_evidence() if source is None else source
    runtime = _runtime_evidence() if runtime is None else runtime
    monkeypatch.setattr(
        browsergym_capture,
        "attest_source_checkouts",
        lambda *_args, **_kwargs: deepcopy(source),
    )
    monkeypatch.setattr(
        browsergym_capture,
        "attest_runtime",
        lambda *_args, **_kwargs: deepcopy(runtime),
    )
    monkeypatch.setattr(
        browsergym_capture,
        "_verify_production_runtime_manifest",
        lambda *_args, **_kwargs: dict(PRODUCTION_RUNTIME_MANIFEST_IDENTITY),
    )

    @contextlib.contextmanager
    def fake_production_scope(*_args: Any, **_kwargs: Any):
        yield

    monkeypatch.setattr(
        browsergym_capture,
        "_production_environment_scope",
        fake_production_scope,
    )
    monkeypatch.setattr(
        browsergym_capture,
        "make_browsergym_environment",
        lambda episode, _settings, **_kwargs: _FakeEnvironment(episode, []),
    )


def _rewrite_receipt_with_valid_self_hash(path: Path, receipt: dict[str, Any]) -> None:
    without_hash = dict(receipt)
    without_hash.pop("receipt_self_sha256", None)
    receipt["receipt_self_sha256"] = hashlib.sha256(
        _canonical_bytes(without_hash)
    ).hexdigest()
    path.write_bytes(_canonical_bytes(receipt, newline=True))


class _FakeEnvironment:
    def __init__(self, episode: Any, closed: list[tuple[str, int]]) -> None:
        self.episode = episode
        self.closed = closed
        self.reset_seeds: list[int] = []
        self.step_calls = 0

    def reset(self, *, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        self.reset_seeds.append(seed)
        return (
            {
                "goal": f"goal::{self.episode.task_name}::{seed}",
                "reward": "observation-secret-that-must-not-be-captured",
                "actions": ["do-not-copy"],
                "labels": {"answer": "do-not-copy"},
            },
            {
                "goal": "poisoned-info-goal",
                "reward": 1.0,
                "action": "poisoned-info-action",
                "label": "poisoned-info-label",
            },
        )

    def step(self, _action: Any) -> None:
        self.step_calls += 1
        raise AssertionError("the reset capture producer must never call step")

    def close(self) -> None:
        self.closed.append((self.episode.task_name, self.episode.seed))


def _write_browsergym_source_fixture(checkout: Path) -> None:
    source_files = {
        "browsergym/core/src/browsergym/core/__init__.py": "",
        "browsergym/core/src/browsergym/core/action/__init__.py": "",
        "browsergym/core/src/browsergym/core/action/highlevel.py": "",
        "browsergym/miniwob/src/browsergym/miniwob/__init__.py": "",
    }
    for relative_path, payload in source_files.items():
        source_file = checkout / relative_path
        source_file.parent.mkdir(parents=True, exist_ok=True)
        source_file.write_text(payload, encoding="utf-8")


def test_capture_exact_plan_goal_provenance_schema_receipt_and_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_constant_evidence(monkeypatch)
    created: list[_FakeEnvironment] = []
    closed: list[tuple[str, int]] = []

    def factory(episode: Any, _settings: Any) -> _FakeEnvironment:
        environment = _FakeEnvironment(episode, closed)
        created.append(environment)
        return environment

    monkeypatch.setattr(
        browsergym_capture,
        "make_browsergym_environment",
        lambda episode, settings, **_kwargs: factory(episode, settings),
    )
    capture = tmp_path / "capture.jsonl"
    receipt_path = tmp_path / "capture.receipt.json"
    receipt = capture_browsergym_goals(
        capture,
        receipt_path,
        browsergym_checkout=tmp_path / "BrowserGym",
        miniwob_checkout=tmp_path / "miniwob-plusplus",
        browser_executable=tmp_path / "chromium",
        browser_installation=tmp_path / "chromium-1117",
        environment_manifest=tmp_path / "runtime-manifest.json",
    )

    plan = production_capture_plan()
    assert len(plan) == PRODUCTION_EPISODES == 240
    assert len(PRODUCTION_TASK_GROUPS) == PRODUCTION_TASK_VARIANTS == 60
    assert len(set(PRODUCTION_TASK_GROUPS.values())) == PRODUCTION_SIMILARITY_GROUPS == 41
    assert tuple(episode.seed for episode in plan[:4]) == PRODUCTION_FIXED_SEEDS
    assert [(env.episode.task_name, env.episode.seed) for env in created] == [
        (episode.task_name, episode.seed) for episode in plan
    ]
    assert all(env.reset_seeds == [env.episode.seed] for env in created)
    assert all(env.step_calls == 0 for env in created)
    assert closed == [(episode.task_name, episode.seed) for episode in plan]

    raw_lines = capture.read_bytes().splitlines(keepends=True)
    rows = [json.loads(line) for line in raw_lines]
    assert len(rows) == 240
    assert all(raw == _canonical_bytes(row, newline=True) for raw, row in zip(raw_lines, rows))
    assert all(
        set(row)
        == {
            "task_name",
            "seed",
            "goal",
            "similarity_group",
            "split",
            "source_pins",
            "runtime_pins",
        }
        for row in rows
    )
    assert rows[0]["goal"] == f"goal::{plan[0].task_name}::{plan[0].seed}"
    assert not any(row["goal"] == "poisoned-info-goal" for row in rows)
    assert not any("observation-secret" in row["goal"] for row in rows)

    without_hash = dict(receipt)
    self_hash = without_hash.pop("receipt_self_sha256")
    assert hashlib.sha256(_canonical_bytes(without_hash)).hexdigest() == self_hash
    assert receipt["producer"] == BROWSERGYM_CAPTURE_PRODUCER
    assert receipt["capture"]["sha256"] == hashlib.sha256(capture.read_bytes()).hexdigest()
    assert verify_browsergym_capture_receipt(capture, receipt_path) == receipt

    adapter_output = tmp_path / "adapter-prompts.jsonl"
    adapter_audit = tmp_path / "adapter-audit.json"
    export_browsergym_prompt_rows(
        capture,
        adapter_output,
        adapter_audit,
        expected_capture_bytes=capture.stat().st_size,
        expected_capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),
        production=False,
    )
    assert len(adapter_output.read_text(encoding="utf-8").splitlines()) == 240

    original_capture = capture.read_bytes()
    capture.write_bytes(original_capture + b"{}\n")
    with pytest.raises(ValueError, match="identity does not match"):
        verify_browsergym_capture_receipt(capture, receipt_path)
    capture.write_bytes(original_capture)

    receipt_value = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_value["capture"]["rows"] = 239
    receipt_path.write_bytes(_canonical_bytes(receipt_value, newline=True))
    with pytest.raises(ValueError, match="self-hash mismatch"):
        verify_browsergym_capture_receipt(capture, receipt_path)


@pytest.mark.parametrize(
    ("drift", "error"),
    [
        ("boundary", "boundary drift"),
        ("publication", "publication contract drift"),
        ("import_scope", "import-scope policy drift"),
        ("capture_extra", "capture schema drift"),
        ("capture_bool_bytes", "capture.bytes must be a positive integer"),
        ("plan_bool_count", "production plan drift"),
        ("control_bool_scale", "control drift"),
        ("source_extra", "browsergym schema drift"),
        ("source_dirty", "worktree is not clean"),
        ("source_tree_bool", "tracked_tree.bytes must be a positive integer"),
        ("runtime_distribution_extra", "installed_distributions schema drift"),
        ("runtime_version", "distribution version drift"),
        ("runtime_browser_count", "installation_entries must be a positive integer"),
        ("runtime_browser_scope", "executable_scope drift"),
        ("runtime_pin_bool_scale", "runtime pin drift"),
    ],
)
def test_receipt_verifier_rejects_semantic_drift_with_valid_outer_self_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
    error: str,
) -> None:
    _patch_constant_evidence(monkeypatch)
    capture = tmp_path / "capture.jsonl"
    receipt_path = tmp_path / "receipt.json"
    capture_browsergym_goals(
        capture,
        receipt_path,
        browsergym_checkout=tmp_path / "BrowserGym",
        miniwob_checkout=tmp_path / "miniwob-plusplus",
        browser_executable=tmp_path / "chromium",
        browser_installation=tmp_path / "chromium-1117",
        environment_manifest=tmp_path / "runtime-manifest.json",
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if drift == "boundary":
        receipt["boundary"] = "fabricated boundary"
    elif drift == "publication":
        receipt["publication"] = "crash-atomic"
    elif drift == "import_scope":
        receipt["import_scope"]["ambient_namespace_paths"] = "allowed"
    elif drift == "capture_extra":
        receipt["capture"]["unreviewed"] = True
    elif drift == "capture_bool_bytes":
        receipt["capture"]["bytes"] = True
    elif drift == "plan_bool_count":
        receipt["plan"]["task_variants"] = True
    elif drift == "control_bool_scale":
        receipt["controls"]["device_scale_factor"] = True
    elif drift == "source_extra":
        receipt["source_attestation"]["browsergym"]["path"] = "/unattested"
    elif drift == "source_dirty":
        receipt["source_attestation"]["browsergym"]["worktree"] = "dirty"
    elif drift == "source_tree_bool":
        receipt["source_attestation"]["browsergym"]["tracked_tree"]["bytes"] = True
    elif drift == "runtime_distribution_extra":
        receipt["runtime_attestation"]["installed_distributions"]["numpy"] = "2.0"
    elif drift == "runtime_version":
        receipt["runtime_attestation"]["installed_distributions"]["playwright"] = "9.9.9"
    elif drift == "runtime_browser_count":
        receipt["runtime_attestation"]["browser"]["installation_entries"] = -1
    elif drift == "runtime_browser_scope":
        receipt["runtime_attestation"]["browser"]["executable_scope"] = "elsewhere"
    elif drift == "runtime_pin_bool_scale":
        receipt["runtime_pins"]["device_scale_factor"] = True
    else:
        raise AssertionError(f"unhandled drift fixture {drift}")
    _rewrite_receipt_with_valid_self_hash(receipt_path, receipt)

    with pytest.raises(ValueError, match=error):
        verify_browsergym_capture_receipt(capture, receipt_path)


def test_receipt_verifier_rejects_same_length_path_swap_during_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_constant_evidence(monkeypatch)
    capture = tmp_path / "capture.jsonl"
    receipt = tmp_path / "receipt.json"
    capture_browsergym_goals(
        capture,
        receipt,
        browsergym_checkout=tmp_path / "BrowserGym",
        miniwob_checkout=tmp_path / "miniwob-plusplus",
        browser_executable=tmp_path / "chromium",
        browser_installation=tmp_path / "chromium-1117",
        environment_manifest=tmp_path / "runtime-manifest.json",
    )
    original_payload = capture.read_bytes()
    replacement_payload = bytearray(original_payload)
    goal_offset = replacement_payload.find(b"goal::")
    assert goal_offset >= 0
    replacement_payload[goal_offset] = ord("h")
    assert len(replacement_payload) == len(original_payload)
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_bytes(replacement_payload)
    capture_inode = capture.stat().st_ino
    original_read = browsergym_capture.os.read
    swapped = False

    def swapping_read(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        chunk = original_read(descriptor, size)
        if (
            not swapped
            and chunk
            and os.fstat(descriptor).st_ino == capture_inode
        ):
            os.replace(replacement, capture)
            swapped = True
        return chunk

    monkeypatch.setattr(browsergym_capture.os, "read", swapping_read)
    with pytest.raises(ValueError, match="path changed while it was read"):
        verify_browsergym_capture_receipt(capture, receipt)
    assert swapped is True


@pytest.mark.parametrize("failure", ["reset_error", "bad_tuple", "missing_goal"])
def test_capture_closes_environment_and_publishes_nothing_on_reset_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    _patch_constant_evidence(monkeypatch)
    closed: list[str] = []

    class BrokenEnvironment:
        def reset(self, *, seed: int) -> Any:
            del seed
            if failure == "reset_error":
                raise RuntimeError("reset exploded")
            if failure == "bad_tuple":
                return {"goal": "not a reset tuple"}
            return ({}, {"goal": "info must not be used as fallback"})

        def close(self) -> None:
            closed.append("closed")

    capture = tmp_path / f"{failure}.jsonl"
    receipt = tmp_path / f"{failure}.receipt.json"
    monkeypatch.setattr(
        browsergym_capture,
        "make_browsergym_environment",
        lambda *_args, **_kwargs: BrokenEnvironment(),
    )
    with pytest.raises((RuntimeError, ValueError)):
        capture_browsergym_goals(
            capture,
            receipt,
            browsergym_checkout=tmp_path / "BrowserGym",
            miniwob_checkout=tmp_path / "miniwob-plusplus",
            browser_executable=tmp_path / "chromium",
            browser_installation=tmp_path / "chromium-1117",
            environment_manifest=tmp_path / "runtime-manifest.json",
        )
    assert closed == ["closed"]
    assert not capture.exists()
    assert not receipt.exists()


@pytest.mark.parametrize("drift", ["source", "runtime"])
def test_capture_rejects_source_or_runtime_drift_after_resets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    _patch_constant_evidence(monkeypatch)
    sources = [_source_evidence(marker="a"), _source_evidence(marker="b")]
    runtimes = [_runtime_evidence(marker="c"), _runtime_evidence(marker="d")]
    if drift == "source":
        runtime = _runtime_evidence()
        runtimes = [runtime, runtime]
    else:
        source = _source_evidence()
        sources = [source, source]
    monkeypatch.setattr(
        browsergym_capture,
        "attest_source_checkouts",
        lambda *_args, **_kwargs: deepcopy(sources.pop(0)),
    )
    monkeypatch.setattr(
        browsergym_capture,
        "attest_runtime",
        lambda *_args, **_kwargs: deepcopy(runtimes.pop(0)),
    )
    closed: list[tuple[str, int]] = []
    capture = tmp_path / f"{drift}.jsonl"
    receipt = tmp_path / f"{drift}.receipt.json"
    monkeypatch.setattr(
        browsergym_capture,
        "make_browsergym_environment",
        lambda episode, _settings, **_kwargs: _FakeEnvironment(episode, closed),
    )

    with pytest.raises(RuntimeError, match=rf"{drift} identity drifted"):
        capture_browsergym_goals(
            capture,
            receipt,
            browsergym_checkout=tmp_path / "BrowserGym",
            miniwob_checkout=tmp_path / "miniwob-plusplus",
            browser_executable=tmp_path / "chromium",
            browser_installation=tmp_path / "chromium-1117",
            environment_manifest=tmp_path / "runtime-manifest.json",
        )
    assert len(closed) == PRODUCTION_EPISODES
    assert not capture.exists()
    assert not receipt.exists()


@pytest.mark.parametrize("existing", ["capture", "receipt"])
def test_capture_is_strictly_non_clobbering_before_environment_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing: str,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        browsergym_capture,
        "attest_source_checkouts",
        lambda *_args, **_kwargs: calls.append("source"),
    )
    capture = tmp_path / "capture.jsonl"
    receipt = tmp_path / "receipt.json"
    target = capture if existing == "capture" else receipt
    target.write_bytes(b"preserve-me")

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        capture_browsergym_goals(
            capture,
            receipt,
            browsergym_checkout=tmp_path / "BrowserGym",
            miniwob_checkout=tmp_path / "miniwob-plusplus",
            browser_executable=tmp_path / "chromium",
            browser_installation=tmp_path / "chromium-1117",
            environment_manifest=tmp_path / "runtime-manifest.json",
        )
    assert target.read_bytes() == b"preserve-me"
    assert calls == []


def test_two_file_publication_rolls_back_ordinary_second_link_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = tmp_path / "capture.jsonl"
    receipt = tmp_path / "receipt.json"
    original_link = browsergym_capture.os.link
    link_calls = 0

    def fail_second_link(source: str | Path, destination: str | Path) -> None:
        nonlocal link_calls
        link_calls += 1
        if link_calls == 2:
            raise FileExistsError("simulated concurrent receipt")
        original_link(source, destination)

    monkeypatch.setattr(browsergym_capture.os, "link", fail_second_link)
    with pytest.raises(RuntimeError, match="concurrently-created"):
        browsergym_capture._publish_pair_no_clobber(
            capture,
            b'{"capture":true}\n',
            receipt,
            b'{"receipt":true}\n',
        )
    assert not capture.exists()
    assert not receipt.exists()

    capture.write_bytes(b'{"capture":true}\n')
    with pytest.raises(ValueError, match="receipt.*readable"):
        verify_browsergym_capture_receipt(capture, receipt)


def _git(checkout: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_git_checkout_attestation_rejects_revision_and_dirty_drift(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _git(checkout, "init")
    _git(checkout, "config", "user.email", "capture@example.invalid")
    _git(checkout, "config", "user.name", "Capture Test")
    tracked = checkout / "tracked.txt"
    tracked.write_text("v1\n", encoding="utf-8")
    _git(checkout, "add", "tracked.txt")
    _git(checkout, "commit", "-m", "v1")
    revision = _git(checkout, "rev-parse", "HEAD")

    attestation = attest_git_checkout(
        checkout,
        expected_revision=revision,
        label="fixture",
    )
    assert attestation["revision"] == revision
    assert attestation["tracked_tree"]["records"] == 1
    assert len(attestation["tracked_tree"]["verified_worktree_sha256"]) == 64

    tracked.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dirty"):
        attest_git_checkout(
            checkout,
            expected_revision=revision,
            label="fixture",
        )
    tracked.write_text("v1\n", encoding="utf-8")
    (checkout / "untracked.txt").write_text("untracked\n", encoding="utf-8")
    with pytest.raises(ValueError, match="untracked"):
        attest_git_checkout(
            checkout,
            expected_revision=revision,
            label="fixture",
        )
    (checkout / "untracked.txt").unlink()

    _git(checkout, "update-index", "--assume-unchanged", "tracked.txt")
    tracked.write_text("v2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="index flags"):
        attest_git_checkout(
            checkout,
            expected_revision=revision,
            label="fixture",
        )
    _git(checkout, "update-index", "--no-assume-unchanged", "tracked.txt")
    tracked.write_text("v1\n", encoding="utf-8")

    _git(checkout, "update-index", "--skip-worktree", "tracked.txt")
    tracked.write_text("v2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="index flags"):
        attest_git_checkout(
            checkout,
            expected_revision=revision,
            label="fixture",
        )
    _git(checkout, "update-index", "--no-skip-worktree", "tracked.txt")
    tracked.write_text("v1\n", encoding="utf-8")

    exclude = checkout / ".git/info/exclude"
    exclude.write_text("ignored-tool\n", encoding="utf-8")
    ignored_tool = checkout / "ignored-tool"
    ignored_tool.write_text("#!/bin/sh\n", encoding="utf-8")
    ignored_tool.chmod(0o755)
    with pytest.raises(ValueError, match="ignored files"):
        attest_git_checkout(
            checkout,
            expected_revision=revision,
            label="fixture",
        )
    ignored_tool.unlink()

    tracked.write_text("v2\n", encoding="utf-8")
    _git(checkout, "add", "tracked.txt")
    _git(checkout, "commit", "-m", "v2")
    with pytest.raises(ValueError, match="revision drift"):
        attest_git_checkout(
            checkout,
            expected_revision=revision,
            label="fixture",
        )


def test_runtime_attestation_binds_executable_tree_and_versions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installation = tmp_path / f"chromium-{PRODUCTION_CHROMIUM_REVISION}"
    executable = installation / "chrome"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"fake-chromium-binary")
    executable.chmod(0o755)
    resource = installation / "resources.pak"
    resource.write_bytes(b"resource-v1")
    sentinel = installation / "DEPENDENCIES_VALIDATED"
    sentinel.write_bytes(b"")
    versions = {
        "browsergym-core": PRODUCTION_BROWSERGYM_VERSION,
        "browsergym-miniwob": PRODUCTION_BROWSERGYM_VERSION,
        "gymnasium": "1.1.1",
        "playwright": PRODUCTION_PLAYWRIGHT_VERSION,
    }
    monkeypatch.setattr(
        browsergym_capture,
        "_installed_version",
        lambda distribution: versions[distribution],
    )
    monkeypatch.setattr(
        browsergym_capture,
        "_reported_chromium_version",
        lambda _path: PRODUCTION_CHROMIUM_VERSION,
    )

    first = attest_runtime(executable, installation)
    assert first["runtime_pins"]["browser_executable"] == {
        "bytes": len(b"fake-chromium-binary"),
        "sha256": hashlib.sha256(b"fake-chromium-binary").hexdigest(),
    }
    first_tree = first["runtime_pins"]["browser_installation"]["sha256"]
    assert first["attestation"]["browser"]["installation_files"] == 3

    sentinel.unlink()
    without_sentinel = attest_runtime(executable, installation)
    assert without_sentinel["runtime_pins"]["browser_installation"]["sha256"] != first_tree
    sentinel.write_bytes(b"")
    restored = attest_runtime(executable, installation)
    assert restored["runtime_pins"]["browser_installation"]["sha256"] == first_tree

    resource.write_bytes(b"resource-v2")
    second = attest_runtime(executable, installation)
    assert second["runtime_pins"]["browser_installation"]["sha256"] != first_tree

    versioned = installation / "versioned"
    versioned.mkdir()
    (versioned / "framework.bin").write_bytes(b"framework")
    internal_link = installation / "Current"
    internal_link.symlink_to("versioned", target_is_directory=True)
    internal = attest_runtime(executable, installation)
    assert internal["attestation"]["browser"]["installation_symlinks"] == 1

    executable_link = installation / "chrome-link"
    executable_link.symlink_to("chrome")
    linked_executable = attest_runtime(executable_link, installation)
    assert linked_executable["runtime_pins"]["browser_executable"] == second[
        "runtime_pins"
    ]["browser_executable"]

    outside = tmp_path / "outside-framework.bin"
    outside.write_bytes(b"outside")
    escaping_link = installation / "escaping"
    escaping_link.symlink_to(outside)
    with pytest.raises(ValueError, match="broken or escaping symbolic link"):
        attest_runtime(executable, installation)
    escaping_link.unlink()

    empty_executable = installation / "empty-chrome"
    empty_executable.write_bytes(b"")
    empty_executable.chmod(0o755)
    with pytest.raises(ValueError, match="non-empty regular file"):
        attest_runtime(empty_executable, installation)

    versions["playwright"] = "9.9.9"
    with pytest.raises(ValueError, match="version drift"):
        attest_runtime(executable, installation)


def test_real_environment_factory_uses_pinned_id_and_all_controls_lazily(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "BrowserGym"
    miniwob_file = checkout / "browsergym/miniwob/__init__.py"
    highlevel_file = checkout / "browsergym/core/action/highlevel.py"
    miniwob_file.parent.mkdir(parents=True)
    highlevel_file.parent.mkdir(parents=True)
    miniwob_file.write_text("", encoding="utf-8")
    highlevel_file.write_text("", encoding="utf-8")
    calls: list[tuple[str, dict[str, Any]]] = []
    expected_environment = object()

    class FakeHighLevelActionSet:
        def to_python_code(self, action: str) -> str:
            return action

    fake_modules = {
        "browsergym.miniwob": SimpleNamespace(__file__=str(miniwob_file)),
        "browsergym.core.action.highlevel": SimpleNamespace(
            __file__=str(highlevel_file),
            HighLevelActionSet=FakeHighLevelActionSet,
        ),
        "gymnasium": SimpleNamespace(
            make=lambda environment_id, **kwargs: (
                calls.append((environment_id, kwargs)) or expected_environment
            )
        ),
    }
    monkeypatch.setattr(
        browsergym_capture.importlib,
        "import_module",
        lambda name: fake_modules[name],
    )
    episode = production_capture_plan()[0]
    executable = tmp_path / "chromium"

    environment = make_browsergym_environment(
        episode,
        BrowserGymCaptureSettings(),
        browsergym_checkout=checkout,
        browser_executable=executable,
    )
    assert environment is expected_environment
    assert len(calls) == 1
    environment_id, kwargs = calls[0]
    assert environment_id == f"browsergym/{episode.task_name}"
    assert kwargs["viewport"] == {"width": 1280, "height": 720}
    assert kwargs["timeout"] == 30_000
    assert kwargs["locale"] == "en-US"
    assert kwargs["timezone_id"] == "UTC"
    assert kwargs["headless"] is True
    assert kwargs["use_raw_page_output"] is False
    assert kwargs["pw_context_kwargs"] == {"device_scale_factor": 1.0}
    assert kwargs["pw_chromium_kwargs"] == {
        "executable_path": str(executable.resolve())
    }
    assert kwargs["max_episode_steps"] == PRODUCTION_MAX_STEPS
    assert callable(kwargs["action_mapping"])

    fake_modules["browsergym.core.action.highlevel"].__file__ = str(
        tmp_path / "unattested/highlevel.py"
    )
    with pytest.raises(RuntimeError, match="not imported from"):
        make_browsergym_environment(
            episode,
            BrowserGymCaptureSettings(),
            browsergym_checkout=checkout,
            browser_executable=executable,
        )


def test_isolated_browsergym_import_scope_excludes_ambient_package_and_restores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "BrowserGym"
    _write_browsergym_source_fixture(checkout)
    ambient = tmp_path / "ambient"
    ambient_package = ambient / "browsergym"
    ambient_package.mkdir(parents=True)
    (ambient_package / "__init__.py").write_text(
        'raise RuntimeError("ambient browsergym package executed")\n',
        encoding="utf-8",
    )
    (ambient_package / "ambient_only.py").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(ambient))
    prior_sys_path = list(sys.path)

    core_root = (checkout / "browsergym/core/src").resolve()
    miniwob_root = (checkout / "browsergym/miniwob/src").resolve()
    expected_namespace_paths = [
        str(core_root / "browsergym"),
        str(miniwob_root / "browsergym"),
    ]
    with browsergym_capture._isolated_browsergym_imports(checkout):
        assert sys.path[:2] == [str(core_root), str(miniwob_root)]
        namespace = sys.modules["browsergym"]
        assert list(namespace.__path__) == expected_namespace_paths
        assert list(namespace.__spec__.submodule_search_locations) == (
            expected_namespace_paths
        )
        with pytest.raises(ModuleNotFoundError):
            browsergym_capture.importlib.import_module("browsergym.ambient_only")

    assert sys.path == prior_sys_path
    assert not any(
        name == "browsergym" or name.startswith("browsergym.")
        for name in sys.modules
    )

    monkeypatch.setitem(
        sys.modules,
        "browsergym",
        SimpleNamespace(__file__=str(ambient_package / "__init__.py")),
    )
    with pytest.raises(RuntimeError, match="loaded before"):
        with browsergym_capture._isolated_browsergym_imports(checkout):
            pytest.fail("preloaded BrowserGym must fail before import-scope mutation")
    assert sys.path == prior_sys_path
    monkeypatch.delitem(sys.modules, "browsergym")

    with pytest.raises(RuntimeError, match="sys.path drifted"):
        with browsergym_capture._isolated_browsergym_imports(checkout):
            sys.path.append(str(tmp_path / "drift"))
    assert sys.path == prior_sys_path


def test_production_environment_scope_disables_bytecode_and_restores_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    miniwob_checkout = tmp_path / "miniwob-plusplus"
    html = miniwob_checkout / "miniwob/html/miniwob"
    html.mkdir(parents=True)
    browsergym_checkout = tmp_path / "BrowserGym"
    _write_browsergym_source_fixture(browsergym_checkout)
    browser_root = tmp_path / "ms-playwright"
    installation = browser_root / f"chromium-{PRODUCTION_CHROMIUM_REVISION}"
    executable = installation / "chrome"
    installation.mkdir(parents=True)
    executable.write_bytes(b"chromium")
    executable.chmod(0o755)
    monkeypatch.delenv("MINIWOB_URL", raising=False)
    monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
    ownership_events: list[str] = []

    @contextlib.contextmanager
    def fake_owned_playwright(_checkout: Path, _executable: Path):
        assert _checkout == browsergym_checkout
        assert _executable == executable
        assert os.environ["PLAYWRIGHT_BROWSERS_PATH"] == str(browser_root.resolve())
        ownership_events.append("start")
        try:
            yield
        finally:
            ownership_events.append("stop")

    monkeypatch.setattr(
        browsergym_capture,
        "_owned_browsergym_playwright",
        fake_owned_playwright,
    )
    prior = sys.dont_write_bytecode
    sys.dont_write_bytecode = False
    try:
        with browsergym_capture._production_environment_scope(
            browsergym_checkout,
            miniwob_checkout,
            executable,
            installation,
        ):
            assert sys.dont_write_bytecode is True
            assert os.environ["MINIWOB_URL"] == html.as_uri().rstrip("/") + "/"
            assert os.environ["PLAYWRIGHT_BROWSERS_PATH"] == str(browser_root.resolve())
        assert sys.dont_write_bytecode is False
        assert "MINIWOB_URL" not in os.environ
        assert "PLAYWRIGHT_BROWSERS_PATH" not in os.environ
        assert ownership_events == ["start", "stop"]
    finally:
        sys.dont_write_bytecode = prior


def test_production_environment_scope_rejects_browser_path_drift_and_restores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    miniwob_checkout = tmp_path / "miniwob-plusplus"
    (miniwob_checkout / "miniwob/html/miniwob").mkdir(parents=True)
    browsergym_checkout = tmp_path / "BrowserGym"
    _write_browsergym_source_fixture(browsergym_checkout)
    browser_root = tmp_path / "ms-playwright"
    installation = browser_root / f"chromium-{PRODUCTION_CHROMIUM_REVISION}"
    executable = installation / "chrome"
    installation.mkdir(parents=True)
    executable.write_bytes(b"chromium")
    executable.chmod(0o755)
    other_root = tmp_path / "other-playwright"
    other_root.mkdir()

    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(other_root))
    with pytest.raises(ValueError, match="outside the attested browser root"):
        with browsergym_capture._production_environment_scope(
            browsergym_checkout,
            miniwob_checkout,
            executable,
            installation,
        ):
            pytest.fail("conflicting browser root must fail before capture")
    assert os.environ["PLAYWRIGHT_BROWSERS_PATH"] == str(other_root)

    monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH")
    wrong_executable = installation / "other-chrome"
    wrong_executable.write_bytes(b"other")
    wrong_executable.chmod(0o755)

    @contextlib.contextmanager
    def wrong_owned_playwright(_checkout: Path, _executable: Path):
        raise ValueError(
            "Playwright's default Chromium executable does not match the "
            "separately attested executable"
        )
        yield

    monkeypatch.setattr(
        browsergym_capture,
        "_owned_browsergym_playwright",
        wrong_owned_playwright,
    )
    with pytest.raises(ValueError, match="does not match"):
        with browsergym_capture._production_environment_scope(
            browsergym_checkout,
            miniwob_checkout,
            executable,
            installation,
        ):
            pytest.fail("wrong Playwright default executable must fail before capture")
    assert "PLAYWRIGHT_BROWSERS_PATH" not in os.environ

    @contextlib.contextmanager
    def valid_owned_playwright(_checkout: Path, _executable: Path):
        yield

    monkeypatch.setattr(
        browsergym_capture,
        "_owned_browsergym_playwright",
        valid_owned_playwright,
    )
    with pytest.raises(RuntimeError, match="drifted during capture"):
        with browsergym_capture._production_environment_scope(
            browsergym_checkout,
            miniwob_checkout,
            executable,
            installation,
        ):
            os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(other_root)
    assert "PLAYWRIGHT_BROWSERS_PATH" not in os.environ


def test_owned_browsergym_playwright_rejects_stale_or_wrong_driver_and_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "BrowserGym"
    core_file = checkout / "browsergym/core/__init__.py"
    core_file.parent.mkdir(parents=True)
    core_file.write_text("", encoding="utf-8")
    executable = tmp_path / "ms-playwright/chromium-1117/chrome"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"chromium")
    executable.chmod(0o755)
    events: list[str] = []

    class FakePlaywright:
        def __init__(self, path: Path) -> None:
            self.chromium = SimpleNamespace(executable_path=str(path))

        def stop(self) -> None:
            events.append("stop")

    class FakeManager:
        def __init__(self, playwright: FakePlaywright) -> None:
            self.playwright = playwright

        def start(self) -> FakePlaywright:
            events.append("start")
            return self.playwright

    core_module = SimpleNamespace(
        __file__=str(core_file),
        _PLAYWRIGHT=None,
    )

    def set_global(playwright: Any) -> None:
        core_module._PLAYWRIGHT = playwright
        events.append("clear" if playwright is None else "set")

    core_module._set_global_playwright = set_global
    owned = FakePlaywright(executable)
    sync_api = SimpleNamespace(sync_playwright=lambda: FakeManager(owned))
    modules = {
        "browsergym.core": core_module,
        "playwright.sync_api": sync_api,
    }
    monkeypatch.setattr(
        browsergym_capture.importlib,
        "import_module",
        lambda name: modules[name],
    )

    with browsergym_capture._owned_browsergym_playwright(checkout, executable):
        assert core_module._PLAYWRIGHT is owned
    assert core_module._PLAYWRIGHT is None
    assert events == ["start", "set", "clear", "stop"]

    core_module._PLAYWRIGHT = object()
    with pytest.raises(RuntimeError, match="already owns"):
        with browsergym_capture._owned_browsergym_playwright(checkout, executable):
            pytest.fail("stale global Playwright must fail before capture")
    core_module._PLAYWRIGHT = None

    wrong = tmp_path / "wrong-chromium"
    wrong.write_bytes(b"wrong")
    sync_api.sync_playwright = lambda: FakeManager(FakePlaywright(wrong))
    with pytest.raises(ValueError, match="does not match"):
        with browsergym_capture._owned_browsergym_playwright(checkout, executable):
            pytest.fail("wrong default executable must fail before capture")
    assert core_module._PLAYWRIGHT is None
    assert events[-3:] == ["start", "clear", "stop"]


def test_capture_controls_cannot_be_relaxed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_constant_evidence(monkeypatch)
    with pytest.raises(ValueError, match="controls are immutable"):
        capture_browsergym_goals(
            tmp_path / "capture.jsonl",
            tmp_path / "receipt.json",
            browsergym_checkout=tmp_path / "BrowserGym",
            miniwob_checkout=tmp_path / "miniwob-plusplus",
            browser_executable=tmp_path / "chromium",
            browser_installation=tmp_path / "chromium-1117",
            environment_manifest=tmp_path / "runtime-manifest.json",
            settings=BrowserGymCaptureSettings(
                playwright_operation_timeout_seconds=31.0
            ),
        )


def test_production_capture_api_rejects_injected_environment_factory(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="environment_factory"):
        capture_browsergym_goals(
            tmp_path / "capture.jsonl",
            tmp_path / "receipt.json",
            browsergym_checkout=tmp_path / "BrowserGym",
            miniwob_checkout=tmp_path / "miniwob-plusplus",
            browser_executable=tmp_path / "chromium",
            browser_installation=tmp_path / "chromium-1117",
            environment_manifest=tmp_path / "runtime-manifest.json",
            environment_factory=lambda *_args: pytest.fail(
                "injected environment must not run"
            ),
        )
