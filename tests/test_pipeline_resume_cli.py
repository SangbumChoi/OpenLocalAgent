"""Operational exact-resume CLI contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from openlocalagent import cli
from openlocalagent import pipeline as flow
from openlocalagent.stages import Stage


def test_train_cli_routes_a_plain_run_to_run_stage(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(flow, "run_stage", lambda stage, config: calls.append((stage, config)))
    monkeypatch.setattr(flow, "resume_stage", lambda *a, **k: pytest.fail("should not resume"))

    cli._train(SimpleNamespace(stage="posttrain", config="posttrain.yaml", resume=False))

    assert calls == [("posttrain", "posttrain.yaml")]


def test_train_cli_passes_resume_without_editing_config(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        flow,
        "resume_stage",
        lambda stage, config, *, git_receipt=None: calls.append((stage, config, git_receipt)),
    )

    cli._train(
        SimpleNamespace(
            stage="posttrain",
            config="configs/posttrain/sft-paper-tier-1m.yaml",
            resume=True,
        )
    )

    assert calls == [("posttrain", "configs/posttrain/sft-paper-tier-1m.yaml", None)]


def test_pipeline_resume_override_is_limited_to_exact_resume_stages(monkeypatch) -> None:
    called = []

    class _Module:
        @staticmethod
        def run(config_path: str, *, resume: bool = False) -> None:
            called.append((config_path, resume))

    monkeypatch.setattr("importlib.import_module", lambda _name: _Module)

    flow.resume_stage("rl", "rl.yaml")
    assert called == [("rl.yaml", True)]

    with pytest.raises(SystemExit, match="does not support exact resume"):
        flow.resume_stage("eval", "eval.yaml")


def test_resume_git_receipt_routes_only_to_posttrain(monkeypatch) -> None:
    calls = []

    class _Module:
        @staticmethod
        def run(
            config_path: str,
            *,
            resume: bool = False,
            resume_git_receipt: str | None = None,
        ) -> None:
            calls.append((config_path, resume, resume_git_receipt))

    monkeypatch.setattr("importlib.import_module", lambda _name: _Module)
    flow.resume_stage(
        "posttrain",
        "posttrain.yaml",
        git_receipt="/private/tmp/posttrain-receipt.json",
    )
    assert calls == [("posttrain.yaml", True, "/private/tmp/posttrain-receipt.json")]

    with pytest.raises(SystemExit, match="only the posttrain stage"):
        flow.resume_stage("rl", "rl.yaml", git_receipt="/private/tmp/posttrain-receipt.json")


def test_train_cli_passes_resume_git_receipt(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        flow,
        "resume_stage",
        lambda stage, config, *, git_receipt=None: calls.append((stage, config, git_receipt)),
    )
    cli._train(
        SimpleNamespace(
            stage="posttrain",
            config="posttrain.yaml",
            resume=True,
            resume_git_receipt="/private/tmp/posttrain-receipt.json",
        )
    )
    assert calls == [("posttrain", "posttrain.yaml", "/private/tmp/posttrain-receipt.json")]


def test_sft_is_accepted_as_the_older_spelling_of_posttrain() -> None:
    assert Stage.parse("sft") is Stage.POSTTRAIN
    assert Stage.parse("posttrain") is Stage.POSTTRAIN
    # Checkpoints and lineage receipts still carry the old spelling.
    assert Stage.POSTTRAIN.checkpoint_stage == "sft"
    assert Stage.PRETRAIN.checkpoint_stage == "pretrain"
    with pytest.raises(SystemExit, match="unknown stage"):
        Stage.parse("finetune")


def test_create_resume_git_receipt_cli_routes_reason_and_evidence(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        "openlocalagent.train.sft.create_resume_git_receipt",
        lambda config, out, *, reason, evidence: calls.append(
            (config, out, reason, evidence)
        ),
    )
    cli._create_resume_git_receipt(
        SimpleNamespace(
            config="posttrain.yaml",
            out="/private/tmp/posttrain-receipt.json",
            reason="non-numerical resume startup optimization",
            evidence=["focused tests passed", "reviewed diff"],
        )
    )
    assert calls == [
        (
            "posttrain.yaml",
            "/private/tmp/posttrain-receipt.json",
            "non-numerical resume startup optimization",
            ["focused tests passed", "reviewed diff"],
        )
    ]
