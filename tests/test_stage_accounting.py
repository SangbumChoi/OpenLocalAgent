from __future__ import annotations

import copy
import hashlib
import importlib
import subprocess
from pathlib import Path

import pytest
import torch
import yaml

from localagent.data.agent_synth import Sample
from localagent.data.render import IGNORE, render_sft
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import ByteTokenizer
from localagent.train.midtrain import MixtureSource, ScheduledMixture, midtrain
from localagent.train.pretrain import pretrain
from localagent.train.sft import sft
from localagent.train.stage_data import (
    assert_checkpoint_tokenizer,
    build_stage_lineage,
    canonical_sha256,
    git_identity,
    tokenizer_identity,
)


def _legacy_strict_conversation_source(split: str) -> dict:
    stem = "agent_sft" if split == "train" else "agent_eval"
    config = "agent_synth.yaml" if split == "train" else "agent_synth_eval.yaml"
    return {
        "path": f"data/synth/{stem}.jsonl",
        "artifact": {
            "generator_config": f"configs/data/{config}",
            "manifest": f"data/synth/{stem}.jsonl.manifest.v1.json",
            "expected_split": split,
            "expected_rule_verified": True,
            "environment_policy": "forbid",
        },
    }


def _paper_v2_train_conversation_source() -> dict:
    stem = "agent_sft_paper_train_v2"
    return {
        "path": f"data/synth/{stem}.jsonl",
        "artifact": {
            "generator_config": "configs/data/agent_synth_paper_train_v2.yaml",
            "manifest": f"data/synth/{stem}.jsonl.manifest.v1.json",
            "expected_split": "train",
            "expected_rule_verified": True,
            "environment_policy": "forbid",
        },
    }


def _without_paper_arm_identity(config: dict) -> dict:
    comparable = copy.deepcopy(config)
    comparable.pop("model_config")
    comparable.pop("init_from")
    comparable["log"].pop("out_dir")
    return comparable


def _model() -> LocalAgentLM:
    cfg = ModelConfig(
        name="stage-accounting-test",
        vocab_size=256,
        d_model=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden=32,
        max_seq_len=64,
        dropout=0.0,
    )
    cfg.assert_within_budget()
    return LocalAgentLM(cfg)


def test_canonical_and_tokenizer_hashes_are_stable_and_content_sensitive(tmp_path):
    assert canonical_sha256({"b": 2, "a": [1]}) == canonical_sha256({"a": [1], "b": 2})
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text('{"version":1}', encoding="utf-8")
    identity = tokenizer_identity("bpe", vocab_size=300, path=tokenizer_path)
    assert identity["sha256"] == hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    tokenizer_path.write_text('{"version":2}', encoding="utf-8")
    assert (
        tokenizer_identity("bpe", vocab_size=300, path=tokenizer_path)["sha256"]
        != identity["sha256"]
    )


def test_checkpoint_tokenizer_validation_uses_lineage_or_artifact_metadata():
    expected = "a" * 64
    different = "b" * 64
    assert_checkpoint_tokenizer(
        {"tokenizer": {"sha256": expected}},
        expected,
    )
    with pytest.raises(ValueError, match="tokenizer lineage"):
        assert_checkpoint_tokenizer(
            {"tokenizer": {"sha256": different}},
            expected,
        )
    with pytest.raises(ValueError, match="conflicting tokenizer identities"):
        assert_checkpoint_tokenizer(
            {
                "lineage": {"tokenizer_sha256": expected},
                "tokenizer": {"sha256": different},
            },
            expected,
        )
    with pytest.raises(ValueError, match="no content-bound tokenizer identity"):
        assert_checkpoint_tokenizer({}, expected)


def test_paper_stage_configs_pin_heldout_inputs_and_feasible_rl_budget():
    root = Path(__file__).resolve().parents[1]
    for arm in ("hybrid", "attn"):
        midtrain_cfg = yaml.safe_load(
            (root / f"configs/train/midtrain-paper-{arm}.yaml").read_text()
        )
        assert {source["name"] for source in midtrain_cfg["data"]["eval_sources"]} == {
            "general_holdout",
            "permissive_python_holdout",
            "structured_web_holdout",
            "agent_holdout",
        }
        assert midtrain_cfg["data"]["mixture"]["unit"] == "loss_tokens"
        assert midtrain_cfg["data"]["strict_conversation_artifacts"] is True
        # The former 2,500-update ceiling replayed to only 7,429,270 supervised tokens.
        assert midtrain_cfg["schedule"]["total_steps"] == 25_000
        agent_train_source = next(
            source
            for source in midtrain_cfg["data"]["sources"]
            if source["type"] == "conversations"
        )
        assert {
            "path": agent_train_source["path"],
            "artifact": agent_train_source["artifact"],
        } == _paper_v2_train_conversation_source()
        agent_eval_source = next(
            source
            for source in midtrain_cfg["data"]["eval_sources"]
            if source["type"] == "conversations"
        )
        assert {
            "path": agent_eval_source["path"],
            "artifact": agent_eval_source["artifact"],
        } == _legacy_strict_conversation_source("eval")
        assert midtrain_cfg["evaluation"]["batches_per_source"] > 0
        assert midtrain_cfg["runtime"]["resume"] is True
        assert 0 < midtrain_cfg["log"]["ckpt_every"] < midtrain_cfg["schedule"]["total_steps"]

        sft_cfg = yaml.safe_load((root / f"configs/train/sft-paper-{arm}.yaml").read_text())
        assert sft_cfg["data"]["strict_conversation_artifacts"] is True
        assert sft_cfg["data"]["conversations"] == [_paper_v2_train_conversation_source()]
        assert sft_cfg["data"]["eval_conversations"] == [
            _legacy_strict_conversation_source("eval")
        ]
        assert sft_cfg["evaluation"]["batch_size"] > 0

        rl_cfg = yaml.safe_load((root / f"configs/train/rl-paper-{arm}.yaml").read_text())
        assert rl_cfg["data"]["strict_conversation_artifacts"] is True
        assert rl_cfg["data"]["conversations"] == [_paper_v2_train_conversation_source()]
        assert rl_cfg["data"]["eval_conversations"] == [
            _legacy_strict_conversation_source("eval")
        ]
        assert rl_cfg["rollout"]["max_new_tokens"] == 256


def test_paper_stage_attention_and_hybrid_arms_keep_matched_data_and_schedules():
    root = Path(__file__).resolve().parents[1]
    for stage in ("midtrain", "sft", "rl"):
        attention = yaml.safe_load(
            (root / f"configs/train/{stage}-paper-attn.yaml").read_text(encoding="utf-8")
        )
        hybrid = yaml.safe_load(
            (root / f"configs/train/{stage}-paper-hybrid.yaml").read_text(encoding="utf-8")
        )
        assert _without_paper_arm_identity(attention) == _without_paper_arm_identity(hybrid)


def test_generic_stage_configs_require_strict_train_conversation_artifacts():
    root = Path(__file__).resolve().parents[1]
    midtrain_cfg = yaml.safe_load(
        (root / "configs/train/midtrain.yaml").read_text(encoding="utf-8")
    )
    assert midtrain_cfg["data"]["strict_conversation_artifacts"] is True
    agent_source = next(
        source for source in midtrain_cfg["data"]["sources"] if source["type"] == "conversations"
    )
    assert {
        "path": agent_source["path"],
        "artifact": agent_source["artifact"],
    } == _legacy_strict_conversation_source("train")

    sft_cfg = yaml.safe_load((root / "configs/train/sft.yaml").read_text(encoding="utf-8"))
    assert sft_cfg["data"]["strict_conversation_artifacts"] is True
    assert sft_cfg["data"]["conversations"] == [
        _legacy_strict_conversation_source("train")
    ]


def test_stage_lineage_hashes_contract_data_tokenizer_parent_and_git(tmp_path, monkeypatch):
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        "localagent.train.stage_data.git_identity",
        lambda _path: {"commit": "abc", "dirty": False, "worktree_sha256": "def"},
    )
    tokenizer = tokenizer_identity("byte", vocab_size=256)
    base = build_stage_lineage(
        stage="pretrain",
        config={"runtime": {"resume": False}, "optim": {"lr": 1e-3}},
        model_config={"vocab_size": 256},
        data_identity={"manifest_sha256": "data-a"},
        tokenizer=tokenizer,
        workspace=tmp_path,
        parent_checkpoint=parent,
    )
    resumed = build_stage_lineage(
        stage="pretrain",
        config={"optim": {"lr": 1e-3}, "runtime": {"resume": True}},
        model_config={"vocab_size": 256},
        data_identity={"manifest_sha256": "data-a"},
        tokenizer=tokenizer,
        workspace=tmp_path,
        parent_checkpoint=parent,
    )
    assert base == resumed
    assert base["parent_checkpoint_sha256"] == hashlib.sha256(b"checkpoint").hexdigest()
    assert base["git"]["commit"] == "abc"
    assert all(
        len(base[key]) == 64
        for key in (
            "config_sha256",
            "model_config_sha256",
            "data_sha256",
            "tokenizer_sha256",
        )
    )


def test_git_identity_accepts_a_file_and_tracks_repo_worktree_content(tmp_path):
    repo = tmp_path / "repo"
    implementation = repo / "src" / "module.py"
    implementation.parent.mkdir(parents=True)
    implementation.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "src/module.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Lineage Test",
            "-c",
            "user.email=lineage@example.invalid",
            "commit",
            "-q",
            "-m",
            "initial",
        ],
        check=True,
    )

    clean = git_identity(implementation)
    assert clean is not None
    assert clean["dirty"] is False
    assert len(clean["commit"]) == 40
    assert len(clean["repository_sha256"]) == 64

    implementation.write_text("VALUE = 2\n", encoding="utf-8")
    tracked_dirty = git_identity(implementation)
    assert tracked_dirty is not None
    assert tracked_dirty["dirty"] is True
    assert tracked_dirty["worktree_sha256"] != clean["worktree_sha256"]

    control_state = repo / ".codex" / "session.json"
    control_state.parent.mkdir()
    control_state.write_text('{"cursor":1}\n', encoding="utf-8")
    with_control_state = git_identity(implementation)
    assert with_control_state is not None
    assert with_control_state["worktree_sha256"] == tracked_dirty["worktree_sha256"]

    untracked = repo / "configs" / "experiment.yaml"
    untracked.parent.mkdir()
    untracked.write_text("seed: 7\n", encoding="utf-8")
    with_untracked = git_identity(implementation)
    assert with_untracked is not None
    assert with_untracked["worktree_sha256"] != tracked_dirty["worktree_sha256"]

    script = repo / "scripts" / "run.sh"
    script.parent.mkdir()
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    non_executable = git_identity(implementation)
    assert non_executable is not None
    script.chmod(0o755)
    executable = git_identity(implementation)
    assert executable is not None
    assert executable["worktree_sha256"] != non_executable["worktree_sha256"]

    first_target = repo / "targets" / "first.txt"
    second_target = repo / "targets" / "second.txt"
    first_target.parent.mkdir()
    first_target.write_text("first\n", encoding="utf-8")
    second_target.write_text("second\n", encoding="utf-8")
    link = repo / "current-target"
    link.symlink_to(first_target.relative_to(repo))
    first_link = git_identity(implementation)
    assert first_link is not None
    link.unlink()
    link.symlink_to(second_target.relative_to(repo))
    second_link = git_identity(implementation)
    assert second_link is not None
    assert second_link["worktree_sha256"] != first_link["worktree_sha256"]


def test_pretrain_records_input_and_loss_tokens_and_rejects_lineage_mismatch(tmp_path):
    stream = list(("accounted training text " * 20).encode())
    checkpoint_path = tmp_path / "pretrain.pt"
    lineage = {
        "version": 1,
        "config_sha256": "config-a",
        "data_sha256": "data-a",
        "tokenizer_sha256": "tokenizer-a",
        "git": {"commit": "abc"},
    }
    _, metrics = pretrain(
        _model(),
        stream,
        ByteTokenizer(),
        steps=1,
        batch_size=2,
        seq_len=8,
        checkpoint_path=checkpoint_path,
        lineage=lineage,
        return_metrics=True,
        log=lambda *_: None,
    )
    assert metrics["token_accounting"] == {
        "input_tokens": 16,
        "loss_tokens": 16,
        "sources": {"train": {"input_tokens": 16, "loss_tokens": 16}},
    }
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["lineage"] == lineage
    assert checkpoint["tokens_seen"] == checkpoint["token_accounting"]["loss_tokens"]

    mismatched = {**lineage, "data_sha256": "data-b"}
    with pytest.raises(ValueError, match="data_sha256"):
        pretrain(
            _model(),
            stream,
            ByteTokenizer(),
            steps=2,
            batch_size=2,
            seq_len=8,
            resume_from=checkpoint_path,
            lineage=mismatched,
            log=lambda *_: None,
        )


class _CountedRows:
    def __init__(self, token: int):
        self.token = token

    def sample_batch_with_counts(self, batch_size, rng, device):
        x = torch.full((batch_size, 6), self.token, dtype=torch.long, device=device)
        y = x.clone()
        y[:, -2:] = IGNORE
        return x, y, batch_size * 5, batch_size * 4


def test_midtrain_accounting_is_per_source_and_resumes(tmp_path):
    mixture = ScheduledMixture(
        [
            MixtureSource("general", _CountedRows(65), 0.5, 0.5),
            MixtureSource("agent", _CountedRows(66), 0.5, 0.5),
        ]
    )
    checkpoint_path = tmp_path / "midtrain.pt"
    lineage = {"version": 1, "config_sha256": "same"}
    _, first_metrics = midtrain(
        _model(),
        mixture,
        steps=2,
        batch_size=2,
        accum_steps=2,
        checkpoint_path=checkpoint_path,
        checkpoint_every=1,
        lineage=lineage,
        return_metrics=True,
        log=lambda *_: None,
    )
    assert sum(first_metrics["source_draws"].values()) == 4
    assert first_metrics["token_accounting"]["input_tokens"] == 40
    assert first_metrics["token_accounting"]["loss_tokens"] == 32
    for source, draws in first_metrics["source_draws"].items():
        assert first_metrics["token_accounting"]["sources"][source] == {
            "input_tokens": draws * 10,
            "loss_tokens": draws * 8,
        }

    _, resumed_metrics = midtrain(
        _model(),
        mixture,
        steps=3,
        batch_size=2,
        accum_steps=2,
        checkpoint_path=checkpoint_path,
        resume_from=checkpoint_path,
        lineage=lineage,
        return_metrics=True,
        log=lambda *_: None,
    )
    assert resumed_metrics["steps_completed"] == 3
    assert sum(resumed_metrics["source_draws"].values()) == 6
    assert resumed_metrics["token_accounting"]["input_tokens"] == 60
    assert resumed_metrics["token_accounting"]["loss_tokens"] == 48


def test_midtrain_records_deterministic_pre_post_heldout_loss_and_accuracy(tmp_path):
    train = ScheduledMixture([MixtureSource("train", _CountedRows(65), 1.0, 1.0)])
    heldout = [MixtureSource("heldout", _CountedRows(66), 1.0, 1.0)]
    checkpoint_path = tmp_path / "midtrain.pt"

    _, metrics = midtrain(
        _model(),
        train,
        steps=1,
        batch_size=2,
        checkpoint_path=checkpoint_path,
        eval_sources=heldout,
        eval_batches=2,
        eval_batch_size=2,
        eval_seed=77,
        return_metrics=True,
        log=lambda *_: None,
    )

    evaluation = metrics["heldout_eval"]
    assert evaluation["contract"] == {
        "kind": "deterministic_teacher_forced_next_token",
        "sources": ["heldout"],
        "batches_per_source": 2,
        "batch_size": 2,
        "seed": 77,
        "same_draws_pre_post": True,
    }
    for phase in ("pre", "post"):
        assert evaluation[phase]["loss_tokens"] == 16
        assert 0.0 <= evaluation[phase]["token_accuracy"] <= 1.0
        assert torch.isfinite(torch.tensor(evaluation[phase]["mean_loss"]))
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert saved["heldout_eval"] == evaluation


def test_midtrain_resume_preserves_original_pre_eval_after_final_periodic_crash(
    tmp_path,
    monkeypatch,
):
    module = importlib.import_module("localagent.train.midtrain")
    train = ScheduledMixture([MixtureSource("train", _CountedRows(65), 1.0, 1.0)])
    heldout = [MixtureSource("heldout", _CountedRows(66), 1.0, 1.0)]
    checkpoint_path = tmp_path / "midtrain.pt"
    real_evaluate = module._evaluate_sources
    calls = 0

    def crash_on_post(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated post-eval crash")
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(module, "_evaluate_sources", crash_on_post)
    with pytest.raises(RuntimeError, match="simulated post-eval crash"):
        module.midtrain(
            _model(),
            train,
            steps=1,
            batch_size=2,
            checkpoint_path=checkpoint_path,
            checkpoint_every=1,
            eval_sources=heldout,
            eval_batches=1,
            eval_batch_size=2,
            eval_seed=88,
            return_metrics=True,
            log=lambda *_: None,
        )
    periodic = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    original_pre = periodic["heldout_eval"]["pre"]
    assert periodic["heldout_eval"]["post"] is None

    resume_calls = 0

    def count_resume_eval(*args, **kwargs):
        nonlocal resume_calls
        resume_calls += 1
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(module, "_evaluate_sources", count_resume_eval)
    _, metrics = module.midtrain(
        _model(),
        train,
        steps=1,
        batch_size=2,
        checkpoint_path=checkpoint_path,
        resume_from=checkpoint_path,
        eval_sources=heldout,
        eval_batches=1,
        eval_batch_size=2,
        eval_seed=88,
        return_metrics=True,
        log=lambda *_: None,
    )

    assert resume_calls == 1  # post only; pre comes from the periodic checkpoint
    assert metrics["heldout_eval"]["pre"] == original_pre
    assert metrics["heldout_eval"]["post"] is not None


def test_sft_reports_dataset_and_realized_assistant_loss_tokens_per_source():
    tokenizer = ByteTokenizer()
    samples = [
        Sample("text", "text", "A", "text", "one"),
        Sample("text", "text", "BB", "text", "two"),
    ]
    rendered = [render_sft(sample, tokenizer) for sample in samples]
    expected_input = sum(len(ids) - 1 for ids, _ in rendered)
    expected_loss = sum(sum(label != IGNORE for label in labels[1:]) for _, labels in rendered)

    _, _, _, metrics = sft(
        _model(),
        samples,
        tokenizer,
        steps=1,
        batch_size=2,
        shuffle=False,
        sample_sources=["file-a.jsonl", "file-b.jsonl"],
        return_metrics=True,
        log=lambda *_: None,
    )
    dataset = metrics["dataset_token_accounting"]["main"]
    realized = metrics["token_accounting"]
    assert dataset["input_tokens"] == realized["input_tokens"] == expected_input
    assert dataset["loss_tokens"] == realized["loss_tokens"] == expected_loss
    assert set(realized["sources"]) == {"file-a.jsonl", "file-b.jsonl"}
    assert all(source["loss_tokens"] > 0 for source in realized["sources"].values())


def test_sft_fixed_input_width_is_used_and_checkpointed(tmp_path) -> None:
    tokenizer = ByteTokenizer()
    model = _model()
    observed_widths: list[int] = []
    original_forward = model.forward

    def recording_forward(tokens, *args, **kwargs):
        observed_widths.append(int(tokens.shape[1]))
        return original_forward(tokens, *args, **kwargs)

    model.forward = recording_forward
    checkpoint_path = tmp_path / "fixed-width-sft.pt"
    sft(
        model,
        [Sample("text", "text", "A", "text", "one")],
        tokenizer,
        steps=1,
        batch_size=1,
        shuffle=False,
        pad_to_input_tokens=48,
        checkpoint_path=checkpoint_path,
        log=lambda *_: None,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert observed_widths == [48]
    assert checkpoint["training_contract"]["pad_to_input_tokens"] == 48
