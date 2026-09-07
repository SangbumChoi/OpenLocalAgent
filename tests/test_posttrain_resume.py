"""Crash-safe exact-resume contracts for the two post-training kernels."""

from __future__ import annotations

import copy
import importlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from openlocalagent.data.synth.agent_synth import Sample
from openlocalagent.data.schema import Conversation, Message, Role
from openlocalagent.model import LocalAgentLM, ModelConfig
from openlocalagent.model.tokenizer import ByteTokenizer
from openlocalagent.train.resume_git_receipt import build_resume_git_receipt
from openlocalagent.train.rl import grpo
from openlocalagent.train.rl import run as run_rl
from openlocalagent.train.sft import run as run_sft
from openlocalagent.train.sft import sft
from openlocalagent.train.stage_data import LINEAGE_VERSION, sha256_file, tokenizer_identity


def _write_unpickle_marker(path: str) -> None:
    Path(path).write_text("unsafe", encoding="utf-8")


class _MaliciousCheckpoint:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return _write_unpickle_marker, (str(self.marker),)


def _sft_config(*, max_seq_len: int = 128) -> ModelConfig:
    return ModelConfig(
        name="posttrain-resume-test",
        vocab_size=256,
        d_model=16,
        embed_dim=8,
        n_layers=1,
        n_loops=1,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden=32,
        max_seq_len=max_seq_len,
        dropout=0.0,
    )


def _tool_sample(prompt: str = "Search for Seoul") -> Sample:
    target = json.dumps(
        {"name": "web_search", "arguments": {"query": "Seoul"}},
        separators=(",", ":"),
        sort_keys=True,
    )
    return Sample(
        category="search",
        group="web_search",
        prompt=prompt,
        kind="tool",
        target=target,
        ref_name="web_search",
        ref_args='{"query":"Seoul"}',
    )


def _text_sample(*, prompt: str = "Say a", target: str = "a") -> Sample:
    return Sample(
        category="text",
        group="text",
        prompt=prompt,
        kind="text",
        target=target,
    )


def _model_from_state(
    cfg: ModelConfig,
    state: dict[str, torch.Tensor],
) -> LocalAgentLM:
    model = LocalAgentLM(cfg)
    model.load_state_dict(state)
    return model


def _assert_state_equal(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> None:
    assert actual.keys() == expected.keys()
    for name, expected_tensor in expected.items():
        torch.testing.assert_close(actual[name], expected_tensor, rtol=0, atol=0)


def _write_yaml(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _write_conversations(path: Path, conversations: list[Conversation]) -> None:
    path.write_text(
        "".join(f"{conversation.to_json()}\n" for conversation in conversations),
        encoding="utf-8",
    )


def _write_parent_checkpoint(
    path: Path,
    cfg: ModelConfig,
    state: dict[str, torch.Tensor],
    *,
    stage: str,
) -> None:
    tokenizer_sha256 = tokenizer_identity("byte", vocab_size=256)["sha256"]
    torch.save(
        {
            "cfg": cfg.__dict__,
            "state_dict": state,
            "stage": stage,
            "tokenizer": {"kind": "byte", "sha256": tokenizer_sha256},
            "lineage": {
                "version": LINEAGE_VERSION,
                "stage": stage,
                "tokenizer_sha256": tokenizer_sha256,
            },
        },
        path,
    )


def _write_sft_runner_with_eval(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    cfg = _sft_config()
    model_config_path = tmp_path / "model.yaml"
    _write_yaml(model_config_path, cfg.__dict__)
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_conversations(
        train_path,
        [
            Conversation(
                messages=[
                    Message(role=Role.user, content="Say a"),
                    Message(role=Role.assistant, content="a"),
                ]
            )
        ],
    )
    _write_conversations(
        eval_path,
        [
            Conversation(
                messages=[
                    Message(role=Role.user, content="Say c"),
                    Message(role=Role.assistant, content="c"),
                ]
            )
        ],
    )
    torch.manual_seed(505)
    parent_path = tmp_path / "midtrain.pt"
    _write_parent_checkpoint(
        parent_path,
        cfg,
        copy.deepcopy(LocalAgentLM(cfg).state_dict()),
        stage="midtrain",
    )
    out_dir = tmp_path / "sft"
    config_path = tmp_path / "sft.yaml"
    _write_yaml(
        config_path,
        {
            "stage": "sft",
            "model_config": str(model_config_path),
            "init_from": str(parent_path),
            "data": {
                "conversations": [str(train_path)],
                "eval_conversations": [str(eval_path)],
                "tokenizer": {"kind": "byte"},
                "seq_len": cfg.max_seq_len,
                "shuffle": True,
            },
            "optim": {"lr": 1e-3},
            "schedule": {"type": "cosine", "warmup_steps": 0, "total_steps": 1},
            "batch": {"micro_batch_size": 1, "grad_accum_steps": 1},
            "heads": {
                "joint_tool_pointer": False,
                "train_route_head": False,
                "train_dense_selector": False,
            },
            "runtime": {"device": "cpu", "dtype": "fp32", "seed": 31},
            "evaluation": {"batch_size": 1},
            "log": {"out_dir": str(out_dir), "ckpt_every": 1},
        },
    )
    return config_path, out_dir, eval_path


def _sft_lineage(worktree_sha256: str) -> dict:
    return {
        "version": LINEAGE_VERSION,
        "stage": "sft",
        "config_sha256": "a" * 64,
        "model_config_sha256": "b" * 64,
        "data_sha256": "c" * 64,
        "tokenizer_sha256": "d" * 64,
        "parent_checkpoint_sha256": "e" * 64,
        "git": {
            "commit": "1" * 64,
            "repository_sha256": "2" * 64,
            "dirty": True,
            "worktree_sha256": worktree_sha256,
        },
    }


def test_sft_periodic_resume_matches_uninterrupted_model_heads_and_accounting(
    tmp_path,
) -> None:
    cfg = _sft_config()
    torch.manual_seed(101)
    initial_state = copy.deepcopy(LocalAgentLM(cfg).state_dict())
    sample = _tool_sample()

    uninterrupted_path = tmp_path / "uninterrupted-sft.pt"
    uninterrupted = _model_from_state(cfg, initial_state)
    expected_history, expected_tool_head, expected_ptr_head, expected_metrics = sft(
        uninterrupted,
        [sample],
        ByteTokenizer(),
        steps=4,
        batch_size=1,
        lr=1e-3,
        warmup=0,
        joint_tool_head=True,
        seed=29,
        checkpoint_path=uninterrupted_path,
        checkpoint_every=1,
        return_metrics=True,
        log=lambda *_: None,
    )

    interrupted_path = tmp_path / "interrupted-sft.pt"
    interrupted = _model_from_state(cfg, initial_state)
    real_forward = interrupted.forward
    forward_calls = 0

    def crash_in_third_step(*args, **kwargs):
        nonlocal forward_calls
        forward_calls += 1
        if forward_calls == 5:
            raise RuntimeError("simulated SFT interruption")
        return real_forward(*args, **kwargs)

    interrupted.forward = crash_in_third_step
    with pytest.raises(RuntimeError, match="simulated SFT interruption"):
        sft(
            interrupted,
            [sample],
            ByteTokenizer(),
            steps=4,
            batch_size=1,
            lr=1e-3,
            warmup=0,
            joint_tool_head=True,
            seed=29,
            checkpoint_path=interrupted_path,
            checkpoint_every=1,
            return_metrics=True,
            log=lambda *_: None,
        )
    periodic = torch.load(interrupted_path, map_location="cpu", weights_only=False)
    assert periodic["step"] == 1

    resumed = _model_from_state(cfg, initial_state)
    actual_history, actual_tool_head, actual_ptr_head, actual_metrics = sft(
        resumed,
        [sample],
        ByteTokenizer(),
        steps=4,
        batch_size=1,
        lr=1e-3,
        warmup=0,
        joint_tool_head=True,
        seed=29,
        checkpoint_path=interrupted_path,
        resume_from=interrupted_path,
        return_metrics=True,
        log=lambda *_: None,
    )

    assert actual_history == pytest.approx(expected_history, rel=0, abs=0)
    assert actual_metrics == expected_metrics
    _assert_state_equal(resumed.state_dict(), uninterrupted.state_dict())
    _assert_state_equal(actual_tool_head.state_dict(), expected_tool_head.state_dict())
    _assert_state_equal(actual_ptr_head.state_dict(), expected_ptr_head.state_dict())
    uninterrupted_checkpoint = torch.load(
        uninterrupted_path,
        map_location="cpu",
        weights_only=False,
    )
    resumed_checkpoint = torch.load(
        interrupted_path,
        map_location="cpu",
        weights_only=False,
    )
    assert (
        resumed_checkpoint["resume_integrity_sha256"]
        == (uninterrupted_checkpoint["resume_integrity_sha256"])
    )
    assert resumed_checkpoint["sampling_state"] == uninterrupted_checkpoint["sampling_state"]
    assert resumed_checkpoint["token_accounting"] == (uninterrupted_checkpoint["token_accounting"])


def test_sft_git_receipt_preserves_exact_resume_and_relabels_first_new_checkpoint(
    tmp_path,
) -> None:
    cfg = _sft_config()
    torch.manual_seed(303)
    initial_state = copy.deepcopy(LocalAgentLM(cfg).state_dict())
    sample = _text_sample()
    recorded_lineage = _sft_lineage("3" * 64)
    current_lineage = _sft_lineage("4" * 64)

    reference_path = tmp_path / "reference.pt"
    reference_model = _model_from_state(cfg, initial_state)
    expected_history, _, _, expected_metrics = sft(
        reference_model,
        [sample],
        ByteTokenizer(),
        steps=4,
        batch_size=1,
        lr=1e-3,
        warmup=0,
        joint_tool_head=False,
        seed=37,
        checkpoint_path=reference_path,
        lineage=current_lineage,
        return_metrics=True,
        log=lambda *_: None,
    )

    resumed_path = tmp_path / "resumed.pt"
    interrupted_model = _model_from_state(cfg, initial_state)
    real_forward = interrupted_model.forward
    forward_calls = 0

    def crash_in_third_step(*args, **kwargs):
        nonlocal forward_calls
        forward_calls += 1
        if forward_calls == 3:
            raise RuntimeError("simulated SFT interruption")
        return real_forward(*args, **kwargs)

    interrupted_model.forward = crash_in_third_step
    with pytest.raises(RuntimeError, match="simulated SFT interruption"):
        sft(
            interrupted_model,
            [sample],
            ByteTokenizer(),
            steps=4,
            batch_size=1,
            lr=1e-3,
            warmup=0,
            joint_tool_head=False,
            seed=37,
            checkpoint_path=resumed_path,
            checkpoint_every=1,
            lineage=recorded_lineage,
            log=lambda *_: None,
        )
    checkpoint_sha256 = sha256_file(resumed_path)
    with pytest.raises(ValueError, match="lineage mismatch: git"):
        sft(
            _model_from_state(cfg, initial_state),
            [sample],
            ByteTokenizer(),
            steps=4,
            batch_size=1,
            lr=1e-3,
            warmup=0,
            joint_tool_head=False,
            seed=37,
            checkpoint_path=resumed_path,
            resume_from=resumed_path,
            lineage=current_lineage,
            log=lambda *_: None,
        )
    assert sha256_file(resumed_path) == checkpoint_sha256

    receipt = build_resume_git_receipt(
        checkpoint_sha256=checkpoint_sha256,
        recorded_lineage=recorded_lineage,
        expected_lineage=current_lineage,
        stage="sft",
        reason="Only fail-closed SFT resume startup validation changed.",
        evidence=["focused exact-resume test"],
    )
    resumed_model = _model_from_state(cfg, initial_state)
    actual_history, _, _, actual_metrics = sft(
        resumed_model,
        [sample],
        ByteTokenizer(),
        steps=4,
        batch_size=1,
        lr=1e-3,
        warmup=0,
        joint_tool_head=False,
        seed=37,
        checkpoint_path=resumed_path,
        resume_from=resumed_path,
        lineage=current_lineage,
        resume_git_receipt=receipt,
        resume_checkpoint_sha256=checkpoint_sha256,
        return_metrics=True,
        log=lambda *_: None,
    )

    assert actual_history == pytest.approx(expected_history, rel=0, abs=0)
    assert actual_metrics == expected_metrics
    _assert_state_equal(resumed_model.state_dict(), reference_model.state_dict())
    resumed_checkpoint = torch.load(resumed_path, map_location="cpu", weights_only=True)
    assert resumed_checkpoint["lineage"] == current_lineage


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_optimizer", "incomplete"),
        ("model_weight", "integrity mismatch"),
        ("token_accounting", "integrity mismatch"),
    ],
)
def test_sft_resume_rejects_incomplete_or_tampered_state(
    tmp_path,
    mutation,
    message,
) -> None:
    cfg = _sft_config()
    torch.manual_seed(44)
    initial_state = copy.deepcopy(LocalAgentLM(cfg).state_dict())
    checkpoint_path = tmp_path / f"{mutation}.pt"
    sft(
        _model_from_state(cfg, initial_state),
        [_text_sample()],
        ByteTokenizer(),
        steps=1,
        batch_size=1,
        joint_tool_head=False,
        seed=7,
        checkpoint_path=checkpoint_path,
        log=lambda *_: None,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if mutation == "missing_optimizer":
        checkpoint.pop("optimizer")
    elif mutation == "model_weight":
        first = next(iter(checkpoint["state_dict"]))
        checkpoint["state_dict"][first].view(-1)[0] += 1
    else:
        checkpoint["token_accounting"]["input_tokens"] += 1
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match=message):
        sft(
            _model_from_state(cfg, initial_state),
            [_text_sample()],
            ByteTokenizer(),
            steps=1,
            batch_size=1,
            joint_tool_head=False,
            seed=7,
            checkpoint_path=checkpoint_path,
            resume_from=checkpoint_path,
            log=lambda *_: None,
        )


@pytest.mark.parametrize("stage", ["sft", "rl"])
def test_posttrain_resume_never_executes_checkpoint_pickle(tmp_path, stage: str) -> None:
    marker = tmp_path / f"{stage}-executed.txt"
    malicious_path = tmp_path / f"{stage}-malicious.pt"
    torch.save(_MaliciousCheckpoint(marker), malicious_path)
    model = LocalAgentLM(_sft_config())
    output_path = tmp_path / f"{stage}-latest.pt"

    with pytest.raises((pickle.UnpicklingError, RuntimeError)):
        if stage == "sft":
            sft(
                model,
                [_text_sample()],
                ByteTokenizer(),
                steps=1,
                batch_size=1,
                warmup=0,
                checkpoint_path=output_path,
                resume_from=malicious_path,
            )
        else:
            grpo(
                model,
                [_text_sample()],
                ByteTokenizer(),
                steps=1,
                prompts_per_step=1,
                group_size=2,
                max_new=4,
                checkpoint_path=output_path,
                resume_from=malicious_path,
            )

    assert not marker.exists()


def test_sft_resume_rejects_data_tokenizer_and_parent_lineage_drift(tmp_path) -> None:
    cfg = _sft_config()
    torch.manual_seed(55)
    initial_state = copy.deepcopy(LocalAgentLM(cfg).state_dict())
    checkpoint_path = tmp_path / "sft-lineage.pt"
    lineage = {
        "version": 1,
        "stage": "sft",
        "parent_checkpoint_sha256": "a" * 64,
    }
    tokenizer_metadata = {"kind": "byte", "sha256": "b" * 64}
    sft(
        _model_from_state(cfg, initial_state),
        [_text_sample()],
        ByteTokenizer(),
        steps=1,
        batch_size=1,
        joint_tool_head=False,
        seed=9,
        checkpoint_path=checkpoint_path,
        lineage=lineage,
        tokenizer_metadata=tokenizer_metadata,
        log=lambda *_: None,
    )

    with pytest.raises(ValueError, match="training contract mismatch"):
        sft(
            _model_from_state(cfg, initial_state),
            [_text_sample(prompt="Say a changed value")],
            ByteTokenizer(),
            steps=1,
            batch_size=1,
            joint_tool_head=False,
            seed=9,
            checkpoint_path=checkpoint_path,
            resume_from=checkpoint_path,
            lineage=lineage,
            tokenizer_metadata=tokenizer_metadata,
            log=lambda *_: None,
        )
    with pytest.raises(ValueError, match="tokenizer metadata mismatch"):
        sft(
            _model_from_state(cfg, initial_state),
            [_text_sample()],
            ByteTokenizer(),
            steps=1,
            batch_size=1,
            joint_tool_head=False,
            seed=9,
            checkpoint_path=checkpoint_path,
            resume_from=checkpoint_path,
            lineage=lineage,
            tokenizer_metadata={"kind": "byte", "sha256": "c" * 64},
            log=lambda *_: None,
        )
    with pytest.raises(ValueError, match="lineage mismatch"):
        sft(
            _model_from_state(cfg, initial_state),
            [_text_sample()],
            ByteTokenizer(),
            steps=1,
            batch_size=1,
            joint_tool_head=False,
            seed=9,
            checkpoint_path=checkpoint_path,
            resume_from=checkpoint_path,
            lineage={
                **lineage,
                "parent_checkpoint_sha256": "d" * 64,
            },
            tokenizer_metadata=tokenizer_metadata,
            log=lambda *_: None,
        )


class _TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(256, 8)
        self.proj = nn.Linear(8, 256)
        self.cfg = SimpleNamespace(max_seq_len=64)

    def forward(self, tokens):
        return self.proj(self.embed(tokens)), None


def _policy_from_state(state: dict[str, torch.Tensor]) -> _TinyPolicy:
    policy = _TinyPolicy()
    policy.load_state_dict(state)
    return policy


def _sampled_ab_rollout(
    _model,
    tok,
    _prompt_ids,
    _max_new,
    _temperature,
    _device,
    generator=None,
    amp_dtype=torch.float32,
):
    del amp_dtype
    if generator is None:
        return [ord("a"), tok.eos_id]
    bit = int(torch.randint(0, 2, (1,), generator=generator))
    return [ord("a") if bit else ord("b"), tok.eos_id]


def test_grpo_periodic_resume_restores_reference_generator_and_optimizer(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("openlocalagent.train.rl._rollout", _sampled_ab_rollout)
    torch.manual_seed(303)
    initial_state = copy.deepcopy(_TinyPolicy().state_dict())
    sample = _text_sample()

    uninterrupted_path = tmp_path / "uninterrupted-rl.pt"
    uninterrupted = _policy_from_state(initial_state)
    expected_history, expected_metrics = grpo(
        uninterrupted,
        [sample],
        ByteTokenizer(),
        steps=4,
        prompts_per_step=1,
        group_size=4,
        lr=1e-3,
        max_new=3,
        seed=17,
        kl_beta=0.1,
        checkpoint_path=uninterrupted_path,
        checkpoint_every=1,
        return_metrics=True,
        log=lambda *_: None,
    )

    interrupted_path = tmp_path / "interrupted-rl.pt"
    rollout_calls = 0

    def crash_in_third_step(*args, **kwargs):
        nonlocal rollout_calls
        rollout_calls += 1
        if rollout_calls == 9:
            raise RuntimeError("simulated RL interruption")
        return _sampled_ab_rollout(*args, **kwargs)

    monkeypatch.setattr("openlocalagent.train.rl._rollout", crash_in_third_step)
    with pytest.raises(RuntimeError, match="simulated RL interruption"):
        grpo(
            _policy_from_state(initial_state),
            [sample],
            ByteTokenizer(),
            steps=4,
            prompts_per_step=1,
            group_size=4,
            lr=1e-3,
            max_new=3,
            seed=17,
            kl_beta=0.1,
            checkpoint_path=interrupted_path,
            checkpoint_every=1,
            return_metrics=True,
            log=lambda *_: None,
        )
    periodic = torch.load(interrupted_path, map_location="cpu", weights_only=False)
    assert periodic["step"] == 1
    assert periodic["reference_state_dict"] is not None

    monkeypatch.setattr("openlocalagent.train.rl._rollout", _sampled_ab_rollout)
    resumed = _policy_from_state(initial_state)
    actual_history, actual_metrics = grpo(
        resumed,
        [sample],
        ByteTokenizer(),
        steps=4,
        prompts_per_step=1,
        group_size=4,
        lr=1e-3,
        max_new=3,
        seed=17,
        kl_beta=0.1,
        checkpoint_path=interrupted_path,
        resume_from=interrupted_path,
        return_metrics=True,
        log=lambda *_: None,
    )

    assert actual_history == pytest.approx(expected_history, rel=0, abs=0)
    assert actual_metrics == expected_metrics
    _assert_state_equal(resumed.state_dict(), uninterrupted.state_dict())
    uninterrupted_checkpoint = torch.load(
        uninterrupted_path,
        map_location="cpu",
        weights_only=False,
    )
    resumed_checkpoint = torch.load(
        interrupted_path,
        map_location="cpu",
        weights_only=False,
    )
    assert (
        resumed_checkpoint["resume_integrity_sha256"]
        == (uninterrupted_checkpoint["resume_integrity_sha256"])
    )
    assert (
        resumed_checkpoint["prompt_accounting"] == (uninterrupted_checkpoint["prompt_accounting"])
    )
    _assert_state_equal(
        resumed_checkpoint["reference_state_dict"],
        uninterrupted_checkpoint["reference_state_dict"],
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_generator", "incomplete"),
        ("reference_weight", "integrity mismatch"),
        ("prompt_accounting", "integrity mismatch"),
    ],
)
def test_grpo_resume_rejects_incomplete_or_tampered_state(
    tmp_path,
    monkeypatch,
    mutation,
    message,
) -> None:
    monkeypatch.setattr("openlocalagent.train.rl._rollout", _sampled_ab_rollout)
    torch.manual_seed(404)
    initial_state = copy.deepcopy(_TinyPolicy().state_dict())
    checkpoint_path = tmp_path / f"{mutation}.pt"
    grpo(
        _policy_from_state(initial_state),
        [_text_sample()],
        ByteTokenizer(),
        steps=1,
        prompts_per_step=1,
        group_size=4,
        max_new=3,
        seed=17,
        kl_beta=0.1,
        checkpoint_path=checkpoint_path,
        log=lambda *_: None,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if mutation == "missing_generator":
        checkpoint.pop("rollout_generator_state")
    elif mutation == "reference_weight":
        first = next(iter(checkpoint["reference_state_dict"]))
        checkpoint["reference_state_dict"][first].view(-1)[0] += 1
    else:
        checkpoint["prompt_accounting"]["generated_tokens"] += 1
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match=message):
        grpo(
            _policy_from_state(initial_state),
            [_text_sample()],
            ByteTokenizer(),
            steps=1,
            prompts_per_step=1,
            group_size=4,
            max_new=3,
            seed=17,
            kl_beta=0.1,
            checkpoint_path=checkpoint_path,
            resume_from=checkpoint_path,
            log=lambda *_: None,
        )


def test_grpo_resume_rejects_reward_row_drift(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("openlocalagent.train.rl._rollout", _sampled_ab_rollout)
    torch.manual_seed(505)
    initial_state = copy.deepcopy(_TinyPolicy().state_dict())
    checkpoint_path = tmp_path / "rl-data-drift.pt"
    grpo(
        _policy_from_state(initial_state),
        [_text_sample()],
        ByteTokenizer(),
        steps=1,
        prompts_per_step=1,
        group_size=4,
        max_new=3,
        seed=17,
        kl_beta=0.1,
        checkpoint_path=checkpoint_path,
        log=lambda *_: None,
    )

    with pytest.raises(ValueError, match="training contract mismatch"):
        grpo(
            _policy_from_state(initial_state),
            [_text_sample(target="b")],
            ByteTokenizer(),
            steps=1,
            prompts_per_step=1,
            group_size=4,
            max_new=3,
            seed=17,
            kl_beta=0.1,
            checkpoint_path=checkpoint_path,
            resume_from=checkpoint_path,
            log=lambda *_: None,
        )


def test_sft_stage_runner_resumes_periodic_checkpoint_without_relabeling_parent(
    tmp_path,
    monkeypatch,
) -> None:
    cfg = _sft_config()
    model_config_path = tmp_path / "model.yaml"
    _write_yaml(model_config_path, cfg.__dict__)
    conversations_path = tmp_path / "train.jsonl"
    _write_conversations(
        conversations_path,
        [
            Conversation(
                messages=[
                    Message(role=Role.user, content="Say a"),
                    Message(role=Role.assistant, content="a"),
                ]
            )
        ],
    )
    torch.manual_seed(606)
    initial_state = copy.deepcopy(LocalAgentLM(cfg).state_dict())
    parent_path = tmp_path / "midtrain.pt"
    _write_parent_checkpoint(parent_path, cfg, initial_state, stage="midtrain")

    def config_for(out_dir: Path) -> dict:
        return {
            "stage": "sft",
            "model_config": str(model_config_path),
            "init_from": str(parent_path),
            "data": {
                "conversations": [str(conversations_path)],
                "tokenizer": {"kind": "byte"},
                "seq_len": cfg.max_seq_len,
                "shuffle": True,
            },
            "optim": {"lr": 1e-3},
            "schedule": {"type": "cosine", "warmup_steps": 0, "total_steps": 4},
            "batch": {"micro_batch_size": 1, "grad_accum_steps": 1},
            "heads": {
                "joint_tool_pointer": False,
                "train_route_head": False,
                "train_dense_selector": False,
            },
            "runtime": {
                "device": "cpu",
                "dtype": "fp32",
                "seed": 23,
            },
            "log": {"out_dir": str(out_dir), "ckpt_every": 1},
        }

    reference_dir = tmp_path / "sft-reference"
    reference_config = tmp_path / "sft-reference.yaml"
    _write_yaml(reference_config, config_for(reference_dir))
    run_sft(str(reference_config))

    resumed_dir = tmp_path / "sft-resumed"
    resumed_config = tmp_path / "sft-resumed.yaml"
    _write_yaml(resumed_config, config_for(resumed_dir))
    original_forward = LocalAgentLM.forward
    forward_calls = 0

    def crash_in_third_step(self, *args, **kwargs):
        nonlocal forward_calls
        forward_calls += 1
        if forward_calls == 3:
            raise RuntimeError("simulated SFT runner interruption")
        return original_forward(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(LocalAgentLM, "forward", crash_in_third_step)
        with pytest.raises(RuntimeError, match="simulated SFT runner interruption"):
            run_sft(str(resumed_config))
    periodic = torch.load(resumed_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert periodic["stage"] == "sft"
    assert periodic["step"] == 1
    assert periodic["lineage"]["parent_checkpoint_sha256"] is not None

    run_sft(str(resumed_config), resume=True)
    reference = torch.load(reference_dir / "latest.pt", map_location="cpu", weights_only=False)
    resumed = torch.load(resumed_dir / "latest.pt", map_location="cpu", weights_only=False)
    _assert_state_equal(resumed["state_dict"], reference["state_dict"])
    assert resumed["loss_history"] == pytest.approx(
        reference["loss_history"],
        rel=0,
        abs=0,
    )
    assert resumed["token_accounting"] == reference["token_accounting"]
    assert resumed["sampling_state"] == reference["sampling_state"]
    assert (
        resumed["lineage"]["parent_checkpoint_sha256"]
        == (reference["lineage"]["parent_checkpoint_sha256"])
    )


def test_sft_stage_runner_starts_fresh_optimizer_child_from_completed_sft(
    tmp_path,
) -> None:
    cfg = _sft_config()
    model_config_path = tmp_path / "model.yaml"
    _write_yaml(model_config_path, cfg.__dict__)
    conversations_path = tmp_path / "train.jsonl"
    _write_conversations(
        conversations_path,
        [
            Conversation(
                messages=[
                    Message(role=Role.user, content="Say a"),
                    Message(role=Role.assistant, content="a"),
                ]
            )
        ],
    )
    torch.manual_seed(607)
    midtrain_parent = tmp_path / "midtrain.pt"
    _write_parent_checkpoint(
        midtrain_parent,
        cfg,
        copy.deepcopy(LocalAgentLM(cfg).state_dict()),
        stage="midtrain",
    )

    def config_for(
        *,
        init_from: Path,
        out_dir: Path,
        steps: int,
        continuation: bool,
    ) -> dict:
        return {
            "stage": "sft",
            "model_config": str(model_config_path),
            "init_from": str(init_from),
            **(
                {"continuation": {"mode": "fresh_optimizer_sft_child_v1"}}
                if continuation
                else {}
            ),
            "data": {
                "conversations": [str(conversations_path)],
                "tokenizer": {"kind": "byte"},
                "seq_len": cfg.max_seq_len,
                "shuffle": True,
            },
            "optim": {"lr": 1e-3},
            "schedule": {"type": "cosine", "warmup_steps": 0, "total_steps": steps},
            "batch": {"micro_batch_size": 1, "grad_accum_steps": 1},
            "heads": {
                "joint_tool_pointer": False,
                "train_route_head": False,
                "train_dense_selector": False,
            },
            "runtime": {"device": "cpu", "dtype": "fp32", "seed": 29},
            "log": {"out_dir": str(out_dir), "ckpt_every": 1},
        }

    parent_dir = tmp_path / "sft-parent"
    parent_config = tmp_path / "sft-parent.yaml"
    _write_yaml(
        parent_config,
        config_for(
            init_from=midtrain_parent,
            out_dir=parent_dir,
            steps=2,
            continuation=False,
        ),
    )
    run_sft(str(parent_config))
    parent_path = parent_dir / "latest.pt"
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)

    child_dir = tmp_path / "sft-child"
    child_config = tmp_path / "sft-child.yaml"
    _write_yaml(
        child_config,
        config_for(
            init_from=parent_path,
            out_dir=child_dir,
            steps=1,
            continuation=True,
        ),
    )
    run_sft(str(child_config))
    child = torch.load(child_dir / "latest.pt", map_location="cpu", weights_only=True)

    parent_optimizer_steps = {
        int(state["step"]) for state in parent["optimizer"]["state"].values()
    }
    child_optimizer_steps = {
        int(state["step"]) for state in child["optimizer"]["state"].values()
    }
    assert parent_optimizer_steps == {2}
    assert child_optimizer_steps == {1}
    assert child["continuation"] == {"mode": "fresh_optimizer_sft_child_v1"}
    assert child["lineage"]["parent_checkpoint_sha256"] == sha256_file(parent_path)
    assert child["training_contract"]["steps"] == 1
    assert child["step"] == 0


def test_sft_continuation_rejects_an_incomplete_parent_fixed_horizon(
    tmp_path,
) -> None:
    module = importlib.import_module("openlocalagent.train.sft")
    cfg = _sft_config()
    model_config_path = tmp_path / "model.yaml"
    _write_yaml(model_config_path, cfg.__dict__)
    conversations_path = tmp_path / "train.jsonl"
    _write_conversations(
        conversations_path,
        [
            Conversation(
                messages=[
                    Message(role=Role.user, content="Say a"),
                    Message(role=Role.assistant, content="a"),
                ]
            )
        ],
    )
    torch.manual_seed(608)
    midtrain_parent = tmp_path / "midtrain.pt"
    _write_parent_checkpoint(
        midtrain_parent,
        cfg,
        copy.deepcopy(LocalAgentLM(cfg).state_dict()),
        stage="midtrain",
    )
    parent_dir = tmp_path / "sft-parent"
    parent_config = tmp_path / "sft-parent.yaml"
    base = {
        "stage": "sft",
        "model_config": str(model_config_path),
        "init_from": str(midtrain_parent),
        "data": {
            "conversations": [str(conversations_path)],
            "tokenizer": {"kind": "byte"},
            "seq_len": cfg.max_seq_len,
            "shuffle": True,
        },
        "optim": {"lr": 1e-3},
        "schedule": {"type": "cosine", "warmup_steps": 0, "total_steps": 2},
        "batch": {"micro_batch_size": 1, "grad_accum_steps": 1},
        "heads": {
            "joint_tool_pointer": False,
            "train_route_head": False,
            "train_dense_selector": False,
        },
        "runtime": {"device": "cpu", "dtype": "fp32", "seed": 31},
        "log": {"out_dir": str(parent_dir), "ckpt_every": 1},
    }
    _write_yaml(parent_config, base)
    run_sft(str(parent_config))
    parent_path = parent_dir / "latest.pt"
    partial = torch.load(parent_path, map_location="cpu", weights_only=True)
    partial["step"] = 0
    partial["loss_history"] = partial["loss_history"][:1]
    partial["sampling_state"]["completed_steps"] = 1
    partial["sampling_state"]["completed_microbatches"] = 1
    partial["resume_integrity_sha256"] = module._sealed_resume_sha256(partial)
    torch.save(partial, parent_path)

    child_config = tmp_path / "sft-child.yaml"
    child = copy.deepcopy(base)
    child["init_from"] = str(parent_path)
    child["continuation"] = {"mode": "fresh_optimizer_sft_child_v1"}
    child["schedule"]["total_steps"] = 1
    child["log"]["out_dir"] = str(tmp_path / "sft-child")
    _write_yaml(child_config, child)

    with pytest.raises(ValueError, match="completed parent fixed horizon"):
        run_sft(str(child_config))


def test_sft_stage_runner_fresh_computes_pre_but_resume_reuses_sealed_baseline(
    tmp_path,
    monkeypatch,
) -> None:
    module = importlib.import_module("openlocalagent.train.sft")
    config_path, out_dir, _ = _write_sft_runner_with_eval(tmp_path)
    real_evaluate = module._evaluate_conversations
    evaluation_calls = 0

    def count_evaluation(*args, **kwargs):
        nonlocal evaluation_calls
        evaluation_calls += 1
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(module, "_evaluate_conversations", count_evaluation)
    run_sft(str(config_path))
    assert evaluation_calls == 2  # fresh pre + post
    fresh = torch.load(out_dir / "latest.pt", map_location="cpu", weights_only=True)
    original_pre = copy.deepcopy(fresh["heldout_baseline"]["pre"])

    evaluation_calls = 0
    run_sft(str(config_path), resume=True)
    assert evaluation_calls == 1  # post only; pre comes from the sealed checkpoint
    resumed = torch.load(out_dir / "latest.pt", map_location="cpu", weights_only=True)
    assert resumed["heldout_baseline"]["pre"] == original_pre
    assert resumed["heldout_eval"]["pre"] == original_pre
    assert resumed["heldout_eval"]["post"] is not None


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("unsealed_pre_tamper", "integrity mismatch"),
        ("sealed_malformed_baseline", "must contain exactly contract and pre"),
    ],
)
def test_sft_stage_runner_rejects_bad_sealed_baseline_before_training(
    tmp_path,
    monkeypatch,
    mutation: str,
    message: str,
) -> None:
    module = importlib.import_module("openlocalagent.train.sft")
    config_path, out_dir, _ = _write_sft_runner_with_eval(tmp_path)
    run_sft(str(config_path))
    checkpoint_path = out_dir / "latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if mutation == "unsealed_pre_tamper":
        checkpoint["heldout_baseline"]["pre"]["mean_loss"] += 1.0
    else:
        checkpoint["heldout_baseline"] = {
            "contract": checkpoint["heldout_baseline"]["contract"]
        }
        checkpoint["resume_integrity_sha256"] = module._sealed_resume_sha256(checkpoint)
    torch.save(checkpoint, checkpoint_path)

    def must_not_evaluate(*_args, **_kwargs):
        raise AssertionError("held-out evaluation started before resume baseline validation")

    def must_not_train(*_args, **_kwargs):
        raise AssertionError("SFT training started before resume baseline validation")

    monkeypatch.setattr(module, "_evaluate_conversations", must_not_evaluate)
    monkeypatch.setattr(module, "sft", must_not_train)
    with pytest.raises(ValueError, match=message):
        run_sft(str(config_path), resume=True)


def test_sft_stage_runner_rejects_current_heldout_contract_drift_before_training(
    tmp_path,
    monkeypatch,
) -> None:
    module = importlib.import_module("openlocalagent.train.sft")
    config_path, _, eval_path = _write_sft_runner_with_eval(tmp_path)
    run_sft(str(config_path))
    _write_conversations(
        eval_path,
        [
            Conversation(
                messages=[
                    Message(role=Role.user, content="Say d"),
                    Message(role=Role.assistant, content="d"),
                ]
            )
        ],
    )

    def must_not_evaluate(*_args, **_kwargs):
        raise AssertionError("held-out evaluation started despite a contract mismatch")

    def must_not_train(*_args, **_kwargs):
        raise AssertionError("SFT training started despite a contract mismatch")

    monkeypatch.setattr(module, "_evaluate_conversations", must_not_evaluate)
    monkeypatch.setattr(module, "sft", must_not_train)
    with pytest.raises(ValueError, match="heldout baseline contract mismatch"):
        run_sft(str(config_path), resume=True)


def test_rl_stage_runner_resumes_with_original_reference_and_prompt_progress(
    tmp_path,
    monkeypatch,
) -> None:
    cfg = _sft_config()
    model_config_path = tmp_path / "model.yaml"
    _write_yaml(model_config_path, cfg.__dict__)
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_conversations(
        train_path,
        [
            Conversation(
                messages=[
                    Message(role=Role.user, content="Say a"),
                    Message(role=Role.assistant, content="a"),
                ]
            )
        ],
    )
    _write_conversations(
        eval_path,
        [
            Conversation(
                messages=[
                    Message(role=Role.user, content="Say c"),
                    Message(role=Role.assistant, content="c"),
                ]
            )
        ],
    )
    torch.manual_seed(707)
    initial_state = copy.deepcopy(LocalAgentLM(cfg).state_dict())
    parent_path = tmp_path / "sft.pt"
    _write_parent_checkpoint(parent_path, cfg, initial_state, stage="sft")

    def config_for(out_dir: Path) -> dict:
        return {
            "stage": "rl",
            "model_config": str(model_config_path),
            "init_from": str(parent_path),
            "data": {
                "conversations": [str(train_path)],
                "eval_conversations": [str(eval_path)],
                "tokenizer": {"kind": "byte"},
            },
            "environment": {
                "name": "canonical_toolcalls",
                "learned_judge": False,
            },
            "rollout": {
                "prompts_per_step": 1,
                "group_size": 4,
                "max_new_tokens": 3,
                "temperature": 1.0,
            },
            "policy": {
                "clip_ratio": 0.2,
                "kl_beta": 0.1,
                "epochs_per_rollout": 1,
            },
            "reward": {"format_weight": 0.1, "truncation_penalty": 0.05},
            "optim": {"lr": 1e-3},
            "schedule": {"total_steps": 4},
            "runtime": {
                "device": "cpu",
                "dtype": "fp32",
                "seed": 17,
            },
            "log": {"out_dir": str(out_dir), "ckpt_every": 1},
        }

    monkeypatch.setattr("openlocalagent.train.rl._rollout", _sampled_ab_rollout)
    reference_dir = tmp_path / "rl-reference"
    reference_config = tmp_path / "rl-reference.yaml"
    _write_yaml(reference_config, config_for(reference_dir))
    run_rl(str(reference_config))

    resumed_dir = tmp_path / "rl-resumed"
    resumed_config = tmp_path / "rl-resumed.yaml"
    _write_yaml(resumed_config, config_for(resumed_dir))
    rollout_calls = 0

    def crash_in_third_step(*args, **kwargs):
        nonlocal rollout_calls
        generator = kwargs.get("generator")
        if generator is None and len(args) >= 7:
            generator = args[6]
        if generator is not None:
            rollout_calls += 1
            if rollout_calls == 9:
                raise RuntimeError("simulated RL runner interruption")
        return _sampled_ab_rollout(*args, **kwargs)

    monkeypatch.setattr("openlocalagent.train.rl._rollout", crash_in_third_step)
    with pytest.raises(RuntimeError, match="simulated RL runner interruption"):
        run_rl(str(resumed_config))
    periodic = torch.load(resumed_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert periodic["stage"] == "rl"
    assert periodic["step"] == 1
    assert periodic["reference_state_dict"] is not None

    monkeypatch.setattr("openlocalagent.train.rl._rollout", _sampled_ab_rollout)
    run_rl(str(resumed_config), resume=True)
    reference = torch.load(reference_dir / "latest.pt", map_location="cpu", weights_only=False)
    resumed = torch.load(resumed_dir / "latest.pt", map_location="cpu", weights_only=False)
    _assert_state_equal(resumed["state_dict"], reference["state_dict"])
    _assert_state_equal(
        resumed["reference_state_dict"],
        reference["reference_state_dict"],
    )
    assert resumed["reward_history"] == pytest.approx(
        reference["reward_history"],
        rel=0,
        abs=0,
    )
    assert resumed["rl_accounting"] == reference["rl_accounting"]
    assert resumed["prompt_accounting"] == reference["prompt_accounting"]
    assert resumed["prompt_schedule_state"] == reference["prompt_schedule_state"]

    def holdout_only_rollout(*args, **kwargs):
        generator = kwargs.get("generator")
        if generator is None and len(args) >= 7:
            generator = args[6]
        if generator is not None:
            pytest.fail("completed RL runner resume must not sample a training rollout")
        return _sampled_ab_rollout(*args, **kwargs)

    monkeypatch.setattr("openlocalagent.train.rl._rollout", holdout_only_rollout)
    run_rl(str(resumed_config), resume=True)
    completed_resume = torch.load(
        resumed_dir / "latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    _assert_state_equal(completed_resume["state_dict"], reference["state_dict"])
    assert completed_resume["rl_accounting"] == reference["rl_accounting"]
