"""Strict exact-name parameter freezing for supervised fine-tuning."""

from __future__ import annotations

import copy
import importlib
from pathlib import Path

import pytest
import torch
import yaml

from openlocalagent.data.synth.agent_synth import Sample
from openlocalagent.data.corpus.decision_quota_order import QUOTA_SAMPLING_MODE
from openlocalagent.data.schema import Conversation, Message, Role
from openlocalagent.model import LocalAgentLM, ModelConfig
from openlocalagent.model.tokenizer import ByteTokenizer
from openlocalagent.train.replay_sampling import (
    PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE,
)
from openlocalagent.train.sft import (
    SFT_CONTINUATION_MODE,
    _sealed_resume_sha256,
    _validate_parent_anchored_sampling_parent,
    _validate_sft_continuation_parent,
    run as run_sft,
    sft,
)
from openlocalagent.train.stage_data import (
    LINEAGE_VERSION,
    canonical_sha256,
    sha256_file,
    tokenizer_identity,
)


def _config() -> ModelConfig:
    return ModelConfig(
        name="sft-parameter-freeze-test",
        vocab_size=256,
        d_model=16,
        embed_dim=8,
        n_layers=1,
        n_loops=1,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden=32,
        max_seq_len=64,
        dropout=0.0,
    )


def _sample() -> Sample:
    return Sample(
        category="text",
        group="text",
        prompt="Say alpha",
        kind="text",
        target="alpha",
    )


def _model_from_state(
    state: dict[str, torch.Tensor],
) -> LocalAgentLM:
    model = LocalAgentLM(_config())
    model.load_state_dict(state)
    return model


def _assert_tensor_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _assert_state_equal(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> None:
    assert actual.keys() == expected.keys()
    for name, expected_tensor in expected.items():
        _assert_tensor_equal(actual[name], expected_tensor)


def _write_yaml(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _write_parent_checkpoint(
    path: Path,
    cfg: ModelConfig,
    state: dict[str, torch.Tensor],
) -> None:
    tokenizer_sha256 = tokenizer_identity("byte", vocab_size=256)["sha256"]
    torch.save(
        {
            "cfg": cfg.__dict__,
            "state_dict": state,
            "stage": "midtrain",
            "tokenizer": {"kind": "byte", "sha256": tokenizer_sha256},
            "lineage": {
                "version": LINEAGE_VERSION,
                "stage": "midtrain",
                "tokenizer_sha256": tokenizer_sha256,
            },
        },
        path,
    )


def _write_completed_quota_sft_parent(
    path: Path,
    cfg: ModelConfig,
    state: dict[str, torch.Tensor],
    *,
    order_sha256: str,
) -> dict:
    tokenizer_sha256 = tokenizer_identity("byte", vocab_size=256)["sha256"]
    model = LocalAgentLM(cfg)
    model.load_state_dict(state)
    sft(
        model,
        [_sample()],
        ByteTokenizer(),
        steps=1,
        batch_size=1,
        warmup=0,
        joint_tool_head=False,
        shuffle=False,
        checkpoint_path=path,
        tokenizer_metadata={"kind": "byte", "sha256": tokenizer_sha256},
        lineage={
            "version": LINEAGE_VERSION,
            "stage": "sft",
            "tokenizer_sha256": tokenizer_sha256,
        },
        log=lambda *_: None,
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint["training_contract"]["lm_sampling"] = {
        "mode": QUOTA_SAMPLING_MODE,
        "no_replacement": True,
        "ordering": {"order_sha256": order_sha256},
    }
    checkpoint["sampling_state"]["lm_cursor"] = 1
    checkpoint["resume_integrity_sha256"] = _sealed_resume_sha256(checkpoint)
    torch.save(checkpoint, path)
    return checkpoint


def _write_runner_fixture(
    tmp_path: Path,
    *,
    freeze_parameters: object,
) -> tuple[Path, Path, dict[str, torch.Tensor]]:
    cfg = _config()
    model_config_path = tmp_path / "model.yaml"
    _write_yaml(model_config_path, cfg.__dict__)

    conversation_path = tmp_path / "train.jsonl"
    conversation = Conversation(
        messages=[
            Message(role=Role.user, content="Say alpha"),
            Message(role=Role.assistant, content="alpha"),
        ],
        meta={"category": "text", "group": "text", "split": "train"},
    )
    conversation_path.write_text(f"{conversation.to_json()}\n", encoding="utf-8")

    torch.manual_seed(901)
    initial_state = copy.deepcopy(LocalAgentLM(cfg).state_dict())
    parent_path = tmp_path / "midtrain.pt"
    _write_parent_checkpoint(parent_path, cfg, initial_state)

    out_dir = tmp_path / "sft"
    config_path = tmp_path / "sft.yaml"
    _write_yaml(
        config_path,
        {
            "stage": "sft",
            "model_config": str(model_config_path),
            "init_from": str(parent_path),
            "data": {
                "conversations": [str(conversation_path)],
                "tokenizer": {"kind": "byte"},
                "seq_len": cfg.max_seq_len,
                "shuffle": False,
            },
            "optim": {
                "name": "adamw",
                "lr": 1.0e-2,
                "weight_decay": 0.125,
                "grad_clip": 0.25,
                "freeze_parameters": freeze_parameters,
            },
            "schedule": {
                "type": "cosine",
                "warmup_steps": 0,
                "total_steps": 1,
            },
            "batch": {"micro_batch_size": 1, "grad_accum_steps": 1},
            "heads": {
                "joint_tool_pointer": False,
                "train_route_head": False,
                "train_dense_selector": False,
                "example_centroids": False,
            },
            "runtime": {"device": "cpu", "dtype": "fp32", "seed": 41},
            "log": {"out_dir": str(out_dir), "ckpt_every": 1},
        },
    )
    return config_path, out_dir, initial_state


def test_runner_freezes_exact_names_and_seals_the_ordered_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen_names = ["out_proj.weight", "embed.weight"]
    config_path, out_dir, initial_state = _write_runner_fixture(
        tmp_path,
        freeze_parameters=frozen_names,
    )

    clipped_max_norms = []
    clip_grad_norm = torch.nn.utils.clip_grad_norm_

    def recording_clip_grad_norm(parameters, max_norm, *args, **kwargs):
        clipped_max_norms.append(max_norm)
        return clip_grad_norm(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        recording_clip_grad_norm,
    )
    run_sft(str(config_path))

    checkpoint = torch.load(
        out_dir / "latest.pt",
        map_location="cpu",
        weights_only=True,
    )
    trained_state = checkpoint["state_dict"]
    assert trained_state.keys() == initial_state.keys()
    assert checkpoint["training_contract"]["freeze_parameters"] == frozen_names
    expected_optimizer_names = [
        name
        for name, _ in _model_from_state(initial_state).named_parameters()
        if name not in frozen_names
    ]
    assert (
        checkpoint["training_contract"]["optimizer_model_parameter_names"]
        == expected_optimizer_names
    )
    assert checkpoint["training_contract"]["optimizer"] == {
        "kind": "AdamW",
        "betas": [0.9, 0.95],
        "weight_decay": 0.125,
        "grad_clip": 0.25,
    }
    assert checkpoint["optimizer"]["param_groups"][0]["weight_decay"] == 0.125
    assert clipped_max_norms == [0.25]
    for name in frozen_names:
        _assert_tensor_equal(trained_state[name], initial_state[name])

    trainable_names = [name for name in initial_state if name not in frozen_names]
    assert any(
        not torch.equal(trained_state[name], initial_state[name])
        for name in trainable_names
    )
    assert len(checkpoint["optimizer"]["param_groups"]) == 1
    assert len(checkpoint["optimizer"]["param_groups"][0]["params"]) == (
        len(dict(_model_from_state(initial_state).named_parameters()))
        - len(frozen_names)
    )


def test_default_sft_does_not_add_a_freeze_contract_field(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "latest.pt"
    sft(
        LocalAgentLM(_config()),
        [_sample()],
        ByteTokenizer(),
        steps=1,
        batch_size=1,
        lr=1.0e-2,
        warmup=0,
        joint_tool_head=False,
        checkpoint_path=checkpoint_path,
        log=lambda *_: None,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert "freeze_parameters" not in checkpoint["training_contract"]
    assert "optimizer_model_parameter_names" not in checkpoint["training_contract"]
    assert checkpoint["training_contract"]["optimizer"] == {
        "kind": "AdamW",
        "betas": [0.9, 0.95],
        "weight_decay": 0.0,
        "grad_clip": 1.0,
    }
    assert len(checkpoint["optimizer"]["param_groups"][0]["params"]) == len(
        dict(LocalAgentLM(_config()).named_parameters())
    )


def test_completed_parent_seal_matches_and_rejects_each_drifted_pin(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "parent.pt"
    sft(
        LocalAgentLM(_config()),
        [_sample()],
        ByteTokenizer(),
        steps=1,
        batch_size=1,
        warmup=0,
        joint_tool_head=False,
        checkpoint_path=checkpoint_path,
        log=lambda *_: None,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    pins = {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "resume_integrity_sha256": checkpoint["resume_integrity_sha256"],
        "training_contract_sha256": canonical_sha256(
            checkpoint["training_contract"]
        ),
        "lm_sampling_sha256": canonical_sha256(
            checkpoint["training_contract"]["lm_sampling"]
        ),
        "completed_steps": checkpoint["sampling_state"]["completed_steps"],
        "completed_lm_cursor": checkpoint["sampling_state"]["lm_cursor"],
    }
    continuation = {"mode": SFT_CONTINUATION_MODE, "parent": pins}

    assert _validate_sft_continuation_parent(
        checkpoint,
        checkpoint_sha256=pins["checkpoint_sha256"],
        continuation=continuation,
    ) == pins

    for field, value in pins.items():
        drifted = copy.deepcopy(continuation)
        drifted["parent"][field] = (
            "0" * 64 if isinstance(value, str) and value != "0" * 64 else (
                "1" * 64 if isinstance(value, str) else value + 1
            )
        )
        with pytest.raises(
            ValueError,
            match=rf"SFT continuation parent {field} mismatch",
        ):
            _validate_sft_continuation_parent(
                checkpoint,
                checkpoint_sha256=pins["checkpoint_sha256"],
                continuation=drifted,
            )


def _parent_anchored_binding_fixture() -> tuple[dict, dict]:
    order_sha256 = "a" * 64
    checkpoint = {
        "training_contract": {
            "batch_size": 2,
            "accum_steps": 2,
            "lm_sampling": {
                "mode": QUOTA_SAMPLING_MODE,
                "no_replacement": True,
                "ordering": {"order_sha256": order_sha256},
            },
        },
        "sampling_state": {
            "completed_steps": 3,
            "lm_cursor": 12,
        },
    }
    sampling_config = {
        "mode": PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE,
        "parent_prefix_decisions": 12,
        "update_decisions": 4,
        "expected_parent_order_sha256": order_sha256,
    }
    return checkpoint, sampling_config


def test_parent_anchored_sampling_parent_binding_returns_observed_evidence() -> None:
    checkpoint, sampling_config = _parent_anchored_binding_fixture()

    assert _validate_parent_anchored_sampling_parent(
        checkpoint,
        sampling_config,
    ) == {
        "parent_lm_sampling_mode": QUOTA_SAMPLING_MODE,
        "parent_no_replacement": True,
        "parent_order_sha256": "a" * 64,
        "parent_completed_steps": 3,
        "parent_completed_lm_cursor": 12,
        "parent_update_decisions": 4,
    }


def test_parent_anchored_sampling_parent_binding_rejects_every_cross_drift() -> None:
    checkpoint, sampling_config = _parent_anchored_binding_fixture()
    cases = []

    drifted_checkpoint = copy.deepcopy(checkpoint)
    drifted_checkpoint["training_contract"]["lm_sampling"]["mode"] = (
        "source_order_wrapping_v1"
    )
    cases.append(
        (
            drifted_checkpoint,
            sampling_config,
            "parent LM sampling mode mismatch",
        )
    )

    drifted_checkpoint = copy.deepcopy(checkpoint)
    drifted_checkpoint["training_contract"]["lm_sampling"]["no_replacement"] = False
    cases.append(
        (
            drifted_checkpoint,
            sampling_config,
            "must be no-replacement",
        )
    )

    drifted_sampling = copy.deepcopy(sampling_config)
    drifted_sampling["parent_prefix_decisions"] = 11
    cases.append(
        (
            checkpoint,
            drifted_sampling,
            "parent_prefix_decisions mismatch",
        )
    )

    drifted_sampling = copy.deepcopy(sampling_config)
    drifted_sampling["update_decisions"] = 5
    cases.append(
        (
            checkpoint,
            drifted_sampling,
            "update_decisions mismatch",
        )
    )

    drifted_sampling = copy.deepcopy(sampling_config)
    drifted_sampling["expected_parent_order_sha256"] = "b" * 64
    cases.append(
        (
            checkpoint,
            drifted_sampling,
            "parent order SHA-256 mismatch",
        )
    )

    drifted_checkpoint = copy.deepcopy(checkpoint)
    drifted_checkpoint["sampling_state"]["completed_steps"] = 2
    cases.append(
        (
            drifted_checkpoint,
            sampling_config,
            "parent cursor arithmetic mismatch",
        )
    )

    for parent, child_sampling, message in cases:
        with pytest.raises(ValueError, match=message):
            _validate_parent_anchored_sampling_parent(
                parent,
                child_sampling,
            )


@pytest.mark.parametrize(
    ("checkpoint", "sampling_config", "error", "message"),
    [
        (
            None,
            {},
            TypeError,
            "checkpoint must be a mapping",
        ),
        (
            {},
            None,
            TypeError,
            "config must be a mapping",
        ),
        (
            {"training_contract": None},
            {
                "mode": PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE,
                "parent_prefix_decisions": 1,
                "update_decisions": 1,
                "expected_parent_order_sha256": "a" * 64,
            },
            ValueError,
            "training contract must be a mapping",
        ),
    ],
)
def test_parent_anchored_sampling_parent_binding_rejects_malformed_mappings(
    checkpoint,
    sampling_config,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        _validate_parent_anchored_sampling_parent(
            checkpoint,
            sampling_config,
        )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"optimizer_name": "AdamW"}, ValueError, "exactly 'adamw'"),
        ({"optimizer_name": "sgd"}, ValueError, "exactly 'adamw'"),
        ({"optimizer_name": None}, TypeError, "must be a string"),
        ({"weight_decay": True}, TypeError, "finite non-negative"),
        ({"weight_decay": -0.1}, ValueError, "finite non-negative"),
        ({"weight_decay": float("inf")}, ValueError, "finite non-negative"),
        ({"grad_clip": True}, TypeError, "finite positive"),
        ({"grad_clip": 0.0}, ValueError, "finite positive"),
        ({"grad_clip": -0.1}, ValueError, "finite positive"),
        ({"grad_clip": float("nan")}, ValueError, "finite positive"),
    ],
)
def test_sft_rejects_invalid_optimizer_contract(
    kwargs: dict,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        sft(
            LocalAgentLM(_config()),
            [_sample()],
            ByteTokenizer(),
            steps=1,
            batch_size=1,
            warmup=0,
            joint_tool_head=False,
            log=lambda *_: None,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("freeze_parameters", "error", "message"),
    [
        ("embed.weight", TypeError, "must be a list"),
        (("embed.weight",), TypeError, "must be a list"),
        ([1], TypeError, r"freeze_parameters\[0\] must be a string"),
        (
            ["embed.weight", "embed.weight"],
            ValueError,
            "contains duplicate names: embed.weight",
        ),
        (
            ["missing.weight"],
            ValueError,
            "contains unknown model parameter names: missing.weight",
        ),
    ],
)
def test_sft_rejects_invalid_freeze_parameter_lists(
    freeze_parameters,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        sft(
            LocalAgentLM(_config()),
            [_sample()],
            ByteTokenizer(),
            steps=1,
            batch_size=1,
            warmup=0,
            joint_tool_head=False,
            freeze_parameters=freeze_parameters,
            log=lambda *_: None,
        )


def test_sft_rejects_freezing_every_model_parameter() -> None:
    model = LocalAgentLM(_config())
    all_names = [name for name, _ in model.named_parameters()]

    with pytest.raises(ValueError, match="cannot freeze every model parameter"):
        sft(
            model,
            [_sample()],
            ByteTokenizer(),
            steps=1,
            batch_size=1,
            warmup=0,
            joint_tool_head=False,
            freeze_parameters=all_names,
            log=lambda *_: None,
        )

    assert all(parameter.requires_grad for parameter in model.parameters())


def test_runner_rejects_explicit_non_list_freeze_config(tmp_path: Path) -> None:
    config_path, _out_dir, _initial_state = _write_runner_fixture(
        tmp_path,
        freeze_parameters=None,
    )

    with pytest.raises(TypeError, match=r"optim\.freeze_parameters must be a list"):
        run_sft(str(config_path))


def test_frozen_sft_exact_resume_succeeds_and_contract_drift_fails(
    tmp_path: Path,
) -> None:
    frozen_names = ["out_proj.weight", "embed.weight"]
    torch.manual_seed(902)
    initial_state = copy.deepcopy(LocalAgentLM(_config()).state_dict())
    common = {
        "steps": 2,
        "batch_size": 1,
        "lr": 1.0e-2,
        "warmup": 0,
        "joint_tool_head": False,
        "shuffle": False,
        "seed": 43,
        "freeze_parameters": frozen_names,
        "log": lambda *_: None,
    }

    reference_path = tmp_path / "reference.pt"
    reference_model = _model_from_state(initial_state)
    expected_history, _, _ = sft(
        reference_model,
        [_sample()],
        ByteTokenizer(),
        checkpoint_path=reference_path,
        **common,
    )

    resumed_path = tmp_path / "resumed.pt"
    partial_model = _model_from_state(initial_state)
    partial_history, _, _ = sft(
        partial_model,
        [_sample()],
        ByteTokenizer(),
        checkpoint_path=resumed_path,
        _max_optimizer_updates=1,
        **common,
    )
    assert len(partial_history) == 1

    resumed_model = _model_from_state(initial_state)
    actual_history, _, _ = sft(
        resumed_model,
        [_sample()],
        ByteTokenizer(),
        checkpoint_path=resumed_path,
        resume_from=resumed_path,
        **common,
    )
    assert actual_history == pytest.approx(expected_history, rel=0, abs=0)
    _assert_state_equal(resumed_model.state_dict(), reference_model.state_dict())
    resumed_parameters = dict(resumed_model.named_parameters())
    assert all(
        not resumed_parameters[name].requires_grad
        for name in frozen_names
    )
    assert all(
        parameter.requires_grad
        for name, parameter in resumed_parameters.items()
        if name not in frozen_names
    )

    checkpoint = torch.load(resumed_path, map_location="cpu", weights_only=True)
    assert checkpoint["training_contract"]["freeze_parameters"] == frozen_names
    assert checkpoint["training_contract"]["optimizer_model_parameter_names"] == [
        name
        for name, _ in resumed_model.named_parameters()
        if name not in frozen_names
    ]
    for name in frozen_names:
        _assert_tensor_equal(checkpoint["state_dict"][name], initial_state[name])

    checkpoint_sha256 = sha256_file(resumed_path)
    with pytest.raises(ValueError, match="training contract mismatch"):
        sft(
            _model_from_state(initial_state),
            [_sample()],
            ByteTokenizer(),
            checkpoint_path=resumed_path,
            resume_from=resumed_path,
            **{
                **common,
                "freeze_parameters": list(reversed(frozen_names)),
            },
        )
    assert sha256_file(resumed_path) == checkpoint_sha256

    for optimizer_drift in (
        {"weight_decay": 0.125},
        {"grad_clip": 0.25},
    ):
        with pytest.raises(ValueError, match="training contract mismatch"):
            sft(
                _model_from_state(initial_state),
                [_sample()],
                ByteTokenizer(),
                checkpoint_path=resumed_path,
                resume_from=resumed_path,
                **{**common, **optimizer_drift},
            )
        assert sha256_file(resumed_path) == checkpoint_sha256


def test_runner_requires_sealed_parent_for_parent_anchored_sampling(
    tmp_path: Path,
) -> None:
    config_path, out_dir, _initial_state = _write_runner_fixture(
        tmp_path,
        freeze_parameters=[],
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["data"]["sampling"] = {
        "mode": PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE
    }
    _write_yaml(config_path, config)

    with pytest.raises(
        ValueError,
        match=r"requires a sealed continuation\.parent",
    ):
        run_sft(str(config_path))
    assert not out_dir.exists()


def test_runner_dispatches_parent_anchored_pulses_and_validates_update_width(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, _out_dir, initial_state = _write_runner_fixture(
        tmp_path,
        freeze_parameters=[],
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    order_sha256 = "a" * 64
    parent_path = tmp_path / "completed-sft-parent.pt"
    parent_checkpoint = _write_completed_quota_sft_parent(
        parent_path,
        _config(),
        initial_state,
        order_sha256=order_sha256,
    )
    config["init_from"] = str(parent_path)
    config["continuation"] = {
        "mode": SFT_CONTINUATION_MODE,
        "parent": {
            "checkpoint_sha256": sha256_file(parent_path),
            "resume_integrity_sha256": parent_checkpoint[
                "resume_integrity_sha256"
            ],
            "training_contract_sha256": canonical_sha256(
                parent_checkpoint["training_contract"]
            ),
            "lm_sampling_sha256": canonical_sha256(
                parent_checkpoint["training_contract"]["lm_sampling"]
            ),
            "completed_steps": 1,
            "completed_lm_cursor": 1,
        },
    }
    training_source = config["data"]["conversations"][0]
    config["data"].update(
        {
            "conversation_prompt_contract": "openai_full_catalog_v1",
            "conversations": [training_source, training_source],
            "sampling": {
                "mode": PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE,
                "parent_prefix_decisions": 1,
                "update_decisions": 1,
                "expected_parent_order_sha256": order_sha256,
            },
            "shuffle": False,
        }
    )
    config["batch"] = {"micro_batch_size": 2, "grad_accum_steps": 2}
    _write_yaml(config_path, config)

    calls = []

    def fake_parent_anchored_window(
        source_conversations,
        *,
        selected_decisions,
        sampling_config,
    ):
        calls.append(
            {
                "source_count": len(source_conversations),
                "selected_decisions": selected_decisions,
                "sampling_config": dict(sampling_config),
            }
        )
        return ((0, 1),), {"update_layout": {"update_decisions": 3}}

    sft_module = importlib.import_module("openlocalagent.train.sft")
    monkeypatch.setattr(
        sft_module,
        "assert_prompt_contract_tokenizer",
        lambda _tokenizer, contract: contract,
    )
    monkeypatch.setattr(
        sft_module,
        "parent_anchored_format_pulse_sampling_window",
        fake_parent_anchored_window,
    )

    with pytest.raises(
        ValueError,
        match=(
            "parent-anchored format pulse update must equal one complete optimizer "
            "update: update_decisions=3, effective_batch=4"
        ),
    ):
        run_sft(str(config_path))

    assert calls == [
        {
            "source_count": 2,
            "selected_decisions": 4,
            "sampling_config": {
                "mode": PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE,
                "parent_prefix_decisions": 1,
                "update_decisions": 1,
                "expected_parent_order_sha256": order_sha256,
            },
        }
    ]
