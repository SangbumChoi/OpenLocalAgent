"""Focused contracts for assistant-token-normalized SFT gradient accumulation."""

from __future__ import annotations

import copy
import importlib

import pytest
import torch

from openlocalagent.data.synth.agent_synth import Sample
from openlocalagent.data.render import IGNORE, render_sft
from openlocalagent.model import LocalAgentLM, ModelConfig
from openlocalagent.model.tokenizer import ByteTokenizer
from openlocalagent.train.sft import sft
from openlocalagent.train.stage_data import sha256_file
from openlocalagent.train.stage_sampling import (
    SFT_LOSS_NORMALIZATION_MICROBATCH,
    SFT_LOSS_NORMALIZATION_UPDATE_TOKENS,
)


def _config() -> ModelConfig:
    return ModelConfig(
        name="sft-loss-normalization-test",
        vocab_size=256,
        d_model=16,
        embed_dim=8,
        n_layers=1,
        n_loops=1,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden=32,
        max_seq_len=128,
        dropout=0.0,
    )


def _samples() -> list[Sample]:
    return [
        Sample("text", "text", "Reply briefly", "text", "x"),
        Sample(
            "text",
            "text",
            "Reply with the longer phrase",
            "text",
            "a substantially longer assistant target",
        ),
    ]


def _model_from_state(
    state: dict[str, torch.Tensor],
) -> LocalAgentLM:
    model = LocalAgentLM(_config())
    model.load_state_dict(state)
    return model


def _assert_state_equal(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> None:
    assert actual.keys() == expected.keys()
    for name, expected_tensor in expected.items():
        torch.testing.assert_close(actual[name], expected_tensor, rtol=0, atol=0)


def test_assistant_token_update_mean_matches_full_effective_batch_ce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Uneven microbatches produce the gradient of one full token-mean CE batch."""

    tokenizer = ByteTokenizer()
    samples = _samples()
    loss_tokens = [
        sum(label != IGNORE for label in labels[1:])
        for _ids, labels in (render_sft(sample, tokenizer) for sample in samples)
    ]
    assert loss_tokens[0] != loss_tokens[1]

    optimizer_instances = []

    class CaptureOptimizer(torch.optim.Optimizer):
        def __init__(self, params, lr, **_kwargs):
            super().__init__(params, {"lr": lr})
            self.captured_gradients: list[torch.Tensor | None] | None = None
            optimizer_instances.append(self)

        @torch.no_grad()
        def step(self, closure=None):
            if closure is not None:
                with torch.enable_grad():
                    closure()
            self.captured_gradients = [
                None if parameter.grad is None else parameter.grad.detach().clone()
                for group in self.param_groups
                for parameter in group["params"]
            ]

    sft_module = importlib.import_module("openlocalagent.train.sft")
    monkeypatch.setattr(sft_module.torch.optim, "AdamW", CaptureOptimizer)

    torch.manual_seed(610)
    initial_state = copy.deepcopy(LocalAgentLM(_config()).state_dict())
    accumulated_model = _model_from_state(initial_state)
    full_batch_model = _model_from_state(initial_state)

    accumulated_history, _, _ = sft(
        accumulated_model,
        samples,
        tokenizer,
        steps=1,
        batch_size=1,
        accum_steps=2,
        shuffle=False,
        warmup=0,
        joint_tool_head=False,
        loss_normalization=SFT_LOSS_NORMALIZATION_UPDATE_TOKENS,
        log=lambda *_: None,
    )
    full_batch_history, _, _ = sft(
        full_batch_model,
        samples,
        tokenizer,
        steps=1,
        batch_size=2,
        accum_steps=1,
        shuffle=False,
        warmup=0,
        joint_tool_head=False,
        loss_normalization=SFT_LOSS_NORMALIZATION_MICROBATCH,
        log=lambda *_: None,
    )
    assert accumulated_history == pytest.approx(full_batch_history, rel=2e-5, abs=2e-7)

    assert len(optimizer_instances) == 2
    accumulated_gradients = optimizer_instances[0].captured_gradients
    full_batch_gradients = optimizer_instances[1].captured_gradients
    assert accumulated_gradients is not None
    assert full_batch_gradients is not None
    assert len(accumulated_gradients) == len(full_batch_gradients)
    for accumulated, full_batch in zip(
        accumulated_gradients,
        full_batch_gradients,
        strict=True,
    ):
        assert accumulated is not None
        assert full_batch is not None
        torch.testing.assert_close(accumulated, full_batch, rtol=2e-5, atol=2e-7)


def test_default_microbatch_normalization_preserves_unweighted_loss_sum() -> None:
    tokenizer = ByteTokenizer()
    samples = _samples()
    torch.manual_seed(612)
    model = LocalAgentLM(_config())
    expected = 0.0
    with torch.no_grad():
        for sample in samples:
            ids, labels = render_sft(sample, tokenizer)
            x = torch.tensor([ids[:-1]], dtype=torch.long)
            y = torch.tensor([labels[1:]], dtype=torch.long)
            _, loss = model(x, targets=y)
            expected += float(loss)

    history, _, _ = sft(
        model,
        samples,
        tokenizer,
        steps=1,
        batch_size=1,
        accum_steps=2,
        shuffle=False,
        warmup=0,
        joint_tool_head=False,
        loss_normalization=SFT_LOSS_NORMALIZATION_MICROBATCH,
        log=lambda *_: None,
    )

    assert history == pytest.approx([expected], rel=2e-5, abs=2e-7)


def test_private_update_limit_keeps_production_horizon_in_partial_checkpoint(
    tmp_path,
) -> None:
    checkpoint_path = tmp_path / "latest.pt"
    history, _, _, metrics = sft(
        LocalAgentLM(_config()),
        _samples(),
        ByteTokenizer(),
        steps=3,
        batch_size=1,
        accum_steps=1,
        shuffle=False,
        warmup=0,
        joint_tool_head=False,
        checkpoint_path=checkpoint_path,
        checkpoint_every=1,
        archive_checkpoints=True,
        return_metrics=True,
        _max_optimizer_updates=1,
        log=lambda *_: None,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    archive = torch.load(
        tmp_path / "latest.step-00000001.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert len(history) == 1
    assert checkpoint["step"] == archive["step"] == 0
    assert checkpoint["training_contract"]["steps"] == archive["training_contract"]["steps"] == 3
    assert checkpoint["sampling_state"]["completed_steps"] == 1
    assert metrics["fixed_horizon_progress"] == {
        "planned_optimizer_updates": 3,
        "completed_optimizer_updates": 1,
        "partial": True,
    }


@pytest.mark.parametrize(
    "incompatible_path",
    ["joint_heads", "knowledge_distillation"],
)
def test_assistant_token_update_mean_rejects_non_lm_loss_paths(
    incompatible_path: str,
) -> None:
    kwargs = (
        {"joint_tool_head": True}
        if incompatible_path == "joint_heads"
        else {"teacher": object()}
    )
    with pytest.raises(ValueError, match="LM-only SFT path"):
        sft(
            LocalAgentLM(_config()),
            _samples(),
            ByteTokenizer(),
            steps=1,
            batch_size=1,
            loss_normalization=SFT_LOSS_NORMALIZATION_UPDATE_TOKENS,
            log=lambda *_: None,
            **kwargs,
        )


def test_assistant_token_update_mean_is_sealed_and_exactly_resumable(
    tmp_path,
) -> None:
    tokenizer = ByteTokenizer()
    samples = _samples()
    torch.manual_seed(611)
    initial_state = copy.deepcopy(LocalAgentLM(_config()).state_dict())

    reference_model = _model_from_state(initial_state)
    expected_history, _, _ = sft(
        reference_model,
        samples,
        tokenizer,
        steps=2,
        batch_size=1,
        accum_steps=1,
        shuffle=False,
        warmup=0,
        joint_tool_head=False,
        loss_normalization=SFT_LOSS_NORMALIZATION_UPDATE_TOKENS,
        log=lambda *_: None,
    )

    checkpoint_path = tmp_path / "token-normalized-sft.pt"
    interrupted_model = _model_from_state(initial_state)
    real_forward = interrupted_model.forward
    forward_calls = 0

    def fail_on_second_update(*args, **kwargs):
        nonlocal forward_calls
        forward_calls += 1
        if forward_calls == 2:
            raise RuntimeError("simulated token-normalized SFT interruption")
        return real_forward(*args, **kwargs)

    interrupted_model.forward = fail_on_second_update
    with pytest.raises(RuntimeError, match="simulated token-normalized SFT interruption"):
        sft(
            interrupted_model,
            samples,
            tokenizer,
            steps=2,
            batch_size=1,
            accum_steps=1,
            shuffle=False,
            warmup=0,
            joint_tool_head=False,
            loss_normalization=SFT_LOSS_NORMALIZATION_UPDATE_TOKENS,
            checkpoint_path=checkpoint_path,
            checkpoint_every=1,
            log=lambda *_: None,
        )

    periodic = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert periodic["step"] == 0
    assert (
        periodic["training_contract"]["loss_normalization"]
        == SFT_LOSS_NORMALIZATION_UPDATE_TOKENS
    )
    periodic_sha256 = sha256_file(checkpoint_path)

    with pytest.raises(ValueError, match="training contract mismatch"):
        sft(
            _model_from_state(initial_state),
            samples,
            tokenizer,
            steps=2,
            batch_size=1,
            accum_steps=1,
            shuffle=False,
            warmup=0,
            joint_tool_head=False,
            loss_normalization=SFT_LOSS_NORMALIZATION_MICROBATCH,
            checkpoint_path=checkpoint_path,
            resume_from=checkpoint_path,
            log=lambda *_: None,
        )
    assert sha256_file(checkpoint_path) == periodic_sha256

    resumed_model = _model_from_state(initial_state)
    actual_history, _, _ = sft(
        resumed_model,
        samples,
        tokenizer,
        steps=2,
        batch_size=1,
        accum_steps=1,
        shuffle=False,
        warmup=0,
        joint_tool_head=False,
        loss_normalization=SFT_LOSS_NORMALIZATION_UPDATE_TOKENS,
        checkpoint_path=checkpoint_path,
        resume_from=checkpoint_path,
        log=lambda *_: None,
    )

    assert actual_history == pytest.approx(expected_history, rel=0, abs=0)
    _assert_state_equal(resumed_model.state_dict(), reference_model.state_dict())
