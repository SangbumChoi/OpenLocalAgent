"""Focused tests for SFT's MPS-only memory cleanup boundaries."""

from __future__ import annotations

import importlib

import pytest
import torch
from torch import nn

from openlocalagent.data.render import IGNORE
from openlocalagent.train.loop import pad_batch as real_pad_batch


class _Tokenizer:
    pad_id = 0


class _PerfectNextTokenModel:
    def __init__(self, events: list[str]) -> None:
        self.training = True
        self._events = events

    def to(self, _device):
        return self

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def __call__(self, tokens: torch.Tensor):
        self._events.append("forward")
        predictions = (tokens + 1).remainder(4)
        logits = torch.full((*tokens.shape, 4), -10.0)
        logits.scatter_(2, predictions.unsqueeze(-1), 10.0)
        return logits, None


@pytest.mark.parametrize(
    ("device", "expected_events"),
    [
        ("cpu", []),
        ("cuda", []),
        ("mps", ["synchronize", "empty_cache"]),
    ],
)
def test_mps_cache_helper_is_backend_specific(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    expected_events: list[str],
) -> None:
    module = importlib.import_module("openlocalagent.train.sft")
    events: list[str] = []
    monkeypatch.setattr(torch.mps, "synchronize", lambda: events.append("synchronize"))
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: events.append("empty_cache"))

    module._mps_synchronize_and_empty_cache(device)

    assert events == expected_events


def test_mps_boundary_clears_gradients_without_changing_cpu_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("openlocalagent.train.sft")
    layer = nn.Linear(2, 2)
    for parameter in layer.parameters():
        parameter.grad = torch.ones_like(parameter)
    events: list[str] = []
    monkeypatch.setattr(torch.mps, "synchronize", lambda: events.append("synchronize"))
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: events.append("empty_cache"))

    module._clear_mps_gradients_and_cache("cpu", layer)
    assert all(parameter.grad is not None for parameter in layer.parameters())
    assert events == []

    module._clear_mps_gradients_and_cache("mps", layer)
    assert all(parameter.grad is None for parameter in layer.parameters())
    assert events == ["synchronize", "empty_cache"]


@pytest.mark.parametrize(
    ("device", "expected_events"),
    [
        ("cpu", ["forward", "forward"]),
        (
            "mps",
            [
                "forward",
                "synchronize",
                "empty_cache",
                "forward",
                "synchronize",
                "empty_cache",
            ],
        ),
    ],
)
def test_heldout_evaluation_cleans_each_mps_batch_without_metric_drift(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    expected_events: list[str],
) -> None:
    module = importlib.import_module("openlocalagent.train.sft")
    events: list[str] = []
    model = _PerfectNextTokenModel(events)
    rows = [
        ([0, 1, 2], [IGNORE, 1, 2]),
        ([1, 2, 3], [IGNORE, 2, 3]),
    ]
    row_iterator = iter(rows)

    monkeypatch.setattr(
        module,
        "render_conversation_rows",
        lambda *_args, **_kwargs: [next(row_iterator)],
    )
    monkeypatch.setattr(
        module,
        "pad_batch",
        lambda batch, pad_id, _device, **kwargs: real_pad_batch(
            batch,
            pad_id,
            "cpu",
            **kwargs,
        ),
    )

    monkeypatch.setattr(torch.mps, "synchronize", lambda: events.append("synchronize"))
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: events.append("empty_cache"))

    metrics = module._evaluate_conversations(
        model,
        [object(), object()],
        _Tokenizer(),
        max_seq_len=8,
        batch_size=1,
        device=device,
    )

    assert events == expected_events
    assert metrics["rows"] == 2
    assert metrics["assistant_loss_tokens"] == 4
    assert metrics["assistant_token_accuracy"] == 1.0
    assert metrics["assistant_sequence_accuracy"] == 1.0
    assert model.training is True
