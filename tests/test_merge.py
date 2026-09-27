"""Tests for weight-space checkpoint merging (model soup + TIES)."""

from __future__ import annotations

import copy

import pytest
import torch

from localagent.model import LocalAgentLM, ModelConfig
from localagent.train.merge import (
    merge_checkpoints,
    model_soup,
    ties_merge,
)


def _tiny_cfg():
    return ModelConfig(vocab_size=256, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
                       ffn_hidden=128, max_seq_len=64)


def _tiny_state_dict(seed: int):
    torch.manual_seed(seed)
    return LocalAgentLM(_tiny_cfg()).state_dict()


# --------------------------------------------------------------------------------------------------
# model soup
# --------------------------------------------------------------------------------------------------
def test_soup_two_is_elementwise_mean():
    a = _tiny_state_dict(1)
    b = _tiny_state_dict(2)
    merged = model_soup([a, b])
    for k in a:
        if a[k].is_floating_point():
            assert torch.allclose(merged[k], (a[k] + b[k]) / 2, atol=1e-6), k
        else:
            assert torch.equal(merged[k], a[k]), k


def test_soup_with_weights_respected():
    a = _tiny_state_dict(1)
    b = _tiny_state_dict(2)
    merged = model_soup([a, b], weights=[0.25, 0.75])
    for k in a:
        if a[k].is_floating_point():
            assert torch.allclose(merged[k], 0.25 * a[k] + 0.75 * b[k], atol=1e-6), k


def test_soup_weights_are_normalized():
    a = _tiny_state_dict(1)
    b = _tiny_state_dict(2)
    # unnormalized weights [1, 3] should behave like [0.25, 0.75]
    merged = model_soup([a, b], weights=[1.0, 3.0])
    expect = model_soup([a, b], weights=[0.25, 0.75])
    for k in a:
        assert torch.allclose(merged[k], expect[k], atol=1e-7), k


def test_soup_of_model_with_itself_is_identity():
    a = _tiny_state_dict(7)
    merged = model_soup([a, copy.deepcopy(a), copy.deepcopy(a)])
    for k in a:
        assert torch.allclose(merged[k], a[k], atol=1e-6), k


def test_soup_key_mismatch_raises():
    a = _tiny_state_dict(1)
    b = _tiny_state_dict(2)
    b2 = dict(b)
    b2.pop(next(iter(b2)))
    with pytest.raises(ValueError):
        model_soup([a, b2])


def test_soup_shape_mismatch_raises():
    a = _tiny_state_dict(1)
    # a model with a different d_model -> different shapes but (mostly) same key names
    torch.manual_seed(2)
    big = LocalAgentLM(ModelConfig(vocab_size=256, d_model=128, n_layers=2, n_heads=4,
                                   n_kv_heads=2, ffn_hidden=128, max_seq_len=64)).state_dict()
    with pytest.raises(ValueError):
        model_soup([a, big])


def test_soup_empty_raises():
    with pytest.raises(ValueError):
        model_soup([])


# --------------------------------------------------------------------------------------------------
# TIES
# --------------------------------------------------------------------------------------------------
def test_ties_density_one_aligned_signs_is_mean_task_vector():
    # Construct base + two checkpoints whose task vectors share a sign on every entry, so with
    # density=1.0 every entry is kept and the elected sign agrees everywhere -> base + mean(tau).
    base = _tiny_state_dict(0)
    a, b = {}, {}
    for k, v in base.items():
        if v.is_floating_point():
            d1 = torch.rand_like(v) + 0.1        # strictly positive task vector
            d2 = torch.rand_like(v) + 0.1
            a[k] = v + d1
            b[k] = v + d2
        else:
            a[k] = v.clone()
            b[k] = v.clone()
    merged = ties_merge([a, b], base, density=1.0)
    for k, v in base.items():
        if v.is_floating_point():
            expect = v + ((a[k] - v) + (b[k] - v)) / 2
            assert torch.allclose(merged[k], expect, atol=1e-6), k


def test_ties_self_merge_recovers_checkpoint():
    base = _tiny_state_dict(0)
    a = _tiny_state_dict(3)
    # density=1.0, single checkpoint: every kept entry agrees with its own sign -> base + tau = a.
    merged = ties_merge([a], base, density=1.0)
    for k, v in a.items():
        assert torch.allclose(merged[k], v, atol=1e-6), k


def test_ties_trim_reduces_to_base_when_density_drops():
    # With density very small only a few entries survive; the rest equal the base exactly.
    base = _tiny_state_dict(0)
    a = _tiny_state_dict(3)
    merged = ties_merge([a], base, density=0.1)
    # at least one tensor must have some entries left untouched (equal to base)
    touched_some_base = False
    for k, v in base.items():
        if v.is_floating_point() and v.numel() > 10:
            at_base = torch.isclose(merged[k], v, atol=1e-9)
            if at_base.any():
                touched_some_base = True
    assert touched_some_base


def test_ties_opposing_signs_cancel_to_base():
    # Two checkpoints with exactly opposing task vectors -> the elected sign is 0 (perfect tie) so
    # no checkpoint agrees and the entry stays at base. Use a zero base so (base ± d) - base is the
    # exact task vector ±d with no float-subtraction residual.
    base = {k: torch.zeros_like(v) for k, v in _tiny_state_dict(0).items()}
    a, b = {}, {}
    for k, v in base.items():
        if v.is_floating_point():
            d = torch.ones_like(v)
            a[k] = v + d
            b[k] = v - d
        else:
            a[k] = v.clone()
            b[k] = v.clone()
    merged = ties_merge([a, b], base, density=1.0)
    for k, v in base.items():
        if v.is_floating_point():
            assert torch.allclose(merged[k], v, atol=1e-6), k


def test_ties_density_out_of_range_raises():
    base = _tiny_state_dict(0)
    a = _tiny_state_dict(1)
    with pytest.raises(ValueError):
        ties_merge([a], base, density=0.0)
    with pytest.raises(ValueError):
        ties_merge([a], base, density=1.5)


# --------------------------------------------------------------------------------------------------
# checkpoint-level merge + heads
# --------------------------------------------------------------------------------------------------
def _ckpt(seed: int, with_heads: bool = True):
    cfg = _tiny_cfg()
    torch.manual_seed(seed)
    model = LocalAgentLM(cfg)
    ck = {"cfg": cfg.__dict__, "state_dict": model.state_dict(),
          "tool_head": None, "ptr_head": None}
    if with_heads:
        from localagent.agent.pointer_head import PointerHead
        from localagent.agent.tool_head import ToolHead
        torch.manual_seed(seed + 100)
        ck["tool_head"] = ToolHead(cfg.d_model).state_dict()
        ck["ptr_head"] = PointerHead(cfg.d_model).state_dict()
    return ck


def test_merge_checkpoints_soup_averages_heads():
    a = _ckpt(1)
    b = _ckpt(2)
    merged = merge_checkpoints([a, b], method="soup")
    assert merged["tool_head"] is not None and merged["ptr_head"] is not None
    for k in a["tool_head"]:
        assert torch.allclose(merged["tool_head"][k],
                              (a["tool_head"][k] + b["tool_head"][k]) / 2, atol=1e-6)
    for k in a["state_dict"]:
        if a["state_dict"][k].is_floating_point():
            assert torch.allclose(merged["state_dict"][k],
                                  (a["state_dict"][k] + b["state_dict"][k]) / 2, atol=1e-6)


def test_merge_checkpoints_ties_requires_base():
    a = _ckpt(1)
    b = _ckpt(2)
    with pytest.raises(ValueError):
        merge_checkpoints([a, b], method="ties", base=None)


def test_merge_checkpoints_ties_with_base_runs():
    base = _ckpt(0)
    a = _ckpt(1)
    b = _ckpt(2)
    merged = merge_checkpoints([a, b], method="ties", base=base, density=0.5)
    assert set(merged["state_dict"].keys()) == set(a["state_dict"].keys())
    assert merged["tool_head"] is not None


def test_merge_checkpoints_arch_mismatch_raises():
    a = _ckpt(1)
    b = _ckpt(2)
    b["cfg"] = dict(b["cfg"])
    b["cfg"]["d_model"] = 128
    with pytest.raises(ValueError):
        merge_checkpoints([a, b], method="soup")


def test_merge_checkpoints_loads_into_model():
    a = _ckpt(1)
    b = _ckpt(2)
    merged = merge_checkpoints([a, b], method="soup")
    cfg = _tiny_cfg()
    model = LocalAgentLM(cfg)
    model.load_state_dict(merged["state_dict"])  # must load cleanly
    logits, _ = model(torch.randint(0, cfg.vocab_size, (1, 8)))
    assert logits.shape[-1] == cfg.vocab_size
