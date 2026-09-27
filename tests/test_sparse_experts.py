import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from openlocalagent.cli import _model_info
from openlocalagent.model import LocalAgentLM, ModelConfig
from openlocalagent.train.loop import router_loss_terms


def _sparse_tiny(**overrides) -> ModelConfig:
    values = {
        "name": "sparse-test",
        "vocab_size": 64,
        "d_model": 32,
        "n_layers": 1,
        "n_heads": 4,
        "n_kv_heads": 2,
        "ffn_hidden": 48,
        "max_seq_len": 32,
        "ffn_num_experts": 4,
        "ffn_top_k": 2,
        "router_aux_loss_coef": 0.25,
    }
    values.update(overrides)
    return ModelConfig(**values)


def test_sparse_param_accounting_counts_all_experts_but_only_top_k_active() -> None:
    cfg = _sparse_tiny(n_layers=3, n_loops=2)
    model = LocalAgentLM(cfg)
    expert_parameters = 3 * cfg.d_model * cfg.ffn_hidden
    inactive_expert_parameters = (
        cfg.n_layers * (cfg.ffn_num_experts - cfg.ffn_top_k) * expert_parameters
    )

    assert model.num_params() == cfg.estimate_params()
    assert model.active_num_params() == cfg.estimate_active_params()
    assert cfg.estimate_params() - cfg.estimate_active_params() == inactive_expert_parameters
    assert cfg.estimate_active_params() < cfg.estimate_params()
    # Recurrent loops share the same banks and therefore do not inflate either unique count.
    no_recurrence = ModelConfig(**{**cfg.__dict__, "n_loops": 1})
    assert cfg.estimate_params() - no_recurrence.estimate_params() == cfg.n_loops * cfg.d_model


def test_stable_top_k_executes_only_selected_experts_and_reports_exact_load() -> None:
    torch.manual_seed(7)
    cfg = _sparse_tiny()
    model = LocalAgentLM(cfg).eval()
    routed_ffn = model.blocks[0].ffn
    with torch.no_grad():
        routed_ffn.router.weight.zero_()

    calls = [0] * cfg.ffn_num_experts
    hooks = []
    for expert_index, expert in enumerate(routed_ffn.experts):
        hooks.append(
            expert.register_forward_hook(
                lambda _module, _inputs, _output, i=expert_index: calls.__setitem__(
                    i, calls[i] + 1
                )
            )
        )
    idx = torch.randint(0, cfg.vocab_size, (2, 5))
    with torch.no_grad():
        first_logits, _ = model(idx)
        first_diagnostics = model.routing_diagnostics()
        second_logits, _ = model(idx)
    for hook in hooks:
        hook.remove()

    # Stable ties choose experts 0 and 1. Experts 2 and 3 are never executed (not compute+mask).
    assert calls == [2, 2, 0, 0]
    assert torch.equal(first_logits, second_logits)
    assert first_diagnostics["tokens"] == 10
    assert first_diagnostics["assignments"] == 20
    assert first_diagnostics["expert_counts"] == [10, 10, 0, 0]
    assert first_diagnostics["expert_load"] == [0.5, 0.5, 0.0, 0.0]
    assert first_diagnostics["expert_token_fraction"] == [1.0, 1.0, 0.0, 0.0]
    assert first_diagnostics["active_experts"] == 2
    assert first_diagnostics["dead_experts"] == [2, 3]
    assert first_diagnostics["router_probability"] == [0.25] * 4
    assert first_diagnostics["router_entropy"] == pytest.approx(math.log(4))
    assert first_diagnostics["router_entropy_normalized"] == pytest.approx(1.0)
    assert first_diagnostics["load_balance_loss"] == pytest.approx(1.0)


def test_model_loss_stays_pure_ce_and_training_helper_adds_router_term() -> None:
    torch.manual_seed(11)
    cfg = _sparse_tiny()
    model = LocalAgentLM(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 6))
    targets = torch.randint(0, cfg.vocab_size, (2, 6))

    logits, lm_loss = model(idx, targets=targets)
    expected_ce = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
    optimization_loss, router_aux, router_weighted = router_loss_terms(model, lm_loss)

    assert torch.equal(lm_loss, expected_ce)
    assert router_aux.requires_grad
    assert torch.allclose(router_weighted, cfg.router_aux_loss_coef * router_aux)
    assert torch.allclose(optimization_loss, lm_loss + router_weighted)
    optimization_loss.backward()
    assert model.blocks[0].ffn.router.weight.grad is not None
    assert torch.isfinite(model.blocks[0].ffn.router.weight.grad).all()


def test_dense_default_has_no_router_or_state_dict_changes() -> None:
    cfg = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden=48,
        max_seq_len=32,
    )
    model = LocalAgentLM(cfg)

    assert not cfg.sparse_ffn
    assert model.num_params() == model.active_num_params() == cfg.estimate_params()
    assert model.routing_aux_loss() is None
    assert model.routing_diagnostics()["enabled"] is False
    assert not any("router" in key or ".experts." in key for key in model.state_dict())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"ffn_num_experts": 0}, "ffn_num_experts"),
        ({"ffn_num_experts": 4, "ffn_top_k": 0}, "ffn_top_k"),
        ({"ffn_num_experts": 4, "ffn_top_k": 5}, "ffn_top_k"),
        ({"ffn_num_experts": 4, "router_aux_loss_coef": -0.1}, "router_aux_loss_coef"),
    ],
)
def test_sparse_config_rejects_invalid_router_settings(overrides, message) -> None:
    with pytest.raises(AssertionError, match=message):
        _sparse_tiny(**overrides)


def test_moe_candidate_is_total_budgeted_and_active_matched_to_dense_control() -> None:
    moe = ModelConfig.from_yaml("configs/model/webgpu-44m-moe.yaml")
    dense = ModelConfig.from_yaml("configs/model/webgpu-17m-dense-moe-control.yaml")

    moe.assert_within_budget()
    dense.assert_within_budget()
    assert moe.estimate_params() == 43_862_464
    assert moe.estimate_active_params() == 17_320_384
    assert dense.estimate_params() == dense.estimate_active_params() == 17_297_344
    assert abs(moe.estimate_active_params() / dense.estimate_params() - 1.0) < 0.002
    assert moe.estimate_params() > 2.5 * dense.estimate_params()


def test_model_info_distinguishes_total_and_active_parameters(capsys) -> None:
    _model_info(SimpleNamespace(config="configs/model/webgpu-44m-moe.yaml"))
    output = capsys.readouterr().out

    assert "webgpu-44m-moe: ~43.86M params" in output
    assert "active/token≈17.32M params" in output
    assert "top-2/8 experts" in output


def test_expert_combine_survives_autocast_dtype_split() -> None:
    """Under autocast the experts return bf16 while the residual stream stays fp32.

    `index_add_` refuses that mix outright, so a sparse model that trains fine in fp32 dies on its
    first autocast step. This ran green on CPU for a whole refactor and then took down the 300M MoE
    arm on the H100 - the assertion worth keeping is that mixed precision is exercised at all.
    """
    torch.manual_seed(0)
    model = LocalAgentLM(_sparse_tiny())
    idx = torch.randint(0, 64, (2, 16))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits, loss = model(idx, targets=idx)
    assert torch.isfinite(loss)
    # The combine widens to the accumulator rather than narrowing the sum.
    assert logits.shape == (2, 16, 64)
