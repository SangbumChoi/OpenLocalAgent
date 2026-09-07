"""The four ladder tiers and the parameter count each one is claimed at.

Counts are written out rather than recomputed from the config, so a config edit that changes
capacity fails here instead of silently re-labelling a published result.
"""

import pytest

from openlocalagent.model.config import ModelConfig

LADDER = [
    ("la-10m", 10_524_544, 100_000_000),
    ("la-45m", 45_114_368, 100_000_000),
    ("la-93m", 93_108_608, 100_000_000),
    ("la-150m", 149_294_784, 350_000_000),
    ("la-300m", 298_535_808, 350_000_000),
    ("la-300m-moe", 298_701_696, 350_000_000),
    ("la-700m", 700_884_352, 750_000_000),
]


@pytest.mark.parametrize("name,params,budget", LADDER)
def test_tier_parameter_count_and_budget(name, params, budget):
    config = ModelConfig.from_yaml(f"configs/model/{name}.yaml")
    assert config.name == name
    assert config.estimate_params() == params
    assert config.param_budget == budget
    config.assert_within_budget()


@pytest.mark.parametrize("name,params,budget", LADDER)
def test_tier_keeps_the_hybrid_ratio(name, params, budget):
    config = ModelConfig.from_yaml(f"configs/model/{name}.yaml")
    kinds = config.block_types()
    assert set(kinds) == {"conv", "attn"}
    assert kinds[-1] == "attn", "the last mixer stays attention so arguments can be copied verbatim"


def test_sparse_arm_keeps_both_of_its_controls_matched():
    """The sparse arm is the architecture experiment, so its two controls are the experiment.

    docs/SPARSE_EXPERTS_REAL_DATA.md set the matched-active convention for this repository, and a
    matched-total control answers a different question. Both are pinned so neither can drift into
    the other and quietly change what the arm measures.
    """
    sparse = ModelConfig.from_yaml("configs/model/la-300m-moe.yaml")
    active_control = ModelConfig.from_yaml("configs/model/la-150m.yaml")
    total_control = ModelConfig.from_yaml("configs/model/la-300m.yaml")

    # primary control: equal compute per token, unequal stored capacity
    assert abs(active_control.estimate_params() - sparse.estimate_active_params()) \
        / sparse.estimate_active_params() < 0.005
    # secondary control: equal stored capacity, unequal compute per token
    assert abs(total_control.estimate_params() - sparse.estimate_params()) \
        / sparse.estimate_params() < 0.001

    assert total_control.ffn_num_experts == active_control.ffn_num_experts == 1, "controls stay dense"
    assert sparse.ffn_num_experts == 8 and sparse.ffn_top_k == 2
    assert sparse.router_aux_loss_coef > 0, "an unregularised router collapses onto one expert"

    # against the total-matched control, everything but the FFN is held fixed
    for field in ("vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads",
                  "max_seq_len", "qk_norm", "conv_kernel", "layer_types"):
        assert getattr(sparse, field) == getattr(total_control, field), field


def test_budget_is_declared_not_inherited_from_the_environment(monkeypatch):
    monkeypatch.setenv("LOCALAGENT_PARAM_BUDGET", "999000000")
    over = ModelConfig(name="over", vocab_size=16384, d_model=1280, n_layers=27,
                       n_heads=20, n_kv_heads=1, ffn_hidden=5120)
    with pytest.raises(ValueError, match="param_budget"):
        over.assert_within_budget()
