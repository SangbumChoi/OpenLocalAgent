from openlocalagent.model.config import PARAM_BUDGET, ModelConfig


def test_yaml_configs_load_and_fit_budget():
    for name in ("ultra-tiny-1m", "tiny-30m", "small-90m", "webgpu-35m-hybrid"):
        cfg = ModelConfig.from_yaml(f"configs/model/{name}.yaml")
        cfg.assert_within_budget()  # raises at or above 100M
        assert 0 < cfg.estimate_params() < PARAM_BUDGET


def test_budget_ceiling_is_exclusive():
    import pytest

    class ExactBudgetConfig(ModelConfig):
        def estimate_params(self) -> int:
            return PARAM_BUDGET

    with pytest.raises(ValueError, match="fewer than 100M"):
        ExactBudgetConfig().assert_within_budget()


def test_tier_sizes_land_in_range():
    sizes = {
        n: ModelConfig.from_yaml(f"configs/model/{n}.yaml").estimate_params()
        for n in ("ultra-tiny-1m", "tiny-30m", "small-90m")
    }
    assert 0.7e6 < sizes["ultra-tiny-1m"] < 1.3e6
    assert 20e6 < sizes["tiny-30m"] < 45e6
    assert 70e6 < sizes["small-90m"] < 100e6


def test_ultra_tiny_uses_structural_levers():
    cfg = ModelConfig.from_yaml("configs/model/ultra-tiny-1m.yaml")
    assert cfg.vocab_size == 256  # byte-level
    assert cfg.factorized  # embed_dim != d_model
    assert cfg.effective_depth == 12  # 2 blocks x 6 loops


def test_head_divisibility_asserts():
    import pytest

    with pytest.raises(AssertionError):
        ModelConfig(d_model=384, n_heads=5)  # 384 not divisible by 5


def test_webgpu_footprint_estimates_account_for_hybrid_cache():
    cfg = ModelConfig.from_yaml("configs/model/webgpu-35m-hybrid.yaml")
    assert 25e6 < cfg.estimate_params() < 45e6
    assert cfg.estimate_weight_bytes(4) < 24 * 1024 * 1024
    hybrid_cache = cfg.estimate_cache_bytes(cfg.max_seq_len)
    all_attention = ModelConfig(
        **{**cfg.__dict__, "name": "all-attn", "layer_types": None}
    ).estimate_cache_bytes(cfg.max_seq_len)
    assert hybrid_cache < all_attention
