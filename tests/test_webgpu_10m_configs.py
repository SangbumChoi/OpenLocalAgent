from dataclasses import asdict

from openlocalagent.model.config import ModelConfig
from openlocalagent.model.transformer import LocalAgentLM


def _load_pair() -> tuple[ModelConfig, ModelConfig]:
    hybrid = ModelConfig.from_yaml("configs/model/webgpu-10m-hybrid.yaml")
    attention = ModelConfig.from_yaml("configs/model/webgpu-10m-attn.yaml")
    return hybrid, attention


def test_webgpu_10m_pair_has_low_dispatch_depth_and_is_parameter_matched() -> None:
    hybrid, attention = _load_pair()
    differing_fields = {
        key
        for key, value in asdict(hybrid).items()
        if value != asdict(attention)[key]
    }

    assert differing_fields == {"name", "ffn_hidden", "layer_types"}
    assert hybrid.block_types() == ["conv", "attn", "conv", "attn"]
    assert attention.block_types() == ["attn"] * 4
    assert hybrid.n_layers == attention.n_layers == 4
    assert hybrid.n_loops == attention.n_loops == 1
    assert hybrid.head_dim == attention.head_dim == 64
    assert hybrid.d_model % 128 == 0
    assert hybrid.ffn_hidden % 16 == attention.ffn_hidden % 16 == 0

    assert hybrid.estimate_params() == 10_524_544
    assert attention.estimate_params() == 10_547_072
    assert 8_000_000 <= hybrid.estimate_params() <= 12_000_000
    assert 8_000_000 <= attention.estimate_params() <= 12_000_000
    relative_delta = abs(hybrid.estimate_params() - attention.estimate_params()) / min(
        hybrid.estimate_params(), attention.estimate_params()
    )
    assert relative_delta < 0.01


def test_webgpu_10m_estimates_match_models_and_cache_contract() -> None:
    hybrid, attention = _load_pair()
    hybrid.assert_within_budget()
    attention.assert_within_budget()

    assert LocalAgentLM(hybrid).num_params() == hybrid.estimate_params()
    assert LocalAgentLM(attention).num_params() == attention.estimate_params()
    assert hybrid.estimate_weight_bytes(bits=16) == 21_049_088
    assert attention.estimate_weight_bytes(bits=16) == 21_094_144
    assert hybrid.estimate_cache_bytes(2048, dtype_bytes=2) == 1_051_648
    assert attention.estimate_cache_bytes(2048, dtype_bytes=2) == 2_097_152
    assert hybrid.estimate_cache_bytes(2048) < attention.estimate_cache_bytes(2048)
