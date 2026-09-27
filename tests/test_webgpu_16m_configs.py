from dataclasses import asdict

from openlocalagent.model.config import ModelConfig
from openlocalagent.model.transformer import LocalAgentLM


def _load_pair() -> tuple[ModelConfig, ModelConfig]:
    hybrid = ModelConfig.from_yaml("configs/model/webgpu-16m-hybrid.yaml")
    attention = ModelConfig.from_yaml("configs/model/webgpu-16m-attn.yaml")
    return hybrid, attention


def test_webgpu_16m_pair_is_strictly_matched_and_below_20m() -> None:
    hybrid, attention = _load_pair()
    hybrid_fields = asdict(hybrid)
    attention_fields = asdict(attention)
    differing_fields = {
        key for key in hybrid_fields if hybrid_fields[key] != attention_fields[key]
    }

    assert differing_fields == {"name", "ffn_hidden", "layer_types"}
    assert hybrid.block_types() == ["conv", "conv", "attn"] * 3
    assert attention.block_types() == ["attn"] * 9
    assert hybrid.estimate_params() == 15_638_464
    assert attention.estimate_params() == 15_618_112
    assert max(hybrid.estimate_params(), attention.estimate_params()) < 20_000_000
    relative_delta = abs(hybrid.estimate_params() - attention.estimate_params()) / (
        hybrid.estimate_params()
    )
    assert relative_delta < 0.005


def test_webgpu_16m_estimates_match_real_models_and_hybrid_cache_is_smaller() -> None:
    hybrid, attention = _load_pair()
    hybrid.assert_within_budget()
    attention.assert_within_budget()

    assert LocalAgentLM(hybrid).num_params() == hybrid.estimate_params()
    assert LocalAgentLM(attention).num_params() == attention.estimate_params()
    assert hybrid.estimate_cache_bytes(hybrid.max_seq_len) < attention.estimate_cache_bytes(
        attention.max_seq_len
    )
