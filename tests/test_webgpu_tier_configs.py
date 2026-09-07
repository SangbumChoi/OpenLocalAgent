import gc
import hashlib
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from openlocalagent.model.config import PARAM_BUDGET, ModelConfig
from openlocalagent.model.transformer import LocalAgentLM


MODEL_DIR = "configs/model"
PRETRAIN_DIR = Path("configs/pretrain")
PAPER_CATALOG_TOKENS = 3_389
PROXY_CATALOG_TOKENS = 3_504
PAPER_MAX_PROMPT_TOKENS = 3_584
PROXY_MAX_PROMPT_TOKENS = 3_703
DECODE_ALLOWANCE = 96
REQUIRED_AGENT_CONTEXT = max(PAPER_MAX_PROMPT_TOKENS, PROXY_MAX_PROMPT_TOKENS) + DECODE_ALLOWANCE
HISTORICAL_10M_SHA256 = {
    "webgpu-10m-hybrid": "8609bca200f40e2f67c154f73a1641c905509f4d69a1ec7f80273b0cc6f12a98",
    "webgpu-10m-attn": "51fbc78ccfa0968512ad24845d5eedb1ce0b6cac125971aaefa705e067f3c133",
}


def _load(name: str) -> ModelConfig:
    return ModelConfig.from_yaml(f"{MODEL_DIR}/{name}.yaml")


def _assert_estimate_matches_model(cfg: ModelConfig) -> None:
    """Instantiate one large model at a time so the invariant stays CI-memory-friendly."""
    model = LocalAgentLM(cfg)
    assert model.num_params() == cfg.estimate_params()
    del model
    gc.collect()


def test_corrected_full_catalog_context_requirement() -> None:
    assert PAPER_CATALOG_TOKENS < PAPER_MAX_PROMPT_TOKENS
    assert PROXY_CATALOG_TOKENS < PROXY_MAX_PROMPT_TOKENS
    assert PAPER_MAX_PROMPT_TOKENS + DECODE_ALLOWANCE == 3_680
    assert PROXY_MAX_PROMPT_TOKENS + DECODE_ALLOWANCE == 3_799
    assert REQUIRED_AGENT_CONTEXT == 3_799


def test_one_million_tier_keeps_byte_model_and_adds_honest_bpe_router() -> None:
    byte = _load("ultra-tiny-1m")
    bpe = _load("webgpu-1m-bpe-router")

    assert byte.vocab_size == 256
    assert bpe.vocab_size == 16_384
    assert bpe.factorized
    assert bpe.embed_dim == 32
    assert bpe.effective_depth == 6
    assert bpe.head_dim == 64
    assert bpe.n_kv_heads == 1
    assert bpe.block_types() == ["conv", "attn"]
    assert bpe.max_seq_len == 4_096
    assert bpe.max_seq_len >= REQUIRED_AGENT_CONTEXT

    assert byte.estimate_params() == 976_960
    assert bpe.estimate_params() == 980_480
    assert bpe.vocab_size * bpe.embed_dim > bpe.estimate_params() / 2
    assert bpe.estimate_weight_bytes(bits=4) == 490_240
    assert bpe.estimate_cache_bytes(4_096, dtype_bytes=2) == 3_147_264
    all_attention = ModelConfig(**{**bpe.__dict__, "layer_types": ["attn", "attn"]})
    assert bpe.estimate_cache_bytes(4_096) < all_attention.estimate_cache_bytes(4_096) * 0.51
    assert 900_000 <= bpe.estimate_params() < 1_000_000

    bpe.assert_within_budget()
    _assert_estimate_matches_model(bpe)


def test_historical_ten_million_pair_remains_byte_for_byte_unchanged() -> None:
    hybrid = _load("webgpu-10m-hybrid")
    attention = _load("webgpu-10m-attn")

    for name, expected_sha256 in HISTORICAL_10M_SHA256.items():
        payload = Path(f"{MODEL_DIR}/{name}.yaml").read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected_sha256

    assert hybrid.estimate_params() == 10_524_544
    assert attention.estimate_params() == 10_547_072
    assert hybrid.vocab_size == attention.vocab_size == 16_384
    assert hybrid.max_seq_len == attention.max_seq_len == 2_048
    assert hybrid.head_dim == attention.head_dim == 64
    assert hybrid.n_kv_heads == attention.n_kv_heads == 1
    assert hybrid.estimate_cache_bytes(2048) < attention.estimate_cache_bytes(2048)


def test_new_ten_million_4k_pair_changes_only_context_from_historical_arms() -> None:
    historical_hybrid = _load("webgpu-10m-hybrid")
    historical_attention = _load("webgpu-10m-attn")
    hybrid = _load("webgpu-10m-hybrid-4k")
    attention = _load("webgpu-10m-attn-4k")

    for historical, extended in (
        (historical_hybrid, hybrid),
        (historical_attention, attention),
    ):
        differing_fields = {
            key
            for key, historical_value in asdict(historical).items()
            if historical_value != asdict(extended)[key]
        }
        assert differing_fields == {"name", "max_seq_len"}

    pair_differences = {
        key for key, value in asdict(hybrid).items() if value != asdict(attention)[key]
    }
    assert pair_differences == {"name", "ffn_hidden", "layer_types"}
    assert hybrid.max_seq_len == attention.max_seq_len == 4_096
    assert hybrid.max_seq_len >= REQUIRED_AGENT_CONTEXT
    assert hybrid.estimate_params() == historical_hybrid.estimate_params() == 10_524_544
    assert attention.estimate_params() == historical_attention.estimate_params() == 10_547_072
    assert hybrid.estimate_cache_bytes(4_096, dtype_bytes=2) == 2_100_224
    assert attention.estimate_cache_bytes(4_096, dtype_bytes=2) == 4_194_304


@pytest.mark.parametrize(
    "name",
    [
        "webgpu-10m-hybrid-4k",
        "webgpu-10m-attn-4k",
    ],
)
def test_new_ten_million_4k_estimates_equal_instantiated_model_counts(name: str) -> None:
    cfg = _load(name)
    cfg.assert_within_budget()
    _assert_estimate_matches_model(cfg)


def test_webgpu_96m_pair_is_strictly_matched_and_below_hard_budget() -> None:
    hybrid = _load("webgpu-96m-hybrid")
    attention = _load("webgpu-96m-attn")
    hybrid_fields = asdict(hybrid)
    attention_fields = asdict(attention)
    differing_fields = {key for key in hybrid_fields if hybrid_fields[key] != attention_fields[key]}

    assert differing_fields == {"name", "ffn_hidden", "layer_types"}
    assert hybrid.vocab_size == attention.vocab_size == 16_384
    assert hybrid.max_seq_len == attention.max_seq_len == 4_096
    assert hybrid.max_seq_len >= REQUIRED_AGENT_CONTEXT
    assert hybrid.head_dim == attention.head_dim == 64
    assert hybrid.n_kv_heads == attention.n_kv_heads == 1
    assert hybrid.qk_norm is attention.qk_norm is True
    assert hybrid.block_types() == ["conv", "conv", "attn"] * 6
    assert attention.block_types() == ["attn"] * 18
    assert hybrid.ffn_hidden % 64 == attention.ffn_hidden % 64 == 0

    assert hybrid.estimate_params() == 95_320_448
    assert attention.estimate_params() == 95_298_944
    assert 95_000_000 <= min(hybrid.estimate_params(), attention.estimate_params())
    assert max(hybrid.estimate_params(), attention.estimate_params()) < PARAM_BUDGET
    relative_delta = abs(hybrid.estimate_params() - attention.estimate_params()) / min(
        hybrid.estimate_params(), attention.estimate_params()
    )
    assert relative_delta < 0.00025

    hybrid.assert_within_budget()
    attention.assert_within_budget()


@pytest.mark.parametrize(
    ("name", "expected_params"),
    [
        ("webgpu-96m-hybrid", 95_320_448),
        ("webgpu-96m-attn", 95_298_944),
    ],
)
def test_webgpu_96m_estimates_equal_instantiated_model_counts(
    name: str, expected_params: int
) -> None:
    cfg = _load(name)
    assert cfg.estimate_params() == expected_params
    _assert_estimate_matches_model(cfg)


def test_webgpu_96m_theoretical_weight_and_cache_estimates() -> None:
    hybrid = _load("webgpu-96m-hybrid")
    attention = _load("webgpu-96m-attn")

    # These are arithmetic packing bounds, not evidence of the currently FP16-only browser
    # exporter supporting int4/Q4 execution.
    assert hybrid.estimate_weight_bytes(bits=4) == 47_660_224
    assert attention.estimate_weight_bytes(bits=4) == 47_649_472
    assert hybrid.estimate_weight_bytes(bits=4) < 48 * 1024 * 1024
    assert attention.estimate_weight_bytes(bits=4) < 48 * 1024 * 1024

    hybrid_cache = hybrid.estimate_cache_bytes(4_096, dtype_bytes=2)
    attention_cache = attention.estimate_cache_bytes(4_096, dtype_bytes=2)
    assert hybrid_cache == 6_322_176
    assert attention_cache == 18_874_368
    assert hybrid_cache < attention_cache * 0.34


def test_new_paper_tier_training_configs_all_select_4k_models() -> None:
    expected_models = {
        "pretrain-paper-tier-1m.yaml": "configs/model/webgpu-1m-bpe-router.yaml",
        "pretrain-paper-tier-10m-hybrid.yaml": "configs/model/webgpu-10m-hybrid-4k.yaml",
        "pretrain-paper-tier-10m-attn.yaml": "configs/model/webgpu-10m-attn-4k.yaml",
        "pretrain-paper-tier-96m-hybrid.yaml": "configs/model/webgpu-96m-hybrid.yaml",
        "pretrain-paper-tier-96m-attn.yaml": "configs/model/webgpu-96m-attn.yaml",
    }

    for training_name, expected_model_path in expected_models.items():
        raw = yaml.safe_load((PRETRAIN_DIR / training_name).read_text(encoding="utf-8"))
        assert raw["model_config"] == expected_model_path
        model = ModelConfig.from_yaml(expected_model_path)
        assert model.max_seq_len == 4_096
        assert model.max_seq_len >= REQUIRED_AGENT_CONTEXT
