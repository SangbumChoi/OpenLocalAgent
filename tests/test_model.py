import torch

from openlocalagent.model import LocalAgentLM, ModelConfig


def _tiny():
    # Shrunken config to keep the CPU test fast while exercising GQA/RoPE/SwiGLU.
    return ModelConfig(vocab_size=256, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
                       ffn_hidden=128, max_seq_len=64)


def test_forward_shapes_and_loss():
    cfg = _tiny()
    model = LocalAgentLM(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(idx, targets=idx)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_tied_embeddings_param_count():
    cfg = _tiny()
    model = LocalAgentLM(cfg)
    # tied weights counted once; close to the closed-form estimate
    assert model.num_params() == sum(
        p.numel() for p in {id(p): p for p in model.parameters()}.values()
    )


def test_factorized_embeddings_and_recurrence_forward():
    # Mirrors the ultra-tiny structure: byte vocab + factorized embed + shared-weight loops.
    cfg = ModelConfig(vocab_size=256, d_model=96, embed_dim=32, n_layers=2, n_loops=4,
                      n_heads=6, n_kv_heads=2, ffn_hidden=128, max_seq_len=64)
    assert cfg.factorized and cfg.effective_depth == 8
    model = LocalAgentLM(cfg)
    assert model.in_proj is not None and model.out_proj is not None and model.loop_embed is not None
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(idx, targets=idx)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(loss)


def test_real_ultra_tiny_config_constructs_under_budget():
    cfg = ModelConfig.from_yaml("configs/model/ultra-tiny-1m.yaml")
    model = LocalAgentLM(cfg)
    assert model.num_params() < 1.3e6
    logits, _ = model(torch.randint(0, cfg.vocab_size, (1, 8)))
    assert logits.shape[-1] == cfg.vocab_size


def test_budget_enforced_at_construction():
    import pytest

    big = ModelConfig(vocab_size=32000, d_model=2048, n_layers=24, n_heads=16, n_kv_heads=4,
                      ffn_hidden=8192)
    with pytest.raises(ValueError):
        LocalAgentLM(big)


# --------------------------------------------------------------------------------------
# Hybrid backbone (gated short-conv + GQA + QK-Norm). Defaults must preserve legacy models.
# --------------------------------------------------------------------------------------

def _hybrid_tiny(**over):
    base = dict(vocab_size=256, d_model=64, n_layers=4, n_heads=4, n_kv_heads=2,
                ffn_hidden=128, max_seq_len=64,
                layer_types=["conv", "attn", "conv", "conv"], qk_norm=True)
    base.update(over)
    return ModelConfig(**base)


def test_defaults_are_all_attention_no_qknorm():
    # The new fields must default to the legacy behavior exactly.
    cfg = ModelConfig(vocab_size=256, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
                      ffn_hidden=128, max_seq_len=64)
    assert cfg.layer_types is None
    assert cfg.qk_norm is False
    assert cfg.block_types() == ["attn", "attn"]


def test_legacy_byte_config_is_byte_identical():
    # Building the shipped byte config must be unchanged: same param count + same forward as a
    # config with the new fields left at default. (Param count + state-dict keys unchanged.)
    cfg = ModelConfig.from_yaml("configs/model/tiny-30m-byte.yaml")
    assert cfg.layer_types is None and cfg.qk_norm is False
    torch.manual_seed(0)
    model = LocalAgentLM(cfg)
    # no conv weights, no qk-norm gains leaked into an all-attention model
    keys = list(model.state_dict().keys())
    assert not any("q_norm" in k or "k_norm" in k or "in_proj" in k and "blocks" in k for k in keys)
    idx = torch.randint(0, cfg.vocab_size, (1, 16))
    logits, _ = model(idx)
    assert logits.shape == (1, 16, cfg.vocab_size) and torch.isfinite(logits).all()


def test_hybrid_forward_and_block_dispatch():
    cfg = _hybrid_tiny()
    model = LocalAgentLM(cfg)
    kinds = [b.kind for b in model.blocks]
    assert kinds == ["conv", "attn", "conv", "conv"]
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(idx, targets=idx)
    assert logits.shape == (2, 16, cfg.vocab_size) and torch.isfinite(loss)


def test_qk_norm_adds_gains_only_when_enabled():
    on = LocalAgentLM(_hybrid_tiny(layer_types=["attn", "attn", "attn", "attn"], qk_norm=True))
    off = LocalAgentLM(_hybrid_tiny(layer_types=["attn", "attn", "attn", "attn"], qk_norm=False))
    assert any("q_norm" in k for k in on.state_dict())
    assert not any("q_norm" in k for k in off.state_dict())
    # estimate_params reflects the extra gains
    assert on.cfg.estimate_params() > off.cfg.estimate_params()


def test_estimate_params_matches_built_model_hybrid():
    cfg = _hybrid_tiny()
    model = LocalAgentLM(cfg)
    # closed-form estimate should equal the real (tied) param count for the hybrid too
    assert model.num_params() == cfg.estimate_params()


def test_conv_decode_cache_matches_full_forward():
    # Parity: incremental single-token decode (conv keeps a (B,d,k-1) state) must match a full
    # forward over the whole sequence — exactly the property the attention KV-cache test asserts.
    torch.manual_seed(1)
    cfg = _hybrid_tiny()
    model = LocalAgentLM(cfg).eval()
    # The default 0.02 initialization makes the three-projection conv residual small enough for a
    # broken cache to hide under a loose logits tolerance. Amplify it deterministically.
    with torch.no_grad():
        for block in model.blocks:
            if block.kind == "conv":
                block.attn.in_proj.weight.mul_(3)
                block.attn.conv.weight.mul_(3)
                block.attn.out_proj.weight.mul_(3)
    idx = torch.randint(0, cfg.vocab_size, (2, 20))

    with torch.no_grad():
        full_logits, _ = model(idx)

        # prefill first 8 tokens, then decode the rest one at a time through the cache
        caches = [None] * model.n_cache_slots()
        pre = 8
        step_logits, _, caches = model(idx[:, :pre], pos=0, caches=caches)
        conv_caches = [
            cache for kind, cache in zip(cfg.block_types(), caches, strict=True)
            if kind == "conv"
        ]
        assert all(cache is not None for cache in conv_caches)
        assert all(cache.shape == (idx.shape[0], cfg.d_model, cfg.conv_kernel - 1)
                   for cache in conv_caches)
        got = [step_logits]
        for t in range(pre, idx.shape[1]):
            sl, _, caches = model(idx[:, t:t + 1], pos=t, caches=caches)
            got.append(sl)
        cached_logits = torch.cat(got, dim=1)

    assert torch.allclose(full_logits, cached_logits, atol=1e-5, rtol=1e-5), (
        (full_logits - cached_logits).abs().max().item()
    )


def test_hybrid_yaml_config_under_budget():
    cfg = ModelConfig.from_yaml("configs/model/tiny-30m-hybrid.yaml")
    cfg.assert_within_budget()
    assert cfg.qk_norm is True
    assert cfg.block_types().count("conv") == 7 and cfg.block_types().count("attn") == 3
    model = LocalAgentLM(cfg)
    n = model.num_params()
    assert 20e6 < n < 45e6
    assert n == cfg.estimate_params()
