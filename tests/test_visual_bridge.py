import pytest
import torch

from openlocalagent.model import LocalAgentLM, ModelConfig


def _cfg(**overrides):
    values = {
        "name": "vision-test",
        "vocab_size": 64,
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 2,
        "ffn_hidden": 128,
        "max_seq_len": 128,
        "vision_enabled": True,
        "vision_image_size": 32,
        "vision_patch_size": 8,
        "vision_width": 16,
    }
    values.update(overrides)
    return ModelConfig(**values)


def test_visual_bridge_is_budgeted_and_shapes_text_logits() -> None:
    cfg = _cfg()
    model = LocalAgentLM(cfg)
    assert model.num_params() == cfg.estimate_params()
    assert model.num_params() < 100_000_000
    idx = torch.randint(0, cfg.vocab_size, (2, 7))
    images = torch.rand(2, 3, 24, 40)
    logits, loss, text_hidden, visual = model.forward_multimodal(
        idx, images, idx, return_hidden=True
    )
    assert logits.shape == (2, 7, cfg.vocab_size)
    assert text_hidden.shape == (2, 7, cfg.d_model)
    assert visual.shape == (2, cfg.vision_tokens, cfg.d_model)
    assert loss is not None and torch.isfinite(loss)


def test_legacy_model_rejects_visual_inputs_without_allocating_bridge() -> None:
    cfg = _cfg(vision_enabled=False)
    model = LocalAgentLM(cfg)
    assert model.vision is None
    with pytest.raises(RuntimeError, match="vision_enabled"):
        model.encode_images(torch.rand(1, 3, 16, 16))


def test_visual_bridge_validates_pixel_range() -> None:
    model = LocalAgentLM(_cfg())
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        model.encode_images(torch.full((1, 3, 8, 8), 2.0))


def test_visual_prefix_respects_context_budget() -> None:
    model = LocalAgentLM(_cfg(max_seq_len=40))
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model.forward_multimodal(
            torch.zeros((1, 30), dtype=torch.long), torch.rand(1, 3, 8, 8)
        )


def test_webgpu_vision_config_stays_under_budget() -> None:
    cfg = ModelConfig.from_yaml("configs/model/webgpu-10m-vision.yaml")
    cfg.assert_within_budget()
    assert cfg.vision_enabled
    assert cfg.estimate_params() == LocalAgentLM(cfg).num_params()
