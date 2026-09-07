import pytest
import torch

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from openlocalagent.inference.export.visual_action import VisualActionExport  # noqa: E402
from openlocalagent.model import LocalAgentLM, ModelConfig  # noqa: E402
from openlocalagent.model.vision import VisualActionHead  # noqa: E402


def _model() -> tuple[LocalAgentLM, VisualActionHead]:
    cfg = ModelConfig(
        name="visual-export-test",
        vocab_size=64,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden=64,
        max_seq_len=64,
        vision_enabled=True,
        vision_image_size=16,
        vision_patch_size=8,
        vision_width=8,
    )
    return LocalAgentLM(cfg).eval(), VisualActionHead(cfg.d_model).eval()


def test_visual_export_wrapper_has_explicit_tensor_contract() -> None:
    model, head = _model()
    wrapper = VisualActionExport(model, head)
    logits, pointer = wrapper(
        torch.zeros((2, 12), dtype=torch.long),
        torch.rand((2, 3, 16, 16)),
        torch.tensor([5, 7]),
    )
    assert logits.shape == (2, 7)
    assert pointer.shape == (2, 2)
