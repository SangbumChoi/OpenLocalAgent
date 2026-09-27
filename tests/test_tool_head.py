import torch

from openlocalagent.agent.tool_head import CLASSES, ToolHead, label_of
from openlocalagent.data.synth.agent_synth import Generator
from openlocalagent.model import LocalAgentLM, ModelConfig


def test_return_hidden_shape():
    cfg = ModelConfig(vocab_size=256, d_model=64, embed_dim=32, n_layers=2, n_loops=2,
                      n_heads=4, n_kv_heads=2, ffn_hidden=128, max_seq_len=64)
    m = LocalAgentLM(cfg)
    logits, feats = m(torch.randint(0, 256, (2, 10)), return_hidden=True)
    assert logits.shape == (2, 10, 256)
    assert feats.shape == (2, 10, cfg.d_model)  # d_model features, not embed_dim


def test_tool_head_predict_in_classes():
    head = ToolHead(d_model=64)
    out = head(torch.randn(64))
    assert out.shape == (len(CLASSES),)


def test_label_of_maps_tool_and_text():
    s = Generator(1, 0, "train").generate(40)
    for sample in s:
        lab = label_of(sample)
        if sample.kind == "tool":
            assert lab == sample.ref_name
        else:
            assert lab == "text"
        assert lab in CLASSES
