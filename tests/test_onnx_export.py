import numpy as np
import pytest
import torch

from openlocalagent.model import LocalAgentLM, ModelConfig

pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")


def test_onnx_export_parity(tmp_path):
    from openlocalagent.inference.export.to_onnx import export
    cfg = ModelConfig(vocab_size=256, d_model=64, embed_dim=32, n_layers=2, n_loops=2,
                      n_heads=4, n_kv_heads=2, ffn_hidden=128, max_seq_len=64, name="t")
    m = LocalAgentLM(cfg)
    ckpt = tmp_path / "m.pt"
    torch.save({"cfg": cfg.__dict__, "state_dict": m.state_dict()}, ckpt)
    out = tmp_path / "m.onnx"
    export(str(ckpt), str(out), check=False)

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = torch.randint(0, 256, (1, 12))
    with torch.no_grad():
        ref = m(x)[0][:, -1, :].numpy()
    got = sess.run(["logits"], {"input_ids": x.numpy()})[0]
    assert got.shape == ref.shape == (1, cfg.vocab_size)
    assert np.abs(ref - got).max() < 1e-3                 # numerical parity
    assert (ref.argmax(-1) == got.argmax(-1)).all()       # same decoded tokens
