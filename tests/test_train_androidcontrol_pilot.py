import pytest
import sys
from pathlib import Path

from localagent.model import ModelConfig
from localagent.model import tokenizer as tk

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from train_androidcontrol_pilot import _checkpoint_tokenizer  # noqa: E402


def test_androidcontrol_pilot_uses_checkpoint_bpe_tokenizer(tmp_path) -> None:
    pytest.importorskip("tokenizers")
    path = tmp_path / "tokenizer.json"
    tok = tk.train_bpe(
        ["mobile action instruction", "accessibility tree and tool call"],
        path,
        vocab_size=320,
        min_frequency=1,
    )
    cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        d_model=32,
        embed_dim=32,
        n_layers=1,
        n_loops=1,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden=64,
        max_seq_len=128,
        name="pilot-tokenizer-test",
    )
    loaded = _checkpoint_tokenizer(
        {"cfg": cfg.__dict__, "tokenizer": {"kind": "bpe", "path": str(path)}}
    )
    assert loaded.vocab_size == cfg.vocab_size
