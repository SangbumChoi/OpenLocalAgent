import torch

from localagent.agent.pointer_head import PointerHead
from localagent.model import LocalAgentLM, ModelConfig
from scripts.train_toolace_action_history_pointer import _pointer_from_state


def test_pointer_adapter_preserves_unnamed_legacy_rows() -> None:
    cfg = ModelConfig(
        vocab_size=256,
        d_model=32,
        embed_dim=32,
        n_layers=1,
        n_loops=1,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden=64,
        max_seq_len=32,
    )
    model = LocalAgentLM(cfg)
    legacy = PointerHead(cfg.d_model, args=[f"legacy_{i}" for i in range(17)])
    checkpoint = {"ptr_head": legacy.state_dict(), "ptr_args": []}
    adapted = _pointer_from_state(model, checkpoint, ["url", "query"])
    assert adapted.arg_emb.weight.shape == (2, cfg.d_model)
    assert torch.equal(adapted.start.weight, legacy.start.weight)
    assert torch.equal(adapted.end.weight, legacy.end.weight)
