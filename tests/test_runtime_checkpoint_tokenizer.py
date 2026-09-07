from pathlib import Path

import pytest
import torch

from openlocalagent.agent.runtime import Agent
from openlocalagent.agent.tools import ToolRegistry
from openlocalagent.data.schema import ToolSpec
from openlocalagent.model import LocalAgentLM, ModelConfig
from openlocalagent.model.tokenizer import ByteTokenizer, BPETokenizer, train_bpe


def _bpe(tmp_path: Path):
    pytest.importorskip("tokenizers")
    path = tmp_path / "tokenizer.json"
    tokenizer = train_bpe(
        [
            "open settings and select the dark theme",
            "send a message to Dana with the quarterly report",
            "structured browser actions copy quoted arguments",
            "<|user|>hello<|assistant|><tool_call>calculator",
        ],
        path,
        vocab_size=320,
        min_frequency=1,
    )
    return tokenizer, path


def _checkpoint(cfg: ModelConfig, *, tokenizer_metadata=None) -> dict:
    checkpoint = {
        "cfg": cfg.__dict__,
        "state_dict": LocalAgentLM(cfg).state_dict(),
    }
    if tokenizer_metadata is not None:
        checkpoint["tokenizer"] = tokenizer_metadata
    return checkpoint


def _tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="echo",
            description="Echo a message.",
            parameters={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        ),
        lambda message: message,
    )
    return registry


def test_agent_from_checkpoint_loads_recorded_bpe_and_path_override(tmp_path):
    tokenizer, tokenizer_path = _bpe(tmp_path)
    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden=64,
        max_seq_len=64,
        name="runtime-bpe-test",
    )
    checkpoint = _checkpoint(
        cfg,
        tokenizer_metadata={"kind": "bpe", "path": str(tokenizer_path)},
    )
    checkpoint_path = tmp_path / "dispatch.pt"
    torch.save(checkpoint, checkpoint_path)

    agent = Agent.from_checkpoint(checkpoint_path, _tools())
    assert isinstance(agent.tokenizer, BPETokenizer)
    assert agent.tokenizer.encode("send Dana the report") == tokenizer.encode(
        "send Dana the report"
    )

    checkpoint["tokenizer"]["path"] = str(tmp_path / "old-location" / "tokenizer.json")
    moved_checkpoint_path = tmp_path / "moved-dispatch.pt"
    torch.save(checkpoint, moved_checkpoint_path)
    moved_agent = Agent.from_checkpoint(
        moved_checkpoint_path,
        _tools(),
        tokenizer_path=tokenizer_path,
    )
    assert moved_agent.tokenizer.encode("open settings") == tokenizer.encode("open settings")


def test_agent_from_checkpoint_keeps_legacy_byte_default_and_validates_vocab(tmp_path):
    cfg = ModelConfig(
        vocab_size=256,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden=64,
        max_seq_len=64,
        name="runtime-byte-test",
    )
    checkpoint = _checkpoint(cfg)
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save(checkpoint, checkpoint_path)

    agent = Agent.from_checkpoint(checkpoint_path, _tools())
    assert isinstance(agent.tokenizer, ByteTokenizer)

    _, tokenizer_path = _bpe(tmp_path)
    checkpoint["tokenizer"] = {"kind": "bpe", "path": str(tokenizer_path)}
    mismatched_path = tmp_path / "mismatched.pt"
    torch.save(checkpoint, mismatched_path)
    with pytest.raises(ValueError, match="tokenizer vocabulary.*does not match model config"):
        Agent.from_checkpoint(mismatched_path, _tools())
