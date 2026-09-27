import json
import hashlib
from pathlib import Path

import torch

from localagent.inference.export.to_hf import export_hf
from localagent.model import LocalAgentLM, ModelConfig


def test_hf_bundle_builds_and_roundtrips(tmp_path):
    cfg = ModelConfig(vocab_size=256, d_model=64, embed_dim=32, n_layers=2, n_loops=2,
                      n_heads=4, n_kv_heads=2, ffn_hidden=128, max_seq_len=64, name="test-tiny")
    m = LocalAgentLM(cfg)
    ckpt = tmp_path / "m.pt"
    torch.save({"cfg": cfg.__dict__, "state_dict": m.state_dict(),
                "tool_head": None, "ptr_head": None}, ckpt)

    out = export_hf(str(ckpt), str(tmp_path / "hf"), push=False)
    cfg_d = json.load(open(f"{out}/config.json"))
    assert cfg_d["name"] == "test-tiny" and cfg_d["vocab_size"] == 256
    weight_file = cfg_d["weights"]["filename"]
    weight_path = tmp_path / "hf" / weight_file
    assert cfg_d["weights"]["bytes"] == weight_path.stat().st_size
    assert cfg_d["weights"]["sha256"] == hashlib.sha256(weight_path.read_bytes()).hexdigest()
    assert f"`{weight_file}` — decoder weights" in (tmp_path / "hf" / "README.md").read_text()
    assert (tmp_path / "hf" / "README.md").exists()

    # reload weights into a fresh model
    from safetensors.torch import load_file
    m2 = LocalAgentLM(cfg)
    m2.load_state_dict(load_file(f"{out}/model.safetensors"))
    x = torch.randint(0, 256, (1, 8))
    assert torch.allclose(m(x)[0], m2(x)[0], atol=1e-5)


def test_hf_bpe_bundle_is_self_contained_and_exports_dispatch_heads(tmp_path):
    cfg = ModelConfig(vocab_size=256, d_model=64, embed_dim=32, n_layers=2, n_loops=1,
                      n_heads=4, n_kv_heads=2, ffn_hidden=128, max_seq_len=64, name="test-bpe")
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("{\"mock\": true}\n", encoding="utf-8")
    checkpoint = tmp_path / "bpe.pt"
    torch.save(
        {
            "cfg": cfg.__dict__,
            "state_dict": LocalAgentLM(cfg).state_dict(),
            "tokenizer": {
                "kind": "bpe",
                "path": str(tokenizer_path),
                "sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
            },
            "tool_head": {"weight": torch.ones(1)},
            "ptr_head": {"weight": torch.ones(1)},
            "route_head": {"weight": torch.ones(1)},
            "dense_selector": {"weight": torch.ones(1)},
            "selector_proj": 1,
            "dispatch_tool_pool": ["mobile_click", "mobile_submit_answer"],
            "ptr_args": ["message"],
            "examples": {"mobile_click": ["tap the button"]},
            "retrieval_examples": {"mobile_click": ["tap the button"]},
        },
        checkpoint,
    )

    out = export_hf(str(checkpoint), str(tmp_path / "hf-bpe"), push=False)
    config = json.loads(Path(out, "config.json").read_text(encoding="utf-8"))
    assert config["tokenizer"]["kind"] == "bpe"
    assert config["tokenizer"]["filename"] == "tokenizer.json"
    assert config["weights"]["filename"] in {"model.safetensors", "pytorch_model.bin"}
    assert config["weights"]["bytes"] == Path(out, config["weights"]["filename"]).stat().st_size
    assert Path(out, "tokenizer.json").read_bytes() == tokenizer_path.read_bytes()
    assert "byte-level" not in Path(out, "README.md").read_text(encoding="utf-8")
    heads = torch.load(Path(out, "agent_heads.bin"), map_location="cpu", weights_only=True)
    assert set(heads) == {
        "tool_head",
        "ptr_head",
        "route_head",
        "dense_selector",
        "selector_proj",
        "tool_pool",
        "ptr_args",
        "examples",
        "retrieval_examples",
    }
    assert config["agent"]["tool_pool"] == ["mobile_click", "mobile_submit_answer"]
    assert config["agent"]["ptr_args"] == ["message"]
    assert "2 tools" in Path(out, "README.md").read_text(encoding="utf-8")


def test_hf_export_infers_standard_dispatch_pool_from_legacy_head_schema(tmp_path):
    cfg = ModelConfig(vocab_size=256, d_model=64, embed_dim=32, n_layers=2, n_loops=1,
                      n_heads=4, n_kv_heads=2, ffn_hidden=128, max_seq_len=64, name="legacy")
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {
            "cfg": cfg.__dict__,
            "state_dict": LocalAgentLM(cfg).state_dict(),
            "tool_head": {"fc.weight": torch.ones(51, 64), "fc.bias": torch.ones(51)},
            "ptr_head": {"arg_emb.weight": torch.ones(2, 64)},
            "ptr_args": ["text", "url"],
            "examples": {"mobile_click": ["tap the button"]},
        },
        checkpoint,
    )

    out = export_hf(str(checkpoint), str(tmp_path / "hf-legacy"), push=False)
    config = json.loads(Path(out, "config.json").read_text(encoding="utf-8"))
    heads = torch.load(Path(out, "agent_heads.bin"), map_location="cpu", weights_only=True)
    assert len(config["agent"]["tool_pool"]) == 63
    assert config["agent"]["ptr_args"] == ["text", "url"]
    assert config["agent"]["provenance"] == "inferred_standard_tool_pool_from_51_class_tool_head"
    assert heads["tool_pool"] == config["agent"]["tool_pool"]
    assert "63 tools" in Path(out, "README.md").read_text(encoding="utf-8")
