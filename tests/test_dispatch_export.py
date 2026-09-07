"""Parity for the exported dispatch heads (route_head + dense_selector) vs PyTorch.

Mirrors test_web_export.py: prove the device can reproduce route selection and dense-selector
top-1 from the exported JSON (route weights + query tower + precomputed tool matrix) applied to the
model's final hidden state, matching the PyTorch RouteHead / BoundSelector within tolerance.
"""

import json

import numpy as np
import pytest
import torch

from openlocalagent.agent.dense_selector import (
    BoundSelector,
    DenseToolSelector,
    tool_embeddings,
)
from openlocalagent.agent.routes import ROUTES, RouteHead
from openlocalagent.agent.tool_head import _feat
from openlocalagent.agent.toolset import REALISTIC_BROWSER_TOOLS, STANDARD_TOOLS
from openlocalagent.inference.export.to_dispatch import (
    dispatch_heads_json,
    parity_dispatch,
    retrieval_tool_matrix,
    selector_tool_matrix,
)
from openlocalagent.model import LocalAgentLM, ModelConfig
from openlocalagent.model import tokenizer as tk

_PROMPTS = [
    "What is the color of a monkey?",
    "Read the file data/loader.py.",
    "What's the weather like in Oslo?",
    "Commit the staged changes with message fix bug.",
    "Email Dana the quarterly report.",
    "What is 18 * 24?",
]


def _make_ck(tmp_path):
    cfg = ModelConfig(vocab_size=256, d_model=64, embed_dim=64, n_layers=2, n_loops=1,
                      n_heads=4, n_kv_heads=2, ffn_hidden=128, max_seq_len=128, name="t")
    m = LocalAgentLM(cfg).eval()
    rh = RouteHead(cfg.d_model)
    emb_dim = tool_embeddings(STANDARD_TOOLS[:1], dim=1024).shape[1]
    sel = DenseToolSelector(cfg.d_model, emb_dim=emb_dim, proj=256)
    examples = {STANDARD_TOOLS[0].name: ["the weather in Paris"]}
    ck = {
        "cfg": cfg.__dict__,
        "state_dict": m.state_dict(),
        "route_head": rh.state_dict(),
        "dense_selector": sel.state_dict(),
        "selector_proj": 256,
        "examples": examples,
    }
    return ck, cfg, m, rh, sel, examples


def _feats(m):
    tok = tk.load_tokenizer("byte")
    with torch.no_grad():
        return torch.stack([_feat(m, tok, p, "cpu") for p in _PROMPTS])


def test_route_head_parity(tmp_path):
    ck, cfg, m, rh, sel, examples = _make_ck(tmp_path)
    heads = dispatch_heads_json(ck)
    feats = _feats(m)

    W = np.array(heads["route_head"]["weight"], dtype=np.float32)   # (5, d_model)
    b = np.array(heads["route_head"]["bias"], dtype=np.float32)     # (5,)
    exp = feats.numpy() @ W.T + b
    with torch.no_grad():
        ref = rh(feats).numpy()

    assert heads["route_head"]["routes"] == list(ROUTES)
    assert heads["route_head"]["routes"][heads["route_head"]["stop_index"]] == "text"
    assert exp.shape == ref.shape == (len(_PROMPTS), 5)
    assert np.abs(exp - ref).max() < 1e-3
    assert (exp.argmax(-1) == ref.argmax(-1)).all()


def test_dense_selector_top1_parity(tmp_path):
    ck, cfg, m, rh, sel, examples = _make_ck(tmp_path)
    heads = dispatch_heads_json(ck)
    feats = _feats(m)

    # PyTorch reference: BoundSelector top-1 over STANDARD_TOOLS (with the same examples).
    bound = BoundSelector(sel, STANDARD_TOOLS, examples=examples)
    ref_top1 = [bound.rank(f)[0] for f in feats]

    # device recipe: q = normalize(feat @ q_proj_W.T + q_proj_b); j = argmax_j q @ T[j].
    qW = np.array(heads["dense_selector"]["q_proj_weight"], dtype=np.float32)
    qb = np.array(heads["dense_selector"]["q_proj_bias"], dtype=np.float32)
    T = np.array(heads["dense_selector"]["tool_matrix"], dtype=np.float32)
    names = heads["dense_selector"]["tool_names"]

    assert T.shape == (len(STANDARD_TOOLS), heads["dense_selector"]["proj"])
    assert names == [t.name for t in STANDARD_TOOLS]
    # shipped rows are L2-normalized (the tool tower side of the score).
    assert np.allclose(np.linalg.norm(T, axis=-1), 1.0, atol=1e-4)

    q = feats.numpy() @ qW.T + qb
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
    scores = q @ T.T
    exp_top1 = [names[int(i)] for i in scores.argmax(-1)]
    assert exp_top1 == ref_top1

    # full-score numerical parity vs the PyTorch selector.
    with torch.no_grad():
        ref_scores = sel(feats, bound.embs).numpy()
    assert np.abs(ref_scores - scores).max() < 1e-3


def test_extended_browser_tool_pool_dispatch_parity(tmp_path):
    ck, cfg, *_ = _make_ck(tmp_path)
    heads = dispatch_heads_json(ck, tools=REALISTIC_BROWSER_TOOLS)
    assert heads["dense_selector"]["tool_names"][-3:] == [
        "web_click",
        "web_type",
        "web_select",
    ]
    result = parity_dispatch(ck, cfg, heads, n_prompts=2, tools=REALISTIC_BROWSER_TOOLS)
    assert result["route_agree"] == 1.0
    assert result["selector_agree"] == 1.0


def test_selector_tool_matrix_matches_bound_embs(tmp_path):
    """The precomputed matrix == normalize(t_proj(tool_embeddings)) the BoundSelector uses."""
    import torch.nn.functional as F
    ck, cfg, m, rh, sel, examples = _make_ck(tmp_path)
    T = selector_tool_matrix(ck, STANDARD_TOOLS, examples)
    bound = BoundSelector(sel, STANDARD_TOOLS, examples=examples)
    with torch.no_grad():
        ref = F.normalize(sel.t_proj(bound.embs), dim=-1)
    assert torch.allclose(T, ref, atol=1e-5)


def test_retrieval_tool_matrix_matches_runtime_retriever():
    """The compact exported retrieval sidecar must match the Python runtime index exactly."""
    from openlocalagent.agent.retriever import ToolRetriever

    examples = {STANDARD_TOOLS[0].name: ["the weather in Paris"]}
    exported = retrieval_tool_matrix(STANDARD_TOOLS, examples, dim=256).numpy()
    runtime = ToolRetriever(STANDARD_TOOLS, examples=examples, dim=256).M
    assert exported.shape == runtime.shape == (len(STANDARD_TOOLS), 256)
    assert np.allclose(exported, runtime, atol=1e-6)


def test_dispatch_json_roundtrips(tmp_path):
    ck, *_ = _make_ck(tmp_path)
    heads = dispatch_heads_json(ck)
    p = tmp_path / "dispatch_heads.json"
    p.write_text(json.dumps(heads))
    back = json.loads(p.read_text())
    assert back["route_head"]["routes"] == list(ROUTES)
    assert back["dense_selector"]["normalize_query"] is True
    assert len(back["dense_selector"]["tool_names"]) == len(STANDARD_TOOLS)
    retrieval = back["retrieval_selector"]
    assert retrieval["algorithm"] == "char_ngram_crc32_v1"
    assert retrieval["dim"] == 256
    assert len(retrieval["tool_matrix"]) == len(STANDARD_TOOLS)
    assert len(retrieval["tool_matrix"][0]) == retrieval["dim"]
    assert retrieval["tool_routes"]


def test_parity_dispatch_uses_checkpoint_bpe_path_override(tmp_path, monkeypatch):
    pytest.importorskip("tokenizers")
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = tk.train_bpe(
        [
            "read a file and send a message",
            "open settings and choose dark mode",
            "calculate a structured browser action",
            "<|user|>hello<|assistant|><tool_call>calculator",
        ],
        tokenizer_path,
        vocab_size=320,
        min_frequency=1,
    )
    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        embed_dim=64,
        n_layers=2,
        n_loops=1,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden=128,
        max_seq_len=128,
        name="dispatch-bpe",
    )
    model = LocalAgentLM(cfg).eval()
    route_head = RouteHead(cfg.d_model)
    emb_dim = tool_embeddings(STANDARD_TOOLS[:1], dim=1024).shape[1]
    selector = DenseToolSelector(cfg.d_model, emb_dim=emb_dim, proj=256)
    checkpoint = {
        "cfg": cfg.__dict__,
        "state_dict": model.state_dict(),
        "route_head": route_head.state_dict(),
        "dense_selector": selector.state_dict(),
        "selector_proj": 256,
        "examples": {},
        "tokenizer": {
            "kind": "bpe",
            "path": str(tmp_path / "old-location" / "tokenizer.json"),
        },
    }
    heads = dispatch_heads_json(checkpoint)

    original_load = tk.load_tokenizer
    calls = []

    def recording_load(kind="byte", path=None):
        calls.append((kind, path))
        return original_load(kind, path)

    monkeypatch.setattr(tk, "load_tokenizer", recording_load)
    result = parity_dispatch(
        checkpoint,
        cfg,
        heads,
        n_prompts=2,
        tokenizer_path=tokenizer_path,
    )

    assert calls == [("bpe", tokenizer_path)]
    assert result["route_agree"] == 1.0
    assert result["selector_agree"] == 1.0


def test_parity_dispatch_legacy_byte_default_and_vocab_validation(tmp_path):
    checkpoint, cfg, *_ = _make_ck(tmp_path)
    heads = dispatch_heads_json(checkpoint)
    result = parity_dispatch(checkpoint, cfg, heads, n_prompts=2)
    assert result["route_agree"] == 1.0
    assert result["selector_agree"] == 1.0

    pytest.importorskip("tokenizers")
    tokenizer_path = tmp_path / "mismatched-tokenizer.json"
    tk.train_bpe(
        ["browser actions and structured tool calls"],
        tokenizer_path,
        vocab_size=320,
        min_frequency=1,
    )
    checkpoint["tokenizer"] = {"kind": "bpe", "path": str(tokenizer_path)}
    with pytest.raises(ValueError, match="tokenizer vocabulary.*does not match model config"):
        parity_dispatch(checkpoint, cfg, heads, n_prompts=1)
