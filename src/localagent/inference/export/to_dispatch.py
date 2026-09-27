"""Export the dispatch heads — RouteHead + DenseToolSelector — for on-device runtimes (Phase 9).

These are the two NEW heads of the generable tool-dispatch pipeline (the existing tool_head /
ptr_head live in ``to_onnx._heads_json``). Both read the model's final hidden state (the post-final
-norm ``hidden`` the onnx graph already exposes), so the device only needs the same forward pass it
already runs for decoding, plus a couple of small matmuls applied to ``hidden[:, -1]``:

* **route_head** — a 5-way linear (``localagent.agent.routes.RouteHead`` == a ``ToolHead`` over
  ``ROUTES``). Exported exactly like ``tool_head``: raw ``weight`` (5, d_model) + ``bias`` (5,) and
  the ordered ``ROUTES``. Device computes ``logits = hidden[:, -1] @ W.T + b``; ``argmax`` is the
  route. JSON, no extra onnx op needed.

* **dense_selector** — a two-tower scorer
  ``score = normalize(q_proj(feat)) @ normalize(t_proj(tool_emb)).T`` (see
  ``dense_selector.BoundSelector``). The tool tower is FIXED per tool set, so we ship only the
  QUERY tower (``q_proj``: Linear d_model->256) and **precompute** the per-tool tower vectors
  ``T_j = normalize(t_proj(tool_embeddings(tools, examples)))`` once, offline. On device the whole
  selection is::

      q = normalize(q_proj(hidden[:, -1]))          # (256,)
      j = argmax_j  q · T[j]                         # T is the shipped (n_tools, 256) matrix

  so the device never touches the 8192-dim char-ngram embedding or ``t_proj`` — it just dots the
  query against a small fixed matrix shipped alongside, with ``tool_names`` giving the column order.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from localagent.model import ModelConfig


def _checkpoint_tokenizer(
    checkpoint: dict,
    cfg: ModelConfig,
    *,
    tokenizer_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
):
    """Load the checkpoint tokenizer for parity, retaining legacy byte compatibility."""
    from localagent.model.tokenizer import load_tokenizer

    metadata = checkpoint.get("tokenizer")
    if metadata is None:
        metadata = {"kind": "byte"}
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint tokenizer metadata must be a mapping")

    kind = str(metadata.get("kind", "byte"))
    path = tokenizer_path if tokenizer_path is not None else metadata.get("path")
    if kind == "bpe" and path is not None and tokenizer_path is None and checkpoint_path is not None:
        recorded = Path(path).expanduser()
        if not recorded.is_absolute() and not recorded.exists():
            colocated = Path(checkpoint_path).resolve().parent / recorded
            if colocated.exists():
                path = colocated

    tokenizer = load_tokenizer(kind, path)
    if tokenizer.vocab_size != cfg.vocab_size:
        raise ValueError(
            "checkpoint tokenizer vocabulary "
            f"({tokenizer.vocab_size}) does not match model config ({cfg.vocab_size})"
        )
    return tokenizer


def _round_nested(x, dp: int = 6):
    """Round a (possibly nested) python list of floats to ``dp`` decimals for JSON size."""
    if isinstance(x, list):
        return [_round_nested(v, dp) for v in x]
    return round(float(x), dp)


def selector_tool_matrix(ck: dict, tools, examples: dict | None = None) -> torch.Tensor:
    """Precompute the per-tool tower matrix ``normalize(t_proj(tool_embeddings(tools, examples)))``.

    Returns a fixed ``(n_tools, proj)`` tensor — exactly the ``t`` side of
    ``DenseToolSelector.forward`` for this tool set. Selection on device is then
    ``argmax_j normalize(q_proj(feat)) @ T.T``, no 8192-dim embedding or ``t_proj`` needed at
    runtime. Examples default to the checkpoint's stored ``examples`` (the paraphrase bridge that
    the trained selector was built with), so the exported columns match the PyTorch BoundSelector.
    """
    import torch.nn.functional as F

    from localagent.agent.dense_selector import DenseToolSelector, tool_embeddings

    sd = ck["dense_selector"]
    d_model = sd["q_proj.weight"].shape[1]
    emb_dim = sd["t_proj.weight"].shape[1]
    proj = ck.get("selector_proj", sd["q_proj.weight"].shape[0])
    sel = DenseToolSelector(d_model, emb_dim=emb_dim, proj=proj).eval()
    sel.load_state_dict(sd)
    if examples is None:
        examples = ck.get("examples")
    with torch.no_grad():
        embs = tool_embeddings(tools, emb_dim, examples=examples)   # (n_tools, emb_dim)
        T = F.normalize(sel.t_proj(embs), dim=-1)                   # (n_tools, proj)
    return T


def retrieval_tool_matrix(tools, examples: dict | None = None, dim: int = 256) -> torch.Tensor:
    """Build a compact, runtime-friendly char-ngram retrieval matrix.

    The dense selector remains the learned primary policy.  This second matrix is an explicit
    zero-training fallback for browser/mobile deployments: it lets the WebGPU demo compare a
    reproducible retrieval policy without a hand-written action guard, while adding only
    ``len(tools) * dim`` floats to the dispatch sidecar.
    """
    import numpy as np

    from localagent.agent.retriever import embed

    rows = []
    for tool in tools:
        vector = embed(f"{tool.name.replace('_', ' ')} {tool.description}", dim)
        tool_examples = (examples or {}).get(tool.name)
        if tool_examples:
            vector = vector + np.mean([embed(query, dim) for query in tool_examples], axis=0)
            norm = np.linalg.norm(vector)
            vector = vector / norm if norm > 0 else vector
        rows.append(vector)
    return torch.tensor(np.stack(rows), dtype=torch.float32)


def dispatch_heads_json(ck: dict, tools=None, examples: dict | None = None) -> dict:
    """Serialize route_head + dense_selector (query tower + precomputed tool matrix) as plain JSON.

    ``tools`` defaults to ``STANDARD_TOOLS`` (the tool set the checkpoint was trained against);
    pass a different list to ship the device a different *fixed* tool pool — the matrix is rebuilt
    for whatever pool is bound, which is the whole point of the generable selector.
    """
    from localagent.agent.routes import ROUTES, route_of
    from localagent.agent.toolset import STANDARD_TOOLS

    if tools is None:
        tools = STANDARD_TOOLS

    rh = ck["route_head"]
    route_w = rh["fc.weight"].cpu().tolist()      # (5, d_model)
    route_b = rh["fc.bias"].cpu().tolist()        # (5,)
    stop_index = ROUTES.index("text")

    sd = ck["dense_selector"]
    q_w = sd["q_proj.weight"].cpu().tolist()       # (proj, d_model)
    q_b = sd["q_proj.bias"].cpu().tolist()         # (proj,)
    proj = sd["q_proj.weight"].shape[0]
    T = selector_tool_matrix(ck, tools, examples)  # (n_tools, proj)
    retrieval_dim = 256
    retrieval_examples = ck.get("retrieval_examples")
    if retrieval_examples is None:
        retrieval_examples = examples if examples is not None else ck.get("examples")
    retrieval_matrix = retrieval_tool_matrix(tools, retrieval_examples, dim=retrieval_dim)

    return {
        "route_head": {
            # weight[r] is the row for ROUTES[r]; logit_r = sum_d hidden[d]*weight[r][d] + bias[r].
            "weight": _round_nested(route_w),          # (5, d_model)
            "bias": _round_nested(route_b),            # (5,)
            "routes": list(ROUTES),                    # ordered, len 5
            "stop_index": stop_index,                  # index of "text" (direct answer / abstain)
        },
        "dense_selector": {
            # QUERY tower only (the tool tower is precomputed into `tool_matrix`).
            # device: q = normalize(hidden[:, -1] @ q_proj_W.T + q_proj_b); j = argmax_j q @ T[j].
            "q_proj_weight": _round_nested(q_w),       # (proj, d_model)
            "q_proj_bias": _round_nested(q_b),         # (proj,)
            "proj": proj,                              # 256
            # precomputed, ALREADY L2-normalized per row, fixed for this tool set.
            "tool_matrix": _round_nested(T.cpu().tolist()),   # (n_tools, proj)
            "tool_names": [t.name for t in tools],            # column order for argmax -> name
            # normalize the query (q) before the dot; tool_matrix rows are pre-normalized so the
            # device does NOT renormalize them. score_j = q · T[j]; pick argmax_j.
            "normalize_query": True,
        },
        "retrieval_selector": {
            "algorithm": "char_ngram_crc32_v1",
            "dim": retrieval_dim,
            "tool_matrix": _round_nested(retrieval_matrix.tolist()),
            "tool_names": [t.name for t in tools],
            "tool_routes": [route_of(t.name) for t in tools],
            "normalize_query": True,
            "compact_instruction_marker": " instruction:",
        },
    }


def export_dispatch(
    checkpoint: str,
    out_path: str,
    tools=None,
    examples: dict | None = None,
    check: bool = True,
    tokenizer_path: str | Path | None = None,
) -> dict:
    """Write ``out_path`` (a ``dispatch_heads.json``) with the route head + dense selector tower.

    Returns the artifact path + size. With ``check=True`` runs the in-process parity check
    (``parity_dispatch``) on a handful of real prompts and prints route/selector agreement.
    ``tokenizer_path`` overrides a moved BPE tokenizer path recorded in the checkpoint.
    """
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    heads = dispatch_heads_json(ck, tools=tools, examples=examples)
    with open(out_path, "w") as f:
        json.dump(heads, f)
    print(f"wrote {out_path}")
    stats = {"dispatch_heads.json": out_path,
             "dispatch_heads.json_MB": os.path.getsize(out_path) / 1e6}

    if check:
        cfg_d = ck["cfg"] if isinstance(ck["cfg"], dict) else ck["cfg"].__dict__
        cfg = ModelConfig(
            **{k: v for k, v in cfg_d.items() if k in ModelConfig.__dataclass_fields__})
        res = parity_dispatch(
            ck,
            cfg,
            heads,
            tokenizer_path=tokenizer_path,
            checkpoint_path=checkpoint,
            tools=tools,
        )
        stats.update(res)
        print(f"route argmax agreement: {res['route_agree']*100:.1f}%  "
              f"max|Δlogits|={res['route_maxdiff']:.2e}")
        print(f"selector top-1 agreement: {res['selector_agree']*100:.1f}%  "
              f"max|Δscore|={res['selector_maxdiff']:.2e}")
    return stats


def parity_dispatch(
    ck: dict,
    cfg: ModelConfig,
    heads: dict,
    n_prompts: int = 12,
    *,
    tokenizer_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    tools=None,
) -> dict:
    """Compare the exported (numpy-style) route head + dense selector against the PyTorch
    ``RouteHead`` / ``BoundSelector`` on a batch of real prompt features.

    Features come from ``tool_head._feat`` on a handful of FREEFORM_EVAL + paraphrase prompts (the
    same final-hidden-state the device reads). Returns agreement rates and max logit/score deltas.
    """
    import numpy as np

    from localagent.agent.dense_selector import BoundSelector, DenseToolSelector
    from localagent.agent.routes import RouteHead
    from localagent.agent.tool_head import _feat
    from localagent.agent.toolset import STANDARD_TOOLS
    from localagent.data.paraphrase import paraphrase_samples
    from localagent.eval.freeform import FREEFORM_EVAL
    from localagent.model import LocalAgentLM

    model = LocalAgentLM(cfg).eval()
    model.load_state_dict(ck["state_dict"])
    tok = _checkpoint_tokenizer(
        ck,
        cfg,
        tokenizer_path=tokenizer_path,
        checkpoint_path=checkpoint_path,
    )

    prompts = [q for q, _ in FREEFORM_EVAL[:n_prompts]]
    if len(prompts) < n_prompts:
        prompts += [s.prompt for s in paraphrase_samples(n=1, split="eval")][: n_prompts - len(prompts)]

    with torch.no_grad():
        feats = torch.stack([_feat(model, tok, p, "cpu") for p in prompts])  # (B, d_model)

    # --- route head: PyTorch reference vs exported weights -------------------------------------
    rh = RouteHead(cfg.d_model)
    rh.load_state_dict(ck["route_head"])
    rh.eval()
    with torch.no_grad():
        ref_route = rh(feats).numpy()
    W = np.array(heads["route_head"]["weight"], dtype=np.float32)
    b = np.array(heads["route_head"]["bias"], dtype=np.float32)
    exp_route = feats.numpy() @ W.T + b
    route_maxdiff = float(np.abs(ref_route - exp_route).max())
    route_agree = float((ref_route.argmax(-1) == exp_route.argmax(-1)).mean())

    # --- dense selector: PyTorch BoundSelector vs precomputed tool matrix ----------------------
    sd = ck["dense_selector"]
    sel = DenseToolSelector(cfg.d_model, emb_dim=sd["t_proj.weight"].shape[1],
                            proj=sd["q_proj.weight"].shape[0])
    sel.load_state_dict(sd)
    bound_tools = STANDARD_TOOLS if tools is None else list(tools)
    bound = BoundSelector(sel, bound_tools, examples=ck.get("examples"))
    ref_top1 = [bound.rank(f)[0] for f in feats]

    qW = np.array(heads["dense_selector"]["q_proj_weight"], dtype=np.float32)
    qb = np.array(heads["dense_selector"]["q_proj_bias"], dtype=np.float32)
    T = np.array(heads["dense_selector"]["tool_matrix"], dtype=np.float32)   # (n_tools, proj)
    names = heads["dense_selector"]["tool_names"]
    q = feats.numpy() @ qW.T + qb                                           # (B, proj)
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
    scores = q @ T.T                                                        # (B, n_tools)
    exp_top1 = [names[int(i)] for i in scores.argmax(-1)]
    selector_agree = float(np.mean([a == r for a, r in zip(exp_top1, ref_top1)]))

    # max|Δscore| against the PyTorch selector's full score row (same normalize convention)
    with torch.no_grad():
        ref_scores = sel(feats, bound.embs).numpy()
    selector_maxdiff = float(np.abs(ref_scores - scores).max())

    return {
        "route_maxdiff": route_maxdiff,
        "route_agree": route_agree,
        "selector_maxdiff": selector_maxdiff,
        "selector_agree": selector_agree,
        "n_prompts": len(prompts),
    }


def export(checkpoint: str, out_path: str) -> None:  # uniform with other export/* modules
    export_dispatch(checkpoint, out_path, check=True)
