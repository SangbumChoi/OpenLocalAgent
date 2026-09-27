"""Dense (two-tower) tool selector — a *trained* selector that is still generable.

The experiments showed: tool selection needs a trained discriminative component (the 51-way head
gets 45%; intrinsic ranking/generation gets ~5-7%), but a fixed-N softmax can't accept tools it
wasn't trained on. This resolves the tension: instead of an N-way output layer, score every tool by
the dot product of a learned **query tower** (over the model's prompt features) and a learned **tool
tower** (over the tool's *description embedding*). Selection becomes `argmax_j q·t_j` over WHATEVER
tools are present — adding/removing a tool is adding/removing a column, no reshape, no retraining,
unseen tools work by embedding their description.

The tool embedding is the zero-training char-ngram vector from `retriever.embed` (the same signal
retrieval already uses), so the only learned parameters are the two projection towers — a cheap
frozen-feature probe, like the route head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from localagent.agent.retriever import embed


class DenseToolSelector(nn.Module):
    """q_proj: model features (d_model) -> p ; t_proj: tool ngram-embedding (dim) -> p.
    score(query, tool) = <q_proj(feat), t_proj(tool_emb)>. Trained by cross-entropy against the gold
    tool over the candidate columns — works for any tool set, including unseen tools at eval."""

    def __init__(self, d_model: int, emb_dim: int = 8192, proj: int = 256):
        super().__init__()
        self.q_proj = nn.Linear(d_model, proj)
        self.t_proj = nn.Linear(emb_dim, proj)
        self.emb_dim = emb_dim

    def forward(self, feats, tool_embs):           # (B,d),(N,emb) -> (B,N) scores
        q = F.normalize(self.q_proj(feats), dim=-1)
        t = F.normalize(self.t_proj(tool_embs), dim=-1)
        return q @ t.T


def tool_embeddings(tools, dim: int = 8192, device="cpu", examples: dict | None = None
                    ) -> torch.Tensor:
    """Fixed char-ngram embedding of each tool's `name + description` (the tool tower's input).
    If `examples` ({tool_name: [query strings]}) is given, add the centroid of the tool's example
    usages and renormalize — the standard paraphrase-bridging trick (queries say 'reserve a flight',
    the API says 'book_flight'), which sharpens selection on out-of-distribution wording."""
    rows = []
    for t in tools:
        v = embed(f"{t.name.replace('_', ' ')} {t.description}", dim)
        ex = (examples or {}).get(t.name)
        if ex:
            v = v + np.mean([embed(q, dim) for q in ex], axis=0)
            nrm = np.linalg.norm(v)
            v = v / nrm if nrm > 0 else v
        rows.append(v)
    M = np.stack(rows)
    return torch.tensor(M, dtype=torch.float32, device=device)


class BoundSelector:
    """A DenseToolSelector bound to a fixed tool list — exposes `rank(feat)` -> ordered tool names,
    so it drops into `hybrid_decode(selector=...)`."""

    def __init__(self, model: DenseToolSelector, tools, device="cpu", examples: dict | None = None):
        self.model = model.to(device).eval()
        self.names = [t.name for t in tools]
        self.embs = tool_embeddings(tools, model.emb_dim, device, examples=examples)

    @torch.no_grad()
    def rank(
        self,
        feat,
        allowed_names: set[str] | None = None,
        *,
        query_text: str | None = None,
        lexical_weight: float = 0.5,
    ) -> list[str]:
        """Return tool names ordered by score, optionally restricted to this turn's candidates.

        Large catalogs are retrieved before decoding.  Restricting the bound selector at this
        boundary is important: otherwise a selector trained over the full catalog can reintroduce
        every tool after retrieval and silently defeat the O(top-k) runtime contract.  When a
        current action/query string is available, blend its deterministic character n-gram
        similarity with the learned dense score.  This keeps the trained backbone signal while
        preventing long-horizon goal text from drowning out the immediate action instruction.
        """
        scores = self.model(feat.unsqueeze(0), self.embs)[0]
        if query_text is not None:
            if lexical_weight < 0:
                raise ValueError("lexical_weight must be non-negative")
            from localagent.agent.retriever import embed

            query = torch.tensor(
                embed(query_text, self.model.emb_dim),
                dtype=self.embs.dtype,
                device=self.embs.device,
            )
            scores = scores + lexical_weight * (self.embs @ query)
        order = [self.names[i] for i in torch.argsort(scores, descending=True).tolist()]
        if allowed_names is not None:
            order = [name for name in order if name in allowed_names]
        return order


def train_dense_selector(model, samples, tok, tools, *, steps=400, batch_size=64, lr=5e-3,
                         proj=256, device="cpu", examples: dict | None = None,
                         log=lambda *a: None) -> DenseToolSelector:
    """Frozen-feature probe (cheap). Only tool samples train selection; the gold tool must be in the
    tool list. CE is over ALL tools, so the towers learn a general query<->description match.
    `examples` enriches the tool-tower embeddings with example-query centroids (paraphrase bridge)."""
    import random

    from localagent.agent.tool_head import _feat

    model.eval()
    name_idx = {t.name: i for i, t in enumerate(tools)}
    rows = [(s, name_idx[s.ref_name]) for s in samples
            if s.kind == "tool" and s.ref_name in name_idx]
    if not rows:
        raise ValueError("dense selector needs a tool decision present in the candidate tool list")
    with torch.no_grad():   # frozen-feature probe: cache detached features (no autograd graph/leak)
        feats = torch.stack(
            [
                _feat(
                    model,
                    tok,
                    sample.prompt,
                    device,
                    framed=bool(getattr(sample, "framed", False)),
                )
                for sample, _ in rows
            ]
        )
    labels = torch.tensor([j for _, j in rows], device=device)
    embs = tool_embeddings(tools, device=device, examples=examples)
    sel = DenseToolSelector(model.cfg.d_model, emb_dim=embs.shape[1], proj=proj).to(device)
    opt = torch.optim.AdamW(sel.parameters(), lr=lr)
    rng = random.Random(0)
    n = len(rows)
    for step in range(steps):
        idx = torch.tensor([rng.randrange(n) for _ in range(batch_size)], device=device)
        loss = F.cross_entropy(sel(feats[idx], embs), labels[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % max(1, steps // 5) == 0 or step == steps - 1:
            top1 = (sel(feats, embs).argmax(-1) == labels).float().mean().item()
            log(f"  [dense-sel] step {step}/{steps} loss {loss.item():.3f} top1 {top1:.3f}")
    return sel
