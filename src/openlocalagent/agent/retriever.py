"""Scalable tool selection by retrieval (for catalogs of 100s–1000s of tools).

A fixed N-way classifier head (agent/tool_head.py) does not scale past a few dozen tools and can
never select a tool it wasn't trained on. The standard large-catalog answer is **retrieval**:
embed the user query and every tool's name+description, retrieve the top-k most similar tools,
then select/ground among that small set. This is O(catalog) memory but O(top-k) for the downstream
selector, adds tools with zero training, and handles unseen tools out of the box.

The embedding here is a deterministic **character n-gram hashing** vector (TF, L2-normalised) — no
training, no dependencies beyond numpy, fast enough for thousands of tools on CPU. It's a strong,
honest baseline for "which method efficiently chooses among many tools".
"""

from __future__ import annotations

import re
import zlib

import numpy as np


def embed(text: str, dim: int = 8192, ngrams=(3, 4, 5)) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    t = " " + re.sub(r"[^a-z0-9]+", " ", text.lower()).strip() + " "
    for n in ngrams:
        for i in range(len(t) - n + 1):
            v[zlib.crc32(t[i:i + n].encode()) % dim] += 1.0
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 0 else v


class ToolRetriever:
    """Index tool specs once; retrieve top-k tool names for a query by cosine similarity."""

    def __init__(self, tools, examples: dict | None = None, dim: int = 8192):
        """`examples`: optional {tool_name: [example query strings]}. Indexing tools by the
        centroid of their example usages (not just the description) is the standard trick that
        bridges the paraphrase gap — queries say 'reserve a flight', the tool API says
        'book_flight'."""
        self.dim = dim
        self.names = [t.name for t in tools]
        vecs = []
        for t in tools:
            v = embed(f"{t.name.replace('_', ' ')} {t.description}", dim)
            ex = (examples or {}).get(t.name)
            if ex:
                v = v + np.mean([embed(q, dim) for q in ex], axis=0)
                nrm = np.linalg.norm(v)
                v = v / nrm if nrm > 0 else v
            vecs.append(v)
        self.M = np.stack(vecs)

    def retrieve(self, query: str, k: int = 10) -> list[str]:
        return [n for n, _ in self.retrieve_scored(query, k)]

    def retrieve_scored(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        sims = self.M @ embed(query, self.dim)
        k = min(k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self.names[i], float(sims[i])) for i in idx]

    def rank_of(self, query: str, gold: str) -> int:
        """1-based rank of `gold` for `query` (len+1 if absent). For recall@k / MRR."""
        order = self.retrieve(query, k=len(self.names))
        return order.index(gold) + 1 if gold in order else len(self.names) + 1
