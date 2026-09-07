"""The dense selector must be *generable*: its scoring is defined over whatever tool list it's
bound to (by description embedding), so it ranks any/unseen tools without a fixed-N reshape."""

import torch

from openlocalagent.agent.dense_selector import BoundSelector, DenseToolSelector, tool_embeddings
from openlocalagent.agent.toolset import STANDARD_TOOLS as TOOLS


def test_scores_shape_matches_tool_count():
    sel = DenseToolSelector(d_model=32, emb_dim=512)
    embs = torch.randn(7, 512)
    scores = sel(torch.randn(3, 32), embs)
    assert scores.shape == (3, 7)   # B x N(tools present), not a fixed N


def test_tool_embeddings_one_row_per_tool():
    embs = tool_embeddings(TOOLS, dim=1024)
    assert embs.shape == (len(TOOLS), 1024)


def test_rank_returns_all_bound_tools_in_order():
    sel = DenseToolSelector(d_model=16, emb_dim=tool_embeddings(TOOLS).shape[1])
    bound = BoundSelector(sel, TOOLS)
    order = bound.rank(torch.randn(16))
    assert sorted(order) == sorted(t.name for t in TOOLS)   # a full permutation, no tool dropped


def test_rank_can_be_restricted_to_retrieved_candidates():
    sel = DenseToolSelector(d_model=16, emb_dim=tool_embeddings(TOOLS).shape[1])
    bound = BoundSelector(sel, TOOLS)
    allowed = {TOOLS[1].name, TOOLS[4].name}
    order = bound.rank(torch.randn(16), allowed_names=allowed)
    assert set(order) == allowed


def test_generalizes_to_an_unseen_tool():
    # bind to a tool list that includes a tool the selector never "saw" — it still scores/ranks it,
    # because scoring is over the description embedding, not a trained output index.
    sel = DenseToolSelector(d_model=16, emb_dim=tool_embeddings(TOOLS[:3]).shape[1])
    bound = BoundSelector(sel, TOOLS[:5])      # 5 tools, selector built for emb_dim only
    order = bound.rank(torch.randn(16))
    assert len(order) == 5 and set(order) == {t.name for t in TOOLS[:5]}
