"""Pointer/copy argument head (ARCHITECTURE_IDEAS §2a, the arg side of the dual head).

Replaces the heuristic span extractors in agent/constrained.py with a *learned* copy mechanism:
conditioned on which argument it is filling, the head predicts a (start, end) span over the
prompt's byte positions, and the argument value is the copied span. Trained jointly with the
model (auxiliary loss in SFT) so it generalizes to free-form values (shell commands, file
contents) that no regex/preposition heuristic can extract reliably.

One pointer arg per sample (our tools have ≤1 free-text string arg; enum/number/arithmetic stay
schema-extracted). Byte-level features make the span a literal byte slice of the prompt.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# string args grounded by the pointer head (enum/number/arithmetic use schema extractors instead)
PTR_ARGS = ["city", "query", "goal", "term", "song", "topic",
            "path", "pattern", "command", "message", "task", "duration",
            "title", "recipient", "url", "content", "summary"]
ARG_IDX = {a: i for i, a in enumerate(PTR_ARGS)}
# Browser/mobile schemas use the same copy mechanism for DOM node identifiers and typed field
# values.  This is deliberately a separate vocabulary: changing ``PTR_ARGS`` would invalidate
# every existing checkpoint's pointer embedding shape.
BROWSER_PTR_ARGS = [*PTR_ARGS, "target_id", "value"]
BROWSER_ARG_IDX = {a: i for i, a in enumerate(BROWSER_PTR_ARGS)}


class PointerHead(nn.Module):
    def __init__(self, d_model: int, args=PTR_ARGS):
        super().__init__()
        # Keep the legacy 17-argument vocabulary as the default, while allowing a checkpoint
        # trained for a grounded UI schema to add arguments such as ``target_id`` and ``value``.
        # The argument order is part of the checkpoint/export contract, so copy it instead of
        # retaining a caller-owned mutable list.
        self.args = tuple(args)
        self.arg_idx = {name: index for index, name in enumerate(self.args)}
        if len(self.arg_idx) != len(self.args):
            raise ValueError("pointer argument names must be unique")
        self.arg_emb = nn.Embedding(len(self.args), d_model)
        self.start = nn.Linear(d_model, d_model, bias=False)
        self.end = nn.Linear(d_model, d_model, bias=False)

    def logits(self, feats: torch.Tensor, arg_idx: torch.Tensor):
        """feats (B,T,d), arg_idx (B,) -> start/end logits (B,T)."""
        q = self.arg_emb(arg_idx)                       # (B,d)
        s = torch.einsum("btd,bd->bt", feats, self.start(q))
        e = torch.einsum("btd,bd->bt", feats, self.end(q))
        return s, e

    @torch.no_grad()
    def predict_span(
        self,
        feats_row: torch.Tensor,
        arg: str,
        span_bounds: tuple[int, int] | None = None,
    ) -> tuple[int, int]:
        """Return a span, optionally restricted to an allowed ``[start, end]`` interval.

        The bounds keep a pointer head conditioned on a serialized tool catalog from copying
        schema prose or tool names instead of values in the user/tool-history grounding text.
        The default remains unrestricted for legacy callers and export parity.
        """
        i = torch.tensor(self.arg_idx[arg], device=feats_row.device)
        q = self.arg_emb(i)
        s = feats_row @ self.start(q)
        e = feats_row @ self.end(q)
        if span_bounds is not None:
            lo, hi = span_bounds
            if not (0 <= lo <= hi < feats_row.shape[0]):
                raise ValueError("pointer span bounds must be within the feature sequence")
            s = s.clone()
            e = e.clone()
            s[:lo] = -1e9
            s[hi + 1 :] = -1e9
            e[:lo] = -1e9
            e[hi + 1 :] = -1e9
        start = int(s.argmax())
        e = e.clone()
        e[:start] = -1e9                                # end must not precede start
        return start, int(e.argmax())


def gold_span(framed_ids: list[int], value_ids: list[int]) -> tuple[int, int] | None:
    """First occurrence of value_ids within framed_ids -> (start, end) inclusive, else None."""
    n, m = len(framed_ids), len(value_ids)
    for i in range(n - m + 1):
        if framed_ids[i:i + m] == value_ids:
            return i, i + m - 1
    return None
