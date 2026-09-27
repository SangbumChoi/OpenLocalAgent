"""Deterministic unit tests for the learned planner rollout (stage 2, planner -> action).

We drive ``plan_rollout`` with a *scripted* tool head (emits a known sequence of tool classes then
the no-tool ``"text"`` STOP class) over a real (tiny, random-init) model so the rest of the path —
context rendering, grounded action decoding (``ptr_head`` + heuristic extractors), the simulated
tool response, and termination — runs for real. We assert:
  * the returned ordered ToolCall tool-names match the scripted sequence,
  * the rollout STOPs on the ``"text"`` class (does not run to ``max_steps``),
  * ``max_steps`` is respected when the head never emits STOP,
  * args are grounded (non-empty, and the literal value when it is present in the query).
"""

from __future__ import annotations

import torch

from openlocalagent.agent.caller import plan_rollout
from openlocalagent.agent.tool_head import CLASSES
from openlocalagent.agent.toolset import STANDARD_TOOLS
from openlocalagent.model import LocalAgentLM, ModelConfig
from openlocalagent.model.tokenizer import load_tokenizer


def _tiny_model():
    cfg = ModelConfig(vocab_size=256, d_model=64, embed_dim=64, n_layers=2, n_loops=1,
                      n_heads=4, n_kv_heads=2, ffn_hidden=128, max_seq_len=512,
                      rope_theta=10000.0, norm_eps=1e-5, tie_embeddings=True, dropout=0.0)
    return LocalAgentLM(cfg).eval()


class ScriptedToolHead(torch.nn.Module):
    """A tool head whose argmax follows a fixed class sequence, then ``"text"`` forever (STOP).

    It ignores the input feature entirely: each call emits a one-hot logit row for the next
    scripted class. This isolates ``plan_rollout``'s control flow from the (untrained) model so the
    test is fully deterministic regardless of weights."""

    def __init__(self, sequence: list[str]):
        super().__init__()
        self.classes = CLASSES
        self.sequence = list(sequence)
        self.i = 0

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        name = self.sequence[self.i] if self.i < len(self.sequence) else "text"
        self.i += 1
        logits = torch.full((len(self.classes),), -10.0)
        logits[self.classes.index(name)] = 10.0
        return logits


def test_rollout_follows_scripted_sequence_and_stops_on_text():
    model, tok = _tiny_model(), load_tokenizer("byte")
    head = ScriptedToolHead(["web_search", "send_email"])   # then implicit "text" STOP
    query = "Look up best hiking trails and email Walter about it."
    plan = plan_rollout(model, tok, query, STANDARD_TOOLS, tool_head=head, ptr_head=None,
                        max_steps=4)
    assert [c.name for c in plan] == ["web_search", "send_email"]
    # STOP came from the "text" class, NOT from exhausting max_steps (which is 4 > 2 emitted).
    assert head.i == 3                                        # 2 tools + 1 STOP read


def test_rollout_respects_max_steps_when_head_never_stops():
    model, tok = _tiny_model(), load_tokenizer("byte")
    # the head keeps asking for a tool; only max_steps bounds the rollout.
    head = ScriptedToolHead(["read_file"] * 10)
    plan = plan_rollout(model, tok, "Read src/main.py please.", STANDARD_TOOLS,
                        tool_head=head, ptr_head=None, max_steps=2)
    assert len(plan) == 2
    assert all(c.name == "read_file" for c in plan)


def test_rollout_stops_immediately_on_text():
    model, tok = _tiny_model(), load_tokenizer("byte")
    head = ScriptedToolHead([])                               # first read is "text"
    plan = plan_rollout(model, tok, "Thanks for your help!", STANDARD_TOOLS,
                        tool_head=head, ptr_head=None, max_steps=4)
    assert plan == []


def test_rollout_grounds_args_present_in_query():
    model, tok = _tiny_model(), load_tokenizer("byte")
    head = ScriptedToolHead(["web_search"])
    plan = plan_rollout(model, tok, "Search the web for best hiking trails.", STANDARD_TOOLS,
                        tool_head=head, ptr_head=None, max_steps=4)
    assert len(plan) == 1 and plan[0].name == "web_search"
    # the query value must be grounded into the call args (non-empty, and the literal phrase).
    assert plan[0].arguments.get("query")
    assert "hiking trails" in plan[0].arguments["query"]
