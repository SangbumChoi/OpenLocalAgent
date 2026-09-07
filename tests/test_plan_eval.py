"""Scoring-math tests for the stage-3 PLANNER eval (`eval.harness.plan_eval`).

These validate the metric arithmetic *independently of any trained model* by monkeypatching the
learned ``plan_rollout`` (stage 2) to return controlled, scripted plans, and by stubbing the
teacher-forced ``multi_turn_eval`` so the free-run planner numbers are isolated. We assert:

  * an exactly-correct rollout  -> whole-plan acc 1.0, step 1.0, grounded 1.0, plan_len 1.0;
  * an off-by-one tool          -> the right per-step / whole / length numbers;
  * an OVER-length plan          -> penalized (whole=0, step<1, plan_len=0);
  * grounded args mismatch on a name-matched step lowers grounded_acc but NOT step_acc;
  * the per-gold-length breakdown buckets episodes correctly.

A final test exercises the REAL teacher-forced path (``multi_turn_eval`` + ``grounded_decode``) on
tiny real episodes to confirm wiring, without asserting trained-model accuracy.
"""

from __future__ import annotations

import openlocalagent.agent.caller as caller_mod
import openlocalagent.eval.harness as harness
from openlocalagent.data.schema import Conversation, Message, Role, ToolCall
from openlocalagent.eval import harness as H


# --- helpers: build an episode by hand so episode_plan/episode_steps read real gold ----------
def _episode(steps: list[tuple[str, dict]], query: str = "do the thing") -> Conversation:
    """A minimal episode: one user turn then one assistant tool-call turn per planned step.

    ``episode_plan`` projects assistant tool-call turns onto names; ``episode_steps`` returns the
    ToolCalls. The first user turn is what ``plan_eval`` feeds the (mocked) rollout."""
    msgs = [Message(role=Role.user, content=query)]
    for name, args in steps:
        msgs.append(Message(role=Role.assistant, tool_calls=[ToolCall(name=name, arguments=args)]))
        msgs.append(Message(role=Role.tool, tool_response="ok"))
    return Conversation(messages=msgs, meta={"plan": [n for n, _ in steps]})


def _calls(steps: list[tuple[str, dict]]) -> list[ToolCall]:
    return [ToolCall(name=n, arguments=a) for n, a in steps]


def _patch_rollout(monkeypatch, fn):
    """Patch the name `plan_rollout` resolves to inside plan_eval (imported from caller)."""
    monkeypatch.setattr(caller_mod, "plan_rollout", fn, raising=True)


def _stub_teacher(monkeypatch, step_acc=0.0, episode_acc=0.0):
    monkeypatch.setattr(
        harness, "multi_turn_eval",
        lambda *a, **k: {"step_acc": step_acc, "episode_acc": episode_acc,
                         "steps": 0, "episodes": 0},
        raising=True,
    )


def test_exact_match_is_perfect(monkeypatch):
    gold = [("web_search", {"query": "trails"}), ("send_email", {"to": "walter"})]
    ep = _episode(gold)
    _patch_rollout(monkeypatch, lambda *a, **k: _calls(gold))
    _stub_teacher(monkeypatch, step_acc=0.5, episode_acc=0.5)
    res = H.plan_eval(None, None, [], [ep], tool_head=None, ptr_head=None)
    assert res["whole_plan_acc"] == 1.0
    assert res["step_acc"] == 1.0
    assert res["grounded_acc"] == 1.0
    assert res["plan_len_acc"] == 1.0
    assert res["by_gold_len"][2] == {"whole": 1.0, "n": 1}
    # teacher-forced passes through from the stub, side by side with the free-run metrics
    assert res["teacher_forced"] == {"step_acc": 0.5, "episode_acc": 0.5}
    assert res["episodes"] == 1 and res["steps"] == 2


def test_off_by_one_tool(monkeypatch):
    gold = [("web_search", {"query": "q"}), ("send_email", {"to": "w"}),
            ("read_file", {"path": "a.py"})]
    pred = [("web_search", {"query": "q"}), ("calc", {"expr": "1+1"}),     # wrong middle tool
            ("read_file", {"path": "a.py"})]
    ep = _episode(gold)
    _patch_rollout(monkeypatch, lambda *a, **k: _calls(pred))
    _stub_teacher(monkeypatch)
    res = H.plan_eval(None, None, [], [ep], tool_head=None, ptr_head=None)
    assert res["whole_plan_acc"] == 0.0              # sequences differ
    assert res["plan_len_acc"] == 1.0               # same length (3)
    assert res["step_acc"] == 2 / 3                 # positions 0 and 2 match, 1 does not
    # grounded checked only on the 2 name-matched positions; both args match -> 1.0
    assert res["grounded_acc"] == 1.0
    assert res["by_gold_len"][3] == {"whole": 0.0, "n": 1}


def test_over_length_plan_is_penalized(monkeypatch):
    gold = [("web_search", {"query": "q"})]
    pred = [("web_search", {"query": "q"}), ("send_email", {"to": "w"})]   # 1 extra step
    ep = _episode(gold)
    _patch_rollout(monkeypatch, lambda *a, **k: _calls(pred))
    _stub_teacher(monkeypatch)
    res = H.plan_eval(None, None, [], [ep], tool_head=None, ptr_head=None)
    assert res["whole_plan_acc"] == 0.0
    assert res["plan_len_acc"] == 0.0               # len 2 != 1
    # span = max(2,1) = 2; only position 0 matches -> 1/2. Over-length can never reach 1.0.
    assert res["step_acc"] == 0.5
    assert res["grounded_acc"] == 1.0               # the single matched position grounds fine


def test_under_length_plan_is_penalized(monkeypatch):
    gold = [("web_search", {"query": "q"}), ("send_email", {"to": "w"})]
    pred = [("web_search", {"query": "q"})]                                  # missing 2nd step
    ep = _episode(gold)
    _patch_rollout(monkeypatch, lambda *a, **k: _calls(pred))
    _stub_teacher(monkeypatch)
    res = H.plan_eval(None, None, [], [ep], tool_head=None, ptr_head=None)
    assert res["whole_plan_acc"] == 0.0
    assert res["plan_len_acc"] == 0.0
    assert res["step_acc"] == 0.5                   # span 2, 1 match
    assert res["grounded_acc"] == 1.0


def test_grounded_args_mismatch_lowers_only_grounded(monkeypatch):
    gold = [("web_search", {"query": "right"})]
    pred = [("web_search", {"query": "WRONG"})]     # name matches, args differ
    ep = _episode(gold)
    _patch_rollout(monkeypatch, lambda *a, **k: _calls(pred))
    _stub_teacher(monkeypatch)
    res = H.plan_eval(None, None, [], [ep], tool_head=None, ptr_head=None)
    assert res["step_acc"] == 1.0                   # tool NAME matches
    assert res["grounded_acc"] == 0.0               # but the args do not AST-match
    assert res["whole_plan_acc"] == 1.0             # whole-plan is NAME-only, so still 1.0


def test_empty_plan_episode_scores_as_correct_abstention(monkeypatch):
    ep = _episode([], query="thanks!")              # 0-step gold plan
    _patch_rollout(monkeypatch, lambda *a, **k: [])  # rollout also abstains
    _stub_teacher(monkeypatch)
    res = H.plan_eval(None, None, [], [ep], tool_head=None, ptr_head=None)
    assert res["whole_plan_acc"] == 1.0
    assert res["plan_len_acc"] == 1.0
    # Whole-plan / plan-len correctly credit the abstention. NOTE: a *pure* 0-step eval set has no
    # positional steps to score, so step_tot==0 and the max(1,.) guard makes step_acc==0.0 (NOT
    # 1.0) and grounded_acc==0.0 — a degenerate edge case; in any mixed set with >=1 real step it
    # washes out. Flagging the doc/code mismatch ("skipped") in the report.
    assert res["step_acc"] == 0.0
    assert res["grounded_acc"] == 0.0
    assert res["by_gold_len"][0] == {"whole": 1.0, "n": 1}


def test_per_gold_length_breakdown_buckets(monkeypatch):
    # three episodes at gold-len 1, 2, 2; rollout returns gold for the len-1 and one len-2, wrong
    # for the other len-2 -> L1=100%(1), L2=50%(2).
    g1 = [("web_search", {"query": "a"})]
    g2a = [("web_search", {"query": "a"}), ("send_email", {"to": "w"})]
    g2b = [("read_file", {"path": "x"}), ("calc", {"expr": "1+1"})]
    eps = [_episode(g1), _episode(g2a), _episode(g2b)]
    # map episode -> the plan to return, keyed by the gold names of its first step
    def rollout(model, tok, query, tools, **k):
        # use the query? all share default; instead key off a counter
        rollout.calls += 1
        return [_calls(g1), _calls(g2a), _calls([("read_file", {"path": "x"})])][rollout.calls - 1]
    rollout.calls = 0
    _patch_rollout(monkeypatch, rollout)
    _stub_teacher(monkeypatch)
    res = H.plan_eval(None, None, [], eps, tool_head=None, ptr_head=None)
    assert res["by_gold_len"][1] == {"whole": 1.0, "n": 1}
    assert res["by_gold_len"][2] == {"whole": 0.5, "n": 2}     # g2a right, g2b wrong (under-length)
    assert res["whole_plan_acc"] == 2 / 3
    # plan_len: g1 ok, g2a ok, g2b pred-len 1 != 2 -> 2/3
    assert res["plan_len_acc"] == 2 / 3


def test_format_plan_eval_is_log_style():
    res = {
        "whole_plan_acc": 0.5, "step_acc": 0.75, "grounded_acc": 0.6, "plan_len_acc": 0.5,
        "by_gold_len": {1: {"whole": 1.0, "n": 2}, 2: {"whole": 0.0, "n": 1}},
        "teacher_forced": {"step_acc": 0.8, "episode_acc": 0.4},
        "episodes": 3, "steps": 4,
    }
    s = H.format_plan_eval(res)
    assert "planner: whole=50% step=75% grounded=60% plan_len=50%" in s
    assert "L1=100%(2)" in s and "L2=0%(1)" in s
    assert "teacher-forced: step_acc=80% episode_acc=40%" in s


def test_real_teacher_forced_path_wires_up(monkeypatch):
    """Exercise the REAL multi_turn_eval + grounded_decode (no stub) on a tiny real model + real
    plan episodes, to confirm the teacher-forced branch is wired correctly. We only mock the
    free-run rollout and assert the metric keys/ranges, not trained accuracy."""
    from openlocalagent.agent.toolset import STANDARD_TOOLS
    from openlocalagent.data.synth.agent_synth import Generator, episode_plan, episode_steps
    from openlocalagent.model import LocalAgentLM, ModelConfig
    from openlocalagent.model.tokenizer import load_tokenizer

    tok = load_tokenizer("byte")
    cfg = ModelConfig(vocab_size=256, d_model=64, embed_dim=64, n_layers=2, n_loops=1,
                      n_heads=4, n_kv_heads=2, ffn_hidden=128, max_seq_len=4096,
                      rope_theta=10000.0, norm_eps=1e-5, tie_embeddings=True, dropout=0.0)
    model = LocalAgentLM(cfg).eval()
    eps = Generator(level=2, seed=5001, split="eval").plan_episodes(3)

    # mock rollout: echo each episode's GOLD calls so free-run metrics are perfect and we can be
    # sure the (untrained) teacher-forced numbers are what vary.
    def rollout(model, tok, query, tools, **k):
        ep = rollout.eps[rollout.i]
        rollout.i += 1
        return episode_steps(ep)
    rollout.eps, rollout.i = eps, 0
    _patch_rollout(monkeypatch, rollout)

    res = H.plan_eval(model, tok, STANDARD_TOOLS, eps, tool_head=None, ptr_head=None, device="cpu")
    assert res["whole_plan_acc"] == 1.0   # rollout echoed gold names
    assert set(res["teacher_forced"]) == {"step_acc", "episode_acc"}
    assert 0.0 <= res["teacher_forced"]["step_acc"] <= 1.0
    # sanity: gold steps counted match episode_plan lengths
    assert res["steps"] == sum(len(episode_plan(e)) for e in eps)
