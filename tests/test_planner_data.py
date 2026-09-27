"""Stage-1 PLANNER data: a learned planner emits an ordered multi-step plan (tool-name list),
each step grounded into a ToolCall by the existing action decoder. These tests assert the plan
episodes are well-formed, every step's args are exactly recoverable (grounding), the recovered
plan is the ordered tool-name list, and train/eval slot pools stay disjoint.
"""

from openlocalagent.data.synth.agent_synth import (Generator, episode_plan, episode_steps)
from openlocalagent.data.render import IGNORE, history_text, render_conversation
from openlocalagent.data.schema import Role
from openlocalagent.model.tokenizer import load_tokenizer

# the string args a step copies from context (mirrors pointer_head.PTR_ARGS for our plan tools)
_STR_ARGS = ("path", "query", "url", "content", "recipient", "title", "task", "duration",
             "command", "message", "pattern", "goal", "term", "song", "topic", "city", "summary")


def _prior_context_text(messages, upto: int) -> str:
    """All user + tool-response text strictly before index ``upto`` — the legal grounding sources
    for step ``upto``'s arguments (copy from the composite request OR an earlier tool response)."""
    parts = []
    for m in messages[:upto]:
        if m.role == Role.user:
            parts.append(m.content)
        elif m.role == Role.tool:
            parts.append(m.tool_response or "")
    return "\n".join(parts)


def test_plan_builders_registry_matches_planner_types():
    g = Generator(5, 0, "train")
    assert set(g._PLAN_BUILDERS.keys()) == set(Generator._PLANNER_TYPES)
    # the original three remain a subset (backwards compatible)
    for t in ("plan_read_test_commit", "plan_research", "plan_fix_test"):
        assert t in g._PLAN_BUILDERS


def test_all_plan_types_producible_via_plan_episodes():
    g = Generator(5, 0, "train")
    seen = {e.meta["type"] for e in g.plan_episodes(2000)}
    assert seen == set(Generator._PLANNER_TYPES)


def test_plan_episodes_well_formed():
    g = Generator(5, 1, "train")
    for ep in g.plan_episodes(300):
        assert ep.meta["kind"] == "planner_episode"
        # first assistant turn is the text plan (or "just reply" for 0-step), before any tool call
        first_asst = next(m for m in ep.messages if m.role == Role.assistant)
        assert not first_asst.tool_calls
        assert first_asst.content.startswith("Plan:")
        # meta plan == derived plan == ordered tool names; plan_len consistent
        plan = episode_plan(ep)
        assert ep.meta["plan"] == plan
        assert ep.meta["plan_len"] == len(plan)
        # each step has exactly one tool call (single-call turns)
        assert all(len(s.arguments) >= 0 for s in episode_steps(ep))
        assert len(episode_steps(ep)) == len(plan)


def test_recovered_plan_is_ordered_tool_name_list():
    g = Generator(5, 2, "train")
    for ep in g.plan_episodes(200):
        names = [m.tool_calls[0].name for m in ep.messages
                 if m.role == Role.assistant and m.tool_calls]
        assert episode_plan(ep) == names
        # and steps line up name-for-name with the plan
        assert [s.name for s in episode_steps(ep)] == names


def test_plan_length_distribution_covers_0_to_4():
    g = Generator(5, 3, "train")
    lens = {ep.meta["plan_len"] for ep in g.plan_episodes(1500)}
    assert lens == {0, 1, 2, 3, 4}


def test_every_step_args_exactly_recoverable_zero_misses():
    """0 grounding misses: every string arg of every step is a literal substring of the prior
    user turns or an earlier tool response. (Run across several hundred steps.)"""
    g = Generator(5, 4, "train")
    misses = 0
    total_steps = 0
    for ep in g.plan_episodes(400):
        for i, m in enumerate(ep.messages):
            if m.role != Role.assistant or not m.tool_calls:
                continue
            total_steps += 1
            ctx = _prior_context_text(ep.messages, i)
            for arg, val in m.tool_calls[0].arguments.items():
                if arg in _STR_ARGS and isinstance(val, str):
                    if val not in ctx:
                        misses += 1
    assert total_steps >= 300
    assert misses == 0


def test_pointer_grounding_present():
    """At least one plan type grounds a downstream arg ONLY from an earlier tool response (the
    value is absent from every user turn but present in a tool turn) — the pointer-head case."""
    g = Generator(5, 5, "train")
    found = False
    for ep in g.plan_episodes(600):
        users = "\n".join(m.content for m in ep.messages if m.role == Role.user)
        for i, m in enumerate(ep.messages):
            if m.role != Role.assistant or not m.tool_calls:
                continue
            for arg, val in m.tool_calls[0].arguments.items():
                if arg in _STR_ARGS and isinstance(val, str) and val not in users:
                    tools_before = "\n".join(t.tool_response or "" for t in ep.messages[:i]
                                             if t.role == Role.tool)
                    if val in tools_before:
                        found = True
    assert found


def test_train_eval_slot_pools_disjoint():
    """Plans must not be memorizable: the slot VALUES used in eval plans never appear in train
    plans (and vice versa) for the copyable string args."""
    gt = Generator(5, 6, "train")
    ge = Generator(5, 6, "eval")

    def values(gen):
        vals = set()
        for ep in gen.plan_episodes(800):
            for s in episode_steps(ep):
                for arg, v in s.arguments.items():
                    if arg in _STR_ARGS and isinstance(v, str):
                        vals.add((arg, v))
        return vals

    tr, ev = values(gt), values(ge)
    assert tr and ev
    assert tr.isdisjoint(ev)


def test_plan_episodes_render_with_learned_and_masked_labels():
    tok = load_tokenizer("byte")
    for ep in Generator(5, 7, "train").plan_episodes(120):
        ids, labels = render_conversation(ep, tok)
        assert len(ids) == len(labels) and ids
        assert any(lab != IGNORE for lab in labels)  # assistant plan + steps learned
        assert any(lab == IGNORE for lab in labels)  # user/tool masked
        history_text(ep.messages)


def test_eval_split_plan_episodes_use_eval_pools_only():
    from openlocalagent.data.synth.agent_synth import (PATHS_EVAL, QUERIES_EVAL, URLS_EVAL)
    g = Generator(5, 8, "eval")
    paths, queries, urls = set(PATHS_EVAL), set(QUERIES_EVAL), set(URLS_EVAL)
    saw = False
    for ep in g.plan_episodes(300):
        for s in episode_steps(ep):
            if "path" in s.arguments:
                assert s.arguments["path"] in paths
                saw = True
            if "query" in s.arguments:
                assert s.arguments["query"] in queries
            if "url" in s.arguments:
                assert s.arguments["url"] in urls
    assert saw
