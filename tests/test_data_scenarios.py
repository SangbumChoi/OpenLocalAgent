"""Tests for the SOTA-agent scenario corpus (``localagent.data.scenarios``).

Note: the test file is named ``test_data_scenarios`` (not ``test_scenarios``) because the latter
already exists for the unrelated ``eval.scenarios_bench`` module; both must stay green.
"""

from localagent.agent.toolset import STANDARD_TOOLS
from localagent.data.schema import Conversation, Role
from localagent.data.scenarios import (
    _CHAINED,
    _ERROR,
    _WORKFLOW,
    ABSTAIN,
    CLARIFY,
    scenario_episodes,
    scenario_samples,
)

TOOL_NAMES = {t.name for t in STANDARD_TOOLS}

SINGLE_SUBTYPES = {"clarify", "abstain", "parallel"}
EPISODE_FAMILIES = {"workflow", "chained", "error_recovery"}


# ---- single-turn -----------------------------------------------------------------------------
def test_single_subtypes_in_both_splits():
    for split in ("train", "eval"):
        cats = {s.category for s in scenario_samples(4, seed=1, split=split)}
        assert SINGLE_SUBTYPES <= cats, (split, cats)


def test_clarify_and_abstain_are_text_no_tool():
    for split in ("train", "eval"):
        for s in scenario_samples(4, seed=2, split=split):
            if s.category in ("clarify", "abstain"):
                assert s.kind == "text"
                assert not s.calls
                assert s.ref_name == ""
                assert s.target  # a non-empty reply / question


def test_parallel_has_two_or_more_calls():
    for split in ("train", "eval"):
        pars = [s for s in scenario_samples(6, seed=3, split=split) if s.category == "parallel"]
        assert pars
        for s in pars:
            assert s.calls is not None and len(s.calls) >= 2
            # ref_name / ref_args are the FIRST call (agent_synth parallel shape)
            assert s.ref_name == s.calls[0]["name"]
            for c in s.calls:
                assert c["name"] in TOOL_NAMES


def test_parallel_arg_values_grounded_in_prompt():
    for split in ("train", "eval"):
        for s in scenario_samples(6, seed=4, split=split):
            if s.category != "parallel":
                continue
            for c in s.calls:
                for v in c["arguments"].values():
                    assert str(v) in s.prompt, (s.prompt, v)


def test_balanced_count_per_subtype():
    samples = scenario_samples(5, seed=5, split="train")
    counts = {st: sum(1 for s in samples if s.category == st) for st in SINGLE_SUBTYPES}
    assert all(c == 5 for c in counts.values()), counts


# ---- multi-turn ------------------------------------------------------------------------------
def _is_valid_episode(ep: Conversation) -> bool:
    msgs = ep.messages
    if not msgs:
        return False
    if msgs[0].role != Role.user:
        return False
    if msgs[-1].role != Role.assistant or msgs[-1].tool_calls:
        return False  # must END on a final assistant TEXT turn
    # every assistant tool-call turn is followed by a tool response turn
    for i, m in enumerate(msgs):
        if m.role == Role.assistant and m.tool_calls:
            assert i + 1 < len(msgs)
            if msgs[i + 1].role != Role.tool:
                return False
        if m.role == Role.tool and (i == 0 or msgs[i - 1].role != Role.assistant):
            return False  # a tool turn must follow an assistant tool-call turn
    return True


def test_episode_families_in_both_splits():
    for split in ("train", "eval"):
        cats = {ep.meta["category"] for ep in scenario_episodes(4, seed=6, split=split)}
        assert EPISODE_FAMILIES <= cats, (split, cats)


def test_episodes_valid_alternating_and_end_assistant():
    for split in ("train", "eval"):
        for ep in scenario_episodes(4, seed=7, split=split):
            assert _is_valid_episode(ep), ep.meta


def test_episode_tools_are_standard():
    for split in ("train", "eval"):
        for ep in scenario_episodes(4, seed=8, split=split):
            for m in ep.messages:
                for tc in m.tool_calls:
                    assert tc.name in TOOL_NAMES, tc.name


def _prior_tool_responses(msgs, idx):
    return [m.tool_response for m in msgs[:idx]
            if m.role == Role.tool and m.tool_response]


def test_chained_and_error_followup_grounded_in_prior_response():
    # at least one follow-up tool-call arg value is a literal substring of a PRIOR tool_response
    for split in ("train", "eval"):
        for fam in ("chained", "error_recovery"):
            eps = [ep for ep in scenario_episodes(6, seed=9, split=split)
                   if ep.meta["category"] == fam]
            assert eps, (split, fam)
            for ep in eps:
                msgs = ep.messages
                found = False
                for i, m in enumerate(msgs):
                    if m.role != Role.assistant or not m.tool_calls:
                        continue
                    priors = _prior_tool_responses(msgs, i)
                    for v in m.tool_calls[0].arguments.values():
                        if any(str(v) in r for r in priors):
                            found = True
                assert found, (fam, ep.meta)


def test_error_recovery_has_error_then_successful_retry():
    for split in ("train", "eval"):
        eps = [ep for ep in scenario_episodes(6, seed=10, split=split)
               if ep.meta["category"] == "error_recovery"]
        assert eps
        for ep in eps:
            msgs = ep.messages
            err_idx = next(i for i, m in enumerate(msgs)
                           if m.role == Role.tool and m.tool_response
                           and ("Error" in m.tool_response or "not found" in m.tool_response
                                or "no matches" in m.tool_response or "failed" in m.tool_response))
            # an assistant tool call AFTER the error, followed by a non-error tool response
            retry = next(j for j in range(err_idx + 1, len(msgs))
                         if msgs[j].role == Role.assistant and msgs[j].tool_calls)
            resp = msgs[retry + 1]
            assert resp.role == Role.tool
            assert "Error" not in (resp.tool_response or "")


# ---- disjointness ----------------------------------------------------------------------------
def _single_slot_values(split):
    """All slot values appearing across single-turn samples (prompts + call args)."""
    vals = set()
    for s in scenario_samples(30, seed=11, split=split):
        if s.calls:
            for c in s.calls:
                vals.update(str(v) for v in c["arguments"].values())
        vals.add(s.prompt)  # clarify/abstain skeletons are their own slots
    return vals


def test_train_eval_disjoint_single():
    tr, ev = _single_slot_values("train"), _single_slot_values("eval")
    assert not (tr & ev)


def _episode_arg_values(split):
    vals = set()
    for ep in scenario_episodes(20, seed=12, split=split):
        for m in ep.messages:
            for tc in m.tool_calls:
                vals.update(str(v) for v in tc.arguments.values())
    return vals


def test_train_eval_disjoint_episode_args():
    tr, ev = _episode_arg_values("train"), _episode_arg_values("eval")
    assert not (tr & ev)


# ---- expanded-volume contracts (B. expand) ---------------------------------------------------
def test_clarify_abstain_expanded_counts():
    # ~2-3x the original variety (was 8 train / 5 eval clarify; 10 train / 6 eval abstain).
    assert len(CLARIFY["train"]) >= 18 and len(CLARIFY["eval"]) >= 9
    assert len(ABSTAIN["train"]) >= 18 and len(ABSTAIN["eval"]) >= 9
    # disjoint skeletons across split for both
    for pool in (CLARIFY, ABSTAIN):
        tr = {p for p, _ in pool["train"]}
        ev = {p for p, _ in pool["eval"]}
        assert not (tr & ev)


def test_episode_skeleton_variety_expanded():
    # was 3 workflow / 2 chained / 3 error_recovery; now substantially more distinct builders.
    assert len(_WORKFLOW) >= 5
    assert len(_CHAINED) >= 4
    assert len(_ERROR) >= 5


def test_parallel_variety_expanded():
    # many distinct parallel skeletons appear when sampling broadly
    seen = set()
    for split in ("train", "eval"):
        for s in scenario_samples(40, seed=21, split=split):
            if s.category != "parallel":
                continue
            seen.add(tuple(c["name"] for c in s.calls))
    assert len(seen) >= 6, seen
