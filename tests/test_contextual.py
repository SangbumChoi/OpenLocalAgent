"""Tests for the referent-conditioned dispatch data source.

Asserts the contracts retrain code relies on: every branch's gold tool appears, literal-substring
arg grounding, a TRUE train/eval split (disjoint skeletons AND slot values), and — the point of
the module — concrete ambiguity coverage: near-identical instructions flip the gold tool with
the referent type.
"""

import json
import random

from localagent.data.contextual import (
    CONTEXTUAL_EXAMPLES,
    GROUPS,
    _render,
    contextual_samples,
)

N_BRANCHES = sum(len(g["branches"]) for g in GROUPS.values())
BRANCH_TOOLS = {(f"ctx_{gname}", b["tool"])
                for gname, g in GROUPS.items() for b in g["branches"]}
ALL_TOOLS = {tool for _, tool in BRANCH_TOOLS}


def test_every_branch_gold_tool_appears():
    for split in ("train", "eval"):
        samples = contextual_samples(n=2, seed=0, split=split)
        covered = {(s.category, s.ref_name) for s in samples}
        assert covered == BRANCH_TOOLS, f"missing branches: {BRANCH_TOOLS - covered}"


def test_balanced_per_branch():
    n = 5
    samples = contextual_samples(n=n, seed=1, split="train")
    assert len(samples) == n * N_BRANCHES
    counts: dict[tuple, int] = {}
    for s in samples:
        counts[(s.category, s.ref_name)] = counts.get((s.category, s.ref_name), 0) + 1
    assert all(c == n for c in counts.values())


def test_targets_canonical_tool_calls():
    for split in ("train", "eval"):
        for s in contextual_samples(n=2, seed=2, split=split):
            assert s.kind == "tool"
            obj = json.loads(s.target)
            assert obj["name"] == s.ref_name
            assert json.loads(s.ref_args) == obj["arguments"]
            # canonical: compact separators, sorted keys
            assert s.target == json.dumps(obj, separators=(",", ":"), sort_keys=True)


def test_arg_values_are_literal_substrings_of_prompt():
    for split in ("train", "eval"):
        for s in contextual_samples(n=6, seed=3, split=split):
            for val in json.loads(s.ref_args).values():
                assert str(val) in s.prompt, (
                    f"{s.ref_name}: {val!r} not in prompt {s.prompt!r}")


def _masked_skeletons(split):
    """Prompt with slot values masked, keyed per (group, tool) branch."""
    sk: dict[tuple, set] = {}
    for s in contextual_samples(n=40, seed=7, split=split):
        masked = s.prompt
        for val in json.loads(s.ref_args).values():
            masked = masked.replace(str(val), "<V>")
        sk.setdefault((s.category, s.ref_name), set()).add(masked)
    return sk


def test_train_eval_skeletons_disjoint():
    tr, ev = _masked_skeletons("train"), _masked_skeletons("eval")
    for branch in BRANCH_TOOLS:
        overlap = tr[branch] & ev[branch]
        assert not overlap, f"{branch} shares phrasing skeletons across split: {overlap}"


def test_train_eval_slot_values_disjoint():
    def values(split):
        v: dict[tuple, set] = {}
        for s in contextual_samples(n=40, seed=9, split=split):
            for val in json.loads(s.ref_args).values():
                v.setdefault((s.category, s.ref_name), set()).add(str(val))
        return v

    tr, ev = values("train"), values("eval")
    for branch in BRANCH_TOOLS:
        if not tr.get(branch) and not ev.get(branch):
            continue  # arg-less branch (run_tests / git_status / git_diff)
        assert not (tr[branch] & ev[branch]), f"{branch} leaks slot values across split"
    # pools must also be disjoint module-wide, not just per branch
    all_tr = set().union(*tr.values())
    all_ev = set().union(*ev.values())
    assert not (all_tr & all_ev)


def test_same_skeleton_flips_tool_by_referent():
    """Concrete ambiguity coverage: render every branch of a group from the SAME skeleton —
    instruction words identical, gold tool different — and require >= 7 such groups."""
    flip_groups = 0
    for gname, gspec in GROUPS.items():
        for split in ("train", "eval"):
            skeleton = gspec["skeletons"][split][0]
            prefix, suffix = skeleton.split("{v}")
            tools = set()
            for branch in gspec["branches"]:
                s = _render(skeleton, branch, random.Random(0), split, gname)
                # near-identical instruction words: same prefix and suffix around the referent
                assert s.prompt.startswith(prefix) and s.prompt.endswith(suffix)
                tools.add(s.ref_name)
            # every branch of the group routes the SAME wording to a DIFFERENT tool
            assert len(tools) == len(gspec["branches"]) >= 3, (gname, split, tools)
        flip_groups += 1
    assert flip_groups >= 7


def test_ambiguity_present_in_sampled_output():
    # the generated corpus itself must contain the flips: per instruction-group (category),
    # samples share near-identical wording but carry >= 3 distinct gold tools
    samples = contextual_samples(n=4, seed=5, split="train")
    by_group: dict[str, set] = {}
    for s in samples:
        by_group.setdefault(s.category, set()).add(s.ref_name)
    assert len(by_group) == len(GROUPS)
    for group, tools in by_group.items():
        assert len(tools) >= 3, f"{group} is not ambiguous: {tools}"


def test_contextual_examples_cover_branch_tools_from_train_pool():
    assert set(CONTEXTUAL_EXAMPLES.keys()) == ALL_TOOLS
    train_skel = {t for g in GROUPS.values() for t in g["skeletons"]["train"]}
    for tool, phrases in CONTEXTUAL_EXAMPLES.items():
        assert phrases, f"{tool} has no example phrasings"
        for p in phrases:
            assert isinstance(p, str) and p
            # each example instantiates a TRAIN skeleton (no eval phrasing leakage)
            assert any(p.startswith(s.split("{v}")[0]) and p.endswith(s.split("{v}")[1])
                       for s in train_skel), p
    # confusable tools share instruction wording in the examples (e.g. "Open ..." for both
    # open_url and open_app) — that is the enrichment the selector needs
    first_words = {t: {p.split()[0] for p in CONTEXTUAL_EXAMPLES[t]}
                   for t in ("open_url", "open_app", "read_file")}
    assert first_words["open_url"] & first_words["open_app"] & first_words["read_file"]


def test_determinism_and_split_validation():
    a = contextual_samples(n=3, seed=42, split="train")
    b = contextual_samples(n=3, seed=42, split="train")
    assert [(s.prompt, s.target) for s in a] == [(s.prompt, s.target) for s in b]
    try:
        contextual_samples(split="dev")
    except ValueError:
        pass
    else:
        raise AssertionError("split='dev' should raise ValueError")
