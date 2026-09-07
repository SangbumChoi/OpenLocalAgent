"""Curriculum ordering (LFM2-style easy->hard) for SFT samples."""

from openlocalagent.data.synth.agent_synth import (
    Generator,
    curriculum_order,
    difficulty_score,
)


def test_difficulty_ordering_easy_to_hard():
    g = Generator(level=2, seed=0, split="train")
    text = g.text()              # no tool, no args  -> easiest
    no_arg = g.run_tests()       # tool call, no args
    single = g.weather()         # single tool, one copy arg
    par = g.parallel()           # two tool calls    -> hardest
    s_text = difficulty_score(text)
    s_noarg = difficulty_score(no_arg)
    s_single = difficulty_score(single)
    s_par = difficulty_score(par)
    # text/no-arg are easiest; a single copy-arg call is harder; parallel hardest.
    assert s_text < s_single
    assert s_noarg < s_single
    assert s_single < s_par


def test_abstention_scores_above_plain_text():
    g = Generator(level=2, seed=1, split="train")
    plain = g.text()
    abstain = g.no_tool()
    assert difficulty_score(abstain) > difficulty_score(plain)


def test_curriculum_order_is_ascending_and_deterministic():
    samples = Generator(level=2, seed=3, split="train").generate(200)
    ordered = curriculum_order(samples)
    scores = [difficulty_score(s) for s in ordered]
    assert scores == sorted(scores)               # ascending easy->hard
    # deterministic + independent of input order
    again = curriculum_order(list(reversed(samples)))
    assert [s.prompt for s in ordered] == [s.prompt for s in again]
    # does not mutate the input
    assert len(samples) == len(ordered)


def test_curriculum_order_preserves_the_set():
    samples = Generator(level=2, seed=4, split="train").generate(120)
    ordered = curriculum_order(samples)
    assert sorted(s.prompt for s in samples) == sorted(s.prompt for s in ordered)
