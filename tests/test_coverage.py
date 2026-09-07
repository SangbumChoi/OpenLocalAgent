"""Tool-selection coverage tests for the synthetic dataset.

These guard the fixes for systematic coverage gaps that caused the deployed 28M agent to pick
wrong tools:
  (a) implicit factual questions ("How tall is X?", "What's the capital of Y?") -> web_search;
  (b) lookup-tool disambiguation: "What does X mean?" -> define, "latest on X" -> get_news,
      arithmetic -> calculator;
  (c) 0 grounding misses (every copyable string arg is an exact prompt substring);
  (d) train/eval slot-pool disjointness for the new entity pools.
"""

import json

from openlocalagent.agent.schema_decode import extract_pools
from openlocalagent.data.synth.agent_synth import (
    ENTITIES_EVAL,
    ENTITIES_TRAIN,
    EVENTS_EVAL,
    EVENTS_TRAIN,
    INVENTIONS_EVAL,
    INVENTIONS_TRAIN,
    PLACES_EVAL,
    PLACES_TRAIN,
    Generator,
)

# copyable string args ground by exact substring; these instead ground via schema extractors
# (arithmetic expr span, weather unit enum), so they are excluded from the substring check.
EXTRACTOR_ARGS = {"expression", "unit"}


def _target_name(sample) -> str:
    return json.loads(sample.target)["name"] if not sample.calls else sample.ref_name


def test_implicit_factual_questions_map_to_web_search_grounded():
    """Bare factual questions (not "search for X" commands) target web_search with a grounded
    query. Covers the four shapes: how-tall/far, capital/population-of, who-invented, when-did."""
    g = Generator(5, 0, "train")
    seen_shapes = {"measure": False, "attr": False, "who": False, "when": False}
    for _ in range(3000):
        s = g.web_search_implicit()
        assert s.ref_name == "web_search"
        query = json.loads(s.ref_args)["query"]
        assert query in s.prompt  # grounded: query is a literal substring
        # the prompt must NOT be an explicit imperative search command
        assert not s.prompt.lower().startswith(("search ", "look up ", "find information"))
        low = s.prompt.lower()
        if low.startswith(
            (
                "how tall",
                "how high",
                "how far",
                "how old",
                "how long",
                "how deep",
                "how heavy",
                "how big",
                "how wide",
                "do you know how",
                "i wonder how",
            )
        ):
            seen_shapes["measure"] = True
        elif "capital of" in low or "population of" in low or "area of" in low:
            seen_shapes["attr"] = True
        elif low.startswith(("who ", "do you know who", "any idea who")):
            seen_shapes["who"] = True
        elif low.startswith(("when did", "what year", "when was")):
            seen_shapes["when"] = True
    assert all(seen_shapes.values()), seen_shapes


def test_define_vs_news_vs_calc_disambiguation():
    """Meaning questions -> define, current-events -> get_news, arithmetic -> calculator."""
    g = Generator(5, 1, "train")
    # define: "What does X mean?" exists and grounds the term
    got_mean = False
    for _ in range(3000):
        s = g.define()
        assert s.ref_name == "define"
        term = json.loads(s.ref_args)["term"]
        assert term in s.prompt
        if "mean?" in s.prompt:
            got_mean = True
    assert got_mean
    # get_news: "What's the latest on X?" exists and grounds the topic
    got_latest = False
    for _ in range(3000):
        s = g.get_news()
        assert s.ref_name == "get_news"
        topic = json.loads(s.ref_args)["topic"]
        assert topic in s.prompt
        if s.prompt.lower().startswith("what's the latest on"):
            got_latest = True
    assert got_latest
    # calculator: arithmetic phrasings ground the expression via the arithmetic extractor
    got_natural = False
    for _ in range(3000):
        s = g.calc()
        assert s.ref_name == "calculator"
        expr = json.loads(s.ref_args)["expression"]
        pools = extract_pools(s.prompt)
        assert pools["arith"] and pools["arith"][0] == expr
        if s.prompt.lower().startswith(("how much is", "can you compute", "work out")):
            got_natural = True
    assert got_natural


def test_zero_grounding_misses_across_new_coverage():
    """Every copyable string arg is an exact substring of the prompt (across both splits)."""
    for split in ("train", "eval"):
        g = Generator(5, 2, split)
        misses = total = 0
        for x in g.generate(4000):
            calls = x.calls or [{"name": x.ref_name, "arguments": json.loads(x.ref_args or "{}")}]
            for c in calls:
                for arg, val in c["arguments"].items():
                    if not isinstance(val, str) or arg in EXTRACTOR_ARGS:
                        continue
                    total += 1
                    if val not in x.prompt:
                        misses += 1
        assert total > 500
        assert misses == 0, f"{split}: {misses} grounding misses"


def test_new_entity_pools_train_eval_disjoint():
    for tr, ev in [
        (ENTITIES_TRAIN, ENTITIES_EVAL),
        (PLACES_TRAIN, PLACES_EVAL),
        (INVENTIONS_TRAIN, INVENTIONS_EVAL),
        (EVENTS_TRAIN, EVENTS_EVAL),
    ]:
        assert tr and ev
        assert set(tr).isdisjoint(set(ev))


def test_agent_synth_single_turn_prompts_are_disjoint_across_splits():
    train = Generator(level=5, seed=101, split="train")
    evaluation = Generator(level=5, seed=202, split="eval")
    train_prompts: set[str] = set()
    eval_prompts: set[str] = set()
    train_makers = train.makers()
    eval_makers = evaluation.makers()
    assert [maker.__name__ for maker in train_makers] == [maker.__name__ for maker in eval_makers]

    for train_maker, eval_maker in zip(train_makers, eval_makers, strict=True):
        train_prompts.update(train_maker().prompt for _ in range(128))
        eval_prompts.update(eval_maker().prompt for _ in range(128))

    assert train_prompts
    assert eval_prompts
    assert train_prompts.isdisjoint(eval_prompts)


def test_eval_split_implicit_questions_use_eval_pools_only():
    g = Generator(5, 3, "eval")
    allowed = set(ENTITIES_EVAL) | set(PLACES_EVAL) | set(INVENTIONS_EVAL) | set(EVENTS_EVAL)
    saw = False
    for _ in range(2000):
        s = g.web_search_implicit()
        query = json.loads(s.ref_args)["query"]
        assert query in allowed
        saw = True
    assert saw


def test_notify_tools_terminal_in_plan_episodes():
    """send_email / notion_write / slack_send only appear as the LAST tool-call in a plan (no
    nonsensical mid-sequence record/notify steps)."""
    g = Generator(5, 4, "train")
    terminal = {"send_email", "notion_write", "slack_send"}
    for ep in g.plan_episodes(3000):
        plan = ep.meta["plan"]
        for i, name in enumerate(plan):
            if name in terminal:
                assert i == len(plan) - 1, f"{name} not terminal in {plan}"
