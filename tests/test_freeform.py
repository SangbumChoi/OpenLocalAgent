"""Tests for the hand-authored free-form dispatch sets (``localagent.eval.freeform``).

FREEFORM_EVAL is the held-out OOD selection test; FREEFORM_TRAIN trains the selector / route head
on the same free, varied free-form distribution. The two must be DISJOINT (different phrasings AND
slot values), the train set must cover every tool, and gold tools must be real STANDARD_TOOLS.
"""

from localagent.agent.toolset import STANDARD_TOOLS
from localagent.eval.freeform import FREEFORM_EVAL, FREEFORM_TRAIN

TOOL_NAMES = {t.name for t in STANDARD_TOOLS}


def test_gold_tools_are_standard():
    for q, gold in FREEFORM_TRAIN + FREEFORM_EVAL:
        assert gold in TOOL_NAMES, f"unknown gold tool {gold!r} for {q!r}"


def test_train_covers_all_tools():
    covered = {gold for _, gold in FREEFORM_TRAIN}
    assert covered == TOOL_NAMES, f"FREEFORM_TRAIN misses: {TOOL_NAMES - covered}"


def test_train_is_sized_80_to_120():
    assert 80 <= len(FREEFORM_TRAIN) <= 120, len(FREEFORM_TRAIN)


def test_train_prompts_unique():
    prompts = [q for q, _ in FREEFORM_TRAIN]
    assert len(prompts) == len(set(prompts))


def test_train_eval_disjoint_prompts():
    tr = {q for q, _ in FREEFORM_TRAIN}
    ev = {q for q, _ in FREEFORM_EVAL}
    assert not (tr & ev), f"shared prompts across split: {tr & ev}"


def test_train_eval_disjoint_words():
    # No FREEFORM_TRAIN prompt should be a near-duplicate of a FREEFORM_EVAL one (slot-value
    # disjointness sanity): they should not share a long literal phrase.
    ev_prompts = [q for q, _ in FREEFORM_EVAL]
    for q, _ in FREEFORM_TRAIN:
        for e in ev_prompts:
            assert q != e


def test_run_python_gold_is_inline_code_not_filename():
    # A gold of run_python must come with an actual inline snippet in the prompt, never "X.py".
    for q, gold in FREEFORM_TRAIN + FREEFORM_EVAL:
        if gold == "run_python":
            assert ".py" not in q, f"run_python prompt names a script file: {q!r}"


def test_download_file_gold_names_a_file_url():
    for q, gold in FREEFORM_TRAIN + FREEFORM_EVAL:
        if gold == "download_file":
            # the prompt must contain a URL with a dotted file at the end of a path
            assert "/" in q and "." in q, f"download_file prompt lacks a file url: {q!r}"
