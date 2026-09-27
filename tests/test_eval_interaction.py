"""The registry has to agree with the files it describes, or it is decoration.

Every fact asserted here was measured on the box before it was written down. The point of the
module under test is that a suite's terms travel with the suite; the point of this test is that
those terms cannot drift from the data.
"""

from pathlib import Path

import pytest

from openlocalagent.eval import suite as S
from openlocalagent.eval.interaction import (
    SPECS,
    Interaction,
    Reproducibility,
    SuiteSpec,
    spec_for,
)


def test_every_scored_suite_has_a_spec():
    """SUITES is what runs; SPECS is what it is allowed to claim. A gap means an unaudited score."""
    missing = set(S.SUITES) - set(SPECS)
    assert not missing, f"scored suites with no declared terms: {sorted(missing)}"


def test_spec_paths_match_the_registry_the_runner_uses():
    for name, path in S.SUITES.items():
        assert Path(spec_for(name).path).name == Path(path).name, (
            f"{name}: spec points at a different row file than the runner loads"
        )


def test_unknown_suite_is_a_hard_error():
    """An unrecognised name must not degrade into a skip - that hides a typo as a missing result."""
    with pytest.raises(KeyError):
        spec_for("androidcontroll")


def test_single_action_needs_nothing_but_the_model():
    mode = Interaction.SINGLE_ACTION
    assert not mode.needs_environment and not mode.needs_user_simulator
    assert "step_success_rate" in mode.metrics


def test_stateful_modes_do_not_offer_step_success():
    """Scoring a stateful suite by its flattened first action is the tau2 mistake, in type form."""
    for mode in (Interaction.STATEFUL, Interaction.DIALOGUE):
        assert mode.needs_environment
        assert "step_success_rate" not in mode.metrics
        assert "task_success_rate" in mode.metrics


def test_dialogue_alone_needs_a_user_simulator():
    assert Interaction.DIALOGUE.needs_user_simulator
    assert not Interaction.STATEFUL.needs_user_simulator


def test_tau2_is_blocked_and_says_why():
    spec = spec_for("tau2")
    assert not spec.scoreable
    assert spec.interaction is Interaction.DIALOGUE
    assert spec.reproducibility is Reproducibility.EXTERNAL_MODEL
    # The reason has to carry the evidence, not just the verdict.
    assert "401/402" in spec.blocked_reason and "48.8%" in spec.blocked_reason


def test_tau2_is_not_one_of_the_ten():
    """Blocking it must not have quietly changed the headline metric's denominator."""
    assert "tau2" not in S.SUITES
    assert len(S.SUITES) == 10


def test_resolution_flags_the_suite_that_a_single_row_swings():
    """toolsandbox: 17 rows at 1/10 weight is 0.59 mean points per row.

    Recorded because the sparse arm beat its dense control by 0.13 - a quarter of one row.
    """
    spec = spec_for("toolsandbox")
    assert spec.resolution(17) == pytest.approx(0.588, abs=0.01)
    assert any("0.59" in note for note in spec.audit(17))


def test_large_suite_needs_no_row_warning():
    assert not any("below" in note for note in spec_for("xlam").audit(2941))


def test_held_out_suites_declare_pure_transfer():
    """BFCL and MCP-Atlas appear in no training stage; the receipt should say so on its own."""
    for name in ("bfcl", "mcpatlas"):
        assert "transfer" in spec_for(name).claim_boundary
        assert "held_out" in spec_for(name).tags


def test_androidcontrol_declares_the_text_only_gap():
    assert "screenshot" in spec_for("androidcontrol").claim_boundary


def test_audit_of_a_clean_suite_is_empty():
    assert spec_for("toolace").audit(949) == []


def test_spec_is_frozen():
    spec = SuiteSpec("x", Path("x"), Interaction.SINGLE_ACTION)
    with pytest.raises(AttributeError):
        spec.name = "y"  # type: ignore[misc]


# --- floors: what a predictor that never reads the observation scores ---------------------

from dataclasses import dataclass as _dc  # noqa: E402


@_dc(frozen=True)
class _T:
    observation: str
    gold_name: str
    gold_arguments: dict


def test_floors_majority_and_constant():
    """agentnet in miniature: three clicks, one type. Always-click scores 75% type_match."""
    from openlocalagent.eval.interaction import Floors

    tasks = [_T("a", "click", {"x": 1}), _T("b", "click", {"x": 2}),
             _T("c", "click", {}), _T("d", "type", {"text": "hi"})]
    f = Floors.measure(tasks)
    assert f.majority_type_match == 0.75
    assert f.constant_step_success == 0.25          # the one argument-free click
    assert f.no_argument_share == 0.25


def test_floors_argument_derivability_counts_values_in_the_observation():
    from openlocalagent.eval.interaction import Floors

    tasks = [_T("weather in Tokyo please", "get_weather", {"city": "Tokyo", "unit": "C"})]
    assert Floors.measure(tasks).argument_derivability == 0.5


def test_floors_reachable_drops_when_identical_observations_disagree():
    """tau2 in miniature: one brief, three different golds - a ceiling below 100%."""
    from openlocalagent.eval.interaction import Floors

    same = "You want to cancel your reservation."
    tasks = [_T(same, "get_user_details", {"user_id": "a"}),
             _T(same, "get_user_details", {"user_id": "b"}),
             _T(same, "find_user_id_by_name_zip", {}),
             _T("different", "x", {})]
    assert Floors.measure(tasks).reachable_step_success == 0.5   # best single answer per group


def test_floors_of_nothing():
    from openlocalagent.eval.interaction import Floors

    assert Floors.measure([]).rows == 0


# --- categories, ceilings and the receipt summary -------------------------------------------

from openlocalagent.eval.interaction import STANDARD_TEN, Category, Floors, summarize  # noqa: E402
from openlocalagent.eval.suite import RowOutcome, rates  # noqa: E402


def test_every_spec_has_a_category_and_only_sound_or_projected_are_averaged():
    for spec in SPECS.values():
        assert isinstance(spec.category, Category)
    assert {n for n, s in SPECS.items() if s.category is Category.SOUND} == {"xlam", "toolace", "bfcl"}
    assert {c for c in Category if c.averaged} == {Category.SOUND, Category.PROJECTED}
    assert SPECS["tau2"].category is Category.BLOCKED
    assert set(STANDARD_TEN) == set(SPECS) - {"tau2"}


def test_floors_fully_derivable_share_is_row_level():
    tasks = [_T("foo and bar", "a", {"x": "foo", "y": "bar"}),
             _T("foo only", "a", {"x": "foo", "y": "zap"})]
    f = Floors.measure(tasks)
    assert f.argument_derivability == 0.75      # 3 of 4 values
    assert f.fully_derivable_share == 0.5       # 1 of 2 rows
    assert "fully_derivable_share" in f.as_record()


def test_summarize_averages_within_category_and_excludes_small_suites():
    suites = {name: {"rows": 500, "step_success_rate": 0.5, "floors": {"constant_step_success": 0.0}}
              for name in STANDARD_TEN}
    suites["xlam"]["step_success_rate"] = 0.6
    suites["toolace"]["step_success_rate"] = 0.3
    suites["bfcl"]["step_success_rate"] = 0.3
    suites["toolsandbox"] = {"rows": 17, "step_success_rate": 1.0, "floors": {}}
    s = summarize(suites)
    assert s["headline"]["metric"] == "sound_tool_calling_mean"
    assert s["headline"]["value"] == pytest.approx(0.4)
    assert s["by_category"]["projected_gui"]["mean"] == pytest.approx(0.5)
    assert s["by_category"]["floor_bound"]["mean"] is None       # reported, never averaged
    assert "toolsandbox" in s["excluded_from_means"]
    assert s["legacy_ten_suite_mean"] == pytest.approx((0.6 + 0.3 + 0.3 + 0.5 * 6 + 1.0) / 10)


def test_summarize_tolerates_partial_receipts():
    s = summarize({"xlam": {"rows": 100, "step_success_rate": 0.2}})
    assert s["headline"]["value"] == pytest.approx(0.2)
    assert s["legacy_ten_suite_mean"] is None


def test_rates_derive_name_given_parse_and_unique_prompt_metrics():
    o = [RowOutcome("h1", True, True, True, None, False),
         RowOutcome("h1", True, True, False, None, False),   # same prompt, half right
         RowOutcome("h2", True, False, False, None, True),
         RowOutcome("h3", False, False, False, None, False)]  # parse failure
    r = rates(o)
    assert r["parse_rate"] == 0.75
    assert r["type_match"] == 0.5
    assert r["step_success_rate"] == 0.25
    assert r["arguments_given_name"] == 0.5             # 1 step of 2 typed
    assert r["step_success_given_parse"] == pytest.approx(1 / 3)
    assert r["unique_prompts"] == 3
    assert r["unique_prompt_step_success"] == pytest.approx((0.5 + 0 + 0) / 3)
    assert rates([])["step_success_rate"] == 0.0


def test_score_reports_an_all_leaked_split_as_empty_not_zero():
    from openlocalagent.eval.suite import score

    class Adapter:
        def predict_many(self, tasks, max_new_tokens):
            return ['{"name": "a", "arguments": {}}' for _ in tasks]

        def predict(self, task, max_new_tokens):
            return '{"name": "a", "arguments": {}}'

    tasks = [_T("obs one", "a", {}), _T("obs two", "a", {})]
    import hashlib
    leaked = {hashlib.sha1(t.observation.encode()).hexdigest() for t in tasks}
    r = score(Adapter(), tasks, max_new_tokens=8, leaked=leaked)
    assert r["step_success_rate"] == 1.0
    assert r["clean"]["rows"] == 0 and r["clean"]["step_success_rate"] is None
    half = score(Adapter(), tasks, max_new_tokens=8, leaked={next(iter(leaked))})
    assert half["clean"] == {"rows": 1, "leaked_rows": 1, "type_match": 1.0, "step_success_rate": 1.0}
