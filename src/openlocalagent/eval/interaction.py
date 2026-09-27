"""What a benchmark actually asks of an agent, and what a score from it is allowed to claim.

The ten-suite harness scores one action: render the observation, parse the first tool call,
compare it to a gold call. That works when the first action is decidable from the first
observation. It is not always true, and when it is false the harness returns 0.0 for every model
at every size while looking like a capability measurement.

tau2 is the worked example. 401 of its 402 observations are the *user simulator's* script
("You recently spoke on the phone... you want to cancel..."), not a request addressed to the
agent; and 62 of its 83 opaque-id arguments never appear in the observation at all, because in
the real benchmark the agent obtains them by asking. Flattened to one turn, the answer is not a
function of the input. Every tier scored exactly 0.0% - 10M and 300M alike - while parse_rate
stayed near 60%, which is the signature of a harness fault rather than a capability limit.

So interaction mode is data, not a comment. A suite declares how it must be run, the runner
refuses to score it any other way, and a receipt carries the boundary of what it may claim.

Single-turn is the degenerate case of the general loop - one step, no environment, no user - so
the existing ten suites keep their exact numbers while stateful suites become expressible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from openlocalagent.data.schema import ToolCall


class Interaction(StrEnum):
    """How many turns a suite needs before its gold answer is determined.

    The ordering is a capability ladder: each mode needs everything the one before it needs, plus
    one more moving part. That is also the order of how much a score costs to trust.
    """

    #: One observation, one action, compare to a gold call. Needs nothing but the model.
    SINGLE_ACTION = "single_action"
    #: Calls execute against a simulated world; the score is whether the final state matches a
    #: goal. Needs an Environment; still fully reproducible offline.
    STATEFUL = "stateful"
    #: Stateful, plus the agent must obtain information by talking to a user it cannot see.
    #: Needs a UserSimulator, which in practice is another language model.
    DIALOGUE = "dialogue"

    @property
    def needs_environment(self) -> bool:
        return self is not Interaction.SINGLE_ACTION

    @property
    def needs_user_simulator(self) -> bool:
        return self is Interaction.DIALOGUE

    @property
    def metrics(self) -> tuple[str, ...]:
        """The metric names a run in this mode may report.

        Reporting step_success_rate for a stateful suite is a category error: the gold "first
        action" is an artifact of flattening, not the benchmark's definition of success.
        """
        match self:
            case Interaction.SINGLE_ACTION:
                return ("parse_rate", "type_match", "step_success_rate", "grounding_accuracy")
            case Interaction.STATEFUL | Interaction.DIALOGUE:
                return ("task_success_rate", "pass_k", "mean_steps", "invalid_call_rate")


class Reproducibility(StrEnum):
    """What a suite's score depends on besides this repository.

    The project's claim is that every number is reproducible from open weights and open data. A
    suite needing a judge or user-simulator model breaks that, and the break has to be visible in
    the receipt rather than discovered later by a reader.
    """

    #: Deterministic given the repo and the released row file.
    SELF_CONTAINED = "self_contained"
    #: Requires a model this repo does not train, so the score is conditional on that model.
    EXTERNAL_MODEL = "external_model"


@runtime_checkable
class Environment(Protocol):
    """A world that tool calls act on.

    Deliberately tiny: a suite's environment is the part nobody can share, so the protocol has to
    be cheap to implement per domain. `snapshot` exists so an episode can be replayed and so
    pass_k can re-run from an identical start rather than a nominally identical one.
    """

    def reset(self, seed: int) -> str:
        """Start an episode; return the first observation."""

    def step(self, call: ToolCall) -> str:
        """Execute one tool call; return the resulting observation.

        Takes the project's own ToolCall rather than (name, arguments), so an environment cannot
        drift from the interchange format the rest of the pipeline moves calls in.

        An invalid call is an observation ("no such tool", a validation error), never an
        exception - a model that calls a tool that does not exist should be scored on recovering,
        which is exactly the behaviour a single-action harness cannot see.
        """

    def goal_reached(self) -> bool:
        """Whether the world is now in the state the task asked for."""

    def snapshot(self) -> bytes:
        """Serialised world state, for replay and for pass_k re-runs."""


@runtime_checkable
class UserSimulator(Protocol):
    """Supplies the utterances a DIALOGUE suite's agent has to elicit.

    This is where reproducibility goes to die, so it is a separate protocol from Environment and
    a suite that needs one is marked EXTERNAL_MODEL. Keeping it separate also means a STATEFUL
    suite never accidentally acquires a model dependency.
    """

    def open(self, brief: str) -> str:
        """First user turn, given the persona brief the agent must never see."""

    def reply(self, transcript: tuple[tuple[str, str], ...]) -> str:
        """Next user turn given (role, text) history."""


class Category(StrEnum):
    """What a suite's number is, once its floor and its split have been measured.

    The ten-suite mean averaged all of these as if they were the same kind of number. They are
    not: a sound suite's score is selection ability, a floor-bound suite's is the label
    distribution, a leaked suite's is memorisation, a wall is a constant, and seventeen rows is
    noise. `summarize` reports each kind on its own terms and only averages within a kind.
    """

    #: Near-unique tools, low floors, arguments copyable from the observation.
    SOUND = "sound_tool_calling"
    #: Sound selection, but a text-only projection of a multimodal benchmark.
    PROJECTED = "projected_gui"
    #: Unique prompts counted with repetition.
    REPETITION_WEIGHTED = "repetition_weighted"
    #: type_match is the label distribution; step needs grounding the text cannot give.
    FLOOR_BOUND = "floor_bound"
    #: Other steps of the eval tasks are in training.
    EPISODE_LEAKED = "episode_leaked"
    #: Held out from every stage and unreached by every tier: a constant, not a gradient.
    ZERO_SHOT_WALL = "zero_shot_wall"
    #: Too few rows for one row to be smaller than the differences being claimed.
    NOISE = "noise"
    #: Not scoreable as a single-action suite.
    BLOCKED = "blocked"

    @property
    def averaged(self) -> bool:
        """Only categories whose scores measure the same thing are averaged together."""
        return self in (Category.SOUND, Category.PROJECTED)


@dataclass(frozen=True, slots=True)
class SuiteSpec:
    """A suite and the terms under which its number may be quoted.

    `rows` and `claim_boundary` sit here rather than in prose because both have already misled a
    reader of this project's own results: toolsandbox holds 17 rows, so a single row moves the
    ten-suite mean by 0.59 - larger than the 0.13 that separated the sparse arm from its dense
    control. A spec that carries its own row count makes that arithmetic checkable instead of
    discoverable.
    """

    name: str
    path: Path
    interaction: Interaction
    reproducibility: Reproducibility = Reproducibility.SELF_CONTAINED
    #: What the number means and how it differs from the upstream published metric.
    claim_boundary: str = ""
    #: Rows below this make the suite's contribution to a mean noise-dominated.
    min_rows_for_claim: int = 100
    #: Set when a suite must not be scored as-is; the runner refuses and prints this.
    blocked_reason: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    #: Which kind of number this suite produces; decides what it may be averaged with.
    category: Category = Category.SOUND

    @property
    def scoreable(self) -> bool:
        return not self.blocked_reason

    def resolution(self, rows: int) -> float:
        """Mean-percentage-points moved by one row, at 1/N weight in an N-suite mean.

        The number to compare against a difference before calling it a result.
        """
        return 100.0 / max(rows, 1) / 10.0

    def audit(self, rows: int) -> list[str]:
        """Everything a reader should be told before believing this suite's score."""
        notes: list[str] = []
        if self.blocked_reason:
            notes.append(f"BLOCKED: {self.blocked_reason}")
        if rows < self.min_rows_for_claim:
            notes.append(
                f"{rows} rows is below {self.min_rows_for_claim}: one row moves the ten-suite "
                f"mean by {self.resolution(rows):.2f} points"
            )
        if self.reproducibility is Reproducibility.EXTERNAL_MODEL:
            notes.append("score is conditional on a model this repository does not train")
        if self.claim_boundary:
            notes.append(self.claim_boundary)
        return notes


@dataclass(frozen=True, slots=True)
class Floors:
    """What a predictor that has never seen the observation scores on this suite.

    A score is only readable against its floor. `type_match` on agentnet is 74% for la-93m-matrix
    and 73% for a predictor that always says `click`; on mind2web 86% versus 80% for always
    `web_click`. Those two numbers looked like selection ability until the floor was measured.
    """

    rows: int
    #: type_match of a predictor that always emits the most frequent gold tool name.
    majority_type_match: float
    #: step_success of that same predictor with empty arguments.
    constant_step_success: float
    #: Share of gold calls that take no arguments at all - name alone is a full answer.
    no_argument_share: float
    #: Share of gold argument values that appear verbatim in the observation.
    argument_derivability: float | None
    #: Upper bound on step_success once identical observations with different golds are counted.
    reachable_step_success: float
    #: Share of rows whose every gold argument value is verbatim in the observation - the
    #: step_success a perfect copier could reach. "Cannot be derived" and "was not derived" stop
    #: being the same number.
    fully_derivable_share: float = 1.0

    @classmethod
    def measure(cls, tasks: Sequence[object]) -> Floors:
        import collections
        import hashlib
        import json

        n = len(tasks)
        if n == 0:
            return cls(0, 0.0, 0.0, 0.0, None, 0.0, 0.0)
        names = collections.Counter(t.gold_name for t in tasks)
        majority, majority_n = names.most_common(1)[0]
        no_arg = sum(1 for t in tasks if not t.gold_arguments)
        constant = sum(1 for t in tasks if t.gold_name == majority and not t.gold_arguments)
        derivable = total = 0
        fully = 0
        for t in tasks:
            observation = t.observation.lower()
            row_ok = True
            for value in (t.gold_arguments or {}).values():
                total += 1
                hit = str(value).lower() in observation
                derivable += hit
                row_ok &= hit
            fully += row_ok
        groups: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for t in tasks:
            key = hashlib.sha1(t.observation.encode()).hexdigest()
            groups[key][(t.gold_name, json.dumps(t.gold_arguments, sort_keys=True))] += 1
        reachable = sum(group.most_common(1)[0][1] for group in groups.values())
        return cls(
            rows=n,
            majority_type_match=round(majority_n / n, 4),
            constant_step_success=round(constant / n, 4),
            no_argument_share=round(no_arg / n, 4),
            argument_derivability=round(derivable / total, 4) if total else None,
            reachable_step_success=round(reachable / n, 4),
            fully_derivable_share=round(fully / n, 4),
        )

    def as_record(self) -> dict[str, float | int | None]:
        return {"rows": self.rows,
                "majority_type_match": self.majority_type_match,
                "constant_step_success": self.constant_step_success,
                "no_argument_share": self.no_argument_share,
                "argument_derivability": self.argument_derivability,
                "reachable_step_success": self.reachable_step_success,
                "fully_derivable_share": self.fully_derivable_share}


#: The registry. Every entry states how the suite must be run and what its score may claim, so
#: those facts live in one greppable place rather than in comments beside SUITES.
SPECS: dict[str, SuiteSpec] = {
    "androidcontrol": SuiteSpec(
        "androidcontrol", Path("data/public/androidcontrol-test.jsonl"),
        Interaction.SINGLE_ACTION, category=Category.PROJECTED, tags=("gui_control",),
        claim_boundary="text-only: the screenshot is omitted, so this is below the published "
                       "multimodal number and is not comparable to it"),
    "mind2web": SuiteSpec(
        "mind2web", Path("data/merged-v2/eval-mind2web.jsonl"),
        Interaction.SINGLE_ACTION, category=Category.EPISODE_LEAKED, tags=("web_control", "episode_leaked"),
        claim_boundary="label-leaked: all 252 eval golds (web_click target_id such as '107') "
                       "are verbatim among 4,508 posttrain golds, while the id is in the "
                       "observation once and the observation is the task sentence alone "
                       "(109 chars, no DOM). Exact prompt overlap is 0 and the 38 rows "
                       "outside the 60-char task-prefix overlap score the same as the rest "
                       "(57.9 vs 58.7 on la-300m-matrix), so the leak is the label space, "
                       "not shared tasks. No clean subset exists; the number is memorised "
                       "label structure. Constant web_click floor: type_match 80%."),
    "toolace": SuiteSpec(
        "toolace", Path("data/public/toolace-eval.jsonl"),
        Interaction.SINGLE_ACTION, category=Category.SOUND, tags=("tool_use",)),
    "xlam": SuiteSpec(
        "xlam", Path("data/public/xlam-test.jsonl"),
        Interaction.SINGLE_ACTION, category=Category.SOUND, tags=("tool_use",),
        claim_boundary="90 of 2,941 eval observations (3.1%) are verbatim in the posttrain split "
                       "(scripts/analyze_harness.py --train-overlap, 2026-09-02); the only exact "
                       "leak measured, in the suite the headline rests on"),
    "bfcl": SuiteSpec(
        "bfcl", Path("data/public/bfcl-eval.jsonl"),
        Interaction.SINGLE_ACTION, category=Category.SOUND, tags=("tool_use", "held_out"),
        claim_boundary="appears in no training stage: pure transfer"),
    "agentnet": SuiteSpec(
        "agentnet", Path("data/public/agentnet-eval.jsonl"),
        Interaction.SINGLE_ACTION, category=Category.FLOOR_BOUND, tags=("gui_control", "grounding_without_pixels"),
        claim_boundary="eval split keeps its empty upstream catalog; only the train split was "
                       "catalog-repaired, so this is not inflated against open baselines. "
                       "type_match is the constant floor: always-click scores 73%, la-93m-matrix "
                       "74%. step_success is a click coordinate inferred from a text screen "
                       "description (8% argument derivability) - a grounding task without the "
                       "image, not a tool-calling one. 12,482 rows, so it dominates nothing at "
                       "equal suite weight but is the largest file"),
    "toolbench": SuiteSpec(
        "toolbench", Path("data/public/toolbench-eval.jsonl"),
        Interaction.SINGLE_ACTION, category=Category.REPETITION_WEIGHTED, tags=("tool_use", "duplicated_rows"),
        claim_boundary="553 of 687 rows are exact-duplicate observations (137 groups, all with "
                       "consistent golds; 0 overlap with the train split), so the suite is ~271 "
                       "unique prompts counted with repetition. 47% of golds take no arguments: "
                       "mostly a selection test among ~200 RapidAPI names"),
    "toolsandbox": SuiteSpec(
        "toolsandbox", Path("data/public/toolsandbox-eval.jsonl"),
        # Natively stateful, but the flattening holds: its user turns are real requests
        # ("Turn off cellular") and the gold first calls take no arguments, so the first action
        # is decidable from the first observation. That is why it scores where tau2 cannot.
        Interaction.SINGLE_ACTION, category=Category.NOISE, min_rows_for_claim=100, tags=("tool_use", "stateful_origin"),
        claim_boundary="single-turn projection of a stateful benchmark; 17 rows, so one row is "
                       "0.59 points of the ten-suite mean - wider than differences we have "
                       "called results"),
    "mcpatlas": SuiteSpec(
        "mcpatlas", Path("data/public/mcpatlas-eval.jsonl"),
        Interaction.SINGLE_ACTION, category=Category.ZERO_SHOT_WALL, tags=("tool_use", "held_out", "zero_shot_wall"),
        claim_boundary="appears in no training stage: pure transfer. In practice a wall, not a "
                       "gradient: parse_rate 96-99% but type_match 3-4% and step 0.4-1.4% at every "
                       "tier from 10M to 300M. 43 MCP tool names the model has never seen in "
                       "15-tool catalogs; it emits well-formed calls to the wrong one"),
    "mobileactions": SuiteSpec(
        "mobileactions", Path("data/public/mobileactions-eval.jsonl"),
        Interaction.SINGLE_ACTION, category=Category.PROJECTED, tags=("gui_control", "parse_bound"),
        claim_boundary="the one suite where parse_rate is the bottleneck (70-83%): 332-char "
                       "observations with a date/time preamble and 1.75 multi-field arguments that "
                       "are 85% copyable from the text. Failures are format, not selection"),
    "tau2": SuiteSpec(
        "tau2", Path("data/public/tau2-eval.jsonl"),
        Interaction.DIALOGUE, Reproducibility.EXTERNAL_MODEL, category=Category.BLOCKED, tags=("tool_use", "supplemental"),
        blocked_reason=(
            "measured, not assumed: 401/402 observations are the user-simulator persona brief "
            "rather than a request to the agent, and 62 of 83 opaque-id gold arguments never "
            "appear in the observation because the agent is meant to obtain them by asking. "
            "Per-argument derivability is 48.8% against xlam's 78.7%, capping a perfect model "
            "near 15%. All six tiers scored 0.0% with parse_rate ~60%, which is a harness fault, "
            "not a capability limit. Needs Interaction.DIALOGUE - an environment and a user "
            "simulator - before any number from it means anything."),
    ),
}


def spec_for(name: str) -> SuiteSpec:
    """The suite's terms, or a hard error - an unknown suite name must never become a skip."""
    try:
        return SPECS[name]
    except KeyError:
        raise KeyError(f"unknown suite {name!r}; known: {sorted(SPECS)}") from None


#: The ten suites the campaign's every receipt averaged, in the order the runner scores them.
STANDARD_TEN = ("androidcontrol", "mind2web", "toolace", "xlam", "bfcl", "agentnet", "toolbench",
                "toolsandbox", "mcpatlas", "mobileactions")


def summarize(suites: dict[str, dict]) -> dict:
    """Report a receipt by category rather than by one mean.

    A mean is taken only within a category whose scores measure the same thing, and only over
    suites with enough rows for one row to move it less than the differences being claimed. The
    ten-suite mean is kept, labelled, because every earlier receipt is on that scale.
    """
    def step(name: str) -> float | None:
        block = suites.get(name)
        return None if block is None else block.get("step_success_rate")

    by_category: dict[str, dict] = {}
    excluded: dict[str, str] = {}
    for name, spec in SPECS.items():
        value = step(name)
        if value is None:
            continue
        rows = suites[name].get("rows", 0)
        if rows < spec.min_rows_for_claim:
            excluded[name] = f"{rows} rows < {spec.min_rows_for_claim}: contributes to no mean"
        entry = by_category.setdefault(
            spec.category.value, {"suites": {}, "averaged": spec.category.averaged})
        floors = suites[name].get("floors", {})
        entry["suites"][name] = {
            "step_success_rate": value,
            "floor": floors.get("constant_step_success"),
            "ceiling": floors.get("fully_derivable_share"),
        }
    for entry in by_category.values():
        members = [v["step_success_rate"] for k, v in entry["suites"].items() if k not in excluded]
        entry["mean"] = (round(sum(members) / len(members), 6)
                         if (members and entry["averaged"]) else None)
    sound = by_category.get(Category.SOUND.value, {}).get("mean")
    ten = [step(name) for name in STANDARD_TEN]
    return {
        "headline": {"metric": "sound_tool_calling_mean", "value": sound,
                     "suites": [n for n, spec in SPECS.items() if spec.category is Category.SOUND]},
        "by_category": by_category,
        "excluded_from_means": excluded,
        "legacy_ten_suite_mean": (round(sum(ten) / len(ten), 6)
                                  if all(v is not None for v in ten) else None),
        "legacy_note": "mean over the standard ten including floor-bound, leaked, wall and noise "
                       "suites; kept for continuity with every earlier receipt, not for reasoning",
    }
