"""Latency and quality metrics for interactive local agents.

Token throughput is not sufficient to characterize a tool-using agent. A tool cannot execute
until the *complete, valid action* is available, so this module treats time-to-first-action (TTFA)
as the primary latency and keeps token throughput as a diagnostic for autoregressive baselines.

The functions are intentionally runtime-agnostic. Browser/WebGPU measurements exported by
``spaces/localagent-webgpu/benchmark.html`` and measurements from native runtimes can therefore be
summarized with the same definitions.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral, Real
from typing import Any

DEFAULT_DEADLINES_MS = (100.0, 250.0, 500.0, 1000.0, 2000.0)
DEFAULT_RATE_CALIBRATION_TPS = (200.0, 400.0, 600.0)


def _finite_float(value: Any, *, name: str, non_negative: bool = False) -> float:
    """Convert a real-valued metric to ``float`` and reject invalid JSON numbers."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if non_negative and result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def percentile(values: Sequence[float], q: float) -> float:
    """Return a linearly interpolated percentile for ``q`` in ``[0, 1]``.

    This matches the common ``(n - 1) * q`` definition and is deterministic for small benchmark
    samples. An empty or non-finite sample is a caller error rather than silently producing NaN.
    """

    if len(values) == 0:
        raise ValueError("percentile requires at least one value")
    quantile = _finite_float(q, name="q")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("q must be between 0 and 1")
    ordered = sorted(
        _finite_float(value, name=f"values[{index}]") for index, value in enumerate(values)
    )
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def latency_summary(values_ms: Sequence[float]) -> dict[str, float | int]:
    """Summarize a non-empty latency sample in milliseconds."""

    if len(values_ms) == 0:
        raise ValueError("latency_summary requires at least one value")
    values = [
        _finite_float(value, name=f"latencies[{index}]", non_negative=True)
        for index, value in enumerate(values_ms)
    ]
    total = sum(values)
    if not math.isfinite(total):
        raise ValueError("sum of latencies must be finite")
    return {
        "count": len(values),
        "min": min(values),
        "mean": total / len(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def autoregressive_ttfa_ms(
    output_tokens: int,
    *,
    ttft_ms: float,
    decode_tokens_per_second: float,
    postprocess_ms: float = 0.0,
) -> float:
    """Estimate when a complete autoregressive action becomes executable.

    ``output_tokens`` is the number of decode steps required before completion is known, including
    the first token produced at TTFT and any required EOS/terminal step. Each remaining step costs
    one time-per-output-token interval, so:

    ``TTFA = TTFT + (output_tokens - 1) / decode_rate + postprocess``.

    A one-token output has no iterative-decode interval, so a zero decode rate is accepted only
    for that edge case. The returned latency is always finite.
    """

    token_count = _positive_integer(output_tokens, name="output_tokens")
    decode_rate = _finite_float(
        decode_tokens_per_second, name="decode_tokens_per_second"
    )
    remaining_tokens = token_count - 1
    if decode_rate < 0 or (remaining_tokens > 0 and decode_rate == 0):
        raise ValueError(
            "decode_tokens_per_second must be non-negative, and positive "
            "when output_tokens is greater than one"
        )
    first_token_ms = _finite_float(ttft_ms, name="ttft_ms", non_negative=True)
    validation_ms = _finite_float(
        postprocess_ms, name="postprocess_ms", non_negative=True
    )
    try:
        decode_ms = (
            0.0 if remaining_tokens == 0 else 1000.0 * remaining_tokens / decode_rate
        )
        result = first_token_ms + decode_ms + validation_ms
    except OverflowError as exc:
        raise ValueError("computed TTFA must be finite") from exc
    if not math.isfinite(result):
        raise ValueError("computed TTFA must be finite")
    return result


def required_decode_tokens_per_second(
    output_tokens: int,
    *,
    deadline_ms: float,
    ttft_ms: float,
    postprocess_ms: float = 0.0,
) -> float:
    """Return the minimum decode rate needed for a complete action by ``deadline_ms``.

    ``math.inf`` means that no finite decode rate can meet the deadline because TTFT and
    post-processing have already consumed it. A one-token output requires no iterative decode
    once the first token is available and therefore returns zero when the fixed costs fit.
    ``output_tokens`` counts decode steps through any required EOS/terminal step, matching
    :func:`autoregressive_ttfa_ms`.
    A mathematically finite rate outside the representable float range is rejected instead of
    being conflated with that ``math.inf`` sentinel.
    """

    token_count = _positive_integer(output_tokens, name="output_tokens")
    deadline = _finite_float(deadline_ms, name="deadline_ms")
    if deadline <= 0:
        raise ValueError("deadline_ms must be finite and positive")
    first_token_ms = _finite_float(ttft_ms, name="ttft_ms", non_negative=True)
    validation_ms = _finite_float(
        postprocess_ms, name="postprocess_ms", non_negative=True
    )

    available_ms = deadline - first_token_ms - validation_ms
    remaining_tokens = token_count - 1
    if available_ms < 0 or (available_ms == 0 and remaining_tokens > 0):
        return math.inf
    if remaining_tokens == 0:
        return 0.0
    try:
        result = 1000.0 * remaining_tokens / available_ms
    except OverflowError as exc:
        raise ValueError("required decode rate exceeds the finite float range") from exc
    if not math.isfinite(result):
        raise ValueError("required decode rate exceeds the finite float range")
    return result


def calibrate_autoregressive_rate_scenarios(
    decode_steps: Sequence[int],
    *,
    decode_rates_tps: Sequence[float] = DEFAULT_RATE_CALIBRATION_TPS,
    deadlines_ms: Sequence[float] = DEFAULT_DEADLINES_MS,
    predecode_ms: float,
    postprocess_ms: float = 0.0,
) -> dict[str, Any]:
    """Evaluate fixed-rate AR counterfactuals without inventing a universal realtime threshold.

    This is a deterministic scenario calculation, not a hardware benchmark or a sampled action
    distribution. ``decode_steps`` includes the first token and every required terminal/EOS step.
    ``predecode_ms`` is the complete fixed cost through the first decoded token and may therefore
    combine tokenization and TTFT for a harness-TTFA scenario. Tool execution, page rendering,
    observation capture, and later agent steps are deliberately outside the complete-action
    estimand.

    The result reports every scenario separately and intentionally emits no mean or percentile
    over the caller-provided grid: treating an arbitrary list of action lengths as an empirical
    distribution would create a misleading performance statistic.
    """

    lengths: list[int] = []
    seen_lengths: set[int] = set()
    for index, value in enumerate(decode_steps):
        length = _positive_integer(value, name=f"decode_steps[{index}]")
        if length in seen_lengths:
            raise ValueError(f"duplicate decode step count {length}")
        seen_lengths.add(length)
        lengths.append(length)
    if not lengths:
        raise ValueError("at least one decode step count is required")

    rates: list[tuple[str, float]] = []
    seen_rate_keys: set[str] = set()
    for index, value in enumerate(decode_rates_tps):
        rate = _finite_float(value, name=f"decode_rates_tps[{index}]")
        if rate <= 0:
            raise ValueError("decode rates must be finite and positive")
        key = f"{rate:.17g}"
        if key in seen_rate_keys:
            raise ValueError(f"duplicate decode rate {rate:g} tokens/s")
        seen_rate_keys.add(key)
        rates.append((key, rate))
    if not rates:
        raise ValueError("at least one decode rate is required")

    deadlines = _validated_deadlines(deadlines_ms)
    fixed_predecode_ms = _finite_float(
        predecode_ms,
        name="predecode_ms",
        non_negative=True,
    )
    fixed_postprocess_ms = _finite_float(
        postprocess_ms,
        name="postprocess_ms",
        non_negative=True,
    )

    by_rate: dict[str, Any] = {}
    for rate_key, rate in rates:
        scenarios: dict[str, Any] = {}
        for length in lengths:
            ttfa_ms = autoregressive_ttfa_ms(
                length,
                ttft_ms=fixed_predecode_ms,
                decode_tokens_per_second=rate,
                postprocess_ms=fixed_postprocess_ms,
            )
            decode_ms = ttfa_ms - fixed_predecode_ms - fixed_postprocess_ms
            deadline_results = {
                deadline_key: {
                    "deadline_ms": deadline,
                    "meets_deadline": ttfa_ms <= deadline,
                    "slack_ms": deadline - ttfa_ms,
                }
                for deadline_key, deadline in deadlines
            }
            scenarios[str(length)] = {
                "decode_steps": length,
                "iterative_decode_intervals": length - 1,
                "iterative_decode_ms": decode_ms,
                "ttfa_ms": ttfa_ms,
                "deadline_attainment_ms": deadline_results,
            }
        by_rate[rate_key] = {
            "decode_tokens_per_second": rate,
            "scenarios_by_decode_steps": scenarios,
        }

    return {
        "schema_version": 1,
        "artifact_type": "autoregressive_decode_rate_counterfactual",
        "empirical_measurement": False,
        "estimand": "complete_action_latency_under_fixed_costs_and_constant_decode_rate",
        "claim_scope": (
            "fixed-cost autoregressive TTFA scenarios only; not a universal computer-use "
            "realtime threshold or a closed-loop browser measurement"
        ),
        "decode_steps_semantics": (
            "model steps through the first token and every required terminal or EOS step"
        ),
        "fixed_costs_ms": {
            "predecode_through_first_token": fixed_predecode_ms,
            "postprocess": fixed_postprocess_ms,
        },
        "decode_steps": lengths,
        "decode_rates_tps": [rate for _, rate in rates],
        "deadlines_ms": [deadline for _, deadline in deadlines],
        "scenario_weighting": None,
        "aggregate_statistics_emitted": False,
        "by_decode_rate_tps": by_rate,
    }


def _optional_boolean_values(
    records: Sequence[Mapping[str, Any]], key: str
) -> list[bool] | None:
    """Return a complete optional boolean column, rejecting partial or truthy data."""

    presence = [key in record for record in records]
    if not any(presence):
        return None
    if not all(presence):
        raise ValueError(f"{key!r} must be present in every record or omitted from every record")
    values = [record[key] for record in records]
    if any(not isinstance(value, bool) for value in values):
        raise ValueError(f"{key!r} must contain JSON booleans")
    return values


def _validated_deadlines(deadlines_ms: Sequence[float]) -> list[tuple[str, float]]:
    deadlines: list[tuple[str, float]] = []
    seen: set[str] = set()
    for index, value in enumerate(deadlines_ms):
        deadline = _finite_float(value, name=f"deadlines_ms[{index}]")
        if deadline <= 0:
            raise ValueError("deadlines must be finite and positive")
        # Seventeen significant digits round-trip every finite IEEE-754 binary64 value. The
        # shorter default ``:g`` precision can collapse two distinct deadlines into one JSON key.
        key = f"{deadline:.17g}"
        if key in seen:
            raise ValueError(f"duplicate deadline {deadline:g} ms")
        seen.add(key)
        deadlines.append((key, deadline))
    if not deadlines:
        raise ValueError("at least one deadline is required")
    return deadlines


def _non_negative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _integer_seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("bootstrap_seed must be an integer")
    return int(value)


def _case_success_counts(
    records: Iterable[Mapping[str, Any]],
    *,
    deadline_ms: float,
    latency_key: str,
    cluster_key: str,
) -> dict[str, tuple[int, int]]:
    """Reduce repeat-level outcomes to useful and attempted counts per held-out case."""

    rows = list(records)
    if not rows:
        raise ValueError("at least one action record is required")
    deadline = _finite_float(deadline_ms, name="deadline_ms")
    if deadline <= 0:
        raise ValueError("deadline_ms must be finite and positive")

    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"record {index} must be a mapping")
    successes = _optional_boolean_values(rows, "success")
    schema_validity = _optional_boolean_values(rows, "schema_valid")
    if successes is None or schema_validity is None:
        raise ValueError("cluster bootstrap requires complete 'success' and 'schema_valid' columns")

    outcomes: dict[str, list[bool]] = {}
    for index, row in enumerate(rows):
        if latency_key not in row:
            raise ValueError(
                f"record {index} missing required latency field {latency_key!r}"
            )
        latency = _finite_float(
            row[latency_key],
            name=f"records[{index}][{latency_key!r}]",
            non_negative=True,
        )
        if cluster_key not in row:
            raise ValueError(
                f"record {index} missing required cluster field {cluster_key!r}"
            )
        case_id = row[cluster_key]
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(
                f"records[{index}][{cluster_key!r}] must be a non-empty string"
            )
        outcomes.setdefault(case_id, []).append(
            successes[index] and schema_validity[index] and latency <= deadline
        )
    return {
        case_id: (sum(case_outcomes), len(case_outcomes))
        for case_id, case_outcomes in sorted(outcomes.items())
    }


def _success_ratio(case_counts: Mapping[str, tuple[int, int]]) -> float:
    useful = sum(counts[0] for counts in case_counts.values())
    opportunities = sum(counts[1] for counts in case_counts.values())
    return useful / opportunities


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    """Return the mean and deterministic percentile-bootstrap 95% interval.

    A single cluster can be summarized descriptively but cannot support resampling uncertainty,
    so confidence-interval callers must provide at least two independent clusters.
    """

    if len(values) < 2:
        raise ValueError("bootstrap confidence intervals require at least two clusters")
    sample_count = _positive_integer(resamples, name="bootstrap_resamples")
    bootstrap_seed = _integer_seed(seed)
    rng = random.Random(bootstrap_seed)
    cluster_count = len(values)
    estimates = [
        sum(values[rng.randrange(cluster_count)] for _ in range(cluster_count)) / cluster_count
        for _ in range(sample_count)
    ]
    estimate = sum(values) / cluster_count
    return estimate, percentile(estimates, 0.025), percentile(estimates, 0.975)


def bootstrap_success_at_deadline_ci(
    records: Iterable[Mapping[str, Any]],
    *,
    deadline_ms: float,
    latency_key: str = "ttfa_ms",
    cluster_key: str = "case_id",
    resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Estimate a task-macro 95% CI by resampling ``case_id`` clusters.

    Repetitions are averaged within each case before resampling, so repeated timings improve the
    estimate of that case's success probability without weighting tasks that happen to have more
    repetitions. This task-macro estimand equals opportunity-level Success@B when every case has
    the same number of measured repetitions.
    """

    case_counts = _case_success_counts(
        records,
        deadline_ms=deadline_ms,
        latency_key=latency_key,
        cluster_key=cluster_key,
    )
    case_rates = [
        useful / opportunities for useful, opportunities in case_counts.values()
    ]
    estimate, lower, upper = _bootstrap_mean_ci(
        case_rates,
        resamples=resamples,
        seed=seed,
    )
    return {
        "estimate": estimate,
        "lower": lower,
        "upper": upper,
        "confidence_level": 0.95,
        "method": "percentile_cluster_bootstrap",
        "estimand": "mean_case_success_at_deadline",
        "opportunity_estimate": _success_ratio(case_counts),
        "cluster_key": cluster_key,
        "cluster_count": len(case_counts),
        "resamples": _positive_integer(resamples, name="bootstrap_resamples"),
        "seed": _integer_seed(seed),
    }


def paired_success_at_deadline_delta_ci(
    baseline_records: Iterable[Mapping[str, Any]],
    candidate_records: Iterable[Mapping[str, Any]],
    *,
    deadline_ms: float,
    latency_key: str = "ttfa_ms",
    cluster_key: str = "case_id",
    resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Return a paired task-bootstrap CI for candidate-minus-baseline Success@B.

    Systems must contain the same task IDs and the same number of opportunities for each task.
    Whole task clusters are drawn once per resample and applied to both systems.
    """

    baseline_counts = _case_success_counts(
        baseline_records,
        deadline_ms=deadline_ms,
        latency_key=latency_key,
        cluster_key=cluster_key,
    )
    candidate_counts = _case_success_counts(
        candidate_records,
        deadline_ms=deadline_ms,
        latency_key=latency_key,
        cluster_key=cluster_key,
    )
    baseline_cases = set(baseline_counts)
    candidate_cases = set(candidate_counts)
    if baseline_cases != candidate_cases:
        missing_candidate = sorted(baseline_cases - candidate_cases)
        missing_baseline = sorted(candidate_cases - baseline_cases)
        raise ValueError(
            "paired bootstrap requires identical case IDs; "
            f"missing from candidate={missing_candidate}, "
            f"missing from baseline={missing_baseline}"
        )

    case_ids = sorted(baseline_cases)
    unequal_opportunities = [
        case_id
        for case_id in case_ids
        if baseline_counts[case_id][1] != candidate_counts[case_id][1]
    ]
    if unequal_opportunities:
        raise ValueError(
            "paired bootstrap requires the same number of opportunities per case; "
            f"mismatched case IDs={unequal_opportunities}"
        )

    baseline_rates = {
        case_id: useful / opportunities
        for case_id, (useful, opportunities) in baseline_counts.items()
    }
    candidate_rates = {
        case_id: useful / opportunities
        for case_id, (useful, opportunities) in candidate_counts.items()
    }
    differences = [
        candidate_rates[case_id] - baseline_rates[case_id] for case_id in case_ids
    ]
    estimate, lower, upper = _bootstrap_mean_ci(
        differences,
        resamples=resamples,
        seed=seed,
    )
    baseline_estimate = sum(baseline_rates.values()) / len(baseline_rates)
    candidate_estimate = sum(candidate_rates.values()) / len(candidate_rates)
    baseline_opportunity_estimate = _success_ratio(baseline_counts)
    candidate_opportunity_estimate = _success_ratio(candidate_counts)
    return {
        "estimate": estimate,
        "lower": lower,
        "upper": upper,
        "baseline_estimate": baseline_estimate,
        "candidate_estimate": candidate_estimate,
        "direction": "candidate_minus_baseline",
        "confidence_level": 0.95,
        "method": "paired_percentile_cluster_bootstrap",
        "estimand": "mean_paired_case_success_delta",
        "baseline_opportunity_estimate": baseline_opportunity_estimate,
        "candidate_opportunity_estimate": candidate_opportunity_estimate,
        "opportunity_delta": (
            candidate_opportunity_estimate - baseline_opportunity_estimate
        ),
        "cluster_key": cluster_key,
        "cluster_count": len(case_ids),
        "opportunity_count_per_system": sum(
            counts[1] for counts in baseline_counts.values()
        ),
        "resamples": _positive_integer(resamples, name="bootstrap_resamples"),
        "seed": _integer_seed(seed),
    }


def _paired_case_observations(
    records: Iterable[Mapping[str, Any]],
    *,
    case_key: str,
    cluster_key: str,
    repetition_key: str,
    success_key: str,
    label: str,
) -> tuple[dict[str, tuple[str, dict[int, bool]]], bool]:
    """Validate raw paired-quality rows and group timing repeats within cases."""

    rows = list(records)
    if not rows:
        raise ValueError(f"{label} requires at least one record")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"{label}[{index}] must be a mapping")
    repetition_presence = [repetition_key in row for row in rows]
    if any(repetition_presence) and not all(repetition_presence):
        raise ValueError(
            f"{label} {repetition_key!r} must be present in every row or omitted from every row"
        )
    has_repetitions = all(repetition_presence)
    cases: dict[str, tuple[str, dict[int, bool]]] = {}
    for index, row in enumerate(rows):
        case_id = row.get(case_key)
        cluster_id = row.get(cluster_key)
        success = row.get(success_key)
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{label}[{index}][{case_key!r}] must be a non-empty string")
        if not isinstance(cluster_id, str) or not cluster_id:
            raise ValueError(
                f"{label}[{index}][{cluster_key!r}] must be a non-empty string"
            )
        if not isinstance(success, bool):
            raise ValueError(f"{label}[{index}][{success_key!r}] must be a JSON boolean")
        if has_repetitions:
            repetition = row[repetition_key]
            if (
                isinstance(repetition, bool)
                or not isinstance(repetition, Integral)
                or repetition < 0
            ):
                raise ValueError(
                    f"{label}[{index}][{repetition_key!r}] must be a non-negative integer"
                )
            repetition_id = int(repetition)
        else:
            repetition_id = 0

        existing = cases.get(case_id)
        if existing is None:
            observations: dict[int, bool] = {}
            cases[case_id] = (cluster_id, observations)
        else:
            existing_cluster, observations = existing
            if existing_cluster != cluster_id:
                raise ValueError(
                    f"{label} case {case_id!r} appears in multiple task clusters"
                )
        if repetition_id in observations:
            raise ValueError(
                f"{label} contains duplicate ({case_key}, {repetition_key}) "
                f"observation ({case_id!r}, {repetition_id})"
            )
        observations[repetition_id] = success
    return cases, has_repetitions


def paired_clustered_exact_action_delta_ci(
    baseline_records: Iterable[Mapping[str, Any]],
    candidate_records: Iterable[Mapping[str, Any]],
    *,
    case_key: str = "case_id",
    cluster_key: str = "task_cluster_id",
    repetition_key: str = "repetition",
    success_key: str = "success",
    resamples: int = 10_000,
    seed: int = 0,
    noninferiority_margin: float = -0.02,
) -> dict[str, Any]:
    """Return a paired task-cluster bootstrap CI for exact-action accuracy.

    Repeated timing observations are first averaged within each case.  Candidate and baseline
    must contain the identical case/repetition opportunities and identical task-cluster mapping.
    Whole task clusters are then sampled with replacement, retaining every case in a sampled
    cluster.  The point estimand is the mean case-level candidate-minus-baseline exact-action
    difference; clusters affect uncertainty, not the point weighting of cases.

    ``passes_noninferiority`` is strict: the percentile interval's lower bound must be greater
    than ``noninferiority_margin``.  This function evaluates only exact action correctness; the
    separately prespecified latency gate must also pass before promotion.
    """

    baseline, baseline_has_repetitions = _paired_case_observations(
        baseline_records,
        case_key=case_key,
        cluster_key=cluster_key,
        repetition_key=repetition_key,
        success_key=success_key,
        label="baseline_records",
    )
    candidate, candidate_has_repetitions = _paired_case_observations(
        candidate_records,
        case_key=case_key,
        cluster_key=cluster_key,
        repetition_key=repetition_key,
        success_key=success_key,
        label="candidate_records",
    )
    if baseline_has_repetitions != candidate_has_repetitions:
        raise ValueError(
            "paired comparison requires repetition IDs in both systems or neither system"
        )
    baseline_cases = set(baseline)
    candidate_cases = set(candidate)
    if baseline_cases != candidate_cases:
        raise ValueError(
            "paired comparison requires identical case IDs; "
            f"missing from candidate={sorted(baseline_cases - candidate_cases)}, "
            f"missing from baseline={sorted(candidate_cases - baseline_cases)}"
        )

    case_ids = sorted(baseline_cases)
    cluster_cases: dict[str, list[str]] = defaultdict(list)
    baseline_rates: dict[str, float] = {}
    candidate_rates: dict[str, float] = {}
    for case_id in case_ids:
        baseline_cluster, baseline_observations = baseline[case_id]
        candidate_cluster, candidate_observations = candidate[case_id]
        if baseline_cluster != candidate_cluster:
            raise ValueError(
                f"paired case {case_id!r} has different task clusters between systems"
            )
        if set(baseline_observations) != set(candidate_observations):
            raise ValueError(
                f"paired case {case_id!r} has different repetition IDs between systems"
            )
        baseline_rates[case_id] = sum(baseline_observations.values()) / len(
            baseline_observations
        )
        candidate_rates[case_id] = sum(candidate_observations.values()) / len(
            candidate_observations
        )
        cluster_cases[baseline_cluster].append(case_id)

    sample_count = _positive_integer(resamples, name="bootstrap_resamples")
    bootstrap_seed = _integer_seed(seed)
    margin = _finite_float(noninferiority_margin, name="noninferiority_margin")
    if not -1.0 <= margin <= 1.0:
        raise ValueError("noninferiority_margin must be in [-1, 1]")

    differences = {
        case_id: candidate_rates[case_id] - baseline_rates[case_id]
        for case_id in case_ids
    }
    estimate = sum(differences.values()) / len(differences)
    cluster_ids = sorted(cluster_cases)
    if len(cluster_ids) < 2:
        raise ValueError(
            "paired task-cluster bootstrap requires at least two task clusters"
        )
    rng = random.Random(bootstrap_seed)
    estimates: list[float] = []
    for _ in range(sample_count):
        sampled_clusters = [
            cluster_ids[rng.randrange(len(cluster_ids))] for _ in cluster_ids
        ]
        sampled_differences = [
            differences[case_id]
            for cluster_id in sampled_clusters
            for case_id in cluster_cases[cluster_id]
        ]
        estimates.append(sum(sampled_differences) / len(sampled_differences))
    lower = percentile(estimates, 0.025)
    upper = percentile(estimates, 0.975)
    baseline_estimate = sum(baseline_rates.values()) / len(baseline_rates)
    candidate_estimate = sum(candidate_rates.values()) / len(candidate_rates)
    observations_per_system = sum(
        len(observations) for _, observations in baseline.values()
    )
    cluster_sizes = [len(cluster_cases[cluster_id]) for cluster_id in cluster_ids]
    return {
        "estimate": estimate,
        "lower": lower,
        "upper": upper,
        "baseline_estimate": baseline_estimate,
        "candidate_estimate": candidate_estimate,
        "direction": "candidate_minus_baseline",
        "confidence_level": 0.95,
        "method": "paired_percentile_task_cluster_bootstrap",
        "estimand": "mean_paired_case_exact_action_delta",
        "case_key": case_key,
        "cluster_key": cluster_key,
        "repetition_key": repetition_key if baseline_has_repetitions else None,
        "success_key": success_key,
        "case_count": len(case_ids),
        "cluster_count": len(cluster_ids),
        "min_cases_per_cluster": min(cluster_sizes),
        "max_cases_per_cluster": max(cluster_sizes),
        "observations_per_system": observations_per_system,
        "resamples": sample_count,
        "seed": bootstrap_seed,
        "noninferiority_margin": margin,
        "passes_noninferiority": lower > margin,
        "strict_decision_rule": "ci95_lower > noninferiority_margin",
        "latency_gate_included": False,
    }


def summarize_action_records(
    records: Iterable[Mapping[str, Any]],
    *,
    latency_key: str = "ttfa_ms",
    deadlines_ms: Sequence[float] = DEFAULT_DEADLINES_MS,
    bootstrap_resamples: int = 0,
    bootstrap_seed: int = 0,
    bootstrap_cluster_key: str = "case_id",
) -> dict[str, Any]:
    """Aggregate action-level benchmark records.

    Every record must contain ``latency_key``. Optional ``success`` and ``schema_valid`` columns
    must each be present in every record or omitted from every record. ``Success@deadline`` and
    useful goodput are emitted only when both quality columns exist, and require exact action,
    independent schema validity, and deadline attainment. Set ``bootstrap_resamples`` to add
    task-macro percentile intervals that resample whole ``bootstrap_cluster_key`` cases. The
    opportunity-level score and task-macro score are labeled separately because unequal
    repetitions can make them differ.
    """

    rows = list(records)
    if not rows:
        raise ValueError("at least one action record is required")

    latencies: list[float] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"record {index} must be a mapping")
        if latency_key not in row:
            raise ValueError(
                f"record {index} missing required latency field {latency_key!r}"
            )
        latencies.append(
            _finite_float(
                row[latency_key],
                name=f"records[{index}][{latency_key!r}]",
                non_negative=True,
            )
        )
    deadlines = _validated_deadlines(deadlines_ms)
    resample_count = _non_negative_integer(
        bootstrap_resamples, name="bootstrap_resamples"
    )
    seed = _integer_seed(bootstrap_seed)
    successes = _optional_boolean_values(rows, "success")
    schema_validity = _optional_boolean_values(rows, "schema_valid")
    if resample_count > 0 and (successes is None or schema_validity is None):
        raise ValueError(
            "cluster bootstrap requires complete 'success' and 'schema_valid' columns"
        )
    total_ms = sum(latencies)
    if not math.isfinite(total_ms):
        raise ValueError("sum of action latencies must be finite")

    summary: dict[str, Any] = {
        "sample_count": len(rows),
        "latency_key": latency_key,
        "latency_ms": latency_summary(latencies),
        "total_measured_ms": total_ms,
    }
    if successes is not None:
        summary["exact_action_accuracy"] = sum(successes) / len(successes)
    if schema_validity is not None:
        summary["schema_valid_rate"] = sum(schema_validity) / len(schema_validity)

    total_minutes = total_ms / 60_000.0
    attainment: dict[str, dict[str, float | int | None]] = {}
    for key, deadline in deadlines:
        on_time_indices = [
            index for index, latency in enumerate(latencies) if latency <= deadline
        ]
        deadline_summary: dict[str, Any] = {
            "deadline_ms": deadline,
            "opportunities": len(rows),
            "on_time": len(on_time_indices),
            "on_time_rate": len(on_time_indices) / len(rows),
        }
        if successes is not None and schema_validity is not None:
            useful = sum(
                successes[index] and schema_validity[index] for index in on_time_indices
            )
            success_at_deadline = useful / len(rows)
            goodput = useful / total_minutes if total_minutes > 0 else None
            if goodput is not None and not math.isfinite(goodput):
                goodput = None
            deadline_summary.update(
                {
                    "useful": useful,
                    "useful_rate": success_at_deadline,
                    "success_at_deadline": success_at_deadline,
                    # A zero measured duration has no finite throughput. ``None`` keeps exported
                    # summaries valid RFC 8259 JSON (rather than Python's non-standard Infinity).
                    "useful_actions_per_minute": goodput,
                }
            )
            if resample_count > 0:
                case_macro_ci = bootstrap_success_at_deadline_ci(
                    rows,
                    deadline_ms=deadline,
                    latency_key=latency_key,
                    cluster_key=bootstrap_cluster_key,
                    resamples=resample_count,
                    seed=seed,
                )
                # Keep the task-macro bootstrap visibly separate from strict opportunity-level
                # Success@B. They are equal for the balanced-repetition paper design but can
                # legitimately differ for an incomplete or otherwise unbalanced exploratory run.
                deadline_summary["case_macro_success_at_deadline"] = case_macro_ci[
                    "estimate"
                ]
                deadline_summary["case_macro_success_at_deadline_ci95"] = case_macro_ci
        attainment[key] = deadline_summary
    summary["deadline_attainment_ms"] = attainment
    return summary


def summarize_grouped_action_records(
    records: Iterable[Mapping[str, Any]],
    *,
    group_key: str = "family",
    latency_key: str = "ttfa_ms",
    deadlines_ms: Sequence[float] = DEFAULT_DEADLINES_MS,
    bootstrap_resamples: int = 0,
    bootstrap_seed: int = 0,
    bootstrap_cluster_key: str = "case_id",
) -> dict[str, dict[str, Any]]:
    """Summarize every action family without hiding a weak group in the aggregate."""

    rows = list(records)
    if not rows:
        raise ValueError("at least one action record is required")
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for index, row in enumerate(rows):
        if group_key not in row:
            raise ValueError(f"record {index} missing required group field {group_key!r}")
        group = row[group_key]
        if not isinstance(group, str) or not group:
            raise ValueError(f"records[{index}][{group_key!r}] must be a non-empty string")
        groups.setdefault(group, []).append(row)
    return {
        group: summarize_action_records(
            group_rows,
            latency_key=latency_key,
            deadlines_ms=deadlines_ms,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
            bootstrap_cluster_key=bootstrap_cluster_key,
        )
        for group, group_rows in sorted(groups.items())
    }
