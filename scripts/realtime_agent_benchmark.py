#!/usr/bin/env python
"""Analyze browser-agent latency or calculate autoregressive action-rate requirements.

Examples:
  python scripts/realtime_agent_benchmark.py requirements --ttft-ms 100 --parse-ms 10
  python scripts/realtime_agent_benchmark.py calibrate --ttft-ms 200 --parse-ms 20
  python scripts/realtime_agent_benchmark.py summarize runs/webgpu/action-benchmark.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Any

from localagent.eval.realtime import (
    calibrate_autoregressive_rate_scenarios,
    latency_summary,
    required_decode_tokens_per_second,
    summarize_action_records,
    summarize_grouped_action_records,
)

_OPTIONAL_BROWSER_LATENCY_KEYS = (
    "harness_ttfa_ms",
    "runtime_ttfa_ms",
    "independent_validate_ms",
    "tokenize_ms",
    "inference_ms",
    "decode_control_ms",
    "dispatch_ms",
    "parse_validate_ms",
    "ttft_ms",
    "tpot_ms",
)


def _csv_ints(raw: str) -> list[int]:
    try:
        values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from exc
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _csv_positive_floats(raw: str) -> list[float]:
    try:
        values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated positive numbers") from exc
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated finite positive numbers")
    return values


def _finite_non_negative_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a number") from exc
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("expected a finite non-negative number")
    return value


def _non_negative_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a non-negative integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return value


def _requirements(args: argparse.Namespace) -> None:
    print(
        f"Minimum iterative decode tok/s (fixed predecode={args.ttft_ms:g} ms, "
        f"fixed postdecode={args.parse_ms:g} ms)"
    )
    header = ["decode steps", *[f"{deadline} ms" for deadline in args.deadlines]]
    print(" | ".join(header))
    print(" | ".join(["---"] * len(header)))
    for output_tokens in args.output_tokens:
        cells = [str(output_tokens)]
        for deadline in args.deadlines:
            try:
                rate = required_decode_tokens_per_second(
                    output_tokens,
                    deadline_ms=deadline,
                    ttft_ms=args.ttft_ms,
                    postprocess_ms=args.parse_ms,
                )
            except ValueError as exc:
                raise SystemExit(f"invalid rate calculation: {exc}") from exc
            cells.append("impossible" if math.isinf(rate) else f"{rate:.1f}")
        print(" | ".join(cells))


def _calibrate(args: argparse.Namespace) -> None:
    """Emit a machine-readable fixed-rate counterfactual, not an empirical benchmark."""

    try:
        calibration = calibrate_autoregressive_rate_scenarios(
            args.decode_steps,
            decode_rates_tps=args.decode_rates,
            deadlines_ms=args.deadlines,
            predecode_ms=args.predecode_ms,
            postprocess_ms=args.parse_ms,
        )
    except ValueError as exc:
        raise SystemExit(f"invalid calibration scenario: {exc}") from exc
    print(json.dumps(calibration, allow_nan=False, indent=2, sort_keys=True))


def _optional_stage_summaries(records: list[dict[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for key in _OPTIONAL_BROWSER_LATENCY_KEYS:
        presence = [key in record for record in records]
        if any(presence) and not all(presence):
            raise ValueError(f"{key!r} must be present in every record or omitted from every record")
        if all(presence):
            available = [record[key] for record in records if record[key] is not None]
            summaries[key] = (
                {
                    **latency_summary(available),
                    "missing_count": len(records) - len(available),
                }
                if available
                else {"count": 0, "missing_count": len(records)}
            )
    return summaries


def _declared_latency_clock(payload: Any) -> str | None:
    """Return the export's declared primary clock, when present."""

    if not isinstance(payload, Mapping):
        return None
    metadata = payload.get("metadata")
    if metadata is None:
        return None
    if not isinstance(metadata, Mapping):
        raise ValueError("'metadata' must be a JSON object or null")
    value = metadata.get("latency_clock")
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("metadata.latency_clock must be a non-empty string or null")
    return value


def _metadata_integer(
    metadata: Mapping[str, Any],
    key: str,
    *,
    minimum: int,
) -> int | None:
    value = metadata.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"metadata.{key} must be a {qualifier} integer or null")
    return int(value)


def _validate_export_integrity(
    payload: Any,
    records: list[dict[str, Any]],
    *,
    latency_key: str,
) -> dict[str, Any] | None:
    """Validate declared browser-run cardinality so failures cannot disappear silently."""

    if not isinstance(payload, dict):
        return None
    metadata = payload.get("metadata")
    if metadata is None:
        return None
    if not isinstance(metadata, Mapping):
        raise ValueError("'metadata' must be a JSON object or null")

    concurrency = _metadata_integer(metadata, "concurrency", minimum=1)
    if concurrency is not None and concurrency != 1:
        raise ValueError(
            "RTAB is defined at concurrency one; metadata.concurrency must equal 1"
        )
    declared_cases = _metadata_integer(metadata, "cases", minimum=1)
    declared_repetitions = _metadata_integer(metadata, "repetitions", minimum=1)
    if (declared_cases is None) != (declared_repetitions is None):
        raise ValueError("metadata.cases and metadata.repetitions must be declared together")

    declared_clock = _declared_latency_clock(payload)
    integrity: dict[str, Any] = {
        "concurrency_one": True if concurrency == 1 else None,
        "declared_latency_clock": declared_clock,
        "selected_latency_key": latency_key,
        "selected_clock_matches_declaration": (
            latency_key == declared_clock if declared_clock is not None else None
        ),
        "declared_cases": declared_cases,
        "declared_repetitions": declared_repetitions,
        "declared_opportunity_count": None,
        "record_count_matches": None,
    }
    if declared_cases is None or declared_repetitions is None:
        return integrity

    expected_count = declared_cases * declared_repetitions
    if len(records) != expected_count:
        raise ValueError(
            f"metadata declares {expected_count} action opportunities but the export contains "
            f"{len(records)} records; failed and invalid trials must remain in the record set"
        )

    case_ids: list[str] = []
    for index, record in enumerate(records):
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"records[{index}]['case_id'] must be a non-empty string")
        case_ids.append(case_id)
    case_counts = Counter(case_ids)
    if len(case_counts) != declared_cases:
        raise ValueError(
            f"metadata declares {declared_cases} cases but records contain "
            f"{len(case_counts)} distinct case IDs"
        )
    unbalanced = {
        case_id: count
        for case_id, count in sorted(case_counts.items())
        if count != declared_repetitions
    }
    if unbalanced:
        raise ValueError(
            "every case must retain exactly metadata.repetitions records; "
            f"mismatched counts={unbalanced}"
        )

    repetition_presence = ["repetition" in record for record in records]
    if any(repetition_presence) and not all(repetition_presence):
        raise ValueError(
            "'repetition' must be present in every record or omitted from every record"
        )
    if all(repetition_presence):
        observed: set[tuple[str, int]] = set()
        for index, (case_id, record) in enumerate(zip(case_ids, records, strict=True)):
            repetition = record["repetition"]
            if (
                isinstance(repetition, bool)
                or not isinstance(repetition, Integral)
                or not 0 <= repetition < declared_repetitions
            ):
                raise ValueError(
                    f"records[{index}]['repetition'] must be an integer in "
                    f"[0, {declared_repetitions})"
                )
            key = (case_id, int(repetition))
            if key in observed:
                raise ValueError(f"duplicate case/repetition opportunity {key!r}")
            observed.add(key)

    integrity.update(
        {
            "declared_opportunity_count": expected_count,
            "record_count_matches": True,
        }
    )
    return integrity


def _summarize(args: argparse.Namespace) -> None:
    path = Path(args.path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get("records", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise SystemExit("expected a JSON list or an object containing a 'records' list")
    if any(not isinstance(record, dict) for record in records):
        raise SystemExit("expected every benchmark record to be a JSON object")
    try:
        declared_clock = _declared_latency_clock(payload)
        latency_key = args.latency_key or declared_clock or "ttfa_ms"
        export_integrity = _validate_export_integrity(
            payload,
            records,
            latency_key=latency_key,
        )
        summary = summarize_action_records(
            records,
            latency_key=latency_key,
            deadlines_ms=args.deadlines,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
        stage_summaries = _optional_stage_summaries(records)
        if stage_summaries:
            summary["stage_latency_ms"] = stage_summaries
        if export_integrity is not None:
            summary["export_integrity"] = export_integrity
        if args.group_key:
            group_presence = [args.group_key in record for record in records]
            if not all(group_presence):
                raise ValueError(
                    f"group field {args.group_key!r} must be present in every record; "
                    "pass an empty --group-key to disable grouped reporting"
                )
            summary[f"by_{args.group_key}"] = summarize_grouped_action_records(
                records,
                group_key=args.group_key,
                latency_key=latency_key,
                deadlines_ms=args.deadlines,
                bootstrap_resamples=args.bootstrap_resamples,
                bootstrap_seed=args.bootstrap_seed,
            )
    except ValueError as exc:
        raise SystemExit(f"invalid benchmark records: {exc}") from exc
    print(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    requirements = subparsers.add_parser(
        "requirements", help="calculate decode throughput needed by action length and deadline"
    )
    requirements.add_argument(
        "--ttft-ms",
        "--predecode-ms",
        dest="ttft_ms",
        type=_finite_non_negative_float,
        default=100.0,
        help=(
            "fixed latency before the first decoded token; use TTFT for model TTFA or "
            "tokenization+TTFT for harness TTFA"
        ),
    )
    requirements.add_argument(
        "--parse-ms",
        "--postdecode-ms",
        dest="parse_ms",
        type=_finite_non_negative_float,
        default=10.0,
        help=(
            "fixed latency after iterative decode; include independent validation for "
            "harness TTFA"
        ),
    )
    requirements.add_argument("--output-tokens", type=_csv_ints, default=[8, 16, 32, 64, 128])
    requirements.add_argument("--deadlines", type=_csv_ints, default=[250, 500, 1000, 2000])
    requirements.set_defaults(func=_requirements)

    calibrate = subparsers.add_parser(
        "calibrate",
        help=(
            "evaluate fixed 200/400/600 tok/s AR scenarios; this is a counterfactual, "
            "not a hardware or closed-loop browser benchmark"
        ),
    )
    calibrate.add_argument(
        "--ttft-ms",
        "--predecode-ms",
        dest="predecode_ms",
        type=_finite_non_negative_float,
        default=200.0,
        help=(
            "fixed latency before the first decoded token; the 200 ms default folds "
            "tokenization and TTFT together for the paper's harness-TTFA illustration"
        ),
    )
    calibrate.add_argument(
        "--parse-ms",
        "--postdecode-ms",
        dest="parse_ms",
        type=_finite_non_negative_float,
        default=20.0,
        help="fixed post-decode parse and independent-validation latency",
    )
    calibrate.add_argument(
        "--decode-steps",
        "--output-tokens",
        dest="decode_steps",
        type=_csv_ints,
        default=[16, 32, 64, 128],
        help="decode steps including every required terminal/EOS step",
    )
    calibrate.add_argument(
        "--decode-rates",
        type=_csv_positive_floats,
        default=[200.0, 400.0, 600.0],
        help="candidate constant iterative decode rates in tokens/s",
    )
    calibrate.add_argument(
        "--deadlines",
        type=_csv_ints,
        default=[250, 500, 1000, 2000],
    )
    calibrate.set_defaults(func=_calibrate)

    summarize = subparsers.add_parser(
        "summarize", help="summarize JSON exported by the WebGPU action benchmark"
    )
    summarize.add_argument("path")
    summarize.add_argument("--deadlines", type=_csv_ints, default=[100, 250, 500, 1000, 2000])
    summarize.add_argument(
        "--latency-key",
        default=None,
        help=(
            "record field to summarize; defaults to metadata.latency_clock, then legacy ttfa_ms"
        ),
    )
    summarize.add_argument(
        "--group-key",
        default="family",
        help="optional record field for per-group metrics; pass an empty string to disable",
    )
    summarize.add_argument(
        "--bootstrap-resamples",
        type=_non_negative_int,
        default=10_000,
        help="case_id cluster-bootstrap draws for Success@deadline CIs; 0 disables",
    )
    summarize.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        help="deterministic case_id cluster-bootstrap seed",
    )
    summarize.set_defaults(func=_summarize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
