"""Confirmatory aggregation of three matched pretraining architecture seeds.

The unit of replication is the training seed. Document bootstraps remain attached to each seed,
but are never pooled or presented as architecture-seed uncertainty.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from localagent.eval.pretrain_compare import (
    COMPARISON_KIND,
    COMPARISON_SCHEMA_VERSION,
)
from localagent.model import ModelConfig
from localagent.train.stage_data import canonical_sha256

SEED_AGGREGATE_KIND = "localagent_pretrain_seed_aggregate"
SEED_AGGREGATE_SCHEMA_VERSION = 1
PRIMARY_METRIC = "bits_per_byte"
METRICS = (
    "cross_entropy_nats_per_token",
    "bits_per_byte",
    "top1_accuracy",
)
_LOWER_IS_BETTER = frozenset(
    {"cross_entropy_nats_per_token", "bits_per_byte"}
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_T_CRITICAL_95_DF2 = 4.302652729911275


@dataclass(frozen=True)
class SeedComparisonSpec:
    """One matched architecture pair and its precomputed document comparison."""

    seed: int
    attention_config: str | Path
    hybrid_config: str | Path
    comparison: str | Path


@dataclass(frozen=True)
class _LoadedSeed:
    seed: int
    attention_config: dict[str, Any]
    hybrid_config: dict[str, Any]
    comparison: dict[str, Any]
    input_record: dict[str, Any]


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: str | Path, *, label: str) -> tuple[Path, dict[str, int | str]]:
    source = Path(path)
    if source.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link: {source}")
    if not source.is_file():
        raise ValueError(f"{label} is missing or is not a file: {source}")
    return source, {
        "path": str(source),
        "bytes": source.stat().st_size,
        "sha256": _sha256_file(source),
    }


def _load_yaml(path: str | Path, *, label: str) -> tuple[dict[str, Any], dict[str, int | str]]:
    source, artifact = _artifact(path, label=label)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"{label} is invalid YAML: {source}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a YAML mapping: {source}")
    return payload, artifact


def _load_json(path: str | Path, *, label: str) -> tuple[dict[str, Any], dict[str, int | str]]:
    source, artifact = _artifact(path, label=label)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON: {source}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {source}")
    return payload, artifact


def _lineage_config_sha256(config: dict[str, Any]) -> str:
    normalized = copy.deepcopy(config)
    runtime = normalized.get("runtime")
    if isinstance(runtime, dict):
        runtime.pop("resume", None)
    return canonical_sha256(normalized)


def _project_path(config_path: Path, declared_path: object, *, label: str) -> Path:
    if not isinstance(declared_path, str) or not declared_path:
        raise ValueError(f"{label} is missing")
    declared = Path(declared_path)
    if declared.is_absolute():
        return declared
    candidates = [
        Path.cwd() / declared,
        config_path.resolve().parents[2] / declared,
    ]
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


def _model_config_sha256(config_path: Path, config: dict[str, Any]) -> str:
    model_path = _project_path(
        config_path,
        config.get("model_config"),
        label=f"{config_path} model_config",
    )
    model_config = ModelConfig.from_yaml(str(model_path))
    model_config.assert_within_budget()
    identity = dict(model_config.__dict__)
    # The tracked 2027–2029 comparison predates the opt-in sparse-FFN fields. Preserve the exact
    # identity of a legacy YAML that did not declare them; newly authored configs (including the
    # sparse candidate) bind the complete current schema.
    raw_model = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    sparse_fields = {
        "ffn_num_experts",
        "ffn_top_k",
        "router_aux_loss_coef",
    }
    vision_fields = {
        "vision_enabled",
        "vision_image_size",
        "vision_patch_size",
        "vision_width",
    }
    if isinstance(raw_model, dict) and sparse_fields.isdisjoint(raw_model):
        for field in sparse_fields:
            identity.pop(field)
    # Archived pretraining comparisons predate the opt-in screenshot bridge. Keep their model
    # identities stable when the legacy YAML does not declare any vision fields; new visual configs
    # bind the complete current schema.
    if isinstance(raw_model, dict) and vision_fields.isdisjoint(raw_model):
        for field in vision_fields:
            identity.pop(field)
    return canonical_sha256(identity)


def _without_pair_identity(config: dict[str, Any]) -> dict[str, Any]:
    comparable = copy.deepcopy(config)
    comparable.pop("model_config", None)
    log = comparable.get("log")
    if isinstance(log, dict):
        log.pop("out_dir", None)
    return comparable


def _without_seed_identity(config: dict[str, Any]) -> dict[str, Any]:
    comparable = copy.deepcopy(config)
    runtime = comparable.get("runtime")
    if isinstance(runtime, dict):
        runtime.pop("seed", None)
    log = comparable.get("log")
    if isinstance(log, dict):
        log.pop("out_dir", None)
    return comparable


def _validate_training_config(
    path: Path,
    config: dict[str, Any],
    *,
    seed: int,
    architecture: str,
    comparison_input: object,
) -> dict[str, Any]:
    if config.get("stage") != "pretrain":
        raise ValueError(f"{path} is not a pretrain configuration")
    runtime = config.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("seed") != seed:
        raise ValueError(f"{path} does not record training seed {seed}")
    if not isinstance(comparison_input, dict):
        raise ValueError(f"seed {seed} comparison has no {architecture} input")
    if comparison_input.get("training_seed") != seed:
        raise ValueError(
            f"seed {seed} comparison {architecture} training seed does not match its config"
        )
    expected_config_sha256 = _lineage_config_sha256(config)
    if comparison_input.get("pretrain_config_sha256") != expected_config_sha256:
        raise ValueError(
            f"seed {seed} comparison {architecture} pretrain config hash mismatch"
        )
    expected_model_sha256 = _model_config_sha256(path, config)
    if comparison_input.get("model_config_sha256") != expected_model_sha256:
        raise ValueError(
            f"seed {seed} comparison {architecture} model config hash mismatch"
        )
    checkpoint_sha256 = comparison_input.get("checkpoint_sha256")
    if not _valid_sha256(checkpoint_sha256):
        raise ValueError(f"seed {seed} comparison {architecture} checkpoint hash is invalid")
    accounting = comparison_input.get("token_accounting")
    if not isinstance(accounting, dict):
        raise ValueError(f"seed {seed} comparison {architecture} lacks token accounting")
    input_tokens = _positive_integer(
        accounting.get("input_tokens"),
        label=f"seed {seed} {architecture} input_tokens",
    )
    loss_tokens = _positive_integer(
        accounting.get("loss_tokens"),
        label=f"seed {seed} {architecture} loss_tokens",
    )
    if loss_tokens > input_tokens:
        raise ValueError(f"seed {seed} {architecture} loss_tokens exceed input_tokens")
    return {
        "config_sha256": expected_config_sha256,
        "model_config_sha256": expected_model_sha256,
        "checkpoint_sha256": str(checkpoint_sha256),
        "token_accounting": {
            "input_tokens": input_tokens,
            "loss_tokens": loss_tokens,
            "source": accounting.get("source"),
        },
    }


def _validate_arm_summary(summary: object, *, label: str) -> dict[str, int | float]:
    if not isinstance(summary, dict):
        raise ValueError(f"{label} must be an object")
    documents = _positive_integer(summary.get("documents"), label=f"{label} documents")
    tokens = _positive_integer(summary.get("tokens"), label=f"{label} tokens")
    utf8_bytes = _positive_integer(summary.get("utf8_bytes"), label=f"{label} utf8_bytes")
    correct_tokens = summary.get("correct_tokens")
    if (
        isinstance(correct_tokens, bool)
        or not isinstance(correct_tokens, int)
        or not 0 <= correct_tokens <= tokens
    ):
        raise ValueError(f"{label} correct_tokens is invalid")
    nll_nats = _finite_number(summary.get("nll_nats"), label=f"{label} nll_nats")
    if nll_nats < 0.0:
        raise ValueError(f"{label} nll_nats is negative")
    expected = {
        "cross_entropy_nats_per_token": nll_nats / tokens,
        "bits_per_byte": nll_nats / (math.log(2.0) * utf8_bytes),
        "top1_accuracy": correct_tokens / tokens,
    }
    for metric, value in expected.items():
        recorded = _finite_number(summary.get(metric), label=f"{label} {metric}")
        if not math.isclose(recorded, value, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"{label} {metric} is inconsistent with its counts")
    return {
        "documents": documents,
        "tokens": tokens,
        "utf8_bytes": utf8_bytes,
        "correct_tokens": correct_tokens,
        "nll_nats": nll_nats,
        **expected,
    }


def _validate_subset(subset: object, *, label: str) -> dict[str, Any]:
    if not isinstance(subset, dict):
        raise ValueError(f"{label} must be an object")
    attention = _validate_arm_summary(subset.get("attention"), label=f"{label} attention")
    hybrid = _validate_arm_summary(subset.get("hybrid"), label=f"{label} hybrid")
    for count in ("documents", "tokens", "utf8_bytes"):
        expected = _positive_integer(subset.get(count), label=f"{label} {count}")
        if attention[count] != expected or hybrid[count] != expected:
            raise ValueError(f"{label} {count} disagrees between arms")
    difference = subset.get("difference_attention_minus_hybrid")
    if not isinstance(difference, dict) or set(difference) != set(METRICS):
        raise ValueError(f"{label} has invalid metric differences")
    validated_difference = {}
    for metric in METRICS:
        metric_result = difference.get(metric)
        if not isinstance(metric_result, dict):
            raise ValueError(f"{label} {metric} difference must be an object")
        estimate = _finite_number(
            metric_result.get("estimate"),
            label=f"{label} {metric} estimate",
        )
        expected_estimate = float(attention[metric]) - float(hybrid[metric])
        if not math.isclose(estimate, expected_estimate, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"{label} {metric} estimate is not attention minus hybrid")
        interval = metric_result.get("percentile_ci")
        if not isinstance(interval, dict):
            raise ValueError(f"{label} {metric} has no document-bootstrap interval")
        confidence = _finite_number(
            interval.get("confidence"),
            label=f"{label} {metric} confidence",
        )
        lower = _finite_number(interval.get("lower"), label=f"{label} {metric} lower")
        upper = _finite_number(interval.get("upper"), label=f"{label} {metric} upper")
        if not 0.0 < confidence < 1.0 or lower > upper:
            raise ValueError(f"{label} {metric} document-bootstrap interval is invalid")
        fractions = {
            name: _finite_number(
                metric_result.get(name),
                label=f"{label} {metric} {name}",
            )
            for name in (
                "attention_win_fraction",
                "hybrid_win_fraction",
                "tie_fraction",
            )
        }
        if any(not 0.0 <= value <= 1.0 for value in fractions.values()) or not math.isclose(
            sum(fractions.values()),
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(f"{label} {metric} bootstrap win fractions are invalid")
        validated_difference[metric] = {
            "estimate": estimate,
            "document_bootstrap_percentile_ci": {
                "confidence": confidence,
                "lower": lower,
                "upper": upper,
            },
        }
    return {
        "documents": int(attention["documents"]),
        "tokens": int(attention["tokens"]),
        "utf8_bytes": int(attention["utf8_bytes"]),
        "attention": attention,
        "hybrid": hybrid,
        "difference": validated_difference,
    }


def _validate_comparison(
    report: dict[str, Any],
    *,
    seed: int,
    attention_path: Path,
    attention_config: dict[str, Any],
    hybrid_path: Path,
    hybrid_config: dict[str, Any],
) -> dict[str, Any]:
    if report.get("kind") != COMPARISON_KIND:
        raise ValueError(f"seed {seed} comparison kind is unsupported")
    if report.get("schema_version") != COMPARISON_SCHEMA_VERSION:
        raise ValueError(f"seed {seed} comparison schema version is unsupported")
    bootstrap = report.get("bootstrap")
    inputs = report.get("inputs")
    if not isinstance(bootstrap, dict) or not isinstance(inputs, dict):
        raise ValueError(f"seed {seed} comparison is incomplete")
    bootstrap_seed = bootstrap.get("seed")
    if (
        isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or not 0 <= bootstrap_seed < 2**63
    ):
        raise ValueError(f"seed {seed} comparison bootstrap seed is invalid")
    resamples = _positive_integer(
        bootstrap.get("resamples"),
        label=f"seed {seed} bootstrap resamples",
    )
    if resamples < 10_000:
        raise ValueError(f"seed {seed} comparison uses fewer than 10000 bootstrap resamples")
    confidence = _finite_number(
        bootstrap.get("confidence"),
        label=f"seed {seed} bootstrap confidence",
    )
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"seed {seed} bootstrap confidence is invalid")
    attention_input = inputs.get("attention")
    hybrid_input = inputs.get("hybrid")
    attention = _validate_training_config(
        attention_path,
        attention_config,
        seed=seed,
        architecture="attention",
        comparison_input=attention_input,
    )
    hybrid = _validate_training_config(
        hybrid_path,
        hybrid_config,
        seed=seed,
        architecture="hybrid",
        comparison_input=hybrid_input,
    )
    if attention["token_accounting"] != hybrid["token_accounting"]:
        raise ValueError(f"seed {seed} paired arms have different token accounting")
    if attention["checkpoint_sha256"] == hybrid["checkpoint_sha256"]:
        raise ValueError(f"seed {seed} paired arms bind the same checkpoint")
    expected_comparison_sha256 = canonical_sha256(
        {
            "kind": COMPARISON_KIND,
            "schema_version": COMPARISON_SCHEMA_VERSION,
            "method": "paired_nonparametric_document_bootstrap",
            "attention_sidecar_sha256": attention_input["sidecar"]["sha256"],
            "hybrid_sidecar_sha256": hybrid_input["sidecar"]["sha256"],
            "seed": bootstrap_seed,
            "resamples": resamples,
            "confidence": confidence,
        }
    )
    if report.get("comparison_sha256") != expected_comparison_sha256:
        raise ValueError(f"seed {seed} comparison identity is invalid")
    bindings = report.get("matched_bindings")
    evaluation = report.get("evaluation")
    groups = report.get("groups")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError(f"seed {seed} comparison has no matched dataset bindings")
    if any(not _valid_sha256(value) for value in bindings.values()):
        raise ValueError(f"seed {seed} comparison has invalid matched dataset bindings")
    if not isinstance(evaluation, dict) or not evaluation:
        raise ValueError(f"seed {seed} comparison has no evaluation protocol")
    if not isinstance(groups, dict) or not groups:
        raise ValueError(f"seed {seed} comparison has no source groups")
    return {
        "bootstrap": {
            "unit": bootstrap.get("unit"),
            "seed": bootstrap_seed,
            "resamples": resamples,
            "confidence": confidence,
        },
        "attention": attention,
        "hybrid": hybrid,
        "matched_bindings": copy.deepcopy(bindings),
        "evaluation": copy.deepcopy(evaluation),
        "overall": _validate_subset(report.get("overall"), label=f"seed {seed} overall"),
        "groups": {
            name: _validate_subset(subset, label=f"seed {seed} group {name}")
            for name, subset in sorted(groups.items())
        },
    }


def _descriptive(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sample_standard_deviation": statistics.stdev(values),
        "standard_error": statistics.stdev(values) / math.sqrt(len(values)),
        "minimum": min(values),
        "maximum": max(values),
    }


def _sign_test(attention_wins: int, hybrid_wins: int) -> dict[str, int | float]:
    non_ties = attention_wins + hybrid_wins
    if non_ties == 0:
        return {
            "non_ties": 0,
            "majority_wins": 0,
            "majority_one_sided_p_value": 1.0,
            "two_sided_p_value": 1.0,
        }
    majority_wins = max(attention_wins, hybrid_wins)
    tail = sum(
        math.comb(non_ties, count)
        for count in range(majority_wins, non_ties + 1)
    ) / 2**non_ties
    return {
        "non_ties": non_ties,
        "majority_wins": majority_wins,
        "majority_one_sided_p_value": tail,
        "two_sided_p_value": min(1.0, 2.0 * tail),
    }


def _aggregate_metric(
    loaded: list[_LoadedSeed],
    *,
    subset_name: str,
    metric: str,
) -> dict[str, Any]:
    seed_estimates = []
    attention_values = []
    hybrid_values = []
    differences = []
    attention_wins = 0
    hybrid_wins = 0
    ties = 0
    for run in loaded:
        subset = (
            run.comparison["overall"]
            if subset_name == "overall"
            else run.comparison["groups"][subset_name]
        )
        attention = float(subset["attention"][metric])
        hybrid = float(subset["hybrid"][metric])
        difference = float(subset["difference"][metric]["estimate"])
        attention_values.append(attention)
        hybrid_values.append(hybrid)
        differences.append(difference)
        if difference == 0.0:
            ties += 1
        elif (metric in _LOWER_IS_BETTER and difference < 0.0) or (
            metric not in _LOWER_IS_BETTER and difference > 0.0
        ):
            attention_wins += 1
        else:
            hybrid_wins += 1
        seed_estimates.append(
            {
                "training_seed": run.seed,
                "attention": attention,
                "hybrid": hybrid,
                "estimate": difference,
                "document_bootstrap_percentile_ci": subset["difference"][metric][
                    "document_bootstrap_percentile_ci"
                ],
            }
        )
    descriptive = _descriptive(differences)
    margin = (
        _T_CRITICAL_95_DF2
        * descriptive["sample_standard_deviation"]
        / math.sqrt(len(differences))
    )
    return {
        "orientation": (
            "lower_is_better" if metric in _LOWER_IS_BETTER else "higher_is_better"
        ),
        "difference": "attention_minus_hybrid",
        "seed_estimates": seed_estimates,
        "attention": _descriptive(attention_values),
        "hybrid": _descriptive(hybrid_values),
        "difference_attention_minus_hybrid": {
            **descriptive,
            "student_t_95_interval": {
                "confidence": 0.95,
                "degrees_of_freedom": 2,
                "critical_value": _T_CRITICAL_95_DF2,
                "lower": descriptive["mean"] - margin,
                "upper": descriptive["mean"] + margin,
                "assumption": (
                    "model-based interval assuming approximately normal seed effects; "
                    "normality cannot be assessed with three seeds"
                ),
            },
            "attention_favoring_seeds": attention_wins,
            "hybrid_favoring_seeds": hybrid_wins,
            "ties": ties,
            "exact_sign_test": _sign_test(attention_wins, hybrid_wins),
        },
    }


def _aggregate_subset(
    loaded: list[_LoadedSeed],
    *,
    subset_name: str,
) -> dict[str, Any]:
    first = (
        loaded[0].comparison["overall"]
        if subset_name == "overall"
        else loaded[0].comparison["groups"][subset_name]
    )
    return {
        "documents_per_seed": first["documents"],
        "tokens_per_seed": first["tokens"],
        "utf8_bytes_per_seed": first["utf8_bytes"],
        "metrics": {
            metric: _aggregate_metric(
                loaded,
                subset_name=subset_name,
                metric=metric,
            )
            for metric in METRICS
        },
    }


def aggregate_pretrain_seeds(
    specifications: list[SeedComparisonSpec],
) -> dict[str, Any]:
    """Validate and aggregate an exactly-three-seed matched architecture experiment."""

    if len(specifications) != 3:
        raise ValueError("confirmatory pretrain aggregation requires exactly three seeds")
    seeds = [specification.seed for specification in specifications]
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63
        for seed in seeds
    ):
        raise ValueError("training seeds must be integers in [0, 2**63)")
    if len(set(seeds)) != len(seeds):
        raise ValueError("training seeds must be unique")

    loaded: list[_LoadedSeed] = []
    for specification in sorted(specifications, key=lambda item: item.seed):
        attention_path, attention_artifact = _artifact(
            specification.attention_config,
            label=f"seed {specification.seed} attention config",
        )
        hybrid_path, hybrid_artifact = _artifact(
            specification.hybrid_config,
            label=f"seed {specification.seed} hybrid config",
        )
        attention_config, _ = _load_yaml(
            attention_path,
            label=f"seed {specification.seed} attention config",
        )
        hybrid_config, _ = _load_yaml(
            hybrid_path,
            label=f"seed {specification.seed} hybrid config",
        )
        if _without_pair_identity(attention_config) != _without_pair_identity(
            hybrid_config
        ):
            raise ValueError(
                f"seed {specification.seed} paired configs differ beyond architecture/output"
            )
        comparison_report, comparison_artifact = _load_json(
            specification.comparison,
            label=f"seed {specification.seed} paired comparison",
        )
        validated = _validate_comparison(
            comparison_report,
            seed=specification.seed,
            attention_path=attention_path,
            attention_config=attention_config,
            hybrid_path=hybrid_path,
            hybrid_config=hybrid_config,
        )
        loaded.append(
            _LoadedSeed(
                seed=specification.seed,
                attention_config=attention_config,
                hybrid_config=hybrid_config,
                comparison=validated,
                input_record={
                    "training_seed": specification.seed,
                    "attention_config": {
                        **attention_artifact,
                        "lineage_sha256": validated["attention"]["config_sha256"],
                    },
                    "hybrid_config": {
                        **hybrid_artifact,
                        "lineage_sha256": validated["hybrid"]["config_sha256"],
                    },
                    "comparison": {
                        **comparison_artifact,
                        "comparison_sha256": comparison_report["comparison_sha256"],
                    },
                    "attention_checkpoint_sha256": validated["attention"][
                        "checkpoint_sha256"
                    ],
                    "hybrid_checkpoint_sha256": validated["hybrid"][
                        "checkpoint_sha256"
                    ],
                    "token_accounting": validated["attention"]["token_accounting"],
                    "document_bootstrap": validated["bootstrap"],
                },
            )
        )

    for architecture in ("attention", "hybrid"):
        configs = [
            run.attention_config if architecture == "attention" else run.hybrid_config
            for run in loaded
        ]
        reference = _without_seed_identity(configs[0])
        if any(_without_seed_identity(config) != reference for config in configs[1:]):
            raise ValueError(
                f"{architecture} configs differ across seeds beyond seed/output"
            )
    checkpoint_hashes = [
        run.input_record[f"{architecture}_checkpoint_sha256"]
        for architecture in ("attention", "hybrid")
        for run in loaded
    ]
    if len(set(checkpoint_hashes)) != len(checkpoint_hashes):
        raise ValueError("checkpoint hashes must be unique across all seed/architecture runs")

    first = loaded[0].comparison
    for run in loaded[1:]:
        if run.comparison["matched_bindings"] != first["matched_bindings"]:
            raise ValueError("seed comparisons use different dataset bindings")
        if run.comparison["evaluation"] != first["evaluation"]:
            raise ValueError("seed comparisons use different evaluation protocols")
        if set(run.comparison["groups"]) != set(first["groups"]):
            raise ValueError("seed comparisons use different source groups")
        for subset_name in ("overall", *sorted(first["groups"])):
            candidate = (
                run.comparison["overall"]
                if subset_name == "overall"
                else run.comparison["groups"][subset_name]
            )
            reference = (
                first["overall"]
                if subset_name == "overall"
                else first["groups"][subset_name]
            )
            for field in ("documents", "tokens", "utf8_bytes"):
                if candidate[field] != reference[field]:
                    raise ValueError(
                        f"seed comparisons disagree on {subset_name} {field}"
                    )

    report = {
        "kind": SEED_AGGREGATE_KIND,
        "schema_version": SEED_AGGREGATE_SCHEMA_VERSION,
        "design": {
            "unit_of_replication": "training_seed",
            "training_seeds": [run.seed for run in loaded],
            "seed_count": len(loaded),
            "difference": "attention_minus_hybrid",
            "primary_metric": f"overall.{PRIMARY_METRIC}",
            "secondary_metrics": [
                "overall.cross_entropy_nats_per_token",
                "overall.top1_accuracy",
            ],
            "exploratory_subgroups": sorted(first["groups"]),
            "document_bootstrap_role": (
                "within-seed held-out-document uncertainty only; intervals are not pooled "
                "or treated as architecture-seed uncertainty"
            ),
            "inference_caveat": (
                "three seeds permit only a fragile model-based t interval and a low-power "
                "exact sign test; report every seed estimate"
            ),
        },
        "inputs": [run.input_record for run in loaded],
        "matched_bindings": first["matched_bindings"],
        "evaluation": first["evaluation"],
        "overall": _aggregate_subset(loaded, subset_name="overall"),
        "groups": {
            name: _aggregate_subset(loaded, subset_name=name)
            for name in sorted(first["groups"])
        },
    }
    report["aggregate_sha256"] = canonical_sha256(report)
    return report
