from __future__ import annotations

import copy
import json
import math
import statistics
from pathlib import Path

import pytest
import yaml

from localagent.eval.pretrain_compare import (
    COMPARISON_KIND,
    COMPARISON_SCHEMA_VERSION,
)
from localagent.eval.pretrain_seed_aggregate import (
    SEED_AGGREGATE_KIND,
    SeedComparisonSpec,
    aggregate_pretrain_seeds,
)
from localagent.model import ModelConfig
from localagent.train.stage_data import canonical_sha256


def _sha(label: str) -> str:
    return canonical_sha256({"label": label})


def _config_sha(config: dict) -> str:
    normalized = copy.deepcopy(config)
    normalized["runtime"].pop("resume", None)
    return canonical_sha256(normalized)


def _arm_summary(*, nll: float, correct: int) -> dict:
    tokens = 100
    utf8_bytes = 200
    return {
        "documents": 10,
        "tokens": tokens,
        "utf8_bytes": utf8_bytes,
        "nll_nats": nll,
        "correct_tokens": correct,
        "cross_entropy_nats_per_token": nll / tokens,
        "bits_per_byte": nll / (math.log(2.0) * utf8_bytes),
        "top1_accuracy": correct / tokens,
    }


def _subset(seed_offset: int) -> dict:
    attention = _arm_summary(nll=200.0 + 10.0 * seed_offset, correct=50 - seed_offset)
    hybrid = _arm_summary(nll=200.0, correct=50)
    differences = {}
    for metric in (
        "cross_entropy_nats_per_token",
        "bits_per_byte",
        "top1_accuracy",
    ):
        estimate = attention[metric] - hybrid[metric]
        differences[metric] = {
            "estimate": estimate,
            "percentile_ci": {
                "confidence": 0.95,
                "lower": estimate - 0.01,
                "upper": estimate + 0.01,
            },
            "attention_win_fraction": 0.0,
            "hybrid_win_fraction": 1.0,
            "tie_fraction": 0.0,
        }
    return {
        "documents": 10,
        "tokens": 100,
        "utf8_bytes": 200,
        "attention": attention,
        "hybrid": hybrid,
        "difference_attention_minus_hybrid": differences,
    }


def _write_inputs(tmp_path: Path) -> list[SeedComparisonSpec]:
    attention_model = ModelConfig(
        name="attention-test",
        vocab_size=256,
        d_model=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden=32,
        max_seq_len=8,
        layer_types=["attn"],
    )
    hybrid_model = ModelConfig(
        name="hybrid-test",
        vocab_size=256,
        d_model=16,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden=32,
        max_seq_len=8,
        layer_types=["conv", "attn"],
    )
    attention_model_path = tmp_path / "attention-model.yaml"
    hybrid_model_path = tmp_path / "hybrid-model.yaml"
    attention_model_path.write_text(
        yaml.safe_dump(attention_model.__dict__),
        encoding="utf-8",
    )
    hybrid_model_path.write_text(
        yaml.safe_dump(hybrid_model.__dict__),
        encoding="utf-8",
    )
    specifications = []
    for offset, seed in enumerate((2027, 2028, 2029), start=1):
        base = {
            "stage": "pretrain",
            "data": {"shards_dir": "fixed", "tokenizer": {"kind": "byte"}},
            "optim": {"lr": 0.001},
            "schedule": {"total_steps": 3},
            "batch": {"micro_batch_size": 1, "grad_accum_steps": 1},
            "runtime": {"device": "cpu", "seed": seed, "resume": True},
            "log": {"eval_every": 1},
        }
        attention_config = copy.deepcopy(base)
        attention_config["model_config"] = str(attention_model_path)
        attention_config["log"]["out_dir"] = f"runs/attention-{seed}"
        hybrid_config = copy.deepcopy(base)
        hybrid_config["model_config"] = str(hybrid_model_path)
        hybrid_config["log"]["out_dir"] = f"runs/hybrid-{seed}"
        attention_config_path = tmp_path / f"attention-{seed}.yaml"
        hybrid_config_path = tmp_path / f"hybrid-{seed}.yaml"
        attention_config_path.write_text(
            yaml.safe_dump(attention_config),
            encoding="utf-8",
        )
        hybrid_config_path.write_text(
            yaml.safe_dump(hybrid_config),
            encoding="utf-8",
        )
        bootstrap_seed = 17
        resamples = 10_000
        confidence = 0.95
        attention_sidecar_sha = _sha(f"attention-sidecar-{seed}")
        hybrid_sidecar_sha = _sha(f"hybrid-sidecar-{seed}")
        comparison = {
            "kind": COMPARISON_KIND,
            "schema_version": COMPARISON_SCHEMA_VERSION,
            "comparison_sha256": canonical_sha256(
                {
                    "kind": COMPARISON_KIND,
                    "schema_version": COMPARISON_SCHEMA_VERSION,
                    "method": "paired_nonparametric_document_bootstrap",
                    "attention_sidecar_sha256": attention_sidecar_sha,
                    "hybrid_sidecar_sha256": hybrid_sidecar_sha,
                    "seed": bootstrap_seed,
                    "resamples": resamples,
                    "confidence": confidence,
                }
            ),
            "inputs": {
                "attention": {
                    "checkpoint_sha256": _sha(f"attention-checkpoint-{seed}"),
                    "model_config_sha256": canonical_sha256(attention_model.__dict__),
                    "pretrain_config_sha256": _config_sha(attention_config),
                    "training_seed": seed,
                    "sidecar": {"sha256": attention_sidecar_sha},
                    "token_accounting": {
                        "input_tokens": 1_000,
                        "loss_tokens": 999,
                        "source": "checkpoint.token_accounting",
                    },
                },
                "hybrid": {
                    "checkpoint_sha256": _sha(f"hybrid-checkpoint-{seed}"),
                    "model_config_sha256": canonical_sha256(hybrid_model.__dict__),
                    "pretrain_config_sha256": _config_sha(hybrid_config),
                    "training_seed": seed,
                    "sidecar": {"sha256": hybrid_sidecar_sha},
                    "token_accounting": {
                        "input_tokens": 1_000,
                        "loss_tokens": 999,
                        "source": "checkpoint.token_accounting",
                    },
                },
            },
            "matched_bindings": {
                "tokenizer_sha256": _sha("tokenizer"),
                "manifest_sha256": _sha("manifest"),
            },
            "evaluation": {"device": "cpu", "dtype": "float32"},
            "bootstrap": {
                "unit": "document",
                "seed": bootstrap_seed,
                "resamples": resamples,
                "confidence": confidence,
            },
            "overall": _subset(offset),
            "groups": {"general": _subset(offset)},
        }
        comparison_path = tmp_path / f"comparison-{seed}.json"
        comparison_path.write_text(
            json.dumps(comparison, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        specifications.append(
            SeedComparisonSpec(
                seed=seed,
                attention_config=attention_config_path,
                hybrid_config=hybrid_config_path,
                comparison=comparison_path,
            )
        )
    return specifications


def test_three_seed_aggregate_reports_seed_level_and_honest_uncertainty(
    tmp_path: Path,
) -> None:
    report = aggregate_pretrain_seeds(_write_inputs(tmp_path))

    assert report["kind"] == SEED_AGGREGATE_KIND
    assert report["design"]["training_seeds"] == [2027, 2028, 2029]
    assert report["design"]["primary_metric"] == "overall.bits_per_byte"
    metric = report["overall"]["metrics"]["cross_entropy_nats_per_token"]
    difference = metric["difference_attention_minus_hybrid"]
    assert [row["estimate"] for row in metric["seed_estimates"]] == pytest.approx(
        [0.1, 0.2, 0.3]
    )
    assert difference["mean"] == pytest.approx(0.2)
    assert difference["sample_standard_deviation"] == pytest.approx(
        statistics.stdev([0.1, 0.2, 0.3])
    )
    expected_margin = 4.302652729911275 * statistics.stdev([0.1, 0.2, 0.3]) / math.sqrt(3)
    assert difference["student_t_95_interval"]["lower"] == pytest.approx(
        0.2 - expected_margin
    )
    assert difference["student_t_95_interval"]["upper"] == pytest.approx(
        0.2 + expected_margin
    )
    assert difference["hybrid_favoring_seeds"] == 3
    assert difference["exact_sign_test"]["majority_one_sided_p_value"] == 0.125
    assert difference["exact_sign_test"]["two_sided_p_value"] == 0.25
    assert len(report["aggregate_sha256"]) == 64


def test_seed_aggregate_rejects_config_hash_mismatch(tmp_path: Path) -> None:
    specifications = _write_inputs(tmp_path)
    comparison_path = Path(specifications[0].comparison)
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    comparison["inputs"]["attention"]["pretrain_config_sha256"] = _sha("wrong")
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")

    with pytest.raises(ValueError, match="pretrain config hash mismatch"):
        aggregate_pretrain_seeds(specifications)


def test_seed_aggregate_rejects_dataset_mismatch(tmp_path: Path) -> None:
    specifications = _write_inputs(tmp_path)
    comparison_path = Path(specifications[-1].comparison)
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    comparison["matched_bindings"]["manifest_sha256"] = _sha("different-manifest")
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")

    with pytest.raises(ValueError, match="different dataset bindings"):
        aggregate_pretrain_seeds(specifications)


def test_seed_aggregate_requires_exactly_three_unique_seeds(tmp_path: Path) -> None:
    specifications = _write_inputs(tmp_path)

    with pytest.raises(ValueError, match="exactly three"):
        aggregate_pretrain_seeds(specifications[:2])
    duplicate = [specifications[0], specifications[0], specifications[2]]
    with pytest.raises(ValueError, match="unique"):
        aggregate_pretrain_seeds(duplicate)
