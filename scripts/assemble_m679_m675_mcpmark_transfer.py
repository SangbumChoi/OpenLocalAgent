#!/usr/bin/env python3
"""Seal matched m675 warm/random MCPMark trajectory continuation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DATASET = "Jakumetsu/mcpmark-trajectory-log"
REVISION = "e50578f0ab904d8e6a7c576c387c1e76ae482c89"
WARM_PARENT = "91eb969628d77099d6b834cf3c1a6bd0a1fa37adefffe80f2d57605e1b775dcc"
RANDOM_PARENT = "ef9e7d978bbb047086259e76a012c16fa15093a33e948fbe6907935b2faf5ad0"


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _identity(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"expected regular file: {path}")
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate(report: dict[str, Any], *, name: str, parent_sha: str) -> None:
    if report.get("kind") != "localagent_public_agent_continuation_report":
        raise ValueError(f"{name} report kind mismatch")
    if report.get("parent", {}).get("sha256") != parent_sha:
        raise ValueError(f"{name} parent checkpoint mismatch")
    if report.get("source", {}).get("dataset") != DATASET:
        raise ValueError(f"{name} dataset mismatch")
    if report.get("source", {}).get("revision") != REVISION:
        raise ValueError(f"{name} revision mismatch")
    if report.get("rows") != {"train": 10, "eval": 5}:
        raise ValueError(f"{name} row counts mismatch")
    expected = {"steps": 32, "batch_size": 2, "max_seq_len": 512, "learning_rate": 1.0e-5}
    for key, value in expected.items():
        if report.get("hyperparameters", {}).get(key) != value:
            raise ValueError(f"{name} hyperparameter mismatch: {key}")
    compatibility = report.get("weight_transfer", {}).get("compatibility", {})
    if compatibility.get("config_mismatches") != {} or compatibility.get("shape_mismatches") != {}:
        raise ValueError(f"{name} tensor compatibility failed")
    if compatibility.get("tokenizer_sha256_equal") is not True:
        raise ValueError(f"{name} tokenizer compatibility failed")


def _arm(report: dict[str, Any]) -> dict[str, Any]:
    transfer = report["weight_transfer"]
    return {
        "parent": report["parent"],
        "child": report["child"],
        "before": report["before"]["eval"],
        "after": report["after"]["eval"],
        "weight_transfer": {
            "compatibility": transfer["compatibility"],
            "groups": transfer["groups"],
            "recommendation": transfer["recommendation"],
        },
    }


def assemble(warm_path: Path, random_path: Path, output: Path) -> dict[str, Any]:
    warm_report = _load(warm_path)
    random_report = _load(random_path)
    _validate(warm_report, name="warm", parent_sha=WARM_PARENT)
    _validate(random_report, name="random", parent_sha=RANDOM_PARENT)
    warm = _arm(warm_report)
    random = _arm(random_report)
    warm_before = float(warm["before"]["assistant_token_accuracy"])
    warm_after = float(warm["after"]["assistant_token_accuracy"])
    random_before = float(random["before"]["assistant_token_accuracy"])
    random_after = float(random["after"]["assistant_token_accuracy"])
    payload: dict[str, Any] = {
        "kind": "localagent_m679_m675_mcpmark_transfer",
        "schema_version": 1,
        "source": {
            "dataset": DATASET,
            "url": "https://huggingface.co/datasets/Jakumetsu/mcpmark-trajectory-log",
            "original_benchmark": "https://github.com/eval-sys/mcpmark",
            "revision": REVISION,
            "train_manifest": warm_report["source"]["manifest"],
            "train_rows": warm_report["train_inputs"],
            "eval_rows": warm_report["eval_inputs"],
            "redaction": "tool_outputs_and_assistant_free_text_redacted",
        },
        "protocol": {
            "train_rows": 10,
            "eval_rows": 5,
            "steps": 32,
            "batch_size": 2,
            "learning_rate": 1.0e-5,
            "max_seq_len": 512,
            "domains": ["filesystem", "notion", "github", "playwright", "postgres"],
            "official_native_score": False,
        },
        "arms": {"warm": warm, "random": random},
        "comparison": {
            "warm_before_eval_token_accuracy": warm_before,
            "warm_after_eval_token_accuracy": warm_after,
            "warm_gain_pp": (warm_after - warm_before) * 100.0,
            "random_before_eval_token_accuracy": random_before,
            "random_after_eval_token_accuracy": random_after,
            "random_gain_pp": (random_after - random_before) * 100.0,
            "warm_minus_random_after_pp": (warm_after - random_after) * 100.0,
            "warm_start_better_after": warm_after > random_after,
            "exact_sequence_accuracy": {
                "warm": warm["after"]["assistant_sequence_accuracy"],
                "random": random["after"]["assistant_sequence_accuracy"],
            },
        },
        "weight_adoption": {
            "config_compatible": True,
            "tokenizer_compatible": True,
            "warm_action_heads_frozen": warm["weight_transfer"]["groups"]["action_heads"]["delta_l2"] == 0.0,
            "warm_body_relative_delta_max": max(
                warm["weight_transfer"]["groups"][name]["relative_delta_l2"]
                for name in ("embedding", "attention_or_mixer", "ffn", "normalization")
            ),
            "random_body_relative_delta_max": max(
                random["weight_transfer"]["groups"][name]["relative_delta_l2"]
                for name in ("embedding", "attention_or_mixer", "ffn", "normalization")
            ),
            "reuse_warm_backbone": warm_after > random_after,
            "recommendation": "retain warm initialization for MCP text continuation; require live server/verifier tests before promotion",
        },
        "raw_reports": {"warm": _identity(warm_path), "random": _identity(random_path)},
        "claim_boundary": (
            "Redacted public MCPMark trajectory continuation with text-only teacher-forced metrics. "
            "This is not an official MCPMark score, native MCP server execution, or real Notion, "
            "browser, filesystem, GitHub, or Postgres side effect."
        ),
    }
    payload["receipt_self_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-report", type=Path, required=True)
    parser.add_argument("--random-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = assemble(args.warm_report, args.random_report, args.out)
    print(json.dumps(payload["comparison"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
