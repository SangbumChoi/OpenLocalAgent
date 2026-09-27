#!/usr/bin/env python3
"""Audit ClawBench V1/V2 task metadata without launching live websites."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


DATASET = "TIGER-Lab/ClawBench"
DATASET_URL = "https://huggingface.co/datasets/TIGER-Lab/ClawBench"
SOURCE_URL = "https://github.com/TIGER-AI-Lab/ClawBench"
REVISION = "cc146e2128724f47f2a7246f1a3057c643b22f70"
LICENSE = "Apache-2.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:  # pragma: no cover - environment-dependent import guard
        raise RuntimeError("pyarrow is required to audit ClawBench parquet metadata") from error
    return [dict(row) for row in parquet.read_table(path).to_pylist()]


def _profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    methods: Counter[str] = Counter()
    placeholders = 0
    extra_info = 0
    for row in rows:
        schema = json.loads(str(row["eval_schema"]))
        methods[str(schema.get("method"))] += 1
        placeholders += int(schema.get("url_pattern") == "__PLACEHOLDER_WILL_NOT_MATCH__")
        extra_info += int(bool(row.get("extra_info")))
    return {
        "tasks": len(rows),
        "unique_task_ids": len({row["task_id"] for row in rows}),
        "task_id_min": min(row["task_id"] for row in rows),
        "task_id_max": max(row["task_id"] for row in rows),
        "metaclasses": dict(sorted(Counter(str(row["metaclass"]) for row in rows).items())),
        "platforms": len({str(row["platform"]) for row in rows}),
        "methods": dict(sorted(methods.items())),
        "interception_placeholder_tasks": placeholders,
        "extra_info_tasks": extra_info,
    }


def audit(v1_path: Path, v2_path: Path, eval_yaml: Path, shared_profile: Path) -> dict[str, Any]:
    v1 = _load_rows(v1_path)
    v2 = _load_rows(v2_path)
    v1_ids = {row["task_id"] for row in v1}
    v2_ids = {row["task_id"] for row in v2}
    body: dict[str, Any] = {
        "kind": "localagent_clawbench_source_audit",
        "schema_version": 1,
        "source": {
            "dataset": DATASET,
            "dataset_url": DATASET_URL,
            "source_url": SOURCE_URL,
            "revision": REVISION,
            "license": LICENSE,
            "files": {
                "v1_tasks": _identity(v1_path),
                "v2_tasks": _identity(v2_path),
                "eval_yaml": _identity(eval_yaml),
                "shared_profile": _identity(shared_profile),
            },
        },
        "corpora": {"v1": _profile(v1), "v2": _profile(v2)},
        "cross_corpus": {
            "v1_v2_task_id_overlap": len(v1_ids & v2_ids),
            "v1_v2_task_id_disjoint": not bool(v1_ids & v2_ids),
        },
        "evaluation_boundary": {
            "public_task_metadata": True,
            "public_live_trace_companions": True,
            "public_static_train_test_split": False,
            "native_live_websites_and_credentials": False,
            "train_policy": "eval_only",
            "reason": (
                "The Hub release registers both corpora as test splits and publishes task metadata. "
                "Scoring requires a live browser harness, request interception, and (for reward) a "
                "judge; no static training split or reproducible local website environment is bundled."
            ),
        },
        "claim_boundary": (
            "Source/protocol audit only. No live website, account, browser harness, intercepted request, "
            "LLM judge, or irreversible side effect was executed; no ClawBench score is claimed."
        ),
    }
    body["receipt_self_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--eval-yaml", type=Path, required=True)
    parser.add_argument("--shared-profile", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    report = audit(args.v1, args.v2, args.eval_yaml, args.shared_profile)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
