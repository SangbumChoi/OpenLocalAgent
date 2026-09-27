#!/usr/bin/env python
"""Plan or acquire a licensed, revision-pinned pretraining mixture.

Examples:
  python scripts/download_pretrain_mixture.py configs/data/pretrain-paper.yaml \
      --out data/raw/paper --dry-run
  python scripts/download_pretrain_mixture.py configs/data/pretrain-paper.yaml \
      --out data/raw/paper --license-evidence smollm-card=data/provenance/smollm.md \
      --license-evidence codeparrot-card=data/provenance/codeparrot.md \
      --license-evidence websight-card=data/provenance/websight.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from localagent.data.hf_corpus import (
    audit_mixture_readiness,
    build_mixture_plan,
    stream_mixture,
)


def _evidence_mapping(values: list[str]) -> dict[str, Path]:
    evidence: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--license-evidence must use ID=PATH")
        evidence_id, raw_path = value.split("=", 1)
        if not evidence_id or not raw_path:
            raise ValueError("--license-evidence must use non-empty ID=PATH")
        if evidence_id in evidence:
            raise ValueError(f"duplicate --license-evidence id {evidence_id!r}")
        evidence[evidence_id] = Path(raw_path)
    return evidence


def _write_plan(path: str, payload: dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="mixture YAML")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument(
        "--target-chars",
        type=int,
        help="override the YAML character bound (exact tokens are known after packing)",
    )
    parser.add_argument(
        "--license-evidence",
        action="append",
        default=[],
        metavar="ID=PATH",
        help="local copy of a config-pinned dataset card/license artifact (repeatable)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate policy, evidence, exact budgets, and disk only; open no dataset streams",
    )
    parser.add_argument(
        "--plan-out",
        help="atomically write the plan/readiness JSON (useful for timestamping)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse verified completed-source spools and replay only the partial source",
    )
    args = parser.parse_args()

    try:
        evidence = _evidence_mapping(args.license_evidence)
    except ValueError as exc:
        parser.error(str(exc))
    plan = build_mixture_plan(args.config, target_chars=args.target_chars)
    readiness = audit_mixture_readiness(
        plan,
        args.out,
        license_evidence=evidence,
        require_stream_runtime=True,
        resume=args.resume,
    )
    preflight = {"plan": plan, "readiness": readiness}
    if args.plan_out:
        _write_plan(args.plan_out, preflight)
    if args.dry_run:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0 if readiness["ready"] else 2
    if not readiness["ready"]:
        parser.error("acquisition is not ready: " + "; ".join(readiness["blockers"]))

    manifest = stream_mixture(
        args.config,
        args.out,
        target_chars=args.target_chars,
        license_evidence=evidence,
        resume=args.resume,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
