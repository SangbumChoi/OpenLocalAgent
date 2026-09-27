#!/usr/bin/env python3
"""Record the local dependency preflight for the realistic-agent catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from localagent.eval.realistic_preflight import preflight_catalog


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def assemble(catalog_path: Path) -> dict[str, Any]:
    report = preflight_catalog(catalog_path)
    body: dict[str, Any] = {
        "kind": "localagent_realistic_agent_preflight_receipt",
        "schema_version": 1,
        "captured_at": "2026-08-06",
        "catalog_sha256": report["catalog_sha256"],
        "catalog_entries": report["catalog_entries"],
        "dependency_probes": report["dependency_probes"],
        "runnable_ids": report["runnable_ids"],
        "blocked_ids": report["blocked_ids"],
        "counts": report["counts"],
        "decision": {
            "native_evaluation_ready": False,
            "train_data_admission_unchanged": True,
            "reason": (
                "Only the four source adapters explicitly marked supported or "
                "supported_text_first_pilot are runnable in this workspace. Native mobile, "
                "browser, desktop, MCP, and container rows remain blocked unless their exact "
                "emulator/VM/service dependency and verifier are available."
            ),
        },
        "claim_boundary": (
            "Read-only dependency preflight. It does not download benchmark data, start an "
            "emulator/browser/VM/MCP service, or establish an official benchmark score."
        ),
    }
    body["receipt_self_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("configs/data/realistic-agent-eval.catalog.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = assemble(args.catalog)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
