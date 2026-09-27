#!/usr/bin/env python3
"""Record a source-linked refresh of the realistic-agent evaluation catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from localagent.data.realistic_catalog import load_catalog


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def assemble(catalog_path: Path) -> dict[str, Any]:
    catalog, fingerprint = load_catalog(catalog_path)
    rows = {entry["id"]: entry for entry in catalog["entries"]}
    required = {"iosworld", "mobile_safety_bench"}
    if not required <= rows.keys():
        raise ValueError("catalog refresh must include iOSWorld and MobileSafetyBench")
    if any(rows[name]["train_policy"] == "train" for name in required):
        raise ValueError("new benchmark rows must remain evaluation-only")
    body: dict[str, Any] = {
        "kind": "localagent_realistic_agent_catalog_refresh_receipt",
        "schema_version": 1,
        "captured_at": "2026-08-06",
        "catalog": {
            "path": str(catalog_path),
            "sha256": fingerprint,
            "entries": len(catalog["entries"]),
            "train_entries": [entry["id"] for entry in catalog["entries"] if entry["train_policy"] == "train"],
            "evaluation_entries": [entry["id"] for entry in catalog["entries"] if entry["train_policy"] != "train"],
        },
        "new_entries": [
            {
                "id": "iosworld",
                "official_project": "https://iosworld.io/",
                "code": "https://github.com/ljang0/iosworld",
                "paper": "https://arxiv.org/abs/2606.09764",
                "source_revision": "e91f4cb2ef4c9dd48fef83a894477b41fd5e209d",
                "contract": {
                    "apps": 26,
                    "tasks": 133,
                    "categories": {"single_app": 27, "multi_app": 60, "memory": 46},
                    "observation_modes": ["vision_only", "vision_plus_xml"],
                    "optional_tool_mode": "MCP",
                    "native_requirements": ["macOS", "Xcode 26+", "iOS 26 simulator", "Appium"],
                },
                "training_policy": "eval_only",
            },
            {
                "id": "mobile_safety_bench",
                "official_project": "https://mobilesafetybench.github.io/",
                "code": "https://github.com/jylee425/mobilesafetybench/tree/release/ver.3",
                "paper": "https://arxiv.org/abs/2410.17520",
                "source_revision": "release/ver.3",
                "contract": {
                    "tasks": 250,
                    "daily_scenario_tasks": 200,
                    "prompt_injection_tasks": 50,
                    "applications": ["messaging", "web_navigation", "social_media", "calendar", "finance"],
                    "native_requirements": ["Android emulator", "ADB", "Appium", "versioned APK assets"],
                },
                "training_policy": "eval_only",
            },
        ],
        "decision": {
            "catalog_admission": True,
            "training_admission": False,
            "reason": (
                "Both sources directly cover deployment-critical mobile behavior, but their task "
                "prompts, seeded state, screenshots, APKs, rubrics, and safety labels are held out. "
                "Use them for native evaluation and text-only safety/protocol diagnostics only; do "
                "not place either benchmark row in SFT or tokenizer training."
            ),
        },
        "claim_boundary": (
            "Source and protocol inventory only. This receipt does not claim an iOSWorld or "
            "MobileSafetyBench score, native emulator execution, or visual grounding result."
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
    print(json.dumps(payload["catalog"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
