#!/usr/bin/env python3
"""Audit the pinned ToolSandbox scenario universe and official execution protocol.

This is a source/protocol audit, not a model score.  ToolSandbox's public CLI resolves all
scenarios when no ``--scenario`` filter is supplied; the repository does not publish a train/test
split.  The audit records that fact explicitly so downstream gates do not mistake a bounded local
smoke for an official benchmark result or invent a split that the source does not define.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

SOURCE_URL = "https://github.com/apple/ToolSandbox"
SOURCE_REVISION = "165848b9a78cead7ca7fe7c89c688b58e6501219"
AUDIT_FILES = (
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "tool_sandbox/cli/utils.py",
    "tool_sandbox/scenarios/__init__.py",
    "tool_sandbox/scenarios/single_tool_call_scenarios.py",
    "tool_sandbox/scenarios/multiple_tool_call_scenarios.py",
    "tool_sandbox/scenarios/multiple_user_turn_scenarios.py",
    "tool_sandbox/scenarios/insufficient_information_scenarios.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def audit(root: Path) -> dict[str, Any]:
    import sys

    sys.path.insert(0, str(root.resolve()))
    from tool_sandbox.common.tool_discovery import ToolBackend
    from tool_sandbox.scenarios import named_scenarios

    scenarios = named_scenarios(preferred_tool_backend=ToolBackend.DEFAULT)
    category_counts = Counter(
        str(category) for scenario in scenarios.values() for category in scenario.categories
    )
    augmented_suffixes = (
        "_3_distraction_tools",
        "_10_distraction_tools",
        "_all_tools",
        "_tool_description_scrambled",
        "_arg_type_scrambled",
        "_arg_description_scrambled",
        "_tool_name_scrambled",
    )
    base_names = [
        name
        for name in scenarios
        if not any(suffix in name for suffix in augmented_suffixes)
    ]
    source_files = []
    for relative in AUDIT_FILES:
        path = root / relative
        source_files.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    revision = _git_revision(root)
    if revision != SOURCE_REVISION:
        raise ValueError(f"unexpected ToolSandbox revision: {revision}")
    return {
        "kind": "localagent_toolsandbox_protocol_audit",
        "schema_version": 1,
        "source": {
            "dataset": "apple/ToolSandbox",
            "source_url": SOURCE_URL,
            "revision": revision,
            "revision_url": f"{SOURCE_URL}/tree/{revision}",
            "license": "Apple Sample Code License",
            "files": source_files,
        },
        "official_protocol": {
            "scenario_resolution": "all scenarios when --scenario is omitted",
            "official_split": None,
            "official_split_verified": False,
            "split_finding": (
                "The pinned repository defines scenario categories and generated tool/argument "
                "augmentations, but no train/test split or static leaderboard subset."
            ),
            "scenario_count": len(scenarios),
            "base_scenario_count": len(base_names),
            "augmentation_factor": len(scenarios) // len(base_names),
            "category_counts": dict(sorted(category_counts.items())),
            "requires_user_simulator": True,
            "requires_external_credentials_for_default_simulator": True,
        },
        "interpretation": {
            "native_smoke_is_official_score": False,
            "training_policy": "eval_only",
            "gate_action": "retain official_split_not_verified until a benchmark owner defines a split or protocol exception",
            "claim_boundary": (
                "This receipt audits the public scenario universe and execution protocol only. "
                "It is not a ToolSandbox model score, user-simulator run, or external-side-effect result."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolsandbox-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    payload = audit(args.toolsandbox_root)
    payload["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
