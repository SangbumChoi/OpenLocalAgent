#!/usr/bin/env python
"""Profile the public EnterpriseOps-Gym card/API without starting its servers.

The Hugging Face API exposes split/config metadata while the dataset card documents the broader
enterprise inventory.  This command binds both public metadata files to a commit and records the
small, reproducible distinction between card-level task counts and downloadable rows.  It never
downloads parquet rows, reads verifiers, starts Docker/MCP services, or treats benchmark tasks as
training data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


DATASET = "ServiceNow-AI/EnterpriseOps-Gym"
SOURCE_URL = "https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym"
CODE_URL = "https://github.com/ServiceNow/EnterpriseOps-Gym"
REVISION = "c8e538eae8a6205294f0a86675fefdc1fac408f6"
LICENSE = "Apache-2.0"
CONFIGS = ("oracle", "plus_5_tools", "plus_10_tools", "plus_15_tools")
DOMAINS = ("calendar", "csm", "drive", "email", "hr", "hybrid", "itsm", "teams")
FEATURES = {
    "task_id",
    "domain",
    "system_prompt",
    "user_prompt",
    "selected_tools",
    "restricted_tools",
    "mcp_endpoint",
    "number_of_runs",
    "reset_database_between_runs",
    "gym_servers_config",
    "verifiers",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read Hugging Face API metadata: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("Hugging Face API payload must be an object")
    return payload


def _card_claims(text: str) -> dict[str, Any]:
    """Extract the dataset card's stated inventory, preserving an internal count discrepancy."""

    about_match = re.search(r"comprises \*\*([\d,]+) expert-curated tasks", text)
    tools_match = re.search(r"\*\*(\d+) tools\*\* across (\d+) enterprise domains", text)
    avg_match = re.search(
        r"\*\*(\d+(?:\.\d+)?) avg steps\*\* per task \(up to (\d+)", text
    ) or re.search(
        r"\*\*(\d+(?:\.\d+)?) avg steps per task\*\* \(up to (\d+)", text
    )
    table_rows = re.findall(
        r"^\| ([A-Za-z]+) \| (\d+) \| ([0-9.]+) \| (\d+) \| ([^|]+) \|$",
        text,
        flags=re.MULTILINE,
    )
    if not about_match or not tools_match or not avg_match or len(table_rows) < 8:
        raise ValueError("EnterpriseOps-Gym card inventory table is missing expected claims")
    domains: dict[str, dict[str, Any]] = {}
    for name, tasks, avg_steps, max_steps, tools in table_rows:
        if name == "Total":
            continue
        domains[name.lower()] = {
            "tasks": int(tasks),
            "average_steps": float(avg_steps),
            "max_steps": int(max_steps),
            "tools": tools.strip() if tools.strip() == "Multi-domain" else int(tools),
        }
    if set(domains) != set(DOMAINS) - {"hybrid"} | {"hybrid"}:
        raise ValueError(f"unexpected EnterpriseOps-Gym domain table: {sorted(domains)}")
    total_match = re.search(r"\| \*\*Total\*\* \| \*\*([\d,]+)\*\* \|", text)
    if not total_match:
        raise ValueError("EnterpriseOps-Gym card total is missing")
    return {
        "about_task_count": int(about_match.group(1).replace(",", "")),
        "tool_count": int(tools_match.group(1)),
        "domain_count": int(tools_match.group(2)),
        "average_steps": float(avg_match.group(1)),
        "max_steps": int(avg_match.group(2)),
        "domain_table_task_count": int(total_match.group(1).replace(",", "")),
        "domains": domains,
        "count_discrepancy": int(about_match.group(1).replace(",", ""))
        != int(total_match.group(1).replace(",", "")),
    }


def profile(api_path: Path, card_path: Path, *, revision: str = REVISION) -> dict[str, Any]:
    """Return a deterministic, metadata-only EnterpriseOps-Gym inventory receipt."""

    api = _read_json(api_path)
    if api.get("id") != DATASET or api.get("sha") != revision:
        raise ValueError("Hugging Face API metadata is not pinned to the expected dataset revision")
    if api.get("private") is not False or api.get("gated") is not False:
        raise ValueError("EnterpriseOps-Gym API metadata must be public and ungated")
    card_data = api.get("cardData")
    if not isinstance(card_data, Mapping) or card_data.get("license") != "apache-2.0":
        raise ValueError("EnterpriseOps-Gym API metadata must report the Apache-2.0 license")
    infos = card_data.get("dataset_info")
    if not isinstance(infos, list):
        raise ValueError("Hugging Face API metadata is missing dataset_info")

    config_rows: dict[str, Any] = {}
    for info in infos:
        if not isinstance(info, Mapping) or not isinstance(info.get("config_name"), str):
            raise ValueError("dataset_info entries must contain config_name")
        name = str(info["config_name"])
        if name in config_rows:
            raise ValueError(f"duplicate EnterpriseOps-Gym config: {name}")
        if name not in CONFIGS:
            raise ValueError(f"unexpected EnterpriseOps-Gym config: {name}")
        features = info.get("features")
        if not isinstance(features, list):
            raise ValueError(f"{name} is missing features")
        feature_names = {item.get("name") for item in features if isinstance(item, Mapping)}
        if feature_names != FEATURES:
            raise ValueError(f"{name} feature schema drift: {sorted(feature_names)}")
        splits = info.get("splits")
        if not isinstance(splits, list) or {item.get("name") for item in splits} != set(DOMAINS):
            raise ValueError(f"{name} split inventory drift")
        split_rows: dict[str, dict[str, int]] = {}
        for split in splits:
            if not isinstance(split, Mapping):
                raise ValueError(f"{name} contains a malformed split")
            split_name = str(split["name"])
            examples = split.get("num_examples")
            num_bytes = split.get("num_bytes")
            if not isinstance(examples, int) or examples <= 0 or not isinstance(num_bytes, int):
                raise ValueError(f"{name}/{split_name} has invalid size metadata")
            split_rows[split_name] = {"examples": examples, "bytes": num_bytes}
        config_rows[name] = {
            "splits": {domain: split_rows[domain] for domain in DOMAINS},
            "total_examples": sum(row["examples"] for row in split_rows.values()),
            "download_size": int(info["download_size"]),
            "dataset_size": int(info["dataset_size"]),
        }
    if set(config_rows) != set(CONFIGS):
        raise ValueError(f"EnterpriseOps-Gym configs missing: {sorted(set(CONFIGS) - set(config_rows))}")

    card_text = card_path.read_text(encoding="utf-8")
    claims = _card_claims(card_text)
    payload: dict[str, Any] = {
        "kind": "localagent_enterpriseopsgym_metadata_receipt",
        "schema_version": 1,
        "dataset": DATASET,
        "source_url": SOURCE_URL,
        "code_url": CODE_URL,
        "api_url": (
            f"https://huggingface.co/api/datasets/{DATASET}/revision/{revision}"
        ),
        "card_url": f"{SOURCE_URL}/resolve/{revision}/README.md",
        "revision": revision,
        "license": LICENSE,
        "source": {
            "api_path": api_path.name,
            "api_bytes": api_path.stat().st_size,
            "api_sha256": _sha256(api_path),
            "card_path": card_path.name,
            "card_bytes": card_path.stat().st_size,
            "card_sha256": _sha256(card_path),
            "metadata_only": True,
            "parquet_rows_downloaded": False,
            "verifiers_read": False,
            "servers_started": False,
            "docker_invoked": False,
        },
        "card_inventory": claims,
        "config_inventory": config_rows,
        "toolset_modes": {
            "oracle": "exact task-required tools",
            "plus_5_tools": "oracle plus five distractors",
            "plus_10_tools": "oracle plus ten distractors",
            "plus_15_tools": "oracle plus fifteen distractors",
        },
        "runtime_requirements": {
            "containerized_mcp_servers": True,
            "docker": True,
            "sql_state_verifiers": True,
            "reset_between_runs": True,
        },
        "claim_boundary": (
            "Public API and dataset-card inventory only; no parquet task text, verifier, server "
            "configuration, Docker/MCP execution, state transition, official task-success score, "
            "leaderboard result, or training artifact is claimed. The card's About count (1,150) "
            "and domain-table total (1,115) are preserved as separate source claims."
        ),
    }
    payload["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", type=Path, required=True)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite metadata receipt")
    payload = profile(args.api, args.card, revision=args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
