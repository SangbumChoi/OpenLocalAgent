#!/usr/bin/env python3
"""Record exact public Hugging Face dataset revisions used by realistic-agent experiments.

The realistic-agent catalog intentionally stores source links and policies, not mutable Hub
metadata.  This read-only audit resolves a small, explicit registry to immutable dataset commits,
licenses, and file inventories without downloading payloads.  It is suitable for a provenance
receipt before selecting bounded train/eval shards; it never treats evaluation-only data as SFT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_REGISTRY: tuple[dict[str, str], ...] = (
    {
        "id": "androidcontrol_mirror",
        "dataset": "OfficerChul/Android-Control-84k",
        "original_url": "https://github.com/google-research/google-research/tree/master/android_control",
        "policy": "train_only_official_train_projection",
    },
    {
        "id": "agentnet",
        "dataset": "xlangai/AgentNet",
        "original_url": "https://github.com/xlang-ai/OpenCUA",
        "policy": "train_only_official_train_projection;images_dropped_for_text_first_model",
    },
    {
        "id": "mind2web",
        "dataset": "osunlp/Mind2Web",
        "original_url": "https://github.com/OSU-NLP-Group/Mind2Web",
        "policy": "train_only_train_split;test_archives_never_training_inputs",
    },
    {
        "id": "xlam_function_calling",
        "dataset": "Salesforce/xlam-function-calling-60k",
        "original_url": "https://github.com/SalesforceAIResearch/xlam",
        "policy": "train_only_official_train_projection;denylist_eval_prompts",
    },
    {
        "id": "computer_agent_arena",
        "dataset": "xlangai/computer-agent-arena",
        "original_url": "https://huggingface.co/datasets/xlangai/computer-agent-arena",
        "policy": "evaluation_only_until_task_and_model_identity_deduplication",
    },
    {
        "id": "osworld2_trajectory",
        "dataset": "xlangai/osworld2.0-trajectory",
        "original_url": "https://github.com/xlang-ai/OSWorld",
        "policy": "evaluation_only;release_matched_vm_required",
    },
    {
        "id": "osworld_verified_trajectories",
        "dataset": "xlangai/ubuntu_osworld_verified_trajs",
        "original_url": "https://github.com/xlang-ai/OSWorld",
        "policy": "evaluation_and_provenance_only;task_identity_leakage_audit_required",
    },
    {
        "id": "enterpriseopsgym",
        "dataset": "ServiceNow-AI/EnterpriseOps-Gym",
        "original_url": "https://github.com/ServiceNow/EnterpriseOps-Gym",
        "policy": "evaluation_only;verifier_and_server_state_never_training_inputs",
    },
    {
        "id": "mcp_trajectory_benchmark",
        "dataset": "obaydata/mcp-agent-trajectory-benchmark",
        "original_url": "https://huggingface.co/datasets/obaydata/mcp-agent-trajectory-benchmark",
        "policy": "train_only_public_train_projection;agent_disjoint_internal_holdout_not_official",
    },
)


def _license(card_data: Any) -> str | None:
    if card_data is None:
        return None
    value = getattr(card_data, "license", None)
    return str(value) if value not in (None, "") else None


def _date(value: Any) -> str | None:
    return value.isoformat() if value is not None and hasattr(value, "isoformat") else None


def snapshot_dataset(api: Any, entry: Mapping[str, str], *, sample_files: int = 8) -> dict[str, Any]:
    """Resolve one public dataset to an immutable Hub revision and bounded file sample."""

    if sample_files < 1:
        raise ValueError("sample_files must be positive")
    dataset = str(entry["dataset"])
    info = api.dataset_info(dataset)
    siblings = list(getattr(info, "siblings", ()) or ())
    names = sorted(str(getattr(item, "rfilename", "")) for item in siblings)
    selected: list[str] = []
    for name in names[:sample_files] + names[-sample_files:]:
        if name not in selected:
            selected.append(name)
    files = []
    by_name = {str(getattr(item, "rfilename", "")): item for item in siblings}
    for name in selected:
        item = by_name[name]
        size = getattr(item, "size", None)
        files.append({"rfilename": name, "size": int(size) if isinstance(size, int) else None})
    return {
        "id": str(entry["id"]),
        "dataset": dataset,
        "hub_url": f"https://huggingface.co/datasets/{dataset}",
        "original_url": str(entry["original_url"]),
        "policy": str(entry["policy"]),
        "revision": str(getattr(info, "sha", "")),
        "created_at": _date(getattr(info, "created_at", None)),
        "last_modified": _date(getattr(info, "last_modified", None)),
        "license": _license(getattr(info, "card_data", None)),
        "private": bool(getattr(info, "private", False)),
        "file_count": len(names),
        "file_sample": files,
    }


def audit(registry: tuple[dict[str, str], ...] = DEFAULT_REGISTRY, *, sample_files: int = 8) -> dict[str, Any]:
    """Resolve the registry through ``huggingface_hub`` without downloading any files."""

    try:
        from huggingface_hub import HfApi
    except ImportError as error:  # pragma: no cover - optional dependency in unit tests.
        raise RuntimeError("huggingface_hub is required for public dataset snapshot audits") from error
    api = HfApi()
    rows = [snapshot_dataset(api, entry, sample_files=sample_files) for entry in registry]
    payload: dict[str, Any] = {
        "kind": "localagent_public_dataset_snapshot_audit",
        "schema_version": 1,
        "registry_policy": "metadata_only;no_payload_download;train_eval_policy_is_explicit",
        "datasets": rows,
    }
    payload["audit_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-files", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    if args.sample_files < 1:
        raise SystemExit("--sample-files must be positive")
    payload = audit(sample_files=args.sample_files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
