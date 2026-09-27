#!/usr/bin/env python3
"""Audit pinned AgentNet trajectory/image provenance without downloading image archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

DATASET = "xlangai/AgentNet"
DATASET_URL = "https://huggingface.co/datasets/xlangai/AgentNet"
REVISION = "d76ee50a63fad81cfdbe576416757d7c2091ed50"
ORIGINAL_REPO = "https://github.com/xlang-ai/OpenCUA"


def _get(url: str, *, start: int | None = None, end: int | None = None) -> bytes:
    headers = {}
    if start is not None and end is not None:
        headers["Range"] = f"bytes={start}-{end}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def _identity(data: bytes) -> dict[str, Any]:
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def audit(*, prefix_bytes: int = 1_048_576) -> dict[str, Any]:
    if prefix_bytes < 1024:
        raise ValueError("prefix_bytes must be at least 1024")
    tree_url = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}?recursive=true&expand=false"
    tree = json.loads(_get(tree_url).decode("utf-8"))
    inventory = [
        {"path": item["path"], "size": item.get("size"), "type": item.get("type")}
        for item in tree
        if item.get("path", "").startswith(("ubuntu_images/", "win_mac_images/"))
        or item.get("path") in {"agentnet_ubuntu_5k.jsonl", "agentnet_win_mac_18k.jsonl", "meta_data_merged.jsonl"}
    ]
    traj_path = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/agentnet_ubuntu_5k.jsonl"
    traj_prefix = _get(traj_path, start=0, end=prefix_bytes - 1)
    rows = []
    for line in traj_prefix.decode("utf-8", errors="ignore").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(rows) >= 3:
            break
    if not rows:
        raise ValueError("no complete AgentNet trajectory row in bounded prefix")
    image_refs: list[str] = []
    action_counts: dict[str, int] = {}
    for row in rows:
        for step in row.get("traj", []):
            image = step.get("image")
            if isinstance(image, str):
                image_refs.append(image)
            value = step.get("value", {})
            code = str(value.get("code", ""))
            family = code.split("(", 1)[0].split(".")[-1] if code else "unknown"
            action_counts[family] = action_counts.get(family, 0) + 1
    metadata_path = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/meta_data_merged.jsonl"
    metadata_prefix = _get(metadata_path, start=0, end=prefix_bytes - 1)
    metadata_rows = []
    for line in metadata_prefix.decode("utf-8", errors="ignore").splitlines():
        try:
            metadata_rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(metadata_rows) >= 3:
            break
    return {
        "kind": "localagent_agentnet_visual_source_audit",
        "schema_version": 1,
        "source": {
            "dataset": DATASET,
            "dataset_url": DATASET_URL,
            "revision": REVISION,
            "original_repository": ORIGINAL_REPO,
            "trajectory_file": "agentnet_ubuntu_5k.jsonl",
            "metadata_file": "meta_data_merged.jsonl",
        },
        "archive_inventory": inventory,
        "bounded_fetch": {
            "prefix_bytes_requested": prefix_bytes,
            "trajectory_prefix": _identity(traj_prefix),
            "metadata_prefix": _identity(metadata_prefix),
        },
        "trajectory_sample": {
            "rows": len(rows),
            "task_ids": [row.get("task_id") for row in rows],
            "systems": [row.get("system") for row in rows],
            "screen_step_count": [len(row.get("traj", [])) for row in rows],
            "image_reference_count": len(image_refs),
            "unique_image_reference_count": len(set(image_refs)),
            "first_image_references": image_refs[:12],
            "action_family_counts": action_counts,
            "sample_row_sha256": [hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest() for row in rows],
            "metadata_sample_rows": metadata_rows,
        },
        "pipeline_boundary": {
            "visual_archive_present": any(
                item["path"] == "ubuntu_images/images.zip" and int(item.get("size") or 0) > 0
                for item in inventory
            ),
            "local_projection_consumes_images": False,
            "current_agentnet_receipts_visual_input_omitted": True,
            "training_admission": "provenance_only_until_image_loader_vision_encoder_and_visual_eval_are_bound",
        },
        "claim_boundary": (
            "This receipt proves that the pinned public AgentNet revision contains image-referenced "
            "computer-use trajectories and multi-gigabyte image archives. It does not download or "
            "train on those images, and it does not claim native desktop, AgentNetBench, OSWorld, "
            "or WebGPU visual success."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-bytes", type=int, default=1_048_576)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(prefix_bytes=args.prefix_bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["trajectory_sample"], sort_keys=True))


if __name__ == "__main__":
    main()
