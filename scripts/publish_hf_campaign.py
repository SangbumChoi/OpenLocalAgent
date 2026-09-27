#!/usr/bin/env python3
"""Publish a verified LocalAgent model and optional train-data acquisition to the Hub.

The command is deliberately opt-in: without ``--push`` it only validates identity and builds a
local model bundle. With ``--push`` it requires an authenticated Hugging Face account, checks that
the destination owner matches that account, and refuses datasets without a hash-bound acquisition
manifest whose rows are all ``train`` or ``model_reference`` policy.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _whoami(api: Any, token: str | None) -> str:
    try:
        info = api.whoami(token=token)
    except Exception as error:  # pragma: no cover - depends on authentication state
        raise RuntimeError("Hugging Face authentication required; run `hf auth login`") from error
    name = info.get("name") if isinstance(info, dict) else None
    if not isinstance(name, str) or not name:
        raise RuntimeError("Hugging Face identity did not include a username")
    return name


def _repo_owner(repo_id: str) -> str:
    parts = repo_id.split("/", 1)
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"repo id must be OWNER/NAME: {repo_id!r}")
    return parts[0]


def _validate_acquisition_manifest(path: Path) -> dict[str, Any]:
    manifest = path / "acquisition-manifest.json"
    if not manifest.is_file():
        raise ValueError(f"dataset directory lacks acquisition-manifest.json: {path}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("kind") != "localagent_hf_source_acquisition" or payload.get("dry_run"):
        raise ValueError("dataset acquisition manifest is not a completed, non-dry-run receipt")
    for row in payload.get("sources", []):
        if row.get("policy") not in {"train", "model_reference"}:
            raise ValueError(f"refusing to publish non-training source: {row.get('id')}")
        if row.get("status") != "downloaded" or not row.get("files"):
            raise ValueError(f"source is not fully hash-bound: {row.get('id')}")
    return payload


def publish(
    *,
    checkpoint: Path,
    model_repo: str,
    model_out: Path,
    dataset_dir: Path | None,
    dataset_repo: str | None,
    token: str | None,
    public: bool,
    push: bool,
) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi
    except ImportError as error:  # pragma: no cover - optional runtime
        raise RuntimeError('install Hub support with: pip install -e ".[hub]"') from error
    api = HfApi(token=token)
    owner = _whoami(api, token) if push else None
    if push and owner != _repo_owner(model_repo):
        raise RuntimeError(
            f"destination owner {_repo_owner(model_repo)!r} does not match authenticated user {owner!r}"
        )
    if dataset_dir is not None:
        if dataset_repo is None:
            raise ValueError("--dataset-repo is required with --dataset-dir")
        dataset_manifest = _validate_acquisition_manifest(dataset_dir)
        if push and owner != _repo_owner(dataset_repo):
            raise RuntimeError("dataset destination owner does not match authenticated user")
    else:
        dataset_manifest = None
    from localagent.inference.export.to_hf import export_hf

    bundle = export_hf(
        str(checkpoint),
        str(model_out),
        repo_id=model_repo if push else None,
        token=token,
        private=not public,
        push=push,
    )
    dataset_url = None
    if push and dataset_dir is not None and dataset_repo is not None:
        api.create_repo(dataset_repo, repo_type="dataset", private=not public, exist_ok=True)
        api.upload_folder(
            repo_id=dataset_repo,
            repo_type="dataset",
            folder_path=str(dataset_dir),
            commit_message="Upload hash-bound LocalAgent training acquisition",
        )
        dataset_url = f"https://huggingface.co/datasets/{dataset_repo}"
    return {
        "kind": "localagent_hf_campaign_publication",
        "schema_version": 1,
        "pushed": push,
        "model_repo": model_repo,
        "model_url": f"https://huggingface.co/{model_repo}" if push else None,
        "model_bundle": str(bundle),
        "dataset_repo": dataset_repo,
        "dataset_url": dataset_url,
        "dataset_manifest_sha256": (
            dataset_manifest.get("receipt_self_sha256") if dataset_manifest is not None else None
        ),
        "authenticated_owner": owner,
        "claim_boundary": (
            "Hub publication identity and upload result only. This does not claim benchmark success, "
            "visual grounding, native mobile/desktop control, or external-account side effects."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--model-out", type=Path, default=Path("runs/hf_campaign_model"))
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--dataset-repo")
    parser.add_argument("--token", default=None)
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    token = args.token or os.environ.get("HF_TOKEN")
    try:
        report = publish(
            checkpoint=args.checkpoint,
            model_repo=args.model_repo,
            model_out=args.model_out,
            dataset_dir=args.dataset_dir,
            dataset_repo=args.dataset_repo,
            token=token,
            public=args.public,
            push=args.push,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
