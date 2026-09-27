#!/usr/bin/env python3
"""Build, verify, and optionally publish the current model + WebGPU Space.

The default mode is local-only and writes two self-contained export directories.  ``--publish``
is an explicit side-effecting operation: it requires a write token, creates the model and static
Space repositories, uploads the verified folders, and performs an anonymous checkpoint-binding
audit.  A reachable Hub URL without a matching ``checkpoint_sha256`` is never reported as a
current release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch

from localagent.eval.demo_deploy import BUNDLE_FILES, sync_demo_bundle, verify_demo_deploy
from localagent.inference.export.to_hf import export_hf
from localagent.inference.export.to_onnx import export_web


def _token(cli_token: str | None) -> str | None:
    return cli_token or os.environ.get("HF_TOKEN")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recorded_tokenizer_path(checkpoint: Path) -> str | None:
    """Resolve the checkpoint's recorded BPE tokenizer without guessing a different asset."""

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = payload.get("tokenizer")
    if not isinstance(metadata, dict) or metadata.get("kind", "byte") == "byte":
        return None
    raw_path = metadata.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("BPE checkpoint tokenizer metadata must contain path")
    recorded = Path(raw_path)
    candidates = (
        recorded,
        checkpoint.parent / recorded,
        Path.cwd() / recorded,
        Path(__file__).resolve().parents[1] / recorded,
    )
    for candidate in candidates:
        if candidate.is_file() and not candidate.is_symlink():
            return str(candidate)
    raise FileNotFoundError(f"recorded BPE tokenizer is missing: {raw_path}")


def _release_tool_specs(checkpoint: Path) -> list[Any] | None:
    """Resolve the browser tool schemas to the checkpoint's HF dispatch pool.

    Older continuation checkpoints omit ``dispatch_tool_pool`` but retain the 51-class fixed
    tool head.  ``export_hf`` already infers the complete 63-name standard/mobile/productivity
    pool for that case; the WebGPU export must use the same pool so ``meta.json`` and
    ``dispatch_heads.json`` cannot silently fall back to the legacy 50-tool catalog.
    """

    from localagent.agent.mobile_toolset import mobile_tools, realistic_productivity_tools
    from localagent.agent.toolset import STANDARD_TOOLS
    from localagent.inference.export.to_hf import _dispatch_metadata

    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = _dispatch_metadata(checkpoint_payload)
    raw_names = metadata.get("tool_pool")
    if not isinstance(raw_names, list) or not raw_names:
        return None
    registry = {
        tool.name: tool
        for tool in [*STANDARD_TOOLS, *mobile_tools(), *realistic_productivity_tools()]
    }
    names = [str(name) for name in raw_names]
    missing = [name for name in names if name not in registry]
    if missing:
        raise ValueError(
            "checkpoint dispatch pool contains schemas unavailable to the WebGPU publisher: "
            + ", ".join(missing)
        )
    if len(set(names)) != len(names):
        raise ValueError("checkpoint dispatch pool contains duplicate tool names")
    return [registry[name] for name in names]


def _copy_static_space(source: Path, target: Path) -> None:
    """Copy app files while excluding generated model files that are supplied by ``bundle``."""

    if target.exists():
        raise FileExistsError(f"refusing to overwrite Space staging directory: {target}")
    excluded = {"bundle-manifest.json", *BUNDLE_FILES, "DEPLOY.md"}
    target.mkdir(parents=True, exist_ok=False)
    for item in source.iterdir():
        if item.name in excluded:
            continue
        destination = target / item.name
        if item.is_dir():
            shutil.copytree(item, destination)
        elif item.is_file() and not item.is_symlink():
            shutil.copy2(item, destination)


def prepare_release(
    *,
    checkpoint: Path,
    model_out: Path,
    web_out: Path,
    demo_source: Path,
    space_out: Path,
    model_repo: str,
) -> dict[str, Any]:
    """Create and verify the model bundle and static Space staging directory."""

    checkpoint = checkpoint.resolve()
    for path in (model_out, web_out, space_out):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite release output: {path}")
    model_out.parent.mkdir(parents=True, exist_ok=True)
    web_out.parent.mkdir(parents=True, exist_ok=True)
    release_tools = _release_tool_specs(checkpoint)
    expected_tool_count = len(release_tools) if release_tools is not None else None
    expected_checkpoint_sha256 = _sha256(checkpoint)
    export_hf(str(checkpoint), str(model_out), repo_id=model_repo, push=False)
    export_web(
        str(checkpoint),
        str(web_out),
        action_only=True,
        tokenizer_path=_recorded_tokenizer_path(checkpoint),
        tools=release_tools,
    )
    source_report = verify_demo_deploy(
        demo_source,
        bundle_dir=web_out,
        require_target_bundle=False,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_tool_count=expected_tool_count,
    )
    if not source_report["verified"]:
        raise RuntimeError(
            "refusing to stage an unverified WebGPU bundle: "
            + ", ".join(source_report["blockers"])
        )
    _copy_static_space(demo_source.resolve(), space_out.resolve())
    sync_demo_bundle(
        web_out,
        space_out,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_tool_count=expected_tool_count,
    )
    target_report = verify_demo_deploy(
        space_out,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_tool_count=expected_tool_count,
    )
    if not target_report["verified"]:
        raise RuntimeError(
            "refusing to publish an unverified Space staging directory: "
            + ", ".join(target_report["blockers"])
        )
    config = json.loads((model_out / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((web_out / "bundle-manifest.json").read_text(encoding="utf-8"))
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "model_out": str(model_out),
        "web_out": str(web_out),
        "space_out": str(space_out),
        "model_parameters": config["parameter_count"],
        "webgpu_bundle_identity": target_report["bundle_identity_sha256"],
        "webgpu_verified": target_report["verified"],
    }


def _upload_folder(*, repo_id: str, folder: Path, repo_type: str, token: str, public: bool) -> None:
    from huggingface_hub import HfApi, create_repo

    create_kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "token": token,
        "private": not public,
        "exist_ok": True,
        "repo_type": repo_type,
    }
    if repo_type == "space":
        create_kwargs["space_sdk"] = "static"
    create_repo(**create_kwargs)
    HfApi().upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
        path_in_repo="",
        ignore_patterns=["*.pyc", "__pycache__", ".DS_Store"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-repo", required=True, help="Hub model repo, e.g. user/localagent-webgpu-10m")
    parser.add_argument("--space-repo", required=True, help="Hub static Space repo, e.g. user/localagent-webgpu")
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--web-out", type=Path, required=True)
    parser.add_argument("--space-out", type=Path, required=True)
    parser.add_argument("--demo-source", type=Path, default=Path("spaces/localagent-webgpu"))
    parser.add_argument("--dataset-url")
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--token", help="HF write token; otherwise HF_TOKEN or hf auth login")
    parser.add_argument("--public", action="store_true", help="publish public repositories")
    parser.add_argument("--publish", action="store_true", help="upload model and Space after local verification")
    args = parser.parse_args()

    if not args.checkpoint.is_file() or args.checkpoint.is_symlink():
        raise SystemExit(f"checkpoint is not a regular file: {args.checkpoint}")
    if args.publish and not _token(args.token):
        raise SystemExit("--publish requires --token, HF_TOKEN, or `hf auth login`")
    if args.publish and args.audit_output is None:
        raise SystemExit("--publish requires --audit-output for anonymous release verification")

    prepared = prepare_release(
        checkpoint=args.checkpoint,
        model_out=args.model_out,
        web_out=args.web_out,
        demo_source=args.demo_source,
        space_out=args.space_out,
        model_repo=args.model_repo,
    )
    print(json.dumps({"prepared": prepared, "published": False}, indent=2, sort_keys=True))
    if not args.publish:
        return 0

    token = _token(args.token)
    assert token is not None
    _upload_folder(
        repo_id=args.model_repo,
        folder=args.model_out,
        repo_type="model",
        token=token,
        public=args.public,
    )
    _upload_folder(
        repo_id=args.space_repo,
        folder=args.space_out,
        repo_type="space",
        token=token,
        public=args.public,
    )

    from scripts.audit_public_hf_release import build_manifest

    manifest = build_manifest(
        model_repo=args.model_repo,
        space_repo=args.space_repo,
        checkpoint=args.checkpoint,
        dataset_url=args.dataset_url,
    )
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if manifest["verification"]["current_checkpoint_match"] is not True:
        raise SystemExit("uploaded Hub artifacts are reachable but do not bind the current checkpoint")
    print(json.dumps({"published": True, "audit": str(args.audit_output), "manifest": manifest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
