#!/usr/bin/env python3
"""Audit a public Hugging Face model/Space release against a local checkpoint.

The audit is intentionally read-only and anonymous.  It never uploads or mutates a Hub repo.  A
release is current only when the public model config exposes the exact local checkpoint SHA-256;
an HTTP-200 legacy model or demo is recorded as public but not current.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


HUB = "https://huggingface.co"
Fetch = Callable[[str], bytes]


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_url(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "LocalAgent-public-audit/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _json(fetch: Fetch, url: str) -> dict[str, Any]:
    raw = fetch(url)
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object from {url}")
    return dict(value)


def _api_url(kind: str, repo: str) -> str:
    return f"{HUB}/api/{kind}/{repo}"


def _resolve_url(kind: str, repo: str, revision: str, filename: str) -> str:
    prefix = "spaces/" if kind == "spaces" else ""
    encoded = urllib.parse.quote(filename, safe="/")
    return f"{HUB}/{prefix}{repo}/resolve/{revision}/{encoded}?download=true"


def _siblings(metadata: Mapping[str, Any]) -> set[str]:
    return {
        item["rfilename"]
        for item in metadata.get("siblings", [])
        if isinstance(item, Mapping) and isinstance(item.get("rfilename"), str)
    }


def _pick_file(siblings: set[str], requested: str | None, candidates: tuple[str, ...]) -> str:
    if requested:
        if requested not in siblings:
            raise ValueError(f"requested Hub file is absent: {requested}")
        return requested
    for candidate in candidates:
        if candidate in siblings:
            return candidate
    raise ValueError(f"none of the expected Hub files are present: {', '.join(candidates)}")


def build_manifest(
    *,
    model_repo: str,
    space_repo: str,
    checkpoint: Path,
    dataset_url: str | None = None,
    model_file: str | None = None,
    demo_file: str | None = None,
    fetch: Fetch = fetch_url,
) -> dict[str, Any]:
    """Fetch public Hub metadata and return a current-checkpoint-aware manifest."""

    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise FileNotFoundError(f"checkpoint is not a regular file: {checkpoint}")
    checkpoint_sha256 = _sha256_file(checkpoint)

    model_metadata = _json(fetch, _api_url("models", model_repo))
    space_metadata = _json(fetch, _api_url("spaces", space_repo))
    model_revision = model_metadata.get("sha")
    space_revision = space_metadata.get("sha")
    if not isinstance(model_revision, str) or not model_revision:
        raise ValueError("public model API did not return a revision")
    if not isinstance(space_revision, str) or not space_revision:
        raise ValueError("public Space API did not return a revision")
    if model_metadata.get("private") is True:
        raise ValueError("model repository is private")
    if space_metadata.get("private") is True:
        raise ValueError("Space repository is private")

    model_name = _pick_file(
        _siblings(model_metadata), model_file, ("model.safetensors", "pytorch_model.bin")
    )
    demo_name = _pick_file(_siblings(space_metadata), demo_file, ("model.fp16.onnx", "model.onnx"))
    model_raw = fetch(_resolve_url("models", model_repo, model_revision, model_name))
    demo_raw = fetch(_resolve_url("spaces", space_repo, space_revision, demo_name))

    config_raw = fetch(_resolve_url("models", model_repo, model_revision, "config.json"))
    config = json.loads(config_raw.decode("utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("public model config must be a JSON object")
    published_checkpoint = config.get("checkpoint_sha256")
    current_match = published_checkpoint == checkpoint_sha256
    manifest: dict[str, Any] = {
        "kind": "localagent_public_model_demo_manifest",
        "schema_version": 2,
        "public": True,
        "model_url": f"{HUB}/{model_repo}",
        "demo_url": f"{HUB}/spaces/{space_repo}",
        "artifact_sha256": _sha256_bytes(model_raw),
        "current_checkpoint_sha256": published_checkpoint
        if isinstance(published_checkpoint, str)
        else None,
        "model": {
            "repo": model_repo,
            "revision": model_revision,
            "file": model_name,
            "bytes": len(model_raw),
            "sha256": _sha256_bytes(model_raw),
            "config_sha256": _sha256_bytes(config_raw),
            "parameters": config.get("parameter_count"),
        },
        "demo": {
            "repo": space_repo,
            "revision": space_revision,
            "model_graph": {
                "file": demo_name,
                "bytes": len(demo_raw),
                "sha256": _sha256_bytes(demo_raw),
            },
        },
        "dataset": {"url": dataset_url} if dataset_url else None,
        "verification": {
            "method": "anonymous Hugging Face API, resolver downloads, and SHA-256",
            "public_model_http_status": 200,
            "public_demo_http_status": 200,
            "model_revision": model_revision,
            "space_revision": space_revision,
            "current_checkpoint_match": current_match,
            "local_checkpoint": str(checkpoint),
            "local_checkpoint_sha256": checkpoint_sha256,
        },
        "claim_boundary": (
            "This receipt verifies public Hub reachability and exact downloaded artifact hashes. "
            "It binds the current release only when current_checkpoint_match is true; otherwise "
            "the public model/demo is legacy or unrelated to the supplied checkpoint. No native "
            "mobile, browser, desktop, MCP, email, or Notion task success is implied."
        ),
    }
    manifest["receipt_self_sha256"] = _sha256_bytes(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "receipt_self_sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--space-repo", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-url")
    parser.add_argument("--model-file")
    parser.add_argument("--demo-file")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit(f"refusing to overwrite receipt: {args.output}")
    try:
        manifest = build_manifest(
            model_repo=args.model_repo,
            space_repo=args.space_repo,
            checkpoint=args.checkpoint,
            dataset_url=args.dataset_url,
            model_file=args.model_file,
            demo_file=args.demo_file,
        )
    except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise SystemExit(f"public release audit failed closed: {error}") from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
