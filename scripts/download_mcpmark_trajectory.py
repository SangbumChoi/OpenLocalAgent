#!/usr/bin/env python3
"""Download a deterministic, provenance-bound MCPMark trajectory slice.

The public trajectory archive is large and contains tool outputs that may include third-party
documents. This command downloads only a small, surface-balanced selection of messages.json
files. It writes a manifest with the Hub revision, selected source paths, byte counts, and
SHA-256 identities; content normalization remains a separate explicit step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


DATASET = "Jakumetsu/mcpmark-trajectory-log"
DATASET_URL = "https://huggingface.co/datasets/Jakumetsu/mcpmark-trajectory-log"
DEFAULT_REVISION = "e50578f0ab904d8e6a7c576c387c1e76ae482c89"
SURFACES = ("filesystem", "notion", "github", "playwright", "postgres")


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": size, "sha256": digest.hexdigest()}


def select_paths(
    files: Sequence[str], *, train_per_surface: int, eval_per_surface: int
) -> dict[str, list[str]]:
    """Select sorted message files without downloading or inspecting trajectory content."""

    if train_per_surface < 1 or eval_per_surface < 1:
        raise ValueError("train_per_surface and eval_per_surface must be positive")
    selected: dict[str, list[str]] = {"train": [], "eval": []}
    for surface in SURFACES:
        candidates = sorted(
            path
            for path in files
            if path.endswith("/messages.json") and f"__{surface}/" in path
        )
        required = train_per_surface + eval_per_surface
        if len(candidates) < required:
            raise ValueError(
                f"MCPMark has only {len(candidates)} {surface} trajectories; {required} required"
            )
        selected["train"].extend(candidates[:train_per_surface])
        selected["eval"].extend(candidates[train_per_surface:required])
    return selected


def download(
    output: Path,
    *,
    revision: str = DEFAULT_REVISION,
    train_per_surface: int = 2,
    eval_per_surface: int = 1,
) -> dict[str, Any]:
    """Download and manifest a balanced public slice; refuse to overwrite an existing directory."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as error:  # pragma: no cover - dependency is optional in unit tests.
        raise RuntimeError("huggingface_hub is required for MCPMark acquisition") from error

    api = HfApi()
    files = api.list_repo_files(DATASET, repo_type="dataset", revision=revision)
    selected = select_paths(
        files, train_per_surface=train_per_surface, eval_per_surface=eval_per_surface
    )
    output.mkdir(parents=True, exist_ok=False)
    entries: list[dict[str, Any]] = []
    for split in ("train", "eval"):
        for source_path in selected[split]:
            local_path = Path(
                hf_hub_download(
                    repo_id=DATASET,
                    repo_type="dataset",
                    filename=source_path,
                    revision=revision,
                    local_dir=str(output),
                )
            )
            entries.append({"split": split, "source_path": source_path, **_identity(local_path)})
    manifest: dict[str, Any] = {
        "kind": "localagent_mcpmark_trajectory_acquisition_manifest",
        "schema_version": 1,
        "dataset": DATASET,
        "dataset_url": DATASET_URL,
        "revision": revision,
        "selection": {
            "surfaces": list(SURFACES),
            "train_per_surface": train_per_surface,
            "eval_per_surface": eval_per_surface,
            "policy": "sorted source paths; train prefix then held-out suffix per surface",
        },
        "entries": entries,
        "content_policy": (
            "Acquisition only. Prompts, arguments, assistant text, and tool outputs are not "
            "eligible for training until the redacting normalizer is run and separately audited."
        ),
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (output / "acquisition-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--train-per-surface", type=int, default=2)
    parser.add_argument("--eval-per-surface", type=int, default=1)
    args = parser.parse_args()
    report = download(
        args.output,
        revision=args.revision,
        train_per_surface=args.train_per_surface,
        eval_per_surface=args.eval_per_surface,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
