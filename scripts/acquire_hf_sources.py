#!/usr/bin/env python3
"""Download approved public Hugging Face sources with revision and file hashes.

Only rows in ``configs/experiments/hf-sources.v1.yaml`` are eligible. The script refuses unknown
source IDs and refuses evaluation-only policy values. Anonymous public downloads need no login; a
Hub-gated source (currently xLAM) needs the user's own ``hf auth login``/``HF_TOKEN``. The resulting
manifest is the only artifact training code should consume, and it records the resolved Hub commit
plus every downloaded file hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Mapping

import yaml


CONFIG = Path("configs/experiments/hf-sources.v1.yaml")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def load_sources(path: str | Path = CONFIG) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("kind") != "localagent_hf_campaign_sources":
        raise ValueError("invalid HF source config kind")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported HF source config version")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("HF source config must contain sources")
    seen: set[str] = set()
    for row in sources:
        if not isinstance(row, Mapping):
            raise ValueError("each HF source must be a mapping")
        source_id = row.get("id")
        if not isinstance(source_id, str) or not source_id or source_id in seen:
            raise ValueError(f"invalid or duplicate HF source id: {source_id!r}")
        seen.add(source_id)
        if row.get("repo_type") not in {"dataset", "model"}:
            raise ValueError(f"unsupported repo_type for {source_id}")
        if row.get("policy") not in {"train", "model_reference"}:
            raise ValueError(f"source {source_id} is not acquisition-approved")
        if row.get("access", "public") not in {"public", "gated"}:
            raise ValueError(f"source {source_id} has unsupported access policy")
        if not isinstance(row.get("allow_patterns"), list) or not row["allow_patterns"]:
            raise ValueError(f"source {source_id} needs allow_patterns")
    return dict(payload)


def _hub_info(api: Any, source: Mapping[str, Any]) -> dict[str, Any]:
    if source["repo_type"] == "dataset":
        info = api.dataset_info(source["repo_id"], revision=source["revision"])
    else:
        info = api.model_info(source["repo_id"], revision=source["revision"])
    return {
        "sha": getattr(info, "sha", None),
        "id": getattr(info, "id", source["repo_id"]),
        "private": bool(getattr(info, "private", False)),
        "gated": bool(getattr(info, "gated", False)),
    }


def acquire(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    source_ids: list[str] | None = None,
    dry_run: bool = False,
    allow_patterns: list[str] | None = None,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    config = load_sources(config_file)
    wanted = set(source_ids or [row["id"] for row in config["sources"]])
    rows = {row["id"]: row for row in config["sources"]}
    unknown = sorted(wanted - set(rows))
    if unknown:
        raise ValueError("unknown HF source id(s): " + ", ".join(unknown))
    out = Path(output_dir).resolve()
    if not dry_run:
        out.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as error:  # pragma: no cover - optional runtime
        raise RuntimeError('install HF support with: pip install -e ".[data,hub]"') from error
    api = HfApi()
    results: list[dict[str, Any]] = []
    for source_id in sorted(wanted):
        source = rows[source_id]
        info = _hub_info(api, source)
        result: dict[str, Any] = {
            "id": source_id,
            "repo_id": source["repo_id"],
            "repo_type": source["repo_type"],
            "requested_revision": source["revision"],
            "resolved_revision": info["sha"],
            "policy": source["policy"],
            "access": source.get("access", "public"),
            "license": source["license"],
            "allow_patterns": allow_patterns or source["allow_patterns"],
            "hub": info,
            "purpose": source["purpose"],
        }
        if dry_run:
            result["status"] = "planned"
            results.append(result)
            continue
        destination = out / source_id
        local_root = Path(
            snapshot_download(
                repo_id=source["repo_id"],
                repo_type=source["repo_type"],
                revision=source["revision"],
                allow_patterns=allow_patterns or source["allow_patterns"],
                local_dir=str(destination),
            )
        )
        files = [
            _identity(path)
            for path in sorted(local_root.rglob("*"))
            if path.is_file() and ".cache" not in path.parts
        ]
        if not files:
            raise RuntimeError(f"HF source {source_id} downloaded no files")
        result.update({"status": "downloaded", "local_dir": str(local_root), "files": files})
        results.append(result)
    payload: dict[str, Any] = {
        "kind": "localagent_hf_source_acquisition",
        "schema_version": 1,
        "config": _identity(config_file),
        "output_dir": str(out),
        "dry_run": dry_run,
        "runtime": {"python": platform.python_version()},
        "sources": results,
        "claim_boundary": (
            "Approved Hugging Face source acquisition only for explicitly approved training/model rows. "
            "Evaluation task prompts, verifiers, credentials, and runtime images are not acquired. "
            "A downloaded source is not evidence of model quality or benchmark success."
        ),
    }
    payload["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if not dry_run:
        manifest = out / "acquisition-manifest.json"
        manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--out", type=Path, default=Path("data/hf-campaign"))
    parser.add_argument("--source", action="append", dest="source_ids")
    parser.add_argument("--allow-pattern", action="append", dest="allow_patterns")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        report = acquire(
            config_path=args.config,
            output_dir=args.out,
            source_ids=args.source_ids,
            dry_run=args.dry_run,
            allow_patterns=args.allow_patterns,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
