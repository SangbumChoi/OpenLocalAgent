"""Verify and stage the generated WebGPU demo bundle without hiding missing artifacts.

The static Space source intentionally keeps large ONNX files out of git.  This module verifies a
generated export before it is copied beside the HTML app, and emits a receipt that can be reviewed
or attached to a deployment.  It never creates a synthetic manifest: a missing or stale bundle is
an explicit failure.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
KIND = "localagent_webgpu_demo_deploy_verification"

STATIC_FILES: tuple[str, ...] = ("index.html", "app.js", "style.css", "tokenizer.js")
BUNDLE_FILES: tuple[str, ...] = (
    "action_model.fp16.onnx",
    "action_model.onnx",
    "dispatch_heads.json",
    "heads.json",
    "meta.json",
    "model.fp16.onnx",
    "model.onnx",
    "tokenizer.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_label(path: Path) -> str:
    """Keep receipts portable when paths are inside the current repository."""

    try:
        return str(path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file() or path.is_symlink():
        return None, "missing_or_non_regular_file"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, "invalid_json"
    if not isinstance(value, Mapping):
        return None, "json_root_must_be_object"
    return dict(value), None


def _bundle_identity(entries: list[dict[str, Any]]) -> str:
    """Hash the verified artifact identities, independent of filesystem paths."""

    payload = "\n".join(
        f"{entry['file']}\0{entry['bytes']}\0{entry['sha256']}"
        for entry in sorted(entries, key=lambda item: str(item["file"]))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _artifact_entry(manifest: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None
    matches = [value for value in artifacts.values() if isinstance(value, Mapping) and value.get("file") == name]
    return matches[0] if len(matches) == 1 else None


def verify_demo_deploy(
    demo_dir: str | Path,
    *,
    bundle_dir: str | Path | None = None,
    require_target_bundle: bool = True,
    expected_checkpoint_sha256: str | None = None,
    expected_tool_count: int | None = None,
) -> dict[str, Any]:
    """Verify app files and a generated bundle, returning a JSON-compatible receipt."""

    demo = Path(demo_dir).resolve()
    bundle = Path(bundle_dir).resolve() if bundle_dir is not None else demo
    blockers: list[str] = []
    static = []
    for name in STATIC_FILES:
        path = demo / name
        exists = path.is_file() and not path.is_symlink()
        static.append({"file": name, "present": exists})
        if not exists:
            blockers.append(f"missing_static:{name}")

    target_bundle_files = ("bundle-manifest.json", *BUNDLE_FILES)
    target_bundle_present = all((demo / name).is_file() and not (demo / name).is_symlink() for name in target_bundle_files)
    if require_target_bundle and bundle != demo and not target_bundle_present:
        blockers.append("deployment_target:bundle_not_staged")

    manifest_path = bundle / "bundle-manifest.json"
    manifest, manifest_error = _read_json(manifest_path)
    manifest_sha = _sha256(manifest_path) if manifest is not None else None
    if manifest is None:
        blockers.append(f"manifest:{manifest_error or 'unreadable'}")
        return {
            "kind": KIND,
            "schema_version": SCHEMA_VERSION,
            "verified": False,
            "demo_dir": _path_label(demo),
            "bundle_dir": _path_label(bundle),
            "deployment_bundle_present": target_bundle_present,
            "static": static,
            "manifest": {"path": _path_label(manifest_path), "sha256": manifest_sha},
            "artifacts": [],
            "blockers": sorted(set(blockers)),
            "claim_boundary": (
                "A clean Space is deployable only when every generated artifact is present and its "
                "exporter manifest, byte count, hash, and parity gate verify; local synthetic or "
                "text-first receipts do not establish native WebGPU deployment."
            ),
        }

    if not isinstance(manifest.get("schema_version"), int) or manifest["schema_version"] < 3:
        blockers.append("manifest:schema_version_lt_3")
    if expected_checkpoint_sha256 is not None and manifest.get("checkpoint_sha256") != expected_checkpoint_sha256:
        blockers.append("manifest:checkpoint_sha256_mismatch")
    parity = manifest.get("parity_gate")
    if not isinstance(parity, Mapping) or parity.get("hard_gate") is not True or parity.get("passed") is not True:
        blockers.append("manifest:parity_gate_not_passed")

    artifacts: list[dict[str, Any]] = []
    for name in BUNDLE_FILES:
        expected = _artifact_entry(manifest, name)
        path = bundle / name
        present = path.is_file() and not path.is_symlink()
        actual_bytes = path.stat().st_size if present else None
        actual_sha = _sha256(path) if present else None
        entry = {
            "file": name,
            "path": _path_label(path),
            "present": present,
            "bytes": actual_bytes,
            "sha256": actual_sha,
            "manifest_bytes": expected.get("bytes") if expected else None,
            "manifest_sha256": expected.get("sha256") if expected else None,
            "verified": False,
        }
        if expected is None:
            blockers.append(f"artifact:{name}:not_bound_by_manifest")
        else:
            manifest_file = expected.get("file")
            if manifest_file != name or not isinstance(expected.get("bytes"), int) or not isinstance(expected.get("sha256"), str):
                blockers.append(f"artifact:{name}:invalid_manifest_identity")
            if not present:
                blockers.append(f"artifact:{name}:missing")
            elif actual_bytes != expected.get("bytes") or actual_sha != expected.get("sha256"):
                blockers.append(f"artifact:{name}:identity_mismatch")
            else:
                entry["verified"] = True
        artifacts.append(entry)

    meta, meta_error = _read_json(bundle / "meta.json")
    if meta is None:
        blockers.append(f"meta:{meta_error or 'unreadable'}")
    elif meta.get("action_model_file") != "action_model.fp16.onnx":
        blockers.append("meta:action_model_file_mismatch")
    if expected_tool_count is not None:
        tool_names = meta.get("tools") if isinstance(meta, Mapping) else None
        if not isinstance(tool_names, list) or len(tool_names) != expected_tool_count:
            actual_count = len(tool_names) if isinstance(tool_names, list) else None
            blockers.append(
                f"meta:tool_count_mismatch:expected={expected_tool_count}:actual={actual_count}"
            )

    if not (demo / "app.js").is_file():
        blockers.append("app:missing_app_js")
    else:
        try:
            app_text = (demo / "app.js").read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            blockers.append("app:unreadable_app_js")
        else:
            if "bundle-manifest.json" not in app_text:
                blockers.append("app:does_not_load_bundle_manifest")

    verified_artifacts = [entry for entry in artifacts if entry["verified"]]
    return {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "verified": not blockers and len(verified_artifacts) == len(BUNDLE_FILES),
        "demo_dir": _path_label(demo),
        "bundle_dir": _path_label(bundle),
        "deployment_bundle_present": target_bundle_present,
        "static": static,
        "manifest": {
            "path": _path_label(manifest_path),
            "sha256": manifest_sha,
            "schema_version": manifest.get("schema_version"),
            "parity_gate_passed": parity.get("passed") is True if isinstance(parity, Mapping) else False,
        },
        "artifacts": artifacts,
        "bundle_identity_sha256": _bundle_identity(verified_artifacts) if verified_artifacts else None,
        "blockers": sorted(set(blockers)),
        "claim_boundary": (
            "A clean Space is deployable only when every generated artifact is present and its "
            "exporter manifest, byte count, hash, and parity gate verify; local synthetic or "
            "text-first receipts do not establish native WebGPU deployment."
        ),
    }


def sync_demo_bundle(
    source_dir: str | Path,
    demo_dir: str | Path,
    *,
    expected_checkpoint_sha256: str | None = None,
    expected_tool_count: int | None = None,
) -> dict[str, Any]:
    """Copy only verified generated bundle files into a static demo directory."""

    source = Path(source_dir).resolve()
    target = Path(demo_dir).resolve()
    source_report = verify_demo_deploy(
        target,
        bundle_dir=source,
        require_target_bundle=False,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_tool_count=expected_tool_count,
    )
    if not source_report["verified"]:
        raise ValueError("refusing to sync an unverified bundle: " + ", ".join(source_report["blockers"]))
    target.mkdir(parents=True, exist_ok=True)
    for name in ("bundle-manifest.json", *BUNDLE_FILES):
        shutil.copy2(source / name, target / name)
    return verify_demo_deploy(
        target,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_tool_count=expected_tool_count,
    )


def write_demo_deploy_receipt(report: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite demo deployment receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
