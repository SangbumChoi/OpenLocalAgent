"""Acquire auditable, size-bounded pretraining mixtures from Hugging Face datasets.

The optional :mod:`datasets` dependency is imported only when a stream is opened. Corpus policy
lives in YAML. This module validates that policy, emits a deterministic acquisition plan, checks
local storage and revision-bound license evidence before network-heavy work, and publishes an
atomic JSONL + manifest handoff to ``scripts/prepare_corpus.py``.

Downloads are resumable only at completed-source boundaries. A partially streamed source is
replayed from its immutable upstream revision; claiming row-exact continuation for an
``IterableDataset`` would be misleading.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
import re
import shutil
import stat
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from contextlib import ExitStack
from dataclasses import asdict
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from localagent.data.pretrain_corpus import CorpusDocument

_EVAL_DENYLIST_MANIFEST_KIND = "localagent_evaluation_denylist_manifest"
_EVAL_DENYLIST_MANIFEST_VERSION = 1
_MIXTURE_PLAN_KIND = "localagent_hf_mixture_acquisition_plan"
_MIXTURE_PLAN_VERSION = 1
_SOURCE_STATE_KIND = "localagent_hf_mixture_source_state"
_SOURCE_STATE_VERSION = 1
_DOWNLOAD_MANIFEST_KIND = "localagent_hf_mixture_download_manifest"
_DOWNLOAD_MANIFEST_VERSION = 3
_STORAGE_ADMISSION_KIND = "localagent_hf_mixture_storage_admission"
_STORAGE_ADMISSION_VERSION = 1
_STORAGE_ADMISSION_FILENAME = "storage_admission.json"
_MAX_STORAGE_ADMISSION_BYTES = 16 * 1024
_SUITE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_SOURCE_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*")
_DATASET_NAME = re.compile(r"[^/\s]+/[^/\s]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_RAW_GZIP_FILE = re.compile(r"file-[0-9]{12}\.json\.gz")
_RAW_STREAM_BACKEND = "hf-jsonl-gzip-v1"
_RAW_PARQUET_STREAM_BACKEND = "hf-parquet-text-v1"
_RAW_STREAM_MANIFEST_KIND = "localagent_hf_raw_jsonl_gzip_file_manifest"
_RAW_STREAM_MANIFEST_VERSION = 1
_RAW_PARQUET_MANIFEST_KIND = "localagent_hf_raw_parquet_file_manifest"
_RAW_PARQUET_MANIFEST_VERSION = 1
_RAW_STREAM_SELECTION = "sha256-seed-path-v1"
_MAX_RAW_JSONL_LINE_BYTES = 64 * 1024 * 1024
_PARQUET_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_PYARROW_VERSION = re.compile(r"[0-9]+(?:\.[0-9]+){2}(?:[A-Za-z0-9.+-]*)?")
_WEBSIGHT_PARQUET_SCHEMA = {
    "fields": [
        {
            "name": "image",
            "nullable": True,
            "type": {
                "fields": [
                    {"name": "bytes", "nullable": True, "type": "binary"},
                    {"name": "path", "nullable": True, "type": "string"},
                ],
                "kind": "struct",
            },
        },
        {"name": "text", "nullable": True, "type": "string"},
        {"name": "llm_generated_idea", "nullable": True, "type": "string"},
    ],
    "text_field": "text",
}
_LICENSE_EVIDENCE_SCOPES = {
    "dataset-distribution",
    "dataset-card-terms",
    "row-level-license-field",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _strict_json_loads(payload: str) -> Any:
    return json.loads(
        payload,
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_verified_raw_artifact(
    path: Path,
    item: Mapping[str, Any],
) -> Any:
    """Open and hash the same regular-file object that the parser will consume."""

    # Hugging Face snapshot entries are normally symlinks into its content-addressed blob cache.
    # Following that link is expected. Security comes from hashing and parsing one held-open
    # regular-file descriptor, so a later path or symlink swap cannot change the parsed object.
    flags = os.O_RDONLY
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(
            f"cannot securely open downloaded raw stream artifact: {item['path']}"
        ) from exc
    try:
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise RuntimeError(
                f"downloaded raw stream artifact is not a regular file: {item['path']}"
            )
        if opened_stat.st_size != item["bytes"]:
            raise RuntimeError(
                f"opened raw stream artifact byte-size mismatch: {item['path']}"
            )
        handle = os.fdopen(descriptor, "rb")
        descriptor = -1
        try:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != item["sha256"]:
                raise RuntimeError(
                    f"opened raw stream artifact SHA-256 mismatch: {item['path']}"
                )
            handle.seek(0)
            return handle
        except BaseException:
            handle.close()
            raise
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise


def acquisition_runtime_identity() -> dict[str, Any]:
    """Fingerprint code, lockfile, interpreter, and streaming dependencies.

    The identity is stored in completed-source state. Resuming across a different shuffle/runtime
    implementation is rejected rather than assumed deterministic.
    """

    package_versions: dict[str, str | None] = {}
    for distribution in ("datasets", "fsspec", "huggingface_hub", "pyarrow", "PyYAML"):
        try:
            package_versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            package_versions[distribution] = None
    module_path = Path(__file__)
    project_root = module_path.parents[3]
    lock_path = project_root / "uv.lock"
    pyproject_path = project_root / "pyproject.toml"
    identity: dict[str, Any] = {
        "localagent_hf_corpus_sha256": _file_sha256(module_path),
        "packages": package_versions,
        "python": {
            "implementation": sys.implementation.name,
            "version": ".".join(str(part) for part in sys.version_info[:3]),
        },
        "pyproject_sha256": _file_sha256(pyproject_path) if pyproject_path.is_file() else None,
        "uv_lock_sha256": _file_sha256(lock_path) if lock_path.is_file() else None,
    }
    identity["runtime_sha256"] = _canonical_sha256(identity)
    return identity


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _normalize_license(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        value = value[0] if len(value) == 1 else ",".join(str(item) for item in value)
    return str(value or "unknown").strip().lower()


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return value


def _load_config(config_path: Path) -> Mapping[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("pretraining mixture config must be a mapping")
    if config.get("version", 1) != 1:
        raise ValueError("pretraining mixture config version must be 1")
    return config


def _stable_source(source: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    dataset = str(source["dataset"])
    fields = source.get("source_fields", [])
    suffix = "/".join(str(row[field]) for field in fields if row.get(field))
    return f"hf://datasets/{dataset}/{suffix}" if suffix else f"hf://datasets/{dataset}"


def normalize_evaluation_decontamination(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Validate and canonicalize a corpus config's evaluation-holdout policy.

    The corpus config, not a caller-authored denylist manifest, is the authority for which suites
    are mandatory. Individual suites may additionally pin the only accepted byte size and digest.
    """

    raw_policy = config.get("evaluation_decontamination")
    if raw_policy is None:
        return None
    if not isinstance(raw_policy, Mapping):
        raise ValueError("evaluation_decontamination must be a mapping")
    if raw_policy.get("manifest_kind") != _EVAL_DENYLIST_MANIFEST_KIND:
        raise ValueError(
            "evaluation_decontamination.manifest_kind must be "
            f"{_EVAL_DENYLIST_MANIFEST_KIND!r}"
        )
    if raw_policy.get("manifest_schema_version") != _EVAL_DENYLIST_MANIFEST_VERSION:
        raise ValueError(
            "evaluation_decontamination.manifest_schema_version must be "
            f"{_EVAL_DENYLIST_MANIFEST_VERSION}"
        )
    raw_suites = raw_policy.get("required_suites")
    if not isinstance(raw_suites, list) or not raw_suites:
        raise ValueError("evaluation_decontamination.required_suites must be a non-empty list")

    suites: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw_suite in enumerate(raw_suites):
        if isinstance(raw_suite, str):
            suite: dict[str, Any] = {"name": raw_suite}
        elif isinstance(raw_suite, Mapping):
            suite = dict(raw_suite)
        else:
            raise ValueError(
                f"evaluation_decontamination.required_suites[{index}] must be a name or mapping"
            )
        name = suite.get("name")
        if not isinstance(name, str) or _SUITE_NAME.fullmatch(name) is None:
            raise ValueError(
                f"evaluation_decontamination.required_suites[{index}].name is invalid"
            )
        if name in names:
            raise ValueError(f"duplicate required evaluation suite name {name!r}")
        names.add(name)
        normalized: dict[str, Any] = {"name": name}
        if "bytes" in suite:
            expected_bytes = suite["bytes"]
            if (
                isinstance(expected_bytes, bool)
                or not isinstance(expected_bytes, int)
                or expected_bytes < 0
            ):
                raise ValueError(f"required evaluation suite {name!r} has invalid bytes")
            normalized["bytes"] = expected_bytes
        if "sha256" in suite:
            expected_sha256 = suite["sha256"]
            if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
                raise ValueError(f"required evaluation suite {name!r} has invalid sha256")
            normalized["sha256"] = expected_sha256
        if ("bytes" in normalized) != ("sha256" in normalized):
            raise ValueError(
                f"required evaluation suite {name!r} must pin both bytes and sha256, or neither"
            )
        suites.append(normalized)
    return {
        "manifest_kind": _EVAL_DENYLIST_MANIFEST_KIND,
        "manifest_schema_version": _EVAL_DENYLIST_MANIFEST_VERSION,
        "required_suites": sorted(suites, key=lambda suite: suite["name"]),
    }


def _normalized_evidence(
    source: Mapping[str, Any],
    *,
    required: bool,
) -> dict[str, Any] | None:
    raw = source.get("license_evidence")
    source_name = source.get("name", source.get("dataset"))
    if raw is None:
        if required:
            raise ValueError(f"source {source_name!r} must pin license_evidence")
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"source {source_name!r} license_evidence must be a mapping")
    evidence_id = raw.get("id")
    url = raw.get("url")
    expected_bytes = raw.get("bytes")
    expected_sha256 = raw.get("sha256")
    scope = raw.get("scope")
    if not isinstance(evidence_id, str) or _SOURCE_NAME.fullmatch(evidence_id) is None:
        raise ValueError(f"source {source_name!r} license_evidence.id is invalid")
    if not isinstance(url, str):
        raise ValueError(f"source {source_name!r} license_evidence.url is invalid")
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc != "huggingface.co":
        raise ValueError(f"source {source_name!r} license_evidence.url must use HTTPS on huggingface.co")
    dataset = str(source["dataset"])
    revision = str(source["revision"])
    expected_prefix = f"/datasets/{dataset}/resolve/{revision}/"
    if not parsed.path.startswith(expected_prefix):
        raise ValueError(
            f"source {source_name!r} license_evidence.url must bind dataset and revision"
        )
    expected_bytes = _positive_int(
        expected_bytes,
        f"source {source_name!r} license_evidence.bytes",
    )
    if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError(f"source {source_name!r} license_evidence.sha256 is invalid")
    if scope not in _LICENSE_EVIDENCE_SCOPES:
        raise ValueError(
            f"source {source_name!r} license_evidence.scope must be one of "
            f"{sorted(_LICENSE_EVIDENCE_SCOPES)}"
        )
    return {
        "bytes": expected_bytes,
        "id": evidence_id,
        "scope": scope,
        "sha256": expected_sha256,
        "url": url,
    }


def _normalized_raw_stream(
    raw_stream: Any,
    *,
    config_dir: Path,
    dataset: str,
    revision: str,
    split: str,
    source_name: str,
    subset: str | None,
    text_field: str,
) -> dict[str, Any] | None:
    if raw_stream is None:
        return None
    if not isinstance(raw_stream, Mapping):
        raise ValueError(f"source {source_name!r} raw_stream must be a mapping")
    expected_keys = {
        "backend",
        "files_manifest",
        "files_manifest_bytes",
        "files_manifest_sha256",
        "interleave_files",
        "selection",
    }
    if set(raw_stream) != expected_keys:
        raise ValueError(
            f"source {source_name!r} raw_stream keys must be exactly "
            f"{sorted(expected_keys)}"
        )
    backend = raw_stream["backend"]
    supported_backends = {_RAW_STREAM_BACKEND, _RAW_PARQUET_STREAM_BACKEND}
    if backend not in supported_backends:
        raise ValueError(
            f"source {source_name!r} raw_stream.backend must be one of "
            f"{sorted(supported_backends)}"
        )
    if raw_stream["selection"] != _RAW_STREAM_SELECTION:
        raise ValueError(
            f"source {source_name!r} raw_stream.selection must be "
            f"{_RAW_STREAM_SELECTION!r}"
        )

    relative_manifest = raw_stream["files_manifest"]
    if not isinstance(relative_manifest, str) or not relative_manifest:
        raise ValueError(
            f"source {source_name!r} raw_stream.files_manifest must be a relative path"
        )
    relative_path = Path(relative_manifest)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(
            f"source {source_name!r} raw_stream.files_manifest must stay under the config directory"
        )
    manifest_path = config_dir / relative_path
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(
            f"source {source_name!r} raw stream manifest must be a regular non-symlink file"
        )
    try:
        manifest_path.resolve().relative_to(config_dir.resolve())
    except ValueError as exc:
        raise ValueError(
            f"source {source_name!r} raw stream manifest escapes the config directory"
        ) from exc

    expected_bytes = _positive_int(
        raw_stream["files_manifest_bytes"],
        f"source {source_name!r} raw_stream.files_manifest_bytes",
    )
    expected_sha256 = raw_stream["files_manifest_sha256"]
    if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError(
            f"source {source_name!r} raw_stream.files_manifest_sha256 is invalid"
        )
    payload = manifest_path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if len(payload) != expected_bytes:
        raise ValueError(f"source {source_name!r} raw stream manifest byte-size mismatch")
    if actual_sha256 != expected_sha256:
        raise ValueError(f"source {source_name!r} raw stream manifest SHA-256 mismatch")
    try:
        manifest = _strict_json_loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"source {source_name!r} raw stream manifest is invalid JSON") from exc

    if not isinstance(manifest, dict):
        raise ValueError(f"source {source_name!r} raw stream manifest must be an object")
    if backend == _RAW_STREAM_BACKEND:
        manifest_keys = {
            "api_url",
            "dataset",
            "files",
            "kind",
            "manifest_sha256",
            "revision",
            "version",
        }
        if set(manifest) != manifest_keys:
            raise ValueError(
                f"source {source_name!r} raw stream manifest keys must be exactly "
                f"{sorted(manifest_keys)}"
            )
        if (
            manifest["kind"] != _RAW_STREAM_MANIFEST_KIND
            or manifest["version"] != _RAW_STREAM_MANIFEST_VERSION
        ):
            raise ValueError(f"source {source_name!r} raw stream manifest kind/version mismatch")
        expected_api_url = (
            f"https://huggingface.co/api/datasets/{dataset}/tree/{revision}"
            "?recursive=true&expand=true"
        )
    else:
        manifest_keys = {
            "api_url",
            "dataset",
            "files",
            "kind",
            "manifest_sha256",
            "parquet_schema",
            "reader_runtime",
            "revision",
            "shard_count",
            "split",
            "subset",
            "total_bytes",
            "version",
        }
        if set(manifest) != manifest_keys:
            raise ValueError(
                f"source {source_name!r} raw stream manifest keys must be exactly "
                f"{sorted(manifest_keys)}"
            )
        if (
            manifest["kind"] != _RAW_PARQUET_MANIFEST_KIND
            or manifest["version"] != _RAW_PARQUET_MANIFEST_VERSION
        ):
            raise ValueError(f"source {source_name!r} raw stream manifest kind/version mismatch")
        canonical_payload = (
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        if payload != canonical_payload:
            raise ValueError(
                f"source {source_name!r} raw parquet manifest must be canonical pretty JSON"
            )
        if (
            not isinstance(subset, str)
            or _PARQUET_LABEL.fullmatch(subset) is None
            or manifest["subset"] != subset
        ):
            raise ValueError(f"source {source_name!r} raw parquet manifest subset mismatch")
        if (
            _PARQUET_LABEL.fullmatch(split) is None
            or manifest["split"] != split
        ):
            raise ValueError(f"source {source_name!r} raw parquet manifest split mismatch")
        if manifest["parquet_schema"] != _WEBSIGHT_PARQUET_SCHEMA:
            raise ValueError(f"source {source_name!r} raw parquet schema contract mismatch")
        if text_field != _WEBSIGHT_PARQUET_SCHEMA["text_field"]:
            raise ValueError(
                f"source {source_name!r} text_field must match its raw parquet projection"
            )
        reader_runtime = manifest["reader_runtime"]
        if (
            not isinstance(reader_runtime, dict)
            or set(reader_runtime) != {"library", "version"}
            or reader_runtime["library"] != "pyarrow"
            or not isinstance(reader_runtime["version"], str)
            or _PYARROW_VERSION.fullmatch(reader_runtime["version"]) is None
        ):
            raise ValueError(f"source {source_name!r} raw parquet reader runtime is invalid")
        expected_api_url = (
            f"https://huggingface.co/api/datasets/{dataset}/tree/{revision}/{subset}"
            "?recursive=true&expand=true"
        )

    if manifest["dataset"] != dataset or manifest["revision"] != revision:
        raise ValueError(f"source {source_name!r} raw stream manifest source mismatch")
    if manifest["api_url"] != expected_api_url:
        raise ValueError(f"source {source_name!r} raw stream manifest API URL mismatch")
    unsigned_manifest = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    manifest_self_hash = manifest["manifest_sha256"]
    if (
        not isinstance(manifest_self_hash, str)
        or _SHA256.fullmatch(manifest_self_hash) is None
        or manifest_self_hash != _canonical_sha256(unsigned_manifest)
    ):
        raise ValueError(f"source {source_name!r} raw stream manifest self-hash mismatch")

    raw_files = manifest["files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError(f"source {source_name!r} raw stream manifest needs files")
    if backend == _RAW_PARQUET_STREAM_BACKEND:
        shard_count = _positive_int(
            manifest["shard_count"],
            f"source {source_name!r} raw parquet manifest shard_count",
        )
        if shard_count > 99_999:
            raise ValueError(
                f"source {source_name!r} raw parquet manifest shard_count is too large"
            )
        if len(raw_files) != shard_count:
            raise ValueError(
                f"source {source_name!r} raw parquet inventory does not match shard_count"
            )
        parquet_file = re.compile(
            rf"{re.escape(str(subset))}/{re.escape(split)}-([0-9]{{5}})-of-"
            rf"{shard_count:05d}-[0-9a-f]{{16}}\.parquet"
        )
    else:
        shard_count = None
        parquet_file = None

    files: list[dict[str, Any]] = []
    paths: list[str] = []
    shard_indexes: list[int] = []
    for index, raw_file in enumerate(raw_files):
        if not isinstance(raw_file, dict) or set(raw_file) != {"bytes", "path", "sha256"}:
            raise ValueError(
                f"source {source_name!r} raw stream file {index} has invalid keys"
            )
        file_path = raw_file["path"]
        file_bytes = raw_file["bytes"]
        file_sha256 = raw_file["sha256"]
        if backend == _RAW_STREAM_BACKEND:
            valid_path = (
                isinstance(file_path, str)
                and _RAW_GZIP_FILE.fullmatch(file_path) is not None
            )
        else:
            match = (
                parquet_file.fullmatch(file_path)
                if isinstance(file_path, str) and parquet_file is not None
                else None
            )
            valid_path = match is not None
            if match is not None:
                shard_indexes.append(int(match.group(1)))
        if not valid_path:
            raise ValueError(
                f"source {source_name!r} raw stream file {index} has invalid path"
            )
        if isinstance(file_bytes, bool) or not isinstance(file_bytes, int) or file_bytes <= 0:
            raise ValueError(
                f"source {source_name!r} raw stream file {file_path!r} has invalid bytes"
            )
        if not isinstance(file_sha256, str) or _SHA256.fullmatch(file_sha256) is None:
            raise ValueError(
                f"source {source_name!r} raw stream file {file_path!r} has invalid SHA-256"
            )
        paths.append(file_path)
        files.append(
            {
                "bytes": file_bytes,
                "path": file_path,
                "sha256": file_sha256,
            }
        )
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError(
            f"source {source_name!r} raw stream manifest files must be unique and sorted"
        )
    total_artifact_bytes = sum(item["bytes"] for item in files)
    if backend == _RAW_PARQUET_STREAM_BACKEND:
        if shard_indexes != list(range(int(shard_count))):
            raise ValueError(
                f"source {source_name!r} raw parquet shard indexes must be contiguous"
            )
        if len({item["sha256"] for item in files}) != len(files):
            raise ValueError(
                f"source {source_name!r} raw parquet artifact hashes must be unique"
            )
        if (
            isinstance(manifest["total_bytes"], bool)
            or not isinstance(manifest["total_bytes"], int)
            or manifest["total_bytes"] <= 0
            or manifest["total_bytes"] != total_artifact_bytes
        ):
            raise ValueError(
                f"source {source_name!r} raw parquet inventory total_bytes mismatch"
            )
    interleave_files = _positive_int(
        raw_stream["interleave_files"],
        f"source {source_name!r} raw_stream.interleave_files",
    )
    if interleave_files > len(files):
        raise ValueError(
            f"source {source_name!r} raw_stream.interleave_files exceeds its inventory"
        )
    inventory = {
        "api_url": manifest["api_url"],
        "bytes": len(payload),
        "files": files,
        "manifest_sha256": manifest_self_hash,
        "sha256": actual_sha256,
    }
    if backend == _RAW_STREAM_BACKEND:
        inventory["total_compressed_bytes"] = total_artifact_bytes
    else:
        inventory["shard_count"] = int(shard_count)
        inventory["total_artifact_bytes"] = total_artifact_bytes
    normalized = {
        "backend": backend,
        "file_inventory": {
            **inventory,
        },
        "interleave_files": interleave_files,
        "selection": _RAW_STREAM_SELECTION,
    }
    if backend == _RAW_PARQUET_STREAM_BACKEND:
        normalized.update(
            {
                "parquet_schema": manifest["parquet_schema"],
                "reader_runtime": manifest["reader_runtime"],
                "text_field": text_field,
            }
        )
    return normalized


def _normalized_source(
    source: Mapping[str, Any],
    *,
    config_dir: Path,
    index: int,
    require_license_evidence: bool,
) -> dict[str, Any]:
    name = source.get("name")
    if not isinstance(name, str) or _SOURCE_NAME.fullmatch(name) is None:
        raise ValueError(f"sources[{index}].name must match {_SOURCE_NAME.pattern!r}")
    dataset = source.get("dataset")
    if not isinstance(dataset, str) or _DATASET_NAME.fullmatch(dataset) is None:
        raise ValueError(f"source {name!r} dataset must be an owner/name identifier")
    revision = source.get("revision")
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ValueError(
            f"source {name!r} must pin a full 40-character lowercase commit revision"
        )
    split = source.get("split", "train")
    text_field = source.get("text_field", "text")
    subset = source.get("subset")
    if not isinstance(split, str) or not split:
        raise ValueError(f"source {name!r} split must be a non-empty string")
    if not isinstance(text_field, str) or not text_field:
        raise ValueError(f"source {name!r} text_field must be a non-empty string")
    if subset is not None and (not isinstance(subset, str) or not subset):
        raise ValueError(f"source {name!r} subset must be a non-empty string or null")
    source_fields = source.get("source_fields", [])
    if (
        not isinstance(source_fields, list)
        or any(not isinstance(field, str) or not field for field in source_fields)
        or len(set(source_fields)) != len(source_fields)
    ):
        raise ValueError(f"source {name!r} source_fields must be unique non-empty strings")
    shuffle_buffer = _positive_int(
        source.get("shuffle_buffer", 10_000),
        f"source {name!r} shuffle_buffer",
    )

    fixed_license = source.get("license")
    license_field = source.get("license_field")
    if (fixed_license is None) == (license_field is None):
        raise ValueError(
            f"source {name!r} must declare exactly one of license or license_field"
        )
    allowed_licenses: list[str] = []
    if license_field is not None:
        if not isinstance(license_field, str) or not license_field:
            raise ValueError(f"source {name!r} license_field must be a non-empty string")
        raw_allowed = source.get("allowed_licenses")
        if not isinstance(raw_allowed, list) or not raw_allowed:
            raise ValueError(
                f"source {name!r} using license_field must declare allowed_licenses"
            )
        allowed_licenses = sorted({_normalize_license(value) for value in raw_allowed})
        if "unknown" in allowed_licenses:
            raise ValueError(f"source {name!r} cannot allow the unknown license")
        if len(allowed_licenses) != len(raw_allowed):
            raise ValueError(f"source {name!r} allowed_licenses must be unique after normalization")
    else:
        normalized_fixed = _normalize_license(fixed_license)
        if normalized_fixed == "unknown":
            raise ValueError(f"source {name!r} fixed license must not be unknown")
        fixed_license = normalized_fixed
        if source.get("allowed_licenses") is not None:
            raise ValueError(
                f"source {name!r} with a fixed license cannot declare allowed_licenses"
            )

    try:
        weight = Decimal(str(source.get("weight", 0)))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"source {name!r} weight must be numeric") from exc
    if not weight.is_finite() or weight <= 0:
        raise ValueError(f"source {name!r} weight must be positive and finite")

    evidence = _normalized_evidence(
        source,
        required=require_license_evidence,
    )
    if evidence is not None:
        if license_field is not None and evidence["scope"] != "row-level-license-field":
            raise ValueError(
                f"source {name!r} using license_field requires row-level-license-field evidence"
            )
        if license_field is None and evidence["scope"] == "row-level-license-field":
            raise ValueError(
                f"source {name!r} with a fixed license cannot use row-level-license-field evidence"
            )

    raw_stream = _normalized_raw_stream(
        source.get("raw_stream"),
        config_dir=config_dir,
        dataset=dataset,
        revision=revision,
        split=split,
        source_name=name,
        subset=subset,
        text_field=text_field,
    )
    return {
        "allowed_licenses": allowed_licenses,
        "dataset": dataset,
        "license": fixed_license,
        "license_evidence": evidence,
        "license_field": license_field,
        "name": name,
        "raw_stream": raw_stream,
        "revision": revision,
        "shuffle_buffer": shuffle_buffer,
        "source_fields": source_fields,
        "split": split,
        "subset": subset,
        "text_field": text_field,
        "weight": format(weight, "f"),
    }


def _allocate_character_budgets(target_chars: int, weights: list[Decimal]) -> list[int]:
    if target_chars < len(weights):
        raise ValueError("target_chars must be at least the number of configured sources")
    total = sum(weights)
    quotas = [Decimal(target_chars) * weight / total for weight in weights]
    budgets = [int(quota.to_integral_value(rounding=ROUND_FLOOR)) for quota in quotas]
    remainder = target_chars - sum(budgets)
    order = sorted(
        range(len(weights)),
        key=lambda index: (-(quotas[index] - budgets[index]), index),
    )
    for index in order[:remainder]:
        budgets[index] += 1
    if sum(budgets) != target_chars or any(budget <= 0 for budget in budgets):
        raise AssertionError("internal character-budget apportionment error")
    return budgets


def build_mixture_plan(
    config_path: str | Path,
    *,
    target_chars: int | None = None,
) -> dict[str, Any]:
    """Return a deterministic, path-independent acquisition plan for a mixture config."""

    path = Path(config_path)
    config = _load_config(path)
    normalize_evaluation_decontamination(config)
    raw_sources = config.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("pretraining mixture needs at least one source")
    if any(not isinstance(source, Mapping) for source in raw_sources):
        raise ValueError("every pretraining mixture source must be a mapping")

    require_license_evidence = config.get("require_license_evidence", False)
    require_full_source_budgets = config.get("require_full_source_budgets", False)
    if not isinstance(require_license_evidence, bool):
        raise ValueError("require_license_evidence must be boolean")
    if not isinstance(require_full_source_budgets, bool):
        raise ValueError("require_full_source_budgets must be boolean")
    normalized_sources = [
        _normalized_source(
            source,
            config_dir=path.parent,
            index=index,
            require_license_evidence=require_license_evidence,
        )
        for index, source in enumerate(raw_sources)
    ]
    names = [source["name"] for source in normalized_sources]
    if len(set(names)) != len(names):
        raise ValueError("pretraining mixture source names must be unique")

    requested_chars_raw = config.get("target_chars", 0) if target_chars is None else target_chars
    requested_chars = _positive_int(requested_chars_raw, "target_chars")
    min_chars = _positive_int(config.get("min_document_chars", 200), "min_document_chars")
    max_chars = _positive_int(config.get("max_document_chars", 2_000_000), "max_document_chars")
    if max_chars < min_chars:
        raise ValueError("max_document_chars must be at least min_document_chars")
    seed = config.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    storage = config.get("storage", {})
    if not isinstance(storage, Mapping):
        raise ValueError("storage must be a mapping")
    minimum_free_bytes = _positive_int(
        storage.get("minimum_free_bytes", 0),
        "storage.minimum_free_bytes",
        allow_zero=True,
    )
    max_raw_jsonl_bytes_raw = storage.get("max_raw_jsonl_bytes")
    max_raw_jsonl_bytes = (
        None
        if max_raw_jsonl_bytes_raw is None
        else _positive_int(max_raw_jsonl_bytes_raw, "storage.max_raw_jsonl_bytes")
    )

    weights = [Decimal(source["weight"]) for source in normalized_sources]
    budgets = _allocate_character_budgets(requested_chars, weights)
    total_weight = sum(weights)
    planned_sources: list[dict[str, Any]] = []
    evidence_by_id: dict[str, dict[str, Any]] = {}
    for source, budget, weight in zip(normalized_sources, budgets, weights, strict=True):
        planned = {
            **source,
            "normalized_weight": format(weight / total_weight, "f"),
            "requested_chars": budget,
        }
        planned_sources.append(planned)
        evidence = source["license_evidence"]
        if evidence is not None:
            previous = evidence_by_id.get(evidence["id"])
            if previous is not None and previous != evidence:
                raise ValueError(f"conflicting license evidence id {evidence['id']!r}")
            evidence_by_id[evidence["id"]] = evidence

    config_bytes = path.read_bytes()
    plan: dict[str, Any] = {
        "config_bytes": len(config_bytes),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "interruption_recovery": {
            "completed_source_resume": True,
            "partial_source_resume": False,
            "partial_source_replay_required": True,
        },
        "kind": _MIXTURE_PLAN_KIND,
        "license_evidence": [evidence_by_id[key] for key in sorted(evidence_by_id)],
        "max_document_chars": max_chars,
        "min_document_chars": min_chars,
        "requested_chars": requested_chars,
        "require_full_source_budgets": require_full_source_budgets,
        "require_license_evidence": require_license_evidence,
        "seed": seed,
        "sources": planned_sources,
        "storage": {
            "max_raw_jsonl_bytes": max_raw_jsonl_bytes,
            "minimum_free_bytes": minimum_free_bytes,
        },
        "version": _MIXTURE_PLAN_VERSION,
    }
    plan["plan_sha256"] = _canonical_sha256(plan)
    return plan


def _nearest_existing_path(path: Path) -> Path:
    candidate = path.absolute()
    while not candidate.exists():
        if candidate.parent == candidate:
            raise ValueError(f"cannot find an existing parent for output path {path}")
        candidate = candidate.parent
    return candidate


def _storage_admission_payload(
    plan: Mapping[str, Any],
    disk: Mapping[str, Any],
) -> dict[str, Any]:
    admission: dict[str, Any] = {
        "available_free_bytes_at_admission": int(disk["available_free_bytes"]),
        "checked_path": str(disk["checked_path"]),
        "filesystem_device": int(disk["filesystem_device"]),
        "kind": _STORAGE_ADMISSION_KIND,
        "minimum_free_bytes": int(plan["storage"]["minimum_free_bytes"]),
        "plan_sha256": str(plan["plan_sha256"]),
        "version": _STORAGE_ADMISSION_VERSION,
    }
    if admission["available_free_bytes_at_admission"] < admission["minimum_free_bytes"]:
        raise ValueError("cannot record storage admission without the configured free-space floor")
    admission["admission_sha256"] = _canonical_sha256(admission)
    return admission


def _verified_storage_admission(
    path: Path,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"storage admission must be a regular non-symlink file: {path}")
    payload = path.read_bytes()
    if len(payload) > _MAX_STORAGE_ADMISSION_BYTES:
        raise ValueError(f"storage admission exceeds {_MAX_STORAGE_ADMISSION_BYTES} bytes")
    try:
        admission = _strict_json_loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"storage admission is invalid JSON: {path}") from exc
    expected_keys = {
        "admission_sha256",
        "available_free_bytes_at_admission",
        "checked_path",
        "filesystem_device",
        "kind",
        "minimum_free_bytes",
        "plan_sha256",
        "version",
    }
    if not isinstance(admission, dict) or set(admission) != expected_keys:
        raise ValueError(f"storage admission has invalid keys: {path}")
    unsigned = {
        key: value for key, value in admission.items() if key != "admission_sha256"
    }
    if (
        not isinstance(admission["admission_sha256"], str)
        or _SHA256.fullmatch(admission["admission_sha256"]) is None
        or admission["admission_sha256"] != _canonical_sha256(unsigned)
    ):
        raise ValueError(f"storage admission self-hash mismatch: {path}")
    if (
        admission["kind"] != _STORAGE_ADMISSION_KIND
        or admission["version"] != _STORAGE_ADMISSION_VERSION
    ):
        raise ValueError(f"storage admission kind/version mismatch: {path}")
    if admission["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError(f"storage admission plan mismatch: {path}")
    configured_minimum = int(plan["storage"]["minimum_free_bytes"])
    if admission["minimum_free_bytes"] != configured_minimum:
        raise ValueError(f"storage admission minimum-free-space mismatch: {path}")
    admitted_free = admission["available_free_bytes_at_admission"]
    if (
        isinstance(admitted_free, bool)
        or not isinstance(admitted_free, int)
        or admitted_free < configured_minimum
    ):
        raise ValueError(f"storage admission did not meet its configured floor: {path}")
    if not isinstance(admission["checked_path"], str) or not admission["checked_path"]:
        raise ValueError(f"storage admission checked path is invalid: {path}")
    filesystem_device = admission["filesystem_device"]
    if (
        isinstance(filesystem_device, bool)
        or not isinstance(filesystem_device, int)
        or filesystem_device < 0
    ):
        raise ValueError(f"storage admission filesystem device is invalid: {path}")
    return admission


def _looks_like_completed_source_for_capacity(
    *,
    plan: Mapping[str, Any],
    source: Mapping[str, Any],
    source_index: int,
    state_dir: Path,
) -> tuple[bool, int]:
    """Return a cheap, conservative completion hint used only for disk estimation.

    Full data hashing still occurs in ``_restore_completed_source`` before a source is trusted.
    A forged hint can therefore only make the subsequent resume fail before opening that source.
    """

    data_path, _, state_path = _source_paths(
        state_dir,
        source_index,
        str(source["name"]),
    )
    if data_path.is_symlink() or state_path.is_symlink():
        return False, 0
    if not data_path.is_file() or not state_path.is_file():
        return False, 0
    try:
        state = _strict_json_loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return False, 0
    if not isinstance(state, dict):
        return False, 0
    unsigned = {key: value for key, value in state.items() if key != "state_sha256"}
    expected = {
        "kind": _SOURCE_STATE_KIND,
        "version": _SOURCE_STATE_VERSION,
        "plan_sha256": plan["plan_sha256"],
        "source_index": source_index,
        "source_plan_sha256": _canonical_sha256(source),
        "data_bytes": data_path.stat().st_size,
    }
    if (
        state.get("state_sha256") != _canonical_sha256(unsigned)
        or any(state.get(key) != value for key, value in expected.items())
    ):
        return False, 0
    return True, data_path.stat().st_size


def _cached_raw_artifact_bytes(
    source: Mapping[str, Any],
    seed: int,
) -> int:
    """Count complete selected Hub-cache artifacts without opening the network."""

    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return 0
    cached_bytes = 0
    for item in _selected_raw_files(source, seed):
        cached = try_to_load_from_cache(
            repo_id=source["dataset"],
            filename=item["path"],
            repo_type="dataset",
            revision=source["revision"],
        )
        if not isinstance(cached, str):
            continue
        cached_path = Path(cached)
        try:
            opened_stat = cached_path.stat()
        except OSError:
            continue
        if stat.S_ISREG(opened_stat.st_mode) and opened_stat.st_size == item["bytes"]:
            cached_bytes += item["bytes"]
    return cached_bytes


def _resume_minimum_free_bytes(
    plan: Mapping[str, Any],
    out: Path,
) -> tuple[int, dict[str, Any]]:
    """Estimate conservative remaining peak bytes for a previously admitted run."""

    configured_minimum = int(plan["storage"]["minimum_free_bytes"])
    max_raw = plan["storage"].get("max_raw_jsonl_bytes")
    if max_raw is None:
        # Canonical document metadata can exceed source text for tiny fixtures. Production plans
        # use an explicit hard cap; this fallback is deliberately conservative.
        max_raw = max(int(plan["requested_chars"]) * 8, 1)
    max_raw = int(max_raw)
    state_dir = out / "download_state"
    completed_indexes: set[int] = set()
    completed_spool_bytes = 0
    for source_index, source in enumerate(plan["sources"]):
        completed, source_bytes = _looks_like_completed_source_for_capacity(
            plan=plan,
            source=source,
            source_index=source_index,
            state_dir=state_dir,
        )
        if completed:
            completed_indexes.add(source_index)
            completed_spool_bytes += source_bytes

    remaining_selected_bytes = 0
    remaining_stream_reserve = 0
    all_selected_bytes = 0
    all_stream_reserve = 0
    cached_selected_bytes = 0
    for source_index, source in enumerate(plan["sources"]):
        raw_stream = source.get("raw_stream")
        if raw_stream is None:
            reserve = int(source["requested_chars"]) * 2
            all_stream_reserve += reserve
            if source_index not in completed_indexes:
                remaining_stream_reserve += reserve
            continue
        selected_bytes = sum(
            int(item["bytes"])
            for item in _selected_raw_files(
                source,
                int(plan["seed"]) + source_index,
            )
        )
        all_selected_bytes += selected_bytes
        if source_index in completed_indexes:
            continue
        cached_bytes = _cached_raw_artifact_bytes(
            source,
            int(plan["seed"]) + source_index,
        )
        cached_selected_bytes += cached_bytes
        remaining_selected_bytes += max(0, selected_bytes - cached_bytes)

    initial_estimate = (
        (2 * max_raw)
        + all_selected_bytes
        + all_stream_reserve
    )
    fixed_overhead = max(0, configured_minimum - initial_estimate)
    remaining_spool_bytes = max(0, max_raw - completed_spool_bytes)
    estimated_remaining_peak = (
        max_raw
        + remaining_spool_bytes
        + remaining_selected_bytes
        + remaining_stream_reserve
        + fixed_overhead
    )
    effective_minimum = min(configured_minimum, estimated_remaining_peak)
    return effective_minimum, {
        "cached_selected_artifact_bytes": cached_selected_bytes,
        "completed_source_count": len(completed_indexes),
        "completed_spool_bytes": completed_spool_bytes,
        "estimated_remaining_peak_bytes": estimated_remaining_peak,
        "fixed_overhead_bytes": fixed_overhead,
        "remaining_selected_artifact_bytes": remaining_selected_bytes,
        "remaining_spool_bytes": remaining_spool_bytes,
        "remaining_stream_reserve_bytes": remaining_stream_reserve,
    }


def audit_mixture_readiness(
    plan: Mapping[str, Any],
    out_dir: str | Path,
    *,
    license_evidence: Mapping[str, str | Path] | None = None,
    require_stream_runtime: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Audit local inputs and disk headroom without opening an upstream dataset stream."""

    if plan.get("kind") != _MIXTURE_PLAN_KIND or plan.get("version") != _MIXTURE_PLAN_VERSION:
        raise ValueError("invalid mixture acquisition plan")
    expected_plan_hash = plan.get("plan_sha256")
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if expected_plan_hash != _canonical_sha256(unsigned):
        raise ValueError("mixture acquisition plan self-hash mismatch")
    provided = dict(license_evidence or {})
    expected_evidence = {
        item["id"]: item for item in plan.get("license_evidence", [])
    }
    unknown = sorted(set(provided) - set(expected_evidence))
    blockers: list[str] = []
    if unknown:
        blockers.append(f"unrecognized license evidence id(s): {', '.join(unknown)}")
    runtime = acquisition_runtime_identity()
    if require_stream_runtime:
        needs_datasets = any(source.get("raw_stream") is None for source in plan["sources"])
        needs_huggingface_hub = any(
            source.get("raw_stream") is not None for source in plan["sources"]
        )
        parquet_sources = [
            source
            for source in plan["sources"]
            if isinstance(source.get("raw_stream"), Mapping)
            and source["raw_stream"].get("backend") == _RAW_PARQUET_STREAM_BACKEND
        ]
        if needs_datasets and runtime["packages"]["datasets"] is None:
            blockers.append('missing datasets runtime; install with: pip install -e ".[data]"')
        if needs_huggingface_hub and runtime["packages"]["huggingface_hub"] is None:
            blockers.append(
                'missing huggingface_hub runtime; install with: pip install -e ".[data]"'
            )
        if parquet_sources and runtime["packages"]["pyarrow"] is None:
            blockers.append('missing pyarrow runtime; install with: pip install -e ".[data]"')
        elif parquet_sources:
            actual_pyarrow = runtime["packages"]["pyarrow"]
            for source in parquet_sources:
                expected_pyarrow = source["raw_stream"]["reader_runtime"]["version"]
                if actual_pyarrow != expected_pyarrow:
                    blockers.append(
                        f"source {source['name']!r} requires pyarrow {expected_pyarrow}, "
                        f"found {actual_pyarrow}"
                    )

    evidence_results: list[dict[str, Any]] = []
    for evidence_id, expected in sorted(expected_evidence.items()):
        provided_path = provided.get(evidence_id)
        if provided_path is None:
            evidence_results.append(
                {
                    **expected,
                    "path": None,
                    "status": "missing",
                }
            )
            blockers.append(f"missing license evidence {evidence_id!r}")
            continue
        path = Path(provided_path)
        if not path.is_file():
            evidence_results.append(
                {
                    **expected,
                    "path": str(path),
                    "status": "not_a_file",
                }
            )
            blockers.append(f"license evidence {evidence_id!r} is not a file")
            continue
        actual_bytes = path.stat().st_size
        actual_sha256 = _file_sha256(path)
        status = "verified"
        if actual_bytes != expected["bytes"]:
            status = "bytes_mismatch"
            blockers.append(f"license evidence {evidence_id!r} byte-size mismatch")
        elif actual_sha256 != expected["sha256"]:
            status = "sha256_mismatch"
            blockers.append(f"license evidence {evidence_id!r} SHA-256 mismatch")
        evidence_results.append(
            {
                **expected,
                "actual_bytes": actual_bytes,
                "actual_sha256": actual_sha256,
                "path": str(path),
                "status": status,
            }
        )

    out = Path(out_dir)
    disk_path = _nearest_existing_path(out)
    disk_usage = shutil.disk_usage(disk_path)
    filesystem_device = os.stat(disk_path).st_dev
    configured_minimum = int(plan["storage"]["minimum_free_bytes"])
    effective_minimum = configured_minimum
    disk_mode = "initial_admission"
    admission: dict[str, Any] | None = None
    remaining_capacity: dict[str, Any] | None = None
    admission_path = out / _STORAGE_ADMISSION_FILENAME
    completed_manifest = out / "download_manifest.json"
    if resume and completed_manifest.is_file():
        # A completed artifact is verified byte-for-byte by ``stream_mixture``. Rechecking the
        # original acquisition headroom would make verification fail after successful storage use.
        effective_minimum = 0
        disk_mode = "completed_verification"
    elif resume and admission_path.exists():
        try:
            admission = _verified_storage_admission(admission_path, plan)
        except ValueError as exc:
            blockers.append(str(exc))
            disk_mode = "invalid_resume_admission"
        else:
            if admission["filesystem_device"] != filesystem_device:
                blockers.append(
                    "storage admission filesystem device disagrees with the resume output"
                )
                disk_mode = "invalid_resume_admission"
            else:
                # The original floor is a one-time admission threshold. Resume still requires a
                # conservative current-space estimate for remaining cache, source spool, and final
                # assembly work, while crediting complete selected Hub artifacts already cached.
                effective_minimum, remaining_capacity = _resume_minimum_free_bytes(
                    plan,
                    out,
                )
                disk_mode = "resume_admission"
    disk_ready = disk_usage.free >= effective_minimum and disk_mode != "invalid_resume_admission"
    if not disk_ready:
        if disk_mode != "invalid_resume_admission":
            blockers.append(
                f"insufficient free disk: need {effective_minimum} bytes, "
                f"found {disk_usage.free} bytes"
            )
    return {
        "blockers": blockers,
        "disk": {
            "admission": admission,
            "available_free_bytes": disk_usage.free,
            "checked_path": str(disk_path),
            "configured_minimum_free_bytes": configured_minimum,
            "filesystem_device": filesystem_device,
            "minimum_free_bytes": effective_minimum,
            "mode": disk_mode,
            "remaining_capacity": remaining_capacity,
            "ready": disk_ready,
        },
        "kind": "localagent_hf_mixture_readiness",
        "license_evidence": evidence_results,
        "plan_sha256": expected_plan_hash,
        "ready": not blockers,
        "runtime": runtime,
        "version": 1,
    }


def _selected_raw_files(
    source: Mapping[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    raw_stream = source.get("raw_stream")
    if not isinstance(raw_stream, Mapping):
        return []
    files = raw_stream["file_inventory"]["files"]
    selected: list[dict[str, Any]] = []
    for item in files:
        selection_sha256 = hashlib.sha256(
            f"{seed}:{item['path']}".encode("utf-8")
        ).hexdigest()
        selected.append({**item, "selection_sha256": selection_sha256})
    selected.sort(key=lambda item: (item["selection_sha256"], item["path"]))
    return selected[: int(raw_stream["interleave_files"])]


def _bounded_shuffle(
    rows: Iterable[Mapping[str, Any]],
    *,
    buffer_size: int,
    seed: int,
) -> Iterable[Mapping[str, Any]]:
    if buffer_size <= 1:
        yield from rows
        return
    rng = random.Random(seed)
    iterator = iter(rows)
    buffer: list[Mapping[str, Any]] = []
    try:
        for row in iterator:
            if len(buffer) < buffer_size:
                buffer.append(row)
                continue
            index = rng.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = row
        while buffer:
            yield buffer.pop(rng.randrange(len(buffer)))
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()


def _download_selected_raw_files(
    source: Mapping[str, Any],
    seed: int,
) -> list[tuple[dict[str, Any], Path]]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - depends on the optional data extra
        raise RuntimeError('install the corpus dependencies with: pip install -e ".[data]"') from exc

    selected = _selected_raw_files(source, seed)
    if not selected:
        raise RuntimeError(f"source {source['name']!r} selected no raw stream files")
    local_files: list[tuple[dict[str, Any], Path]] = []
    for item in selected:
        downloaded = Path(
            hf_hub_download(
                repo_id=source["dataset"],
                filename=item["path"],
                repo_type="dataset",
                revision=source["revision"],
            )
        )
        if not downloaded.is_file():
            raise RuntimeError(
                f"downloaded raw stream artifact is not a file: {item['path']}"
            )
        if downloaded.stat().st_size != item["bytes"]:
            raise RuntimeError(
                f"downloaded raw stream artifact byte-size mismatch: {item['path']}"
            )
        if _file_sha256(downloaded) != item["sha256"]:
            raise RuntimeError(
                f"downloaded raw stream artifact SHA-256 mismatch: {item['path']}"
            )
        local_files.append((item, downloaded))
    return local_files


def _load_raw_jsonl_gzip(
    source: Mapping[str, Any],
    seed: int,
) -> Iterable[Mapping[str, Any]]:
    local_files = _download_selected_raw_files(source, seed)

    def iter_interleaved() -> Iterable[Mapping[str, Any]]:
        with ExitStack() as stack:
            active: list[tuple[dict[str, Any], Any, int]] = [
                (item, stack.enter_context(gzip.open(path, "rb")), 0)
                for item, path in local_files
            ]
            while active:
                next_active: list[tuple[dict[str, Any], Any, int]] = []
                for item, handle, line_number in active:
                    raw_line = handle.readline()
                    if not raw_line:
                        continue
                    line_number += 1
                    if len(raw_line) > _MAX_RAW_JSONL_LINE_BYTES:
                        raise RuntimeError(
                            f"{item['path']}:{line_number}: raw JSONL row exceeds "
                            f"{_MAX_RAW_JSONL_LINE_BYTES} bytes"
                        )
                    try:
                        row = _strict_json_loads(raw_line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                        raise RuntimeError(
                            f"{item['path']}:{line_number}: invalid raw JSONL object"
                        ) from exc
                    if not isinstance(row, dict):
                        raise RuntimeError(
                            f"{item['path']}:{line_number}: raw JSONL row must be an object"
                        )
                    next_active.append((item, handle, line_number))
                    yield row
                active = next_active

    return _bounded_shuffle(
        iter_interleaved(),
        buffer_size=int(source.get("shuffle_buffer", 10_000)),
        seed=seed,
    )


def _load_raw_parquet_text(
    source: Mapping[str, Any],
    seed: int,
) -> Iterable[Mapping[str, Any]]:
    raw_stream = source.get("raw_stream")
    if not isinstance(raw_stream, Mapping):
        raise RuntimeError(f"source {source['name']!r} has no raw parquet stream")
    expected_runtime = raw_stream["reader_runtime"]
    expected_version = str(expected_runtime["version"])
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on the optional data extra
        raise RuntimeError('install the corpus dependencies with: pip install -e ".[data]"') from exc
    try:
        distribution_version = importlib_metadata.version("pyarrow")
    except importlib_metadata.PackageNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("the pinned pyarrow runtime is not installed") from exc
    module_version = getattr(pa, "__version__", None)
    if distribution_version != expected_version or module_version != expected_version:
        raise RuntimeError(
            f"source {source['name']!r} requires pyarrow {expected_version}, "
            f"found distribution={distribution_version!r} module={module_version!r}"
        )

    expected_schema = pa.schema(
        [
            pa.field(
                "image",
                pa.struct(
                    [
                        pa.field("bytes", pa.binary(), nullable=True),
                        pa.field("path", pa.string(), nullable=True),
                    ]
                ),
                nullable=True,
            ),
            pa.field("text", pa.string(), nullable=True),
            pa.field("llm_generated_idea", pa.string(), nullable=True),
        ]
    )
    text_field = str(raw_stream["text_field"])
    local_files = _download_selected_raw_files(source, seed)

    def iter_file_rows(
        item: Mapping[str, Any],
        parquet_file: Any,
    ) -> Iterable[Mapping[str, Any]]:
        try:
            batches = parquet_file.iter_batches(
                batch_size=1024,
                columns=[text_field],
                use_threads=False,
            )
            for batch in batches:
                if batch.schema.names != [text_field] or batch.num_columns != 1:
                    raise RuntimeError(
                        f"{item['path']}: parquet projection returned unexpected columns"
                    )
                for scalar in batch.column(0):
                    yield {text_field: scalar.as_py()}
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"{item['path']}: failed reading raw parquet text column"
            ) from exc

    def iter_interleaved() -> Iterable[Mapping[str, Any]]:
        with ExitStack() as stack:
            active: list[tuple[dict[str, Any], Iterable[Mapping[str, Any]]]] = []
            for item, path in local_files:
                # Hash the same held-open regular-file object that Arrow consumes. This closes the
                # path-replacement window between Hub-cache verification and parsing.
                handle = stack.enter_context(_open_verified_raw_artifact(path, item))
                try:
                    parquet_file = pq.ParquetFile(handle)
                except Exception as exc:
                    raise RuntimeError(f"{item['path']}: invalid raw parquet artifact") from exc
                if not parquet_file.schema_arrow.equals(expected_schema, check_metadata=False):
                    raise RuntimeError(f"{item['path']}: raw parquet schema mismatch")
                active.append((item, iter(iter_file_rows(item, parquet_file))))

            while active:
                next_active: list[
                    tuple[dict[str, Any], Iterable[Mapping[str, Any]]]
                ] = []
                for item, rows in active:
                    try:
                        row = next(rows)
                    except StopIteration:
                        continue
                    next_active.append((item, rows))
                    yield row
                active = next_active

    return _bounded_shuffle(
        iter_interleaved(),
        buffer_size=int(source.get("shuffle_buffer", 10_000)),
        seed=seed,
    )


def _load_stream(source: Mapping[str, Any], seed: int) -> Iterable[Mapping[str, Any]]:
    raw_stream = source.get("raw_stream")
    if raw_stream is not None:
        backend = raw_stream["backend"]
        if backend == _RAW_STREAM_BACKEND:
            return _load_raw_jsonl_gzip(source, seed)
        if backend == _RAW_PARQUET_STREAM_BACKEND:
            return _load_raw_parquet_text(source, seed)
        raise RuntimeError(f"unsupported normalized raw stream backend {backend!r}")
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on the optional data extra
        raise RuntimeError('install the corpus dependencies with: pip install -e ".[data]"') from exc

    kwargs: dict[str, Any] = {
        "path": source["dataset"],
        "split": source.get("split", "train"),
        "streaming": True,
        "revision": source["revision"],
    }
    if source.get("subset"):
        kwargs["name"] = source["subset"]
    stream = load_dataset(**kwargs)
    buffer_size = int(source.get("shuffle_buffer", 10_000))
    if buffer_size > 1:
        stream = stream.shuffle(seed=seed, buffer_size=buffer_size)
    return stream


def _source_paths(state_dir: Path, index: int, name: str) -> tuple[Path, Path, Path]:
    stem = f"{index:03d}-{name}"
    data_path = state_dir / f"{stem}.jsonl"
    return data_path, data_path.with_suffix(".jsonl.tmp"), state_dir / f"{stem}.manifest.json"


def _source_state_payload(
    *,
    plan: Mapping[str, Any],
    runtime: Mapping[str, Any],
    source: Mapping[str, Any],
    source_index: int,
    data_path: Path,
    stats: Mapping[str, Any],
    license_counts: Mapping[str, int],
) -> dict[str, Any]:
    state = {
        "data_bytes": data_path.stat().st_size,
        "data_sha256": _file_sha256(data_path),
        "kind": _SOURCE_STATE_KIND,
        "license_counts": dict(sorted(license_counts.items())),
        "plan_sha256": plan["plan_sha256"],
        "runtime_sha256": runtime["runtime_sha256"],
        "source_index": source_index,
        "source_plan_sha256": _canonical_sha256(source),
        "stats": dict(stats),
        "version": _SOURCE_STATE_VERSION,
    }
    state["state_sha256"] = _canonical_sha256(state)
    return state


def _restore_completed_source(
    *,
    plan: Mapping[str, Any],
    runtime: Mapping[str, Any],
    source: Mapping[str, Any],
    source_index: int,
    data_path: Path,
    state_path: Path,
    seen: set[str],
) -> tuple[dict[str, Any], Counter[str], int]:
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid completed-source state: {state_path}") from exc
    unsigned_state = {key: value for key, value in state.items() if key != "state_sha256"}
    if state.get("state_sha256") != _canonical_sha256(unsigned_state):
        raise RuntimeError(f"completed-source state self-hash mismatch: {state_path}")
    expected = {
        "kind": _SOURCE_STATE_KIND,
        "version": _SOURCE_STATE_VERSION,
        "plan_sha256": plan["plan_sha256"],
        "runtime_sha256": runtime["runtime_sha256"],
        "source_index": source_index,
        "source_plan_sha256": _canonical_sha256(source),
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise RuntimeError(f"completed-source state mismatch for {source['name']!r}: {key}")
    if not data_path.is_file():
        raise RuntimeError(f"completed-source data is missing: {data_path}")
    if state.get("data_bytes") != data_path.stat().st_size:
        raise RuntimeError(f"completed-source byte-size mismatch: {data_path}")
    if state.get("data_sha256") != _file_sha256(data_path):
        raise RuntimeError(f"completed-source SHA-256 mismatch: {data_path}")

    actual_chars = 0
    actual_documents = 0
    actual_licenses: Counter[str] = Counter()
    with data_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{data_path}:{line_number}: invalid completed-source JSON"
                ) from exc
            text = row.get("text")
            license_name = row.get("license")
            if not isinstance(text, str) or not isinstance(license_name, str):
                raise RuntimeError(
                    f"{data_path}:{line_number}: invalid completed-source document"
                )
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen:
                raise RuntimeError(
                    f"{data_path}:{line_number}: duplicate crosses a completed-source boundary"
                )
            expected_doc_id = f"{source['name']}:{digest}"
            if row.get("doc_id") != expected_doc_id:
                raise RuntimeError(f"{data_path}:{line_number}: document identity mismatch")
            seen.add(digest)
            actual_chars += len(text)
            actual_documents += 1
            actual_licenses[license_name] += 1
    stats = state.get("stats")
    if not isinstance(stats, dict):
        raise RuntimeError(f"completed-source stats are invalid: {state_path}")
    if stats.get("accepted_chars") != actual_chars:
        raise RuntimeError(f"completed-source character count mismatch: {data_path}")
    if stats.get("accepted_documents") != actual_documents:
        raise RuntimeError(f"completed-source document count mismatch: {data_path}")
    if state.get("license_counts") != dict(sorted(actual_licenses.items())):
        raise RuntimeError(f"completed-source license counts mismatch: {data_path}")
    return stats, actual_licenses, data_path.stat().st_size


def _committed_source_artifacts(
    plan: Mapping[str, Any],
    state_dir: Path,
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for source_index, source in enumerate(plan["sources"]):
        data_path, _, state_path = _source_paths(
            state_dir,
            source_index,
            str(source["name"]),
        )
        if not data_path.is_file() or not state_path.is_file():
            raise RuntimeError(
                f"committed source artifact pair is incomplete for {source['name']!r}"
            )
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid completed-source state: {state_path}") from exc
        unsigned_state = {key: value for key, value in state.items() if key != "state_sha256"}
        if state.get("state_sha256") != _canonical_sha256(unsigned_state):
            raise RuntimeError(f"completed-source state self-hash mismatch: {state_path}")
        if state.get("plan_sha256") != plan["plan_sha256"]:
            raise RuntimeError(f"completed-source plan mismatch: {state_path}")
        if state.get("source_index") != source_index:
            raise RuntimeError(f"completed-source index mismatch: {state_path}")
        if state.get("source_plan_sha256") != _canonical_sha256(source):
            raise RuntimeError(f"completed-source plan identity mismatch: {state_path}")
        data_bytes = data_path.stat().st_size
        data_sha256 = _file_sha256(data_path)
        if state.get("data_bytes") != data_bytes or state.get("data_sha256") != data_sha256:
            raise RuntimeError(f"completed-source data identity mismatch: {data_path}")
        artifacts.append(
            {
                "data_jsonl": {
                    "bytes": data_bytes,
                    "path": str(data_path),
                    "sha256": data_sha256,
                },
                "name": source["name"],
                "source_index": source_index,
                "state_manifest": {
                    "bytes": state_path.stat().st_size,
                    "path": str(state_path),
                    "sha256": _file_sha256(state_path),
                    "state_sha256": state["state_sha256"],
                },
            }
        )
    return artifacts


def _verified_completed_manifest(
    manifest_path: Path,
    raw_path: Path,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid completed download manifest: {manifest_path}") from exc
    unsigned_manifest = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if manifest.get("manifest_sha256") != _canonical_sha256(unsigned_manifest):
        raise RuntimeError(f"completed download manifest self-hash mismatch: {manifest_path}")
    if manifest.get("kind") != _DOWNLOAD_MANIFEST_KIND:
        raise RuntimeError(f"unexpected completed download manifest kind: {manifest_path}")
    if manifest.get("version") != _DOWNLOAD_MANIFEST_VERSION:
        raise RuntimeError(f"unexpected completed download manifest version: {manifest_path}")
    if manifest.get("plan_sha256") != plan["plan_sha256"]:
        raise RuntimeError("completed download was produced from a different acquisition plan")
    if manifest.get("acquisition_plan") != plan:
        raise RuntimeError("completed download embedded acquisition plan mismatch")
    if not raw_path.is_file():
        raise RuntimeError(f"completed download data is missing: {raw_path}")
    if manifest.get("raw_jsonl_bytes") != raw_path.stat().st_size:
        raise RuntimeError("completed download byte-size mismatch")
    if manifest.get("raw_jsonl_sha256") != _file_sha256(raw_path):
        raise RuntimeError("completed download SHA-256 mismatch")
    expected_source_artifacts = _committed_source_artifacts(
        plan,
        raw_path.parent / "download_state",
    )
    if manifest.get("source_artifacts") != expected_source_artifacts:
        raise RuntimeError("completed download source-artifact identities mismatch")
    declared_admission = manifest.get("storage_admission")
    if not isinstance(declared_admission, dict) or set(declared_admission) != {
        "admission_sha256",
        "bytes",
        "path",
        "sha256",
    }:
        raise RuntimeError("completed download storage-admission identity is invalid")
    admission_path = manifest_path.parent / Path(str(declared_admission["path"])).name
    try:
        admission = _verified_storage_admission(admission_path, plan)
    except ValueError as exc:
        raise RuntimeError("completed download storage admission is invalid") from exc
    actual_admission = {
        "admission_sha256": admission["admission_sha256"],
        "bytes": admission_path.stat().st_size,
        "path": admission_path.name,
        "sha256": _file_sha256(admission_path),
    }
    if declared_admission != actual_admission:
        raise RuntimeError("completed download storage-admission identity mismatch")
    return manifest


def stream_mixture(
    config_path: str | Path,
    out_dir: str | Path,
    *,
    target_chars: int | None = None,
    license_evidence: Mapping[str, str | Path] | None = None,
    resume: bool = False,
    loader: Callable[[Mapping[str, Any], int], Iterable[Mapping[str, Any]]] = _load_stream,
) -> dict[str, Any]:
    """Acquire the configured mixture and return its committed audit manifest.

    ``target_chars`` is a download/storage bound, not the final training-token count. The exact
    post-tokenization count is recorded later by shard packing. ``resume=True`` reuses only
    checksum-verified, completed source spools; a partial source is deterministically replayed.
    """

    config_path = Path(config_path)
    config = _load_config(config_path)
    decontamination_policy = normalize_evaluation_decontamination(config)
    plan = build_mixture_plan(config_path, target_chars=target_chars)
    out = Path(out_dir)
    raw_path = out / "mixture.jsonl"
    manifest_path = out / "download_manifest.json"
    state_dir = out / "download_state"
    admission_path = out / _STORAGE_ADMISSION_FILENAME
    if manifest_path.exists():
        if not resume:
            raise RuntimeError(
                f"download manifest already exists: {manifest_path}; pass resume=True to verify/reuse"
            )
        return _verified_completed_manifest(manifest_path, raw_path, plan)
    if not resume and (raw_path.exists() or state_dir.exists()):
        raise RuntimeError(
            f"incomplete or unpublished acquisition exists under {out}; "
            "pass resume=True to verify completed sources and replay the partial source"
        )
    runtime = acquisition_runtime_identity()
    readiness = audit_mixture_readiness(
        plan,
        out_dir,
        license_evidence=license_evidence,
        require_stream_runtime=loader is _load_stream,
        resume=resume,
    )
    if not readiness["ready"]:
        raise RuntimeError("mixture acquisition is not ready: " + "; ".join(readiness["blockers"]))

    out.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    if admission_path.exists():
        storage_admission = _verified_storage_admission(admission_path, plan)
    else:
        storage_admission = _storage_admission_payload(plan, readiness["disk"])
        _atomic_json(admission_path, storage_admission)

    seen: set[str] = set()
    source_stats: dict[str, dict[str, Any]] = {}
    license_counts: Counter[str] = Counter()
    total_chars = 0
    total_documents = 0
    completed_data_bytes = 0
    max_raw_bytes = plan["storage"]["max_raw_jsonl_bytes"]

    for source_index, source in enumerate(plan["sources"]):
        name = str(source["name"])
        data_path, tmp_path, state_path = _source_paths(state_dir, source_index, name)
        if resume and data_path.exists() and state_path.exists():
            stats, restored_licenses, source_bytes = _restore_completed_source(
                plan=plan,
                runtime=runtime,
                source=source,
                source_index=source_index,
                data_path=data_path,
                state_path=state_path,
                seen=seen,
            )
            source_stats[name] = stats
            license_counts.update(restored_licenses)
            total_chars += int(stats["accepted_chars"])
            total_documents += int(stats["accepted_documents"])
            completed_data_bytes += source_bytes
            continue
        if data_path.exists() != state_path.exists():
            if not resume:
                raise RuntimeError(f"incomplete completed-source pair for {name!r}")
            data_path.unlink(missing_ok=True)
            state_path.unlink(missing_ok=True)
        if tmp_path.exists():
            if not resume:
                raise RuntimeError(
                    f"partial source stream exists: {tmp_path}; pass resume=True to replay it"
                )
            tmp_path.unlink()

        text_field = str(source["text_field"])
        fixed_license = source["license"]
        license_field = source["license_field"]
        allowed = set(source["allowed_licenses"])
        accepted_chars = 0
        accepted_documents = 0
        source_licenses: Counter[str] = Counter()
        skipped: Counter[str] = Counter()
        exhausted = True
        source_digest = hashlib.sha256()
        source_bytes = 0

        try:
            with tmp_path.open("wb") as handle:
                row_stream = iter(loader(source, int(plan["seed"]) + source_index))
                try:
                    for row in row_stream:
                        text = row.get(text_field)
                        if not isinstance(text, str):
                            skipped["missing_text"] += 1
                            continue
                        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
                        if not int(plan["min_document_chars"]) <= len(text) <= int(
                            plan["max_document_chars"]
                        ):
                            skipped["length"] += 1
                            continue
                        license_name = _normalize_license(
                            row.get(str(license_field)) if license_field else fixed_license
                        )
                        if allowed and license_name not in allowed:
                            skipped["license"] += 1
                            continue
                        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                        if digest in seen:
                            skipped["duplicate"] += 1
                            continue
                        evidence = source["license_evidence"]
                        document = CorpusDocument(
                            text=text,
                            source=_stable_source(source, row),
                            doc_id=f"{name}:{digest}",
                            license=license_name,
                            meta={
                                "dataset": source["dataset"],
                                "license_evidence_id": evidence["id"] if evidence else None,
                                "mixture_source": name,
                                "revision": source["revision"],
                                "subset": source.get("subset"),
                            },
                        )
                        encoded = (_canonical_json(asdict(document)) + "\n").encode("utf-8")
                        if (
                            max_raw_bytes is not None
                            and completed_data_bytes + source_bytes + len(encoded) > max_raw_bytes
                        ):
                            raise RuntimeError(
                                "storage.max_raw_jsonl_bytes would be exceeded before "
                                f"source {name!r} reached its character budget"
                            )
                        handle.write(encoded)
                        source_digest.update(encoded)
                        source_bytes += len(encoded)
                        seen.add(digest)
                        accepted_chars += len(text)
                        accepted_documents += 1
                        source_licenses[license_name] += 1
                        if accepted_chars >= int(source["requested_chars"]):
                            exhausted = False
                            break
                finally:
                    close = getattr(row_stream, "close", None)
                    if callable(close):
                        close()
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            # Keep the partial spool as interruption evidence. ``resume=True`` explicitly replays
            # this source from its immutable revision and never treats the partial bytes as valid.
            raise

        if plan["require_full_source_budgets"] and exhausted:
            shortfall = int(source["requested_chars"]) - accepted_chars
            raise RuntimeError(
                f"source {name!r} exhausted {shortfall} characters before its required budget"
            )
        if accepted_documents == 0:
            raise RuntimeError(f"no documents were accepted from source {name!r}")

        tmp_path.replace(data_path)
        stats = {
            "accepted_chars": accepted_chars,
            "accepted_documents": accepted_documents,
            "dataset": source["dataset"],
            "license_evidence": source["license_evidence"],
            "normalized_weight": source["normalized_weight"],
            "requested_chars": source["requested_chars"],
            "revision": source["revision"],
            "skipped": dict(sorted(skipped.items())),
            "stream_exhausted_before_budget": exhausted,
            "subset": source.get("subset"),
            "weight": float(source["weight"]),
        }
        if source.get("raw_stream") is not None:
            selected_files = _selected_raw_files(
                source,
                int(plan["seed"]) + source_index,
            )
            stats["raw_stream"] = {
                "backend": source["raw_stream"]["backend"],
                "file_inventory": {
                    key: value
                    for key, value in source["raw_stream"]["file_inventory"].items()
                    if key != "files"
                },
                "selected_file_count": len(selected_files),
                "selected_files": selected_files,
                "selected_total_bytes": sum(item["bytes"] for item in selected_files),
                "selection": source["raw_stream"]["selection"],
            }
            if source["raw_stream"]["backend"] == _RAW_PARQUET_STREAM_BACKEND:
                stats["raw_stream"].update(
                    {
                        "parquet_schema": source["raw_stream"]["parquet_schema"],
                        "projection": {
                            "columns": [source["text_field"]],
                            "materializes_only_configured_text": True,
                        },
                        "reader_runtime": source["raw_stream"]["reader_runtime"],
                    }
                )
        state = _source_state_payload(
            plan=plan,
            runtime=runtime,
            source=source,
            source_index=source_index,
            data_path=data_path,
            stats=stats,
            license_counts=source_licenses,
        )
        if state["data_sha256"] != source_digest.hexdigest():
            raise AssertionError("source spool digest changed before commit")
        _atomic_json(state_path, state)
        source_stats[name] = stats
        license_counts.update(source_licenses)
        total_chars += accepted_chars
        total_documents += accepted_documents
        completed_data_bytes += data_path.stat().st_size

    if total_documents == 0:
        raise RuntimeError("no documents were accepted from the configured sources")

    raw_tmp = raw_path.with_suffix(".jsonl.tmp")
    raw_digest = hashlib.sha256()
    raw_bytes = 0
    with raw_tmp.open("wb") as output:
        for source_index, source in enumerate(plan["sources"]):
            data_path, _, _ = _source_paths(state_dir, source_index, str(source["name"]))
            with data_path.open("rb") as input_handle:
                for chunk in iter(lambda: input_handle.read(1024 * 1024), b""):
                    output.write(chunk)
                    raw_digest.update(chunk)
                    raw_bytes += len(chunk)
        output.flush()
        os.fsync(output.fileno())
    if raw_bytes != completed_data_bytes:
        raise AssertionError("combined raw JSONL byte count differs from completed source spools")
    raw_tmp.replace(raw_path)

    verified_evidence = [
        result
        for result in readiness["license_evidence"]
        if result["status"] == "verified"
    ]
    manifest: dict[str, Any] = {
        "accepted_chars": total_chars,
        "accepted_documents": total_documents,
        "acquisition_plan": plan,
        "config": str(config_path),
        "config_bytes": plan["config_bytes"],
        "config_sha256": plan["config_sha256"],
        "kind": _DOWNLOAD_MANIFEST_KIND,
        "license_counts": dict(sorted(license_counts.items())),
        "license_evidence": verified_evidence,
        "plan_sha256": plan["plan_sha256"],
        "raw_jsonl": str(raw_path),
        "raw_jsonl_bytes": raw_bytes,
        "raw_jsonl_sha256": raw_digest.hexdigest(),
        "requested_chars": plan["requested_chars"],
        "runtime": runtime,
        "seed": plan["seed"],
        "source_artifacts": _committed_source_artifacts(plan, state_dir),
        "sources": source_stats,
        "storage_admission": {
            "admission_sha256": storage_admission["admission_sha256"],
            "bytes": admission_path.stat().st_size,
            "path": admission_path.name,
            "sha256": _file_sha256(admission_path),
        },
        "version": _DOWNLOAD_MANIFEST_VERSION,
    }
    if decontamination_policy is not None:
        manifest["evaluation_decontamination"] = decontamination_policy
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    _atomic_json(manifest_path, manifest)
    return manifest
