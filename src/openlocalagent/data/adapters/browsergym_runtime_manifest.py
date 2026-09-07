"""Strict, path-independent runtime manifests for controlled BrowserGym capture.

The manifest binds the active Python executable, interpreter/OS identity, and every installed
distribution's declared regular files below the active ``purelib``/``platlib`` roots. Generated
console-script launchers are deliberately outside that byte identity because installers commonly
embed absolute interpreter paths in them. They are accepted only as declared entry-point launchers
directly below the active ``scripts`` root, checked as stable regular files, and recorded by a
path-independent logical identity.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib.metadata
import json
import os
import platform
import re
import stat
import sys
import sysconfig
import tempfile
import urllib.parse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

BROWSERGYM_RUNTIME_MANIFEST_KIND = "localagent_browsergym_capture_environment"
BROWSERGYM_RUNTIME_MANIFEST_SCHEMA_VERSION = 1
DEFAULT_MAX_MANIFEST_BYTES = 8 * 1024 * 1024

_MAX_DISTRIBUTIONS = 10_000
_MAX_DECLARED_FILES_PER_DISTRIBUTION = 250_000
_MAX_TOTAL_DECLARED_FILES = 1_000_000
_MAX_FILE_BYTES = 4 * 1024 * 1024 * 1024
_MAX_TOTAL_FILE_BYTES = 64 * 1024 * 1024 * 1024
_MAX_DIRECT_URL_BYTES = 1024 * 1024
_MAX_NAME_BYTES = 512
_MAX_VERSION_BYTES = 4096
_MAX_PATH_BYTES = 16 * 1024
_READ_CHUNK_BYTES = 1024 * 1024

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_NORMALIZED_NAME_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_WINDOWS_DRIVE_RE = re.compile(r"[A-Za-z]:")
_ENTRY_POINT_GROUPS = frozenset({"console_scripts", "gui_scripts"})
_BYTECODE_SUFFIX = ".pyc"
_BYTECODE_DIRECTORY = "__pycache__"

_DISTRIBUTION_FILE_POLICY = (
    "sha256 of canonical JSONL records containing logical import-root-relative path, POSIX "
    "permission mode, identity byte count, and content sha256 for every "
    "importlib.metadata-declared regular file below active purelib/platlib or the exact non-symlink "
    "sysconfig data/include ancillary root, subject only to the signed bytecode and "
    "installer-metadata normalization rules"
)
_OUTSIDE_ROOT_POLICY = (
    "only declared console_scripts/gui_scripts launchers directly below the active sysconfig "
    "scripts root, plus installer aliases suffixed by the exact active Python major or major.minor, "
    "are allowed outside distribution roots; launchers are validated as stable regular files and "
    "recorded by logical entry-point identity but not byte-hashed because generated wrappers may "
    "embed absolute installation paths"
)
_INSTALLER_METADATA_POLICY = (
    "dist-info/RECORD is validated as a declared stable regular file but excluded because it "
    "redundantly embeds hashes and byte counts for path-dependent generated launchers and "
    "direct_url.json; non-editable file: direct_url.json is identity-hashed after replacing its "
    "absolute archive URL with file:///<local-archive>, and is accepted only with a matching "
    "sha256 archive_info hash"
)
_PLAYWRIGHT_DRIVER_RELATIVE_ROOT = "playwright/driver"

_TOP_LEVEL_KEYS = frozenset(
    {
        "kind",
        "schema_version",
        "scope",
        "python",
        "operating_system",
        "installed_distributions",
        "playwright_driver",
        "manifest_self_sha256",
    }
)
_SCOPE_KEYS = frozenset(
    {
        "import_roots",
        "ancillary_roots",
        "distribution_file_identity",
        "excluded_bytecode",
        "outside_import_roots",
        "installer_metadata",
    }
)
_PYTHON_KEYS = frozenset(
    {"implementation", "version", "cache_tag", "soabi", "executable"}
)
_OS_KEYS = frozenset({"system", "release", "version", "machine", "macos"})
_MACOS_KEYS = frozenset({"release", "version_info", "machine"})
_DISTRIBUTION_KEYS = frozenset(
    {"name", "version", "content", "excluded_entry_point_launchers"}
)
_CONTENT_KEYS = frozenset({"files", "bytes", "sha256"})
_LAUNCHER_KEYS = frozenset({"group", "name", "artifact"})
_DRIVER_KEYS = frozenset(
    {"distribution", "import_root", "relative_root", "content"}
)
_IDENTITY_KEYS = frozenset({"bytes", "sha256"})


@dataclass(frozen=True)
class _ImportRoot:
    label: str
    path: Path


@dataclass(frozen=True)
class _FileRecord:
    logical_path: str
    root_relative_path: str
    root: _ImportRoot
    mode: int
    bytes: int
    sha256: str

    def aggregate_record(self, *, logical_path: str | None = None) -> dict[str, Any]:
        return {
            "path": self.logical_path if logical_path is None else logical_path,
            "mode": self.mode,
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class _DistributionCandidate:
    name: str
    version: str
    distribution: importlib.metadata.Distribution


def _canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_json_loads(payload: bytes, *, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-finite JSON number {value!r}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not valid UTF-8") from error
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(label):
            raise
        raise ValueError(f"{label} is not strict JSON") from error


def _exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    observed = frozenset(value)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(f"{label} schema drift: missing={missing}, extra={extra}")


def _bounded_string(
    value: Any,
    *,
    label: str,
    maximum_bytes: int,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str) or (not value and not allow_empty):
        requirement = "a string" if allow_empty else "a non-empty string"
        raise ValueError(f"{label} must be {requirement}")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError as error:
        raise ValueError(f"{label} is not valid UTF-8") from error
    if len(encoded) > maximum_bytes:
        raise ValueError(f"{label} exceeds the {maximum_bytes}-byte limit")
    return value


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _positive_int(value: Any, *, label: str) -> int:
    result = _nonnegative_int(value, label=label)
    if result == 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _sha256_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase 64-hex SHA-256")
    return value


def _normalize_distribution_name(name: str) -> str:
    normalized = re.sub(r"[-_.]+", "-", name).lower()
    if _NORMALIZED_NAME_RE.fullmatch(normalized) is None:
        raise ValueError(f"invalid installed distribution name {name!r}")
    return normalized


def _resolved_directory(path_value: Any, *, label: str) -> Path:
    try:
        raw = Path(os.fspath(path_value))
    except TypeError as error:
        raise ValueError(f"{label} is not a filesystem path") from error
    if not raw.is_absolute():
        raw = Path(os.path.abspath(raw))
    try:
        initial = raw.lstat()
    except OSError as error:
        raise ValueError(f"{label} is missing or unreadable") from error
    if stat.S_ISLNK(initial.st_mode) or not stat.S_ISDIR(initial.st_mode):
        raise ValueError(f"{label} must be a non-symlink directory")
    try:
        resolved = raw.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"{label} cannot be resolved safely") from error
    if not resolved.is_dir():
        raise ValueError(f"{label} must resolve to a directory")
    return resolved


def _active_import_roots() -> tuple[tuple[_ImportRoot, ...], Path]:
    paths = sysconfig.get_paths()
    if not isinstance(paths, Mapping):
        raise ValueError("sysconfig paths must be a mapping")
    try:
        purelib = _resolved_directory(paths["purelib"], label="active purelib")
        platlib = _resolved_directory(paths["platlib"], label="active platlib")
    except KeyError as error:
        raise ValueError("sysconfig must provide purelib and platlib paths") from error
    if purelib == platlib:
        roots = (_ImportRoot("purelib+platlib", purelib),)
    else:
        try:
            purelib.relative_to(platlib)
            overlap = True
        except ValueError:
            try:
                platlib.relative_to(purelib)
                overlap = True
            except ValueError:
                overlap = False
        if overlap:
            raise ValueError("active purelib and platlib roots overlap ambiguously")
        roots = tuple(
            sorted(
                (
                    _ImportRoot("purelib", purelib),
                    _ImportRoot("platlib", platlib),
                ),
                key=lambda root: root.label,
            )
        )
    try:
        scripts_value = paths["scripts"]
    except KeyError as error:
        raise ValueError("sysconfig must provide the active scripts path") from error
    scripts = Path(os.path.abspath(Path(os.fspath(scripts_value))))
    try:
        data_value = paths["data"]
    except KeyError as error:
        raise ValueError("sysconfig must provide the active data path") from error
    data_include = Path(os.path.abspath(Path(os.fspath(data_value)))) / "include"
    if data_include.exists() or data_include.is_symlink():
        ancillary = _resolved_directory(
            data_include,
            label="active sysconfig data/include",
        )
        if any(
            ancillary == root.path
            or ancillary.is_relative_to(root.path)
            or root.path.is_relative_to(ancillary)
            for root in roots
        ):
            raise ValueError(
                "active sysconfig data/include overlaps an import root ambiguously"
            )
        roots = (*roots, _ImportRoot("data-include", ancillary))
    return roots, scripts


def _stable_regular_file_identity_by_path(
    root: Path,
    relative: PurePosixPath,
    *,
    label: str,
    maximum_bytes: int,
    return_payload: bool,
) -> tuple[dict[str, Any], bytes | None]:
    """Use lstat-before/open/fstat/lstat-after where descriptor-relative opens are unavailable."""

    directories: list[tuple[Path, os.stat_result]] = []
    current = root
    try:
        root_stat = current.lstat()
    except OSError as error:
        raise ValueError(f"{label} import root is unreadable") from error
    directories.append((current, root_stat))
    for part in relative.parts[:-1]:
        current /= part
        try:
            directory_stat = current.lstat()
        except OSError as error:
            raise ValueError(f"{label} is missing or unreadable") from error
        if stat.S_ISLNK(directory_stat.st_mode) or not stat.S_ISDIR(
            directory_stat.st_mode
        ):
            raise ValueError(f"{label} contains a symlink or non-directory component")
        directories.append((current, directory_stat))
    target = current / relative.parts[-1]
    try:
        path_before = target.lstat()
    except OSError as error:
        raise ValueError(f"{label} is missing or unreadable") from error
    if stat.S_ISLNK(path_before.st_mode) or not stat.S_ISREG(path_before.st_mode):
        raise ValueError(f"{label} must be a non-symlink regular file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags)
    except OSError as error:
        raise ValueError(f"{label} could not be opened safely") from error
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if return_payload else None
    observed_bytes = 0
    try:
        before = os.fstat(descriptor)
        identity_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
        if any(
            getattr(path_before, field) != getattr(before, field)
            for field in identity_fields
        ):
            raise ValueError(f"{label} changed before it was opened")
        if before.st_size < 0 or before.st_size > maximum_bytes:
            raise ValueError(f"{label} exceeds the {maximum_bytes}-byte limit")
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            observed_bytes += len(chunk)
            if observed_bytes > maximum_bytes:
                raise ValueError(f"{label} exceeds the {maximum_bytes}-byte limit")
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if any(getattr(before, field) != getattr(after, field) for field in identity_fields):
        raise ValueError(f"{label} changed while it was hashed")
    if observed_bytes != before.st_size:
        raise ValueError(f"{label} changed byte count while it was hashed")
    try:
        path_after = target.lstat()
    except OSError as error:
        raise ValueError(f"{label} disappeared while it was hashed") from error
    if stat.S_ISLNK(path_after.st_mode) or any(
        getattr(before, field) != getattr(path_after, field)
        for field in identity_fields
    ):
        raise ValueError(f"{label} changed path identity while it was hashed")
    directory_fields = ("st_dev", "st_ino", "st_mode", "st_mtime_ns")
    for directory, directory_before in directories:
        try:
            directory_after = directory.lstat()
        except OSError as error:
            raise ValueError(f"{label} parent changed while it was hashed") from error
        if stat.S_ISLNK(directory_after.st_mode) or any(
            getattr(directory_before, field) != getattr(directory_after, field)
            for field in directory_fields
        ):
            raise ValueError(f"{label} parent changed while it was hashed")
    payload = b"".join(chunks) if chunks is not None else None
    return (
        {
            "mode": stat.S_IMODE(before.st_mode),
            "bytes": observed_bytes,
            "sha256": digest.hexdigest(),
        },
        payload,
    )


def _stable_regular_file_identity(
    root: Path,
    relative: PurePosixPath,
    *,
    label: str,
    maximum_bytes: int = _MAX_FILE_BYTES,
    return_payload: bool = False,
) -> tuple[dict[str, Any], bytes | None]:
    parts = relative.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{label} has an unsafe relative path")
    if os.open not in os.supports_dir_fd:
        return _stable_regular_file_identity_by_path(
            root,
            relative,
            label=label,
            maximum_bytes=maximum_bytes,
            return_payload=return_payload,
        )
    flags_directory = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags_nofollow = getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = []
    try:
        root_descriptor = os.open(root, flags_directory | flags_nofollow)
        descriptors.append(root_descriptor)
        current_descriptor = root_descriptor
        for part in parts[:-1]:
            next_descriptor = os.open(
                part,
                flags_directory | flags_nofollow,
                dir_fd=current_descriptor,
            )
            descriptors.append(next_descriptor)
            current_descriptor = next_descriptor
        file_descriptor = os.open(
            parts[-1],
            os.O_RDONLY | flags_nofollow,
            dir_fd=current_descriptor,
        )
        descriptors.append(file_descriptor)
    except (NotImplementedError, OSError) as error:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise ValueError(f"{label} is missing, escaping, or contains a symlink") from error

    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if return_payload else None
    observed_bytes = 0
    try:
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        if before.st_size < 0 or before.st_size > maximum_bytes:
            raise ValueError(f"{label} exceeds the {maximum_bytes}-byte limit")
        while True:
            chunk = os.read(file_descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            observed_bytes += len(chunk)
            if observed_bytes > maximum_bytes:
                raise ValueError(f"{label} exceeds the {maximum_bytes}-byte limit")
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(file_descriptor)
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise ValueError(f"{label} changed while it was hashed")
    if observed_bytes != before.st_size:
        raise ValueError(f"{label} changed byte count while it was hashed")
    target = root.joinpath(*parts)
    try:
        final = target.lstat()
    except OSError as error:
        raise ValueError(f"{label} disappeared while it was hashed") from error
    if stat.S_ISLNK(final.st_mode) or any(
        getattr(before, field) != getattr(final, field) for field in fields
    ):
        raise ValueError(f"{label} changed path identity while it was hashed")
    payload = b"".join(chunks) if chunks is not None else None
    return (
        {
            "mode": stat.S_IMODE(before.st_mode),
            "bytes": observed_bytes,
            "sha256": digest.hexdigest(),
        },
        payload,
    )


def _read_bounded_stable_file(
    path: Path,
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    try:
        parent = _resolved_directory(path.parent, label=f"{label} parent")
    except ValueError as error:
        raise ValueError(f"{label} has an unsafe parent directory") from error
    identity, payload = _stable_regular_file_identity(
        parent,
        PurePosixPath(path.name),
        label=label,
        maximum_bytes=maximum_bytes,
        return_payload=True,
    )
    if identity["bytes"] == 0:
        raise ValueError(f"{label} must be non-empty")
    assert payload is not None
    return payload


def _manifest_content(records: Sequence[_FileRecord], *, driver_relative: bool = False) -> dict:
    digest = hashlib.sha256()
    total_bytes = 0
    for record in sorted(records, key=lambda item: item.logical_path):
        logical_path = record.logical_path
        if driver_relative:
            prefix = f"{record.root.label}/{_PLAYWRIGHT_DRIVER_RELATIVE_ROOT}/"
            if not logical_path.startswith(prefix):
                raise RuntimeError("Playwright driver record escaped its logical subtree")
            logical_path = logical_path.removeprefix(prefix)
        digest.update(
            _canonical_json_bytes(
                record.aggregate_record(logical_path=logical_path),
                newline=True,
            )
        )
        total_bytes += record.bytes
        if total_bytes > _MAX_TOTAL_FILE_BYTES:
            raise ValueError(
                f"manifest file bytes exceed the {_MAX_TOTAL_FILE_BYTES}-byte limit"
            )
    return {"files": len(records), "bytes": total_bytes, "sha256": digest.hexdigest()}


def _entry_point_names(
    distribution: importlib.metadata.Distribution,
    *,
    distribution_name: str,
) -> frozenset[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    try:
        entry_points: Iterable[Any] = distribution.entry_points
    except Exception as error:
        raise ValueError(
            f"could not read entry points for distribution {distribution_name!r}"
        ) from error
    for entry_point in entry_points:
        group = getattr(entry_point, "group", None)
        if group not in _ENTRY_POINT_GROUPS:
            continue
        name = _bounded_string(
            getattr(entry_point, "name", None),
            label=f"distribution {distribution_name!r} entry-point name",
            maximum_bytes=_MAX_NAME_BYTES,
        )
        if "/" in name or "\\" in name or name in {".", ".."}:
            raise ValueError(
                f"distribution {distribution_name!r} has an unsafe entry-point name"
            )
        key = (group, name)
        if key in result:
            raise ValueError(
                f"distribution {distribution_name!r} has duplicate entry point {key!r}"
            )
        result.add(key)
    return frozenset(result)


def _launcher_identity(
    path: Path,
    *,
    declared_text: str,
    scripts_root: Path,
    entry_points: frozenset[tuple[str, str]],
    distribution_name: str,
) -> dict[str, str] | None:
    normalized = Path(os.path.abspath(path))
    try:
        relative = normalized.relative_to(scripts_root)
    except ValueError:
        return None
    if len(relative.parts) != 1:
        return None
    basename = relative.name
    declared_parts = PurePosixPath(declared_text).parts
    if (
        len(declared_parts) < 3
        or any(part != ".." for part in declared_parts[:-2])
        or declared_parts[-2].lower() not in {"bin", "scripts"}
        or declared_parts[-1] != basename
    ):
        return None

    def launcher_forms(stem: str, artifact: str) -> dict[str, str]:
        return {
            stem: artifact,
            f"{stem}.exe": f"{artifact}-exe",
            f"{stem}-script.py": f"{artifact}-script-py",
            f"{stem}.cmd": f"{artifact}-cmd",
        }

    exact_matches: list[dict[str, str]] = []
    for group, name in sorted(entry_points):
        artifact = launcher_forms(name, "script").get(basename)
        if artifact is not None:
            exact_matches.append(
                {"group": group, "name": name, "artifact": artifact}
            )
    if len(exact_matches) > 1:
        raise ValueError(
            f"distribution {distribution_name!r} launcher {basename!r} is ambiguous"
        )
    if exact_matches:
        return exact_matches[0]

    version_aliases = (
        (str(sys.version_info.major), "python-major"),
        (
            f"{sys.version_info.major}.{sys.version_info.minor}",
            "python-major-minor",
        ),
    )
    alias_matches: list[dict[str, str]] = []
    for group, name in sorted(entry_points):
        for suffix, artifact_prefix in version_aliases:
            artifact = launcher_forms(
                f"{name}{suffix}",
                artifact_prefix,
            ).get(basename)
            if artifact is not None:
                alias_matches.append(
                    {"group": group, "name": name, "artifact": artifact}
                )
    if len(alias_matches) > 1:
        raise ValueError(
            f"distribution {distribution_name!r} launcher {basename!r} is ambiguous"
        )
    return alias_matches[0] if alias_matches else None


def _validate_excluded_launcher(
    path: Path,
    *,
    scripts_root: Path,
    label: str,
) -> None:
    root = _resolved_directory(scripts_root, label="active scripts root")
    normalized = Path(os.path.abspath(path))
    try:
        relative = normalized.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the active scripts root") from error
    if len(relative.parts) != 1:
        raise ValueError(f"{label} is not directly below the active scripts root")
    _stable_regular_file_identity(
        root,
        PurePosixPath(*relative.parts),
        label=label,
    )


def _declared_path_text(path: Any, *, distribution_name: str) -> str:
    try:
        value = str(path)
    except Exception as error:
        raise ValueError(
            f"distribution {distribution_name!r} has an invalid declared path"
        ) from error
    value = _bounded_string(
        value,
        label=f"distribution {distribution_name!r} declared path",
        maximum_bytes=_MAX_PATH_BYTES,
    )
    if (
        "\x00" in value
        or "\\" in value
        or value.startswith("/")
        or _WINDOWS_DRIVE_RE.match(value) is not None
    ):
        raise ValueError(
            f"distribution {distribution_name!r} has an unsafe declared path {value!r}"
        )
    return value


def _root_for_located_path(
    path: Path,
    *,
    roots: Sequence[_ImportRoot],
) -> tuple[_ImportRoot, PurePosixPath] | None:
    normalized = Path(os.path.abspath(path))
    matches: list[tuple[_ImportRoot, Path]] = []
    for root in roots:
        try:
            matches.append((root, normalized.relative_to(root.path)))
        except ValueError:
            continue
    if len(matches) > 1:
        raise ValueError("declared distribution path maps to multiple active import roots")
    if not matches:
        return None
    root, relative = matches[0]
    return root, PurePosixPath(*relative.parts)


def _valid_ancillary_declared_path(
    declared_text: str,
    *,
    root: _ImportRoot,
    relative: PurePosixPath,
) -> bool:
    if root.label != "data-include":
        return False
    parts = PurePosixPath(declared_text).parts
    first_non_parent = next(
        (index for index, part in enumerate(parts) if part != ".."),
        None,
    )
    if first_non_parent is None or first_non_parent == 0:
        return False
    return (
        parts[first_non_parent] == "include"
        and parts[first_non_parent + 1 :] == relative.parts
    )


def _reject_editable_direct_url(
    distribution: importlib.metadata.Distribution,
    *,
    distribution_name: str,
) -> dict[str, Any] | None:
    try:
        direct_url = distribution.read_text("direct_url.json")
    except Exception as error:
        raise ValueError(
            f"could not read direct_url.json for distribution {distribution_name!r}"
        ) from error
    if direct_url is None:
        return None
    if not isinstance(direct_url, str):
        raise ValueError(
            f"distribution {distribution_name!r} direct_url.json is not text"
        )
    raw = direct_url.encode("utf-8")
    if len(raw) > _MAX_DIRECT_URL_BYTES:
        raise ValueError(
            f"distribution {distribution_name!r} direct_url.json exceeds "
            f"{_MAX_DIRECT_URL_BYTES} bytes"
        )
    value = _strict_json_loads(
        raw,
        label=f"distribution {distribution_name!r} direct_url.json",
    )
    if not isinstance(value, dict):
        raise ValueError(
            f"distribution {distribution_name!r} direct_url.json must be an object"
        )
    directory_info = value.get("dir_info")
    if directory_info is not None and not isinstance(directory_info, dict):
        raise ValueError(
            f"distribution {distribution_name!r} direct_url.json dir_info must be an object"
        )
    if isinstance(directory_info, dict) and directory_info.get("editable") is True:
        raise ValueError(
            f"editable distribution {distribution_name!r} is forbidden"
        )
    return value


def _canonical_direct_url_identity(
    value: Mapping[str, Any],
    *,
    distribution_name: str,
) -> dict[str, Any]:
    url = _bounded_string(
        value.get("url"),
        label=f"distribution {distribution_name!r} direct_url.json url",
        maximum_bytes=_MAX_PATH_BYTES,
    )
    parsed = urllib.parse.urlsplit(url)
    if not parsed.scheme:
        raise ValueError(
            f"distribution {distribution_name!r} direct_url.json URL has no scheme"
        )
    if parsed.scheme.lower() != "file":
        return dict(value)
    if parsed.netloc not in {"", "localhost"} or parsed.query or parsed.fragment:
        raise ValueError(
            f"distribution {distribution_name!r} has an unsupported local archive URL"
        )
    if not parsed.path:
        raise ValueError(
            f"distribution {distribution_name!r} local archive URL has no path"
        )
    archive_info = value.get("archive_info")
    if not isinstance(archive_info, dict):
        raise ValueError(
            f"distribution {distribution_name!r} local direct URL must bind archive_info"
        )
    if not set(archive_info).issubset({"hash", "hashes"}) or "hashes" not in archive_info:
        raise ValueError(
            f"distribution {distribution_name!r} archive_info schema is unsupported"
        )
    hashes = archive_info["hashes"]
    if not isinstance(hashes, dict):
        raise ValueError(
            f"distribution {distribution_name!r} archive_info.hashes must be an object"
        )
    archive_sha256 = _sha256_string(
        hashes.get("sha256"),
        label=f"distribution {distribution_name!r} archive_info sha256",
    )
    for algorithm, digest in hashes.items():
        _bounded_string(
            algorithm,
            label=f"distribution {distribution_name!r} archive hash algorithm",
            maximum_bytes=_MAX_NAME_BYTES,
        )
        _bounded_string(
            digest,
            label=f"distribution {distribution_name!r} archive hash value",
            maximum_bytes=_MAX_VERSION_BYTES,
        )
    legacy_hash = archive_info.get("hash")
    if legacy_hash is not None and legacy_hash != f"sha256={archive_sha256}":
        raise ValueError(
            f"distribution {distribution_name!r} archive_info hash fields disagree"
        )
    if "dir_info" in value or "vcs_info" in value:
        raise ValueError(
            f"distribution {distribution_name!r} local archive identity cannot contain "
            "directory or VCS metadata"
        )
    normalized = dict(value)
    normalized["url"] = "file:///<local-archive>"
    return normalized


def _distribution_candidates(
    import_roots: Sequence[_ImportRoot],
) -> list[_DistributionCandidate]:
    candidates: list[_DistributionCandidate] = []
    observed_names: set[str] = set()
    search_paths = [
        os.fspath(root.path)
        for root in import_roots
        if root.label != "data-include"
    ]
    if not search_paths:
        raise ValueError("at least one active purelib/platlib search path is required")
    try:
        distributions = importlib.metadata.distributions(path=search_paths)
    except Exception as error:
        raise ValueError(
            "could not enumerate distributions from active purelib/platlib"
        ) from error
    for index, distribution in enumerate(distributions):
        if index >= _MAX_DISTRIBUTIONS:
            raise ValueError(
                f"installed distribution count exceeds {_MAX_DISTRIBUTIONS}"
            )
        try:
            raw_name = distribution.metadata["Name"]
            raw_version = distribution.version
        except Exception as error:
            raise ValueError("installed distribution metadata is incomplete") from error
        raw_name = _bounded_string(
            raw_name,
            label="installed distribution name",
            maximum_bytes=_MAX_NAME_BYTES,
        )
        version = _bounded_string(
            raw_version,
            label=f"installed distribution {raw_name!r} version",
            maximum_bytes=_MAX_VERSION_BYTES,
        )
        if raw_name != raw_name.strip() or version != version.strip():
            raise ValueError("installed distribution names and versions must be stripped")
        name = _normalize_distribution_name(raw_name)
        if name in observed_names:
            raise ValueError(f"duplicate normalized installed distribution name {name!r}")
        observed_names.add(name)
        candidates.append(_DistributionCandidate(name, version, distribution))
    return sorted(candidates, key=lambda candidate: candidate.name)


def _scan_distribution(
    candidate: _DistributionCandidate,
    *,
    roots: Sequence[_ImportRoot],
    scripts_root: Path,
) -> tuple[dict[str, Any], tuple[_FileRecord, ...]]:
    distribution = candidate.distribution
    direct_url_metadata = _reject_editable_direct_url(
        distribution,
        distribution_name=candidate.name,
    )
    entry_points = _entry_point_names(
        distribution,
        distribution_name=candidate.name,
    )
    try:
        declared_files = distribution.files
    except Exception as error:
        raise ValueError(
            f"could not enumerate files for distribution {candidate.name!r}"
        ) from error
    if declared_files is None:
        raise ValueError(
            f"distribution {candidate.name!r} has no declared file inventory"
        )

    records: list[_FileRecord] = []
    launchers: list[dict[str, str]] = []
    observed_logical_paths: set[str] = set()
    observed_launchers: set[tuple[str, str, str]] = set()
    observed_direct_url = False
    declared_file_count = 0
    for index, declared in enumerate(declared_files):
        if index >= _MAX_DECLARED_FILES_PER_DISTRIBUTION:
            raise ValueError(
                f"distribution {candidate.name!r} exceeds "
                f"{_MAX_DECLARED_FILES_PER_DISTRIBUTION} declared files"
            )
        declared_file_count += 1
        declared_text = _declared_path_text(
            declared,
            distribution_name=candidate.name,
        )
        parts = PurePosixPath(declared_text).parts
        excluded_bytecode = (
            any(part == _BYTECODE_DIRECTORY for part in parts)
            or declared_text.endswith(_BYTECODE_SUFFIX)
        )
        try:
            located = Path(os.fspath(distribution.locate_file(declared)))
        except Exception as error:
            raise ValueError(
                f"could not locate declared file {declared_text!r} for "
                f"distribution {candidate.name!r}"
            ) from error
        located = Path(os.path.abspath(located))
        rooted = _root_for_located_path(located, roots=roots)
        if rooted is None:
            launcher = _launcher_identity(
                located,
                declared_text=declared_text,
                scripts_root=scripts_root,
                entry_points=entry_points,
                distribution_name=candidate.name,
            )
            if launcher is None:
                raise ValueError(
                    f"distribution {candidate.name!r} declared file "
                    f"{declared_text!r} escapes active import roots and is not an "
                    "active entry-point launcher"
                )
            key = (launcher["group"], launcher["name"], launcher["artifact"])
            if key in observed_launchers:
                raise ValueError(
                    f"distribution {candidate.name!r} has duplicate launcher {key!r}"
                )
            _validate_excluded_launcher(
                located,
                scripts_root=scripts_root,
                label=f"distribution {candidate.name!r} excluded launcher {key!r}",
            )
            observed_launchers.add(key)
            launchers.append(launcher)
            continue
        root, relative = rooted
        if any(part in {"", ".", ".."} for part in parts) and not (
            _valid_ancillary_declared_path(
                declared_text,
                root=root,
                relative=relative,
            )
        ):
            raise ValueError(
                f"distribution {candidate.name!r} declared path {declared_text!r} "
                "contains an escape component"
            )
        if excluded_bytecode:
            continue
        if not relative.parts:
            raise ValueError(
                f"distribution {candidate.name!r} declares an import root as a file"
            )
        logical_path = f"{root.label}/{relative.as_posix()}"
        if logical_path in observed_logical_paths:
            raise ValueError(
                f"distribution {candidate.name!r} has duplicate declared path "
                f"{logical_path!r}"
            )
        observed_logical_paths.add(logical_path)
        return_payload = (
            relative.name == "direct_url.json"
            and relative.parent.name.endswith((".dist-info", ".egg-info"))
        )
        excluded_record = (
            relative.name == "RECORD"
            and relative.parent.name.endswith(".dist-info")
        )
        identity, payload = _stable_regular_file_identity(
            root.path,
            relative,
            label=f"distribution {candidate.name!r} file {relative.as_posix()!r}",
            maximum_bytes=(
                _MAX_DIRECT_URL_BYTES if return_payload else _MAX_FILE_BYTES
            ),
            return_payload=return_payload,
        )
        if excluded_record:
            continue
        if return_payload:
            assert payload is not None
            observed_direct_url = True
            direct_value = _strict_json_loads(
                payload,
                label=f"distribution {candidate.name!r} direct_url.json",
            )
            if not isinstance(direct_value, dict):
                raise ValueError(
                    f"distribution {candidate.name!r} direct_url.json must be an object"
                )
            directory_info = direct_value.get("dir_info")
            if directory_info is not None and not isinstance(directory_info, dict):
                raise ValueError(
                    f"distribution {candidate.name!r} direct_url.json dir_info "
                    "must be an object"
                )
            if isinstance(directory_info, dict) and directory_info.get("editable") is True:
                raise ValueError(
                    f"editable distribution {candidate.name!r} is forbidden"
                )
            if (
                direct_url_metadata is not None
                and _canonical_json_bytes(direct_value)
                != _canonical_json_bytes(direct_url_metadata)
            ):
                raise ValueError(
                    f"distribution {candidate.name!r} direct_url.json changed while "
                    "its inventory was hashed"
                )
            normalized_direct_url = _canonical_direct_url_identity(
                direct_value,
                distribution_name=candidate.name,
            )
            normalized_payload = _canonical_json_bytes(normalized_direct_url)
            identity = {
                **identity,
                "bytes": len(normalized_payload),
                "sha256": _sha256(normalized_payload),
            }
        records.append(
            _FileRecord(
                logical_path=logical_path,
                root_relative_path=relative.as_posix(),
                root=root,
                mode=identity["mode"],
                bytes=identity["bytes"],
                sha256=identity["sha256"],
            )
        )

    if direct_url_metadata is not None and not observed_direct_url:
        raise ValueError(
            f"distribution {candidate.name!r} direct_url.json is not declared in "
            "its file inventory"
        )
    if declared_file_count == 0:
        raise ValueError(
            f"distribution {candidate.name!r} has an empty declared file inventory"
        )
    if not records:
        raise ValueError(
            f"distribution {candidate.name!r} has no identity-bearing declared files"
        )
    return (
        {
            "name": candidate.name,
            "version": candidate.version,
            "content": _manifest_content(records),
            "excluded_entry_point_launchers": sorted(
                launchers,
                key=lambda item: (item["group"], item["name"], item["artifact"]),
            ),
        },
        tuple(records),
    )


def _is_excluded_bytecode_path(relative: PurePosixPath) -> bool:
    return (
        any(part == _BYTECODE_DIRECTORY for part in relative.parts)
        or relative.name.endswith(_BYTECODE_SUFFIX)
    )


def _walk_driver_tree(
    root: _ImportRoot,
    expected_records: Sequence[_FileRecord],
) -> tuple[_FileRecord, ...]:
    driver_relative = PurePosixPath(_PLAYWRIGHT_DRIVER_RELATIVE_ROOT)
    driver_path = root.path.joinpath(*driver_relative.parts)
    try:
        initial = driver_path.lstat()
    except OSError as error:
        raise ValueError("installed Playwright driver directory is missing") from error
    if stat.S_ISLNK(initial.st_mode) or not stat.S_ISDIR(initial.st_mode):
        raise ValueError("installed Playwright driver must be a non-symlink directory")

    actual: list[_FileRecord] = []

    def visit(directory: Path, relative_directory: PurePosixPath) -> None:
        try:
            before = directory.stat(follow_symlinks=False)
            children = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as error:
            raise ValueError("installed Playwright driver could not be traversed") from error
        for child in children:
            try:
                child.name.encode("utf-8")
                child_stat = child.stat(follow_symlinks=False)
            except (OSError, UnicodeError) as error:
                raise ValueError("installed Playwright driver has an unreadable entry") from error
            child_relative = relative_directory / child.name
            if stat.S_ISLNK(child_stat.st_mode):
                raise ValueError(
                    "installed Playwright driver contains a symbolic link: "
                    f"{child_relative.as_posix()!r}"
                )
            if stat.S_ISDIR(child_stat.st_mode):
                if child.name != _BYTECODE_DIRECTORY:
                    visit(Path(child.path), child_relative)
                continue
            if not stat.S_ISREG(child_stat.st_mode):
                raise ValueError(
                    "installed Playwright driver contains a non-regular entry: "
                    f"{child_relative.as_posix()!r}"
                )
            if _is_excluded_bytecode_path(child_relative):
                continue
            root_relative = driver_relative / child_relative
            identity, _ = _stable_regular_file_identity(
                root.path,
                root_relative,
                label=(
                    "installed Playwright driver file "
                    f"{child_relative.as_posix()!r}"
                ),
            )
            actual.append(
                _FileRecord(
                    logical_path=f"{root.label}/{root_relative.as_posix()}",
                    root_relative_path=root_relative.as_posix(),
                    root=root,
                    mode=identity["mode"],
                    bytes=identity["bytes"],
                    sha256=identity["sha256"],
                )
            )
        try:
            after = directory.stat(follow_symlinks=False)
        except OSError as error:
            raise ValueError("installed Playwright driver changed during traversal") from error
        fields = ("st_dev", "st_ino", "st_mode", "st_mtime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in fields):
            raise ValueError("installed Playwright driver changed during traversal")

    visit(driver_path, PurePosixPath())
    expected = {
        record.logical_path: (record.mode, record.bytes, record.sha256)
        for record in expected_records
    }
    observed = {
        record.logical_path: (record.mode, record.bytes, record.sha256)
        for record in actual
    }
    if expected != observed:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        changed = sorted(
            path for path in set(expected) & set(observed) if expected[path] != observed[path]
        )
        raise ValueError(
            "installed Playwright driver does not exactly match its declared distribution "
            f"files: missing={missing}, extra={extra}, changed={changed}"
        )
    if not actual:
        raise ValueError("installed Playwright driver contains no identity-bearing files")
    return tuple(actual)


def _python_identity() -> dict[str, Any]:
    implementation = _bounded_string(
        platform.python_implementation(),
        label="Python implementation",
        maximum_bytes=_MAX_NAME_BYTES,
    )
    version = _bounded_string(
        platform.python_version(),
        label="Python version",
        maximum_bytes=_MAX_VERSION_BYTES,
    )
    cache_tag = _bounded_string(
        getattr(sys.implementation, "cache_tag", None),
        label="Python cache tag",
        maximum_bytes=_MAX_NAME_BYTES,
    )
    soabi = _bounded_string(
        sysconfig.get_config_var("SOABI"),
        label="Python SOABI",
        maximum_bytes=_MAX_NAME_BYTES,
    )
    try:
        executable = Path(sys.executable).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("Python executable cannot be resolved") from error
    identity, _ = _stable_regular_file_identity(
        executable.parent,
        PurePosixPath(executable.name),
        label="resolved Python executable",
    )
    return {
        "implementation": implementation,
        "version": version,
        "cache_tag": cache_tag,
        "soabi": soabi,
        "executable": {"bytes": identity["bytes"], "sha256": identity["sha256"]},
    }


def _operating_system_identity() -> dict[str, Any]:
    macos_release, macos_version_info, macos_machine = platform.mac_ver()
    if len(macos_version_info) != 3:
        raise ValueError("platform.mac_ver() must return a three-part version tuple")
    return {
        "system": _bounded_string(
            platform.system(),
            label="OS system",
            maximum_bytes=_MAX_VERSION_BYTES,
        ),
        "release": _bounded_string(
            platform.release(),
            label="OS release",
            maximum_bytes=_MAX_VERSION_BYTES,
        ),
        "version": _bounded_string(
            platform.version(),
            label="OS version",
            maximum_bytes=_MAX_VERSION_BYTES,
        ),
        "machine": _bounded_string(
            platform.machine(),
            label="OS machine",
            maximum_bytes=_MAX_VERSION_BYTES,
        ),
        "macos": {
            "release": _bounded_string(
                macos_release,
                label="macOS release",
                maximum_bytes=_MAX_VERSION_BYTES,
                allow_empty=True,
            ),
            "version_info": [
                _bounded_string(
                    item,
                    label=f"macOS version_info[{index}]",
                    maximum_bytes=_MAX_VERSION_BYTES,
                    allow_empty=True,
                )
                for index, item in enumerate(macos_version_info)
            ],
            "machine": _bounded_string(
                macos_machine,
                label="macOS machine",
                maximum_bytes=_MAX_VERSION_BYTES,
                allow_empty=True,
            ),
        },
    }


def _validate_content(value: Any, *, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    _exact_keys(value, _CONTENT_KEYS, label=label)
    _nonnegative_int(value["files"], label=f"{label}.files")
    _nonnegative_int(value["bytes"], label=f"{label}.bytes")
    _sha256_string(value["sha256"], label=f"{label}.sha256")


def _validate_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("BrowserGym runtime manifest must be a JSON object")
    _exact_keys(manifest, _TOP_LEVEL_KEYS, label="BrowserGym runtime manifest")
    if manifest["kind"] != BROWSERGYM_RUNTIME_MANIFEST_KIND:
        raise ValueError("BrowserGym runtime manifest kind mismatch")
    if manifest["schema_version"] != BROWSERGYM_RUNTIME_MANIFEST_SCHEMA_VERSION:
        raise ValueError("BrowserGym runtime manifest schema version mismatch")

    scope = manifest["scope"]
    if not isinstance(scope, dict):
        raise ValueError("BrowserGym runtime manifest scope must be an object")
    _exact_keys(scope, _SCOPE_KEYS, label="BrowserGym runtime manifest scope")
    roots = scope["import_roots"]
    if roots not in (["purelib+platlib"], ["platlib", "purelib"]):
        raise ValueError("BrowserGym runtime manifest import-root labels are invalid")
    ancillary_roots = scope["ancillary_roots"]
    if ancillary_roots not in ([], ["data-include"]):
        raise ValueError("BrowserGym runtime manifest ancillary-root labels are invalid")
    if scope["distribution_file_identity"] != _DISTRIBUTION_FILE_POLICY:
        raise ValueError("BrowserGym runtime manifest distribution-file policy drift")
    if scope["excluded_bytecode"] != [".pyc", "__pycache__"]:
        raise ValueError("BrowserGym runtime manifest bytecode exclusion drift")
    if scope["outside_import_roots"] != _OUTSIDE_ROOT_POLICY:
        raise ValueError("BrowserGym runtime manifest outside-root policy drift")
    if scope["installer_metadata"] != _INSTALLER_METADATA_POLICY:
        raise ValueError("BrowserGym runtime manifest installer-metadata policy drift")

    python_identity = manifest["python"]
    if not isinstance(python_identity, dict):
        raise ValueError("BrowserGym runtime manifest Python identity must be an object")
    _exact_keys(python_identity, _PYTHON_KEYS, label="Python identity")
    for key in ("implementation", "version", "cache_tag", "soabi"):
        _bounded_string(
            python_identity[key],
            label=f"Python identity {key}",
            maximum_bytes=_MAX_VERSION_BYTES,
        )
    executable = python_identity["executable"]
    if not isinstance(executable, dict):
        raise ValueError("Python executable identity must be an object")
    _exact_keys(executable, _IDENTITY_KEYS, label="Python executable identity")
    _positive_int(executable["bytes"], label="Python executable bytes")
    _sha256_string(executable["sha256"], label="Python executable sha256")

    operating_system = manifest["operating_system"]
    if not isinstance(operating_system, dict):
        raise ValueError("operating-system identity must be an object")
    _exact_keys(operating_system, _OS_KEYS, label="operating-system identity")
    for key in ("system", "release", "version", "machine"):
        _bounded_string(
            operating_system[key],
            label=f"operating-system identity {key}",
            maximum_bytes=_MAX_VERSION_BYTES,
        )
    macos = operating_system["macos"]
    if not isinstance(macos, dict):
        raise ValueError("macOS identity must be an object")
    _exact_keys(macos, _MACOS_KEYS, label="macOS identity")
    for key in ("release", "machine"):
        _bounded_string(
            macos[key],
            label=f"macOS identity {key}",
            maximum_bytes=_MAX_VERSION_BYTES,
            allow_empty=True,
        )
    version_info = macos["version_info"]
    if not isinstance(version_info, list) or len(version_info) != 3:
        raise ValueError("macOS version_info must be a three-item list")
    for index, item in enumerate(version_info):
        _bounded_string(
            item,
            label=f"macOS version_info[{index}]",
            maximum_bytes=_MAX_VERSION_BYTES,
            allow_empty=True,
        )

    distributions = manifest["installed_distributions"]
    if not isinstance(distributions, list) or not distributions:
        raise ValueError("installed_distributions must be a non-empty list")
    if len(distributions) > _MAX_DISTRIBUTIONS:
        raise ValueError("installed_distributions exceeds the hard count limit")
    previous_name: str | None = None
    playwright_distribution: Mapping[str, Any] | None = None
    for index, distribution in enumerate(distributions):
        label = f"installed_distributions[{index}]"
        if not isinstance(distribution, dict):
            raise ValueError(f"{label} must be an object")
        _exact_keys(distribution, _DISTRIBUTION_KEYS, label=label)
        name = _bounded_string(
            distribution["name"],
            label=f"{label}.name",
            maximum_bytes=_MAX_NAME_BYTES,
        )
        if _normalize_distribution_name(name) != name:
            raise ValueError(f"{label}.name is not normalized")
        if previous_name is not None and name <= previous_name:
            raise ValueError("installed_distributions must be sorted with unique names")
        previous_name = name
        _bounded_string(
            distribution["version"],
            label=f"{label}.version",
            maximum_bytes=_MAX_VERSION_BYTES,
        )
        _validate_content(distribution["content"], label=f"{label}.content")
        if distribution["content"]["files"] == 0:
            raise ValueError(f"{label}.content must contain identity-bearing files")
        launchers = distribution["excluded_entry_point_launchers"]
        if not isinstance(launchers, list):
            raise ValueError(f"{label}.excluded_entry_point_launchers must be a list")
        launcher_keys: list[tuple[str, str, str]] = []
        for launcher_index, launcher in enumerate(launchers):
            launcher_label = (
                f"{label}.excluded_entry_point_launchers[{launcher_index}]"
            )
            if not isinstance(launcher, dict):
                raise ValueError(f"{launcher_label} must be an object")
            _exact_keys(launcher, _LAUNCHER_KEYS, label=launcher_label)
            if launcher["group"] not in _ENTRY_POINT_GROUPS:
                raise ValueError(f"{launcher_label}.group is invalid")
            launcher_keys.append(
                (
                    launcher["group"],
                    _bounded_string(
                        launcher["name"],
                        label=f"{launcher_label}.name",
                        maximum_bytes=_MAX_NAME_BYTES,
                    ),
                    _bounded_string(
                        launcher["artifact"],
                        label=f"{launcher_label}.artifact",
                        maximum_bytes=_MAX_NAME_BYTES,
                    ),
                )
            )
        if launcher_keys != sorted(set(launcher_keys)):
            raise ValueError(f"{label} launcher identities must be sorted and unique")
        if name == "playwright":
            playwright_distribution = distribution

    if playwright_distribution is None:
        raise ValueError("installed Playwright distribution is required")
    driver = manifest["playwright_driver"]
    if not isinstance(driver, dict):
        raise ValueError("playwright_driver must be an object")
    _exact_keys(driver, _DRIVER_KEYS, label="playwright_driver")
    if driver["distribution"] != "playwright":
        raise ValueError("playwright_driver distribution mismatch")
    if driver["import_root"] not in roots:
        raise ValueError("playwright_driver import root is outside manifest scope")
    if driver["relative_root"] != _PLAYWRIGHT_DRIVER_RELATIVE_ROOT:
        raise ValueError("playwright_driver relative root mismatch")
    _validate_content(driver["content"], label="playwright_driver.content")
    if driver["content"]["files"] == 0:
        raise ValueError("playwright_driver must contain identity-bearing files")
    if driver["content"]["bytes"] > playwright_distribution["content"]["bytes"]:
        raise ValueError("playwright_driver bytes exceed Playwright distribution bytes")
    if driver["content"]["files"] > playwright_distribution["content"]["files"]:
        raise ValueError("playwright_driver files exceed Playwright distribution files")

    declared_self_hash = _sha256_string(
        manifest["manifest_self_sha256"],
        label="BrowserGym runtime manifest self-hash",
    )
    without_hash = dict(manifest)
    without_hash.pop("manifest_self_sha256")
    observed_self_hash = _sha256(_canonical_json_bytes(without_hash))
    if not hmac.compare_digest(declared_self_hash, observed_self_hash):
        raise ValueError("BrowserGym runtime manifest self-hash mismatch")
    return manifest


def build_active_environment_manifest() -> dict[str, Any]:
    """Build a deterministic, self-hashed manifest of the active capture environment.

    The builder fails closed on incomplete distribution inventories, normalized-name collisions,
    editable installs, declared files outside the active import roots (apart from validated
    entry-point launchers), symlinks, special files, missing files, unsafe paths, and observable
    file or directory changes during hashing.
    """

    roots, scripts_root = _active_import_roots()
    distributions: list[dict[str, Any]] = []
    playwright_records: tuple[_FileRecord, ...] | None = None
    total_files = 0
    total_bytes = 0
    for candidate in _distribution_candidates(roots):
        distribution, records = _scan_distribution(
            candidate,
            roots=roots,
            scripts_root=scripts_root,
        )
        total_files += distribution["content"]["files"]
        total_bytes += distribution["content"]["bytes"]
        if total_files > _MAX_TOTAL_DECLARED_FILES:
            raise ValueError(
                f"manifest declared files exceed {_MAX_TOTAL_DECLARED_FILES}"
            )
        if total_bytes > _MAX_TOTAL_FILE_BYTES:
            raise ValueError(
                f"manifest file bytes exceed the {_MAX_TOTAL_FILE_BYTES}-byte limit"
            )
        distributions.append(distribution)
        if candidate.name == "playwright":
            playwright_records = records
    if playwright_records is None:
        raise ValueError("installed Playwright distribution is required")

    driver_prefix = f"{_PLAYWRIGHT_DRIVER_RELATIVE_ROOT}/"
    driver_declared = tuple(
        record
        for record in playwright_records
        if record.root_relative_path.startswith(driver_prefix)
    )
    driver_roots = {record.root for record in driver_declared}
    if len(driver_roots) != 1:
        raise ValueError(
            "Playwright driver declared files must occupy exactly one active import root"
        )
    driver_root = next(iter(driver_roots))
    driver_records = _walk_driver_tree(driver_root, driver_declared)

    manifest_without_hash: dict[str, Any] = {
        "kind": BROWSERGYM_RUNTIME_MANIFEST_KIND,
        "schema_version": BROWSERGYM_RUNTIME_MANIFEST_SCHEMA_VERSION,
        "scope": {
            "import_roots": [
                root.label for root in roots if root.label != "data-include"
            ],
            "ancillary_roots": [
                root.label for root in roots if root.label == "data-include"
            ],
            "distribution_file_identity": _DISTRIBUTION_FILE_POLICY,
            "excluded_bytecode": [".pyc", "__pycache__"],
            "outside_import_roots": _OUTSIDE_ROOT_POLICY,
            "installer_metadata": _INSTALLER_METADATA_POLICY,
        },
        "python": _python_identity(),
        "operating_system": _operating_system_identity(),
        "installed_distributions": distributions,
        "playwright_driver": {
            "distribution": "playwright",
            "import_root": driver_root.label,
            "relative_root": _PLAYWRIGHT_DRIVER_RELATIVE_ROOT,
            "content": _manifest_content(driver_records, driver_relative=True),
        },
    }
    manifest = {
        **manifest_without_hash,
        "manifest_self_sha256": _sha256(_canonical_json_bytes(manifest_without_hash)),
    }
    payload = _canonical_json_bytes(manifest, newline=True)
    if len(payload) > DEFAULT_MAX_MANIFEST_BYTES:
        raise ValueError(
            f"BrowserGym runtime manifest exceeds {DEFAULT_MAX_MANIFEST_BYTES} bytes"
        )
    return _validate_manifest(manifest)


def _unlink_if_same_file(path: Path, *, device: int, inode: int) -> None:
    try:
        observed = path.lstat()
    except FileNotFoundError:
        return
    if (
        not stat.S_ISLNK(observed.st_mode)
        and observed.st_dev == device
        and observed.st_ino == inode
    ):
        path.unlink()


def freeze_active_environment_manifest(path: str | Path) -> dict[str, Any]:
    """Build and publish one new canonical runtime manifest without overwriting any path.

    Publication uses a fully written and synced temporary file followed by a same-directory hard
    link, so concurrent creation is also no-clobber. A failure after publication removes only the
    inode created by this call.
    """

    manifest = build_active_environment_manifest()
    payload = _canonical_json_bytes(manifest, newline=True)
    if len(payload) > DEFAULT_MAX_MANIFEST_BYTES:
        raise ValueError(
            f"BrowserGym runtime manifest exceeds {DEFAULT_MAX_MANIFEST_BYTES} bytes"
        )
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite existing BrowserGym runtime manifest: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    published_identity: tuple[int, int] | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o644)
            temporary_stat = os.fstat(handle.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as error:
            raise FileExistsError(
                "refusing to overwrite concurrently created BrowserGym runtime "
                f"manifest: {destination}"
            ) from error
        published_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
        verified = load_and_verify_environment_manifest(
            destination,
            expected_sha256=manifest["manifest_self_sha256"],
            verify_active=False,
        )
        if _canonical_json_bytes(verified, newline=True) != payload:
            raise RuntimeError("published BrowserGym runtime manifest changed")
    except Exception:
        if published_identity is not None:
            _unlink_if_same_file(
                destination,
                device=published_identity[0],
                inode=published_identity[1],
            )
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return manifest


def load_and_verify_environment_manifest(
    path: str | Path,
    expected_sha256: str | None = None,
    verify_active: bool = True,
) -> dict[str, Any]:
    """Load and strictly verify a frozen runtime manifest.

    ``expected_sha256`` pins the manifest's lowercase self SHA-256. When ``verify_active`` is true,
    the validated manifest must be canonical-byte identical to a freshly rebuilt active manifest.
    Reads are bounded, symlink-safe, and checked for observable changes during the snapshot.
    """

    manifest_path = Path(path)
    raw = _read_bounded_stable_file(
        manifest_path,
        maximum_bytes=DEFAULT_MAX_MANIFEST_BYTES,
        label="BrowserGym runtime manifest",
    )
    value = _strict_json_loads(raw, label="BrowserGym runtime manifest")
    manifest = _validate_manifest(value)
    canonical = _canonical_json_bytes(manifest, newline=True)
    if raw != canonical:
        raise ValueError("BrowserGym runtime manifest is not canonical JSON")
    if expected_sha256 is not None:
        expected = _sha256_string(
            expected_sha256,
            label="expected BrowserGym runtime manifest SHA-256",
        )
        if not hmac.compare_digest(expected, manifest["manifest_self_sha256"]):
            raise ValueError("BrowserGym runtime manifest expected SHA-256 mismatch")
    if verify_active:
        active = build_active_environment_manifest()
        if canonical != _canonical_json_bytes(active, newline=True):
            raise ValueError(
                "BrowserGym runtime manifest differs from the active environment"
            )
    return manifest


__all__ = [
    "BROWSERGYM_RUNTIME_MANIFEST_KIND",
    "BROWSERGYM_RUNTIME_MANIFEST_SCHEMA_VERSION",
    "DEFAULT_MAX_MANIFEST_BYTES",
    "build_active_environment_manifest",
    "freeze_active_environment_manifest",
    "load_and_verify_environment_manifest",
]
