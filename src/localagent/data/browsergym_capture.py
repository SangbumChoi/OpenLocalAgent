"""Controlled BrowserGym/MiniWoB reset-goal capture with reproducibility receipts.

This module is intentionally dependency-light. BrowserGym, Gymnasium, and Playwright are imported
only when the real environment factory is invoked. The producer calls ``reset(seed=...)`` once per
planned episode, reads only ``observation["goal"]``, and closes the environment without taking an
action. It does not score episodes or create fresh-evaluation evidence.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from localagent.data.browsergym_prompts import (
    DEFAULT_MAX_CAPTURE_BYTES,
    DEFAULT_MAX_LINE_BYTES,
    DEFAULT_MAX_PROMPT_BYTES,
    PRODUCTION_BROWSERGYM_REVISION,
    PRODUCTION_BROWSERGYM_VERSION,
    PRODUCTION_CHROMIUM_REVISION,
    PRODUCTION_CHROMIUM_VERSION,
    PRODUCTION_EPISODES,
    PRODUCTION_FIXED_SEEDS,
    PRODUCTION_LOCAL_POLICY_EXCLUSIONS,
    PRODUCTION_MAX_STEPS,
    PRODUCTION_MINIWOB_REVISION,
    PRODUCTION_PLAYWRIGHT_VERSION,
    PRODUCTION_RUNTIME_MANIFEST_IDENTITY,
    PRODUCTION_SIMILARITY_GROUPS,
    PRODUCTION_TASK_GROUPS,
    PRODUCTION_TASK_VARIANTS,
)
from localagent.data.browsergym_runtime_manifest import (
    load_and_verify_environment_manifest,
)

BROWSERGYM_CAPTURE_PRODUCER = "browsergym-miniwob-controlled-reset-goals-v3"
BROWSERGYM_CAPTURE_RECEIPT_KIND = "localagent_browsergym_capture_producer_receipt"
BROWSERGYM_CAPTURE_RECEIPT_SCHEMA_VERSION = 3

PRODUCTION_LOCALE = "en-US"
PRODUCTION_TIMEZONE_ID = "UTC"
PRODUCTION_HEADLESS = True
PRODUCTION_VIEWPORT = {"width": 1280, "height": 720}
PRODUCTION_DEVICE_SCALE_FACTOR = 1.0
PRODUCTION_ACTION_SET = "highlevel-default-unused-reset-only"
PRODUCTION_OBSERVATION_MODE = "processed-dom-axtree-screenshot"
PRODUCTION_TIMEOUT_SECONDS = 30.0

DEFAULT_MAX_RECEIPT_BYTES = 4 * 1024 * 1024

_CAPTURE_BOUNDARY = (
    "reset-returned observation.goal only; reset info discarded; no actions, rewards, "
    "labels, episode steps, scores, or fresh-evaluation claim"
)
_GOAL_PROVENANCE = "observation.goal returned by env.reset(seed)"
_PUBLICATION_CONTRACT = (
    "two-file non-clobber publication with rollback on ordinary errors; abrupt process "
    "termination may leave capture without receipt, so consumers must require receipt verification"
)
_IMPORT_SCOPE = {
    "policy": "isolated-attested-browsergym-checkout-src-v1",
    "browsergym_namespace_paths": [
        "browsergym/core/src/browsergym",
        "browsergym/miniwob/src/browsergym",
    ],
    "required_modules": {
        "browsergym.core": "browsergym/core/src/browsergym/core/__init__.py",
        "browsergym.core.action.highlevel": (
            "browsergym/core/src/browsergym/core/action/highlevel.py"
        ),
        "browsergym.miniwob": "browsergym/miniwob/src/browsergym/miniwob/__init__.py",
    },
    "preloaded_browsergym_modules": "forbidden",
    "ambient_namespace_paths": "excluded",
    "sys_path_restored_after_capture": True,
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA1_RE = re.compile(r"[0-9a-f]{40}\Z")
_CHROMIUM_INSTALLATION_RE = re.compile(r"(?:chromium|chromium_headless_shell)-([0-9]+)\Z")
_CHROMIUM_VERSION_RE = re.compile(
    r"(?:Chromium|Google Chrome|HeadlessChrome)\s+([0-9]+(?:\.[0-9]+){3})"
)
_CAPTURE_KEYS = frozenset(
    {
        "task_name",
        "seed",
        "goal",
        "similarity_group",
        "split",
        "source_pins",
        "runtime_pins",
    }
)
_SOURCE_PIN_KEYS = frozenset(
    {"browsergym_revision", "browsergym_version", "miniwob_revision"}
)
_SOURCE_EVIDENCE_KEYS = frozenset({"source_pins", "repositories"})
_REPOSITORY_KEYS = frozenset({"browsergym", "miniwob"})
_REPOSITORY_ATTESTATION_KEYS = frozenset(
    {"revision", "git_tree_sha1", "tracked_tree", "worktree"}
)
_TRACKED_TREE_KEYS = frozenset(
    {"bytes", "records", "sha256", "verified_worktree_sha256"}
)
_RUNTIME_PIN_KEYS = frozenset(
    {
        "playwright_version",
        "chromium_revision",
        "chromium_version",
        "python_version",
        "os",
        "architecture",
        "locale",
        "timezone_id",
        "headless",
        "viewport",
        "device_scale_factor",
        "action_set",
        "observation_mode",
        "max_steps",
        "playwright_operation_timeout_seconds",
        "browser_executable",
        "browser_installation",
        "environment_manifest",
    }
)
_RUNTIME_EVIDENCE_KEYS = frozenset({"runtime_pins", "attestation"})
_RUNTIME_ATTESTATION_KEYS = frozenset(
    {"installed_distributions", "python_implementation", "browser"}
)
_DISTRIBUTION_KEYS = frozenset(
    {"browsergym-core", "browsergym-miniwob", "gymnasium", "playwright"}
)
_BROWSER_ATTESTATION_KEYS = frozenset(
    {
        "reported_version",
        "installation_entries",
        "installation_files",
        "installation_symlinks",
        "executable_scope",
    }
)
_CAPTURE_RECEIPT_KEYS = frozenset(
    {
        "bytes",
        "sha256",
        "rows",
        "row_keys",
        "canonical_sorted_jsonl",
        "goal_provenance",
    }
)
_RECEIPT_KEYS = frozenset(
    {
        "kind",
        "schema_version",
        "producer",
        "boundary",
        "capture",
        "plan",
        "controls",
        "source_pins",
        "source_attestation",
        "runtime_pins",
        "runtime_attestation",
        "environment_manifest",
        "import_scope",
        "publication",
        "receipt_self_sha256",
    }
)


@dataclass(frozen=True)
class BrowserGymCaptureSettings:
    """Fixed controls for the paper's BrowserGym reset capture."""

    locale: str = PRODUCTION_LOCALE
    timezone_id: str = PRODUCTION_TIMEZONE_ID
    headless: bool = PRODUCTION_HEADLESS
    viewport_width: int = PRODUCTION_VIEWPORT["width"]
    viewport_height: int = PRODUCTION_VIEWPORT["height"]
    device_scale_factor: float = PRODUCTION_DEVICE_SCALE_FACTOR
    action_set: str = PRODUCTION_ACTION_SET
    observation_mode: str = PRODUCTION_OBSERVATION_MODE
    max_steps: int = PRODUCTION_MAX_STEPS
    playwright_operation_timeout_seconds: float = PRODUCTION_TIMEOUT_SECONDS

    def controls(self) -> dict[str, Any]:
        """Return the row-level runtime controls."""

        return {
            "locale": self.locale,
            "timezone_id": self.timezone_id,
            "headless": self.headless,
            "viewport": {
                "width": self.viewport_width,
                "height": self.viewport_height,
            },
            "device_scale_factor": self.device_scale_factor,
            "action_set": self.action_set,
            "observation_mode": self.observation_mode,
            "max_steps": self.max_steps,
            "playwright_operation_timeout_seconds": (
                self.playwright_operation_timeout_seconds
            ),
        }

    def validate_production(self) -> None:
        """Fail if any capture control differs from the frozen production protocol."""

        expected = {
            "locale": PRODUCTION_LOCALE,
            "timezone_id": PRODUCTION_TIMEZONE_ID,
            "headless": PRODUCTION_HEADLESS,
            "viewport": dict(PRODUCTION_VIEWPORT),
            "device_scale_factor": PRODUCTION_DEVICE_SCALE_FACTOR,
            "action_set": PRODUCTION_ACTION_SET,
            "observation_mode": PRODUCTION_OBSERVATION_MODE,
            "max_steps": PRODUCTION_MAX_STEPS,
            "playwright_operation_timeout_seconds": PRODUCTION_TIMEOUT_SECONDS,
        }
        if not _canonical_equal(self.controls(), expected):
            raise ValueError(
                "BrowserGym production controls are immutable; "
                f"expected {expected}, observed {self.controls()}"
            )


@dataclass(frozen=True)
class BrowserGymCaptureEpisode:
    """One reset-only episode in the exact production plan."""

    task_name: str
    seed: int
    similarity_group: int
    split: str = "test"


class ResetEnvironment(Protocol):
    """Narrow environment surface used by the capture producer."""

    def reset(self, *, seed: int) -> tuple[Mapping[str, Any], Any]: ...

    def close(self) -> None: ...


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


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is forbidden")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _strict_json_loads(payload: str, *, label: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"{label} is not strict JSON") from error


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], *, label: str) -> None:
    observed = frozenset(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{label} schema drift: missing={missing}, extra={extra}")


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _finite_positive_number(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{label} must be a finite positive number")
    return float(value)


def _canonical_equal(left: Any, right: Any) -> bool:
    return _canonical_json_bytes(left) == _canonical_json_bytes(right)


def _frozen_json(value: Any, *, label: str) -> Any:
    try:
        payload = _canonical_json_bytes(value)
        return _strict_json_loads(payload.decode("utf-8"), label=label)
    except (TypeError, UnicodeError, ValueError) as error:
        raise ValueError(f"{label} is not canonical-JSON-compatible") from error


def production_capture_plan() -> tuple[BrowserGymCaptureEpisode, ...]:
    """Return the exact 60-task by four-seed plan in canonical order."""

    if len(PRODUCTION_TASK_GROUPS) != PRODUCTION_TASK_VARIANTS:
        raise RuntimeError("BrowserGym task-variant constants are inconsistent")
    if len(set(PRODUCTION_TASK_GROUPS.values())) != PRODUCTION_SIMILARITY_GROUPS:
        raise RuntimeError("BrowserGym similarity-group constants are inconsistent")
    episodes = tuple(
        BrowserGymCaptureEpisode(
            task_name=task_name,
            seed=seed,
            similarity_group=PRODUCTION_TASK_GROUPS[task_name],
        )
        for task_name in sorted(PRODUCTION_TASK_GROUPS)
        for seed in PRODUCTION_FIXED_SEEDS
    )
    if len(episodes) != PRODUCTION_EPISODES:
        raise RuntimeError("BrowserGym episode-count constants are inconsistent")
    return episodes


def _plan_receipt() -> dict[str, Any]:
    return {
        "split": "test",
        "fixed_seeds": list(PRODUCTION_FIXED_SEEDS),
        "episode_rows": PRODUCTION_EPISODES,
        "task_variants": PRODUCTION_TASK_VARIANTS,
        "similarity_group_count": PRODUCTION_SIMILARITY_GROUPS,
        "task_groups": dict(sorted(PRODUCTION_TASK_GROUPS.items())),
        "localagent_policy_exclusions": list(PRODUCTION_LOCAL_POLICY_EXCLUSIONS),
    }


def _run_git(checkout: Path, *arguments: str) -> bytes:
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "LANG": "C",
            "LC_ALL": "C",
        }
    )
    command = ["git", "-C", str(checkout), *arguments]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            timeout=30,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"failed to inspect git checkout {checkout}") from error
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(
            f"git inspection failed for {checkout}: {detail or 'unknown git error'}"
        )
    return result.stdout


def _reject_git_index_and_ignored_bypasses(checkout: Path, *, label: str) -> None:
    tagged = _run_git(checkout, "ls-files", "-v", "-z")
    for record in tagged.split(b"\0"):
        if not record:
            continue
        if len(record) < 3 or record[1:2] != b" ":
            raise ValueError(f"{label} git index flag output is malformed")
        tag = chr(record[0])
        if tag != "H":
            raise ValueError(
                f"{label} git index flags are unsupported ({tag!r}); "
                "assume-unchanged and skip-worktree entries are forbidden"
            )
    ignored = _run_git(
        checkout,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "-z",
    )
    if ignored:
        raise ValueError(
            f"{label} checkout contains ignored files outside the attested revision"
        )


def _git_blob_sha1(payload: bytes) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {len(payload)}\0".encode("ascii"))
    digest.update(payload)
    return digest.hexdigest()


def _stable_git_regular_identity(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link")
    try:
        initial = path.stat()
    except OSError as error:
        raise ValueError(f"{label} is missing or unreadable") from error
    if not stat.S_ISREG(initial.st_mode):
        raise ValueError(f"{label} must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} could not be opened safely") from error
    sha1 = hashlib.sha1()
    sha1.update(f"blob {initial.st_size}\0".encode("ascii"))
    sha256 = hashlib.sha256()
    observed_bytes = 0
    try:
        before = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            observed_bytes += len(chunk)
            sha1.update(chunk)
            sha256.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity_fields):
        raise ValueError(f"{label} changed while it was hashed")
    if observed_bytes != before.st_size:
        raise ValueError(f"{label} byte count changed while it was hashed")
    return {
        "bytes": observed_bytes,
        "git_blob_sha1": sha1.hexdigest(),
        "mode": stat.S_IMODE(before.st_mode),
        "sha256": sha256.hexdigest(),
    }


def _verified_worktree_sha256(
    checkout: Path,
    *,
    listing: bytes,
    label: str,
) -> str:
    digest = hashlib.sha256()
    for raw_record in listing.split(b"\0"):
        if not raw_record:
            continue
        try:
            metadata, raw_path = raw_record.split(b"\t", 1)
            raw_mode, raw_kind, raw_object_id = metadata.split(b" ", 2)
            relative_name = raw_path.decode("utf-8")
            mode = raw_mode.decode("ascii")
            kind = raw_kind.decode("ascii")
            object_id = raw_object_id.decode("ascii")
        except (UnicodeError, ValueError) as error:
            raise ValueError(f"{label} git tree listing is malformed") from error
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts or relative_name == "":
            raise ValueError(f"{label} git tree contains an unsafe path")
        candidate = checkout.joinpath(*relative.parts)
        try:
            candidate.parent.resolve().relative_to(checkout)
        except ValueError as error:
            raise ValueError(f"{label} tracked path escapes the checkout") from error
        entry_label = f"{label} tracked entry {relative_name!r}"
        if kind != "blob":
            raise ValueError(f"{entry_label} has unsupported git object kind {kind!r}")
        if mode in {"100644", "100755"}:
            identity = _stable_git_regular_identity(candidate, label=entry_label)
            expected_executable = mode == "100755"
            observed_executable = bool(identity["mode"] & 0o111)
            if observed_executable != expected_executable:
                raise ValueError(f"{entry_label} executable-mode drift")
            if identity["git_blob_sha1"] != object_id:
                raise ValueError(f"{entry_label} content differs from HEAD")
            digest.update(
                _canonical_json_bytes(
                    {
                        "bytes": identity["bytes"],
                        "kind": "file",
                        "mode": mode,
                        "path": relative_name,
                        "sha256": identity["sha256"],
                    },
                    newline=True,
                )
            )
        elif mode == "120000":
            if not candidate.is_symlink():
                raise ValueError(f"{entry_label} symlink type differs from HEAD")
            try:
                target = os.readlink(candidate)
                target_payload = os.fsencode(target)
            except OSError as error:
                raise ValueError(f"{entry_label} symlink is unreadable") from error
            if _git_blob_sha1(target_payload) != object_id:
                raise ValueError(f"{entry_label} symlink target differs from HEAD")
            digest.update(
                _canonical_json_bytes(
                    {
                        "kind": "symlink",
                        "path": relative_name,
                        "target_sha256": _sha256(target_payload),
                    },
                    newline=True,
                )
            )
        else:
            raise ValueError(f"{entry_label} has unsupported git mode {mode!r}")
    return digest.hexdigest()


def attest_git_checkout(
    checkout: str | Path,
    *,
    expected_revision: str,
    label: str,
) -> dict[str, Any]:
    """Attest an exact, clean local git checkout without trusting its path."""

    root = Path(checkout)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"{label} checkout must be a non-symlink directory: {root}")
    if _GIT_SHA1_RE.fullmatch(expected_revision) is None:
        raise ValueError(f"{label} expected revision must be lowercase 40-hex")
    resolved = root.resolve()
    top_level_payload = _run_git(resolved, "rev-parse", "--show-toplevel")
    try:
        top_level = Path(top_level_payload.decode("utf-8").strip()).resolve()
    except UnicodeError as error:
        raise ValueError(f"{label} git top-level path is not UTF-8") from error
    if top_level != resolved:
        raise ValueError(
            f"{label} checkout must name the git top-level exactly; got {top_level}"
        )
    revision = _run_git(resolved, "rev-parse", "--verify", "HEAD^{commit}").decode(
        "ascii"
    ).strip()
    if revision != expected_revision:
        raise ValueError(
            f"{label} revision drift: expected {expected_revision}, observed {revision}"
        )
    dirty = _run_git(
        resolved,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    if dirty:
        raise ValueError(f"{label} checkout is dirty or contains untracked files")
    _reject_git_index_and_ignored_bypasses(resolved, label=label)
    git_tree_sha1 = _run_git(resolved, "rev-parse", "HEAD^{tree}").decode("ascii").strip()
    if _GIT_SHA1_RE.fullmatch(git_tree_sha1) is None:
        raise ValueError(f"{label} git tree identity is invalid")
    listing = _run_git(resolved, "ls-tree", "-r", "-z", "--full-tree", "HEAD")
    verified_worktree_sha256 = _verified_worktree_sha256(
        resolved,
        listing=listing,
        label=label,
    )
    final_revision = _run_git(
        resolved,
        "rev-parse",
        "--verify",
        "HEAD^{commit}",
    ).decode("ascii").strip()
    final_dirty = _run_git(
        resolved,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    _reject_git_index_and_ignored_bypasses(resolved, label=label)
    final_listing = _run_git(
        resolved,
        "ls-tree",
        "-r",
        "-z",
        "--full-tree",
        "HEAD",
    )
    if (
        final_revision != revision
        or final_dirty
        or final_listing != listing
    ):
        raise ValueError(f"{label} checkout changed while it was attested")
    return {
        "revision": revision,
        "git_tree_sha1": git_tree_sha1,
        "tracked_tree": {
            "bytes": len(listing),
            "records": listing.count(b"\0"),
            "sha256": _sha256(listing),
            "verified_worktree_sha256": verified_worktree_sha256,
        },
        "worktree": "verified_against_head_no_extra_files_or_index_flags",
    }


def attest_source_checkouts(
    browsergym_checkout: str | Path,
    miniwob_checkout: str | Path,
) -> dict[str, Any]:
    """Attest both source repositories required by a production capture."""

    browsergym = Path(browsergym_checkout)
    miniwob = Path(miniwob_checkout)
    if browsergym.resolve() == miniwob.resolve():
        raise ValueError("BrowserGym and MiniWoB checkouts must be different repositories")
    return {
        "source_pins": {
            "browsergym_revision": PRODUCTION_BROWSERGYM_REVISION,
            "browsergym_version": PRODUCTION_BROWSERGYM_VERSION,
            "miniwob_revision": PRODUCTION_MINIWOB_REVISION,
        },
        "repositories": {
            "browsergym": attest_git_checkout(
                browsergym,
                expected_revision=PRODUCTION_BROWSERGYM_REVISION,
                label="BrowserGym",
            ),
            "miniwob": attest_git_checkout(
                miniwob,
                expected_revision=PRODUCTION_MINIWOB_REVISION,
                label="MiniWoB",
            ),
        },
    }


def _stable_regular_file_identity(
    path: Path,
    *,
    label: str,
    require_executable: bool = False,
    allow_empty: bool = False,
) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link")
    try:
        initial = path.stat()
    except OSError as error:
        raise ValueError(f"{label} is missing or unreadable: {path}") from error
    if not stat.S_ISREG(initial.st_mode) or (
        initial.st_size <= 0 and not allow_empty
    ):
        raise ValueError(f"{label} must be a non-empty regular file")
    if require_executable and os.name != "nt" and not os.access(path, os.X_OK):
        raise ValueError(f"{label} is not executable")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    digest = hashlib.sha256()
    observed_bytes = 0
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} could not be opened safely") from error
    try:
        before = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            observed_bytes += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity_fields):
        raise ValueError(f"{label} changed while it was hashed")
    if observed_bytes != before.st_size:
        raise ValueError(f"{label} byte count changed while it was hashed")
    return {"bytes": observed_bytes, "sha256": digest.hexdigest()}


def _directory_tree_identity(root: Path, *, label: str) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"{label} must be a non-symlink directory")
    root = root.resolve()
    digest = hashlib.sha256()
    total_bytes = 0
    entries = 0
    files = 0
    symlinks = 0

    def visit(directory: Path, relative_directory: Path) -> None:
        nonlocal entries, files, symlinks, total_bytes
        before = directory.stat()
        try:
            children = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as error:
            raise ValueError(f"{label} could not be traversed") from error
        for child in children:
            relative = relative_directory / child.name
            relative_name = relative.as_posix()
            try:
                relative_name.encode("utf-8")
                child_stat = child.stat(follow_symlinks=False)
            except (OSError, UnicodeError) as error:
                raise ValueError(f"{label} contains an unreadable entry") from error
            entries += 1
            if stat.S_ISDIR(child_stat.st_mode):
                digest.update(
                    _canonical_json_bytes(
                        {"kind": "directory", "path": relative_name},
                        newline=True,
                    )
                )
                visit(Path(child.path), relative)
            elif stat.S_ISREG(child_stat.st_mode):
                identity = _stable_regular_file_identity(
                    Path(child.path),
                    label=f"{label} entry {relative_name!r}",
                    allow_empty=True,
                )
                files += 1
                total_bytes += identity["bytes"]
                digest.update(
                    _canonical_json_bytes(
                        {
                            "bytes": identity["bytes"],
                            "kind": "file",
                            "mode": stat.S_IMODE(child_stat.st_mode),
                            "path": relative_name,
                            "sha256": identity["sha256"],
                        },
                        newline=True,
                    )
                )
            elif stat.S_ISLNK(child_stat.st_mode):
                link_path = Path(child.path)
                try:
                    target = os.readlink(link_path)
                    target.encode("utf-8")
                    resolved_target = link_path.resolve(strict=True)
                    resolved_relative = resolved_target.relative_to(root).as_posix()
                except (OSError, RuntimeError, UnicodeError, ValueError) as error:
                    raise ValueError(
                        f"{label} contains a broken or escaping symbolic link: "
                        f"{relative_name!r}"
                    ) from error
                if not resolved_target.is_file() and not resolved_target.is_dir():
                    raise ValueError(
                        f"{label} symbolic link targets an unsupported entry: "
                        f"{relative_name!r}"
                    )
                symlinks += 1
                digest.update(
                    _canonical_json_bytes(
                        {
                            "kind": "symlink",
                            "path": relative_name,
                            "resolved_path": resolved_relative,
                            "target": target,
                        },
                        newline=True,
                    )
                )
            else:
                raise ValueError(
                    f"{label} contains unsupported non-file entry {relative_name!r}"
                )
        after = directory.stat()
        identity_fields = ("st_dev", "st_ino", "st_mode", "st_mtime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in identity_fields):
            raise ValueError(f"{label} changed while it was traversed")

    visit(root, Path())
    if total_bytes <= 0 or files <= 0:
        raise ValueError(f"{label} must contain at least one non-empty regular file")
    return {
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
        "entries": entries,
        "files": files,
        "symlinks": symlinks,
    }


def _installed_version(distribution: str) -> str:
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError(f"required distribution {distribution!r} is not installed") from error
    if not version:
        raise ValueError(f"required distribution {distribution!r} has an empty version")
    return version


def _reported_chromium_version(executable: Path) -> str:
    try:
        result = subprocess.run(
            [str(executable), "--version"],
            check=False,
            capture_output=True,
            timeout=15,
            env={**os.environ, "LANG": "C", "LC_ALL": "C"},
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError("failed to query the Chromium executable version") from error
    output = (result.stdout + b"\n" + result.stderr).decode("utf-8", errors="replace").strip()
    match = _CHROMIUM_VERSION_RE.search(output)
    if result.returncode != 0 or match is None:
        raise ValueError("Chromium executable did not report a parseable version")
    return match.group(1)


def attest_runtime(
    browser_executable: str | Path,
    browser_installation: str | Path,
    *,
    settings: BrowserGymCaptureSettings = BrowserGymCaptureSettings(),
) -> dict[str, Any]:
    """Build immutable runtime evidence and enforce all production version pins."""

    settings.validate_production()
    executable = Path(browser_executable)
    installation = Path(browser_installation)
    if not installation.is_dir() or installation.is_symlink():
        raise ValueError("browser installation must be a non-symlink directory")
    installation = installation.resolve()
    try:
        executable = executable.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("browser executable is missing or has a broken link") from error
    try:
        executable.relative_to(installation)
    except ValueError as error:
        raise ValueError("browser executable must be inside browser installation") from error
    revision_match = _CHROMIUM_INSTALLATION_RE.fullmatch(installation.name)
    if revision_match is None:
        raise ValueError(
            "browser installation directory must be named chromium-<revision> "
            "or chromium_headless_shell-<revision>"
        )
    chromium_revision = revision_match.group(1)
    executable_identity = _stable_regular_file_identity(
        executable,
        label="browser executable",
        require_executable=True,
    )
    installation_identity = _directory_tree_identity(
        installation,
        label="browser installation",
    )
    chromium_version = _reported_chromium_version(executable)
    distributions = {
        "browsergym-core": _installed_version("browsergym-core"),
        "browsergym-miniwob": _installed_version("browsergym-miniwob"),
        "gymnasium": _installed_version("gymnasium"),
        "playwright": _installed_version("playwright"),
    }
    expected_versions = {
        "browsergym-core": PRODUCTION_BROWSERGYM_VERSION,
        "browsergym-miniwob": PRODUCTION_BROWSERGYM_VERSION,
        "playwright": PRODUCTION_PLAYWRIGHT_VERSION,
    }
    observed_versions = {key: distributions[key] for key in expected_versions}
    if observed_versions != expected_versions:
        raise ValueError(
            "installed BrowserGym/Playwright version drift: "
            f"expected {expected_versions}, observed {observed_versions}"
        )
    if chromium_revision != PRODUCTION_CHROMIUM_REVISION:
        raise ValueError(
            "Chromium revision drift: "
            f"expected {PRODUCTION_CHROMIUM_REVISION}, observed {chromium_revision}"
        )
    if chromium_version != PRODUCTION_CHROMIUM_VERSION:
        raise ValueError(
            "Chromium version drift: "
            f"expected {PRODUCTION_CHROMIUM_VERSION}, observed {chromium_version}"
        )

    controls = settings.controls()
    runtime_pins = {
        "playwright_version": distributions["playwright"],
        "chromium_revision": chromium_revision,
        "chromium_version": chromium_version,
        "python_version": platform.python_version(),
        "os": platform.system().lower(),
        "architecture": platform.machine().lower(),
        **controls,
        "browser_executable": executable_identity,
        "browser_installation": {
            "bytes": installation_identity["bytes"],
            "sha256": installation_identity["sha256"],
        },
        "environment_manifest": dict(PRODUCTION_RUNTIME_MANIFEST_IDENTITY),
    }
    return {
        "runtime_pins": runtime_pins,
        "attestation": {
            "installed_distributions": distributions,
            "python_implementation": platform.python_implementation(),
            "browser": {
                "reported_version": chromium_version,
                "installation_entries": installation_identity["entries"],
                "installation_files": installation_identity["files"],
                "installation_symlinks": installation_identity["symlinks"],
                "executable_scope": "inside_attested_installation",
            },
        },
    }


def _validate_source_evidence(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("source attestation must be an object")
    _require_exact_keys(
        value,
        _SOURCE_EVIDENCE_KEYS,
        label="source attestation",
    )
    source_pins = value["source_pins"]
    if not isinstance(source_pins, dict):
        raise ValueError("source_pins must be an object")
    _require_exact_keys(source_pins, _SOURCE_PIN_KEYS, label="source_pins")
    expected = {
        "browsergym_revision": PRODUCTION_BROWSERGYM_REVISION,
        "browsergym_version": PRODUCTION_BROWSERGYM_VERSION,
        "miniwob_revision": PRODUCTION_MINIWOB_REVISION,
    }
    if not _canonical_equal(source_pins, expected):
        raise ValueError(f"source pin drift: expected {expected}, observed {source_pins}")
    repositories = value["repositories"]
    if not isinstance(repositories, dict):
        raise ValueError("source repositories attestation must be an object")
    _require_exact_keys(repositories, _REPOSITORY_KEYS, label="source repositories")
    expected_revisions = {
        "browsergym": PRODUCTION_BROWSERGYM_REVISION,
        "miniwob": PRODUCTION_MINIWOB_REVISION,
    }
    for repository_name, expected_revision in expected_revisions.items():
        repository = repositories[repository_name]
        label = f"source repositories.{repository_name}"
        if not isinstance(repository, dict):
            raise ValueError(f"{label} must be an object")
        _require_exact_keys(
            repository,
            _REPOSITORY_ATTESTATION_KEYS,
            label=label,
        )
        if repository["revision"] != expected_revision:
            raise ValueError(f"{label}.revision drift")
        git_tree_sha1 = repository["git_tree_sha1"]
        if not isinstance(git_tree_sha1, str) or _GIT_SHA1_RE.fullmatch(git_tree_sha1) is None:
            raise ValueError(f"{label}.git_tree_sha1 must be lowercase 40-hex")
        if (
            repository["worktree"]
            != "verified_against_head_no_extra_files_or_index_flags"
        ):
            raise ValueError(f"{label}.worktree is not clean")
        tracked_tree = repository["tracked_tree"]
        if not isinstance(tracked_tree, dict):
            raise ValueError(f"{label}.tracked_tree must be an object")
        _require_exact_keys(
            tracked_tree,
            _TRACKED_TREE_KEYS,
            label=f"{label}.tracked_tree",
        )
        _positive_int(
            tracked_tree["bytes"],
            label=f"{label}.tracked_tree.bytes",
        )
        _positive_int(
            tracked_tree["records"],
            label=f"{label}.tracked_tree.records",
        )
        tracked_sha256 = tracked_tree["sha256"]
        if (
            not isinstance(tracked_sha256, str)
            or _SHA256_RE.fullmatch(tracked_sha256) is None
        ):
            raise ValueError(f"{label}.tracked_tree.sha256 must be lowercase 64-hex")
        verified_worktree_sha256 = tracked_tree["verified_worktree_sha256"]
        if (
            not isinstance(verified_worktree_sha256, str)
            or _SHA256_RE.fullmatch(verified_worktree_sha256) is None
        ):
            raise ValueError(
                f"{label}.tracked_tree.verified_worktree_sha256 "
                "must be lowercase 64-hex"
            )
    return value


def _validate_identity(value: Any, *, label: str) -> None:
    if not isinstance(value, dict) or set(value) != {"bytes", "sha256"}:
        raise ValueError(f"{label} must contain exactly bytes and sha256")
    byte_count = value["bytes"]
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
        raise ValueError(f"{label}.bytes must be a positive integer")
    if not isinstance(value["sha256"], str) or _SHA256_RE.fullmatch(value["sha256"]) is None:
        raise ValueError(f"{label}.sha256 must be lowercase 64-hex")


def _validate_runtime_manifest_identity(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    _require_exact_keys(
        value,
        frozenset(PRODUCTION_RUNTIME_MANIFEST_IDENTITY),
        label=label,
    )
    if not _canonical_equal(value, PRODUCTION_RUNTIME_MANIFEST_IDENTITY):
        raise ValueError(f"{label} drift")
    return value


def _verify_production_runtime_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    expected = PRODUCTION_RUNTIME_MANIFEST_IDENTITY
    if manifest_path.name != expected["file"]:
        raise ValueError(
            "BrowserGym runtime manifest filename does not match the production pin"
        )
    manifest = load_and_verify_environment_manifest(
        manifest_path,
        expected_sha256=expected["self_sha256"],
        verify_active=True,
    )
    payload = _canonical_json_bytes(manifest, newline=True)
    observed = {
        "kind": manifest["kind"],
        "schema_version": manifest["schema_version"],
        "file": manifest_path.name,
        "bytes": len(payload),
        "sha256": _sha256(payload),
        "self_sha256": manifest["manifest_self_sha256"],
        "distributions": len(manifest["installed_distributions"]),
        "playwright_driver_sha256": manifest["playwright_driver"]["content"]["sha256"],
    }
    return dict(
        _validate_runtime_manifest_identity(observed, label="runtime manifest")
    )


def _validate_runtime_evidence(
    value: Any,
    *,
    settings: BrowserGymCaptureSettings,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("runtime attestation must be an object")
    _require_exact_keys(
        value,
        _RUNTIME_EVIDENCE_KEYS,
        label="runtime attestation",
    )
    runtime_pins = value["runtime_pins"]
    if not isinstance(runtime_pins, dict):
        raise ValueError("runtime_pins must be an object")
    _require_exact_keys(runtime_pins, _RUNTIME_PIN_KEYS, label="runtime_pins")
    expected_fixed = {
        "playwright_version": PRODUCTION_PLAYWRIGHT_VERSION,
        "chromium_revision": PRODUCTION_CHROMIUM_REVISION,
        "chromium_version": PRODUCTION_CHROMIUM_VERSION,
        "environment_manifest": dict(PRODUCTION_RUNTIME_MANIFEST_IDENTITY),
        **settings.controls(),
    }
    observed_fixed = {key: runtime_pins.get(key) for key in expected_fixed}
    if not _canonical_equal(observed_fixed, expected_fixed):
        raise ValueError(
            f"runtime pin drift: expected {expected_fixed}, observed {observed_fixed}"
        )
    for key in ("python_version", "os", "architecture"):
        _nonempty_string(runtime_pins[key], label=f"runtime_pins.{key}")
    for key in (
        "playwright_version",
        "chromium_revision",
        "chromium_version",
        "locale",
        "timezone_id",
        "action_set",
        "observation_mode",
    ):
        _nonempty_string(runtime_pins[key], label=f"runtime_pins.{key}")
    if not isinstance(runtime_pins["headless"], bool):
        raise ValueError("runtime_pins.headless must be boolean")
    viewport = runtime_pins["viewport"]
    if not isinstance(viewport, dict):
        raise ValueError("runtime_pins.viewport must be an object")
    _require_exact_keys(
        viewport,
        frozenset({"width", "height"}),
        label="runtime_pins.viewport",
    )
    _positive_int(viewport["width"], label="runtime_pins.viewport.width")
    _positive_int(viewport["height"], label="runtime_pins.viewport.height")
    _finite_positive_number(
        runtime_pins["device_scale_factor"],
        label="runtime_pins.device_scale_factor",
    )
    _positive_int(runtime_pins["max_steps"], label="runtime_pins.max_steps")
    _finite_positive_number(
        runtime_pins["playwright_operation_timeout_seconds"],
        label="runtime_pins.playwright_operation_timeout_seconds",
    )
    _validate_identity(runtime_pins["browser_executable"], label="browser_executable")
    _validate_identity(runtime_pins["browser_installation"], label="browser_installation")
    _validate_runtime_manifest_identity(
        runtime_pins["environment_manifest"],
        label="runtime_pins.environment_manifest",
    )

    attestation = value["attestation"]
    if not isinstance(attestation, dict):
        raise ValueError("runtime attestation details must be an object")
    _require_exact_keys(
        attestation,
        _RUNTIME_ATTESTATION_KEYS,
        label="runtime attestation details",
    )
    distributions = attestation["installed_distributions"]
    if not isinstance(distributions, dict):
        raise ValueError("installed_distributions must be an object")
    _require_exact_keys(
        distributions,
        _DISTRIBUTION_KEYS,
        label="installed_distributions",
    )
    for distribution, version in distributions.items():
        _nonempty_string(
            version,
            label=f"installed_distributions.{distribution}",
        )
    expected_distributions = {
        "browsergym-core": PRODUCTION_BROWSERGYM_VERSION,
        "browsergym-miniwob": PRODUCTION_BROWSERGYM_VERSION,
        "playwright": PRODUCTION_PLAYWRIGHT_VERSION,
    }
    observed_distributions = {
        key: distributions.get(key) for key in expected_distributions
    }
    if not _canonical_equal(observed_distributions, expected_distributions):
        raise ValueError("installed BrowserGym/Playwright distribution version drift")
    _nonempty_string(
        attestation["python_implementation"],
        label="runtime attestation details.python_implementation",
    )
    browser = attestation["browser"]
    if not isinstance(browser, dict):
        raise ValueError("runtime browser attestation must be an object")
    _require_exact_keys(
        browser,
        _BROWSER_ATTESTATION_KEYS,
        label="runtime browser attestation",
    )
    if browser["reported_version"] != PRODUCTION_CHROMIUM_VERSION:
        raise ValueError("runtime browser reported_version drift")
    if browser["executable_scope"] != "inside_attested_installation":
        raise ValueError("runtime browser executable_scope drift")
    entries = _positive_int(
        browser["installation_entries"],
        label="runtime browser installation_entries",
    )
    files = _positive_int(
        browser["installation_files"],
        label="runtime browser installation_files",
    )
    symlinks = _nonnegative_int(
        browser["installation_symlinks"],
        label="runtime browser installation_symlinks",
    )
    if entries < files + symlinks:
        raise ValueError(
            "runtime browser installation_entries is smaller than files plus symlinks"
        )
    return value


def _module_must_resolve_inside(module: Any, checkout: Path, *, label: str) -> None:
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        raise RuntimeError(f"{label} does not expose a filesystem source path")
    try:
        Path(module_file).resolve().relative_to(checkout.resolve())
    except ValueError as error:
        raise RuntimeError(
            f"{label} is not imported from the attested BrowserGym checkout"
        ) from error


def make_browsergym_environment(
    episode: BrowserGymCaptureEpisode,
    settings: BrowserGymCaptureSettings,
    *,
    browsergym_checkout: str | Path,
    browser_executable: str | Path,
) -> ResetEnvironment:
    """Create one real BrowserGym environment using only lazy optional imports."""

    settings.validate_production()
    try:
        miniwob_module = importlib.import_module("browsergym.miniwob")
        highlevel_module = importlib.import_module("browsergym.core.action.highlevel")
        gym = importlib.import_module("gymnasium")
    except ImportError as error:
        raise RuntimeError(
            "controlled capture requires browsergym-miniwob, gymnasium, and Playwright"
        ) from error
    checkout = Path(browsergym_checkout).resolve()
    _module_must_resolve_inside(
        miniwob_module,
        checkout,
        label="browsergym.miniwob",
    )
    _module_must_resolve_inside(
        highlevel_module,
        checkout,
        label="browsergym.core.action.highlevel",
    )
    action_set_type = getattr(highlevel_module, "HighLevelActionSet", None)
    if action_set_type is None:
        raise RuntimeError("pinned BrowserGym HighLevelActionSet API is unavailable")
    action_mapping = action_set_type().to_python_code
    try:
        return gym.make(
            f"browsergym/{episode.task_name}",
            viewport={
                "width": settings.viewport_width,
                "height": settings.viewport_height,
            },
            timeout=int(settings.playwright_operation_timeout_seconds * 1000),
            locale=settings.locale,
            timezone_id=settings.timezone_id,
            headless=settings.headless,
            action_mapping=action_mapping,
            use_raw_page_output=False,
            pw_chromium_kwargs={"executable_path": str(Path(browser_executable).resolve())},
            pw_context_kwargs={"device_scale_factor": settings.device_scale_factor},
            max_episode_steps=settings.max_steps,
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "pinned BrowserGym constructor contract drifted; refusing a capture with "
            "partially applied controls"
        ) from error


def _miniwob_html_directory(checkout: Path) -> Path:
    # BrowserGym appends ``<subdomain>.html`` to MINIWOB_URL.  The pinned
    # MiniWoB++ checkout stores those task pages one level below the shared
    # ``html`` assets directory.
    html = checkout.resolve() / "miniwob" / "html" / "miniwob"
    if not html.is_dir() or html.is_symlink():
        raise ValueError(
            "MiniWoB checkout must contain a non-symlink "
            "miniwob/html/miniwob task directory"
        )
    return html


@contextlib.contextmanager
def _fixed_miniwob_url(checkout: Path):
    html = _miniwob_html_directory(checkout)
    expected = html.as_uri().rstrip("/") + "/"
    prior = os.environ.get("MINIWOB_URL")
    if prior is not None and prior != expected:
        raise ValueError(
            "MINIWOB_URL already points outside the attested MiniWoB checkout"
        )
    os.environ["MINIWOB_URL"] = expected
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("MINIWOB_URL", None)
        else:
            os.environ["MINIWOB_URL"] = prior


@contextlib.contextmanager
def _fixed_playwright_browsers_path(
    browser_installation: Path,
    browser_executable: Path,
):
    """Force BrowserGym's task and chat browsers into the attested installation."""

    try:
        installation = browser_installation.resolve(strict=True)
        executable = browser_executable.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("attested Playwright browser paths must exist") from error
    if not installation.is_dir() or installation.is_symlink():
        raise ValueError("attested Playwright browser installation must be a directory")
    try:
        executable.relative_to(installation)
    except ValueError as error:
        raise ValueError(
            "attested Chromium executable must be inside its installation"
        ) from error

    browser_root = installation.parent
    expected = str(browser_root)
    prior = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if prior is not None:
        try:
            prior_root = Path(prior).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ValueError(
                "PLAYWRIGHT_BROWSERS_PATH does not resolve to the attested browser root"
            ) from error
        if prior_root != browser_root:
            raise ValueError(
                "PLAYWRIGHT_BROWSERS_PATH points outside the attested browser root"
            )

    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = expected
    try:
        yield
        if os.environ.get("PLAYWRIGHT_BROWSERS_PATH") != expected:
            raise RuntimeError("PLAYWRIGHT_BROWSERS_PATH drifted during capture")
    finally:
        if prior is None:
            os.environ.pop("PLAYWRIGHT_BROWSERS_PATH", None)
        else:
            os.environ["PLAYWRIGHT_BROWSERS_PATH"] = prior


@contextlib.contextmanager
def _owned_browsergym_playwright(
    browsergym_checkout: Path,
    browser_executable: Path,
):
    """Own BrowserGym's global driver and bind both of its Chromium launches."""

    try:
        core_module = importlib.import_module("browsergym.core")
        sync_api = importlib.import_module("playwright.sync_api")
    except ImportError as error:
        raise RuntimeError(
            "controlled capture requires BrowserGym core and Playwright"
        ) from error
    _module_must_resolve_inside(
        core_module,
        browsergym_checkout,
        label="browsergym.core",
    )
    set_global = getattr(core_module, "_set_global_playwright", None)
    if not callable(set_global) or not hasattr(core_module, "_PLAYWRIGHT"):
        raise RuntimeError("pinned BrowserGym global Playwright API is unavailable")
    if getattr(core_module, "_PLAYWRIGHT") is not None:
        raise RuntimeError(
            "BrowserGym already owns a Playwright driver; refusing stale browser resolution"
        )
    sync_playwright = getattr(sync_api, "sync_playwright", None)
    if not callable(sync_playwright):
        raise RuntimeError("pinned Playwright sync API is unavailable")

    playwright = None
    try:
        playwright = sync_playwright().start()
        default_path = getattr(getattr(playwright, "chromium", None), "executable_path", None)
        if not isinstance(default_path, str) or not default_path:
            raise RuntimeError("Playwright returned an invalid Chromium executable path")
        try:
            default_executable = Path(default_path).resolve(strict=True)
            expected_executable = browser_executable.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise RuntimeError(
                "Playwright's default Chromium executable is missing"
            ) from error
        if default_executable != expected_executable:
            raise ValueError(
                "Playwright's default Chromium executable does not match the "
                "separately attested executable"
            )
        set_global(playwright)
        yield
        if getattr(core_module, "_PLAYWRIGHT") is not playwright:
            raise RuntimeError("BrowserGym's global Playwright driver drifted during capture")
    finally:
        set_global(None)
        if playwright is not None:
            playwright.stop()


def _browsergym_source_roots(checkout: Path) -> tuple[Path, Path]:
    checkout = checkout.resolve()
    logical_roots = (
        Path("browsergym/core/src"),
        Path("browsergym/miniwob/src"),
    )
    roots: list[Path] = []
    for logical_root in logical_roots:
        source_root = checkout / logical_root
        if (
            not source_root.is_dir()
            or source_root.is_symlink()
            or not (source_root / "browsergym").is_dir()
            or (source_root / "browsergym").is_symlink()
        ):
            raise ValueError(
                "attested BrowserGym checkout is missing a non-symlink "
                f"{logical_root.as_posix()} package root"
            )
        resolved_root = source_root.resolve()
        try:
            resolved_root.relative_to(checkout)
            (resolved_root / "browsergym").resolve().relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(
                f"attested BrowserGym {logical_root.as_posix()} package root escapes its checkout"
            ) from error
        roots.append(resolved_root)
    return roots[0], roots[1]


def _path_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_isolated_browsergym_modules(
    checkout: Path,
    namespace_paths: tuple[Path, Path],
) -> None:
    expected_namespace_paths = [str(path) for path in namespace_paths]
    namespace = sys.modules.get("browsergym")
    if namespace is None or list(getattr(namespace, "__path__", ())) != expected_namespace_paths:
        raise RuntimeError("BrowserGym namespace search path drifted during capture")
    namespace_spec = getattr(namespace, "__spec__", None)
    if (
        namespace_spec is None
        or list(getattr(namespace_spec, "submodule_search_locations", ()))
        != expected_namespace_paths
    ):
        raise RuntimeError("BrowserGym namespace import specification drifted during capture")

    required_modules = _IMPORT_SCOPE["required_modules"]
    if not isinstance(required_modules, dict):
        raise RuntimeError("internal BrowserGym import-scope policy is invalid")
    for module_name, logical_file in required_modules.items():
        module = sys.modules.get(module_name)
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str) or not module_file:
            raise RuntimeError(f"{module_name} was not loaded from the attested checkout")
        try:
            observed = Path(module_file).resolve(strict=True)
            expected = (checkout / logical_file).resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise RuntimeError(
                f"{module_name} source path disappeared during capture"
            ) from error
        if observed != expected:
            raise RuntimeError(
                f"{module_name} is not the exact attested BrowserGym source file"
            )

    for module_name, module in tuple(sys.modules.items()):
        if module_name != "browsergym" and not module_name.startswith("browsergym."):
            continue
        module_file = getattr(module, "__file__", None)
        if isinstance(module_file, str) and module_file:
            try:
                resolved_file = Path(module_file).resolve(strict=True)
            except (OSError, RuntimeError) as error:
                raise RuntimeError(
                    f"{module_name} source path disappeared during capture"
                ) from error
            if not any(
                _path_inside(resolved_file, namespace_path)
                for namespace_path in namespace_paths
            ):
                raise RuntimeError(
                    f"{module_name} was imported outside the attested BrowserGym checkout"
                )
            continue
        module_search_paths = getattr(module, "__path__", None)
        if module_search_paths is None:
            raise RuntimeError(
                f"{module_name} has neither a source file nor a package search path"
            )
        for search_path in module_search_paths:
            try:
                resolved_search_path = Path(search_path).resolve(strict=True)
            except (OSError, RuntimeError) as error:
                raise RuntimeError(
                    f"{module_name} package search path disappeared during capture"
                ) from error
            if not any(
                _path_inside(resolved_search_path, namespace_path)
                for namespace_path in namespace_paths
            ):
                raise RuntimeError(
                    f"{module_name} exposes an ambient package search path"
                )


@contextlib.contextmanager
def _isolated_browsergym_imports(browsergym_checkout: Path):
    """Load BrowserGym only from two exact source roots in the attested checkout."""

    checkout = browsergym_checkout.resolve()
    source_roots = _browsergym_source_roots(checkout)
    namespace_paths = tuple(root / "browsergym" for root in source_roots)
    preloaded = sorted(
        name
        for name in sys.modules
        if name == "browsergym" or name.startswith("browsergym.")
    )
    if preloaded:
        raise RuntimeError(
            "BrowserGym modules were loaded before the isolated capture scope: "
            + ", ".join(preloaded[:8])
        )

    prior_sys_path = list(sys.path)
    root_strings = [str(root) for root in source_roots]
    filtered_sys_path: list[str] = []
    for entry in prior_sys_path:
        try:
            resolved_entry = Path(entry or os.curdir).resolve()
        except (OSError, RuntimeError):
            filtered_sys_path.append(entry)
            continue
        if resolved_entry not in source_roots:
            filtered_sys_path.append(entry)
    scoped_sys_path = [*root_strings, *filtered_sys_path]
    missing_cache_entry = object()
    prior_importer_cache = {
        root: sys.path_importer_cache.get(root, missing_cache_entry)
        for root in root_strings
    }
    # Import the PEP 420 namespace with only the two attested roots visible. If ambient
    # site-packages remained visible here, a later regular ``browsergym/__init__.py`` would take
    # precedence over the namespace portions even though the trusted roots appear first.
    sys.path[:] = root_strings
    importlib.invalidate_caches()
    try:
        namespace = importlib.import_module("browsergym")
        if getattr(namespace, "__file__", None) is not None:
            raise RuntimeError(
                "attested BrowserGym root must resolve as a source-only namespace package"
            )
        namespace.__path__ = [str(path) for path in namespace_paths]
        namespace_spec = getattr(namespace, "__spec__", None)
        if namespace_spec is None:
            raise RuntimeError("BrowserGym namespace does not expose an import specification")
        namespace_spec.submodule_search_locations = [
            str(path) for path in namespace_paths
        ]
        sys.path[:] = scoped_sys_path
        importlib.invalidate_caches()
        importlib.import_module("browsergym.core")
        importlib.import_module("browsergym.core.action.highlevel")
        importlib.import_module("browsergym.miniwob")
        _validate_isolated_browsergym_modules(checkout, namespace_paths)
        yield
        if sys.path != scoped_sys_path:
            raise RuntimeError("sys.path drifted during BrowserGym capture")
        _validate_isolated_browsergym_modules(checkout, namespace_paths)
    finally:
        for module_name in tuple(sys.modules):
            if module_name == "browsergym" or module_name.startswith("browsergym."):
                sys.modules.pop(module_name, None)
        sys.path[:] = prior_sys_path
        for root, prior_cache_entry in prior_importer_cache.items():
            if prior_cache_entry is missing_cache_entry:
                sys.path_importer_cache.pop(root, None)
            else:
                sys.path_importer_cache[root] = prior_cache_entry
        importlib.invalidate_caches()


@contextlib.contextmanager
def _production_environment_scope(
    browsergym_checkout: Path,
    miniwob_checkout: Path,
    browser_executable: Path,
    browser_installation: Path,
):
    prior_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        with (
            _isolated_browsergym_imports(browsergym_checkout),
            _fixed_miniwob_url(miniwob_checkout),
            _fixed_playwright_browsers_path(
                browser_installation,
                browser_executable,
            ),
            _owned_browsergym_playwright(
                browsergym_checkout,
                browser_executable,
            ),
        ):
            yield
    finally:
        sys.dont_write_bytecode = prior_dont_write_bytecode


def _goal_from_reset(result: Any, *, episode: BrowserGymCaptureEpisode) -> str:
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError(
            f"{episode.task_name} seed {episode.seed} reset must return (observation, info)"
        )
    observation = result[0]
    if not isinstance(observation, Mapping):
        raise ValueError(
            f"{episode.task_name} seed {episode.seed} reset observation must be a mapping"
        )
    goal = observation.get("goal")
    if not isinstance(goal, str) or not goal:
        raise ValueError(
            f"{episode.task_name} seed {episode.seed} observation.goal "
            "must be a non-empty string"
        )
    try:
        goal_bytes = goal.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(
            f"{episode.task_name} seed {episode.seed} observation.goal is not UTF-8"
        ) from error
    if len(goal_bytes) > DEFAULT_MAX_PROMPT_BYTES:
        raise ValueError(
            f"{episode.task_name} seed {episode.seed} observation.goal exceeds "
            f"{DEFAULT_MAX_PROMPT_BYTES} bytes"
        )
    return goal


def _capture_row(
    episode: BrowserGymCaptureEpisode,
    goal: str,
    *,
    source_pins: Mapping[str, Any],
    runtime_pins: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "task_name": episode.task_name,
        "seed": episode.seed,
        "goal": goal,
        "similarity_group": episode.similarity_group,
        "split": episode.split,
        "source_pins": source_pins,
        "runtime_pins": runtime_pins,
    }


def _matches_payload(path: Path, payload: bytes) -> bool:
    if not path.is_file() or path.is_symlink() or path.stat().st_size != len(payload):
        return False
    offset = 0
    with path.open("rb") as handle:
        while offset < len(payload):
            chunk = handle.read(min(1024 * 1024, len(payload) - offset))
            if not chunk or chunk != payload[offset : offset + len(chunk)]:
                return False
            offset += len(chunk)
        return handle.read(1) == b""


def _unlink_if_inode(path: Path, *, device: int, inode: int) -> None:
    try:
        observed = path.stat()
    except FileNotFoundError:
        return
    if observed.st_dev == device and observed.st_ino == inode and not path.is_symlink():
        path.unlink()


def _publish_pair_no_clobber(
    capture_path: Path,
    capture_payload: bytes,
    receipt_path: Path,
    receipt_payload: bytes,
) -> None:
    for path in (capture_path, receipt_path):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"refusing to overwrite existing capture artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    published: list[tuple[Path, int, int]] = []
    try:
        for destination, payload in (
            (capture_path, capture_payload),
            (receipt_path, receipt_payload),
        ):
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                temporary_paths.append(temporary)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary_stat = temporary.stat()
            try:
                os.link(temporary, destination)
            except FileExistsError as error:
                raise RuntimeError(
                    f"refusing to overwrite concurrently-created artifact: {destination}"
                ) from error
            published.append(
                (destination, temporary_stat.st_dev, temporary_stat.st_ino)
            )
        if not _matches_payload(capture_path, capture_payload):
            raise RuntimeError("published BrowserGym capture failed verification")
        if not _matches_payload(receipt_path, receipt_payload):
            raise RuntimeError("published BrowserGym receipt failed verification")
    except Exception:
        for destination, device, inode in reversed(published):
            _unlink_if_inode(destination, device=device, inode=inode)
        raise
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)


def _preflight_outputs(capture_path: Path, receipt_path: Path) -> None:
    if capture_path.resolve() == receipt_path.resolve():
        raise ValueError("capture and receipt paths must be different")
    for path in (capture_path, receipt_path):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"refusing to overwrite existing capture artifact: {path}")


def capture_browsergym_goals(
    capture_path: str | Path,
    receipt_path: str | Path,
    *,
    browsergym_checkout: str | Path,
    miniwob_checkout: str | Path,
    browser_executable: str | Path,
    browser_installation: str | Path,
    environment_manifest: str | Path,
    settings: BrowserGymCaptureSettings = BrowserGymCaptureSettings(),
) -> dict[str, Any]:
    """Run the exact reset-only plan and publish capture plus receipt without clobbering.

    The production entry point always selects the real BrowserGym factory, isolates imports to the
    attested checkout, and forces ``MINIWOB_URL`` to the attested MiniWoB checkout. Source, browser
    runtime, and the complete Python environment manifest are sampled both before and after all
    resets; any drift aborts publication. Ordinary publication errors roll back files created by
    this call. Portable crash-atomicity across two paths is not claimed: abrupt process death may
    leave capture without receipt, and consumers must require
    :func:`verify_browsergym_capture_receipt`.
    """

    settings.validate_production()
    capture = Path(capture_path)
    receipt_file = Path(receipt_path)
    _preflight_outputs(capture, receipt_file)
    environment_before = _frozen_json(
        _verify_production_runtime_manifest(environment_manifest),
        label="runtime manifest identity",
    )

    source_before = _frozen_json(
        _validate_source_evidence(
            attest_source_checkouts(browsergym_checkout, miniwob_checkout)
        ),
        label="source attestation",
    )
    runtime_before = _frozen_json(
        _validate_runtime_evidence(
            attest_runtime(
                browser_executable,
                browser_installation,
                settings=settings,
            ),
            settings=settings,
        ),
        label="runtime attestation",
    )
    episodes = production_capture_plan()

    def production_environment_factory(
        episode: BrowserGymCaptureEpisode,
        capture_settings: BrowserGymCaptureSettings,
    ) -> ResetEnvironment:
        return make_browsergym_environment(
            episode,
            capture_settings,
            browsergym_checkout=browsergym_checkout,
            browser_executable=browser_executable,
        )

    rows: list[dict[str, Any]] = []
    with _production_environment_scope(
        Path(browsergym_checkout),
        Path(miniwob_checkout),
        Path(browser_executable),
        Path(browser_installation),
    ):
        for episode in episodes:
            environment = production_environment_factory(episode, settings)
            if environment is None:
                raise RuntimeError("BrowserGym environment factory returned None")
            try:
                goal = _goal_from_reset(
                    environment.reset(seed=episode.seed),
                    episode=episode,
                )
            finally:
                environment.close()
            rows.append(
                _capture_row(
                    episode,
                    goal,
                    source_pins=source_before["source_pins"],
                    runtime_pins=runtime_before["runtime_pins"],
                )
            )

    source_after = _frozen_json(
        _validate_source_evidence(
            attest_source_checkouts(browsergym_checkout, miniwob_checkout)
        ),
        label="post-capture source attestation",
    )
    if not _canonical_equal(source_after, source_before):
        raise RuntimeError("BrowserGym source identity drifted during capture")
    runtime_after = _frozen_json(
        _validate_runtime_evidence(
            attest_runtime(
                browser_executable,
                browser_installation,
                settings=settings,
            ),
            settings=settings,
        ),
        label="post-capture runtime attestation",
    )
    if not _canonical_equal(runtime_after, runtime_before):
        raise RuntimeError("BrowserGym runtime identity drifted during capture")
    environment_after = _frozen_json(
        _verify_production_runtime_manifest(environment_manifest),
        label="post-capture runtime manifest identity",
    )
    if not _canonical_equal(environment_after, environment_before):
        raise RuntimeError("BrowserGym runtime manifest identity drifted during capture")

    capture_payload = b"".join(
        _canonical_json_bytes(row, newline=True) for row in rows
    )
    if len(capture_payload) > DEFAULT_MAX_CAPTURE_BYTES:
        raise ValueError(
            f"BrowserGym capture exceeds {DEFAULT_MAX_CAPTURE_BYTES} bytes"
        )
    receipt_without_hash: dict[str, Any] = {
        "kind": BROWSERGYM_CAPTURE_RECEIPT_KIND,
        "schema_version": BROWSERGYM_CAPTURE_RECEIPT_SCHEMA_VERSION,
        "producer": BROWSERGYM_CAPTURE_PRODUCER,
        "boundary": _CAPTURE_BOUNDARY,
        "capture": {
            "bytes": len(capture_payload),
            "sha256": _sha256(capture_payload),
            "rows": len(rows),
            "row_keys": sorted(_CAPTURE_KEYS),
            "canonical_sorted_jsonl": True,
            "goal_provenance": _GOAL_PROVENANCE,
        },
        "plan": _plan_receipt(),
        "controls": settings.controls(),
        "source_pins": source_before["source_pins"],
        "source_attestation": source_before["repositories"],
        "runtime_pins": runtime_before["runtime_pins"],
        "runtime_attestation": runtime_before["attestation"],
        "environment_manifest": environment_before,
        "import_scope": _frozen_json(
            _IMPORT_SCOPE,
            label="BrowserGym import-scope policy",
        ),
        "publication": _PUBLICATION_CONTRACT,
    }
    receipt = {
        **receipt_without_hash,
        "receipt_self_sha256": _sha256(_canonical_json_bytes(receipt_without_hash)),
    }
    receipt_payload = _canonical_json_bytes(receipt, newline=True)
    if len(receipt_payload) > DEFAULT_MAX_RECEIPT_BYTES:
        raise ValueError(
            f"BrowserGym receipt exceeds {DEFAULT_MAX_RECEIPT_BYTES} bytes"
        )
    _preflight_outputs(capture, receipt_file)
    _publish_pair_no_clobber(
        capture,
        capture_payload,
        receipt_file,
        receipt_payload,
    )
    return receipt


def _read_stable_file_snapshot(
    path: Path,
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} must be a readable non-symlink file") from error
    chunks: list[bytes] = []
    observed_bytes = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        if before.st_size <= 0:
            raise ValueError(f"{label} must be non-empty")
        if before.st_size > maximum_bytes:
            raise ValueError(f"{label} exceeds the {maximum_bytes}-byte limit")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            observed_bytes += len(chunk)
            if observed_bytes > maximum_bytes:
                raise ValueError(f"{label} exceeds the {maximum_bytes}-byte limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        try:
            current_path = os.stat(path, follow_symlinks=False)
        except OSError as error:
            raise ValueError(f"{label} path changed while it was read") from error
    finally:
        os.close(descriptor)
    identity_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity_fields):
        raise ValueError(f"{label} changed while it was read")
    if any(
        getattr(before, field) != getattr(current_path, field)
        for field in identity_fields
    ):
        raise ValueError(f"{label} path changed while it was read")
    if observed_bytes != before.st_size:
        raise ValueError(f"{label} byte count changed while it was read")
    return b"".join(chunks)


def verify_browsergym_capture_receipt(
    capture_path: str | Path,
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Verify receipt self-hash, exact plan, canonical rows, and capture identity."""

    capture = Path(capture_path)
    receipt_file = Path(receipt_path)
    receipt_payload = _read_stable_file_snapshot(
        receipt_file,
        maximum_bytes=DEFAULT_MAX_RECEIPT_BYTES,
        label="BrowserGym receipt",
    )
    try:
        receipt_text = receipt_payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("BrowserGym receipt is not UTF-8") from error
    receipt = _strict_json_loads(receipt_text, label="BrowserGym receipt")
    if not isinstance(receipt, dict):
        raise ValueError("BrowserGym receipt must be an object")
    _require_exact_keys(receipt, _RECEIPT_KEYS, label="BrowserGym receipt")
    if receipt_payload != _canonical_json_bytes(receipt, newline=True):
        raise ValueError("BrowserGym receipt is not canonical sorted JSON")
    without_hash = dict(receipt)
    declared_self_hash = without_hash.pop("receipt_self_sha256")
    if (
        not isinstance(declared_self_hash, str)
        or _SHA256_RE.fullmatch(declared_self_hash) is None
        or _sha256(_canonical_json_bytes(without_hash)) != declared_self_hash
    ):
        raise ValueError("BrowserGym receipt self-hash mismatch")
    schema_version = receipt["schema_version"]
    if (
        receipt["kind"] != BROWSERGYM_CAPTURE_RECEIPT_KIND
        or isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != BROWSERGYM_CAPTURE_RECEIPT_SCHEMA_VERSION
        or receipt["producer"] != BROWSERGYM_CAPTURE_PRODUCER
    ):
        raise ValueError("BrowserGym receipt producer schema drift")
    if receipt["boundary"] != _CAPTURE_BOUNDARY:
        raise ValueError("BrowserGym receipt boundary drift")
    if receipt["publication"] != _PUBLICATION_CONTRACT:
        raise ValueError("BrowserGym receipt publication contract drift")
    if not _canonical_equal(receipt["import_scope"], _IMPORT_SCOPE):
        raise ValueError("BrowserGym receipt import-scope policy drift")
    if not _canonical_equal(receipt["plan"], _plan_receipt()):
        raise ValueError("BrowserGym receipt production plan drift")
    settings = BrowserGymCaptureSettings()
    if not _canonical_equal(receipt["controls"], settings.controls()):
        raise ValueError("BrowserGym receipt control drift")
    _validate_source_evidence(
        {
            "source_pins": receipt["source_pins"],
            "repositories": receipt["source_attestation"],
        }
    )
    _validate_runtime_evidence(
        {
            "runtime_pins": receipt["runtime_pins"],
            "attestation": receipt["runtime_attestation"],
        },
        settings=settings,
    )
    _validate_runtime_manifest_identity(
        receipt["environment_manifest"],
        label="BrowserGym receipt environment_manifest",
    )
    if not _canonical_equal(
        receipt["environment_manifest"],
        receipt["runtime_pins"]["environment_manifest"],
    ):
        raise ValueError(
            "BrowserGym receipt runtime manifest identity does not match runtime pins"
        )

    capture_payload = _read_stable_file_snapshot(
        capture,
        maximum_bytes=DEFAULT_MAX_CAPTURE_BYTES,
        label="BrowserGym capture",
    )
    capture_identity = {
        "bytes": len(capture_payload),
        "sha256": _sha256(capture_payload),
    }
    capture_section = receipt["capture"]
    if not isinstance(capture_section, dict):
        raise ValueError("BrowserGym receipt capture section must be an object")
    _require_exact_keys(
        capture_section,
        _CAPTURE_RECEIPT_KEYS,
        label="BrowserGym receipt capture",
    )
    _positive_int(capture_section["bytes"], label="BrowserGym receipt capture.bytes")
    capture_sha256 = capture_section["sha256"]
    if not isinstance(capture_sha256, str) or _SHA256_RE.fullmatch(capture_sha256) is None:
        raise ValueError("BrowserGym receipt capture.sha256 must be lowercase 64-hex")
    _positive_int(capture_section["rows"], label="BrowserGym receipt capture.rows")
    if capture_section["row_keys"] != sorted(_CAPTURE_KEYS):
        raise ValueError("BrowserGym receipt capture.row_keys drift")
    if capture_section["canonical_sorted_jsonl"] is not True:
        raise ValueError("BrowserGym receipt does not require canonical sorted JSONL")
    if capture_section["goal_provenance"] != _GOAL_PROVENANCE:
        raise ValueError("BrowserGym receipt goal provenance drift")
    if (
        capture_section["bytes"] != capture_identity["bytes"]
        or capture_section["sha256"] != capture_identity["sha256"]
        or capture_section["rows"] != PRODUCTION_EPISODES
    ):
        raise ValueError("BrowserGym capture identity does not match receipt")

    expected_episodes = production_capture_plan()
    observed_rows = 0
    with io.BytesIO(capture_payload) as handle:
        for episode in expected_episodes:
            raw_line = handle.readline(DEFAULT_MAX_LINE_BYTES + 1)
            observed_rows += 1
            if not raw_line or len(raw_line) > DEFAULT_MAX_LINE_BYTES:
                raise ValueError("BrowserGym capture row count or line limit drift")
            try:
                row_text = raw_line.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError("BrowserGym capture row is not UTF-8") from error
            row = _strict_json_loads(
                row_text,
                label=f"BrowserGym capture row {observed_rows}",
            )
            if not isinstance(row, dict):
                raise ValueError("BrowserGym capture row must be an object")
            _require_exact_keys(
                row,
                _CAPTURE_KEYS,
                label=f"BrowserGym capture row {observed_rows}",
            )
            if raw_line != _canonical_json_bytes(row, newline=True):
                raise ValueError("BrowserGym capture row is not canonical sorted JSON")
            expected_episode_fields = {
                "task_name": episode.task_name,
                "seed": episode.seed,
                "similarity_group": episode.similarity_group,
                "split": episode.split,
            }
            if (
                not isinstance(row["task_name"], str)
                or isinstance(row["seed"], bool)
                or not isinstance(row["seed"], int)
                or isinstance(row["similarity_group"], bool)
                or not isinstance(row["similarity_group"], int)
                or not isinstance(row["split"], str)
                or any(
                    row[key] != value
                    for key, value in expected_episode_fields.items()
                )
            ):
                raise ValueError("BrowserGym capture episode order or plan drift")
            if not isinstance(row["goal"], str) or not row["goal"]:
                raise ValueError("BrowserGym capture goal must be a non-empty string")
            if len(row["goal"].encode("utf-8")) > DEFAULT_MAX_PROMPT_BYTES:
                raise ValueError("BrowserGym capture goal exceeds the prompt limit")
            if not _canonical_equal(row["source_pins"], receipt["source_pins"]):
                raise ValueError("BrowserGym capture source pins do not match receipt")
            if not _canonical_equal(row["runtime_pins"], receipt["runtime_pins"]):
                raise ValueError("BrowserGym capture runtime pins do not match receipt")
        if handle.read(1):
            raise ValueError("BrowserGym capture contains unexpected extra rows")
    return receipt


__all__ = [
    "BROWSERGYM_CAPTURE_PRODUCER",
    "BROWSERGYM_CAPTURE_RECEIPT_KIND",
    "BROWSERGYM_CAPTURE_RECEIPT_SCHEMA_VERSION",
    "BrowserGymCaptureEpisode",
    "BrowserGymCaptureSettings",
    "PRODUCTION_ACTION_SET",
    "PRODUCTION_DEVICE_SCALE_FACTOR",
    "PRODUCTION_HEADLESS",
    "PRODUCTION_LOCALE",
    "PRODUCTION_OBSERVATION_MODE",
    "PRODUCTION_TIMEOUT_SECONDS",
    "PRODUCTION_TIMEZONE_ID",
    "PRODUCTION_VIEWPORT",
    "attest_git_checkout",
    "attest_runtime",
    "attest_source_checkouts",
    "capture_browsergym_goals",
    "make_browsergym_environment",
    "production_capture_plan",
    "verify_browsergym_capture_receipt",
]
