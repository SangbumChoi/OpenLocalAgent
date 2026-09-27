"""Offline BrowserGym/MiniWoB reset-capture validation for decontamination prompts.

The adapter never imports or runs BrowserGym.  A separate, controlled environment must first
capture reset-returned goals together with immutable source and runtime pins.  This module checks
that capture and emits only ``source_case_id`` and ``prompt`` rows for corpus decontamination.
It is not an episode scorer and is not evidence of chronologically fresh evaluation.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

BROWSERGYM_PROMPT_AUDIT_KIND = "localagent_browsergym_miniwob_prompt_export_audit"
BROWSERGYM_PROMPT_AUDIT_SCHEMA_VERSION = 2
BROWSERGYM_PROMPT_ADAPTER = "browsergym-miniwob-reset-capture-prompt-rows-v1"

PRODUCTION_BROWSERGYM_REVISION = "9e779f087de9a65668b6974d11f9ce9816026e96"
PRODUCTION_BROWSERGYM_VERSION = "0.14.3"
PRODUCTION_MINIWOB_REVISION = "7fd85d71a4b60325c6585396ec4f48377d049838"
PRODUCTION_PLAYWRIGHT_VERSION = "1.44.0"
PRODUCTION_CHROMIUM_REVISION = "1117"
PRODUCTION_CHROMIUM_VERSION = "125.0.6422.26"
PRODUCTION_RUNTIME_MANIFEST_IDENTITY: Mapping[str, Any] = {
    "kind": "localagent_browsergym_capture_environment",
    "schema_version": 1,
    "file": "browsergym-capture-runtime-darwin-arm64-py312.json",
    "bytes": 12_996,
    "sha256": "5edf3987b09db987eabbef52324ef6d0eb87d69e7c36e94d5f88cdccddf21382",
    "self_sha256": "274802850e0bef5635b906a668f49e3c540e459dee1841b20a54e55ccc3863c7",
    "distributions": 51,
    "playwright_driver_sha256": (
        "cb628761b7355e456bd2581f8a0b008d200ca3e2a6f53c466c3c245b63db26da"
    ),
}
PRODUCTION_FIXED_SEEDS = (11, 17, 23, 29)
# Frozen only after two sequential controlled acquisitions were byte-identical and independently
# verified against the producer receipt.
PRODUCTION_CAPTURE_FILE = "browsergym-miniwob-reset-goals.jsonl"
PRODUCTION_CAPTURE_BYTES: int | None = 348_513
PRODUCTION_CAPTURE_SHA256: str | None = (
    "128f7f6be8d5b52f745523b0bca4517fdaf8107044eee5a76366464ac10079ff"
)
PRODUCTION_CAPTURE_RECEIPT_IDENTITY: Mapping[str, Any] = {
    "status": "frozen_controlled_acquisition",
    "file": "browsergym-miniwob-reset-goals.receipt.json",
    "bytes": 6_538,
    "sha256": "b04318c36579a05d3f61a40ea09c1f1c0bd1e004a534b2b5d18305b50e68ebea",
    "receipt_self_sha256": (
        "e8cece5a8acf0f5e2333e004c33b035b4b31fa7ec3e3d501c43fcbbac341611a"
    ),
    "kind": "localagent_browsergym_capture_producer_receipt",
    "schema_version": 3,
    "producer": "browsergym-miniwob-controlled-reset-goals-v3",
}
PRODUCTION_LOCAL_POLICY_EXCLUSIONS = (
    "click-pie",
    "click-pie-nodelay",
    "terminal",
)
# Backward-compatible public alias. The pinned upstream metadata does not itself mark these tasks
# nondeterministic; their omission is a LocalAgent production-plan policy choice.
PRODUCTION_EXCLUDED_NONDETERMINISTIC = PRODUCTION_LOCAL_POLICY_EXCLUSIONS

# BrowserGym metadata at PRODUCTION_BROWSERGYM_REVISION, browsergym_split=test, after applying the
# three LocalAgent production-plan exclusions above.
PRODUCTION_TASK_GROUPS: Mapping[str, int] = {
    "miniwob.ascending-numbers": 0,
    "miniwob.bisect-angle": 1,
    "miniwob.choose-list": 5,
    "miniwob.click-button": 7,
    "miniwob.click-button-sequence": 8,
    "miniwob.click-color": 11,
    "miniwob.click-link": 13,
    "miniwob.click-menu": 14,
    "miniwob.click-menu-2": 14,
    "miniwob.click-scroll-list": 17,
    "miniwob.click-shape": 19,
    "miniwob.click-tab": 20,
    "miniwob.click-tab-2": 21,
    "miniwob.click-tab-2-easy": 21,
    "miniwob.click-tab-2-hard": 21,
    "miniwob.click-tab-2-medium": 21,
    "miniwob.click-test": 22,
    "miniwob.click-widget": 24,
    "miniwob.copy-paste": 25,
    "miniwob.copy-paste-2": 25,
    "miniwob.count-sides": 27,
    "miniwob.drag-circle": 30,
    "miniwob.drag-cube": 31,
    "miniwob.drag-items-grid": 33,
    "miniwob.email-inbox": 37,
    "miniwob.email-inbox-delete": 37,
    "miniwob.email-inbox-forward": 37,
    "miniwob.email-inbox-forward-nl": 37,
    "miniwob.email-inbox-forward-nl-turk": 37,
    "miniwob.email-inbox-important": 37,
    "miniwob.email-inbox-nl-turk": 37,
    "miniwob.email-inbox-noscroll": 37,
    "miniwob.email-inbox-reply": 37,
    "miniwob.email-inbox-star-reply": 37,
    "miniwob.find-greatest": 39,
    "miniwob.find-word": 41,
    "miniwob.form-sequence-2": 44,
    "miniwob.form-sequence-3": 45,
    "miniwob.generate-number": 46,
    "miniwob.grid-coordinate": 47,
    "miniwob.guess-number": 48,
    "miniwob.highlight-text": 49,
    "miniwob.highlight-text-2": 49,
    "miniwob.hot-cold": 50,
    "miniwob.number-checkboxes": 55,
    "miniwob.order-food": 57,
    "miniwob.phone-book": 58,
    "miniwob.resize-textarea": 60,
    "miniwob.right-angle": 61,
    "miniwob.scroll-text": 62,
    "miniwob.scroll-text-2": 62,
    "miniwob.sign-agreement": 64,
    "miniwob.simple-algebra": 65,
    "miniwob.simple-arithmetic": 65,
    "miniwob.social-media": 66,
    "miniwob.social-media-all": 66,
    "miniwob.social-media-some": 66,
    "miniwob.stock-market": 67,
    "miniwob.tic-tac-toe": 71,
    "miniwob.use-spinner": 76,
}
PRODUCTION_TASK_VARIANTS = 60
PRODUCTION_SIMILARITY_GROUPS = 41
PRODUCTION_EPISODES = 240
PRODUCTION_MAX_STEPS = 10

DEFAULT_MAX_CAPTURE_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_LINE_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_PROMPT_BYTES = 512 * 1024
DEFAULT_MAX_CAPTURE_ROWS = 10_000

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
_TASK_NAME_RE = re.compile(r"miniwob\.[a-z0-9][a-z0-9-]*\Z")
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
_VIEWPORT_KEYS = frozenset({"width", "height"})
_IDENTITY_KEYS = frozenset({"bytes", "sha256"})
_RUNTIME_MANIFEST_IDENTITY_KEYS = frozenset(PRODUCTION_RUNTIME_MANIFEST_IDENTITY)
_CAPTURE_RECEIPT_IDENTITY_KEYS = frozenset(
    {
        "bytes",
        "file",
        "kind",
        "producer",
        "receipt_self_sha256",
        "schema_version",
        "sha256",
    }
)
_CAPTURE_RECEIPT_POLICY_KEYS = _CAPTURE_RECEIPT_IDENTITY_KEYS | {"status"}


@dataclass(frozen=True)
class BrowserGymPromptLimits:
    """Resource limits for bounded local reset-capture parsing."""

    max_capture_bytes: int = DEFAULT_MAX_CAPTURE_BYTES
    max_line_bytes: int = DEFAULT_MAX_LINE_BYTES
    max_prompt_bytes: int = DEFAULT_MAX_PROMPT_BYTES
    max_capture_rows: int = DEFAULT_MAX_CAPTURE_ROWS

    def validate(self) -> None:
        for name, value in (
            ("max_capture_bytes", self.max_capture_bytes),
            ("max_line_bytes", self.max_line_bytes),
            ("max_prompt_bytes", self.max_prompt_bytes),
            ("max_capture_rows", self.max_capture_rows),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class _EpisodePrompt:
    task_name: str
    seed: int
    goal: str
    similarity_group: int
    split: str

    @property
    def source_case_id(self) -> str:
        return f"browsergym:{self.task_name}:{self.seed}"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


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


def _nonempty_string(value: Any, *, label: str, max_length: int = 1024) -> str:
    if not isinstance(value, str) or not value or len(value) > max_length:
        raise ValueError(f"{label} must be a non-empty string of at most {max_length} characters")
    return value


def _validate_source_pins(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    _require_exact_keys(value, _SOURCE_PIN_KEYS, label=label)
    browsergym_revision = value["browsergym_revision"]
    miniwob_revision = value["miniwob_revision"]
    if (
        not isinstance(browsergym_revision, str)
        or _GIT_REVISION_RE.fullmatch(browsergym_revision) is None
    ):
        raise ValueError(f"{label}.browsergym_revision must be lowercase 40-hex")
    if (
        not isinstance(miniwob_revision, str)
        or _GIT_REVISION_RE.fullmatch(miniwob_revision) is None
    ):
        raise ValueError(f"{label}.miniwob_revision must be lowercase 40-hex")
    _nonempty_string(value["browsergym_version"], label=f"{label}.browsergym_version")
    return value


def _validate_runtime_pins(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    _require_exact_keys(value, _RUNTIME_PIN_KEYS, label=label)
    for key in (
        "playwright_version",
        "chromium_revision",
        "chromium_version",
        "python_version",
        "os",
        "architecture",
        "locale",
        "timezone_id",
        "action_set",
        "observation_mode",
    ):
        _nonempty_string(value[key], label=f"{label}.{key}")
    if not isinstance(value["headless"], bool):
        raise ValueError(f"{label}.headless must be boolean")
    _positive_int(value["max_steps"], label=f"{label}.max_steps")
    timeout = value["playwright_operation_timeout_seconds"]
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError(
            f"{label}.playwright_operation_timeout_seconds must be finite and positive"
        )
    viewport = value["viewport"]
    if not isinstance(viewport, dict):
        raise ValueError(f"{label}.viewport must be an object")
    _require_exact_keys(viewport, _VIEWPORT_KEYS, label=f"{label}.viewport")
    _positive_int(viewport["width"], label=f"{label}.viewport.width")
    _positive_int(viewport["height"], label=f"{label}.viewport.height")
    scale = value["device_scale_factor"]
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError(f"{label}.device_scale_factor must be finite and positive")
    for key in ("browser_executable", "browser_installation"):
        identity = value[key]
        if not isinstance(identity, dict):
            raise ValueError(f"{label}.{key} must be an object")
        _require_exact_keys(identity, _IDENTITY_KEYS, label=f"{label}.{key}")
        _positive_int(identity["bytes"], label=f"{label}.{key}.bytes")
        sha256 = identity["sha256"]
        if not isinstance(sha256, str) or _SHA256_RE.fullmatch(sha256) is None:
            raise ValueError(f"{label}.{key}.sha256 must be lowercase 64-hex")
    manifest_identity = value["environment_manifest"]
    if not isinstance(manifest_identity, dict):
        raise ValueError(f"{label}.environment_manifest must be an object")
    _require_exact_keys(
        manifest_identity,
        _RUNTIME_MANIFEST_IDENTITY_KEYS,
        label=f"{label}.environment_manifest",
    )
    for key in ("kind", "file"):
        _nonempty_string(
            manifest_identity[key],
            label=f"{label}.environment_manifest.{key}",
        )
    for key in ("schema_version", "bytes", "distributions"):
        _positive_int(
            manifest_identity[key],
            label=f"{label}.environment_manifest.{key}",
        )
    for key in ("sha256", "self_sha256", "playwright_driver_sha256"):
        digest = manifest_identity[key]
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(
                f"{label}.environment_manifest.{key} must be lowercase 64-hex"
            )
    return value


def _validate_production_pins(
    source_pins: Mapping[str, Any],
    runtime_pins: Mapping[str, Any],
) -> None:
    expected_source = {
        "browsergym_revision": PRODUCTION_BROWSERGYM_REVISION,
        "browsergym_version": PRODUCTION_BROWSERGYM_VERSION,
        "miniwob_revision": PRODUCTION_MINIWOB_REVISION,
    }
    if source_pins != expected_source:
        raise ValueError(
            f"production BrowserGym source pins mismatch: expected {expected_source}"
        )
    expected_runtime = {
        "playwright_version": PRODUCTION_PLAYWRIGHT_VERSION,
        "chromium_revision": PRODUCTION_CHROMIUM_REVISION,
        "chromium_version": PRODUCTION_CHROMIUM_VERSION,
    }
    observed = {key: runtime_pins.get(key) for key in expected_runtime}
    if observed != expected_runtime:
        raise ValueError(
            f"production BrowserGym runtime pins mismatch: expected {expected_runtime}"
        )
    if runtime_pins.get("max_steps") != PRODUCTION_MAX_STEPS:
        raise ValueError(
            "production BrowserGym max_steps mismatch: "
            f"expected {PRODUCTION_MAX_STEPS}"
        )
    if runtime_pins.get("environment_manifest") != PRODUCTION_RUNTIME_MANIFEST_IDENTITY:
        raise ValueError("production BrowserGym runtime manifest identity mismatch")


def _parse_capture(
    capture_payload: bytes,
    *,
    limits: BrowserGymPromptLimits,
) -> tuple[list[_EpisodePrompt], dict[str, Any], dict[str, Any]]:
    episodes: list[_EpisodePrompt] = []
    source_pins: dict[str, Any] | None = None
    runtime_pins: dict[str, Any] | None = None
    seen_pairs: set[tuple[str, int]] = set()
    task_groups: dict[str, int] = {}

    with io.BytesIO(capture_payload) as handle:
        while True:
            raw_line = handle.readline(limits.max_line_bytes + 1)
            if not raw_line:
                break
            line_number = len(episodes) + 1
            if line_number > limits.max_capture_rows:
                raise ValueError(
                    "BrowserGym reset capture exceeds "
                    f"the {limits.max_capture_rows}-row limit"
                )
            if len(raw_line) > limits.max_line_bytes:
                raise ValueError(
                    f"BrowserGym reset capture line {line_number} exceeds "
                    f"the {limits.max_line_bytes}-byte line limit"
                )
            if not raw_line.endswith(b"\n"):
                raise ValueError(
                    f"BrowserGym reset capture line {line_number} is not newline-terminated"
                )
            if not raw_line.strip():
                raise ValueError(f"BrowserGym reset capture line {line_number} is blank")
            try:
                decoded = raw_line.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(
                    f"BrowserGym reset capture line {line_number} is not UTF-8"
                ) from error
            row = _strict_json_loads(
                decoded,
                label=f"BrowserGym reset capture line {line_number}",
            )
            if not isinstance(row, dict):
                raise ValueError(
                    f"BrowserGym reset capture line {line_number} must be an object"
                )
            _require_exact_keys(
                row,
                _CAPTURE_KEYS,
                label=f"BrowserGym reset capture line {line_number}",
            )
            task_name = row["task_name"]
            if (
                not isinstance(task_name, str)
                or _TASK_NAME_RE.fullmatch(task_name) is None
            ):
                raise ValueError(
                    f"BrowserGym reset capture line {line_number} has an invalid task_name"
                )
            seed = _nonnegative_int(
                row["seed"],
                label=f"BrowserGym reset capture line {line_number}.seed",
            )
            if seed >= 2**32:
                raise ValueError(
                    f"BrowserGym reset capture line {line_number}.seed must be below 2**32"
                )
            goal = row["goal"]
            if not isinstance(goal, str) or not goal:
                raise ValueError(
                    f"BrowserGym reset capture line {line_number}.goal "
                    "must be a non-empty string"
                )
            if len(goal.encode("utf-8")) > limits.max_prompt_bytes:
                raise ValueError(
                    f"BrowserGym reset capture line {line_number}.goal exceeds "
                    f"the {limits.max_prompt_bytes}-byte prompt limit"
                )
            similarity_group = _nonnegative_int(
                row["similarity_group"],
                label=f"BrowserGym reset capture line {line_number}.similarity_group",
            )
            split = row["split"]
            if split not in {"train", "test"}:
                raise ValueError(
                    f"BrowserGym reset capture line {line_number}.split "
                    "must be 'train' or 'test'"
                )
            observed_source_pins = _validate_source_pins(
                row["source_pins"],
                label=f"BrowserGym reset capture line {line_number}.source_pins",
            )
            observed_runtime_pins = _validate_runtime_pins(
                row["runtime_pins"],
                label=f"BrowserGym reset capture line {line_number}.runtime_pins",
            )
            if source_pins is None:
                source_pins = observed_source_pins
                runtime_pins = observed_runtime_pins
            elif (
                observed_source_pins != source_pins
                or observed_runtime_pins != runtime_pins
            ):
                raise ValueError("BrowserGym reset capture pins differ between rows")

            pair = (task_name, seed)
            if pair in seen_pairs:
                raise ValueError(
                    f"duplicate BrowserGym reset capture episode {task_name!r} seed {seed}"
                )
            seen_pairs.add(pair)
            prior_group = task_groups.setdefault(task_name, similarity_group)
            if prior_group != similarity_group:
                raise ValueError(
                    f"BrowserGym task {task_name!r} has inconsistent similarity_group values"
                )
            episodes.append(
                _EpisodePrompt(
                    task_name=task_name,
                    seed=seed,
                    goal=goal,
                    similarity_group=similarity_group,
                    split=split,
                )
            )
    if not episodes or source_pins is None or runtime_pins is None:
        raise ValueError("BrowserGym reset capture must contain at least one row")
    return episodes, source_pins, runtime_pins


def _validate_production_plan(episodes: list[_EpisodePrompt]) -> None:
    if len(episodes) != PRODUCTION_EPISODES:
        raise ValueError(
            "production BrowserGym capture episode-count mismatch: "
            f"expected {PRODUCTION_EPISODES}, observed {len(episodes)}"
        )
    if {episode.split for episode in episodes} != {"test"}:
        raise ValueError("production BrowserGym capture must contain only the test split")
    observed_task_groups = {
        episode.task_name: episode.similarity_group for episode in episodes
    }
    if observed_task_groups != dict(PRODUCTION_TASK_GROUPS):
        missing = sorted(set(PRODUCTION_TASK_GROUPS) - set(observed_task_groups))
        extra = sorted(set(observed_task_groups) - set(PRODUCTION_TASK_GROUPS))
        mismatched = sorted(
            task
            for task in set(observed_task_groups) & set(PRODUCTION_TASK_GROUPS)
            if observed_task_groups[task] != PRODUCTION_TASK_GROUPS[task]
        )
        raise ValueError(
            "production BrowserGym task/group plan mismatch: "
            f"missing={missing}, extra={extra}, mismatched={mismatched}"
        )
    observed_groups = set(observed_task_groups.values())
    if len(observed_groups) != PRODUCTION_SIMILARITY_GROUPS:
        raise ValueError(
            "production BrowserGym similarity-group count mismatch: "
            f"expected {PRODUCTION_SIMILARITY_GROUPS}, observed {len(observed_groups)}"
        )
    expected_pairs = {
        (task_name, seed)
        for task_name in PRODUCTION_TASK_GROUPS
        for seed in PRODUCTION_FIXED_SEEDS
    }
    observed_pairs = {(episode.task_name, episode.seed) for episode in episodes}
    if observed_pairs != expected_pairs:
        missing = sorted(expected_pairs - observed_pairs)
        extra = sorted(observed_pairs - expected_pairs)
        raise ValueError(
            "production BrowserGym task/seed plan mismatch: "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )


def _matches_payload(path: Path, payload: bytes) -> bool:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != len(payload)
    ):
        return False
    offset = 0
    with path.open("rb") as handle:
        while offset < len(payload):
            chunk = handle.read(min(1024 * 1024, len(payload) - offset))
            if not chunk or chunk != payload[offset : offset + len(chunk)]:
                return False
            offset += len(chunk)
        return handle.read(1) == b""


def _assert_existing_exact(path: Path, payload: bytes) -> None:
    if path.exists() and not _matches_payload(path, payload):
        raise RuntimeError(f"refusing to overwrite drifted derived artifact: {path}")


def _publish_atomic(path: Path, payload: bytes) -> None:
    if path.exists():
        if not _matches_payload(path, payload):
            raise RuntimeError(f"refusing to overwrite drifted derived artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError:
            if not _matches_payload(path, payload):
                raise RuntimeError(
                    f"refusing to overwrite concurrently-created artifact: {path}"
                )
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    if not _matches_payload(path, payload):
        raise RuntimeError(f"published artifact failed verification: {path}")


def _rows_payload(episodes: Iterable[_EpisodePrompt]) -> bytes:
    return b"".join(
        _canonical_json_bytes(
            {"source_case_id": episode.source_case_id, "prompt": episode.goal},
            newline=True,
        )
        for episode in episodes
    )


def _grouping(episodes: Iterable[_EpisodePrompt]) -> dict[str, list[str]]:
    groups: dict[int, set[str]] = {}
    for episode in episodes:
        groups.setdefault(episode.similarity_group, set()).add(episode.task_name)
    return {
        str(group): sorted(tasks)
        for group, tasks in sorted(groups.items(), key=lambda item: item[0])
    }


def _read_stable_regular_file(
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
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1))
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


def _verified_receipt_identity(
    capture: Path,
    receipt_file: Path,
    *,
    capture_bytes: int,
    capture_sha256: str,
) -> dict[str, Any]:
    # browsergym_capture imports this module for the immutable production plan, so this import must
    # remain local to the production consumer path.
    from localagent.data.browsergym_capture import (
        DEFAULT_MAX_RECEIPT_BYTES,
        verify_browsergym_capture_receipt,
    )

    verified = verify_browsergym_capture_receipt(capture, receipt_file)
    if not isinstance(verified, Mapping):
        raise ValueError("BrowserGym receipt verifier did not return an object")
    receipt_payload = _read_stable_regular_file(
        receipt_file,
        maximum_bytes=DEFAULT_MAX_RECEIPT_BYTES,
        label="BrowserGym capture receipt",
    )
    try:
        receipt_text = receipt_payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("BrowserGym capture receipt is not UTF-8") from error
    observed = _strict_json_loads(receipt_text, label="BrowserGym capture receipt")
    if (
        not isinstance(observed, dict)
        or observed != dict(verified)
        or receipt_payload != _canonical_json_bytes(dict(verified), newline=True)
    ):
        raise ValueError("BrowserGym capture receipt changed after verification")

    kind = _nonempty_string(verified.get("kind"), label="BrowserGym receipt.kind")
    schema_version = _positive_int(
        verified.get("schema_version"),
        label="BrowserGym receipt.schema_version",
    )
    producer = _nonempty_string(
        verified.get("producer"),
        label="BrowserGym receipt.producer",
    )
    receipt_self_sha256 = verified.get("receipt_self_sha256")
    if (
        not isinstance(receipt_self_sha256, str)
        or _SHA256_RE.fullmatch(receipt_self_sha256) is None
    ):
        raise ValueError("BrowserGym receipt.receipt_self_sha256 must be lowercase 64-hex")
    receipt_capture = verified.get("capture")
    if not isinstance(receipt_capture, Mapping):
        raise ValueError("BrowserGym receipt.capture must be an object")
    if (
        receipt_capture.get("bytes") != capture_bytes
        or receipt_capture.get("sha256") != capture_sha256
    ):
        raise ValueError("BrowserGym receipt capture identity disagrees with the prompt input")
    return {
        "bytes": len(receipt_payload),
        "file": receipt_file.name,
        "kind": kind,
        "producer": producer,
        "receipt_self_sha256": receipt_self_sha256,
        "schema_version": schema_version,
        "sha256": _sha256(receipt_payload),
    }


def _validate_frozen_production_identities(
    *,
    capture_file: str,
    capture_bytes: int,
    capture_sha256: str,
    receipt_identity: Mapping[str, Any],
) -> None:
    if (
        PRODUCTION_CAPTURE_BYTES is None
        or PRODUCTION_CAPTURE_SHA256 is None
    ):
        raise ValueError(
            "production BrowserGym capture origin is not frozen; "
            "set its audited byte size and SHA-256 only after controlled acquisition"
        )
    if (
        capture_file != PRODUCTION_CAPTURE_FILE
        or capture_bytes != PRODUCTION_CAPTURE_BYTES
        or capture_sha256 != PRODUCTION_CAPTURE_SHA256
    ):
        raise ValueError(
            "production BrowserGym capture identity does not match the "
            "controlled acquisition pins"
        )

    policy = PRODUCTION_CAPTURE_RECEIPT_IDENTITY
    if not isinstance(policy, Mapping):
        raise ValueError("production BrowserGym receipt identity policy must be an object")
    _require_exact_keys(
        policy,
        _CAPTURE_RECEIPT_POLICY_KEYS,
        label="production BrowserGym receipt identity policy",
    )
    expected = {key: policy[key] for key in _CAPTURE_RECEIPT_IDENTITY_KEYS}
    if (
        policy.get("status") != "frozen_controlled_acquisition"
        or any(value is None for value in expected.values())
    ):
        raise ValueError(
            "production BrowserGym receipt origin is not frozen; "
            "freeze its file bytes, SHA-256, self-hash, and producer evidence "
            "after controlled acquisition"
        )
    if dict(receipt_identity) != expected:
        raise ValueError(
            "production BrowserGym receipt identity does not match the "
            "controlled acquisition pins"
        )


def export_browsergym_prompt_rows(
    capture_path: str | Path,
    output_path: str | Path,
    audit_path: str | Path,
    *,
    expected_capture_bytes: int,
    expected_capture_sha256: str,
    receipt_path: str | Path | None = None,
    production: bool = True,
    limits: BrowserGymPromptLimits = BrowserGymPromptLimits(),
) -> dict[str, Any]:
    """Validate an offline reset capture and publish two-field prompt-only JSONL.

    Production mode requires the capture producer's verified receipt and enforces the pinned
    60-task, 41-group, four-seed MiniWoB test plan. ``production=False`` exists for adapter
    fixtures and exploratory captures; it does not weaken the default CLI behavior.
    """

    limits.validate()
    capture = Path(capture_path)
    output = Path(output_path)
    audit_file = Path(audit_path)
    receipt_file = Path(receipt_path) if receipt_path is not None else None
    if production and receipt_file is None:
        raise ValueError("production BrowserGym prompt export requires receipt_path")
    distinct_paths = {capture.resolve(), output.resolve(), audit_file.resolve()}
    if receipt_file is not None:
        distinct_paths.add(receipt_file.resolve())
    expected_distinct = 4 if receipt_file is not None else 3
    if len(distinct_paths) != expected_distinct:
        raise ValueError(
            "capture_path, receipt_path, output_path, and audit_path must be different files"
        )
    if (
        isinstance(expected_capture_bytes, bool)
        or not isinstance(expected_capture_bytes, int)
        or expected_capture_bytes < 0
    ):
        raise ValueError("expected_capture_bytes must be a nonnegative integer")
    if (
        not isinstance(expected_capture_sha256, str)
        or _SHA256_RE.fullmatch(expected_capture_sha256) is None
    ):
        raise ValueError("expected_capture_sha256 must be lowercase 64-hex")
    if not capture.is_file():
        raise ValueError(f"BrowserGym reset capture is missing or not a file: {capture}")
    if capture.is_symlink():
        raise ValueError("BrowserGym reset capture must not be a symbolic link")
    observed_bytes = capture.stat().st_size
    if observed_bytes > limits.max_capture_bytes:
        raise ValueError(
            "BrowserGym reset capture exceeds "
            f"the {limits.max_capture_bytes}-byte capture limit"
        )
    with capture.open("rb") as capture_handle:
        capture_payload = capture_handle.read(limits.max_capture_bytes + 1)
    if len(capture_payload) > limits.max_capture_bytes:
        raise ValueError(
            "BrowserGym reset capture exceeds "
            f"the {limits.max_capture_bytes}-byte capture limit"
        )
    observed_bytes = len(capture_payload)
    if observed_bytes != expected_capture_bytes:
        raise ValueError(
            "BrowserGym reset capture byte-size mismatch: "
            f"expected {expected_capture_bytes}, got {observed_bytes}"
        )
    observed_sha256 = _sha256(capture_payload)
    if observed_sha256 != expected_capture_sha256:
        raise ValueError("BrowserGym reset capture SHA-256 mismatch")

    receipt_identity: dict[str, Any] | None = None
    if production:
        assert receipt_file is not None
        receipt_identity = _verified_receipt_identity(
            capture,
            receipt_file,
            capture_bytes=observed_bytes,
            capture_sha256=observed_sha256,
        )

    episodes, source_pins, runtime_pins = _parse_capture(
        capture_payload,
        limits=limits,
    )
    if production:
        assert receipt_identity is not None
        _validate_production_pins(source_pins, runtime_pins)
        _validate_production_plan(episodes)
        _validate_frozen_production_identities(
            capture_file=capture.name,
            capture_bytes=observed_bytes,
            capture_sha256=observed_sha256,
            receipt_identity=receipt_identity,
        )

    ordered = sorted(episodes, key=lambda episode: (episode.task_name, episode.seed))
    source_case_ids = [episode.source_case_id for episode in ordered]
    if len(source_case_ids) != len(set(source_case_ids)):
        raise ValueError("derived BrowserGym prompt source_case_id collision")
    output_payload = _rows_payload(ordered)
    grouping = _grouping(ordered)
    task_groups = {
        episode.task_name: episode.similarity_group
        for episode in sorted(ordered, key=lambda item: item.task_name)
    }
    grouping_payload = _canonical_json_bytes(task_groups)
    seeds = sorted({episode.seed for episode in ordered})
    splits = sorted({episode.split for episode in ordered})
    split = "+".join(splits)
    mode = "production" if production else "fixture"
    audit_without_hash: dict[str, Any] = {
        "kind": BROWSERGYM_PROMPT_AUDIT_KIND,
        "schema_version": BROWSERGYM_PROMPT_AUDIT_SCHEMA_VERSION,
        "adapter": BROWSERGYM_PROMPT_ADAPTER,
        "freeze_binding": {
            "adapter": BROWSERGYM_PROMPT_ADAPTER,
            "benchmark": "browsergym-miniwob",
            "mode": mode,
            "revision": source_pins["browsergym_revision"],
            "split": split,
            "prompt_only": True,
            "contains_current_step_labels": False,
            "output": {
                "bytes": len(output_payload),
                "sha256": _sha256(output_payload),
                "records": len(ordered),
            },
        },
        "purpose": "prompt_only_corpus_decontamination",
        "benchmark": "browsergym-miniwob",
        "mode": mode,
        "revision": source_pins["browsergym_revision"],
        "split": split,
        "capture": {
            "bytes": observed_bytes,
            "file": capture.name,
            "rows": len(ordered),
            "sha256": observed_sha256,
        },
        **(
            {"capture_receipt": receipt_identity}
            if receipt_identity is not None
            else {}
        ),
        "source_pins": source_pins,
        "runtime_pins": runtime_pins,
        "plan": {
            "episode_rows": len(ordered),
            "localagent_policy_exclusions": (
                list(PRODUCTION_LOCAL_POLICY_EXCLUSIONS) if production else []
            ),
            "fixed_seeds": seeds,
            "grouping_sha256": _sha256(grouping_payload),
            "similarity_group_count": len(grouping),
            "similarity_groups": grouping,
            "splits": splits,
            "task_groups": dict(sorted(task_groups.items())),
            "task_variants": len(task_groups),
        },
        "output": {
            "bytes": len(output_payload),
            "file": output.name,
            "rows": len(ordered),
            "sha256": _sha256(output_payload),
            "row_keys": ["prompt", "source_case_id"],
        },
        "limits": {
            "max_capture_bytes": limits.max_capture_bytes,
            "max_capture_rows": limits.max_capture_rows,
            "max_line_bytes": limits.max_line_bytes,
            "max_prompt_bytes": limits.max_prompt_bytes,
        },
        "boundary": (
            "offline reset-goal decontamination only; no environment execution, episode scoring, "
            "or chronologically fresh-evaluation claim"
        ),
    }
    audit = {
        **audit_without_hash,
        "audit_self_sha256": _sha256(_canonical_json_bytes(audit_without_hash)),
    }
    audit_payload = _canonical_json_bytes(audit, newline=True)

    _assert_existing_exact(output, output_payload)
    _assert_existing_exact(audit_file, audit_payload)
    _publish_atomic(output, output_payload)
    _publish_atomic(audit_file, audit_payload)
    return audit


__all__ = [
    "BROWSERGYM_PROMPT_AUDIT_KIND",
    "BROWSERGYM_PROMPT_ADAPTER",
    "BrowserGymPromptLimits",
    "PRODUCTION_BROWSERGYM_REVISION",
    "PRODUCTION_BROWSERGYM_VERSION",
    "PRODUCTION_CAPTURE_BYTES",
    "PRODUCTION_CAPTURE_FILE",
    "PRODUCTION_CAPTURE_RECEIPT_IDENTITY",
    "PRODUCTION_CAPTURE_SHA256",
    "PRODUCTION_CHROMIUM_REVISION",
    "PRODUCTION_CHROMIUM_VERSION",
    "PRODUCTION_EPISODES",
    "PRODUCTION_EXCLUDED_NONDETERMINISTIC",
    "PRODUCTION_FIXED_SEEDS",
    "PRODUCTION_LOCAL_POLICY_EXCLUSIONS",
    "PRODUCTION_MINIWOB_REVISION",
    "PRODUCTION_MAX_STEPS",
    "PRODUCTION_PLAYWRIGHT_VERSION",
    "PRODUCTION_RUNTIME_MANIFEST_IDENTITY",
    "PRODUCTION_SIMILARITY_GROUPS",
    "PRODUCTION_TASK_GROUPS",
    "PRODUCTION_TASK_VARIANTS",
    "export_browsergym_prompt_rows",
]
