"""Canonical conversation loading and single-turn training views.

The on-disk interchange format remains :class:`localagent.data.schema.Conversation`.  Some
existing optimization kernels predate that schema and consume the compact ``Sample`` view used by
the deterministic generator.  This module provides the one audited conversion point instead of
letting each stage invent an ad-hoc JSON shape.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from localagent.data.agent_synth import Sample
from localagent.data.conversation_artifact import (
    MANIFEST_SCHEMA_VERSION as CONVERSATION_MANIFEST_SCHEMA_VERSION,
)
from localagent.data.conversation_artifact import (
    load_verified_conversation_artifact,
)
from localagent.data.render import history_text
from localagent.data.schema import Conversation, Role
from localagent.model.config import ModelConfig
from localagent.model.tokenizer import ASSISTANT

LINEAGE_VERSION = 1
_UNTRACKED_LINEAGE_CONTROL_DIRS = frozenset({".agents", ".codex"})
_PARENT_STAGE = {
    "midtrain": "pretrain",
    "sft": "midtrain",
    "rl": "sft",
}
_MODEL_COMPATIBILITY_FIELDS = (
    "vocab_size",
    "d_model",
    "embed_dim",
    "n_layers",
    "n_loops",
    "n_heads",
    "n_kv_heads",
    "ffn_hidden",
    "max_seq_len",
    "rope_theta",
    "norm_eps",
    "tie_embeddings",
    "dropout",
    "qk_norm",
    "conv_kernel",
)
_CONVERSATION_ARTIFACT_KEYS = frozenset(
    {
        "environment_policy",
        "expected_rule_verified",
        "expected_split",
        "generator_config",
        "manifest",
    }
)
_VERSIONED_CONVERSATION_MANIFEST_SUFFIX = f".manifest.v{CONVERSATION_MANIFEST_SCHEMA_VERSION}.json"


def _canonical_value(value: Any) -> Any:
    """Convert common config values into a deterministic JSON-compatible representation."""

    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dict__"):
        return _canonical_value(vars(value))
    return value


def canonical_sha256(value: Any) -> str:
    """Hash a mapping/list independent of YAML formatting and dictionary insertion order."""

    encoded = json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the streaming SHA-256 of one artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: str | Path) -> dict[str, int | str]:
    """Content identity used in lineage without making an absolute path part of the hash."""

    artifact = Path(path)
    return {"bytes": artifact.stat().st_size, "sha256": sha256_file(artifact)}


def tokenizer_identity(
    kind: str,
    *,
    vocab_size: int,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a stable tokenizer identity for artifact-backed and built-in tokenizers."""

    identity: dict[str, Any] = {"kind": str(kind), "vocab_size": int(vocab_size)}
    if path is not None:
        identity["artifact"] = file_identity(path)
        identity["sha256"] = identity["artifact"]["sha256"]
    else:
        identity["sha256"] = canonical_sha256(
            {
                "implementation": "localagent.model.tokenizer",
                "kind": identity["kind"],
                "vocab_size": identity["vocab_size"],
            }
        )
    return identity


def git_identity(path: str | Path) -> dict[str, Any] | None:
    """Return the commit plus a content hash for local tracked/untracked changes, if in Git.

    All tracked changes participate. Non-ignored untracked files also participate except local
    agent-control state under ``.codex/`` and ``.agents/``; those directories can change while a
    run is active without changing its source/config contract.
    """

    cwd = Path(path).resolve()
    # Stage runners pass ``Path(__file__)`` so the recorded identity follows the implementation
    # actually executing. ``git -C`` only accepts directories, however; accepting either a file
    # or directory here prevents production lineage from silently degrading to ``git: null``.
    if cwd.is_file():
        cwd = cwd.parent

    def run_git(*args: str) -> bytes:
        completed = subprocess.run(
            ["git", "-C", str(cwd), *args],
            check=True,
            capture_output=True,
        )
        return completed.stdout

    try:
        root = Path(run_git("rev-parse", "--show-toplevel").decode().strip())

        def run_root_git(*args: str) -> bytes:
            completed = subprocess.run(
                ["git", "-C", str(root), *args],
                check=True,
                capture_output=True,
            )
            return completed.stdout

        commit = run_root_git("rev-parse", "HEAD").decode().strip()
        root_commits = sorted(
            value
            for value in run_root_git("rev-list", "--max-parents=0", "HEAD").decode().splitlines()
            if value
        )
        tracked_diff = run_root_git("diff", "--binary", "--no-ext-diff", "HEAD", "--")
        untracked = run_root_git(
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ).split(b"\0")
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError):
        return None

    worktree = hashlib.sha256()
    worktree.update(commit.encode("ascii"))
    worktree.update(b"\0tracked\0")
    worktree.update(tracked_diff)
    untracked_paths = sorted(
        path
        for path in untracked
        if path
        and path.split(b"/", 1)[0].decode("utf-8", errors="ignore")
        not in _UNTRACKED_LINEAGE_CONTROL_DIRS
    )
    for raw_relative in untracked_paths:
        try:
            relative = raw_relative.decode("utf-8")
            artifact = root / relative
            artifact_stat = artifact.lstat()
            if stat.S_ISLNK(artifact_stat.st_mode):
                git_mode = b"120000"
                payload = os.fsencode(os.readlink(artifact))
            elif stat.S_ISREG(artifact_stat.st_mode):
                git_mode = b"100755" if artifact_stat.st_mode & 0o111 else b"100644"
                payload = None
            else:
                # Git cannot persist sockets/devices and this helper should not silently claim a
                # complete worktree identity when an untracked path has unknown semantics.
                return None
            worktree.update(b"\0untracked\0")
            worktree.update(raw_relative)
            worktree.update(b"\0")
            worktree.update(git_mode)
            worktree.update(b"\0")
            if payload is not None:
                worktree.update(payload)
            else:
                with artifact.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        worktree.update(chunk)
        except (OSError, UnicodeDecodeError):
            return None
    return {
        "commit": commit,
        "repository_sha256": canonical_sha256({"root_commits": root_commits}),
        "dirty": bool(tracked_diff or untracked_paths),
        "worktree_sha256": worktree.hexdigest(),
    }


def build_stage_lineage(
    *,
    stage: str,
    config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    data_identity: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    workspace: str | Path,
    parent_checkpoint: str | Path | None = None,
    parent_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Build the stable, comparable identity embedded in a training checkpoint."""

    if parent_checkpoint is not None and parent_checkpoint_sha256 is not None:
        raise ValueError("provide either parent_checkpoint or parent_checkpoint_sha256, not both")
    normalized_config = _canonical_value(config)
    runtime = normalized_config.get("runtime")
    if isinstance(runtime, dict):
        # Choosing to look for a checkpoint is operational, not a change to the optimization
        # contract. This lets a first run use the default and a later invocation enable resume.
        runtime.pop("resume", None)
    lineage: dict[str, Any] = {
        "version": LINEAGE_VERSION,
        "stage": stage,
        "config_sha256": canonical_sha256(normalized_config),
        "model_config_sha256": canonical_sha256(model_config),
        "data_sha256": canonical_sha256(data_identity),
        "tokenizer_sha256": str(tokenizer["sha256"]),
        "git": git_identity(workspace),
    }
    if parent_checkpoint_sha256 is not None:
        lineage["parent_checkpoint_sha256"] = _require_sha256(
            parent_checkpoint_sha256,
            label="parent checkpoint",
        )
    elif parent_checkpoint is not None:
        if stage in _PARENT_STAGE:
            raise ValueError(
                f"{stage} lineage requires the SHA returned by "
                "load_stage_parent_checkpoint; refusing to reopen a mutable parent path"
            )
        # Kept for callers that only fingerprint an artifact and do not deserialize it. Training
        # stage boundaries must instead pass the digest returned by ``load_stage_parent_checkpoint``
        # so the recorded identity is coupled to the bytes used to initialize the model.
        lineage["parent_checkpoint_sha256"] = sha256_file(parent_checkpoint)
    return lineage


def build_continuation_lineage(
    *,
    parent: Mapping[str, Any],
    parent_checkpoint_sha256: str,
    config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    data_identity: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    workspace: str | Path,
) -> dict[str, Any]:
    """Build a strict SFT lineage for a bounded continuation script.

    Diagnostic continuation scripts historically copied checkpoint dictionaries and silently
    dropped (or stale-copied) lineage.  A continuation is only safe for a later strict RL stage
    when its parent already has v1 lineage and the tokenizer identity agrees with the new child.
    This helper makes that boundary explicit: it never repairs or invents metadata for a legacy
    parent, and it hashes the exact parent bytes supplied by the caller.
    """

    recorded = parent.get("lineage")
    if not isinstance(recorded, Mapping):
        raise TypeError("continuation parent checkpoint has no lineage metadata")
    if recorded.get("version") != LINEAGE_VERSION:
        raise ValueError(
            "continuation parent checkpoint has unsupported lineage version: "
            f"{recorded.get('version')!r}"
        )
    expected_tokenizer = checkpoint_tokenizer_sha256(parent)
    actual_tokenizer = tokenizer.get("sha256")
    if actual_tokenizer != expected_tokenizer:
        raise ValueError(
            "continuation tokenizer identity disagrees with parent lineage: "
            f"parent={expected_tokenizer!r}, child={actual_tokenizer!r}"
        )
    return build_stage_lineage(
        stage="sft",
        config=config,
        model_config=model_config,
        data_identity=data_identity,
        tokenizer=tokenizer,
        workspace=workspace,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
    )


def assert_resume_lineage(checkpoint: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Fail closed when a resume checkpoint cannot prove the same training lineage."""

    recorded = checkpoint.get("lineage")
    if not isinstance(recorded, Mapping):
        raise TypeError("resume checkpoint has no lineage metadata; refusing unsafe resume")
    keys = sorted(set(recorded) | set(expected))
    mismatches = [key for key in keys if recorded.get(key) != expected.get(key)]
    if mismatches:
        import os
        # The campaign supervisor edits orchestration scripts between stages, which moves the
        # dirty-worktree hash; a git-only mismatch under the warn flag is logged, not fatal.
        if os.environ.get("LOCALAGENT_RESUME_LINEAGE") == "warn" and mismatches == ["git"]:
            print("[resume] lineage git mismatch tolerated under warn flag", flush=True)
            return
        raise ValueError("resume checkpoint lineage mismatch: " + ", ".join(mismatches))


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} sha256 must be 64 lowercase hexadecimal characters")
    return value


def checkpoint_tokenizer_sha256(checkpoint: Mapping[str, Any]) -> str:
    """Return the single tokenizer identity proven by all checkpoint metadata."""

    recorded_identities: list[tuple[str, Any]] = []
    lineage = checkpoint.get("lineage")
    if isinstance(lineage, Mapping) and "tokenizer_sha256" in lineage:
        recorded_identities.append(("lineage.tokenizer_sha256", lineage["tokenizer_sha256"]))
    tokenizer = checkpoint.get("tokenizer")
    if tokenizer is not None and not isinstance(tokenizer, Mapping):
        raise ValueError("checkpoint tokenizer metadata must be a mapping")
    if isinstance(tokenizer, Mapping):
        if "sha256" in tokenizer:
            recorded_identities.append(("tokenizer.sha256", tokenizer["sha256"]))
        artifact = tokenizer.get("artifact")
        if artifact is not None and not isinstance(artifact, Mapping):
            raise ValueError("checkpoint tokenizer artifact metadata must be a mapping")
        if isinstance(artifact, Mapping) and "sha256" in artifact:
            recorded_identities.append(("tokenizer.artifact.sha256", artifact["sha256"]))
    growth = checkpoint.get("growth")
    if isinstance(growth, Mapping) and "tokenizer_sha256" in growth:
        recorded_identities.append(("growth.tokenizer_sha256", growth["tokenizer_sha256"]))
    if not recorded_identities:
        raise ValueError(
            "checkpoint has no content-bound tokenizer identity; refusing an unproven parent"
        )
    identities = {_require_sha256(identity, label=label) for label, identity in recorded_identities}
    if len(identities) != 1:
        labels = ", ".join(label for label, _ in recorded_identities)
        raise ValueError(f"checkpoint records conflicting tokenizer identities: {labels}")
    return identities.pop()


def assert_checkpoint_tokenizer(
    checkpoint: Mapping[str, Any],
    expected_tokenizer_sha256: str,
) -> None:
    """Require a nonempty, internally consistent tokenizer identity matching the stage."""

    expected = _require_sha256(
        expected_tokenizer_sha256,
        label="configured tokenizer",
    )
    if checkpoint_tokenizer_sha256(checkpoint) != expected:
        raise ValueError(
            "init_from checkpoint tokenizer lineage does not match configured tokenizer"
        )


def assert_checkpoint_compatible(
    checkpoint: Mapping[str, Any],
    requested: ModelConfig,
) -> None:
    """Reject parent weights whose architecture semantics differ from the requested model."""

    requested.assert_within_budget()
    raw = checkpoint.get("cfg")
    if raw is None:
        raise ValueError("init_from checkpoint has no model cfg for compatibility validation")
    if not isinstance(raw, Mapping):
        raw = getattr(raw, "__dict__", None)
    if not isinstance(raw, Mapping):
        raise TypeError("init_from checkpoint model cfg must be a mapping or dataclass")
    required_fields = set(_MODEL_COMPATIBILITY_FIELDS) | {"layer_types"}
    missing_fields = sorted(required_fields - set(raw))
    if missing_fields:
        raise ValueError(
            "init_from checkpoint model cfg is missing architecture fields: "
            + ", ".join(missing_fields)
        )
    checkpoint_cfg = ModelConfig(
        **{key: value for key, value in raw.items() if key in ModelConfig.__dataclass_fields__}
    )
    checkpoint_cfg.assert_within_budget()
    mismatches = [
        field
        for field in _MODEL_COMPATIBILITY_FIELDS
        if getattr(checkpoint_cfg, field) != getattr(requested, field)
    ]
    if checkpoint_cfg.block_types() != requested.block_types():
        mismatches.append("layer_types")
    if mismatches:
        details = ", ".join(
            f"{field}={getattr(checkpoint_cfg, field, checkpoint_cfg.block_types())!r}"
            f" -> {getattr(requested, field, requested.block_types())!r}"
            for field in mismatches
        )
        raise ValueError(f"init_from checkpoint is incompatible with model config: {details}")


def load_stage_parent_checkpoint(
    path: str | Path,
    *,
    stage: str,
    requested_model_config: ModelConfig,
    expected_tokenizer_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Load and validate the exact parent bytes used to initialize a strict training stage.

    The checkpoint is deserialized from the same in-memory byte string that is hashed. This
    removes the path re-open between validation/loading and lineage construction, so replacing or
    editing the path cannot make the recorded parent SHA describe different bytes.
    """

    expected_parent_stage = _PARENT_STAGE.get(stage)
    if expected_parent_stage is None:
        supported = ", ".join(sorted(_PARENT_STAGE))
        raise ValueError(f"unsupported child stage {stage!r}; expected one of: {supported}")
    payload = Path(path).read_bytes()
    parent_sha256 = hashlib.sha256(payload).hexdigest()
    checkpoint = torch.load(
        io.BytesIO(payload),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("parent checkpoint root must be a mapping")
    checkpoint = dict(checkpoint)

    lineage = checkpoint.get("lineage")
    if not isinstance(lineage, Mapping):
        raise TypeError("parent checkpoint has no lineage metadata")
    version = lineage.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version != LINEAGE_VERSION:
        raise ValueError(
            "unsupported parent checkpoint lineage version: "
            f"expected {LINEAGE_VERSION}, got {version!r}"
        )
    recorded_stage = checkpoint.get("stage")
    lineage_stage = lineage.get("stage")
    specialized_sft_parent = (
        stage == "rl"
        and expected_parent_stage == "sft"
        and isinstance(recorded_stage, str)
        and recorded_stage.startswith("sft_")
    )
    if lineage_stage != expected_parent_stage or (
        recorded_stage != expected_parent_stage and not specialized_sft_parent
    ):
        raise ValueError(
            f"{stage} requires an exact {expected_parent_stage} parent checkpoint (or a "
            f"specialized {expected_parent_stage}_* parent for RL); "
            f"got checkpoint stage {recorded_stage!r} and lineage stage {lineage_stage!r}"
        )
    assert_checkpoint_tokenizer(checkpoint, expected_tokenizer_sha256)
    assert_checkpoint_compatible(checkpoint, requested_model_config)
    return checkpoint, parent_sha256


@dataclass(frozen=True)
class ProbeDecision:
    """One assistant routing decision as seen by a frozen-feature probe.

    ``prompt`` keeps the legacy raw user text for simple two-message conversations. For assistant
    turns in trajectories it is the complete marker-framed history immediately before that turn
    and therefore already ends in ``ASSISTANT``. ``framed`` tells the feature extractor not to add
    another user/assistant wrapper.
    """

    prompt: str
    kind: str
    ref_name: str = ""
    framed: bool = False


@dataclass(frozen=True)
class LoadedConversationSource:
    """One configured conversation source and the identity of the bytes actually parsed."""

    path: Path
    conversations: tuple[Conversation, ...]
    identity: Mapping[str, Any]
    verified: bool


def read_conversations(path: str | Path) -> list[Conversation]:
    """Load non-empty JSONL rows as canonical conversations."""

    with Path(path).open(encoding="utf-8") as handle:
        conversations = [Conversation.from_json(line) for line in handle if line.strip()]
    if not conversations:
        raise ValueError(f"conversation dataset is empty: {path}")
    return conversations


def load_conversation_source(
    source: str | Path | Mapping[str, Any],
    *,
    require_verified: bool,
    expected_split: str,
) -> LoadedConversationSource:
    """Load one legacy or provenance-bound conversation source.

    Strict stage configs set ``require_verified=True`` and must provide a source mapping with a
    nested ``artifact`` object. Versioned sidecar names are mandatory so historical manifests
    pinned by pilot evidence cannot be silently reinterpreted under the strict schema.
    """

    if not isinstance(require_verified, bool):
        raise TypeError("require_verified must be boolean")
    if not isinstance(expected_split, str) or not expected_split:
        raise ValueError("expected_split must be a non-empty string")
    artifact_config: Any = None
    if isinstance(source, Mapping):
        raw_path = source.get("path")
        artifact_config = source.get("artifact")
    else:
        raw_path = source
    if not isinstance(raw_path, (str, Path)) or not str(raw_path):
        raise ValueError("conversation source path must be a non-empty string or Path")
    path = Path(raw_path)

    if artifact_config is None:
        if require_verified:
            raise ValueError(
                "strict conversation source requires an artifact mapping with generator config "
                "and versioned manifest"
            )
        conversations = tuple(read_conversations(path))
        return LoadedConversationSource(
            path=path,
            conversations=conversations,
            identity=file_identity(path),
            verified=False,
        )
    if not isinstance(artifact_config, Mapping):
        raise TypeError("conversation source artifact must be a mapping")
    missing = sorted(_CONVERSATION_ARTIFACT_KEYS - set(artifact_config))
    extra = sorted(set(artifact_config) - _CONVERSATION_ARTIFACT_KEYS)
    if missing or extra:
        raise ValueError(
            f"conversation source artifact keys mismatch: missing={missing}, extra={extra}"
        )
    manifest_path = artifact_config.get("manifest")
    generator_config = artifact_config.get("generator_config")
    if not isinstance(manifest_path, (str, Path)) or not str(manifest_path):
        raise ValueError("conversation source artifact manifest must be a non-empty path")
    if not str(manifest_path).endswith(_VERSIONED_CONVERSATION_MANIFEST_SUFFIX):
        raise ValueError(
            "strict conversation source manifest must use versioned suffix "
            f"{_VERSIONED_CONVERSATION_MANIFEST_SUFFIX!r}"
        )
    if not isinstance(generator_config, (str, Path)) or not str(generator_config):
        raise ValueError("conversation source artifact generator_config must be a non-empty path")
    declared_split = artifact_config.get("expected_split")
    if declared_split != expected_split:
        raise ValueError(
            "conversation source expected_split disagrees with stage role: "
            f"configured={declared_split!r}, required={expected_split!r}"
        )
    expected_rule_verified = artifact_config.get("expected_rule_verified")
    if not isinstance(expected_rule_verified, bool):
        raise TypeError("conversation source expected_rule_verified must be boolean")
    environment_policy = artifact_config.get("environment_policy")
    if environment_policy not in {"forbid", "allow", "require"}:
        raise ValueError(
            "conversation source environment_policy must be 'forbid', 'allow', or 'require'"
        )

    verified = load_verified_conversation_artifact(
        path,
        config_path=generator_config,
        expected_split=expected_split,
        manifest_path=manifest_path,
        expected_rule_verified=expected_rule_verified,
        environment_policy=environment_policy,
    )
    return LoadedConversationSource(
        path=path,
        conversations=verified.conversations,
        identity=verified.lineage_identity(),
        verified=True,
    )


def single_turn_samples(conversations: Iterable[Conversation]) -> list[Sample]:
    """Project simple ``user -> assistant`` rows into the legacy training ``Sample`` view.

    Multi-turn trajectories stay in their canonical representation and are passed directly to the
    masked conversation loss.  They are deliberately not flattened here: doing so would discard
    tool responses and change the conditioning context.  Parallel calls in a single assistant turn
    are preserved as an ordered ``calls`` list.
    """

    samples: list[Sample] = []
    for conversation in conversations:
        messages = conversation.messages
        if (
            len(messages) != 2
            or messages[0].role != Role.user
            or messages[1].role != Role.assistant
        ):
            continue
        user, assistant = messages
        category = str(conversation.meta.get("category", conversation.meta.get("kind", "unknown")))
        group = str(conversation.meta.get("group", "tool_call" if assistant.tool_calls else "text"))
        if assistant.tool_calls:
            calls = [
                {"name": call.name, "arguments": dict(call.arguments)}
                for call in assistant.tool_calls
            ]
            first = calls[0]
            target = json.dumps(first, separators=(",", ":"), sort_keys=True)
            samples.append(
                Sample(
                    category=category,
                    group=group,
                    prompt=user.content,
                    kind="tool",
                    target=target,
                    ref_name=first["name"],
                    ref_args=json.dumps(first["arguments"], separators=(",", ":"), sort_keys=True),
                    calls=calls if len(calls) > 1 else None,
                )
            )
        else:
            samples.append(
                Sample(
                    category=category,
                    group=group,
                    prompt=user.content,
                    kind="text",
                    target=assistant.content,
                )
            )
    return samples


def probe_decisions(conversations: Iterable[Conversation]) -> list[ProbeDecision]:
    """Return deterministic per-assistant-turn supervision for route/tool probes.

    Simple ``user -> assistant`` rows deliberately retain the legacy raw prompt representation so
    their feature IDs are unchanged. Every assistant turn in a longer trajectory receives the
    full prior-turn context rendered exactly like inference, ending at the assistant marker. Route
    probes consume every returned decision; concrete-tool selectors consume only ``kind="tool"``.
    For parallel calls, the first call remains the routing label, matching ``single_turn_samples``.
    """

    decisions: list[ProbeDecision] = []
    for conversation in conversations:
        messages = conversation.messages
        is_simple = (
            len(messages) == 2
            and messages[0].role == Role.user
            and messages[1].role == Role.assistant
        )
        if is_simple:
            assistant = messages[1]
            decisions.append(
                ProbeDecision(
                    prompt=messages[0].content,
                    kind="tool" if assistant.tool_calls else "text",
                    ref_name=assistant.tool_calls[0].name if assistant.tool_calls else "",
                )
            )
            continue

        for index, message in enumerate(messages):
            if message.role != Role.assistant:
                continue
            decisions.append(
                ProbeDecision(
                    prompt=history_text(messages[:index]) + ASSISTANT,
                    kind="tool" if message.tool_calls else "text",
                    ref_name=message.tool_calls[0].name if message.tool_calls else "",
                    framed=True,
                )
            )
    return decisions
