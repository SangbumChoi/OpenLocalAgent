"""Supervised fine-tune on agent samples (Phase 4, implemented).

Loss is masked to the assistant body + EOS (render.render_sft); the model only learns to produce
tool calls / text given the prompt, not to echo the user. Function masking is an opt-in,
schema-preserving tool-name augmentation that retains original rows and adds deterministic opaque
name variants; held-out conversations are never masked.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from localagent.data.conversation_artifact import (
    assert_no_conversation_overlap,
    conversation_semantic_sha256,
)
from localagent.data.decision_quota_order import (
    QUOTA_SAMPLING_MODE,
    order_assistant_decisions,
    quota_sampling_contract,
)
from localagent.data.prompt_contract import (
    LEGACY_CONVERSATION_PROMPT_CONTRACT,
    assert_prompt_contract_tokenizer,
    resolve_conversation_prompt_contract,
)
from localagent.data.render import (
    IGNORE,
    CatalogTokenCache,
    render_conversation_rows,
    shifted_token_counts,
    token_row_length,
)
from localagent.train.device import autocast_ctx
from localagent.train.function_masking import augment_conversations
from localagent.train.loop import (
    cosine_lr,
    pad_batch,
    router_loss_terms,
    set_lr,
    validate_pad_to_input_tokens,
    wsd_lr,
)
from localagent.train.replay_sampling import (
    MIXED_REPLAY_SAMPLING_MODE,
    PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE,
    mixed_replay_sampling_window,
    parent_anchored_format_pulse_sampling_window,
)
from localagent.train.stage_sampling import (
    SFT_LOSS_NORMALIZATION_MICROBATCH,
    SFT_LOSS_NORMALIZATION_UPDATE_TOKENS,
    SFTSamplingSchedule,
    decision_keys_to_row_order,
    framed_prompt_ids,
    prepare_sft_data,
    validate_sft_loss_normalization,
    validate_multi_turn_batch_size,
)
from localagent.agent.pointer_head import PTR_ARGS
from localagent.train.stage_sampling import (
    add_row_accounting as _add_row_accounting,
)
from localagent.train.stage_sampling import (
    empty_token_accounting as _empty_token_accounting,
)
from localagent.train.stage_sampling import (
    encode_with_value_span as _encode_with_value_span,
)

_SFT_RESUME_FORMAT = "localagent.sft_resume"
_SFT_RESUME_VERSION = 1
SFT_CONTINUATION_MODE = "fresh_optimizer_sft_child_v1"
_SFT_CONTINUATION_PARENT_FIELDS = (
    "checkpoint_sha256",
    "resume_integrity_sha256",
    "training_contract_sha256",
    "lm_sampling_sha256",
    "completed_steps",
    "completed_lm_cursor",
)
_SFT_CONTINUATION_PARENT_HASH_FIELDS = _SFT_CONTINUATION_PARENT_FIELDS[:4]
_SFT_RESUME_SEALED_FIELDS = (
    "resume_format",
    "resume_version",
    "cfg",
    "state_dict",
    "tool_head",
    "ptr_head",
    "optimizer",
    "grad_scaler",
    "step",
    "loss_history",
    "dataset_token_accounting",
    "token_accounting",
    "token_accounting_scope",
    "sampling_state",
    "torch_rng_state",
    "cuda_rng_state_all",
    "mps_rng_state",
    "xpu_rng_state_all",
    "stage",
    "training_seed",
    "training_contract",
    "lineage",
    "conversation_prompt_contract",
    "tokenizer",
    "data",
    "execution",
    "heldout_baseline",
)


def _mps_synchronize_and_empty_cache(device: str | torch.device) -> None:
    """Release only unoccupied MPS allocator cache after queued work has completed."""

    if torch.device(device).type != "mps":
        return
    torch.mps.synchronize()
    torch.mps.empty_cache()


def _clear_mps_gradients_and_cache(
    device: str | torch.device,
    *modules: torch.nn.Module | None,
) -> None:
    """Drop stale training gradients before an MPS evaluation stage."""

    if torch.device(device).type != "mps":
        return
    for module in modules:
        if module is not None:
            module.zero_grad(set_to_none=True)
    _mps_synchronize_and_empty_cache(device)


def validate_sft_continuation_parent(parent: Any) -> dict[str, Any]:
    """Validate and normalize the optional exact continuation-parent seal."""

    if not isinstance(parent, Mapping):
        raise TypeError("continuation.parent must be a mapping")
    if set(parent) != set(_SFT_CONTINUATION_PARENT_FIELDS):
        raise ValueError(
            "continuation.parent must contain exactly "
            + ", ".join(_SFT_CONTINUATION_PARENT_FIELDS)
        )
    for field in _SFT_CONTINUATION_PARENT_HASH_FIELDS:
        if not _valid_sha256(parent.get(field)):
            raise ValueError(
                f"continuation.parent.{field} must be a lowercase SHA-256 digest"
            )
    completed_steps = parent.get("completed_steps")
    if (
        isinstance(completed_steps, bool)
        or not isinstance(completed_steps, int)
        or completed_steps < 1
    ):
        raise ValueError("continuation.parent.completed_steps must be a positive integer")
    completed_lm_cursor = parent.get("completed_lm_cursor")
    if (
        isinstance(completed_lm_cursor, bool)
        or not isinstance(completed_lm_cursor, int)
        or completed_lm_cursor < 0
    ):
        raise ValueError(
            "continuation.parent.completed_lm_cursor must be a non-negative integer"
        )
    return {
        field: parent[field]
        for field in _SFT_CONTINUATION_PARENT_FIELDS
    }


def resolve_sft_continuation(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Validate the optional fresh-optimizer SFT-child initialization contract.

    Exact resume continues one interrupted fixed horizon, whereas this contract starts a new
    independently planned SFT horizon from a *completed* SFT checkpoint.  Keeping the modes
    separate prevents an optimizer/schedule extension from being mislabeled as exact resume.
    """

    raw = config.get("continuation")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise TypeError("continuation must be a mapping")
    if set(raw) not in ({"mode"}, {"mode", "parent"}):
        raise ValueError("continuation must contain exactly mode, with optional parent")
    mode = raw.get("mode")
    if mode != SFT_CONTINUATION_MODE:
        raise ValueError(
            f"continuation.mode must be {SFT_CONTINUATION_MODE!r}, got {mode!r}"
        )
    resolved: dict[str, Any] = {"mode": SFT_CONTINUATION_MODE}
    if "parent" in raw:
        resolved["parent"] = validate_sft_continuation_parent(raw["parent"])
    return resolved


def _validate_sft_optimizer_contract(
    optimizer_name: Any,
    weight_decay: Any,
    grad_clip: Any,
) -> tuple[str, float, float]:
    """Validate the pure-PyTorch SFT optimizer knobs before constructing AdamW."""

    if not isinstance(optimizer_name, str):
        raise TypeError("optimizer_name must be a string")
    if optimizer_name != "adamw":
        raise ValueError("optimizer_name must be exactly 'adamw'")

    if isinstance(weight_decay, bool) or not isinstance(weight_decay, (int, float)):
        raise TypeError("weight_decay must be a finite non-negative number")
    weight_decay = float(weight_decay)
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be a finite non-negative number")

    if isinstance(grad_clip, bool) or not isinstance(grad_clip, (int, float)):
        raise TypeError("grad_clip must be a finite positive number")
    grad_clip = float(grad_clip)
    if not math.isfinite(grad_clip) or grad_clip <= 0:
        raise ValueError("grad_clip must be a finite positive number")
    return optimizer_name, weight_decay, grad_clip


def _validate_sft_freeze_parameters(
    model,
    freeze_parameters: list[str] | None,
) -> tuple[str, ...] | None:
    """Validate and apply an exact-name model-parameter freeze contract."""

    if freeze_parameters is None:
        return None
    if not isinstance(freeze_parameters, list):
        raise TypeError("freeze_parameters must be a list of exact model parameter names")

    for index, name in enumerate(freeze_parameters):
        if not isinstance(name, str):
            raise TypeError(f"freeze_parameters[{index}] must be a string")

    seen: set[str] = set()
    duplicates: list[str] = []
    for name in freeze_parameters:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise ValueError(
            "freeze_parameters contains duplicate names: " + ", ".join(duplicates)
        )

    named_parameters = dict(model.named_parameters())
    unknown = [name for name in freeze_parameters if name not in named_parameters]
    if unknown:
        raise ValueError(
            "freeze_parameters contains unknown model parameter names: " + ", ".join(unknown)
        )
    if len(freeze_parameters) == len(named_parameters):
        raise ValueError("freeze_parameters cannot freeze every model parameter")

    for name in freeze_parameters:
        named_parameters[name].requires_grad_(False)
    return tuple(freeze_parameters)


def quota_sampling_window(
    ordering,
    *,
    selected_decisions: int,
    start_decision: int = 0,
) -> tuple[tuple[tuple[int, int], ...], dict[str, Any]]:
    """Resolve a deterministic no-replacement decision window and its sealed contract.

    ``start_decision`` is a zero-based offset into the canonical quota order.  The returned key
    sequence remains a complete permutation because :class:`SFTSamplingSchedule` validates that
    invariant, but its prefix begins at the requested offset.  The fixed horizon must end before
    the original epoch ends, so a continuation can consume unseen rows without wrapping.

    The zero-offset path delegates directly to :func:`quota_sampling_contract`; existing configs
    therefore retain their exact sampling metadata and validation behavior.
    """

    if isinstance(start_decision, bool) or not isinstance(start_decision, int):
        raise TypeError("data.sampling.start_decision must be an integer")
    if start_decision < 0:
        raise ValueError("data.sampling.start_decision must be non-negative")
    if isinstance(selected_decisions, bool) or not isinstance(selected_decisions, int):
        raise TypeError("selected_decisions must be an integer")
    if selected_decisions < 0:
        raise ValueError("selected_decisions must be non-negative")

    audit = ordering.audit
    end_decision = start_decision + selected_decisions
    if end_decision > audit.ordered_decision_count:
        raise ValueError(
            "quota SFT decision window exceeds the available no-replacement epoch: "
            f"start={start_decision}, selected={selected_decisions}, "
            f"available={audit.ordered_decision_count}"
        )
    if start_decision == 0:
        return tuple(ordering.keys), quota_sampling_contract(
            ordering,
            selected_decisions=selected_decisions,
        )

    start_counts = audit.prefix_counts(start_decision)
    end_counts = audit.prefix_counts(end_decision)
    window_counts = {
        stratum_id: end_counts[stratum_id] - start_counts[stratum_id]
        for stratum_id in start_counts
    }
    compact_audit = audit.as_dict()
    compact_audit.pop("ordered_stratum_ids")
    contract = {
        "mode": QUOTA_SAMPLING_MODE,
        "no_replacement": True,
        "require_all_observed_strata": False,
        "ordering": compact_audit,
        "start_decision": start_decision,
        "selected_window": {
            "decisions": selected_decisions,
            "end_decision_exclusive": end_decision,
            "covered_strata": sum(count > 0 for count in window_counts.values()),
            "all_observed_strata_covered": all(
                count > 0 for count in window_counts.values()
            ),
            "stratum_counts": window_counts,
        },
    }
    rotated_keys = tuple(ordering.keys[start_decision:]) + tuple(
        ordering.keys[:start_decision]
    )
    return rotated_keys, contract


def _digest_chunk(digest: Any, tag: bytes, payload: bytes = b"") -> None:
    digest.update(len(tag).to_bytes(4, "big"))
    digest.update(tag)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _update_resume_digest(digest: Any, value: Any) -> None:
    """Hash nested checkpoint state without depending on ``torch.save`` container bytes."""

    if value is None:
        _digest_chunk(digest, b"none")
    elif isinstance(value, bool):
        _digest_chunk(digest, b"bool", b"1" if value else b"0")
    elif isinstance(value, int):
        _digest_chunk(digest, b"int", str(value).encode("ascii"))
    elif isinstance(value, float):
        _digest_chunk(digest, b"float", value.hex().encode("ascii"))
    elif isinstance(value, str):
        _digest_chunk(digest, b"str", value.encode("utf-8"))
    elif isinstance(value, bytes):
        _digest_chunk(digest, b"bytes", value)
    elif isinstance(value, Path):
        _digest_chunk(digest, b"path", str(value).encode("utf-8"))
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        _digest_chunk(
            digest,
            b"tensor",
            f"{tensor.dtype}:{tuple(tensor.shape)}".encode("ascii"),
        )
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        _digest_chunk(digest, b"tensor-bytes", raw)
    elif isinstance(value, Mapping):
        _digest_chunk(digest, b"mapping", str(len(value)).encode("ascii"))
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _update_resume_digest(digest, key)
            _update_resume_digest(digest, value[key])
    elif isinstance(value, Sequence):
        _digest_chunk(digest, b"sequence", str(len(value)).encode("ascii"))
        for item in value:
            _update_resume_digest(digest, item)
    elif hasattr(value, "__dict__"):
        _digest_chunk(
            digest,
            f"object:{type(value).__module__}.{type(value).__qualname__}".encode(),
        )
        _update_resume_digest(digest, vars(value))
    else:
        raise TypeError(f"unsupported resume-digest value {type(value).__name__}")


def _resume_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _update_resume_digest(digest, value)
    return digest.hexdigest()


def _sealed_resume_sha256(payload: Mapping[str, Any]) -> str:
    # v1 checkpoints predate component-level router/LM reporting. Preserve validation of those
    # envelopes while sealing the optional histories whenever a newly written checkpoint carries
    # them. Their presence, rather than a format-version guess, selects the extended envelope.
    fields = list(_SFT_RESUME_SEALED_FIELDS)
    if "loss_components" in payload:
        fields.append("loss_components")
    return _resume_sha256({field: payload.get(field) for field in fields})


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sft_checkpoint_archive_path(
    checkpoint_path: str | Path,
    *,
    completed_steps: int,
) -> Path:
    """Return the immutable periodic-archive path beside the rolling checkpoint."""

    if isinstance(completed_steps, bool) or not isinstance(completed_steps, int):
        raise TypeError("completed_steps must be an integer")
    if completed_steps < 1:
        raise ValueError("completed_steps must be positive")
    path = Path(checkpoint_path)
    return path.with_name(
        f"{path.stem}.step-{completed_steps:08d}{path.suffix}"
    )


def _assert_matching_sft_checkpoint_archive(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    """Accept an idempotent retry, but fail closed on any existing different archive."""

    try:
        existing = _load_validated_sft_resume_checkpoint(path)
    except Exception as exc:
        raise FileExistsError(
            f"refusing to overwrite invalid or different SFT checkpoint archive: {path}"
        ) from exc
    if (
        set(existing) != set(payload)
        or existing.get("resume_integrity_sha256")
        != payload.get("resume_integrity_sha256")
    ):
        raise FileExistsError(
            f"refusing to overwrite different SFT checkpoint archive: {path}"
        )


def _write_immutable_sft_checkpoint_archive(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    """Atomically create ``path`` once, or verify an already-identical sealed archive."""

    if path.exists():
        _assert_matching_sft_checkpoint_archive(path, payload)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        torch.save(payload, temporary_path)
        try:
            # A hard link publishes the completed file atomically and fails rather than
            # replacing a concurrently created archive.
            os.link(temporary_path, path)
        except FileExistsError:
            _assert_matching_sft_checkpoint_archive(path, payload)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_validated_sft_resume_checkpoint(
    resume_from: str | Path | Mapping[str, Any],
) -> Mapping[str, Any]:
    """Load and validate the integrity-sealed SFT resume envelope.

    A mapping is accepted so the stage runner can validate a checkpoint before an expensive
    held-out baseline evaluation and then pass the same read-only payload into ``sft()``.
    ``sft()`` still calls this helper itself before its complete resume-contract validation.
    """

    checkpoint = (
        resume_from
        if isinstance(resume_from, Mapping)
        else torch.load(resume_from, map_location="cpu", weights_only=True)
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("SFT resume checkpoint root must be a mapping")
    required = {*_SFT_RESUME_SEALED_FIELDS, "resume_integrity_sha256"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError("SFT resume checkpoint is incomplete; missing: " + ", ".join(missing))
    recorded_integrity = checkpoint.get("resume_integrity_sha256")
    if not _valid_sha256(recorded_integrity):
        raise ValueError("SFT resume checkpoint integrity digest is invalid")
    if recorded_integrity != _sealed_resume_sha256(checkpoint):
        raise ValueError("SFT resume checkpoint integrity mismatch")
    if (
        checkpoint.get("resume_format") != _SFT_RESUME_FORMAT
        or checkpoint.get("resume_version") != _SFT_RESUME_VERSION
    ):
        raise ValueError("SFT resume checkpoint format/version is unsupported")
    if checkpoint.get("stage") != "sft":
        raise ValueError("SFT resume checkpoint stage must be 'sft'")
    return checkpoint


def _validated_completed_sft_parent(
    checkpoint: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Require a sealed SFT checkpoint at the end of its original fixed horizon."""

    checkpoint = _load_validated_sft_resume_checkpoint(checkpoint)
    training_contract = checkpoint.get("training_contract")
    if not isinstance(training_contract, Mapping):
        raise ValueError("SFT continuation parent training contract is invalid")
    steps = training_contract.get("steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("SFT continuation parent fixed horizon is invalid")
    step = checkpoint.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step != steps - 1:
        raise ValueError(
            "SFT continuation requires a completed parent fixed horizon: "
            f"completed_step={step!r}, planned_final_step={steps - 1}"
        )
    _validated_sft_history(checkpoint.get("loss_history"), completed_steps=steps)
    sampling_state = checkpoint.get("sampling_state")
    if not isinstance(sampling_state, Mapping):
        raise ValueError("SFT continuation parent sampling state is invalid")
    completed_steps = sampling_state.get("completed_steps")
    if (
        isinstance(completed_steps, bool)
        or not isinstance(completed_steps, int)
        or completed_steps != steps
    ):
        raise ValueError("SFT continuation parent sampling state is incomplete")
    accum_steps = training_contract.get("accum_steps")
    completed_microbatches = sampling_state.get("completed_microbatches")
    if (
        isinstance(accum_steps, bool)
        or not isinstance(accum_steps, int)
        or accum_steps < 1
        or isinstance(completed_microbatches, bool)
        or not isinstance(completed_microbatches, int)
        or completed_microbatches != steps * accum_steps
    ):
        raise ValueError("SFT continuation parent microbatch accounting is incomplete")
    return checkpoint


def _validate_sft_continuation_parent(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    continuation: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify an exact parent seal against one completed SFT checkpoint.

    This helper is intentionally reusable by preflight: it performs the same completed-parent
    validation as the runner, derives every observable pin from the checkpoint, and returns the
    normalized configured seal only after all six values match.
    """

    resolved_continuation = resolve_sft_continuation(
        {"continuation": continuation}
    )
    if resolved_continuation is None or "parent" not in resolved_continuation:
        raise ValueError("SFT continuation parent seal is required")
    expected = resolved_continuation["parent"]
    checkpoint = _validated_completed_sft_parent(checkpoint)
    if not _valid_sha256(checkpoint_sha256):
        raise ValueError("SFT continuation parent checkpoint SHA-256 is invalid")

    training_contract = checkpoint["training_contract"]
    lm_sampling = training_contract.get("lm_sampling")
    if not isinstance(lm_sampling, Mapping):
        raise ValueError("SFT continuation parent LM sampling contract is invalid")
    sampling_state = checkpoint["sampling_state"]
    completed_lm_cursor = sampling_state.get("lm_cursor")
    if (
        isinstance(completed_lm_cursor, bool)
        or not isinstance(completed_lm_cursor, int)
        or completed_lm_cursor < 0
    ):
        raise ValueError("SFT continuation parent LM cursor is invalid")

    from localagent.train.stage_data import canonical_sha256

    observed = {
        "checkpoint_sha256": checkpoint_sha256,
        "resume_integrity_sha256": checkpoint["resume_integrity_sha256"],
        "training_contract_sha256": canonical_sha256(training_contract),
        "lm_sampling_sha256": canonical_sha256(lm_sampling),
        "completed_steps": sampling_state["completed_steps"],
        "completed_lm_cursor": completed_lm_cursor,
    }
    for field in _SFT_CONTINUATION_PARENT_FIELDS:
        if observed[field] != expected[field]:
            raise ValueError(
                f"SFT continuation parent {field} mismatch: "
                f"expected={expected[field]!r}, observed={observed[field]!r}"
            )
    return dict(expected)


def _validate_parent_anchored_sampling_parent(
    checkpoint: Mapping[str, Any],
    sampling_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Cross-bind a v3 parent-anchored sampling plan to its completed SFT parent."""

    if not isinstance(checkpoint, Mapping):
        raise TypeError("parent-anchored sampling checkpoint must be a mapping")
    if not isinstance(sampling_config, Mapping):
        raise TypeError("parent-anchored sampling config must be a mapping")

    sampling_mode = sampling_config.get("mode")
    if not isinstance(sampling_mode, str):
        raise TypeError("data.sampling.mode must be a string")
    if sampling_mode != PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE:
        raise ValueError(
            "parent-anchored sampling config mode mismatch: "
            f"expected={PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE!r}, "
            f"observed={sampling_mode!r}"
        )

    parent_prefix_decisions = sampling_config.get("parent_prefix_decisions")
    if (
        isinstance(parent_prefix_decisions, bool)
        or not isinstance(parent_prefix_decisions, int)
        or parent_prefix_decisions < 0
    ):
        raise ValueError(
            "data.sampling.parent_prefix_decisions must be a non-negative integer"
        )
    configured_update_decisions = sampling_config.get("update_decisions")
    if (
        isinstance(configured_update_decisions, bool)
        or not isinstance(configured_update_decisions, int)
        or configured_update_decisions < 1
    ):
        raise ValueError("data.sampling.update_decisions must be a positive integer")
    expected_parent_order_sha256 = sampling_config.get(
        "expected_parent_order_sha256"
    )
    if not _valid_sha256(expected_parent_order_sha256):
        raise ValueError(
            "data.sampling.expected_parent_order_sha256 must be a lowercase SHA-256 digest"
        )

    training_contract = checkpoint.get("training_contract")
    if not isinstance(training_contract, Mapping):
        raise ValueError(
            "parent-anchored sampling parent training contract must be a mapping"
        )
    parent_lm_sampling = training_contract.get("lm_sampling")
    if not isinstance(parent_lm_sampling, Mapping):
        raise ValueError(
            "parent-anchored sampling parent LM sampling contract must be a mapping"
        )
    parent_lm_sampling_mode = parent_lm_sampling.get("mode")
    if not isinstance(parent_lm_sampling_mode, str):
        raise TypeError("parent-anchored sampling parent LM sampling mode must be a string")
    if parent_lm_sampling_mode != QUOTA_SAMPLING_MODE:
        raise ValueError(
            "parent-anchored sampling parent LM sampling mode mismatch: "
            f"expected={QUOTA_SAMPLING_MODE!r}, observed={parent_lm_sampling_mode!r}"
        )
    parent_no_replacement = parent_lm_sampling.get("no_replacement")
    if not isinstance(parent_no_replacement, bool):
        raise TypeError(
            "parent-anchored sampling parent no_replacement must be boolean"
        )
    if not parent_no_replacement:
        raise ValueError(
            "parent-anchored sampling parent LM sampling must be no-replacement"
        )
    parent_ordering = parent_lm_sampling.get("ordering")
    if not isinstance(parent_ordering, Mapping):
        raise ValueError(
            "parent-anchored sampling parent ordering contract must be a mapping"
        )
    parent_order_sha256 = parent_ordering.get("order_sha256")
    if not _valid_sha256(parent_order_sha256):
        raise ValueError(
            "parent-anchored sampling parent order_sha256 must be a lowercase SHA-256 digest"
        )

    batch_size = training_contract.get("batch_size")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError(
            "parent-anchored sampling parent batch_size must be a positive integer"
        )
    accum_steps = training_contract.get("accum_steps")
    if (
        isinstance(accum_steps, bool)
        or not isinstance(accum_steps, int)
        or accum_steps < 1
    ):
        raise ValueError(
            "parent-anchored sampling parent accum_steps must be a positive integer"
        )
    parent_update_decisions = batch_size * accum_steps

    sampling_state = checkpoint.get("sampling_state")
    if not isinstance(sampling_state, Mapping):
        raise ValueError(
            "parent-anchored sampling parent sampling state must be a mapping"
        )
    completed_steps = sampling_state.get("completed_steps")
    if (
        isinstance(completed_steps, bool)
        or not isinstance(completed_steps, int)
        or completed_steps < 1
    ):
        raise ValueError(
            "parent-anchored sampling parent completed_steps must be a positive integer"
        )
    completed_lm_cursor = sampling_state.get("lm_cursor")
    if (
        isinstance(completed_lm_cursor, bool)
        or not isinstance(completed_lm_cursor, int)
        or completed_lm_cursor < 0
    ):
        raise ValueError(
            "parent-anchored sampling parent lm_cursor must be a non-negative integer"
        )

    if parent_prefix_decisions != completed_lm_cursor:
        raise ValueError(
            "parent-anchored sampling parent_prefix_decisions mismatch: "
            f"configured={parent_prefix_decisions}, observed={completed_lm_cursor}"
        )
    if configured_update_decisions != parent_update_decisions:
        raise ValueError(
            "parent-anchored sampling update_decisions mismatch: "
            f"configured={configured_update_decisions}, observed={parent_update_decisions}"
        )
    if expected_parent_order_sha256 != parent_order_sha256:
        raise ValueError(
            "parent-anchored sampling parent order SHA-256 mismatch: "
            f"configured={expected_parent_order_sha256}, observed={parent_order_sha256}"
        )
    expected_lm_cursor = completed_steps * parent_update_decisions
    if completed_lm_cursor != expected_lm_cursor:
        raise ValueError(
            "parent-anchored sampling parent cursor arithmetic mismatch: "
            f"cursor={completed_lm_cursor}, completed_steps={completed_steps}, "
            f"update_decisions={parent_update_decisions}, expected={expected_lm_cursor}"
        )

    return {
        "parent_lm_sampling_mode": parent_lm_sampling_mode,
        "parent_no_replacement": parent_no_replacement,
        "parent_order_sha256": parent_order_sha256,
        "parent_completed_steps": completed_steps,
        "parent_completed_lm_cursor": completed_lm_cursor,
        "parent_update_decisions": parent_update_decisions,
    }


def _validated_resume_heldout_baseline(
    checkpoint: Mapping[str, Any],
    *,
    expected_contract: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Extract a sealed held-out baseline only when it matches current verified eval data."""

    recorded = checkpoint.get("heldout_baseline")
    if expected_contract is None:
        if recorded is not None:
            raise ValueError(
                "SFT resume checkpoint heldout baseline does not match current evaluation contract"
            )
        return None
    if not isinstance(recorded, Mapping):
        raise ValueError("SFT resume checkpoint heldout baseline is missing or malformed")
    if set(recorded) != {"contract", "pre"}:
        raise ValueError(
            "SFT resume checkpoint heldout baseline must contain exactly contract and pre"
        )
    recorded_contract = recorded.get("contract")
    if not isinstance(recorded_contract, Mapping):
        raise ValueError("SFT resume checkpoint heldout baseline contract is malformed")
    if dict(recorded_contract) != dict(expected_contract):
        raise ValueError("SFT resume checkpoint heldout baseline contract mismatch")

    pre = recorded.get("pre")
    if not isinstance(pre, Mapping):
        raise ValueError("SFT resume checkpoint heldout baseline pre metrics are malformed")
    expected_pre_fields = {
        "rows",
        "assistant_loss_tokens",
        "mean_loss",
        "assistant_token_accuracy",
        "assistant_sequence_accuracy",
    }
    if set(pre) != expected_pre_fields:
        raise ValueError("SFT resume checkpoint heldout baseline pre metrics are malformed")
    for key in ("rows", "assistant_loss_tokens"):
        value = pre[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("SFT resume checkpoint heldout baseline pre metrics are malformed")
    mean_loss = pre["mean_loss"]
    if (
        isinstance(mean_loss, bool)
        or not isinstance(mean_loss, (int, float))
        or not math.isfinite(float(mean_loss))
        or float(mean_loss) < 0.0
    ):
        raise ValueError("SFT resume checkpoint heldout baseline pre metrics are malformed")
    for key in ("assistant_token_accuracy", "assistant_sequence_accuracy"):
        value = pre[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError("SFT resume checkpoint heldout baseline pre metrics are malformed")
    return {
        "contract": dict(expected_contract),
        "pre": dict(pre),
    }


def _model_config_mapping(model) -> dict[str, Any]:
    value = getattr(model.cfg, "__dict__", None)
    if not isinstance(value, Mapping):
        raise TypeError("sft() model cfg must expose a mapping-compatible __dict__")
    return dict(value)


def _prepared_sft_sha256(prepared) -> str:
    """Bind every rendered LM/auxiliary row and its source without retaining a second copy."""

    digest = hashlib.sha256()
    for label, entries in (
        ("main", prepared.main_entries),
        ("decay", prepared.decay_entries),
    ):
        _update_resume_digest(digest, label)
        _update_resume_digest(digest, len(entries))
        for row, source in entries:
            _update_resume_digest(digest, row[0])
            _update_resume_digest(digest, row[1])
            _update_resume_digest(digest, source)
    _update_resume_digest(digest, "rows")
    for row in prepared.rows:
        _update_resume_digest(digest, row[0])
        _update_resume_digest(digest, row[1])
    _update_resume_digest(digest, prepared.head_items)
    _update_resume_digest(digest, prepared.multi_turn_items)
    _update_resume_digest(digest, prepared.dataset_accounting)
    return digest.hexdigest()


def _tokenizer_contract(tok) -> dict[str, Any]:
    return {
        "class": f"{type(tok).__module__}.{type(tok).__qualname__}",
        "vocab_size": int(tok.vocab_size),
        "pad_id": int(tok.pad_id),
        "eos_id": int(tok.eos_id),
    }


def _assert_optional_metadata(
    checkpoint: Mapping[str, Any],
    *,
    key: str,
    expected: Mapping[str, Any] | None,
) -> None:
    recorded = checkpoint.get(key)
    if expected is None:
        if recorded is not None:
            raise ValueError(f"resume checkpoint records {key} metadata but none was provided")
        return
    if not isinstance(recorded, Mapping):
        raise TypeError(f"resume checkpoint has no valid {key} metadata")
    if dict(recorded) != dict(expected):
        raise ValueError(f"resume checkpoint {key} metadata mismatch")


def _validated_sft_history(value: Any, *, completed_steps: int) -> list[float]:
    if not isinstance(value, list) or len(value) != completed_steps:
        raise ValueError("resume checkpoint loss_history length disagrees with completed steps")
    history = []
    for loss in value:
        if (
            isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(float(loss))
        ):
            raise ValueError("resume checkpoint loss_history contains an invalid value")
        history.append(float(loss))
    return history


def _dense_selector_dimensions(
    state: dict[str, torch.Tensor],
    *,
    expected_d_model: int,
    configured_proj: int,
) -> tuple[int, int]:
    """Infer a restored dense selector's input/projection widths from its state."""

    query_weight = state.get("q_proj.weight")
    tool_weight = state.get("t_proj.weight")
    if (
        not isinstance(query_weight, torch.Tensor)
        or query_weight.ndim != 2
        or not isinstance(tool_weight, torch.Tensor)
        or tool_weight.ndim != 2
    ):
        raise ValueError("dense_selector checkpoint lacks 2-D q_proj/t_proj weights")
    projection_dim, model_dim = query_weight.shape
    tool_projection_dim, embedding_dim = tool_weight.shape
    if model_dim != expected_d_model:
        raise ValueError(
            "dense_selector checkpoint d_model does not match model config: "
            f"{model_dim} != {expected_d_model}"
        )
    if tool_projection_dim != projection_dim:
        raise ValueError("dense_selector checkpoint projection widths disagree")
    if projection_dim != configured_proj:
        raise ValueError(
            "dense_selector checkpoint projection width does not match selector_proj: "
            f"{projection_dim} != {configured_proj}"
        )
    return int(embedding_dim), int(projection_dim)


def _structured_prediction_invariance(
    natural_records: Sequence[dict[str, Any]],
    trailing_records: Sequence[dict[str, Any]],
    *,
    reference_condition: str,
    comparison_condition: str,
) -> dict[str, Any]:
    """Compare route, selector, and dispatched predictions on the same held-out decisions."""

    def index_records(records: Sequence[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        indexed = {int(record["configured_index"]): record for record in records}
        if len(indexed) != len(records):
            raise ValueError("structured evaluation returned duplicate configured indices")
        return indexed

    reference = index_records(natural_records)
    comparison = index_records(trailing_records)
    comparable_indices = sorted(reference.keys() & comparison.keys())

    route_mismatches = 0
    selector_mismatches = 0
    dispatch_mismatches = 0
    for index in comparable_indices:
        left = reference[index]
        right = comparison[index]
        route_mismatches += left["predicted_route"] != right["predicted_route"]
        selector_mismatches += left["predicted_tool"] != right["predicted_tool"]
        left_dispatch = None if left["predicted_route"] == "text" else left["predicted_tool"]
        right_dispatch = None if right["predicted_route"] == "text" else right["predicted_tool"]
        dispatch_mismatches += left_dispatch != right_dispatch

    return {
        "reference_condition": reference_condition,
        "comparison_condition": comparison_condition,
        "comparable_rows": len(comparable_indices),
        "reference_only_rows": len(reference.keys() - comparison.keys()),
        "comparison_only_rows": len(comparison.keys() - reference.keys()),
        "route_prediction_mismatches": route_mismatches,
        "selector_prediction_mismatches": selector_mismatches,
        "dispatched_prediction_mismatches": dispatch_mismatches,
        "all_comparable_predictions_match": not (
            route_mismatches or selector_mismatches or dispatch_mismatches
        ),
    }


@torch.no_grad()
def _evaluate_conversations(
    model,
    conversations,
    tok,
    *,
    max_seq_len: int,
    batch_size: int,
    device: str,
    amp_dtype=torch.float32,
    conversation_prompt_contract: str | None = None,
    pad_to_input_tokens: int | None = None,
) -> dict[str, Any]:
    """Evaluate an explicit held-out Conversation set in deterministic file order."""

    if max_seq_len < 2 or batch_size < 1:
        raise ValueError("held-out max_seq_len must be >= 2 and batch_size must be positive")
    fixed_input_width = validate_pad_to_input_tokens(
        pad_to_input_tokens,
        label="evaluation.pad_to_input_tokens",
    )
    if fixed_input_width is not None and fixed_input_width > max_seq_len:
        raise ValueError(
            "evaluation.pad_to_input_tokens cannot exceed the held-out sequence limit: "
            f"configured={fixed_input_width}, limit={max_seq_len}"
        )
    rows = []
    catalog_token_cache = CatalogTokenCache(tok)
    for conversation in conversations:
        for row in render_conversation_rows(
            conversation,
            tok,
            prompt_contract=conversation_prompt_contract,
            max_seq_len=max_seq_len,
            catalog_cache=catalog_token_cache,
        ):
            _, loss_tokens = shifted_token_counts(row)
            if token_row_length(row) >= 2 and loss_tokens > 0:
                rows.append(row)
    if not rows:
        raise ValueError("SFT held-out data has no trainable assistant targets")
    required_input_width = max(token_row_length(row) - 1 for row in rows)
    if fixed_input_width is not None and required_input_width > fixed_input_width:
        raise ValueError(
            "held-out row requires more input tokens than "
            "evaluation.pad_to_input_tokens: "
            f"required={required_input_width}, configured={fixed_input_width}"
        )

    was_training = model.training
    model.to(device).eval()
    device_obj = torch.device(device)
    loss_sum = 0.0
    correct_tokens = 0
    loss_tokens = 0
    exact_rows = 0
    for start in range(0, len(rows), batch_size):
        x = y = logits = flat_targets = flat_logits = mask = predictions = None
        token_correct = None
        try:
            x, y = pad_batch(
                rows[start : start + batch_size],
                tok.pad_id,
                device,
                pad_to_input_tokens=fixed_input_width,
            )
            with autocast_ctx(device_obj, amp_dtype):
                logits, _ = model(x)
            logits = logits.float()
            flat_targets = y.reshape(-1)
            flat_logits = logits.reshape(-1, logits.shape[-1])
            mask = flat_targets != IGNORE
            batch_loss_tokens = int(mask.sum())
            loss_sum += float(
                F.cross_entropy(
                    flat_logits,
                    flat_targets,
                    ignore_index=IGNORE,
                    reduction="sum",
                )
            )
            predictions = logits.argmax(dim=-1)
            token_correct = (predictions == y) & (y != IGNORE)
            correct_tokens += int(token_correct.sum())
            loss_tokens += batch_loss_tokens
            exact_rows += int(((predictions == y) | (y == IGNORE)).all(dim=1).sum())
        finally:
            del (
                token_correct,
                predictions,
                mask,
                flat_logits,
                flat_targets,
                logits,
                y,
                x,
            )
            _mps_synchronize_and_empty_cache(device_obj)
    model.train(was_training)
    return {
        "rows": len(rows),
        "assistant_loss_tokens": loss_tokens,
        "mean_loss": loss_sum / loss_tokens,
        "assistant_token_accuracy": correct_tokens / loss_tokens,
        "assistant_sequence_accuracy": exact_rows / len(rows),
    }


def _framed_full(model, tok, prompts, device, max_seq_len=None):
    """Per-token features for a batch of framed prompts (with grad). Returns
    (feats (B,Tmax,d), lengths (B,), framed_ids list)."""
    limit = min(max_seq_len or model.cfg.max_seq_len, model.cfg.max_seq_len)
    enc = [framed_prompt_ids(tok, prompt, limit) for prompt in prompts]
    maxlen = max(len(e) for e in enc)
    X = torch.full((len(enc), maxlen), tok.pad_id, dtype=torch.long, device=device)
    for i, e in enumerate(enc):
        X[i, : len(e)] = torch.tensor(e, device=device)
    _, feats = model(X, return_hidden=True)
    lengths = torch.tensor([len(e) for e in enc], device=device)
    return feats, lengths, enc


def sft(
    model,
    samples,
    tok,
    *,
    steps=1200,
    batch_size=32,
    lr=1e-3,
    optimizer_name="adamw",
    weight_decay=0.0,
    grad_clip=1.0,
    warmup=40,
    device="cpu",
    log=print,
    joint_tool_head=False,
    ptr_args=None,
    aux_weight=1.0,
    ptr_weight=0.15,
    conversations=None,
    accum_steps=1,
    mt_weight=1.0,
    multi_turn_batch_size=12,
    teacher=None,
    kd_type="topk",
    kd_k=16,
    kd_weight=0.5,
    kd_temperature=2.0,
    lr_schedule="cosine",
    decay_frac=0.2,
    decay_samples=None,
    shuffle=True,
    init_tool_head=None,
    init_ptr_head=None,
    max_seq_len=None,
    pad_to_input_tokens=None,
    seed=0,
    amp_dtype=torch.float32,
    sample_sources=None,
    conversation_sources=None,
    decay_sample_sources=None,
    decay_conversations=None,
    decay_conversation_sources=None,
    conversation_prompt_contract=None,
    return_metrics=False,
    checkpoint_path=None,
    checkpoint_every=0,
    resume_from=None,
    lineage=None,
    tokenizer_metadata=None,
    data_metadata=None,
    execution=None,
    heldout_baseline=None,
    lm_order_keys=None,
    sampling_contract=None,
    loss_normalization=SFT_LOSS_NORMALIZATION_MICROBATCH,
    freeze_parameters=None,
    resume_git_receipt=None,
    resume_checkpoint_sha256=None,
    archive_checkpoints=False,
    _max_optimizer_updates=None,
):
    """SFT with masked LM loss over single-turn samples + optional multi-turn `conversations`
    (which teach tool->response->follow-up continuation). With `joint_tool_head`, also trains
    jointly a tool-selection head AND a pointer/copy argument head (on the single-turn samples).

    `steps` is the number of OPTIMIZER steps; each runs `accum_steps` micro-batches of size
    `batch_size` (effective batch = batch_size * accum_steps). Each micro-batch's combined loss is
    divided by `accum_steps` and backward()'d immediately, so peak memory stays at one micro-batch
    regardless of accumulation. `mt_weight` scales ONLY the multi-turn head-training losses (tool +
    pointer CE on episode contexts); the LM loss on rendered conversations is unaffected.
    `multi_turn_batch_size` controls that auxiliary episode-context forward independently and may
    be zero to disable it without changing the conversation LM stream.

    **distill-throughout-SFT** (optional, default OFF): if `teacher` is given, the teacher's
    Top-K next-token targets are cached ONCE on the single-turn SFT `samples` (reusing
    distill.py's `cache_teacher_topk`, memory = K/pos not full vocab), and each step adds
    `kd_weight * topk_kd_loss(student_logits, teacher_topk)` on the assistant spans alongside the
    LM/head/pointer losses. The backbone keeps matching the teacher's distribution WHILE the heads
    train, so it is not pulled away from verbatim arg-copying. Only `kd_type="topk"` is supported
    here (it reuses distill.py's `_topk_kd_loss`). When `teacher is None` the path is inert and
    every existing caller is byte-for-byte unchanged.

    **WSD schedule** (opt-in, MiniCPM 2404.06395): `lr_schedule="wsd"` switches the per-step LR
    from cosine to Warmup-Stable-Decay — linear warmup -> flat `lr` plateau -> exponential
    `lr*0.5^((s-S)/T)` over the last `decay_frac` of steps (T = decay-window length). Default
    `lr_schedule="cosine"` is byte-for-byte the old schedule. `decay_samples` (a separate, ideally
    cleaner/curated sample pool) is OPTIONAL: when given AND on WSD, the single-turn LM rows drawn
    during the decay window come from `decay_samples` instead of the main pool — the on-device
    "inject your cleanest data in the decay window" trick. Multi-turn `conversations` and the head
    items are unchanged (heads keep their full training distribution).

    **Ordered passes** (opt-in): with `shuffle=False`, the single-turn LM micro-batches walk
    `lm_rows` in source order and wrap at the end. Supplying ``lm_order_keys`` instead maps a
    canonical assistant-decision permutation onto the rendered full-catalog rows and consumes its
    prefix without replacement; the fixed horizon must fit within one complete decision epoch.
    Stage configs may select either a quota prefix (optionally beginning at a sealed decision
    offset) or a complete mixed-replay permutation whose prefix has an exact per-update cycle.
    Default `shuffle=True` retains i.i.d. sampling with replacement. Only the LM stream is affected;
    head / pointer / multi-turn / KD micro-batches stay i.i.d. The opt-in
    ``assistant_token_mean_per_update_v1`` loss normalization weights each accumulated
    micro-batch by its exact assistant-token count, preventing short targets from receiving the
    same update mass as long tool calls. It is restricted to the LM-only training path.
    ``optimizer_name`` is intentionally restricted to exact lowercase ``"adamw"``.
    ``weight_decay`` and ``grad_clip`` are applied as configured and sealed into exact-resume
    checkpoints. ``freeze_parameters`` optionally names exact model parameters to mark
    non-trainable and omit from AdamW. Its order and the resulting ordered optimizer model
    parameter names are sealed into exact-resume checkpoints; auxiliary heads are unaffected.
    ``pad_to_input_tokens`` optionally fixes the post-shift LM tensor width. Rows must fit exactly;
    the runner only right-pads and never truncates.
    Returns ``(loss_hist, tool_head, ptr_head)``; heads are None unless ``joint_tool_head``.
    ``return_metrics=True`` appends deterministic dataset and realized LM-token accounting.

    When ``checkpoint_path`` is set, checkpoints are written atomically after complete optimizer
    steps. With ``archive_checkpoints=True``, every ``checkpoint_every`` boundary also creates an
    immutable ``<stem>.step-<completed_steps:08d><suffix>`` resume envelope beside the rolling
    checkpoint. ``resume_from`` is an exact continuation of the same fixed horizon, parent state,
    data, prompt contract, and execution environment; it is not a warm-start mechanism.
    """
    if steps < 1 or batch_size < 1 or accum_steps < 1:
        raise ValueError("steps, batch_size, and accum_steps must be positive")
    optimizer_name, weight_decay, grad_clip = _validate_sft_optimizer_contract(
        optimizer_name,
        weight_decay,
        grad_clip,
    )
    if _max_optimizer_updates is not None and (
        isinstance(_max_optimizer_updates, bool)
        or not isinstance(_max_optimizer_updates, int)
        or _max_optimizer_updates < 1
    ):
        raise ValueError("_max_optimizer_updates must be a positive integer when provided")
    if resume_git_receipt is not None and resume_from is None:
        raise ValueError("resume_git_receipt requires resume_from")
    if resume_git_receipt is None and resume_checkpoint_sha256 is not None:
        raise ValueError("resume_checkpoint_sha256 requires resume_git_receipt")
    multi_turn_batch_size = validate_multi_turn_batch_size(multi_turn_batch_size)
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be non-negative")
    if not isinstance(archive_checkpoints, bool):
        raise TypeError("archive_checkpoints must be boolean")
    if archive_checkpoints and checkpoint_path is None:
        raise ValueError("archive_checkpoints requires checkpoint_path")
    if archive_checkpoints and (
        isinstance(checkpoint_every, bool)
        or not isinstance(checkpoint_every, int)
        or checkpoint_every < 1
    ):
        raise ValueError("archive_checkpoints requires positive checkpoint_every")
    loss_normalization = validate_sft_loss_normalization(loss_normalization)
    if loss_normalization == SFT_LOSS_NORMALIZATION_UPDATE_TOKENS and (
        joint_tool_head or teacher is not None
    ):
        raise ValueError(
            "assistant-token update normalization supports the LM-only SFT path"
        )
    torch.manual_seed(seed)
    model.train()
    model.to(device)
    frozen_parameter_names = _validate_sft_freeze_parameters(model, freeze_parameters)
    device_obj = torch.device(device)
    exact_resume_enabled = checkpoint_path is not None or resume_from is not None
    if exact_resume_enabled and device_obj.type not in {
        "cpu",
        "cuda",
        "mps",
        "xpu",
    }:
        raise ValueError(
            "exact SFT resume supports CPU, CUDA, MPS, and XPU RNG state only; "
            f"got {device_obj.type!r}"
        )
    initial_model_sha256 = _resume_sha256(model.state_dict()) if exact_resume_enabled else None
    use_grad_scaler = device_obj.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    seq_limit = min(max_seq_len or model.cfg.max_seq_len, model.cfg.max_seq_len)
    fixed_input_width = validate_pad_to_input_tokens(
        pad_to_input_tokens,
        label="batch.pad_to_input_tokens",
    )
    if fixed_input_width is not None and fixed_input_width > seq_limit:
        raise ValueError(
            "batch.pad_to_input_tokens cannot exceed the SFT sequence limit: "
            f"configured={fixed_input_width}, limit={seq_limit}"
        )
    prepared = prepare_sft_data(
        samples,
        tok,
        conversations=conversations,
        sample_sources=sample_sources,
        conversation_sources=conversation_sources,
        decay_samples=decay_samples,
        decay_sample_sources=decay_sample_sources,
        lr_schedule=lr_schedule,
        max_seq_len=seq_limit,
        joint_tool_head=joint_tool_head,
        ptr_args=ptr_args,
        conversation_prompt_contract=conversation_prompt_contract,
        decay_conversations=decay_conversations,
        decay_conversation_sources=decay_conversation_sources,
    )
    samples = list(prepared.samples)
    conversations = list(prepared.conversations)
    rows = prepared.rows
    lm_rows = prepared.main_entries
    decay_lm_rows = prepared.decay_entries
    required_input_width = max(
        token_row_length(row) - 1
        for pool in (lm_rows, decay_lm_rows)
        for row, _ in pool
    )
    if fixed_input_width is not None and required_input_width > fixed_input_width:
        raise ValueError(
            "SFT row requires more input tokens than batch.pad_to_input_tokens: "
            f"required={required_input_width}, configured={fixed_input_width}"
        )
    dataset_accounting = prepared.dataset_accounting
    lm_order = None
    if lm_order_keys is not None:
        if prepared.conversation_prompt_contract == LEGACY_CONVERSATION_PROMPT_CONTRACT:
            raise ValueError("quota decision ordering requires a full-catalog prompt contract")
        lm_order = decision_keys_to_row_order(
            prepared.conversations,
            lm_order_keys,
            expected_rows=len(lm_rows),
        )
        required_rows = steps * accum_steps * batch_size
        if required_rows > len(lm_order):
            raise ValueError(
                "quota no-replacement SFT horizon exceeds the available decision rows: "
                f"required={required_rows}, available={len(lm_order)}"
            )
        if not isinstance(sampling_contract, Mapping):
            raise TypeError("quota decision ordering requires sampling_contract metadata")
    elif sampling_contract is not None:
        raise ValueError("sampling_contract requires lm_order_keys")
    training_accounting = _empty_token_accounting(
        [source for pool in (lm_rows, decay_lm_rows) for _, source in pool]
    )

    # --- distill-throughout-SFT: cache teacher Top-K targets ONCE on the SFT rows ---
    kd_cache = None
    teacher_state_sha256 = (
        _resume_sha256(teacher.state_dict())
        if exact_resume_enabled and teacher is not None
        else None
    )
    if teacher is not None:
        if kd_type != "topk":
            raise ValueError("sft() distillation only supports kd_type='topk'")
        from localagent.train.distill import _topk_kd_loss, cache_teacher_topk

        log(f"  [sft] caching teacher top-{kd_k} targets for distill-throughout-SFT ...")
        kd_cache = cache_teacher_topk(
            teacher, rows, tok, device=device, temperature=kd_temperature, k=kd_k, log=log
        )
        V_kd = model.cfg.vocab_size
    tool_head = ptr_head = None
    optimizer_model_parameter_names = None
    if frozen_parameter_names is None:
        params = list(model.parameters())
    else:
        frozen_parameter_set = set(frozen_parameter_names)
        optimizer_model_parameter_names = [
            name
            for name, _ in model.named_parameters()
            if name not in frozen_parameter_set
        ]
        params = [
            parameter
            for name, parameter in model.named_parameters()
            if name not in frozen_parameter_set
        ]
    if joint_tool_head:
        from localagent.agent.pointer_head import PointerHead
        from localagent.agent.tool_head import ToolHead

        tool_head = ToolHead(model.cfg.d_model).to(device)
        ptr_head = PointerHead(model.cfg.d_model, args=ptr_args or PTR_ARGS).to(device)
        # Warm-start the heads from a prior checkpoint to CONTINUE-train (keep learned
        # selection/grounding and adapt to new data) instead of resetting them.
        if init_tool_head is not None:
            tool_head.load_state_dict(init_tool_head)
        if init_ptr_head is not None:
            ptr_head.load_state_dict(init_ptr_head)
        params += list(tool_head.parameters()) + list(ptr_head.parameters())
        head_items = prepared.head_items
        mt = prepared.multi_turn_items
    initial_tool_head_sha256 = (
        _resume_sha256(tool_head.state_dict())
        if exact_resume_enabled and tool_head is not None
        else None
    )
    initial_ptr_head_sha256 = (
        _resume_sha256(ptr_head.state_dict())
        if exact_resume_enabled and ptr_head is not None
        else None
    )
    opt = torch.optim.AdamW(
        params,
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )
    hist: list[float] = []
    lm_loss_history: list[float | None] = []
    router_aux_loss_history: list[float | None] = []
    router_weighted_loss_history: list[float | None] = []
    sampling_schedule = SFTSamplingSchedule(
        prepared,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        lr_schedule=lr_schedule,
        decay_frac=decay_frac,
        kd_enabled=kd_cache is not None,
        joint_tool_head=joint_tool_head,
        multi_turn_batch_size=multi_turn_batch_size,
        lm_order=lm_order,
    )
    training_contract = {
        "version": 1,
        "steps": int(steps),
        "batch_size": int(batch_size),
        "accum_steps": int(accum_steps),
        "lr": float(lr),
        "warmup": int(warmup),
        "lr_schedule": str(lr_schedule),
        "decay_frac": float(decay_frac),
        "shuffle": bool(shuffle),
        "lm_sampling": (
            dict(sampling_contract)
            if sampling_contract is not None
            else {
                "mode": (
                    "iid_with_replacement_v1" if shuffle else "source_order_wrapping_v1"
                )
            }
        ),
        "joint_tool_head": bool(joint_tool_head),
        "aux_weight": float(aux_weight),
        "ptr_weight": float(ptr_weight),
        "mt_weight": float(mt_weight),
        "multi_turn_batch_size": int(multi_turn_batch_size),
        "kd_type": str(kd_type),
        "kd_k": int(kd_k),
        "kd_weight": float(kd_weight),
        "kd_temperature": float(kd_temperature),
        "kd_enabled": kd_cache is not None,
        "teacher_state_sha256": teacher_state_sha256,
        "teacher_cache_sha256": (
            _resume_sha256(kd_cache) if exact_resume_enabled and kd_cache is not None else None
        ),
        "max_seq_len": int(seq_limit),
        **(
            {"pad_to_input_tokens": fixed_input_width}
            if fixed_input_width is not None
            else {}
        ),
        "amp_dtype": str(amp_dtype),
        "seed": int(seed),
        "conversation_prompt_contract": prepared.conversation_prompt_contract,
        "tokenizer": _tokenizer_contract(tok),
        "prepared_data_sha256": (_prepared_sft_sha256(prepared) if exact_resume_enabled else None),
        "initial_model_sha256": initial_model_sha256,
        "initial_tool_head_sha256": initial_tool_head_sha256,
        "initial_ptr_head_sha256": initial_ptr_head_sha256,
        "optimizer": {
            "kind": "AdamW",
            "betas": [0.9, 0.95],
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
        },
        "loss_normalization": loss_normalization,
        **(
            {
                "freeze_parameters": list(frozen_parameter_names),
                "optimizer_model_parameter_names": optimizer_model_parameter_names,
            }
            if frozen_parameter_names is not None
            else {}
        ),
        **(
            {
                "archive_checkpoints": True,
                "checkpoint_archive_every": int(checkpoint_every),
                "checkpoint_archive_format": "immutable_periodic_sft_v1",
            }
            if archive_checkpoints
            else {}
        ),
    }
    start_step = 0

    if resume_from is not None:
        from localagent.train.resume_git_receipt import assert_resume_git_receipt
        from localagent.train.stage_data import assert_resume_lineage

        checkpoint = _load_validated_sft_resume_checkpoint(resume_from)
        if checkpoint.get("cfg") != _model_config_mapping(model):
            raise ValueError("SFT resume checkpoint model config mismatch")
        if checkpoint.get("training_seed") != seed:
            raise ValueError("SFT resume checkpoint training seed mismatch")
        if checkpoint.get("training_contract") != training_contract:
            raise ValueError("SFT resume checkpoint training contract mismatch")
        if checkpoint.get("conversation_prompt_contract") != (
            prepared.conversation_prompt_contract
        ):
            raise ValueError("SFT resume checkpoint conversation prompt contract mismatch")

        recorded_lineage = checkpoint.get("lineage")
        if lineage is not None:
            if resume_git_receipt is None:
                assert_resume_lineage(checkpoint, lineage)
            else:
                if not isinstance(recorded_lineage, Mapping):
                    raise TypeError(
                        "resume checkpoint has no lineage metadata; refusing unsafe resume"
                    )
                assert_resume_git_receipt(
                    resume_git_receipt,
                    checkpoint_sha256=resume_checkpoint_sha256,
                    recorded_lineage=recorded_lineage,
                    expected_lineage=lineage,
                    stage="sft",
                )
        elif recorded_lineage is not None:
            raise ValueError(
                "SFT resume checkpoint records lineage but no expected lineage was provided"
            )
        _assert_optional_metadata(
            checkpoint,
            key="tokenizer",
            expected=tokenizer_metadata,
        )
        _assert_optional_metadata(checkpoint, key="data", expected=data_metadata)
        _assert_optional_metadata(checkpoint, key="execution", expected=execution)
        _assert_optional_metadata(
            checkpoint,
            key="heldout_baseline",
            expected=heldout_baseline,
        )

        state = checkpoint.get("state_dict")
        if not isinstance(state, Mapping):
            raise ValueError("SFT resume checkpoint state_dict is invalid")
        recorded_tool_head = checkpoint.get("tool_head")
        recorded_ptr_head = checkpoint.get("ptr_head")
        if joint_tool_head:
            if not isinstance(recorded_tool_head, Mapping) or not isinstance(
                recorded_ptr_head,
                Mapping,
            ):
                raise ValueError("SFT resume checkpoint is missing joint tool/pointer heads")
        elif recorded_tool_head is not None or recorded_ptr_head is not None:
            raise ValueError("SFT resume checkpoint has unexpected joint tool/pointer heads")
        recorded_optimizer = checkpoint.get("optimizer")
        if not isinstance(recorded_optimizer, Mapping):
            raise ValueError("SFT resume checkpoint optimizer state is invalid")
        recorded_scaler = checkpoint.get("grad_scaler")
        if use_grad_scaler:
            if not isinstance(recorded_scaler, Mapping):
                raise ValueError("SFT resume checkpoint gradient-scaler state is missing")
        elif recorded_scaler is not None:
            raise ValueError("SFT resume checkpoint has unexpected gradient-scaler state")

        checkpoint_step = checkpoint.get("step")
        if (
            isinstance(checkpoint_step, bool)
            or not isinstance(checkpoint_step, int)
            or checkpoint_step < 0
        ):
            raise ValueError("SFT resume checkpoint step is invalid")
        start_step = checkpoint_step + 1
        if start_step > steps:
            raise ValueError(
                f"SFT resume checkpoint is already at step {checkpoint_step}, "
                f"beyond total steps {steps}"
            )
        hist = _validated_sft_history(
            checkpoint.get("loss_history"),
            completed_steps=start_step,
        )
        loss_components = checkpoint.get("loss_components")
        if loss_components is None:
            # Legacy v1 envelopes did not record the decomposition. Preserve the unknown prefix
            # explicitly rather than deriving it from the combined objective.
            lm_loss_history = [None] * start_step
            router_aux_loss_history = [None] * start_step
            router_weighted_loss_history = [None] * start_step
        else:
            if not isinstance(loss_components, Mapping):
                raise ValueError("SFT resume loss_components must be a mapping")
            destinations = {
                "lm_loss": lm_loss_history,
                "router_aux": router_aux_loss_history,
                "router_weighted": router_weighted_loss_history,
            }
            for name, destination in destinations.items():
                values = loss_components.get(name)
                if not isinstance(values, list) or len(values) != start_step:
                    raise ValueError(
                        f"SFT resume loss_components[{name!r}] is inconsistent"
                    )
                if any(
                    value is not None
                    and (isinstance(value, bool) or not isinstance(value, (int, float)))
                    for value in values
                ):
                    raise ValueError(
                        f"SFT resume loss_components[{name!r}] must contain numbers or null"
                    )
                destination.extend(
                    None if value is None else float(value) for value in values
                )
        if checkpoint.get("dataset_token_accounting") != dataset_accounting:
            raise ValueError("SFT resume checkpoint dataset token accounting mismatch")
        if checkpoint.get("token_accounting_scope") != "language_model_microbatches":
            raise ValueError("SFT resume checkpoint token accounting scope mismatch")

        sampling_state = checkpoint.get("sampling_state")
        if not isinstance(sampling_state, Mapping):
            raise ValueError("SFT resume checkpoint sampling state is invalid")
        expected_schedule = SFTSamplingSchedule(
            prepared,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            lr_schedule=lr_schedule,
            decay_frac=decay_frac,
            kd_enabled=kd_cache is not None,
            joint_tool_head=joint_tool_head,
            multi_turn_batch_size=multi_turn_batch_size,
            lm_order=lm_order,
        )
        expected_accounting = _empty_token_accounting(
            [source for pool in (lm_rows, decay_lm_rows) for _, source in pool]
        )
        for replay_step in range(start_step):
            for _ in range(accum_steps):
                selection = expected_schedule.next_microbatch(
                    step=replay_step,
                    total_steps=steps,
                )
                replay_pool = decay_lm_rows if selection.pool == "decay" else lm_rows
                for row_index in selection.lm_indices:
                    replay_row, replay_source = replay_pool[row_index]
                    _add_row_accounting(
                        expected_accounting,
                        replay_row,
                        replay_source,
                    )
        expected_sampling_state = {
            "rng_state": expected_schedule.rng.getstate(),
            "lm_cursor": int(expected_schedule.lm_cursor),
            "completed_steps": start_step,
            "completed_microbatches": start_step * accum_steps,
        }
        if dict(sampling_state) != expected_sampling_state:
            raise ValueError("SFT resume checkpoint sampler state mismatch")
        if checkpoint.get("token_accounting") != expected_accounting:
            raise ValueError("SFT resume checkpoint token accounting mismatch")
        training_accounting = expected_accounting
        sampling_schedule.rng.setstate(sampling_state["rng_state"])
        sampling_schedule.lm_cursor = int(sampling_state["lm_cursor"])

        recorded_torch_rng = checkpoint.get("torch_rng_state")
        if (
            not isinstance(recorded_torch_rng, torch.Tensor)
            or recorded_torch_rng.dtype != torch.uint8
            or recorded_torch_rng.ndim != 1
        ):
            raise ValueError("SFT resume checkpoint Torch RNG state is invalid")
        recorded_cuda_rng = checkpoint.get("cuda_rng_state_all")
        recorded_mps_rng = checkpoint.get("mps_rng_state")
        recorded_xpu_rng = checkpoint.get("xpu_rng_state_all")
        if device_obj.type == "cuda":
            if not isinstance(recorded_cuda_rng, (list, tuple)):
                raise ValueError("SFT resume checkpoint CUDA RNG state is missing")
            if recorded_mps_rng is not None or recorded_xpu_rng is not None:
                raise ValueError("SFT resume checkpoint has unexpected accelerator RNG state")
        elif device_obj.type == "mps":
            if not isinstance(recorded_mps_rng, torch.Tensor):
                raise ValueError("SFT resume checkpoint MPS RNG state is missing")
            if recorded_cuda_rng is not None or recorded_xpu_rng is not None:
                raise ValueError("SFT resume checkpoint has unexpected accelerator RNG state")
        elif device_obj.type == "xpu":
            if not isinstance(recorded_xpu_rng, (list, tuple)):
                raise ValueError("SFT resume checkpoint XPU RNG state is missing")
            if recorded_cuda_rng is not None or recorded_mps_rng is not None:
                raise ValueError("SFT resume checkpoint has unexpected accelerator RNG state")
        elif any(
            value is not None for value in (recorded_cuda_rng, recorded_mps_rng, recorded_xpu_rng)
        ):
            raise ValueError("SFT resume checkpoint has unexpected accelerator RNG state")

        model.load_state_dict(state)
        if joint_tool_head:
            tool_head.load_state_dict(recorded_tool_head)
            ptr_head.load_state_dict(recorded_ptr_head)
        opt.load_state_dict(recorded_optimizer)
        if use_grad_scaler:
            scaler.load_state_dict(recorded_scaler)
        torch.set_rng_state(recorded_torch_rng.cpu())
        if device_obj.type == "cuda":
            torch.cuda.set_rng_state_all(recorded_cuda_rng)
        elif device_obj.type == "mps":
            torch.mps.set_rng_state(recorded_mps_rng.cpu())
        elif device_obj.type == "xpu":
            torch.xpu.set_rng_state_all(recorded_xpu_rng)

    def _micro_loss(selection):
        """Full combined loss (LM + head + ptr + mt) for ONE micro-batch of `batch_size`.
        `lm_pool` is the LM-row pool to sample (swapped to curated rows in the WSD decay window)."""
        lm_pool = decay_lm_rows if selection.pool == "decay" else lm_rows
        selected_entries = [lm_pool[index] for index in selection.lm_indices]
        for row, source in selected_entries:
            _add_row_accounting(training_accounting, row, source)
        x, y = pad_batch(
            [row for row, _ in selected_entries],
            tok.pad_id,
            device,
            pad_to_input_tokens=fixed_input_width,
        )
        _, lm_loss = model(x, targets=y)
        loss, router_aux, router_weighted = router_loss_terms(model, lm_loss)
        if kd_cache is not None:
            # KD micro-batch sampled from the SFT rows (the only rows with cached teacher
            # targets), batched exactly like distill.py: inputs = row[:-1], mask on labels[1:].
            bi = list(selection.kd_indices)
            seqs = [rows[j][0][:-1] for j in bi]
            labs = [rows[j][1][1:] for j in bi]
            ml = max(len(s) for s in seqs)
            Xk = torch.full((len(bi), ml), tok.pad_id, dtype=torch.long, device=device)
            mk = torch.zeros(len(bi), ml, device=device)
            for r in range(len(bi)):
                Xk[r, : len(seqs[r])] = torch.tensor(seqs[r], device=device)
                lab_t = torch.tensor(labs[r], device=device)
                mk[r, : len(lab_t)] = (lab_t != IGNORE).float()
            klogits, _ = model(Xk)
            kd = _topk_kd_loss(klogits, kd_cache, bi, Xk, mk, V_kd, kd_temperature, device)
            loss = loss + kd_weight * kd
        if joint_tool_head:
            from localagent.model.tokenizer import ASSISTANT, USER

            batch = [head_items[index] for index in selection.head_indices]
            feats, lengths, enc = _framed_full(
                model,
                tok,
                [b[0] for b in batch],
                device,
                seq_limit,
            )
            last = feats[torch.arange(len(batch)), lengths - 1]
            loss = loss + aux_weight * F.cross_entropy(
                tool_head(last), torch.tensor([b[1] for b in batch], device=device)
            )
            # pointer head: rows in the batch that have a copy arg with a locatable gold span
            rws, gs, ge, ai = [], [], [], []
            for bi, (prompt, _, parg, pval) in enumerate(batch):
                if parg is None:
                    continue
                contextual_ids, span = _encode_with_value_span(
                    tok,
                    f"{USER}{prompt}{ASSISTANT}",
                    pval,
                    seq_limit,
                )
                if contextual_ids != enc[bi]:
                    raise ValueError("contextual pointer encoding disagrees with framed prompt")
                if span is None:
                    continue
                rws.append(bi)
                gs.append(span[0])
                ge.append(span[1])
                ai.append(ptr_head.arg_idx[parg])
            if rws:
                sub = feats[rws]  # (k,Tmax,d)
                sl, el = ptr_head.logits(sub, torch.tensor(ai, device=device))
                for r, bi in enumerate(rws):  # mask padding positions
                    sl[r, lengths[bi] :] = torch.finfo(sl.dtype).min
                    el[r, lengths[bi] :] = torch.finfo(el.dtype).min
                loss = loss + ptr_weight * (  # down-weighted so it can't swamp tool selection
                    F.cross_entropy(sl, torch.tensor(gs, device=device))
                    + F.cross_entropy(el, torch.tensor(ge, device=device))
                )
            # --- multi-turn head training (episode contexts), scaled by mt_weight ---
            if selection.multi_turn_indices:
                mb = [mt[index] for index in selection.multi_turn_indices]
                ml = max(len(e[0]) for e in mb)
                X = torch.full((len(mb), ml), tok.pad_id, dtype=torch.long, device=device)
                for r, e in enumerate(mb):
                    X[r, : len(e[0])] = torch.tensor(list(e[0]), device=device)
                _, mfeats = model(X, return_hidden=True)
                mlast = mfeats[torch.arange(len(mb)), torch.tensor([len(e[0]) - 1 for e in mb])]
                loss = loss + mt_weight * aux_weight * F.cross_entropy(
                    tool_head(mlast), torch.tensor([e[1] for e in mb], device=device)
                )
                prw = [r for r, e in enumerate(mb) if e[2] >= 0]
                if prw:
                    sl, el = ptr_head.logits(
                        mfeats[prw], torch.tensor([mb[r][2] for r in prw], device=device)
                    )
                    for j, r in enumerate(prw):
                        sl[j, len(mb[r][0]) :] = torch.finfo(sl.dtype).min
                        el[j, len(mb[r][0]) :] = torch.finfo(el.dtype).min
                    loss = loss + mt_weight * ptr_weight * (
                        F.cross_entropy(sl, torch.tensor([mb[r][3] for r in prw], device=device))
                        + F.cross_entropy(el, torch.tensor([mb[r][4] for r in prw], device=device))
                    )
        return loss, lm_loss, router_aux, router_weighted

    def save(step: int) -> None:
        if checkpoint_path is None:
            return
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if device_obj.type == "mps":
            torch.mps.synchronize()
        payload = {
            "resume_format": _SFT_RESUME_FORMAT,
            "resume_version": _SFT_RESUME_VERSION,
            "cfg": _model_config_mapping(model),
            "state_dict": model.state_dict(),
            "tool_head": tool_head.state_dict() if tool_head is not None else None,
            "ptr_head": ptr_head.state_dict() if ptr_head is not None else None,
            "optimizer": opt.state_dict(),
            "grad_scaler": scaler.state_dict() if use_grad_scaler else None,
            "step": step,
            "loss_history": hist,
            "loss_components": {
                "lm_loss": lm_loss_history,
                "router_aux": router_aux_loss_history,
                "router_weighted": router_weighted_loss_history,
            },
            "dataset_token_accounting": dataset_accounting,
            "token_accounting": training_accounting,
            "token_accounting_scope": "language_model_microbatches",
            "sampling_state": {
                "rng_state": sampling_schedule.rng.getstate(),
                "lm_cursor": int(sampling_schedule.lm_cursor),
                "completed_steps": step + 1,
                "completed_microbatches": (step + 1) * accum_steps,
            },
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if device_obj.type == "cuda" else None
            ),
            "mps_rng_state": (torch.mps.get_rng_state() if device_obj.type == "mps" else None),
            "xpu_rng_state_all": (
                torch.xpu.get_rng_state_all() if device_obj.type == "xpu" else None
            ),
            "stage": "sft",
            "training_seed": seed,
            "training_contract": training_contract,
            "lineage": dict(lineage) if lineage is not None else None,
            "conversation_prompt_contract": prepared.conversation_prompt_contract,
            "tokenizer": (dict(tokenizer_metadata) if tokenizer_metadata is not None else None),
            "data": dict(data_metadata) if data_metadata is not None else None,
            "execution": dict(execution) if execution is not None else None,
            "heldout_baseline": (dict(heldout_baseline) if heldout_baseline is not None else None),
        }
        payload["resume_integrity_sha256"] = _sealed_resume_sha256(payload)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        if archive_checkpoints and (step + 1) % checkpoint_every == 0:
            archive_path = _sft_checkpoint_archive_path(
                path,
                completed_steps=step + 1,
            )
            _write_immutable_sft_checkpoint_archive(archive_path, payload)

    stop_step = (
        steps
        if _max_optimizer_updates is None
        else min(steps, start_step + _max_optimizer_updates)
    )
    for step in range(start_step, stop_step):
        if lr_schedule == "wsd":
            set_lr(opt, wsd_lr(step, steps, lr, warmup, decay_frac, min_ratio=0.0))
        else:
            set_lr(opt, cosine_lr(step, steps, lr, warmup, 0.1))
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_lm_loss = 0.0
        step_router_aux = 0.0
        step_router_weighted = 0.0
        selections = [
            sampling_schedule.next_microbatch(step=step, total_steps=steps)
            for _ in range(accum_steps)
        ]
        if loss_normalization == SFT_LOSS_NORMALIZATION_UPDATE_TOKENS:
            microbatch_loss_tokens = []
            for selection in selections:
                lm_pool = (
                    decay_lm_rows if selection.pool == "decay" else lm_rows
                )
                selected_entries = [lm_pool[index] for index in selection.lm_indices]
                loss_tokens = sum(
                    shifted_token_counts(row)[1] for row, _ in selected_entries
                )
                if loss_tokens < 1:
                    raise RuntimeError(
                        "SFT selected a microbatch without assistant loss tokens"
                    )
                microbatch_loss_tokens.append(loss_tokens)
            update_loss_tokens = sum(microbatch_loss_tokens)
            loss_weights = [
                loss_tokens / update_loss_tokens
                for loss_tokens in microbatch_loss_tokens
            ]
        else:
            loss_weights = [1.0 / accum_steps] * accum_steps
        for selection, loss_weight in zip(
            selections,
            loss_weights,
            strict=True,
        ):
            with autocast_ctx(device_obj, amp_dtype):
                (
                    unscaled_loss,
                    lm_loss,
                    router_aux,
                    router_weighted,
                ) = _micro_loss(selection)
                loss = unscaled_loss * loss_weight
            scaler.scale(loss).backward()  # free this micro-batch's graph before the next forward
            unscaled_loss_value = unscaled_loss.item()
            report_weight = (
                loss_weight
                if loss_normalization == SFT_LOSS_NORMALIZATION_UPDATE_TOKENS
                else 1.0
            )
            step_lm_loss += float(lm_loss.detach()) * report_weight
            step_router_aux += float(router_aux.detach()) * report_weight
            step_router_weighted += float(router_weighted.detach()) * report_weight
            if loss_normalization == SFT_LOSS_NORMALIZATION_UPDATE_TOKENS:
                step_loss += unscaled_loss_value * loss_weight
            else:
                # Preserve the historical microbatch-loss sum for the default contract.
                step_loss += unscaled_loss_value
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        scaler.step(opt)
        scaler.update()
        hist.append(step_loss)
        lm_loss_history.append(step_lm_loss)
        router_aux_loss_history.append(step_router_aux)
        router_weighted_loss_history.append(step_router_weighted)
        if step % max(1, steps // 8) == 0 or step == steps - 1:
            message = f"  [sft] step {step:4d}/{steps}  loss {step_loss:.3f}"
            if model.cfg.sparse_ffn:
                message += (
                    f"  lm {step_lm_loss:.3f}  router_aux {step_router_aux:.3f}  "
                    f"router_weighted {step_router_weighted:.3f}"
                )
            log(message)
        if checkpoint_every and (step + 1) % checkpoint_every == 0:
            save(step)
    save(stop_step - 1)
    metrics = {
        "dataset_token_accounting": dataset_accounting,
        "token_accounting": training_accounting,
        "token_accounting_scope": "language_model_microbatches",
        "lm_sampling": training_contract["lm_sampling"],
        "fixed_horizon_progress": {
            "planned_optimizer_updates": steps,
            "completed_optimizer_updates": len(hist),
            "partial": len(hist) < steps,
        },
        "loss_components": {
            "optimization": hist,
            "lm_loss": lm_loss_history,
            "router_aux": router_aux_loss_history,
            "router_weighted": router_weighted_loss_history,
        },
    }
    if prepared.conversation_prompt_contract != LEGACY_CONVERSATION_PROMPT_CONTRACT:
        metrics["conversation_prompt_contract"] = prepared.conversation_prompt_contract
    if return_metrics:
        return hist, tool_head, ptr_head, metrics
    return hist, tool_head, ptr_head


def run(
    config_path: str,
    *,
    resume: bool | None = None,
    resume_git_receipt: str | Path | None = None,
    _receipt_creation: Mapping[str, Any] | None = None,
    _max_optimizer_updates: int | None = None,
) -> None | dict[str, Any]:
    """Run masked SFT and structured heads; ``runtime.resume`` continues ``log.out_dir/latest.pt``."""

    import json

    import yaml

    from localagent.agent.dense_selector import (
        BoundSelector,
        DenseToolSelector,
        train_dense_selector,
    )
    from localagent.agent.routes import RouteHead, train_route_head
    from localagent.agent.toolset import STANDARD_TOOLS
    from localagent.data.stratified_eval_selector import (
        ALGORITHM as STRATIFIED_EVAL_ALGORITHM,
    )
    from localagent.data.stratified_eval_selector import (
        select_stratified_eval_subset,
    )
    from localagent.eval.structured_context import evaluate_decisions
    from localagent.model import LocalAgentLM, ModelConfig
    from localagent.model.tokenizer import load_tokenizer
    from localagent.train.device import execution_metadata, resolve_device, resolve_dtype
    from localagent.train.stage_data import (
        build_stage_lineage,
        canonical_sha256,
        load_conversation_source,
        load_stage_parent_checkpoint,
        probe_decisions,
        sha256_file,
        single_turn_samples,
        tokenizer_identity,
    )

    config = yaml.safe_load(Path(config_path).read_text())
    if config.get("stage", "sft") != "sft":
        raise ValueError(f"expected stage='sft', got {config.get('stage')!r}")
    cfg = ModelConfig.from_yaml(config["model_config"])
    cfg.assert_within_budget()
    continuation = resolve_sft_continuation(config)

    data_cfg = config["data"]
    configured_sampling = data_cfg.get("sampling")
    if (
        isinstance(configured_sampling, Mapping)
        and configured_sampling.get("mode")
        == PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE
        and (continuation is None or "parent" not in continuation)
    ):
        raise ValueError(
            "parent-anchored format pulse sampling requires a sealed continuation.parent"
        )
    conversation_prompt_contract = resolve_conversation_prompt_contract(
        data_cfg.get("conversation_prompt_contract")
    )
    heads_cfg = config.get("heads", {})
    runtime_cfg = config.get("runtime", {})
    if not isinstance(runtime_cfg, Mapping):
        raise TypeError("runtime must be a mapping")
    function_masking_seed = runtime_cfg.get("seed", 0)
    strict_conversation_artifacts = data_cfg.get("strict_conversation_artifacts", False)
    if not isinstance(strict_conversation_artifacts, bool):
        raise TypeError("data.strict_conversation_artifacts must be boolean")

    def source_specs(value, *, label: str) -> list:
        if isinstance(value, (str, Path, Mapping)):
            return [value]
        if not isinstance(value, list):
            raise TypeError(f"{label} must be a source or list of sources")
        return value

    conversation_specs = source_specs(
        data_cfg["conversations"],
        label="data.conversations",
    )
    loaded_conversation_sources = [
        load_conversation_source(
            source,
            require_verified=strict_conversation_artifacts,
            expected_split="train",
        )
        for source in conversation_specs
    ]
    conversation_paths = [source.path for source in loaded_conversation_sources]
    raw_conversations = [
        conversation
        for source in loaded_conversation_sources
        for conversation in source.conversations
    ]
    raw_conversation_source_paths = [
        str(source.path)
        for source in loaded_conversation_sources
        for _ in source.conversations
    ]
    conversations, masked_source_indices, function_masking_main_audit = augment_conversations(
        raw_conversations,
        data_cfg.get("function_masking", False),
        seed=function_masking_seed,
    )
    function_masking_enabled = bool(function_masking_main_audit["enabled"])
    if function_masking_enabled and any(
        bool(heads_cfg.get(key, default))
        for key, default in (
            ("joint_tool_pointer", True),
            ("train_route_head", True),
            ("train_dense_selector", True),
        )
    ):
        raise ValueError(
            "function_masking is an LM augmentation and requires all structured heads to be "
            "disabled so opaque aliases cannot be mislabelled as canonical tool classes"
        )
    conversation_sources = [
        raw_conversation_source_paths[index] for index in masked_source_indices
    ]
    decay_specs = source_specs(
        data_cfg.get("decay_conversations", []),
        label="data.decay_conversations",
    )
    loaded_decay_sources = [
        load_conversation_source(
            source,
            require_verified=strict_conversation_artifacts,
            expected_split="train",
        )
        for source in decay_specs
    ]
    decay_paths = [source.path for source in loaded_decay_sources]
    decay_conversations_by_path = [
        (source.path, list(source.conversations)) for source in loaded_decay_sources
    ]
    raw_decay_conversations = [
        conversation for _, rows in decay_conversations_by_path for conversation in rows
    ]
    raw_decay_source_paths = [
        f"decay:{path}"
        for path, rows in decay_conversations_by_path
        for _ in rows
    ]
    decay_conversations, masked_decay_source_indices, function_masking_decay_audit = (
        augment_conversations(
            raw_decay_conversations,
            data_cfg.get("function_masking", False),
            seed=function_masking_seed,
        )
    )
    decay_conversation_sources = [
        raw_decay_source_paths[index] for index in masked_decay_source_indices
    ]
    function_masking_audit = {
        "enabled": function_masking_enabled
        or bool(function_masking_decay_audit["enabled"]),
        "main": function_masking_main_audit,
        "decay": function_masking_decay_audit,
    }
    eval_conversation_specs = source_specs(
        data_cfg.get("eval_conversations", []),
        label="data.eval_conversations",
    )
    loaded_eval_sources = [
        load_conversation_source(
            source,
            require_verified=strict_conversation_artifacts,
            expected_split="eval",
        )
        for source in eval_conversation_specs
    ]
    eval_conversation_paths = [source.path for source in loaded_eval_sources]
    eval_conversations = [
        conversation for source in loaded_eval_sources for conversation in source.conversations
    ]
    full_eval_conversation_rows = len(eval_conversations)
    conversation_overlap_audit = assert_no_conversation_overlap(
        [*conversations, *decay_conversations],
        eval_conversations,
        left_label="SFT main/decay training content",
        right_label="held-out",
        conversation_prompt_contract=conversation_prompt_contract,
    )
    evaluation_cfg = config.get("evaluation", {})
    if not isinstance(evaluation_cfg, Mapping):
        raise TypeError("evaluation must be a mapping")
    eval_selection_audit = None
    max_eval_conversations = evaluation_cfg.get("max_conversations")
    eval_selection_mode = evaluation_cfg.get("selection")
    if max_eval_conversations is None:
        if eval_selection_mode is not None:
            raise ValueError("evaluation.selection requires evaluation.max_conversations")
    else:
        if eval_selection_mode != STRATIFIED_EVAL_ALGORITHM:
            raise ValueError(
                "evaluation.selection must be "
                f"{STRATIFIED_EVAL_ALGORITHM!r} when max_conversations is configured"
            )
        selection = select_stratified_eval_subset(
            eval_conversations,
            max_rows=max_eval_conversations,
        )
        eval_conversations = list(selection.conversations)
        eval_selection_audit = selection.audit.as_dict()
    eval_content_rows = [
        conversation_semantic_sha256(conversation) for conversation in eval_conversations
    ]
    samples = []
    sample_sources = []
    multi_turn_conversations = []
    multi_turn_sources = []
    for conversation, source in zip(conversations, conversation_sources, strict=True):
        projected = single_turn_samples([conversation])
        if projected:
            samples.extend(projected)
            sample_sources.extend([source] * len(projected))
        else:
            multi_turn_conversations.append(conversation)
            multi_turn_sources.append(source)
    decision_samples = probe_decisions(conversations)
    joint_heads = bool(heads_cfg.get("joint_tool_pointer", True))
    multi_turn_batch_size = validate_multi_turn_batch_size(
        heads_cfg.get("multi_turn_batch_size", 12)
    )
    train_route = bool(heads_cfg.get("train_route_head", True))
    train_dense = bool(heads_cfg.get("train_dense_selector", True))
    if joint_heads and not samples:
        raise ValueError("joint tool/pointer heads need simple user -> assistant conversations")
    if train_route and not decision_samples:
        raise ValueError("route head needs at least one assistant decision")
    standard_tool_names = {tool.name for tool in STANDARD_TOOLS}
    if train_dense and not any(
        decision.kind == "tool" and decision.ref_name in standard_tool_names
        for decision in decision_samples
    ):
        raise ValueError("dense selector needs at least one tool decision in the standard tool set")
    tok_cfg = data_cfg.get("tokenizer", {"kind": "byte"})
    tokenizer_kind = str(tok_cfg.get("kind", "byte"))
    tokenizer_path = tok_cfg.get("path")
    tokenizer = load_tokenizer(tokenizer_kind, tokenizer_path)
    if tokenizer.vocab_size != cfg.vocab_size:
        raise ValueError("tokenizer vocabulary does not match model config")
    assert_prompt_contract_tokenizer(tokenizer, conversation_prompt_contract)
    tokenizer_lineage = tokenizer_identity(
        tokenizer_kind,
        vocab_size=tokenizer.vocab_size,
        path=tokenizer_path,
    )

    runtime = config.get("runtime", {})
    seed = int(runtime.get("seed", 0))
    requested_device = runtime.get("device", "auto")
    requested_dtype = runtime.get("dtype", "auto")
    device = resolve_device(requested_device)
    dtype = resolve_dtype(device, requested_dtype)
    execution = execution_metadata(
        requested_device=requested_device,
        resolved_device=device,
        requested_dtype=requested_dtype,
        resolved_dtype=dtype,
    )
    init_from = Path(config["init_from"])
    # The shared strict loader already defines the exact SFT -> RL boundary.  Reusing that
    # validation for an explicitly declared SFT child preserves the same one-read byte binding,
    # architecture checks, and tokenizer checks without weakening the default midtrain -> SFT
    # boundary.
    parent_loader_stage = "rl" if continuation is not None else "sft"
    checkpoint, parent_checkpoint_sha256 = load_stage_parent_checkpoint(
        init_from,
        stage=parent_loader_stage,
        requested_model_config=cfg,
        expected_tokenizer_sha256=str(tokenizer_lineage["sha256"]),
    )
    if continuation is not None:
        if "parent" in continuation:
            _validate_sft_continuation_parent(
                checkpoint,
                checkpoint_sha256=parent_checkpoint_sha256,
                continuation=continuation,
            )
        else:
            _validated_completed_sft_parent(checkpoint)
        checkpoint = dict(checkpoint)
    state = checkpoint.get("state_dict", checkpoint.get("model"))
    if state is None:
        raise ValueError("init_from checkpoint has no state_dict/model")
    torch.manual_seed(seed)
    model = LocalAgentLM(cfg)
    model.load_state_dict(state)

    schedule = config.get("schedule", {})
    batch_cfg = config.get("batch", {})
    optim = config.get("optim", {})
    if "freeze_parameters" in optim and not isinstance(
        optim["freeze_parameters"],
        list,
    ):
        raise TypeError("optim.freeze_parameters must be a list of exact model parameter names")
    log_cfg = config.get("log", {})
    sampling_cfg = data_cfg.get("sampling")
    decision_ordering = None
    decision_sampling_contract = None
    if sampling_cfg is not None:
        if not isinstance(sampling_cfg, Mapping):
            raise TypeError("data.sampling must be a mapping")
        sampling_mode = sampling_cfg.get("mode")
        if sampling_mode not in {
            QUOTA_SAMPLING_MODE,
            MIXED_REPLAY_SAMPLING_MODE,
            PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE,
        }:
            raise ValueError(
                "data.sampling.mode must be one of "
                f"{QUOTA_SAMPLING_MODE!r}, {MIXED_REPLAY_SAMPLING_MODE!r}, "
                f"{PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE!r}"
            )
        if conversation_prompt_contract == LEGACY_CONVERSATION_PROMPT_CONTRACT:
            raise ValueError("decision sampling requires openai_full_catalog_v1")
        if bool(data_cfg.get("shuffle", True)):
            raise ValueError("decision sampling requires data.shuffle=false")
        if decay_paths:
            raise ValueError("decision sampling does not support decay_conversations")
        selected_decisions = (
            int(schedule.get("total_steps", 3_000))
            * int(batch_cfg.get("micro_batch_size", 8))
            * int(batch_cfg.get("grad_accum_steps", 1))
        )
        if sampling_mode == QUOTA_SAMPLING_MODE:
            decision_ordering = order_assistant_decisions(conversations)
            ordered_decision_keys, decision_sampling_contract = quota_sampling_window(
                decision_ordering,
                selected_decisions=selected_decisions,
                start_decision=sampling_cfg.get("start_decision", 0),
            )
        elif sampling_mode == MIXED_REPLAY_SAMPLING_MODE:
            ordered_decision_keys, decision_sampling_contract = (
                mixed_replay_sampling_window(
                    [source.conversations for source in loaded_conversation_sources],
                    selected_decisions=selected_decisions,
                    sampling_config=sampling_cfg,
                )
            )
            effective_batch = int(batch_cfg.get("micro_batch_size", 8)) * int(
                batch_cfg.get("grad_accum_steps", 1)
            )
            if (
                decision_sampling_contract["cycle"]["length"]
                != effective_batch
            ):
                raise ValueError(
                    "mixed replay cycle must equal one complete optimizer update: "
                    f"cycle={decision_sampling_contract['cycle']['length']}, "
                    f"effective_batch={effective_batch}"
                )
        else:
            parent_checkpoint_binding = _validate_parent_anchored_sampling_parent(
                checkpoint,
                sampling_cfg,
            )
            ordered_decision_keys, decision_sampling_contract = (
                parent_anchored_format_pulse_sampling_window(
                    [source.conversations for source in loaded_conversation_sources],
                    selected_decisions=selected_decisions,
                    sampling_config=sampling_cfg,
                )
            )
            effective_batch = int(batch_cfg.get("micro_batch_size", 8)) * int(
                batch_cfg.get("grad_accum_steps", 1)
            )
            update_decisions = decision_sampling_contract["update_layout"][
                "update_decisions"
            ]
            if update_decisions != effective_batch:
                raise ValueError(
                    "parent-anchored format pulse update must equal one complete "
                    "optimizer update: "
                    f"update_decisions={update_decisions}, "
                    f"effective_batch={effective_batch}"
                )
            decision_sampling_contract = dict(decision_sampling_contract)
            decision_sampling_contract["parent_checkpoint_binding"] = (
                parent_checkpoint_binding
            )
    out_dir = Path(log_cfg.get("out_dir", "runs/sft"))
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "latest.pt"
    configured_resume = runtime.get("resume", False)
    if not isinstance(configured_resume, bool):
        raise TypeError("runtime.resume must be boolean")
    resume_requested = configured_resume if resume is None else resume
    if not isinstance(resume_requested, bool):
        raise TypeError("resume override must be boolean or None")
    if resume_git_receipt is not None and not resume_requested:
        raise ValueError("resume_git_receipt is valid only when exact resume is requested")
    if _receipt_creation is not None and not resume_requested:
        raise ValueError("resume Git receipt creation requires exact resume")
    if resume_git_receipt is not None and _receipt_creation is not None:
        raise ValueError("cannot consume and create a resume Git receipt in one invocation")
    if resume_requested and not checkpoint_path.exists():
        raise FileNotFoundError(
            f"SFT resume requested but checkpoint does not exist: {checkpoint_path}"
        )
    seq_limit = min(int(data_cfg.get("seq_len", cfg.max_seq_len)), cfg.max_seq_len)
    pad_to_input_tokens = validate_pad_to_input_tokens(
        batch_cfg.get("pad_to_input_tokens"),
        label="batch.pad_to_input_tokens",
    )
    if pad_to_input_tokens is not None and pad_to_input_tokens > seq_limit:
        raise ValueError(
            "batch.pad_to_input_tokens cannot exceed the SFT sequence limit: "
            f"configured={pad_to_input_tokens}, limit={seq_limit}"
        )
    evaluation_pad_to_input_tokens = validate_pad_to_input_tokens(
        evaluation_cfg.get("pad_to_input_tokens"),
        label="evaluation.pad_to_input_tokens",
    )
    if (
        evaluation_pad_to_input_tokens is not None
        and evaluation_pad_to_input_tokens > seq_limit
    ):
        raise ValueError(
            "evaluation.pad_to_input_tokens cannot exceed the SFT sequence limit: "
            f"configured={evaluation_pad_to_input_tokens}, limit={seq_limit}"
        )
    heldout_contract = (
        {
            "kind": "deterministic_teacher_forced_assistant_tokens",
            "row_order": (
                "configured_jsonl_order"
                if conversation_prompt_contract == LEGACY_CONVERSATION_PROMPT_CONTRACT
                else "configured_jsonl_assistant_decision_order"
            ),
            "same_rows_pre_post": True,
            "max_seq_len": seq_limit,
            **(
                {"pad_to_input_tokens": evaluation_pad_to_input_tokens}
                if evaluation_pad_to_input_tokens is not None
                else {}
            ),
            "dataset_sha256": canonical_sha256(sorted(eval_content_rows)),
            **(
                {"selection": eval_selection_audit}
                if eval_selection_audit is not None
                else {}
            ),
            **(
                {"conversation_prompt_contract": conversation_prompt_contract}
                if conversation_prompt_contract != LEGACY_CONVERSATION_PROMPT_CONTRACT
                else {}
            ),
        }
        if eval_conversations
        else None
    )

    decay_samples = None
    decay_sample_sources = None
    if decay_paths:
        if conversation_prompt_contract == LEGACY_CONVERSATION_PROMPT_CONTRACT:
            decay_samples = []
            decay_sample_sources = []
            for conversation, source_name in zip(
                decay_conversations,
                decay_conversation_sources,
                strict=True,
            ):
                projected = single_turn_samples([conversation])
                decay_samples.extend(projected)
                decay_sample_sources.extend([source_name] * len(projected))
            if not decay_samples:
                raise ValueError("decay_conversations has no simple user -> assistant rows")

    data_metadata = {
        "conversation_rows": len(conversations),
        "single_turn_rows": len(samples),
        "probe_decision_rows": len(decision_samples),
        "paths": [str(path) for path in conversation_paths],
        "conversation_overlap_audit": conversation_overlap_audit.as_dict(),
        "function_masking": function_masking_audit,
    }
    if conversation_prompt_contract != LEGACY_CONVERSATION_PROMPT_CONTRACT:
        data_metadata["conversation_prompt_contract"] = conversation_prompt_contract
    if decision_sampling_contract is not None:
        data_metadata["decision_sampling"] = decision_sampling_contract
    if decay_paths:
        data_metadata.update(
            {
                "decay_conversation_rows": len(decay_conversations),
                "decay_paths": [str(path) for path in decay_paths],
            }
        )
    if eval_conversations:
        data_metadata.update(
            {
                "eval_conversation_rows": len(eval_conversations),
                "eval_source_conversation_rows": full_eval_conversation_rows,
                "eval_paths": [str(path) for path in eval_conversation_paths],
                "heldout_content_overlap": 0,
                "heldout_rendered_prompt_overlap": 0,
                **(
                    {"eval_selection": eval_selection_audit}
                    if eval_selection_audit is not None
                    else {}
                ),
            }
        )
    tokenizer_metadata = {
        "kind": tokenizer_kind,
        "path": tokenizer_path,
        "sha256": tokenizer_lineage["sha256"],
    }
    data_identity = {
        "conversations": [dict(source.identity) for source in loaded_conversation_sources],
        "eval_conversations": [dict(source.identity) for source in loaded_eval_sources],
        "decay_conversations": (
            [dict(source.identity) for source in loaded_decay_sources] if decay_paths else []
        ),
        "conversation_overlap_audit": conversation_overlap_audit.as_dict(),
        "function_masking": function_masking_audit,
        **(
            {"eval_selection": eval_selection_audit}
            if eval_selection_audit is not None
            else {}
        ),
    }
    if conversation_prompt_contract != LEGACY_CONVERSATION_PROMPT_CONTRACT:
        data_identity["conversation_prompt_contract"] = conversation_prompt_contract
    if decision_sampling_contract is not None:
        data_identity["decision_sampling"] = decision_sampling_contract
    lineage = build_stage_lineage(
        stage="sft",
        config=config,
        model_config=cfg.__dict__,
        data_identity=data_identity,
        tokenizer=tokenizer_lineage,
        workspace=Path(__file__).resolve(),
        parent_checkpoint_sha256=parent_checkpoint_sha256,
    )

    resume_checkpoint = None
    resume_checkpoint_sha256 = None
    validated_resume_git_receipt = None
    if resume_requested:
        from localagent.train.resume_git_receipt import (
            assert_resume_git_receipt,
            load_resume_git_receipt,
            write_resume_git_receipt,
        )

        resume_checkpoint_sha256 = sha256_file(checkpoint_path)
        resume_checkpoint = _load_validated_sft_resume_checkpoint(checkpoint_path)
        recorded_lineage = resume_checkpoint.get("lineage")
        if not isinstance(recorded_lineage, Mapping):
            raise TypeError("resume checkpoint has no lineage metadata; refusing unsafe resume")
        if _receipt_creation is not None:
            output = _receipt_creation.get("path")
            if not isinstance(output, (str, Path)):
                raise TypeError("resume Git receipt output path is required")
            reason = _receipt_creation.get("reason")
            evidence = _receipt_creation.get("evidence")
            receipt = write_resume_git_receipt(
                output,
                checkpoint_sha256=resume_checkpoint_sha256,
                recorded_lineage=recorded_lineage,
                expected_lineage=lineage,
                stage="sft",
                reason=reason,
                evidence=evidence,
            )
            print(
                json.dumps(
                    {
                        "receipt": str(output),
                        "receipt_self_sha256": receipt["receipt_self_sha256"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return receipt
        if resume_git_receipt is not None:
            validated_resume_git_receipt = load_resume_git_receipt(resume_git_receipt)
            assert_resume_git_receipt(
                validated_resume_git_receipt,
                checkpoint_sha256=resume_checkpoint_sha256,
                recorded_lineage=recorded_lineage,
                expected_lineage=lineage,
                stage="sft",
            )
        heldout_baseline = _validated_resume_heldout_baseline(
            resume_checkpoint,
            expected_contract=heldout_contract,
        )
    elif heldout_contract is not None:
        heldout_baseline = {
            "contract": heldout_contract,
            "pre": _evaluate_conversations(
                model,
                eval_conversations,
                tokenizer,
                max_seq_len=seq_limit,
                batch_size=int(evaluation_cfg.get("batch_size", 8)),
                device=device,
                amp_dtype=dtype,
                conversation_prompt_contract=conversation_prompt_contract,
                pad_to_input_tokens=evaluation_pad_to_input_tokens,
            ),
        }
    else:
        heldout_baseline = None
    heldout_eval = (
        {
            "contract": heldout_baseline["contract"],
            "pre": heldout_baseline["pre"],
            "post": None,
            "delta": None,
        }
        if heldout_baseline is not None
        else None
    )

    _mps_synchronize_and_empty_cache(device)
    loss_history, tool_head, ptr_head, training_metrics = sft(
        model,
        samples,
        tokenizer,
        steps=int(schedule.get("total_steps", 3_000)),
        batch_size=int(batch_cfg.get("micro_batch_size", 8)),
        accum_steps=int(batch_cfg.get("grad_accum_steps", 1)),
        lr=float(optim.get("lr", 1e-4)),
        optimizer_name=optim.get("name", "adamw"),
        weight_decay=optim.get("weight_decay", 0.0),
        grad_clip=optim.get("grad_clip", 1.0),
        warmup=int(schedule.get("warmup_steps", 50)),
        device=device,
        joint_tool_head=joint_heads,
        aux_weight=float(heads_cfg.get("tool_loss_weight", 1.0)),
        ptr_weight=float(heads_cfg.get("pointer_loss_weight", 0.15)),
        mt_weight=float(heads_cfg.get("multi_turn_head_weight", 1.0)),
        multi_turn_batch_size=multi_turn_batch_size,
        conversations=(
            multi_turn_conversations
            if conversation_prompt_contract == LEGACY_CONVERSATION_PROMPT_CONTRACT
            else conversations
        ),
        sample_sources=sample_sources,
        conversation_sources=(
            multi_turn_sources
            if conversation_prompt_contract == LEGACY_CONVERSATION_PROMPT_CONTRACT
            else conversation_sources
        ),
        lr_schedule=str(schedule.get("type", "cosine")),
        decay_frac=float(schedule.get("decay_frac", 0.2)),
        decay_samples=decay_samples,
        decay_sample_sources=decay_sample_sources,
        decay_conversations=(
            decay_conversations
            if conversation_prompt_contract != LEGACY_CONVERSATION_PROMPT_CONTRACT and decay_paths
            else None
        ),
        decay_conversation_sources=decay_conversation_sources,
        conversation_prompt_contract=conversation_prompt_contract,
        shuffle=bool(data_cfg.get("shuffle", True)),
        init_tool_head=checkpoint.get("tool_head"),
        init_ptr_head=checkpoint.get("ptr_head"),
        max_seq_len=seq_limit,
        pad_to_input_tokens=pad_to_input_tokens,
        seed=seed,
        amp_dtype=dtype,
        return_metrics=True,
        checkpoint_path=checkpoint_path,
        checkpoint_every=int(log_cfg.get("ckpt_every", 0)),
        archive_checkpoints=log_cfg.get("archive_checkpoints", False),
        resume_from=resume_checkpoint,
        lineage=lineage,
        tokenizer_metadata=tokenizer_metadata,
        data_metadata=data_metadata,
        execution=execution,
        heldout_baseline=heldout_baseline,
        lm_order_keys=(
            ordered_decision_keys
            if decision_sampling_contract is not None
            else None
        ),
        sampling_contract=decision_sampling_contract,
        loss_normalization=optim.get(
            "loss_normalization",
            SFT_LOSS_NORMALIZATION_MICROBATCH,
        ),
        freeze_parameters=(
            optim["freeze_parameters"]
            if "freeze_parameters" in optim
            else None
        ),
        resume_git_receipt=validated_resume_git_receipt,
        resume_checkpoint_sha256=(
            resume_checkpoint_sha256
            if validated_resume_git_receipt is not None
            else None
        ),
        _max_optimizer_updates=_max_optimizer_updates,
    )
    if eval_conversations:
        _clear_mps_gradients_and_cache(device, model, tool_head, ptr_head)
    training_checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if eval_conversations:
        heldout_post = _evaluate_conversations(
            model,
            eval_conversations,
            tokenizer,
            max_seq_len=seq_limit,
            batch_size=int(evaluation_cfg.get("batch_size", 8)),
            device=device,
            amp_dtype=dtype,
            conversation_prompt_contract=conversation_prompt_contract,
            pad_to_input_tokens=evaluation_pad_to_input_tokens,
        )
        heldout_eval["post"] = heldout_post
        heldout_eval["delta"] = {
            "mean_loss": heldout_post["mean_loss"] - heldout_eval["pre"]["mean_loss"],
            "assistant_token_accuracy": (
                heldout_post["assistant_token_accuracy"]
                - heldout_eval["pre"]["assistant_token_accuracy"]
            ),
            "assistant_sequence_accuracy": (
                heldout_post["assistant_sequence_accuracy"]
                - heldout_eval["pre"]["assistant_sequence_accuracy"]
            ),
        }

    rebuild_examples = bool(heads_cfg.get("example_centroids", True))
    examples: dict[str, list[str]] = (
        {} if rebuild_examples else dict(checkpoint.get("examples") or {})
    )
    if rebuild_examples:
        max_examples = int(heads_cfg.get("examples_per_tool", 16))
        for sample in samples:
            if sample.kind != "tool":
                continue
            bucket = examples.setdefault(sample.ref_name, [])
            if len(bucket) < max_examples:
                bucket.append(sample.prompt)

    selector_proj = int(heads_cfg.get("selector_proj", checkpoint.get("selector_proj") or 256))
    route_head = dense_selector = None
    if train_route:
        torch.manual_seed(seed + 1)
        route_head = train_route_head(
            model,
            decision_samples,
            tokenizer,
            steps=int(heads_cfg.get("route_steps", 300)),
            batch_size=int(heads_cfg.get("probe_batch_size", 64)),
            lr=float(heads_cfg.get("probe_lr", 5e-3)),
            device=device,
            log=print,
        )
    if train_dense:
        torch.manual_seed(seed + 2)
        dense_selector = train_dense_selector(
            model,
            decision_samples,
            tokenizer,
            STANDARD_TOOLS,
            steps=int(heads_cfg.get("selector_steps", 400)),
            batch_size=int(heads_cfg.get("probe_batch_size", 64)),
            lr=float(heads_cfg.get("probe_lr", 5e-3)),
            proj=selector_proj,
            device=device,
            examples=examples or None,
            log=print,
        )

    # A disabled head means "do not retrain it in this stage", not "delete it".  Carry prior
    # structured heads through so an LM-only SFT pass remains a valid input to export or RL.
    tool_head_state = (
        tool_head.state_dict() if tool_head is not None else checkpoint.get("tool_head")
    )
    ptr_head_state = ptr_head.state_dict() if ptr_head is not None else checkpoint.get("ptr_head")
    route_head_state = (
        route_head.state_dict() if route_head is not None else checkpoint.get("route_head")
    )
    dense_selector_state = (
        dense_selector.state_dict()
        if dense_selector is not None
        else checkpoint.get("dense_selector")
    )
    heldout_structured_eval = None
    if eval_conversations and route_head_state is not None and dense_selector_state is not None:
        selector_emb_dim, selector_state_proj = _dense_selector_dimensions(
            dense_selector_state,
            expected_d_model=cfg.d_model,
            configured_proj=selector_proj,
        )
        eval_route_head = RouteHead(cfg.d_model).to(device)
        eval_route_head.load_state_dict(route_head_state)
        eval_route_head.eval()
        eval_selector_model = DenseToolSelector(
            cfg.d_model,
            emb_dim=selector_emb_dim,
            proj=selector_state_proj,
        ).to(device)
        eval_selector_model.load_state_dict(dense_selector_state)
        eval_selector = BoundSelector(
            eval_selector_model,
            STANDARD_TOOLS,
            device=device,
            examples=examples or None,
        )
        eval_decisions = probe_decisions(eval_conversations)
        structured_batch_size = int(evaluation_cfg.get("batch_size", 8))
        natural_condition = evaluate_decisions(
            model=model,
            tokenizer=tokenizer,
            route_head=eval_route_head,
            selector=eval_selector,
            decisions=eval_decisions,
            target_input_tokens=None,
            batch_size=structured_batch_size,
            device=device,
            include_records=True,
        )
        fixed_compute_tokens = min(512, cfg.max_seq_len)
        trailing_condition = evaluate_decisions(
            model=model,
            tokenizer=tokenizer,
            route_head=eval_route_head,
            selector=eval_selector,
            decisions=eval_decisions,
            target_input_tokens=fixed_compute_tokens,
            batch_size=structured_batch_size,
            device=device,
            materialization="trailing_compute",
            include_records=True,
        )
        natural_records = natural_condition.pop("records")
        trailing_records = trailing_condition.pop("records")
        heldout_structured_eval = {
            "contract": {
                "kind": "frozen_route_and_dense_selector",
                "split": "explicit_disjoint_eval_conversations",
                "row_order": "configured_jsonl_assistant_decision_order",
                "dataset_sha256": canonical_sha256(sorted(eval_content_rows)),
                "decision_rows": len(eval_decisions),
                "candidate_tools": len(STANDARD_TOOLS),
                "selector_embedding_dim": selector_emb_dim,
                "selector_projection_dim": selector_state_proj,
                "fixed_compute_tokens": fixed_compute_tokens,
            },
            "conditions": [natural_condition, trailing_condition],
            "prediction_invariance": _structured_prediction_invariance(
                natural_records,
                trailing_records,
                reference_condition=str(natural_condition["condition"]),
                comparison_condition=str(trailing_condition["condition"]),
            ),
        }
    payload = {
        **training_checkpoint,
        "cfg": cfg.__dict__,
        "state_dict": model.state_dict(),
        "tool_head": tool_head_state,
        "ptr_head": ptr_head_state,
        "route_head": route_head_state,
        "dense_selector": dense_selector_state,
        "selector_proj": selector_proj,
        "examples": examples,
        "tokenizer": tokenizer_metadata,
        "stage": "sft",
        "step": training_checkpoint["step"],
        "loss_history": loss_history,
        "lineage": lineage,
        "dataset_token_accounting": training_metrics["dataset_token_accounting"],
        "token_accounting": training_metrics["token_accounting"],
        "token_accounting_scope": training_metrics["token_accounting_scope"],
        "fixed_horizon_progress": training_metrics["fixed_horizon_progress"],
        "data": data_metadata,
        "heldout_eval": heldout_eval,
        "heldout_structured_eval": heldout_structured_eval,
        "execution": execution,
        **({"continuation": continuation} if continuation is not None else {}),
    }
    if conversation_prompt_contract != LEGACY_CONVERSATION_PROMPT_CONTRACT:
        payload["conversation_prompt_contract"] = conversation_prompt_contract
    tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(checkpoint_path)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "stage": "sft",
                "checkpoint": str(checkpoint_path),
                "conversation_rows": len(conversations),
                "single_turn_rows": len(samples),
                "probe_decision_rows": len(decision_samples),
                "loss_last": loss_history[-1] if loss_history else None,
                "loss_steps": len(loss_history),
                **training_metrics,
                "lineage": lineage,
                "data": data_metadata,
                "heldout_eval": heldout_eval,
                "heldout_structured_eval": heldout_structured_eval,
                "execution": execution,
                **({"continuation": continuation} if continuation is not None else {}),
                "structured_heads": {
                    "tool_pointer": tool_head_state is not None and ptr_head_state is not None,
                    "route": route_head_state is not None,
                    "dense_selector": dense_selector_state is not None,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"checkpoint": str(checkpoint_path), "metrics": str(metrics_path)}, indent=2))


def create_resume_git_receipt(
    config_path: str,
    output_path: str | Path,
    *,
    reason: str,
    evidence: Sequence[str],
) -> dict[str, Any]:
    """Create one external Git-only lineage receipt for the configured SFT checkpoint."""

    result = run(
        config_path,
        resume=True,
        _receipt_creation={
            "path": output_path,
            "reason": reason,
            "evidence": list(evidence),
        },
    )
    if not isinstance(result, dict):
        raise RuntimeError("SFT resume Git receipt creation did not return a receipt")
    return result
