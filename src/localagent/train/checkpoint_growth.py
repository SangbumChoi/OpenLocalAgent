"""Audited depth growth for compatible LocalAgent checkpoints.

This module deliberately implements one narrow operation: copy every layer in a deeper target
from an explicitly named source layer while keeping all other model semantics identical. Reusing
a block changes how many residual transformations are applied, so this is a warm start and is
explicitly not a function-preserving model transformation.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from localagent.model import LocalAgentLM, ModelConfig
from localagent.train.stage_data import canonical_sha256

GROWTH_FORMAT = "localagent.checkpoint_growth"
GROWTH_SCHEMA_VERSION = 1
_ALLOWED_CONFIG_DIFFERENCES = frozenset({"name", "n_layers", "layer_types"})
_BASE_CHECKPOINT_STAGES = frozenset({None, "pretrain", "midtrain", "checkpoint_growth"})
_STRUCTURED_AUXILIARY_FIELDS = (
    "tool_head",
    "ptr_head",
    "route_head",
    "dense_selector",
    "selector_proj",
    "structured_heads_available",
    "invalidated_structured_heads",
)


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def parse_layer_map(value: str) -> dict[int, int]:
    """Parse ``TARGET:SOURCE`` comma pairs into an explicit layer map."""

    if not value.strip():
        raise ValueError("layer map must not be empty")
    parsed: dict[int, int] = {}
    for entry in value.split(","):
        pair = entry.strip().split(":")
        if len(pair) != 2:
            raise ValueError(
                "layer map entries must use TARGET:SOURCE syntax, for example 0:0,1:0,2:1"
            )
        try:
            target_layer, source_layer = (int(item.strip()) for item in pair)
        except ValueError as error:
            raise ValueError("layer map indices must be base-10 integers") from error
        if target_layer < 0 or source_layer < 0:
            raise ValueError("layer map indices must be non-negative")
        if target_layer in parsed:
            raise ValueError(f"target layer {target_layer} appears more than once")
        parsed[target_layer] = source_layer
    return parsed


def load_checkpoint_with_sha256(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load exactly the checkpoint bytes identified by the returned SHA-256."""

    artifact = Path(path)
    payload = artifact.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    del payload
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint root must be a mapping")
    return dict(checkpoint), digest


def checkpoint_tokenizer_sha256(checkpoint: Mapping[str, Any]) -> str:
    """Return one proven tokenizer identity, rejecting missing or conflicting metadata."""

    identities: list[Any] = []
    tokenizer = checkpoint.get("tokenizer")
    if isinstance(tokenizer, Mapping) and tokenizer.get("sha256") is not None:
        identities.append(tokenizer["sha256"])
    lineage = checkpoint.get("lineage")
    if isinstance(lineage, Mapping) and lineage.get("tokenizer_sha256") is not None:
        identities.append(lineage["tokenizer_sha256"])
    growth = checkpoint.get("growth")
    if isinstance(growth, Mapping) and growth.get("tokenizer_sha256") is not None:
        identities.append(growth["tokenizer_sha256"])
    if not identities:
        raise ValueError(
            "checkpoint has no content-bound tokenizer identity; refusing an unproven warm start"
        )
    if any(not isinstance(identity, str) for identity in identities):
        raise ValueError("checkpoint tokenizer sha256 identities must be strings")
    normalized = set(identities)
    if len(normalized) != 1:
        raise ValueError("checkpoint records conflicting tokenizer identities")
    identity = normalized.pop()
    if len(identity) != 64 or any(character not in "0123456789abcdef" for character in identity):
        raise ValueError("checkpoint tokenizer sha256 must be 64 lowercase hexadecimal characters")
    return identity


def _checkpoint_model_config(checkpoint: Mapping[str, Any]) -> ModelConfig:
    raw = checkpoint.get("cfg")
    if raw is None:
        raise ValueError("source checkpoint has no model cfg")
    if not isinstance(raw, Mapping):
        raw = getattr(raw, "__dict__", None)
    if not isinstance(raw, Mapping):
        raise ValueError("source checkpoint model cfg must be a mapping or dataclass")
    required = set(ModelConfig.__dataclass_fields__)
    actual = set(raw)
    missing = sorted(required - actual)
    unexpected = sorted(actual - required)
    if missing or unexpected:
        raise ValueError(
            "source checkpoint model cfg must use the complete current schema: "
            f"missing={missing}, unexpected={unexpected}"
        )
    cfg = ModelConfig(**dict(raw))
    cfg.assert_within_budget()
    lineage = checkpoint.get("lineage")
    if lineage is not None:
        if not isinstance(lineage, Mapping):
            raise ValueError("source checkpoint lineage must be a mapping or null")
        recorded_config_sha256 = lineage.get("model_config_sha256")
        if not _valid_sha256(recorded_config_sha256):
            raise ValueError(
                "source checkpoint lineage has no valid model-config identity"
            )
        if recorded_config_sha256 != canonical_sha256(cfg.__dict__):
            raise ValueError(
                "source checkpoint lineage model-config identity mismatch"
            )
    return cfg


def _normalized_layer_map(
    layer_map: Mapping[int, int],
    *,
    source_layers: int,
    target_layers: int,
) -> dict[int, int]:
    normalized: dict[int, int] = {}
    for target_layer, source_layer in layer_map.items():
        if (
            isinstance(target_layer, bool)
            or not isinstance(target_layer, int)
            or isinstance(source_layer, bool)
            or not isinstance(source_layer, int)
        ):
            raise ValueError("layer map keys and values must be integers")
        normalized[target_layer] = source_layer
    expected_targets = set(range(target_layers))
    actual_targets = set(normalized)
    if actual_targets != expected_targets:
        missing = sorted(expected_targets - actual_targets)
        extra = sorted(actual_targets - expected_targets)
        raise ValueError(
            "layer map must name every target layer exactly once: "
            f"missing={missing}, out_of_range={extra}"
        )
    invalid_sources = sorted(
        {source_layer for source_layer in normalized.values() if source_layer not in range(source_layers)}
    )
    if invalid_sources:
        raise ValueError(f"layer map has out-of-range source layers: {invalid_sources}")
    return normalized


def assert_growth_compatible(
    source: ModelConfig,
    target: ModelConfig,
    layer_map: Mapping[int, int],
) -> dict[int, int]:
    """Validate the only supported growth contract and return its normalized layer map."""

    source.assert_within_budget()
    target.assert_within_budget()
    if target.n_layers <= source.n_layers:
        raise ValueError(
            "checkpoint growth requires a strictly deeper target: "
            f"source n_layers={source.n_layers}, target n_layers={target.n_layers}"
        )
    mismatches = [
        field
        for field in ModelConfig.__dataclass_fields__
        if field not in _ALLOWED_CONFIG_DIFFERENCES
        and getattr(source, field) != getattr(target, field)
    ]
    if mismatches:
        details = ", ".join(
            f"{field}={getattr(source, field)!r} -> {getattr(target, field)!r}"
            for field in mismatches
        )
        raise ValueError(f"growth model configs are incompatible: {details}")
    normalized = _normalized_layer_map(
        layer_map,
        source_layers=source.n_layers,
        target_layers=target.n_layers,
    )
    source_kinds = source.block_types()
    target_kinds = target.block_types()
    kind_mismatches = [
        (target_layer, target_kinds[target_layer], source_layer, source_kinds[source_layer])
        for target_layer, source_layer in sorted(normalized.items())
        if target_kinds[target_layer] != source_kinds[source_layer]
    ]
    if kind_mismatches:
        details = ", ".join(
            f"target {target_layer} ({target_kind}) -> "
            f"source {source_layer} ({source_kind})"
            for target_layer, target_kind, source_layer, source_kind in kind_mismatches
        )
        raise ValueError(f"mapped block kinds must match: {details}")
    return normalized


def _state_contract(cfg: ModelConfig) -> dict[str, tuple[int, ...]]:
    with torch.device("meta"):
        state = LocalAgentLM(cfg).state_dict()
    return {name: tuple(tensor.shape) for name, tensor in state.items()}


def _validate_state_dict(
    state_dict: Mapping[str, Any],
    cfg: ModelConfig,
    *,
    label: str,
) -> None:
    expected = _state_contract(cfg)
    actual_names = set(state_dict)
    expected_names = set(expected)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise ValueError(
            f"{label} state_dict keys do not match its model cfg: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name, expected_shape in expected.items():
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{label} state_dict value {name!r} is not a tensor")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{label} state_dict shape mismatch for {name}: "
                f"{tuple(tensor.shape)} != {expected_shape}"
            )


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash names, tensor contracts, and exact CPU bytes in stable key order."""

    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        header = json.dumps(
            {"dtype": str(value.dtype), "name": name, "shape": list(value.shape)},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        raw = value.view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _manifest_layer_map(value: Any) -> dict[int, int]:
    if not isinstance(value, list) or not value:
        raise ValueError("growth manifest layer map must be a non-empty list")
    layer_map: dict[int, int] = {}
    for index, record in enumerate(value):
        if not isinstance(record, Mapping) or set(record) != {
            "source_layer",
            "target_layer",
        }:
            raise ValueError(f"growth manifest layer-map record {index} is invalid")
        target_layer = record.get("target_layer")
        source_layer = record.get("source_layer")
        if (
            isinstance(target_layer, bool)
            or not isinstance(target_layer, int)
            or isinstance(source_layer, bool)
            or not isinstance(source_layer, int)
            or target_layer in layer_map
        ):
            raise ValueError(f"growth manifest layer-map record {index} is invalid")
        layer_map[target_layer] = source_layer
    return layer_map


def verify_growth_checkpoint(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Verify every content-bound contract carried by a grown checkpoint."""

    if checkpoint.get("stage") != "checkpoint_growth":
        raise ValueError("growth checkpoint has an invalid stage")
    manifest = checkpoint.get("growth")
    if not isinstance(manifest, Mapping):
        raise ValueError("growth checkpoint has no growth manifest")
    manifest_sha256 = manifest.get("manifest_sha256")
    unsigned_manifest = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if not _valid_sha256(manifest_sha256):
        raise ValueError("growth manifest has an invalid self-hash")
    if manifest_sha256 != canonical_sha256(unsigned_manifest):
        raise ValueError("growth manifest self-hash mismatch")
    if (
        manifest.get("format") != GROWTH_FORMAT
        or manifest.get("schema_version") != GROWTH_SCHEMA_VERSION
    ):
        raise ValueError("growth manifest format/version is unsupported")

    source_model_config = manifest.get("source_model_config")
    target_model_config = manifest.get("target_model_config")
    if not isinstance(source_model_config, Mapping) or not isinstance(
        target_model_config, Mapping
    ):
        raise ValueError("growth manifest model configs are invalid")
    source_cfg = _checkpoint_model_config({"cfg": source_model_config})
    target_cfg = _checkpoint_model_config({"cfg": target_model_config})
    if manifest.get("source_model_config_sha256") != canonical_sha256(
        source_cfg.__dict__
    ):
        raise ValueError("growth manifest source-model config hash mismatch")
    if manifest.get("target_model_config_sha256") != canonical_sha256(
        target_cfg.__dict__
    ):
        raise ValueError("growth manifest target-model config hash mismatch")
    top_level_cfg = checkpoint.get("cfg")
    if not isinstance(top_level_cfg, Mapping) or dict(top_level_cfg) != target_cfg.__dict__:
        raise ValueError("growth checkpoint top-level cfg differs from its manifest")

    layer_map = _manifest_layer_map(manifest.get("target_layer_to_source_layer"))
    normalized_map = assert_growth_compatible(source_cfg, target_cfg, layer_map)
    expected_records = [
        {"target_layer": target_layer, "source_layer": source_layer}
        for target_layer, source_layer in sorted(normalized_map.items())
    ]
    if manifest.get("target_layer_to_source_layer") != expected_records:
        raise ValueError("growth manifest layer map is not canonical")
    if manifest.get("allowed_model_config_differences") != sorted(
        _ALLOWED_CONFIG_DIFFERENCES
    ):
        raise ValueError("growth manifest allowed-config contract mismatch")
    if manifest.get("strict_model_config_fields") != sorted(
        set(ModelConfig.__dataclass_fields__) - _ALLOWED_CONFIG_DIFFERENCES
    ):
        raise ValueError("growth manifest strict-config contract mismatch")
    if (
        manifest.get("mapped_block_kinds_match") is not True
        or manifest.get("function_preserving") is not False
        or manifest.get("optimizer_state") != "discarded"
    ):
        raise ValueError("growth manifest transformation semantics are invalid")

    tokenizer_sha256 = checkpoint_tokenizer_sha256(checkpoint)
    if manifest.get("tokenizer_sha256") != tokenizer_sha256:
        raise ValueError("growth manifest tokenizer identity mismatch")
    raw_state = checkpoint.get("state_dict")
    if not isinstance(raw_state, Mapping):
        raise ValueError("growth checkpoint has no target state_dict")
    target_state = dict(raw_state)
    _validate_state_dict(target_state, target_cfg, label="growth target")
    if manifest.get("target_state_dict_sha256") != state_dict_sha256(target_state):
        raise ValueError("growth checkpoint target-state hash mismatch")
    for field in (
        "source_checkpoint_sha256",
        "source_state_dict_sha256",
        "target_state_dict_sha256",
    ):
        if not _valid_sha256(manifest.get(field)):
            raise ValueError(f"growth manifest {field} is invalid")
    discarded = manifest.get("discarded_payload_keys")
    if (
        not isinstance(discarded, list)
        or any(not isinstance(key, str) for key in discarded)
        or discarded != sorted(set(discarded))
    ):
        raise ValueError("growth manifest discarded-payload record is invalid")
    return dict(manifest)


def _source_key(target_key: str, layer_map: Mapping[int, int]) -> str:
    parts = target_key.split(".", 2)
    if len(parts) != 3 or parts[0] != "blocks":
        return target_key
    target_layer = int(parts[1])
    return f"blocks.{layer_map[target_layer]}.{parts[2]}"


def _grow_checkpoint_payload(
    source_checkpoint: Mapping[str, Any],
    target_cfg: ModelConfig,
    layer_map: Mapping[int, int],
    *,
    source_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Produce a target-shaped warm-start checkpoint with no optimizer/training state."""

    if not _valid_sha256(source_checkpoint_sha256):
        raise ValueError("source_checkpoint_sha256 must be 64 lowercase hexadecimal characters")
    source_stage = source_checkpoint.get("stage")
    if source_stage not in _BASE_CHECKPOINT_STAGES:
        raise ValueError(
            "checkpoint growth accepts only pretrain/midtrain backbone checkpoints; "
            f"got stage={source_stage!r}"
        )
    present_auxiliary = [
        field
        for field in _STRUCTURED_AUXILIARY_FIELDS
        if source_checkpoint.get(field) is not None
    ]
    if present_auxiliary:
        raise ValueError(
            "checkpoint growth refuses to silently discard structured auxiliary state: "
            + ", ".join(present_auxiliary)
        )
    if source_checkpoint.get("growth") is not None or source_stage == "checkpoint_growth":
        verify_growth_checkpoint(source_checkpoint)
    source_cfg = _checkpoint_model_config(source_checkpoint)
    target_cfg.assert_within_budget()
    normalized_map = assert_growth_compatible(source_cfg, target_cfg, layer_map)
    tokenizer_sha256 = checkpoint_tokenizer_sha256(source_checkpoint)
    raw_state = source_checkpoint.get("state_dict", source_checkpoint.get("model"))
    if not isinstance(raw_state, Mapping):
        raise ValueError("source checkpoint has no state_dict/model mapping")
    source_state = dict(raw_state)
    _validate_state_dict(source_state, source_cfg, label="source")

    target_contract = _state_contract(target_cfg)
    target_state: dict[str, torch.Tensor] = {}
    for target_key, expected_shape in target_contract.items():
        source_key = _source_key(target_key, normalized_map)
        source_tensor = source_state[source_key]
        if tuple(source_tensor.shape) != expected_shape:
            raise ValueError(
                f"mapped tensor shape mismatch for {target_key}: "
                f"source {source_key} has {tuple(source_tensor.shape)}, "
                f"target expects {expected_shape}"
            )
        target_state[target_key] = source_tensor.detach().cpu().clone()
    _validate_state_dict(target_state, target_cfg, label="target")

    source_cfg_dict = dict(source_cfg.__dict__)
    target_cfg_dict = dict(target_cfg.__dict__)
    map_records = [
        {"target_layer": target_layer, "source_layer": source_layer}
        for target_layer, source_layer in sorted(normalized_map.items())
    ]
    manifest_core: dict[str, Any] = {
        "format": GROWTH_FORMAT,
        "schema_version": GROWTH_SCHEMA_VERSION,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "source_state_dict_sha256": state_dict_sha256(source_state),
        "target_state_dict_sha256": state_dict_sha256(target_state),
        "source_model_config": source_cfg_dict,
        "source_model_config_sha256": canonical_sha256(source_cfg_dict),
        "target_model_config": target_cfg_dict,
        "target_model_config_sha256": canonical_sha256(target_cfg_dict),
        "allowed_model_config_differences": sorted(_ALLOWED_CONFIG_DIFFERENCES),
        "strict_model_config_fields": sorted(
            set(ModelConfig.__dataclass_fields__) - _ALLOWED_CONFIG_DIFFERENCES
        ),
        "tokenizer_sha256": tokenizer_sha256,
        "target_layer_to_source_layer": map_records,
        "mapped_block_kinds_match": True,
        "function_preserving": False,
        "optimizer_state": "discarded",
        "source_stage": source_stage,
        "discarded_payload_keys": sorted(
            set(source_checkpoint) - {"cfg", "model", "state_dict", "tokenizer"}
        ),
        "warning": (
            "NOT function-preserving: repeated/reordered residual blocks change the model "
            "function. Training progress, optimizer/RNG state, lineage, and other unlisted "
            "payload state are discarded; use only as a fresh-optimizer backbone warm start."
        ),
    }
    manifest = {
        **manifest_core,
        "manifest_sha256": canonical_sha256(manifest_core),
    }
    tokenizer_metadata = source_checkpoint.get("tokenizer")
    tokenizer = (
        dict(tokenizer_metadata)
        if isinstance(tokenizer_metadata, Mapping)
        else {"sha256": tokenizer_sha256}
    )
    tokenizer["sha256"] = tokenizer_sha256
    return {
        "cfg": target_cfg_dict,
        "state_dict": target_state,
        "stage": "checkpoint_growth",
        "tokenizer": tokenizer,
        "growth": manifest,
    }


def grow_checkpoint(
    source_path: str | Path,
    target_cfg: ModelConfig,
    layer_map: Mapping[int, int],
) -> dict[str, Any]:
    """Load content-bound source bytes and build the target-shaped warm-start payload."""

    checkpoint, source_sha256 = load_checkpoint_with_sha256(source_path)
    return _grow_checkpoint_payload(
        checkpoint,
        target_cfg,
        layer_map,
        source_checkpoint_sha256=source_sha256,
    )


def write_grown_checkpoint(
    source_path: str | Path,
    target_cfg: ModelConfig,
    layer_map: Mapping[int, int],
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Grow one checkpoint and atomically write the auditable target artifact."""

    source = Path(source_path)
    output = Path(output_path)
    if source.resolve() == output.resolve():
        raise ValueError("source and output checkpoints must be different paths")
    if output.exists() and not overwrite:
        raise FileExistsError(f"output checkpoint already exists: {output}")
    grown = grow_checkpoint(source, target_cfg, layer_map)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("w+b") as handle:
            torch.save(grown, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, output)
        else:
            try:
                os.link(temporary, output)
            except FileExistsError as error:
                raise FileExistsError(
                    f"output checkpoint already exists: {output}"
                ) from error
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    return grown["growth"]


__all__ = [
    "GROWTH_FORMAT",
    "GROWTH_SCHEMA_VERSION",
    "assert_growth_compatible",
    "checkpoint_tokenizer_sha256",
    "grow_checkpoint",
    "load_checkpoint_with_sha256",
    "parse_layer_map",
    "state_dict_sha256",
    "verify_growth_checkpoint",
    "write_grown_checkpoint",
]
