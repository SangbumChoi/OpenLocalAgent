"""Deterministic, no-model token-budget plans for post-training stages.

Plans are content-addressed canonical JSON.  They load and verify the same tokenizer and declared
data artifacts as the stage runners, but deliberately never open a parent checkpoint or construct
``LocalAgentLM``.  Midtrain and SFT plans count the exact shifted/masked language-model tokens
selected by the shared schedules.  RL plans report only deterministic prompt coverage and bounded
action-token opportunities because generated action lengths depend on the policy.
"""

from __future__ import annotations

import copy
import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from localagent.data.conversation_artifact import assert_no_conversation_overlap
from localagent.data.decision_quota_order import (
    QUOTA_SAMPLING_MODE,
    order_assistant_decisions,
    quota_sampling_contract,
)
from localagent.data.pretrain_corpus import PackedShardDataset
from localagent.data.prompt_contract import (
    LEGACY_CONVERSATION_PROMPT_CONTRACT,
    OPENAI_FULL_CATALOG_V1,
    assert_prompt_contract_tokenizer,
    resolve_conversation_prompt_contract,
)
from localagent.data.render import prompt_text
from localagent.data.schema import Role
from localagent.data.stratified_eval_selector import (
    ALGORITHM as STRATIFIED_EVAL_ALGORITHM,
)
from localagent.data.stratified_eval_selector import (
    select_stratified_eval_subset,
)
from localagent.model import ModelConfig
from localagent.model.tokenizer import batched_token_lengths, load_tokenizer
from localagent.train.midtrain import (
    ConversationTokenCountDataset,
    MixtureSource,
    ScheduledMixture,
    _audit_packed_holdout_splits,
    validate_packed_source,
)
from localagent.train.loop import validate_pad_to_input_tokens
from localagent.train.function_masking import augment_conversations
from localagent.train.replay_sampling import (
    MIXED_REPLAY_SAMPLING_MODE,
    PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE,
    mixed_replay_sampling_window,
    parent_anchored_format_pulse_sampling_window,
)
from localagent.train.rl import (
    CatalogStringCache,
    _assert_gold_outputs_fit,
    _audit_data_splits,
    _decision_prompt_text,
    _preflight_full_context,
    decision_keys_to_prompt_order,
    project_rl_decisions,
)
from localagent.train.sft import quota_sampling_window, resolve_sft_continuation
from localagent.train.stage_data import (
    canonical_sha256,
    file_identity,
    load_conversation_source,
    probe_decisions,
    single_turn_samples,
    tokenizer_identity,
)
from localagent.train.stage_sampling import (
    SFT_FORWARD_SLOT_KEYS,
    SFT_LOSS_NORMALIZATION_MICROBATCH,
    SFT_LOSS_NORMALIZATION_UPDATE_TOKENS,
    RLPromptSchedule,
    SFTSamplingSchedule,
    add_row_accounting,
    decision_keys_to_row_order,
    empty_token_accounting,
    next_midtrain_microbatch_counts,
    prepare_sft_data,
    sft_microbatch_forward_token_slots,
    validate_sft_loss_normalization,
    validate_multi_turn_batch_size,
)

PLAN_KIND = "localagent_stage_budget_plan"
PLAN_SCHEMA_VERSION = 2


def canonical_plan_bytes(plan: Mapping[str, Any]) -> bytes:
    """Return the single accepted on-disk JSON representation."""

    return (
        json.dumps(
            plan,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _without_self_hash(plan: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(plan))
    payload.pop("plan_self_sha256", None)
    return payload


def seal_stage_budget_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Attach the canonical self-hash, replacing no pre-existing seal."""

    if "plan_self_sha256" in plan:
        raise ValueError("cannot seal a stage budget plan that already has a self-hash")
    sealed = copy.deepcopy(dict(plan))
    sealed["plan_self_sha256"] = canonical_sha256(sealed)
    return sealed


def assert_stage_budget_self_hash(plan: Mapping[str, Any]) -> None:
    """Fail when the embedded self-hash is absent, malformed, or stale."""

    recorded = plan.get("plan_self_sha256")
    if (
        not isinstance(recorded, str)
        or len(recorded) != 64
        or any(character not in "0123456789abcdef" for character in recorded)
    ):
        raise ValueError("stage budget plan has no valid plan_self_sha256")
    expected = canonical_sha256(_without_self_hash(plan))
    if recorded != expected:
        raise ValueError("stage budget plan self-hash mismatch")


def _source_specs(value: Any, *, label: str) -> list[Any]:
    if isinstance(value, (str, Path, Mapping)):
        return [value]
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a source or list of sources")
    return value


def _nonnegative_budget(value: int | None, *, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _positive_steps(value: int, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _base_context(config_path: str | Path) -> tuple[dict[str, Any], ModelConfig, Any, dict]:
    import yaml

    path = Path(config_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise TypeError("training config root must be a mapping")
    config = dict(config)
    model_config_path = Path(config["model_config"])
    model_config = ModelConfig.from_yaml(model_config_path)
    model_config.assert_within_budget()

    data_cfg = config["data"]
    if not isinstance(data_cfg, Mapping):
        raise TypeError("training data config must be a mapping")
    tok_cfg = data_cfg.get("tokenizer", {"kind": "byte"})
    if not isinstance(tok_cfg, Mapping):
        raise TypeError("data.tokenizer must be a mapping")
    tokenizer_kind = str(tok_cfg.get("kind", "byte"))
    tokenizer_path = tok_cfg.get("path")
    tokenizer = load_tokenizer(tokenizer_kind, tokenizer_path)
    if tokenizer.vocab_size != model_config.vocab_size:
        raise ValueError("tokenizer vocabulary does not match model config")
    tokenizer_lineage = tokenizer_identity(
        tokenizer_kind,
        vocab_size=tokenizer.vocab_size,
        path=tokenizer_path,
    )
    identity = {
        "config": {
            "path": str(path),
            **file_identity(path),
            "canonical_sha256": canonical_sha256(config),
        },
        "model_config": {
            "path": str(model_config_path),
            **file_identity(model_config_path),
            "canonical_sha256": canonical_sha256(model_config.__dict__),
        },
        "tokenizer": {
            "path": str(tokenizer_path) if tokenizer_path is not None else None,
            **tokenizer_lineage,
        },
    }
    return config, model_config, tokenizer, identity


def _loaded_source_identity(source) -> dict[str, Any]:
    return {
        "path": str(source.path),
        "verified": bool(source.verified),
        "artifact": dict(source.identity),
    }


def _packed_source_identity(dataset: PackedShardDataset) -> dict[str, Any]:
    entries = dataset.manifest["splits"][dataset.split]["shards"]
    artifacts = []
    for entry in entries:
        artifacts.append(
            {
                "tokens": {
                    "path": str(entry["tokens"]),
                    "bytes": int(entry["bytes"]),
                    "sha256": str(entry["sha256"]),
                },
                "lengths": {
                    "path": str(entry["lengths"]),
                    "bytes": int(entry["lengths_bytes"]),
                    "sha256": str(entry["lengths_sha256"]),
                },
                "rows": int(entry["rows"]),
            }
        )
    manifest_path = Path(dataset.root) / "manifest.json"
    return {
        "path": str(dataset.root),
        "split": dataset.split,
        "manifest": {
            "path": str(manifest_path),
            **file_identity(manifest_path),
            "canonical_sha256": canonical_sha256(dataset.manifest),
        },
        "artifacts": artifacts,
        "rows": len(dataset),
        "seq_len": dataset.seq_len,
    }


def _zero_update_sources(source_names: Sequence[str]) -> dict[str, dict[str, int]]:
    return {
        source: {"draws": 0, "rows": 0, "input_tokens": 0, "loss_tokens": 0}
        for source in dict.fromkeys(source_names)
    }


def _sum_updates(
    updates: Sequence[Mapping[str, Any]],
    source_names: Sequence[str],
    *,
    steps: int | None = None,
) -> dict[str, Any]:
    selected = updates if steps is None else updates[:steps]
    totals = {
        "updates": len(selected),
        "input_tokens": 0,
        "loss_tokens": 0,
        "sources": _zero_update_sources(source_names),
    }
    for update in selected:
        totals["input_tokens"] += int(update["input_tokens"])
        totals["loss_tokens"] += int(update["loss_tokens"])
        for source, metrics in update["sources"].items():
            destination = totals["sources"][source]
            for key in ("draws", "rows", "input_tokens", "loss_tokens"):
                destination[key] += int(metrics[key])
    return totals


def _empty_forward_token_slots() -> dict[str, int]:
    return {**{key: 0 for key in SFT_FORWARD_SLOT_KEYS}, "total": 0}


def _add_forward_token_slots(
    destination: dict[str, int],
    values: Mapping[str, Any],
) -> None:
    for key in SFT_FORWARD_SLOT_KEYS:
        destination[key] += int(values[key])
    destination["total"] = sum(destination[key] for key in SFT_FORWARD_SLOT_KEYS)


def _sum_sft_updates(
    updates: Sequence[Mapping[str, Any]],
    source_names: Sequence[str],
    *,
    steps: int | None = None,
) -> dict[str, Any]:
    totals = _sum_updates(updates, source_names, steps=steps)
    selected = updates if steps is None else updates[:steps]
    forward_slots = _empty_forward_token_slots()
    for update in selected:
        _add_forward_token_slots(forward_slots, update["model_forward_token_slots"])
    totals["model_forward_token_slots"] = forward_slots
    return totals


_RL_FORWARD_PHASES = (
    "rollout_prefill",
    "rollout_cached_decode",
    "old_policy_scoring",
    "reference_policy_scoring",
    "current_policy_optimization",
)


def _rl_forward_slot_bounds(
    *,
    prompt_input_tokens: int,
    prompt_groups: int,
    group_size: int,
    max_new_tokens: int,
    reference_enabled: bool,
    policy_epochs: int,
) -> dict[str, Any]:
    """Bound exact tensor elements presented to every RL backbone forward.

    Rollout decoding uses one full-prompt prefill and then one-token cached forwards.  Scoring
    concatenates each prompt and generated action and drops the final input token.  Reward
    diversity and generated lengths are policy-dependent, so only safe lower/upper bounds can be
    replayed without loading the policy.
    """

    rollout_prefill = group_size * prompt_input_tokens
    rollout_decode_upper = prompt_groups * group_size * max_new_tokens
    scoring_upper = group_size * (
        prompt_input_tokens + prompt_groups * (max_new_tokens - 1)
    )
    phases = {
        "rollout_prefill": {"lower": rollout_prefill, "upper": rollout_prefill},
        "rollout_cached_decode": {"lower": 0, "upper": rollout_decode_upper},
        "old_policy_scoring": {"lower": 0, "upper": scoring_upper},
        "reference_policy_scoring": {
            "lower": 0,
            "upper": scoring_upper if reference_enabled else 0,
        },
        "current_policy_optimization": {
            "lower": 0,
            "upper": scoring_upper * policy_epochs,
        },
    }
    return {
        "phases": phases,
        "total": {
            "lower": sum(phases[phase]["lower"] for phase in _RL_FORWARD_PHASES),
            "upper": sum(phases[phase]["upper"] for phase in _RL_FORWARD_PHASES),
        },
    }


def _sum_rl_forward_slot_bounds(updates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    phases = {
        phase: {"lower": 0, "upper": 0}
        for phase in _RL_FORWARD_PHASES
    }
    for update in updates:
        bounds = update["model_forward_token_slot_bounds"]
        for phase in _RL_FORWARD_PHASES:
            phases[phase]["lower"] += int(bounds["phases"][phase]["lower"])
            phases[phase]["upper"] += int(bounds["phases"][phase]["upper"])
    return {
        "phases": phases,
        "total": {
            "lower": sum(phases[phase]["lower"] for phase in _RL_FORWARD_PHASES),
            "upper": sum(phases[phase]["upper"] for phase in _RL_FORWARD_PHASES),
        },
    }


def _conversation_count_only_totals(dataset: ConversationTokenCountDataset) -> dict[str, int]:
    """Summarize an existing count-only dataset without constructing token tensors."""

    row_counts = tuple(
        (input_tokens, loss_tokens)
        for input_tokens, loss_tokens in dataset._row_token_counts
        if input_tokens > 0 and loss_tokens > 0
    )
    if not row_counts:
        raise ValueError("SFT held-out data has no trainable assistant targets")
    return {
        "rows": len(row_counts),
        "input_tokens": sum(input_tokens for input_tokens, _ in row_counts),
        "loss_tokens": sum(loss_tokens for _, loss_tokens in row_counts),
    }


def calibrate_supervised_prefix(
    updates: Sequence[Mapping[str, Any]],
    *,
    min_supervised_tokens: int | None,
    max_supervised_tokens: int | None,
) -> dict[str, Any]:
    """Choose the smallest fixed-horizon prefix satisfying explicit token bounds."""

    minimum = _nonnegative_budget(
        min_supervised_tokens,
        label="min_supervised_tokens",
    )
    maximum = _nonnegative_budget(
        max_supervised_tokens,
        label="max_supervised_tokens",
    )
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("min_supervised_tokens cannot exceed max_supervised_tokens")

    if minimum is None:
        selected_steps = len(updates)
        selected_tokens = sum(int(update["loss_tokens"]) for update in updates)
        previous_tokens = selected_tokens - int(updates[-1]["loss_tokens"]) if updates else 0
        mode = "full_horizon"
    else:
        selected_steps = 0
        selected_tokens = 0
        previous_tokens = 0
        for update in updates:
            if selected_tokens >= minimum:
                break
            previous_tokens = selected_tokens
            selected_tokens += int(update["loss_tokens"])
            selected_steps += 1
        if selected_tokens < minimum:
            raise ValueError(
                "fixed max_steps horizon cannot reach min_supervised_tokens: "
                f"planned={selected_tokens}, required={minimum}"
            )
        mode = "smallest_fixed_horizon_prefix"

    if maximum is not None and selected_tokens > maximum:
        raise ValueError(
            "smallest qualifying prefix exceeds max_supervised_tokens: "
            f"planned={selected_tokens}, maximum={maximum}"
        )
    return {
        "contract": (
            "smallest prefix of the fixed max_steps horizon; changing the horizon "
            "reparameterizes scheduled weights/decay and requires a new plan"
        ),
        "mode": mode,
        "min_supervised_tokens": minimum,
        "max_supervised_tokens": maximum,
        "selected_steps": selected_steps,
        "previous_prefix_loss_tokens": previous_tokens,
        "selected_prefix_loss_tokens": selected_tokens,
    }


def _midtrain_plan(
    config: Mapping[str, Any],
    model_config: ModelConfig,
    tokenizer,
    *,
    max_steps: int,
    min_supervised_tokens: int | None,
    max_supervised_tokens: int | None,
) -> dict[str, Any]:
    data_cfg = config["data"]
    conversation_prompt_contract = resolve_conversation_prompt_contract(
        data_cfg.get("conversation_prompt_contract")
    )
    assert_prompt_contract_tokenizer(tokenizer, conversation_prompt_contract)
    strict = data_cfg.get("strict_conversation_artifacts", False)
    if not isinstance(strict, bool):
        raise TypeError("data.strict_conversation_artifacts must be boolean")
    tok_cfg = data_cfg.get("tokenizer", {"kind": "byte"})
    tokenizer_path = tok_cfg.get("path")
    configured_tokenizer_sha256 = (
        str(file_identity(tokenizer_path)["sha256"]) if tokenizer_path is not None else None
    )
    mixture_cfg = data_cfg.get("mixture", {})
    if not isinstance(mixture_cfg, Mapping):
        raise TypeError("midtrain data.mixture must be a mapping")
    mixture_unit = str(mixture_cfg.get("unit", "draws"))

    sources: list[MixtureSource] = []
    source_identities: list[dict[str, Any]] = []
    source_names: list[str] = []
    train_keys: set[tuple[str, str, str | None]] = set()
    train_fingerprints: set[str] = set()
    train_conversations = []
    packed_train: list[tuple[str, object]] = []
    shard_fingerprints: set[str] = set()
    for source_cfg in data_cfg["sources"]:
        source_type = str(source_cfg["type"])
        source_name = str(source_cfg["name"])
        source_path = source_cfg["path"]
        source_key = (
            source_type,
            str(Path(source_path).resolve()),
            source_cfg.get("split", "train") if source_type == "shards" else None,
        )
        if source_key in train_keys:
            raise ValueError("midtrain training sources contain a duplicate artifact/split")
        train_keys.add(source_key)
        if source_type == "shards":
            dataset = PackedShardDataset(
                source_path,
                split=source_cfg.get("split", "train"),
            )
            packed_train.append((source_name, dataset))
            fingerprint = validate_packed_source(
                dataset,
                model_config,
                source_name=source_name,
                configured_tokenizer_sha256=configured_tokenizer_sha256,
            )
            if fingerprint is not None:
                shard_fingerprints.add(fingerprint)
            identity = _packed_source_identity(dataset)
            runner_artifact_identity = {
                "manifest_sha256": canonical_sha256(dataset.manifest),
                "split": dataset.split,
            }
        elif source_type == "conversations":
            loaded = load_conversation_source(
                source_cfg,
                require_verified=strict,
                expected_split="train",
            )
            rows = list(loaded.conversations)
            dataset = ConversationTokenCountDataset(
                rows,
                tokenizer,
                model_config.max_seq_len,
                conversation_prompt_contract=conversation_prompt_contract,
            )
            train_conversations.extend(rows)
            identity = _loaded_source_identity(loaded)
            runner_artifact_identity = dict(loaded.identity)
        else:
            raise ValueError(f"unknown midtrain source type {source_type!r}")
        content_fingerprint = canonical_sha256(
            {"type": source_type, "artifact": runner_artifact_identity}
        )
        if content_fingerprint in train_fingerprints:
            raise ValueError("midtrain training sources contain duplicate content identities")
        train_fingerprints.add(content_fingerprint)
        start_weight = float(source_cfg["weight"])
        end_weight = float(source_cfg.get("end_weight", start_weight))
        sources.append(
            MixtureSource(
                name=source_name,
                dataset=dataset,
                start_weight=start_weight,
                end_weight=end_weight,
            )
        )
        source_names.append(source_name)
        source_identities.append(
            {
                "name": source_name,
                "type": source_type,
                "start_weight": start_weight,
                "end_weight": end_weight,
                "identity": identity,
            }
        )
    if len(shard_fingerprints) > 1:
        raise ValueError("midtrain shard sources use different tokenizer fingerprints")

    eval_identities: list[dict[str, Any]] = []
    eval_keys: set[tuple[str, str, str | None]] = set()
    eval_fingerprints: set[str] = set()
    eval_names: set[str] = set()
    eval_conversations = []
    packed_eval: list[tuple[str, object]] = []
    for source_cfg in data_cfg.get("eval_sources", []):
        source_type = str(source_cfg["type"])
        source_name = str(source_cfg["name"])
        if source_name in eval_names:
            raise ValueError("midtrain held-out source names must be unique")
        eval_names.add(source_name)
        source_path = source_cfg["path"]
        source_key = (
            source_type,
            str(Path(source_path).resolve()),
            source_cfg.get("split", "val") if source_type == "shards" else None,
        )
        if source_key in train_keys:
            raise ValueError("midtrain held-out source exactly overlaps a training source")
        if source_key in eval_keys:
            raise ValueError("midtrain held-out sources contain a duplicate artifact/split")
        eval_keys.add(source_key)
        if source_type == "shards":
            dataset = PackedShardDataset(
                source_path,
                split=source_cfg.get("split", "val"),
            )
            packed_eval.append((source_name, dataset))
            fingerprint = validate_packed_source(
                dataset,
                model_config,
                source_name=source_name,
                configured_tokenizer_sha256=configured_tokenizer_sha256,
            )
            if fingerprint is not None:
                shard_fingerprints.add(fingerprint)
            identity = _packed_source_identity(dataset)
            runner_artifact_identity = {
                "manifest_sha256": canonical_sha256(dataset.manifest),
                "split": dataset.split,
            }
        elif source_type == "conversations":
            loaded = load_conversation_source(
                source_cfg,
                require_verified=strict,
                expected_split="eval",
            )
            eval_rows = list(loaded.conversations)
            ConversationTokenCountDataset(
                eval_rows,
                tokenizer,
                model_config.max_seq_len,
                conversation_prompt_contract=conversation_prompt_contract,
            )
            eval_conversations.extend(eval_rows)
            identity = _loaded_source_identity(loaded)
            runner_artifact_identity = dict(loaded.identity)
        else:
            raise ValueError(f"unknown midtrain eval source type {source_type!r}")
        content_fingerprint = canonical_sha256(
            {"type": source_type, "artifact": runner_artifact_identity}
        )
        if content_fingerprint in train_fingerprints:
            raise ValueError("midtrain held-out source content overlaps a training source")
        if content_fingerprint in eval_fingerprints:
            raise ValueError("midtrain held-out sources contain duplicate content identities")
        eval_fingerprints.add(content_fingerprint)
        eval_identities.append({"name": source_name, "type": source_type, "identity": identity})
    if len(shard_fingerprints) > 1:
        raise ValueError("midtrain train/eval shard sources use different tokenizer fingerprints")
    conversation_overlap = assert_no_conversation_overlap(
        train_conversations,
        eval_conversations,
        left_label="midtrain training",
        right_label="held-out",
        conversation_prompt_contract=conversation_prompt_contract,
    )
    packed_overlap = _audit_packed_holdout_splits(packed_train, packed_eval)

    batch_cfg = config.get("batch", {})
    batch_size = int(batch_cfg.get("micro_batch_size", 8))
    accum_steps = int(batch_cfg.get("grad_accum_steps", 1))
    if batch_size < 1 or accum_steps < 1:
        raise ValueError("midtraining batch and accumulation sizes must be positive")
    schedule_type = str(config.get("schedule", {}).get("type", "wsd"))
    if schedule_type not in {"cosine", "wsd"}:
        raise ValueError(f"midtrain lr_schedule must be 'cosine' or 'wsd', got {schedule_type!r}")
    if eval_identities:
        evaluation_cfg = config.get("evaluation", {})
        eval_batches = int(evaluation_cfg.get("batches_per_source", 8))
        eval_batch_size = int(evaluation_cfg.get("batch_size", batch_size))
        if eval_batches < 1 or eval_batch_size < 1:
            raise ValueError("held-out batches_per_source and batch_size must be positive")
    seed = int(config.get("runtime", {}).get("seed", 0))
    rng = random.Random(seed)
    mixture = ScheduledMixture(sources, unit=mixture_unit)
    mixture_state = mixture.initial_state()
    updates: list[dict[str, Any]] = []
    for step in range(max_steps):
        update_sources = _zero_update_sources(source_names)
        input_tokens = 0
        loss_tokens = 0
        for _ in range(accum_steps):
            microbatch = next_midtrain_microbatch_counts(
                mixture,
                mixture_state,
                rng,
                step=step,
                total_steps=max_steps,
                batch_size=batch_size,
            )
            source_metrics = update_sources[microbatch.source.name]
            source_metrics["draws"] += 1
            source_metrics["rows"] += batch_size
            source_metrics["input_tokens"] += microbatch.input_tokens
            source_metrics["loss_tokens"] += microbatch.loss_tokens
            input_tokens += microbatch.input_tokens
            loss_tokens += microbatch.loss_tokens
        updates.append(
            {
                "step": step,
                "input_tokens": input_tokens,
                "loss_tokens": loss_tokens,
                "sources": update_sources,
            }
        )

    calibration = calibrate_supervised_prefix(
        updates,
        min_supervised_tokens=min_supervised_tokens,
        max_supervised_tokens=max_supervised_tokens,
    )
    selected_steps = int(calibration["selected_steps"])
    return {
        "data": {
            **(
                {"conversation_prompt_contract": conversation_prompt_contract}
                if conversation_prompt_contract != LEGACY_CONVERSATION_PROMPT_CONTRACT
                else {}
            ),
            "sources": source_identities,
            "eval_sources": eval_identities,
            "conversation_overlap_audit": conversation_overlap.as_dict(),
            "packed_holdout_audit": packed_overlap,
        },
        "schedule": {
            **(
                {"conversation_prompt_contract": conversation_prompt_contract}
                if conversation_prompt_contract != LEGACY_CONVERSATION_PROMPT_CONTRACT
                else {}
            ),
            "seed": seed,
            "mixture_unit": mixture_unit,
            "lr_schedule": schedule_type,
            "micro_batch_size": batch_size,
            "grad_accum_steps": accum_steps,
        },
        "planned": {
            "accounting_kind": "exact_shifted_masked_language_model_tokens",
            "updates": updates,
            "horizon_totals": _sum_updates(updates, source_names),
            "selected_totals": _sum_updates(
                updates,
                source_names,
                steps=selected_steps,
            ),
            "final_mixture_state": mixture_state,
        },
        "calibration": calibration,
    }


def _sft_plan(
    config: Mapping[str, Any],
    model_config: ModelConfig,
    tokenizer,
    *,
    max_steps: int,
    min_supervised_tokens: int | None,
    max_supervised_tokens: int | None,
) -> dict[str, Any]:
    continuation = resolve_sft_continuation(config)
    data_cfg = config["data"]
    conversation_prompt_contract = resolve_conversation_prompt_contract(
        data_cfg.get("conversation_prompt_contract")
    )
    assert_prompt_contract_tokenizer(tokenizer, conversation_prompt_contract)
    strict = data_cfg.get("strict_conversation_artifacts", False)
    if not isinstance(strict, bool):
        raise TypeError("data.strict_conversation_artifacts must be boolean")
    conversation_specs = _source_specs(
        data_cfg["conversations"],
        label="data.conversations",
    )
    loaded_sources = [
        load_conversation_source(
            source,
            require_verified=strict,
            expected_split="train",
        )
        for source in conversation_specs
    ]
    decay_specs = _source_specs(
        data_cfg.get("decay_conversations", []),
        label="data.decay_conversations",
    )
    loaded_decay = [
        load_conversation_source(
            source,
            require_verified=strict,
            expected_split="train",
        )
        for source in decay_specs
    ]
    eval_specs = _source_specs(
        data_cfg.get("eval_conversations", []),
        label="data.eval_conversations",
    )
    loaded_eval = [
        load_conversation_source(
            source,
            require_verified=strict,
            expected_split="eval",
        )
        for source in eval_specs
    ]

    raw_conversations = [
        conversation for source in loaded_sources for conversation in source.conversations
    ]
    raw_conversation_source_paths = [
        str(source.path) for source in loaded_sources for _ in source.conversations
    ]
    runtime_cfg = config.get("runtime", {})
    if not isinstance(runtime_cfg, Mapping):
        raise TypeError("runtime must be a mapping")
    function_masking_seed = int(runtime_cfg.get("seed", 0))
    conversations, masked_source_indices, function_masking_main_audit = augment_conversations(
        raw_conversations,
        data_cfg.get("function_masking", False),
        seed=function_masking_seed,
    )
    raw_decay_conversations = [
        conversation for source in loaded_decay for conversation in source.conversations
    ]
    decay_conversations, masked_decay_source_indices, function_masking_decay_audit = (
        augment_conversations(
            raw_decay_conversations,
            data_cfg.get("function_masking", False),
            seed=function_masking_seed,
        )
    )
    function_masking_audit = {
        "enabled": bool(function_masking_main_audit["enabled"])
        or bool(function_masking_decay_audit["enabled"]),
        "main": function_masking_main_audit,
        "decay": function_masking_decay_audit,
    }
    conversation_sources = [
        raw_conversation_source_paths[index] for index in masked_source_indices
    ]
    raw_decay_source_paths = [
        f"decay:{source.path}" for source in loaded_decay for _ in source.conversations
    ]
    decay_conversation_sources = [
        raw_decay_source_paths[index] for index in masked_decay_source_indices
    ]
    eval_conversations = [
        conversation for source in loaded_eval for conversation in source.conversations
    ]
    full_eval_conversation_rows = len(eval_conversations)
    overlap = assert_no_conversation_overlap(
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

    samples = []
    sample_sources = []
    multi_turn = []
    multi_turn_sources = []
    for conversation, source_name in zip(
        conversations,
        conversation_sources,
        strict=True,
    ):
        projected = single_turn_samples([conversation])
        if projected:
            samples.extend(projected)
            sample_sources.extend([source_name] * len(projected))
        else:
            multi_turn.append(conversation)
            multi_turn_sources.append(source_name)

    heads_cfg = config.get("heads", {})
    function_masking_enabled = bool(function_masking_audit["enabled"])
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
    joint_heads = bool(heads_cfg.get("joint_tool_pointer", True))
    multi_turn_batch_size = validate_multi_turn_batch_size(
        heads_cfg.get("multi_turn_batch_size", 12)
    )
    decisions = probe_decisions(conversations)
    train_route = bool(heads_cfg.get("train_route_head", True))
    train_dense = bool(heads_cfg.get("train_dense_selector", True))
    if joint_heads and not samples:
        raise ValueError("joint tool/pointer heads need simple user -> assistant conversations")
    if train_route and not decisions:
        raise ValueError("route head needs at least one assistant decision")
    if train_dense:
        from localagent.agent.toolset import STANDARD_TOOLS

        standard_tool_names = {tool.name for tool in STANDARD_TOOLS}
        if not any(
            decision.kind == "tool" and decision.ref_name in standard_tool_names
            for decision in decisions
        ):
            raise ValueError(
                "dense selector needs at least one tool decision in the standard tool set"
            )

    decay_samples = None
    decay_sample_sources = None
    if loaded_decay and conversation_prompt_contract == LEGACY_CONVERSATION_PROMPT_CONTRACT:
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

    schedule_cfg = config.get("schedule", {})
    lr_schedule = str(schedule_cfg.get("type", "cosine"))
    decay_frac = float(schedule_cfg.get("decay_frac", 0.2))
    optim_cfg = config.get("optim", {})
    if not isinstance(optim_cfg, Mapping):
        raise TypeError("optim must be a mapping")
    freeze_parameters = optim_cfg.get("freeze_parameters")
    if freeze_parameters is not None:
        if not isinstance(freeze_parameters, list):
            raise TypeError(
                "optim.freeze_parameters must be a list of exact model parameter names"
            )
        if any(not isinstance(name, str) for name in freeze_parameters):
            raise TypeError("optim.freeze_parameters entries must be strings")
        if len(set(freeze_parameters)) != len(freeze_parameters):
            raise ValueError("optim.freeze_parameters must not contain duplicates")
    optimizer_name = optim_cfg.get("name", "adamw")
    if optimizer_name != "adamw":
        raise ValueError("optim.name must be exactly 'adamw' for SFT")
    weight_decay = optim_cfg.get("weight_decay", 0.0)
    if (
        isinstance(weight_decay, bool)
        or not isinstance(weight_decay, (int, float))
        or not math.isfinite(float(weight_decay))
        or float(weight_decay) < 0.0
    ):
        raise ValueError("optim.weight_decay must be a finite non-negative number")
    grad_clip = optim_cfg.get("grad_clip", 1.0)
    if (
        isinstance(grad_clip, bool)
        or not isinstance(grad_clip, (int, float))
        or not math.isfinite(float(grad_clip))
        or float(grad_clip) <= 0.0
    ):
        raise ValueError("optim.grad_clip must be a finite positive number")
    loss_normalization = validate_sft_loss_normalization(
        optim_cfg.get(
            "loss_normalization",
            SFT_LOSS_NORMALIZATION_MICROBATCH,
        )
    )
    if (
        loss_normalization == SFT_LOSS_NORMALIZATION_UPDATE_TOKENS
        and joint_heads
    ):
        raise ValueError(
            "assistant-token update normalization supports the LM-only SFT path"
        )
    seq_limit = min(
        int(data_cfg.get("seq_len", model_config.max_seq_len)),
        model_config.max_seq_len,
    )
    heldout_eval_token_accounting = None
    if eval_conversations:
        eval_count_dataset = ConversationTokenCountDataset(
            eval_conversations,
            tokenizer,
            seq_limit,
            conversation_prompt_contract=conversation_prompt_contract,
        )
        heldout_eval_token_accounting = {
            "accounting_kind": "exact_shifted_masked_language_model_tokens",
            **_conversation_count_only_totals(eval_count_dataset),
        }
    prepared = prepare_sft_data(
        samples,
        tokenizer,
        conversations=(
            multi_turn
            if conversation_prompt_contract == LEGACY_CONVERSATION_PROMPT_CONTRACT
            else conversations
        ),
        sample_sources=sample_sources,
        conversation_sources=(
            multi_turn_sources
            if conversation_prompt_contract == LEGACY_CONVERSATION_PROMPT_CONTRACT
            else conversation_sources
        ),
        decay_samples=decay_samples,
        decay_sample_sources=decay_sample_sources,
        lr_schedule=lr_schedule,
        max_seq_len=seq_limit,
        joint_tool_head=joint_heads,
        conversation_prompt_contract=conversation_prompt_contract,
        decay_conversations=(
            decay_conversations
            if conversation_prompt_contract != LEGACY_CONVERSATION_PROMPT_CONTRACT and loaded_decay
            else None
        ),
        decay_conversation_sources=(
            decay_conversation_sources
            if conversation_prompt_contract != LEGACY_CONVERSATION_PROMPT_CONTRACT
            else None
        ),
    )
    batch_cfg = config.get("batch", {})
    batch_size = int(batch_cfg.get("micro_batch_size", 8))
    accum_steps = int(batch_cfg.get("grad_accum_steps", 1))
    if batch_size < 1 or accum_steps < 1:
        raise ValueError("SFT batch and accumulation sizes must be positive")
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
    seed = int(config.get("runtime", {}).get("seed", 0))
    sampling_cfg = data_cfg.get("sampling")
    lm_order = None
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
        if loaded_decay:
            raise ValueError("decision sampling does not support decay_conversations")
        selected_decisions = max_steps * batch_size * accum_steps
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
                    [source.conversations for source in loaded_sources],
                    selected_decisions=selected_decisions,
                    sampling_config=sampling_cfg,
                )
            )
            effective_batch = batch_size * accum_steps
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
            ordered_decision_keys, decision_sampling_contract = (
                parent_anchored_format_pulse_sampling_window(
                    [source.conversations for source in loaded_sources],
                    selected_decisions=selected_decisions,
                    sampling_config=sampling_cfg,
                )
            )
            effective_batch = batch_size * accum_steps
            if (
                decision_sampling_contract["update_layout"]["update_decisions"]
                != effective_batch
            ):
                raise ValueError(
                    "parent-anchored replay update size must equal one complete "
                    "optimizer update: "
                    f"update_decisions="
                    f"{decision_sampling_contract['update_layout']['update_decisions']}, "
                    f"effective_batch={effective_batch}"
                )
        lm_order = decision_keys_to_row_order(
            conversations,
            ordered_decision_keys,
            expected_rows=len(prepared.main_entries),
        )
    sampling = SFTSamplingSchedule(
        prepared,
        batch_size=batch_size,
        shuffle=bool(data_cfg.get("shuffle", True)),
        seed=seed,
        lr_schedule=lr_schedule,
        decay_frac=decay_frac,
        kd_enabled=False,
        joint_tool_head=joint_heads,
        multi_turn_batch_size=multi_turn_batch_size,
        lm_order=lm_order,
    )
    source_names = list(
        dict.fromkeys(
            source for pool in (prepared.main_entries, prepared.decay_entries) for _, source in pool
        )
    )
    mixed_cycle_labels: tuple[str, ...] = ()
    if decision_sampling_contract is not None:
        sampling_mode = decision_sampling_contract.get("mode")
        if sampling_mode == MIXED_REPLAY_SAMPLING_MODE:
            mixed_cycle_labels = tuple(decision_sampling_contract["cycle"]["labels"])
        elif sampling_mode == PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE:
            update_layout = decision_sampling_contract["update_layout"]
            pulse_positions = set(update_layout["pulse_positions_zero_based"])
            phase_order = tuple(
                decision_sampling_contract["format_pulses"]["phase_order"]
            )
            update_decisions = int(update_layout["update_decisions"])
            mixed_cycle_labels = tuple(
                label
                for update_index in range(int(update_layout["total_updates"]))
                for label in (
                    tuple(
                        phase_order[row_index % len(phase_order)]
                        for row_index in range(update_decisions)
                    )
                    if update_index in pulse_positions
                    else ("general",) * update_decisions
                )
            )
    mixed_label_names = list(dict.fromkeys(mixed_cycle_labels))
    mixed_label_totals = (
        empty_token_accounting(mixed_label_names)
        if mixed_cycle_labels
        else None
    )
    mixed_label_cursor = 0
    updates: list[dict[str, Any]] = []
    for step in range(max_steps):
        update_accounting = empty_token_accounting(source_names)
        update_label_accounting = (
            empty_token_accounting(mixed_label_names)
            if mixed_cycle_labels
            else None
        )
        update_forward_slots = _empty_forward_token_slots()
        update_draws = Counter()
        for _ in range(accum_steps):
            selection = sampling.next_microbatch(step=step, total_steps=max_steps)
            _add_forward_token_slots(
                update_forward_slots,
                sft_microbatch_forward_token_slots(
                    prepared,
                    selection,
                    pad_to_input_tokens=pad_to_input_tokens,
                ),
            )
            pool = prepared.decay_entries if selection.pool == "decay" else prepared.main_entries
            for index in selection.lm_indices:
                row, source = pool[index]
                add_row_accounting(update_accounting, row, source)
                update_draws[source] += 1
                if (
                    update_label_accounting is not None
                    and mixed_label_totals is not None
                ):
                    label = mixed_cycle_labels[
                        mixed_label_cursor % len(mixed_cycle_labels)
                    ]
                    mixed_label_cursor += 1
                    add_row_accounting(update_label_accounting, row, label)
                    add_row_accounting(mixed_label_totals, row, label)
        update_sources = _zero_update_sources(source_names)
        for source in source_names:
            row_metrics = update_accounting["sources"][source]
            update_sources[source] = {
                "draws": int(update_draws[source]),
                "rows": int(row_metrics["rows"]),
                "input_tokens": int(row_metrics["input_tokens"]),
                "loss_tokens": int(row_metrics["loss_tokens"]),
            }
        updates.append(
            {
                "step": step,
                "input_tokens": int(update_accounting["input_tokens"]),
                "loss_tokens": int(update_accounting["loss_tokens"]),
                "model_forward_token_slots": update_forward_slots,
                "sources": update_sources,
                **(
                    {"mixed_replay_labels": update_label_accounting["sources"]}
                    if update_label_accounting is not None
                    else {}
                ),
            }
        )

    calibration = calibrate_supervised_prefix(
        updates,
        min_supervised_tokens=min_supervised_tokens,
        max_supervised_tokens=max_supervised_tokens,
    )
    selected_steps = int(calibration["selected_steps"])
    return {
        "data": {
            **(
                {"conversation_prompt_contract": conversation_prompt_contract}
                if conversation_prompt_contract != LEGACY_CONVERSATION_PROMPT_CONTRACT
                else {}
            ),
            "conversations": [_loaded_source_identity(source) for source in loaded_sources],
            "decay_conversations": [_loaded_source_identity(source) for source in loaded_decay],
            "eval_conversations": [_loaded_source_identity(source) for source in loaded_eval],
            "conversation_overlap_audit": overlap.as_dict(),
            "function_masking": function_masking_audit,
            "eval_source_conversation_rows": full_eval_conversation_rows,
            "eval_selected_conversation_rows": len(eval_conversations),
            **(
                {"eval_selection": eval_selection_audit}
                if eval_selection_audit is not None
                else {}
            ),
            "dataset_token_accounting": prepared.dataset_accounting,
            "heldout_eval_token_accounting": heldout_eval_token_accounting,
            **(
                {"decision_sampling": decision_sampling_contract}
                if decision_sampling_contract is not None
                else {}
            ),
            "rendered_pool_rows": {
                "single_turn": len(prepared.rows),
                "main": len(prepared.main_entries),
                "decay": len(prepared.decay_entries),
                "head_items": len(prepared.head_items),
                "multi_turn_head_items": len(prepared.multi_turn_items),
            },
        },
        "schedule": {
            **(
                {"conversation_prompt_contract": conversation_prompt_contract}
                if conversation_prompt_contract != LEGACY_CONVERSATION_PROMPT_CONTRACT
                else {}
            ),
            "seed": seed,
            "lr_schedule": lr_schedule,
            "decay_frac": decay_frac,
            "loss_normalization": loss_normalization,
            **(
                {"freeze_parameters": list(freeze_parameters)}
                if freeze_parameters is not None
                else {}
            ),
            "shuffle": bool(data_cfg.get("shuffle", True)),
            "lm_sampling": (
                decision_sampling_contract
                if decision_sampling_contract is not None
                else {
                    "mode": (
                        "iid_with_replacement_v1"
                        if bool(data_cfg.get("shuffle", True))
                        else "source_order_wrapping_v1"
                    )
                }
            ),
            "micro_batch_size": batch_size,
            "grad_accum_steps": accum_steps,
            **(
                {"pad_to_input_tokens": pad_to_input_tokens}
                if pad_to_input_tokens is not None
                else {}
            ),
            **(
                {"evaluation_pad_to_input_tokens": evaluation_pad_to_input_tokens}
                if evaluation_pad_to_input_tokens is not None
                else {}
            ),
            "joint_tool_pointer": joint_heads,
            "multi_turn_batch_size": multi_turn_batch_size,
            "distillation_enabled": False,
            **({"continuation": continuation} if continuation is not None else {}),
        },
        "planned": {
            "accounting_kind": "exact_shifted_masked_language_model_tokens",
            "model_forward_token_slots_kind": (
                "exact padded input-tensor elements for each explicit backbone forward"
            ),
            "updates": updates,
            "horizon_totals": _sum_sft_updates(updates, source_names),
            **(
                {"mixed_replay_label_totals": mixed_label_totals}
                if mixed_label_totals is not None
                else {}
            ),
            "selected_totals": _sum_sft_updates(
                updates,
                source_names,
                steps=selected_steps,
            ),
        },
        "calibration": calibration,
    }


def _rl_plan(
    config: Mapping[str, Any],
    model_config: ModelConfig,
    tokenizer,
    *,
    max_steps: int,
    min_supervised_tokens: int | None,
    max_supervised_tokens: int | None,
) -> dict[str, Any]:
    if min_supervised_tokens is not None or max_supervised_tokens is not None:
        raise ValueError(
            "RL generation-dependent supervised token totals cannot be calibrated without a policy"
        )
    environment = config.get("environment", {})
    if environment.get("name", "canonical_toolcalls") != "canonical_toolcalls":
        raise NotImplementedError("only canonical_toolcalls RL can be budget-planned")
    if environment.get("learned_judge", False):
        raise NotImplementedError("learned reward judges cannot be budget-planned")

    data_cfg = config["data"]
    conversation_prompt_contract = resolve_conversation_prompt_contract(
        data_cfg.get("conversation_prompt_contract")
    )
    assert_prompt_contract_tokenizer(tokenizer, conversation_prompt_contract)
    strict = data_cfg.get("strict_conversation_artifacts", False)
    if not isinstance(strict, bool):
        raise TypeError("data.strict_conversation_artifacts must be boolean")
    train_specs = _source_specs(
        data_cfg["conversations"],
        label="data.conversations",
    )
    raw_eval = data_cfg.get("eval_conversations")
    if not raw_eval:
        raise ValueError("RL requires explicit data.eval_conversations")
    eval_specs = _source_specs(raw_eval, label="data.eval_conversations")
    loaded_train = [
        load_conversation_source(
            source,
            require_verified=strict,
            expected_split="train",
        )
        for source in train_specs
    ]
    loaded_eval = [
        load_conversation_source(
            source,
            require_verified=strict,
            expected_split="eval",
        )
        for source in eval_specs
    ]
    train_conversations = []
    train_conversation_sources = []
    for source in loaded_train:
        source_rows = list(source.conversations)
        train_conversations.extend(source_rows)
        train_conversation_sources.extend([str(source.path)] * len(source_rows))
    eval_conversations = []
    eval_conversation_sources = []
    for source in loaded_eval:
        source_rows = list(source.conversations)
        eval_conversations.extend(source_rows)
        eval_conversation_sources.extend([str(source.path)] * len(source_rows))
    single_turn_rows = sum(
        len(conversation.messages) == 2
        and conversation.messages[0].role == Role.user
        and conversation.messages[1].role == Role.assistant
        for conversation in train_conversations
    )
    eval_single_turn_rows = sum(
        len(conversation.messages) == 2
        and conversation.messages[0].role == Role.user
        and conversation.messages[1].role == Role.assistant
        for conversation in eval_conversations
    )
    full_eval_conversation_rows = len(eval_conversations)
    full_eval_single_turn_rows = eval_single_turn_rows
    catalog_cache = CatalogStringCache()
    if conversation_prompt_contract == OPENAI_FULL_CATALOG_V1:
        samples = project_rl_decisions(
            train_conversations,
            sources=train_conversation_sources,
        )
        eval_samples = project_rl_decisions(
            eval_conversations,
            sources=eval_conversation_sources,
        )
        sample_sources = [decision.source for decision in samples]
        if not samples:
            raise ValueError("RL data has no assistant decisions")
        if not eval_samples:
            raise ValueError("RL eval data has no assistant decisions")
    else:
        samples = []
        sample_sources = []
        for source in loaded_train:
            projected = single_turn_samples(source.conversations)
            samples.extend(projected)
            sample_sources.extend([str(source.path)] * len(projected))
        eval_samples = single_turn_samples(eval_conversations)
        if not samples:
            raise ValueError("RL data has no simple user -> assistant exact-reward rows")
        if not eval_samples:
            raise ValueError("RL eval data has no simple user -> assistant exact-reward rows")
    split_audit = _audit_data_splits(
        train_conversations,
        eval_conversations,
        samples,
        eval_samples,
        conversation_prompt_contract=conversation_prompt_contract,
        catalog_cache=catalog_cache,
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
        selected_indices = [row_number - 1 for row_number in selection.source_row_numbers]
        eval_conversations = list(selection.conversations)
        eval_conversation_sources = [
            eval_conversation_sources[index] for index in selected_indices
        ]
        eval_selection_audit = selection.audit.as_dict()
        eval_single_turn_rows = sum(
            len(conversation.messages) == 2
            and conversation.messages[0].role == Role.user
            and conversation.messages[1].role == Role.assistant
            for conversation in eval_conversations
        )
        if conversation_prompt_contract == OPENAI_FULL_CATALOG_V1:
            eval_samples = project_rl_decisions(
                eval_conversations,
                sources=eval_conversation_sources,
            )
        else:
            eval_samples = single_turn_samples(eval_conversations)
        if not eval_samples:
            raise ValueError("selected RL evaluation subset has no exact-reward rows")
    selected_split_audit = _audit_data_splits(
        train_conversations,
        eval_conversations,
        samples,
        eval_samples,
        conversation_prompt_contract=conversation_prompt_contract,
        catalog_cache=catalog_cache,
    )

    rollout_cfg = config.get("rollout", {})
    warmup_steps = int(config.get("schedule", {}).get("warmup_steps", 5))
    prompts_per_step = int(rollout_cfg.get("prompts_per_step", 8))
    group_size = int(rollout_cfg.get("group_size", 4))
    max_new = int(rollout_cfg.get("max_new_tokens", 64))
    temperature = float(rollout_cfg.get("temperature", 1.0))
    policy_cfg = config.get("policy", {})
    clip_ratio = float(policy_cfg.get("clip_ratio", 0.2))
    kl_beta = float(policy_cfg.get("kl_beta", 0.02))
    policy_epochs = int(policy_cfg.get("epochs_per_rollout", 1))
    reward_cfg = config.get("reward", {})
    format_weight = float(reward_cfg.get("format_weight", 0.1))
    truncation_penalty = float(reward_cfg.get("truncation_penalty", 0.05))
    if group_size < 2:
        raise ValueError("group_size must be >= 2")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if prompts_per_step < 1:
        raise ValueError("prompts_per_step must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if policy_epochs < 1:
        raise ValueError("policy_epochs must be >= 1")
    if clip_ratio < 0 or clip_ratio >= 1:
        raise ValueError("clip_ratio must be in [0, 1)")
    if kl_beta < 0:
        raise ValueError("kl_beta must be non-negative")
    if format_weight < 0 or truncation_penalty < 0:
        raise ValueError("reward weights must be non-negative")
    if max_new < 1 or max_new >= model_config.max_seq_len:
        raise ValueError("max_new must be in [1, model.max_seq_len)")
    gold_output_budget = {
        "train": _assert_gold_outputs_fit(
            samples,
            tokenizer,
            max_new,
            split="training split",
            catalog_cache=catalog_cache,
        ),
        "eval": _assert_gold_outputs_fit(
            eval_samples,
            tokenizer,
            max_new,
            split="held-out split",
            catalog_cache=catalog_cache,
        ),
    }
    max_prompt = model_config.max_seq_len - max_new
    context_preflight = None
    if conversation_prompt_contract == OPENAI_FULL_CATALOG_V1:
        prompt_lengths = batched_token_lengths(
            tokenizer,
            (_decision_prompt_text(decision, catalog_cache) for decision in samples),
        )
        eval_prompt_lengths = batched_token_lengths(
            tokenizer,
            (_decision_prompt_text(decision, catalog_cache) for decision in eval_samples),
        )
        context_preflight = {
            "train": _preflight_full_context(
                samples,
                tokenizer,
                max_new=max_new,
                max_seq_len=model_config.max_seq_len,
                split="training split",
                catalog_cache=catalog_cache,
                prompt_lengths=prompt_lengths,
            ),
            "eval": _preflight_full_context(
                eval_samples,
                tokenizer,
                max_new=max_new,
                max_seq_len=model_config.max_seq_len,
                split="held-out split",
                catalog_cache=catalog_cache,
                prompt_lengths=eval_prompt_lengths,
            ),
        }
    else:
        prompt_lengths = [
            len(tokenizer.encode(prompt_text(sample))[-max_prompt:]) for sample in samples
        ]
        eval_prompt_lengths = [
            len(tokenizer.encode(prompt_text(sample))[-max_prompt:]) for sample in eval_samples
        ]
    seed = int(config.get("runtime", {}).get("seed", 0))
    prompt_sampling_cfg = rollout_cfg.get("prompt_sampling")
    prompt_order = None
    prompt_sampling_contract = None
    if prompt_sampling_cfg is not None:
        if not isinstance(prompt_sampling_cfg, Mapping):
            raise TypeError("rollout.prompt_sampling must be a mapping")
        sampling_mode = prompt_sampling_cfg.get("mode")
        if sampling_mode != QUOTA_SAMPLING_MODE:
            raise ValueError(
                "rollout.prompt_sampling.mode must be "
                f"{QUOTA_SAMPLING_MODE!r} when configured"
            )
        if conversation_prompt_contract != OPENAI_FULL_CATALOG_V1:
            raise ValueError("quota prompt sampling requires openai_full_catalog_v1")
        decision_ordering = order_assistant_decisions(train_conversations)
        prompt_order = decision_keys_to_prompt_order(samples, decision_ordering.keys)
        prompt_sampling_contract = quota_sampling_contract(
            decision_ordering,
            selected_decisions=max_steps * prompts_per_step,
            require_all_strata=False,
        )
    sampling = RLPromptSchedule(
        len(samples),
        prompts_per_step,
        seed=seed,
        prompt_order=prompt_order,
    )
    source_names = list(dict.fromkeys(sample_sources))
    coverage = [0] * len(samples)
    updates = []
    for step in range(max_steps):
        indices = sampling.indices_for_step(step)
        per_source = {
            source: {
                "prompt_groups": 0,
                "rollouts": 0,
                "prompt_input_tokens": 0,
                "rollout_prompt_token_opportunities": 0,
                "min_action_token_opportunities": 0,
                "max_action_token_opportunities": 0,
            }
            for source in source_names
        }
        prompt_input_tokens = 0
        rollout_prompt_opportunities = 0
        for index in indices:
            coverage[index] += 1
            source = sample_sources[index]
            prompt_length = prompt_lengths[index]
            source_metrics = per_source[source]
            source_metrics["prompt_groups"] += 1
            source_metrics["rollouts"] += group_size
            source_metrics["prompt_input_tokens"] += prompt_length
            source_metrics["rollout_prompt_token_opportunities"] += prompt_length * group_size
            source_metrics["min_action_token_opportunities"] += group_size
            source_metrics["max_action_token_opportunities"] += group_size * max_new
            prompt_input_tokens += prompt_length
            rollout_prompt_opportunities += prompt_length * group_size
        for source_metrics in per_source.values():
            source_metrics["model_forward_token_slot_bounds"] = _rl_forward_slot_bounds(
                prompt_input_tokens=source_metrics["prompt_input_tokens"],
                prompt_groups=source_metrics["prompt_groups"],
                group_size=group_size,
                max_new_tokens=max_new,
                reference_enabled=kl_beta > 0,
                policy_epochs=policy_epochs,
            )
        forward_slot_bounds = _rl_forward_slot_bounds(
            prompt_input_tokens=prompt_input_tokens,
            prompt_groups=len(indices),
            group_size=group_size,
            max_new_tokens=max_new,
            reference_enabled=kl_beta > 0,
            policy_epochs=policy_epochs,
        )
        updates.append(
            {
                "step": step,
                "prompt_indices": list(indices),
                "prompt_groups": len(indices),
                "unique_prompt_rows": len(set(indices)),
                "rollouts": len(indices) * group_size,
                "prompt_input_tokens": prompt_input_tokens,
                "rollout_prompt_token_opportunities": rollout_prompt_opportunities,
                "min_action_token_opportunities": len(indices) * group_size,
                "max_action_token_opportunities": len(indices) * group_size * max_new,
                "generated_tokens": {
                    "lower": len(indices) * group_size,
                    "upper": len(indices) * group_size * max_new,
                },
                "informative_groups": {"lower": 0, "upper": len(indices)},
                "optimizer_step_attempts": {
                    "lower": 0,
                    "upper": policy_epochs,
                },
                "model_forward_token_slot_bounds": forward_slot_bounds,
                "sources": per_source,
            }
        )
    horizon = {
        "updates": max_steps,
        "prompt_groups": sum(update["prompt_groups"] for update in updates),
        "rollouts": sum(update["rollouts"] for update in updates),
        "prompt_input_tokens": sum(update["prompt_input_tokens"] for update in updates),
        "rollout_prompt_token_opportunities": sum(
            update["rollout_prompt_token_opportunities"] for update in updates
        ),
        "min_action_token_opportunities": sum(
            update["min_action_token_opportunities"] for update in updates
        ),
        "max_action_token_opportunities": sum(
            update["max_action_token_opportunities"] for update in updates
        ),
        "unique_prompt_rows_sampled": sum(count > 0 for count in coverage),
        "prompt_rows_never_sampled": sum(count == 0 for count in coverage),
        "minimum_prompt_draws": min(coverage),
        "maximum_prompt_draws": max(coverage),
        "prompt_draw_counts": coverage,
        "generated_tokens": {
            "lower": sum(update["generated_tokens"]["lower"] for update in updates),
            "upper": sum(update["generated_tokens"]["upper"] for update in updates),
        },
        "informative_groups": {
            "lower": 0,
            "upper": sum(update["informative_groups"]["upper"] for update in updates),
        },
        "optimizer_step_attempts": {
            "lower": 0,
            "upper": sum(update["optimizer_step_attempts"]["upper"] for update in updates),
        },
        "model_forward_token_slot_bounds": _sum_rl_forward_slot_bounds(updates),
    }
    heldout_prompt_tokens = sum(eval_prompt_lengths)
    heldout_rows = len(eval_prompt_lengths)
    heldout_evaluation_bounds = {
        "scope": "greedy held-out evaluation before and after training",
        "passes": 2,
        "prompt_rows_per_pass": heldout_rows,
        "prompt_tokens_per_pass": heldout_prompt_tokens,
        "model_forward_token_slots": {
            "lower": 2 * heldout_prompt_tokens,
            "upper": 2 * (heldout_prompt_tokens + heldout_rows * max_new),
        },
    }
    return {
        "data": {
            **(
                {
                    "conversation_prompt_contract": conversation_prompt_contract,
                    "assistant_decision_rows": len(samples),
                    "eval_assistant_decision_rows": len(eval_samples),
                    "context_preflight": context_preflight,
                    "prompt_truncation": "forbidden",
                    "prompt_token_lengths": {
                        "train": prompt_lengths,
                        "eval": eval_prompt_lengths,
                    },
                    "prompt_length_retention": "integer_token_counts_only",
                    "retains_complete_prompts": False,
                    "retains_prompt_token_ids": False,
                    "catalog_cache": {
                        "identity": "sha256(exact rendered catalog text + EOS)",
                        "unique_catalogs": catalog_cache.unique_catalogs,
                        "retained_characters": catalog_cache.retained_characters,
                    },
                    "schema_validation": (
                        "validate_tool_catalog + recursive schema_matches "
                        "(including additionalProperties)"
                    ),
                    "output_validation": ("strict parse_tool_output before exact/schema scoring"),
                    "gold_output_contract": ("shared AssistantTrainingTurn body + terminal EOS"),
                }
                if conversation_prompt_contract != LEGACY_CONVERSATION_PROMPT_CONTRACT
                else {}
            ),
            "conversations": [_loaded_source_identity(source) for source in loaded_train],
            "eval_conversations": [_loaded_source_identity(source) for source in loaded_eval],
            "split_audit": split_audit,
            "selected_eval_split_audit": selected_split_audit,
            **(
                {"eval_selection": eval_selection_audit}
                if eval_selection_audit is not None
                else {}
            ),
            "gold_output_budget": gold_output_budget,
            "single_turn_rows": single_turn_rows,
            "eval_single_turn_rows": eval_single_turn_rows,
            "eval_source_conversation_rows": full_eval_conversation_rows,
            "eval_source_single_turn_rows": full_eval_single_turn_rows,
            "eval_selected_conversation_rows": len(eval_conversations),
            **(
                {"prompt_sampling": prompt_sampling_contract}
                if prompt_sampling_contract is not None
                else {}
            ),
        },
        "schedule": {
            **(
                {
                    "conversation_prompt_contract": conversation_prompt_contract,
                    "prompt_materialization": "lazy per selected assistant decision",
                    "prompt_truncation": "forbidden",
                    "context_preflight": context_preflight["train"],
                    "schema_validation": "validate_tool_catalog + recursive schema_matches",
                    "output_validation": ("strict parse_tool_output before exact/schema scoring"),
                }
                if conversation_prompt_contract != LEGACY_CONVERSATION_PROMPT_CONTRACT
                else {}
            ),
            "seed": seed,
            "warmup_steps": warmup_steps,
            "prompts_per_step": prompts_per_step,
            "group_size": group_size,
            "max_new_tokens": max_new,
            "temperature": temperature,
            "clip_ratio": clip_ratio,
            "kl_beta": kl_beta,
            "policy_epochs": policy_epochs,
            "format_weight": format_weight,
            "truncation_penalty": truncation_penalty,
            "prompt_sampling": (
                prompt_sampling_contract
                if prompt_sampling_contract is not None
                else {"mode": "iid_with_replacement_v1"}
            ),
        },
        "planned": {
            "accounting_kind": "deterministic_prompt_and_action_token_opportunities",
            "model_forward_token_slot_bounds_kind": (
                "safe cumulative tensor-element bounds across cached rollout generation, "
                "old/reference scoring, and current-policy optimization forwards"
            ),
            "model_forward_token_slot_bound_assumptions": {
                "lower": (
                    "every rollout samples EOS first and every reward group has zero variance"
                ),
                "upper": (
                    "every rollout reaches max_new_tokens without EOS and every reward group is "
                    "informative in every policy epoch"
                ),
                "rollout_forward_slots_per_action": (
                    "prompt_tokens + generated_tokens - sampled_eos_indicator"
                ),
                "scoring_forward_slots_per_action": (
                    "prompt_tokens + generated_tokens - 1"
                ),
                "reference_enabled": kl_beta > 0,
            },
            "scope": "grpo_training_loop_only",
            "generation_dependent_loss_tokens": None,
            "generation_dependent_loss_tokens_reason": (
                "rollout lengths and informative optimizer groups require model generation"
            ),
            "updates": updates,
            "rollout_steps": max_steps,
            "horizon_totals": horizon,
            "heldout_evaluation": heldout_evaluation_bounds,
        },
        "calibration": None,
    }


def build_stage_budget_plan(
    config_path: str | Path,
    *,
    max_steps: int | None = None,
    min_supervised_tokens: int | None = None,
    max_supervised_tokens: int | None = None,
) -> dict[str, Any]:
    """Build and seal a deterministic plan without loading a model or parent checkpoint."""

    config_path = Path(config_path)
    config, model_config, tokenizer, identity = _base_context(config_path)
    stage = str(config.get("stage", ""))
    if stage not in {"midtrain", "sft", "rl"}:
        raise ValueError(f"unsupported stage budget plan {stage!r}")
    default_steps = {"midtrain": 4_000, "sft": 3_000, "rl": 60}[stage]
    configured_steps = int(config.get("schedule", {}).get("total_steps", default_steps))
    horizon_steps = _positive_steps(
        configured_steps if max_steps is None else max_steps,
        label="max_steps",
    )
    minimum = _nonnegative_budget(
        min_supervised_tokens,
        label="min_supervised_tokens",
    )
    maximum = _nonnegative_budget(
        max_supervised_tokens,
        label="max_supervised_tokens",
    )
    builder = {
        "midtrain": _midtrain_plan,
        "sft": _sft_plan,
        "rl": _rl_plan,
    }[stage]
    stage_plan = builder(
        config,
        model_config,
        tokenizer,
        max_steps=horizon_steps,
        min_supervised_tokens=minimum,
        max_supervised_tokens=maximum,
    )
    plan = {
        "kind": PLAN_KIND,
        "schema_version": PLAN_SCHEMA_VERSION,
        "stage": stage,
        "request": {
            "config_path": str(config_path),
            "configured_steps": configured_steps,
            "max_steps": horizon_steps,
            "min_supervised_tokens": minimum,
            "max_supervised_tokens": maximum,
        },
        "identity": identity,
        **stage_plan,
    }
    return seal_stage_budget_plan(plan)


def verify_stage_budget_plan(path: str | Path) -> dict[str, Any]:
    """Verify canonical encoding, self-hash, and a fresh replay against current artifacts."""

    plan_path = Path(path)
    raw = plan_path.read_bytes()
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("stage budget plan is not valid JSON") from exc
    if not isinstance(plan, Mapping):
        raise TypeError("stage budget plan root must be a mapping")
    plan = dict(plan)
    if plan.get("kind") != PLAN_KIND or plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported stage budget plan kind/schema version")
    if raw != canonical_plan_bytes(plan):
        raise ValueError("stage budget plan is not canonical JSON")
    assert_stage_budget_self_hash(plan)
    request = plan.get("request")
    if not isinstance(request, Mapping):
        raise TypeError("stage budget plan request must be a mapping")
    rebuilt = build_stage_budget_plan(
        request["config_path"],
        max_steps=int(request["max_steps"]),
        min_supervised_tokens=request.get("min_supervised_tokens"),
        max_supervised_tokens=request.get("max_supervised_tokens"),
    )
    if canonical_plan_bytes(plan) != canonical_plan_bytes(rebuilt):
        raise ValueError("stage budget plan drifted from current config/tokenizer/data artifacts")
    return plan


def write_stage_budget_plan(path: str | Path, plan: Mapping[str, Any]) -> None:
    """Atomically write an already sealed canonical plan."""

    assert_stage_budget_self_hash(plan)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(canonical_plan_bytes(plan))
    temporary.replace(destination)
