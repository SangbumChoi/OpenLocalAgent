"""GRPO RL fine-tuning (Phase 10, implemented) with a *verifiable* reward.

For each prompt we sample G rollouts, score tool calls by exact normalized AST and text responses
by exact match, normalize advantages within the group, and optimize a clipped old-policy ratio
with an optional reference KL. A small format reward can provide signal before exact success
appears. This offline stage does not execute BrowserGym tasks and uses no learned reward model or
LLM judge.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from localagent.agent.parser import extract_tool_calls
from localagent.agent.schema_decode import validate as validate_arguments
from localagent.data.agent_synth import Sample
from localagent.data.conversation_artifact import (
    audit_conversation_overlap,
    conversation_semantic_sha256,
)
from localagent.data.decision_quota_order import (
    QUOTA_SAMPLING_MODE,
    order_assistant_decisions,
    quota_sampling_contract,
)
from localagent.data.prompt_contract import (
    LEGACY_CONVERSATION_PROMPT_CONTRACT,
    OPENAI_FULL_CATALOG_V1,
    AssistantTrainingTurn,
    FunctionCatalogCache,
    assert_prompt_contract_tokenizer,
    assistant_training_turns,
    resolve_conversation_prompt_contract,
    schema_matches,
    validate_tool_catalog,
)
from localagent.data.render import assistant_body, prompt_text
from localagent.data.schema import Conversation, Role, ToolCall, ToolSpec
from localagent.data.stratified_eval_selector import (
    ALGORITHM as STRATIFIED_EVAL_ALGORITHM,
)
from localagent.data.stratified_eval_selector import select_stratified_eval_subset
from localagent.eval.harness import _correct
from localagent.eval.tool_eval import match_calls, parse_tool_output
from localagent.model.tokenizer import BPE_EOS, batched_token_lengths
from localagent.train.device import autocast_ctx
from localagent.train.loop import cosine_lr, set_lr
from localagent.train.stage_sampling import RLPromptSchedule

_RL_RESUME_FORMAT = "localagent.rl_resume"
_RL_RESUME_VERSION = 1
_RL_RESUME_SEALED_FIELDS = (
    "resume_format",
    "resume_version",
    "cfg",
    "state_dict",
    "optimizer",
    "grad_scaler",
    "reference_state_dict",
    "step",
    "reward_history",
    "rl_accounting",
    "prompt_accounting",
    "prompt_schedule_state",
    "rollout_generator_state",
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


def _rl_digest_chunk(digest: Any, tag: bytes, payload: bytes = b"") -> None:
    digest.update(len(tag).to_bytes(4, "big"))
    digest.update(tag)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _rl_update_resume_digest(digest: Any, value: Any) -> None:
    """Hash nested checkpoint state independently of PyTorch's archive representation."""

    if value is None:
        _rl_digest_chunk(digest, b"none")
    elif isinstance(value, bool):
        _rl_digest_chunk(digest, b"bool", b"1" if value else b"0")
    elif isinstance(value, int):
        _rl_digest_chunk(digest, b"int", str(value).encode("ascii"))
    elif isinstance(value, float):
        _rl_digest_chunk(digest, b"float", value.hex().encode("ascii"))
    elif isinstance(value, str):
        _rl_digest_chunk(digest, b"str", value.encode("utf-8"))
    elif isinstance(value, bytes):
        _rl_digest_chunk(digest, b"bytes", value)
    elif isinstance(value, Path):
        _rl_digest_chunk(digest, b"path", str(value).encode("utf-8"))
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        _rl_digest_chunk(
            digest,
            b"tensor",
            f"{tensor.dtype}:{tuple(tensor.shape)}".encode("ascii"),
        )
        _rl_digest_chunk(
            digest,
            b"tensor-bytes",
            tensor.reshape(-1).view(torch.uint8).numpy().tobytes(),
        )
    elif isinstance(value, Mapping):
        _rl_digest_chunk(digest, b"mapping", str(len(value)).encode("ascii"))
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _rl_update_resume_digest(digest, key)
            _rl_update_resume_digest(digest, value[key])
    elif isinstance(value, Sequence):
        _rl_digest_chunk(digest, b"sequence", str(len(value)).encode("ascii"))
        for item in value:
            _rl_update_resume_digest(digest, item)
    elif hasattr(value, "__dict__"):
        _rl_digest_chunk(
            digest,
            f"object:{type(value).__module__}.{type(value).__qualname__}".encode(),
        )
        _rl_update_resume_digest(digest, vars(value))
    else:
        raise TypeError(f"unsupported RL resume-digest value {type(value).__name__}")


def _rl_resume_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _rl_update_resume_digest(digest, value)
    return digest.hexdigest()


def _rl_sealed_resume_sha256(payload: Mapping[str, Any]) -> str:
    return _rl_resume_sha256({field: payload.get(field) for field in _RL_RESUME_SEALED_FIELDS})


def _rl_valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _rl_model_config_mapping(model) -> dict[str, Any]:
    value = getattr(model.cfg, "__dict__", None)
    if not isinstance(value, Mapping):
        raise TypeError("grpo() model cfg must expose a mapping-compatible __dict__")
    return dict(value)


def _rl_tokenizer_contract(tok) -> dict[str, Any]:
    return {
        "class": f"{type(tok).__module__}.{type(tok).__qualname__}",
        "vocab_size": int(tok.vocab_size),
        "pad_id": int(tok.pad_id),
        "eos_id": int(tok.eos_id),
    }


def _rl_training_data_sha256(
    samples,
    tool_schemas,
    *,
    prompt_contract: str,
    catalog_cache,
) -> str:
    """Bind exact reward rows, decode prompts, and row-local schemas."""

    digest = hashlib.sha256()
    _rl_update_resume_digest(digest, prompt_contract)
    _rl_update_resume_digest(digest, len(samples))
    for index, value in enumerate(samples):
        _rl_update_resume_digest(digest, index)
        if isinstance(value, RLDecision):
            _validate_decision(value)
            sample_identity = {
                "source": value.source,
                "decision_sha256": _decision_fingerprint(value),
                "prompt_sha256": _sha256(_decision_prompt_text(value, catalog_cache)),
                "reward": vars(value.reward),
            }
        else:
            sample_identity = {
                "sample": vars(value),
                "prompt_sha256": _sha256(prompt_text(value)),
            }
        _rl_update_resume_digest(digest, sample_identity)
        if tool_schemas is not None:
            _rl_update_resume_digest(digest, tool_schemas[index])
    return digest.hexdigest()


def _rl_assert_optional_metadata(
    checkpoint: Mapping[str, Any],
    *,
    key: str,
    expected: Mapping[str, Any] | None,
) -> None:
    recorded = checkpoint.get(key)
    if expected is None:
        if recorded is not None:
            raise ValueError(f"RL resume checkpoint records {key} metadata but none was provided")
        return
    if not isinstance(recorded, Mapping):
        raise TypeError(f"RL resume checkpoint has no valid {key} metadata")
    if dict(recorded) != dict(expected):
        raise ValueError(f"RL resume checkpoint {key} metadata mismatch")


def _validated_reward_history(value: Any, *, completed_steps: int) -> list[float]:
    if not isinstance(value, list) or len(value) != completed_steps:
        raise ValueError(
            "RL resume checkpoint reward_history length disagrees with completed steps"
        )
    history = []
    for reward in value:
        if (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(float(reward))
        ):
            raise ValueError("RL resume checkpoint reward_history contains an invalid value")
        history.append(float(reward))
    return history


def _rl_nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"RL resume checkpoint {label} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class RLDecision:
    """One canonical assistant decision without retaining its complete catalog prompt.

    ``conversation`` deliberately remains a reference to the canonical interchange row.  The
    catalog-independent ``turn`` carries the exact history suffix and gold assistant body; the
    compact legacy ``Sample`` is retained only as the deterministic reward target.
    """

    source: str
    conversation_index: int
    message_index: int
    conversation: Conversation
    turn: AssistantTrainingTurn
    reward: Sample

    @property
    def training_turn(self) -> AssistantTrainingTurn:
        """Descriptive alias for callers that do not use the shorter ``turn`` name."""

        return self.turn


class CatalogStringCache:
    """Intern exact rendered catalog strings while keeping decision prompts lazy."""

    def __init__(self) -> None:
        self._entries: dict[str, str] = {}
        self._identity_entries: dict[int, tuple[Sequence[ToolSpec], str]] = {}
        self._catalog_cache = FunctionCatalogCache()
        self._validated_decisions: dict[int, RLDecision] = {}
        self._conversation_sha256: dict[int, tuple[Conversation, str]] = {}
        self._prompt_sha256: dict[int, tuple[RLDecision, str]] = {}

    def text(self, tools: Sequence[ToolSpec]) -> str:
        cacheable = getattr(tools, "_localagent_verified_read_only", False) is True
        key = id(tools)
        if cacheable:
            cached_identity = self._identity_entries.get(key)
            if cached_identity is not None and cached_identity[0] is tools:
                return cached_identity[1]
        catalog = self._catalog_cache.entry(tools).text + BPE_EOS
        fingerprint = _sha256(catalog)
        cached = self._entries.get(fingerprint)
        if cached is not None:
            if cached != catalog:
                raise RuntimeError("function catalog SHA-256 collision")
            result = cached
        else:
            self._entries[fingerprint] = catalog
            result = catalog
        if cacheable:
            self._identity_entries[key] = (tools, result)
        return result

    def registry(self, tools: Sequence[ToolSpec]) -> Mapping[str, ToolSpec]:
        """Return validated tools, reusing only verified immutable catalog objects."""

        return self._catalog_cache.entry(tools).registry

    def training_turns(self, conversation: Conversation) -> tuple[AssistantTrainingTurn, ...]:
        """Validate/render turns while sharing the same immutable catalog cache."""

        return assistant_training_turns(
            conversation,
            catalog_cache=self._catalog_cache,
        )

    def decision_is_validated(self, decision: RLDecision) -> bool:
        """Return true only for the same cached decision object."""

        return self._validated_decisions.get(id(decision)) is decision

    def mark_decision_validated(self, decision: RLDecision) -> None:
        """Cache only decisions whose conversation-owned sequences cannot mutate."""

        conversation = decision.conversation
        if (
            getattr(conversation.messages, "_localagent_verified_read_only", False) is True
            and getattr(conversation.tools, "_localagent_verified_read_only", False) is True
        ):
            self._validated_decisions[id(decision)] = decision

    def conversation_sha256(self, conversation: Conversation) -> str:
        """Hash semantic content once for a verified immutable conversation graph."""

        cacheable = (
            getattr(conversation.messages, "_localagent_verified_read_only", False) is True
            and getattr(conversation.tools, "_localagent_verified_read_only", False) is True
        )
        key = id(conversation)
        if cacheable:
            cached = self._conversation_sha256.get(key)
            if cached is not None and cached[0] is conversation:
                return cached[1]
        result = conversation_semantic_sha256(conversation)
        if cacheable:
            self._conversation_sha256[key] = (conversation, result)
        return result

    def prompt_sha256(self, decision: RLDecision) -> str:
        """Hash one prompt without retaining its full text between audit passes."""

        cacheable = (
            getattr(
                decision.conversation.messages,
                "_localagent_verified_read_only",
                False,
            )
            is True
            and getattr(
                decision.conversation.tools,
                "_localagent_verified_read_only",
                False,
            )
            is True
        )
        key = id(decision)
        if cacheable:
            cached = self._prompt_sha256.get(key)
            if cached is not None and cached[0] is decision:
                return cached[1]
        result = _sha256(_decision_prompt_text(decision, self))
        if cacheable:
            self._prompt_sha256[key] = (decision, result)
        return result

    @property
    def unique_catalogs(self) -> int:
        return len(self._entries)

    @property
    def retained_characters(self) -> int:
        return sum(len(value) for value in self._entries.values())


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _conversation_fingerprint(conversation: Conversation) -> str:
    """Hash semantic row content independently of provenance metadata and JSON key ordering."""

    payload = json.loads(conversation.to_json())
    payload.pop("meta", None)
    canonical = json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
    )
    return _sha256(canonical)


def _fingerprint_set(values: Sequence[str]) -> str:
    """Return an order-independent fingerprint for a multiset of row fingerprints."""

    return _sha256("\n".join(sorted(values)))


def _reward_sample(
    conversation: Conversation,
    turn: AssistantTrainingTurn,
) -> Sample:
    """Build the compact exact-reward view for one already validated assistant turn."""

    message = conversation.messages[turn.message_index]
    category = str(conversation.meta.get("category", conversation.meta.get("kind", "unknown")))
    group = str(conversation.meta.get("group", "tool_call" if message.tool_calls else "text"))
    if not message.tool_calls:
        sample = Sample(
            category=category,
            group=group,
            prompt="",
            kind="text",
            target=message.content,
        )
    else:
        calls = [
            {"name": call.name, "arguments": dict(call.arguments)} for call in message.tool_calls
        ]
        first = calls[0]
        sample = Sample(
            category=category,
            group=group,
            prompt="",
            kind="tool",
            target=json.dumps(first, separators=(",", ":"), sort_keys=True),
            ref_name=first["name"],
            ref_args=json.dumps(first["arguments"], separators=(",", ":"), sort_keys=True),
            calls=calls if len(calls) > 1 else None,
        )
    if assistant_body(sample) != turn.body:
        raise RuntimeError(
            "RL reward projection disagrees with the shared assistant-body contract: "
            f"assistant_message_index={turn.message_index}"
        )
    return sample


def project_rl_decisions(
    conversations: Sequence[Conversation],
    *,
    sources: Sequence[str | Path] | None = None,
) -> tuple[RLDecision, ...]:
    """Project every assistant turn in canonical file/turn order for full-contract RL."""

    if isinstance(conversations, (str, bytes)) or not isinstance(conversations, Sequence):
        raise TypeError("conversations must be a sequence of Conversation values")
    if sources is not None and (
        isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence)
    ):
        raise TypeError("sources must be a sequence aligned with conversations")
    if sources is not None and len(sources) != len(conversations):
        raise ValueError("sources must align one-to-one with conversations")
    decisions: list[RLDecision] = []
    catalog_cache = FunctionCatalogCache()
    for conversation_index, conversation in enumerate(conversations):
        source = "" if sources is None else str(sources[conversation_index])
        turns = assistant_training_turns(
            conversation,
            catalog_cache=catalog_cache,
        )
        for turn in turns:
            decisions.append(
                RLDecision(
                    source=source,
                    conversation_index=conversation_index,
                    message_index=turn.message_index,
                    conversation=conversation,
                    turn=turn,
                    reward=_reward_sample(conversation, turn),
                )
            )
    return tuple(decisions)


def decision_keys_to_prompt_order(
    decisions: Sequence[RLDecision],
    ordered_keys: Sequence[tuple[int, int]],
) -> tuple[int, ...]:
    """Map canonical assistant-decision keys to projected RL prompt-row indices."""

    natural_keys = tuple(
        (decision.conversation_index, decision.message_index) for decision in decisions
    )
    if len(set(natural_keys)) != len(natural_keys):
        raise RuntimeError("projected RL assistant-decision keys are not unique")
    normalized_keys = tuple(ordered_keys)
    if len(normalized_keys) != len(natural_keys):
        raise ValueError("RL decision order must contain every prompt row exactly once")
    if len(set(normalized_keys)) != len(normalized_keys):
        raise ValueError("RL decision order contains duplicate prompt keys")
    if set(normalized_keys) != set(natural_keys):
        raise ValueError("RL decision order does not match projected prompt rows")
    prompt_by_key = {key: index for index, key in enumerate(natural_keys)}
    return tuple(prompt_by_key[key] for key in normalized_keys)


# Kept as a private spelling for stage-local callers and focused contract tests.
_project_rl_decisions = project_rl_decisions


def _validate_decision(
    decision: RLDecision,
    catalog_cache: CatalogStringCache | None = None,
) -> None:
    """Fail closed if a purported frozen decision no longer matches its conversation."""

    if not isinstance(decision, RLDecision):
        raise TypeError(
            "openai_full_catalog_v1 RL requires RLDecision inputs projected from Conversations"
        )
    if decision.conversation_index < 0:
        raise ValueError("RLDecision conversation_index must be non-negative")
    if decision.message_index != decision.turn.message_index:
        raise ValueError("RLDecision message_index disagrees with its AssistantTrainingTurn")
    if not 0 <= decision.message_index < len(decision.conversation.messages):
        raise ValueError("RLDecision message_index is outside its Conversation")
    message = decision.conversation.messages[decision.message_index]
    if message.role != Role.assistant:
        raise ValueError("RLDecision must reference an assistant message")
    cache = catalog_cache if catalog_cache is not None else CatalogStringCache()
    if cache.decision_is_validated(decision):
        if assistant_body(decision.reward) != decision.turn.body:
            raise ValueError(
                "RLDecision reward target disagrees with its AssistantTrainingTurn body"
            )
        return
    matching_turns = tuple(
        turn
        for turn in cache.training_turns(decision.conversation)
        if turn.message_index == decision.message_index
    )
    if len(matching_turns) != 1 or matching_turns[0] != decision.turn:
        raise ValueError(
            "RLDecision AssistantTrainingTurn no longer matches its current Conversation "
            "prompt_suffix/body/index"
        )
    registry = cache.registry(decision.conversation.tools)
    for call in message.tool_calls:
        spec = registry.get(call.name)
        if spec is None:
            raise ValueError(f"RLDecision assistant message references unknown tool {call.name!r}")
        if not schema_matches(call.arguments, spec.parameters):
            raise ValueError(f"RLDecision assistant arguments violate the schema for {call.name!r}")
    if assistant_body(decision.reward) != decision.turn.body:
        raise ValueError("RLDecision reward target disagrees with its AssistantTrainingTurn body")
    cache.mark_decision_validated(decision)


def _decision_prompt_text(
    decision: RLDecision,
    catalog_cache: CatalogStringCache | None = None,
) -> str:
    """Materialize one exact full-contract decode prompt only for immediate use."""

    cache = catalog_cache if catalog_cache is not None else CatalogStringCache()
    _validate_decision(decision, cache)
    return cache.text(decision.conversation.tools) + decision.turn.prompt_suffix


def _decision_fingerprint(
    decision: RLDecision,
    catalog_cache: CatalogStringCache | None = None,
) -> str:
    """Hash semantic conversation content together with the scored assistant position."""

    return _sha256(
        json.dumps(
            {
                "conversation_sha256": (
                    catalog_cache.conversation_sha256(decision.conversation)
                    if catalog_cache is not None
                    else conversation_semantic_sha256(decision.conversation)
                ),
                "message_index": decision.message_index,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _full_contract_split_audit(
    train_conversations: Sequence[Conversation],
    eval_conversations: Sequence[Conversation],
    train_decisions: Sequence[RLDecision],
    eval_decisions: Sequence[RLDecision],
    *,
    catalog_cache: CatalogStringCache,
) -> dict[str, Any]:
    """Audit exact full-contract prompts without retaining prompt text or token lists."""

    for label, conversations, decisions in (
        ("train", train_conversations, train_decisions),
        ("eval", eval_conversations, eval_decisions),
    ):
        expected = [
            (conversation_index, turn.message_index, conversation)
            for conversation_index, conversation in enumerate(conversations)
            for turn in catalog_cache.training_turns(conversation)
        ]
        if len(expected) != len(decisions):
            raise RuntimeError(
                f"RL {label} assistant-decision projection does not align with raw conversations"
            )
        for decision, (conversation_index, message_index, conversation) in zip(
            decisions,
            expected,
            strict=True,
        ):
            if (
                decision.conversation_index != conversation_index
                or decision.message_index != message_index
                or decision.conversation is not conversation
            ):
                raise RuntimeError(
                    f"RL {label} assistant-decision projection does not align with raw "
                    "conversations"
                )
    for decision in chain(train_decisions, eval_decisions):
        _validate_decision(decision, catalog_cache)
    train_rows = [
        _decision_fingerprint(decision, catalog_cache) for decision in train_decisions
    ]
    eval_rows = [
        _decision_fingerprint(decision, catalog_cache) for decision in eval_decisions
    ]
    row_overlap = sorted(set(train_rows) & set(eval_rows))
    if row_overlap:
        raise ValueError(
            "RL train/eval contamination: "
            f"{len(row_overlap)} semantic conversation + assistant message index "
            "fingerprint(s) overlap"
        )

    train_prompts = [catalog_cache.prompt_sha256(decision) for decision in train_decisions]
    eval_prompts = [catalog_cache.prompt_sha256(decision) for decision in eval_decisions]
    prompt_overlap = sorted(set(train_prompts) & set(eval_prompts))
    if prompt_overlap:
        raise ValueError(
            "RL train/eval contamination: "
            f"{len(prompt_overlap)} exact openai_full_catalog_v1 prompt fingerprint(s) overlap"
        )

    return {
        "fingerprint": (
            "sha256(canonical semantic conversation sha256 + assistant message index); "
            "sha256(exact openai_full_catalog_v1 decode prompt excluding current gold); "
            "order-independent multiset aggregation"
        ),
        "conversation_prompt_contract": OPENAI_FULL_CATALOG_V1,
        "dataset_fingerprint": (
            "shared conversation_semantic_sha256 once per raw conversation row"
        ),
        "train_dataset_sha256": _fingerprint_set(
            [
                catalog_cache.conversation_sha256(conversation)
                for conversation in train_conversations
            ]
        ),
        "eval_dataset_sha256": _fingerprint_set(
            [
                catalog_cache.conversation_sha256(conversation)
                for conversation in eval_conversations
            ]
        ),
        "train_scored_rows_sha256": _fingerprint_set(train_rows),
        "eval_scored_rows_sha256": _fingerprint_set(eval_rows),
        "train_scored_prompts_sha256": _fingerprint_set(train_prompts),
        "eval_scored_prompts_sha256": _fingerprint_set(eval_prompts),
        "train_scored_rows": len(train_decisions),
        "eval_scored_rows": len(eval_decisions),
        "train_dataset_rows": len(train_conversations),
        "eval_dataset_rows": len(eval_conversations),
        "row_overlap": 0,
        "prompt_overlap": 0,
        "current_gold_in_prompt_fingerprint": False,
        "catalog_cache": {
            "unique_catalogs": catalog_cache.unique_catalogs,
            "retained_characters": catalog_cache.retained_characters,
            "retains_complete_prompts": False,
            "retains_prompt_token_ids": False,
        },
    }


def _audit_data_splits(
    train_conversations: Sequence[Conversation],
    eval_conversations: Sequence[Conversation],
    train_samples,
    eval_samples,
    *,
    conversation_prompt_contract: str | None = None,
    catalog_cache: CatalogStringCache | None = None,
) -> dict:
    """Fail closed on held-out contamination and describe the frozen split identities."""

    prompt_contract = resolve_conversation_prompt_contract(conversation_prompt_contract)
    if prompt_contract == OPENAI_FULL_CATALOG_V1:
        cache = catalog_cache if catalog_cache is not None else CatalogStringCache()
        return _full_contract_split_audit(
            train_conversations,
            eval_conversations,
            train_samples,
            eval_samples,
            catalog_cache=cache,
        )

    conversation_overlap_audit = audit_conversation_overlap(
        train_conversations,
        eval_conversations,
    )
    if conversation_overlap_audit.semantic_overlap_sha256:
        raise ValueError(
            "RL train/eval contamination: "
            f"{len(conversation_overlap_audit.semantic_overlap_sha256)} "
            "canonical conversation row fingerprint(s) overlap"
        )
    if conversation_overlap_audit.rendered_prompt_overlap_sha256:
        raise ValueError(
            "RL train/eval contamination: "
            f"{len(conversation_overlap_audit.rendered_prompt_overlap_sha256)} "
            "rendered assistant prompt fingerprint(s) overlap"
        )

    train_rows = [_conversation_fingerprint(row) for row in train_conversations]
    eval_rows = [_conversation_fingerprint(row) for row in eval_conversations]

    train_scored_rows = [
        _conversation_fingerprint(row)
        for row in train_conversations
        if (
            len(row.messages) == 2
            and row.messages[0].role == Role.user
            and row.messages[1].role == Role.assistant
        )
    ]
    eval_scored_rows = [
        _conversation_fingerprint(row)
        for row in eval_conversations
        if (
            len(row.messages) == 2
            and row.messages[0].role == Role.user
            and row.messages[1].role == Role.assistant
        )
    ]
    if train_conversations and len(train_scored_rows) != len(train_samples):
        raise RuntimeError("RL train scored-row projection does not align with samples")
    if eval_conversations and len(eval_scored_rows) != len(eval_samples):
        raise RuntimeError("RL eval scored-row projection does not align with samples")

    train_prompt_rows = [_sha256(prompt_text(sample)) for sample in train_samples]
    eval_prompt_rows = [_sha256(prompt_text(sample)) for sample in eval_samples]
    return {
        "fingerprint": (
            "sha256(canonical_json_row); sha256(rendered_single_turn_prompt); "
            "order-independent multiset aggregation"
        ),
        "train_dataset_sha256": _fingerprint_set(train_rows),
        "eval_dataset_sha256": _fingerprint_set(eval_rows),
        "train_scored_rows_sha256": _fingerprint_set(train_scored_rows),
        "eval_scored_rows_sha256": _fingerprint_set(eval_scored_rows),
        "train_scored_prompts_sha256": _fingerprint_set(train_prompt_rows),
        "eval_scored_prompts_sha256": _fingerprint_set(eval_prompt_rows),
        "train_scored_rows": len(train_scored_rows),
        "eval_scored_rows": len(eval_scored_rows),
        "row_overlap": 0,
        "prompt_overlap": 0,
        "conversation_overlap_audit": conversation_overlap_audit.as_dict(),
    }


def _reward_target(value) -> Sample:
    if isinstance(value, RLDecision):
        return value.reward
    return value


def _assert_gold_outputs_fit(
    samples,
    tok,
    max_new: int,
    *,
    split: str,
    catalog_cache: CatalogStringCache | None = None,
) -> dict[str, int]:
    """Fail before rollout when exact-reward targets cannot fit the decoding budget."""

    rows = 0
    embedded_eos = 0
    longest = 0
    too_long = 0
    cache = catalog_cache if catalog_cache is not None else CatalogStringCache()
    for value in samples:
        if isinstance(value, RLDecision):
            _validate_decision(value, cache)
        body = tok.encode(assistant_body(_reward_target(value)))
        rows += 1
        embedded_eos += int(tok.eos_id in body)
        length = len(body) + 1
        longest = max(longest, length)
        too_long += int(length > max_new)
    if embedded_eos:
        raise ValueError(
            f"RL {split} has {embedded_eos} gold output(s) containing an embedded EOS token; "
            "exact reward would decode only the prefix before the terminal target"
        )
    if too_long:
        raise ValueError(
            f"RL {split} has {too_long} gold output(s) longer than max_new_tokens={max_new}; "
            f"longest requires {longest} tokens including EOS"
        )
    return {
        "rows": rows,
        "max_gold_tokens_including_eos": longest,
        "decoding_budget": max_new,
    }


def _preflight_full_context(
    decisions: Sequence[RLDecision],
    tok,
    *,
    max_new: int,
    max_seq_len: int,
    split: str,
    catalog_cache: CatalogStringCache | None = None,
    prompt_lengths: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Measure exact prompts and reject overflow; full-contract prompts are never sliced."""

    if max_new < 1 or max_new >= max_seq_len:
        raise ValueError("max_new must be in [1, model.max_seq_len)")
    if not decisions:
        raise ValueError(f"RL {split} has no assistant decisions")
    assert_prompt_contract_tokenizer(tok, OPENAI_FULL_CATALOG_V1)
    supplied_prompt_lengths = prompt_lengths is not None
    if prompt_lengths is not None and len(prompt_lengths) != len(decisions):
        raise ValueError("prompt_lengths must align one-to-one with RL decisions")
    cache = catalog_cache if catalog_cache is not None else CatalogStringCache()
    if prompt_lengths is None:
        prompt_lengths = batched_token_lengths(
            tok,
            (_decision_prompt_text(decision, cache) for decision in decisions),
        )
    rows = 0
    min_prompt: int | None = None
    max_prompt = 0
    min_reserve: int | None = None
    max_reserve = 0
    over_limit = 0
    first_over_limit: tuple[RLDecision, int] | None = None
    for index, decision in enumerate(decisions):
        if supplied_prompt_lengths:
            _validate_decision(decision, cache)
        prompt_length = prompt_lengths[index]
        if (
            isinstance(prompt_length, bool)
            or not isinstance(prompt_length, int)
            or prompt_length < 1
        ):
            raise ValueError("prompt_lengths must contain positive integers")
        reserve = max_seq_len - prompt_length
        rows += 1
        min_prompt = prompt_length if min_prompt is None else min(min_prompt, prompt_length)
        max_prompt = max(max_prompt, prompt_length)
        min_reserve = reserve if min_reserve is None else min(min_reserve, reserve)
        max_reserve = max(max_reserve, reserve)
        if prompt_length + max_new > max_seq_len:
            over_limit += 1
            if first_over_limit is None:
                first_over_limit = (decision, prompt_length)
    if over_limit:
        assert first_over_limit is not None
        first, first_length = first_over_limit
        raise ValueError(
            f"RL {split} openai_full_catalog_v1 context exceeds max_seq_len and cannot be "
            f"truncated: {over_limit} decision(s), source={first.source!r}, "
            f"conversation_index={first.conversation_index}, "
            f"assistant_message_index={first.message_index}, prompt_tokens={first_length}, "
            f"max_new_tokens={max_new}, max_seq_len={max_seq_len}"
        )
    return {
        "rows": rows,
        "min_prompt_tokens": int(min_prompt or 0),
        "max_prompt_tokens": max_prompt,
        "required_decode_reserve_tokens": max_new,
        "min_available_decode_reserve_tokens": int(min_reserve or 0),
        "max_available_decode_reserve_tokens": max_reserve,
        "max_seq_len": max_seq_len,
        "prompt_plus_reserve_max_tokens": max_prompt + max_new,
        "truncation": "forbidden",
        "truncated_rows": 0,
        "preflight_before_model_load": True,
    }


def _assert_parent_prompt_contract(
    checkpoint: Mapping[str, Any],
    conversation_prompt_contract: str | None,
) -> str:
    """Require the RL policy parent to use the same prompt contract (missing means legacy)."""

    expected = resolve_conversation_prompt_contract(conversation_prompt_contract)
    parent = resolve_conversation_prompt_contract(checkpoint.get("conversation_prompt_contract"))
    if parent != expected:
        raise ValueError(
            "RL parent checkpoint conversation_prompt_contract mismatch: "
            f"configured={expected!r}, parent={parent!r}"
        )
    return parent


def _single_turn_tool_schemas(
    conversations: Sequence[Conversation],
) -> list[tuple[ToolSpec, ...]]:
    """Keep each projected sample paired with the registry embedded in its source row."""

    return [
        tuple(conversation.tools)
        for conversation in conversations
        if (
            len(conversation.messages) == 2
            and conversation.messages[0].role == Role.user
            and conversation.messages[1].role == Role.assistant
        )
    ]


def _correct_for_contract(
    sample: Sample,
    text: str,
    *,
    conversation_prompt_contract: str | None = None,
) -> bool:
    """Apply strict full-contract output parsing while preserving the legacy evaluator."""

    prompt_contract = resolve_conversation_prompt_contract(conversation_prompt_contract)
    if prompt_contract == LEGACY_CONVERSATION_PROMPT_CONTRACT:
        return bool(_correct(sample, text))

    parsed = parse_tool_output(text)
    if not parsed.format_valid:
        return False
    if sample.kind == "tool":
        if not parsed.tool_syntax_present or not parsed.calls:
            return False
        if sample.calls:
            expected = [ToolCall(**call) for call in sample.calls]
        else:
            expected = [ToolCall(**json.loads(sample.target))]
        return match_calls(list(parsed.calls), expected)
    return not parsed.tool_syntax_present and text == sample.target


def _valid_tool_call_format(
    text: str,
    tool_specs: Sequence[ToolSpec] | None = None,
    *,
    conversation_prompt_contract: str | None = None,
) -> bool:
    """Validate parsed tool-call ASTs, consulting the row's registry when it is available."""

    prompt_contract = resolve_conversation_prompt_contract(conversation_prompt_contract)
    if prompt_contract == OPENAI_FULL_CATALOG_V1:
        parsed = parse_tool_output(text)
        if not parsed.format_valid or not parsed.tool_syntax_present or not parsed.calls:
            return False
        calls = parsed.calls
        registry = validate_tool_catalog(
            () if tool_specs is None else tool_specs,
            label="RL reward function catalog",
        )
        return all(
            call.name in registry and schema_matches(call.arguments, registry[call.name].parameters)
            for call in calls
        )
    calls = extract_tool_calls(text)
    if not calls:
        return False
    if any(
        not isinstance(call.name, str) or not call.name or not isinstance(call.arguments, dict)
        for call in calls
    ):
        return False
    if not tool_specs:
        return True

    registry = {spec.name: spec for spec in tool_specs}
    for call in calls:
        spec = registry.get(call.name)
        if spec is None or not validate_arguments(call.arguments, spec.parameters or {}):
            return False
    return True


@torch.no_grad()
def _rollout(
    model,
    tok,
    prompt_ids,
    max_new,
    temperature,
    device,
    generator=None,
    amp_dtype=torch.float32,
):
    ids = list(prompt_ids)
    caches = [None] * model.n_cache_slots()
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with autocast_ctx(torch.device(device), amp_dtype):
        logits, _, caches = model(x, pos=0, caches=caches)
    pos = len(ids)
    gen = []
    for _ in range(max_new):
        if temperature == 0:
            nxt = int(logits[0, -1].float().argmax())
        else:
            probs = F.softmax(logits[0, -1].float() / temperature, dim=-1)
            nxt = int(torch.multinomial(probs, 1, generator=generator))
        # Keep EOS in the sampled action so GRPO can reinforce when to stop. ``tok.decode`` strips
        # it for reward evaluation, while ``_token_logprobs`` includes its policy probability.
        gen.append(nxt)
        if nxt == tok.eos_id:
            break
        step = torch.tensor([[nxt]], dtype=torch.long, device=device)
        with autocast_ctx(torch.device(device), amp_dtype):
            logits, _, caches = model(step, pos=pos, caches=caches)
        pos += 1
    return gen


def _token_logprobs(
    model,
    prompt_ids,
    gen_ids,
    device,
    amp_dtype=torch.float32,
    temperature: float = 1.0,
):
    """Return log probabilities for sampled generation tokens only.

    Prompt tokens condition the policy but never enter the returned loss vector. A sampled EOS is
    part of ``gen_ids`` and is therefore included. Empty generations return an empty vector so the
    caller can guard them without invoking the model on a zero-length sequence.
    """
    if not prompt_ids:
        raise ValueError("prompt_ids must contain at least one token")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not gen_ids:
        return torch.empty(0, dtype=torch.float32, device=device)
    full = torch.tensor([prompt_ids + gen_ids], dtype=torch.long, device=device)
    with autocast_ctx(torch.device(device), amp_dtype):
        logits, _ = model(full[:, :-1])
    logp = F.log_softmax(logits[0].float() / temperature, dim=-1)
    targets = full[0, 1:]
    positions = torch.arange(targets.shape[0], device=targets.device)
    tok_lp = logp[positions, targets]
    start = len(prompt_ids) - 1  # first position predicting a gen token
    return tok_lp[start:]


def _grpo_token_loss(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    advantage: torch.Tensor | float,
    *,
    clip_ratio: float,
    kl_beta: float,
) -> torch.Tensor:
    """Clipped GRPO surrogate and sampled reference KL, averaged over generated tokens.

    The importance ratio must be formed independently for every sampled token. Exponentiating a
    sequence-mean log-ratio would instead optimize a geometric-mean sequence ratio and could hide
    large, opposing token-level policy changes.
    """
    if current_logprobs.ndim != 1 or current_logprobs.numel() == 0:
        raise ValueError("GRPO needs a non-empty one-dimensional generation log-prob vector")
    if old_logprobs.shape != current_logprobs.shape:
        raise ValueError("old-policy log-probs must match current-policy log-probs")
    if reference_logprobs.shape != current_logprobs.shape:
        raise ValueError("reference-policy log-probs must match current-policy log-probs")
    if clip_ratio < 0 or clip_ratio >= 1:
        raise ValueError("clip_ratio must be in [0, 1)")
    if kl_beta < 0:
        raise ValueError("kl_beta must be non-negative")

    old_logprobs = old_logprobs.detach()
    reference_logprobs = reference_logprobs.detach()
    advantage_t = torch.as_tensor(
        advantage,
        dtype=current_logprobs.dtype,
        device=current_logprobs.device,
    )
    if advantage_t.numel() != 1:
        raise ValueError("advantage must be scalar for one sampled sequence")

    token_ratio = torch.exp(current_logprobs - old_logprobs)
    unclipped = token_ratio * advantage_t
    clipped = torch.clamp(token_ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage_t
    token_loss = -torch.minimum(unclipped, clipped)
    if kl_beta > 0:
        # k3 estimator for KL(policy || reference) on the sampled token. Rollouts are generated
        # by the old policy, so after the first update this is the usual GRPO off-policy estimate.
        log_reference_ratio = reference_logprobs - current_logprobs
        sampled_kl = torch.exp(log_reference_ratio) - 1 - log_reference_ratio
        token_loss = token_loss + kl_beta * sampled_kl
    return token_loss.mean()


def _rollout_reward(
    sample,
    text: str,
    *,
    format_weight: float,
    truncated: bool,
    truncation_penalty: float = 0.05,
    tool_specs: Sequence[ToolSpec] | None = None,
    conversation_prompt_contract: str | None = None,
    reward_environment: str = "canonical_toolcalls",
) -> float:
    if reward_environment == "stateful_productivity":
        reward = _stateful_productivity_reward(sample, text, tool_specs)
    else:
        if reward_environment != "canonical_toolcalls":
            raise ValueError(f"unsupported RL reward environment: {reward_environment!r}")
        reward = float(
            _correct_for_contract(
                sample,
                text,
                conversation_prompt_contract=conversation_prompt_contract,
            )
        )
        if sample.kind == "tool" and _valid_tool_call_format(
            text,
            tool_specs,
            conversation_prompt_contract=conversation_prompt_contract,
        ):
            reward += format_weight
    if truncated:
        reward -= truncation_penalty
    return reward


def _stateful_productivity_reward(sample, text: str, tool_specs: Sequence[ToolSpec] | None) -> float:
    """Return shaped one-step reward for the deterministic local productivity fixture.

    This is deliberately opt-in.  The normal RL stage keeps exact canonical-toolcall rewards;
    this branch gives the local email/Notion/browser simulation enough schema/tool/argument signal
    to produce informative GRPO groups before exact calls are learned.
    """

    if sample.kind != "tool":
        return 1.0 if text == sample.target else 0.0
    parsed = parse_tool_output(text)
    calls = extract_tool_calls(text)
    if not calls:
        return 0.0
    call = calls[0]
    exact_tool = call.name == sample.ref_name
    try:
        expected_args = json.loads(sample.ref_args or "{}")
    except json.JSONDecodeError:
        expected_args = {}
    exact_args = json.dumps(call.arguments, sort_keys=True, separators=(",", ":")) == json.dumps(
        expected_args, sort_keys=True, separators=(",", ":")
    )
    registry = {spec.name: spec for spec in (tool_specs or ())}
    spec = registry.get(call.name)
    schema_valid = bool(spec and validate_arguments(call.arguments, spec.parameters or {}))
    state_transition = exact_tool and exact_args
    # Reserve a small term for a complete, strict envelope.  This is intentionally weaker than
    # tool/argument/transition correctness so malformed or unknown calls cannot look successful,
    # while still giving the policy a learnable distinction from plain malformed text.
    envelope_valid = bool(parsed.format_valid and parsed.tool_syntax_present and parsed.calls)
    return float(
        0.10 * envelope_valid
        + 0.10 * schema_valid
        + 0.20 * exact_tool
        + 0.20 * exact_args
        + 0.25 * state_transition
        + 0.15 * state_transition
    )


def _prompt_ids_for_policy(
    value,
    tok,
    *,
    max_prompt: int,
    conversation_prompt_contract: str,
    catalog_cache: CatalogStringCache | None,
) -> list[int]:
    """Encode one selected policy prompt, preserving legacy tail slicing exactly."""

    if conversation_prompt_contract == LEGACY_CONVERSATION_PROMPT_CONTRACT:
        return tok.encode(prompt_text(value))[-max_prompt:]
    if not isinstance(value, RLDecision):
        raise TypeError(
            "openai_full_catalog_v1 RL requires RLDecision inputs projected from Conversations"
        )
    prompt_ids = tok.encode(_decision_prompt_text(value, catalog_cache))
    if len(prompt_ids) > max_prompt:
        raise ValueError(
            "openai_full_catalog_v1 prompt exceeds the reserved context and cannot be truncated: "
            f"source={value.source!r}, conversation_index={value.conversation_index}, "
            f"assistant_message_index={value.message_index}, prompt_tokens={len(prompt_ids)}, "
            f"prompt_budget={max_prompt}"
        )
    return prompt_ids


def _tool_specs_for_policy(
    value,
    *,
    conversation_prompt_contract: str,
    legacy_specs: Sequence[ToolSpec],
) -> Sequence[ToolSpec]:
    if conversation_prompt_contract == OPENAI_FULL_CATALOG_V1:
        if not isinstance(value, RLDecision):
            raise TypeError(
                "openai_full_catalog_v1 RL requires RLDecision inputs projected from Conversations"
            )
        return value.conversation.tools
    return legacy_specs


@torch.no_grad()
def _evaluate_holdout(
    model,
    samples,
    tok,
    *,
    max_new: int,
    device: str,
    format_weight: float,
    truncation_penalty: float,
    tool_schemas: Sequence[Sequence[ToolSpec]] | None = None,
    amp_dtype: torch.dtype = torch.float32,
    conversation_prompt_contract: str | None = None,
    catalog_cache: CatalogStringCache | None = None,
    reward_environment: str = "canonical_toolcalls",
) -> dict:
    """Greedily score a frozen split; no sampling or training RNG is consumed."""

    prompt_contract = resolve_conversation_prompt_contract(conversation_prompt_contract)
    if not samples:
        raise ValueError("held-out RL evaluation needs at least one exact-reward sample")
    if max_new < 1 or max_new >= model.cfg.max_seq_len:
        raise ValueError("max_new must be in [1, model.max_seq_len)")
    if (
        prompt_contract == LEGACY_CONVERSATION_PROMPT_CONTRACT
        and tool_schemas is not None
        and len(tool_schemas) != len(samples)
    ):
        raise ValueError("tool_schemas must align one-to-one with held-out samples")
    if prompt_contract == OPENAI_FULL_CATALOG_V1:
        if tool_schemas is not None:
            raise ValueError(
                "full-contract held-out evaluation reads schemas from RLDecision conversations"
            )
        assert_prompt_contract_tokenizer(tok, prompt_contract)
        _assert_gold_outputs_fit(samples, tok, max_new, split="held-out split")
        _preflight_full_context(
            samples,
            tok,
            max_new=max_new,
            max_seq_len=model.cfg.max_seq_len,
            split="held-out split",
            catalog_cache=catalog_cache,
        )

    model.to(device).eval()
    exact = 0
    reward_total = 0.0
    truncated_rows = 0
    tool_rows = 0
    text_rows = 0
    tool_exact = 0
    text_exact = 0
    valid_tool_formats = 0
    schema_covered_tool_rows = 0
    max_prompt = model.cfg.max_seq_len - max_new
    for index, sample in enumerate(samples):
        legacy_specs = tool_schemas[index] if tool_schemas is not None else ()
        specs = _tool_specs_for_policy(
            sample,
            conversation_prompt_contract=prompt_contract,
            legacy_specs=legacy_specs,
        )
        prompt_ids = _prompt_ids_for_policy(
            sample,
            tok,
            max_prompt=max_prompt,
            conversation_prompt_contract=prompt_contract,
            catalog_cache=catalog_cache,
        )
        rollout = _rollout(
            model,
            tok,
            prompt_ids,
            max_new,
            0.0,
            device,
            amp_dtype=amp_dtype,
        )
        text = tok.decode(rollout)
        truncated = len(rollout) == max_new and rollout[-1] != tok.eos_id
        reward_sample = _reward_target(sample)
        ok = int(
            _correct_for_contract(
                reward_sample,
                text,
                conversation_prompt_contract=prompt_contract,
            )
        )
        exact += ok
        truncated_rows += int(truncated)
        reward_total += _rollout_reward(
            reward_sample,
            text,
            format_weight=format_weight,
            truncated=truncated,
            truncation_penalty=truncation_penalty,
            tool_specs=specs,
            conversation_prompt_contract=prompt_contract,
            reward_environment=reward_environment,
        )
        if reward_sample.kind == "tool":
            tool_rows += 1
            tool_exact += ok
            valid_tool_formats += int(
                _valid_tool_call_format(
                    text,
                    specs,
                    conversation_prompt_contract=prompt_contract,
                )
            )
            schema_covered_tool_rows += int(bool(specs))
        else:
            text_rows += 1
            text_exact += ok

    return {
        "n": len(samples),
        "exact_match_accuracy": exact / len(samples),
        "mean_reward": reward_total / len(samples),
        "tool_rows": tool_rows,
        "text_rows": text_rows,
        "tool_exact_match_accuracy": tool_exact / tool_rows if tool_rows else None,
        "text_exact_match_accuracy": text_exact / text_rows if text_rows else None,
        "tool_format_valid_rate": valid_tool_formats / tool_rows if tool_rows else None,
        "schema_covered_tool_rows": schema_covered_tool_rows,
        "truncated_rows": truncated_rows,
    }


def grpo(
    model,
    samples,
    tok,
    *,
    steps=60,
    prompts_per_step=8,
    group_size=4,
    lr=2e-4,
    warmup_steps=5,
    temperature=1.0,
    max_new=64,
    device="cpu",
    log=print,
    seed=0,
    clip_ratio=0.2,
    kl_beta=0.02,
    policy_epochs=1,
    format_weight=0.1,
    truncation_penalty=0.05,
    tool_schemas=None,
    amp_dtype=torch.float32,
    return_metrics=False,
    conversation_prompt_contract=None,
    catalog_cache: CatalogStringCache | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 0,
    resume_from: str | Path | None = None,
    lineage: Mapping[str, Any] | None = None,
    tokenizer_metadata: Mapping[str, Any] | None = None,
    data_metadata: Mapping[str, Any] | None = None,
    execution: Mapping[str, Any] | None = None,
    heldout_baseline: Mapping[str, Any] | None = None,
    prompt_order: Sequence[int] | None = None,
    prompt_sampling_contract: Mapping[str, Any] | None = None,
    execution_rollout_step_limit: int | None = None,
    reward_environment: str = "canonical_toolcalls",
):
    """Group-relative policy optimization against deterministic exact-target rewards.

    ``policy_epochs>1`` reuses sampled rollouts and makes clipping active; one epoch is the
    low-memory REINFORCE-like setting. KL is estimated on sampled actions against a frozen copy
    of the starting policy. With ``checkpoint_path`` set, saves happen atomically only after a
    complete rollout step and all policy epochs. ``resume_from`` requires the same fixed horizon,
    reward/prompt/data contract, parent policy, and execution environment.
    """
    prompt_contract = resolve_conversation_prompt_contract(conversation_prompt_contract)
    if reward_environment not in {"canonical_toolcalls", "stateful_productivity"}:
        raise ValueError(f"unsupported RL reward environment: {reward_environment!r}")
    if group_size < 2:
        raise ValueError("group_size must be >= 2")
    if policy_epochs < 1:
        raise ValueError("policy_epochs must be >= 1")
    if steps < 1 or prompts_per_step < 1:
        raise ValueError("steps and prompts_per_step must be positive")
    if execution_rollout_step_limit is None:
        execution_rollout_steps = steps
    elif (
        isinstance(execution_rollout_step_limit, bool)
        or not isinstance(execution_rollout_step_limit, int)
        or execution_rollout_step_limit < 1
        or execution_rollout_step_limit > steps
    ):
        raise ValueError("execution_rollout_step_limit must be in [1, steps]")
    else:
        execution_rollout_steps = execution_rollout_step_limit
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be non-negative")
    if not samples:
        raise ValueError("grpo() needs at least one exact-reward sample")
    if (
        prompt_contract == LEGACY_CONVERSATION_PROMPT_CONTRACT
        and tool_schemas is not None
        and len(tool_schemas) != len(samples)
    ):
        raise ValueError("tool_schemas must align one-to-one with training samples")
    if prompt_contract == OPENAI_FULL_CATALOG_V1 and tool_schemas is not None:
        raise ValueError("full-contract GRPO reads schemas from RLDecision conversations")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if clip_ratio < 0 or clip_ratio >= 1:
        raise ValueError("clip_ratio must be in [0, 1)")
    if kl_beta < 0:
        raise ValueError("kl_beta must be non-negative")
    if format_weight < 0 or truncation_penalty < 0:
        raise ValueError("reward weights must be non-negative")
    if max_new < 1 or max_new >= model.cfg.max_seq_len:
        raise ValueError("max_new must be in [1, model.max_seq_len)")
    if prompt_order is not None:
        if not isinstance(prompt_sampling_contract, Mapping):
            raise TypeError("prompt_order requires prompt_sampling_contract metadata")
        required_prompts = steps * prompts_per_step
        if required_prompts > len(prompt_order):
            raise ValueError(
                "quota no-replacement RL horizon exceeds the available prompt rows: "
                f"required={required_prompts}, available={len(prompt_order)}"
            )
    elif prompt_sampling_contract is not None:
        raise ValueError("prompt_sampling_contract requires prompt_order")
    if prompt_contract == OPENAI_FULL_CATALOG_V1:
        assert_prompt_contract_tokenizer(tok, prompt_contract)
        _assert_gold_outputs_fit(samples, tok, max_new, split="training split")
        _preflight_full_context(
            samples,
            tok,
            max_new=max_new,
            max_seq_len=model.cfg.max_seq_len,
            split="training split",
            catalog_cache=catalog_cache,
        )
    torch.manual_seed(seed)
    model.to(device)
    device_obj = torch.device(device)
    exact_resume_enabled = checkpoint_path is not None or resume_from is not None
    if exact_resume_enabled and device_obj.type not in {
        "cpu",
        "cuda",
        "mps",
        "xpu",
    }:
        raise ValueError(
            "exact RL resume supports CPU, CUDA, MPS, and XPU RNG state only; "
            f"got {device_obj.type!r}"
        )
    initial_model_sha256 = _rl_resume_sha256(model.state_dict()) if exact_resume_enabled else None
    use_grad_scaler = device_obj.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    reference = copy.deepcopy(model).to(device).eval() if kl_beta > 0 else None
    if reference is not None:
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
    prompt_schedule = RLPromptSchedule(
        len(samples),
        prompts_per_step,
        seed=seed,
        prompt_order=prompt_order,
    )
    device_type = torch.device(device).type
    generator = torch.Generator(device=device) if device_type in {"cpu", "cuda"} else None
    if generator is not None:
        generator.manual_seed(seed)
    else:
        torch.manual_seed(seed)
    training_contract = {
        "version": 1,
        "steps": int(steps),
        "prompts_per_step": int(prompts_per_step),
        "group_size": int(group_size),
        "lr": float(lr),
        "warmup_steps": int(warmup_steps),
        "temperature": float(temperature),
        "max_new": int(max_new),
        "clip_ratio": float(clip_ratio),
        "kl_beta": float(kl_beta),
        "policy_epochs": int(policy_epochs),
        "format_weight": float(format_weight),
        "truncation_penalty": float(truncation_penalty),
        "amp_dtype": str(amp_dtype),
        "seed": int(seed),
        "prompt_sampling": (
            dict(prompt_sampling_contract)
            if prompt_sampling_contract is not None
            else {"mode": "iid_with_replacement_v1"}
        ),
        "conversation_prompt_contract": prompt_contract,
        "tokenizer": _rl_tokenizer_contract(tok),
        "training_data_sha256": (
            _rl_training_data_sha256(
                samples,
                tool_schemas,
                prompt_contract=prompt_contract,
                catalog_cache=catalog_cache,
            )
            if exact_resume_enabled
            else None
        ),
        "initial_model_sha256": initial_model_sha256,
        "optimizer": {
            "kind": "AdamW",
            "betas": [0.9, 0.95],
            "weight_decay": 0.01,
            "grad_clip": 1.0,
            "warmup_steps": int(warmup_steps),
            "lr_schedule": "cosine",
            "min_lr_ratio": 0.1,
        },
        "policy_contract": {
            "objective": "sampled_token_clipped_grpo",
            "ratio_scope": "generated_tokens_only",
            "reference_kl": "sampled_token_k3",
            "includes_sampled_eos": True,
        },
    }
    hist: list[float] = []
    attempted_rollout_steps = 0
    zero_signal_steps = 0
    informative_groups = 0
    realized_optimizer_updates = 0
    informative_steps = 0
    sample_draws = [0] * len(samples)
    selected_prompt_tokens = 0
    rollout_prompt_tokens = 0
    generated_tokens = 0
    generated_eos_tokens = 0
    truncated_rollouts = 0
    reward_value_counts: dict[str, int] = {}
    parser_format_valid_rollouts = 0
    complete_parser_format_valid_rollouts = 0
    parser_tool_syntax_rollouts = 0
    exact_reward_success_rollouts = 0
    tool_reward_rollouts = 0
    text_reward_rollouts = 0
    strict_tool_format_valid_rollouts = 0
    informative_scoring_input_slots = 0
    start_step = 0

    def current_rl_accounting() -> dict[str, int]:
        return {
            "attempted_rollout_steps": attempted_rollout_steps,
            "attempted_groups": attempted_rollout_steps * prompts_per_step,
            "attempted_rollouts": attempted_rollout_steps * prompts_per_step * group_size,
            "zero_signal_steps": zero_signal_steps,
            "informative_groups": informative_groups,
            "realized_optimizer_updates": realized_optimizer_updates,
            "policy_epochs_per_informative_batch": policy_epochs,
        }

    def current_prompt_accounting() -> dict[str, Any]:
        phases = {
            "rollout_prefill": rollout_prompt_tokens,
            "rollout_cached_decode": generated_tokens - generated_eos_tokens,
            "old_policy_scoring": informative_scoring_input_slots,
            "reference_policy_scoring": (
                informative_scoring_input_slots if reference is not None else 0
            ),
            "current_policy_optimization": (
                informative_scoring_input_slots * policy_epochs
            ),
        }
        return {
            "selected_prompts": sum(sample_draws),
            "selected_prompt_tokens": selected_prompt_tokens,
            "rollout_prompt_tokens": rollout_prompt_tokens,
            "generated_tokens": generated_tokens,
            "generated_eos_tokens": generated_eos_tokens,
            "truncated_rollouts": truncated_rollouts,
            "reward_value_counts": dict(sorted(reward_value_counts.items())),
            "parser_format_valid_rollouts": parser_format_valid_rollouts,
            "complete_parser_format_valid_rollouts": complete_parser_format_valid_rollouts,
            "parser_tool_syntax_rollouts": parser_tool_syntax_rollouts,
            "exact_reward_success_rollouts": exact_reward_success_rollouts,
            "tool_reward_rollouts": tool_reward_rollouts,
            "text_reward_rollouts": text_reward_rollouts,
            "strict_tool_format_valid_rollouts": strict_tool_format_valid_rollouts,
            "informative_steps": informative_steps,
            "informative_scoring_input_slots": informative_scoring_input_slots,
            "model_forward_token_slots": {
                "phases": phases,
                "total": sum(phases.values()),
            },
            "sample_draws": list(sample_draws),
        }

    if resume_from is not None:
        from localagent.train.stage_data import assert_resume_lineage

        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, Mapping):
            raise ValueError("RL resume checkpoint root must be a mapping")
        required = {*_RL_RESUME_SEALED_FIELDS, "resume_integrity_sha256"}
        missing = sorted(required - set(checkpoint))
        if missing:
            raise ValueError("RL resume checkpoint is incomplete; missing: " + ", ".join(missing))
        recorded_integrity = checkpoint.get("resume_integrity_sha256")
        if not _rl_valid_sha256(recorded_integrity):
            raise ValueError("RL resume checkpoint integrity digest is invalid")
        if recorded_integrity != _rl_sealed_resume_sha256(checkpoint):
            raise ValueError("RL resume checkpoint integrity mismatch")
        if (
            checkpoint.get("resume_format") != _RL_RESUME_FORMAT
            or checkpoint.get("resume_version") != _RL_RESUME_VERSION
        ):
            raise ValueError("RL resume checkpoint format/version is unsupported")
        if checkpoint.get("stage") != "rl":
            raise ValueError("RL resume checkpoint stage must be 'rl'")
        if checkpoint.get("cfg") != _rl_model_config_mapping(model):
            raise ValueError("RL resume checkpoint model config mismatch")
        if checkpoint.get("training_seed") != seed:
            raise ValueError("RL resume checkpoint training seed mismatch")
        if checkpoint.get("training_contract") != training_contract:
            raise ValueError("RL resume checkpoint training contract mismatch")
        if checkpoint.get("conversation_prompt_contract") != prompt_contract:
            raise ValueError("RL resume checkpoint conversation prompt contract mismatch")

        recorded_lineage = checkpoint.get("lineage")
        if lineage is not None:
            assert_resume_lineage(checkpoint, lineage)
        elif recorded_lineage is not None:
            raise ValueError(
                "RL resume checkpoint records lineage but no expected lineage was provided"
            )
        _rl_assert_optional_metadata(
            checkpoint,
            key="tokenizer",
            expected=tokenizer_metadata,
        )
        _rl_assert_optional_metadata(checkpoint, key="data", expected=data_metadata)
        _rl_assert_optional_metadata(checkpoint, key="execution", expected=execution)
        _rl_assert_optional_metadata(
            checkpoint,
            key="heldout_baseline",
            expected=heldout_baseline,
        )

        state = checkpoint.get("state_dict")
        optimizer_state = checkpoint.get("optimizer")
        if not isinstance(state, Mapping):
            raise ValueError("RL resume checkpoint state_dict is invalid")
        if not isinstance(optimizer_state, Mapping):
            raise ValueError("RL resume checkpoint optimizer state is invalid")
        recorded_reference = checkpoint.get("reference_state_dict")
        if reference is not None:
            if not isinstance(recorded_reference, Mapping):
                raise ValueError("RL resume checkpoint frozen reference policy is missing")
        elif recorded_reference is not None:
            raise ValueError("RL resume checkpoint has an unexpected frozen reference policy")
        recorded_scaler = checkpoint.get("grad_scaler")
        if use_grad_scaler:
            if not isinstance(recorded_scaler, Mapping):
                raise ValueError("RL resume checkpoint gradient-scaler state is missing")
        elif recorded_scaler is not None:
            raise ValueError("RL resume checkpoint has unexpected gradient-scaler state")

        checkpoint_step = checkpoint.get("step")
        if (
            isinstance(checkpoint_step, bool)
            or not isinstance(checkpoint_step, int)
            or checkpoint_step < 0
        ):
            raise ValueError("RL resume checkpoint step is invalid")
        start_step = checkpoint_step + 1
        if start_step > steps:
            raise ValueError(
                f"RL resume checkpoint is already at step {checkpoint_step}, "
                f"beyond total steps {steps}"
            )
        hist = _validated_reward_history(
            checkpoint.get("reward_history"),
            completed_steps=start_step,
        )

        recorded_schedule = checkpoint.get("prompt_schedule_state")
        if not isinstance(recorded_schedule, Mapping):
            raise ValueError("RL resume checkpoint prompt schedule state is invalid")
        replay_schedule = RLPromptSchedule(
            len(samples),
            prompts_per_step,
            seed=seed,
            prompt_order=prompt_order,
        )
        expected_draws = [0] * len(samples)
        expected_prompt_tokens = 0
        max_prompt = model.cfg.max_seq_len - max_new
        for replay_step in range(start_step):
            replay_indices = replay_schedule.indices_for_step(replay_step)
            for sample_index in replay_indices:
                expected_draws[sample_index] += 1
                expected_prompt_tokens += len(
                    _prompt_ids_for_policy(
                        samples[sample_index],
                        tok,
                        max_prompt=max_prompt,
                        conversation_prompt_contract=prompt_contract,
                        catalog_cache=catalog_cache,
                    )
                )
        expected_schedule_state = {
            "rng_state": replay_schedule.rng.getstate(),
            "next_step": start_step,
        }
        if dict(recorded_schedule) != expected_schedule_state:
            raise ValueError("RL resume checkpoint prompt schedule state mismatch")

        recorded_prompt_accounting = checkpoint.get("prompt_accounting")
        if not isinstance(recorded_prompt_accounting, Mapping):
            raise ValueError("RL resume checkpoint prompt accounting is invalid")
        expected_prompt_fields = {
            "selected_prompts": start_step * prompts_per_step,
            "selected_prompt_tokens": expected_prompt_tokens,
            "rollout_prompt_tokens": expected_prompt_tokens * group_size,
            "sample_draws": expected_draws,
        }
        if any(
            recorded_prompt_accounting.get(key) != value
            for key, value in expected_prompt_fields.items()
        ):
            raise ValueError("RL resume checkpoint prompt accounting mismatch")
        generated_tokens = _rl_nonnegative_int(
            recorded_prompt_accounting.get("generated_tokens"),
            label="prompt_accounting.generated_tokens",
        )
        generated_eos_tokens = _rl_nonnegative_int(
            recorded_prompt_accounting.get("generated_eos_tokens"),
            label="prompt_accounting.generated_eos_tokens",
        )
        truncated_rollouts = _rl_nonnegative_int(
            recorded_prompt_accounting.get("truncated_rollouts"),
            label="prompt_accounting.truncated_rollouts",
        )
        recorded_reward_counts = recorded_prompt_accounting.get("reward_value_counts")
        if not isinstance(recorded_reward_counts, Mapping):
            raise ValueError("RL resume checkpoint reward value counts are invalid")
        reward_value_counts = {}
        for reward_hex, count in recorded_reward_counts.items():
            if not isinstance(reward_hex, str):
                raise ValueError("RL resume checkpoint reward value key must be text")
            try:
                reward_value = float.fromhex(reward_hex)
            except ValueError as exc:
                raise ValueError(
                    "RL resume checkpoint reward value key is not a hexadecimal float"
                ) from exc
            if not math.isfinite(reward_value) or reward_value.hex() != reward_hex:
                raise ValueError("RL resume checkpoint reward value key is not canonical")
            reward_value_counts[reward_hex] = _rl_nonnegative_int(
                count,
                label=f"prompt_accounting.reward_value_counts[{reward_hex!r}]",
            )
        parser_format_valid_rollouts = _rl_nonnegative_int(
            recorded_prompt_accounting.get("parser_format_valid_rollouts"),
            label="prompt_accounting.parser_format_valid_rollouts",
        )
        complete_parser_format_valid_rollouts = _rl_nonnegative_int(
            recorded_prompt_accounting.get("complete_parser_format_valid_rollouts"),
            label="prompt_accounting.complete_parser_format_valid_rollouts",
        )
        parser_tool_syntax_rollouts = _rl_nonnegative_int(
            recorded_prompt_accounting.get("parser_tool_syntax_rollouts"),
            label="prompt_accounting.parser_tool_syntax_rollouts",
        )
        exact_reward_success_rollouts = _rl_nonnegative_int(
            recorded_prompt_accounting.get("exact_reward_success_rollouts"),
            label="prompt_accounting.exact_reward_success_rollouts",
        )
        tool_reward_rollouts = _rl_nonnegative_int(
            recorded_prompt_accounting.get("tool_reward_rollouts"),
            label="prompt_accounting.tool_reward_rollouts",
        )
        text_reward_rollouts = _rl_nonnegative_int(
            recorded_prompt_accounting.get("text_reward_rollouts"),
            label="prompt_accounting.text_reward_rollouts",
        )
        strict_tool_format_valid_rollouts = _rl_nonnegative_int(
            recorded_prompt_accounting.get("strict_tool_format_valid_rollouts"),
            label="prompt_accounting.strict_tool_format_valid_rollouts",
        )
        informative_steps = _rl_nonnegative_int(
            recorded_prompt_accounting.get("informative_steps"),
            label="prompt_accounting.informative_steps",
        )
        informative_scoring_input_slots = _rl_nonnegative_int(
            recorded_prompt_accounting.get("informative_scoring_input_slots"),
            label="prompt_accounting.informative_scoring_input_slots",
        )
        attempted_rollouts = start_step * prompts_per_step * group_size
        scoring_upper = group_size * (
            expected_prompt_tokens + start_step * prompts_per_step * (max_new - 1)
        )
        if (
            generated_tokens > attempted_rollouts * max_new
            or generated_eos_tokens > attempted_rollouts
            or truncated_rollouts > attempted_rollouts
            or sum(reward_value_counts.values()) != attempted_rollouts
            or parser_format_valid_rollouts > attempted_rollouts
            or complete_parser_format_valid_rollouts > parser_format_valid_rollouts
            or parser_tool_syntax_rollouts > attempted_rollouts
            or exact_reward_success_rollouts > attempted_rollouts
            or tool_reward_rollouts + text_reward_rollouts != attempted_rollouts
            or strict_tool_format_valid_rollouts > tool_reward_rollouts
            or informative_steps > start_step
            or informative_scoring_input_slots > scoring_upper
        ):
            raise ValueError("RL resume checkpoint prompt accounting bounds are invalid")
        expected_forward_slots = {
            "rollout_prefill": expected_prompt_tokens * group_size,
            "rollout_cached_decode": generated_tokens - generated_eos_tokens,
            "old_policy_scoring": informative_scoring_input_slots,
            "reference_policy_scoring": (
                informative_scoring_input_slots if reference is not None else 0
            ),
            "current_policy_optimization": informative_scoring_input_slots * policy_epochs,
        }
        if recorded_prompt_accounting.get("model_forward_token_slots") != {
            "phases": expected_forward_slots,
            "total": sum(expected_forward_slots.values()),
        }:
            raise ValueError("RL resume checkpoint forward-slot accounting mismatch")
        sample_draws = expected_draws
        selected_prompt_tokens = expected_prompt_tokens
        rollout_prompt_tokens = expected_prompt_tokens * group_size

        recorded_accounting = checkpoint.get("rl_accounting")
        if not isinstance(recorded_accounting, Mapping):
            raise ValueError("RL resume checkpoint RL accounting is invalid")
        attempted_rollout_steps = _rl_nonnegative_int(
            recorded_accounting.get("attempted_rollout_steps"),
            label="rl_accounting.attempted_rollout_steps",
        )
        attempted_groups = _rl_nonnegative_int(
            recorded_accounting.get("attempted_groups"),
            label="rl_accounting.attempted_groups",
        )
        recorded_attempted_rollouts = _rl_nonnegative_int(
            recorded_accounting.get("attempted_rollouts"),
            label="rl_accounting.attempted_rollouts",
        )
        zero_signal_steps = _rl_nonnegative_int(
            recorded_accounting.get("zero_signal_steps"),
            label="rl_accounting.zero_signal_steps",
        )
        informative_groups = _rl_nonnegative_int(
            recorded_accounting.get("informative_groups"),
            label="rl_accounting.informative_groups",
        )
        realized_optimizer_updates = _rl_nonnegative_int(
            recorded_accounting.get("realized_optimizer_updates"),
            label="rl_accounting.realized_optimizer_updates",
        )
        if (
            attempted_rollout_steps != start_step
            or attempted_groups != start_step * prompts_per_step
            or recorded_attempted_rollouts != attempted_rollouts
            or zero_signal_steps + informative_steps != start_step
            or informative_groups < informative_steps
            or informative_groups > attempted_groups
            or realized_optimizer_updates > informative_steps * policy_epochs
            or recorded_accounting.get("policy_epochs_per_informative_batch") != policy_epochs
        ):
            raise ValueError("RL resume checkpoint RL accounting mismatch")

        recorded_generator_state = checkpoint.get("rollout_generator_state")
        if generator is not None:
            if (
                not isinstance(recorded_generator_state, torch.Tensor)
                or recorded_generator_state.dtype != torch.uint8
                or recorded_generator_state.ndim != 1
            ):
                raise ValueError("RL resume checkpoint rollout generator state is invalid")
        elif recorded_generator_state is not None:
            raise ValueError("RL resume checkpoint has unexpected rollout generator state")
        recorded_torch_rng = checkpoint.get("torch_rng_state")
        if (
            not isinstance(recorded_torch_rng, torch.Tensor)
            or recorded_torch_rng.dtype != torch.uint8
            or recorded_torch_rng.ndim != 1
        ):
            raise ValueError("RL resume checkpoint Torch RNG state is invalid")
        recorded_cuda_rng = checkpoint.get("cuda_rng_state_all")
        recorded_mps_rng = checkpoint.get("mps_rng_state")
        recorded_xpu_rng = checkpoint.get("xpu_rng_state_all")
        if device_obj.type == "cuda":
            if not isinstance(recorded_cuda_rng, (list, tuple)):
                raise ValueError("RL resume checkpoint CUDA RNG state is missing")
            if recorded_mps_rng is not None or recorded_xpu_rng is not None:
                raise ValueError("RL resume checkpoint has unexpected accelerator RNG state")
        elif device_obj.type == "mps":
            if not isinstance(recorded_mps_rng, torch.Tensor):
                raise ValueError("RL resume checkpoint MPS RNG state is missing")
            if recorded_cuda_rng is not None or recorded_xpu_rng is not None:
                raise ValueError("RL resume checkpoint has unexpected accelerator RNG state")
        elif device_obj.type == "xpu":
            if not isinstance(recorded_xpu_rng, (list, tuple)):
                raise ValueError("RL resume checkpoint XPU RNG state is missing")
            if recorded_cuda_rng is not None or recorded_mps_rng is not None:
                raise ValueError("RL resume checkpoint has unexpected accelerator RNG state")
        elif any(
            value is not None for value in (recorded_cuda_rng, recorded_mps_rng, recorded_xpu_rng)
        ):
            raise ValueError("RL resume checkpoint has unexpected accelerator RNG state")

        model.load_state_dict(state)
        opt.load_state_dict(optimizer_state)
        if reference is not None:
            reference.load_state_dict(recorded_reference)
        if use_grad_scaler:
            scaler.load_state_dict(recorded_scaler)
        prompt_schedule.rng.setstate(recorded_schedule["rng_state"])
        prompt_schedule.next_step = int(recorded_schedule["next_step"])
        if generator is not None:
            generator.set_state(recorded_generator_state.cpu())
        torch.set_rng_state(recorded_torch_rng.cpu())
        if device_obj.type == "cuda":
            torch.cuda.set_rng_state_all(recorded_cuda_rng)
        elif device_obj.type == "mps":
            torch.mps.set_rng_state(recorded_mps_rng.cpu())
        elif device_obj.type == "xpu":
            torch.xpu.set_rng_state_all(recorded_xpu_rng)

    if (
        execution_rollout_step_limit is not None
        and start_step >= execution_rollout_steps
    ):
        raise ValueError(
            "RL execution rollout-step limit must exceed the completed resume prefix: "
            f"completed={start_step}, limit={execution_rollout_steps}"
        )
    learning_rate_history = [
        float(cosine_lr(step, steps, lr, warmup_steps, 0.1))
        for step in range(start_step)
    ]

    def save(step: int) -> None:
        if checkpoint_path is None:
            return
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if device_obj.type == "mps":
            torch.mps.synchronize()
        payload = {
            "resume_format": _RL_RESUME_FORMAT,
            "resume_version": _RL_RESUME_VERSION,
            "cfg": _rl_model_config_mapping(model),
            "state_dict": model.state_dict(),
            "optimizer": opt.state_dict(),
            "grad_scaler": scaler.state_dict() if use_grad_scaler else None,
            "reference_state_dict": (reference.state_dict() if reference is not None else None),
            "step": step,
            "reward_history": hist,
            "rl_accounting": current_rl_accounting(),
            "prompt_accounting": current_prompt_accounting(),
            "prompt_schedule_state": {
                "rng_state": prompt_schedule.rng.getstate(),
                "next_step": int(prompt_schedule.next_step),
            },
            "rollout_generator_state": (generator.get_state() if generator is not None else None),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if device_obj.type == "cuda" else None
            ),
            "mps_rng_state": (torch.mps.get_rng_state() if device_obj.type == "mps" else None),
            "xpu_rng_state_all": (
                torch.xpu.get_rng_state_all() if device_obj.type == "xpu" else None
            ),
            "stage": "rl",
            "training_seed": seed,
            "training_contract": training_contract,
            "lineage": dict(lineage) if lineage is not None else None,
            "conversation_prompt_contract": prompt_contract,
            "tokenizer": (dict(tokenizer_metadata) if tokenizer_metadata is not None else None),
            "data": dict(data_metadata) if data_metadata is not None else None,
            "execution": dict(execution) if execution is not None else None,
            "heldout_baseline": (dict(heldout_baseline) if heldout_baseline is not None else None),
        }
        payload["resume_integrity_sha256"] = _rl_sealed_resume_sha256(payload)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

    for step in range(start_step, execution_rollout_steps):
        attempted_rollout_steps += 1
        scheduled_lr = float(cosine_lr(step, steps, lr, warmup_steps, 0.1))
        set_lr(opt, scheduled_lr)
        applied_lrs = [float(group["lr"]) for group in opt.param_groups]
        if not applied_lrs or any(
            applied_lr != scheduled_lr for applied_lr in applied_lrs
        ):
            raise RuntimeError("RL optimizer learning rate drifted from the production schedule")
        learning_rate_history.append(applied_lrs[0])
        batch_indices = prompt_schedule.indices_for_step(step)
        groups = []
        rewards_log = []
        for sample_index in batch_indices:
            sample_draws[sample_index] += 1
            s = samples[sample_index]
            legacy_specs = tool_schemas[sample_index] if tool_schemas is not None else ()
            specs = _tool_specs_for_policy(
                s,
                conversation_prompt_contract=prompt_contract,
                legacy_specs=legacy_specs,
            )
            max_prompt = model.cfg.max_seq_len - max_new
            pid = _prompt_ids_for_policy(
                s,
                tok,
                max_prompt=max_prompt,
                conversation_prompt_contract=prompt_contract,
                catalog_cache=catalog_cache,
            )
            selected_prompt_tokens += len(pid)
            rollout_prompt_tokens += len(pid) * group_size
            model.eval()
            rollouts = [
                _rollout(
                    model,
                    tok,
                    pid,
                    max_new,
                    temperature,
                    device,
                    generator,
                    amp_dtype,
                )
                for _ in range(group_size)
            ]
            generated_tokens += sum(len(rollout) for rollout in rollouts)
            generated_eos_tokens += sum(
                bool(rollout and rollout[-1] == tok.eos_id) for rollout in rollouts
            )
            rollout_texts = [tok.decode(rollout) for rollout in rollouts]
            rollout_truncated = [
                len(rollout) == max_new and rollout[-1] != tok.eos_id
                for rollout in rollouts
            ]
            truncated_rollouts += sum(rollout_truncated)
            reward_sample = _reward_target(s)
            reward_values = [
                _rollout_reward(
                    reward_sample,
                    text,
                    format_weight=format_weight,
                    truncated=truncated,
                    truncation_penalty=truncation_penalty,
                    tool_specs=specs,
                    conversation_prompt_contract=prompt_contract,
                    reward_environment=reward_environment,
                )
                for text, truncated in zip(rollout_texts, rollout_truncated)
            ]
            for text, truncated, reward_value in zip(
                rollout_texts,
                rollout_truncated,
                reward_values,
            ):
                reward_hex = float(reward_value).hex()
                reward_value_counts[reward_hex] = reward_value_counts.get(reward_hex, 0) + 1
                parsed = parse_tool_output(text)
                parser_format_valid_rollouts += int(parsed.format_valid)
                complete_parser_format_valid_rollouts += int(
                    parsed.format_valid and not truncated
                )
                parser_tool_syntax_rollouts += int(parsed.tool_syntax_present)
                exact_reward_success_rollouts += int(
                    _correct_for_contract(
                        reward_sample,
                        text,
                        conversation_prompt_contract=prompt_contract,
                    )
                )
                if reward_sample.kind == "tool":
                    tool_reward_rollouts += 1
                    strict_tool_format_valid_rollouts += int(
                        _valid_tool_call_format(
                            text,
                            specs,
                            conversation_prompt_contract=prompt_contract,
                        )
                    )
                else:
                    text_reward_rollouts += 1
            rewards = torch.tensor(reward_values, device=device)
            rewards_log.append(rewards.mean().item())
            reward_std = rewards.std(unbiased=False)
            if reward_std < 1e-6:
                continue
            informative_scoring_input_slots += sum(
                len(pid) + len(rollout) - 1 for rollout in rollouts
            )
            adv = (rewards - rewards.mean()) / (reward_std + 1e-6)
            scored = []
            with torch.no_grad():
                for rollout, advantage in zip(rollouts, adv):
                    if not rollout:
                        continue
                    old_lp = _token_logprobs(
                        model,
                        pid,
                        rollout,
                        device,
                        amp_dtype,
                        temperature,
                    )
                    ref_lp = (
                        _token_logprobs(
                            reference,
                            pid,
                            rollout,
                            device,
                            amp_dtype,
                            temperature,
                        )
                        if reference is not None
                        else old_lp
                    )
                    scored.append((rollout, advantage.detach(), old_lp.detach(), ref_lp.detach()))
            if scored:
                groups.append((pid, scored))

        informative_groups += len(groups)
        if not groups:
            zero_signal_steps += 1
        else:
            informative_steps += 1
        for _ in range(policy_epochs):
            if not groups:
                break
            # Keep dropout disabled so current-policy log-probs use the same policy that sampled
            # the rollouts. eval() does not disable gradients.
            model.eval()
            opt.zero_grad(set_to_none=True)
            n_scored = sum(len(scored) for _, scored in groups)
            for pid, scored in groups:
                for rollout, advantage, old_lp, ref_lp in scored:
                    current_lp = _token_logprobs(
                        model,
                        pid,
                        rollout,
                        device,
                        amp_dtype,
                        temperature,
                    )
                    loss = _grpo_token_loss(
                        current_lp,
                        old_lp,
                        ref_lp,
                        advantage,
                        clip_ratio=clip_ratio,
                        kl_beta=kl_beta,
                    )
                    # Backward one rollout at a time so a paper-scale batch does not retain every
                    # transformer activation graph simultaneously.
                    scaler.scale(loss / n_scored).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if not use_grad_scaler or scaler.get_scale() >= scale_before:
                realized_optimizer_updates += 1
        avg_r = sum(rewards_log) / max(1, len(rewards_log))
        hist.append(avg_r)
        if step % max(1, steps // 6) == 0 or step == steps - 1:
            log(
                f"  [grpo] step {step:3d}/{steps}  mean_reward {avg_r:.3f}  "
                f"informative_groups={len(groups)}/{len(batch_indices)}"
            )
        if checkpoint_every and (step + 1) % checkpoint_every == 0:
            save(step)
    save(execution_rollout_steps - 1)
    metrics = current_rl_accounting()
    prompt_metrics = current_prompt_accounting()
    metrics.update(
        {
            "generated_tokens": prompt_metrics["generated_tokens"],
            "generated_eos_tokens": prompt_metrics["generated_eos_tokens"],
            "truncated_rollouts": prompt_metrics["truncated_rollouts"],
            "informative_scoring_input_slots": prompt_metrics[
                "informative_scoring_input_slots"
            ],
            "model_forward_token_slots": prompt_metrics["model_forward_token_slots"],
            "learning_rate_history": learning_rate_history,
            "fixed_horizon_progress": {
                "planned_rollout_steps": steps,
                "completed_rollout_steps": attempted_rollout_steps,
                "execution_rollout_step_limit": execution_rollout_steps,
                "bounded_prefix": execution_rollout_steps < steps,
            },
            "rollout_observability": {
                "reward": {
                    "distribution": [
                        {
                            "reward": float.fromhex(reward_hex),
                            "reward_hex": reward_hex,
                            "count": count,
                        }
                        for reward_hex, count in sorted(
                            prompt_metrics["reward_value_counts"].items(),
                            key=lambda item: float.fromhex(item[0]),
                        )
                    ],
                    "unique_values": len(prompt_metrics["reward_value_counts"]),
                    "exact_success_rollouts": prompt_metrics[
                        "exact_reward_success_rollouts"
                    ],
                },
                "parsing": {
                    "parser_format_valid_rollouts": prompt_metrics[
                        "parser_format_valid_rollouts"
                    ],
                    "complete_parser_format_valid_rollouts": prompt_metrics[
                        "complete_parser_format_valid_rollouts"
                    ],
                    "parser_tool_syntax_rollouts": prompt_metrics[
                        "parser_tool_syntax_rollouts"
                    ],
                    "tool_reward_rollouts": prompt_metrics["tool_reward_rollouts"],
                    "text_reward_rollouts": prompt_metrics["text_reward_rollouts"],
                    "strict_tool_format_valid_rollouts": prompt_metrics[
                        "strict_tool_format_valid_rollouts"
                    ],
                },
                "truncation": {
                    "truncated_rollouts": prompt_metrics["truncated_rollouts"],
                },
                "tokens": {
                    "selected_prompt_tokens": prompt_metrics["selected_prompt_tokens"],
                    "rollout_prompt_tokens": prompt_metrics["rollout_prompt_tokens"],
                    "generated_tokens": prompt_metrics["generated_tokens"],
                    "generated_eos_tokens": prompt_metrics["generated_eos_tokens"],
                    "model_forward_token_slots": prompt_metrics[
                        "model_forward_token_slots"
                    ],
                },
            },
        }
    )
    if prompt_sampling_contract is not None:
        metrics["prompt_sampling"] = training_contract["prompt_sampling"]
    if prompt_contract != LEGACY_CONVERSATION_PROMPT_CONTRACT:
        metrics["conversation_prompt_contract"] = prompt_contract
        metrics["prompt_truncation"] = "forbidden"
    return (hist, metrics) if return_metrics else hist


def run(
    config_path: str,
    *,
    resume: bool | None = None,
    _execution_rollout_step_limit: int | None = None,
    _expected_config_canonical_sha256: str | None = None,
    _expected_parent_checkpoint_sha256: str | None = None,
    _expected_execution: Mapping[str, str] | None = None,
    _require_fresh_output_dir: bool = False,
) -> None:
    """Run verifiable GRPO; ``runtime.resume`` continues ``log.out_dir/latest.pt`` exactly."""

    import yaml

    from localagent.model import LocalAgentLM, ModelConfig
    from localagent.model.tokenizer import load_tokenizer
    from localagent.train.device import execution_metadata, resolve_device, resolve_dtype
    from localagent.train.stage_data import (
        build_stage_lineage,
        canonical_sha256,
        load_conversation_source,
        load_stage_parent_checkpoint,
        single_turn_samples,
        tokenizer_identity,
    )

    config = yaml.safe_load(Path(config_path).read_text())
    if _expected_config_canonical_sha256 is not None:
        if canonical_sha256(config) != _expected_config_canonical_sha256:
            raise ValueError("RL config canonical SHA-256 does not match guarded readiness")
    if config.get("stage", "rl") != "rl":
        raise ValueError(f"expected stage='rl', got {config.get('stage')!r}")
    environment = config.get("environment", {})
    environment_name = environment.get("name", "canonical_toolcalls")
    if environment_name not in {"canonical_toolcalls", "stateful_productivity"}:
        raise NotImplementedError(
            "TODO(phase-10): only deterministic canonical_toolcalls and local stateful_productivity "
            "rewards are wired; "
            f"requested {environment_name!r}"
        )
    if environment.get("learned_judge", False):
        raise NotImplementedError(
            "TODO(phase-10): learned reward judges are intentionally not wired"
        )

    cfg = ModelConfig.from_yaml(config["model_config"])
    cfg.assert_within_budget()
    data_cfg = config["data"]
    conversation_prompt_contract = resolve_conversation_prompt_contract(
        data_cfg.get("conversation_prompt_contract")
    )
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
    raw_eval_conversations = data_cfg.get("eval_conversations")
    if not raw_eval_conversations:
        raise ValueError(
            "RL requires explicit data.eval_conversations for frozen held-out evaluation"
        )
    eval_conversation_specs = source_specs(
        raw_eval_conversations,
        label="data.eval_conversations",
    )
    loaded_conversation_sources = [
        load_conversation_source(
            source,
            require_verified=strict_conversation_artifacts,
            expected_split="train",
        )
        for source in conversation_specs
    ]
    loaded_eval_sources = [
        load_conversation_source(
            source,
            require_verified=strict_conversation_artifacts,
            expected_split="eval",
        )
        for source in eval_conversation_specs
    ]
    conversation_paths = [source.path for source in loaded_conversation_sources]
    eval_conversation_paths = [source.path for source in loaded_eval_sources]
    conversations: list[Conversation] = []
    conversation_sources: list[str] = []
    for source in loaded_conversation_sources:
        source_rows = list(source.conversations)
        conversations.extend(source_rows)
        conversation_sources.extend([str(source.path)] * len(source_rows))
    eval_conversations: list[Conversation] = []
    eval_conversation_sources: list[str] = []
    for source in loaded_eval_sources:
        source_rows = list(source.conversations)
        eval_conversations.extend(source_rows)
        eval_conversation_sources.extend([str(source.path)] * len(source_rows))
    full_eval_conversation_rows = len(eval_conversations)
    single_turn_rows = sum(
        len(conversation.messages) == 2
        and conversation.messages[0].role == Role.user
        and conversation.messages[1].role == Role.assistant
        for conversation in conversations
    )
    eval_single_turn_rows = sum(
        len(conversation.messages) == 2
        and conversation.messages[0].role == Role.user
        and conversation.messages[1].role == Role.assistant
        for conversation in eval_conversations
    )
    full_eval_single_turn_rows = eval_single_turn_rows

    catalog_cache = CatalogStringCache()
    if conversation_prompt_contract == OPENAI_FULL_CATALOG_V1:
        samples = project_rl_decisions(
            conversations,
            sources=conversation_sources,
        )
        eval_samples = project_rl_decisions(
            eval_conversations,
            sources=eval_conversation_sources,
        )
        if not samples:
            raise ValueError("RL data has no assistant decisions")
        if not eval_samples:
            raise ValueError("RL eval data has no assistant decisions")
        train_tool_schemas = None
        eval_tool_schemas = None
    else:
        samples = single_turn_samples(conversations)
        if not samples:
            raise ValueError("RL data has no simple user -> assistant exact-reward rows")
        eval_samples = single_turn_samples(eval_conversations)
        if not eval_samples:
            raise ValueError("RL eval data has no simple user -> assistant exact-reward rows")
        train_tool_schemas = _single_turn_tool_schemas(conversations)
        eval_tool_schemas = _single_turn_tool_schemas(eval_conversations)
        if len(train_tool_schemas) != len(samples) or len(eval_tool_schemas) != len(eval_samples):
            raise RuntimeError(
                "single-turn schema projection is inconsistent with sample projection"
            )
    split_audit = _audit_data_splits(
        conversations,
        eval_conversations,
        samples,
        eval_samples,
        conversation_prompt_contract=conversation_prompt_contract,
        catalog_cache=catalog_cache,
    )
    evaluation_cfg = config.get("evaluation", {})
    if not isinstance(evaluation_cfg, Mapping):
        raise TypeError("evaluation must be a mapping")
    raw_preflight_coverage = evaluation_cfg.get("preflight_minimum_coverage")
    preflight_coverage = None
    if raw_preflight_coverage is not None:
        if not isinstance(raw_preflight_coverage, Mapping):
            raise TypeError("evaluation.preflight_minimum_coverage must be a mapping")
        preflight_coverage = dict(raw_preflight_coverage)
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
            eval_tool_schemas = None
        else:
            eval_samples = single_turn_samples(eval_conversations)
            eval_tool_schemas = _single_turn_tool_schemas(eval_conversations)
            if len(eval_tool_schemas) != len(eval_samples):
                raise RuntimeError(
                    "selected single-turn schema projection is inconsistent with sample projection"
                )
        if not eval_samples:
            raise ValueError("selected RL evaluation subset has no exact-reward rows")
    if preflight_coverage is not None:
        expected_keys = {
            "kind",
            "schema_version",
            "selector",
            "production_max_conversations",
            "minimum_coverage_rows",
            "mandatory_strata",
            "verified_eval_artifacts",
            "selection_audit",
            "derivation_sha256",
        }
        if set(preflight_coverage) != expected_keys:
            raise ValueError("RL preflight minimum-coverage contract keys mismatch")
        recorded_derivation = preflight_coverage.pop("derivation_sha256")
        if (
            not _rl_valid_sha256(recorded_derivation)
            or recorded_derivation != canonical_sha256(preflight_coverage)
        ):
            raise ValueError("RL preflight minimum-coverage derivation hash mismatch")
        preflight_coverage["derivation_sha256"] = recorded_derivation
        observed_eval_artifacts = [
            {"path": str(source.path), **dict(source.identity)}
            for source in loaded_eval_sources
        ]
        minimum_rows = preflight_coverage.get("minimum_coverage_rows")
        production_max = preflight_coverage.get("production_max_conversations")
        if (
            preflight_coverage.get("kind")
            != "localagent_rl_preflight_minimum_eval_coverage"
            or preflight_coverage.get("schema_version") != 1
            or preflight_coverage.get("selector") != STRATIFIED_EVAL_ALGORITHM
            or isinstance(minimum_rows, bool)
            or not isinstance(minimum_rows, int)
            or minimum_rows < 1
            or isinstance(production_max, bool)
            or not isinstance(production_max, int)
            or production_max < minimum_rows
            or max_eval_conversations != minimum_rows
            or preflight_coverage.get("verified_eval_artifacts")
            != observed_eval_artifacts
            or preflight_coverage.get("selection_audit") != eval_selection_audit
            or eval_selection_audit is None
            or eval_selection_audit["capacity"]
            != {
                "max_rows": minimum_rows,
                "coverage_rows": minimum_rows,
                "fill_rows": 0,
            }
            or preflight_coverage.get("mandatory_strata")
            != eval_selection_audit["mandatory_strata"]
        ):
            raise ValueError(
                "RL preflight minimum-coverage derivation drifted from verified "
                "eval artifacts or selector output"
            )
    selected_split_audit = _audit_data_splits(
        conversations,
        eval_conversations,
        samples,
        eval_samples,
        conversation_prompt_contract=conversation_prompt_contract,
        catalog_cache=catalog_cache,
    )

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
    optim = config.get("optim", {})
    schedule = config.get("schedule", {})
    rollout = config.get("rollout", {})
    policy = config.get("policy", {})
    reward = config.get("reward", {})
    log_cfg = config.get("log", {})
    max_new = int(rollout.get("max_new_tokens", 64))
    format_weight = float(reward.get("format_weight", 0.1))
    truncation_penalty = float(reward.get("truncation_penalty", 0.05))
    prompt_sampling_cfg = rollout.get("prompt_sampling")
    decision_ordering = None
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
        decision_ordering = order_assistant_decisions(conversations)
        prompt_order = decision_keys_to_prompt_order(samples, decision_ordering.keys)
        selected_decisions = int(schedule.get("total_steps", 60)) * int(
            rollout.get("prompts_per_step", 8)
        )
        prompt_sampling_contract = quota_sampling_contract(
            decision_ordering,
            selected_decisions=selected_decisions,
            require_all_strata=False,
        )
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
    context_preflight = None
    if conversation_prompt_contract == OPENAI_FULL_CATALOG_V1:
        context_preflight = {
            "train": _preflight_full_context(
                samples,
                tokenizer,
                max_new=max_new,
                max_seq_len=cfg.max_seq_len,
                split="training split",
                catalog_cache=catalog_cache,
            ),
            "eval": _preflight_full_context(
                eval_samples,
                tokenizer,
                max_new=max_new,
                max_seq_len=cfg.max_seq_len,
                split="held-out split",
                catalog_cache=catalog_cache,
            ),
        }

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
    if _expected_execution is not None:
        guarded_execution = {
            key: execution[key]
            for key in (
                "requested_device",
                "resolved_device",
                "requested_dtype",
                "resolved_dtype",
            )
        }
        if dict(_expected_execution) != guarded_execution:
            raise ValueError("RL execution identity does not match guarded readiness")
    out_dir = Path(log_cfg.get("out_dir", "runs/rl"))
    checkpoint_path = out_dir / "latest.pt"
    configured_resume = runtime.get("resume", False)
    if not isinstance(configured_resume, bool):
        raise TypeError("runtime.resume must be boolean")
    resume_requested = configured_resume if resume is None else resume
    if not isinstance(resume_requested, bool):
        raise TypeError("resume override must be boolean or None")
    if not isinstance(_require_fresh_output_dir, bool):
        raise TypeError("_require_fresh_output_dir must be boolean")
    if _require_fresh_output_dir and resume_requested:
        raise ValueError("guarded fresh-output RL cannot resume")
    if resume_requested and not checkpoint_path.exists():
        raise FileNotFoundError(
            f"RL resume requested but checkpoint does not exist: {checkpoint_path}"
        )
    resume_from = checkpoint_path if resume_requested else None
    init_from = Path(config["init_from"])
    checkpoint, parent_checkpoint_sha256 = load_stage_parent_checkpoint(
        init_from,
        stage="rl",
        requested_model_config=cfg,
        expected_tokenizer_sha256=str(tokenizer_lineage["sha256"]),
    )
    if (
        _expected_parent_checkpoint_sha256 is not None
        and parent_checkpoint_sha256 != _expected_parent_checkpoint_sha256
    ):
        raise ValueError("RL parent checkpoint SHA-256 does not match guarded readiness")
    parent_prompt_contract = _assert_parent_prompt_contract(
        checkpoint,
        conversation_prompt_contract,
    )
    state = checkpoint.get("state_dict", checkpoint.get("model"))
    if state is None:
        raise ValueError("init_from checkpoint has no state_dict/model")
    torch.manual_seed(seed)
    model = LocalAgentLM(cfg)
    model.load_state_dict(state)
    out_dir.mkdir(
        parents=True,
        exist_ok=not _require_fresh_output_dir,
    )

    train_artifacts = [
        {"path": str(source.path), **dict(source.identity)}
        for source in loaded_conversation_sources
    ]
    eval_artifacts = [
        {"path": str(source.path), **dict(source.identity)} for source in loaded_eval_sources
    ]
    data_metadata = {
        "conversation_rows": len(conversations),
        "single_turn_rows": single_turn_rows,
        "paths": [str(path) for path in conversation_paths],
        "eval_conversation_rows": len(eval_conversations),
        "eval_single_turn_rows": eval_single_turn_rows,
        "eval_source_conversation_rows": full_eval_conversation_rows,
        "eval_source_single_turn_rows": full_eval_single_turn_rows,
        "eval_paths": [str(path) for path in eval_conversation_paths],
        "train_artifacts": train_artifacts,
        "eval_artifacts": eval_artifacts,
        "gold_output_budget": gold_output_budget,
        "split_audit": split_audit,
        "selected_eval_split_audit": selected_split_audit,
        **(
            {"prompt_sampling": prompt_sampling_contract}
            if prompt_sampling_contract is not None
            else {}
        ),
        **(
            {"eval_selection": eval_selection_audit}
            if eval_selection_audit is not None
            else {}
        ),
        **(
            {"preflight_minimum_coverage": preflight_coverage}
            if preflight_coverage is not None
            else {}
        ),
    }
    if conversation_prompt_contract != LEGACY_CONVERSATION_PROMPT_CONTRACT:
        data_metadata.update(
            {
                "conversation_prompt_contract": conversation_prompt_contract,
                "parent_conversation_prompt_contract": parent_prompt_contract,
                "assistant_decision_rows": len(samples),
                "eval_assistant_decision_rows": len(eval_samples),
                "context_preflight": context_preflight,
                "prompt_truncation": "forbidden",
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
            }
        )
    tokenizer_metadata = {
        "kind": tokenizer_kind,
        "path": tokenizer_path,
        "sha256": tokenizer_lineage["sha256"],
    }
    lineage_data_identity: dict[str, Any] = {
        "train_artifacts": train_artifacts,
        "eval_artifacts": eval_artifacts,
        "split_audit": split_audit,
        "selected_eval_split_audit": selected_split_audit,
        **(
            {"prompt_sampling": prompt_sampling_contract}
            if prompt_sampling_contract is not None
            else {}
        ),
        **(
            {"eval_selection": eval_selection_audit}
            if eval_selection_audit is not None
            else {}
        ),
        **(
            {"preflight_minimum_coverage": preflight_coverage}
            if preflight_coverage is not None
            else {}
        ),
    }
    if conversation_prompt_contract != LEGACY_CONVERSATION_PROMPT_CONTRACT:
        lineage_data_identity.update(
            {
                "conversation_prompt_contract": conversation_prompt_contract,
                "parent_conversation_prompt_contract": parent_prompt_contract,
                "context_preflight": context_preflight,
                "prompt_truncation": "forbidden",
                "schema_validation": (
                    "validate_tool_catalog + recursive schema_matches "
                    "(including additionalProperties)"
                ),
            }
        )
    lineage = build_stage_lineage(
        stage="rl",
        config=config,
        model_config=cfg.__dict__,
        data_identity=lineage_data_identity,
        tokenizer=tokenizer_lineage,
        workspace=Path(__file__).resolve(),
        parent_checkpoint_sha256=parent_checkpoint_sha256,
    )
    heldout_pre = _evaluate_holdout(
        model,
        eval_samples,
        tokenizer,
        max_new=max_new,
        device=device,
        format_weight=format_weight,
        truncation_penalty=truncation_penalty,
        tool_schemas=eval_tool_schemas,
        amp_dtype=dtype,
        conversation_prompt_contract=conversation_prompt_contract,
        catalog_cache=catalog_cache,
    )
    heldout_contract = {
        "split": "explicit_disjoint_eval_conversations",
        "dataset_sha256": selected_split_audit["eval_scored_rows_sha256"],
        "decoding": "greedy_argmax",
        "max_new_tokens": max_new,
        "same_rows_pre_post": True,
    }
    if conversation_prompt_contract != LEGACY_CONVERSATION_PROMPT_CONTRACT:
        heldout_contract.update(
            {
                "conversation_prompt_contract": conversation_prompt_contract,
                "parent_conversation_prompt_contract": parent_prompt_contract,
                "row_order": "configured_jsonl_assistant_decision_order",
                "assistant_decision_rows": len(eval_samples),
                **(
                    {"selection": eval_selection_audit}
                    if eval_selection_audit is not None
                    else {}
                ),
                "context_preflight": context_preflight["eval"],
                "prompt_truncation": "forbidden",
                "current_gold_in_prompt": False,
                "schema_validation": "validate_tool_catalog + recursive schema_matches",
                "output_validation": "strict parse_tool_output before exact/schema scoring",
            }
        )
    heldout_baseline = {"contract": heldout_contract, "pre": heldout_pre}
    reward_history, rl_accounting = grpo(
        model,
        samples,
        tokenizer,
        steps=int(schedule.get("total_steps", 60)),
        prompts_per_step=int(rollout.get("prompts_per_step", 8)),
        group_size=int(rollout.get("group_size", 4)),
        lr=float(optim.get("lr", 2e-4)),
        warmup_steps=int(schedule.get("warmup_steps", 5)),
        temperature=float(rollout.get("temperature", 1.0)),
        max_new=max_new,
        device=device,
        seed=seed,
        clip_ratio=float(policy.get("clip_ratio", 0.2)),
        kl_beta=float(policy.get("kl_beta", 0.02)),
        policy_epochs=int(policy.get("epochs_per_rollout", 1)),
        format_weight=format_weight,
        truncation_penalty=truncation_penalty,
        tool_schemas=train_tool_schemas,
        amp_dtype=dtype,
        return_metrics=True,
        conversation_prompt_contract=conversation_prompt_contract,
        catalog_cache=catalog_cache,
        checkpoint_path=checkpoint_path,
        checkpoint_every=int(log_cfg.get("ckpt_every", 0)),
        resume_from=resume_from,
        lineage=lineage,
        tokenizer_metadata=tokenizer_metadata,
        data_metadata=data_metadata,
        execution=execution,
        heldout_baseline=heldout_baseline,
        prompt_order=prompt_order,
        prompt_sampling_contract=prompt_sampling_contract,
        execution_rollout_step_limit=_execution_rollout_step_limit,
        reward_environment=environment_name,
    )
    training_checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    heldout_post = _evaluate_holdout(
        model,
        eval_samples,
        tokenizer,
        max_new=max_new,
        device=device,
        format_weight=format_weight,
        truncation_penalty=truncation_penalty,
        tool_schemas=eval_tool_schemas,
        amp_dtype=dtype,
        conversation_prompt_contract=conversation_prompt_contract,
        catalog_cache=catalog_cache,
        reward_environment=environment_name,
    )
    delta_keys = (
        "exact_match_accuracy",
        "mean_reward",
        "tool_exact_match_accuracy",
        "text_exact_match_accuracy",
        "tool_format_valid_rate",
    )
    heldout_delta = {
        key: heldout_post[key] - heldout_pre[key]
        for key in delta_keys
        if heldout_pre[key] is not None and heldout_post[key] is not None
    }
    heldout_eval = {
        "contract": heldout_contract,
        "pre": heldout_pre,
        "post": heldout_post,
        "delta": heldout_delta,
    }

    structured_head_keys = ("tool_head", "ptr_head", "route_head", "dense_selector")
    invalidated_heads = [key for key in structured_head_keys if checkpoint.get(key) is not None]
    policy_contract = {
        "objective": "sampled_token_clipped_grpo",
        "ratio_scope": "generated_tokens_only",
        "reference_kl": "sampled_token_k3",
        "includes_sampled_eos": True,
        "epochs_per_rollout": int(policy.get("epochs_per_rollout", 1)),
    }
    if conversation_prompt_contract != LEGACY_CONVERSATION_PROMPT_CONTRACT:
        policy_contract.update(
            {
                "conversation_prompt_contract": conversation_prompt_contract,
                "parent_conversation_prompt_contract": parent_prompt_contract,
                "conditioning": (
                    "full function catalog + EOS + role-preserving history + assistant marker"
                ),
                "prompt_materialization": "lazy per selected assistant decision",
                "prompt_truncation": "forbidden",
                "context_preflight": context_preflight["train"],
            }
        )
    payload = {
        **training_checkpoint,
        "cfg": cfg.__dict__,
        "state_dict": model.state_dict(),
        "tokenizer": tokenizer_metadata,
        "stage": "rl",
        "step": training_checkpoint["step"],
        "reward_history": reward_history,
        "structured_heads_available": False,
        "invalidated_structured_heads": invalidated_heads,
        "reward_contract": {
            "environment": environment_name,
            "correctness": (
                "stateful schema/tool/argument/transition shaped reward"
                if environment_name == "stateful_productivity"
                else "exact normalized tool AST; exact text match"
            ),
            "format_weight": format_weight,
            "format_validation": (
                "registry name + argument schema when available; parsed AST fallback"
            ),
            "truncation_penalty": truncation_penalty,
            "learned_judge": False,
            "policy_scope": "autoregressive_lm_only",
        },
        "policy_contract": policy_contract,
        "heldout_eval": heldout_eval,
        "rl_accounting": rl_accounting,
        "lineage": lineage,
        "data": data_metadata,
        "execution": execution,
    }
    if conversation_prompt_contract != LEGACY_CONVERSATION_PROMPT_CONTRACT:
        payload["conversation_prompt_contract"] = conversation_prompt_contract
        payload["assistant_decision_rows"] = len(samples)
        payload["reward_contract"].update(
            {
                "conversation_prompt_contract": conversation_prompt_contract,
                "correctness": (
                    "strict parse_tool_output gate + exact normalized tool AST; "
                    "byte-exact text response"
                ),
                "format_validation": (
                    "strict parse_tool_output + shared validate_tool_catalog + recursive "
                    "schema_matches; outside text, duplicate keys, extra object keys, malformed "
                    "markers, unknown tools, and schema violations rejected"
                ),
                "output_parser": "localagent.eval.tool_eval.parse_tool_output",
                "schema_additional_properties": "enforced recursively",
                "gold_output_contract": "shared AssistantTrainingTurn body + terminal EOS",
                "prompt_truncation": "forbidden",
                "context_preflight": context_preflight,
            }
        )
        payload["context_preflight"] = context_preflight
    payload["resume_integrity_sha256"] = _rl_sealed_resume_sha256(payload)
    tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(checkpoint_path)
    metrics_path = out_dir / "metrics.json"
    metrics_payload = {
        "stage": "rl",
        "checkpoint": str(checkpoint_path),
        "conversation_rows": len(conversations),
        "single_turn_rows": single_turn_rows,
        "mean_reward_last": reward_history[-1],
        "reward_steps": len(reward_history),
        "rl_accounting": rl_accounting,
        "prompt_accounting": payload["prompt_accounting"],
        "data": data_metadata,
        "heldout_eval": heldout_eval,
        "reward_contract": payload["reward_contract"],
        "policy_contract": policy_contract,
        "structured_heads_available": False,
        "invalidated_structured_heads": invalidated_heads,
        "lineage": lineage,
        "execution": execution,
    }
    if conversation_prompt_contract != LEGACY_CONVERSATION_PROMPT_CONTRACT:
        metrics_payload["conversation_prompt_contract"] = conversation_prompt_contract
        metrics_payload["assistant_decision_rows"] = len(samples)
        metrics_payload["context_preflight"] = context_preflight
    metrics_path.write_text(
        json.dumps(
            metrics_payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"checkpoint": str(checkpoint_path), "metrics": str(metrics_path)}, indent=2))
