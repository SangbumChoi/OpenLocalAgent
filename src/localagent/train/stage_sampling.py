"""Shared deterministic sampling primitives for training and no-model budget planning.

The post-training runners and :mod:`localagent.train.stage_budget` consume the objects in this
module directly.  Keeping every Python-RNG draw here prevents an audit plan from approximating a
runner's schedule and then silently drifting when auxiliary SFT sampling changes.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

from localagent.data.prompt_contract import (
    LEGACY_CONVERSATION_PROMPT_CONTRACT,
    assert_prompt_contract_tokenizer,
    assistant_training_turns,
)
from localagent.data.render import (
    IGNORE,
    CatalogTokenCache,
    RenderedTokenRow,
    SharedPrefixTokenSequence,
    history_text,
    render_conversation_rows_batch,
    render_sft,
    shifted_token_counts,
    token_row_length,
)
from localagent.data.schema import Role
from localagent.model.tokenizer import ASSISTANT, BPE_EOS
from localagent.train.loop import in_decay_window, validate_pad_to_input_tokens

TokenRow = RenderedTokenRow
SFTEntry = tuple[TokenRow, str]
SFTHeadItem = tuple[str, int, str | None, str | None]
SFTMultiTurnItem = tuple[Sequence[int], int, int, int, int]
SFT_FORWARD_SLOT_KEYS = (
    "padded_lm",
    "distillation",
    "short_joint_head",
    "multi_turn_head",
)
SFT_LOSS_NORMALIZATION_MICROBATCH = "microbatch_mean_v1"
SFT_LOSS_NORMALIZATION_UPDATE_TOKENS = "assistant_token_mean_per_update_v1"
SFT_LOSS_NORMALIZATIONS = frozenset(
    {
        SFT_LOSS_NORMALIZATION_MICROBATCH,
        SFT_LOSS_NORMALIZATION_UPDATE_TOKENS,
    }
)


def validate_sft_loss_normalization(value: Any) -> str:
    """Return one explicit SFT LM-loss normalization contract."""

    if not isinstance(value, str):
        raise TypeError("optim.loss_normalization must be a string")
    if value not in SFT_LOSS_NORMALIZATIONS:
        raise ValueError(
            "optim.loss_normalization must be one of "
            + ", ".join(sorted(SFT_LOSS_NORMALIZATIONS))
        )
    return value


def validate_multi_turn_batch_size(value: Any) -> int:
    """Return a valid auxiliary multi-turn batch size without accepting booleans."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("SFT multi_turn_batch_size must be a non-negative integer")
    return value


def framed_prompt_ids(tok, prompt: str, max_seq_len: int) -> list[int]:
    """Encode the exact short joint-head input consumed by :func:`sft._framed_full`."""

    if max_seq_len < 1:
        raise ValueError("max_seq_len must be positive")
    from localagent.model.tokenizer import ASSISTANT, USER

    return tok.encode(f"{USER}{prompt}{ASSISTANT}")[-max_seq_len:]


def _padded_token_slots(lengths: Sequence[int], *, trim_right: int = 0) -> int:
    """Count tensor slots after right-padding and an optional shared right trim."""

    if not lengths:
        return 0
    if trim_right < 0:
        raise ValueError("trim_right must be non-negative")
    max_length = max(lengths)
    if max_length < trim_right:
        raise ValueError("cannot trim beyond a padded sequence")
    return len(lengths) * (max_length - trim_right)


def empty_token_accounting(source_names: Sequence[str]) -> dict[str, Any]:
    """Return zeroed input/loss token counters in deterministic source order."""

    return {
        "input_tokens": 0,
        "loss_tokens": 0,
        "sources": {
            name: {"input_tokens": 0, "loss_tokens": 0, "rows": 0}
            for name in dict.fromkeys(source_names)
        },
    }


def add_row_accounting(
    accounting: dict[str, Any],
    row: TokenRow,
    source: str,
) -> None:
    """Add one rendered row using the same shift and mask as next-token training."""

    input_tokens, loss_tokens = shifted_token_counts(row)
    source_metrics = accounting["sources"].setdefault(
        source,
        {"input_tokens": 0, "loss_tokens": 0, "rows": 0},
    )
    accounting["input_tokens"] += input_tokens
    accounting["loss_tokens"] += loss_tokens
    source_metrics["input_tokens"] += input_tokens
    source_metrics["loss_tokens"] += loss_tokens
    source_metrics["rows"] += 1


def dataset_token_accounting(entries: Sequence[SFTEntry]) -> dict[str, Any]:
    """Account for every row in one rendered SFT pool exactly once."""

    accounting = empty_token_accounting([source for _, source in entries])
    for row, source in entries:
        add_row_accounting(accounting, row, source)
    return accounting


def encode_with_value_span(
    tok,
    text: str,
    value: str,
    max_seq_len: int,
) -> tuple[list[int], tuple[int, int] | None]:
    """Encode text and map a copied string value to its contextual token span."""

    if max_seq_len < 1:
        raise ValueError("max_seq_len must be positive")
    full_ids = tok.encode(text)
    cut = max(0, len(full_ids) - max_seq_len)
    kept_ids = full_ids[cut:]
    if not value:
        return kept_ids, None

    occurrences: list[tuple[int, int]] = []
    search_from = 0
    while (found := text.find(value, search_from)) >= 0:
        occurrences.append((found, found + len(value)))
        search_from = found + max(1, len(value))

    native = getattr(tok, "_tokenizer", None)
    if native is not None:
        encoded = native.encode(text, add_special_tokens=False)
        if list(encoded.ids) != full_ids:
            raise ValueError("tokenizer native encoding disagrees with tokenizer wrapper")
        offsets = list(encoded.offsets)
        for char_start, char_end in occurrences:
            overlapping = [
                index
                for index, (token_start, token_end) in enumerate(offsets)
                if token_end > char_start and token_start < char_end
            ]
            if overlapping and overlapping[0] >= cut:
                return kept_ids, (overlapping[0] - cut, overlapping[-1] - cut)
        return kept_ids, None

    # ByteTokenizer has one token per UTF-8 byte.
    for char_start, char_end in occurrences:
        byte_start = len(text[:char_start].encode("utf-8"))
        byte_end = len(text[:char_end].encode("utf-8"))
        if byte_start >= cut and byte_end > byte_start:
            return kept_ids, (byte_start - cut, byte_end - cut - 1)
    return kept_ids, None


@dataclass(frozen=True)
class SFTPreparedData:
    """Rendered LM pools plus auxiliary examples that affect SFT RNG consumption."""

    samples: tuple[Any, ...]
    conversations: tuple[Any, ...]
    rows: tuple[TokenRow, ...]
    main_entries: tuple[SFTEntry, ...]
    decay_entries: tuple[SFTEntry, ...]
    has_decay_pool: bool
    head_items: tuple[SFTHeadItem, ...]
    head_input_lengths: tuple[int, ...]
    multi_turn_items: tuple[SFTMultiTurnItem, ...]
    dataset_accounting: Mapping[str, Any]
    conversation_prompt_contract: str
    catalog_token_cache: CatalogTokenCache | None


def decision_keys_to_row_order(
    conversations: Sequence[Any],
    ordered_keys: Sequence[tuple[int, int]],
    *,
    expected_rows: int | None = None,
) -> tuple[int, ...]:
    """Map canonical assistant-decision keys to full-catalog rendered-row indices.

    ``render_conversation_rows_batch`` and ``assistant_training_turns`` share the same source-order
    projection.  This helper makes that relationship explicit and fails closed before a quota
    order can silently select the wrong language-model row.
    """

    natural_keys = tuple(
        (conversation_index, turn.message_index)
        for conversation_index, conversation in enumerate(conversations)
        for turn in assistant_training_turns(conversation)
    )
    if expected_rows is not None and len(natural_keys) != expected_rows:
        raise ValueError(
            "assistant-decision projection does not match rendered SFT rows: "
            f"decisions={len(natural_keys)}, rows={expected_rows}"
        )
    if len(set(natural_keys)) != len(natural_keys):
        raise RuntimeError("canonical assistant-decision keys are not unique")

    normalized_keys = tuple(ordered_keys)
    if len(normalized_keys) != len(natural_keys):
        raise ValueError(
            "quota decision order must contain every rendered assistant decision exactly once"
        )
    if len(set(normalized_keys)) != len(normalized_keys):
        raise ValueError("quota decision order contains duplicate assistant-decision keys")
    if set(normalized_keys) != set(natural_keys):
        raise ValueError("quota decision order does not match the rendered assistant decisions")
    row_by_key = {key: index for index, key in enumerate(natural_keys)}
    return tuple(row_by_key[key] for key in normalized_keys)


def prepare_sft_data(
    samples: Sequence[Any],
    tok,
    *,
    conversations: Sequence[Any] | None,
    sample_sources: Sequence[str] | None,
    conversation_sources: Sequence[str] | None,
    decay_samples: Sequence[Any] | None,
    decay_sample_sources: Sequence[str] | None,
    lr_schedule: str,
    max_seq_len: int,
    joint_tool_head: bool,
    ptr_args: Sequence[str] | None = None,
    conversation_prompt_contract: str | None = None,
    decay_conversations: Sequence[Any] | None = None,
    decay_conversation_sources: Sequence[str] | None = None,
) -> SFTPreparedData:
    """Render and validate every SFT pool used by both training and budget planning."""

    if lr_schedule not in {"cosine", "wsd"}:
        raise ValueError(f"sft() lr_schedule must be 'cosine' or 'wsd', got {lr_schedule!r}")
    if max_seq_len < 2:
        raise ValueError("sft() max_seq_len must be at least 2")
    prompt_contract = assert_prompt_contract_tokenizer(tok, conversation_prompt_contract)
    if ptr_args is None:
        from localagent.agent.pointer_head import PTR_ARGS

        ptr_args = tuple(PTR_ARGS)
    else:
        ptr_args = tuple(ptr_args)
    ptr_arg_idx = {arg: index for index, arg in enumerate(ptr_args)}
    if len(ptr_arg_idx) != len(ptr_args):
        raise ValueError("pointer argument names must be unique")
    catalog_token_cache = (
        None if prompt_contract == LEGACY_CONVERSATION_PROMPT_CONTRACT else CatalogTokenCache(tok)
    )

    sample_rows = tuple(samples)
    conversation_rows = tuple(conversations or ())
    if sample_sources is None:
        sample_sources = ["single_turn"] * len(sample_rows)
    if conversation_sources is None:
        conversation_sources = ["multi_turn"] * len(conversation_rows)
    if len(sample_sources) != len(sample_rows):
        raise ValueError("sample_sources length must match samples")
    if len(conversation_sources) != len(conversation_rows):
        raise ValueError("conversation_sources length must match conversations")

    def bounded(row: TokenRow) -> TokenRow:
        ids, labels = row
        return ids[-max_seq_len:], labels[-max_seq_len:]

    def usable(row: TokenRow) -> bool:
        _, loss_tokens = shifted_token_counts(row)
        return token_row_length(row) >= 2 and loss_tokens > 0

    def render_conversation_entries(
        values: Sequence[Any],
        sources: Sequence[str],
    ) -> tuple[SFTEntry, ...]:
        rendered = render_conversation_rows_batch(
            values,
            tok,
            prompt_contract=prompt_contract,
            max_seq_len=max_seq_len,
            catalog_cache=catalog_token_cache,
        )
        if prompt_contract == LEGACY_CONVERSATION_PROMPT_CONTRACT:
            row_sources = tuple(str(source) for source in sources)
        else:
            row_sources = tuple(
                str(source)
                for conversation, source in zip(values, sources, strict=True)
                for message in conversation.messages
                if message.role == Role.assistant
            )
        if len(row_sources) != len(rendered):
            raise RuntimeError(
                "conversation source projection does not align with rendered SFT rows"
            )
        return tuple(
            (row, source)
            for row, source in zip(rendered, row_sources, strict=True)
            if usable(row)
        )

    if prompt_contract == LEGACY_CONVERSATION_PROMPT_CONTRACT:
        sample_entries = tuple(
            entry
            for entry in (
                (bounded(render_sft(sample, tok)), str(source))
                for sample, source in zip(sample_rows, sample_sources, strict=True)
            )
            if usable(entry[0])
        )
    else:
        if not conversation_rows:
            raise ValueError("openai_full_catalog_v1 SFT requires canonical Conversation rows")
        sample_entries = ()
    conversation_entries = render_conversation_entries(
        conversation_rows,
        conversation_sources,
    )
    main_entries = sample_entries + conversation_entries
    if not main_entries:
        raise ValueError("sft() needs at least one conversation with a trainable assistant target")
    rows = (
        tuple(row for row, _ in sample_entries)
        if prompt_contract == LEGACY_CONVERSATION_PROMPT_CONTRACT
        else tuple(row for row, _ in main_entries)
    )

    has_decay_pool = lr_schedule == "wsd" and (
        decay_samples is not None
        if prompt_contract == LEGACY_CONVERSATION_PROMPT_CONTRACT
        else decay_conversations is not None
    )
    decay_entries = main_entries
    if has_decay_pool:
        if prompt_contract == LEGACY_CONVERSATION_PROMPT_CONTRACT:
            decay_rows = tuple(decay_samples or ())
            if decay_sample_sources is None:
                decay_sample_sources = ["decay_single_turn"] * len(decay_rows)
            if len(decay_sample_sources) != len(decay_rows):
                raise ValueError("decay_sample_sources length must match decay_samples")
            rendered_decay = tuple(
                entry
                for entry in (
                    (bounded(render_sft(sample, tok)), str(source))
                    for sample, source in zip(decay_rows, decay_sample_sources, strict=True)
                )
                if usable(entry[0])
            )
            decay_entries = rendered_decay + conversation_entries
        else:
            decay_conversation_rows = tuple(decay_conversations or ())
            if decay_conversation_sources is None:
                decay_conversation_sources = ["decay_conversation"] * len(decay_conversation_rows)
            if len(decay_conversation_sources) != len(decay_conversation_rows):
                raise ValueError("decay_conversation_sources length must match decay_conversations")
            decay_entries = render_conversation_entries(
                decay_conversation_rows,
                decay_conversation_sources,
            )
        if not decay_entries:
            raise ValueError("decay sample pool has no trainable assistant targets")

    head_items: tuple[SFTHeadItem, ...] = ()
    head_input_lengths: tuple[int, ...] = ()
    multi_turn_items: tuple[SFTMultiTurnItem, ...] = ()
    if joint_tool_head:
        if not sample_rows:
            raise ValueError("joint tool/pointer heads need simple user -> assistant conversations")
        from localagent.agent.tool_head import CLASSES, canonical_tool_name, label_of

        mutable_head_items: list[SFTHeadItem] = []

        def ptr_of(arguments: Mapping[str, Any]) -> tuple[str | None, str | None]:
            for key, value in arguments.items():
                if key in ptr_arg_idx and isinstance(value, str):
                    return key, value
            return None, None

        for sample in sample_rows:
            if sample.calls:
                conjuncts = sample.prompt.split(" and ")
                if len(conjuncts) == len(sample.calls):
                    for conjunct, call in zip(conjuncts, sample.calls, strict=True):
                        label = (
                            CLASSES.index(canonical_tool_name(call["name"]))
                            if canonical_tool_name(call["name"]) in CLASSES
                            else CLASSES.index("text")
                        )
                        pointer_arg, pointer_value = ptr_of(call["arguments"])
                        mutable_head_items.append(
                            (conjunct.strip(), label, pointer_arg, pointer_value)
                        )
            else:
                arguments = json.loads(sample.ref_args) if sample.kind == "tool" else {}
                pointer_arg, pointer_value = ptr_of(arguments)
                mutable_head_items.append(
                    (
                        sample.prompt,
                        CLASSES.index(label_of(sample)),
                        pointer_arg,
                        pointer_value,
                    )
                )
        head_items = tuple(mutable_head_items)
        head_input_lengths = tuple(
            len(framed_prompt_ids(tok, item[0], max_seq_len)) for item in head_items
        )

        mutable_multi_turn: list[SFTMultiTurnItem] = []
        for conversation in conversation_rows:
            turns_by_index = {}
            catalog_text = ""
            catalog_ids: tuple[int, ...] = ()
            if prompt_contract != LEGACY_CONVERSATION_PROMPT_CONTRACT:
                catalog_entry = catalog_token_cache.catalog_cache.entry(conversation.tools)
                catalog_text = catalog_entry.text + BPE_EOS
                catalog_ids = catalog_token_cache.tokens(conversation.tools)
                turns_by_index = {
                    turn.message_index: turn
                    for turn in assistant_training_turns(
                        conversation,
                        catalog_cache=catalog_token_cache.catalog_cache,
                    )
                }
            for index, message in enumerate(conversation.messages):
                if message.role.value != "assistant" or not message.tool_calls:
                    continue
                call = message.tool_calls[0]
                if prompt_contract == LEGACY_CONVERSATION_PROMPT_CONTRACT:
                    context = history_text(conversation.messages[:index]) + ASSISTANT
                    context_ids = tok.encode(context)[-max_seq_len:]
                else:
                    turn = turns_by_index[index]
                    prompt_suffix = turn.prompt_suffix
                    suffix_ids = tuple(tok.encode(prompt_suffix))
                    context_ids = SharedPrefixTokenSequence(
                        catalog_ids,
                        suffix_ids,
                    )
                    if len(context_ids) > max_seq_len:
                        raise ValueError(
                            "openai_full_catalog_v1 auxiliary prompt exceeds max_seq_len "
                            "and cannot be truncated: "
                            f"assistant_message_index={index}, tokens={len(context_ids)}, "
                            f"max_seq_len={max_seq_len}"
                        )
                canonical_name = canonical_tool_name(call.name)
                label = (
                    CLASSES.index(canonical_name)
                    if canonical_name in CLASSES
                    else CLASSES.index("text")
                )
                pointer_arg_index, gold_start, gold_end = -1, -1, -1
                for key, value in call.arguments.items():
                    if key not in ptr_arg_idx or not isinstance(value, str):
                        continue
                    if prompt_contract == LEGACY_CONVERSATION_PROMPT_CONTRACT:
                        contextual_ids, span = encode_with_value_span(
                            tok,
                            context,
                            value,
                            max_seq_len,
                        )
                        if contextual_ids != list(context_ids):
                            raise ValueError(
                                "contextual pointer encoding disagrees with conversation"
                            )
                    elif value in catalog_text:
                        contextual_ids, span = encode_with_value_span(
                            tok,
                            catalog_text + prompt_suffix,
                            value,
                            max_seq_len,
                        )
                        if contextual_ids != list(context_ids):
                            raise ValueError(
                                "contextual pointer encoding disagrees with conversation"
                            )
                    else:
                        contextual_suffix_ids, suffix_span = encode_with_value_span(
                            tok,
                            prompt_suffix,
                            value,
                            max_seq_len,
                        )
                        if contextual_suffix_ids != list(suffix_ids):
                            raise ValueError(
                                "contextual pointer suffix encoding disagrees with conversation"
                            )
                        span = (
                            None
                            if suffix_span is None
                            else (
                                len(catalog_ids) + suffix_span[0],
                                len(catalog_ids) + suffix_span[1],
                            )
                        )
                    if span is not None:
                        pointer_arg_index, gold_start, gold_end = (
                            ptr_arg_idx[key],
                            span[0],
                            span[1],
                        )
                        break
                mutable_multi_turn.append(
                    (
                        context_ids,
                        label,
                        pointer_arg_index,
                        gold_start,
                        gold_end,
                    )
                )
        multi_turn_items = tuple(mutable_multi_turn)

    accounting = {
        "main": dataset_token_accounting(main_entries),
        "decay": dataset_token_accounting(decay_entries) if has_decay_pool else None,
    }
    return SFTPreparedData(
        samples=sample_rows,
        conversations=conversation_rows,
        rows=rows,
        main_entries=main_entries,
        decay_entries=decay_entries,
        has_decay_pool=has_decay_pool,
        head_items=head_items,
        head_input_lengths=head_input_lengths,
        multi_turn_items=multi_turn_items,
        dataset_accounting=accounting,
        conversation_prompt_contract=prompt_contract,
        catalog_token_cache=catalog_token_cache,
    )


@dataclass(frozen=True)
class SFTMicrobatchSelection:
    """All indices selected for one SFT microbatch in exact RNG-consumption order."""

    pool: Literal["main", "decay"]
    lm_indices: tuple[int, ...]
    kd_indices: tuple[int, ...]
    head_indices: tuple[int, ...]
    multi_turn_indices: tuple[int, ...]


class SFTSamplingSchedule:
    """Stateful SFT sampler shared by the runner and deterministic planner."""

    def __init__(
        self,
        prepared: SFTPreparedData,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
        lr_schedule: str,
        decay_frac: float,
        kd_enabled: bool,
        joint_tool_head: bool,
        multi_turn_batch_size: int = 12,
        lm_order: Sequence[int] | None = None,
    ):
        if batch_size < 1:
            raise ValueError("SFT batch_size must be positive")
        if lr_schedule not in {"cosine", "wsd"}:
            raise ValueError(f"sft() lr_schedule must be 'cosine' or 'wsd', got {lr_schedule!r}")
        self.prepared = prepared
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.lr_schedule = lr_schedule
        self.decay_frac = decay_frac
        self.kd_enabled = kd_enabled
        self.joint_tool_head = joint_tool_head
        self.multi_turn_batch_size = validate_multi_turn_batch_size(multi_turn_batch_size)
        self.rng = random.Random(seed)
        self.lm_cursor = 0
        self.lm_order = None if lm_order is None else tuple(lm_order)
        if self.lm_order is not None:
            if shuffle:
                raise ValueError("quota no-replacement SFT sampling requires shuffle=false")
            if prepared.has_decay_pool:
                raise ValueError("quota no-replacement SFT sampling does not support a decay pool")
            expected = set(range(len(prepared.main_entries)))
            if len(self.lm_order) != len(expected) or set(self.lm_order) != expected:
                raise ValueError("SFT lm_order must be an exact permutation of main entries")

    def next_microbatch(self, *, step: int, total_steps: int) -> SFTMicrobatchSelection:
        """Return one selection and advance the single shared Python RNG stream."""

        if total_steps < 1 or not 0 <= step < total_steps:
            raise ValueError("SFT sampling step must be within the positive fixed horizon")
        use_decay = (
            self.lr_schedule == "wsd"
            and self.prepared.has_decay_pool
            and in_decay_window(step, total_steps, self.decay_frac)
        )
        pool: Literal["main", "decay"] = "decay" if use_decay else "main"
        entries = self.prepared.decay_entries if pool == "decay" else self.prepared.main_entries
        if self.lm_order is not None:
            end = self.lm_cursor + self.batch_size
            if end > len(self.lm_order):
                raise ValueError(
                    "quota no-replacement SFT horizon exceeds the available decision rows"
                )
            lm_indices = self.lm_order[self.lm_cursor : end]
            self.lm_cursor = end
        elif self.shuffle:
            lm_indices = tuple(self.rng.randrange(len(entries)) for _ in range(self.batch_size))
        else:
            lm_indices = tuple(
                (self.lm_cursor + offset) % len(entries) for offset in range(self.batch_size)
            )
            self.lm_cursor = (self.lm_cursor + self.batch_size) % len(entries)

        kd_indices: tuple[int, ...] = ()
        if self.kd_enabled:
            kd_indices = tuple(
                self.rng.randrange(len(self.prepared.rows)) for _ in range(self.batch_size)
            )

        head_indices: tuple[int, ...] = ()
        multi_turn_indices: tuple[int, ...] = ()
        if self.joint_tool_head:
            head_indices = tuple(
                self.rng.randrange(len(self.prepared.head_items)) for _ in range(self.batch_size)
            )
            if self.prepared.multi_turn_items:
                multi_turn_indices = tuple(
                    self.rng.randrange(len(self.prepared.multi_turn_items))
                    for _ in range(
                        min(
                            self.multi_turn_batch_size,
                            len(self.prepared.multi_turn_items),
                        )
                    )
                )
        return SFTMicrobatchSelection(
            pool=pool,
            lm_indices=lm_indices,
            kd_indices=kd_indices,
            head_indices=head_indices,
            multi_turn_indices=multi_turn_indices,
        )


def sft_microbatch_forward_token_slots(
    prepared: SFTPreparedData,
    selection: SFTMicrobatchSelection,
    *,
    pad_to_input_tokens: int | None = None,
) -> dict[str, int]:
    """Count exact padded input slots for every model forward in one SFT microbatch."""

    entries = prepared.decay_entries if selection.pool == "decay" else prepared.main_entries
    lm_lengths = [token_row_length(entries[index][0]) for index in selection.lm_indices]
    kd_lengths = [token_row_length(prepared.rows[index]) - 1 for index in selection.kd_indices]
    short_head_lengths = [prepared.head_input_lengths[index] for index in selection.head_indices]
    multi_turn_lengths = [
        len(prepared.multi_turn_items[index][0]) for index in selection.multi_turn_indices
    ]
    fixed_input_width = validate_pad_to_input_tokens(
        pad_to_input_tokens,
        label="batch.pad_to_input_tokens",
    )
    if fixed_input_width is None:
        padded_lm = _padded_token_slots(lm_lengths, trim_right=1)
    else:
        required_input_width = max(lm_lengths, default=1) - 1
        if required_input_width > fixed_input_width:
            raise ValueError(
                "SFT row requires more input tokens than batch.pad_to_input_tokens: "
                f"required={required_input_width}, configured={fixed_input_width}"
            )
        padded_lm = len(lm_lengths) * fixed_input_width
    values = {
        "padded_lm": padded_lm,
        "distillation": _padded_token_slots(kd_lengths),
        "short_joint_head": _padded_token_slots(short_head_lengths),
        "multi_turn_head": _padded_token_slots(multi_turn_lengths),
    }
    values["total"] = sum(values.values())
    return values


@dataclass(frozen=True)
class MidtrainMicrobatch:
    """One sampled and token-counted midtraining microbatch."""

    source: Any
    x: torch.Tensor
    y: torch.Tensor
    input_tokens: int
    loss_tokens: int


@dataclass(frozen=True)
class MidtrainMicrobatchCounts:
    """Count-only equivalent of one deterministic midtraining microbatch."""

    source: Any
    input_tokens: int
    loss_tokens: int


def sample_counted_batch(source, batch_size: int, rng: random.Random, device: str):
    """Sample once and return the runner's exact pre-padding input/loss counts."""

    counted_sampler = getattr(source.dataset, "sample_batch_with_counts", None)
    if counted_sampler is not None:
        return counted_sampler(batch_size, rng, device)

    x, y = source.dataset.sample_batch(batch_size, rng, device)
    loss_per_row = (y != IGNORE).sum(dim=1)
    loss_tokens = int(loss_per_row.sum())
    if hasattr(source.dataset, "manifest"):
        input_tokens = int(torch.clamp(loss_per_row + 1, max=x.shape[1]).sum())
    else:
        input_tokens = x.numel()
    return x, y, input_tokens, loss_tokens


def sample_batch_token_counts(
    source,
    batch_size: int,
    rng: random.Random,
) -> tuple[int, int]:
    """Sample once and return exact counts without materializing token tensors."""

    count_sampler = getattr(source.dataset, "sample_batch_token_counts", None)
    if count_sampler is None:
        raise TypeError(
            f"training source {source.name!r} has no count-only sampling implementation"
        )
    input_tokens, loss_tokens = count_sampler(batch_size, rng)
    return int(input_tokens), int(loss_tokens)


def next_midtrain_microbatch(
    mixture,
    mixture_state: dict[str, Any],
    rng: random.Random,
    *,
    step: int,
    total_steps: int,
    batch_size: int,
    device: str,
) -> MidtrainMicrobatch:
    """Choose, sample, validate, and observe one exact midtraining microbatch."""

    if total_steps < 1 or not 0 <= step < total_steps:
        raise ValueError("midtraining sampling step must be within the positive fixed horizon")
    progress = step / max(1, total_steps - 1)
    source = mixture.choose(progress, rng, mixture_state)
    x, y, input_tokens, loss_tokens = sample_counted_batch(
        source,
        batch_size,
        rng,
        device,
    )
    observed_loss_tokens = int((y != IGNORE).sum())
    if observed_loss_tokens != loss_tokens:
        raise ValueError(f"training source {source.name!r} token accounting disagrees with mask")
    if input_tokens <= 0 or loss_tokens <= 0:
        raise ValueError(f"training source {source.name!r} produced an empty supervised batch")
    mixture.observe(
        mixture_state,
        source,
        progress=progress,
        input_tokens=input_tokens,
        loss_tokens=loss_tokens,
    )
    return MidtrainMicrobatch(
        source=source,
        x=x,
        y=y,
        input_tokens=input_tokens,
        loss_tokens=loss_tokens,
    )


def next_midtrain_microbatch_counts(
    mixture,
    mixture_state: dict[str, Any],
    rng: random.Random,
    *,
    step: int,
    total_steps: int,
    batch_size: int,
) -> MidtrainMicrobatchCounts:
    """Choose, count, validate, and observe the tensor-free equivalent microbatch."""

    if total_steps < 1 or not 0 <= step < total_steps:
        raise ValueError("midtraining sampling step must be within the positive fixed horizon")
    progress = step / max(1, total_steps - 1)
    source = mixture.choose(progress, rng, mixture_state)
    input_tokens, loss_tokens = sample_batch_token_counts(
        source,
        batch_size,
        rng,
    )
    if input_tokens <= 0 or loss_tokens <= 0:
        raise ValueError(f"training source {source.name!r} produced an empty supervised batch")
    mixture.observe(
        mixture_state,
        source,
        progress=progress,
        input_tokens=input_tokens,
        loss_tokens=loss_tokens,
    )
    return MidtrainMicrobatchCounts(
        source=source,
        input_tokens=input_tokens,
        loss_tokens=loss_tokens,
    )


class RLPromptSchedule:
    """Deterministic GRPO prompt-row sampler; rollout RNG remains model-dependent."""

    def __init__(
        self,
        sample_count: int,
        prompts_per_step: int,
        *,
        seed: int,
        prompt_order: Sequence[int] | None = None,
    ):
        if sample_count < 1:
            raise ValueError("RL sample_count must be positive")
        if prompts_per_step < 1:
            raise ValueError("RL prompts_per_step must be positive")
        self.sample_count = sample_count
        self.prompts_per_step = prompts_per_step
        self.rng = random.Random(seed)
        self.next_step = 0
        self.prompt_order = None if prompt_order is None else tuple(prompt_order)
        if self.prompt_order is not None:
            expected = set(range(sample_count))
            if len(self.prompt_order) != sample_count or set(self.prompt_order) != expected:
                raise ValueError("RL prompt_order must be an exact permutation of prompt rows")

    def indices_for_step(self, step: int) -> tuple[int, ...]:
        """Draw one prompt batch and reject out-of-order schedule consumption."""

        if step != self.next_step:
            raise ValueError(f"RL prompt schedule expected step {self.next_step}, got {step}")
        self.next_step += 1
        if self.prompt_order is not None:
            start = step * self.prompts_per_step
            end = start + self.prompts_per_step
            if end > len(self.prompt_order):
                raise ValueError(
                    "quota no-replacement RL horizon exceeds the available prompt rows"
                )
            return self.prompt_order[start:end]
        return tuple(self.rng.randrange(self.sample_count) for _ in range(self.prompts_per_step))
