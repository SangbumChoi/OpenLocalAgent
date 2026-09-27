"""Render Samples into token ids for training (Phase 3/4).

SFT target framing (byte-level, markers are literal text):
    <|user|>{prompt}<|assistant|>{body}<EOS>
where body is `<tool_call>{json}</tool_call>` for tool samples or the plain text for text
samples. The loss is masked over the prompt (we only learn the assistant body + EOS).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from localagent.data.agent_synth import Sample
from localagent.data.prompt_contract import (
    LEGACY_CONVERSATION_PROMPT_CONTRACT,
    OPENAI_FULL_CATALOG_V1,
    FunctionCatalogCache,
    assert_prompt_contract_tokenizer,
    assistant_training_turns,
)
from localagent.data.schema import Conversation, Role
from localagent.model.tokenizer import (
    ASSISTANT,
    BPE_EOS,
    TOOL,
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    TOOL_RESPONSE_CLOSE,
    TOOL_RESPONSE_OPEN,
    USER,
)

IGNORE = -100
TokenRow = tuple[list[int], list[int]]


@dataclass(frozen=True)
class SharedPrefixTokenSequence(Sequence[int]):
    """A logical token sequence whose large immutable prefix may be shared by many rows."""

    shared_prefix: tuple[int, ...]
    suffix: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.shared_prefix) + len(self.suffix)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.materialize()[index]
        prefix_len = len(self.shared_prefix)
        normalized = index if index >= 0 else len(self) + index
        if not 0 <= normalized < len(self):
            raise IndexError(index)
        if normalized < prefix_len:
            return self.shared_prefix[normalized]
        return self.suffix[normalized - prefix_len]

    def __iter__(self) -> Iterator[int]:
        yield from self.shared_prefix
        yield from self.suffix

    def materialize(self) -> list[int]:
        return [*self.shared_prefix, *self.suffix]


@dataclass(frozen=True)
class LazyCatalogTokenRow:
    """A masked LM row that retains one shared catalog prefix until batch collation."""

    prompt_ids: SharedPrefixTokenSequence
    body_ids: tuple[int, ...]
    message_index: int

    @property
    def token_count(self) -> int:
        return len(self.prompt_ids) + len(self.body_ids)

    @property
    def input_token_count(self) -> int:
        return max(0, self.token_count - 1)

    @property
    def shifted_loss_token_count(self) -> int:
        return len(self.body_ids)

    def materialize(self) -> TokenRow:
        prompt = self.prompt_ids.materialize()
        body = list(self.body_ids)
        return prompt + body, [IGNORE] * len(prompt) + body

    def __iter__(self):
        return iter(self.materialize())

    def __getitem__(self, index: int) -> list[int]:
        return self.materialize()[index]

    def __len__(self) -> int:
        return 2


RenderedTokenRow = TokenRow | LazyCatalogTokenRow


class CatalogTokenCache:
    """Intern canonical catalog tokens by content fingerprint for one tokenizer."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._entries: dict[str, tuple[str, tuple[int, ...]]] = {}
        self.catalog_cache = FunctionCatalogCache()

    def tokens(self, tools) -> tuple[int, ...]:
        catalog = self.catalog_cache.entry(tools).text + BPE_EOS
        fingerprint = hashlib.sha256(catalog.encode("utf-8")).hexdigest()
        cached = self._entries.get(fingerprint)
        if cached is not None:
            if cached[0] != catalog:  # cryptographic collision guard
                raise RuntimeError("function catalog SHA-256 collision")
            return cached[1]
        ids = tuple(self._tokenizer.encode(catalog))
        self._entries[fingerprint] = (catalog, ids)
        return ids

    @property
    def unique_catalogs(self) -> int:
        return len(self._entries)

    @property
    def retained_token_count(self) -> int:
        return sum(len(ids) for _, ids in self._entries.values())


def materialize_token_row(row: RenderedTokenRow) -> TokenRow:
    """Materialize a lazy row, leaving legacy tuple rows unchanged."""

    return row.materialize() if isinstance(row, LazyCatalogTokenRow) else row


def token_row_length(row: RenderedTokenRow) -> int:
    """Return the unshifted row length without materializing a shared catalog."""

    return row.token_count if isinstance(row, LazyCatalogTokenRow) else len(row[0])


def shifted_token_counts(row: RenderedTokenRow) -> tuple[int, int]:
    """Return exact next-token input/loss counts without materializing a shared catalog."""

    if isinstance(row, LazyCatalogTokenRow):
        return row.input_token_count, row.shifted_loss_token_count
    ids, labels = row
    return max(0, len(ids) - 1), sum(label != IGNORE for label in labels[1:])


@dataclass(frozen=True)
class _FullCatalogTokenInput:
    catalog_ids: tuple[int, ...]
    prompt_suffix: str
    body: str
    message_index: int


def _validate_max_seq_len(max_seq_len: int | None) -> None:
    if max_seq_len is not None and (
        isinstance(max_seq_len, bool) or not isinstance(max_seq_len, int) or max_seq_len < 2
    ):
        raise ValueError("max_seq_len must be an integer >= 2")


def _validate_batch_size(batch_size: int) -> None:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")


def _assert_full_catalog_row_fits(
    *,
    token_count: int,
    message_index: int,
    max_seq_len: int | None,
) -> None:
    if max_seq_len is not None and token_count > max_seq_len:
        raise ValueError(
            "openai_full_catalog_v1 row exceeds max_seq_len and cannot be truncated: "
            f"assistant_message_index={message_index}, "
            f"tokens={token_count}, max_seq_len={max_seq_len}"
        )


def _full_catalog_token_inputs(
    conversation: Conversation,
    cache: CatalogTokenCache,
) -> tuple[_FullCatalogTokenInput, ...]:
    """Validate one conversation and retain its shared catalog token tuple."""

    catalog_ids = cache.tokens(conversation.tools)
    return tuple(
        _FullCatalogTokenInput(
            catalog_ids=catalog_ids,
            prompt_suffix=turn.prompt_suffix,
            body=turn.body,
            message_index=turn.message_index,
        )
        for turn in assistant_training_turns(
            conversation,
            catalog_cache=cache.catalog_cache,
        )
    )


def _full_catalog_row(
    item: _FullCatalogTokenInput,
    prompt_suffix_ids: Sequence[int],
    body_ids: Sequence[int],
    *,
    eos_id: int,
    max_seq_len: int | None,
) -> LazyCatalogTokenRow:
    row = LazyCatalogTokenRow(
        prompt_ids=SharedPrefixTokenSequence(
            item.catalog_ids,
            tuple(prompt_suffix_ids),
        ),
        body_ids=(*body_ids, eos_id),
        message_index=item.message_index,
    )
    _assert_full_catalog_row_fits(
        token_count=row.token_count,
        message_index=item.message_index,
        max_seq_len=max_seq_len,
    )
    return row


def _render_full_catalog_inputs_scalar(
    inputs: Sequence[_FullCatalogTokenInput],
    tok,
    *,
    max_seq_len: int | None,
) -> list[RenderedTokenRow]:
    """Render descriptors in the precise suffix/body/check order of the scalar API."""

    rows: list[RenderedTokenRow] = []
    for item in inputs:
        rows.append(
            _full_catalog_row(
                item,
                tok.encode(item.prompt_suffix),
                tok.encode(item.body),
                eos_id=tok.eos_id,
                max_seq_len=max_seq_len,
            )
        )
    return rows


def _render_full_catalog_inputs_batch(
    inputs: Sequence[_FullCatalogTokenInput],
    tok,
    *,
    max_seq_len: int | None,
) -> list[RenderedTokenRow]:
    """Encode one bounded descriptor batch while preserving scalar error precedence."""

    if not inputs:
        return []
    texts = [
        text
        for item in inputs
        for text in (
            item.prompt_suffix,
            item.body,
        )
    ]
    encode_batch = getattr(tok, "encode_batch", None)
    if not callable(encode_batch):
        return _render_full_catalog_inputs_scalar(
            inputs,
            tok,
            max_seq_len=max_seq_len,
        )
    try:
        encoded_output = encode_batch(texts, add_eos=False)
    except Exception:
        # A batch encoder may surface a later input first. Replaying the bounded batch through
        # the scalar path restores suffix/body/length-check source order. If scalar encoding is
        # valid, it is also an exact compatibility fallback for a batch-only tokenizer failure.
        return _render_full_catalog_inputs_scalar(
            inputs,
            tok,
            max_seq_len=max_seq_len,
        )
    try:
        encoded_iterator = iter(encoded_output)
    except TypeError as exc:
        raise RuntimeError("tokenizer batch output does not align with its input texts") from exc
    try:
        encoded = list(encoded_iterator)
    except Exception:
        # Some tokenizer adapters defer their work until a returned iterator is consumed.
        # Treat those lazy failures exactly like a direct encode_batch failure.
        return _render_full_catalog_inputs_scalar(
            inputs,
            tok,
            max_seq_len=max_seq_len,
        )
    if len(encoded) != len(texts):
        raise RuntimeError("tokenizer batch output does not align with its input texts")

    rows: list[RenderedTokenRow] = []
    for index, item in enumerate(inputs):
        rows.append(
            _full_catalog_row(
                item,
                encoded[index * 2],
                encoded[index * 2 + 1],
                eos_id=tok.eos_id,
                max_seq_len=max_seq_len,
            )
        )
    return rows


def conversation_row_token_counts(
    conversations: Sequence[Conversation],
    tok,
    *,
    prompt_contract: str | None = None,
    max_seq_len: int | None = None,
    catalog_cache: CatalogTokenCache | None = None,
    batch_size: int = 256,
) -> list[tuple[int, int]]:
    """Count exact shifted input/loss tokens for a conversation catalog.

    Legacy rows deliberately take the scalar renderer path, preserving whole-trajectory
    left-truncation exactly. Full-catalog rows reuse the validated catalog cache while encoding
    catalog-independent prompt suffixes and assistant bodies in bounded batches. Result order is
    conversation order followed by assistant-message order.
    """

    contract = assert_prompt_contract_tokenizer(tok, prompt_contract)
    _validate_max_seq_len(max_seq_len)
    _validate_batch_size(batch_size)

    cache = catalog_cache if catalog_cache is not None else CatalogTokenCache(tok)
    if contract == LEGACY_CONVERSATION_PROMPT_CONTRACT:
        counts: list[tuple[int, int]] = []
        for conversation in conversations:
            counts.extend(
                shifted_token_counts(row)
                for row in render_conversation_rows(
                    conversation,
                    tok,
                    prompt_contract=contract,
                    max_seq_len=max_seq_len,
                    catalog_cache=cache,
                )
            )
        return counts

    if contract != OPENAI_FULL_CATALOG_V1:  # pragma: no cover - resolver guards this branch
        raise AssertionError(f"unhandled conversation prompt contract {contract!r}")

    counts = []
    pending: list[_FullCatalogTokenInput] = []

    def flush() -> None:
        if not pending:
            return
        batch_rows = _render_full_catalog_inputs_batch(
            pending,
            tok,
            max_seq_len=max_seq_len,
        )
        counts.extend(shifted_token_counts(row) for row in batch_rows)
        pending.clear()

    for conversation in conversations:
        try:
            conversation_inputs = _full_catalog_token_inputs(conversation, cache)
        except Exception:
            # Scalar rendering would finish every earlier conversation before validating this
            # one. Let an earlier encoding/length error win over this later schema error.
            flush()
            raise
        for item in conversation_inputs:
            pending.append(item)
            if len(pending) == batch_size:
                flush()
    flush()
    return counts


def _canon(name: str, args: dict) -> str:
    return json.dumps({"name": name, "arguments": args}, separators=(",", ":"), sort_keys=True)


def history_text(messages) -> str:
    """Render a message prefix to text (markers, no EOS) — the multi-turn decode/training context."""
    parts = []
    for m in messages:
        if m.role == Role.user:
            parts.append(USER + m.content)
        elif m.role == Role.tool:
            parts.append(TOOL + TOOL_RESPONSE_OPEN + (m.tool_response or "") + TOOL_RESPONSE_CLOSE)
        elif m.role == Role.assistant:
            if m.tool_calls:
                body = "".join(
                    TOOL_CALL_OPEN + _canon(c.name, c.arguments) + TOOL_CALL_CLOSE
                    for c in m.tool_calls
                )
            else:
                body = m.content
            parts.append(ASSISTANT + body)
    return "".join(parts)


def render_conversation(conv: Conversation, tok) -> tuple[list[int], list[int]]:
    """Render a multi-turn Conversation to (input_ids, labels); loss is on every assistant turn
    (tool calls + final text + per-turn EOS). User and tool-response tokens are masked."""
    ids: list[int] = []
    labels: list[int] = []

    def add(text: str, learn: bool):
        t = tok.encode(text)
        ids.extend(t)
        labels.extend(t if learn else [IGNORE] * len(t))

    for m in conv.messages:
        if m.role == Role.user:
            add(USER + m.content, False)
        elif m.role == Role.tool:
            add(TOOL + TOOL_RESPONSE_OPEN + (m.tool_response or "") + TOOL_RESPONSE_CLOSE, False)
        elif m.role == Role.assistant:
            add(ASSISTANT, False)  # marker is part of the prompt
            if m.tool_calls:
                body = "".join(
                    TOOL_CALL_OPEN + _canon(c.name, c.arguments) + TOOL_CALL_CLOSE
                    for c in m.tool_calls
                )
            else:
                body = m.content
            b = tok.encode(body) + [tok.eos_id]
            ids.extend(b)
            labels.extend(b)  # learn the assistant body + end-of-turn
    return ids, labels


def render_conversation_rows(
    conv: Conversation,
    tok,
    *,
    prompt_contract: str | None = None,
    max_seq_len: int | None = None,
    catalog_cache: CatalogTokenCache | None = None,
) -> list[RenderedTokenRow]:
    """Render one or more masked LM rows under an explicit conversation prompt contract.

    The default legacy contract wraps :func:`render_conversation` and retains its historical
    whole-trajectory, left-truncating behavior. ``openai_full_catalog_v1`` instead emits one exact
    eval-parity row per assistant decision. Its complete prompt is masked and only that decision's
    assistant body plus EOS is learned. Full-catalog rows fail closed rather than truncate.
    """

    contract = assert_prompt_contract_tokenizer(tok, prompt_contract)
    _validate_max_seq_len(max_seq_len)

    if contract == LEGACY_CONVERSATION_PROMPT_CONTRACT:
        ids, labels = render_conversation(conv, tok)
        if max_seq_len is not None:
            ids = ids[-max_seq_len:]
            labels = labels[-max_seq_len:]
        return [(ids, labels)]

    if contract != OPENAI_FULL_CATALOG_V1:  # pragma: no cover - resolver guards this branch
        raise AssertionError(f"unhandled conversation prompt contract {contract!r}")

    cache = catalog_cache if catalog_cache is not None else CatalogTokenCache(tok)
    return _render_full_catalog_inputs_scalar(
        _full_catalog_token_inputs(conv, cache),
        tok,
        max_seq_len=max_seq_len,
    )


def render_conversation_rows_batch(
    conversations: Sequence[Conversation],
    tok,
    *,
    prompt_contract: str | None = None,
    max_seq_len: int | None = None,
    catalog_cache: CatalogTokenCache | None = None,
    batch_size: int = 256,
) -> list[RenderedTokenRow]:
    """Render conversations in exact source order with bounded tokenizer batches.

    Legacy rendering delegates to the singular API one conversation at a time. The full-catalog
    contract batches catalog-independent suffix/body pairs while retaining each interned catalog
    tuple as a shared lazy prefix. Validation and scalar replay keep observable failures ordered
    as though :func:`render_conversation_rows` had been called for each conversation in turn.
    """

    contract = assert_prompt_contract_tokenizer(tok, prompt_contract)
    _validate_max_seq_len(max_seq_len)
    _validate_batch_size(batch_size)

    cache = catalog_cache if catalog_cache is not None else CatalogTokenCache(tok)
    if contract == LEGACY_CONVERSATION_PROMPT_CONTRACT:
        rows: list[RenderedTokenRow] = []
        for conversation in conversations:
            rows.extend(
                render_conversation_rows(
                    conversation,
                    tok,
                    prompt_contract=contract,
                    max_seq_len=max_seq_len,
                    catalog_cache=cache,
                )
            )
        return rows

    if contract != OPENAI_FULL_CATALOG_V1:  # pragma: no cover - resolver guards this branch
        raise AssertionError(f"unhandled conversation prompt contract {contract!r}")

    rows = []
    pending: list[_FullCatalogTokenInput] = []

    def flush() -> None:
        if not pending:
            return
        rows.extend(
            _render_full_catalog_inputs_batch(
                pending,
                tok,
                max_seq_len=max_seq_len,
            )
        )
        pending.clear()

    for conversation in conversations:
        try:
            conversation_inputs = _full_catalog_token_inputs(conversation, cache)
        except Exception:
            # Finish earlier scalar-equivalent work before surfacing a later catalog/message
            # validation failure. If that earlier work fails, its source-order error wins.
            flush()
            raise
        for item in conversation_inputs:
            pending.append(item)
            if len(pending) == batch_size:
                flush()
    flush()
    return rows


def assistant_body(s: Sample) -> str:
    if s.kind != "tool":
        return s.target
    if s.calls:  # parallel: one <tool_call> block per call
        return "".join(
            f"{TOOL_CALL_OPEN}{_canon(c['name'], c['arguments'])}{TOOL_CALL_CLOSE}" for c in s.calls
        )
    return f"{TOOL_CALL_OPEN}{s.target}{TOOL_CALL_CLOSE}"


def prompt_text(s: Sample) -> str:
    return f"{USER}{s.prompt}{ASSISTANT}"


def render_sft(s: Sample, tok) -> tuple[list[int], list[int]]:
    """Return (input_ids, labels) of equal length; labels masked over the prompt."""
    p = tok.encode(prompt_text(s))
    b = tok.encode(assistant_body(s)) + [tok.eos_id]
    ids = p + b
    labels = [IGNORE] * len(p) + b
    return ids, labels


def render_full_text(s: Sample) -> str:
    """Full conversation text (for pretraining as a plain LM stream)."""
    return prompt_text(s) + assistant_body(s)


def build_pretrain_stream(samples: list[Sample], tok) -> list[int]:
    stream: list[int] = []
    for s in samples:
        stream.extend(tok.encode(render_full_text(s)))
        stream.append(tok.eos_id)  # document separator
    return stream
