"""Tokenizer — byte-level deployment default plus an optional trained BPE tier.

The ultra-tiny tier is **byte-level**: vocab is exactly 256 (one id per UTF-8 byte), so the
model pays no embedding tax and needs no trained tokenizer. Agent markers (``<|user|>``,
``<tool_call>`` …) are just literal UTF-8 text the model learns to emit — they stay in the 256
byte space. Byte ``0x00`` is reserved as EOS/PAD (it never appears in valid UTF-8 of our data).

The BPE wrapper uses the small ``tokenizers`` dependency directly (never ``transformers``).
Agent markers are registered as indivisible special tokens and ``<|end|>`` is id 0, preserving
the shared EOS/PAD convention used by the byte tokenizer.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

EOS_ID = 0      # reserved byte, end-of-sequence
PAD_ID = 0      # same byte doubles as pad (masked out of the loss)

# Literal text markers used to frame conversations (plain bytes, not new vocab ids).
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
TOOL = "<|tool|>"
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"

SPECIAL_MARKERS = [
    USER, ASSISTANT, TOOL,
    TOOL_CALL_OPEN, TOOL_CALL_CLOSE,
    TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE,
]
BPE_EOS = "<|end|>"
BPE_SPECIAL_TOKENS = [BPE_EOS, *SPECIAL_MARKERS]


class ByteTokenizer:
    """UTF-8 byte tokenizer. vocab_size == 256."""

    vocab_size = 256
    eos_id = EOS_ID
    pad_id = PAD_ID

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = list(text.encode("utf-8"))
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def encode_batch(
        self,
        texts: Iterable[str],
        add_eos: bool = False,
    ) -> list[list[int]]:
        """Encode a finite text batch with the exact scalar semantics."""

        return [self.encode(text, add_eos=add_eos) for text in texts]

    def decode(self, ids: list[int], stop_at_eos: bool = True) -> str:
        out = []
        for i in ids:
            if stop_at_eos and i == EOS_ID:
                break
            out.append(i)
        return bytes(out).decode("utf-8", errors="replace")


class BPETokenizer:
    """Thin serializable wrapper around a Hugging Face ``tokenizers.Tokenizer``."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        eos_id = tokenizer.token_to_id(BPE_EOS)
        if eos_id != 0:
            raise ValueError(f"BPE tokenizer must assign {BPE_EOS!r} id 0, got {eos_id}")
        self.eos_id = 0
        self.pad_id = 0
        self.vocab_size = tokenizer.get_vocab_size()

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = self._tokenizer.encode(text, add_special_tokens=False).ids
        return [*ids, self.eos_id] if add_eos else ids

    def encode_batch(
        self,
        texts: Iterable[str],
        add_eos: bool = False,
    ) -> list[list[int]]:
        """Encode a finite batch through the tokenizer's deterministic Rust batch path."""

        values = list(texts)
        encodings = self._tokenizer.encode_batch(values, add_special_tokens=False)
        if add_eos:
            return [[*encoding.ids, self.eos_id] for encoding in encodings]
        return [encoding.ids for encoding in encodings]

    def decode(self, ids: list[int], stop_at_eos: bool = True) -> str:
        if stop_at_eos and self.eos_id in ids:
            ids = ids[: ids.index(self.eos_id)]
        return self._tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(path))

    @classmethod
    def from_file(cls, path: str | Path) -> BPETokenizer:
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_file(str(path)))


def train_bpe(
    documents: Iterable[str],
    path: str | Path,
    *,
    vocab_size: int = 16_384,
    min_frequency: int = 2,
) -> BPETokenizer:
    """Train byte-fallback BPE from an iterable and save it as one inspectable JSON file."""

    if vocab_size <= len(BPE_SPECIAL_TOKENS) + 256:
        raise ValueError("vocab_size is too small for special tokens plus byte fallback")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE(unk_token=None, byte_fallback=True))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=BPE_SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(documents, trainer=trainer)
    wrapped = BPETokenizer(tokenizer)
    wrapped.save(path)
    return wrapped


def load_tokenizer(kind: str = "byte", path: str | Path | None = None):
    if kind == "byte":
        return ByteTokenizer()
    if kind == "bpe":
        if path is None:
            raise ValueError("BPE tokenizer requires path=")
        return BPETokenizer.from_file(path)
    raise ValueError(f"unknown tokenizer kind {kind!r}")


def batched_token_lengths(
    tokenizer,
    texts: Iterable[str],
    *,
    batch_size: int = 256,
) -> list[int]:
    """Return scalar-equivalent token lengths without retaining the complete text corpus."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    lengths: list[int] = []
    batch: list[str] = []

    def flush() -> None:
        if not batch:
            return
        encode_batch = getattr(tokenizer, "encode_batch", None)
        encoded = (
            encode_batch(batch, add_eos=False)
            if callable(encode_batch)
            else [tokenizer.encode(text, add_eos=False) for text in batch]
        )
        if len(encoded) != len(batch):
            raise RuntimeError("tokenizer batch output does not align with its input texts")
        lengths.extend(len(token_ids) for token_ids in encoded)
        batch.clear()

    for text in texts:
        if not isinstance(text, str):
            raise TypeError("batched_token_lengths inputs must be text")
        batch.append(text)
        if len(batch) == batch_size:
            flush()
    flush()
    return lengths
