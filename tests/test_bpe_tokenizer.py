from openlocalagent.model.tokenizer import (
    ASSISTANT,
    TOOL,
    TOOL_CALL_OPEN,
    USER,
    BPETokenizer,
    ByteTokenizer,
    batched_token_lengths,
    train_bpe,
)
from openlocalagent.train.sft import _encode_with_value_span


def test_bpe_roundtrip_special_markers_and_reload(tmp_path):
    corpus = [
        "Move src/app.py into backup/app.py and run the tests.",
        "Use tools carefully. " * 30,
        f"{ASSISTANT}{TOOL_CALL_OPEN}read_file",
    ] * 10
    path = tmp_path / "tokenizer.json"
    tok = train_bpe(corpus, path, vocab_size=320, min_frequency=1)
    text = f"{ASSISTANT}{TOOL_CALL_OPEN}read_file src/app.py"
    ids = tok.encode(text, add_eos=True)
    assert ids[-1] == tok.eos_id == 0
    assert tok.decode(ids) == text
    marker_id = tok._tokenizer.token_to_id(TOOL_CALL_OPEN)
    assert marker_id in tok.encode(TOOL_CALL_OPEN)
    loaded = BPETokenizer.from_file(path)
    assert loaded.encode(text) == tok.encode(text)


def test_batched_token_lengths_match_scalar_byte_and_bpe(tmp_path):
    texts = [
        "",
        "plain text",
        f"{USER}Unicode 서울{ASSISTANT}",
        f"{TOOL_CALL_OPEN}read_file",
    ]
    bpe_path = tmp_path / "tokenizer.json"
    bpe = train_bpe(texts * 10, bpe_path, vocab_size=320, min_frequency=1)

    for tokenizer in (ByteTokenizer(), bpe):
        expected = [len(tokenizer.encode(text)) for text in texts]
        assert batched_token_lengths(tokenizer, iter(texts), batch_size=2) == expected
        assert tokenizer.encode_batch(texts) == [tokenizer.encode(text) for text in texts]


def test_bpe_pointer_span_uses_contextual_leading_space_token(tmp_path):
    path = tmp_path / "tokenizer.json"
    corpus = [f"{USER}Weather in Seoul{ASSISTANT} Seoul weather forecast"] * 30
    tok = train_bpe(corpus, path, vocab_size=320, min_frequency=1)
    text = f"{USER}Weather in Seoul{ASSISTANT}"

    ids, span = _encode_with_value_span(tok, text, "Seoul", max_seq_len=128)

    assert span is not None
    start, end = span
    assert tok.decode(ids[start:end + 1]).strip() == "Seoul"
    isolated = tok.encode("Seoul")
    assert not any(ids[index:index + len(isolated)] == isolated for index in range(len(ids)))


def test_bpe_pointer_span_selects_retained_multi_turn_occurrence(tmp_path):
    path = tmp_path / "tokenizer.json"
    corpus = [
        f"{USER}Search for Busan{ASSISTANT}{TOOL}The result path is Busan{ASSISTANT}"
    ] * 30
    tok = train_bpe(corpus, path, vocab_size=320, min_frequency=1)
    text = (
        f"{USER}Search for Busan{ASSISTANT}"
        + ("old context " * 20)
        + f"{TOOL}The result path is Busan{ASSISTANT}"
    )

    ids, span = _encode_with_value_span(tok, text, "Busan", max_seq_len=12)

    assert span is not None
    start, end = span
    assert tok.decode(ids[start:end + 1]).strip() == "Busan"
