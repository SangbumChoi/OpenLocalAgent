from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from openlocalagent.model.tokenizer import ASSISTANT, TOOL_CALL_OPEN, USER, train_bpe


ROOT = Path(__file__).parents[1]
WEB_TOKENIZER = ROOT / "demos" / "webgpu" / "tokenizer.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_browser_bytelevel_bpe_matches_training_tokenizer(tmp_path):
    pytest.importorskip("tokenizers")
    corpus = [
        "Hello world! Move src/app.py into backup/app.py.",
        f"{USER}click 'the Confirm button'{ASSISTANT}{TOOL_CALL_OPEN}",
        "Unicode: café 한글 😀 — tabs\tand\nnewlines.",
        "Multiple  spaces, contractions: we're testing; don't stop.",
    ] * 30
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = train_bpe(corpus, tokenizer_path, vocab_size=384, min_frequency=1)
    samples = [
        "Hello world!",
        f"{USER}click the button{ASSISTANT}",
        "Move src/app.py to dst/app.py",
        "café 한글 😀",
        " leading  spaces\n",
        "we're testing; don't stop.",
        f"prefix{TOOL_CALL_OPEN}read_file suffix",
        "unseen bytes: Ω≈ç√∫˜µ≤≥÷",
    ]
    expected = [{"ids": tokenizer.encode(text), "decoded": text} for text in samples]
    samples_path = tmp_path / "samples.json"
    samples_path.write_text(json.dumps(samples, ensure_ascii=False), encoding="utf-8")

    script = """
const fs = require("fs");
const { ByteLevelBPETokenizer } = require(process.argv[1]);
const spec = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const samples = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const tok = new ByteLevelBPETokenizer(
  { vocab_size: Object.keys(spec.model.vocab).length, eos_id: 0 },
  spec
);
process.stdout.write(JSON.stringify(samples.map((text) => {
  const ids = tok.encode(text);
  return { ids, decoded: tok.decode(ids, false) };
})));
"""
    result = subprocess.run(
        [
            shutil.which("node"),
            "-e",
            script,
            str(WEB_TOKENIZER),
            str(tokenizer_path),
            str(samples_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == expected


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_browser_byte_tokenizer_roundtrip(tmp_path):
    sample = f"{USER}café 한글 😀{ASSISTANT}"
    sample_path = tmp_path / "sample.txt"
    sample_path.write_text(sample, encoding="utf-8")
    script = """
const fs = require("fs");
const { Utf8ByteTokenizer } = require(process.argv[1]);
const text = fs.readFileSync(process.argv[2], "utf8");
const tok = new Utf8ByteTokenizer({ vocab_size: 256, eos_id: 0 });
const ids = tok.encode(text);
process.stdout.write(JSON.stringify({ ids, decoded: tok.decode(ids, false) }));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(WEB_TOKENIZER), str(sample_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["ids"] == list(sample.encode("utf-8"))
    assert payload["decoded"] == sample
