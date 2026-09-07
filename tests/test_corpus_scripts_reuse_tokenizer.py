"""A corpus script that packs against the shared tokenizer must reuse it, never retrain it.

build_corpus.py retrains the BPE and OVERWRITES --tokenizer-path unless --reuse-tokenizer is
passed, and the overwrite is silent. fetch_pretrain_4b.sh omitted it, packing rewrote
data/tokenizer-h100-16k.json mid-run, and every checkpoint trained against the old file became
unloadable by the next stage - the lineage guard fired correctly, on damage already done. The
original tokenizer had no other copy on the box.

The tokenizer is also what makes shards comparable: a corpus packed against a different BPE
cannot be compared with any run that used the old one, so the flag is a correctness requirement
and not just a safety one.
"""

from pathlib import Path

import pytest

SCRIPTS = sorted(Path("scripts").glob("*.sh"))
PACKERS = [p for p in SCRIPTS if "--tokenizer-path" in p.read_text(encoding="utf-8")]


def test_there_is_something_to_check():
    assert PACKERS, "no script packs a corpus; this guard would be vacuous"


@pytest.mark.parametrize("path", PACKERS, ids=lambda p: p.name)
def test_packer_reuses_the_shared_tokenizer(path: Path):
    body = path.read_text(encoding="utf-8")
    assert "--reuse-tokenizer" in body, (
        f"{path.name} passes --tokenizer-path without --reuse-tokenizer, so build_corpus.py will "
        "retrain the BPE and overwrite that file. Every checkpoint trained against the old "
        "tokenizer then fails its lineage check, and the old file has no backup."
    )
