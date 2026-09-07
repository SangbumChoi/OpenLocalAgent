"""Evaluation must refuse a checkpoint whose vocabulary is not the one being loaded.

Training already refuses to chain onto such a checkpoint. Evaluation did not, which is the worse
half of the asymmetry: a training failure stops a run, a scoring failure becomes a published
number. Decoding under the wrong BPE table yields well-formed nonsense - measured once at 0.0
across every suite when 64k checkpoints met the 16k table.
"""

import hashlib

import pytest

from openlocalagent.eval.suite import TokenizerLineage, tokenizer_lineage


def _payload(sha: str) -> dict:
    return {"tokenizer": {"kind": "bpe", "path": "data/tokenizer-h100-16k.json", "sha256": sha}}


def test_matching_lineage_is_silent(tmp_path):
    body = b'{"model": {}}'
    tok = tmp_path / "tokenizer-h100-16k.json"
    tok.write_bytes(body)
    sha = hashlib.sha256(body).hexdigest()
    lineage = tokenizer_lineage(_payload(sha), str(tok))
    assert lineage is not None and lineage.matches


def test_mismatch_is_detected_and_explains_itself(tmp_path):
    tok = tmp_path / "tokenizer-h100-16k.json"
    tok.write_bytes(b"a different vocabulary")
    lineage = tokenizer_lineage(_payload("236945b5" + "0" * 56), str(tok))
    assert lineage is not None and not lineage.matches
    # The message has to say what to do, not just that something is wrong.
    assert "236945b5" in lineage.complaint
    assert "--allow-tokenizer-mismatch" in lineage.complaint


def test_receipt_note_carries_both_identities(tmp_path):
    tok = tmp_path / "t.json"
    tok.write_bytes(b"x")
    note = tokenizer_lineage(_payload("f" * 64), str(tok)).as_receipt_note()
    assert note["checkpoint_tokenizer_sha256"] == "f" * 64
    assert note["loaded_tokenizer_sha256"] == hashlib.sha256(b"x").hexdigest()
    assert "not comparable" in note["warning"]


def test_checkpoint_without_recorded_identity_is_not_blocked(tmp_path):
    """Older artifacts predate the metadata; refusing them would retire valid past results."""
    tok = tmp_path / "t.json"
    tok.write_bytes(b"x")
    assert tokenizer_lineage({"cfg": {}}, str(tok)) is None


def test_missing_tokenizer_file_is_not_a_lineage_verdict(tmp_path):
    assert tokenizer_lineage(_payload("f" * 64), str(tmp_path / "absent.json")) is None


@pytest.mark.parametrize("same", [True, False])
def test_matches_is_pure_comparison(same):
    other = "a" * 64 if same else "b" * 64
    assert TokenizerLineage("a" * 64, other, "p").matches is same
