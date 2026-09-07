"""Fail-closed contracts for explicit SFT Git-lineage migration receipts."""

from __future__ import annotations

import json

import pytest

from openlocalagent.train.resume_git_receipt import (
    assert_resume_git_receipt,
    build_resume_git_receipt,
    load_resume_git_receipt,
    write_resume_git_receipt,
)


def _lineage(
    worktree: str,
    *,
    dirty: bool,
    data_sha256: str = "d" * 64,
) -> dict:
    return {
        "version": 1,
        "stage": "sft",
        "config_sha256": "a" * 64,
        "model_config_sha256": "b" * 64,
        "data_sha256": data_sha256,
        "tokenizer_sha256": "c" * 64,
        "parent_checkpoint_sha256": "e" * 64,
        "git": {
            # Production Git repositories commonly still use 40-hex SHA-1 object IDs.
            "commit": "1" * 40,
            "repository_sha256": "2" * 64,
            "dirty": dirty,
            "worktree_sha256": worktree,
        },
    }


def _receipt() -> tuple[dict, dict, dict]:
    recorded = _lineage("3" * 64, dirty=True)
    expected = _lineage("4" * 64, dirty=True)
    receipt = build_resume_git_receipt(
        checkpoint_sha256="5" * 64,
        recorded_lineage=recorded,
        expected_lineage=expected,
        stage="sft",
        reason="SFT resume startup optimization changed no numerical path.",
        evidence=["focused exact-resume equivalence passed"],
    )
    return receipt, recorded, expected


def test_resume_git_receipt_is_deterministic_self_hashed_and_never_overwrites(
    tmp_path,
) -> None:
    _, recorded, expected = _receipt()
    path = tmp_path / "receipt.json"
    first = write_resume_git_receipt(
        path,
        checkpoint_sha256="5" * 64,
        recorded_lineage=recorded,
        expected_lineage=expected,
        stage="sft",
        reason="SFT resume startup optimization changed no numerical path.",
        evidence=["focused exact-resume equivalence passed"],
    )
    loaded = load_resume_git_receipt(path)
    assert loaded == first
    assert_resume_git_receipt(
        loaded,
        checkpoint_sha256="5" * 64,
        recorded_lineage=recorded,
        expected_lineage=expected,
        stage="sft",
    )
    with pytest.raises(FileExistsError):
        write_resume_git_receipt(
            path,
            checkpoint_sha256="5" * 64,
            recorded_lineage=recorded,
            expected_lineage=expected,
            stage="sft",
            reason="same request",
            evidence=["must not overwrite"],
        )


def test_resume_git_receipt_rejects_tamper_stale_current_and_wrong_checkpoint(
    tmp_path,
) -> None:
    receipt, recorded, expected = _receipt()
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    tampered = dict(load_resume_git_receipt(path))
    tampered["reason"] = "forged"
    with pytest.raises(ValueError, match="self-hash mismatch"):
        assert_resume_git_receipt(
            tampered,
            checkpoint_sha256="5" * 64,
            recorded_lineage=recorded,
            expected_lineage=expected,
            stage="sft",
        )
    with pytest.raises(ValueError, match="checkpoint sha256 mismatch"):
        assert_resume_git_receipt(
            receipt,
            checkpoint_sha256="6" * 64,
            recorded_lineage=recorded,
            expected_lineage=expected,
            stage="sft",
        )
    stale_expected = _lineage("7" * 64, dirty=True)
    with pytest.raises(ValueError, match="expected current lineage mismatch"):
        assert_resume_git_receipt(
            receipt,
            checkpoint_sha256="5" * 64,
            recorded_lineage=recorded,
            expected_lineage=stale_expected,
            stage="sft",
        )


@pytest.mark.parametrize(
    ("recorded", "expected", "message"),
    [
        (
            _lineage("3" * 64, dirty=True),
            _lineage("4" * 64, dirty=True, data_sha256="f" * 64),
            "non-Git lineage drift",
        ),
        (
            _lineage("3" * 64, dirty=True),
            {
                **_lineage("4" * 64, dirty=True),
                "git": {
                    **_lineage("4" * 64, dirty=True)["git"],
                    "commit": "8" * 40,
                },
            },
            "lineage.git.commit drift",
        ),
        (
            _lineage("3" * 64, dirty=True),
            {
                **_lineage("4" * 64, dirty=True),
                "git": {
                    **_lineage("4" * 64, dirty=True)["git"],
                    "repository_sha256": "9" * 64,
                },
            },
            "lineage.git.repository_sha256 drift",
        ),
    ],
)
def test_resume_git_receipt_never_authorizes_non_worktree_drift(
    recorded,
    expected,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_resume_git_receipt(
            checkpoint_sha256="5" * 64,
            recorded_lineage=recorded,
            expected_lineage=expected,
            stage="sft",
            reason="not sufficient",
            evidence=["must fail closed"],
        )
