"""Immutable periodic checkpoint archives for supervised fine-tuning."""

from __future__ import annotations

import copy
import shutil

import pytest
import torch

from openlocalagent.data.synth.agent_synth import Sample
from openlocalagent.model import LocalAgentLM, ModelConfig
from openlocalagent.model.tokenizer import ByteTokenizer
from openlocalagent.train.sft import sft
from openlocalagent.train.stage_data import sha256_file


def _config() -> ModelConfig:
    return ModelConfig(
        name="sft-checkpoint-archive-test",
        vocab_size=256,
        d_model=16,
        embed_dim=8,
        n_layers=1,
        n_loops=1,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden=32,
        max_seq_len=64,
        dropout=0.0,
    )


def _sample() -> Sample:
    return Sample(
        category="text",
        group="text",
        prompt="Say a",
        kind="text",
        target="a",
    )


def _model(state: dict[str, torch.Tensor]) -> LocalAgentLM:
    model = LocalAgentLM(_config())
    model.load_state_dict(state)
    return model


def test_sft_periodic_archives_are_sealed_complete_and_retry_idempotent(
    tmp_path,
) -> None:
    torch.manual_seed(811)
    initial_state = copy.deepcopy(LocalAgentLM(_config()).state_dict())
    checkpoint_path = tmp_path / "latest.pt"

    sft(
        _model(initial_state),
        [_sample()],
        ByteTokenizer(),
        steps=2,
        batch_size=1,
        warmup=0,
        joint_tool_head=False,
        seed=17,
        checkpoint_path=checkpoint_path,
        checkpoint_every=1,
        archive_checkpoints=True,
        log=lambda *_: None,
    )

    first_archive = tmp_path / "latest.step-00000001.pt"
    final_archive = tmp_path / "latest.step-00000002.pt"
    assert first_archive.is_file()
    assert final_archive.is_file()
    first = torch.load(first_archive, map_location="cpu", weights_only=True)
    final = torch.load(final_archive, map_location="cpu", weights_only=True)
    assert first["step"] == 0
    assert first["sampling_state"]["completed_steps"] == 1
    assert final["step"] == 1
    assert final["sampling_state"]["completed_steps"] == 2
    assert final["resume_integrity_sha256"]
    assert final["training_contract"]["archive_checkpoints"] is True
    assert final["training_contract"]["checkpoint_archive_every"] == 1
    assert (
        final["training_contract"]["checkpoint_archive_format"]
        == "immutable_periodic_sft_v1"
    )

    archive_sha256 = {
        first_archive: sha256_file(first_archive),
        final_archive: sha256_file(final_archive),
    }
    # The completed-checkpoint resume takes no optimizer steps, then reaches the existing
    # final-save path again. Both periodic archives must remain byte-for-byte untouched.
    sft(
        _model(initial_state),
        [_sample()],
        ByteTokenizer(),
        steps=2,
        batch_size=1,
        warmup=0,
        joint_tool_head=False,
        seed=17,
        checkpoint_path=checkpoint_path,
        checkpoint_every=1,
        archive_checkpoints=True,
        resume_from=checkpoint_path,
        log=lambda *_: None,
    )
    assert {
        first_archive: sha256_file(first_archive),
        final_archive: sha256_file(final_archive),
    } == archive_sha256


def test_sft_archive_creation_refuses_to_overwrite_different_valid_checkpoint(
    tmp_path,
) -> None:
    torch.manual_seed(812)
    initial_state = copy.deepcopy(LocalAgentLM(_config()).state_dict())
    checkpoint_path = tmp_path / "latest.pt"
    common = {
        "steps": 1,
        "batch_size": 1,
        "warmup": 0,
        "joint_tool_head": False,
        "seed": 19,
        "checkpoint_every": 1,
        "archive_checkpoints": True,
        "log": lambda *_: None,
    }
    sft(
        _model(initial_state),
        [_sample()],
        ByteTokenizer(),
        checkpoint_path=checkpoint_path,
        **common,
    )

    torch.manual_seed(813)
    different_state = copy.deepcopy(LocalAgentLM(_config()).state_dict())
    different_checkpoint = tmp_path / "different.pt"
    sft(
        _model(different_state),
        [_sample()],
        ByteTokenizer(),
        checkpoint_path=different_checkpoint,
        **common,
    )
    archive_path = tmp_path / "latest.step-00000001.pt"
    shutil.copyfile(
        tmp_path / "different.step-00000001.pt",
        archive_path,
    )
    conflicting_sha256 = sha256_file(archive_path)

    # A fresh run, not a resume: resuming a checkpoint that already completed its steps runs
    # nothing and writes nothing (the final save is guarded), so it never reaches the archive.
    # The property under test is that a run which does reach step 1 refuses a different,
    # valid archive sitting at that step.
    with pytest.raises(FileExistsError, match="refusing to overwrite different"):
        sft(
            _model(initial_state),
            [_sample()],
            ByteTokenizer(),
            checkpoint_path=checkpoint_path,
            **common,
        )
    assert sha256_file(archive_path) == conflicting_sha256


def test_sft_archive_default_is_off_and_preserves_legacy_contract(tmp_path) -> None:
    torch.manual_seed(814)
    checkpoint_path = tmp_path / "latest.pt"
    sft(
        LocalAgentLM(_config()),
        [_sample()],
        ByteTokenizer(),
        steps=1,
        batch_size=1,
        warmup=0,
        joint_tool_head=False,
        seed=23,
        checkpoint_path=checkpoint_path,
        checkpoint_every=1,
        log=lambda *_: None,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert "archive_checkpoints" not in checkpoint["training_contract"]
    assert not (tmp_path / "latest.step-00000001.pt").exists()


@pytest.mark.parametrize(
    ("checkpoint_path", "checkpoint_every", "archive_checkpoints", "message"),
    [
        (None, 1, True, "requires checkpoint_path"),
        ("latest.pt", 0, True, "requires positive checkpoint_every"),
        ("latest.pt", 1, "yes", "must be boolean"),
    ],
)
def test_sft_archive_contract_rejects_incomplete_or_ambiguous_configuration(
    tmp_path,
    checkpoint_path,
    checkpoint_every,
    archive_checkpoints,
    message,
) -> None:
    resolved_path = (
        None if checkpoint_path is None else tmp_path / checkpoint_path
    )
    with pytest.raises((TypeError, ValueError), match=message):
        sft(
            LocalAgentLM(_config()),
            [_sample()],
            ByteTokenizer(),
            steps=1,
            batch_size=1,
            warmup=0,
            joint_tool_head=False,
            checkpoint_path=resolved_path,
            checkpoint_every=checkpoint_every,
            archive_checkpoints=archive_checkpoints,
            log=lambda *_: None,
        )


def test_run_level_no_op_resume_is_decided_from_the_checkpoint_step(tmp_path) -> None:
    from openlocalagent.train.sft import _sft_already_complete

    out = tmp_path / "run"
    out.mkdir()
    config = {"runtime": {"resume": True}, "log": {"out_dir": str(out)},
              "schedule": {"total_steps": 10}}
    assert not _sft_already_complete(config, None)          # no checkpoint yet
    torch.save({"step": 8}, out / "latest.pt")
    assert not _sft_already_complete(config, None)          # one step left
    torch.save({"step": 9}, out / "latest.pt")
    assert _sft_already_complete(config, None)              # last step recorded
    assert not _sft_already_complete(config, False)         # resume not requested
    assert not _sft_already_complete({**config, "runtime": {}}, None)

