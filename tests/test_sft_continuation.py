"""Frozen contracts for the 1M SFT recovery child."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from localagent.model import ModelConfig
from localagent.train.sft import (
    resolve_sft_continuation,
    validate_sft_continuation_parent,
)

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "configs" / "train"
PARENT_PATH = TRAIN / "sft-paper-tier-1m.yaml"
CONTINUATION_PATH = TRAIN / "sft-paper-tier-1m-continuation.yaml"
FORMAT_BOOTSTRAP_PATH = TRAIN / "sft-paper-tier-1m-format-bootstrap.yaml"
MIXED_REPLAY_PATH = TRAIN / "sft-paper-tier-1m-mixed-replay-pilot.yaml"
PARENT_ANCHOR_PULSE_PATH = (
    TRAIN / "sft-paper-tier-1m-parent-anchor-pulse-pilot.yaml"
)
RL_PATH = TRAIN / "rl-paper-tier-1m.yaml"
TOTAL_TRAIN_DECISIONS = 93_504
FORMAT_BOOTSTRAP_DECISIONS = 8_192
MIXED_REPLAY_DECISIONS = 4_096
PARENT_ANCHOR_PULSE_DECISIONS = 5_952
SEALED_PARENT = {
    "checkpoint_sha256": (
        "1913123ea0982f675f0add7c5b23154faf6adda99424a0a2009130104c32021f"
    ),
    "resume_integrity_sha256": (
        "71bdebf928b157a695b0a6ebf4b5dac0fea72a9e30b916f72cba71834b1df693"
    ),
    "training_contract_sha256": (
        "6032ac576b96655aa5df6646bf0bf655aa7b64f056083a1a29dc67c50e13700b"
    ),
    "lm_sampling_sha256": (
        "4fe6c5e1b275981575514b0c5c33d5b022b7884ad596fb1050892fa16f9dbba1"
    ),
    "completed_steps": 348,
    "completed_lm_cursor": 5_568,
}


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _selected_decisions(config: dict) -> int:
    return (
        int(config["schedule"]["total_steps"])
        * int(config["batch"]["micro_batch_size"])
        * int(config["batch"]["grad_accum_steps"])
    )


def test_mode_only_continuation_contracts_remain_byte_compatible() -> None:
    for path in (
        CONTINUATION_PATH,
        FORMAT_BOOTSTRAP_PATH,
        MIXED_REPLAY_PATH,
    ):
        config = _load(path)
        assert resolve_sft_continuation(config) == config["continuation"] == {
            "mode": "fresh_optimizer_sft_child_v1"
        }


def test_sealed_parent_contract_is_exact_and_normalized() -> None:
    config = {
        "continuation": {
            "mode": "fresh_optimizer_sft_child_v1",
            "parent": copy.deepcopy(SEALED_PARENT),
        }
    }

    assert validate_sft_continuation_parent(SEALED_PARENT) == SEALED_PARENT
    assert resolve_sft_continuation(config) == config["continuation"]


@pytest.mark.parametrize(
    ("parent", "error", "message"),
    [
        (None, TypeError, r"continuation\.parent must be a mapping"),
        (
            {
                key: value
                for key, value in SEALED_PARENT.items()
                if key != "checkpoint_sha256"
            },
            ValueError,
            "must contain exactly",
        ),
        (
            {**SEALED_PARENT, "extra": "forbidden"},
            ValueError,
            "must contain exactly",
        ),
        (
            {**SEALED_PARENT, "checkpoint_sha256": "A" * 64},
            ValueError,
            "lowercase SHA-256",
        ),
        (
            {**SEALED_PARENT, "completed_steps": True},
            ValueError,
            "positive integer",
        ),
        (
            {**SEALED_PARENT, "completed_lm_cursor": -1},
            ValueError,
            "non-negative integer",
        ),
    ],
)
def test_sealed_parent_contract_rejects_schema_and_type_drift(
    parent,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        validate_sft_continuation_parent(parent)


def test_1m_sft_continuation_is_a_distinct_fresh_optimizer_tail_plan() -> None:
    parent = _load(PARENT_PATH)
    continuation = _load(CONTINUATION_PATH)
    rl = _load(RL_PATH)

    parent_decisions = _selected_decisions(parent)
    continuation_decisions = _selected_decisions(continuation)
    assert parent_decisions == 5_568
    assert continuation["continuation"] == {
        "mode": "fresh_optimizer_sft_child_v1"
    }
    assert continuation["init_from"] == f"{parent['log']['out_dir']}/latest.pt"
    assert continuation["log"]["out_dir"] != parent["log"]["out_dir"]
    assert continuation["data"]["sampling"] == {
        "mode": "quota_stratified_no_replacement_v1",
        "start_decision": parent_decisions,
    }
    assert parent_decisions + continuation_decisions == TOTAL_TRAIN_DECISIONS
    assert continuation["optim"]["lr"] < parent["optim"]["lr"]
    assert continuation["schedule"]["warmup_steps"] > 0

    # Promotion remains an explicit gate decision; merely adding the recovery plan must not
    # silently relabel the current failed-RL parent.
    assert rl["init_from"] == f"{parent['log']['out_dir']}/latest.pt"
    assert rl["init_from"] != f"{continuation['log']['out_dir']}/latest.pt"


def test_1m_sft_continuation_preserves_model_data_and_evaluation_contracts() -> None:
    parent = _load(PARENT_PATH)
    continuation = _load(CONTINUATION_PATH)

    assert continuation["model_config"] == parent["model_config"]
    model = ModelConfig.from_yaml(str(ROOT / continuation["model_config"]))
    model.assert_within_budget()
    for field in (
        "conversation_prompt_contract",
        "strict_conversation_artifacts",
        "conversations",
        "eval_conversations",
        "tokenizer",
        "seq_len",
        "function_masking",
        "shuffle",
    ):
        assert continuation["data"][field] == parent["data"][field]
    assert continuation["batch"] == parent["batch"]
    assert continuation["evaluation"] == parent["evaluation"]
    assert continuation["heads"] == parent["heads"]
    assert continuation["runtime"] == parent["runtime"]


def test_1m_format_bootstrap_is_one_ordered_derived_corpus_pass() -> None:
    parent = _load(PARENT_PATH)
    bootstrap = _load(FORMAT_BOOTSTRAP_PATH)
    rl = _load(RL_PATH)

    assert bootstrap["continuation"] == {
        "mode": "fresh_optimizer_sft_child_v1"
    }
    assert bootstrap["init_from"] == f"{parent['log']['out_dir']}/latest.pt"
    assert bootstrap["log"]["out_dir"] not in {
        parent["log"]["out_dir"],
        _load(CONTINUATION_PATH)["log"]["out_dir"],
    }
    assert _selected_decisions(bootstrap) == FORMAT_BOOTSTRAP_DECISIONS
    assert bootstrap["data"]["shuffle"] is False
    assert "sampling" not in bootstrap["data"]
    assert bootstrap["data"]["conversations"] == [
        {
            "path": "data/synth/agent_sft_format_bootstrap_v1.jsonl",
            "artifact": {
                "generator_config": "configs/data/agent_sft_format_bootstrap_v1.yaml",
                "manifest": (
                    "data/synth/agent_sft_format_bootstrap_v1.jsonl.manifest.v1.json"
                ),
                "expected_split": "train",
                "expected_rule_verified": True,
                "environment_policy": "forbid",
            },
        }
    ]
    assert bootstrap["optim"]["lr"] < parent["optim"]["lr"]
    assert {
        key: value
        for key, value in bootstrap["batch"].items()
        if key != "pad_to_input_tokens"
    } == parent["batch"]
    assert bootstrap["batch"]["pad_to_input_tokens"] == 3_529
    assert {
        key: value
        for key, value in bootstrap["evaluation"].items()
        if key != "pad_to_input_tokens"
    } == parent["evaluation"]
    assert bootstrap["evaluation"]["pad_to_input_tokens"] == 3_598
    assert (
        bootstrap["schedule"]["total_steps"]
        * bootstrap["batch"]["grad_accum_steps"]
        * bootstrap["batch"]["micro_batch_size"]
        * bootstrap["batch"]["pad_to_input_tokens"]
        == 28_909_568
    )
    assert bootstrap["heads"] == parent["heads"]

    # Adding a targeted recovery child is not itself authorization to change the RL parent.
    assert rl["init_from"] == f"{parent['log']['out_dir']}/latest.pt"
    assert rl["init_from"] != f"{bootstrap['log']['out_dir']}/latest.pt"


def test_1m_mixed_replay_is_a_token_normalized_bounded_parent_child() -> None:
    parent = _load(PARENT_PATH)
    failed_bootstrap = _load(FORMAT_BOOTSTRAP_PATH)
    replay = _load(MIXED_REPLAY_PATH)
    rl = _load(RL_PATH)

    assert replay["continuation"] == {"mode": "fresh_optimizer_sft_child_v1"}
    assert replay["init_from"] == f"{parent['log']['out_dir']}/latest.pt"
    assert replay["init_from"] != f"{failed_bootstrap['log']['out_dir']}/latest.pt"
    assert _selected_decisions(replay) == MIXED_REPLAY_DECISIONS
    assert replay["schedule"] == {
        "type": "cosine",
        "warmup_steps": 16,
        "total_steps": 256,
    }
    assert replay["optim"]["lr"] == 5.0e-6
    assert (
        replay["optim"]["loss_normalization"]
        == "assistant_token_mean_per_update_v1"
    )
    assert replay["batch"] == {
        "micro_batch_size": 2,
        "grad_accum_steps": 8,
        "pad_to_input_tokens": 3_684,
    }
    assert replay["evaluation"]["pad_to_input_tokens"] == 3_598
    assert replay["log"]["ckpt_every"] == 32
    assert replay["log"]["archive_checkpoints"] is True

    sampling = replay["data"]["sampling"]
    assert sampling["mode"] == "general_format_mixed_no_replacement_v2"
    assert sampling["general_source_index"] == 0
    assert sampling["format_source_index"] == 1
    assert sampling["exclude_format_semantic_overlap"] is True
    effective_batch = (
        replay["batch"]["micro_batch_size"]
        * replay["batch"]["grad_accum_steps"]
    )
    assert len(sampling["cycle"]) == effective_batch
    assert sampling["cycle"].count("general") == 12
    for phase in ("format_core", "multi_argument", "parallel", "text"):
        assert sampling["cycle"].count(phase) == 1
    for start in range(0, effective_batch, replay["batch"]["micro_batch_size"]):
        microbatch = sampling["cycle"][
            start : start + replay["batch"]["micro_batch_size"]
        ]
        if "text" in microbatch:
            assert "general" in microbatch

    assert rl["init_from"] == f"{parent['log']['out_dir']}/latest.pt"
    assert rl["init_from"] != f"{replay['log']['out_dir']}/latest.pt"


def test_1m_parent_anchor_pulse_preserves_parent_blocks_and_freezes_token_io() -> None:
    parent = _load(PARENT_PATH)
    failed_bootstrap = _load(FORMAT_BOOTSTRAP_PATH)
    failed_replay = _load(MIXED_REPLAY_PATH)
    replay = _load(PARENT_ANCHOR_PULSE_PATH)
    rl = _load(RL_PATH)

    assert replay["continuation"] == {
        "mode": "fresh_optimizer_sft_child_v1",
        "parent": SEALED_PARENT,
    }
    assert resolve_sft_continuation(replay) == replay["continuation"]
    assert replay["init_from"] == f"{parent['log']['out_dir']}/latest.pt"
    assert replay["init_from"] not in {
        f"{failed_bootstrap['log']['out_dir']}/latest.pt",
        f"{failed_replay['log']['out_dir']}/latest.pt",
    }
    assert _selected_decisions(replay) == PARENT_ANCHOR_PULSE_DECISIONS
    assert replay["schedule"] == {
        "type": "cosine",
        "warmup_steps": 24,
        "total_steps": 372,
    }
    assert replay["optim"] == {
        "name": "adamw",
        "lr": 1.0e-6,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "loss_normalization": "microbatch_mean_v1",
        "freeze_parameters": [
            "loop_embed",
            "embed.weight",
            "in_proj.weight",
            "out_proj.weight",
        ],
    }
    assert replay["batch"] == {
        "micro_batch_size": 2,
        "grad_accum_steps": 8,
        "pad_to_input_tokens": 3_684,
    }
    assert replay["evaluation"]["pad_to_input_tokens"] == 3_598
    assert replay["log"]["ckpt_every"] == 12
    assert replay["log"]["archive_checkpoints"] is True
    assert replay["schedule"]["total_steps"] % replay["log"]["ckpt_every"] == 0

    sampling = replay["data"]["sampling"]
    assert sampling == {
        "mode": "parent_quota_update_blocks_with_format_pulses_v3",
        "general_source_index": 0,
        "format_source_index": 1,
        "parent_prefix_decisions": 5_568,
        "update_decisions": 16,
        "expected_parent_order_sha256": (
            "b1f78dc4b3a08647fa91f1643c40c625804ee63c38728f198dc7fa013aad5c9d"
        ),
        "expected_parent_prefix_sha256": (
            "d7cf1bfcc61cf6b8882cfe149f78f7c2536855536f1190685d77569c5a3adedb"
        ),
        "format_pulses": {
            "count": 24,
            "rows_per_phase": 4,
            "phase_order": [
                "format_core",
                "multi_argument",
                "parallel",
                "text",
            ],
            "within_pulse_order": "phase_round_robin_v1",
            "position_contract": "centered_update_quantiles_v1",
        },
    }
    assert sampling["parent_prefix_decisions"] == _selected_decisions(parent)
    assert sampling["parent_prefix_decisions"] + (
        sampling["format_pulses"]["count"] * sampling["update_decisions"]
    ) == PARENT_ANCHOR_PULSE_DECISIONS
    assert sampling["update_decisions"] == (
        replay["batch"]["micro_batch_size"]
        * replay["batch"]["grad_accum_steps"]
    )

    # Adding another recovery experiment is not promotion authorization.
    assert rl["init_from"] == f"{parent['log']['out_dir']}/latest.pt"
    assert rl["init_from"] != f"{replay['log']['out_dir']}/latest.pt"
