"""Deterministic post-training budget-plan contracts."""

from __future__ import annotations

import copy
import random
from pathlib import Path

import pytest
import yaml

from localagent.data.decision_quota_order import (
    order_assistant_decisions,
    quota_sampling_contract,
)
from localagent.data.pretrain_corpus import CorpusDocument, PackedShardDataset, pack_shards
from localagent.data.prompt_contract import OPENAI_FULL_CATALOG_V1
from localagent.data.schema import Conversation, Message, Role, ToolCall
from localagent.data.stratified_eval_selector import (
    ALGORITHM as STRATIFIED_EVAL_ALGORITHM,
)
from localagent.data.stratified_eval_selector import (
    select_stratified_eval_subset,
)
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import ByteTokenizer, train_bpe
from localagent.train.midtrain import (
    ConversationDataset,
    ConversationTokenCountDataset,
    MixtureSource,
    ScheduledMixture,
    midtrain,
)
from localagent.train.rl import (
    CatalogStringCache,
    _decision_prompt_text,
    _preflight_full_context,
    grpo,
    project_rl_decisions,
)
from localagent.train.sft import quota_sampling_window, sft
from localagent.train.stage_budget import (
    build_stage_budget_plan,
    calibrate_supervised_prefix,
    canonical_plan_bytes,
    seal_stage_budget_plan,
    verify_stage_budget_plan,
    write_stage_budget_plan,
)
from localagent.train.stage_data import single_turn_samples
from localagent.train.stage_sampling import (
    RLPromptSchedule,
    SFTSamplingSchedule,
    decision_keys_to_row_order,
    prepare_sft_data,
)


def _model_config(tmp_path: Path) -> tuple[ModelConfig, Path]:
    config = ModelConfig(
        name="stage-budget-test",
        vocab_size=256,
        d_model=16,
        embed_dim=8,
        n_layers=1,
        n_loops=1,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden=32,
        max_seq_len=128,
        dropout=0.0,
    )
    config.assert_within_budget()
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(config.__dict__, sort_keys=False), encoding="utf-8")
    return config, path


def _bpe_model_config(
    tmp_path: Path,
    *,
    vocab_size: int,
    max_seq_len: int,
    name: str,
) -> tuple[ModelConfig, Path]:
    config = ModelConfig(
        name=name,
        vocab_size=vocab_size,
        d_model=16,
        embed_dim=8,
        n_layers=1,
        n_loops=1,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden=32,
        max_seq_len=max_seq_len,
        dropout=0.0,
    )
    config.assert_within_budget()
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config.__dict__, sort_keys=False), encoding="utf-8")
    return config, path


def _bpe_tokenizer(tmp_path: Path):
    path = tmp_path / "tokenizer.json"
    tokenizer = train_bpe(
        [
            (
                "catalog system user assistant first second train held out "
                "history decision alpha beta gamma source"
            )
        ],
        path,
        vocab_size=300,
        min_frequency=1,
    )
    return tokenizer, path


def _write_conversations(path: Path, conversations: list[Conversation]) -> None:
    path.write_text(
        "".join(f"{conversation.to_json()}\n" for conversation in conversations),
        encoding="utf-8",
    )


def _write_config(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _training_conversations() -> list[Conversation]:
    return [
        Conversation(
            messages=[
                Message(role=Role.user, content="A"),
                Message(role=Role.assistant, content="one"),
            ]
        ),
        Conversation(
            messages=[
                Message(role=Role.user, content="This prompt is deliberately longer"),
                Message(role=Role.assistant, content="a much longer target"),
            ]
        ),
        Conversation(
            messages=[
                Message(role=Role.user, content="Search Seoul"),
                Message(
                    role=Role.assistant,
                    tool_calls=[ToolCall(name="web_search", arguments={"query": "Seoul"})],
                ),
                Message(role=Role.tool, tool_response='{"result":"Seoul"}'),
                Message(
                    role=Role.assistant,
                    tool_calls=[ToolCall(name="web_search", arguments={"query": "Seoul"})],
                ),
            ]
        ),
    ]


def _quota_training_conversations() -> list[Conversation]:
    return [
        Conversation(
            messages=[
                Message(role=Role.user, content=f"Quota prompt {index}"),
                Message(role=Role.assistant, content=f"quota target {index}"),
            ],
            meta={"category": "text", "group": "text", "kind": "text"},
        )
        for index in range(3)
    ]


def test_calibration_chooses_smallest_prefix_and_enforces_boundaries() -> None:
    updates = [{"loss_tokens": value} for value in (4, 6, 8)]

    exact = calibrate_supervised_prefix(
        updates,
        min_supervised_tokens=10,
        max_supervised_tokens=10,
    )
    assert exact["selected_steps"] == 2
    assert exact["previous_prefix_loss_tokens"] == 4
    assert exact["selected_prefix_loss_tokens"] == 10

    zero = calibrate_supervised_prefix(
        updates,
        min_supervised_tokens=0,
        max_supervised_tokens=0,
    )
    assert zero["selected_steps"] == 0
    assert zero["selected_prefix_loss_tokens"] == 0

    with pytest.raises(ValueError, match="exceeds max_supervised_tokens"):
        calibrate_supervised_prefix(
            updates,
            min_supervised_tokens=11,
            max_supervised_tokens=17,
        )
    with pytest.raises(ValueError, match="cannot reach"):
        calibrate_supervised_prefix(
            updates,
            min_supervised_tokens=19,
            max_supervised_tokens=None,
        )


def test_sft_schedule_preserves_lm_kd_head_multiturn_rng_order() -> None:
    tokenizer = ByteTokenizer()
    conversations = _training_conversations()
    samples = single_turn_samples(conversations[:2])
    multi_turn = conversations[2:]
    prepared = prepare_sft_data(
        samples,
        tokenizer,
        conversations=multi_turn,
        sample_sources=["main-a", "main-b"],
        conversation_sources=["main-mt"],
        decay_samples=[samples[1]],
        decay_sample_sources=["decay"],
        lr_schedule="wsd",
        max_seq_len=128,
        joint_tool_head=True,
    )
    schedule = SFTSamplingSchedule(
        prepared,
        batch_size=2,
        shuffle=True,
        seed=29,
        lr_schedule="wsd",
        decay_frac=0.5,
        kd_enabled=True,
        joint_tool_head=True,
        multi_turn_batch_size=1,
    )
    expected_rng = random.Random(29)

    for step, pool_name in ((0, "main"), (1, "decay")):
        pool = prepared.main_entries if pool_name == "main" else prepared.decay_entries
        expected_lm = tuple(expected_rng.randrange(len(pool)) for _ in range(2))
        expected_kd = tuple(expected_rng.randrange(len(prepared.rows)) for _ in range(2))
        expected_head = tuple(expected_rng.randrange(len(prepared.head_items)) for _ in range(2))
        expected_mt = tuple(
            expected_rng.randrange(len(prepared.multi_turn_items))
            for _ in range(min(1, len(prepared.multi_turn_items)))
        )
        selection = schedule.next_microbatch(step=step, total_steps=2)
        assert selection.pool == pool_name
        assert selection.lm_indices == expected_lm
        assert selection.kd_indices == expected_kd
        assert selection.head_indices == expected_head
        assert selection.multi_turn_indices == expected_mt


def test_sft_schedule_defaults_to_twelve_multiturn_items_and_zero_consumes_no_draws() -> None:
    tokenizer = ByteTokenizer()
    conversations = _training_conversations()
    prepared = prepare_sft_data(
        single_turn_samples(conversations[:2]),
        tokenizer,
        conversations=conversations[2:],
        sample_sources=["main-a", "main-b"],
        conversation_sources=["main-mt"],
        decay_samples=None,
        decay_sample_sources=None,
        lr_schedule="cosine",
        max_seq_len=128,
        joint_tool_head=True,
    )
    default = SFTSamplingSchedule(
        prepared,
        batch_size=1,
        shuffle=True,
        seed=31,
        lr_schedule="cosine",
        decay_frac=0.2,
        kd_enabled=False,
        joint_tool_head=True,
    )
    assert default.multi_turn_batch_size == 12
    assert len(default.next_microbatch(step=0, total_steps=1).multi_turn_indices) == min(
        12,
        len(prepared.multi_turn_items),
    )

    zero = SFTSamplingSchedule(
        prepared,
        batch_size=1,
        shuffle=True,
        seed=31,
        lr_schedule="cosine",
        decay_frac=0.2,
        kd_enabled=False,
        joint_tool_head=True,
        multi_turn_batch_size=0,
    )
    selection = zero.next_microbatch(step=0, total_steps=1)
    assert selection.multi_turn_indices == ()

    expected_rng = random.Random(31)
    expected_rng.randrange(len(prepared.main_entries))
    expected_rng.randrange(len(prepared.head_items))
    assert zero.rng.getstate() == expected_rng.getstate()

    with pytest.raises(ValueError, match="non-negative integer"):
        SFTSamplingSchedule(
            prepared,
            batch_size=1,
            shuffle=True,
            seed=31,
            lr_schedule="cosine",
            decay_frac=0.2,
            kd_enabled=False,
            joint_tool_head=True,
            multi_turn_batch_size=-1,
        )
    with pytest.raises(ValueError, match="non-negative integer"):
        SFTSamplingSchedule(
            prepared,
            batch_size=1,
            shuffle=True,
            seed=31,
            lr_schedule="cosine",
            decay_frac=0.2,
            kd_enabled=False,
            joint_tool_head=True,
            multi_turn_batch_size=True,
        )


def test_quota_schedules_consume_exact_permutations_without_replacement() -> None:
    tokenizer = ByteTokenizer()
    conversations = _training_conversations()[:2]
    prepared = prepare_sft_data(
        single_turn_samples(conversations),
        tokenizer,
        conversations=None,
        sample_sources=["first", "second"],
        conversation_sources=None,
        decay_samples=None,
        decay_sample_sources=None,
        lr_schedule="cosine",
        max_seq_len=128,
        joint_tool_head=False,
    )
    schedule = SFTSamplingSchedule(
        prepared,
        batch_size=1,
        shuffle=False,
        seed=7,
        lr_schedule="cosine",
        decay_frac=0.2,
        kd_enabled=False,
        joint_tool_head=False,
        lm_order=(1, 0),
    )

    assert schedule.next_microbatch(step=0, total_steps=3).lm_indices == (1,)
    assert schedule.next_microbatch(step=1, total_steps=3).lm_indices == (0,)
    with pytest.raises(ValueError, match="horizon exceeds"):
        schedule.next_microbatch(step=2, total_steps=3)
    with pytest.raises(ValueError, match="shuffle=false"):
        SFTSamplingSchedule(
            prepared,
            batch_size=1,
            shuffle=True,
            seed=7,
            lr_schedule="cosine",
            decay_frac=0.2,
            kd_enabled=False,
            joint_tool_head=False,
            lm_order=(1, 0),
        )

    rl_schedule = RLPromptSchedule(
        3,
        2,
        seed=7,
        prompt_order=(2, 0, 1),
    )
    assert rl_schedule.indices_for_step(0) == (2, 0)
    with pytest.raises(ValueError, match="horizon exceeds"):
        rl_schedule.indices_for_step(1)


def test_sft_quota_window_is_offset_bound_and_default_contract_is_unchanged() -> None:
    ordering = order_assistant_decisions(_quota_training_conversations())

    default_keys, default_contract = quota_sampling_window(
        ordering,
        selected_decisions=ordering.audit.frontload_decision_count,
    )
    assert default_keys == ordering.keys
    assert default_contract == quota_sampling_contract(
        ordering,
        selected_decisions=ordering.audit.frontload_decision_count,
    )

    window_keys, window_contract = quota_sampling_window(
        ordering,
        start_decision=1,
        selected_decisions=2,
    )
    assert window_keys == ordering.keys[1:] + ordering.keys[:1]
    assert window_contract["start_decision"] == 1
    assert window_contract["selected_window"]["decisions"] == 2
    assert window_contract["selected_window"]["end_decision_exclusive"] == 3
    assert sum(window_contract["selected_window"]["stratum_counts"].values()) == 2
    assert window_contract["require_all_observed_strata"] is False

    with pytest.raises(TypeError, match="start_decision must be an integer"):
        quota_sampling_window(ordering, start_decision=True, selected_decisions=1)
    with pytest.raises(ValueError, match="start_decision must be non-negative"):
        quota_sampling_window(ordering, start_decision=-1, selected_decisions=1)
    with pytest.raises(ValueError, match="decision window exceeds"):
        quota_sampling_window(
            ordering,
            start_decision=len(ordering.keys) - 1,
            selected_decisions=2,
        )


def test_decision_key_order_maps_to_rendered_assistant_turn_order() -> None:
    conversations = _training_conversations()[:2]
    natural_keys = ((0, 1), (1, 1))
    reversed_keys = tuple(reversed(natural_keys))

    assert decision_keys_to_row_order(
        conversations,
        reversed_keys,
        expected_rows=2,
    ) == (1, 0)
    with pytest.raises(ValueError, match="every rendered"):
        decision_keys_to_row_order(
            conversations,
            reversed_keys[:-1],
            expected_rows=2,
        )


def test_sft_runner_and_planner_share_masking_and_interleaved_schedule(tmp_path) -> None:
    model_config, model_path = _model_config(tmp_path)
    conversation_path = tmp_path / "train.jsonl"
    conversations = _training_conversations()
    _write_conversations(conversation_path, conversations)
    config_path = _write_config(
        tmp_path / "sft.yaml",
        {
            "stage": "sft",
            "model_config": str(model_path),
            "init_from": str(tmp_path / "unused-parent.pt"),
            "data": {
                "conversations": [str(conversation_path)],
                "tokenizer": {"kind": "byte"},
                "seq_len": model_config.max_seq_len,
                "shuffle": True,
            },
            "heads": {"joint_tool_pointer": True, "multi_turn_batch_size": 1},
            "schedule": {"type": "wsd", "total_steps": 3, "decay_frac": 0.5},
            "batch": {"micro_batch_size": 1, "grad_accum_steps": 2},
            "runtime": {"seed": 37},
        },
    )
    plan = build_stage_budget_plan(config_path)

    samples = []
    multi_turn = []
    for conversation in conversations:
        projected = single_turn_samples([conversation])
        (samples if projected else multi_turn).extend(projected or [conversation])
    source = str(conversation_path)
    model = LocalAgentLM(model_config)
    observed_forward_slots = []
    real_forward = model.forward

    def record_forward(input_ids, *args, **kwargs):
        observed_forward_slots.append(int(input_ids.numel()))
        return real_forward(input_ids, *args, **kwargs)

    model.forward = record_forward
    _, _, _, metrics = sft(
        model,
        samples,
        ByteTokenizer(),
        steps=3,
        batch_size=1,
        accum_steps=2,
        joint_tool_head=True,
        multi_turn_batch_size=1,
        conversations=multi_turn,
        sample_sources=[source] * len(samples),
        conversation_sources=[source] * len(multi_turn),
        lr_schedule="wsd",
        decay_frac=0.5,
        shuffle=True,
        max_seq_len=model_config.max_seq_len,
        seed=37,
        return_metrics=True,
        log=lambda *_: None,
    )

    runner = metrics["token_accounting"]
    planned = plan["planned"]["horizon_totals"]
    assert planned["input_tokens"] == runner["input_tokens"]
    assert planned["loss_tokens"] == runner["loss_tokens"]
    assert planned["sources"][source]["rows"] == runner["sources"][source]["rows"]
    assert planned["sources"][source]["input_tokens"] == runner["sources"][source]["input_tokens"]
    assert planned["sources"][source]["loss_tokens"] == runner["sources"][source]["loss_tokens"]
    assert plan["data"]["dataset_token_accounting"] == metrics["dataset_token_accounting"]
    assert plan["schedule"]["multi_turn_batch_size"] == 1

    calls_per_update = 2 * 3
    for step, update in enumerate(plan["planned"]["updates"]):
        calls = observed_forward_slots[step * calls_per_update : (step + 1) * calls_per_update]
        slots = update["model_forward_token_slots"]
        assert slots == {
            "padded_lm": calls[0] + calls[3],
            "distillation": 0,
            "short_joint_head": calls[1] + calls[4],
            "multi_turn_head": calls[2] + calls[5],
            "total": sum(calls),
        }
    horizon_slots = plan["planned"]["horizon_totals"]["model_forward_token_slots"]
    assert horizon_slots["total"] == sum(observed_forward_slots)
    for key in (
        "padded_lm",
        "distillation",
        "short_joint_head",
        "multi_turn_head",
    ):
        assert horizon_slots[key] == sum(
            update["model_forward_token_slots"][key] for update in plan["planned"]["updates"]
        )


def test_sft_fixed_input_width_changes_only_lm_forward_slots(tmp_path) -> None:
    model_config, model_path = _model_config(tmp_path)
    conversation_path = tmp_path / "train.jsonl"
    _write_conversations(conversation_path, _training_conversations()[:2])
    base_config = {
        "stage": "sft",
        "model_config": str(model_path),
        "init_from": str(tmp_path / "unused-parent.pt"),
        "data": {
            "conversations": [str(conversation_path)],
            "tokenizer": {"kind": "byte"},
            "seq_len": model_config.max_seq_len,
            "shuffle": False,
        },
        "heads": {
            "joint_tool_pointer": False,
            "train_route_head": False,
            "train_dense_selector": False,
        },
        "schedule": {"type": "cosine", "total_steps": 2},
        "batch": {"micro_batch_size": 2, "grad_accum_steps": 2},
        "runtime": {"seed": 47},
    }
    default_path = _write_config(tmp_path / "sft-default-width.yaml", base_config)
    fixed_config = copy.deepcopy(base_config)
    fixed_config["batch"]["pad_to_input_tokens"] = 127
    fixed_config["evaluation"] = {"pad_to_input_tokens": 126}
    fixed_path = _write_config(tmp_path / "sft-fixed-width.yaml", fixed_config)

    default_plan = build_stage_budget_plan(default_path)
    fixed_plan = build_stage_budget_plan(fixed_path)

    default_totals = default_plan["planned"]["horizon_totals"]
    fixed_totals = fixed_plan["planned"]["horizon_totals"]
    assert fixed_totals["input_tokens"] == default_totals["input_tokens"]
    assert fixed_totals["loss_tokens"] == default_totals["loss_tokens"]
    assert fixed_totals["sources"] == default_totals["sources"]
    assert fixed_totals["model_forward_token_slots"]["padded_lm"] == 2 * 2 * 2 * 127
    assert fixed_totals["model_forward_token_slots"]["distillation"] == 0
    assert fixed_totals["model_forward_token_slots"]["short_joint_head"] == 0
    assert fixed_totals["model_forward_token_slots"]["multi_turn_head"] == 0
    assert fixed_totals["model_forward_token_slots"]["total"] == 2 * 2 * 2 * 127
    assert fixed_plan["schedule"]["pad_to_input_tokens"] == 127
    assert fixed_plan["schedule"]["evaluation_pad_to_input_tokens"] == 126
    assert "pad_to_input_tokens" not in default_plan["schedule"]
    assert "evaluation_pad_to_input_tokens" not in default_plan["schedule"]


@pytest.mark.parametrize(
    ("optim", "match"),
    [
        ({"name": "sgd"}, "optim.name must be exactly 'adamw'"),
        ({"weight_decay": -0.1}, "optim.weight_decay"),
        ({"weight_decay": True}, "optim.weight_decay"),
        ({"grad_clip": 0.0}, "optim.grad_clip"),
        ({"grad_clip": "1.0"}, "optim.grad_clip"),
    ],
)
def test_sft_plan_rejects_unimplemented_optimizer_drift(
    tmp_path: Path,
    optim: dict,
    match: str,
) -> None:
    model_config, model_path = _model_config(tmp_path)
    conversation_path = tmp_path / "train.jsonl"
    _write_conversations(conversation_path, _training_conversations()[:1])
    config_path = _write_config(
        tmp_path / "sft-invalid-optimizer.yaml",
        {
            "stage": "sft",
            "model_config": str(model_path),
            "init_from": str(tmp_path / "unused-parent.pt"),
            "data": {
                "conversations": [str(conversation_path)],
                "tokenizer": {"kind": "byte"},
                "seq_len": model_config.max_seq_len,
            },
            "optim": optim,
            "heads": {
                "joint_tool_pointer": False,
                "train_route_head": False,
                "train_dense_selector": False,
            },
            "schedule": {"type": "cosine", "total_steps": 1},
            "batch": {"micro_batch_size": 1, "grad_accum_steps": 1},
            "runtime": {"seed": 53},
        },
    )

    with pytest.raises((TypeError, ValueError), match=match):
        build_stage_budget_plan(config_path)


@pytest.mark.parametrize(
    ("section", "value", "match"),
    [
        ("batch", 0, "batch.pad_to_input_tokens must be a positive integer"),
        ("batch", True, "batch.pad_to_input_tokens must be a positive integer"),
        ("batch", 129, "cannot exceed the SFT sequence limit"),
        ("evaluation", 0, "evaluation.pad_to_input_tokens must be a positive integer"),
        ("evaluation", 129, "cannot exceed the SFT sequence limit"),
    ],
)
def test_sft_plan_rejects_invalid_fixed_input_widths(
    tmp_path,
    section,
    value,
    match,
) -> None:
    model_config, model_path = _model_config(tmp_path)
    conversation_path = tmp_path / "train.jsonl"
    _write_conversations(conversation_path, _training_conversations()[:1])
    config = {
        "stage": "sft",
        "model_config": str(model_path),
        "init_from": str(tmp_path / "unused-parent.pt"),
        "data": {
            "conversations": [str(conversation_path)],
            "tokenizer": {"kind": "byte"},
            "seq_len": model_config.max_seq_len,
        },
        "heads": {
            "joint_tool_pointer": False,
            "train_route_head": False,
            "train_dense_selector": False,
        },
        "schedule": {"type": "cosine", "total_steps": 1},
        "batch": {"micro_batch_size": 1, "grad_accum_steps": 1},
        "runtime": {"seed": 53},
    }
    config.setdefault(section, {})["pad_to_input_tokens"] = value
    config_path = _write_config(tmp_path / f"sft-invalid-{section}.yaml", config)

    with pytest.raises(ValueError, match=match):
        build_stage_budget_plan(config_path)


def test_sft_plan_rejects_a_row_wider_than_the_fixed_input_width(tmp_path) -> None:
    model_config, model_path = _model_config(tmp_path)
    conversation_path = tmp_path / "train.jsonl"
    _write_conversations(conversation_path, _training_conversations()[:1])
    config_path = _write_config(
        tmp_path / "sft-undersized-width.yaml",
        {
            "stage": "sft",
            "model_config": str(model_path),
            "init_from": str(tmp_path / "unused-parent.pt"),
            "data": {
                "conversations": [str(conversation_path)],
                "tokenizer": {"kind": "byte"},
                "seq_len": model_config.max_seq_len,
                "shuffle": False,
            },
            "heads": {
                "joint_tool_pointer": False,
                "train_route_head": False,
                "train_dense_selector": False,
            },
            "schedule": {"type": "cosine", "total_steps": 1},
            "batch": {
                "micro_batch_size": 1,
                "grad_accum_steps": 1,
                "pad_to_input_tokens": 1,
            },
            "runtime": {"seed": 59},
        },
    )

    with pytest.raises(
        ValueError,
        match="row requires more input tokens than batch.pad_to_input_tokens",
    ):
        build_stage_budget_plan(config_path)


def test_sft_offset_window_planner_and_runner_select_the_same_rows(tmp_path: Path) -> None:
    tokenizer, tokenizer_path = _bpe_tokenizer(tmp_path)
    model_config, model_path = _bpe_model_config(
        tmp_path,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=512,
        name="sft-offset-window",
    )
    conversations = _quota_training_conversations()
    conversation_path = tmp_path / "train.jsonl"
    _write_conversations(conversation_path, conversations)
    start_decision = 1
    steps = 2
    config_path = _write_config(
        tmp_path / "sft-offset-window-train.yaml",
        {
            "stage": "sft",
            "model_config": str(model_path),
            "init_from": str(tmp_path / "unused-parent.pt"),
            "continuation": {"mode": "fresh_optimizer_sft_child_v1"},
            "data": {
                "conversation_prompt_contract": OPENAI_FULL_CATALOG_V1,
                "conversations": [str(conversation_path)],
                "tokenizer": {"kind": "bpe", "path": str(tokenizer_path)},
                "seq_len": model_config.max_seq_len,
                "shuffle": False,
                "sampling": {
                    "mode": "quota_stratified_no_replacement_v1",
                    "start_decision": start_decision,
                },
            },
            "heads": {
                "joint_tool_pointer": False,
                "train_route_head": False,
                "train_dense_selector": False,
            },
            "schedule": {"type": "cosine", "total_steps": steps},
            "batch": {"micro_batch_size": 1, "grad_accum_steps": 1},
            "runtime": {"seed": 41},
        },
    )
    plan = build_stage_budget_plan(config_path)
    ordering = order_assistant_decisions(conversations)
    ordered_keys, sampling_contract = quota_sampling_window(
        ordering,
        start_decision=start_decision,
        selected_decisions=steps,
    )

    _, _, _, metrics = sft(
        LocalAgentLM(model_config),
        [],
        tokenizer,
        steps=steps,
        batch_size=1,
        conversations=conversations,
        conversation_sources=[str(conversation_path)] * len(conversations),
        conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
        shuffle=False,
        max_seq_len=model_config.max_seq_len,
        seed=41,
        lm_order_keys=ordered_keys,
        sampling_contract=sampling_contract,
        return_metrics=True,
        log=lambda *_: None,
    )

    assert plan["schedule"]["lm_sampling"] == sampling_contract
    assert plan["schedule"]["continuation"] == {
        "mode": "fresh_optimizer_sft_child_v1"
    }
    assert plan["data"]["decision_sampling"] == sampling_contract
    assert metrics["lm_sampling"] == sampling_contract
    planned = plan["planned"]["horizon_totals"]
    assert planned["input_tokens"] == metrics["token_accounting"]["input_tokens"]
    assert planned["loss_tokens"] == metrics["token_accounting"]["loss_tokens"]
    assert planned["sources"][str(conversation_path)]["rows"] == steps


def test_sft_zero_multiturn_batch_skips_forward_and_reports_heldout_semantic_totals(
    tmp_path,
) -> None:
    model_config, model_path = _model_config(tmp_path)
    conversations = _training_conversations()
    heldout = Conversation(
        messages=[
            Message(role=Role.user, content="A distinct held-out request"),
            Message(role=Role.assistant, content="a held-out answer"),
        ]
    )
    conversation_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_conversations(conversation_path, conversations)
    _write_conversations(eval_path, [heldout])
    config_path = _write_config(
        tmp_path / "sft-zero-multiturn.yaml",
        {
            "stage": "sft",
            "model_config": str(model_path),
            "init_from": str(tmp_path / "unused-parent.pt"),
            "data": {
                "conversations": [str(conversation_path)],
                "eval_conversations": [str(eval_path)],
                "tokenizer": {"kind": "byte"},
                "seq_len": model_config.max_seq_len,
                "shuffle": True,
            },
            "heads": {"joint_tool_pointer": True, "multi_turn_batch_size": 0},
            "schedule": {"type": "cosine", "total_steps": 1},
            "batch": {"micro_batch_size": 1, "grad_accum_steps": 1},
            "runtime": {"seed": 43},
        },
    )
    plan = build_stage_budget_plan(config_path)

    samples = []
    multi_turn = []
    for conversation in conversations:
        projected = single_turn_samples([conversation])
        (samples if projected else multi_turn).extend(projected or [conversation])
    model = LocalAgentLM(model_config)
    observed_forward_slots = []
    real_forward = model.forward

    def record_forward(input_ids, *args, **kwargs):
        observed_forward_slots.append(int(input_ids.numel()))
        return real_forward(input_ids, *args, **kwargs)

    model.forward = record_forward
    sft(
        model,
        samples,
        ByteTokenizer(),
        steps=1,
        batch_size=1,
        joint_tool_head=True,
        multi_turn_batch_size=0,
        conversations=multi_turn,
        max_seq_len=model_config.max_seq_len,
        seed=43,
        log=lambda *_: None,
    )

    slots = plan["planned"]["updates"][0]["model_forward_token_slots"]
    assert len(observed_forward_slots) == 2
    assert slots["padded_lm"] == observed_forward_slots[0]
    assert slots["short_joint_head"] == observed_forward_slots[1]
    assert slots["multi_turn_head"] == 0
    assert slots["total"] == sum(observed_forward_slots)

    heldout_counts = ConversationTokenCountDataset(
        [heldout],
        ByteTokenizer(),
        model_config.max_seq_len,
    )._row_token_counts
    assert plan["data"]["heldout_eval_token_accounting"] == {
        "accounting_kind": "exact_shifted_masked_language_model_tokens",
        "rows": len(heldout_counts),
        "input_tokens": sum(value[0] for value in heldout_counts),
        "loss_tokens": sum(value[1] for value in heldout_counts),
    }


def test_midtrain_runner_and_planner_share_source_and_row_sampling(tmp_path) -> None:
    model_config, model_path = _model_config(tmp_path)
    tokenizer = ByteTokenizer()
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    first_rows = [
        Conversation(
            messages=[
                Message(role=Role.user, content="short"),
                Message(role=Role.assistant, content="x"),
            ]
        )
    ]
    second_rows = [
        Conversation(
            messages=[
                Message(role=Role.user, content="a longer prompt for accounting"),
                Message(role=Role.assistant, content="a longer answer"),
            ]
        )
    ]
    _write_conversations(first_path, first_rows)
    _write_conversations(second_path, second_rows)
    source_config = [
        {
            "name": "first",
            "type": "conversations",
            "path": str(first_path),
            "weight": 0.7,
            "end_weight": 0.3,
        },
        {
            "name": "second",
            "type": "conversations",
            "path": str(second_path),
            "weight": 0.3,
            "end_weight": 0.7,
        },
    ]
    config_path = _write_config(
        tmp_path / "midtrain.yaml",
        {
            "stage": "midtrain",
            "model_config": str(model_path),
            "init_from": str(tmp_path / "unused-parent.pt"),
            "data": {
                "mixture": {"unit": "loss_tokens"},
                "tokenizer": {"kind": "byte"},
                "sources": source_config,
            },
            "schedule": {"total_steps": 3},
            "batch": {"micro_batch_size": 1, "grad_accum_steps": 2},
            "runtime": {"seed": 11},
        },
    )
    plan = build_stage_budget_plan(config_path)
    mixture = ScheduledMixture(
        [
            MixtureSource(
                "first",
                ConversationDataset(first_rows, tokenizer, model_config.max_seq_len),
                0.7,
                0.3,
            ),
            MixtureSource(
                "second",
                ConversationDataset(second_rows, tokenizer, model_config.max_seq_len),
                0.3,
                0.7,
            ),
        ],
        unit="loss_tokens",
    )
    _, metrics = midtrain(
        LocalAgentLM(model_config),
        mixture,
        steps=3,
        batch_size=1,
        accum_steps=2,
        seed=11,
        return_metrics=True,
        log=lambda *_: None,
    )

    runner = metrics["token_accounting"]
    planned = plan["planned"]["horizon_totals"]
    assert planned["input_tokens"] == runner["input_tokens"]
    assert planned["loss_tokens"] == runner["loss_tokens"]
    assert metrics["source_draws"] == {
        source: values["draws"] for source, values in planned["sources"].items()
    }
    for source in ("first", "second"):
        assert (
            planned["sources"][source]["input_tokens"] == runner["sources"][source]["input_tokens"]
        )
        assert planned["sources"][source]["loss_tokens"] == runner["sources"][source]["loss_tokens"]


def test_midtrain_planner_counts_packed_and_conversation_rows_without_batches(
    tmp_path,
    monkeypatch,
) -> None:
    model_config, model_path = _model_config(tmp_path)
    shards = tmp_path / "shards"
    pack_shards(
        [
            CorpusDocument("short planner row", doc_id="short"),
            CorpusDocument("longer planner row " * 12, doc_id="long"),
        ],
        ByteTokenizer(),
        seq_len=32,
        shards_dir=str(shards),
        rows_per_shard=3,
        val_fraction=0.0,
        seed=13,
    )
    conversation_path = tmp_path / "train.jsonl"
    _write_conversations(conversation_path, _training_conversations())
    config_path = _write_config(
        tmp_path / "midtrain-count-only.yaml",
        {
            "stage": "midtrain",
            "model_config": str(model_path),
            "init_from": str(tmp_path / "unused-parent.pt"),
            "data": {
                "mixture": {"unit": "loss_tokens"},
                "tokenizer": {"kind": "byte"},
                "sources": [
                    {
                        "name": "packed",
                        "type": "shards",
                        "path": str(shards),
                        "split": "train",
                        "weight": 0.5,
                    },
                    {
                        "name": "conversation",
                        "type": "conversations",
                        "path": str(conversation_path),
                        "weight": 0.5,
                    },
                ],
            },
            "schedule": {"total_steps": 4},
            "batch": {"micro_batch_size": 2, "grad_accum_steps": 2},
            "runtime": {"seed": 17},
        },
    )

    def reject_materialized_batch(*_args, **_kwargs):
        raise AssertionError("budget planner materialized a training batch")

    monkeypatch.setattr(PackedShardDataset, "sample_batch", reject_materialized_batch)
    monkeypatch.setattr(
        ConversationDataset,
        "__init__",
        reject_materialized_batch,
    )
    assert not hasattr(ConversationTokenCountDataset, "sample_batch")
    plan = build_stage_budget_plan(config_path)

    assert plan["planned"]["horizon_totals"]["updates"] == 4
    assert plan["planned"]["horizon_totals"]["input_tokens"] > 0
    assert plan["planned"]["horizon_totals"]["loss_tokens"] > 0
    assert (
        sum(source["draws"] for source in plan["planned"]["horizon_totals"]["sources"].values())
        == 8
    )
    assert model_config.max_seq_len >= 32


def test_rl_plan_matches_runner_prompt_coverage_without_fake_loss_tokens(
    tmp_path,
    monkeypatch,
) -> None:
    model_config, model_path = _model_config(tmp_path)
    train_path = tmp_path / "rl-train.jsonl"
    eval_path = tmp_path / "rl-eval.jsonl"
    train_rows = [
        Conversation(
            messages=[
                Message(role=Role.user, content="Say a"),
                Message(role=Role.assistant, content="a"),
            ]
        ),
        Conversation(
            messages=[
                Message(role=Role.user, content="Say b"),
                Message(role=Role.assistant, content="b"),
            ]
        ),
    ]
    eval_rows = [
        Conversation(
            messages=[
                Message(role=Role.user, content="Say c"),
                Message(role=Role.assistant, content="c"),
            ]
        )
    ]
    _write_conversations(train_path, train_rows)
    _write_conversations(eval_path, eval_rows)
    config_path = _write_config(
        tmp_path / "rl.yaml",
        {
            "stage": "rl",
            "model_config": str(model_path),
            "init_from": str(tmp_path / "unused-parent.pt"),
            "data": {
                "conversations": [str(train_path)],
                "eval_conversations": [str(eval_path)],
                "tokenizer": {"kind": "byte"},
            },
            "environment": {"name": "canonical_toolcalls", "learned_judge": False},
            "rollout": {
                "prompts_per_step": 2,
                "group_size": 2,
                "max_new_tokens": 2,
                "temperature": 1.0,
            },
            "schedule": {"total_steps": 3, "warmup_steps": 7},
            "runtime": {"seed": 19},
        },
    )
    plan = build_stage_budget_plan(config_path)
    samples = single_turn_samples(train_rows)
    observed_indices = []
    real_indices_for_step = RLPromptSchedule.indices_for_step

    def record_indices(self, step):
        indices = real_indices_for_step(self, step)
        observed_indices.extend(indices)
        return indices

    monkeypatch.setattr(RLPromptSchedule, "indices_for_step", record_indices)
    _, metrics = grpo(
        LocalAgentLM(model_config),
        samples,
        ByteTokenizer(),
        steps=3,
        prompts_per_step=2,
        group_size=2,
        max_new=2,
        seed=19,
        return_metrics=True,
        log=lambda *_: None,
    )

    horizon = plan["planned"]["horizon_totals"]
    observed_counts = [observed_indices.count(index) for index in range(len(samples))]
    assert horizon["prompt_draw_counts"] == observed_counts
    assert horizon["prompt_groups"] == metrics["attempted_groups"]
    assert horizon["rollouts"] == metrics["attempted_rollouts"]
    assert horizon["min_action_token_opportunities"] == 12
    assert horizon["max_action_token_opportunities"] == 24
    prompt_tokens = horizon["prompt_input_tokens"]
    rollout_lower = 2 * prompt_tokens
    rollout_upper = rollout_lower + 24
    scoring_upper = 2 * (prompt_tokens + 6)
    bounds = horizon["model_forward_token_slot_bounds"]
    assert bounds["phases"] == {
        "rollout_prefill": {"lower": rollout_lower, "upper": rollout_lower},
        "rollout_cached_decode": {"lower": 0, "upper": rollout_upper - rollout_lower},
        "old_policy_scoring": {"lower": 0, "upper": scoring_upper},
        "reference_policy_scoring": {"lower": 0, "upper": scoring_upper},
        "current_policy_optimization": {"lower": 0, "upper": scoring_upper},
    }
    assert bounds["total"] == {
        "lower": rollout_lower,
        "upper": rollout_upper + 3 * scoring_upper,
    }
    assert plan["schedule"]["warmup_steps"] == 7
    assert plan["planned"]["generation_dependent_loss_tokens"] is None
    assert "loss_tokens" not in horizon


def test_rl_plan_binds_the_same_bounded_eval_selection_and_selected_split(
    tmp_path: Path,
) -> None:
    model_config, model_path = _model_config(tmp_path)
    train_path = tmp_path / "rl-train.jsonl"
    eval_path = tmp_path / "rl-eval.jsonl"
    train_rows = [
        Conversation(
            messages=[
                Message(role=Role.user, content="Training request"),
                Message(role=Role.assistant, content="training answer"),
            ]
        )
    ]
    eval_rows = [
        Conversation(
            messages=[
                Message(role=Role.user, content=f"Held-out request {index}"),
                Message(role=Role.assistant, content=f"held-out answer {index}"),
            ],
            meta={"category": "text"},
        )
        for index in range(3)
    ]
    _write_conversations(train_path, train_rows)
    _write_conversations(eval_path, eval_rows)
    config_path = _write_config(
        tmp_path / "rl-bounded-eval.yaml",
        {
            "stage": "rl",
            "model_config": str(model_path),
            "init_from": str(tmp_path / "unused-parent.pt"),
            "data": {
                "conversations": [str(train_path)],
                "eval_conversations": [str(eval_path)],
                "tokenizer": {"kind": "byte"},
            },
            "evaluation": {
                "max_conversations": 2,
                "selection": STRATIFIED_EVAL_ALGORITHM,
            },
            "environment": {"name": "canonical_toolcalls", "learned_judge": False},
            "rollout": {
                "prompts_per_step": 1,
                "group_size": 2,
                "max_new_tokens": 32,
                "temperature": 1.0,
            },
            "schedule": {"total_steps": 1},
            "runtime": {"seed": 19},
        },
    )

    plan = build_stage_budget_plan(config_path)
    expected = select_stratified_eval_subset(eval_rows, max_rows=2)

    assert plan["data"]["eval_source_conversation_rows"] == 3
    assert plan["data"]["eval_selected_conversation_rows"] == 2
    assert plan["data"]["eval_selection"] == expected.audit.as_dict()
    assert plan["data"]["eval_single_turn_rows"] == 2
    assert plan["data"]["split_audit"]["eval_scored_rows"] == 3
    assert plan["data"]["selected_eval_split_audit"]["eval_scored_rows"] == 2
    assert (
        plan["data"]["split_audit"]["eval_scored_rows_sha256"]
        != plan["data"]["selected_eval_split_audit"]["eval_scored_rows_sha256"]
    )
    assert model_config.max_seq_len == 128


def test_full_rl_plan_projects_all_decisions_and_matches_runner_context_and_sources(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tokenizer, tokenizer_path = _bpe_tokenizer(tmp_path)
    model_config, model_path = _bpe_model_config(
        tmp_path,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=4096,
        name="full-rl-budget",
    )
    first_path = tmp_path / "train-first.jsonl"
    second_path = tmp_path / "train-second.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    first_rows = [
        Conversation(
            messages=[
                Message(role=Role.system, content="Preserve exact history."),
                Message(role=Role.user, content="First decision."),
                Message(role=Role.assistant, content="alpha"),
                Message(role=Role.user, content="Second decision."),
                Message(role=Role.assistant, content="beta"),
            ]
        )
    ]
    second_rows = [
        Conversation(
            messages=[
                Message(role=Role.user, content="Third decision."),
                Message(role=Role.assistant, content="gamma"),
            ]
        )
    ]
    eval_rows = [
        Conversation(
            messages=[
                Message(role=Role.user, content="Held-out decision."),
                Message(role=Role.assistant, content="held"),
            ]
        )
    ]
    _write_conversations(first_path, first_rows)
    _write_conversations(second_path, second_rows)
    _write_conversations(eval_path, eval_rows)
    max_new = 8
    seed = 23
    steps = 3
    prompts_per_step = 2
    config_path = _write_config(
        tmp_path / "full-rl.yaml",
        {
            "stage": "rl",
            "model_config": str(model_path),
            "init_from": str(tmp_path / "unused-parent.pt"),
            "data": {
                "conversation_prompt_contract": OPENAI_FULL_CATALOG_V1,
                "conversations": [str(first_path), str(second_path)],
                "eval_conversations": [str(eval_path)],
                "tokenizer": {"kind": "bpe", "path": str(tokenizer_path)},
            },
            "environment": {"name": "canonical_toolcalls", "learned_judge": False},
            "rollout": {
                "prompts_per_step": prompts_per_step,
                "group_size": 2,
                "max_new_tokens": max_new,
                "temperature": 1.0,
            },
            "schedule": {"total_steps": steps},
            "runtime": {"seed": seed},
        },
    )

    plan = build_stage_budget_plan(config_path)
    train_conversations = [*first_rows, *second_rows]
    train_sources = [
        *([str(first_path)] * len(first_rows)),
        *([str(second_path)] * len(second_rows)),
    ]
    decisions = project_rl_decisions(train_conversations, sources=train_sources)
    eval_decisions = project_rl_decisions(eval_rows, sources=[str(eval_path)])
    cache = CatalogStringCache()
    prompt_lengths = [
        len(tokenizer.encode(_decision_prompt_text(decision, cache))) for decision in decisions
    ]
    eval_prompt_lengths = [
        len(tokenizer.encode(_decision_prompt_text(decision, cache))) for decision in eval_decisions
    ]
    runner_context = {
        "train": _preflight_full_context(
            decisions,
            tokenizer,
            max_new=max_new,
            max_seq_len=model_config.max_seq_len,
            split="training split",
            catalog_cache=cache,
            prompt_lengths=prompt_lengths,
        ),
        "eval": _preflight_full_context(
            eval_decisions,
            tokenizer,
            max_new=max_new,
            max_seq_len=model_config.max_seq_len,
            split="held-out split",
            catalog_cache=cache,
            prompt_lengths=eval_prompt_lengths,
        ),
    }

    assert plan["data"]["conversation_prompt_contract"] == OPENAI_FULL_CATALOG_V1
    assert plan["schedule"]["conversation_prompt_contract"] == OPENAI_FULL_CATALOG_V1
    assert plan["data"]["assistant_decision_rows"] == len(decisions) == 3
    assert plan["data"]["eval_assistant_decision_rows"] == len(eval_decisions) == 1
    assert plan["data"]["single_turn_rows"] == 1
    assert plan["data"]["prompt_token_lengths"] == {
        "train": prompt_lengths,
        "eval": eval_prompt_lengths,
    }
    assert plan["data"]["context_preflight"] == runner_context
    assert plan["data"]["retains_complete_prompts"] is False
    assert plan["data"]["retains_prompt_token_ids"] is False

    expected_draws = {str(first_path): 0, str(second_path): 0}
    schedule = RLPromptSchedule(len(decisions), prompts_per_step, seed=seed)
    for step in range(steps):
        for index in schedule.indices_for_step(step):
            expected_draws[decisions[index].source] += 1
    actual_draws = {
        source: sum(
            update["sources"][source]["prompt_groups"] for update in plan["planned"]["updates"]
        )
        for source in expected_draws
    }
    assert actual_draws == expected_draws

    observed_prompt_lengths = []

    def record_rollout(_model, _tok, prompt_ids, *_args, **_kwargs):
        observed_prompt_lengths.append(len(prompt_ids))
        return [_tok.eos_id]

    monkeypatch.setattr("localagent.train.rl._rollout", record_rollout)
    _, runner_metrics = grpo(
        LocalAgentLM(model_config),
        decisions,
        tokenizer,
        steps=steps,
        prompts_per_step=prompts_per_step,
        group_size=2,
        max_new=max_new,
        kl_beta=0.0,
        seed=seed,
        conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
        return_metrics=True,
        log=lambda *_: None,
    )
    expected_selected_lengths = []
    schedule = RLPromptSchedule(len(decisions), prompts_per_step, seed=seed)
    for step in range(steps):
        for index in schedule.indices_for_step(step):
            expected_selected_lengths.extend([prompt_lengths[index]] * 2)
    assert observed_prompt_lengths == expected_selected_lengths
    assert plan["planned"]["horizon_totals"]["prompt_groups"] == runner_metrics["attempted_groups"]
    assert plan["planned"]["horizon_totals"]["rollouts"] == runner_metrics["attempted_rollouts"]


def test_full_rl_plan_passes_4k_context_and_rejects_same_rows_at_2k(
    tmp_path: Path,
) -> None:
    tokenizer, tokenizer_path = _bpe_tokenizer(tmp_path)
    long_system = " ".join(f"segment-{index:04d}" for index in range(280))
    train_rows = [
        Conversation(
            messages=[
                Message(role=Role.system, content=long_system),
                Message(role=Role.user, content="Long-context train decision."),
                Message(role=Role.assistant, content="ok"),
            ]
        )
    ]
    eval_rows = [
        Conversation(
            messages=[
                Message(role=Role.system, content=long_system + " heldout"),
                Message(role=Role.user, content="Long-context eval decision."),
                Message(role=Role.assistant, content="ok"),
            ]
        )
    ]
    train_path = tmp_path / "long-train.jsonl"
    eval_path = tmp_path / "long-eval.jsonl"
    _write_conversations(train_path, train_rows)
    _write_conversations(eval_path, eval_rows)
    max_new = 8
    decision = project_rl_decisions(train_rows, sources=[str(train_path)])[0]
    prompt_length = len(tokenizer.encode(_decision_prompt_text(decision)))
    assert 2048 - max_new < prompt_length <= 4096 - max_new

    def config_for(max_seq_len: int) -> Path:
        _, model_path = _bpe_model_config(
            tmp_path,
            vocab_size=tokenizer.vocab_size,
            max_seq_len=max_seq_len,
            name=f"full-rl-{max_seq_len}",
        )
        return _write_config(
            tmp_path / f"rl-{max_seq_len}.yaml",
            {
                "stage": "rl",
                "model_config": str(model_path),
                "init_from": str(tmp_path / "unused-parent.pt"),
                "data": {
                    "conversation_prompt_contract": OPENAI_FULL_CATALOG_V1,
                    "conversations": [str(train_path)],
                    "eval_conversations": [str(eval_path)],
                    "tokenizer": {"kind": "bpe", "path": str(tokenizer_path)},
                },
                "rollout": {
                    "prompts_per_step": 1,
                    "group_size": 2,
                    "max_new_tokens": max_new,
                    "temperature": 1.0,
                },
                "schedule": {"total_steps": 1},
            },
        )

    plan = build_stage_budget_plan(config_for(4096))
    assert plan["data"]["context_preflight"]["train"]["max_prompt_tokens"] == prompt_length
    assert plan["data"]["context_preflight"]["train"]["truncated_rows"] == 0

    with pytest.raises(ValueError, match="cannot be truncated"):
        build_stage_budget_plan(config_for(2048))


@pytest.mark.parametrize("stage", ["midtrain", "sft"])
def test_full_contract_budget_overlap_uses_system_aware_prompt_hashes(
    tmp_path: Path,
    stage: str,
) -> None:
    tokenizer, tokenizer_path = _bpe_tokenizer(tmp_path)
    model_config, model_path = _bpe_model_config(
        tmp_path,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=512,
        name=f"{stage}-overlap-contract",
    )
    train_path = tmp_path / f"{stage}-train.jsonl"
    eval_path = tmp_path / f"{stage}-eval.jsonl"
    _write_conversations(
        train_path,
        [
            Conversation(
                messages=[
                    Message(role=Role.system, content="Train-only system."),
                    Message(role=Role.user, content="Shared user request."),
                    Message(role=Role.assistant, content="train"),
                ]
            )
        ],
    )
    _write_conversations(
        eval_path,
        [
            Conversation(
                messages=[
                    Message(role=Role.system, content="Held-out system."),
                    Message(role=Role.user, content="Shared user request."),
                    Message(role=Role.assistant, content="eval"),
                ]
            )
        ],
    )
    if stage == "midtrain":
        data = {
            "conversation_prompt_contract": OPENAI_FULL_CATALOG_V1,
            "tokenizer": {"kind": "bpe", "path": str(tokenizer_path)},
            "mixture": {"unit": "loss_tokens"},
            "sources": [
                {
                    "name": "train",
                    "type": "conversations",
                    "path": str(train_path),
                    "weight": 1.0,
                }
            ],
            "eval_sources": [
                {
                    "name": "eval",
                    "type": "conversations",
                    "path": str(eval_path),
                }
            ],
        }
        extra = {
            "batch": {"micro_batch_size": 1, "grad_accum_steps": 1},
            "evaluation": {"batches_per_source": 1, "batch_size": 1},
        }
    else:
        data = {
            "conversation_prompt_contract": OPENAI_FULL_CATALOG_V1,
            "tokenizer": {"kind": "bpe", "path": str(tokenizer_path)},
            "conversations": [str(train_path)],
            "eval_conversations": [str(eval_path)],
            "seq_len": model_config.max_seq_len,
        }
        extra = {
            "batch": {"micro_batch_size": 1, "grad_accum_steps": 1},
            "heads": {
                "joint_tool_pointer": False,
                "train_route_head": False,
                "train_dense_selector": False,
            },
        }
    config_path = _write_config(
        tmp_path / f"{stage}-overlap.yaml",
        {
            "stage": stage,
            "model_config": str(model_path),
            "init_from": str(tmp_path / "unused-parent.pt"),
            "data": data,
            "schedule": {"type": "cosine", "total_steps": 1},
            **extra,
        },
    )

    plan = build_stage_budget_plan(config_path)

    assert plan["data"]["conversation_prompt_contract"] == OPENAI_FULL_CATALOG_V1
    overlap = plan["data"]["conversation_overlap_audit"]
    assert overlap["semantic_overlap"] == 0
    assert overlap["rendered_prompt_overlap"] == 0


def test_plan_self_hash_tamper_and_artifact_drift_fail_verification(tmp_path) -> None:
    _, model_path = _model_config(tmp_path)
    conversation_path = tmp_path / "train.jsonl"
    rows = _training_conversations()[:2]
    _write_conversations(conversation_path, rows)
    config_path = _write_config(
        tmp_path / "sft.yaml",
        {
            "stage": "sft",
            "model_config": str(model_path),
            "init_from": str(tmp_path / "unused-parent.pt"),
            "data": {
                "conversations": [str(conversation_path)],
                "tokenizer": {"kind": "byte"},
                "shuffle": False,
            },
            "heads": {
                "joint_tool_pointer": False,
                "train_route_head": False,
                "train_dense_selector": False,
            },
            "schedule": {"total_steps": 2},
            "batch": {"micro_batch_size": 1, "grad_accum_steps": 1},
            "runtime": {"seed": 3},
        },
    )
    plan = build_stage_budget_plan(
        config_path,
        min_supervised_tokens=1,
        max_supervised_tokens=100,
    )
    plan_path = tmp_path / "plan.json"
    write_stage_budget_plan(plan_path, plan)
    assert verify_stage_budget_plan(plan_path) == plan

    tampered = copy.deepcopy(plan)
    tampered["calibration"]["selected_steps"] += 1
    plan_path.write_bytes(canonical_plan_bytes(tampered))
    with pytest.raises(ValueError, match="self-hash mismatch"):
        verify_stage_budget_plan(plan_path)

    resealed = seal_stage_budget_plan(
        {key: value for key, value in tampered.items() if key != "plan_self_sha256"}
    )
    plan_path.write_bytes(canonical_plan_bytes(resealed))
    with pytest.raises(ValueError, match="drifted"):
        verify_stage_budget_plan(plan_path)

    write_stage_budget_plan(plan_path, plan)
    rows.append(
        Conversation(
            messages=[
                Message(role=Role.user, content="new prompt"),
                Message(role=Role.assistant, content="new answer"),
            ]
        )
    )
    _write_conversations(conversation_path, rows)
    with pytest.raises(ValueError, match="drifted"):
        verify_stage_budget_plan(plan_path)
