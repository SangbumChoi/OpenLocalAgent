"""Focused contracts for config-driven SFT and verifiable-RL stage runners."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import pytest
import torch
import yaml

from openlocalagent.agent.dense_selector import DenseToolSelector
from openlocalagent.agent.pointer_head import PointerHead
from openlocalagent.agent.routes import RouteHead
from openlocalagent.agent.tool_head import ToolHead, _feat
from openlocalagent.agent.toolset import STANDARD_TOOLS
from openlocalagent.data.conversation_artifact import (
    CONVERSATION_FORMAT,
    CONVERSATION_SERIALIZATION,
    MANIFEST_KIND,
    MANIFEST_SCHEMA_VERSION,
    FileIdentity,
    self_hashed_manifest,
)
from openlocalagent.data.schema import Conversation, Message, Role, ToolCall, ToolSpec
from openlocalagent.model import LocalAgentLM, ModelConfig
from openlocalagent.model.tokenizer import ASSISTANT, USER, ByteTokenizer
from openlocalagent.train.midtrain import run as run_midtrain
from openlocalagent.train.rl import _rollout_reward
from openlocalagent.train.rl import run as run_rl
from openlocalagent.train.sft import run as run_sft
from openlocalagent.train.stage_data import (
    LINEAGE_VERSION,
    probe_decisions,
    read_conversations,
    single_turn_samples,
    tokenizer_identity,
)


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        name="stage-runner-test",
        vocab_size=256,
        d_model=16,
        embed_dim=8,
        n_layers=1,
        n_loops=1,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden=32,
        max_seq_len=256,
        dropout=0.0,
    )


def _write_yaml(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _write_conversations(path: Path, conversations: list[Conversation]) -> None:
    path.write_text(
        "".join(f"{conversation.to_json()}\n" for conversation in conversations),
        encoding="utf-8",
    )


def _write_verified_conversation_source(
    tmp_path: Path,
    *,
    name: str,
    conversations: list[Conversation],
    split: str,
) -> dict:
    """Publish a compact strict-artifact fixture using the production manifest schema."""

    path = tmp_path / f"{name}.jsonl"
    config_path = tmp_path / f"{name}.generator.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "kind": "stage_runner_conversation_fixture",
                "schema_version": 1,
                "out": str(path),
                "split": split,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    rows = []
    for conversation in conversations:
        row = Conversation.from_json(conversation.to_json())
        row.meta.update(
            {
                "split": split,
                "rule_verified": True,
                "environment_executed": False,
            }
        )
        rows.append(row)
    _write_conversations(path, rows)
    output_identity = FileIdentity.from_bytes(path.read_bytes())
    config_identity = FileIdentity.from_bytes(config_path.read_bytes())
    manifest_core = {
        "kind": MANIFEST_KIND,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "format": CONVERSATION_FORMAT,
        "conversation_serialization": CONVERSATION_SERIALIZATION,
        "generator_config": config_identity.as_dict(),
        "rows": len(rows),
        "output_bytes": output_identity.bytes,
        "output_sha256": output_identity.sha256,
        "single_turn": sum(len(row.messages) == 2 for row in rows),
        "multi_turn": sum(len(row.messages) != 2 for row in rows),
        "irrelevance": 0,
        "seed": 0,
        "level": 0,
        "split": split,
        "rule_verified": True,
        "rule_verification_scope": ["canonical_conversation_schema"],
        "model_verified": False,
        "environment_executed": False,
        "verification_claim": "test_fixture_rule_audited_not_environment_executed",
        "split_contract": {"fixture": True},
        "exact_prompt_holdouts": {"fixture": True},
        "structural_counts": {},
        "behavior_counts": {},
        "behavior_definitions": {},
        "argument_value_counts": {},
        "argument_schema_coverage": {},
        "plan_length_counts": {},
        "complexity_contract": {"fixture": True},
        "coverage_contract": {"fixture": True},
    }
    _manifest, manifest_payload = self_hashed_manifest(manifest_core)
    manifest_path = path.with_suffix(path.suffix + ".manifest.v1.json")
    manifest_path.write_bytes(manifest_payload)
    return {
        "path": str(path),
        "artifact": {
            "generator_config": str(config_path),
            "manifest": str(manifest_path),
            "expected_split": split,
            "expected_rule_verified": True,
            "environment_policy": "forbid",
        },
    }


def _stage_parent(payload: dict, *, stage: str) -> dict:
    tokenizer_sha256 = tokenizer_identity("byte", vocab_size=256)["sha256"]
    return {
        **payload,
        "stage": stage,
        "tokenizer": {"kind": "byte", "sha256": tokenizer_sha256},
        "lineage": {
            "version": LINEAGE_VERSION,
            "stage": stage,
            "tokenizer_sha256": tokenizer_sha256,
        },
    }


def _runner_conversations() -> list[Conversation]:
    return [
        Conversation(
            messages=[
                Message(role=Role.user, content="Search for Seoul"),
                Message(
                    role=Role.assistant,
                    tool_calls=[ToolCall(name="web_search", arguments={"query": "Seoul"})],
                ),
            ],
            meta={"category": "search", "group": "web_search"},
        ),
        Conversation(
            messages=[
                Message(role=Role.user, content="Say ready"),
                Message(role=Role.assistant, content="ready"),
            ],
            meta={"category": "text", "group": "text"},
        ),
        Conversation(
            messages=[
                Message(role=Role.user, content="Search for Busan"),
                Message(
                    role=Role.assistant,
                    tool_calls=[ToolCall(name="web_search", arguments={"query": "Busan"})],
                ),
                Message(
                    role=Role.tool,
                    tool_response=json.dumps({"result": f"Busan is in South Korea. {'x' * 300}"}),
                ),
                Message(
                    role=Role.assistant,
                    tool_calls=[ToolCall(name="web_search", arguments={"query": "Busan"})],
                ),
            ],
            meta={"category": "episode", "group": "multi_turn"},
        ),
    ]


def _assert_state_dict_equal(left: dict, right: dict) -> None:
    assert left.keys() == right.keys()
    for key, value in left.items():
        assert torch.equal(value, right[key]), key


def _assert_cpu_fp32_execution(checkpoint: dict, metrics: dict) -> None:
    execution = checkpoint["execution"]
    mps_backend = getattr(torch.backends, "mps", None)
    assert metrics["execution"] == execution
    assert execution == {
        "requested_device": "cpu",
        "resolved_device": "cpu",
        "requested_dtype": "fp32",
        "resolved_dtype": "fp32",
        "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_built": bool(mps_backend and mps_backend.is_built()),
        "mps_available": bool(mps_backend and mps_backend.is_available()),
        "torch_intraop_threads": int(torch.get_num_threads()),
        "torch_interop_threads": int(torch.get_num_interop_threads()),
    }
    json.dumps(execution, allow_nan=False)


def test_conversation_projection_preserves_parallel_calls_and_skips_multi_turn() -> None:
    conversations = [
        Conversation(
            messages=[
                Message(role=Role.user, content="Weather in Seoul"),
                Message(
                    role=Role.assistant,
                    tool_calls=[
                        ToolCall(
                            name="get_weather",
                            arguments={"unit": "celsius", "city": "Seoul"},
                        )
                    ],
                ),
            ],
            meta={"category": "weather", "group": "tool_call"},
        ),
        Conversation(
            messages=[
                Message(role=Role.user, content="Open example.com and take a screenshot"),
                Message(
                    role=Role.assistant,
                    tool_calls=[
                        ToolCall(name="open_url", arguments={"url": "example.com"}),
                        ToolCall(name="screenshot", arguments={}),
                    ],
                ),
            ],
            meta={"kind": "parallel", "group": "parallel"},
        ),
        Conversation(
            messages=[
                Message(role=Role.user, content="Say hello"),
                Message(role=Role.assistant, content="hello"),
            ],
        ),
        Conversation(
            messages=[
                Message(role=Role.user, content="Search for Seoul"),
                Message(
                    role=Role.assistant,
                    tool_calls=[ToolCall(name="web_search", arguments={"query": "Seoul"})],
                ),
                Message(role=Role.tool, tool_response='{"result":"found"}'),
                Message(role=Role.assistant, content="I found it."),
            ],
        ),
    ]

    samples = single_turn_samples(conversations)

    assert len(samples) == 3
    assert samples[0].category == "weather"
    assert samples[0].group == "tool_call"
    assert samples[0].ref_name == "get_weather"
    assert samples[0].ref_args == '{"city":"Seoul","unit":"celsius"}'
    assert samples[0].target == (
        '{"arguments":{"city":"Seoul","unit":"celsius"},"name":"get_weather"}'
    )
    assert samples[0].calls is None

    assert samples[1].category == "parallel"
    assert samples[1].ref_name == "open_url"
    assert samples[1].calls == [
        {"name": "open_url", "arguments": {"url": "example.com"}},
        {"name": "screenshot", "arguments": {}},
    ]

    assert samples[2].kind == "text"
    assert samples[2].target == "hello"
    assert all(sample.prompt != "Search for Seoul" for sample in samples)


def test_probe_decisions_preserve_single_turn_and_frame_each_trajectory_turn() -> None:
    decisions = probe_decisions(_runner_conversations())

    assert len(decisions) == 4
    assert (decisions[0].prompt, decisions[0].kind, decisions[0].ref_name) == (
        "Search for Seoul",
        "tool",
        "web_search",
    )
    assert decisions[0].framed is False
    assert (decisions[1].prompt, decisions[1].kind, decisions[1].ref_name) == (
        "Say ready",
        "text",
        "",
    )
    assert decisions[1].framed is False

    first_turn, follow_up = decisions[2:]
    assert first_turn.framed is True
    assert first_turn.prompt == f"{USER}Search for Busan{ASSISTANT}"
    assert follow_up.framed is True
    assert follow_up.kind == "tool"
    assert follow_up.ref_name == "web_search"
    assert follow_up.prompt.startswith(f"{USER}Search for Busan{ASSISTANT}<tool_call>")
    assert "<|tool|><tool_response>" in follow_up.prompt
    assert follow_up.prompt.endswith(ASSISTANT)


def test_probe_feature_does_not_double_wrap_framed_history() -> None:
    cfg = _tiny_config()
    torch.manual_seed(5)
    model = LocalAgentLM(cfg).eval()
    tokenizer = ByteTokenizer()
    raw = "Search for Seoul"
    framed = f"{USER}{raw}{ASSISTANT}"

    legacy_feature = _feat(model, tokenizer, raw, "cpu")
    framed_feature = _feat(model, tokenizer, framed, "cpu", framed=True)

    torch.testing.assert_close(framed_feature, legacy_feature, rtol=0, atol=0)
    long_history = probe_decisions(_runner_conversations())[-1].prompt
    expected_ids = torch.tensor(
        [tokenizer.encode(long_history)[-cfg.max_seq_len :]],
        dtype=torch.long,
    )
    with torch.no_grad():
        _, expected_hidden = model(expected_ids, return_hidden=True)
    actual = _feat(model, tokenizer, long_history, "cpu", framed=True)
    torch.testing.assert_close(actual, expected_hidden[0, -1], rtol=0, atol=0)


def test_read_conversations_uses_canonical_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "conversations.jsonl"
    expected = _runner_conversations()[:2]
    path.write_text(
        f"\n{expected[0].to_json()}\n\n{expected[1].to_json()}\n",
        encoding="utf-8",
    )

    actual = read_conversations(path)

    assert [conversation.to_json() for conversation in actual] == [
        conversation.to_json() for conversation in expected
    ]


def test_midtrain_runner_records_cpu_fp32_execution(tmp_path: Path) -> None:
    cfg = _tiny_config()
    model_config_path = tmp_path / "model.yaml"
    _write_yaml(model_config_path, cfg.__dict__)
    conversation_source = _write_verified_conversation_source(
        tmp_path,
        name="conversations",
        conversations=_runner_conversations()[:2],
        split="train",
    )
    init_checkpoint = tmp_path / "init.pt"
    torch.save(
        _stage_parent(
            {"cfg": cfg.__dict__, "state_dict": LocalAgentLM(cfg).state_dict()},
            stage="pretrain",
        ),
        init_checkpoint,
    )
    out_dir = tmp_path / "midtrain"
    config_path = tmp_path / "midtrain.yaml"
    _write_yaml(
        config_path,
        {
            "stage": "midtrain",
            "model_config": str(model_config_path),
            "init_from": str(init_checkpoint),
            "data": {
                "strict_conversation_artifacts": True,
                "tokenizer": {"kind": "byte"},
                "sources": [
                    {
                        "name": "agent",
                        "type": "conversations",
                        **conversation_source,
                        "weight": 1.0,
                    }
                ],
            },
            "optim": {"lr": 1e-3},
            "schedule": {"type": "cosine", "warmup_steps": 0, "total_steps": 1},
            "batch": {"micro_batch_size": 1, "grad_accum_steps": 1},
            "runtime": {"device": "cpu", "dtype": "fp32", "seed": 5},
            "log": {"out_dir": str(out_dir)},
        },
    )

    run_midtrain(str(config_path))

    checkpoint = torch.load(out_dir / "latest.pt", map_location="cpu", weights_only=False)
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert checkpoint["data"]["conversation_overlap_audit"]["semantic_overlap"] == 0
    assert checkpoint["data"]["conversation_overlap_audit"]["rendered_prompt_overlap"] == 0
    _assert_cpu_fp32_execution(checkpoint, metrics)


def test_rl_runner_requires_an_explicit_eval_dataset(tmp_path: Path) -> None:
    cfg = _tiny_config()
    model_config_path = tmp_path / "model.yaml"
    _write_yaml(model_config_path, cfg.__dict__)
    conversations_path = tmp_path / "conversations.jsonl"
    _write_conversations(conversations_path, _runner_conversations()[:2])
    config_path = tmp_path / "rl.yaml"
    _write_yaml(
        config_path,
        {
            "stage": "rl",
            "model_config": str(model_config_path),
            "data": {
                "conversations": [str(conversations_path)],
                "tokenizer": {"kind": "byte"},
            },
        },
    )

    with pytest.raises(ValueError, match="data.eval_conversations"):
        run_rl(str(config_path))


def test_rl_runner_rejects_embedded_gold_eos_before_loading_checkpoint(
    tmp_path: Path,
) -> None:
    cfg = _tiny_config()
    model_config_path = tmp_path / "model.yaml"
    _write_yaml(model_config_path, cfg.__dict__)
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_conversations(
        train_path,
        [
            Conversation(
                messages=[
                    Message(role=Role.user, content="Return the unsafe value"),
                    Message(role=Role.assistant, content="prefix\x00unreachable"),
                ]
            )
        ],
    )
    _write_conversations(
        eval_path,
        [
            Conversation(
                messages=[
                    Message(role=Role.user, content="Return the safe value"),
                    Message(role=Role.assistant, content="safe"),
                ]
            )
        ],
    )
    config_path = tmp_path / "rl.yaml"
    _write_yaml(
        config_path,
        {
            "stage": "rl",
            "model_config": str(model_config_path),
            "init_from": str(tmp_path / "does-not-exist.pt"),
            "data": {
                "conversations": [str(train_path)],
                "eval_conversations": [str(eval_path)],
                "tokenizer": {"kind": "byte"},
            },
            "rollout": {"max_new_tokens": 64},
        },
    )

    with pytest.raises(ValueError, match="embedded EOS token"):
        run_rl(str(config_path))


def test_verifiable_reward_scores_exact_tool_call_without_a_judge() -> None:
    sample = single_turn_samples(_runner_conversations())[0]
    exact = f"<tool_call>{sample.target}</tool_call>"

    assert _rollout_reward(sample, exact, format_weight=0.1, truncated=False) == 1.1
    assert (
        _rollout_reward(sample, "<tool_call>{}</tool_call>", format_weight=0.1, truncated=False)
        == 0
    )
    assert (
        _rollout_reward(
            sample,
            exact,
            format_weight=0.1,
            truncated=True,
            truncation_penalty=0.05,
        )
        == 1.05
    )


def test_sft_runner_seed_reproduces_backbone_and_all_new_heads(tmp_path: Path) -> None:
    cfg = _tiny_config()
    model_config_path = tmp_path / "model.yaml"
    _write_yaml(model_config_path, cfg.__dict__)
    conversations_path = tmp_path / "conversations.jsonl"
    _write_conversations(conversations_path, _runner_conversations()[:2])

    torch.manual_seed(3)
    init_checkpoint = tmp_path / "init.pt"
    torch.save(
        _stage_parent(
            {"cfg": cfg.__dict__, "state_dict": LocalAgentLM(cfg).state_dict()},
            stage="midtrain",
        ),
        init_checkpoint,
    )

    checkpoints = []
    for run_index in range(2):
        out_dir = tmp_path / f"sft-{run_index}"
        config_path = tmp_path / f"sft-{run_index}.yaml"
        _write_yaml(
            config_path,
            {
                "stage": "sft",
                "model_config": str(model_config_path),
                "init_from": str(init_checkpoint),
                "data": {
                    "conversations": [str(conversations_path)],
                    "tokenizer": {"kind": "byte"},
                    "seq_len": cfg.max_seq_len,
                    "function_masking": False,
                    "shuffle": True,
                },
                "optim": {"lr": 1e-3},
                "schedule": {"type": "cosine", "warmup_steps": 0, "total_steps": 1},
                "batch": {"micro_batch_size": 1, "grad_accum_steps": 1},
                "heads": {
                    "joint_tool_pointer": True,
                    "train_route_head": True,
                    "train_dense_selector": True,
                    "selector_proj": 4,
                    "route_steps": 1,
                    "selector_steps": 1,
                    "probe_batch_size": 1,
                    "example_centroids": True,
                },
                "runtime": {"device": "cpu", "dtype": "fp32", "seed": 41},
                "log": {"out_dir": str(out_dir)},
            },
        )
        torch.manual_seed(900 + run_index)
        run_sft(str(config_path))
        checkpoints.append(
            torch.load(out_dir / "latest.pt", map_location="cpu", weights_only=False)
        )

    for state_name in (
        "state_dict",
        "tool_head",
        "ptr_head",
        "route_head",
        "dense_selector",
    ):
        _assert_state_dict_equal(checkpoints[0][state_name], checkpoints[1][state_name])
    assert checkpoints[0]["loss_history"] == checkpoints[1]["loss_history"]


def test_sft_trains_route_and_dense_heads_from_multiturn_only_data(tmp_path: Path) -> None:
    cfg = _tiny_config()
    model_config_path = tmp_path / "model.yaml"
    _write_yaml(model_config_path, cfg.__dict__)
    conversations_path = tmp_path / "conversations.jsonl"
    conversation = Conversation(
        messages=[
            Message(role=Role.user, content="Search for Seoul"),
            Message(
                role=Role.assistant,
                tool_calls=[ToolCall(name="web_search", arguments={"query": "Seoul"})],
            ),
            Message(role=Role.tool, tool_response='{"result":"Seoul"}'),
            Message(role=Role.user, content="Now search for Busan"),
            Message(
                role=Role.assistant,
                tool_calls=[ToolCall(name="web_search", arguments={"query": "Busan"})],
            ),
        ]
    )
    _write_conversations(conversations_path, [conversation])

    torch.manual_seed(3)
    init_checkpoint = tmp_path / "init.pt"
    torch.save(
        _stage_parent(
            {"cfg": cfg.__dict__, "state_dict": LocalAgentLM(cfg).state_dict()},
            stage="midtrain",
        ),
        init_checkpoint,
    )
    out_dir = tmp_path / "sft"
    config_path = tmp_path / "sft.yaml"
    _write_yaml(
        config_path,
        {
            "stage": "sft",
            "model_config": str(model_config_path),
            "init_from": str(init_checkpoint),
            "data": {
                "conversations": [str(conversations_path)],
                "tokenizer": {"kind": "byte"},
                "seq_len": cfg.max_seq_len,
            },
            "optim": {"lr": 1e-3},
            "schedule": {"type": "cosine", "warmup_steps": 0, "total_steps": 1},
            "batch": {"micro_batch_size": 1, "grad_accum_steps": 1},
            "heads": {
                "joint_tool_pointer": False,
                "train_route_head": True,
                "train_dense_selector": True,
                "selector_proj": 4,
                "route_steps": 1,
                "selector_steps": 1,
                "probe_batch_size": 1,
            },
            "runtime": {"device": "cpu", "dtype": "fp32", "seed": 17},
            "log": {"out_dir": str(out_dir)},
        },
    )

    run_sft(str(config_path))

    checkpoint = torch.load(out_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert checkpoint["data"]["single_turn_rows"] == 0
    assert checkpoint["data"]["probe_decision_rows"] == 2
    assert checkpoint["route_head"] is not None
    assert checkpoint["dense_selector"] is not None


def test_sft_runner_rejects_same_shape_incompatible_checkpoint_config(
    tmp_path: Path,
) -> None:
    cfg = _tiny_config()
    model_config_path = tmp_path / "model.yaml"
    _write_yaml(model_config_path, cfg.__dict__)
    conversations_path = tmp_path / "conversations.jsonl"
    _write_conversations(conversations_path, _runner_conversations()[:2])

    incompatible_cfg = dict(cfg.__dict__)
    incompatible_cfg["rope_theta"] = cfg.rope_theta * 2
    init_checkpoint = tmp_path / "init.pt"
    torch.save(
        _stage_parent(
            {
                "cfg": incompatible_cfg,
                "state_dict": LocalAgentLM(cfg).state_dict(),
            },
            stage="midtrain",
        ),
        init_checkpoint,
    )
    config_path = tmp_path / "sft.yaml"
    _write_yaml(
        config_path,
        {
            "stage": "sft",
            "model_config": str(model_config_path),
            "init_from": str(init_checkpoint),
            "data": {
                "conversations": [str(conversations_path)],
                "tokenizer": {"kind": "byte"},
            },
            "heads": {
                "joint_tool_pointer": False,
                "train_route_head": False,
                "train_dense_selector": False,
            },
            "schedule": {"total_steps": 1},
            "runtime": {"device": "cpu", "seed": 1},
            "log": {"out_dir": str(tmp_path / "out")},
        },
    )

    with pytest.raises(ValueError, match="rope_theta"):
        run_sft(str(config_path))


def test_sft_runner_rejects_eval_overlap_from_decay_conversations(
    tmp_path: Path,
) -> None:
    cfg = _tiny_config()
    model_config_path = tmp_path / "model.yaml"
    _write_yaml(model_config_path, cfg.__dict__)
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    decay_path = tmp_path / "decay.jsonl"
    _write_conversations(
        train_path,
        [
            Conversation(
                messages=[
                    Message(role=Role.user, content="Say train"),
                    Message(role=Role.assistant, content="train"),
                ],
                meta={"split": "train"},
            )
        ],
    )
    heldout = Conversation(
        messages=[
            Message(role=Role.user, content="Say held out"),
            Message(role=Role.assistant, content="held out"),
        ],
        meta={"split": "eval"},
    )
    _write_conversations(eval_path, [heldout])
    _write_conversations(
        decay_path,
        [
            Conversation(
                messages=heldout.messages,
                meta={"split": "decay", "different_provenance": True},
            )
        ],
    )
    init_checkpoint = tmp_path / "init.pt"
    torch.save(
        _stage_parent(
            {"cfg": cfg.__dict__, "state_dict": LocalAgentLM(cfg).state_dict()},
            stage="midtrain",
        ),
        init_checkpoint,
    )
    config_path = tmp_path / "sft.yaml"
    _write_yaml(
        config_path,
        {
            "stage": "sft",
            "model_config": str(model_config_path),
            "init_from": str(init_checkpoint),
            "data": {
                "conversations": [str(train_path)],
                "decay_conversations": [str(decay_path)],
                "eval_conversations": [str(eval_path)],
                "tokenizer": {"kind": "byte"},
            },
            "heads": {
                "joint_tool_pointer": False,
                "train_route_head": False,
                "train_dense_selector": False,
            },
            "schedule": {"type": "wsd", "total_steps": 1},
            "runtime": {"device": "cpu", "seed": 1},
            "log": {"out_dir": str(tmp_path / "out")},
        },
    )

    with pytest.raises(ValueError, match="main/decay training content"):
        run_sft(str(config_path))


def test_sft_then_rl_invalidates_stale_structured_heads(tmp_path: Path) -> None:
    cfg = _tiny_config()
    model_config_path = tmp_path / "model.yaml"
    _write_yaml(model_config_path, cfg.__dict__)

    train_conversations = _runner_conversations()
    eval_tools = [
        ToolSpec(
            name="web_search",
            description="Search the web.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
    ]
    eval_conversations = [
        Conversation(
            messages=[
                Message(role=Role.user, content="Search for Jeju"),
                Message(
                    role=Role.assistant,
                    tool_calls=[ToolCall(name="web_search", arguments={"query": "Jeju"})],
                ),
            ],
            tools=eval_tools,
            meta={"category": "search", "group": "web_search", "split": "eval"},
        ),
        Conversation(
            messages=[
                Message(role=Role.user, content="Say complete"),
                Message(role=Role.assistant, content="complete"),
            ],
            tools=eval_tools,
            meta={"category": "text", "group": "text", "split": "eval"},
        ),
    ]
    conversation_source = _write_verified_conversation_source(
        tmp_path,
        name="conversations",
        conversations=train_conversations,
        split="train",
    )
    eval_conversation_source = _write_verified_conversation_source(
        tmp_path,
        name="eval-conversations",
        conversations=eval_conversations,
        split="eval",
    )
    conversations_path = Path(conversation_source["path"])
    eval_conversations_path = Path(eval_conversation_source["path"])

    torch.manual_seed(7)
    model = LocalAgentLM(cfg)
    inherited_heads = {
        "tool_head": ToolHead(cfg.d_model).state_dict(),
        "ptr_head": PointerHead(cfg.d_model).state_dict(),
        "route_head": RouteHead(cfg.d_model).state_dict(),
        "dense_selector": DenseToolSelector(
            cfg.d_model,
            emb_dim=8,
            proj=4,
        ).state_dict(),
    }
    init_checkpoint = tmp_path / "init.pt"
    torch.save(
        _stage_parent(
            {
                "cfg": cfg.__dict__,
                "state_dict": model.state_dict(),
                **inherited_heads,
                "selector_proj": 4,
                "examples": {"web_search": ["find a city"]},
            },
            stage="midtrain",
        ),
        init_checkpoint,
    )

    sft_dir = tmp_path / "sft"
    sft_config_path = tmp_path / "sft.yaml"
    _write_yaml(
        sft_config_path,
        {
            "stage": "sft",
            "model_config": str(model_config_path),
            "init_from": str(init_checkpoint),
            "data": {
                "strict_conversation_artifacts": True,
                "conversations": [conversation_source],
                "eval_conversations": [eval_conversation_source],
                "tokenizer": {"kind": "byte"},
                "seq_len": cfg.max_seq_len,
                "function_masking": False,
                "shuffle": False,
            },
            "optim": {"lr": 1e-3},
            "schedule": {"type": "cosine", "warmup_steps": 0, "total_steps": 1},
            "batch": {"micro_batch_size": 1, "grad_accum_steps": 1},
            "heads": {
                "joint_tool_pointer": True,
                "train_route_head": False,
                "train_dense_selector": False,
                "example_centroids": False,
            },
            "runtime": {"device": "cpu", "dtype": "fp32", "seed": 1},
            "evaluation": {"batch_size": 1},
            "log": {"out_dir": str(sft_dir)},
        },
    )

    run_sft(str(sft_config_path))

    sft_checkpoint = torch.load(sft_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert sft_checkpoint["stage"] == "sft"
    assert sft_checkpoint["step"] == 0
    assert len(sft_checkpoint["loss_history"]) == 1
    assert torch.isfinite(torch.tensor(sft_checkpoint["loss_history"])).all()
    sft_overlap_audit = sft_checkpoint["data"]["conversation_overlap_audit"]
    assert sft_overlap_audit["semantic_overlap"] == 0
    assert sft_overlap_audit["rendered_prompt_overlap"] == 0
    sft_function_masking = sft_checkpoint["data"].pop("function_masking")
    assert sft_function_masking["enabled"] is False
    assert sft_function_masking["main"]["enabled"] is False
    assert sft_function_masking["decay"]["enabled"] is False
    assert sft_checkpoint["data"] == {
        "conversation_rows": 3,
        "single_turn_rows": 2,
        "probe_decision_rows": 4,
        "paths": [str(conversations_path)],
            "conversation_overlap_audit": sft_overlap_audit,
            "eval_conversation_rows": 2,
            "eval_source_conversation_rows": 2,
            "eval_paths": [str(eval_conversations_path)],
        "heldout_content_overlap": 0,
        "heldout_rendered_prompt_overlap": 0,
    }
    assert sft_checkpoint["heldout_eval"]["pre"]["rows"] == 2
    assert sft_checkpoint["heldout_eval"]["post"]["rows"] == 2
    assert sft_checkpoint["heldout_eval"]["contract"]["same_rows_pre_post"] is True
    structured_eval = sft_checkpoint["heldout_structured_eval"]
    assert structured_eval["contract"] == {
        "kind": "frozen_route_and_dense_selector",
        "split": "explicit_disjoint_eval_conversations",
        "row_order": "configured_jsonl_assistant_decision_order",
        "dataset_sha256": sft_checkpoint["heldout_eval"]["contract"]["dataset_sha256"],
        "decision_rows": 2,
        "candidate_tools": len(STANDARD_TOOLS),
        "selector_embedding_dim": 8,
        "selector_projection_dim": 4,
        "fixed_compute_tokens": 256,
    }
    conditions = {condition["condition"]: condition for condition in structured_eval["conditions"]}
    assert set(conditions) == {"natural", "fixed_trailing_compute_256"}
    assert conditions["natural"]["configured_rows"] == 2
    assert conditions["natural"]["eligible_rows"] == 2
    assert conditions["natural"]["tool_rows"] == 1
    assert conditions["fixed_trailing_compute_256"]["eligible_rows"] == 2
    assert structured_eval["prediction_invariance"] == {
        "reference_condition": "natural",
        "comparison_condition": "fixed_trailing_compute_256",
        "comparable_rows": 2,
        "reference_only_rows": 0,
        "comparison_only_rows": 0,
        "route_prediction_mismatches": 0,
        "selector_prediction_mismatches": 0,
        "dispatched_prediction_mismatches": 0,
        "all_comparable_predictions_match": True,
    }
    for name in ("route_head", "dense_selector"):
        state = inherited_heads[name]
        _assert_state_dict_equal(sft_checkpoint[name], state)
    for name in ("tool_head", "ptr_head"):
        assert sft_checkpoint[name].keys() == inherited_heads[name].keys()
        assert any(
            not torch.equal(sft_checkpoint[name][key], inherited_heads[name][key])
            for key in inherited_heads[name]
        )
    assert sft_checkpoint["selector_proj"] == 4
    assert sft_checkpoint["examples"] == {"web_search": ["find a city"]}

    sft_metrics = json.loads((sft_dir / "metrics.json").read_text(encoding="utf-8"))
    assert sft_metrics["loss_steps"] == 1
    assert sft_metrics["conversation_rows"] == 3
    assert sft_metrics["single_turn_rows"] == 2
    assert sft_metrics["probe_decision_rows"] == 4
    assert sft_metrics["heldout_structured_eval"] == structured_eval
    _assert_cpu_fp32_execution(sft_checkpoint, sft_metrics)

    rl_dir = tmp_path / "rl"
    rl_config_path = tmp_path / "rl.yaml"
    _write_yaml(
        rl_config_path,
        {
            "stage": "rl",
            "model_config": str(model_config_path),
            "init_from": str(sft_dir / "latest.pt"),
            "data": {
                "strict_conversation_artifacts": True,
                "conversations": [conversation_source],
                "eval_conversations": [eval_conversation_source],
                "tokenizer": {"kind": "byte"},
            },
            "environment": {
                "name": "canonical_toolcalls",
                "learned_judge": False,
            },
            "rollout": {
                "prompts_per_step": 1,
                "group_size": 2,
                "max_new_tokens": 80,
                "temperature": 1.0,
            },
            "policy": {
                "clip_ratio": 0.2,
                "kl_beta": 0.0,
                "epochs_per_rollout": 1,
            },
            "reward": {"format_weight": 0.1, "truncation_penalty": 0.05},
            "optim": {"lr": 1e-4},
            "schedule": {"total_steps": 1},
            "runtime": {"device": "cpu", "dtype": "fp32", "seed": 13},
            "log": {"out_dir": str(rl_dir)},
        },
    )

    run_rl(str(rl_config_path))

    rl_checkpoint = torch.load(rl_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert rl_checkpoint["stage"] == "rl"
    assert rl_checkpoint["step"] == 0
    assert len(rl_checkpoint["reward_history"]) == 1
    assert torch.isfinite(torch.tensor(rl_checkpoint["reward_history"])).all()
    assert rl_checkpoint["reward_contract"] == {
        "environment": "canonical_toolcalls",
        "correctness": "exact normalized tool AST; exact text match",
        "format_weight": 0.1,
        "format_validation": (
            "registry name + argument schema when available; parsed AST fallback"
        ),
        "truncation_penalty": 0.05,
        "learned_judge": False,
        "policy_scope": "autoregressive_lm_only",
    }
    assert rl_checkpoint["policy_contract"] == {
        "objective": "sampled_token_clipped_grpo",
        "ratio_scope": "generated_tokens_only",
        "reference_kl": "sampled_token_k3",
        "includes_sampled_eos": True,
        "epochs_per_rollout": 1,
    }
    assert rl_checkpoint["structured_heads_available"] is False
    assert rl_checkpoint["invalidated_structured_heads"] == [
        "tool_head",
        "ptr_head",
        "route_head",
        "dense_selector",
    ]
    for name in inherited_heads:
        assert name not in rl_checkpoint
    assert rl_checkpoint["data"]["conversation_rows"] == 3
    assert rl_checkpoint["data"]["single_turn_rows"] == 2
    assert rl_checkpoint["data"]["eval_conversation_rows"] == 2
    assert rl_checkpoint["data"]["eval_single_turn_rows"] == 2
    assert rl_checkpoint["data"]["eval_paths"] == [str(eval_conversations_path)]
    split_audit = rl_checkpoint["data"]["split_audit"]
    assert split_audit["row_overlap"] == 0
    assert split_audit["prompt_overlap"] == 0
    assert split_audit["conversation_overlap_audit"]["semantic_overlap"] == 0
    assert split_audit["conversation_overlap_audit"]["rendered_prompt_overlap"] == 0
    assert len(split_audit["train_dataset_sha256"]) == 64
    assert len(split_audit["eval_dataset_sha256"]) == 64
    assert split_audit["train_scored_rows"] == 2
    assert split_audit["eval_scored_rows"] == 2
    assert split_audit["train_dataset_sha256"] != split_audit["train_scored_rows_sha256"]
    assert len(split_audit["eval_scored_rows_sha256"]) == 64
    heldout = rl_checkpoint["heldout_eval"]
    assert heldout["contract"] == {
        "split": "explicit_disjoint_eval_conversations",
        "dataset_sha256": split_audit["eval_scored_rows_sha256"],
        "decoding": "greedy_argmax",
        "max_new_tokens": 80,
        "same_rows_pre_post": True,
    }
    assert heldout["pre"]["n"] == 2
    assert heldout["post"]["n"] == 2
    assert heldout["pre"]["tool_rows"] == 1
    assert heldout["pre"]["text_rows"] == 1
    assert heldout["pre"]["schema_covered_tool_rows"] == 1
    assert set(heldout["delta"]) >= {"exact_match_accuracy", "mean_reward"}
    assert rl_checkpoint["rl_accounting"]["attempted_rollout_steps"] == 1
    assert rl_checkpoint["rl_accounting"]["attempted_groups"] == 1
    assert rl_checkpoint["rl_accounting"]["attempted_rollouts"] == 2
    assert (
        rl_checkpoint["rl_accounting"]["realized_optimizer_updates"]
        <= rl_checkpoint["policy_contract"]["epochs_per_rollout"]
    )
    assert rl_checkpoint["lineage"]["stage"] == "rl"
    assert len(rl_checkpoint["lineage"]["parent_checkpoint_sha256"]) == 64
    assert len(rl_checkpoint["lineage"]["tokenizer_sha256"]) == 64
    assert rl_checkpoint["lineage"]["git"] is not None

    rl_metrics = json.loads((rl_dir / "metrics.json").read_text(encoding="utf-8"))
    assert rl_metrics["reward_steps"] == 1
    assert rl_metrics["reward_contract"] == rl_checkpoint["reward_contract"]
    assert rl_metrics["policy_contract"] == rl_checkpoint["policy_contract"]
    assert rl_metrics["data"] == rl_checkpoint["data"]
    assert rl_metrics["heldout_eval"] == heldout
    assert rl_metrics["rl_accounting"] == rl_checkpoint["rl_accounting"]
    assert rl_metrics["lineage"] == rl_checkpoint["lineage"]
    assert rl_metrics["structured_heads_available"] is False
    assert (
        rl_metrics["invalidated_structured_heads"]
        == (rl_checkpoint["invalidated_structured_heads"])
    )
    _assert_cpu_fp32_execution(rl_checkpoint, rl_metrics)
