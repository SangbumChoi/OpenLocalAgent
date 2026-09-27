from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch

from localagent.data.prompt_contract import assistant_training_examples
from localagent.data.schema import Conversation, Message, Role, ToolCall, ToolSpec
from localagent.eval.real_use import (
    RealUseRequirements,
    audit_public_real_use_cases,
    collect_router_diagnostics,
    score_public_real_use,
    score_public_real_use_dataset,
    summarize_router_diagnostics,
)
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import ByteTokenizer


def _tools() -> list[ToolSpec]:
    string = {"type": "string"}
    return [
        ToolSpec(
            name="search",
            description="Search public records.",
            parameters={
                "type": "object",
                "properties": {"query": string},
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="send",
            description="Send one message.",
            parameters={
                "type": "object",
                "properties": {"to": string, "text": string},
                "required": ["to", "text"],
                "additionalProperties": False,
            },
        ),
    ]


def _meta(
    *,
    index: int,
    category: str,
    behavior: str,
    action_count: int,
    capabilities: list[str],
) -> dict:
    return {
        "category": category,
        "group": "public_agent",
        "kind": "public_agent_trace",
        "split": "eval",
        "generator": "public_agent_snapshot_v1",
        "public_data": True,
        "behavior": behavior,
        "capabilities": capabilities,
        "action_count": action_count,
        "enrichment_level": 1,
        "parent_record_id": f"parent-{index}",
        "rule_verified": True,
        "model_verified": False,
        "environment_executed": False,
        "verification_scope": "schema_catalog_arguments_sequence_and_split_slots",
        "provenance": {
            "dataset": "public/example-agent",
            "subset": "heldout",
            "revision": "revision-2026-07",
            "record_id": f"record-{index}",
            "url": "https://example.org/public-agent-dataset",
            "license": "Apache-2.0",
            "file_sha256": f"{index + 1:064x}",
            "source_line": index + 1,
        },
    }


def _conversations() -> list[Conversation]:
    tools = _tools()
    return [
        Conversation(
            messages=[
                Message(role=Role.user, content="Find the incident and notify Ada."),
                Message(
                    role=Role.assistant,
                    tool_calls=[ToolCall(name="search", arguments={"query": "incident 42"})],
                ),
                Message(role=Role.tool, tool_response='{"title":"Incident 42"}'),
                Message(role=Role.user, content="Send that title."),
                Message(
                    role=Role.assistant,
                    tool_calls=[
                        ToolCall(
                            name="send",
                            arguments={"to": "Ada", "text": "Incident 42"},
                        )
                    ],
                ),
            ],
            tools=tools,
            meta=_meta(
                index=0,
                category="productivity",
                behavior="action",
                action_count=2,
                capabilities=["message_send", "retrieval"],
            ),
        ),
        Conversation(
            messages=[
                Message(role=Role.user, content="Do not send yet; wait for approval."),
                Message(role=Role.assistant, content="I will wait for approval."),
            ],
            tools=tools,
            meta=_meta(
                index=1,
                category="safety",
                behavior="abstention",
                action_count=0,
                capabilities=["safe_abstention"],
            ),
        ),
        Conversation(
            messages=[
                Message(role=Role.user, content="Tell me a short joke."),
                Message(role=Role.assistant, content="Why did the byte cross the bus?"),
            ],
            tools=tools,
            meta=_meta(
                index=2,
                category="irrelevant",
                behavior="irrelevance",
                action_count=0,
                capabilities=["irrelevance_detection"],
            ),
        ),
    ]


def _requirements(*, require_sparse_router: bool = True) -> RealUseRequirements:
    return RealUseRequirements(
        min_action_exact=1.0,
        min_tool_call_exact=1.0,
        min_abstention_accuracy=1.0,
        min_irrelevance_accuracy=1.0,
        min_multi_turn_step_exact=1.0,
        min_multi_turn_episode_exact=1.0,
        min_category_action_exact=1.0,
        min_conversations=3,
        min_datasets=1,
        min_categories=3,
        min_action_conversations=1,
        min_abstention_conversations=1,
        min_irrelevance_conversations=1,
        min_multi_turn_action_conversations=1,
        required_capabilities=(
            "irrelevance_detection",
            "message_send",
            "retrieval",
            "safe_abstention",
        ),
        require_sparse_router=require_sparse_router,
        min_router_utilization=1.0,
        min_router_normalized_entropy=0.8,
        min_router_category_divergence=0.6,
    )


def _gold_predictor(conversations: list[Conversation]):
    outputs = {
        example.prompt: example.body
        for conversation in conversations
        for example in assistant_training_examples(conversation)
    }
    return lambda prompt, _tools: outputs[prompt]


def _diagnostics(counts: list[int]) -> dict:
    assignments = sum(counts)
    return {
        "enabled": True,
        "num_experts": len(counts),
        "top_k": 1,
        "invocations": 1,
        "tokens": assignments,
        "assignments": assignments,
        "expert_counts": counts,
        "expert_load": [count / assignments for count in counts],
        "router_probability": [count / assignments for count in counts],
        "router_entropy": 0.0,
        "load_balance_loss": 1.0,
    }


def test_public_real_use_gate_scores_multiturn_abstention_categories_and_router():
    conversations = _conversations()
    router = summarize_router_diagnostics(
        [
            ("productivity", _diagnostics([8, 0])),
            ("safety", _diagnostics([0, 8])),
            ("irrelevant", _diagnostics([0, 8])),
        ]
    )

    result = score_public_real_use(
        conversations,
        _gold_predictor(conversations),
        _requirements(),
        router_report=router,
    )

    assert result["gates"]["all_passed"] is True
    assert result["coverage"]["behaviors"] == {
        "abstention": 1,
        "action": 1,
        "irrelevance": 1,
    }
    assert result["coverage"]["multi_turn_action_conversations"] == 1
    assert result["coverage"]["parallel_action_conversations"] == 0
    assert result["score"]["metrics"]["whole_call_exact"]["accuracy"] == 1.0
    assert result["score"]["metrics"]["abstention"]["accuracy"] == 1.0
    assert result["by_behavior"] == {
        "abstention": {"correct": 1, "total": 1, "accuracy": 1.0},
        "action": {"correct": 2, "total": 2, "accuracy": 1.0},
        "irrelevance": {"correct": 1, "total": 1, "accuracy": 1.0},
    }
    assert result["score"]["metrics"]["teacher_forced_tool_multi_turn"] == {
        "tool_step_exact": {"correct": 2, "total": 2, "accuracy": 1.0},
        "tool_episode_exact": {"correct": 1, "total": 1, "accuracy": 1.0},
    }
    assert set(result["score"]["by_category"]) == {
        "irrelevant",
        "productivity",
        "safety",
    }
    assert router["expert_counts"] == [8, 16]
    assert router["utilization"] == 1.0
    assert router["normalized_entropy"] > 0.9
    assert (
        router["category_distribution_divergence"]["mean_normalized_jensen_shannon"]
        == pytest.approx(2 / 3)
    )


def test_public_contract_rejects_train_rows_and_action_count_drift():
    conversations = _conversations()
    conversations[0].meta["split"] = "train"
    with pytest.raises(ValueError, match=r"meta.split must equal 'eval'"):
        audit_public_real_use_cases(conversations)

    conversations = _conversations()
    conversations[0].meta["action_count"] = 1
    with pytest.raises(ValueError, match="does not match 2 assistant tool calls"):
        audit_public_real_use_cases(conversations)


def test_parallel_calls_do_not_masquerade_as_multiturn_coverage():
    conversation = Conversation(
        messages=[
            Message(role=Role.user, content="Search and notify in parallel."),
            Message(
                role=Role.assistant,
                tool_calls=[
                    ToolCall(name="search", arguments={"query": "incident 42"}),
                    ToolCall(
                        name="send",
                        arguments={"to": "Ada", "text": "Starting lookup"},
                    ),
                ],
            ),
        ],
        tools=_tools(),
        meta=_meta(
            index=3,
            category="productivity",
            behavior="action",
            action_count=2,
            capabilities=["message_send", "retrieval"],
        ),
    )

    coverage = audit_public_real_use_cases([conversation])

    assert coverage["tool_decisions"] == 1
    assert coverage["parallel_action_conversations"] == 1
    assert coverage["multi_turn_action_conversations"] == 0


def test_dense_or_missing_router_never_receives_diversity_credit():
    conversations = _conversations()
    dense = summarize_router_diagnostics(
        [
            ("productivity", {"enabled": False}),
            ("safety", {"enabled": False}),
            ("irrelevant", {"enabled": False}),
        ]
    )

    result = score_public_real_use(
        conversations,
        _gold_predictor(conversations),
        _requirements(),
        router_report=dense,
    )

    assert dense == {
        "available": True,
        "enabled": False,
        "reason": "dense_model",
        "cases": 3,
    }
    assert result["gates"]["all_passed"] is False
    gate = next(
        record
        for record in result["gates"]["records"]
        if record["name"] == "sparse_router_telemetry"
    )
    assert gate["observed"] is False
    assert gate["passed"] is False
    assert "utilization" not in dense


def test_quality_gate_reports_exact_failure_without_rounding_or_hiding_categories():
    conversations = _conversations()
    result = score_public_real_use(
        conversations,
        lambda _prompt, _tools: "No tool.",
        _requirements(require_sparse_router=False),
    )

    assert result["gates"]["all_passed"] is False
    assert result["score"]["metrics"]["action_exact"] == {
        "correct": 2,
        "total": 4,
        "accuracy": 0.5,
    }
    assert result["score"]["metrics"]["whole_call_exact"] == {
        "correct": 0,
        "total": 2,
        "accuracy": 0.0,
    }
    assert result["score"]["by_category"]["productivity"]["action_exact"]["accuracy"] == 0.0
    category_gate = next(
        record
        for record in result["gates"]["records"]
        if record["name"] == "minimum_per_category_action_exact"
    )
    assert category_gate["passed"] is False
    assert category_gate["observed"]["productivity"] == 0.0


def test_router_summary_rejects_inconsistent_assignment_telemetry():
    invalid = _diagnostics([3, 1])
    invalid["assignments"] = 3
    with pytest.raises(ValueError, match=r"assignments != sum\(expert_counts\)"):
        summarize_router_diagnostics([("category", invalid)])


class _FakeRouterModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cfg = SimpleNamespace(max_seq_len=16_384)
        self._diagnostics = {"enabled": False}

    def forward(self, indices: torch.Tensor):
        tokens = int(indices.numel())
        self._diagnostics = _diagnostics([tokens, 0])
        return indices

    def routing_diagnostics(self) -> dict:
        return self._diagnostics


def test_collect_router_diagnostics_uses_model_interface_and_restores_mode():
    model = _FakeRouterModel()
    model.train()
    conversations = _conversations()[1:]

    report = collect_router_diagnostics(
        model,
        conversations,
        ByteTokenizer(),
    )

    assert model.training is True
    assert report["available"] is True
    assert report["enabled"] is True
    assert report["cases"] == 2
    assert report["expert_counts"][0] == report["tokens"]
    assert report["expert_counts"][1] == 0
    assert report["utilization"] == 0.5


def test_collect_router_diagnostics_integrates_with_sparse_localagent_model():
    config = ModelConfig(
        name="router-eval-test",
        vocab_size=256,
        d_model=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden=16,
        max_seq_len=4096,
        ffn_num_experts=2,
        ffn_top_k=1,
    )
    model = LocalAgentLM(config)

    report = collect_router_diagnostics(
        model,
        _conversations()[1:],
        ByteTokenizer(),
    )

    assert report["available"] is True
    assert report["enabled"] is True
    assert report["num_experts"] == 2
    assert report["top_k"] == 1
    assert report["assignments"] == report["tokens"]
    assert report["total_parameters"] == model.num_params()
    assert report["active_parameters"] == model.active_num_params()
    assert report["active_parameter_fraction"] < 1.0


def test_frozen_public_dataset_entrypoint_binds_exact_jsonl_bytes(tmp_path):
    conversations = _conversations()
    payload = "".join(conversation.to_json() + "\n" for conversation in conversations).encode()
    dataset = tmp_path / "public-real-use-eval.jsonl"
    dataset.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    result = score_public_real_use_dataset(
        dataset,
        digest,
        _gold_predictor(conversations),
        _requirements(require_sparse_router=False),
    )

    assert result["gates"]["all_passed"] is True
    assert result["dataset_artifact"] == {
        "path": str(dataset),
        "bytes": len(payload),
        "sha256": digest,
        "frozen_identity_verified": True,
    }
    with pytest.raises(ValueError, match="does not match the frozen identity"):
        score_public_real_use_dataset(
            dataset,
            "0" * 64,
            _gold_predictor(conversations),
            _requirements(require_sparse_router=False),
        )
