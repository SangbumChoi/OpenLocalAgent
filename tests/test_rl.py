"""Focused policy-rollout contracts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from openlocalagent.data.synth.agent_synth import Generator, Sample
from openlocalagent.data.conversation_artifact import conversation_semantic_sha256
from openlocalagent.data.prompt_contract import (
    LEGACY_CONVERSATION_PROMPT_CONTRACT,
    OPENAI_FULL_CATALOG_V1,
    render_agent_decode_prompt,
)
from openlocalagent.data.render import assistant_body, prompt_text
from openlocalagent.data.schema import Conversation, Message, Role, ToolCall, ToolSpec
from openlocalagent.model import LocalAgentLM, ModelConfig
from openlocalagent.model.tokenizer import ASSISTANT, BPE_EOS, ByteTokenizer, train_bpe
from openlocalagent.train.rl import (
    CatalogStringCache,
    _assert_gold_outputs_fit,
    _assert_parent_prompt_contract,
    _audit_data_splits,
    _correct_for_contract,
    _decision_fingerprint,
    _decision_prompt_text,
    _evaluate_holdout,
    _fingerprint_set,
    _grpo_token_loss,
    _preflight_full_context,
    _prompt_ids_for_policy,
    _rollout,
    _rollout_reward,
    _stateful_productivity_reward,
    _token_logprobs,
    _valid_tool_call_format,
    grpo,
    project_rl_decisions,
    run,
)
from openlocalagent.train.stage_data import canonical_sha256, tokenizer_identity


class _ImmediateEosModel:
    def n_cache_slots(self) -> int:
        return 1

    def __call__(self, tokens, *, pos=0, caches=None):
        logits = torch.full(
            (tokens.shape[0], tokens.shape[1], 256),
            float("-inf"),
            device=tokens.device,
        )
        logits[..., ByteTokenizer.eos_id] = 0.0
        return logits, None, caches


class _UniformNonEosModel:
    def n_cache_slots(self) -> int:
        return 1

    def __call__(self, tokens, *, pos=0, caches=None):
        logits = torch.zeros(
            (tokens.shape[0], tokens.shape[1], 256),
            device=tokens.device,
        )
        logits[..., ByteTokenizer.eos_id] = float("-inf")
        return logits, None, caches


class _TrainablePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(256, 8)
        self.proj = nn.Linear(8, 256)
        self.cfg = SimpleNamespace(max_seq_len=64)

    def forward(self, tokens):
        return self.proj(self.embed(tokens)), None


class _EvaluationShell:
    cfg = SimpleNamespace(max_seq_len=4096)

    def to(self, _device):
        return self

    def eval(self):
        return self


def _text_conversation(prompt: str, target: str, *, source: str) -> Conversation:
    return Conversation(
        messages=[
            Message(role=Role.user, content=prompt),
            Message(role=Role.assistant, content=target),
        ],
        meta={"source": source},
    )


def _text_sample(prompt: str, target: str) -> Sample:
    return Sample(
        category="text",
        group="text",
        prompt=prompt,
        kind="text",
        target=target,
    )


@pytest.fixture(scope="module")
def rl_bpe_tokenizer(tmp_path_factory):
    path: Path = tmp_path_factory.mktemp("rl-full-contract") / "tokenizer.json"
    return train_bpe(
        [
            (
                "catalog system inspect telemetry satellite samples result summarize "
                "stable query nested object assistant tool response alpha beta gamma"
            )
        ],
        path,
        vocab_size=300,
        min_frequency=1,
    )


def _nested_tool() -> ToolSpec:
    return ToolSpec(
        name="inspect_telemetry",
        description="Inspect exact telemetry.",
        parameters={
            "type": "object",
            "properties": {
                "satellite": {"type": "string"},
                "window": {
                    "type": "object",
                    "properties": {
                        "samples": {
                            "type": "array",
                            "items": {"type": "number"},
                        }
                    },
                    "required": ["samples"],
                    "additionalProperties": False,
                },
            },
            "required": ["satellite", "window"],
            "additionalProperties": False,
        },
    )


def _trajectory(
    *, final: str = "Telemetry is stable.", system: str = "Stay exact."
) -> Conversation:
    tool = _nested_tool()
    return Conversation(
        tools=[tool],
        messages=[
            Message(role=Role.system, content=system),
            Message(role=Role.user, content="Inspect Asteria."),
            Message(
                role=Role.assistant,
                tool_calls=[
                    ToolCall(
                        name=tool.name,
                        arguments={
                            "satellite": "Asteria",
                            "window": {"samples": [1, 2.5]},
                        },
                    )
                ],
            ),
            Message(role=Role.tool, tool_response='{"samples":[1,2.5]}'),
            Message(role=Role.user, content="Summarize."),
            Message(role=Role.assistant, content=final),
        ],
    )


def test_rollout_retains_eos_for_policy_gradient() -> None:
    tokenizer = ByteTokenizer()
    generated = _rollout(
        _ImmediateEosModel(),
        tokenizer,
        tokenizer.encode("prompt"),
        max_new=4,
        temperature=1.0,
        device="cpu",
        generator=torch.Generator().manual_seed(0),
    )

    assert generated == [tokenizer.eos_id]
    assert tokenizer.decode(generated) == ""


def test_rollout_is_seeded_and_reports_a_full_non_eos_generation_as_truncated() -> None:
    tokenizer = ByteTokenizer()
    model = _UniformNonEosModel()

    first = _rollout(
        model,
        tokenizer,
        tokenizer.encode("prompt"),
        max_new=8,
        temperature=0.7,
        device="cpu",
        generator=torch.Generator().manual_seed(42),
    )
    torch.rand(20)  # the explicit rollout generator is independent of global RNG consumption
    second = _rollout(
        model,
        tokenizer,
        tokenizer.encode("prompt"),
        max_new=8,
        temperature=0.7,
        device="cpu",
        generator=torch.Generator().manual_seed(42),
    )

    assert first == second
    assert len(first) == 8
    assert tokenizer.eos_id not in first


def test_rl_data_audit_rejects_row_and_prompt_overlap() -> None:
    row = _text_conversation("Say ready", "ready", source="same")
    sample = _text_sample("Say ready", "ready")
    with pytest.raises(ValueError, match="conversation row fingerprint"):
        _audit_data_splits(
            [row],
            [_text_conversation("Say ready", "ready", source="different provenance")],
            [sample],
            [sample],
        )

    train_row = _text_conversation("Say ready", "ready", source="train")
    eval_row = _text_conversation("Say ready", "different", source="eval")
    with pytest.raises(ValueError, match="prompt fingerprint"):
        _audit_data_splits(
            [train_row],
            [eval_row],
            [sample],
            [_text_sample("Say ready", "different")],
        )


def test_rl_data_audit_fingerprints_disjoint_rows_stably() -> None:
    train_rows = [
        _text_conversation("Say alpha", "alpha", source="train"),
        _text_conversation("Say beta", "beta", source="train"),
        Conversation(
            messages=[
                Message(role=Role.user, content="Start an episode"),
                Message(role=Role.assistant, content="started"),
                Message(role=Role.user, content="Finish the episode"),
                Message(role=Role.assistant, content="finished"),
            ],
            meta={"source": "train", "kind": "episode"},
        ),
    ]
    eval_rows = [_text_conversation("Say gamma", "gamma", source="eval")]
    train_samples = [
        _text_sample("Say alpha", "alpha"),
        _text_sample("Say beta", "beta"),
    ]
    eval_samples = [_text_sample("Say gamma", "gamma")]

    first = _audit_data_splits(train_rows, eval_rows, train_samples, eval_samples)
    reordered = _audit_data_splits(
        list(reversed(train_rows)),
        eval_rows,
        list(reversed(train_samples)),
        eval_samples,
    )

    assert first == reordered
    assert first["row_overlap"] == 0
    assert first["prompt_overlap"] == 0
    assert len(first["train_dataset_sha256"]) == 64
    assert len(first["eval_dataset_sha256"]) == 64
    assert first["train_scored_rows"] == 2
    assert first["eval_scored_rows"] == 1
    assert first["train_dataset_sha256"] != first["train_scored_rows_sha256"]
    assert len(first["eval_scored_rows_sha256"]) == 64
    assert len(first["eval_scored_prompts_sha256"]) == 64


def test_rl_data_audit_accepts_real_disjoint_agent_synth_prompt_pools() -> None:
    train_samples = Generator(level=5, seed=11, split="train").generate_balanced(8)
    eval_samples = Generator(level=5, seed=909, split="eval").generate_balanced(8)

    audit = _audit_data_splits([], [], train_samples, eval_samples)

    assert audit["prompt_overlap"] == 0


def test_rl_rejects_gold_outputs_that_cannot_fit_exact_reward_budget() -> None:
    tokenizer = ByteTokenizer()
    samples = [_text_sample("Repeat", "a target longer than one token")]

    with pytest.raises(ValueError, match="longer than max_new_tokens=1"):
        _assert_gold_outputs_fit(samples, tokenizer, 1, split="eval")

    budget = _assert_gold_outputs_fit(samples, tokenizer, 64, split="eval")
    assert budget["rows"] == 1
    assert budget["max_gold_tokens_including_eos"] <= budget["decoding_budget"]


def test_rl_rejects_gold_with_embedded_eos_before_terminal_eos() -> None:
    tokenizer = ByteTokenizer()
    samples = [_text_sample("Repeat safely", "prefix\x00unreachable suffix")]

    with pytest.raises(ValueError, match="embedded EOS token"):
        _assert_gold_outputs_fit(samples, tokenizer, 64, split="eval")


def test_grpo_reports_zero_signal_steps_without_claiming_optimizer_updates(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "openlocalagent.train.rl._rollout",
        lambda *_args, **_kwargs: [ByteTokenizer.eos_id],
    )

    history, metrics = grpo(
        _TrainablePolicy(),
        [_text_sample("Say ready", "ready")],
        ByteTokenizer(),
        steps=2,
        prompts_per_step=1,
        group_size=2,
        max_new=4,
        kl_beta=0.0,
        return_metrics=True,
        log=lambda *_: None,
    )

    assert history == [0.0, 0.0]
    expected = {
        "attempted_rollout_steps": 2,
        "attempted_groups": 2,
        "attempted_rollouts": 4,
        "zero_signal_steps": 2,
        "informative_groups": 0,
        "realized_optimizer_updates": 0,
        "policy_epochs_per_informative_batch": 1,
    }
    assert {key: metrics[key] for key in expected} == expected
    assert metrics["generated_tokens"] == 4
    assert metrics["generated_eos_tokens"] == 4
    assert metrics["truncated_rollouts"] == 0
    assert metrics["informative_scoring_input_slots"] == 0
    slots = metrics["model_forward_token_slots"]
    assert slots["phases"]["rollout_prefill"] > 0
    assert slots["phases"]["rollout_cached_decode"] == 0
    assert slots["phases"]["old_policy_scoring"] == 0
    assert slots["phases"]["reference_policy_scoring"] == 0
    assert slots["phases"]["current_policy_optimization"] == 0
    assert slots["total"] == slots["phases"]["rollout_prefill"]
    observation = metrics["rollout_observability"]
    assert observation["reward"] == {
        "distribution": [{"reward": 0.0, "reward_hex": "0x0.0p+0", "count": 4}],
        "unique_values": 1,
        "exact_success_rollouts": 0,
    }
    assert observation["parsing"]["parser_format_valid_rollouts"] == 4
    assert observation["parsing"]["complete_parser_format_valid_rollouts"] == 4
    assert observation["parsing"]["text_reward_rollouts"] == 4
    assert observation["truncation"]["truncated_rollouts"] == 0
    assert observation["tokens"]["generated_tokens"] == 4


def test_grpo_counts_informative_groups_and_realized_policy_epochs(monkeypatch) -> None:
    generations = iter(
        [
            [ord("a"), ByteTokenizer.eos_id],
            [ord("b"), ByteTokenizer.eos_id],
        ]
    )
    monkeypatch.setattr(
        "openlocalagent.train.rl._rollout",
        lambda *_args, **_kwargs: next(generations),
    )

    _, metrics = grpo(
        _TrainablePolicy(),
        [_text_sample("Say a", "a")],
        ByteTokenizer(),
        steps=1,
        prompts_per_step=1,
        group_size=2,
        max_new=4,
        kl_beta=0.0,
        policy_epochs=2,
        return_metrics=True,
        log=lambda *_: None,
    )

    assert metrics["zero_signal_steps"] == 0
    assert metrics["informative_groups"] == 1
    assert metrics["realized_optimizer_updates"] == 2
    observation = metrics["rollout_observability"]
    assert observation["reward"]["unique_values"] == 2
    assert observation["reward"]["exact_success_rollouts"] == 1
    assert sum(row["count"] for row in observation["reward"]["distribution"]) == 2
    assert observation["parsing"]["parser_format_valid_rollouts"] == 2
    assert observation["tokens"]["generated_tokens"] == 4


def test_grpo_bounded_prefix_preserves_full_schedule_and_reaches_real_update(
    monkeypatch,
) -> None:
    generations = iter(
        [
            [ord("a"), ByteTokenizer.eos_id],
            [ord("b"), ByteTokenizer.eos_id],
            [ord("a"), ByteTokenizer.eos_id],
            [ord("b"), ByteTokenizer.eos_id],
        ]
    )
    monkeypatch.setattr(
        "openlocalagent.train.rl._rollout",
        lambda *_args, **_kwargs: next(generations),
    )
    model = _TrainablePolicy()
    initial = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    history, metrics = grpo(
        model,
        [_text_sample("Say a", "a")],
        ByteTokenizer(),
        steps=3,
        execution_rollout_step_limit=2,
        warmup_steps=1,
        lr=1.0e-3,
        prompts_per_step=1,
        group_size=2,
        max_new=4,
        kl_beta=0.0,
        return_metrics=True,
        log=lambda *_: None,
    )

    assert len(history) == 2
    assert metrics["learning_rate_history"] == [0.0, 1.0e-3]
    assert metrics["fixed_horizon_progress"] == {
        "planned_rollout_steps": 3,
        "completed_rollout_steps": 2,
        "execution_rollout_step_limit": 2,
        "bounded_prefix": True,
    }
    assert metrics["realized_optimizer_updates"] == 2
    assert any(
        not torch.equal(initial[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )


def test_grpo_default_resume_preserves_completed_horizon_noop(monkeypatch, tmp_path) -> None:
    generations = iter(
        [
            [ord("a"), ByteTokenizer.eos_id],
            [ord("b"), ByteTokenizer.eos_id],
        ]
    )
    monkeypatch.setattr(
        "openlocalagent.train.rl._rollout",
        lambda *_args, **_kwargs: next(generations),
    )
    model = _TrainablePolicy()
    initial = {
        name: parameter.detach().clone()
        for name, parameter in model.state_dict().items()
    }
    checkpoint_path = tmp_path / "completed.pt"
    expected_history, expected_metrics = grpo(
        model,
        [_text_sample("Say a", "a")],
        ByteTokenizer(),
        steps=1,
        prompts_per_step=1,
        group_size=2,
        max_new=4,
        kl_beta=0.0,
        checkpoint_path=checkpoint_path,
        return_metrics=True,
        log=lambda *_: None,
    )
    resumed = _TrainablePolicy()
    resumed.load_state_dict(initial)
    monkeypatch.setattr(
        "openlocalagent.train.rl._rollout",
        lambda *_args, **_kwargs: pytest.fail(
            "completed-horizon resume must not sample another rollout"
        ),
    )

    actual_history, actual_metrics = grpo(
        resumed,
        [_text_sample("Say a", "a")],
        ByteTokenizer(),
        steps=1,
        prompts_per_step=1,
        group_size=2,
        max_new=4,
        kl_beta=0.0,
        checkpoint_path=checkpoint_path,
        resume_from=checkpoint_path,
        return_metrics=True,
        log=lambda *_: None,
    )

    assert actual_history == expected_history
    assert actual_metrics == expected_metrics
    assert all(
        torch.equal(model.state_dict()[name], resumed.state_dict()[name])
        for name in model.state_dict()
    )


def test_format_reward_validates_registry_and_argument_schema() -> None:
    sample = Sample(
        category="search",
        group="web_search",
        prompt="Search for Seoul",
        kind="tool",
        target='{"arguments":{"query":"Seoul"},"name":"web_search"}',
        ref_name="web_search",
        ref_args='{"query":"Seoul"}',
    )
    registry = [
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
    exact = f"<tool_call>{sample.target}</tool_call>"
    schema_valid_but_wrong = (
        '<tool_call>{"name":"web_search","arguments":{"query":"Busan"}}</tool_call>'
    )
    unknown = '<tool_call>{"name":"not_registered","arguments":{}}</tool_call>'
    invalid_args = '<tool_call>{"name":"web_search","arguments":{"query":7}}</tool_call>'

    assert (
        _rollout_reward(
            sample,
            exact,
            format_weight=0.1,
            truncated=False,
            tool_specs=registry,
        )
        == 1.1
    )
    assert (
        _rollout_reward(
            sample,
            schema_valid_but_wrong,
            format_weight=0.1,
            truncated=False,
            tool_specs=registry,
        )
        == 0.1
    )
    for invalid in (unknown, invalid_args):
        assert (
            _rollout_reward(
                sample,
                invalid,
                format_weight=0.1,
                truncated=False,
                tool_specs=registry,
            )
            == 0.0
        )

    # Legacy rows without embedded schemas retain parse-only shaping.
    assert (
        _rollout_reward(
            sample,
            unknown,
            format_weight=0.1,
            truncated=False,
        )
        == 0.1
    )


def test_format_reward_never_turns_honest_text_or_abstention_into_a_tool_reward() -> None:
    sample = _text_sample("Do not use a tool", "No action needed.")
    registry = [
        ToolSpec(
            name="web_search",
            description="Search the web.",
            parameters={"type": "object", "properties": {}, "required": []},
        )
    ]

    assert (
        _rollout_reward(
            sample,
            "No action needed.",
            format_weight=0.1,
            truncated=False,
            tool_specs=registry,
        )
        == 1.0
    )
    assert (
        _rollout_reward(
            sample,
            '<tool_call>{"name":"web_search","arguments":{}}</tool_call>',
            format_weight=0.1,
            truncated=False,
            tool_specs=registry,
        )
        == 0.0
    )


def test_stateful_productivity_reward_shapes_strict_envelope_and_exact_transition() -> None:
    sample = Sample(
        category="stateful_productivity",
        group="email",
        prompt="Send the message",
        kind="tool",
        target='{"arguments":{"body":"Build is green","subject":"Weekly rollout","to":"maya@example.com"},"name":"email_send"}',
        ref_name="email_send",
        ref_args='{"body":"Build is green","subject":"Weekly rollout","to":"maya@example.com"}',
    )
    spec = ToolSpec(
        name="email_send",
        description="Send an email.",
        parameters={
            "type": "object",
            "properties": {
                "body": {"type": "string"},
                "subject": {"type": "string"},
                "to": {"type": "string"},
            },
            "required": ["body", "subject", "to"],
            "additionalProperties": False,
        },
    )
    exact = (
        '<tool_call>{"name":"email_send","arguments":{"body":"Build is green",'
        '"subject":"Weekly rollout","to":"maya@example.com"}}</tool_call>'
    )
    unknown = '<tool_call>{"name":"unknown","arguments":{}}</tool_call>'
    assert _stateful_productivity_reward(sample, exact, [spec]) == pytest.approx(1.0)
    assert _stateful_productivity_reward(sample, unknown, [spec]) == pytest.approx(0.1)
    assert _stateful_productivity_reward(sample, "malformed", [spec]) == 0.0


def test_token_logprobs_exclude_prompt_and_include_sampled_eos() -> None:
    torch.manual_seed(3)
    model = _TrainablePolicy()
    prompt = [11, 12, 13]
    generation = [14, ByteTokenizer.eos_id]

    actual = _token_logprobs(
        model,
        prompt,
        generation,
        "cpu",
        amp_dtype=torch.float32,
        temperature=2.0,
    )

    full = torch.tensor([prompt + generation])
    logits, _ = model(full[:, :-1])
    expected_all = torch.log_softmax(logits[0] / 2.0, dim=-1)
    targets = full[0, 1:]
    expected = expected_all[torch.arange(targets.numel()), targets][-len(generation) :]
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2,)


def test_token_logprobs_guard_empty_prompt_and_generation() -> None:
    model = _TrainablePolicy()

    empty = _token_logprobs(model, [1], [], "cpu")

    assert empty.shape == (0,)
    assert empty.dtype == torch.float32
    with pytest.raises(ValueError, match="prompt_ids"):
        _token_logprobs(model, [], [ByteTokenizer.eos_id], "cpu")


def test_token_logprobs_cpu_autocast_returns_float32_with_finite_gradients() -> None:
    torch.manual_seed(4)
    model = _TrainablePolicy()

    logprobs = _token_logprobs(
        model,
        [1, 2],
        [3, ByteTokenizer.eos_id],
        "cpu",
        amp_dtype=torch.bfloat16,
    )
    (-logprobs.mean()).backward()

    assert logprobs.dtype == torch.float32
    assert torch.isfinite(logprobs).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_grpo_clips_each_sampled_token_not_a_geometric_mean_sequence_ratio() -> None:
    # The two token ratios are 2.0 and 0.5. Their geometric mean is 1.0, which would hide both
    # changes if log-ratios were averaged before exponentiation.
    current = torch.tensor([math.log(2.0), math.log(0.5)], requires_grad=True)
    old = torch.zeros(2, requires_grad=True)
    reference = current.detach().clone().requires_grad_(True)

    loss = _grpo_token_loss(
        current,
        old,
        reference,
        advantage=1.0,
        clip_ratio=0.2,
        kl_beta=0.0,
    )
    loss.backward()

    # Positive-advantage token surrogates are min(2.0, 1.2) and min(0.5, 0.8).
    assert loss.item() == pytest.approx(-(1.2 + 0.5) / 2)
    assert current.grad is not None and torch.isfinite(current.grad).all()
    assert old.grad is None
    assert reference.grad is None


def test_grpo_sampled_token_k3_kl_is_nonnegative_and_detaches_reference() -> None:
    current = torch.tensor([0.0, 0.0], requires_grad=True)
    old = torch.zeros(2, requires_grad=True)
    reference = torch.tensor([math.log(2.0), math.log(0.5)], requires_grad=True)
    beta = 0.3

    loss = _grpo_token_loss(
        current,
        old,
        reference,
        advantage=0.0,
        clip_ratio=0.2,
        kl_beta=beta,
    )
    expected_k3 = ((2.0 - 1.0 - math.log(2.0)) + (0.5 - 1.0 - math.log(0.5))) / 2
    loss.backward()

    assert loss.item() == pytest.approx(beta * expected_k3)
    assert loss.item() >= 0
    assert current.grad is not None and torch.isfinite(current.grad).all()
    assert old.grad is None
    assert reference.grad is None


def test_full_rl_projects_every_assistant_decision_and_matches_shared_prompt_contract() -> None:
    conversation = _trajectory()
    cache = CatalogStringCache()

    decisions = project_rl_decisions([conversation], sources=["trajectory.jsonl"])

    assert [decision.message_index for decision in decisions] == [2, 5]
    assert all(decision.conversation is conversation for decision in decisions)
    assert all(decision.source == "trajectory.jsonl" for decision in decisions)
    assert [decision.conversation_index for decision in decisions] == [0, 0]
    assert [assistant_body(decision.reward) for decision in decisions] == [
        decision.turn.body for decision in decisions
    ]
    assert all(decision.reward.prompt == "" for decision in decisions)
    with pytest.raises(FrozenInstanceError):
        decisions[0].message_index = 3

    for decision in decisions:
        expected = render_agent_decode_prompt(
            conversation.messages[: decision.message_index],
            conversation.tools,
        )
        assert _decision_prompt_text(decision, cache) == expected
        assert decision.turn.body not in expected
    assert cache.unique_catalogs == 1
    assert BPE_EOS in _decision_prompt_text(decisions[1], cache)
    assert decisions[0].turn.body + BPE_EOS in _decision_prompt_text(decisions[1], cache)


def test_full_rl_catalog_cache_never_hides_mutable_schema_drift() -> None:
    conversation = _trajectory()
    decision = project_rl_decisions([conversation], sources=["trajectory.jsonl"])[0]
    cache = CatalogStringCache()
    _decision_prompt_text(decision, cache)

    conversation.tools[0].description = "mutated " + ASSISTANT

    with pytest.raises(ValueError, match="reserved prompt marker"):
        _decision_prompt_text(decision, cache)


def test_full_rl_prompt_hash_excludes_current_gold_but_includes_contract_context() -> None:
    train_conversation = _text_conversation(
        "Shared request",
        "train gold",
        source="train",
    )
    eval_conversation = _text_conversation(
        "Shared request",
        "eval gold",
        source="eval",
    )
    train = project_rl_decisions(
        [train_conversation],
        sources=["train.jsonl"],
    )
    eval_same_prompt = project_rl_decisions(
        [eval_conversation],
        sources=["eval.jsonl"],
    )

    assert _decision_prompt_text(train[0]) == _decision_prompt_text(eval_same_prompt[0])
    assert _decision_fingerprint(train[0]) != _decision_fingerprint(eval_same_prompt[0])
    with pytest.raises(ValueError, match="exact openai_full_catalog_v1 prompt fingerprint"):
        _audit_data_splits(
            [train_conversation],
            [eval_conversation],
            train,
            eval_same_prompt,
            conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
        )

    different_system_conversation = Conversation(
        messages=[
            Message(role=Role.system, content="A distinct system contract."),
            Message(role=Role.user, content="Shared request"),
            Message(role=Role.assistant, content="eval gold"),
        ]
    )
    different_system = project_rl_decisions(
        [different_system_conversation],
        sources=["eval.jsonl"],
    )
    audit = _audit_data_splits(
        [train_conversation],
        [different_system_conversation],
        train,
        different_system,
        conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
    )
    assert audit["prompt_overlap"] == 0
    assert audit["current_gold_in_prompt_fingerprint"] is False
    assert audit["train_scored_rows"] == audit["eval_scored_rows"] == 1


def test_full_rl_preflight_never_truncates_while_legacy_keeps_tail_slice(
    rl_bpe_tokenizer,
) -> None:
    decision = project_rl_decisions([_trajectory()], sources=["train.jsonl"])[-1]
    prompt_tokens = len(rl_bpe_tokenizer.encode(_decision_prompt_text(decision)))
    max_new = 4

    with pytest.raises(ValueError, match="cannot be truncated"):
        _preflight_full_context(
            [decision],
            rl_bpe_tokenizer,
            max_new=max_new,
            max_seq_len=prompt_tokens + max_new - 1,
            split="training split",
        )

    context = _preflight_full_context(
        [decision],
        rl_bpe_tokenizer,
        max_new=max_new,
        max_seq_len=prompt_tokens + max_new,
        split="training split",
    )
    assert context["rows"] == 1
    assert context["min_prompt_tokens"] == context["max_prompt_tokens"] == prompt_tokens
    assert context["min_available_decode_reserve_tokens"] == max_new
    assert context["truncation"] == "forbidden"
    assert context["truncated_rows"] == 0

    tokenizer = ByteTokenizer()
    sample = _text_sample("a deliberately long legacy prompt", "ready")
    expected = tokenizer.encode(prompt_text(sample))[-7:]
    assert (
        _prompt_ids_for_policy(
            sample,
            tokenizer,
            max_prompt=7,
            conversation_prompt_contract=LEGACY_CONVERSATION_PROMPT_CONTRACT,
            catalog_cache=None,
        )
        == expected
    )


def test_full_rl_requires_atomic_bpe_eos_contract() -> None:
    decision = project_rl_decisions([_trajectory()], sources=["train.jsonl"])[0]

    with pytest.raises(ValueError, match="requires a BPE tokenizer"):
        _preflight_full_context(
            [decision],
            ByteTokenizer(),
            max_new=4,
            max_seq_len=4096,
            split="training split",
        )


def test_full_rl_format_reward_uses_recursive_shared_schema_validation() -> None:
    tool = _nested_tool()
    valid = (
        '<tool_call>{"name":"inspect_telemetry","arguments":'
        '{"satellite":"Asteria","window":{"samples":[1,2.5]}}}</tool_call>'
    )
    nested_extra = (
        '<tool_call>{"name":"inspect_telemetry","arguments":'
        '{"satellite":"Asteria","window":{"samples":[1],"unexpected":true}}}</tool_call>'
    )
    root_extra = (
        '<tool_call>{"name":"inspect_telemetry","arguments":'
        '{"satellite":"Asteria","window":{"samples":[1]},"unexpected":true}}</tool_call>'
    )

    assert _valid_tool_call_format(
        valid,
        [tool],
        conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
    )
    assert not _valid_tool_call_format(
        nested_extra,
        [tool],
        conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
    )
    assert not _valid_tool_call_format(
        root_extra,
        [tool],
        conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
    )


def test_full_rl_strict_output_gate_rejects_permissive_tool_call_adversaries() -> None:
    tool = _nested_tool()
    sample = Sample(
        category="telemetry",
        group="tool_call",
        prompt="",
        kind="tool",
        target=(
            '{"arguments":{"satellite":"Asteria","window":{"samples":[1]}},'
            '"name":"inspect_telemetry"}'
        ),
        ref_name="inspect_telemetry",
        ref_args='{"satellite":"Asteria","window":{"samples":[1]}}',
    )
    canonical = f"<tool_call>{sample.target}</tool_call>"
    adversaries = [
        "outside " + canonical,
        (
            '<tool_call>{"name":"wrong","name":"inspect_telemetry","arguments":'
            '{"satellite":"Asteria","window":{"samples":[1]}}}</tool_call>'
        ),
        (
            '<tool_call>{"name":"inspect_telemetry","arguments":'
            '{"satellite":"Asteria","window":{"samples":[1]}},"extra":true}</tool_call>'
        ),
        canonical + "</tool_call>",
        canonical.replace("</tool_call>", ""),
    ]

    assert _correct_for_contract(
        sample,
        canonical,
        conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
    )
    assert _valid_tool_call_format(
        canonical,
        [tool],
        conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
    )
    assert (
        _rollout_reward(
            sample,
            canonical,
            format_weight=0.1,
            truncated=False,
            tool_specs=[tool],
            conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
        )
        == 1.1
    )
    for generated in adversaries:
        assert not _correct_for_contract(
            sample,
            generated,
            conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
        )
        assert not _valid_tool_call_format(
            generated,
            [tool],
            conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
        )
        assert (
            _rollout_reward(
                sample,
                generated,
                format_weight=0.1,
                truncated=False,
                tool_specs=[tool],
                conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
            )
            == 0.0
        )

    # The legacy parser deliberately keeps its historical permissive outside-text behavior.
    assert _correct_for_contract(
        sample,
        adversaries[0],
        conversation_prompt_contract=LEGACY_CONVERSATION_PROMPT_CONTRACT,
    )
    assert _valid_tool_call_format(
        adversaries[0],
        [tool],
        conversation_prompt_contract=LEGACY_CONVERSATION_PROMPT_CONTRACT,
    )


def test_full_rl_text_exactness_does_not_strip_generated_output() -> None:
    sample = _text_sample("Answer exactly", "ready")

    assert _correct_for_contract(
        sample,
        "ready",
        conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
    )
    assert not _correct_for_contract(
        sample,
        " ready ",
        conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
    )
    assert _correct_for_contract(
        sample,
        " ready ",
        conversation_prompt_contract=LEGACY_CONVERSATION_PROMPT_CONTRACT,
    )


def test_full_rl_holdout_exact_reward_and_format_rate_share_strict_gate(
    monkeypatch,
    rl_bpe_tokenizer,
) -> None:
    decision = project_rl_decisions([_trajectory()], sources=["eval.jsonl"])[0]
    canonical = assistant_body(decision.reward)
    outside_text = "outside " + canonical

    monkeypatch.setattr(
        "openlocalagent.train.rl._rollout",
        lambda *_args, **_kwargs: [
            *rl_bpe_tokenizer.encode(outside_text),
            rl_bpe_tokenizer.eos_id,
        ],
    )
    rejected = _evaluate_holdout(
        _EvaluationShell(),
        [decision],
        rl_bpe_tokenizer,
        max_new=512,
        device="cpu",
        format_weight=0.1,
        truncation_penalty=0.05,
        conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
    )
    assert rejected["exact_match_accuracy"] == 0.0
    assert rejected["mean_reward"] == 0.0
    assert rejected["tool_format_valid_rate"] == 0.0

    monkeypatch.setattr(
        "openlocalagent.train.rl._rollout",
        lambda *_args, **_kwargs: [
            *rl_bpe_tokenizer.encode(canonical),
            rl_bpe_tokenizer.eos_id,
        ],
    )
    accepted = _evaluate_holdout(
        _EvaluationShell(),
        [decision],
        rl_bpe_tokenizer,
        max_new=512,
        device="cpu",
        format_weight=0.1,
        truncation_penalty=0.05,
        conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
    )
    assert accepted["exact_match_accuracy"] == 1.0
    assert accepted["mean_reward"] == 1.1
    assert accepted["tool_format_valid_rate"] == 1.0


def test_rl_decision_rederives_turn_and_rejects_stale_history_or_forgery() -> None:
    conversation = _trajectory()
    decision = project_rl_decisions([conversation], sources=["train.jsonl"])[1]
    forged = replace(
        decision,
        turn=replace(decision.turn, prompt_suffix=decision.turn.prompt_suffix + "forged"),
    )

    with pytest.raises(ValueError, match="prompt_suffix/body/index"):
        _decision_prompt_text(forged)

    conversation.messages[1].content = "Mutated prior user history."
    with pytest.raises(ValueError, match="prompt_suffix/body/index"):
        _decision_prompt_text(decision)


def test_full_rl_dataset_hash_uses_shared_unicode_semantics_once_per_raw_row() -> None:
    train_conversation = Conversation(
        messages=[
            Message(role=Role.system, content="Préserve 한글."),
            Message(role=Role.user, content="Résume 서울."),
            Message(role=Role.assistant, content="첫 번째"),
            Message(role=Role.user, content="Encore."),
            Message(role=Role.assistant, content="두 번째"),
        ]
    )
    eval_conversation = _text_conversation("Distinct", "held out", source="eval")
    train_decisions = project_rl_decisions([train_conversation], sources=["train.jsonl"])
    eval_decisions = project_rl_decisions([eval_conversation], sources=["eval.jsonl"])

    audit = _audit_data_splits(
        [train_conversation],
        [eval_conversation],
        train_decisions,
        eval_decisions,
        conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
    )
    semantic = conversation_semantic_sha256(train_conversation)
    assert len(train_decisions) == 2
    assert audit["train_dataset_rows"] == 1
    assert audit["train_scored_rows"] == 2
    assert audit["train_dataset_sha256"] == _fingerprint_set([semantic])
    assert audit["train_dataset_sha256"] != _fingerprint_set([semantic, semantic])


def test_rl_parent_prompt_contract_missing_means_legacy_and_mismatch_fails() -> None:
    assert (
        _assert_parent_prompt_contract({}, LEGACY_CONVERSATION_PROMPT_CONTRACT)
        == LEGACY_CONVERSATION_PROMPT_CONTRACT
    )
    assert (
        _assert_parent_prompt_contract(
            {"conversation_prompt_contract": OPENAI_FULL_CATALOG_V1},
            OPENAI_FULL_CATALOG_V1,
        )
        == OPENAI_FULL_CATALOG_V1
    )
    with pytest.raises(ValueError, match="conversation_prompt_contract mismatch"):
        _assert_parent_prompt_contract({}, OPENAI_FULL_CATALOG_V1)
    with pytest.raises(ValueError, match="conversation_prompt_contract mismatch"):
        _assert_parent_prompt_contract(
            {"conversation_prompt_contract": OPENAI_FULL_CATALOG_V1},
            LEGACY_CONVERSATION_PROMPT_CONTRACT,
        )


def test_full_rl_gold_output_contract_rejects_embedded_canonical_eos() -> None:
    conversation = _text_conversation(
        "Return a marker safely",
        f"prefix{BPE_EOS}unreachable",
        source="train",
    )
    with pytest.raises(ValueError, match="reserved prompt marker"):
        project_rl_decisions([conversation], sources=["train.jsonl"])


def test_full_rl_runner_persists_prompt_context_schema_and_output_contracts(
    tmp_path: Path,
    rl_bpe_tokenizer,
) -> None:
    train = Conversation(
        messages=[
            Message(role=Role.system, content="Answer one character."),
            Message(role=Role.user, content="First train decision."),
            Message(role=Role.assistant, content="A"),
            Message(role=Role.user, content="Second train decision."),
            Message(role=Role.assistant, content="B"),
        ]
    )
    heldout = Conversation(
        messages=[
            Message(role=Role.system, content="Answer one character."),
            Message(role=Role.user, content="Held-out decision."),
            Message(role=Role.assistant, content="C"),
        ]
    )
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    train_path.write_text(train.to_json() + "\n", encoding="utf-8")
    eval_path.write_text(heldout.to_json() + "\n", encoding="utf-8")
    tokenizer_path = tmp_path / "tokenizer.json"
    rl_bpe_tokenizer.save(tokenizer_path)
    cfg = ModelConfig(
        name="rl-full-contract-test",
        vocab_size=rl_bpe_tokenizer.vocab_size,
        d_model=16,
        embed_dim=8,
        n_layers=1,
        n_loops=1,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden=32,
        max_seq_len=512,
        dropout=0.0,
    )
    cfg.assert_within_budget()
    model_config_path = tmp_path / "model.yaml"
    model_config_path.write_text(
        yaml.safe_dump(cfg.__dict__, sort_keys=False),
        encoding="utf-8",
    )
    tokenizer_lineage = tokenizer_identity(
        "bpe",
        vocab_size=rl_bpe_tokenizer.vocab_size,
        path=tokenizer_path,
    )
    parent_path = tmp_path / "sft.pt"
    torch.save(
        {
            "stage": "sft",
            "lineage": {
                "version": 1,
                "stage": "sft",
                "tokenizer_sha256": tokenizer_lineage["sha256"],
            },
            "cfg": cfg.__dict__,
            "state_dict": LocalAgentLM(cfg).state_dict(),
            "tokenizer": {
                "kind": "bpe",
                "path": str(tokenizer_path),
                "sha256": tokenizer_lineage["sha256"],
            },
            "conversation_prompt_contract": OPENAI_FULL_CATALOG_V1,
        },
        parent_path,
    )
    out_dir = tmp_path / "rl"
    config_path = tmp_path / "rl.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "stage": "rl",
                "model_config": str(model_config_path),
                "init_from": str(parent_path),
                "data": {
                    "conversation_prompt_contract": OPENAI_FULL_CATALOG_V1,
                    "conversations": [str(train_path)],
                    "eval_conversations": [str(eval_path)],
                    "tokenizer": {"kind": "bpe", "path": str(tokenizer_path)},
                },
                "rollout": {
                    "prompts_per_step": 1,
                    "group_size": 2,
                    "max_new_tokens": 4,
                    "temperature": 1.0,
                },
                "policy": {
                    "clip_ratio": 0.2,
                    "kl_beta": 0.0,
                    "epochs_per_rollout": 1,
                },
                "schedule": {"total_steps": 1},
                "runtime": {"device": "cpu", "dtype": "fp32", "seed": 17},
                "log": {"out_dir": str(out_dir)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    production_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_sha256 = canonical_sha256(production_config)
    parent_sha256 = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    expected_execution = {
        "requested_device": "cpu",
        "resolved_device": "cpu",
        "requested_dtype": "fp32",
        "resolved_dtype": "fp32",
    }
    with pytest.raises(ValueError, match="config canonical SHA-256"):
        run(
            str(config_path),
            _expected_config_canonical_sha256="0" * 64,
            _require_fresh_output_dir=True,
        )
    assert not out_dir.exists()
    with pytest.raises(ValueError, match="execution identity"):
        run(
            str(config_path),
            _expected_config_canonical_sha256=config_sha256,
            _expected_execution={
                **expected_execution,
                "resolved_device": "mps",
            },
            _require_fresh_output_dir=True,
        )
    assert not out_dir.exists()
    with pytest.raises(ValueError, match="parent checkpoint SHA-256"):
        run(
            str(config_path),
            _expected_config_canonical_sha256=config_sha256,
            _expected_parent_checkpoint_sha256="0" * 64,
            _expected_execution=expected_execution,
            _require_fresh_output_dir=True,
        )
    assert not out_dir.exists()

    run(
        str(config_path),
        _expected_config_canonical_sha256=config_sha256,
        _expected_parent_checkpoint_sha256=parent_sha256,
        _expected_execution=expected_execution,
        _require_fresh_output_dir=True,
    )

    checkpoint = torch.load(out_dir / "latest.pt", map_location="cpu", weights_only=False)
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert checkpoint["conversation_prompt_contract"] == OPENAI_FULL_CATALOG_V1
    assert metrics["conversation_prompt_contract"] == OPENAI_FULL_CATALOG_V1
    assert checkpoint["assistant_decision_rows"] == 2
    assert checkpoint["data"]["single_turn_rows"] == 0
    assert checkpoint["data"]["assistant_decision_rows"] == 2
    assert checkpoint["data"]["eval_assistant_decision_rows"] == 1
    assert checkpoint["context_preflight"]["train"]["rows"] == 2
    assert checkpoint["context_preflight"]["eval"]["rows"] == 1
    assert checkpoint["context_preflight"]["train"]["truncation"] == "forbidden"
    assert checkpoint["reward_contract"]["schema_additional_properties"] == ("enforced recursively")
    assert checkpoint["reward_contract"]["gold_output_contract"].endswith("terminal EOS")
    assert checkpoint["policy_contract"]["prompt_materialization"] == (
        "lazy per selected assistant decision"
    )
    assert checkpoint["heldout_eval"]["contract"]["current_gold_in_prompt"] is False
    assert checkpoint["lineage"]["stage"] == "rl"
    assert metrics["data"] == checkpoint["data"]
    assert metrics["reward_contract"] == checkpoint["reward_contract"]
    assert metrics["policy_contract"] == checkpoint["policy_contract"]
    assert metrics["heldout_eval"] == checkpoint["heldout_eval"]

    checkpoint_bytes = (out_dir / "latest.pt").read_bytes()
    with pytest.raises(FileExistsError):
        run(
            str(config_path),
            _expected_config_canonical_sha256=config_sha256,
            _expected_parent_checkpoint_sha256=parent_sha256,
            _expected_execution=expected_execution,
            _require_fresh_output_dir=True,
        )
    assert (out_dir / "latest.pt").read_bytes() == checkpoint_bytes
