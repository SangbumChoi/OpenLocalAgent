import hashlib
import json
from pathlib import Path

import pytest
import yaml

from openlocalagent.agent.schema_decode import fill_tool
from openlocalagent.agent.toolset import STANDARD_TOOLS
from openlocalagent.data.synth import agent_synth
from openlocalagent.data.synth.agent_synth import Generator, synthesize
from openlocalagent.data.synth.agent_synth_paper_v2 import (
    PAPER_TRAIN_V2_MODE,
    PAPER_TRAIN_V2_MODE_VERSION,
    build_paper_train_v2_tools,
)
from openlocalagent.data.conversation_artifact import (
    canonical_json_bytes,
    load_verified_conversation_artifact,
)
from openlocalagent.data.schema import Conversation


def _bounded_v2_config(out: Path) -> dict:
    return {
        "out": str(out),
        "n_samples": 240,
        "seed": 2026,
        "level": 5,
        "split": "train",
        "generator": {
            "backend": "deterministic_templates",
            "mode": PAPER_TRAIN_V2_MODE,
            "mode_version": PAPER_TRAIN_V2_MODE_VERSION,
        },
        "complexity": {"multi_turn": 0.25},
        "irrelevance_fraction": 0.20,
        "coverage": {
            "minimum_conversations": {
                "parallel_calls": 4,
                "integer_arguments": 3,
                "enum_arguments": 3,
                "boolean_arguments": 12,
                "number_arguments": 12,
                "tool_response_grounded_followups": 5,
                "verified_error_recovery": 5,
                "paper_v2_schema_trajectories": 5,
            },
            "plan_length_minimums": {0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
        },
        "verification": {"rule_based": True, "model_based": False},
    }


def test_paper_v2_overlay_preserves_standard_tool_names_order_and_registry():
    original_names = [tool.name for tool in STANDARD_TOOLS]
    tools = build_paper_train_v2_tools(STANDARD_TOOLS)
    assert [tool.name for tool in tools] == original_names
    assert len(tools) == len(STANDARD_TOOLS) == 50

    original_scroll = next(tool for tool in STANDARD_TOOLS if tool.name == "scroll")
    v2_scroll = next(tool for tool in tools if tool.name == "scroll")
    assert set(original_scroll.parameters["properties"]) == {"direction"}
    assert v2_scroll.parameters["required"] == ["direction"]
    assert v2_scroll.parameters["properties"]["amount"]["type"] == "number"
    assert v2_scroll.parameters["properties"]["smooth"]["type"] == "boolean"


def test_paper_v2_generator_is_train_only_and_legacy_default_is_unchanged():
    implicit = Generator(level=5, seed=19, split="train")
    explicit = Generator(level=5, seed=19, split="train", mode=Generator._LEGACY_MODE)
    assert implicit.generate(300) == explicit.generate(300)
    assert "precise_scroll_v2" not in {maker.__name__ for maker in implicit.makers()}

    v2 = Generator(level=5, seed=19, split="train", mode=PAPER_TRAIN_V2_MODE)
    assert "precise_scroll_v2" in {maker.__name__ for maker in v2.makers()}
    sample = v2.precise_scroll_v2()
    assert sample.target == json.dumps(
        {"name": "scroll", "arguments": json.loads(sample.ref_args)},
        separators=(",", ":"),
        sort_keys=True,
    )
    args = json.loads(sample.ref_args)
    assert isinstance(args["amount"], float)
    assert isinstance(args["smooth"], bool)
    v2_scroll = next(
        tool
        for tool in build_paper_train_v2_tools(STANDARD_TOOLS)
        if tool.name == "scroll"
    )
    assert fill_tool(sample.prompt, v2_scroll) == args

    with pytest.raises(ValueError, match="train-only"):
        Generator(level=5, seed=19, split="eval", mode=PAPER_TRAIN_V2_MODE)


def test_paper_v2_slot_audit_fails_closed_on_frozen_eval_collision(monkeypatch):
    audit = agent_synth._paper_train_v2_slot_audit()
    assert audit["overlap"] == 0
    assert audit["paired_legacy_train_eval_pools"] >= 30
    monkeypatch.setattr(
        agent_synth,
        "PAPER_TRAIN_V2_SLOT_POOLS",
        {"deliberate_collision": (agent_synth.CITIES_EVAL[0],)},
    )
    with pytest.raises(ValueError, match="collides with frozen eval slots"):
        agent_synth._paper_train_v2_slot_audit()


def test_bounded_paper_v2_export_is_deterministic_canonical_and_covered(tmp_path, capsys):
    out = tmp_path / "paper-v2.jsonl"
    config = _bounded_v2_config(out)
    config_path = tmp_path / "paper-v2.yaml"
    config_path.write_text(yaml.safe_dump(config))

    synthesize(str(config_path))
    capsys.readouterr()
    first_payload = out.read_bytes()
    first_manifest_payload = out.with_suffix(".jsonl.manifest.json").read_bytes()
    synthesize(str(config_path))
    capsys.readouterr()
    assert out.read_bytes() == first_payload
    assert out.with_suffix(".jsonl.manifest.json").read_bytes() == first_manifest_payload
    verified = load_verified_conversation_artifact(
        out,
        config_path=config_path,
        expected_split="train",
        environment_policy="forbid",
    )
    assert len(verified.conversations) == 240

    raw_lines = first_payload.decode("utf-8").splitlines()
    conversations = [Conversation.from_json(line) for line in raw_lines]
    assert len(raw_lines) == len(set(raw_lines)) == 240
    assert sum(len(conversation.messages) > 2 for conversation in conversations) == 60
    assert sum(
        conversation.meta.get("category") == "no_tool" for conversation in conversations
    ) == 48
    assert all(
        conversation.meta["template_mode"] == PAPER_TRAIN_V2_MODE
        and conversation.meta["template_mode_version"] == PAPER_TRAIN_V2_MODE_VERSION
        and conversation.meta["environment_executed"] is False
        for conversation in conversations
    )
    standard_names = [tool.name for tool in STANDARD_TOOLS]
    assert all(
        [tool.name for tool in conversation.tools] == standard_names
        for conversation in conversations
    )

    scroll_calls = [
        call
        for conversation in conversations
        for message in conversation.messages
        for call in message.tool_calls
        if call.name == "scroll" and "amount" in call.arguments
    ]
    assert len(scroll_calls) >= 29
    assert all(
        isinstance(call.arguments["amount"], float)
        and isinstance(call.arguments["smooth"], bool)
        for call in scroll_calls
    )

    manifest = json.loads(first_manifest_payload)
    assert first_manifest_payload == canonical_json_bytes(manifest)
    unsigned = dict(manifest)
    self_hash = unsigned.pop("manifest_self_sha256")
    assert self_hash == hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    assert manifest["output_sha256"] == hashlib.sha256(first_payload).hexdigest()
    assert manifest["rows"] == 240
    assert manifest["environment_executed"] is False
    assert manifest["argument_schema_coverage"]["absent_primitive_types"] == []
    assert manifest["argument_schema_coverage"]["schema_overlay"]["properties"] == {
        "amount": "number",
        "smooth": "boolean",
    }
    for name, minimum in config["coverage"]["minimum_conversations"].items():
        assert manifest["behavior_counts"][name] >= minimum
    assert manifest["argument_value_counts"]["boolean"] > 0
    assert manifest["argument_value_counts"]["number"] > 0
    preflight = manifest["coverage_contract"]["paper_train_v2_preflight"]
    assert preflight["status"] == "passed"
    assert preflight["requirements"]["irrelevance_conversations"] == 48
    assert manifest["split_contract"]["paper_train_v2"]["frozen_eval_slot_audit"][
        "overlap"
    ] == 0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"split": "eval"}, "train-only"),
        (
            {"generator": {"backend": "deterministic_templates", "mode": PAPER_TRAIN_V2_MODE}},
            "mode_version=2",
        ),
        (
            {
                "generator": {
                    "backend": "deterministic_templates",
                    "mode": PAPER_TRAIN_V2_MODE,
                    "mode_version": 3,
                }
            },
            "mode_version=2",
        ),
    ],
)
def test_paper_v2_rejects_eval_or_missing_wrong_version(tmp_path, change, message):
    out = tmp_path / "rejected.jsonl"
    config = _bounded_v2_config(out)
    config.update(change)
    config_path = tmp_path / "rejected.yaml"
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match=message):
        synthesize(str(config_path))
    assert not out.exists()


def test_paper_v2_preflight_rejects_infeasible_capacity_before_output(tmp_path):
    out = tmp_path / "too-many.jsonl"
    config = _bounded_v2_config(out)
    config["n_samples"] = 50_000
    config["coverage"]["minimum_conversations"]["boolean_arguments"] = 17_281
    config["coverage"]["minimum_conversations"]["number_arguments"] = 1
    config_path = tmp_path / "too-many.yaml"
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="precise_scroll_single_turn"):
        synthesize(str(config_path))
    assert not out.exists()


def test_paper_v2_paper_config_freezes_50k_train_only_strata():
    root = Path(__file__).parents[1]
    config = yaml.safe_load(
        (root / "configs/data/agent_synth_paper_train_v2.yaml").read_text()
    )
    assert config["out"] == "data/synth/agent_sft_paper_train_v2.jsonl"
    assert config["n_samples"] == 50_000
    assert config["split"] == "train"
    assert config["generator"] == {
        "backend": "deterministic_templates",
        "mode": PAPER_TRAIN_V2_MODE,
        "mode_version": PAPER_TRAIN_V2_MODE_VERSION,
    }
    assert config["complexity"]["multi_turn"] == 0.25
    assert config["irrelevance_fraction"] == 0.20
    assert config["coverage"]["minimum_conversations"]["boolean_arguments"] == 2500
    assert config["coverage"]["minimum_conversations"]["number_arguments"] == 2500
    assert {entry["name"] for entry in config["exact_prompt_holdouts"]} == {
        "local-browser-tasks",
        "local-realtime-actions",
    }
    generator = Generator(
        level=config["level"],
        seed=config["seed"],
        split=config["split"],
        mode=config["generator"]["mode"],
    )
    preflight = agent_synth._paper_train_v2_preflight(
        generator,
        round(config["n_samples"] * config["irrelevance_fraction"]),
        config["coverage"]["minimum_conversations"],
        config["coverage"]["plan_length_minimums"],
    )
    assert preflight["status"] == "passed"
    assert preflight["requirements"]["paper_v2_precise_scroll_single_turn"] == 5000
