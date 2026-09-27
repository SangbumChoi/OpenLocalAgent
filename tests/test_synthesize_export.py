import hashlib
import json
from pathlib import Path

import yaml

from openlocalagent.data.synth import agent_synth
from openlocalagent.data.synth.agent_synth import synthesize
from openlocalagent.data.conversation_artifact import (
    CONVERSATION_SERIALIZATION,
    MANIFEST_KIND,
    MANIFEST_SCHEMA_VERSION,
    canonical_json_bytes,
)
from openlocalagent.data.render import render_conversation
from openlocalagent.data.schema import Conversation, Role
from openlocalagent.model.tokenizer import ByteTokenizer


def test_holdout_prompt_canonicalization_covers_case_whitespace_and_unicode():
    normalize = agent_synth._canonical_holdout_prompt
    assert normalize("  CAFÉ\u00a0Status\tNOW  ") == normalize("cafe\u0301 status now")
    assert normalize("Ｆｕｌｌ－Ｗｉｄｔｈ") == normalize("full-width")


def test_train_eval_synth_configs_share_generation_contract():
    root = Path(__file__).parents[1]
    train = yaml.safe_load((root / "configs/data/agent_synth.yaml").read_text())
    evaluation = yaml.safe_load((root / "configs/data/agent_synth_eval.yaml").read_text())
    assert train["split"] == "train"
    assert evaluation["split"] == "eval"
    assert train["out"] != evaluation["out"]
    holdouts = train.pop("exact_prompt_holdouts")
    assert {entry["name"] for entry in holdouts} == {
        "local-browser-tasks",
        "local-realtime-actions",
    }
    for config in (train, evaluation):
        for key in ("out", "seed", "split"):
            config.pop(key)
    assert train == evaluation


def test_all_declared_train_eval_slot_pools_are_disjoint():
    checked = 0
    for name, train_values in vars(agent_synth).items():
        if not name.endswith("_TRAIN"):
            continue
        eval_name = f"{name[:-6]}_EVAL"
        if not hasattr(agent_synth, eval_name):
            continue
        eval_values = getattr(agent_synth, eval_name)
        assert set(train_values).isdisjoint(eval_values), name
        checked += 1
    assert checked >= 30


def test_synthesize_exports_rule_audited_canonical_conversations(tmp_path):
    out = tmp_path / "agent.jsonl"
    config = {
        "out": str(out),
        "n_samples": 40,
        "seed": 7,
        "level": 5,
        "split": "train",
        "generator": {"backend": "deterministic_templates"},
        "complexity": {"multi_turn": 0.2},
        "irrelevance_fraction": 0.15,
        "coverage": {
            "minimum_conversations": {
                "parallel_calls": 2,
                "integer_arguments": 2,
                "enum_arguments": 2,
                "tool_response_grounded_followups": 1,
                "verified_error_recovery": 1,
            },
            "plan_length_minimums": {0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
        },
        "verification": {"rule_based": True, "model_based": False},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    synthesize(str(path))
    conversations = [Conversation.from_json(line) for line in out.read_text().splitlines() if line]
    assert len(conversations) == 40
    assert len(set(out.read_text().splitlines())) == 40
    assert sum(conv.meta.get("category") == "no_tool" for conv in conversations) == 6
    assert sum(len(conv.messages) > 2 for conv in conversations) == 8
    assert all(conv.tools for conv in conversations)
    assert all(conv.meta["rule_verified"] is True for conv in conversations)
    assert all(conv.meta["verified"] is False for conv in conversations)
    assert all(conv.meta["environment_executed"] is False for conv in conversations)
    manifest_path = tmp_path / "agent.jsonl.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest_path.read_bytes() == canonical_json_bytes(manifest)
    assert manifest["kind"] == MANIFEST_KIND
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["conversation_serialization"] == CONVERSATION_SERIALIZATION
    assert manifest["generator_config"] == {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    unsigned_manifest = dict(manifest)
    manifest_self_sha256 = unsigned_manifest.pop("manifest_self_sha256")
    assert (
        manifest_self_sha256 == hashlib.sha256(canonical_json_bytes(unsigned_manifest)).hexdigest()
    )
    assert manifest["rows"] == 40 and manifest["rule_verified"] is True
    assert manifest["model_verified"] is False
    assert manifest["environment_executed"] is False
    assert manifest["verification_claim"] == "rule_audited_not_environment_executed"
    assert manifest["output_bytes"] == out.stat().st_size
    assert manifest["output_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert manifest["rule_verification_scope"] == [
        "canonical_conversation_schema",
        "registered_tool_name",
        "required_argument_presence",
        "primitive_argument_type",
        "argument_enum",
        "planner_metadata_matches_tool_sequence",
    ]
    assert manifest["complexity_contract"] == {
        "enforced": {"multi_turn": 0.2},
        "regular_sampling": "deterministic_maker_registry",
        "quota_sampling": True,
    }
    assert manifest["coverage_contract"] == {
        "minimum_conversations": {
            "enum_arguments": 2,
            "integer_arguments": 2,
            "parallel_calls": 2,
            "tool_response_grounded_followups": 1,
            "verified_error_recovery": 1,
        },
        "plan_length_minimums": {"0": 1, "1": 1, "2": 1, "3": 1, "4": 1},
        "sampling": "deterministic_reserved_batches_then_registry_fill",
        "semantics": "minimums; deterministic fill may increase realized behavior counts",
    }
    assert manifest["split_contract"]["primitive_value_disjointness_claimed"] is False
    assert manifest["split_contract"]["template_disjointness_claimed"] is False
    assert manifest["exact_prompt_holdouts"] == {
        "artifact_count": 0,
        "artifacts": [],
        "match_mode": "canonical normalized equality",
        "match_scope": "all Conversation user-message content fields",
        "normalization": {
            "case": "Unicode casefold",
            "case_sensitive": False,
            "unicode": "NFKC",
            "whitespace": "Unicode split then single ASCII-space join",
        },
        "normalized_prompts_sha256": hashlib.sha256(b"[]").hexdigest(),
        "normalized_unique_prompts": 0,
        "output_matches": 0,
    }
    assert manifest["structural_counts"]["multi_turn_conversations"] == 8
    assert manifest["structural_counts"]["irrelevance_conversations"] == 6
    assert manifest["structural_counts"]["assistant_tool_calls"] > 0
    assert manifest["behavior_counts"]["parallel_calls"] >= 2
    assert manifest["behavior_counts"]["integer_arguments"] >= 2
    assert manifest["behavior_counts"]["enum_arguments"] >= 2
    assert manifest["behavior_counts"]["tool_response_grounded_followups"] >= 1
    assert manifest["behavior_counts"]["verified_error_recovery"] >= 1
    assert manifest["behavior_counts"]["explicit_restraint"] >= 7
    assert manifest["behavior_definitions"]["tool_response_grounded_followups"] == (
        "string argument absent from the initial user turn and copied from a prior tool response"
    )
    assert manifest["behavior_definitions"]["verified_error_recovery"] == (
        "compatibility quota name: rule-audited scripted trace with a FAILED tool response, "
        "remediation call, and later All tests passed response; tool outcomes are template "
        "literals and are not executed"
    )
    assert manifest["argument_schema_coverage"]["absent_primitive_types"] == [
        "boolean",
        "number",
    ]
    assert "does not claim training coverage" in manifest["argument_schema_coverage"]["note"]
    assert all(manifest["plan_length_counts"][str(length)] >= 1 for length in range(5))
    assert manifest["argument_value_counts"]["integer"] >= 2


def test_train_config_excludes_every_tracked_suite_query_and_pins_artifacts(tmp_path, capsys):
    root = Path(__file__).parents[1]
    source_config = yaml.safe_load((root / "configs/data/agent_synth.yaml").read_text())
    source_config["out"] = str(tmp_path / "agent.jsonl")
    source_config["n_samples"] = 200
    source_config["coverage"]["minimum_conversations"] = {
        "parallel_calls": 2,
        "integer_arguments": 2,
        "enum_arguments": 2,
        "tool_response_grounded_followups": 2,
        "verified_error_recovery": 1,
    }
    source_config["coverage"]["plan_length_minimums"] = {
        0: 1,
        1: 1,
        2: 1,
        3: 1,
        4: 1,
    }
    for entry in source_config["exact_prompt_holdouts"]:
        entry["path"] = str((root / "configs" / "data" / entry["path"]).resolve())
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(yaml.safe_dump(source_config))

    synthesize(str(config_path))
    capsys.readouterr()
    conversations = [
        Conversation.from_json(line) for line in (tmp_path / "agent.jsonl").read_text().splitlines()
    ]
    heldout_queries: set[str] = set()
    for filename in ("benchmark-cases.json", "browser-task-cases.json"):
        suite = json.loads((root / "demos" / "webgpu" / filename).read_text())
        assert suite["holdout_contract"]["primitive_value_disjointness_claimed"] is False
        heldout_queries.update(case["query"] for case in suite["cases"])
    generated_user_prompts = {
        message.content
        for conversation in conversations
        for message in conversation.messages
        if message.role == Role.user
    }
    assert heldout_queries.isdisjoint(generated_user_prompts)

    manifest = json.loads((tmp_path / "agent.jsonl.manifest.json").read_text())
    assert manifest["exact_prompt_holdouts"]["artifact_count"] == 2
    assert manifest["exact_prompt_holdouts"]["normalized_unique_prompts"] == len(heldout_queries)
    assert manifest["exact_prompt_holdouts"]["output_matches"] == 0
    assert [entry["name"] for entry in manifest["exact_prompt_holdouts"]["artifacts"]] == [
        "local-browser-tasks",
        "local-realtime-actions",
    ]


def test_synthesize_fails_closed_on_tampered_exact_prompt_holdout(tmp_path):
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [{"query": "A frozen benchmark prompt."}],
            }
        )
    )
    config = {
        "out": str(tmp_path / "agent.jsonl"),
        "n_samples": 10,
        "seed": 1,
        "level": 5,
        "split": "train",
        "complexity": {"multi_turn": 0},
        "irrelevance_fraction": 0,
        "exact_prompt_holdouts": [
            {
                "name": "frozen",
                "path": str(suite),
                "bytes": suite.stat().st_size,
                "sha256": "0" * 64,
            }
        ],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    try:
        synthesize(str(path))
    except ValueError as exc:
        assert "holdout 'frozen' SHA-256 mismatch" in str(exc)
    else:
        raise AssertionError("expected a tampered holdout artifact to be rejected")


def test_synthesize_excludes_case_whitespace_normalized_prompt_variant(tmp_path):
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [{"query": "  RUN\u00a0THE\tTESTS. "}],
            }
        )
    )
    config = {
        "out": str(tmp_path / "agent.jsonl"),
        "n_samples": 100,
        "seed": 7,
        "level": 5,
        "split": "train",
        "complexity": {"multi_turn": 0},
        "irrelevance_fraction": 0,
        "exact_prompt_holdouts": [
            {
                "name": "frozen",
                "path": str(suite),
                "bytes": suite.stat().st_size,
                "sha256": hashlib.sha256(suite.read_bytes()).hexdigest(),
            }
        ],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    synthesize(str(path))
    conversations = [
        Conversation.from_json(line) for line in (tmp_path / "agent.jsonl").read_text().splitlines()
    ]
    normalize = agent_synth._canonical_holdout_prompt
    generated = {
        normalize(message.content)
        for conversation in conversations
        for message in conversation.messages
        if message.role == Role.user and message.content is not None
    }
    assert normalize("Run the tests.") not in generated


def test_synthesize_rejects_coverage_quotas_that_do_not_fit(tmp_path):
    config = {
        "out": str(tmp_path / "agent.jsonl"),
        "n_samples": 10,
        "seed": 1,
        "level": 5,
        "split": "train",
        "complexity": {"multi_turn": 0.2},
        "irrelevance_fraction": 0.2,
        "coverage": {
            "minimum_conversations": {
                "tool_response_grounded_followups": 2,
                "verified_error_recovery": 1,
            }
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    try:
        synthesize(str(path))
    except ValueError as exc:
        assert "multi-turn coverage quotas need 3 rows, only 2 available" in str(exc)
    else:
        raise AssertionError("expected infeasible coverage quotas to be rejected")


def test_parallel_conversation_renderer_keeps_every_call(tmp_path):
    config = {
        "out": str(tmp_path / "agent.jsonl"),
        "n_samples": 100,
        "seed": 2,
        "level": 5,
        "generator": {"backend": "deterministic_templates"},
        "complexity": {"multi_turn": 0},
        "irrelevance_fraction": 0,
        "verification": {"rule_based": True, "model_based": False},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    synthesize(str(path))
    conversations = [
        Conversation.from_json(line) for line in (tmp_path / "agent.jsonl").read_text().splitlines()
    ]
    parallel = next(
        conv
        for conv in conversations
        if any(
            message.role == Role.assistant and len(message.tool_calls) == 2
            for message in conv.messages
        )
    )
    ids, _ = render_conversation(parallel, ByteTokenizer())
    rendered = ByteTokenizer().decode(ids, stop_at_eos=False)
    assert rendered.count("<tool_call>") == 2
