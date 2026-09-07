import hashlib
import json
from pathlib import Path

import pytest
import yaml

from openlocalagent.data.synth.agent_synth import synthesize
from openlocalagent.data.conversation_artifact import (
    FileIdentity,
    assert_no_conversation_overlap,
    audit_conversation_overlap,
    canonical_json_bytes,
    load_verified_conversation_artifact,
    rendered_assistant_prompts,
    self_hashed_manifest,
)
from openlocalagent.data.prompt_contract import (
    LEGACY_CONVERSATION_PROMPT_CONTRACT,
    OPENAI_FULL_CATALOG_V1,
    FunctionCatalogCache,
    assistant_training_examples,
    render_function_catalog,
)
from openlocalagent.data.render import history_text
from openlocalagent.data.schema import Conversation, Message, Role, ToolCall, ToolSpec
from openlocalagent.model.tokenizer import ASSISTANT, BPE_EOS, USER
from openlocalagent.train.stage_data import load_conversation_source


def _synthesize_small(
    tmp_path: Path,
    *,
    split: str = "train",
    rule_verified: bool = True,
) -> tuple[Path, Path, Path]:
    output = tmp_path / f"agent-{split}.jsonl"
    config = {
        "out": str(output),
        "n_samples": 12,
        "seed": 17,
        "level": 5,
        "split": split,
        "generator": {"backend": "deterministic_templates"},
        "complexity": {"multi_turn": 0},
        "irrelevance_fraction": 0,
        "verification": {
            "rule_based": rule_verified,
            "model_based": False,
        },
    }
    config_path = tmp_path / f"agent-{split}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    synthesize(str(config_path))
    return (
        output,
        output.with_suffix(output.suffix + ".manifest.json"),
        config_path,
    )


def _reseal_manifest(path: Path, mutate) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.pop("manifest_self_sha256")
    mutate(manifest)
    _sealed, payload = self_hashed_manifest(manifest)
    path.write_bytes(payload)


def test_verified_loader_binds_jsonl_sidecar_config_and_policy(tmp_path: Path):
    output, manifest_path, config_path = _synthesize_small(tmp_path)

    artifact = load_verified_conversation_artifact(
        output,
        config_path=config_path,
        expected_split="train",
    )

    assert len(artifact.conversations) == 12
    assert artifact.rule_verified is True
    assert artifact.environment_executed is False
    assert artifact.identity.jsonl == FileIdentity.from_bytes(output.read_bytes())
    assert artifact.identity.sidecar == FileIdentity.from_bytes(manifest_path.read_bytes())
    assert artifact.identity.generator_config == FileIdentity.from_bytes(config_path.read_bytes())
    assert artifact.manifest_path == manifest_path
    assert artifact.lineage_identity() == {
        "kind": "localagent_synthetic_conversation_artifact",
        "schema_version": 1,
        "split": "train",
        "jsonl": artifact.identity.jsonl.as_dict(),
        "sidecar": {
            **artifact.identity.sidecar.as_dict(),
            "manifest_self_sha256": artifact.identity.manifest_self_sha256,
        },
        "generator_config": artifact.identity.generator_config.as_dict(),
    }
    assert manifest_path.read_bytes() == canonical_json_bytes(artifact.manifest)

    rebound = load_verified_conversation_artifact(
        output,
        config_path=config_path,
        expected_split="train",
        environment_policy="allow",
        expected_manifest_identity=artifact.identity.sidecar,
    )
    assert rebound.identity == artifact.identity


def test_catalog_cache_reuses_verified_interned_catalog_but_never_mutable_lists(
    tmp_path: Path,
):
    output, _manifest_path, config_path = _synthesize_small(tmp_path)
    artifact = load_verified_conversation_artifact(
        output,
        config_path=config_path,
        expected_split="train",
    )
    verified_tools = artifact.conversations[0].tools
    cache = FunctionCatalogCache()

    first = cache.entry(verified_tools)
    assert cache.entry(verified_tools) is first
    assert cache.unique_catalogs == 1

    mutable_tools = [_contract_tool()]
    mutable_cache = FunctionCatalogCache()
    mutable_cache.entry(mutable_tools)
    assert mutable_cache.unique_catalogs == 0
    mutable_tools[0].description = "escape " + ASSISTANT
    with pytest.raises(ValueError, match="reserved prompt marker"):
        mutable_cache.entry(mutable_tools)


def test_synthesize_republishes_identical_jsonl_and_canonical_sidecar(tmp_path: Path):
    output, manifest_path, config_path = _synthesize_small(tmp_path)
    first = (output.read_bytes(), manifest_path.read_bytes())

    synthesize(str(config_path))

    assert (output.read_bytes(), manifest_path.read_bytes()) == first


def test_stage_source_loader_requires_versioned_strict_artifact_contract(
    tmp_path: Path,
):
    output, legacy_manifest, config_path = _synthesize_small(tmp_path)
    versioned_manifest = output.with_suffix(output.suffix + ".manifest.v1.json")
    versioned_manifest.write_bytes(legacy_manifest.read_bytes())
    artifact = {
        "generator_config": str(config_path),
        "manifest": str(versioned_manifest),
        "expected_split": "train",
        "expected_rule_verified": True,
        "environment_policy": "forbid",
    }

    loaded = load_conversation_source(
        {"path": str(output), "artifact": artifact},
        require_verified=True,
        expected_split="train",
    )

    assert loaded.verified is True
    assert len(loaded.conversations) == 12
    assert loaded.identity["jsonl"] == FileIdentity.from_bytes(output.read_bytes()).as_dict()
    assert (
        loaded.identity["sidecar"]["sha256"]
        == hashlib.sha256(versioned_manifest.read_bytes()).hexdigest()
    )
    assert (
        loaded.identity["generator_config"]
        == FileIdentity.from_bytes(config_path.read_bytes()).as_dict()
    )

    with pytest.raises(ValueError, match="requires an artifact mapping"):
        load_conversation_source(
            output,
            require_verified=True,
            expected_split="train",
        )
    with pytest.raises(ValueError, match="versioned suffix"):
        load_conversation_source(
            {
                "path": str(output),
                "artifact": {**artifact, "manifest": str(legacy_manifest)},
            },
            require_verified=True,
            expected_split="train",
        )
    with pytest.raises(ValueError, match="disagrees with stage role"):
        load_conversation_source(
            {"path": str(output), "artifact": artifact},
            require_verified=True,
            expected_split="eval",
        )

    legacy = load_conversation_source(
        output,
        require_verified=False,
        expected_split="train",
    )
    assert legacy.verified is False
    assert legacy.identity == {
        "bytes": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def test_verified_loader_rejects_wrong_split_rule_environment_and_sidecar_identity(
    tmp_path: Path,
):
    output, _manifest_path, config_path = _synthesize_small(tmp_path)
    artifact = load_verified_conversation_artifact(
        output,
        config_path=config_path,
        expected_split="train",
    )

    with pytest.raises(ValueError, match="split mismatch"):
        load_verified_conversation_artifact(
            output,
            config_path=config_path,
            expected_split="eval",
        )
    with pytest.raises(ValueError, match="rule-verification state mismatch"):
        load_verified_conversation_artifact(
            output,
            config_path=config_path,
            expected_split="train",
            expected_rule_verified=False,
        )
    with pytest.raises(ValueError, match="environment-executed.*required"):
        load_verified_conversation_artifact(
            output,
            config_path=config_path,
            expected_split="train",
            environment_policy="require",
        )
    wrong_sidecar = FileIdentity(
        bytes=artifact.identity.sidecar.bytes + 1,
        sha256=artifact.identity.sidecar.sha256,
    )
    with pytest.raises(ValueError, match="sidecar byte identity mismatch"):
        load_verified_conversation_artifact(
            output,
            config_path=config_path,
            expected_split="train",
            expected_manifest_identity=wrong_sidecar,
        )


def test_verified_loader_rejects_config_jsonl_and_manifest_tampering(tmp_path: Path):
    output, manifest_path, config_path = _synthesize_small(tmp_path)
    original_config = config_path.read_bytes()
    original_output = output.read_bytes()
    original_manifest = manifest_path.read_bytes()

    config_path.write_bytes(original_config + b"# changed after generation\n")
    with pytest.raises(ValueError, match="generator config byte identity mismatch"):
        load_verified_conversation_artifact(
            output,
            config_path=config_path,
            expected_split="train",
        )
    config_path.write_bytes(original_config)

    output.write_bytes(original_output + b"\n")
    with pytest.raises(ValueError, match="JSONL byte identity mismatch"):
        load_verified_conversation_artifact(
            output,
            config_path=config_path,
            expected_split="train",
        )
    output.write_bytes(original_output)

    tampered_manifest = json.loads(original_manifest)
    tampered_manifest["rows"] += 1
    manifest_path.write_bytes(canonical_json_bytes(tampered_manifest))
    with pytest.raises(ValueError, match="manifest_self_sha256 mismatch"):
        load_verified_conversation_artifact(
            output,
            config_path=config_path,
            expected_split="train",
        )

    manifest_path.write_bytes(original_manifest)
    _reseal_manifest(manifest_path, lambda manifest: manifest.__setitem__("rows", 13))
    with pytest.raises(ValueError, match="row count mismatch"):
        load_verified_conversation_artifact(
            output,
            config_path=config_path,
            expected_split="train",
        )


def test_verified_loader_rejects_noncanonical_conversation_after_valid_reseal(
    tmp_path: Path,
):
    output, manifest_path, config_path = _synthesize_small(tmp_path)
    lines = output.read_bytes().splitlines(keepends=True)
    first = json.loads(lines[0])
    lines[0] = (
        json.dumps(first, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")
    payload = b"".join(lines)
    output.write_bytes(payload)

    def bind_modified_output(manifest: dict) -> None:
        manifest["output_bytes"] = len(payload)
        manifest["output_sha256"] = hashlib.sha256(payload).hexdigest()

    _reseal_manifest(manifest_path, bind_modified_output)
    with pytest.raises(ValueError, match="canonical Conversation schema serialization"):
        load_verified_conversation_artifact(
            output,
            config_path=config_path,
            expected_split="train",
        )


def test_overlap_audit_distinguishes_semantic_rows_from_rendered_prompts():
    left = Conversation(
        messages=[
            Message(role=Role.user, content="Inspect the file."),
            Message(role=Role.assistant, content="Done."),
        ],
        meta={"split": "train"},
    )
    same_semantics_new_meta = Conversation(
        messages=[
            Message(role=Role.user, content="Inspect the file."),
            Message(role=Role.assistant, content="Done."),
        ],
        meta={"split": "eval"},
    )
    same_prompt_new_target = Conversation(
        messages=[
            Message(role=Role.user, content="Inspect the file."),
            Message(role=Role.assistant, content="I cannot."),
        ],
        meta={"split": "eval"},
    )
    clean = Conversation(
        messages=[
            Message(role=Role.user, content="Open the calendar."),
            Message(role=Role.assistant, content="Done."),
        ],
        meta={"split": "eval"},
    )

    semantic_overlap = audit_conversation_overlap([left], [same_semantics_new_meta])
    assert len(semantic_overlap.semantic_overlap_sha256) == 1
    assert len(semantic_overlap.rendered_prompt_overlap_sha256) == 1

    prompt_only_overlap = audit_conversation_overlap([left], [same_prompt_new_target])
    assert prompt_only_overlap.semantic_overlap_sha256 == ()
    assert len(prompt_only_overlap.rendered_prompt_overlap_sha256) == 1
    with pytest.raises(
        ValueError,
        match="semantic_rows=0, rendered_prompts=1",
    ):
        assert_no_conversation_overlap([left], [same_prompt_new_target])

    clean_audit = assert_no_conversation_overlap([left], [clean])
    assert clean_audit.clean
    assert clean_audit.as_dict()["semantic_overlap"] == 0
    assert clean_audit.as_dict()["rendered_prompt_overlap"] == 0


def test_rendered_prompt_audit_matches_multiturn_renderer_prefixes():
    conversation = Conversation(
        messages=[
            Message(role=Role.user, content="Find the config."),
            Message(
                role=Role.assistant,
                tool_calls=[ToolCall(name="search_files", arguments={"query": "config"})],
            ),
            Message(role=Role.tool, tool_response="config/app.yaml"),
            Message(role=Role.assistant, content="Found it."),
        ]
    )

    prompts = rendered_assistant_prompts(conversation)
    assert prompts == (
        history_text(conversation.messages[:1]) + ASSISTANT,
        history_text(conversation.messages[:3]) + ASSISTANT,
    )


def _contract_tool(*, description: str = "Inspect one path.") -> ToolSpec:
    return ToolSpec(
        name="inspect_path",
        description=description,
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    )


def _contract_conversation(
    *,
    system: str = "Follow the inspection policy.",
    tool_description: str = "Inspect one path.",
    first_answer: str = "First result.",
) -> Conversation:
    return Conversation(
        tools=[_contract_tool(description=tool_description)],
        messages=[
            Message(role=Role.system, content=system),
            Message(role=Role.user, content="Inspect the first path."),
            Message(role=Role.assistant, content=first_answer),
            Message(role=Role.user, content="Inspect another path."),
            Message(role=Role.assistant, content="Second result."),
        ],
    )


def test_rendered_prompt_contract_defaults_to_exact_legacy_bytes():
    conversation = _contract_conversation()
    expected = (
        USER + "Inspect the first path." + ASSISTANT,
        (
            USER
            + "Inspect the first path."
            + ASSISTANT
            + "First result."
            + USER
            + "Inspect another path."
            + ASSISTANT
        ),
    )

    assert rendered_assistant_prompts(conversation) == expected
    assert (
        rendered_assistant_prompts(
            conversation,
            conversation_prompt_contract=LEGACY_CONVERSATION_PROMPT_CONTRACT,
        )
        == expected
    )


def test_full_prompt_contract_matches_training_and_is_context_sensitive():
    conversation = _contract_conversation()
    prompts = rendered_assistant_prompts(
        conversation,
        conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
    )

    assert prompts == tuple(example.prompt for example in assistant_training_examples(conversation))
    assert prompts[0].startswith(render_function_catalog(conversation.tools) + BPE_EOS)
    assert ASSISTANT + "First result." + BPE_EOS + USER in prompts[1]

    system_changed = _contract_conversation(system="Use a different system policy.")
    catalog_changed = _contract_conversation(tool_description="Inspect a path differently.")
    history_changed = _contract_conversation(first_answer="A different first result.")
    assert (
        rendered_assistant_prompts(
            system_changed,
            conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
        )
        != prompts
    )
    assert (
        rendered_assistant_prompts(
            catalog_changed,
            conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
        )
        != prompts
    )
    changed_history_prompts = rendered_assistant_prompts(
        history_changed,
        conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
    )
    assert changed_history_prompts[0] == prompts[0]
    assert changed_history_prompts[1] != prompts[1]


def test_full_contract_overlap_records_semantics_and_forwards_through_assertion():
    left = _contract_conversation()
    right = _contract_conversation(system="Use a held-out system policy.")

    legacy = audit_conversation_overlap([left], [right])
    assert len(legacy.rendered_prompt_overlap_sha256) == 2
    assert legacy.conversation_prompt_contract == LEGACY_CONVERSATION_PROMPT_CONTRACT

    full = assert_no_conversation_overlap(
        [left],
        [right],
        conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
    )
    assert full.clean
    assert full.conversation_prompt_contract == OPENAI_FULL_CATALOG_V1
    metadata = full.as_dict()["fingerprint_contract"]
    assert metadata["conversation_prompt_contract"] == OPENAI_FULL_CATALOG_V1
    assert "catalog+EOS" in metadata["rendered_prompt"]


def test_full_contract_rejects_marker_injection_that_collides_under_legacy():
    injected = Conversation(
        messages=[
            Message(role=Role.user, content="x" + ASSISTANT + "y"),
            Message(role=Role.assistant, content="injected target"),
        ]
    )
    structural = Conversation(
        messages=[
            Message(role=Role.user, content="x"),
            Message(role=Role.assistant, content="y"),
            Message(role=Role.assistant, content="structural target"),
        ]
    )
    injected_legacy = set(rendered_assistant_prompts(injected))
    structural_legacy = set(rendered_assistant_prompts(structural))
    assert injected_legacy & structural_legacy

    with pytest.raises(ValueError, match="reserved prompt marker"):
        rendered_assistant_prompts(
            injected,
            conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
        )
