import hashlib
import json
from pathlib import Path

import pytest
import yaml

from localagent.data.flywheel import build_training_pool, ingest
from localagent.data.prompt_contract import assistant_training_turns
from localagent.data.public_agent import (
    MIND2WEB_REVISION,
    PUBLIC_AGENT_GENERATOR,
    VERIFICATION_SCOPE,
    XLAM_REVISION,
    build_public_agent_dataset,
)
from localagent.data.render import IGNORE, render_conversation
from localagent.data.schema import Conversation, Role
from localagent.model.tokenizer import ByteTokenizer


def _tools() -> list[dict]:
    return [
        {
            "name": "notify_channel",
            "description": "Post an operational update.",
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["channel", "message"],
                "additionalProperties": False,
            },
        },
        {
            "name": "search_tickets",
            "description": "Search support tickets.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        {
            "name": "update_ticket",
            "description": "Update one support ticket.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["investigating", "resolved"],
                    },
                    "ticket_id": {"type": "string"},
                },
                "required": ["ticket_id", "status"],
                "additionalProperties": False,
            },
        },
    ]


def _action_record(
    record_id: str,
    *,
    incident: str,
    ticket: str,
    channel: str,
) -> dict:
    return {
        "record_id": record_id,
        "domain": "customer_support",
        "behavior": "action",
        "capabilities": ["update_ticket", "search_tickets", "notify_channel", "search_tickets"],
        "slot_values": {
            "channel": [channel],
            "incident": [incident],
            "ticket": [ticket],
        },
        "quality": {"accepted": True, "source_trace": "operator-reviewed"},
        "tools": _tools(),
        "messages": [
            {
                "role": "user",
                "content": f"Investigate {incident}, update {ticket}, and notify {channel}.",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {"name": "search_tickets", "arguments": {"query": incident}}
                ],
            },
            {"role": "tool", "tool_response": f"Found {ticket} for {incident}."},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "name": "update_ticket",
                        # Deliberately non-sorted input; canonical rendering must sort it.
                        "arguments": {
                            "ticket_id": ticket,
                            "status": "investigating",
                        },
                    }
                ],
            },
            {"role": "tool", "tool_response": f"Updated {ticket}."},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "name": "notify_channel",
                        "arguments": {
                            "message": f"{incident} is under investigation",
                            "channel": channel,
                        },
                    }
                ],
            },
            {"role": "tool", "tool_response": f"Posted to {channel}."},
            {"role": "assistant", "content": "The incident was triaged and the team was notified."},
        ],
    }


def _negative_record(record_id: str, *, ticket: str) -> dict:
    return {
        "record_id": record_id,
        "domain": "customer_support",
        "behavior": "irrelevance",
        "capabilities": [],
        "slot_values": {"ticket": [ticket]},
        "quality": {"accepted": True, "source_trace": "operator-reviewed"},
        "tools": _tools(),
        "messages": [
            {
                "role": "user",
                "content": (
                    f"The request to resolve {ticket} is quoted background only. "
                    "Do not update anything; acknowledge."
                ),
            },
            {"role": "assistant", "content": "Acknowledged. No action was taken."},
        ],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> tuple[int, str]:
    payload = b"".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
        + b"\n"
        for row in rows
    )
    path.write_bytes(payload)
    return len(payload), hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    path.write_bytes(payload)
    return len(payload), hashlib.sha256(payload).hexdigest()


def _source(
    path: Path,
    *,
    source_id: str,
    split: str,
    byte_count: int,
    sha256: str,
) -> dict:
    return {
        "source_id": source_id,
        "dataset": "Fixture/PublicActions",
        "subset": split,
        "revision": "0123456789abcdef",
        "url": "https://example.test/public-actions",
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "adapter": "localagent_v1",
        "split": split,
        "path": path.name,
        "bytes": byte_count,
        "sha256": sha256,
    }


def _fixture_config(tmp_path: Path, *, overlap_slots: bool = False) -> Path:
    train_path = tmp_path / "public-train.jsonl"
    eval_path = tmp_path / "public-eval.jsonl"
    train_identity = _write_jsonl(
        train_path,
        [
            _action_record(
                "train-incident",
                incident="INC-104",
                ticket="T-104",
                channel="#ops-east",
            ),
            _negative_record("train-negative", ticket="T-204"),
        ],
    )
    eval_negative = _negative_record(
        "eval-negative",
        ticket="T-204" if overlap_slots else "T-908",
    )
    if overlap_slots:
        eval_negative["messages"][0]["content"] = (
            "Evaluation-only wording. " + eval_negative["messages"][0]["content"]
        )
    eval_identity = _write_jsonl(
        eval_path,
        [
            _action_record(
                "eval-incident",
                incident="INC-907",
                ticket="T-907",
                channel="#ops-west",
            ),
            eval_negative,
        ],
    )
    holdout = tmp_path / "heldout.jsonl"
    holdout_identity = _write_jsonl(holdout, [{"prompt": "A frozen unrelated request."}])
    config = {
        "schema_version": 1,
        "seed": 73,
        "enrichment_level": 3,
        "outputs": {
            "train": "normalized-train.jsonl",
            "eval": "normalized-eval.jsonl",
        },
        "manifest": "normalized.manifest.json",
        "sources": [
            _source(
                train_path,
                source_id="fixture-train",
                split="train",
                byte_count=train_identity[0],
                sha256=train_identity[1],
            ),
            _source(
                eval_path,
                source_id="fixture-eval",
                split="eval",
                byte_count=eval_identity[0],
                sha256=eval_identity[1],
            ),
        ],
        "exact_prompt_holdouts": [
            {
                "name": "frozen-fixture",
                "path": holdout.name,
                "bytes": holdout_identity[0],
                "sha256": holdout_identity[1],
            }
        ],
    }
    path = tmp_path / "public-agent.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return path


def test_public_ingestion_is_deterministic_provenance_bound_and_balanced(tmp_path: Path):
    config = _fixture_config(tmp_path)
    first = build_public_agent_dataset(config)
    first_bytes = {
        path: path.read_bytes()
        for path in [*first.outputs.values(), first.manifest_path]
    }
    second = build_public_agent_dataset(config)

    assert {
        path: path.read_bytes()
        for path in [*second.outputs.values(), second.manifest_path]
    } == first_bytes
    assert len(first.conversations["train"]) == 7
    assert len(first.conversations["eval"]) == 2
    train_counts = first.manifest["outputs"]["train"]
    assert train_counts["behavior"] == {"action": 3, "irrelevance": 4}
    assert train_counts["multi_step_rows"] == 3
    assert train_counts["derivations"] == {
        "counterfactual_cancel_v1": 1,
        "domain_context_wrapper_v1": 2,
        "operational_wrapper_v1": 2,
        "source": 2,
    }
    assert first.manifest["outputs"]["eval"]["derivations"] == {"source": 2}
    assert first.manifest["split_audit"]["slot_overlap"] == 0
    assert first.manifest["licenses"] == {"CC-BY-4.0": 2}
    assert first.manifest_path.read_bytes().endswith(b"\n")

    action = first.conversations["train"][0]
    assert action.meta["generator"] == PUBLIC_AGENT_GENERATOR
    assert action.meta["verification_scope"] == VERIFICATION_SCOPE
    assert action.meta["capabilities"] == [
        "notify_channel",
        "search_tickets",
        "update_ticket",
    ]
    assert action.meta["action_count"] == 3
    assert set(action.meta["provenance"]) == {
        "dataset",
        "subset",
        "revision",
        "record_id",
        "url",
        "license",
        "file_sha256",
        "source_line",
    }
    assert action.meta["rule_verified"] is True
    assert action.meta["model_verified"] is False
    assert action.meta["environment_executed"] is False


def test_public_actions_render_canonical_targets_and_mask_non_assistant_tokens(tmp_path: Path):
    result = build_public_agent_dataset(_fixture_config(tmp_path))
    action = next(
        row
        for row in result.conversations["train"]
        if row.meta["derivation"] == "source" and row.meta["behavior"] == "action"
    )
    turns = assistant_training_turns(action)
    assert (
        turns[1].body
        == '<tool_call>{"arguments":{"status":"investigating","ticket_id":"T-104"},'
        '"name":"update_ticket"}</tool_call>'
    )
    ids, labels = render_conversation(action, ByteTokenizer())
    assert len(ids) == len(labels)
    assert sum(label == 0 for label in labels) == sum(
        message.role == Role.assistant for message in action.messages
    )
    assert any(label == IGNORE for label in labels)
    for message in action.messages:
        if message.role == Role.tool:
            encoded = ByteTokenizer().encode(message.tool_response or "")
            assert encoded
            # Tool response text is context, never an SFT target.
            response_start = _find_subsequence(ids, encoded)
            assert all(
                label == IGNORE
                for label in labels[response_start : response_start + len(encoded)]
            )


def _find_subsequence(values: list[int], needle: list[int]) -> int:
    for index in range(len(values) - len(needle) + 1):
        if values[index : index + len(needle)] == needle:
            return index
    raise AssertionError("subsequence not found")


def test_public_ingestion_rejects_train_eval_slot_leakage(tmp_path: Path):
    with pytest.raises(ValueError, match="declared slot values must be disjoint"):
        build_public_agent_dataset(_fixture_config(tmp_path, overlap_slots=True))


def test_public_ingestion_rejects_holdout_prompt_and_source_drift(tmp_path: Path):
    config = _fixture_config(tmp_path)
    raw = yaml.safe_load(config.read_text())
    train_source = tmp_path / raw["sources"][0]["path"]
    rows = [json.loads(line) for line in train_source.read_text().splitlines()]
    rows[0]["messages"][0]["content"] = "A frozen unrelated request."
    identity = _write_jsonl(train_source, rows)
    raw["sources"][0]["bytes"], raw["sources"][0]["sha256"] = identity
    config.write_text(yaml.safe_dump(raw, sort_keys=True))
    with pytest.raises(ValueError, match="matches an exact eval prompt holdout"):
        build_public_agent_dataset(config)

    config = _fixture_config(tmp_path)
    raw = yaml.safe_load(config.read_text())
    raw["sources"][0]["sha256"] = "0" * 64
    config.write_text(yaml.safe_dump(raw, sort_keys=True))
    with pytest.raises(ValueError, match="SHA-256 identity mismatch"):
        build_public_agent_dataset(config)


@pytest.mark.parametrize(
    ("dataset", "adapter", "revision", "license", "split", "message"),
    [
        (
            "Salesforce/xlam-function-calling-60k",
            "xlam_v1",
            "bad-revision",
            "cc-by-4.0",
            "train",
            "pinned TRAIN",
        ),
        (
            "osunlp/Mind2Web",
            "mind2web_v1",
            "17ece8eb89862368edc0cc806acee6fca5163474",
            "cc-by-4.0",
            "eval",
            "pinned data/train",
        ),
        (
            "McGill-NLP/WebLINX",
            "localagent_v1",
            "a30ff2",
            "CC-BY-NC-SA-4.0",
            "train",
            "eval-only",
        ),
        (
            "gorilla-llm/BFCL",
            "localagent_v1",
            "v4",
            "Apache-2.0",
            "train",
            "never be used",
        ),
    ],
)
def test_known_public_source_policy_fails_closed(
    tmp_path: Path,
    dataset: str,
    adapter: str,
    revision: str,
    license: str,
    split: str,
    message: str,
):
    config = _fixture_config(tmp_path)
    raw = yaml.safe_load(config.read_text())
    raw["sources"][0].update(
        {
            "dataset": dataset,
            "adapter": adapter,
            "revision": revision,
            "license": license,
            "split": split,
        }
    )
    config.write_text(yaml.safe_dump(raw, sort_keys=True))
    with pytest.raises(ValueError, match=message):
        build_public_agent_dataset(config)


def test_flywheel_mines_public_train_rows_and_is_idempotent(tmp_path: Path):
    result = build_public_agent_dataset(_fixture_config(tmp_path))
    source = result.outputs["train"]
    pool = tmp_path / "pool.jsonl"

    assert build_training_pool(pool, store_path=source) == 7
    assert build_training_pool(pool, store_path=source) == 0
    loaded = ingest(pool)
    assert len(loaded) == 7
    assert all(row.meta["split"] == "train" for row in loaded)
    assert all(row.meta["public_data"] is True for row in loaded)

    # Evaluation rows are rejected by the flywheel even when they are otherwise rule-verified.
    assert (
        build_training_pool(
            pool,
            conversations=list(result.conversations["eval"]),
        )
        == 0
    )
    assert len(ingest(pool)) == 7


def test_flywheel_rejects_tampered_public_provenance(tmp_path: Path):
    result = build_public_agent_dataset(_fixture_config(tmp_path))
    row = Conversation.from_json(result.conversations["train"][0].to_json())
    row.meta["provenance"]["source_line"] = 0
    row.meta["environment_executed"] = True

    assert build_training_pool(tmp_path / "pool.jsonl", conversations=[row]) == 0
    assert ingest(tmp_path / "pool.jsonl") == []


def test_conversation_roundtrip_keeps_public_provenance(tmp_path: Path):
    result = build_public_agent_dataset(_fixture_config(tmp_path))
    row = result.conversations["eval"][0]
    roundtrip = Conversation.from_json(row.to_json())
    assert roundtrip == row
    assert roundtrip.meta["provenance"]["license"] == "CC-BY-4.0"


def test_xlam_adapter_normalizes_stringified_parallel_calls(tmp_path: Path):
    source_path = tmp_path / "xlam_function_calling_60k.json"
    tools = [
        {
            "name": "math.sum",
            "description": "Sum integers.",
            "parameters": {
                "numbers": {
                    "type": "list[int]",
                    "description": "Numbers to add.",
                    "required": True,
                }
            },
        },
        {
            "name": "math.power",
            "description": "Raise a base to a power.",
            "parameters": {
                "base": {"type": "int", "required": True},
                "exponent": {"type": "int", "required": True},
            },
        },
    ]
    answers = [
        {"name": "math.sum", "arguments": {"numbers": [3, 5, 8]}},
        {"name": "math.power", "arguments": {"exponent": 3, "base": 2}},
    ]
    identity = _write_json(
        source_path,
        [
            {
                "id": 17,
                "query": json.dumps("Add 3, 5, and 8, then calculate 2 cubed."),
                "tools": json.dumps(tools),
                "answers": json.dumps(answers),
            }
        ],
    )
    config = {
        "schema_version": 1,
        "seed": 1,
        "enrichment_level": 0,
        "outputs": {"train": "xlam-train.jsonl"},
        "manifest": "xlam.manifest.json",
        "sources": [
            {
                "source_id": "xlam-train",
                "dataset": "Salesforce/xlam-function-calling-60k",
                "subset": "default",
                "revision": XLAM_REVISION,
                "url": "https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k",
                "license": "cc-by-4.0",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "adapter": "xlam_v1",
                "split": "train",
                "path": source_path.name,
                "bytes": identity[0],
                "sha256": identity[1],
            }
        ],
    }
    config_path = tmp_path / "xlam.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True))
    result = build_public_agent_dataset(config_path)
    row = result.conversations["train"][0]

    assert row.meta["action_count"] == 2
    assert row.meta["capabilities"] == ["math.power", "math.sum"]
    assert row.messages[1].tool_calls[0].arguments == {"numbers": [3, 5, 8]}
    assert result.manifest["outputs"]["train"]["multi_step_rows"] == 0
    assert result.manifest["outputs"]["train"]["parallel_action_rows"] == 1
    assert result.manifest["split_audit"]["paired_splits_present"] is False
    body = assistant_training_turns(row)[0].body
    assert body.count("<tool_call>") == 2
    assert '"arguments":{"base":2,"exponent":3}' in body


def test_mind2web_adapter_emits_grounded_multistep_action_trace(tmp_path: Path):
    source_path = tmp_path / "data" / "train" / "train_0.json"
    candidate = {
        "backend_node_id": "node-42",
        "is_original_target": True,
        "is_top_level_target": True,
    }
    identity = _write_json(
        source_path,
        [
            {
                "annotation_id": "m2w-train-17",
                "website": "shop.example",
                "domain": "shopping",
                "subdomain": "checkout",
                "confirmed_task": "Search for hiking boots and set the size to 9.",
                "action_reprs": [
                    "[search] Search -> CLICK",
                    "[search] Search -> TYPE: hiking boots",
                ],
                "actions": [
                    {
                        "action_uid": "a1",
                        "operation": {"op": "CLICK", "value": ""},
                        "pos_candidates": [candidate],
                    },
                    {
                        "action_uid": "a2",
                        "operation": {"op": "TYPE", "value": "hiking boots"},
                        "pos_candidates": [
                            {
                                **candidate,
                                "backend_node_id": "node-43",
                            }
                        ],
                    },
                ],
            }
        ],
    )
    config = {
        "schema_version": 1,
        "seed": 1,
        "enrichment_level": 0,
        "outputs": {"train": "mind2web-train.jsonl"},
        "manifest": "mind2web.manifest.json",
        "sources": [
            {
                "source_id": "mind2web-train-0",
                "dataset": "osunlp/Mind2Web",
                "subset": "train",
                "revision": MIND2WEB_REVISION,
                "url": "https://huggingface.co/datasets/osunlp/Mind2Web",
                "license": "cc-by-4.0",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "adapter": "mind2web_v1",
                "split": "train",
                "path": "data/train/train_0.json",
                "bytes": identity[0],
                "sha256": identity[1],
                "max_actions_per_record": 8,
            }
        ],
    }
    config_path = tmp_path / "mind2web.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True))
    result = build_public_agent_dataset(config_path)
    row = result.conversations["train"][0]

    assert row.meta["category"] == "shopping"
    assert row.meta["action_count"] == 2
    assert row.meta["capabilities"] == ["web_click", "web_type"]
    assert [message.role for message in row.messages] == [
        Role.user,
        Role.assistant,
        Role.tool,
        Role.assistant,
        Role.tool,
    ]
    assert row.messages[1].tool_calls[0].arguments == {"target_id": "node-42"}
    assert row.messages[3].tool_calls[0].arguments == {
        "target_id": "node-43",
        "text": "hiking boots",
    }
    assert result.manifest["outputs"]["train"]["multi_step_rows"] == 1
