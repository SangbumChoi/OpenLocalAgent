import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

import openlocalagent.data.conversation_artifact as artifact_module
from openlocalagent.data.synth.agent_synth import synthesize
from openlocalagent.data.conversation_artifact import (
    canonical_json_bytes,
    load_verified_conversation_artifact,
    self_hashed_manifest,
)
from openlocalagent.data.schema import Conversation, Message, ToolCall, ToolSpec


def _synthesize_artifact(tmp_path: Path) -> tuple[Path, Path, Path]:
    output = tmp_path / "agent-train.jsonl"
    config = {
        "out": str(output),
        "n_samples": 12,
        "seed": 17,
        "level": 5,
        "split": "train",
        "generator": {"backend": "deterministic_templates"},
        "complexity": {"multi_turn": 0},
        "irrelevance_fraction": 0,
        "verification": {"rule_based": True, "model_based": False},
    }
    config_path = tmp_path / "agent-train.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    synthesize(str(config_path))
    return output, output.with_suffix(output.suffix + ".manifest.json"), config_path


def _bind_output(manifest_path: Path, payload: bytes) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("manifest_self_sha256")
    manifest["output_bytes"] = len(payload)
    manifest["output_sha256"] = hashlib.sha256(payload).hexdigest()
    _manifest, sealed = self_hashed_manifest(manifest)
    manifest_path.write_bytes(sealed)


def test_jsonl_uses_one_streaming_descriptor_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    output, _manifest_path, config_path = _synthesize_artifact(tmp_path)
    original_open = artifact_module._open_regular_file
    data_opens = 0

    @contextmanager
    def tracked_open(path: Path, *, label: str):
        nonlocal data_opens
        if path == output:
            data_opens += 1
        with original_open(path, label=label) as opened:
            yield opened

    monkeypatch.setattr(artifact_module, "_open_regular_file", tracked_open)
    loaded = load_verified_conversation_artifact(
        output,
        config_path=config_path,
        expected_split="train",
    )

    assert len(loaded.conversations) == 12
    assert data_opens == 1


def test_streaming_byte_and_line_caps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    output, _manifest_path, config_path = _synthesize_artifact(tmp_path)

    with pytest.raises(ValueError, match="exceeds hard byte cap"):
        load_verified_conversation_artifact(
            output,
            config_path=config_path,
            expected_split="train",
            max_jsonl_bytes=output.stat().st_size - 1,
        )

    first_line_bytes = len(output.read_bytes().splitlines(keepends=True)[0])
    monkeypatch.setattr(artifact_module, "_MAX_ROW_BYTES", first_line_bytes - 1)
    with pytest.raises(ValueError, match=r"line 1 exceeds hard byte cap"):
        load_verified_conversation_artifact(
            output,
            config_path=config_path,
            expected_split="train",
        )


def test_loader_rejects_symlink_and_open_path_swap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    output, _manifest_path, config_path = _synthesize_artifact(tmp_path)
    real_output = tmp_path / "real.jsonl"
    output.replace(real_output)
    output.symlink_to(real_output)

    with pytest.raises(ValueError, match="regular non-symlink"):
        load_verified_conversation_artifact(
            output,
            config_path=config_path,
            expected_split="train",
        )

    output.unlink()
    real_output.replace(output)
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_bytes(output.read_bytes())
    original_os_open = artifact_module.os.open
    swapped = False

    def swapping_open(path: os.PathLike[str], flags: int) -> int:
        nonlocal swapped
        descriptor = original_os_open(path, flags)
        if Path(path) == output and not swapped:
            swapped = True
            replacement.replace(output)
        return descriptor

    monkeypatch.setattr(artifact_module.os, "open", swapping_open)
    with pytest.raises(ValueError, match="regular non-symlink"):
        load_verified_conversation_artifact(
            output,
            config_path=config_path,
            expected_split="train",
        )


@pytest.mark.parametrize("drift_kind", ["metadata", "rename"])
def test_loader_rejects_in_place_and_rename_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift_kind: str,
):
    output, _manifest_path, config_path = _synthesize_artifact(tmp_path)
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_bytes(output.read_bytes())
    original_parse = artifact_module._parse_conversation_line
    drifted = False

    def drifting_parse(line: bytes, *, line_number: int, cache):
        nonlocal drifted
        conversation = original_parse(line, line_number=line_number, cache=cache)
        if not drifted:
            drifted = True
            if drift_kind == "metadata":
                current = output.stat()
                os.utime(
                    output,
                    ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000),
                )
            else:
                replacement.replace(output)
        return conversation

    monkeypatch.setattr(artifact_module, "_parse_conversation_line", drifting_parse)
    with pytest.raises(ValueError, match="changed while it was being verified"):
        load_verified_conversation_artifact(
            output,
            config_path=config_path,
            expected_split="train",
        )


def test_complete_verified_snapshot_is_read_only_and_legacy_serialization_matches(
    tmp_path: Path,
):
    output, manifest_path, config_path = _synthesize_artifact(tmp_path)
    legacy = [
        Conversation.from_json(line.decode("utf-8")) for line in output.read_bytes().splitlines()
    ]
    legacy[0].meta["nested"] = {"items": [{"value": 1}]}
    first_call = next(
        call
        for conversation in legacy
        for message in conversation.messages
        for call in message.tool_calls
    )
    first_call.arguments["nested"] = {"items": [{"value": 1}]}
    payload = "".join(row.to_json() + "\n" for row in legacy).encode("utf-8")
    output.write_bytes(payload)
    _bind_output(manifest_path, payload)
    loaded = load_verified_conversation_artifact(
        output,
        config_path=config_path,
        expected_split="train",
    )

    assert loaded.conversations == tuple(legacy)
    assert [row.to_json() for row in loaded.conversations] == [row.to_json() for row in legacy]
    assert [asdict(row) for row in loaded.conversations] == [asdict(row) for row in legacy]
    assert loaded.conversations[0].tools is loaded.conversations[1].tools
    assert isinstance(loaded.conversations[0], Conversation)
    assert isinstance(loaded.conversations[0].messages, list)
    assert isinstance(loaded.conversations[0].messages[0], Message)
    assert isinstance(loaded.conversations[0].tools, list)
    assert isinstance(loaded.conversations[0].tools[0], ToolSpec)
    assert (
        loaded.conversations[0].tools[0].parameters is loaded.conversations[1].tools[0].parameters
    )
    loaded_call = next(
        call
        for conversation in loaded.conversations
        for message in conversation.messages
        for call in message.tool_calls
    )
    loaded_call_message = next(
        message
        for conversation in loaded.conversations
        for message in conversation.messages
        if loaded_call in message.tool_calls
    )
    assert isinstance(loaded_call, ToolCall)

    with pytest.raises(TypeError, match="read-only"):
        loaded.conversations[0].messages = []
    with pytest.raises(TypeError, match="read-only"):
        loaded.conversations[0].tools = []
    with pytest.raises(TypeError, match="read-only"):
        loaded.conversations[0].meta = {}
    with pytest.raises(TypeError, match="read-only"):
        loaded.conversations[0].messages.append(loaded.conversations[0].messages[0])
    with pytest.raises(TypeError, match="read-only"):
        loaded.conversations[0].messages[0].content = "mutated"
    with pytest.raises(TypeError, match="read-only"):
        loaded_call_message.tool_calls = []
    with pytest.raises(TypeError, match="read-only"):
        loaded_call_message.tool_calls.append(loaded_call)
    with pytest.raises(TypeError, match="read-only"):
        loaded_call.name = "mutated"
    with pytest.raises(TypeError, match="read-only"):
        loaded_call.arguments = {}
    with pytest.raises(TypeError, match="read-only"):
        loaded_call.arguments["nested"]["items"][0]["value"] = 2
    with pytest.raises(TypeError, match="read-only"):
        loaded.conversations[0].meta["nested"]["items"].append({"value": 2})
    with pytest.raises(TypeError, match="read-only"):
        loaded.conversations[0].tools.append(loaded.conversations[0].tools[0])
    with pytest.raises(TypeError, match="read-only"):
        loaded.conversations[0].tools[0].description = "mutated"
    with pytest.raises(TypeError, match="read-only"):
        loaded.conversations[0].tools[0].parameters["mutated"] = True
    with pytest.raises(TypeError, match="read-only"):
        loaded.manifest["rows"] = 0
    with pytest.raises(TypeError, match="read-only"):
        loaded.manifest["generator_config"]["bytes"] = 0

    mutable = Conversation.from_json(loaded.conversations[0].to_json())
    mutable.messages[0].content = "mutable"
    mutable.meta["nested"]["items"][0]["value"] = 2
    assert mutable.messages[0].content == "mutable"
    assert mutable.meta["nested"]["items"][0]["value"] == 2


def test_catalog_digest_collision_compares_exact_serialized_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output, manifest_path, config_path = _synthesize_artifact(tmp_path)
    rows = [
        Conversation.from_json(line.decode("utf-8")) for line in output.read_bytes().splitlines()
    ]
    rows[1].tools[0].description += " collision variant"
    payload = "".join(row.to_json() + "\n" for row in rows).encode("utf-8")
    output.write_bytes(payload)
    _bind_output(manifest_path, payload)
    monkeypatch.setattr(artifact_module, "_catalog_digest", lambda _serialized: b"collision")

    loaded = load_verified_conversation_artifact(
        output,
        config_path=config_path,
        expected_split="train",
    )

    assert loaded.conversations[0].tools is not loaded.conversations[1].tools
    assert (
        loaded.conversations[0].tools[0].description != loaded.conversations[1].tools[0].description
    )
    assert loaded.conversations[0].tools is loaded.conversations[2].tools


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.replace(b"\n", b"\r\n", 1), "end in exactly one LF"),
        (lambda payload: payload[:-1], "end in exactly one LF"),
        (lambda payload: b"\n" + payload, "line 1 must not be empty"),
        (
            lambda payload: payload.replace(
                b'{"messages":',
                b'{"messages":[],"messages":',
                1,
            ),
            "duplicate JSON key",
        ),
        (lambda payload: b'{"nonfinite":NaN,' + payload[1:], "non-finite JSON number"),
    ],
)
def test_streaming_rows_fail_closed_with_strict_jsonl_rules(
    tmp_path: Path,
    mutate,
    message: str,
):
    output, manifest_path, config_path = _synthesize_artifact(tmp_path)
    payload = mutate(output.read_bytes())
    output.write_bytes(payload)
    _bind_output(manifest_path, payload)

    with pytest.raises(ValueError, match=message):
        load_verified_conversation_artifact(
            output,
            config_path=config_path,
            expected_split="train",
        )


def test_loaded_rows_reproduce_the_canonical_artifact_bytes(tmp_path: Path):
    output, _manifest_path, config_path = _synthesize_artifact(tmp_path)
    loaded = load_verified_conversation_artifact(
        output,
        config_path=config_path,
        expected_split="train",
    )

    reproduced = b"".join(
        (conversation.to_json() + "\n").encode("utf-8") for conversation in loaded.conversations
    )
    assert reproduced == output.read_bytes()
    assert canonical_json_bytes(loaded.manifest) == loaded.manifest_path.read_bytes()
