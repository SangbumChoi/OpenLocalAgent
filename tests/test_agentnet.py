import hashlib
import json
from pathlib import Path

import pytest
import yaml

from localagent.data.agentnet import normalize_agentnet_record, parse_pyautogui_actions
from localagent.data.public_agent import build_public_agent_dataset
from scripts.ingest_agentnet_text import parse_action, project


def test_text_projection_maps_supported_actions_and_rejects_termination() -> None:
    assert parse_action("pyautogui.write(message='hello')") == (
        "type_text",
        {"text": "hello"},
    )
    assert parse_action("pyautogui.moveTo(x=0.1, y=0.2)\npyautogui.dragTo(x=0.3, y=0.4)") == (
        "drag",
        {"source": "x=0.100000;y=0.200000", "dest": "x=0.300000;y=0.400000"},
    )
    assert parse_action("computer.terminate(status='success')") is None
    assert parse_action("computer.wait()") == ("wait", {"seconds": 1})


def test_published_agentnet_text_projection_receipt_is_explicitly_offline() -> None:
    receipt = json.loads(
        Path("docs/paper/results/raw/m62-agentnet-text-projection-eval-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["dataset"]["official_split_verified"] is False
    assert receipt["dataset"]["images_consumed"] is False
    assert receipt["dataset"]["desktop_runtime_executed"] is False
    assert receipt["transfer_arms"]["pretrained_sft"]["first_action_type_rate"] == 0.75
    assert receipt["transfer_arms"]["matched_random_backbone"]["first_action_type_rate"] == 0.0


def test_m112_surface_selector_receipt_preserves_negative_transfer_boundary() -> None:
    receipt = json.loads(
        Path("docs/paper/results/raw/m112-agentnet-surface-selector-repair-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["source"]["official_split_verified"] is False
    assert receipt["source"]["images_consumed"] is False
    assert receipt["training"]["backbone_frozen"] is True
    assert receipt["training"]["warm"]["eval_selector_top1"] == pytest.approx(0.7067669034004211)
    assert receipt["training"]["random"]["eval_selector_top1"] == pytest.approx(0.7142857313156128)
    assert receipt["end_to_end_text_projection"]["first_action_type_rate"] == 1.0
    assert receipt["end_to_end_text_projection"]["success_rate"] == 0.0


def test_published_mixed_public_continuation_receipt_is_hash_bound_and_negative() -> None:
    path = Path("docs/paper/results/raw/m68-mixed-public-agent-continuation-v1.json")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    expected = receipt.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(
            receipt,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert actual == expected
    assert {item["dataset"] for item in receipt["source"]["datasets"]} == {
        "xlangai/AgentNet",
        "osunlp/Mind2Web",
    }
    assert receipt["source"]["public_training_text_used"] is True
    assert receipt["source"]["native_runtime_executed"] is False
    training = receipt["training"]
    assert training["rows"] == {"train": 533, "eval": 137}
    assert training["after"]["eval"]["assistant_token_accuracy"] == pytest.approx(
        0.6004706734250543
    )
    assert training["heads"]["after"]["selector_top1_accuracy"] == 0.0
    assert receipt["weight_transfer"]["compatibility"]["tokenizer_sha256_equal"] is True
    weight_path = Path("docs/paper/results/raw/m68-mixed-public-agent-weight-transfer-v1.json")
    assert hashlib.sha256(weight_path.read_bytes()).hexdigest() == receipt["weight_transfer"]["report"]["sha256"]


def test_published_mixed_head_adaptation_receipt_is_head_only_and_hash_bound() -> None:
    path = Path("docs/paper/results/raw/m69-mixed-public-agent-head-adaptation-v1.json")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    expected = receipt.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert actual == expected
    assert receipt["configuration"]["head_steps"] == 800
    assert receipt["training"]["heads"]["before"]["selector_top1_accuracy"] == 0.0
    assert receipt["training"]["heads"]["after"]["selector_top1_accuracy"] == pytest.approx(
        0.5783132530120482
    )
    groups = receipt["weight_transfer"]["groups"]
    assert groups["embedding"]["relative_delta_l2"] == 0.0
    assert groups["attention_or_mixer"]["relative_delta_l2"] == 0.0
    assert groups["action_heads"]["relative_delta_l2"] > 0.8
    weight_path = Path("docs/paper/results/raw/m69-mixed-public-agent-head-weight-transfer-v1.json")
    assert hashlib.sha256(weight_path.read_bytes()).hexdigest() == receipt["weight_transfer"]["report"]["sha256"]


def test_published_random_head_control_matches_parent_head_result() -> None:
    path = Path("docs/paper/results/raw/m70-mixed-public-agent-random-head-control-v1.json")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    expected = receipt.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(
            receipt,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert actual == expected
    assert receipt["configuration"]["head_init"] == "random"
    assert receipt["training"]["heads"]["after"] == {
        "rows": 167,
        "tool_rows": 166,
        "route_accuracy": pytest.approx(0.9940119760479041),
        "selector_top1_accuracy": pytest.approx(0.5783132530120482),
    }
    assert receipt["matched_parent_head_control"]["after_selector_delta"] == 0.0
    assert receipt["matched_parent_head_control"]["after_route_delta"] == 0.0
    groups = receipt["weight_transfer"]["groups"]
    assert groups["embedding"]["relative_delta_l2"] == 0.0
    assert groups["action_heads"]["relative_delta_l2"] > 0.8
    weight_path = Path("docs/paper/results/raw/m70-mixed-public-agent-random-head-weight-transfer-v1.json")
    assert hashlib.sha256(weight_path.read_bytes()).hexdigest() == receipt["weight_transfer"]["report"]["sha256"]


def test_text_projection_keeps_parent_records_disjoint(tmp_path: Path) -> None:
    source = tmp_path / "agentnet.jsonl"
    rows = [
        {
            "task_id": f"task-{index}",
            "instruction": "Click the button.",
            "traj": [
                {
                    "value": {
                        "observation": "A button is visible.",
                        "code": "pyautogui.click(x=0.1, y=0.2)",
                    }
                }
            ],
        }
        for index in range(5)
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    train, evaluation, metadata = project(
        source,
        dataset="fixture/AgentNet",
        revision="fixture-revision",
        eval_fraction=0.4,
        seed=7,
        max_observation_chars=10,
    )
    train_ids = {row.meta["parent_record_id"] for row in train}
    eval_ids = {row.meta["parent_record_id"] for row in evaluation}
    assert train_ids.isdisjoint(eval_ids)
    assert metadata["complete_parent_records"] == 5
    assert metadata["train_rows"] == len(train)
    assert metadata["eval_rows"] == len(evaluation)
    assert "observation truncated" in train[0].messages[0].content


def test_parse_agentnet_literal_pyautogui_sequence() -> None:
    actions = parse_pyautogui_actions(
        "pyautogui.click(x=0.1, y=0.2)\npyautogui.write(message='hello')\n"
    )
    assert actions == [
        {"action_type": "click", "arguments": {"x": 0.1, "y": 0.2}},
        {"action_type": "write", "arguments": {"text": "hello"}},
    ]
    assert parse_pyautogui_actions("pyautogui.hotkey('ctrl', 'c')")[0]["arguments"] == {
        "keys": ["ctrl", "c"]
    }


def test_normalize_official_steps_shape_preserves_coordinate_actions() -> None:
    row = normalize_agentnet_record(
        {
            "task_id": "task-1",
            "user_task_description": "Open the editor and type hello.",
            "steps": [
                {
                    "inner_monologue": {"observation": "Editor window is visible."},
                    "action": "pyautogui.click(x=0.2, y=0.3)",
                    "low_level_instruction": "Focus the editor.",
                },
                {
                    "inner_monologue": {"observation": "The editor is focused."},
                    "ground_truth_actions": [
                        {"type": "write", "params": {"text": "hello"}, "metadata": {}},
                        {"type": "press", "params": {"keys": ["enter"]}, "metadata": {}},
                    ],
                },
            ],
        }
    )
    assert row["record_id"] == "agentnet:task-1"
    assert row["quality"]["text_first"] is True
    assert row["quality"]["action_count"] == 3
    assert row["capabilities"] == [
        "agentnet_click",
        "agentnet_key_press",
        "agentnet_type_text",
    ]
    calls = [
        message["tool_calls"][0]
        for message in row["messages"]
        if message["role"] == "assistant" and message.get("tool_calls")
    ]
    assert calls == [
        {"name": "agentnet_click", "arguments": {"x": 0.2, "y": 0.3}},
        {"name": "agentnet_type_text", "arguments": {"text": "hello"}},
        {"name": "agentnet_key_press", "arguments": {"keys": ["enter"]}},
    ]


def test_normalize_huggingface_traj_shape_and_reject_screenshot_only() -> None:
    row = normalize_agentnet_record(
        {
            "task_id": "task-2",
            "instruction": "Press enter.",
            "traj": [
                {
                    "value": {
                        "observation": "A dialog is open.",
                        "code": "pyautogui.press(keys=['enter'])",
                    }
                }
            ],
        }
    )
    assert row["messages"][2]["tool_calls"] == [
        {"name": "agentnet_key_press", "arguments": {"keys": ["enter"]}}
    ]

    with pytest.raises(ValueError, match="screenshot-only"):
        normalize_agentnet_record(
            {
                "task_id": "task-3",
                "instruction": "Click the button.",
                "traj": [
                    {
                        "value": {
                            "image": "frame.png",
                            "code": "pyautogui.click(x=0.1, y=0.1)",
                        }
                    }
                ],
            }
        )


def test_public_builder_accepts_agentnet_only_as_eval(tmp_path: Path) -> None:
    train_row = {
        "record_id": "fixture-train",
        "domain": "fixture",
        "behavior": "action",
        "capabilities": ["ping"],
        "slot_values": {},
        "tools": [
            {
                "name": "ping",
                "description": "Ping.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            }
        ],
        "messages": [
            {"role": "user", "content": "Ping."},
            {"role": "assistant", "tool_calls": [{"name": "ping", "arguments": {}}]},
        ],
    }
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(json.dumps(train_row, separators=(",", ":")) + "\n")
    eval_path = tmp_path / "agentnet.jsonl"
    eval_path.write_text(
        json.dumps(
            {
                "task_id": "agentnet-1",
                "instruction": "Press enter.",
                "traj": [
                    {
                        "value": {
                            "observation": "A dialog is open.",
                            "code": "pyautogui.press(keys=['enter'])",
                        }
                    }
                ],
            },
            separators=(",", ":"),
        )
        + "\n"
    )

    def identity(path: Path) -> tuple[int, str]:
        payload = path.read_bytes()
        return len(payload), hashlib.sha256(payload).hexdigest()

    train_bytes, train_sha = identity(train_path)
    eval_bytes, eval_sha = identity(eval_path)
    config = {
        "schema_version": 1,
        "seed": 1,
        "enrichment_level": 0,
        "outputs": {"train": "out-train.jsonl", "eval": "out-eval.jsonl"},
        "manifest": "manifest.json",
        "sources": [
            {
                "source_id": "fixture-train",
                "dataset": "Fixture/PublicActions",
                "subset": "train",
                "revision": "0123456789abcdef",
                "url": "https://example.test/public-actions",
                "license": "CC-BY-4.0",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "adapter": "localagent_v1",
                "split": "train",
                "path": train_path.name,
                "bytes": train_bytes,
                "sha256": train_sha,
            },
            {
                "source_id": "agentnet-eval",
                "dataset": "xlangai/AgentNet",
                "subset": "official",
                "revision": "d76ee50a63fad81cfdbe576416757d7c2091ed50",
                "url": "https://huggingface.co/datasets/xlangai/AgentNet",
                "license": "MIT",
                "license_url": "https://github.com/xlang-ai/OpenCUA/blob/main/LICENSE",
                "adapter": "agentnet_v1",
                "split": "eval",
                "path": eval_path.name,
                "bytes": eval_bytes,
                "sha256": eval_sha,
            },
        ],
    }
    config_path = tmp_path / "public-agent.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True))
    result = build_public_agent_dataset(config_path)
    assert len(result.conversations["eval"]) == 1
    assert result.conversations["eval"][0].meta["action_count"] == 1
