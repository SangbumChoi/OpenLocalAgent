import json

import pytest

from localagent.eval.agentnet_results import aggregate_agentnet_results


def _record(task_id: str, *, platform: str = "macos") -> dict:
    return {
        "task_id": task_id,
        "platform": platform,
        "steps": [
            {
                "ground_truth_actions": [
                    {
                        "type": "click",
                        "params": {"position": {"x": 0.25, "y": 0.5}},
                        "metadata": {},
                    },
                    {
                        "type": "press",
                        "params": {"keys": ["enter"]},
                        "metadata": {},
                    },
                ]
            }
        ],
    }


def _prediction(task_id: str, *, correct: bool, platform: str = "macos") -> dict:
    return {
        "task_id": task_id,
        "platform": platform,
        "predicted_actions": [
            {
                "name": "agentnet_click",
                "arguments": {"x": 0.25 if correct else 0.9, "y": 0.5},
            },
            {
                "name": "agentnet_key_press",
                "arguments": {"keys": ["enter"] if correct else ["escape"]},
            },
        ],
    }


def test_agentnet_result_aggregate_reports_platform_and_exact_coverage(tmp_path) -> None:
    ground = tmp_path / "ground.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    ground.write_text(
        chr(10).join(
            json.dumps(row)
            for row in [_record("one"), _record("two", platform="windows")]
        )
        + chr(10),
        encoding="utf-8",
    )
    predictions.write_text(
        chr(10).join(
            json.dumps(row)
            for row in [
                _prediction("one", correct=True),
                _prediction("two", correct=False, platform="windows"),
            ]
        )
        + chr(10),
        encoding="utf-8",
    )
    receipt = aggregate_agentnet_results(
        ground,
        predictions,
        expected_ids=["one", "two"],
        source_revision="agentnet-revision",
    )
    assert receipt["status"] == "complete"
    assert receipt["overall"]["tasks"] == 2
    assert receipt["overall"]["success_rate"] == 0.5
    assert receipt["by_platform"]["macos"]["success_rate"] == 1.0
    assert receipt["by_platform"]["windows"]["success_rate"] == 0.0


def test_agentnet_result_aggregate_requires_prediction_coverage(tmp_path) -> None:
    ground = tmp_path / "ground.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    ground.write_text(json.dumps(_record("one")) + chr(10), encoding="utf-8")
    predictions.write_text(json.dumps(_prediction("other", correct=True)) + chr(10), encoding="utf-8")
    with pytest.raises(ValueError, match="coverage"):
        aggregate_agentnet_results(ground, predictions)


def test_agentnet_result_aggregate_marks_missing_expected_id_incomplete(tmp_path) -> None:
    ground = tmp_path / "ground.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    ground.write_text(json.dumps(_record("one")) + chr(10), encoding="utf-8")
    predictions.write_text(json.dumps(_prediction("one", correct=True)) + chr(10), encoding="utf-8")
    receipt = aggregate_agentnet_results(
        ground,
        predictions,
        expected_ids=["one", "two"],
    )
    assert receipt["status"] == "incomplete"
    assert receipt["completeness"]["missing_expected_ids"] == ["two"]


def test_agentnet_result_aggregate_rejects_duplicate_prediction_ids(tmp_path) -> None:
    ground = tmp_path / "ground.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    ground.write_text(json.dumps(_record("one")) + chr(10), encoding="utf-8")
    predictions.write_text(
        chr(10).join(
            json.dumps(_prediction("one", correct=True))
            for _ in range(2)
        )
        + chr(10),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate prediction"):
        aggregate_agentnet_results(ground, predictions)


def test_agentnet_result_aggregate_accepts_current_pyautogui_code_rows(tmp_path) -> None:
    ground = tmp_path / "ground.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    ground.write_text(
        json.dumps(
            {
                "task_id": "code-row",
                "platform": "ubuntu",
                "steps": [
                    {
                        "action": "pyautogui.click(x=0.25, y=0.5)",
                    }
                ],
            }
        )
        + chr(10),
        encoding="utf-8",
    )
    predictions.write_text(
        json.dumps(
            {
                "task_id": "code-row",
                "predicted_actions": [
                    {
                        "name": "agentnet_click",
                        "arguments": {"x": 0.25, "y": 0.5},
                    }
                ],
            }
        )
        + chr(10),
        encoding="utf-8",
    )
    receipt = aggregate_agentnet_results(
        ground,
        predictions,
        expected_ids=["code-row"],
    )
    assert receipt["status"] == "complete"
    assert receipt["overall"]["mean_total"] == 1.0
