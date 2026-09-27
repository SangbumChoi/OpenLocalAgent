from localagent.eval.agentnet import score_agentnet_actions, score_agentnet_record


def _ground_truth() -> list[dict]:
    return [
        {
            "type": "click",
            "params": {"position": {"x": 0.25, "y": 0.5}},
            "metadata": {},
        },
        {"type": "write", "params": {"text": "hello"}, "metadata": {}},
        {"type": "press", "params": {"keys": ["enter"]}, "metadata": {}},
    ]


def test_agentnet_scorer_accepts_normalized_localagent_calls_and_merges_enter() -> None:
    result = score_agentnet_actions(
        _ground_truth(),
        [
            {"name": "agentnet_click", "arguments": {"x": 0.25, "y": 0.5}},
            {"name": "agentnet_type_text", "arguments": {"text": "hello"}},
            {"name": "agentnet_key_press", "arguments": {"keys": ["enter"]}},
        ],
    )
    assert result["total"] == 1.0
    assert result["actions"] == {"click": 1.0, "write": 1.0}
    assert result["ground_truth_count"] == 2
    assert result["predicted_count"] == 2


def test_agentnet_scorer_reports_coordinate_and_text_errors() -> None:
    result = score_agentnet_actions(
        _ground_truth(),
        [
            {"name": "agentnet_click", "arguments": {"x": 0.9, "y": 0.9}},
            {"name": "agentnet_type_text", "arguments": {"text": "different"}},
            {"name": "agentnet_key_press", "arguments": {"keys": ["escape"]}},
        ],
    )
    assert 0.0 < result["total"] < 1.0
    assert result["actions"]["click"] < 1.0
    assert result["actions"]["write"] < 1.0


def test_agentnet_record_flattens_steps_and_rejects_wrong_first_action() -> None:
    record = {
        "task_id": "task-1",
        "steps": [{"ground_truth_actions": _ground_truth()}],
    }
    result = score_agentnet_record(
        record,
        [{"name": "agentnet_key_press", "arguments": {"keys": ["escape"]}}],
    )
    assert result["total"] == 0.0
    assert result["first_action_type_match"] is False
    assert result["actions"] == {"click": 0.0, "write": 0.0}


def test_agentnet_scorer_accepts_points_inside_public_target_bbox() -> None:
    truth = [
        {
            "type": "click",
            "params": {"position": {"x": 0.25, "y": 0.25}},
            "metadata": {"bboxes": [{"rel_bbox": [0.2, 0.2, 0.2, 0.2]}]},
        }
    ]
    result = score_agentnet_actions(
        truth,
        [{"name": "agentnet_click", "arguments": {"x": 0.35, "y": 0.35}}],
    )
    assert result["total"] == 1.0
    assert result["actions"] == {"click": 1.0}


def test_agentnet_scorer_penalizes_extra_actions_but_not_missing_suffix() -> None:
    truth = [
        {"type": "press", "params": {"keys": ["a"]}, "metadata": {}},
        {"type": "press", "params": {"keys": ["b"]}, "metadata": {}},
    ]
    short = score_agentnet_actions(
        truth,
        [{"name": "agentnet_key_press", "arguments": {"keys": ["a"]}}],
    )
    long = score_agentnet_actions(
        truth,
        [
            {"name": "agentnet_key_press", "arguments": {"keys": ["a"]}},
            {"name": "agentnet_key_press", "arguments": {"keys": ["b"]}},
            {"name": "agentnet_key_press", "arguments": {"keys": ["c"]}},
        ],
    )
    assert short["total"] == 0.5
    assert long["total"] == 2 / 3


def test_agentnet_scorer_accepts_wait_actions() -> None:
    result = score_agentnet_actions(
        [{"type": "wait", "params": {"seconds": 1}, "metadata": {}}],
        [{"name": "agentnet_wait", "arguments": {"seconds": 1}}],
    )
    assert result["total"] == 1.0
    assert result["actions"] == {"wait": 1.0}
