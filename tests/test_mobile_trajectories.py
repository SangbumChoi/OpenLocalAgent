from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_mobile_trajectories_are_ordered_and_have_unique_steps() -> None:
    script = """
const suite = require(process.argv[1]);
const trajectoryIds = suite.MOBILE_TRAJECTORIES.map((trajectory) => trajectory.id);
const stepIds = suite.MOBILE_TRAJECTORIES.flatMap((trajectory) =>
  trajectory.steps.map((step) => `${trajectory.id}:${step.id}`));
const summary = suite.trajectorySummary(suite.MOBILE_TRAJECTORIES.flatMap((trajectory) =>
  trajectory.steps.map((step, index) => ({
    trajectory_id: trajectory.id,
    exact_action: true,
    schema_valid: true,
    state_transition: true,
    closed_loop_success: true,
    step_index: index,
  }))));
process.stdout.write(JSON.stringify({trajectoryIds, stepIds, summary}));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, "./demos/webgpu/mobile-trajectories.js"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert len(payload["trajectoryIds"]) == len(set(payload["trajectoryIds"])) == 3
    assert len(payload["stepIds"]) == len(set(payload["stepIds"])) == 13
    assert payload["summary"]["trajectories"] == 3
    assert payload["summary"]["steps"] == 13
    assert payload["summary"]["pass_at_1"] == 1


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_mobile_trajectory_prompts_include_state_and_goal() -> None:
    script = """
const suite = require(process.argv[1]);
const trajectory = suite.MOBILE_TRAJECTORIES[0];
const prompt = suite.trajectoryPrompt(trajectory, trajectory.steps[0], suite.trajectoryInitialState());
process.stdout.write(JSON.stringify({prompt, state: suite.trajectoryInitialState()}));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, "./demos/webgpu/mobile-trajectories.js"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert "Goal:" in payload["prompt"]
    assert "Current state JSON:" in payload["prompt"]
    assert payload["state"]["email"]["sent"] is False


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_mobile_trajectory_exact_action_ignores_runtime_metadata() -> None:
    script = """
const suite = require(process.argv[1]);
const action = {tool: "open_url", args: {url: "https://example.local/mail"}, route: "web_search", conf: 0.9};
const expected = {tool: "open_url", args: {url: "https://example.local/mail"}};
process.stdout.write(JSON.stringify({exact: suite.trajectoryActionExact(action, expected)}));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, "./demos/webgpu/mobile-trajectories.js"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["exact"] is True


def test_mobile_trajectory_page_loads_bundle_tool_schemas() -> None:
    html = Path("demos/webgpu/mobile-trajectories.html").read_text()
    assert 'fetch("meta.json")' in html
    assert "META_FOR_MOBILE_TASKS" in html
