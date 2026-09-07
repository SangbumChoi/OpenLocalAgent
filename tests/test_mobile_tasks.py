from __future__ import annotations

import json
import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_mobile_task_suite_has_unique_expected_actions_and_state_helpers():
    script = """
const suite = require(process.argv[1]);
const ids = suite.MOBILE_TASKS.map((task) => task.id);
const families = suite.MOBILE_TASKS.reduce((out, task) => {
  out[task.family] = (out[task.family] || 0) + 1;
  return out;
}, {});
const state = suite.mobileTaskInitialState();
const summary = suite.mobileTaskSummary(suite.MOBILE_TASKS.map((task) => ({
  family: task.family,
  exact_tool: true,
  exact_args: true,
  exact_action: true,
  schema_valid: true,
  state_transition: true,
  closed_loop_success: true,
  selection_policy: "mobile_lexical_guard",
})));
process.stdout.write(JSON.stringify({ ids, families, state, summary }));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, "./demos/webgpu/mobile-tasks.js"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert len(payload["ids"]) == len(set(payload["ids"])) == 9
    assert payload["families"] == {"mobile": 7, "productivity": 2}
    assert payload["state"]["screen"] == "home"
    assert payload["summary"]["closed_loop_success_rate"] == 1
    assert payload["summary"]["selection_policy_counts"] == {"mobile_lexical_guard": 9}
