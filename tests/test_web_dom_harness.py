from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
WEB_DIR = ROOT / "demos" / "webgpu"
DOM_CASES = WEB_DIR / "browser-task-cases.json"
DOM_HARNESS = WEB_DIR / "browser-tasks.js"
DOM_HTML = WEB_DIR / "browser-tasks.html"


def test_dom_suite_is_versioned_and_covers_each_dispatch_family_once():
    suite = json.loads(DOM_CASES.read_text())
    expected = {
        "click",
        "double_click",
        "type_text",
        "key_press",
        "scroll",
        "drag",
        "move_cursor",
        "open_url",
    }
    assert suite["schema_version"] == 1
    assert suite["holdout_contract"] == {
        "case_sensitive": False,
        "normalization": "Unicode NFKC, Unicode whitespace collapse, Unicode casefold",
        "primitive_value_disjointness_claimed": False,
        "template_disjointness_claimed": False,
        "training_exclusion": "canonical normalized query equality",
    }
    assert suite["fixture_contract"]["version"] == 1
    assert {case["expected"]["tool"] for case in suite["cases"]} == expected
    assert len({case["id"] for case in suite["cases"]}) == len(suite["cases"])
    assert all(case["expected_dom"] for case in suite["cases"])
    html = DOM_HTML.read_text()
    assert 'id="browser-task-warmups"' in html
    assert 'value="3"' in html
    assert 'id="browser-task-repetitions"' in html
    assert 'value="30"' in html
    assert 'id="browser-task-seed"' in html
    assert 'value="dom-loop-v2-trailing"' in html
    for context_tokens in ("128", "512", "1024", "1536"):
        assert f'value="{context_tokens}"' in html


def test_dom_payload_binds_verified_fetched_graph_bytes_and_provider_scope():
    harness = DOM_HARNESS.read_text()
    assert "const modelByteEvidence = modelArtifactEvidence(MODEL_URL)" in harness
    assert "if (!modelByteEvidence?.manifest_verified)" in harness
    assert "graph_hash: modelByteEvidence.sha256" in harness
    assert "model_bytes: modelByteEvidence.bytes" in harness
    assert "model_byte_evidence: modelByteEvidence" in harness
    assert "runtime_asset_evidence: runtimeAssets" in harness
    assert "suite_byte_evidence: BROWSER_TASK_SUITE_BYTE_EVIDENCE" in harness
    assert "bundle_manifest_byte_evidence: bundleManifestByteEvidence()" in harness
    assert 'per_node_placement: "unknown"' in harness
    assert 'per_node_fallback_status: "unknown"' in harness
    assert 'session_provider_count: 1' in harness
    assert "explicit-webgpu-no-whole-session-retry" in harness
    assert "hashes copied from bundle-manifest.json" not in harness
    assert 'benchmark_version: "rtab-dom-0.4"' in harness
    assert "fixed_compute_tokens_natural_decision_feature" in harness
    assert "decision_feature_index" in harness
    assert "after_natural_assistant_marker" in harness
    assert "predicted_route: action?.route ?? null" in harness
    assert "route_confidence: action?.conf ?? null" in harness
    assert "predicted_action: predictedAction" in harness
    assert "expected_action: expectedAction" in harness
    assert "parse_evidence: parseEvidence" in harness
    assert "action_timeout_ms: BROWSER_TASK_ACTION_TIMEOUT_MS" in harness
    assert "no_subsequent_policy_call_started: true" in harness


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_dom_harness_independently_rejects_invalid_schema_arguments():
    script = """
global.window = { __localAgentSkipInit: true };
global.META = {
  tools: [{
    name: "key_press",
    schema: {
      type: "object",
      properties: { key: { type: "string", enum: ["Enter", "Tab", "Escape"] } },
      required: ["key"],
      additionalProperties: false,
    },
  }],
};
const { browserTaskValidateActionSchema } = require(process.argv[1]);
const values = [
  browserTaskValidateActionSchema({ tool: "key_press", args: { key: "Escape" } }),
  browserTaskValidateActionSchema({ tool: "key_press", args: { key: "Delete" } }),
  browserTaskValidateActionSchema({
    tool: "key_press",
    args: { key: "Escape", undeclared: true },
  }),
];
process.stdout.write(JSON.stringify(values));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(DOM_HARNESS)],
        check=True,
        capture_output=True,
        text=True,
    )
    valid, bad_enum, extra_arg = json.loads(result.stdout)
    assert valid["valid"] is True
    assert bad_enum["valid"] is False
    assert any("enum" in error for error in bad_enum["errors"])
    assert extra_arg["valid"] is False
    assert any("not declared" in error for error in extra_arg["errors"])


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_dom_success_at_deadline_keeps_every_opportunity_in_denominator():
    latency = {
        "runtime_ttfa_ms": 99,
        "independent_validate_ms": 1,
        "model_wall_ms": 99,
        "tool_ms": 2,
        "paint_wait_ms": 4,
        "closed_loop_ms": 106,
    }
    records = [
        {
            "expected": {"tool": "click"},
            "score": {
                "exact_tool": True,
                "exact_args": True,
                "exact_action": True,
                "schema_valid": True,
                "final_dom_valid": True,
                "state_transition": True,
                "closed_loop_success": True,
            },
            "latency_ms": {**latency, "harness_ttfa_ms": 100},
        },
        {
            "expected": {"tool": "click"},
            "score": {
                "exact_tool": True,
                "exact_args": True,
                "exact_action": True,
                "schema_valid": False,
                "final_dom_valid": False,
                "state_transition": False,
                "closed_loop_success": False,
            },
            "latency_ms": {**latency, "harness_ttfa_ms": 80},
        },
        {
            "expected": {"tool": "click"},
            "score": {
                "exact_tool": False,
                "exact_args": False,
                "exact_action": False,
                "schema_valid": False,
                "final_dom_valid": False,
                "state_transition": False,
                "closed_loop_success": False,
            },
            "latency_ms": {**latency, "harness_ttfa_ms": 60},
        },
    ]
    script = """
global.window = { __localAgentSkipInit: true };
const { browserTaskSummarize } = require(process.argv[1]);
const records = JSON.parse(process.argv[2]);
process.stdout.write(JSON.stringify(browserTaskSummarize(records)));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(DOM_HARNESS), json.dumps(records)],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(result.stdout)
    deadline = summary["deadline_attainment_ms"]["100"]
    assert deadline["opportunities"] == 3
    assert deadline["on_time"] == 3
    assert deadline["useful"] == 1
    assert deadline["success_at_deadline"] == pytest.approx(1 / 3)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_dom_v04_action_evidence_is_normalized_and_independently_scored():
    script = """
global.window = { __localAgentSkipInit: true };
global.META = {
  tools: [{
    name: "key_press",
    schema: {
      type: "object",
      properties: { key: { type: "string", enum: ["Enter", "Escape"] } },
      required: ["key"],
      additionalProperties: false,
    },
  }],
};
global.groundedArgsValid = () => true;
const {
  browserTaskNormalizeAction, browserTaskScoreAction, browserTaskValidateActionSchema
} = require(process.argv[1]);
const raw = {
  tool: "key_press", args: { key: "Escape" }, timing: { omitted: true }, route: "tool",
};
const predicted = browserTaskNormalizeAction(raw);
const expected = { tool: "key_press", args: { key: "Escape" } };
const schema = browserTaskValidateActionSchema(predicted);
const score = browserTaskScoreAction(predicted, { expected }, schema);
process.stdout.write(JSON.stringify({ predicted, expected, schema, score }));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(DOM_HARNESS)],
        check=True,
        capture_output=True,
        text=True,
    )
    evidence = json.loads(result.stdout)
    assert evidence["predicted"] == evidence["expected"] == {
        "tool": "key_press",
        "args": {"key": "Escape"},
    }
    assert evidence["schema"]["validator"] == "browser-task-json-schema-subset-v2"
    assert evidence["schema"]["valid"] is True
    assert evidence["schema"]["errors"] == []
    assert evidence["schema"]["schema_tool"] == "key_press"
    assert evidence["score"]["exact_tool"] is True
    assert evidence["score"]["exact_args"] is True
    assert evidence["score"]["exact_action"] is True
    assert evidence["score"]["schema_valid"] is True


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_dom_v04_watchdog_is_finite_and_timeout_is_fail_stop():
    script = """
global.window = { __localAgentSkipInit: true };
const {
  BROWSER_TASK_ACTION_TIMEOUT_MS, browserTaskWithWatchdog, browserTaskIsActionTimeout
} = require(process.argv[1]);
(async () => {
  const completed = await browserTaskWithWatchdog(() => Promise.resolve("ok"), 20);
  let timeout = null;
  let subsequentStarted = false;
  try {
    await browserTaskWithWatchdog(() => new Promise(() => {}), 5);
    subsequentStarted = true;
  } catch (error) {
    timeout = {
      name: error.name,
      code: error.code,
      timeout_ms: error.timeout_ms,
      recognized: browserTaskIsActionTimeout(error),
    };
  }
  process.stdout.write(JSON.stringify({
    configured: BROWSER_TASK_ACTION_TIMEOUT_MS, completed, timeout, subsequentStarted
  }));
})().catch((error) => { console.error(error); process.exit(1); });
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(DOM_HARNESS)],
        check=True,
        capture_output=True,
        text=True,
    )
    evidence = json.loads(result.stdout)
    assert evidence == {
        "configured": 10_000,
        "completed": "ok",
        "timeout": {
            "name": "BrowserTaskActionTimeoutError",
            "code": "browser_task_action_timeout",
            "timeout_ms": 5,
            "recognized": True,
        },
        "subsequentStarted": False,
    }
