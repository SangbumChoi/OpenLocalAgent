/*
 * Deterministic multi-step stateful pilot for the text-first WebGPU policy.
 *
 * This is a local contract test, not AndroidWorld, BrowserGym, OSWorld, or a live MCP account.
 * Each trajectory feeds the current in-memory state back into the next prompt, validates the
 * emitted action independently, applies only the expected transition, and fails closed on an
 * out-of-order or malformed action.
 */

"use strict";

const MOBILE_TRAJECTORY_SUITE_ID = "localagent-mobile-productivity-trajectory-v1";

const MOBILE_TRAJECTORIES = Object.freeze([
  {
    id: "gmail_compose_send",
    family: "email",
    goal: "Open Gmail, compose an email, fill its fields, and send it.",
    steps: [
      {
        id: "open_gmail",
        prompt: "Open the Gmail app on the Android phone.",
        expected: { tool: "mobile_open_app", args: { app_name: "Gmail" } },
        apply: (state) => ({ ...state, app: "Gmail", screen: "inbox" }),
      },
      {
        id: "tap_compose",
        prompt: "Tap Compose at x=120 y=220 on the Android phone.",
        expected: { tool: "mobile_click", args: { x: 120, y: 220 } },
        guard: (state) => state.app === "Gmail" && state.screen === "inbox",
        apply: (state) => ({ ...state, screen: "compose", focus: "to" }),
      },
      {
        id: "fill_recipient",
        prompt: "Type 'alice@example.com' into the focused recipient field.",
        expected: { tool: "mobile_input_text", args: { text: "alice@example.com" } },
        guard: (state) => state.screen === "compose" && state.focus === "to",
        apply: (state) => ({
          ...state,
          email: { ...state.email, to: "alice@example.com" },
          focus: "subject",
        }),
      },
      {
        id: "fill_subject",
        prompt: "Type 'Mobile pilot' into the focused subject field.",
        expected: { tool: "mobile_input_text", args: { text: "Mobile pilot" } },
        guard: (state) => state.screen === "compose" && state.focus === "subject",
        apply: (state) => ({
          ...state,
          email: { ...state.email, subject: "Mobile pilot" },
          focus: "body",
        }),
      },
      {
        id: "fill_body",
        prompt: "Type 'The WebGPU state loop passed' into the focused message body.",
        expected: { tool: "mobile_input_text", args: { text: "The WebGPU state loop passed" } },
        guard: (state) => state.screen === "compose" && state.focus === "body",
        apply: (state) => ({
          ...state,
          email: { ...state.email, body: "The WebGPU state loop passed" },
          focus: "send",
        }),
      },
      {
        id: "send_email",
        prompt: "Send the composed email now.",
        expected: {
          tool: "email_send",
          args: {
            to: "alice@example.com",
            subject: "Mobile pilot",
            body: "The WebGPU state loop passed",
          },
        },
        guard: (state) => state.focus === "send" && state.email.body !== null,
        apply: (state) => ({
          ...state,
          screen: "inbox",
          focus: null,
          email: { ...state.email, sent: true },
        }),
      },
    ],
  },
  {
    id: "notion_capture",
    family: "notion",
    goal: "Open Notion and create a page containing the deployment note.",
    steps: [
      {
        id: "open_notion",
        prompt: "Open the Notion app on the Android phone.",
        expected: { tool: "mobile_open_app", args: { app_name: "Notion" } },
        apply: (state) => ({ ...state, app: "Notion", screen: "home" }),
      },
      {
        id: "create_notion_page",
        prompt: "Create a Notion page titled 'Mobile pilot' with content 'WebGPU state loop passed'.",
        expected: {
          tool: "notion_create_page",
          args: { title: "Mobile pilot", content: "WebGPU state loop passed" },
        },
        guard: (state) => state.app === "Notion",
        apply: (state) => ({
          ...state,
          notion: { title: "Mobile pilot", content: "WebGPU state loop passed" },
        }),
      },
    ],
  },
  {
    id: "browser_search_open",
    family: "browser",
    goal: "Open the local mail page, search for the quarterly report, and open the first result.",
    steps: [
      {
        id: "open_mail_page",
        prompt: "Open the local mail page at https://example.local/mail.",
        expected: { tool: "open_url", args: { url: "https://example.local/mail" } },
        apply: (state) => ({ ...state, page: "mail", focused: null }),
      },
      {
        id: "focus_search",
        prompt: "Click the Search field on the local mail page.",
        expected: { tool: "click", args: { target: "the Search field" } },
        guard: (state) => state.page === "mail",
        apply: (state) => ({ ...state, focused: "search" }),
      },
      {
        id: "type_query",
        prompt: "Type 'quarterly report' into the focused Search field.",
        expected: { tool: "type_text", args: { text: "quarterly report" } },
        guard: (state) => state.focused === "search",
        apply: (state) => ({ ...state, query: "quarterly report" }),
      },
      {
        id: "submit_query",
        prompt: "Press Enter to submit the search.",
        expected: { tool: "key_press", args: { key: "Enter" } },
        guard: (state) => state.query === "quarterly report",
        apply: (state) => ({ ...state, results: ["quarterly report"] }),
      },
      {
        id: "open_first_result",
        prompt: "Click the first result in the search results.",
        expected: { tool: "click", args: { target: "the first result" } },
        guard: (state) => Array.isArray(state.results) && state.results.length === 1,
        apply: (state) => ({ ...state, opened: "quarterly report" }),
      },
    ],
  },
]);

function trajectoryCanonical(value) {
  if (Array.isArray(value)) return `[${value.map(trajectoryCanonical).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) =>
      `${JSON.stringify(key)}:${trajectoryCanonical(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function trajectoryExact(left, right) {
  return trajectoryCanonical(left) === trajectoryCanonical(right);
}

function trajectoryActionExact(action, expected) {
  return trajectoryExact(
    { tool: action?.tool || null, args: action?.args || null },
    expected,
  );
}

function trajectoryInitialState() {
  return {
    app: "home",
    screen: "home",
    focus: null,
    email: { to: null, subject: null, body: null, sent: false },
    notion: null,
    page: null,
    focused: null,
    query: null,
    results: null,
    opened: null,
  };
}

function trajectoryPrompt(trajectory, step, state) {
  return [
    `Goal: ${trajectory.goal}`,
    `Current state JSON: ${trajectoryCanonical(state)}`,
    `Next required action: ${step.prompt}`,
    "Return exactly one structured action.",
  ].join("\n");
}

function trajectorySchema(action) {
  const spec = (window.META_FOR_MOBILE_TASKS || []).find((tool) => tool.name === action?.tool);
  if (!spec) return { valid: false, error: "unknown_tool" };
  const args = action?.args;
  if (!args || typeof args !== "object" || Array.isArray(args)) {
    return { valid: false, error: "args_not_object" };
  }
  const schema = spec.schema || {};
  for (const required of schema.required || []) {
    if (!Object.hasOwn(args, required)) return { valid: false, error: `missing_${required}` };
  }
  for (const [name, value] of Object.entries(args)) {
    const prop = schema.properties?.[name];
    if (!prop) return { valid: false, error: `unknown_${name}` };
    if (prop.enum && !prop.enum.includes(value)) return { valid: false, error: `enum_${name}` };
    if (prop.type === "string" && typeof value !== "string") return { valid: false, error: `type_${name}` };
    if (prop.type === "number" && (typeof value !== "number" || !Number.isFinite(value))) {
      return { valid: false, error: `type_${name}` };
    }
  }
  return { valid: true, error: null };
}

function trajectoryApply(state, action, step) {
  if (!trajectoryActionExact(action, step.expected)) {
    return { state, transitioned: false, error: "wrong_action" };
  }
  if (step.guard && !step.guard(state)) return { state, transitioned: false, error: "precondition_failed" };
  return { state: step.apply(state), transitioned: true, error: null };
}

function trajectorySummary(records) {
  const rate = (key, rows) => rows.length
    ? rows.filter((row) => row[key] === true).length / rows.length : 0;
  const byTrajectory = {};
  for (const trajectory of MOBILE_TRAJECTORIES) {
    const rows = records.filter((row) => row.trajectory_id === trajectory.id);
    const firstFailure = rows.find((row) => !row.closed_loop_success);
    byTrajectory[trajectory.id] = {
      family: trajectory.family,
      steps: rows.length,
      exact_action_rate: rate("exact_action", rows),
      schema_valid_rate: rate("schema_valid", rows),
      state_transition_rate: rate("state_transition", rows),
      closed_loop_success_rate: rate("closed_loop_success", rows),
      trajectory_success: rows.length === trajectory.steps.length && rows.every((row) => row.closed_loop_success),
      first_failure: firstFailure
        ? {
          step_index: firstFailure.step_index,
          step_id: firstFailure.step_id,
          expected_tool: firstFailure.expected.tool,
          observed_tool: firstFailure.observed?.tool || null,
          observed_args: firstFailure.observed?.args || null,
          schema_error: firstFailure.schema_error,
          transition_error: firstFailure.transition_error,
        }
        : null,
    };
  }
  return {
    suite: MOBILE_TRAJECTORY_SUITE_ID,
    trajectories: MOBILE_TRAJECTORIES.length,
    steps: records.length,
    exact_action_rate: rate("exact_action", records),
    schema_valid_rate: rate("schema_valid", records),
    state_transition_rate: rate("state_transition", records),
    closed_loop_success_rate: rate("closed_loop_success", records),
    pass_at_1: Object.values(byTrajectory).filter((row) => row.trajectory_success).length /
      MOBILE_TRAJECTORIES.length,
    by_trajectory: byTrajectory,
  };
}

async function runMobileTrajectoryPilot() {
  await window.__localAgentReady;
  const records = [];
  const started = performance.now();
  for (const trajectory of MOBILE_TRAJECTORIES) {
    let state = trajectoryInitialState();
    for (let index = 0; index < trajectory.steps.length; index++) {
      const step = trajectory.steps[index];
      const before = JSON.parse(JSON.stringify(state));
      let action = null;
      let modelError = null;
      const stepStarted = performance.now();
      try {
        action = await window.__localAgentStructuredAction(trajectoryPrompt(trajectory, step, before));
      } catch (error) {
        modelError = String(error?.message || error);
      }
      const schema = trajectorySchema(action);
      const applied = trajectoryApply(state, action, step);
      state = applied.state;
      records.push({
        trajectory_id: trajectory.id,
        family: trajectory.family,
        step_index: index,
        step_id: step.id,
        expected: step.expected,
        observed: action,
        model_error: modelError,
        schema_valid: schema.valid,
        schema_error: schema.error,
        exact_action: trajectoryActionExact(action, step.expected),
        state_transition: applied.transitioned,
        transition_error: applied.error,
        closed_loop_success: schema.valid && trajectoryActionExact(action, step.expected) && applied.transitioned,
        latency_ms: performance.now() - stepStarted,
        state_before: before,
        state_after: state,
      });
    }
  }
  const result = {
    suite: MOBILE_TRAJECTORY_SUITE_ID,
    model_ready_ms: window.__localAgentModelReadyMs ?? null,
    run_ms: performance.now() - started,
    backend: (await window.__localAgentReady).backend,
    records,
    summary: trajectorySummary(records),
  };
  if (typeof window !== "undefined") window.__localAgentMobileTrajectoryResult = result;
  const output = document.querySelector("#mobile-trajectory-output");
  if (output) output.textContent = JSON.stringify(result.summary, null, 2);
  const progress = document.querySelector("#mobile-trajectory-progress");
  if (progress) progress.textContent = `${result.summary.pass_at_1 * 100}% trajectory pass@1 · ${result.backend}`;
  return result;
}

if (typeof window !== "undefined") {
  window.__localAgentRunMobileTrajectoryPilot = runMobileTrajectoryPilot;
  window.__localAgentMobileTrajectorySuite = MOBILE_TRAJECTORIES;
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    MOBILE_TRAJECTORIES,
    MOBILE_TRAJECTORY_SUITE_ID,
    trajectoryCanonical,
    trajectoryExact,
    trajectoryActionExact,
    trajectoryInitialState,
    trajectoryPrompt,
    trajectorySummary,
  };
}
