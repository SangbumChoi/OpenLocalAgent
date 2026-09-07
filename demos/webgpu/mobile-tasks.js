/*
 * Deterministic, text-first closed-loop pilot for the optional mobile tool pool.
 *
 * The state machine is intentionally local and inspectable. It models Android-style navigation
 * and two productivity records (email + Notion) without pretending to be AndroidWorld, AITW,
 * BrowserGym, or a live account integration. Every action is independently checked against the
 * declared schema in META and against the expected state transition.
 */

"use strict";

const MOBILE_TASK_SUITE_ID = "localagent-mobile-productivity-v1";
const MOBILE_TASKS = Object.freeze([
  {
    id: "open_gmail",
    family: "mobile",
    prompt: "On the Android phone, open the 'Gmail' app.",
    expected: { tool: "mobile_open_app", args: { app_name: "Gmail" } },
    transition: (state) => ({ ...state, app: "Gmail", screen: "inbox" }),
  },
  {
    id: "tap_compose",
    family: "mobile",
    prompt: "On the Android phone screen, tap Compose at x=120 y=220.",
    expected: { tool: "mobile_click", args: { x: 120, y: 220 } },
    transition: (state) => ({ ...state, screen: "compose", focus: "to" }),
  },
  {
    id: "fill_recipient",
    family: "mobile",
    prompt: "On the mobile screen, type 'alice@example.com' into the focused recipient field.",
    expected: { tool: "mobile_input_text", args: { text: "alice@example.com" } },
    transition: (state) => ({ ...state, email: { ...state.email, to: "alice@example.com" }, focus: "subject" }),
  },
  {
    id: "fill_subject",
    family: "mobile",
    prompt: "On the Android phone screen, type 'Mobile pilot' into the focused subject field.",
    expected: { tool: "mobile_input_text", args: { text: "Mobile pilot" } },
    transition: (state) => ({ ...state, email: { ...state.email, subject: "Mobile pilot" }, focus: "body" }),
  },
  {
    id: "scroll_inbox",
    family: "mobile",
    prompt: "On the Android phone screen, scroll down to older messages.",
    expected: { tool: "mobile_scroll", args: { direction: "down" } },
    transition: (state) => ({ ...state, scroll: state.scroll + 1 }),
  },
  {
    id: "swipe_messages",
    family: "mobile",
    prompt: "On the mobile screen, swipe from x=100 y=700 to x=100 y=300.",
    expected: { tool: "mobile_swipe", args: { start_x: 100, start_y: 700, end_x: 100, end_y: 300 } },
    transition: (state) => ({ ...state, scroll: state.scroll + 1 }),
  },
  {
    id: "write_notion",
    family: "productivity",
    prompt: "Create a Notion page titled 'Mobile pilot' with content 'WebGPU state loop passed'.",
    expected: {
      tool: "notion_create_page",
      args: { title: "Mobile pilot", content: "WebGPU state loop passed" },
    },
    transition: (state) => ({
      ...state,
      notion: { title: "Mobile pilot", content: "WebGPU state loop passed" },
    }),
  },
  {
    id: "send_email",
    family: "productivity",
    prompt: "Send an email to 'alice@example.com' with subject 'Mobile pilot' and body 'The WebGPU state loop passed'.",
    expected: {
      tool: "email_send",
      args: {
        to: "alice@example.com",
        subject: "Mobile pilot",
        body: "The WebGPU state loop passed",
      },
    },
    transition: (state) => ({
      ...state,
      email: {
        ...state.email,
        sent: true,
        body: "The WebGPU state loop passed",
      },
    }),
  },
  {
    id: "go_back",
    family: "mobile",
    prompt: "On the Android phone screen, navigate back from the compose window.",
    expected: { tool: "mobile_navigate_back", args: {} },
    transition: (state) => ({ ...state, screen: "inbox", focus: null }),
  },
]);

let LAST_MOBILE_TASK_RUN = null;
let MOBILE_TASK_MODEL_READY_MS = null;

function mobileTaskCanonical(value) {
  if (Array.isArray(value)) return `[${value.map(mobileTaskCanonical).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) =>
      `${JSON.stringify(key)}:${mobileTaskCanonical(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function mobileTaskExact(left, right) {
  return mobileTaskCanonical(left) === mobileTaskCanonical(right);
}

function mobileTaskInitialState() {
  return {
    app: "home",
    screen: "home",
    focus: null,
    scroll: 0,
    email: { to: null, subject: null, body: null, sent: false },
    notion: null,
  };
}

function mobileTaskIndependentSchema(action) {
  const spec = (window.META_FOR_MOBILE_TASKS || []).find((tool) => tool.name === action?.tool);
  if (!spec) return { valid: false, error: "unknown_tool" };
  const args = action.args;
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

function mobileTaskStateApply(state, action, task) {
  if (!mobileTaskExact({ tool: action?.tool, args: action?.args }, task.expected)) {
    return { state, transitioned: false };
  }
  return { state: task.transition(state), transitioned: true };
}

function mobileTaskRender(state, prompt) {
  $("mobile-task-current-prompt").textContent = prompt;
  $("mobile-task-state").textContent = JSON.stringify(state, null, 2);
}

function mobileTaskRecord(task, action, before, after, started, modelError = null) {
  const expected = task.expected;
  const schema = mobileTaskIndependentSchema(action);
  const exactTool = action?.tool === expected.tool;
  const exactArgs = mobileTaskExact(action?.args, expected.args);
  const exactAction = exactTool && exactArgs;
  const transitioned = exactAction && mobileTaskExact(after, task.transition(before));
  return {
    id: task.id,
    family: task.family,
    prompt: task.prompt,
    expected,
    observed: action,
    model_error: modelError,
    schema_valid: schema.valid,
    schema_error: schema.error,
    exact_tool: exactTool,
    exact_args: exactArgs,
    exact_action: exactAction,
    state_transition: transitioned,
    closed_loop_success: exactAction && schema.valid && transitioned,
    selection_policy: action?.selection_policy || null,
    latency_ms: {
      model_wall_ms: performance.now() - started,
      closed_loop_ms: performance.now() - started,
    },
    state_before: before,
    state_after: after,
  };
}

function mobileTaskSummary(records) {
  const rate = (key, subset = records) => subset.length
    ? subset.filter((record) => record[key] === true).length / subset.length : 0;
  const byFamily = {};
  for (const family of ["mobile", "productivity"]) {
    const rows = records.filter((record) => record.family === family);
    if (rows.length) byFamily[family] = {
      records: rows.length,
      exact_action_rate: rate("exact_action", rows),
      schema_valid_rate: rate("schema_valid", rows),
      closed_loop_success_rate: rate("closed_loop_success", rows),
    };
  }
  const policyCounts = {};
  for (const record of records) {
    const key = record.selection_policy || "unknown";
    policyCounts[key] = (policyCounts[key] || 0) + 1;
  }
  return {
    suite: MOBILE_TASK_SUITE_ID,
    records: records.length,
    exact_tool_rate: rate("exact_tool"),
    exact_args_rate: rate("exact_args"),
    exact_action_rate: rate("exact_action"),
    schema_valid_rate: rate("schema_valid"),
    state_transition_rate: rate("state_transition"),
    closed_loop_success_rate: rate("closed_loop_success"),
    selection_policy_counts: policyCounts,
    by_family: byFamily,
  };
}

async function runMobileTaskPilot() {
  await window.__localAgentReady;
  const startedRun = performance.now();
  const records = [];
  let state = mobileTaskInitialState();
  for (const task of MOBILE_TASKS) {
    const before = JSON.parse(JSON.stringify(state));
    mobileTaskRender(before, task.prompt);
    const started = performance.now();
    let action = null;
    let modelError = null;
    try {
      action = await window.__localAgentStructuredAction(task.prompt);
    } catch (error) {
      modelError = String(error?.message || error);
    }
    const applied = mobileTaskStateApply(state, action, task);
    state = applied.state;
    records.push(mobileTaskRecord(task, action, before, state, started, modelError));
  }
  const result = {
    suite: MOBILE_TASK_SUITE_ID,
    model_ready_ms: MOBILE_TASK_MODEL_READY_MS,
    run_ms: performance.now() - startedRun,
    backend: await window.__localAgentReady.then((value) => value.backend),
    mobile_guard_enabled: window.__localAgentMobileLexicalGuardEnabled ?? null,
    tool_pool_size: window.META_FOR_MOBILE_TASKS?.length || null,
    summary: mobileTaskSummary(records),
    records,
  };
  LAST_MOBILE_TASK_RUN = result;
  $("mobile-task-progress").textContent =
    `${result.summary.closed_loop_success_rate * 100}% closed-loop success · ${result.backend}`;
  $("mobile-task-output").textContent = JSON.stringify(result.summary, null, 2);
  $("download-mobile-tasks").disabled = false;
  return result;
}

async function mobileTaskDownload() {
  if (!LAST_MOBILE_TASK_RUN) return;
  const blob = new Blob([JSON.stringify(LAST_MOBILE_TASK_RUN, null, 2)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${MOBILE_TASK_SUITE_ID}.json`;
  link.click();
  URL.revokeObjectURL(link.href);
}

async function mobileTaskInit() {
  const readyStarted = performance.now();
  try {
    await window.__localAgentReady;
    MOBILE_TASK_MODEL_READY_MS = performance.now() - readyStarted;
    window.META_FOR_MOBILE_TASKS = Array.isArray(window.META_FOR_MOBILE_TASKS)
      ? window.META_FOR_MOBILE_TASKS : [];
    // app.js keeps META private; use the same public action call and infer the catalog from the
    // exported bundle metadata fetched by this page, so independent validation stays separate.
    const response = await fetch("meta.json");
    window.META_FOR_MOBILE_TASKS = (await response.json()).tools || [];
    $("start-mobile-tasks").disabled = false;
    $("start-mobile-tasks").addEventListener("click", runMobileTaskPilot, { once: true });
    $("download-mobile-tasks").addEventListener("click", mobileTaskDownload);
    await runMobileTaskPilot();
  } catch (error) {
    $("mobile-task-progress").textContent = `error: ${error.message}`;
    throw error;
  }
}

if (typeof window !== "undefined") {
  window.__localAgentMobileTaskRun = runMobileTaskPilot;
  window.__localAgentMobileTaskResult = () => LAST_MOBILE_TASK_RUN;
}
if (typeof window !== "undefined" && typeof document !== "undefined") mobileTaskInit();

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    MOBILE_TASKS,
    mobileTaskCanonical,
    mobileTaskExact,
    mobileTaskInitialState,
    mobileTaskSummary,
  };
}
