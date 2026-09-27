/* Deterministic, single-step DOM microtask harness for LocalAgent.
 *
 * Each measured loop is:
 *   fresh versioned fixture -> structured action policy -> independent schema validation ->
 *   local semantic DOM dispatch -> next paint -> exact action + final DOM scoring.
 *
 * Events created here are deliberately synthetic (`isTrusted === false`). The harness measures
 * same-page DOM behavior, not OS input, visual grounding, multi-step planning, or browser-wide
 * automation.
 */

"use strict";

const BROWSER_TASK_FIXTURE_CONTRACT_VERSION = 1;
const BROWSER_TASK_DEADLINES_MS = [100, 250, 500, 1000, 2000];
const BROWSER_TASK_ACTION_TIMEOUT_MS = 10_000;
const BROWSER_TASK_SUITE_IDENTITY = Object.freeze({
  file: "browser-task-cases.json",
  bytes: 6285,
  sha256: "4c46b5b347257b81e716ec0a20a6c6116df716466e1ba8e8a74a117bb5708971",
  identity_source:
    "configs/data/pretrain-paper.yaml:evaluation_decontamination/local-browser-tasks",
});
const BROWSER_TASK_SUPPORTED_TOOLS = new Set([
  "click",
  "double_click",
  "type_text",
  "key_press",
  "scroll",
  "drag",
  "move_cursor",
  "open_url",
]);

let BROWSER_TASK_SUITE = null;
let BROWSER_TASK_SUITE_BYTE_EVIDENCE = null;
let BROWSER_TASK_MODEL_READY_MS = null;
let LAST_BROWSER_TASK_RUN = null;
let ACTIVE_BROWSER_TASK_FIXTURE = null;

function browserTaskWithWatchdog(operation, timeoutMs = BROWSER_TASK_ACTION_TIMEOUT_MS) {
  let timeoutId = null;
  const timeout = new Promise((resolve, reject) => {
    timeoutId = setTimeout(() => {
      const error = new Error(
        `Policy call exceeded the ${timeoutMs} ms watchdog; aborting the entire page run.`
      );
      error.name = "BrowserTaskActionTimeoutError";
      error.code = "browser_task_action_timeout";
      error.timeout_ms = timeoutMs;
      reject(error);
    }, timeoutMs);
  });
  const pending = Promise.resolve().then(operation);
  return Promise.race([pending, timeout]).finally(() => clearTimeout(timeoutId));
}

function browserTaskIsActionTimeout(error) {
  return error?.code === "browser_task_action_timeout";
}

function browserTaskCanonicalJson(value) {
  if (Array.isArray(value)) {
    return `[${value.map(browserTaskCanonicalJson).join(",")}]`;
  }
  if (value && typeof value === "object") {
    const pairs = Object.keys(value).sort().map((key) =>
      `${JSON.stringify(key)}:${browserTaskCanonicalJson(value[key])}`
    );
    return `{${pairs.join(",")}}`;
  }
  return JSON.stringify(value);
}

async function browserTaskSha256(text) {
  if (!crypto.subtle) return null;
  const bytes = new TextEncoder().encode(text);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(
    new Uint8Array(digest),
    (value) => value.toString(16).padStart(2, "0")
  ).join("");
}

function browserTaskSeededRandom(seedText) {
  let state = 2166136261;
  for (let index = 0; index < seedText.length; index++) {
    state ^= seedText.charCodeAt(index);
    state = Math.imul(state, 16777619);
  }
  return function random() {
    state += 0x6D2B79F5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function browserTaskShuffle(cases, seedText) {
  const shuffled = [...cases];
  const random = browserTaskSeededRandom(seedText);
  for (let index = shuffled.length - 1; index > 0; index--) {
    const other = Math.floor(random() * (index + 1));
    [shuffled[index], shuffled[other]] = [shuffled[other], shuffled[index]];
  }
  return shuffled;
}

function browserTaskPercentile(values, quantile) {
  if (!values.length) return null;
  const ordered = [...values].sort((left, right) => left - right);
  const position = (ordered.length - 1) * quantile;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return ordered[lower];
  return ordered[lower] * (upper - position) + ordered[upper] * (position - lower);
}

function browserTaskLatencySummary(values) {
  const finite = values.filter((value) => Number.isFinite(value));
  if (!finite.length) {
    return {
      count: 0,
      min: null,
      mean: null,
      p50: null,
      p90: null,
      p95: null,
      p99: null,
      max: null,
    };
  }
  const sum = finite.reduce((total, value) => total + value, 0);
  return {
    count: finite.length,
    min: Math.min(...finite),
    mean: sum / finite.length,
    p50: browserTaskPercentile(finite, 0.50),
    p90: browserTaskPercentile(finite, 0.90),
    p95: browserTaskPercentile(finite, 0.95),
    p99: browserTaskPercentile(finite, 0.99),
    max: Math.max(...finite),
  };
}

function browserTaskRate(records, key) {
  if (!records.length) return 0;
  return records.filter((record) => record.score[key] === true).length / records.length;
}

function browserTaskContextAudit(records, targetInputTokens) {
  let missing = 0;
  let mismatched = 0;
  for (const record of records) {
    if (!Number.isInteger(record.input_tokens)) {
      missing += 1;
      continue;
    }
    const matchesLength = targetInputTokens == null
      ? record.input_tokens === record.natural_input_tokens &&
        record.context_padding_tokens === 0
      : record.input_tokens === targetInputTokens &&
        record.natural_input_tokens + record.context_padding_tokens === record.input_tokens;
    const matchesFeatureContract =
      record.decision_input_tokens === record.natural_input_tokens &&
      record.decision_feature_index === record.natural_input_tokens - 1 &&
      record.context_padding_placement === (
        targetInputTokens == null ? "none" : "after_natural_assistant_marker"
      );
    if (!matchesLength || !matchesFeatureContract) mismatched += 1;
  }
  return {
    requested_input_tokens: targetInputTokens,
    verified_records: records.length - missing,
    missing_records: missing,
    mismatched_records: mismatched,
  };
}

function browserTaskSummarize(records) {
  if (!records.length) throw new Error("At least one measured DOM action record is required.");
  const harnessLatencies = records.map(
    (record) => record.latency_ms.harness_ttfa_ms
  );
  if (harnessLatencies.some((value) => !Number.isFinite(value) || value < 0)) {
    throw new Error(
      "Every measured DOM opportunity must retain a finite non-negative harness TTFA."
    );
  }
  const latencyKeys = [
    "harness_ttfa_ms",
    "runtime_ttfa_ms",
    "independent_validate_ms",
    "model_wall_ms",
    "tool_ms",
    "paint_wait_ms",
    "closed_loop_ms",
  ];
  const latency = {};
  for (const key of latencyKeys) {
    latency[key] = browserTaskLatencySummary(records.map((record) => record.latency_ms[key]));
  }

  const byAction = {};
  for (const tool of BROWSER_TASK_SUPPORTED_TOOLS) {
    const actionRecords = records.filter((record) => record.expected.tool === tool);
    if (!actionRecords.length) continue;
    byAction[tool] = {
      count: actionRecords.length,
      exact_action_rate: browserTaskRate(actionRecords, "exact_action"),
      schema_valid_rate: browserTaskRate(actionRecords, "schema_valid"),
      final_dom_rate: browserTaskRate(actionRecords, "final_dom_valid"),
      closed_loop_success_rate: browserTaskRate(actionRecords, "closed_loop_success"),
      closed_loop_ms: browserTaskLatencySummary(
        actionRecords.map((record) => record.latency_ms.closed_loop_ms)
      ),
    };
  }

  const totalHarnessMs = harnessLatencies.reduce((total, value) => total + value, 0);
  const deadlineAttainment = {};
  for (const deadline of BROWSER_TASK_DEADLINES_MS) {
    const onTime = records.filter(
      (record) => record.latency_ms.harness_ttfa_ms <= deadline
    );
    const useful = onTime.filter(
      (record) => record.score.exact_action && record.score.schema_valid
    );
    deadlineAttainment[deadline] = {
      opportunities: records.length,
      on_time: onTime.length,
      on_time_rate: onTime.length / records.length,
      useful: useful.length,
      success_at_deadline: useful.length / records.length,
      useful_actions_per_minute:
        totalHarnessMs > 0 ? useful.length / (totalHarnessMs / 60000) : null,
    };
  }

  return {
    records: records.length,
    exact_tool_rate: browserTaskRate(records, "exact_tool"),
    exact_args_rate: browserTaskRate(records, "exact_args"),
    exact_action_rate: browserTaskRate(records, "exact_action"),
    schema_valid_rate: browserTaskRate(records, "schema_valid"),
    final_dom_rate: browserTaskRate(records, "final_dom_valid"),
    state_transition_rate: browserTaskRate(records, "state_transition"),
    closed_loop_success_rate: browserTaskRate(records, "closed_loop_success"),
    latency_ms: latency,
    deadline_attainment_ms: deadlineAttainment,
    by_action: byAction,
  };
}

function browserTaskCreateElement(tag, attributes = {}, text = null) {
  const element = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) {
    if (name === "className") {
      element.className = value;
    } else if (name === "dataset") {
      for (const [dataName, dataValue] of Object.entries(value)) {
        element.dataset[dataName] = dataValue;
      }
    } else if (name in element && name !== "style") {
      element[name] = value;
    } else {
      element.setAttribute(name, value);
    }
  }
  if (text != null) element.textContent = text;
  return element;
}

function browserTaskFixtureIntro(title, description) {
  const intro = browserTaskCreateElement("div", { className: "browser-task-fixture-copy" });
  intro.append(
    browserTaskCreateElement("strong", {}, title),
    browserTaskCreateElement("span", {}, description)
  );
  return intro;
}

function browserTaskSemanticElement(tag, id, label, text) {
  return browserTaskCreateElement(
    tag,
    {
      id,
      dataset: { semanticTarget: label },
      "aria-label": label,
    },
    text
  );
}

function browserTaskClearFixture() {
  if (ACTIVE_BROWSER_TASK_FIXTURE?.cleanup) {
    ACTIVE_BROWSER_TASK_FIXTURE.cleanup();
  }
  ACTIVE_BROWSER_TASK_FIXTURE = null;
  $("browser-task-stage").replaceChildren(
    browserTaskCreateElement(
      "p",
      { className: "browser-task-placeholder" },
      "No fixture is active."
    )
  );
}

function browserTaskBuildConfirmFixture(stage) {
  const button = browserTaskSemanticElement(
    "button",
    "fixture-confirm",
    "the Confirm button",
    "Confirm"
  );
  button.className = "browser-task-fixture-button";
  button.dataset.state = "idle";
  button.addEventListener("click", () => {
    button.dataset.state = "confirmed";
    button.textContent = "Confirmed";
  });
  stage.append(
    browserTaskFixtureIntro("Confirmation", "The button changes state after a click event."),
    button
  );
}

function browserTaskBuildBackFixture(stage) {
  const button = browserTaskSemanticElement(
    "button",
    "fixture-back",
    "the Back arrow",
    "← Back"
  );
  button.className = "browser-task-fixture-button";
  button.setAttribute("aria-expanded", "false");
  button.addEventListener("dblclick", () => {
    button.setAttribute("aria-expanded", "true");
    button.textContent = "← History opened";
  });
  stage.append(
    browserTaskFixtureIntro("History", "The history drawer opens only on dblclick."),
    button
  );
}

function browserTaskBuildFeedbackFixture(stage) {
  const label = browserTaskCreateElement("label", { htmlFor: "fixture-feedback" }, "Feedback");
  const input = browserTaskCreateElement("input", {
    id: "fixture-feedback",
    className: "browser-task-fixture-input",
    type: "text",
    value: "",
    dataset: { inputSeen: "false" },
  });
  input.addEventListener("input", () => {
    input.dataset.inputSeen = "true";
  });
  stage.append(
    browserTaskFixtureIntro("Feedback form", "The empty field is focused before inference."),
    label,
    input
  );
  input.focus();
}

function browserTaskBuildEscapeFixture(stage) {
  const input = browserTaskCreateElement("input", {
    id: "fixture-key-target",
    className: "browser-task-fixture-input",
    type: "text",
    value: "focus is here",
  });
  const dialog = browserTaskCreateElement(
    "div",
    {
      id: "fixture-dialog",
      className: "browser-task-dialog",
      role: "dialog",
      hidden: false,
    },
    "Press Escape to close this local dialog."
  );
  input.addEventListener("keydown", (event) => {
    input.dataset.lastKey = event.key;
    if (event.key === "Escape") dialog.hidden = true;
  });
  stage.append(
    browserTaskFixtureIntro("Keyboard fixture", "The focused field closes the dialog on Escape."),
    input,
    dialog
  );
  input.focus();
}

function browserTaskBuildScrollFixture(stage) {
  const pane = browserTaskCreateElement("div", {
    id: "fixture-scroll",
    className: "browser-task-scroll-pane",
    tabIndex: 0,
  });
  const content = browserTaskCreateElement("div", { className: "browser-task-scroll-content" });
  content.append(
    browserTaskCreateElement("p", {}, "Top of the local document"),
    browserTaskCreateElement("p", {}, "Middle content"),
    browserTaskCreateElement("p", {}, "Bottom of the local document")
  );
  pane.append(content);
  pane.addEventListener("scroll", () => {
    if (pane.scrollTop > 0) pane.dataset.scrollDirection = "down";
  });
  stage.append(
    browserTaskFixtureIntro("Scrollable document", "The pane starts at scrollTop = 0."),
    pane
  );
  pane.focus();
}

function browserTaskBuildDragFixture(stage) {
  const board = browserTaskCreateElement("div", { className: "browser-task-drag-board" });
  const sourceZone = browserTaskCreateElement(
    "div",
    { id: "fixture-inbox", className: "browser-task-drop-zone" },
    "Inbox"
  );
  const doneZone = browserTaskSemanticElement(
    "div",
    "fixture-done",
    "the Done button",
    "Done"
  );
  doneZone.className = "browser-task-drop-zone";
  const bell = browserTaskSemanticElement(
    "button",
    "fixture-bell",
    "the notification bell",
    "🔔 Notification"
  );
  bell.className = "browser-task-drag-card";
  bell.draggable = true;
  sourceZone.append(bell);
  board.append(sourceZone, doneZone);

  bell.addEventListener("dragstart", (event) => {
    event.dataTransfer?.setData("text/plain", bell.id);
  });
  doneZone.addEventListener("dragover", (event) => event.preventDefault());
  doneZone.addEventListener("drop", (event) => {
    event.preventDefault();
    const sourceId = event.dataTransfer?.getData("text/plain") || bell.id;
    const source = document.getElementById(sourceId);
    if (source) doneZone.append(source);
    doneZone.dataset.dropSeen = "true";
  });
  stage.append(
    browserTaskFixtureIntro("Notification board", "The bell begins in Inbox."),
    board
  );
}

function browserTaskBuildBookmarkFixture(stage) {
  const bookmark = browserTaskSemanticElement(
    "button",
    "fixture-bookmark",
    "the Bookmark star",
    "☆ Bookmark"
  );
  bookmark.className = "browser-task-fixture-button";
  const markPointer = () => {
    bookmark.dataset.pointerSeen = "true";
    bookmark.textContent = "★ Bookmark";
  };
  bookmark.addEventListener("pointermove", markPointer);
  bookmark.addEventListener("mousemove", markPointer);
  stage.append(
    browserTaskFixtureIntro(
      "Bookmark control",
      "A synthetic pointer/mouse move changes the local control."
    ),
    bookmark
  );
}

async function browserTaskBuildNavigationFixture(stage) {
  const frame = browserTaskCreateElement("iframe", {
    id: "fixture-navigation-frame",
    className: "browser-task-navigation-frame",
    title: "Disposable local navigation fixture",
    sandbox: "allow-same-origin",
  });
  const loaded = new Promise((resolve, reject) => {
    const timeout = setTimeout(
      () => reject(new Error("Local navigation fixture timed out.")),
      3000
    );
    frame.addEventListener("load", () => {
      clearTimeout(timeout);
      resolve();
    }, { once: true });
  });
  frame.srcdoc = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <style>
    body { margin: 0; padding: 16px; color: #e8e8f2; background: #0f1020;
           font: 15px/1.5 system-ui, sans-serif; }
    a { color: #5ad1c0; }
    section { margin-top: 12px; padding: 12px; border: 1px solid #232450; border-radius: 8px; }
    [hidden] { display: none; }
  </style>
</head>
<body data-route="home">
  <strong>Local fixture site</strong>
  <a id="details-link" href="about:srcdoc#details"
     data-local-url="fixture.local/details">Details</a>
  <section id="home-page">Home route</section>
  <section id="details-page" data-active="false" hidden>Details route</section>
</body>
</html>`;
  stage.append(
    browserTaskFixtureIntro(
      "Local navigation",
      "open_url is restricted to the link inside this disposable iframe."
    ),
    frame
  );
  await loaded;

  const frameWindow = frame.contentWindow;
  const frameDocument = frame.contentDocument;
  if (!frameWindow || !frameDocument) {
    throw new Error("Local navigation fixture is not same-origin.");
  }
  const activateRoute = () => {
    const details = frameDocument.getElementById("details-page");
    const home = frameDocument.getElementById("home-page");
    const isDetails = frameWindow.location.hash === "#details";
    frameDocument.body.dataset.route = isDetails ? "details" : "home";
    home.hidden = isDetails;
    details.hidden = !isDetails;
    details.dataset.active = String(isDetails);
  };
  frameWindow.addEventListener("hashchange", activateRoute);
  activateRoute();
  return () => frameWindow.removeEventListener("hashchange", activateRoute);
}

const BROWSER_TASK_FIXTURE_BUILDERS = new Map([
  ["confirm-button", browserTaskBuildConfirmFixture],
  ["back-arrow", browserTaskBuildBackFixture],
  ["feedback-field", browserTaskBuildFeedbackFixture],
  ["escape-dialog", browserTaskBuildEscapeFixture],
  ["scroll-pane", browserTaskBuildScrollFixture],
  ["notification-drag", browserTaskBuildDragFixture],
  ["bookmark-hover", browserTaskBuildBookmarkFixture],
  ["local-navigation", browserTaskBuildNavigationFixture],
]);

async function browserTaskRenderFixture(benchmarkCase) {
  browserTaskClearFixture();
  const fixture = benchmarkCase.fixture;
  if (fixture.version !== BROWSER_TASK_FIXTURE_CONTRACT_VERSION) {
    throw new Error(
      `Fixture ${fixture.id} requests v${fixture.version}; ` +
      `this harness implements v${BROWSER_TASK_FIXTURE_CONTRACT_VERSION}.`
    );
  }
  const builder = BROWSER_TASK_FIXTURE_BUILDERS.get(fixture.id);
  if (!builder) throw new Error(`Unknown local fixture ${fixture.id}.`);

  const stage = $("browser-task-stage");
  stage.replaceChildren();
  stage.dataset.fixtureId = fixture.id;
  stage.dataset.fixtureVersion = String(fixture.version);
  $("browser-task-current-prompt").textContent = benchmarkCase.query;
  $("browser-task-fixture-badge").textContent = `${fixture.id} · v${fixture.version}`;
  const cleanup = await builder(stage);
  ACTIVE_BROWSER_TASK_FIXTURE = {
    id: fixture.id,
    version: fixture.version,
    cleanup: typeof cleanup === "function" ? cleanup : null,
  };
}

function browserTaskAssertionDocument(assertion) {
  if (assertion.document === "main") {
    return { document, window };
  }
  if (assertion.document?.startsWith("frame:")) {
    const selector = assertion.document.slice("frame:".length);
    const frame = $("browser-task-stage").querySelector(selector);
    if (!frame) throw new Error(`Assertion frame not found: ${selector}`);
    if (!frame.contentDocument || !frame.contentWindow) {
      throw new Error(`Assertion frame is inaccessible: ${selector}`);
    }
    return { document: frame.contentDocument, window: frame.contentWindow };
  }
  throw new Error(`Unsupported assertion document ${assertion.document}.`);
}

function browserTaskInspectAssertion(assertion) {
  try {
    const context = browserTaskAssertionDocument(assertion);
    if (assertion.kind === "location_hash") {
      const actual = context.window.location.hash;
      return { ...assertion, found: true, actual, passed: actual === assertion.equals };
    }

    const element = context.document.querySelector(assertion.selector);
    if (!element) {
      return {
        ...assertion,
        found: false,
        actual: null,
        passed: false,
        error: `Selector not found: ${assertion.selector}`,
      };
    }

    let actual;
    if (assertion.kind === "attribute") {
      actual = element.getAttribute(assertion.name);
    } else if (assertion.kind === "property") {
      actual = element[assertion.name];
    } else if (assertion.kind === "parent") {
      actual = element.parentElement?.id ? `#${element.parentElement.id}` : null;
    } else {
      throw new Error(`Unsupported DOM assertion kind ${assertion.kind}.`);
    }

    const passed = Object.hasOwn(assertion, "greater_than")
      ? typeof actual === "number" && actual > assertion.greater_than
      : actual === assertion.equals;
    return { ...assertion, found: true, actual, passed };
  } catch (error) {
    return {
      ...assertion,
      found: false,
      actual: null,
      passed: false,
      error: error.message,
    };
  }
}

function browserTaskInspectState(benchmarkCase) {
  const assertions = benchmarkCase.expected_dom.map(browserTaskInspectAssertion);
  return {
    passed: assertions.every((assertion) => assertion.passed),
    assertions,
  };
}

function browserTaskAssertionValues(state) {
  return state.assertions.map((assertion) => ({
    document: assertion.document,
    selector: assertion.selector || null,
    kind: assertion.kind,
    actual: assertion.actual,
  }));
}

function browserTaskJsonValue(value) {
  if (value === undefined) return null;
  try {
    return JSON.parse(JSON.stringify(value));
  } catch {
    return null;
  }
}

function browserTaskNormalizeAction(action) {
  if (action?.abstain === true) return { abstain: true };
  return {
    tool: typeof action?.tool === "string" ? action.tool : null,
    args: browserTaskJsonValue(action?.args),
  };
}

function browserTaskParseEvidence(action, modelError = null) {
  return {
    policy: action?.policy ?? null,
    inference_passes: action?.inference_passes ?? null,
    parse_kind: action?.parse_kind ?? null,
    parse_failure: action?.parse_failure === true || modelError != null,
    parse_error: action?.parse_error ?? browserTaskErrorRecord(modelError)?.message ?? null,
    runtime_validation_failure: action?.validation_failure === true,
    runtime_validation_error: action?.validation_error ?? null,
    runtime_error: browserTaskErrorRecord(modelError),
  };
}

function browserTaskValueMatchesSchema(value, schema, path, errors) {
  if (schema.enum && !schema.enum.some(
    (candidate) => browserTaskCanonicalJson(candidate) === browserTaskCanonicalJson(value)
  )) {
    errors.push(`${path} is not in the declared enum.`);
  }
  if (Object.hasOwn(schema, "const") &&
      browserTaskCanonicalJson(value) !== browserTaskCanonicalJson(schema.const)) {
    errors.push(`${path} does not equal the declared constant.`);
  }

  const type = schema.type;
  const typeMatches =
    !type ||
    (type === "object" && value !== null && typeof value === "object" &&
      !Array.isArray(value)) ||
    (type === "array" && Array.isArray(value)) ||
    (type === "string" && typeof value === "string") ||
    (type === "integer" && Number.isInteger(value)) ||
    (type === "number" && typeof value === "number" && Number.isFinite(value)) ||
    (type === "boolean" && typeof value === "boolean") ||
    (type === "null" && value === null);
  if (!typeMatches) {
    errors.push(`${path} does not have JSON Schema type ${type}.`);
    return;
  }

  if (type === "object") {
    const properties = schema.properties || {};
    for (const required of schema.required || []) {
      if (!Object.hasOwn(value, required)) errors.push(`${path}.${required} is required.`);
    }
    for (const [key, child] of Object.entries(value)) {
      if (properties[key]) {
        browserTaskValueMatchesSchema(child, properties[key], `${path}.${key}`, errors);
      } else if (schema.additionalProperties !== true) {
        errors.push(`${path}.${key} is not declared by the tool schema.`);
      }
    }
  }
  if (type === "array") {
    if (Number.isInteger(schema.minItems) && value.length < schema.minItems) {
      errors.push(`${path} has fewer than ${schema.minItems} items.`);
    }
    if (Number.isInteger(schema.maxItems) && value.length > schema.maxItems) {
      errors.push(`${path} has more than ${schema.maxItems} items.`);
    }
    if (schema.items) {
      value.forEach((item, index) =>
        browserTaskValueMatchesSchema(item, schema.items, `${path}[${index}]`, errors)
      );
    }
  }
  if (type === "string") {
    if (Number.isInteger(schema.minLength) && value.length < schema.minLength) {
      errors.push(`${path} is shorter than minLength ${schema.minLength}.`);
    }
    if (Number.isInteger(schema.maxLength) && value.length > schema.maxLength) {
      errors.push(`${path} is longer than maxLength ${schema.maxLength}.`);
    }
    if (schema.pattern) {
      try {
        if (!(new RegExp(schema.pattern)).test(value)) {
          errors.push(`${path} does not match the declared pattern.`);
        }
      } catch {
        errors.push(`${path} has an invalid schema pattern.`);
      }
    }
  }
  if (type === "integer" || type === "number") {
    if (Number.isFinite(schema.minimum) && value < schema.minimum) {
      errors.push(`${path} is below minimum ${schema.minimum}.`);
    }
    if (Number.isFinite(schema.maximum) && value > schema.maximum) {
      errors.push(`${path} is above maximum ${schema.maximum}.`);
    }
  }
}

function browserTaskValidateActionSchema(action) {
  const normalized = browserTaskNormalizeAction(action);
  const errors = [];
  if (normalized.abstain === true) {
    return {
      validator: "browser-task-json-schema-subset-v2",
      valid: false,
      errors: ["Abstention has no executable tool schema for this task suite."],
      schema_tool: null,
      tool_schema: null,
    };
  }
  const spec = (META.tools || []).find((tool) => tool.name === normalized.tool);
  if (!spec) {
    return {
      validator: "browser-task-json-schema-subset-v2",
      valid: false,
      errors: [`Unknown tool ${JSON.stringify(normalized.tool)}.`],
      schema_tool: null,
      tool_schema: null,
    };
  }
  const schema = browserTaskJsonValue(spec.schema || {});
  browserTaskValueMatchesSchema(normalized.args, schema, "$.args", errors);
  return {
    validator: "browser-task-json-schema-subset-v2",
    valid: errors.length === 0,
    errors,
    schema_tool: spec.name,
    tool_schema: schema,
  };
}

function browserTaskScoreAction(action, benchmarkCase, schemaResult) {
  const normalized = browserTaskNormalizeAction(action);
  const expected = benchmarkCase.expected;
  const exactTool = normalized.abstain !== true && normalized.tool === expected.tool;
  const exactArgs = exactTool &&
    browserTaskCanonicalJson(normalized.args) === browserTaskCanonicalJson(expected.args);
  let appSchemaValid = false;
  if (schemaResult.tool_schema && normalized.args) {
    try {
      appSchemaValid = groundedArgsValid(normalized.args, schemaResult.tool_schema);
    } catch {
      appSchemaValid = false;
    }
  }
  return {
    exact_tool: exactTool,
    exact_args: exactArgs,
    exact_action: exactTool && exactArgs,
    schema_valid: schemaResult.valid,
    app_schema_valid_diagnostic: appSchemaValid,
    schema_validator_agreement: schemaResult.valid === appSchemaValid,
  };
}

function browserTaskNormalizeTarget(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/^the\s+/, "")
    .replace(/\s+/g, " ");
}

function browserTaskResolveTarget(label, root = document) {
  const wanted = browserTaskNormalizeTarget(label);
  const candidates = root.querySelectorAll("[data-semantic-target]");
  for (const candidate of candidates) {
    if (browserTaskNormalizeTarget(candidate.dataset.semanticTarget) === wanted) {
      return candidate;
    }
  }
  throw new Error(`No local semantic target matches ${JSON.stringify(label)}.`);
}

function browserTaskMouseEvent(type, options = {}) {
  return new MouseEvent(type, {
    bubbles: true,
    cancelable: true,
    composed: true,
    view: window,
    button: 0,
    buttons: type === "mousedown" ? 1 : 0,
    ...options,
  });
}

function browserTaskDispatchClick(element, detail = 1) {
  element.focus?.();
  if (typeof PointerEvent === "function") {
    element.dispatchEvent(new PointerEvent("pointerdown", {
      bubbles: true,
      cancelable: true,
      pointerId: 1,
      pointerType: "mouse",
      isPrimary: true,
      button: 0,
      buttons: 1,
      detail,
    }));
  }
  element.dispatchEvent(browserTaskMouseEvent("mousedown", { detail }));
  if (typeof PointerEvent === "function") {
    element.dispatchEvent(new PointerEvent("pointerup", {
      bubbles: true,
      cancelable: true,
      pointerId: 1,
      pointerType: "mouse",
      isPrimary: true,
      button: 0,
      buttons: 0,
      detail,
    }));
  }
  element.dispatchEvent(browserTaskMouseEvent("mouseup", { detail }));
  element.dispatchEvent(browserTaskMouseEvent("click", { detail }));
}

function browserTaskSetInputValue(element, value) {
  const prototype = element instanceof HTMLTextAreaElement
    ? HTMLTextAreaElement.prototype
    : HTMLInputElement.prototype;
  const setter = Object.getOwnPropertyDescriptor(prototype, "value")?.set;
  if (setter) setter.call(element, value);
  else element.value = value;
}

function browserTaskCreateDragEvent(type, dataTransfer) {
  try {
    return new DragEvent(type, {
      bubbles: true,
      cancelable: true,
      dataTransfer,
    });
  } catch {
    const event = new Event(type, { bubbles: true, cancelable: true });
    Object.defineProperty(event, "dataTransfer", { value: dataTransfer });
    return event;
  }
}

function browserTaskDataTransfer() {
  try {
    return new DataTransfer();
  } catch {
    const values = new Map();
    return {
      dropEffect: "move",
      effectAllowed: "all",
      setData: (type, value) => values.set(type, String(value)),
      getData: (type) => values.get(type) || "",
      clearData: (type) => type ? values.delete(type) : values.clear(),
    };
  }
}

function browserTaskNormalizeLocalUrl(value) {
  const raw = String(value || "").trim();
  const parsed = new URL(raw.includes("://") ? raw : `https://${raw}`);
  const path = parsed.pathname.replace(/\/+$/, "") || "/";
  return `${parsed.hostname.toLowerCase()}${path}`;
}

function browserTaskDispatchAction(action) {
  if (!action || action.abstain) {
    throw new Error("No executable action was produced.");
  }
  if (!BROWSER_TASK_SUPPORTED_TOOLS.has(action.tool)) {
    throw new Error(`Tool ${JSON.stringify(action.tool)} is outside this local DOM harness.`);
  }
  const args = action.args || {};
  const eventLog = [];

  if (action.tool === "click") {
    const target = browserTaskResolveTarget(args.target, $("browser-task-stage"));
    browserTaskDispatchClick(target);
    eventLog.push("pointerdown", "mousedown", "pointerup", "mouseup", "click");
    return {
      ok: true,
      synthetic_events: true,
      target: target.dataset.semanticTarget,
      events: eventLog,
    };
  }

  if (action.tool === "double_click") {
    const target = browserTaskResolveTarget(args.target, $("browser-task-stage"));
    browserTaskDispatchClick(target, 1);
    browserTaskDispatchClick(target, 2);
    target.dispatchEvent(browserTaskMouseEvent("dblclick", { detail: 2 }));
    eventLog.push("click", "click", "dblclick");
    return {
      ok: true,
      synthetic_events: true,
      target: target.dataset.semanticTarget,
      events: eventLog,
    };
  }

  if (action.tool === "type_text") {
    if (typeof args.text !== "string") throw new Error("type_text requires a string text value.");
    const target = document.activeElement;
    if (!(target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement)) {
      throw new Error("The local fixture has no focused text field.");
    }
    let value = "";
    for (const character of args.text) {
      const beforeInput = typeof InputEvent === "function"
        ? new InputEvent("beforeinput", {
            bubbles: true,
            cancelable: true,
            data: character,
            inputType: "insertText",
          })
        : new Event("beforeinput", { bubbles: true, cancelable: true });
      if (!target.dispatchEvent(beforeInput)) continue;
      value += character;
      browserTaskSetInputValue(target, value);
      const input = typeof InputEvent === "function"
        ? new InputEvent("input", {
            bubbles: true,
            data: character,
            inputType: "insertText",
          })
        : new Event("input", { bubbles: true });
      target.dispatchEvent(input);
    }
    return {
      ok: true,
      synthetic_events: true,
      target: `#${target.id}`,
      events: ["beforeinput", "input"],
      characters: Array.from(args.text).length,
    };
  }

  if (action.tool === "key_press") {
    if (typeof args.key !== "string") throw new Error("key_press requires a string key.");
    const target = document.activeElement || $("browser-task-stage");
    const key = args.key === "Space" ? " " : args.key;
    const code = args.key === "Space" ? "Space" : args.key;
    target.dispatchEvent(new KeyboardEvent("keydown", {
      key,
      code,
      bubbles: true,
      cancelable: true,
    }));
    target.dispatchEvent(new KeyboardEvent("keyup", {
      key,
      code,
      bubbles: true,
      cancelable: true,
    }));
    return {
      ok: true,
      synthetic_events: true,
      target: target.id ? `#${target.id}` : target.tagName,
      events: ["keydown", "keyup"],
      key: args.key,
    };
  }

  if (action.tool === "scroll") {
    const pane = $("browser-task-stage").querySelector("#fixture-scroll");
    if (!pane) throw new Error("The active fixture has no local scroll pane.");
    const horizontal = args.direction === "left" || args.direction === "right";
    const sign = args.direction === "up" || args.direction === "left" ? -1 : 1;
    if (!["up", "down", "left", "right"].includes(args.direction)) {
      throw new Error(`Unsupported scroll direction ${JSON.stringify(args.direction)}.`);
    }
    const amount = Math.max(
      48,
      horizontal ? pane.clientWidth * 0.8 : pane.clientHeight * 0.8
    );
    if (horizontal) pane.scrollLeft += sign * amount;
    else pane.scrollTop += sign * amount;
    pane.dispatchEvent(new Event("scroll", { bubbles: false }));
    return {
      ok: true,
      synthetic_events: true,
      target: "#fixture-scroll",
      events: ["scroll"],
      direction: args.direction,
      scroll_top: pane.scrollTop,
      scroll_left: pane.scrollLeft,
    };
  }

  if (action.tool === "drag") {
    const source = browserTaskResolveTarget(args.source, $("browser-task-stage"));
    const destination = browserTaskResolveTarget(args.dest, $("browser-task-stage"));
    const dataTransfer = browserTaskDataTransfer();
    dataTransfer.setData("text/plain", source.id);
    for (const [target, eventName] of [
      [source, "dragstart"],
      [destination, "dragenter"],
      [destination, "dragover"],
      [destination, "drop"],
      [source, "dragend"],
    ]) {
      target.dispatchEvent(browserTaskCreateDragEvent(eventName, dataTransfer));
      eventLog.push(eventName);
    }
    return {
      ok: true,
      synthetic_events: true,
      source: source.dataset.semanticTarget,
      destination: destination.dataset.semanticTarget,
      events: eventLog,
    };
  }

  if (action.tool === "move_cursor") {
    const target = browserTaskResolveTarget(args.target, $("browser-task-stage"));
    if (typeof PointerEvent === "function") {
      target.dispatchEvent(new PointerEvent("pointermove", {
        bubbles: true,
        cancelable: true,
        pointerId: 1,
        pointerType: "mouse",
        isPrimary: true,
      }));
      eventLog.push("pointermove");
    }
    target.dispatchEvent(browserTaskMouseEvent("mousemove"));
    eventLog.push("mousemove");
    return {
      ok: true,
      synthetic_events: true,
      physical_cursor_moved: false,
      target: target.dataset.semanticTarget,
      events: eventLog,
    };
  }

  const frame = $("browser-task-stage").querySelector("#fixture-navigation-frame");
  if (!frame?.contentDocument) {
    throw new Error("The active fixture has no local navigation frame.");
  }
  const wantedUrl = browserTaskNormalizeLocalUrl(args.url);
  if (!wantedUrl.startsWith("fixture.local/")) {
    throw new Error(`Navigation outside fixture.local is blocked: ${JSON.stringify(args.url)}.`);
  }
  const links = Array.from(frame.contentDocument.querySelectorAll("[data-local-url]"));
  const link = links.find(
    (candidate) => browserTaskNormalizeLocalUrl(candidate.dataset.localUrl) === wantedUrl
  );
  if (!link) throw new Error(`No local fixture route matches ${JSON.stringify(args.url)}.`);
  browserTaskDispatchClick(link);
  return {
    ok: true,
    synthetic_events: true,
    browser_wide_navigation: false,
    navigation_scope: "disposable same-document iframe fragment",
    target: link.dataset.localUrl,
    events: ["pointerdown", "mousedown", "pointerup", "mouseup", "click"],
  };
}

async function browserTaskAfterNextPaint() {
  const start = performance.now();
  await new Promise((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(resolve));
  });
  return performance.now() - start;
}

function browserTaskErrorRecord(error) {
  if (!error) return null;
  return {
    name: error.name || "Error",
    message: error.message || String(error),
  };
}

async function browserTaskRunCase(
  benchmarkCase,
  repetition,
  orderIndex,
  measured,
  targetInputTokens = null
) {
  if (document.visibilityState !== "visible") {
    throw new Error(
      "Keep this benchmark tab visible so requestAnimationFrame can observe a paint."
    );
  }
  await browserTaskRenderFixture(benchmarkCase);
  await browserTaskAfterNextPaint();
  const beforeState = browserTaskInspectState(benchmarkCase);
  const fixtureClean = beforeState.assertions.every((assertion) => !assertion.passed);

  const startedAtEpochMs = Date.now();
  const closedLoopStart = performance.now();
  const modelStart = performance.now();
  let action = null;
  let modelError = null;
  try {
    action = await browserTaskWithWatchdog(
      () => callPolicyOnce(
        benchmarkCase.query,
        ACTION_POLICIES.STRUCTURED,
        { targetInputTokens }
      )
    );
  } catch (error) {
    if (browserTaskIsActionTimeout(error)) throw error;
    modelError = error;
  }
  const modelEnd = performance.now();

  const predictedAction = browserTaskNormalizeAction(action);
  const expectedAction = browserTaskJsonValue(benchmarkCase.expected);
  const parseEvidence = browserTaskParseEvidence(action, modelError);
  const validateStart = performance.now();
  const schemaResult = browserTaskValidateActionSchema(predictedAction);
  const independentValidateMs = performance.now() - validateStart;
  const actionScore = browserTaskScoreAction(predictedAction, benchmarkCase, schemaResult);
  const toolStart = performance.now();
  let execution = null;
  let toolError = null;
  try {
    execution = browserTaskDispatchAction(predictedAction);
  } catch (error) {
    toolError = error;
    execution = {
      ok: false,
      synthetic_events: true,
      events: [],
      error: browserTaskErrorRecord(error),
    };
  }
  const toolEnd = performance.now();
  const paintWaitMs = await browserTaskAfterNextPaint();
  const afterState = browserTaskInspectState(benchmarkCase);
  const closedLoopEnd = performance.now();

  const stateChanged =
    browserTaskCanonicalJson(browserTaskAssertionValues(beforeState)) !==
    browserTaskCanonicalJson(browserTaskAssertionValues(afterState));
  const score = {
    ...actionScore,
    fixture_clean: fixtureClean,
    execution_ok: execution?.ok === true,
    final_dom_valid: afterState.passed,
    state_transition: fixtureClean && stateChanged && afterState.passed,
  };
  score.closed_loop_success =
    score.exact_action &&
    score.schema_valid &&
    score.execution_ok &&
    score.state_transition;

  const runtimeTtfaMs = action?.timing?.ttfa_ms ?? (modelEnd - modelStart);
  const harnessTtfaMs = runtimeTtfaMs + independentValidateMs;
  const latency = {
    harness_ttfa_ms: harnessTtfaMs,
    runtime_ttfa_ms: runtimeTtfaMs,
    independent_validate_ms: independentValidateMs,
    model_wall_ms: modelEnd - modelStart,
    tokenize_ms: action?.timing?.tokenize_ms ?? null,
    inference_ms: action?.timing?.inference_ms ?? null,
    decode_control_ms: action?.timing?.decode_control_ms ?? null,
    model_dispatch_ms: action?.timing?.dispatch_ms ?? null,
    parse_validate_ms: action?.timing?.parse_validate_ms ?? null,
    ttft_ms: action?.timing?.ttft_ms ?? null,
    tpot_ms: action?.timing?.tpot_ms ?? null,
    tool_ms: toolEnd - toolStart,
    paint_wait_ms: paintWaitMs,
    closed_loop_ms: closedLoopEnd - closedLoopStart,
  };
  return {
    case_id: benchmarkCase.id,
    family: benchmarkCase.family,
    fixture: { ...benchmarkCase.fixture },
    query: benchmarkCase.query,
    expected: expectedAction,
    expected_action: expectedAction,
    repetition,
    order_index: orderIndex,
    measured,
    action_timeout_ms: BROWSER_TASK_ACTION_TIMEOUT_MS,
    watchdog_outcome: "completed_before_timeout",
    started_at_epoch_ms: startedAtEpochMs,
    backend: BACKEND,
    input_tokens: action?.input_tokens ?? null,
    input_bytes: action?.input_bytes ?? null,
    natural_input_tokens: action?.natural_input_tokens ?? null,
    context_padding_tokens: action?.context_padding_tokens ?? null,
    context_padding_placement: action?.context_padding_placement ?? null,
    decision_input_tokens: action?.decision_input_tokens ?? null,
    decision_feature_index: action?.decision_feature_index ?? null,
    output_tokens: action?.output_tokens ?? 0,
    predicted_action: predictedAction,
    predicted_tool: predictedAction.abstain ? null : predictedAction.tool,
    predicted_route: action?.route ?? null,
    route_confidence: action?.conf ?? null,
    expected_tool: expectedAction.tool,
    parse_evidence: parseEvidence,
    parse_failure: parseEvidence.parse_failure,
    validation_failure: !schemaResult.valid,
    success: score.exact_action,
    schema_valid: score.schema_valid,
    harness_ttfa_ms: harnessTtfaMs,
    runtime_ttfa_ms: runtimeTtfaMs,
    // Backward-compatible alias. metadata.latency_clock names the unambiguous primary field.
    ttfa_ms: harnessTtfaMs,
    tokenize_ms: latency.tokenize_ms,
    inference_ms: latency.inference_ms,
    decode_control_ms: latency.decode_control_ms,
    dispatch_ms: latency.model_dispatch_ms,
    parse_validate_ms: latency.parse_validate_ms,
    independent_validate_ms: latency.independent_validate_ms,
    ttft_ms: latency.ttft_ms,
    tpot_ms: latency.tpot_ms,
    model_error: browserTaskErrorRecord(modelError),
    independent_schema: schemaResult,
    execution,
    tool_error: browserTaskErrorRecord(toolError),
    dom_before: beforeState,
    dom_after: afterState,
    score,
    latency_ms: latency,
  };
}

function browserTaskValidateSuite(suite) {
  if (suite.schema_version !== 1) {
    throw new Error(`Unsupported browser-task suite schema ${suite.schema_version}.`);
  }
  if (suite.scope?.steps_per_case !== 1 || suite.scope?.input_modality !== "text") {
    throw new Error("Browser-task suite must declare single-step, text-only scope.");
  }
  if (suite.fixture_contract?.version !== BROWSER_TASK_FIXTURE_CONTRACT_VERSION) {
    throw new Error(
      `Suite fixture contract is v${suite.fixture_contract?.version}; ` +
      `harness implements v${BROWSER_TASK_FIXTURE_CONTRACT_VERSION}.`
    );
  }
  if (!Array.isArray(suite.cases) || !suite.cases.length) {
    throw new Error("Browser-task suite has no cases.");
  }
  const ids = new Set();
  const coveredTools = new Set();
  for (const benchmarkCase of suite.cases) {
    if (!benchmarkCase.id || ids.has(benchmarkCase.id)) {
      throw new Error(`Duplicate or missing browser-task case id ${benchmarkCase.id}.`);
    }
    ids.add(benchmarkCase.id);
    if (typeof benchmarkCase.query !== "string" || !benchmarkCase.query.trim()) {
      throw new Error(`Case ${benchmarkCase.id} has no text query.`);
    }
    if (!BROWSER_TASK_SUPPORTED_TOOLS.has(benchmarkCase.expected?.tool)) {
      throw new Error(
        `Case ${benchmarkCase.id} has unsupported tool ${benchmarkCase.expected?.tool}.`
      );
    }
    coveredTools.add(benchmarkCase.expected.tool);
    if (!benchmarkCase.expected.args ||
        typeof benchmarkCase.expected.args !== "object" ||
        Array.isArray(benchmarkCase.expected.args)) {
      throw new Error(`Case ${benchmarkCase.id} expected args must be an object.`);
    }
    if (!BROWSER_TASK_FIXTURE_BUILDERS.has(benchmarkCase.fixture?.id)) {
      throw new Error(`Case ${benchmarkCase.id} has unknown fixture ${benchmarkCase.fixture?.id}.`);
    }
    if (benchmarkCase.fixture.version !== BROWSER_TASK_FIXTURE_CONTRACT_VERSION) {
      throw new Error(`Case ${benchmarkCase.id} has an incompatible fixture version.`);
    }
    if (!Array.isArray(benchmarkCase.expected_dom) || !benchmarkCase.expected_dom.length) {
      throw new Error(`Case ${benchmarkCase.id} has no final DOM assertions.`);
    }
    for (const assertion of benchmarkCase.expected_dom) {
      const supportedKind = [
        "attribute",
        "property",
        "parent",
        "location_hash",
      ].includes(assertion.kind);
      const supportedDocument =
        assertion.document === "main" || assertion.document?.startsWith("frame:");
      const hasComparison =
        Object.hasOwn(assertion, "equals") || Object.hasOwn(assertion, "greater_than");
      if (!supportedKind || !supportedDocument || !hasComparison) {
        throw new Error(`Case ${benchmarkCase.id} has an invalid DOM assertion.`);
      }
      if (assertion.kind !== "location_hash" && !assertion.selector) {
        throw new Error(`Case ${benchmarkCase.id} has a selector-less DOM assertion.`);
      }
    }
  }
  for (const tool of BROWSER_TASK_SUPPORTED_TOOLS) {
    if (!coveredTools.has(tool)) {
      throw new Error(`Browser-task suite does not cover required action ${tool}.`);
    }
  }
}

function browserTaskValidateExpectedActions(suite) {
  for (const benchmarkCase of suite.cases) {
    const result = browserTaskValidateActionSchema(benchmarkCase.expected);
    if (!result.valid) {
      throw new Error(
        `Case ${benchmarkCase.id} expected action violates loaded tool schema: ` +
        result.errors.join(" ")
      );
    }
  }
}

async function browserTaskAdapterInfo() {
  if (!navigator.gpu) return null;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;
    const info = adapter.info || {};
    return {
      vendor: info.vendor || null,
      architecture: info.architecture || null,
      device: info.device || null,
      description: info.description || null,
      is_fallback_adapter: adapter.isFallbackAdapter || false,
    };
  } catch {
    return null;
  }
}

function browserTaskModelResourceTiming() {
  const entries = performance.getEntriesByType("resource");
  const entry = entries.find((item) => {
    try {
      return new URL(item.name, location.href).pathname.endsWith(`/${MODEL_URL}`);
    } catch {
      return item.name.endsWith(`/${MODEL_URL}`);
    }
  });
  if (!entry) return null;
  return {
    duration_ms: entry.duration,
    transfer_bytes: entry.transferSize || null,
    encoded_bytes: entry.encodedBodySize || null,
    decoded_bytes: entry.decodedBodySize || null,
    cache_or_transfer_interpretation:
      entry.transferSize === 0 ? "cache-or-opaque-response" : "network-transfer-reported",
  };
}

function browserTaskBundleArtifact(fileName) {
  if (!fileName || !BUNDLE_MANIFEST?.artifacts) return null;
  return Object.values(BUNDLE_MANIFEST.artifacts)
    .find((artifact) => artifact?.file === fileName) || null;
}

function browserTaskSetProgress(text) {
  $("browser-task-progress").textContent = text;
}

function browserTaskReadBoundedInteger(id, minimum, maximum) {
  const parsed = Number.parseInt($(id).value, 10);
  const value = Number.isFinite(parsed) ? parsed : minimum;
  return Math.min(maximum, Math.max(minimum, value));
}

function browserTaskFixed(value, digits = 1) {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : "—";
}

function browserTaskMetric(label, value) {
  const metric = browserTaskCreateElement("div", { className: "metric" });
  metric.append(
    browserTaskCreateElement("span", {}, label),
    browserTaskCreateElement("strong", {}, value)
  );
  return metric;
}

function browserTaskRenderSummary(payload) {
  const output = $("browser-task-output");
  output.replaceChildren();
  const summary = payload.summary;
  const metrics = browserTaskCreateElement("div", { className: "metric-grid" });
  metrics.append(
    browserTaskMetric(
      "Closed-loop p50",
      `${browserTaskFixed(summary.latency_ms.closed_loop_ms.p50)} ms`
    ),
    browserTaskMetric(
      "Harness TTFA p50",
      `${browserTaskFixed(summary.latency_ms.harness_ttfa_ms.p50)} ms`
    ),
    browserTaskMetric(
      "Tool p50",
      `${browserTaskFixed(summary.latency_ms.tool_ms.p50, 2)} ms`
    ),
    browserTaskMetric("Exact action", `${browserTaskFixed(summary.exact_action_rate * 100)}%`),
    browserTaskMetric("Schema valid", `${browserTaskFixed(summary.schema_valid_rate * 100)}%`),
    browserTaskMetric("Final DOM", `${browserTaskFixed(summary.final_dom_rate * 100)}%`),
    browserTaskMetric(
      "End-to-end success",
      `${browserTaskFixed(summary.closed_loop_success_rate * 100)}%`
    )
  );
  output.append(metrics);

  const table = browserTaskCreateElement("table", { className: "benchmark-table" });
  const head = browserTaskCreateElement("thead");
  const headerRow = browserTaskCreateElement("tr");
  for (const heading of ["Action", "n", "Exact", "Schema", "DOM", "Loop", "p50 loop"]) {
    headerRow.append(browserTaskCreateElement("th", {}, heading));
  }
  head.append(headerRow);
  const body = browserTaskCreateElement("tbody");
  for (const [tool, values] of Object.entries(summary.by_action)) {
    const row = browserTaskCreateElement("tr");
    const cells = [
      tool,
      String(values.count),
      `${browserTaskFixed(values.exact_action_rate * 100)}%`,
      `${browserTaskFixed(values.schema_valid_rate * 100)}%`,
      `${browserTaskFixed(values.final_dom_rate * 100)}%`,
      `${browserTaskFixed(values.closed_loop_success_rate * 100)}%`,
      `${browserTaskFixed(values.closed_loop_ms.p50)} ms`,
    ];
    for (const cell of cells) row.append(browserTaskCreateElement("td", {}, cell));
    body.append(row);
  }
  table.append(head, body);
  output.append(table);

  const details = browserTaskCreateElement("details");
  details.append(browserTaskCreateElement("summary", {}, "Summary and honest-run metadata"));
  details.append(
    browserTaskCreateElement(
      "pre",
      {},
      JSON.stringify({ metadata: payload.metadata, summary: payload.summary }, null, 2)
    )
  );
  output.append(details);
}

async function runBrowserTasks() {
  const startButton = $("start-browser-tasks");
  const downloadButton = $("download-browser-tasks");
  const warmupRecords = [];
  const records = [];
  const recordedOrder = [];
  let activeOpportunity = null;
  startButton.disabled = true;
  downloadButton.disabled = true;
  $("browser-task-output").replaceChildren();

  try {
    if (!BROWSER_TASK_SUITE) throw new Error("Browser-task suite is not loaded.");
    if (!BROWSER_TASK_SUITE_BYTE_EVIDENCE?.identity_verified) {
      throw new Error("Browser-task suite lacks verified raw-byte identity evidence.");
    }
    if (document.visibilityState !== "visible") {
      throw new Error("Keep this benchmark tab visible before starting the run.");
    }
    if (BACKEND !== REQUESTED_BACKEND) {
      throw new Error(
        `Explicit ${REQUESTED_BACKEND} run initialized unexpected backend ${BACKEND}.`
      );
    }
    const modelByteEvidence = modelArtifactEvidence(MODEL_URL);
    if (!modelByteEvidence?.manifest_verified) {
      throw new Error(
        `DOM benchmark model ${MODEL_URL} was not byte-verified against bundle-manifest.json.`
      );
    }
    const runtimeAssets = runtimeAssetEvidence();

    const warmups = browserTaskReadBoundedInteger("browser-task-warmups", 0, 20);
    const repetitions = browserTaskReadBoundedInteger("browser-task-repetitions", 1, 100);
    const contextValue = $("browser-task-context-tokens").value;
    const targetInputTokens = contextValue ? Number.parseInt(contextValue, 10) : null;
    const seed = $("browser-task-seed").value.trim() || "dom-loop-v1";
    const warmupCases = browserTaskShuffle(BROWSER_TASK_SUITE.cases, `${seed}:warmup`);
    for (let index = 0; index < warmups; index++) {
      const benchmarkCase = warmupCases[index % warmupCases.length];
      browserTaskSetProgress(`Warm-up ${index + 1}/${warmups}: ${benchmarkCase.id}`);
      activeOpportunity = {
        phase: "warmup",
        index,
        case_id: benchmarkCase.id,
      };
      const warmupRecord = await browserTaskRunCase(
        benchmarkCase, -1, index, false, targetInputTokens
      );
      warmupRecord.phase = index === 0 ? "first_inference" : "warmup";
      warmupRecords.push(warmupRecord);
      activeOpportunity = null;
    }

    const total = repetitions * BROWSER_TASK_SUITE.cases.length;
    for (let repetition = 0; repetition < repetitions; repetition++) {
      const repetitionCases = browserTaskShuffle(
        BROWSER_TASK_SUITE.cases,
        `${seed}:${repetition}`
      );
      for (let orderIndex = 0; orderIndex < repetitionCases.length; orderIndex++) {
        const benchmarkCase = repetitionCases[orderIndex];
        if (document.visibilityState !== "visible") {
          throw new Error("Run stopped because the benchmark tab became hidden.");
        }
        browserTaskSetProgress(
          `Measured DOM loop ${records.length + 1}/${total}: ${benchmarkCase.id}`
        );
        activeOpportunity = {
          phase: "measured",
          repetition,
          order_index: orderIndex,
          case_id: benchmarkCase.id,
        };
        recordedOrder.push({
          repetition,
          order_index: orderIndex,
          case_id: benchmarkCase.id,
        });
        records.push(
          await browserTaskRunCase(
            benchmarkCase,
            repetition,
            orderIndex,
            true,
            targetInputTokens
          )
        );
        activeOpportunity = null;
      }
    }

    const modelMetaCanonicalHash = await browserTaskSha256(
      browserTaskCanonicalJson(META)
    );
    const bundleManifestCanonicalHash = BUNDLE_MANIFEST
      ? await browserTaskSha256(browserTaskCanonicalJson(BUNDLE_MANIFEST))
      : null;
    const modelArtifact = browserTaskBundleArtifact(MODEL_URL);
    const tokenizerArtifact = browserTaskBundleArtifact(META.tokenizer_file);
    const gpuAdapter = await browserTaskAdapterInfo();
    const contextAudit = browserTaskContextAudit(records, targetInputTokens);
    if (contextAudit.mismatched_records > 0) {
      throw new Error(
        `${contextAudit.mismatched_records} measured records violate the context condition.`
      );
    }
    LAST_BROWSER_TASK_RUN = {
      schema_version: 2,
      benchmark: BROWSER_TASK_SUITE.name,
      created_at: new Date().toISOString(),
      metadata: {
        benchmark_version: "rtab-dom-0.4",
        backend: BACKEND,
        requested_backend: REQUESTED_BACKEND,
        backend_requirement:
          REQUESTED_BACKEND === "webgpu"
            ? "explicit-webgpu-no-whole-session-retry"
            : REQUESTED_BACKEND === "wasm"
              ? "explicit-wasm-no-whole-session-retry-control"
              : "invalid-benchmark-provider-condition",
        execution_provider_request: {
          requested: REQUESTED_BACKEND,
          session_provider_count: 1,
          whole_session_retry: false,
          single_provider_session_creation_succeeded: true,
          per_node_placement: "unknown",
          per_node_fallback_status: "unknown",
          note:
            "ORT Web does not expose per-node placement; this proves the requested session " +
            "provider and does not claim every node executed on the GPU.",
        },
        benchmark_grade: true,
        bundle_manifest_required: true,
        model_url: MODEL_URL,
        precision: MODEL_URL.includes("fp16") ? "fp16" : "unknown-or-fp32",
        git_commit: BUNDLE_MANIFEST?.git_commit ?? null,
        model_hash: modelByteEvidence.sha256,
        checkpoint_hash: BUNDLE_MANIFEST?.checkpoint_sha256 ?? null,
        tokenizer_hash: runtimeAssets.tokenizer?.sha256 ?? null,
        heads_hash: runtimeAssets.heads_json.sha256,
        dispatch_heads_hash: runtimeAssets.dispatch_heads_json.sha256,
        meta_file_hash: runtimeAssets.meta_json.sha256,
        runtime_asset_evidence: runtimeAssets,
        graph_hash: modelByteEvidence.sha256,
        model_bytes: modelByteEvidence.bytes,
        model_byte_evidence: modelByteEvidence,
        manifest_graph_hash: modelArtifact?.sha256 ?? null,
        manifest_model_bytes: modelArtifact?.bytes ?? null,
        manifest_tokenizer_hash: tokenizerArtifact?.sha256 ?? null,
        artifact_hash_contract:
          "model, heads, metadata, dispatch, and tokenizer assets are fetched as bytes, SHA-256 " +
          "and size checked in-browser against bundle-manifest.json before parsing or ORT session " +
          "creation; ORT receives the verified in-memory model bytes",
        model_meta_canonical_sha256: modelMetaCanonicalHash,
        bundle_manifest: BUNDLE_MANIFEST,
        bundle_manifest_byte_evidence: bundleManifestByteEvidence(),
        bundle_manifest_canonical_sha256: bundleManifestCanonicalHash,
        model_vocab_size: META.vocab_size,
        model_d_model: META.d_model,
        model_layers: META.n_layers ?? null,
        model_parameters: META.model_parameters ?? null,
        model_encoding: META.encoding,
        model_tool_count: (META.tools || []).length,
        ort_web_version: ort.env?.versions?.web || null,
        onnxruntime_version: ort.env?.versions?.web || null,
        user_agent: navigator.userAgent,
        browser: navigator.userAgent,
        os: null,
        language: navigator.language,
        power_mode: null,
        hardware_concurrency: navigator.hardwareConcurrency || null,
        device_memory_gb: navigator.deviceMemory || null,
        webgpu_adapter: gpuAdapter,
        gpu_adapter: gpuAdapter,
        webgpu_adapter_note:
          "separate navigator.gpu query; onnxruntime-web does not expose its selected adapter",
        page_to_model_ready_ms: BROWSER_TASK_MODEL_READY_MS,
        bundle_load_timing_ms: { ...BUNDLE_LOAD_TIMING },
        model_resource: browserTaskModelResourceTiming(),
        suite_schema_version: BROWSER_TASK_SUITE.schema_version,
        suite_sha256: BROWSER_TASK_SUITE_BYTE_EVIDENCE.sha256,
        suite_bytes: BROWSER_TASK_SUITE_BYTE_EVIDENCE.bytes,
        suite_byte_evidence: BROWSER_TASK_SUITE_BYTE_EVIDENCE,
        suite_expected_actions_schema_validated: true,
        fixture_contract: { ...BROWSER_TASK_SUITE.fixture_contract },
        case_order_seed: seed,
        recorded_case_order: recordedOrder,
        target_input_tokens: targetInputTokens,
        context_condition:
          targetInputTokens == null
            ? "natural"
            : "fixed_compute_tokens_natural_decision_feature",
        context_padding:
          targetInputTokens == null
            ? "none"
            : "single-token spaces appended after the natural assistant marker",
        decision_feature_contract:
          "hidden[natural_input_tokens - 1]; pointer scan bounded to natural_input_tokens",
        context_audit: contextAudit,
        warmups,
        repetitions,
        cases: BROWSER_TASK_SUITE.cases.length,
        measured_records: records.length,
        concurrency: 1,
        latency_clock: "harness_ttfa_ms",
        timeout_ms: BROWSER_TASK_ACTION_TIMEOUT_MS,
        action_timeout_ms: BROWSER_TASK_ACTION_TIMEOUT_MS,
        watchdog_scope: "every warmup and measured policy call",
        timeout_contract:
          "a timeout aborts the entire page collection; ORT session.run is not cancellable, " +
          "so no subsequent policy call starts while timed-out inference may still be live",
        timer: "performance.now",
        paint_barrier: "two consecutive requestAnimationFrame callbacks",
        latency_boundaries: {
          harness_ttfa:
            "prompt tokenization through independent schema validation; observation readiness excluded",
          runtime_ttfa:
            "runtime prompt tokenization through runtime schema validation",
          ttfa_ms:
            "backward-compatible exact alias of harness_ttfa_ms; not an additional clock",
          tool: "synchronous local semantic event dispatch",
          closed_loop: "immediately before policy call through post-paint DOM inspection",
        },
        schema_source: "loaded META.tools[].schema",
        schema_validator:
          "independent browser-tasks.js JSON Schema subset v2; app validator is diagnostic only",
        raw_action_evidence_contract:
          "each row stores normalized predicted_action, full expected_action, parse_evidence, " +
          "and independent_schema including errors and the exact selected tool schema",
        dispatch_contract: {
          input_modality: "text-only prompt",
          actions_per_case: 1,
          target_resolution: "local data-semantic-target labels",
          event_trust: "synthetic; Event.isTrusted is false",
          physical_cursor_control: false,
          visual_grounding: false,
          multi_step_planning: false,
          browser_wide_control: false,
          external_navigation: false,
          local_navigation: "fragment route in a disposable same-origin iframe",
        },
      },
      summary: browserTaskSummarize(records),
      warmup_records: warmupRecords,
      records,
    };
    browserTaskRenderSummary(LAST_BROWSER_TASK_RUN);
    browserTaskSetProgress(
      `Complete: ${records.length} measured single-step DOM loops on ${BACKEND}.`
    );
    downloadButton.disabled = false;
  } catch (error) {
    console.error(error);
    if (browserTaskIsActionTimeout(error)) {
      LAST_BROWSER_TASK_RUN = {
        schema_version: 2,
        benchmark: "localagent-single-step-dom-microtasks",
        created_at: new Date().toISOString(),
        status: "aborted_incomplete",
        metadata: {
          benchmark_version: "rtab-dom-0.4",
          action_timeout_ms: BROWSER_TASK_ACTION_TIMEOUT_MS,
          watchdog_scope: "every warmup and measured policy call",
          timeout_contract:
            "a timeout aborts the entire page collection; ORT session.run is not cancellable, " +
            "so no subsequent policy call starts while timed-out inference may still be live",
          completed_warmups: warmupRecords.length,
          completed_measured_records: records.length,
          recorded_case_order: recordedOrder,
        },
        failure: {
          kind: "action_timeout",
          fatal_to_page_collection: true,
          inference_cancellation_supported: false,
          no_subsequent_policy_call_started: true,
          timeout_ms: error.timeout_ms ?? BROWSER_TASK_ACTION_TIMEOUT_MS,
          active_opportunity: activeOpportunity,
          error: browserTaskErrorRecord(error),
        },
        warmup_records: warmupRecords,
        records,
      };
      downloadButton.disabled = false;
    }
    browserTaskSetProgress(`DOM task run failed: ${error.message}`);
  } finally {
    startButton.disabled = false;
  }
}

function downloadBrowserTasks() {
  if (!LAST_BROWSER_TASK_RUN) return;
  const payload = JSON.stringify(LAST_BROWSER_TASK_RUN, null, 2);
  const blob = new Blob([payload], { type: "application/json" });
  const anchor = document.createElement("a");
  anchor.href = URL.createObjectURL(blob);
  anchor.download = `localagent-dom-loop-${Date.now()}.json`;
  anchor.click();
  URL.revokeObjectURL(anchor.href);
}

// Read-only inspection hook for the local browser harness and automated receipt capture.
// The page still owns execution; callers receive the immutable-in-practice result object
// produced by the latest run and must not treat it as native browser/OS evidence.
if (typeof window !== "undefined") {
  window.__localAgentBrowserTaskResult = () => LAST_BROWSER_TASK_RUN;
}

async function initBrowserTasks() {
  $("start-browser-tasks").disabled = true;
  $("download-browser-tasks").disabled = true;
  $("start-browser-tasks").addEventListener("click", runBrowserTasks);
  $("download-browser-tasks").addEventListener("click", downloadBrowserTasks);
  try {
    const suitePromise = fetchPinnedJsonArtifact(
      BROWSER_TASK_SUITE_IDENTITY.file,
      BROWSER_TASK_SUITE_IDENTITY
    ).then((document) => {
      BROWSER_TASK_SUITE = document.value;
      BROWSER_TASK_SUITE_BYTE_EVIDENCE = document.evidence;
      browserTaskValidateSuite(BROWSER_TASK_SUITE);
    });
    await Promise.all([LOCALAGENT_READY, suitePromise]);
    browserTaskValidateExpectedActions(BROWSER_TASK_SUITE);
    BROWSER_TASK_MODEL_READY_MS =
      performance.now() - window.__localAgentBrowserTasksStart;
    $("start-browser-tasks").disabled = false;
    browserTaskSetProgress(
      `Model and fixture contract v${BROWSER_TASK_FIXTURE_CONTRACT_VERSION} ready. ` +
      "Runs use concurrency 1."
    );
  } catch (error) {
    console.error(error);
    browserTaskSetProgress(`DOM benchmark unavailable: ${error.message}`);
  }
}

if (!(typeof window !== "undefined" && window.__localAgentSkipInit)) {
  initBrowserTasks();
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    BROWSER_TASK_ACTION_TIMEOUT_MS,
    browserTaskWithWatchdog,
    browserTaskIsActionTimeout,
    browserTaskCanonicalJson,
    browserTaskNormalizeAction,
    browserTaskScoreAction,
    browserTaskShuffle,
    browserTaskSummarize,
    browserTaskValidateActionSchema,
    browserTaskValidateSuite,
  };
}
