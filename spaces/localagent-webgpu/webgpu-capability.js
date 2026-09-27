/* Fail-closed native WebGPU capability and action-dispatch receipt harness. */
"use strict";

const CAPABILITY_WARMUPS = 3;
const CAPABILITY_REPETITIONS = 30;
const CAPABILITY_QUERIES = Object.freeze([
  {
    prompt: "Email Dana the quarterly report",
    expected_tool: "send_email",
    expected_argument: "recipient",
  },
  {
    prompt: "Open https://example.com",
    expected_tool: "open_url",
    expected_argument: "url",
  },
  {
    prompt: "Write 'WebGPU state loop passed' to Notion",
    expected_tool: "notion_write",
    expected_argument: "content",
  },
]);

function capabilityElement(id) {
  return document.getElementById(id);
}

function percentile(values, quantile) {
  const ordered = [...values].sort((left, right) => left - right);
  if (!ordered.length) return null;
  const position = (ordered.length - 1) * quantile;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return ordered[lower];
  return ordered[lower] * (upper - position) + ordered[upper] * (position - lower);
}

function finiteSummary(values) {
  const finite = values.filter((value) => Number.isFinite(value) && value >= 0);
  return {
    count: finite.length,
    min_ms: finite.length ? Math.min(...finite) : null,
    mean_ms: finite.length ? finite.reduce((sum, value) => sum + value, 0) / finite.length : null,
    p50_ms: percentile(finite, 0.50),
    p95_ms: percentile(finite, 0.95),
    max_ms: finite.length ? Math.max(...finite) : null,
  };
}

function adapterIdentity(adapter) {
  const info = adapter?.info || {};
  const fields = [
    ["vendor", info.vendor],
    ["architecture", info.architecture],
    ["device", info.device],
    ["description", info.description],
    ["adapter", adapter?.name],
  ].filter(([, value]) => typeof value === "string" && value.trim());
  const identity = fields.map(([key, value]) => `${key}=${value.trim()}`).join("; ");
  return {
    identity: identity || null,
    vendor: info.vendor || null,
    architecture: info.architecture || null,
    device: info.device || null,
    description: info.description || null,
    adapter_name: adapter?.name || null,
    is_fallback_adapter: adapter?.isFallbackAdapter ?? null,
    identity_fields: fields.map(([key]) => key),
  };
}

async function queryAdapter() {
  if (!globalThis.navigator?.gpu) {
    throw new Error("navigator.gpu is unavailable; this is not a native WebGPU execution.");
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("navigator.gpu.requestAdapter() returned no adapter.");
  const identity = adapterIdentity(adapter);
  if (!identity.identity) {
    throw new Error("WebGPU adapter exists but exposes no non-empty hardware identity.");
  }
  const device = await adapter.requestDevice();
  if (!device) throw new Error("WebGPU adapter failed to create a GPUDevice.");
  const features = [...device.features].sort();
  const limits = {
    max_storage_buffer_binding_size: device.limits.maxStorageBufferBindingSize,
    max_buffer_size: device.limits.maxBufferSize,
    max_compute_workgroups_per_dimension: device.limits.maxComputeWorkgroupsPerDimension,
  };
  device.destroy();
  return { adapter, identity, features, limits };
}

async function graphByteEstimate() {
  const path = "action_model.fp16.onnx";
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Failed to fetch ${path}: HTTP ${response.status}.`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  const sha256 = Array.from(new Uint8Array(digest), (value) =>
    value.toString(16).padStart(2, "0")
  ).join("");
  return { path, bytes: bytes.byteLength, sha256 };
}

async function manifestIdentity() {
  const response = await fetch("bundle-manifest.json");
  if (!response.ok) throw new Error(`Failed to fetch bundle-manifest.json: HTTP ${response.status}.`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return {
    path: "bundle-manifest.json",
    bytes: bytes.byteLength,
    sha256: Array.from(new Uint8Array(digest), (value) =>
      value.toString(16).padStart(2, "0")
    ).join(""),
  };
}

function checkpointIdentity() {
  const manifest = BUNDLE_MANIFEST || {};
  return {
    source: "bundle-manifest.json",
    sha256: manifest.checkpoint_sha256 || null,
    stage: manifest.checkpoint_stage || null,
    step: Number.isInteger(manifest.checkpoint_step) ? manifest.checkpoint_step : null,
    config_name: manifest.config_name || null,
    parameters: Number.isInteger(manifest.model_parameters)
      ? manifest.model_parameters
      : (Number.isInteger(META?.model_parameters) ? META.model_parameters : null),
  };
}

async function runDispatchProbe(query) {
  const measured = [];
  let last = null;
  for (let index = 0; index < CAPABILITY_WARMUPS + CAPABILITY_REPETITIONS; index += 1) {
    const started = performance.now();
    const result = await window.__localAgentStructuredAction(query.prompt);
    const elapsed = performance.now() - started;
    last = result;
    if (index >= CAPABILITY_WARMUPS) measured.push({ elapsed_ms: elapsed, result });
  }
  const exact = measured.filter(({ result }) =>
    result?.tool === query.expected_tool &&
    result?.schema_valid === true &&
    typeof result?.args?.[query.expected_argument] === "string" &&
    result.args[query.expected_argument].length > 0
  ).length;
  const timings = finiteSummary(measured.map(({ elapsed_ms }) => elapsed_ms));
  const inference = finiteSummary(measured.map(({ result }) => result?.timing?.inference_ms));
  const inputTokens = Number(last?.input_tokens || 0);
  return {
    prompt: query.prompt,
    expected_tool: query.expected_tool,
    expected_argument: query.expected_argument,
    measured_repetitions: measured.length,
    exact_actions: exact,
    exact_action_rate: measured.length ? exact / measured.length : 0,
    last_result: last,
    wall_latency_ms: timings,
    model_inference_ms: inference,
    input_tokens: inputTokens,
    tokens_per_second_p50: inference.p50_ms > 0 ? (inputTokens * 1000) / inference.p50_ms : null,
  };
}

async function runCapabilityReceipt() {
  const started = performance.now();
  const runtime = await queryAdapter();
  const graph = await graphByteEstimate();
  const manifest = await manifestIdentity();
  await window.__localAgentReady;
  const cases = [];
  for (const query of CAPABILITY_QUERIES) cases.push(await runDispatchProbe(query));
  const allInference = cases.flatMap((item) =>
    Array.from({ length: item.measured_repetitions }, () => item.model_inference_ms.p50_ms)
  );
  const tokensPerSecond = cases
    .map((item) => item.tokens_per_second_p50)
    .filter((value) => Number.isFinite(value));
  // ORT exposes output tensors to JavaScript but not driver VRAM counters.  This is a
  // conservative, reproducible allocation estimate (graph bytes + largest input/output), not
  // a claim about total GPU-driver memory.
  const largestInputBytes = Math.max(...cases.map((item) => item.input_tokens * 8), 0);
  const largestOutputBytes = Math.max(...cases.map((item) => item.input_tokens * 384 * 4), 0);
  const peakMemoryBytes = graph.bytes + largestInputBytes + largestOutputBytes;
  const payload = {
    kind: "localagent_webgpu_native_capability_receipt",
    schema_version: 1,
    backend: "webgpu",
    environment_executed: true,
    hardware_adapter: runtime.identity.identity,
    adapter: runtime.identity,
    capability: {
      evaluated_cases: cases.length,
      exact_actions: cases.reduce((sum, item) => sum + (item.exact_actions > 0 ? 1 : 0), 0),
      closed_loop_success: 0,
      external_side_effects_executed: false,
      cases,
    },
    protocol: {
      page: "spaces/localagent-webgpu/webgpu-capability.html",
      ort_version: globalThis.ort?.version || globalThis.ort?.env?.versions?.web || null,
      requested_provider: ["webgpu"],
      session_provider_retry: false,
      warmups_per_case: CAPABILITY_WARMUPS,
      measured_repetitions_per_case: CAPABILITY_REPETITIONS,
      action_graph: graph,
      bundle_manifest: manifest,
    },
    performance: {
      tokens_per_second_p50: tokensPerSecond.length ? percentile(tokensPerSecond, 0.5) : null,
      tokens_per_second_definition: "action-graph input tokens divided by p50 model inference ms",
      latency_ms_p50: percentile(cases.map((item) => item.wall_latency_ms.p50_ms), 0.5),
      peak_memory_mb: peakMemoryBytes / (1024 * 1024),
      peak_memory_method: "conservative action-graph bytes plus largest host input/output tensors; driver VRAM counter unavailable",
      graph: graph,
      largest_input_bytes: largestInputBytes,
      largest_output_bytes: largestOutputBytes,
    },
    runtime: {
      ort_version: globalThis.ort?.version || globalThis.ort?.env?.versions?.web || null,
      requested_provider: ["webgpu"],
      session_provider_retry: false,
      navigator_gpu_available: true,
      device_features: runtime.features,
      device_limits: runtime.limits,
      user_agent: navigator.userAgent,
      elapsed_ms: performance.now() - started,
    },
    checkpoint: checkpointIdentity(),
    claim_boundary: "Native browser WebGPU adapter and action-graph dispatch receipt. The calls are local predictions only: no real email, browser navigation, or Notion account was touched; closed_loop_success is therefore zero.",
  };
  return payload;
}

(async () => {
  const output = capabilityElement("capability-output");
  const status = capabilityElement("capability-status");
  try {
    const payload = await runCapabilityReceipt();
    window.__localAgentWebGpuCapabilityResult = payload;
    const raw = JSON.stringify(payload, null, 2);
    output.textContent = raw;
    capabilityElement("capability-result-json").textContent = raw;
    status.textContent = "Native WebGPU capability receipt complete.";
  } catch (error) {
    const payload = {
      kind: "localagent_webgpu_native_capability_receipt",
      schema_version: 1,
      backend: "webgpu",
      environment_executed: false,
      error: { name: error?.name || "Error", message: error?.message || String(error) },
      claim_boundary: "Fail-closed harness failure; no native WebGPU capability or latency claim is valid.",
    };
    window.__localAgentWebGpuCapabilityResult = payload;
    const raw = JSON.stringify(payload, null, 2);
    output.textContent = raw;
    capabilityElement("capability-result-json").textContent = raw;
    status.textContent = "Receipt failed closed; no native claim is valid.";
    console.error(error);
  }
})();
