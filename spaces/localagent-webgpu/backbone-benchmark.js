/* LocalAgent matched-backbone latency benchmark.
 *
 * This runner is intentionally narrower than benchmark.js:
 *   - it accepts exactly two hidden-only ONNX graphs;
 *   - both graphs receive the same deterministic, pre-tokenized integer IDs;
 *   - it measures backbone forwards at 128/512/1024/1536 actual tensor tokens;
 *   - it never evaluates actions, text, accuracy, or any other quality signal.
 *
 * The exporter-produced matched-backbones.json pins both provenance files and every graph/config
 * artifact by SHA-256. Graph bytes are hashed in the browser before they are passed to ORT.
 */
"use strict";

const BACKBONE_CONTEXT_LENGTHS = Object.freeze([128, 512, 1024, 1536]);
const BACKBONE_DEFAULT_MANIFEST_URL =
  "../../runs/webgpu/random-backbone-latency-seed-20260728/matched-backbones.json";
const BACKBONE_DEFAULT_SEED = "slmw2026-backbone-v1";
const BACKBONE_MIN_WARMUPS = 3;
const BACKBONE_MIN_REPETITIONS = 30;
const BACKBONE_ORT_VERSION = "1.27.0";
const BACKBONE_ORT_SCRIPT_URL =
  `https://cdn.jsdelivr.net/npm/onnxruntime-web@${BACKBONE_ORT_VERSION}/dist/ort.webgpu.min.js`;
const BACKBONE_ORT_WASM_BASE_URL =
  `https://cdn.jsdelivr.net/npm/onnxruntime-web@${BACKBONE_ORT_VERSION}/dist/`;
const BACKBONE_LABELS = Object.freeze({
  latency_only: true,
  untrained_random_weights: true,
  quality_evaluation: false,
});

let BACKBONE_STATE = {
  manifest: null,
  manifestUrl: null,
  manifestSha256: null,
  inputs: new Map(),
  arms: new Map(),
  bundleRecords: [],
  sessionRecords: [],
  inputPreparationRecord: null,
  readyAtMs: null,
};
let LAST_BACKBONE_BENCHMARK = null;
let BACKBONE_RUN_STARTED = false;

function backboneElement(id) {
  return document.getElementById(id);
}

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value && typeof value === "object") {
    const fields = Object.keys(value).sort().map(
      (key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`
    );
    return `{${fields.join(",")}}`;
  }
  return JSON.stringify(value);
}

function pythonExponent(value) {
  const [mantissa, rawExponent] = value.toExponential().split("e");
  const exponent = Number.parseInt(rawExponent, 10);
  const sign = exponent < 0 ? "-" : "+";
  return `${mantissa}e${sign}${Math.abs(exponent).toString().padStart(2, "0")}`;
}

function modelConfigCanonicalJson(config) {
  // The exporter hashes Python json.dumps(asdict(config), separators=(",", ":"), sort_keys=True).
  // JSON parsing loses the lexical distinction between 0 and 0.0, so retain the ModelConfig
  // schema's three float fields explicitly while reproducing Python's exponent spelling.
  const floatFields = new Set(["dropout", "norm_eps", "rope_theta"]);
  const fields = Object.keys(config).sort().map((key) => {
    let encoded;
    const value = config[key];
    if (floatFields.has(key)) {
      if (!Number.isFinite(value)) throw new Error(`Model config float ${key} is not finite.`);
      if (
        key === "norm_eps" &&
        value !== 0 &&
        (Math.abs(value) < 1e-4 || Math.abs(value) >= 1e16)
      ) {
        encoded = pythonExponent(value);
      } else if (Number.isInteger(value)) {
        encoded = `${value}.0`;
      } else {
        encoded = JSON.stringify(value);
      }
    } else {
      encoded = canonicalJson(value);
    }
    return `${JSON.stringify(key)}:${encoded}`;
  });
  return `{${fields.join(",")}}`;
}

function seededRandom(seedText) {
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

function shuffled(values, seedText) {
  const result = [...values];
  const random = seededRandom(seedText);
  for (let index = result.length - 1; index > 0; index--) {
    const replacement = Math.floor(random() * (index + 1));
    [result[index], result[replacement]] = [result[replacement], result[index]];
  }
  return result;
}

function buildBackboneSchedule(armIds, contextLengths, repetitions, seedText, phase) {
  if (!Array.isArray(armIds) || armIds.length !== 2) {
    throw new Error("A matched backbone schedule requires exactly two graph arms.");
  }
  if (!Array.isArray(contextLengths) || !contextLengths.length) {
    throw new Error("At least one context length is required.");
  }
  if (!Number.isInteger(repetitions) || repetitions < 1) {
    throw new Error("Schedule repetitions must be a positive integer.");
  }
  const conditions = armIds.flatMap((armId) =>
    contextLengths.map((inputTokens) => ({ arm_id: armId, input_tokens: inputTokens }))
  );
  const schedule = [];
  for (let repetition = 0; repetition < repetitions; repetition++) {
    const order = shuffled(conditions, `${seedText}:${phase}:${repetition}`);
    order.forEach((condition, orderIndex) => {
      schedule.push({ ...condition, repetition, order_index: orderIndex });
    });
  }
  return schedule;
}

function percentile(values, quantile) {
  const ordered = [...values].sort((left, right) => left - right);
  const position = (ordered.length - 1) * quantile;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return ordered[lower];
  return ordered[lower] * (upper - position) + ordered[upper] * (position - lower);
}

function latencySummary(values) {
  const finite = values.filter((value) => Number.isFinite(value) && value >= 0);
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
  return {
    count: finite.length,
    min: Math.min(...finite),
    mean: finite.reduce((sum, value) => sum + value, 0) / finite.length,
    p50: percentile(finite, 0.50),
    p90: percentile(finite, 0.90),
    p95: percentile(finite, 0.95),
    p99: percentile(finite, 0.99),
    max: Math.max(...finite),
  };
}

function summarizeBackboneRecords(records) {
  const groups = {};
  for (const record of records) {
    const key = `${record.arm_id}:${record.input_tokens}`;
    if (!groups[key]) {
      groups[key] = {
        arm_id: record.arm_id,
        input_tokens: record.input_tokens,
        attempted: 0,
        completed: 0,
        failed: 0,
        inference_ms: [],
      };
    }
    const group = groups[key];
    group.attempted += 1;
    if (record.run_ok) {
      group.completed += 1;
      group.inference_ms.push(record.inference_ms);
    } else {
      group.failed += 1;
    }
  }
  const conditions = Object.values(groups)
    .sort((left, right) =>
      left.arm_id.localeCompare(right.arm_id) || left.input_tokens - right.input_tokens
    )
    .map((group) => ({
      arm_id: group.arm_id,
      input_tokens: group.input_tokens,
      attempted: group.attempted,
      completed: group.completed,
      failed: group.failed,
      inference_latency_ms: latencySummary(group.inference_ms),
    }));
  return {
    estimand: "hidden_only_backbone_forward_latency",
    quality_metrics_included: false,
    attempted: records.length,
    completed: records.filter((record) => record.run_ok).length,
    failed: records.filter((record) => !record.run_ok).length,
    conditions,
  };
}

function errorDetail(error) {
  return {
    name: error?.name || "Error",
    message: error?.message || String(error),
    stack: typeof error?.stack === "string" ? error.stack : null,
  };
}

function isSha256(value) {
  return typeof value === "string" && /^[0-9a-f]{64}$/i.test(value);
}

function assertSha256(value, field) {
  if (!isSha256(value)) throw new Error(`${field} must be a 64-character SHA-256 hex digest.`);
}

function validateBackboneManifest(manifest) {
  if (!manifest || typeof manifest !== "object" || Array.isArray(manifest)) {
    throw new Error("Backbone benchmark manifest must be a JSON object.");
  }
  if (manifest.schema_version !== 1) {
    throw new Error(`Unsupported backbone manifest schema ${manifest.schema_version}.`);
  }
  if (
    manifest.artifact_type !== "matched_random_backbone_latency_suite" ||
    manifest.latency_only !== true ||
    manifest.trained !== false ||
    manifest.capability_artifact !== false ||
    !Array.isArray(manifest.quality_claims) ||
    manifest.quality_claims.length !== 0
  ) {
    throw new Error(
      "Pair manifest must be the exporter-produced latency-only, untrained, " +
      "non-capability suite with no quality claims."
    );
  }
  const modelEntries = Object.entries(manifest.models || {});
  if (modelEntries.length !== 2) {
    throw new Error("Pair manifest must contain exactly two model provenance entries.");
  }
  const roles = modelEntries.map(([role]) => role).sort();
  if (canonicalJson(roles) !== canonicalJson([
    "all_attention_control", "hybrid_treatment",
  ])) {
    throw new Error("Pair manifest must contain hybrid_treatment and all_attention_control.");
  }
  if (!Number.isInteger(manifest.shared_random_seed) || manifest.shared_random_seed < 0) {
    throw new Error("Pair manifest must declare a non-negative shared_random_seed.");
  }
  const intentionalDifferences = Object.keys(manifest.intentional_differences || {}).sort();
  if (canonicalJson(intentionalDifferences) !== canonicalJson([
    "ffn_hidden", "layer_types", "name",
  ])) {
    throw new Error(
      "Pair manifest must declare exactly name, ffn_hidden, and layer_types as differences."
    );
  }
  const requiredControlledFields = [
    "conv_kernel", "d_model", "dropout", "embed_dim", "max_seq_len", "n_heads",
    "n_kv_heads", "n_layers", "n_loops", "norm_eps", "qk_norm", "rope_theta",
    "tie_embeddings", "vocab_size",
  ];
  if (canonicalJson([...(manifest.controlled_fields || [])].sort()) !==
      canonicalJson(requiredControlledFields.sort())) {
    throw new Error("Pair manifest controlled_fields do not match the architecture control.");
  }
  for (const [role, model] of modelEntries) {
    if (
      !model ||
      typeof model !== "object" ||
      typeof model.name !== "string" ||
      !model.name ||
      typeof model.directory !== "string" ||
      !model.directory ||
      typeof model.provenance !== "string" ||
      !model.provenance
    ) {
      throw new Error(`models.${role} must name a model and provenance file.`);
    }
    const artifact = manifest.artifacts?.[model.provenance];
    if (!artifact || !isSha256(artifact.sha256) || !Number.isInteger(artifact.bytes)) {
      throw new Error(`Pair manifest does not pin ${model.provenance}.`);
    }
  }
  return manifest;
}

function validateBackboneProvenance(provenance, role, modelEntry, sharedRandomSeed = null) {
  if (!provenance || typeof provenance !== "object" || Array.isArray(provenance)) {
    throw new Error(`Provenance for ${role} must be a JSON object.`);
  }
  if (
    provenance.schema_version !== 1 ||
    provenance.artifact_type !== "random_weight_hidden_backbone_onnx" ||
    provenance.latency_only !== true ||
    provenance.trained !== false ||
    provenance.training_steps !== 0 ||
    provenance.capability_artifact !== false ||
    !Array.isArray(provenance.quality_claims) ||
    provenance.quality_claims.length !== 0
  ) {
    throw new Error(`${role} provenance is not an untrained latency-only artifact.`);
  }
  const contract = provenance.graph_contract;
  if (
    contract?.input?.name !== "input_ids" ||
    contract?.input?.dtype !== "int64" ||
    contract?.output?.name !== "hidden" ||
    contract?.tokenizer_asset_included !== false
  ) {
    throw new Error(`${role} provenance does not declare the pre-tokenized hidden-only contract.`);
  }
  if (
    provenance.model?.name !== modelEntry.name ||
    provenance.model?.pair_role !== role ||
    !provenance.model?.config ||
    !isSha256(provenance.model?.config_canonical_sha256) ||
    !Number.isInteger(provenance.model?.full_model_parameters)
  ) {
    throw new Error(`${role} provenance has an invalid model/config contract.`);
  }
  if (
    provenance.weights?.source !== "deterministic_random_initialization" ||
    provenance.weights?.checkpoint !== null ||
    (sharedRandomSeed != null && provenance.weights?.seed !== sharedRandomSeed) ||
    !isSha256(provenance.weights?.state_dict_sha256)
  ) {
    throw new Error(`${role} provenance is not tied to deterministic random initialization.`);
  }
  const graphArtifact =
    provenance.artifacts?.["backbone.fp16.onnx"] ||
    provenance.artifacts?.["backbone.fp32.onnx"];
  if (
    !graphArtifact ||
    !isSha256(graphArtifact.sha256) ||
    !Number.isInteger(graphArtifact.bytes)
  ) {
    throw new Error(`${role} provenance has no pinned hidden-only ONNX graph.`);
  }
  return provenance;
}

function requireExplicitProvider(provider) {
  if (provider !== "webgpu" && provider !== "wasm") {
    throw new Error(`Unknown provider '${provider}'; expected exactly webgpu or wasm.`);
  }
  return provider;
}

function backboneSessionOptions(provider) {
  requireExplicitProvider(provider);
  return { executionProviders: [provider] };
}

async function sha256ArrayBuffer(buffer) {
  if (!globalThis.crypto?.subtle) {
    throw new Error("Web Crypto SHA-256 is unavailable; artifact verification cannot continue.");
  }
  const digest = await globalThis.crypto.subtle.digest("SHA-256", buffer);
  return Array.from(
    new Uint8Array(digest),
    (value) => value.toString(16).padStart(2, "0")
  ).join("");
}

async function sha256Text(text) {
  return sha256ArrayBuffer(new TextEncoder().encode(text));
}

function requestedManifestUrl() {
  if (typeof window === "undefined") return BACKBONE_DEFAULT_MANIFEST_URL;
  return window.__localAgentBackboneManifestUrl ||
    new URLSearchParams(window.location.search).get("manifest") ||
    BACKBONE_DEFAULT_MANIFEST_URL;
}

function resolveArtifactUrl(path, baseUrl = document.baseURI) {
  return new URL(path, baseUrl).href;
}

async function fetchVerifiedArtifact(
  path,
  expectedSha256,
  artifactKind,
  artifactId,
  baseUrl = document.baseURI
) {
  assertSha256(expectedSha256, `${artifactKind}.expected_sha256`);
  const url = resolveArtifactUrl(path, baseUrl);
  const started = performance.now();
  const response = await fetch(url);
  const responseAt = performance.now();
  if (!response.ok) {
    throw new Error(`Failed to fetch ${path}: HTTP ${response.status}.`);
  }
  const buffer = await response.arrayBuffer();
  const readFinished = performance.now();
  const hashStarted = performance.now();
  const actualSha256 = await sha256ArrayBuffer(buffer);
  const finished = performance.now();
  const record = {
    phase: "bundle",
    artifact_kind: artifactKind,
    artifact_id: artifactId,
    url,
    fetch_ms: responseAt - started,
    read_ms: readFinished - responseAt,
    hash_ms: finished - hashStarted,
    fetch_and_read_ms: readFinished - started,
    total_fetch_read_hash_ms: finished - started,
    bytes: buffer.byteLength,
    expected_sha256: expectedSha256.toLowerCase(),
    actual_sha256: actualSha256,
    hash_computed: true,
    hash_verified: actualSha256 === expectedSha256.toLowerCase(),
    content_type: response.headers.get("content-type") || null,
    etag: response.headers.get("etag") || null,
    last_modified: response.headers.get("last-modified") || null,
    cache_control: response.headers.get("cache-control") || null,
    browser_cache_state: "unknown",
    browser_cache_state_reason:
      "Fetch response does not expose a reliable network-versus-browser-cache classification",
    ...BACKBONE_LABELS,
  };
  BACKBONE_STATE.bundleRecords.push(record);
  if (!record.hash_verified) {
    throw new Error(
      `${artifactKind} ${artifactId} SHA-256 mismatch: ` +
      `expected ${record.expected_sha256}, got ${record.actual_sha256}.`
    );
  }
  return { buffer, record };
}

function decodeJsonBuffer(buffer, label) {
  try {
    return JSON.parse(new TextDecoder().decode(buffer));
  } catch (error) {
    throw new Error(`${label} is not valid JSON: ${error.message}`);
  }
}

async function prepareBackboneInputs(vocabSize) {
  const started = performance.now();
  if (!Number.isInteger(vocabSize) || vocabSize < 1) {
    throw new Error("A positive model vocabulary size is required for deterministic input IDs.");
  }
  const publicInputs = [];
  for (const inputTokens of BACKBONE_CONTEXT_LENGTHS) {
    const tokenIds = Array.from(
      { length: inputTokens },
      (_, index) => (131 * index + 17) % vocabSize
    );
    const tensorData = BigInt64Array.from(tokenIds, (tokenId) => BigInt(tokenId));
    const tensor = new ort.Tensor(
      "int64",
      tensorData,
      [1, inputTokens]
    );
    const input = {
      input_tokens: inputTokens,
      actual_tensor_tokens: tensor.dims[1],
      token_ids: tokenIds,
      input_ids_int64_sha256: await sha256ArrayBuffer(tensorData.buffer),
      tensor_dtype: tensor.type,
      tensor_dims: [...tensor.dims],
      input_semantics: "deterministic_pretokenized_ids",
      fixture_contract: "ids[i]=(131*i+17) mod vocab_size",
      vocab_size: vocabSize,
      tokenizer_asset: null,
      tensor,
    };
    BACKBONE_STATE.inputs.set(inputTokens, input);
    publicInputs.push({
      ...input,
      tensor: undefined,
    });
  }
  BACKBONE_STATE.inputPreparationRecord = {
    phase: "input_preparation",
    duration_ms: performance.now() - started,
    input_semantics: "deterministic_pretokenized_ids",
    fixture_contract: "ids[i]=(131*i+17) mod vocab_size",
    vocab_size: vocabSize,
    tokenizer_asset: null,
    requested_context_lengths: [...BACKBONE_CONTEXT_LENGTHS],
    all_actual_lengths_verified: publicInputs.every(
      (input) => input.input_tokens === input.actual_tensor_tokens
    ),
    ...BACKBONE_LABELS,
  };
  return publicInputs;
}

function validateConfig(config, armId) {
  if (!config || typeof config !== "object" || Array.isArray(config)) {
    throw new Error(`Config for ${armId} must be a JSON object.`);
  }
  if (config.name !== armId) {
    throw new Error(`Config name ${config.name} does not match manifest model ${armId}.`);
  }
  if (!Number.isInteger(config.d_model) || config.d_model < 1) {
    throw new Error(`${armId} config has no valid d_model.`);
  }
  if (!Number.isInteger(config.vocab_size) || config.vocab_size < 1) {
    throw new Error(`${armId} config has no valid vocab_size.`);
  }
  if (
    !Number.isInteger(config.max_seq_len) ||
    config.max_seq_len < Math.max(...BACKBONE_CONTEXT_LENGTHS)
  ) {
    throw new Error(`${armId} max_seq_len does not cover every benchmark context.`);
  }
}

function validateMatchedConfigs(assets) {
  const [left, right] = assets;
  for (const field of [
    "vocab_size", "d_model", "embed_dim", "n_layers", "n_loops", "n_heads", "n_kv_heads",
    "max_seq_len", "rope_theta", "norm_eps", "tie_embeddings", "dropout", "qk_norm",
    "conv_kernel",
  ]) {
    if (canonicalJson(left.config[field]) !== canonicalJson(right.config[field])) {
      throw new Error(`Matched backbone configs differ in required control field '${field}'.`);
    }
  }
  const hybrid = assets.find((asset) => asset.arm.pair_role === "hybrid_treatment");
  const attention = assets.find(
    (asset) => asset.arm.pair_role === "all_attention_control"
  );
  if (!hybrid || !attention) {
    throw new Error("Matched assets are missing the declared treatment/control roles.");
  }
  if (
    !hybrid.config.layer_types?.includes("conv") ||
    !hybrid.config.layer_types?.includes("attn")
  ) {
    throw new Error("Hybrid treatment must contain both convolution and attention layers.");
  }
  if (
    !Array.isArray(attention.config.layer_types) ||
    attention.config.layer_types.some((kind) => kind !== "attn")
  ) {
    throw new Error("All-attention control must contain attention layers only.");
  }
}

function validatePairAccounting(manifest, assets) {
  const hybrid = assets.find((asset) => asset.arm.pair_role === "hybrid_treatment");
  const attention = assets.find(
    (asset) => asset.arm.pair_role === "all_attention_control"
  );
  const hybridParameters = hybrid.provenance.model.full_model_parameters;
  const attentionParameters = attention.provenance.model.full_model_parameters;
  const relativeDelta = Math.abs(attentionParameters - hybridParameters) / hybridParameters;
  if (
    manifest.match?.hybrid_parameters !== hybridParameters ||
    manifest.match?.attention_parameters !== attentionParameters ||
    !Number.isFinite(manifest.match?.relative_parameter_delta) ||
    Math.abs(manifest.match.relative_parameter_delta - relativeDelta) > 1e-15
  ) {
    throw new Error("Pair manifest parameter accounting disagrees with model provenance.");
  }
  for (const field of ["ffn_hidden", "layer_types", "name"]) {
    const declared = manifest.intentional_differences[field];
    if (
      canonicalJson(declared?.hybrid_treatment) !== canonicalJson(hybrid.config[field]) ||
      canonicalJson(declared?.all_attention_control) !== canonicalJson(attention.config[field])
    ) {
      throw new Error(`Pair manifest intentional difference '${field}' is inconsistent.`);
    }
  }
}

function validateHiddenOnlySession(session, armId) {
  const inputNames = [...(session.inputNames || [])];
  const outputNames = [...(session.outputNames || [])];
  if (inputNames.length !== 1 || inputNames[0] !== "input_ids") {
    throw new Error(
      `${armId} graph inputs are ${JSON.stringify(inputNames)}; expected only input_ids.`
    );
  }
  if (outputNames.length !== 1 || outputNames[0] !== "hidden") {
    throw new Error(
      `${armId} graph outputs are ${JSON.stringify(outputNames)}; ` +
      "latency harness accepts hidden-only graphs and rejects logits."
    );
  }
  return { inputNames, outputNames };
}

function ortWebGpuEvidence() {
  const webgpu = globalThis.ort?.env?.webgpu;
  const adapter = webgpu?.adapter || null;
  const device = webgpu?.device || null;
  const info = adapter?.info || {};
  return {
    ort_adapter_available: Boolean(adapter),
    ort_device_available: Boolean(device),
    adapter_info: adapter ? {
      vendor: info.vendor || null,
      architecture: info.architecture || null,
      device: info.device || null,
      description: info.description || null,
      is_fallback_adapter: adapter.isFallbackAdapter ?? null,
    } : null,
    profiling_callback_configured: typeof webgpu?.profiling?.ondata === "function",
  };
}

function reportedOrtVersion() {
  const version = globalThis.ort?.version || globalThis.ort?.env?.versions?.web;
  return typeof version === "string" && version ? version : null;
}

function ortVersionEvidence() {
  const reported = reportedOrtVersion();
  return {
    ort_version_reported: reported,
    ort_version_verified:
      reported == null ? null : reported === BACKBONE_ORT_VERSION,
    ort_version_verification_status:
      reported == null
        ? "unknown_runtime_did_not_report_version"
        : reported === BACKBONE_ORT_VERSION
          ? "matches_script_pin"
          : "mismatch",
  };
}

function verifyOrtVersionPin() {
  const evidence = ortVersionEvidence();
  if (evidence.ort_version_verified === false) {
    throw new Error(
      `Loaded ONNX Runtime Web ${evidence.ort_version_reported}, ` +
      `expected pinned ${BACKBONE_ORT_VERSION}.`
    );
  }
  return evidence;
}

function providerEvidence(provider) {
  const webgpu = ortWebGpuEvidence();
  if (provider !== "webgpu" && provider !== "wasm") {
    return {
      provider_requested: provider,
      provider_verified: false,
      provider_verification_method: "provider request rejected before session creation",
      provider_verified_scope: "invalid provider request",
      ort_webgpu: webgpu,
    };
  }
  if (provider === "webgpu") {
    const directlyObserved = webgpu.ort_adapter_available && webgpu.ort_device_available;
    return {
      provider_requested: provider,
      provider_verified: directlyObserved ? true : null,
      provider_verification_method: directlyObserved
        ? "ORT env exposes the adapter and device used after exact-one-provider session creation"
        : "exact-one-provider session creation succeeded; this ORT build exposes no adapter/device",
      provider_verified_scope:
        "session provider and ORT WebGPU device only; per-node execution placement is not claimed",
      ort_webgpu: webgpu,
    };
  }
  return {
    provider_requested: provider,
    provider_verified: true,
    provider_verification_method:
      "session created from an exact-one-provider executionProviders:['wasm'] request",
    provider_verified_scope:
      "session provider contract only; per-node execution placement is not exposed",
    ort_webgpu: webgpu,
  };
}

async function loadBackboneBundle(provider) {
  const loadStarted = performance.now();
  ort.env.wasm.wasmPaths = BACKBONE_ORT_WASM_BASE_URL;
  verifyOrtVersionPin();

  const manifestUrl = resolveArtifactUrl(requestedManifestUrl());
  BACKBONE_STATE.manifestUrl = manifestUrl;
  const manifestStarted = performance.now();
  const manifestResponse = await fetch(manifestUrl);
  if (!manifestResponse.ok) {
    throw new Error(
      `Missing matched-backbones manifest ${manifestUrl} (HTTP ${manifestResponse.status}); ` +
      "export the two pinned hidden-only graphs first."
    );
  }
  const manifestText = await manifestResponse.text();
  const manifestReadFinished = performance.now();
  const manifest = validateBackboneManifest(JSON.parse(manifestText));
  const manifestHashStarted = performance.now();
  const manifestSha256 = await sha256Text(manifestText);
  const manifestFinished = performance.now();
  BACKBONE_STATE.manifest = manifest;
  BACKBONE_STATE.manifestSha256 = manifestSha256;
  BACKBONE_STATE.bundleRecords.push({
    phase: "bundle",
    artifact_kind: "benchmark_manifest",
    artifact_id: "matched_backbones",
    url: manifestUrl,
    fetch_and_read_ms: manifestReadFinished - manifestStarted,
    hash_ms: manifestFinished - manifestHashStarted,
    total_fetch_read_hash_ms: manifestFinished - manifestStarted,
    bytes: new TextEncoder().encode(manifestText).length,
    actual_sha256: manifestSha256,
    hash_computed: true,
    hash_verified: null,
    hash_verification_status: "unknown_no_external_expected_digest",
    content_type: manifestResponse.headers.get("content-type") || null,
    browser_cache_state: "unknown",
    browser_cache_state_reason:
      "Fetch response does not expose a reliable network-versus-browser-cache classification",
    ...BACKBONE_LABELS,
  });

  const assets = [];
  for (const [role, modelEntry] of Object.entries(manifest.models)) {
    const provenanceEntry = manifest.artifacts[modelEntry.provenance];
    const provenanceArtifact = await fetchVerifiedArtifact(
      modelEntry.provenance,
      provenanceEntry.sha256,
      "model_provenance",
      role,
      manifestUrl
    );
    if (provenanceArtifact.record.bytes !== provenanceEntry.bytes) {
      throw new Error(`${role} provenance byte count does not match the pair manifest.`);
    }
    const provenance = validateBackboneProvenance(
      decodeJsonBuffer(provenanceArtifact.buffer, modelEntry.provenance),
      role,
      modelEntry,
      manifest.shared_random_seed
    );
    const config = provenance.model.config;
    validateConfig(config, modelEntry.name);
    const computedConfigSha256 = await sha256Text(modelConfigCanonicalJson(config));
    if (computedConfigSha256 !== provenance.model.config_canonical_sha256) {
      throw new Error(
        `${role} embedded config SHA-256 mismatch: expected ` +
        `${provenance.model.config_canonical_sha256}, got ${computedConfigSha256}.`
      );
    }

    const graphEntry =
      provenance.artifacts["backbone.fp16.onnx"] ||
      provenance.artifacts["backbone.fp32.onnx"];
    const graphRelativePath = `${modelEntry.directory}/${graphEntry.file}`;
    const pairGraphEntry = manifest.artifacts[graphRelativePath];
    if (
      !pairGraphEntry ||
      pairGraphEntry.sha256 !== graphEntry.sha256 ||
      pairGraphEntry.bytes !== graphEntry.bytes
    ) {
      throw new Error(`${role} graph provenance disagrees with the pair manifest.`);
    }
    const graphArtifact = await fetchVerifiedArtifact(
      graphRelativePath,
      graphEntry.sha256,
      "hidden_only_onnx_graph",
      role,
      manifestUrl
    );
    if (graphArtifact.record.bytes !== graphEntry.bytes) {
      throw new Error(`${role} graph byte count does not match provenance.`);
    }

    const configEntry = provenance.artifacts["model-config.yaml"];
    const configRelativePath = `${modelEntry.directory}/${configEntry.file}`;
    const pairConfigEntry = manifest.artifacts[configRelativePath];
    if (
      !pairConfigEntry ||
      pairConfigEntry.sha256 !== configEntry.sha256 ||
      pairConfigEntry.bytes !== configEntry.bytes ||
      configEntry.sha256 !== provenance.model.config_source_sha256
    ) {
      throw new Error(`${role} config provenance disagrees with the pair manifest.`);
    }
    const configArtifact = await fetchVerifiedArtifact(
      configRelativePath,
      configEntry.sha256,
      "model_config_source",
      role,
      manifestUrl
    );
    if (configArtifact.record.bytes !== configEntry.bytes) {
      throw new Error(`${role} config byte count does not match provenance.`);
    }

    const arm = {
      id: modelEntry.name,
      pair_role: role,
      graph_kind: "hidden_only",
      graph_file: graphRelativePath,
      provenance_file: modelEntry.provenance,
      config_file: configRelativePath,
      model_parameters: provenance.model.full_model_parameters,
      precision: graphEntry.precision,
    };
    assets.push({
      arm,
      config,
      provenance,
      provenanceSha256: provenanceArtifact.record.actual_sha256,
      graphBuffer: graphArtifact.buffer,
      graphBytes: graphArtifact.record.bytes,
      graphSha256: graphArtifact.record.actual_sha256,
      configSha256: provenance.model.config_canonical_sha256,
      configSourceSha256: configArtifact.record.actual_sha256,
    });
  }
  validateMatchedConfigs(assets);
  validatePairAccounting(manifest, assets);
  await prepareBackboneInputs(assets[0].config.vocab_size);

  const sessionOrder = shuffled(assets, `${BACKBONE_DEFAULT_SEED}:session-create`);
  for (let orderIndex = 0; orderIndex < sessionOrder.length; orderIndex++) {
    const asset = sessionOrder[orderIndex];
    const started = performance.now();
    let session;
    let sessionError = null;
    try {
      session = await ort.InferenceSession.create(
        asset.graphBuffer,
        backboneSessionOptions(provider)
      );
      const contract = validateHiddenOnlySession(session, asset.arm.id);
      const evidence = providerEvidence(provider);
      BACKBONE_STATE.sessionRecords.push({
        phase: "session_create",
        arm_id: asset.arm.id,
        order_index: orderIndex,
        session_create_ms: performance.now() - started,
        graph_sha256: asset.graphSha256,
        config_sha256: asset.configSha256,
        config_source_sha256: asset.configSourceSha256,
        provenance_sha256: asset.provenanceSha256,
        graph_bytes: asset.graphBytes,
        input_names: contract.inputNames,
        output_names: contract.outputNames,
        ...evidence,
        ...BACKBONE_LABELS,
        error: null,
      });
    } catch (error) {
      sessionError = errorDetail(error);
      BACKBONE_STATE.sessionRecords.push({
        phase: "session_create",
        arm_id: asset.arm.id,
        order_index: orderIndex,
        session_create_ms: performance.now() - started,
        graph_sha256: asset.graphSha256,
        config_sha256: asset.configSha256,
        config_source_sha256: asset.configSourceSha256,
        provenance_sha256: asset.provenanceSha256,
        graph_bytes: asset.graphBytes,
        ...providerEvidence(provider),
        ...BACKBONE_LABELS,
        error: sessionError,
      });
      throw error;
    }
    BACKBONE_STATE.arms.set(asset.arm.id, {
      ...asset,
      session,
      graphBuffer: null,
      sessionError,
    });
  }
  BACKBONE_STATE.readyAtMs = performance.now();
  return {
    bundle_and_session_ms: BACKBONE_STATE.readyAtMs - loadStarted,
    provider: providerEvidence(provider),
  };
}

function phaseRecordBase(phase, condition, armAsset, provider, globalOrder) {
  return {
    phase,
    global_order_index: globalOrder,
    arm_id: condition.arm_id,
    input_tokens: condition.input_tokens,
    actual_tensor_tokens: BACKBONE_STATE.inputs.get(condition.input_tokens)?.actual_tensor_tokens,
    repetition: condition.repetition,
    order_index: condition.order_index,
    graph_sha256: armAsset.graphSha256,
    config_sha256: armAsset.configSha256,
    provider_requested: provider,
    provider_verified: providerEvidence(provider).provider_verified,
    ...BACKBONE_LABELS,
  };
}

async function runBackboneInference(phase, condition, provider, globalOrder) {
  const armAsset = BACKBONE_STATE.arms.get(condition.arm_id);
  const input = BACKBONE_STATE.inputs.get(condition.input_tokens);
  if (!armAsset || !input) throw new Error("Unknown benchmark condition.");
  const base = phaseRecordBase(phase, condition, armAsset, provider, globalOrder);
  const started = performance.now();
  let hidden = null;
  try {
    const outputs = await armAsset.session.run({ input_ids: input.tensor });
    const inferenceMs = performance.now() - started;
    hidden = outputs.hidden;
    if (!hidden || !Array.isArray(hidden.dims)) {
      throw new Error(`${condition.arm_id} did not return a hidden tensor.`);
    }
    const expectedDims = [1, condition.input_tokens, armAsset.config.d_model];
    if (
      hidden.dims.length !== expectedDims.length ||
      hidden.dims.some((value, index) => value !== expectedDims[index])
    ) {
      throw new Error(
        `${condition.arm_id} returned hidden dims ${JSON.stringify(hidden.dims)}, ` +
        `expected ${JSON.stringify(expectedDims)}.`
      );
    }
    return {
      ...base,
      run_ok: true,
      inference_ms: inferenceMs,
      output_name: "hidden",
      output_dtype: hidden.type,
      output_dims: [...hidden.dims],
      output_location: "onnxruntime_default_cpu_output",
      output_dispose_api_available: typeof hidden.dispose === "function",
      output_release_attempted_after_validation: true,
      error: null,
    };
  } catch (error) {
    return {
      ...base,
      run_ok: false,
      inference_ms: performance.now() - started,
      output_name: null,
      output_dtype: null,
      output_dims: null,
      output_location: "onnxruntime_default_cpu_output",
      output_dispose_api_available: typeof hidden?.dispose === "function",
      output_release_attempted_after_validation: Boolean(hidden),
      error: errorDetail(error),
    };
  } finally {
    // The largest hidden output is several MiB. Retaining one tensor per run would turn the tail
    // benchmark into a GC/memory-growth benchmark. ORT defaults graph outputs to CPU unless an
    // output-location override is supplied; this runner deliberately uses that action-graph
    // contract and releases each output immediately after shape validation.
    try {
      hidden?.dispose?.();
    } catch {
      // Disposal support varies by ORT build. Dropping the final reference remains the fallback.
    }
  }
}

function setBackboneStatus(kind, text, provider = null) {
  const status = backboneElement("status");
  if (status) status.className = `status ${kind}`;
  const label = backboneElement("status-text");
  if (label) label.textContent = text;
  const badge = backboneElement("backend-badge");
  if (badge && provider) {
    badge.textContent = provider;
    badge.hidden = false;
  }
}

function setBackboneProgress(text) {
  const progress = backboneElement("backbone-progress");
  if (progress) progress.textContent = text;
}

function fixed(value, digits = 2) {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : "—";
}

function renderBackboneSummary(payload) {
  const output = backboneElement("backbone-output");
  if (!output || !payload.summary) return;
  const rows = payload.summary.conditions.map((condition) => {
    const latency = condition.inference_latency_ms;
    return (
      `<tr><td>${condition.arm_id}</td><td>${condition.input_tokens}</td>` +
      `<td>${condition.completed}/${condition.attempted}</td>` +
      `<td>${fixed(latency.p50)}</td><td>${fixed(latency.p95)}</td>` +
      `<td>${fixed(latency.p99)}</td></tr>`
    );
  }).join("");
  output.innerHTML = `
    <div class="metric-grid">
      <div class="metric"><span>Measured forwards</span><strong>${payload.summary.attempted}</strong></div>
      <div class="metric"><span>Completed forwards</span><strong>${payload.summary.completed}</strong></div>
      <div class="metric"><span>Failed forwards</span><strong>${payload.summary.failed}</strong></div>
      <div class="metric"><span>Quality metrics</span><strong>none</strong></div>
    </div>
    <table class="benchmark-table">
      <thead><tr><th>Backbone</th><th>Tokens</th><th>Completed</th>
      <th>p50 ms</th><th>p95 ms</th><th>p99 ms</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <details><summary>Protocol metadata</summary><pre>${JSON.stringify({
      latency_only: payload.latency_only,
      untrained_random_weights: payload.untrained_random_weights,
      metadata: payload.metadata,
    }, null, 2)}</pre></details>`;
}

function publicArmMetadata() {
  return Array.from(BACKBONE_STATE.arms.values()).map((asset) => ({
    id: asset.arm.id,
    pair_role: asset.arm.pair_role,
    graph_kind: asset.arm.graph_kind,
    graph_file: asset.arm.graph_file,
    graph_sha256: asset.graphSha256,
    graph_bytes: asset.graphBytes,
    config_file: asset.arm.config_file,
    config_sha256: asset.configSha256,
    config_source_sha256: asset.configSourceSha256,
    provenance_file: asset.arm.provenance_file,
    provenance_sha256: asset.provenanceSha256,
    model_parameters: asset.arm.model_parameters,
    precision: asset.arm.precision || null,
    config: asset.config,
    provenance: asset.provenance,
    ...BACKBONE_LABELS,
  }));
}

function publishBackbonePayload(payload) {
  LAST_BACKBONE_BENCHMARK = payload;
  if (typeof window !== "undefined") {
    window.__localAgentBackboneBenchmarkResult = payload;
  }
  const resultNode = typeof document !== "undefined"
    ? backboneElement("backbone-result-json")
    : null;
  if (resultNode) resultNode.textContent = JSON.stringify(payload);
  const download = typeof document !== "undefined"
    ? backboneElement("download-backbone-benchmark")
    : null;
  if (download) download.disabled = false;
  return payload;
}

function makeBackbonePayload({
  status,
  provider,
  seed,
  warmups,
  repetitions,
  firstInferenceRecords,
  warmupRecords,
  records,
  errors,
}) {
  const evidence = providerEvidence(provider);
  const ortVersion = ortVersionEvidence();
  const manifest = BACKBONE_STATE.manifest;
  return {
    schema_version: 1,
    benchmark: "localagent_matched_hidden_backbone_latency",
    status,
    created_at: new Date().toISOString(),
    ...BACKBONE_LABELS,
    metadata: {
      protocol_version: "backbone-latency-0.1",
      latency_boundary:
        "performance.now immediately before session.run through resolved hidden tensor output",
      output_location:
        "ONNX Runtime default CPU output; WebGPU timing therefore includes hidden-state readback",
      output_release:
        "hidden.dispose() is attempted after shape validation on every run; outputs are never retained",
      excluded_from_latency:
        "bundle fetch/hash, provenance/config load, deterministic ID generation, tensor construction, " +
        "session creation, output-shape validation, and all quality evaluation",
      provider_requested: provider,
      provider_verified: evidence.provider_verified,
      provider_verification_method: evidence.provider_verification_method,
      provider_verified_scope: evidence.provider_verified_scope,
      execution_provider_list:
        provider === "webgpu" || provider === "wasm" ? [provider] : [],
      whole_session_provider_retry: false,
      per_node_placement_verified: false,
      per_node_fallback_status: "unknown",
      ort_webgpu: evidence.ort_webgpu,
      ort_script_url: BACKBONE_ORT_SCRIPT_URL,
      ort_version_pin: BACKBONE_ORT_VERSION,
      ...ortVersion,
      cross_origin_isolated: globalThis.crossOriginIsolated ?? null,
      shared_array_buffer_available: typeof globalThis.SharedArrayBuffer !== "undefined",
      ort_wasm_num_threads: globalThis.ort?.env?.wasm?.numThreads ?? null,
      user_agent: globalThis.navigator?.userAgent || null,
      language: globalThis.navigator?.language || null,
      hardware_concurrency: globalThis.navigator?.hardwareConcurrency || null,
      device_memory_gb: globalThis.navigator?.deviceMemory || null,
      timer: "performance.now",
      concurrency: 1,
      tab_visibility_required: true,
      manifest_url: BACKBONE_STATE.manifestUrl,
      manifest_sha256: BACKBONE_STATE.manifestSha256,
      manifest: manifest,
      context_lengths: [...BACKBONE_CONTEXT_LENGTHS],
      context_condition: "exact_input_tensor_sequence_length",
      input_semantics: "deterministic_pretokenized_ids",
      input_fixture_contract: "ids[i]=(131*i+17) mod vocab_size",
      tokenizer_asset: null,
      arm_count: BACKBONE_STATE.arms.size,
      arms: publicArmMetadata(),
      case_order_seed: seed,
      session_order_seed: `${BACKBONE_DEFAULT_SEED}:session-create`,
      warmups_per_condition: warmups,
      measured_repetitions_per_condition: repetitions,
      first_inference_definition:
        "first session.run for every graph-by-length condition; eight randomized raw records",
      first_ever_for_graph_definition:
        "true only on the earliest first-inference condition encountered for each graph",
      first_for_condition_definition:
        "true on all eight cold-shape records; no earlier run used that graph and input length",
      warm_summary_excludes:
        "bundle, session creation, first inference, and every warmup record",
      page_to_ready_ms: BACKBONE_STATE.readyAtMs != null &&
        globalThis.window?.__localAgentBackboneBenchmarkStart != null
        ? BACKBONE_STATE.readyAtMs - window.__localAgentBackboneBenchmarkStart
        : null,
    },
    bundle_records: [...BACKBONE_STATE.bundleRecords],
    session_records: [...BACKBONE_STATE.sessionRecords],
    input_preparation_record: BACKBONE_STATE.inputPreparationRecord,
    inputs: Array.from(BACKBONE_STATE.inputs.values()).map((input) => ({
      ...input,
      tensor: undefined,
    })),
    first_inference_records: firstInferenceRecords,
    warmup_records: warmupRecords,
    records,
    summary: records.length ? summarizeBackboneRecords(records) : null,
    errors,
  };
}

async function runBackboneBenchmark() {
  if (BACKBONE_RUN_STARTED) {
    return LAST_BACKBONE_BENCHMARK;
  }
  const runButton = backboneElement("start-backbone-benchmark");
  let provider =
    window.__localAgentBackboneRequestedProvider ||
    new URLSearchParams(window.location.search).get("backend") ||
    "webgpu";
  const warmups = Number.parseInt(backboneElement("backbone-warmups").value, 10);
  const repetitions = Number.parseInt(backboneElement("backbone-repetitions").value, 10);
  const seed = backboneElement("backbone-seed").value.trim() || BACKBONE_DEFAULT_SEED;
  const firstInferenceRecords = [];
  const warmupRecords = [];
  const records = [];
  const errors = [];
  try {
    provider = requireExplicitProvider(provider);
    if (!Number.isInteger(warmups) || warmups < BACKBONE_MIN_WARMUPS) {
      throw new Error(`Warmups must be at least ${BACKBONE_MIN_WARMUPS} per condition.`);
    }
    if (!Number.isInteger(repetitions) || repetitions < BACKBONE_MIN_REPETITIONS) {
      throw new Error(
        `Measured repetitions must be at least ${BACKBONE_MIN_REPETITIONS} per condition.`
      );
    }
    const armIds = Array.from(BACKBONE_STATE.arms.keys());
    if (armIds.length !== 2) {
      throw new Error("Two verified sessions must be ready before running the benchmark.");
    }
    if (document.visibilityState !== "visible") {
      throw new Error("Run cannot start while the benchmark tab is hidden.");
    }
    BACKBONE_RUN_STARTED = true;
    if (runButton) runButton.disabled = true;

    const firstSchedule = buildBackboneSchedule(
      armIds, BACKBONE_CONTEXT_LENGTHS, 1, seed, "first_inference"
    );
    const graphsAlreadyInferred = new Set();
    for (let index = 0; index < firstSchedule.length; index++) {
      if (document.visibilityState !== "visible") {
        throw new Error("Run stopped because the benchmark tab became hidden.");
      }
      const condition = firstSchedule[index];
      setBackboneProgress(
        `First condition inference ${index + 1}/${firstSchedule.length}: ` +
        `${condition.arm_id} @ ${condition.input_tokens}`
      );
      const record = await runBackboneInference(
        "first_inference", condition, provider, index
      );
      record.first_for_condition = true;
      record.first_ever_for_graph = !graphsAlreadyInferred.has(condition.arm_id);
      graphsAlreadyInferred.add(condition.arm_id);
      firstInferenceRecords.push(record);
      if (!record.run_ok) {
        throw new Error(
          `First inference failed for ${condition.arm_id} @ ${condition.input_tokens}: ` +
          record.error.message
        );
      }
    }

    const warmSchedule = buildBackboneSchedule(
      armIds, BACKBONE_CONTEXT_LENGTHS, warmups, seed, "warmup"
    );
    for (let index = 0; index < warmSchedule.length; index++) {
      if (document.visibilityState !== "visible") {
        throw new Error("Run stopped because the benchmark tab became hidden.");
      }
      const condition = warmSchedule[index];
      setBackboneProgress(
        `Warmup ${index + 1}/${warmSchedule.length}: ` +
        `${condition.arm_id} @ ${condition.input_tokens}`
      );
      const record = await runBackboneInference("warmup", condition, provider, index);
      warmupRecords.push(record);
      if (!record.run_ok) {
        throw new Error(`Warmup failed for ${condition.arm_id}: ${record.error.message}`);
      }
    }

    const measuredSchedule = buildBackboneSchedule(
      armIds, BACKBONE_CONTEXT_LENGTHS, repetitions, seed, "measured"
    );
    for (let index = 0; index < measuredSchedule.length; index++) {
      if (document.visibilityState !== "visible") {
        throw new Error("Run stopped because the benchmark tab became hidden.");
      }
      const condition = measuredSchedule[index];
      setBackboneProgress(
        `Measured ${index + 1}/${measuredSchedule.length}: ` +
        `${condition.arm_id} @ ${condition.input_tokens}`
      );
      const record = await runBackboneInference("measured", condition, provider, index);
      records.push(record);
      if (!record.run_ok) errors.push(record.error);
    }
    if (errors.length) {
      throw new Error(
        `${errors.length} measured hidden-only forward(s) failed; run is not complete.`
      );
    }
    const payload = makeBackbonePayload({
      status: "complete",
      provider,
      seed,
      warmups,
      repetitions,
      firstInferenceRecords,
      warmupRecords,
      records,
      errors,
    });
    publishBackbonePayload(payload);
    renderBackboneSummary(payload);
    setBackboneStatus("ready", "Latency-only backbone run complete.", provider);
    setBackboneProgress(
      `Complete: ${records.length} measured hidden-only forwards; raw result is globally available.`
    );
    return payload;
  } catch (error) {
    errors.push(errorDetail(error));
    const payload = makeBackbonePayload({
      status: "failed",
      provider,
      seed,
      warmups,
      repetitions,
      firstInferenceRecords,
      warmupRecords,
      records,
      errors,
    });
    publishBackbonePayload(payload);
    renderBackboneSummary(payload);
    setBackboneStatus("error", `Benchmark failed: ${error.message}`, provider);
    setBackboneProgress(
      "Partial raw records and error details are available for download and in the result global."
    );
    if (!BACKBONE_RUN_STARTED && runButton) runButton.disabled = false;
    return payload;
  }
}

function downloadBackboneBenchmark() {
  if (!LAST_BACKBONE_BENCHMARK) return;
  const blob = new Blob(
    [JSON.stringify(LAST_BACKBONE_BENCHMARK, null, 2)],
    { type: "application/json" }
  );
  const anchor = document.createElement("a");
  anchor.href = URL.createObjectURL(blob);
  anchor.download = `localagent-backbone-latency-${Date.now()}.json`;
  anchor.click();
  URL.revokeObjectURL(anchor.href);
}

async function initializeBackboneBenchmark() {
  let provider =
    window.__localAgentBackboneRequestedProvider ||
    new URLSearchParams(window.location.search).get("backend") ||
    "webgpu";
  setBackboneStatus("loading", "Verifying two hidden-only graph bundles…", provider);
  setBackboneProgress("No inference is issued during bundle or session setup.");
  try {
    provider = requireExplicitProvider(provider);
    await loadBackboneBundle(provider);
    const readyPayload = makeBackbonePayload({
      status: "ready",
      provider,
      seed: BACKBONE_DEFAULT_SEED,
      warmups: BACKBONE_MIN_WARMUPS,
      repetitions: BACKBONE_MIN_REPETITIONS,
      firstInferenceRecords: [],
      warmupRecords: [],
      records: [],
      errors: [],
    });
    publishBackbonePayload(readyPayload);
    backboneElement("start-backbone-benchmark").disabled = false;
    setBackboneStatus(
      "ready",
      "Two verified sessions ready; no inference has run yet.",
      provider
    );
    setBackboneProgress(
      "Ready. The first click preserves one first-inference record per graph and length."
    );
  } catch (error) {
    const payload = makeBackbonePayload({
      status: "load_failed",
      provider,
      seed: BACKBONE_DEFAULT_SEED,
      warmups: BACKBONE_MIN_WARMUPS,
      repetitions: BACKBONE_MIN_REPETITIONS,
      firstInferenceRecords: [],
      warmupRecords: [],
      records: [],
      errors: [errorDetail(error)],
    });
    publishBackbonePayload(payload);
    setBackboneStatus("error", `Bundle/session setup failed: ${error.message}`, provider);
    setBackboneProgress(
      "No whole-session provider retry was attempted. Error details are in the result global."
    );
  }
}

if (typeof window !== "undefined") {
  window.__localAgentBackboneBenchmarkResult = null;
}

if (
  typeof window !== "undefined" &&
  typeof document !== "undefined" &&
  !window.__localAgentSkipInit
) {
  backboneElement("start-backbone-benchmark").addEventListener(
    "click", runBackboneBenchmark
  );
  backboneElement("download-backbone-benchmark").addEventListener(
    "click", downloadBackboneBenchmark
  );
  initializeBackboneBenchmark();
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    BACKBONE_CONTEXT_LENGTHS,
    BACKBONE_LABELS,
    buildBackboneSchedule,
    latencySummary,
    modelConfigCanonicalJson,
    seededRandom,
    shuffled,
    summarizeBackboneRecords,
    validateBackboneManifest,
    validateBackboneProvenance,
  };
}
