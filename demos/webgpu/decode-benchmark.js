/* LocalAgent cached-decode latency benchmark.
 *
 * Accepted artifacts are the legacy deterministic random-weight pair, a strictly pinned trained
 * pair, or one explicitly selected and cryptographically pinned trained-checkpoint export. This
 * runner measures only graph latency. It does not tokenize text, score generated token IDs, or
 * make a language, tool-use, action, or agent-capability claim.
 */
"use strict";

const DECODE_CONTEXT_LENGTHS = Object.freeze([128, 512, 1024, 1536]);
const DECODE_DEFAULT_MANIFEST_URL =
  "../../runs/webgpu/random-cached-decode-latency-seed-20260728-v2/matched-decode.json";
const DECODE_DEFAULT_SEED = "slmw2026-cached-decode-v1";
const DECODE_MIN_WARMUPS = 3;
const DECODE_MIN_REPETITIONS = 30;
const DECODE_DEFAULT_OUTPUT_TOKENS = 32;
const DECODE_ACCEPTANCE_PROTOCOL = Object.freeze({
  id: "cached-decode-acceptance-1",
  context_lengths: DECODE_CONTEXT_LENGTHS,
  output_tokens_per_condition: 32,
  warmups_per_condition: 3,
  measured_repetitions_per_condition: 30,
  case_order_seed: DECODE_DEFAULT_SEED,
});
const DECODE_ORT_VERSION = "1.27.0";
const DECODE_ORT_VENDOR_BASE_PATH =
  `vendor/onnxruntime-web-${DECODE_ORT_VERSION}/`;
const DECODE_ORT_SCRIPT_PATH = `${DECODE_ORT_VENDOR_BASE_PATH}ort.webgpu.min.js`;
const DECODE_ORT_WASM_FILE = "ort-wasm-simd-threaded.jsep.wasm";
const DECODE_ORT_WASM_PATH = `${DECODE_ORT_VENDOR_BASE_PATH}${DECODE_ORT_WASM_FILE}`;
const DECODE_HARNESS_SCHEMA_VERSION = 2;
const DECODE_HARNESS_HTML_FILE = "decode-benchmark.html";
const DECODE_HARNESS_JAVASCRIPT_FILE = "decode-benchmark.js";
const DECODE_EVIDENCE_SCOPE = Object.freeze({
  acquisition_bytes_externally_rooted: true,
  browser_execution_attested: false,
  gpu_hardware_attested: false,
  scope: "externally_rooted_acquisition_bytes_with_unattested_browser_gpu_execution",
});
const DECODE_LABELS = Object.freeze({
  latency_only: true,
  untrained_random_weights: true,
  capability_artifact: false,
  quality_evaluation: false,
});
const DECODE_TRAINED_LABEL =
  "trained weights, latency only; quality scored separately";
const DECODE_TRAINED_LABELS = Object.freeze({
  latency_only: true,
  untrained_random_weights: false,
  trained_weights: true,
  capability_artifact: false,
  action_capability_claimed: false,
  action_capability_evaluation: false,
  quality_evaluation: false,
  quality_scored_separately: true,
  artifact_manifest_latency_only: false,
  benchmark_label: DECODE_TRAINED_LABEL,
});
const DECODE_UNVERIFIED_LABELS = Object.freeze({
  latency_only: true,
  untrained_random_weights: null,
  trained_weights: null,
  capability_artifact: false,
  action_capability_claimed: false,
  quality_evaluation: false,
  artifact_mode: "unverified",
});
const DECODE_BENCHMARK_MODES = Object.freeze(["matched", "single"]);
const DECODE_SINGLE_MANIFEST_TYPE = "single_trained_cached_decode_suite";
const DECODE_DECISION_ABI_LOGITS =
  "final_logits_argmax_with_next_token_crosscheck";
const DECODE_DECISION_ABI_LEGACY = "legacy_exported_next_token_only";

let DECODE_STATE = {
  benchmarkMode: null,
  acceptanceMode: false,
  acceptanceRootSha256: null,
  manifest: null,
  artifactMode: null,
  manifestUrl: null,
  manifestSha256: null,
  manifestRawText: null,
  artifacts: [],
  sessions: [],
  arms: new Map(),
  inputs: new Map(),
  inputPreparationRecord: null,
  providerVerification: null,
  benchmarkSessionId: null,
  runId: null,
  runChallenge: null,
  externalMachineConditionSha256: null,
  acceptanceAcquisitionRoots: null,
  harnessIdentity: null,
  readyAtMs: null,
};
let LAST_DECODE_BENCHMARK = null;
let DECODE_RUN_STARTED = false;

function decodeLabelsForMode(mode) {
  if (mode === "random") return DECODE_LABELS;
  if (mode === "trained") return DECODE_TRAINED_LABELS;
  return DECODE_UNVERIFIED_LABELS;
}

function currentDecodeLabels() {
  return decodeLabelsForMode(DECODE_STATE.artifactMode);
}

function decodeElement(id) {
  return typeof document === "undefined" ? null : document.getElementById(id);
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

function buildDecodeSchedule(
  armIds,
  contextLengths,
  repetitions,
  seedText,
  phase,
  benchmarkMode = "matched"
) {
  const expectedArms = benchmarkMode === "single" ? 1 : 2;
  if (!DECODE_BENCHMARK_MODES.includes(benchmarkMode)) {
    throw new Error(`Unknown decode benchmark mode '${benchmarkMode}'.`);
  }
  if (!Array.isArray(armIds) || armIds.length !== expectedArms) {
    throw new Error(
      `A ${benchmarkMode} decode schedule requires exactly ${expectedArms} model arm(s).`
    );
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

function summarizeDecodeRecords(records, artifactMode = "random") {
  if (artifactMode !== "random" && artifactMode !== "trained") {
    throw new Error(`Cannot summarize unverified artifact mode '${artifactMode}'.`);
  }
  const groups = {};
  for (const record of records) {
    const key = `${record.arm_id}:${record.input_tokens}`;
    if (!groups[key]) {
      groups[key] = {
        arm_id: record.arm_id,
        input_tokens: record.input_tokens,
        output_tokens: record.output_tokens_requested,
        attempted: 0,
        completed: 0,
        failed: 0,
        prefill_graph_passes: 0,
        decode_graph_passes: 0,
        ttft_ms: [],
        tpot_ms: [],
        decode_tokens_per_second: [],
        prefill_ms: [],
        decode_inference_ms: [],
        model_decode_tokens_per_second: [],
        final_cache_logical_bytes: [],
      };
    }
    const group = groups[key];
    group.attempted += 1;
    group.prefill_graph_passes += record.graph_pass_counts?.prefill || 0;
    group.decode_graph_passes += record.graph_pass_counts?.decode || 0;
    if (record.run_ok) {
      group.completed += 1;
      for (const field of [
        "ttft_ms", "tpot_ms", "decode_tokens_per_second", "prefill_ms",
        "decode_inference_ms", "model_decode_tokens_per_second",
      ]) {
        group[field].push(record[field]);
      }
      group.final_cache_logical_bytes.push(record.cache?.final_logical_bytes);
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
      prompt_tokens: group.input_tokens,
      output_tokens: group.output_tokens,
      attempted: group.attempted,
      completed: group.completed,
      failed: group.failed,
      graph_pass_counts: {
        prefill: group.prefill_graph_passes,
        decode: group.decode_graph_passes,
        total: group.prefill_graph_passes + group.decode_graph_passes,
      },
      ttft_ms: latencySummary(group.ttft_ms),
      tpot_ms: latencySummary(group.tpot_ms),
      decode_tokens_per_second: latencySummary(group.decode_tokens_per_second),
      prefill_ms: latencySummary(group.prefill_ms),
      decode_inference_ms: latencySummary(group.decode_inference_ms),
      model_decode_tokens_per_second: latencySummary(
        group.model_decode_tokens_per_second
      ),
      final_cache_logical_bytes: latencySummary(group.final_cache_logical_bytes),
    }));
  return {
    estimand: artifactMode === "trained"
      ? "trained_weight_cached_autoregressive_graph_latency"
      : "random_weight_cached_autoregressive_graph_latency",
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

function arrayEquals(left, right) {
  return canonicalJson(left) === canonicalJson(right);
}

function validateNoContradictoryCapabilityFields(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be a JSON object.`);
  }
  for (const field of [
    "capability_artifact",
    "action_capability_claimed",
    "action_capability_artifact",
    "action_capability_evaluated",
    "action_capability_evaluation",
  ]) {
    if (Object.hasOwn(value, field) && value[field] !== false) {
      throw new Error(`${label}.${field} must be exactly false when present.`);
    }
  }
  for (const field of ["action_capability_claims", "capability_claims"]) {
    if (
      Object.hasOwn(value, field) &&
      (!Array.isArray(value[field]) || value[field].length !== 0)
    ) {
      throw new Error(`${label}.${field} must be exactly an empty array when present.`);
    }
  }
  for (const field of ["capability_metrics", "capability_artifact_type"]) {
    if (Object.hasOwn(value, field) && value[field] !== null) {
      throw new Error(`${label}.${field} must be exactly null when present.`);
    }
  }
  return value;
}

function requireExplicitProvider(provider) {
  if (provider !== "webgpu" && provider !== "wasm") {
    throw new Error(`Unknown provider '${provider}'; expected exactly webgpu or wasm.`);
  }
  return provider;
}

function validateArtifactPin(pin, field) {
  if (
    !pin ||
    typeof pin !== "object" ||
    !Number.isInteger(pin.bytes) ||
    pin.bytes < 1 ||
    !isSha256(pin.sha256)
  ) {
    throw new Error(`${field} must pin positive bytes and a SHA-256 digest.`);
  }
  return pin;
}

function validateCheckpointPin(pin, field) {
  const requiredFields = [
    "bytes", "checkpoint", "sha256", "stage", "step", "tokens_seen", "training_steps",
  ];
  if (
    !pin ||
    typeof pin !== "object" ||
    Array.isArray(pin) ||
    !arrayEquals(Object.keys(pin).sort(), [...requiredFields].sort()) ||
    typeof pin.checkpoint !== "string" ||
    !pin.checkpoint ||
    !["pretrain", "midtrain", "sft", "rl"].includes(pin.stage) ||
    !Number.isInteger(pin.step) ||
    pin.step < 0 ||
    pin.training_steps !== pin.step + 1
  ) {
    throw new Error(`${field} is not an exact supported trained-checkpoint pin.`);
  }
  if (
    (pin.stage === "rl" && pin.tokens_seen !== null) ||
    (
      pin.stage !== "rl" &&
      (!Number.isInteger(pin.tokens_seen) || pin.tokens_seen < 1)
    )
  ) {
    throw new Error(`${field} has invalid stage-specific token accounting.`);
  }
  validateArtifactPin(pin, field);
  return pin;
}

function validateTokenizerProvenancePin(pin, field) {
  const requiredFields = [
    "artifact", "artifact_identity", "checkpoint_metadata_present", "kind", "sha256",
    "verified", "vocab_size",
  ];
  const optionalExporterFields = [
    "bundled_artifact_identity", "encoding", "eos_id", "file", "pad_id",
  ];
  const keys = Object.keys(pin || {});
  if (
    !pin ||
    typeof pin !== "object" ||
    Array.isArray(pin) ||
    requiredFields.some((key) => !Object.hasOwn(pin, key)) ||
    keys.some((key) => !requiredFields.includes(key) && !optionalExporterFields.includes(key)) ||
    pin.checkpoint_metadata_present !== true ||
    pin.verified !== true ||
    (pin.kind !== "byte" && pin.kind !== "bpe") ||
    !isSha256(pin.sha256) ||
    !Number.isInteger(pin.vocab_size) ||
    pin.vocab_size < 1
  ) {
    throw new Error(`${field} is not an exact verified tokenizer-provenance pin.`);
  }
  if (pin.kind === "byte") {
    if (
      pin.vocab_size !== 256 ||
      pin.artifact !== null ||
      pin.artifact_identity !== null
    ) {
      throw new Error(`${field} has an invalid built-in byte-tokenizer identity.`);
    }
  } else {
    if (
      pin.vocab_size <= 256 ||
      typeof pin.artifact !== "string" ||
      !pin.artifact
    ) {
      throw new Error(`${field} has an invalid BPE tokenizer artifact label.`);
    }
    const artifact = validateArtifactPin(
      pin.artifact_identity,
      `${field}.artifact_identity`
    );
    if (artifact.sha256.toLowerCase() !== pin.sha256.toLowerCase()) {
      throw new Error(`${field} artifact identity disagrees with its tokenizer SHA-256.`);
    }
    if (
      Object.hasOwn(pin, "encoding") &&
      pin.encoding !== "bytelevel-bpe"
    ) {
      throw new Error(`${field} has an invalid BPE tokenizer encoding.`);
    }
    if (
      Object.hasOwn(pin, "file") &&
      pin.file !== "tokenizer.json"
    ) {
      throw new Error(`${field} must use the bundled tokenizer.json filename.`);
    }
    if (Object.hasOwn(pin, "bundled_artifact_identity")) {
      const bundled = validateArtifactPin(
        pin.bundled_artifact_identity,
        `${field}.bundled_artifact_identity`
      );
      if (
        bundled.bytes !== artifact.bytes ||
        bundled.sha256.toLowerCase() !== artifact.sha256.toLowerCase()
      ) {
        throw new Error(`${field} bundled tokenizer identity disagrees with its source.`);
      }
    }
  }
  if (
    Object.hasOwn(pin, "eos_id") &&
    (!Number.isInteger(pin.eos_id) || pin.eos_id < 0 || pin.eos_id >= pin.vocab_size)
  ) {
    throw new Error(`${field}.eos_id is outside the tokenizer vocabulary.`);
  }
  if (
    Object.hasOwn(pin, "pad_id") &&
    (!Number.isInteger(pin.pad_id) || pin.pad_id < 0 || pin.pad_id >= pin.vocab_size)
  ) {
    throw new Error(`${field}.pad_id is outside the tokenizer vocabulary.`);
  }
  return pin;
}

function tokenizerIdentityCore(pin) {
  return Object.fromEntries([
    "artifact", "artifact_identity", "checkpoint_metadata_present", "kind", "sha256",
    "verified", "vocab_size",
  ].map((key) => [key, pin?.[key] ?? null]));
}

function tokenizerPinsMatch(left, right) {
  return canonicalJson(tokenizerIdentityCore(left)) ===
    canonicalJson(tokenizerIdentityCore(right));
}

function decodeManifestMode(manifest) {
  const qualityClaimsEmpty =
    Array.isArray(manifest?.quality_claims) && manifest.quality_claims.length === 0;
  const noActionCapabilityClaim =
    manifest?.action_capability_claimed !== true &&
    manifest?.action_capability_artifact !== true &&
    (
      !Array.isArray(manifest?.action_capability_claims) ||
      manifest.action_capability_claims.length === 0
    );
  const randomContract =
    manifest?.artifact_type === "matched_random_cached_decode_latency_suite" &&
    manifest.latency_only === true &&
    manifest.trained === false &&
    manifest.capability_artifact === false &&
    qualityClaimsEmpty &&
    noActionCapabilityClaim &&
    Number.isInteger(manifest.shared_random_seed) &&
    manifest.shared_random_seed >= 0 &&
    !Object.hasOwn(manifest, "checkpoints") &&
    !Object.hasOwn(manifest, "tokenizer") &&
    !Object.hasOwn(manifest, "quality_evaluation");
  const trainedContract =
    manifest?.artifact_type === "matched_trained_cached_decode_suite" &&
    manifest.latency_only === false &&
    manifest.trained === true &&
    manifest.capability_artifact === false &&
    qualityClaimsEmpty &&
    noActionCapabilityClaim &&
    !Object.hasOwn(manifest, "shared_random_seed") &&
    manifest.quality_evaluation?.included === false &&
    manifest.quality_evaluation?.required_separately === true &&
    arrayEquals(
      Object.keys(manifest.quality_evaluation || {}).sort(),
      ["included", "required_separately"]
    );
  if (randomContract === trainedContract) {
    throw new Error(
      "Pair manifest has a mixed or ambiguous random/trained artifact contract."
    );
  }
  return trainedContract ? "trained" : "random";
}

function validateDecodeManifest(manifest) {
  if (!manifest || typeof manifest !== "object" || Array.isArray(manifest)) {
    throw new Error("Cached decode manifest must be a JSON object.");
  }
  validateNoContradictoryCapabilityFields(manifest, "cached decode manifest");
  if (manifest.schema_version !== 1) {
    throw new Error(`Unsupported cached decode manifest schema ${manifest.schema_version}.`);
  }
  const artifactMode = decodeManifestMode(manifest);
  const requiredControlledFields = [
    "conv_kernel", "d_model", "dropout", "embed_dim", "max_seq_len", "n_heads",
    "n_kv_heads", "n_layers", "n_loops", "norm_eps", "qk_norm", "rope_theta",
    "tie_embeddings", "vocab_size",
  ];
  if (!arrayEquals(
    [...(manifest.controlled_fields || [])].sort(),
    [...requiredControlledFields].sort()
  )) {
    throw new Error("Pair manifest controlled_fields do not match the architecture control.");
  }
  if (!arrayEquals(
    Object.keys(manifest.intentional_differences || {}).sort(),
    ["ffn_hidden", "layer_types", "name"]
  )) {
    throw new Error(
      "Pair manifest must declare exactly name, ffn_hidden, and layer_types as differences."
    );
  }
  if (
    !Number.isInteger(manifest.match?.hybrid_parameters) ||
    !Number.isInteger(manifest.match?.attention_parameters) ||
    !Number.isFinite(manifest.match?.relative_parameter_delta) ||
    manifest.match.relative_parameter_delta < 0 ||
    manifest.match.relative_parameter_delta >= 0.01
  ) {
    throw new Error("Pair manifest does not satisfy the <1% parameter-match contract.");
  }
  const models = Object.entries(manifest.models || {});
  if (models.length !== 2) {
    throw new Error("Pair manifest must contain exactly two model entries.");
  }
  const roles = models.map(([role]) => role).sort();
  if (!arrayEquals(roles, ["all_attention_control", "hybrid_treatment"])) {
    throw new Error(
      "Pair manifest must contain hybrid_treatment and all_attention_control roles."
    );
  }
  for (const [role, model] of models) {
    if (
      !model ||
      typeof model.name !== "string" ||
      !model.name ||
      typeof model.directory !== "string" ||
      !model.directory ||
      typeof model.provenance !== "string" ||
      !model.provenance
    ) {
      throw new Error(`models.${role} must name a model directory and provenance file.`);
    }
    validateArtifactPin(
      manifest.artifacts?.[model.provenance],
      `artifacts.${model.provenance}`
    );
  }
  if (artifactMode === "trained") {
    const checkpointRoles = Object.keys(manifest.checkpoints || {}).sort();
    if (!arrayEquals(checkpointRoles, roles)) {
      throw new Error("Trained pair must pin exactly one checkpoint for each model role.");
    }
    const checkpoints = checkpointRoles.map((role) =>
      validateCheckpointPin(
        manifest.checkpoints[role],
        `checkpoints.${role}`
      )
    );
    for (const field of ["stage", "step", "tokens_seen", "training_steps"]) {
      if (checkpoints[0][field] !== checkpoints[1][field]) {
        throw new Error(`Trained pair checkpoint pins disagree on controlled field ${field}.`);
      }
    }
    validateTokenizerProvenancePin(manifest.tokenizer, "tokenizer");
  }
  return manifest;
}

function validateSingleDecodeManifest(manifest) {
  if (!manifest || typeof manifest !== "object" || Array.isArray(manifest)) {
    throw new Error("Single-model cached decode manifest must be a JSON object.");
  }
  validateNoContradictoryCapabilityFields(manifest, "single-model manifest");
  const qualityClaimsEmpty =
    Array.isArray(manifest.quality_claims) && manifest.quality_claims.length === 0;
  const exactQualityEvaluation =
    manifest.quality_evaluation?.included === false &&
    manifest.quality_evaluation?.required_separately === true &&
    arrayEquals(
      Object.keys(manifest.quality_evaluation || {}).sort(),
      ["included", "required_separately"]
    );
  if (
    manifest.schema_version !== 1 ||
    manifest.artifact_type !== DECODE_SINGLE_MANIFEST_TYPE ||
    manifest.latency_only !== false ||
    manifest.trained !== true ||
    manifest.capability_artifact !== false ||
    manifest.action_capability_claimed === true ||
    !qualityClaimsEmpty ||
    !exactQualityEvaluation
  ) {
    throw new Error(
      "Single-model manifest must describe one trained, non-capability, quality-separate export."
    );
  }
  const model = manifest.model;
  if (
    !model ||
    typeof model !== "object" ||
    Array.isArray(model) ||
    typeof model.name !== "string" ||
    !model.name ||
    typeof model.pair_role !== "string" ||
    !model.pair_role ||
    typeof model.provenance !== "string" ||
    !model.provenance ||
    new URL(model.provenance, "https://localagent.invalid/").origin !==
      "https://localagent.invalid"
  ) {
    throw new Error(
      "Single-model manifest model must name one relative, pinned provenance file."
    );
  }
  const artifactKeys = Object.keys(manifest.artifacts || {});
  if (!arrayEquals(artifactKeys, [model.provenance])) {
    throw new Error("Single-model manifest must pin exactly its provenance file.");
  }
  validateArtifactPin(
    manifest.artifacts[model.provenance],
    `artifacts.${model.provenance}`
  );
  return manifest;
}

function checkpointPinFromProvenance(provenance) {
  const weights = provenance?.weights || {};
  return {
    bytes: weights.checkpoint_bytes,
    checkpoint: weights.checkpoint,
    sha256: weights.checkpoint_sha256,
    stage: weights.checkpoint_stage,
    step: weights.checkpoint_step,
    tokens_seen: weights.tokens_seen,
    training_steps: provenance?.training_steps,
  };
}

function singleDecodeProvenanceContext(provenance) {
  return {
    single_model: true,
    artifactMode: "trained",
    checkpoint: checkpointPinFromProvenance(provenance),
    tokenizer: provenance?.tokenizer,
  };
}

function validateCacheSlot(slot, index) {
  if (
    !slot ||
    typeof slot !== "object" ||
    slot.slot !== index ||
    !Number.isInteger(slot.loop) ||
    slot.loop < 0 ||
    !Number.isInteger(slot.layer) ||
    slot.layer < 0 ||
    (slot.kind !== "attn" && slot.kind !== "conv") ||
    !Array.isArray(slot.past_inputs) ||
    !Array.isArray(slot.present_outputs) ||
    !Array.isArray(slot.shape) ||
    !slot.dtype_by_precision ||
    slot.dtype_by_precision.fp32 !== "float32"
  ) {
    throw new Error(`cache_slots[${index}] has an invalid cache contract.`);
  }
  const expectedCount = slot.kind === "attn" ? 2 : 1;
  if (
    slot.past_inputs.length !== expectedCount ||
    slot.present_outputs.length !== expectedCount
  ) {
    throw new Error(`${slot.kind} cache slot ${index} has the wrong tensor count.`);
  }
  const expectedPast = slot.kind === "attn"
    ? [`past_${index}_key`, `past_${index}_value`]
    : [`past_${index}_conv`];
  const expectedPresent = slot.kind === "attn"
    ? [`present_${index}_key`, `present_${index}_value`]
    : [`present_${index}_conv`];
  if (!arrayEquals(slot.past_inputs, expectedPast) ||
      !arrayEquals(slot.present_outputs, expectedPresent)) {
    throw new Error(`cache slot ${index} does not use the canonical past/present names.`);
  }
  const expectedUpdate = slot.kind === "attn"
    ? "append_one_token_along_axis_2"
    : "replace_with_latest_fixed_width_tail";
  if (slot.update !== expectedUpdate) {
    throw new Error(`cache slot ${index} has an invalid update rule.`);
  }
  return slot;
}

function validateCacheSlotsAgainstConfig(contract, config, role) {
  if (
    config.d_model % config.n_heads !== 0 ||
    config.n_heads % config.n_kv_heads !== 0 ||
    !Array.isArray(config.layer_types) ||
    config.layer_types.length !== config.n_layers
  ) {
    throw new Error(`${role} config cannot define the declared GQA/cache layout.`);
  }
  const expectedSlots = config.n_loops * config.n_layers;
  if (contract.cache_slots.length !== expectedSlots) {
    throw new Error(
      `${role} cache slot count ${contract.cache_slots.length} != ${expectedSlots}.`
    );
  }
  contract.cache_slots.forEach((slot, index) => {
    const expectedLoop = Math.floor(index / config.n_layers);
    const expectedLayer = index % config.n_layers;
    const expectedKind = config.layer_types[expectedLayer];
    const expectedShape = expectedKind === "attn"
      ? ["batch", config.n_kv_heads, "cache_sequence", config.d_model / config.n_heads]
      : ["batch", config.d_model, config.conv_kernel - 1];
    if (
      slot.loop !== expectedLoop ||
      slot.layer !== expectedLayer ||
      slot.kind !== expectedKind ||
      !arrayEquals(slot.shape, expectedShape)
    ) {
      throw new Error(`${role} cache slot ${index} disagrees with the embedded model config.`);
    }
  });
}

function graphIoNames(graph, direction, role) {
  const values = graph?.[`${direction}_names`];
  if (!Array.isArray(values) || values.some((value) => typeof value !== "string" || !value)) {
    throw new Error(`${role}.${direction}_names must be a string array.`);
  }
  return values;
}

function expectedIoDtype(name, expectedCacheDtype) {
  return name === "input_ids" || name === "next_token" ? "int64" : expectedCacheDtype;
}

function validateTypedIo(entries, expectedNames, expectedCacheDtype, role, direction) {
  if (!Array.isArray(entries) || entries.length !== expectedNames.length) {
    throw new Error(`${role}.${direction} typed I/O length does not match its names.`);
  }
  entries.forEach((entry, index) => {
    const expectedDtype = expectedIoDtype(expectedNames[index], expectedCacheDtype);
    if (
      !entry ||
      entry.name !== expectedNames[index] ||
      entry.dtype !== expectedDtype ||
      !Array.isArray(entry.shape)
    ) {
      throw new Error(
        `${role}.${direction}[${index}] does not match ${expectedNames[index]} ` +
        `${expectedDtype}.`
      );
    }
  });
}

function validatePrecisionGraphContract(contract, precision, role, vocabSize) {
  const graph = contract.graphs?.[precision];
  const expectedCacheDtype = precision === "fp16" ? "float16" : "float32";
  if (!graph || graph.cache_dtype !== expectedCacheDtype) {
    throw new Error(`${role} has no valid ${precision} graph contract.`);
  }
  const pastNames = contract.cache_slots.flatMap((slot) => slot.past_inputs);
  const presentNames = contract.cache_slots.flatMap((slot) => slot.present_outputs);
  const prefillInputs = graphIoNames(graph.prefill, "input", `${role}.${precision}.prefill`);
  const prefillOutputs = graphIoNames(
    graph.prefill, "output", `${role}.${precision}.prefill`
  );
  const decodeInputs = graphIoNames(graph.decode, "input", `${role}.${precision}.decode`);
  const decodeOutputs = graphIoNames(graph.decode, "output", `${role}.${precision}.decode`);
  const logitsAbi =
    arrayEquals(prefillOutputs, ["next_token", "logits", ...presentNames]) &&
    arrayEquals(decodeOutputs, ["next_token", "logits", ...presentNames]);
  const legacyAbi =
    arrayEquals(prefillOutputs, ["next_token", ...presentNames]) &&
    arrayEquals(decodeOutputs, ["next_token", ...presentNames]);
  if (!logitsAbi && !legacyAbi) {
    throw new Error(`${role} ${precision} graph outputs do not match a supported cache ABI.`);
  }
  const decisionAbi = logitsAbi
    ? DECODE_DECISION_ABI_LOGITS
    : DECODE_DECISION_ABI_LEGACY;
  if (
    graph.prefill.file !== `prefill.${precision}.onnx` ||
    graph.decode.file !== `decode.${precision}.onnx` ||
    !arrayEquals(prefillInputs, ["input_ids"]) ||
    !arrayEquals(decodeInputs, ["input_ids", ...pastNames]) ||
    !arrayEquals(prefillOutputs, logitsAbi
      ? ["next_token", "logits", ...presentNames]
      : ["next_token", ...presentNames]) ||
    !arrayEquals(decodeOutputs, logitsAbi
      ? ["next_token", "logits", ...presentNames]
      : ["next_token", ...presentNames])
  ) {
    throw new Error(`${role} ${precision} graph names do not match its cache ABI.`);
  }
  validateTypedIo(
    graph.prefill.inputs,
    prefillInputs,
    expectedCacheDtype,
    `${role}.${precision}.prefill`,
    "inputs"
  );
  validateTypedIo(
    graph.prefill.outputs,
    prefillOutputs,
    expectedCacheDtype,
    `${role}.${precision}.prefill`,
    "outputs"
  );
  validateTypedIo(
    graph.decode.inputs,
    decodeInputs,
    expectedCacheDtype,
    `${role}.${precision}.decode`,
    "inputs"
  );
  validateTypedIo(
    graph.decode.outputs,
    decodeOutputs,
    expectedCacheDtype,
    `${role}.${precision}.decode`,
    "outputs"
  );
  const shapeByCacheName = new Map();
  for (const slot of contract.cache_slots) {
    for (const name of [...slot.past_inputs, ...slot.present_outputs]) {
      shapeByCacheName.set(name, slot.shape);
    }
  }
  const shapeChecks = [
    [graph.prefill.inputs, "input_ids", ["batch", "prompt_sequence"]],
    [graph.prefill.outputs, "next_token", ["batch"]],
    [graph.decode.inputs, "input_ids", ["batch", 1]],
    [graph.decode.outputs, "next_token", ["batch"]],
  ];
  if (logitsAbi) {
    shapeChecks.push(
      [graph.prefill.outputs, "logits", ["batch", "vocab_size"]],
      [graph.decode.outputs, "logits", ["batch", "vocab_size"]]
    );
  }
  const outputCacheOffset = logitsAbi ? 2 : 1;
  for (const entries of [
    graph.prefill.outputs.slice(outputCacheOffset),
    graph.decode.inputs.slice(1),
    graph.decode.outputs.slice(outputCacheOffset),
  ]) {
    for (const entry of entries) {
      shapeChecks.push([entries, entry.name, shapeByCacheName.get(entry.name)]);
    }
  }
  for (const [entries, name, expectedShape] of shapeChecks) {
    const entry = entries.find((candidate) => candidate.name === name);
    if (!entry || !expectedShape || !arrayEquals(entry.shape, expectedShape)) {
      throw new Error(`${role} ${precision} tensor ${name} has an invalid declared shape.`);
    }
  }
  for (const slot of contract.cache_slots) {
    if (slot.dtype_by_precision?.[precision] !== expectedCacheDtype) {
      throw new Error(`${role} slot ${slot.slot} has the wrong ${precision} dtype.`);
    }
  }
  if (logitsAbi) {
    if (
      !Number.isInteger(vocabSize) ||
      vocabSize < 1 ||
      contract.logits?.name !== "logits" ||
      !arrayEquals(contract.logits?.shape, ["batch", vocabSize]) ||
      contract.logits?.dtype_by_precision?.[precision] !== expectedCacheDtype
    ) {
      throw new Error(`${role} ${precision} logits metadata does not match the model vocabulary.`);
    }
  } else if (Object.hasOwn(contract, "logits")) {
    throw new Error(`${role} ${precision} legacy ABI must not declare logits metadata.`);
  }
  return { decisionAbi, graph };
}

function decodeDecisionAbi(provenance) {
  return provenance?.graph_contract?.logits
    ? DECODE_DECISION_ABI_LOGITS
    : DECODE_DECISION_ABI_LEGACY;
}

function validateLegacyTrajectoryParity(provenance, role, precisions) {
  let expectedPromptLengths = null;
  for (const precision of precisions) {
    const result = provenance.parity.results[precision];
    const cacheDtype = precision === "fp16" ? "float16" : "float32";
    const cacheAtolCeiling = precision === "fp16" ? 0.1 : 0.001;
    if (
      !result ||
      result.hard_gate !== true ||
      result.passed !== true ||
      result.greedy_next_token_exact !== true ||
      result.cache_dtype !== cacheDtype ||
      !Number.isInteger(result.decode_steps) ||
      result.decode_steps < 3 ||
      !Number.isFinite(result.cache_atol) ||
      result.cache_atol < 0 ||
      result.cache_atol > cacheAtolCeiling ||
      !Number.isFinite(result.max_cache_abs_diff) ||
      result.max_cache_abs_diff < 0 ||
      result.max_cache_abs_diff > result.cache_atol ||
      result.provider !== "CPUExecutionProvider" ||
      result.reference !== "exact in-memory LocalAgentLM random initialization" ||
      result.reference_independence?.onnx_vs_pytorch_cached_path !== true ||
      result.reference_independence
        ?.pytorch_cached_vs_fresh_full_context_greedy_token !== true ||
      !Array.isArray(result.per_fixture) ||
      result.per_fixture.length < 2
    ) {
      throw new Error(`${role} ${precision} legacy trajectory parity failed.`);
    }
    const promptLengths = result.per_fixture
      .map((fixture) => fixture.prompt_length)
      .sort((left, right) => left - right);
    const fixtureDigests = result.per_fixture.map((fixture) => fixture.input_ids_sha256);
    if (
      new Set(promptLengths).size !== promptLengths.length ||
      new Set(fixtureDigests).size !== fixtureDigests.length
    ) {
      throw new Error(`${role} ${precision} legacy parity fixtures are not distinct.`);
    }
    if (expectedPromptLengths == null) {
      expectedPromptLengths = promptLengths;
    } else if (!arrayEquals(promptLengths, expectedPromptLengths)) {
      throw new Error(`${role} legacy parity prompt-length coverage differs by precision.`);
    }
    for (const fixture of result.per_fixture) {
      if (
        !Number.isInteger(fixture.prompt_length) ||
        fixture.prompt_length < 1 ||
        !isSha256(fixture.input_ids_sha256) ||
        fixture.prefill_next_token_exact !== true ||
        fixture.prefill_cached_vs_full_context_next_token_exact !== true ||
        !Number.isFinite(fixture.prefill_cache_max_abs_diff) ||
        fixture.prefill_cache_max_abs_diff < 0 ||
        fixture.prefill_cache_max_abs_diff > result.cache_atol ||
        !Array.isArray(fixture.decode) ||
        fixture.decode.length !== result.decode_steps
      ) {
        throw new Error(`${role} ${precision} legacy parity fixture failed.`);
      }
      fixture.decode.forEach((step, index) => {
        if (
          step.decode_step !== index + 1 ||
          step.next_token_exact !== true ||
          step.cached_vs_full_context_next_token_exact !== true ||
          !Number.isFinite(step.cache_max_abs_diff) ||
          step.cache_max_abs_diff < 0 ||
          step.cache_max_abs_diff > result.cache_atol
        ) {
          throw new Error(
            `${role} ${precision} legacy parity decode step ${index + 1} failed.`
          );
        }
      });
    }
    for (const graphKind of ["prefill", "decode"]) {
      const file = provenance.graph_contract.graphs[precision][graphKind].file;
      const artifact = validateArtifactPin(
        provenance.artifacts?.[file],
        `${role}.artifacts.${file}`
      );
      const parityArtifact = validateArtifactPin(
        result.artifacts?.[graphKind],
        `${role}.parity.${precision}.artifacts.${graphKind}`
      );
      if (
        artifact.bytes !== parityArtifact.bytes ||
        artifact.sha256.toLowerCase() !== parityArtifact.sha256.toLowerCase()
      ) {
        throw new Error(
          `${role} ${precision} legacy ${graphKind} identity is not parity-bound.`
        );
      }
    }
  }
}

function validateTrajectoryParity(provenance, role, artifactMode, decisionAbi) {
  const parity = provenance.parity;
  const precisions = Object.keys(provenance.graph_contract.graphs || {}).sort();
  if (
    !parity ||
    parity.hard_gate !== true ||
    parity.fixture_contract !==
      "ids[i]=(131*i+17+977*fixture_index) mod vocab_size" ||
    !parity.results ||
    !arrayEquals(Object.keys(parity.results).sort(), precisions)
  ) {
    throw new Error(`${role} provenance has no complete hard trajectory-parity gate.`);
  }
  const fixtureLengthsDeclared = Object.hasOwn(parity, "fixture_lengths");
  const fixtureRequirementDeclared = Object.hasOwn(
    parity, "fixture_length_requirement"
  );
  if (fixtureLengthsDeclared !== fixtureRequirementDeclared) {
    throw new Error(`${role} parity fixture-length metadata is only partially declared.`);
  }
  const hasFixtureLengths = fixtureLengthsDeclared && fixtureRequirementDeclared;
  if (hasFixtureLengths) {
    if (
      !Array.isArray(parity.fixture_lengths) ||
      typeof parity.fixture_length_requirement !== "string" ||
      parity.fixture_length_requirement !==
        "at least two positive, distinct prompt lengths; order is preserved" ||
      parity.fixture_lengths.length < 2 ||
      parity.fixture_lengths.some((value) => !Number.isInteger(value) || value < 1) ||
      new Set(parity.fixture_lengths).size !== parity.fixture_lengths.length
    ) {
      throw new Error(`${role} parity fixture-length metadata is invalid.`);
    }
  }
  if (
    Object.hasOwn(parity, "cache_atol_ceiling_by_precision") &&
    !arrayEquals(
      parity.cache_atol_ceiling_by_precision,
      { fp16: 0.1, fp32: 0.001 }
    )
  ) {
    throw new Error(`${role} parity tolerance-ceiling metadata is invalid.`);
  }
  if (decisionAbi === DECODE_DECISION_ABI_LEGACY) {
    if (artifactMode !== "random") {
      throw new Error(`${role} trained provenance cannot use the legacy next-token-only ABI.`);
    }
    validateLegacyTrajectoryParity(provenance, role, precisions);
    return;
  }
  let expectedPromptLengths = null;
  const expectedReference = artifactMode === "trained"
    ? "exact in-memory LocalAgentLM checkpoint weights"
    : "exact in-memory LocalAgentLM random initialization";
  for (const precision of precisions) {
    const result = parity.results[precision];
    const cacheDtype = precision === "fp16" ? "float16" : "float32";
    const requiredCacheAtol = precision === "fp16" ? 0.1 : 0.001;
    if (
      !result ||
      result.hard_gate !== true ||
      result.passed !== true ||
      result.greedy_next_token_exact !== true ||
      result.cache_dtype !== cacheDtype ||
      !Number.isInteger(result.decode_steps) ||
      result.decode_steps < 3 ||
      !Number.isFinite(result.cache_atol) ||
      result.cache_atol < 0 ||
      result.cache_atol > requiredCacheAtol ||
      !Number.isFinite(result.max_cache_abs_diff) ||
      result.max_cache_abs_diff < 0 ||
      result.max_cache_abs_diff > result.cache_atol ||
      result.logits_atol !== result.cache_atol ||
      !arrayEquals(
        result.final_token_logits_shape,
        ["batch", provenance.model.config.vocab_size]
      ) ||
      !Number.isFinite(result.max_logits_abs_diff) ||
      result.max_logits_abs_diff < 0 ||
      result.max_logits_abs_diff > result.logits_atol ||
      !Number.isFinite(result.max_cached_vs_full_context_logits_abs_diff) ||
      result.max_cached_vs_full_context_logits_abs_diff < 0 ||
      result.max_cached_vs_full_context_logits_abs_diff > 0.001 ||
      result.provider !== "CPUExecutionProvider" ||
      result.reference !== expectedReference ||
      result.reference_independence?.onnx_logits_vs_pytorch_cached_path !== true ||
      result.reference_independence?.onnx_vs_pytorch_cached_path !== true ||
      result.reference_independence
        ?.pytorch_cached_vs_fresh_full_context_logits !== true ||
      result.reference_independence
        ?.pytorch_cached_vs_fresh_full_context_greedy_token !== true ||
      !Array.isArray(result.per_fixture) ||
      !result.per_fixture.length
    ) {
      throw new Error(`${role} ${precision} trajectory parity did not pass its hard gate.`);
    }
    const promptLengths = result.per_fixture
      .map((fixture) => fixture.prompt_length)
      .sort((left, right) => left - right);
    const fixtureDigests = result.per_fixture.map((fixture) => fixture.input_ids_sha256);
    if (
      new Set(promptLengths).size !== promptLengths.length ||
      promptLengths.length < 2 ||
      new Set(fixtureDigests).size !== fixtureDigests.length
    ) {
      throw new Error(`${role} ${precision} parity lacks distinct multi-length fixtures.`);
    }
    if (
      hasFixtureLengths &&
      !arrayEquals(
        result.per_fixture.map((fixture) => fixture.prompt_length),
        parity.fixture_lengths
      )
    ) {
      throw new Error(`${role} ${precision} parity fixtures contradict declared lengths.`);
    }
    if (expectedPromptLengths == null) {
      expectedPromptLengths = promptLengths;
    } else if (!arrayEquals(promptLengths, expectedPromptLengths)) {
      throw new Error(`${role} parity prompt-length coverage differs by precision.`);
    }
    for (const fixture of result.per_fixture) {
      if (
        !Number.isInteger(fixture.prompt_length) ||
        fixture.prompt_length < 1 ||
        !isSha256(fixture.input_ids_sha256) ||
        fixture.prefill_next_token_exact !== true ||
        fixture.prefill_cached_vs_full_context_next_token_exact !== true ||
        !Number.isFinite(fixture.prefill_cache_max_abs_diff) ||
        fixture.prefill_cache_max_abs_diff < 0 ||
        fixture.prefill_cache_max_abs_diff > result.cache_atol ||
        !Number.isFinite(fixture.prefill_logits_max_abs_diff) ||
        fixture.prefill_logits_max_abs_diff < 0 ||
        fixture.prefill_logits_max_abs_diff > result.logits_atol ||
        !Number.isFinite(
          fixture.prefill_cached_vs_full_context_logits_max_abs_diff
        ) ||
        fixture.prefill_cached_vs_full_context_logits_max_abs_diff < 0 ||
        fixture.prefill_cached_vs_full_context_logits_max_abs_diff > 0.001 ||
        !Array.isArray(fixture.decode) ||
        fixture.decode.length !== result.decode_steps
      ) {
        throw new Error(`${role} ${precision} parity fixture is incomplete or failed.`);
      }
      fixture.decode.forEach((step, index) => {
        if (
          step.decode_step !== index + 1 ||
          step.next_token_exact !== true ||
          step.cached_vs_full_context_next_token_exact !== true ||
          !Number.isFinite(step.cache_max_abs_diff) ||
          step.cache_max_abs_diff < 0 ||
          step.cache_max_abs_diff > result.cache_atol ||
          !Number.isFinite(step.logits_max_abs_diff) ||
          step.logits_max_abs_diff < 0 ||
          step.logits_max_abs_diff > result.logits_atol ||
          !Number.isFinite(step.cached_vs_full_context_logits_max_abs_diff) ||
          step.cached_vs_full_context_logits_max_abs_diff < 0 ||
          step.cached_vs_full_context_logits_max_abs_diff > 0.001
        ) {
          throw new Error(
            `${role} ${precision} parity fixture decode step ${index + 1} failed.`
          );
        }
      });
    }
    for (const graphKind of ["prefill", "decode"]) {
      const file = provenance.graph_contract.graphs[precision][graphKind].file;
      const artifact = validateArtifactPin(
        provenance.artifacts?.[file],
        `${role}.artifacts.${file}`
      );
      const parityArtifact = validateArtifactPin(
        result.artifacts?.[graphKind],
        `${role}.parity.${precision}.artifacts.${graphKind}`
      );
      if (
        artifact.bytes !== parityArtifact.bytes ||
        artifact.sha256.toLowerCase() !== parityArtifact.sha256.toLowerCase()
      ) {
        throw new Error(
          `${role} ${precision} ${graphKind} graph identity is not bound to parity.`
        );
      }
    }
  }
}

function normalizeDecodeProvenanceContext(value, role) {
  if (value == null || Number.isInteger(value)) {
    return {
      artifactMode: "random",
      sharedRandomSeed: value,
      checkpoint: null,
      tokenizer: null,
    };
  }
  if (typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${role} provenance validation context is invalid.`);
  }
  if (value.single_model === true) {
    if (
      value.artifactMode !== "trained" ||
      !value.checkpoint ||
      !value.tokenizer
    ) {
      throw new Error(`${role} single-model provenance context is incomplete.`);
    }
    return {
      artifactMode: "trained",
      sharedRandomSeed: null,
      checkpoint: validateCheckpointPin(
        value.checkpoint,
        "single_model.checkpoint"
      ),
      tokenizer: validateTokenizerProvenancePin(
        value.tokenizer,
        "single_model.tokenizer"
      ),
      singleModel: true,
    };
  }
  validateDecodeManifest(value);
  const artifactMode = decodeManifestMode(value);
  return {
    artifactMode,
    sharedRandomSeed:
      artifactMode === "random" ? value.shared_random_seed : null,
    checkpoint:
      artifactMode === "trained" ? value.checkpoints?.[role] : null,
    tokenizer:
      artifactMode === "trained" ? value.tokenizer : null,
    singleModel: false,
  };
}

function validateDecodeProvenance(provenance, role, modelEntry, validationContext = null) {
  if (!provenance || typeof provenance !== "object" || Array.isArray(provenance)) {
    throw new Error(`Provenance for ${role} must be a JSON object.`);
  }
  validateNoContradictoryCapabilityFields(provenance, `provenance.${role}`);
  const context = normalizeDecodeProvenanceContext(validationContext, role);
  const qualityClaimsEmpty =
    Array.isArray(provenance.quality_claims) && provenance.quality_claims.length === 0;
  if (context.artifactMode === "random") {
    if (
      provenance.schema_version !== 1 ||
      provenance.artifact_type !== "random_weight_cached_decode_onnx" ||
      provenance.latency_only !== true ||
      provenance.trained !== false ||
      provenance.training_steps !== 0 ||
      provenance.capability_artifact !== false ||
      !qualityClaimsEmpty ||
      Object.hasOwn(provenance, "checkpoint_step") ||
      Object.hasOwn(provenance, "tokens_seen") ||
      Object.hasOwn(provenance, "tokenizer") ||
      Object.hasOwn(provenance, "quality_evaluation")
    ) {
      throw new Error(`${role} provenance is not an untrained latency-only decode artifact.`);
    }
    if (
      provenance.weights?.source !== "deterministic_random_initialization" ||
      provenance.weights?.checkpoint !== null ||
      (
        context.sharedRandomSeed != null &&
        provenance.weights?.seed !== context.sharedRandomSeed
      ) ||
      !isSha256(provenance.weights?.state_dict_sha256)
    ) {
      throw new Error(`${role} provenance is not tied to deterministic random initialization.`);
    }
  } else {
    const exactQualityEvaluation =
      provenance.quality_evaluation?.included === false &&
      provenance.quality_evaluation?.required_separately === true &&
      provenance.quality_evaluation?.scope ===
        "Export validates graph parity only; held-out CE/BPB and downstream " +
        "capability metrics are separate artifacts." &&
      arrayEquals(
        Object.keys(provenance.quality_evaluation || {}).sort(),
        ["included", "required_separately", "scope"]
      );
    if (
      provenance.schema_version !== 1 ||
      provenance.artifact_type !== "trained_checkpoint_cached_decode_onnx" ||
      provenance.latency_only !== false ||
      provenance.trained !== true ||
      !Number.isInteger(provenance.training_steps) ||
      provenance.training_steps < 1 ||
      provenance.capability_artifact !== false ||
      provenance.capability_metrics !== null ||
      provenance.action_capability_claimed === true ||
      provenance.action_capability_artifact === true ||
      (
        Array.isArray(provenance.action_capability_claims) &&
        provenance.action_capability_claims.length > 0
      ) ||
      !qualityClaimsEmpty ||
      !exactQualityEvaluation ||
      !Number.isInteger(provenance.checkpoint_step) ||
      provenance.checkpoint_step < 0 ||
      provenance.training_steps !== provenance.checkpoint_step + 1
    ) {
      throw new Error(
        `${role} provenance is not a trained, non-capability, quality-separate decode artifact.`
      );
    }
    const checkpoint = validateCheckpointPin(
      context.checkpoint,
      `checkpoints.${role}`
    );
    const tokenizer = validateTokenizerProvenancePin(
      context.tokenizer,
      "tokenizer"
    );
    const stageUsesTokenAccounting = checkpoint.stage !== "rl";
    if (
      (
        stageUsesTokenAccounting &&
        (
          !Number.isInteger(provenance.tokens_seen) ||
          provenance.tokens_seen < 1 ||
          !Number.isInteger(provenance.input_tokens_seen) ||
          provenance.input_tokens_seen < provenance.tokens_seen
        )
      ) ||
      (
        !stageUsesTokenAccounting &&
        (
          provenance.tokens_seen !== null ||
          provenance.input_tokens_seen !== null ||
          !provenance.rl_accounting ||
          !Number.isInteger(provenance.rl_accounting.realized_optimizer_updates) ||
          provenance.rl_accounting.realized_optimizer_updates < 0
        )
      )
    ) {
      throw new Error(`${role} provenance has invalid ${checkpoint.stage} token accounting.`);
    }
    if (
    provenance.weights?.source !== "strict_lineage_validated_lm_checkpoint" ||
      provenance.weights?.checkpoint !== checkpoint.checkpoint ||
      provenance.weights?.checkpoint_bytes !== checkpoint.bytes ||
      provenance.weights?.checkpoint_sha256?.toLowerCase() !==
        checkpoint.sha256.toLowerCase() ||
      provenance.weights?.checkpoint_stage !== checkpoint.stage ||
      provenance.weights?.checkpoint_step !== checkpoint.step ||
      provenance.weights?.tokens_seen !== checkpoint.tokens_seen ||
      provenance.weights?.input_tokens_seen !== provenance.input_tokens_seen ||
      provenance.checkpoint_step !== checkpoint.step ||
      provenance.tokens_seen !== checkpoint.tokens_seen ||
      provenance.training_steps !== checkpoint.training_steps ||
      !isSha256(provenance.weights?.state_dict_sha256)
    ) {
      throw new Error(`${role} trained weights are not bound to the pair checkpoint pin.`);
    }
    validateTokenizerProvenancePin(
      provenance.tokenizer,
      `${role}.tokenizer`
    );
    if (!tokenizerPinsMatch(provenance.tokenizer, tokenizer)) {
      throw new Error(`${role} tokenizer provenance disagrees with the trained identity pin.`);
    }
  }
  if (
    provenance.model?.name !== modelEntry.name ||
    provenance.model?.pair_role !== role ||
    !provenance.model?.config ||
    !isSha256(provenance.model?.config_canonical_sha256) ||
    !isSha256(provenance.model?.config_source_sha256) ||
    !Number.isInteger(provenance.model?.full_model_parameters)
  ) {
    throw new Error(`${role} provenance has an invalid model/config identity.`);
  }
  if (
    context.artifactMode === "trained" &&
    provenance.tokenizer.vocab_size !== provenance.model.config.vocab_size
  ) {
    throw new Error(`${role} tokenizer vocabulary disagrees with its model config.`);
  }
  const contract = provenance.graph_contract;
  const decisionAbi = contract?.logits
    ? DECODE_DECISION_ABI_LOGITS
    : DECODE_DECISION_ABI_LEGACY;
  const decisionContractValid = decisionAbi === DECODE_DECISION_ABI_LOGITS
    ? (
      contract?.next_token?.decode ===
        "compatibility argmax over the exported final-token logits" &&
      contract?.logits?.name === "logits" &&
      contract?.logits?.description ===
        "unnormalized LM scores for the final input token only" &&
      arrayEquals(
        contract?.logits?.shape,
        ["batch", provenance.model.config.vocab_size]
      )
    )
    : (
      context.artifactMode === "random" &&
      contract?.next_token?.decode === "greedy argmax over final-token LM logits" &&
      !Object.hasOwn(contract || {}, "logits")
    );
  if (
    !contract ||
    contract.next_token?.name !== "next_token" ||
    contract.next_token?.dtype !== "int64" ||
    !arrayEquals(contract.next_token?.shape, ["batch"]) ||
    !decisionContractValid ||
    contract.prefill_projection !==
      "only the final normalized prompt feature is projected to vocabulary logits" ||
    contract.cache_update_strategy !==
      "attention K/V append one token; short-conv state replaces its fixed-width tail" ||
    contract.decode_token_axis_fixed_one !== true ||
    contract.decode_position?.caller_position_input !== false ||
    typeof contract.decode_position?.derived_from !== "string" ||
    contract.decode_position?.rule !==
      "RoPE position = first attention past-key axis-2 length"
  ) {
    throw new Error(`${role} provenance has an invalid pre-tokenized greedy decode contract.`);
  }
  if (!Array.isArray(contract.cache_slots) || !contract.cache_slots.length) {
    throw new Error(`${role} graph contract must expose at least one cache slot.`);
  }
  const precisionKeys = Object.keys(contract.graphs || {}).sort();
  if (
    !arrayEquals(precisionKeys, ["fp32"]) &&
    !arrayEquals(precisionKeys, ["fp16", "fp32"])
  ) {
    throw new Error(`${role} graph contract has unsupported precision keys.`);
  }
  contract.cache_slots.forEach(validateCacheSlot);
  validateCacheSlotsAgainstConfig(contract, provenance.model.config, role);
  const firstAttentionSlot = contract.cache_slots.find((slot) => slot.kind === "attn");
  if (
    !firstAttentionSlot ||
    contract.decode_position.derived_from !== firstAttentionSlot.past_inputs[0]
  ) {
    throw new Error(`${role} decode position is not derived from its first attention cache.`);
  }
  const pastNames = contract.cache_slots.flatMap((slot) => slot.past_inputs);
  const presentNames = contract.cache_slots.flatMap((slot) => slot.present_outputs);
  if (
    new Set(pastNames).size !== pastNames.length ||
    new Set(presentNames).size !== presentNames.length
  ) {
    throw new Error(`${role} cache tensor names must be unique.`);
  }
  const fp32Contract = validatePrecisionGraphContract(
    contract,
    "fp32",
    role,
    provenance.model.config.vocab_size
  );
  if (fp32Contract.decisionAbi !== decisionAbi) {
    throw new Error(`${role} fp32 graph decision ABI disagrees with provenance metadata.`);
  }
  for (const name of [
    "prefill.fp32.onnx", "decode.fp32.onnx", "model-config.yaml",
  ]) {
    const artifact = validateArtifactPin(
      provenance.artifacts?.[name], `${role}.artifacts.${name}`
    );
    const expectedPrecision = name === "model-config.yaml" ? "text" : "fp32";
    if (artifact.file !== name || artifact.precision !== expectedPrecision) {
      throw new Error(`${role} artifact ${name} has an invalid file/precision label.`);
    }
  }
  const fp16Presence = [
    Boolean(provenance.artifacts?.["prefill.fp16.onnx"]),
    Boolean(provenance.artifacts?.["decode.fp16.onnx"]),
  ];
  if (fp16Presence[0] !== fp16Presence[1]) {
    throw new Error(`${role} provenance must include both fp16 graphs or neither.`);
  }
  if (fp16Presence[0]) {
    const prefillFp16 = validateArtifactPin(
      provenance.artifacts["prefill.fp16.onnx"],
      `${role}.artifacts.prefill.fp16.onnx`
    );
    const decodeFp16 = validateArtifactPin(
      provenance.artifacts["decode.fp16.onnx"],
      `${role}.artifacts.decode.fp16.onnx`
    );
    if (
      prefillFp16.file !== "prefill.fp16.onnx" ||
      prefillFp16.precision !== "fp16" ||
      decodeFp16.file !== "decode.fp16.onnx" ||
      decodeFp16.precision !== "fp16"
    ) {
      throw new Error(`${role} fp16 artifact labels are inconsistent.`);
    }
    const fp16Contract = validatePrecisionGraphContract(
      contract,
      "fp16",
      role,
      provenance.model.config.vocab_size
    );
    if (fp16Contract.decisionAbi !== decisionAbi) {
      throw new Error(`${role} fp16/fp32 graph decision ABIs differ.`);
    }
  } else if (contract.graphs?.fp16) {
    throw new Error(`${role} declares fp16 graph I/O without fp16 artifacts.`);
  }
  validateTrajectoryParity(provenance, role, context.artifactMode, decisionAbi);
  return provenance;
}

function validateSingleDecodeProvenance(provenance, modelEntry = null) {
  const role = provenance?.model?.pair_role;
  const name = provenance?.model?.name;
  if (typeof role !== "string" || !role || typeof name !== "string" || !name) {
    throw new Error("Single-model provenance must name its model and export role.");
  }
  if (
    modelEntry &&
    (modelEntry.name !== name || modelEntry.pair_role !== role)
  ) {
    throw new Error("Single-model manifest identity disagrees with export provenance.");
  }
  const accepted = validateDecodeProvenance(
    provenance,
    role,
    { name },
    singleDecodeProvenanceContext(provenance)
  );
  const metadata = validateArtifactPin(
    accepted.artifacts?.["meta.json"],
    "single_model.artifacts.meta.json"
  );
  if (metadata.file !== "meta.json" || metadata.precision !== "metadata") {
    throw new Error("Single-model runtime metadata artifact has an invalid label.");
  }
  const tokenizer = accepted.tokenizer;
  if (tokenizer.kind === "bpe") {
    const bundled = validateArtifactPin(
      accepted.artifacts?.["tokenizer.json"],
      "single_model.artifacts.tokenizer.json"
    );
    if (
      tokenizer.file !== "tokenizer.json" ||
      bundled.file !== "tokenizer.json" ||
      bundled.precision !== "tokenizer" ||
      bundled.bytes !== tokenizer.artifact_identity.bytes ||
      bundled.sha256.toLowerCase() !== tokenizer.sha256.toLowerCase()
    ) {
      throw new Error("Single-model bundled tokenizer is not bound to tokenizer provenance.");
    }
    if (
      tokenizer.bundled_artifact_identity &&
      (
        bundled.bytes !== tokenizer.bundled_artifact_identity.bytes ||
        bundled.sha256.toLowerCase() !==
          tokenizer.bundled_artifact_identity.sha256.toLowerCase()
      )
    ) {
      throw new Error("Single-model bundled tokenizer identity is inconsistent.");
    }
  } else if (
    Object.hasOwn(accepted.artifacts || {}, "tokenizer.json") ||
    tokenizer.file != null
  ) {
    throw new Error("Byte-tokenizer single-model export must not bundle tokenizer.json.");
  }
  if (accepted.training_lineage_export != null) {
    if (accepted.training_lineage_export !== "training-lineage.json") {
      throw new Error("Single-model training-lineage export has an invalid filename.");
    }
    const lineage = validateArtifactPin(
      accepted.artifacts?.["training-lineage.json"],
      "single_model.artifacts.training-lineage.json"
    );
    if (
      lineage.file !== "training-lineage.json" ||
      lineage.precision !== "metadata"
    ) {
      throw new Error("Single-model training-lineage artifact has an invalid label.");
    }
  }
  return accepted;
}

function validateAcceptanceTrainingLineage(provenance, lineage, acceptanceMode = false) {
  if (!acceptanceMode || provenance.weights?.checkpoint_stage === "pretrain") {
    return lineage;
  }
  if (!["midtrain", "sft", "rl"].includes(provenance.weights?.checkpoint_stage)) {
    throw new Error("Acceptance mode requires a recognized checkpoint training stage.");
  }
  const trainingArtifacts = lineage?.training_artifacts;
  const artifactIdentitiesValid =
    Array.isArray(trainingArtifacts) &&
    trainingArtifacts.length > 0 &&
    trainingArtifacts.length === lineage?.training_artifact_sha256?.length &&
    new Set(trainingArtifacts.map((identity) => identity.path)).size ===
      trainingArtifacts.length &&
    trainingArtifacts.every((identity, index) =>
      identity &&
      typeof identity === "object" &&
      !Array.isArray(identity) &&
      arrayEquals(
        Object.keys(identity).sort(),
        ["artifact_kind", "bytes", "path", "sha256"]
      ) &&
      typeof identity.artifact_kind === "string" &&
      identity.artifact_kind.length > 0 &&
      typeof identity.path === "string" &&
      identity.path.startsWith("/") &&
      Number.isInteger(identity.bytes) &&
      identity.bytes > 0 &&
      isSha256(identity.sha256) &&
      identity.sha256 === lineage.training_artifact_sha256[index]
    );
  if (
    provenance.training_lineage_export !== "training-lineage.json" ||
    !lineage ||
    lineage.kind !== "localagent_training_lineage_export" ||
    lineage.schema_version !== 1 ||
    lineage.stage !== provenance.weights.checkpoint_stage ||
    lineage.checkpoint_sha256 !== provenance.weights.checkpoint_sha256 ||
    !lineage.lineage ||
    lineage.lineage.stage !== provenance.weights.checkpoint_stage ||
    canonicalJson(lineage.lineage) !== canonicalJson(provenance.checkpoint_lineage) ||
    lineage.lineage.tokenizer_sha256 !== provenance.tokenizer?.sha256 ||
    lineage.conversation_prompt_contract !== "openai_full_catalog_v1" ||
    !Array.isArray(lineage.training_artifact_sha256) ||
    lineage.training_artifact_sha256.length < 1 ||
    new Set(lineage.training_artifact_sha256).size !==
      lineage.training_artifact_sha256.length ||
    lineage.training_artifact_sha256.some((digest) => !isSha256(digest)) ||
    !artifactIdentitiesValid
  ) {
    throw new Error(
      "Accepted posttraining provenance requires a non-empty, unique training-artifact " +
      "file-identity/SHA-256 lineage bound to the checkpoint and " +
      "openai_full_catalog_v1 prompt contract."
    );
  }
  return lineage;
}

function validateCachedRuntimeMetadata(metadata, provenance, precision) {
  const expectedTokenizerFile =
    provenance.tokenizer?.kind === "bpe" ? "tokenizer.json" : null;
  if (
    !metadata ||
    metadata.schema_version !== 1 ||
    metadata.artifact_type !== "localagent_cached_autoregressive_onnx" ||
    metadata.default_precision !== precision ||
    canonicalJson(metadata.graph_contract) !== canonicalJson(provenance.graph_contract) ||
    canonicalJson(metadata.model?.config) !== canonicalJson(provenance.model.config) ||
    metadata.model?.config_canonical_sha256 !==
      provenance.model.config_canonical_sha256 ||
    metadata.model?.config_file !== "model-config.yaml" ||
    metadata.model?.parameters !== provenance.model.full_model_parameters ||
    metadata.checkpoint?.sha256 !== provenance.weights.checkpoint_sha256 ||
    metadata.checkpoint?.stage !== provenance.weights.checkpoint_stage ||
    metadata.checkpoint?.step !== provenance.weights.checkpoint_step ||
    metadata.tokenizer?.kind !== provenance.tokenizer.kind ||
    metadata.tokenizer?.sha256 !== provenance.tokenizer.sha256 ||
    metadata.tokenizer?.vocab_size !== provenance.tokenizer.vocab_size ||
    metadata.tokenizer?.verified !== true ||
    (metadata.tokenizer?.file ?? null) !== expectedTokenizerFile
  ) {
    throw new Error("Cached runtime metadata disagrees with trained export provenance.");
  }
  return metadata;
}

function outputLocationsForContract(
  provider,
  presentNames,
  decisionAbi = DECODE_DECISION_ABI_LOGITS
) {
  requireExplicitProvider(provider);
  const cacheLocation = provider === "webgpu" ? "gpu-buffer" : "cpu";
  const locations = [["next_token", "cpu"]];
  if (decisionAbi === DECODE_DECISION_ABI_LOGITS) locations.push(["logits", "cpu"]);
  locations.push(...presentNames.map((name) => [name, cacheLocation]));
  return Object.fromEntries(locations);
}

function decodeSessionOptions(
  provider,
  presentNames,
  decisionAbi = DECODE_DECISION_ABI_LOGITS
) {
  return {
    executionProviders: [requireExplicitProvider(provider)],
    preferredOutputLocation: outputLocationsForContract(
      provider,
      presentNames,
      decisionAbi
    ),
  };
}

async function sha256ArrayBuffer(buffer) {
  if (!globalThis.crypto?.subtle) {
    throw new Error("Web Crypto SHA-256 is unavailable; verification cannot continue.");
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

function newBenchmarkIdentity(label) {
  if (typeof globalThis.crypto?.randomUUID !== "function") {
    throw new Error(`Web Crypto randomUUID is unavailable; ${label} identity cannot be proven.`);
  }
  return globalThis.crypto.randomUUID();
}

async function observeHarnessResource(relativePath, url, externalExpectedSha256 = null) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Could not read harness resource ${relativePath}: HTTP ${response.status}.`);
  }
  const payload = await response.arrayBuffer();
  const sha256 = await sha256ArrayBuffer(payload);
  const hashVerified =
    externalExpectedSha256 == null ? false : sha256 === externalExpectedSha256;
  if (externalExpectedSha256 != null && !hashVerified) {
    throw new Error(
      `${relativePath} SHA-256 ${sha256} differs from external root ` +
      `${externalExpectedSha256}.`
    );
  }
  return {
    relative_path: relativePath,
    url,
    bytes: payload.byteLength,
    sha256,
    external_expected_sha256: externalExpectedSha256,
    hash_verified: hashVerified,
  };
}

async function captureDecodeHarnessIdentity(externalRoots = null) {
  if (typeof document === "undefined" || typeof window === "undefined") {
    throw new Error("Harness identity can be captured only in the benchmark page.");
  }
  const htmlUrl = new URL(DECODE_HARNESS_HTML_FILE, document.baseURI).href;
  const javascriptMatches = Array.from(document.scripts).filter((script) => {
    if (!script.src) return false;
    return new URL(script.src, document.baseURI).pathname.endsWith(
      `/${DECODE_HARNESS_JAVASCRIPT_FILE}`
    );
  });
  if (javascriptMatches.length !== 1) {
    throw new Error("Expected exactly one decode-benchmark.js harness script element.");
  }
  const javascriptUrl = new URL(javascriptMatches[0].src, document.baseURI).href;
  const ortMatches = Array.from(document.scripts).filter((script) => {
    if (!script.src) return false;
    return new URL(script.src, document.baseURI).pathname.endsWith(
      `/${DECODE_ORT_SCRIPT_PATH}`
    );
  });
  if (ortMatches.length !== 1) {
    throw new Error("The page does not contain exactly one pinned ONNX Runtime Web script URL.");
  }
  const ortJavascriptUrl = new URL(ortMatches[0].src, document.baseURI).href;
  const ortWasmUrl = new URL(DECODE_ORT_WASM_PATH, document.baseURI).href;
  const [html, javascript, ortJavascript, ortWasm] = await Promise.all([
    observeHarnessResource(
      DECODE_HARNESS_HTML_FILE,
      htmlUrl,
      externalRoots?.html_sha256 || null
    ),
    observeHarnessResource(
      DECODE_HARNESS_JAVASCRIPT_FILE,
      javascriptUrl,
      externalRoots?.javascript_sha256 || null
    ),
    observeHarnessResource(
      "ort.webgpu.min.js",
      ortJavascriptUrl,
      externalRoots?.ort_javascript_sha256 || null
    ),
    observeHarnessResource(
      DECODE_ORT_WASM_FILE,
      ortWasmUrl,
      externalRoots?.ort_wasm_sha256 || null
    ),
  ]);
  const version = verifyOrtVersionPin();
  const resources = [html, javascript, ortJavascript, ortWasm];
  const sameOrigin = resources.every(
    (resource) => new URL(resource.url).origin === window.location.origin
  );
  if (externalRoots && !sameOrigin) {
    throw new Error(
      "Acceptance requires self-hosted same-origin HTML, harness JavaScript, ORT JavaScript, " +
      "and ORT WASM acquisition bytes."
    );
  }
  return {
    schema_version: DECODE_HARNESS_SCHEMA_VERSION,
    html,
    javascript,
    ort: {
      javascript: ortJavascript,
      wasm: ortWasm,
      self_hosted_same_origin: sameOrigin,
      version_pin: DECODE_ORT_VERSION,
      version_reported: version.ort_version_reported,
      version_verified: version.ort_version_verified,
    },
  };
}

function normalizeDecodeBenchmarkMode(mode) {
  const normalized = mode || "matched";
  if (!DECODE_BENCHMARK_MODES.includes(normalized)) {
    throw new Error(`Unknown decode benchmark mode '${normalized}'; expected matched or single.`);
  }
  return normalized;
}

function normalizeDecodeAcceptanceMode(value) {
  if (value === true || value === "1" || value === "true") return true;
  if (
    value === false ||
    value == null ||
    value === "" ||
    value === "0" ||
    value === "false"
  ) {
    return false;
  }
  throw new Error("Acceptance mode must be exactly 1/true or 0/false.");
}

function validateDecodeProtocolSettings(
  { outputTokens, warmups, repetitions, seed = DECODE_DEFAULT_SEED },
  acceptanceMode = false
) {
  if (acceptanceMode) {
    if (
      outputTokens !== DECODE_ACCEPTANCE_PROTOCOL.output_tokens_per_condition ||
      warmups !== DECODE_ACCEPTANCE_PROTOCOL.warmups_per_condition ||
      repetitions !== DECODE_ACCEPTANCE_PROTOCOL.measured_repetitions_per_condition ||
      seed !== DECODE_ACCEPTANCE_PROTOCOL.case_order_seed
    ) {
      throw new Error(
        "Acceptance mode requires exactly 32 output tokens, 3 warmups, and 30 measured " +
        `repetitions per context with seed ${DECODE_DEFAULT_SEED}.`
      );
    }
  } else {
    if (!Number.isInteger(outputTokens) || outputTokens < 2 || outputTokens > 256) {
      throw new Error("Output tokens must be an integer from 2 through 256.");
    }
    if (!Number.isInteger(warmups) || warmups < DECODE_MIN_WARMUPS) {
      throw new Error(`Warmups must be at least ${DECODE_MIN_WARMUPS} per condition.`);
    }
    if (!Number.isInteger(repetitions) || repetitions < DECODE_MIN_REPETITIONS) {
      throw new Error(
        `Measured repetitions must be at least ${DECODE_MIN_REPETITIONS} per condition.`
      );
    }
  }
  return { outputTokens, warmups, repetitions };
}

function configureAcceptanceProtocolSettings(acceptanceMode) {
  if (!acceptanceMode) return;
  const settings = [
    ["decode-output-tokens", DECODE_ACCEPTANCE_PROTOCOL.output_tokens_per_condition],
    ["decode-warmups", DECODE_ACCEPTANCE_PROTOCOL.warmups_per_condition],
    [
      "decode-repetitions",
      DECODE_ACCEPTANCE_PROTOCOL.measured_repetitions_per_condition,
    ],
    ["decode-seed", DECODE_ACCEPTANCE_PROTOCOL.case_order_seed],
  ];
  for (const [id, value] of settings) {
    const input = decodeElement(id);
    if (!input) continue;
    input.value = String(value);
    input.readOnly = true;
    input.setAttribute("aria-readonly", "true");
  }
}

function requestedDecodeAcceptanceMode() {
  if (typeof window === "undefined") return false;
  return normalizeDecodeAcceptanceMode(
    window.__localAgentDecodeAcceptanceMode ??
    new URLSearchParams(window.location.search).get("acceptance")
  );
}

function requestedDecodeAcceptanceRootSha256() {
  if (typeof window === "undefined") return null;
  const value =
    window.__localAgentDecodeAcceptanceRootSha256 ??
    new URLSearchParams(window.location.search).get("acceptance_root_sha256");
  if (value == null || value === "") return null;
  assertSha256(value, "acceptance wrapper-manifest root SHA-256");
  return value.toLowerCase();
}

function requestedAcceptanceSha256(parameter, globalName, label) {
  if (typeof window === "undefined") return null;
  const value =
    window[globalName] ??
    new URLSearchParams(window.location.search).get(parameter);
  if (value == null || value === "") return null;
  assertSha256(value, label);
  return value.toLowerCase();
}

function requestedDecodeAcceptanceEvidence() {
  if (!requestedDecodeAcceptanceMode()) return null;
  const evidence = {
    run_challenge: requestedAcceptanceSha256(
      "run_challenge",
      "__localAgentDecodeRunChallenge",
      "external run challenge"
    ),
    machine_condition_sha256: requestedAcceptanceSha256(
      "machine_condition_sha256",
      "__localAgentDecodeMachineConditionSha256",
      "external machine/GPU condition SHA-256"
    ),
    html_sha256: requestedAcceptanceSha256(
      "harness_html_sha256",
      "__localAgentDecodeHarnessHtmlSha256",
      "external harness HTML SHA-256"
    ),
    javascript_sha256: requestedAcceptanceSha256(
      "harness_js_sha256",
      "__localAgentDecodeHarnessJavascriptSha256",
      "external harness JavaScript SHA-256"
    ),
    ort_javascript_sha256: requestedAcceptanceSha256(
      "ort_js_sha256",
      "__localAgentDecodeOrtJavascriptSha256",
      "external ORT JavaScript SHA-256"
    ),
    ort_wasm_sha256: requestedAcceptanceSha256(
      "ort_wasm_sha256",
      "__localAgentDecodeOrtWasmSha256",
      "external ORT WASM SHA-256"
    ),
  };
  const missing = Object.entries(evidence)
    .filter(([, value]) => value == null)
    .map(([field]) => field);
  if (missing.length) {
    throw new Error(
      `Acceptance requires external run, machine, and acquisition roots: ${missing.join(", ")}.`
    );
  }
  return evidence;
}

function parsePositiveInteger(value, field) {
  if (typeof value === "number" && Number.isInteger(value) && value > 0) return value;
  if (typeof value === "string" && /^[1-9][0-9]*$/.test(value)) {
    const parsed = Number(value);
    if (Number.isSafeInteger(parsed)) return parsed;
  }
  throw new Error(`${field} must be a positive integer.`);
}

function validateSingleDecodeQuery(query) {
  if (!query || typeof query !== "object" || Array.isArray(query)) {
    throw new Error("Single-model direct provenance query is invalid.");
  }
  if (typeof query.provenance !== "string" || !query.provenance) {
    throw new Error("Single-model direct mode requires a provenance URL.");
  }
  assertSha256(query.sha256, "single-model provenance SHA-256");
  return {
    provenance: query.provenance,
    sha256: query.sha256.toLowerCase(),
    bytes: parsePositiveInteger(query.bytes, "single-model provenance bytes"),
  };
}

function requestedDecodeBenchmarkMode() {
  if (typeof window === "undefined") return "matched";
  return normalizeDecodeBenchmarkMode(
    window.__localAgentDecodeBenchmarkMode ||
    new URLSearchParams(window.location.search).get("mode") ||
    "matched"
  );
}

function requestedDirectProvenanceQuery() {
  if (typeof window === "undefined") return null;
  const parameters = new URLSearchParams(window.location.search);
  const provenanceSha256 =
    window.__localAgentDecodeProvenanceSha256 ||
    parameters.get("provenance_sha256") ||
    parameters.get("manifest_sha256");
  const provenance =
    window.__localAgentDecodeProvenanceUrl ||
    parameters.get("provenance") ||
    (provenanceSha256 ? requestedDecodeManifestUrl() : null);
  if (!provenance) return null;
  return validateSingleDecodeQuery({
    provenance,
    sha256: provenanceSha256,
    bytes:
      window.__localAgentDecodeProvenanceBytes ||
      parameters.get("provenance_bytes") ||
      parameters.get("manifest_bytes"),
  });
}

function requestedDecodeManifestUrl() {
  if (typeof window === "undefined") return DECODE_DEFAULT_MANIFEST_URL;
  return window.__localAgentDecodeManifestUrl ||
    new URLSearchParams(window.location.search).get("manifest") ||
    DECODE_DEFAULT_MANIFEST_URL;
}

function resolveArtifactUrl(path, baseUrl = document.baseURI) {
  return new URL(path, baseUrl).href;
}

async function fetchVerifiedArtifact(
  path,
  expectedPin,
  artifactKind,
  artifactId,
  baseUrl = document.baseURI
) {
  validateArtifactPin(expectedPin, `${artifactKind}.${artifactId}`);
  const url = resolveArtifactUrl(path, baseUrl);
  const started = performance.now();
  const response = await fetch(url);
  const responseAt = performance.now();
  if (!response.ok) throw new Error(`Failed to fetch ${path}: HTTP ${response.status}.`);
  const buffer = await response.arrayBuffer();
  const readFinished = performance.now();
  const hashStarted = performance.now();
  const actualSha256 = await sha256ArrayBuffer(buffer);
  const finished = performance.now();
  const record = {
    phase: "artifact_verification",
    artifact_kind: artifactKind,
    artifact_id: artifactId,
    relative_path: path,
    url,
    fetch_ms: responseAt - started,
    read_ms: readFinished - responseAt,
    hash_ms: finished - hashStarted,
    total_fetch_read_hash_ms: finished - started,
    bytes: buffer.byteLength,
    expected_bytes: expectedPin.bytes,
    expected_sha256: expectedPin.sha256.toLowerCase(),
    actual_sha256: actualSha256,
    bytes_verified: buffer.byteLength === expectedPin.bytes,
    hash_verified: actualSha256 === expectedPin.sha256.toLowerCase(),
    verification_before_parse_or_ort: true,
    content_type: response.headers.get("content-type") || null,
    etag: response.headers.get("etag") || null,
    last_modified: response.headers.get("last-modified") || null,
    cache_control: response.headers.get("cache-control") || null,
    browser_cache_state: "unknown",
    browser_cache_state_reason:
      "Fetch does not expose a reliable network-versus-browser-cache classification",
    ...currentDecodeLabels(),
  };
  DECODE_STATE.artifacts.push(record);
  if (!record.bytes_verified || !record.hash_verified) {
    throw new Error(
      `${artifactKind} ${artifactId} verification failed: expected ` +
      `${expectedPin.bytes} bytes / ${expectedPin.sha256}, got ` +
      `${buffer.byteLength} bytes / ${actualSha256}.`
    );
  }
  return { buffer, record };
}

function decodeJsonBuffer(buffer, label) {
  try {
    return JSON.parse(decodeUtf8Buffer(buffer, label));
  } catch (error) {
    throw new Error(`${label} is not valid UTF-8 JSON: ${error.message}`);
  }
}

function decodeUtf8Buffer(buffer, label) {
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(buffer);
  } catch (error) {
    throw new Error(`${label} is not valid UTF-8: ${error.message}`);
  }
}

function validateConfig(config, armId) {
  if (!config || typeof config !== "object" || Array.isArray(config)) {
    throw new Error(`Config for ${armId} must be a JSON object.`);
  }
  if (config.name !== armId) {
    throw new Error(`Config name ${config.name} does not match manifest model ${armId}.`);
  }
  for (const field of [
    "vocab_size", "d_model", "n_layers", "n_loops", "n_heads", "n_kv_heads",
    "max_seq_len", "conv_kernel",
  ]) {
    if (!Number.isInteger(config[field]) || config[field] < 1) {
      throw new Error(`${armId} config field ${field} must be a positive integer.`);
    }
  }
  if (config.max_seq_len < Math.max(...DECODE_CONTEXT_LENGTHS) + 256) {
    throw new Error(
      `${armId} max_seq_len cannot cover the longest context plus configured decode ceiling.`
    );
  }
}

function validateMatchedConfigs(assets, manifest) {
  const [left, right] = assets;
  const allowedDifferences = new Set(["name", "ffn_hidden", "layer_types"]);
  const allFields = new Set([...Object.keys(left.config), ...Object.keys(right.config)]);
  for (const field of allFields) {
    if (allowedDifferences.has(field)) continue;
    if (canonicalJson(left.config[field]) !== canonicalJson(right.config[field])) {
      throw new Error(`Matched decode configs differ in controlled field '${field}'.`);
    }
  }
  const hybrid = assets.find((asset) => asset.role === "hybrid_treatment");
  const attention = assets.find((asset) => asset.role === "all_attention_control");
  if (!hybrid || !attention) throw new Error("Matched treatment/control roles are missing.");
  if (
    !hybrid.config.layer_types?.includes("conv") ||
    !hybrid.config.layer_types?.includes("attn")
  ) {
    throw new Error("Hybrid treatment must contain convolution and attention layers.");
  }
  if (
    !Array.isArray(attention.config.layer_types) ||
    attention.config.layer_types.some((kind) => kind !== "attn")
  ) {
    throw new Error("All-attention control must contain attention layers only.");
  }
  for (const field of allowedDifferences) {
    const declared = manifest.intentional_differences[field];
    if (
      canonicalJson(declared?.hybrid_treatment) !== canonicalJson(hybrid.config[field]) ||
      canonicalJson(declared?.all_attention_control) !== canonicalJson(attention.config[field])
    ) {
      throw new Error(`Pair manifest intentional difference '${field}' is inconsistent.`);
    }
  }
  const hybridParameters = hybrid.provenance.model.full_model_parameters;
  const attentionParameters = attention.provenance.model.full_model_parameters;
  const relativeDelta = Math.abs(attentionParameters - hybridParameters) / hybridParameters;
  if (
    manifest.match.hybrid_parameters !== hybridParameters ||
    manifest.match.attention_parameters !== attentionParameters ||
    Math.abs(manifest.match.relative_parameter_delta - relativeDelta) > 1e-15 ||
    relativeDelta >= 0.01
  ) {
    throw new Error("Pair parameter accounting disagrees with model provenance.");
  }
}

function selectGraphArtifacts(provenance) {
  const hasFp16 = Boolean(provenance.artifacts["prefill.fp16.onnx"]);
  const precision = hasFp16 ? "fp16" : "fp32";
  const precisionContract = provenance.graph_contract.graphs?.[precision];
  if (!precisionContract) {
    throw new Error(`Provenance has no ${precision} graph contract.`);
  }
  const prefillName = precisionContract.prefill.file;
  const decodeName = precisionContract.decode.file;
  const prefill = provenance.artifacts[prefillName];
  const decode = provenance.artifacts[decodeName];
  if (!prefill || !decode) {
    throw new Error(`Provenance has no complete ${precision} prefill/decode graph pair.`);
  }
  return {
    precision,
    prefillName,
    decodeName,
    prefill,
    decode,
    contract: precisionContract,
  };
}

function crossCheckPairArtifact(manifest, relativePath, provenancePin, role) {
  const pairPin = validateArtifactPin(
    manifest.artifacts?.[relativePath],
    `pair.artifacts.${relativePath}`
  );
  if (
    pairPin.bytes !== provenancePin.bytes ||
    pairPin.sha256.toLowerCase() !== provenancePin.sha256.toLowerCase()
  ) {
    throw new Error(`${role} artifact ${relativePath} disagrees with pair provenance.`);
  }
  return pairPin;
}

function validateSessionContract(session, contract, graphKind, armId) {
  const expectedInputs = graphIoNames(
    contract[graphKind], "input", `${armId}.${graphKind}`
  );
  const expectedOutputs = graphIoNames(
    contract[graphKind], "output", `${armId}.${graphKind}`
  );
  const actualInputs = [...(session.inputNames || [])];
  const actualOutputs = [...(session.outputNames || [])];
  if (!arrayEquals(actualInputs, expectedInputs)) {
    throw new Error(
      `${armId} ${graphKind} inputs ${JSON.stringify(actualInputs)} do not match ` +
      `${JSON.stringify(expectedInputs)}.`
    );
  }
  if (!arrayEquals(actualOutputs, expectedOutputs)) {
    throw new Error(
      `${armId} ${graphKind} outputs ${JSON.stringify(actualOutputs)} do not match ` +
      `${JSON.stringify(expectedOutputs)}.`
    );
  }
  return { input_names: actualInputs, output_names: actualOutputs };
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
    ort_version_verified: reported == null ? null : reported === DECODE_ORT_VERSION,
    ort_version_verification_status:
      reported == null
        ? "unknown_runtime_did_not_report_version"
        : reported === DECODE_ORT_VERSION
          ? "matches_script_pin"
          : "mismatch",
  };
}

function verifyOrtVersionPin() {
  const evidence = ortVersionEvidence();
  if (evidence.ort_version_verified !== true) {
    throw new Error(
      `Could not verify ONNX Runtime Web version ${evidence.ort_version_reported}; ` +
      `expected pinned ${DECODE_ORT_VERSION}.`
    );
  }
  return evidence;
}

function providerEvidence(provider, sessionCreated = false) {
  requireExplicitProvider(provider);
  const webgpu = ortWebGpuEvidence();
  return {
    provider_requested: provider,
    provider_actual: null,
    provider_actual_observation: sessionCreated
      ? "not exposed by ONNX Runtime Web; exact provider request and session creation observed"
      : "no successfully created session observed",
    provider_actual_scope:
      "provider request plus session creation only; graph-wide and per-node placement are unknown",
    exact_provider_request_and_session_creation_observed:
      sessionCreated && provider === "webgpu"
        ? (webgpu.ort_adapter_available && webgpu.ort_device_available ? true : null)
        : sessionCreated,
    execution_provider_list: [provider],
    whole_session_provider_retry: false,
    per_node_placement_verified: false,
    graph_wide_provider_verified: false,
    per_node_placement_status: "unknown",
    per_node_fallback_status: "unknown",
    ort_webgpu: webgpu,
  };
}

function validateRequiredWebGpuEvidence(provider, evidence, successfulSessionCount) {
  if (provider !== "webgpu") {
    throw new Error("Single-model trained decode mode requires backend=webgpu.");
  }
  if (
    !evidence ||
    evidence.provider_requested !== "webgpu" ||
    evidence.provider_actual !== null ||
    !arrayEquals(evidence.execution_provider_list, ["webgpu"]) ||
    evidence.whole_session_provider_retry !== false ||
    evidence.exact_provider_request_and_session_creation_observed !== true ||
    evidence.ort_webgpu?.ort_device_available !== true ||
    successfulSessionCount !== 2
  ) {
    throw new Error(
      "Single-model WebGPU verification requires two exact-provider sessions and an ORT GPUDevice."
    );
  }
  return {
    ...evidence,
    required_for_single_model: true,
    required_verification_passed: true,
    cache_output_location_verification_required: true,
    verification_method:
      "two sessions created from executionProviders=['webgpu']; ORT exposed a GPUDevice; " +
      "cache tensors must report gpu-buffer; graph-wide/per-node placement remains unknown",
  };
}

function gpuRuntimeMetadata() {
  const evidence = ortWebGpuEvidence();
  const device = globalThis.ort?.env?.webgpu?.device || null;
  const navigatorGpu = globalThis.navigator?.gpu || null;
  let deviceFeatures = null;
  try {
    deviceFeatures = device?.features ? [...device.features].sort() : null;
  } catch {
    deviceFeatures = null;
  }
  return {
    navigator_gpu_available: Boolean(navigatorGpu),
    ort_webgpu: evidence,
    device_label: typeof device?.label === "string" ? device.label || null : null,
    device_features: deviceFeatures,
  };
}

function browserRuntimeMetadata() {
  const userAgentData = globalThis.navigator?.userAgentData;
  return {
    user_agent: globalThis.navigator?.userAgent || null,
    user_agent_brands: Array.isArray(userAgentData?.brands)
      ? userAgentData.brands.map((brand) => ({ ...brand }))
      : null,
    mobile: userAgentData?.mobile ?? null,
    platform: userAgentData?.platform || globalThis.navigator?.platform || null,
    language: globalThis.navigator?.language || null,
    languages: Array.isArray(globalThis.navigator?.languages)
      ? [...globalThis.navigator.languages]
      : null,
    hardware_concurrency: globalThis.navigator?.hardwareConcurrency || null,
    device_memory_gb: globalThis.navigator?.deviceMemory || null,
  };
}

async function prepareDecodeInputs(vocabSize) {
  const started = performance.now();
  if (!Number.isInteger(vocabSize) || vocabSize < 1) {
    throw new Error("A positive vocabulary size is required for deterministic input IDs.");
  }
  const publicInputs = [];
  for (const inputTokens of DECODE_CONTEXT_LENGTHS) {
    const tokenIds = Array.from(
      { length: inputTokens },
      (_, index) => (131 * index + 17) % vocabSize
    );
    const tensorData = BigInt64Array.from(tokenIds, (tokenId) => BigInt(tokenId));
    const tensor = new ort.Tensor("int64", tensorData, [1, inputTokens]);
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
    DECODE_STATE.inputs.set(inputTokens, input);
    publicInputs.push({ ...input, tensor: undefined });
  }
  DECODE_STATE.inputPreparationRecord = {
    phase: "input_preparation",
    duration_ms: performance.now() - started,
    input_semantics: "deterministic_pretokenized_ids",
    fixture_contract: "ids[i]=(131*i+17) mod vocab_size",
    vocab_size: vocabSize,
    tokenizer_asset: null,
    requested_context_lengths: [...DECODE_CONTEXT_LENGTHS],
    all_actual_lengths_verified: publicInputs.every(
      (input) => input.input_tokens === input.actual_tensor_tokens
    ),
    ...currentDecodeLabels(),
  };
  return publicInputs;
}

async function loadDecodeBundle(provider) {
  const loadStarted = performance.now();
  requireExplicitProvider(provider);
  const benchmarkMode = requestedDecodeBenchmarkMode();
  const acceptanceMode = requestedDecodeAcceptanceMode();
  const acceptanceRootSha256 = requestedDecodeAcceptanceRootSha256();
  if (acceptanceMode && benchmarkMode !== "single") {
    throw new Error("Acceptance mode requires mode=single.");
  }
  if (acceptanceMode && acceptanceRootSha256 == null) {
    throw new Error(
      "Acceptance mode requires acceptance_root_sha256 for the wrapper manifest."
    );
  }
  if (!acceptanceMode && acceptanceRootSha256 != null) {
    throw new Error("acceptance_root_sha256 is valid only with acceptance=1.");
  }
  const directQuery = benchmarkMode === "single"
    ? requestedDirectProvenanceQuery()
    : null;
  if (acceptanceMode && directQuery) {
    throw new Error(
      "Acceptance mode requires the exporter-produced single-decode wrapper manifest."
    );
  }
  if (benchmarkMode === "single" && provider !== "webgpu") {
    throw new Error("Single-model trained decode mode requires backend=webgpu.");
  }
  DECODE_STATE.benchmarkMode = benchmarkMode;
  DECODE_STATE.acceptanceMode = acceptanceMode;
  DECODE_STATE.acceptanceRootSha256 = acceptanceRootSha256;
  ort.env.wasm.wasmPaths = new URL(DECODE_ORT_VENDOR_BASE_PATH, document.baseURI).href;
  verifyOrtVersionPin();

  const requestedSource = directQuery?.provenance || requestedDecodeManifestUrl();
  const sourceUrl = resolveArtifactUrl(requestedSource);
  DECODE_STATE.manifestUrl = sourceUrl;
  const sourceStarted = performance.now();
  const response = await fetch(sourceUrl);
  if (!response.ok) {
    throw new Error(
      `Missing cached-decode ${benchmarkMode} source ${sourceUrl} ` +
      `(HTTP ${response.status}).`
    );
  }
  const sourceBuffer = await response.arrayBuffer();
  const readFinished = performance.now();
  const hashStarted = performance.now();
  const sourceSha256 = await sha256ArrayBuffer(sourceBuffer);
  if (acceptanceMode && sourceSha256 !== acceptanceRootSha256) {
    throw new Error(
      `Acceptance wrapper-manifest root SHA-256 mismatch: expected ` +
      `${acceptanceRootSha256}, got ${sourceSha256}.`
    );
  }
  const hashFinished = performance.now();
  if (
    directQuery &&
    (
      sourceBuffer.byteLength !== directQuery.bytes ||
      sourceSha256 !== directQuery.sha256
    )
  ) {
    throw new Error(
      "Direct single-model provenance verification failed before JSON parsing: expected " +
      `${directQuery.bytes} bytes / ${directQuery.sha256}, got ` +
      `${sourceBuffer.byteLength} bytes / ${sourceSha256}.`
    );
  }
  const sourceText = new TextDecoder("utf-8", { fatal: true }).decode(sourceBuffer);
  let sourceJson;
  try {
    sourceJson = JSON.parse(sourceText);
  } catch (error) {
    throw new Error(`Cached-decode source is not valid JSON: ${error.message}`);
  }

  let manifest = null;
  let artifactMode = null;
  let sourceKind = null;
  let sourceRecord = null;
  const assets = [];
  if (benchmarkMode === "matched") {
    manifest = validateDecodeManifest(sourceJson);
    artifactMode = decodeManifestMode(manifest);
    sourceKind = "matched_decode_manifest";
  } else if (directQuery) {
    artifactMode = "trained";
    sourceKind = "direct_pinned_model_provenance";
  } else {
    manifest = validateSingleDecodeManifest(sourceJson);
    artifactMode = "trained";
    sourceKind = "single_decode_manifest";
  }
  DECODE_STATE.manifest = manifest || {
    artifact_type: "direct_pinned_trained_cached_decode_provenance",
    bytes: directQuery.bytes,
    provenance: requestedSource,
    schema_version: 1,
    sha256: directQuery.sha256,
  };
  DECODE_STATE.artifactMode = artifactMode;
  DECODE_STATE.manifestRawText = sourceText;
  DECODE_STATE.manifestSha256 = sourceSha256;
  applyDecodeArtifactModeUi(artifactMode);
  sourceRecord = {
    phase: "artifact_verification",
    artifact_kind: sourceKind,
    artifact_id: benchmarkMode === "matched" ? "matched_decode" : "single_decode",
    relative_path: requestedSource,
    url: sourceUrl,
    fetch_and_read_ms: readFinished - sourceStarted,
    hash_ms: hashFinished - hashStarted,
    total_fetch_read_hash_ms: hashFinished - sourceStarted,
    bytes: sourceBuffer.byteLength,
    expected_bytes: directQuery?.bytes ?? null,
    actual_sha256: sourceSha256,
    expected_sha256: directQuery?.sha256 ?? acceptanceRootSha256,
    bytes_verified: directQuery ? sourceBuffer.byteLength === directQuery.bytes : null,
    hash_verified:
      directQuery || acceptanceMode
        ? sourceSha256 === (directQuery?.sha256 ?? acceptanceRootSha256)
        : null,
    hash_verification_status: directQuery
      ? "verified_by_direct_query_pin"
      : acceptanceMode
        ? "verified_by_external_acceptance_root"
        : "unknown_no_external_expected_digest",
    verification_before_parse_or_ort: true,
    content_type: response.headers.get("content-type") || null,
    browser_cache_state: "unknown",
    browser_cache_state_reason:
      "Fetch does not expose a reliable network-versus-browser-cache classification",
    ...currentDecodeLabels(),
  };
  DECODE_STATE.artifacts.push(sourceRecord);

  if (benchmarkMode === "matched") {
    for (const [role, modelEntry] of Object.entries(manifest.models)) {
      const provenancePin = manifest.artifacts[modelEntry.provenance];
      const provenanceArtifact = await fetchVerifiedArtifact(
        modelEntry.provenance,
        provenancePin,
        "model_provenance",
        role,
        sourceUrl
      );
      const provenance = validateDecodeProvenance(
        decodeJsonBuffer(provenanceArtifact.buffer, modelEntry.provenance),
        role,
        modelEntry,
        manifest
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
      const configPin = provenance.artifacts["model-config.yaml"];
      const configPath = `${modelEntry.directory}/${configPin.file || "model-config.yaml"}`;
      crossCheckPairArtifact(manifest, configPath, configPin, role);
      const configArtifact = await fetchVerifiedArtifact(
        configPath,
        configPin,
        "model_config_source",
        role,
        sourceUrl
      );
      if (
        configArtifact.record.actual_sha256 !==
        provenance.model.config_source_sha256.toLowerCase()
      ) {
        throw new Error(`${role} source config digest disagrees with model provenance.`);
      }
      const graphSelection = selectGraphArtifacts(provenance);
      const prefillPath =
        `${modelEntry.directory}/${graphSelection.prefill.file || graphSelection.prefillName}`;
      const decodePath =
        `${modelEntry.directory}/${graphSelection.decode.file || graphSelection.decodeName}`;
      crossCheckPairArtifact(manifest, prefillPath, graphSelection.prefill, role);
      crossCheckPairArtifact(manifest, decodePath, graphSelection.decode, role);
      const prefillArtifact = await fetchVerifiedArtifact(
        prefillPath,
        graphSelection.prefill,
        "cached_prefill_onnx_graph",
        role,
        sourceUrl
      );
      const decodeArtifact = await fetchVerifiedArtifact(
        decodePath,
        graphSelection.decode,
        "cached_decode_onnx_graph",
        role,
        sourceUrl
      );
      assets.push({
        role,
        id: modelEntry.name,
        modelEntry,
        config,
        provenance,
        provenance_file: modelEntry.provenance,
        provenance_sha256: provenanceArtifact.record.actual_sha256,
        config_file: configPath,
        config_source_sha256: configArtifact.record.actual_sha256,
        config_canonical_sha256: computedConfigSha256,
        decision_output_abi: decodeDecisionAbi(provenance),
        precision: graphSelection.precision,
        prefill_file: prefillPath,
        prefill_sha256: prefillArtifact.record.actual_sha256,
        prefill_bytes: prefillArtifact.record.bytes,
        prefill_buffer: prefillArtifact.buffer,
        decode_file: decodePath,
        decode_sha256: decodeArtifact.record.actual_sha256,
        decode_bytes: decodeArtifact.record.bytes,
        decode_buffer: decodeArtifact.buffer,
      });
    }
    validateMatchedConfigs(assets, manifest);
  } else {
    let provenanceArtifact;
    let provenance;
    let provenanceUrl;
    let provenanceRawText;
    let trainingLineage = null;
    let trainingLineageRawText = null;
    const modelEntry = manifest?.model || null;
    if (directQuery) {
      provenanceArtifact = { buffer: sourceBuffer, record: sourceRecord };
      provenanceUrl = sourceUrl;
      provenanceRawText = sourceText;
      provenance = validateSingleDecodeProvenance(sourceJson);
      sourceRecord.artifact_id = provenance.model.pair_role;
    } else {
      const provenancePin = manifest.artifacts[modelEntry.provenance];
      provenanceArtifact = await fetchVerifiedArtifact(
        modelEntry.provenance,
        provenancePin,
        "model_provenance",
        modelEntry.pair_role,
        sourceUrl
      );
      provenanceUrl = provenanceArtifact.record.url;
      provenanceRawText = decodeUtf8Buffer(
        provenanceArtifact.buffer,
        modelEntry.provenance
      );
      provenance = validateSingleDecodeProvenance(
        JSON.parse(provenanceRawText),
        modelEntry
      );
    }
    const role = provenance.model.pair_role;
    const config = provenance.model.config;
    validateConfig(config, provenance.model.name);
    const computedConfigSha256 = await sha256Text(modelConfigCanonicalJson(config));
    if (computedConfigSha256 !== provenance.model.config_canonical_sha256) {
      throw new Error(
        `Single-model embedded config SHA-256 mismatch: expected ` +
        `${provenance.model.config_canonical_sha256}, got ${computedConfigSha256}.`
      );
    }
    const configPin = provenance.artifacts["model-config.yaml"];
    const configArtifact = await fetchVerifiedArtifact(
      configPin.file,
      configPin,
      "model_config_source",
      role,
      provenanceUrl
    );
    if (
      configArtifact.record.actual_sha256 !==
      provenance.model.config_source_sha256.toLowerCase()
    ) {
      throw new Error("Single-model source config digest disagrees with provenance.");
    }
    const graphSelection = selectGraphArtifacts(provenance);
    const prefillArtifact = await fetchVerifiedArtifact(
      graphSelection.prefill.file,
      graphSelection.prefill,
      "cached_prefill_onnx_graph",
      role,
      provenanceUrl
    );
    const decodeArtifact = await fetchVerifiedArtifact(
      graphSelection.decode.file,
      graphSelection.decode,
      "cached_decode_onnx_graph",
      role,
      provenanceUrl
    );
    const metadataPin = provenance.artifacts["meta.json"];
    const metadataArtifact = await fetchVerifiedArtifact(
      metadataPin.file,
      metadataPin,
      "cached_runtime_metadata",
      role,
      provenanceUrl
    );
    const runtimeMetadata = validateCachedRuntimeMetadata(
      decodeJsonBuffer(metadataArtifact.buffer, metadataPin.file),
      provenance,
      graphSelection.precision
    );
    if (provenance.tokenizer.kind === "bpe") {
      const tokenizerPin = provenance.artifacts["tokenizer.json"];
      await fetchVerifiedArtifact(
        tokenizerPin.file,
        tokenizerPin,
        "tokenizer",
        role,
        provenanceUrl
      );
    }
    if (provenance.training_lineage_export != null) {
      const lineagePin = provenance.artifacts[provenance.training_lineage_export];
      const lineageArtifact = await fetchVerifiedArtifact(
        lineagePin.file,
        lineagePin,
        "training_lineage",
        role,
        provenanceUrl
      );
      const lineage = decodeJsonBuffer(lineageArtifact.buffer, lineagePin.file);
      trainingLineageRawText = decodeUtf8Buffer(lineageArtifact.buffer, lineagePin.file);
      if (
        lineage.kind !== "localagent_training_lineage_export" ||
        lineage.schema_version !== 1 ||
        lineage.checkpoint_sha256 !== provenance.weights.checkpoint_sha256
      ) {
        throw new Error("Training-lineage sidecar disagrees with model provenance.");
      }
      trainingLineage = lineage;
    }
    validateAcceptanceTrainingLineage(
      provenance,
      trainingLineage,
      DECODE_STATE.acceptanceMode
    );
    assets.push({
      role,
      id: provenance.model.name,
      modelEntry: modelEntry || {
        name: provenance.model.name,
        pair_role: role,
        provenance: requestedSource,
      },
      config,
      provenance,
      runtime_metadata: runtimeMetadata,
      training_lineage: trainingLineage,
      training_lineage_raw_text: trainingLineageRawText,
      provenance_raw_text: provenanceRawText,
      provenance_bytes: provenanceArtifact.record.bytes,
      provenance_file: modelEntry?.provenance || requestedSource,
      provenance_sha256: provenanceArtifact.record.actual_sha256,
      config_file: configPin.file,
      config_source_sha256: configArtifact.record.actual_sha256,
      config_canonical_sha256: computedConfigSha256,
      decision_output_abi: decodeDecisionAbi(provenance),
      precision: graphSelection.precision,
      prefill_file: graphSelection.prefill.file,
      prefill_sha256: prefillArtifact.record.actual_sha256,
      prefill_bytes: prefillArtifact.record.bytes,
      prefill_buffer: prefillArtifact.buffer,
      decode_file: graphSelection.decode.file,
      decode_sha256: decodeArtifact.record.actual_sha256,
      decode_bytes: decodeArtifact.record.bytes,
      decode_buffer: decodeArtifact.buffer,
    });
  }

  await prepareDecodeInputs(assets[0].config.vocab_size);
  const sessionOrderSeed = benchmarkMode === "matched"
    ? `${DECODE_DEFAULT_SEED}:session-create`
    : `${DECODE_DEFAULT_SEED}:single:session-create`;
  const sessionOrder = shuffled(
    assets.flatMap((asset) => [
      { asset, graphKind: "prefill" },
      { asset, graphKind: "decode" },
    ]),
    sessionOrderSeed
  );
  const partialSessions = new Map();
  for (let orderIndex = 0; orderIndex < sessionOrder.length; orderIndex++) {
    const { asset, graphKind } = sessionOrder[orderIndex];
    const cacheContract = asset.provenance.graph_contract;
    const graphContract = cacheContract.graphs[asset.precision];
    const presentNames = cacheContract.cache_slots.flatMap((slot) => slot.present_outputs);
    const options = decodeSessionOptions(
      provider,
      presentNames,
      asset.decision_output_abi
    );
    const graphBuffer =
      graphKind === "prefill" ? asset.prefill_buffer : asset.decode_buffer;
    const graphSha256 =
      graphKind === "prefill" ? asset.prefill_sha256 : asset.decode_sha256;
    const graphBytes = graphKind === "prefill" ? asset.prefill_bytes : asset.decode_bytes;
    const started = performance.now();
    try {
      const session = await ort.InferenceSession.create(graphBuffer, options);
      const observedContract = validateSessionContract(
        session, graphContract, graphKind, asset.id
      );
      const evidence = providerEvidence(provider, true);
      DECODE_STATE.sessions.push({
        phase: "session_create",
        benchmark_session_id: DECODE_STATE.benchmarkSessionId,
        run_challenge: DECODE_STATE.runChallenge,
        order_index: orderIndex,
        arm_id: asset.id,
        graph_kind: graphKind,
        session_create_ms: performance.now() - started,
        graph_sha256: graphSha256,
        graph_bytes: graphBytes,
        preferred_output_location: options.preferredOutputLocation,
        cache_residency_requested: provider === "webgpu" ? "gpu-buffer" : "cpu",
        next_token_residency_requested: "cpu",
        logits_residency_requested:
          asset.decision_output_abi === DECODE_DECISION_ABI_LOGITS ? "cpu" : null,
        ...observedContract,
        ...evidence,
        ...currentDecodeLabels(),
        error: null,
      });
      if (!partialSessions.has(asset.id)) partialSessions.set(asset.id, {});
      partialSessions.get(asset.id)[graphKind] = session;
    } catch (error) {
      DECODE_STATE.sessions.push({
        phase: "session_create",
        benchmark_session_id: DECODE_STATE.benchmarkSessionId,
        run_challenge: DECODE_STATE.runChallenge,
        order_index: orderIndex,
        arm_id: asset.id,
        graph_kind: graphKind,
        session_create_ms: performance.now() - started,
        graph_sha256: graphSha256,
        graph_bytes: graphBytes,
        preferred_output_location: options.preferredOutputLocation,
        ...providerEvidence(provider, false),
        ...currentDecodeLabels(),
        error: errorDetail(error),
      });
      throw error;
    }
  }

  for (const asset of assets) {
    const sessions = partialSessions.get(asset.id);
    if (!sessions?.prefill || !sessions?.decode) {
      throw new Error(`${asset.id} did not create both required sessions.`);
    }
    DECODE_STATE.arms.set(asset.id, {
      ...asset,
      prefill_session: sessions.prefill,
      decode_session: sessions.decode,
      prefill_buffer: null,
      decode_buffer: null,
    });
  }
  const evidence = providerEvidence(provider, true);
  DECODE_STATE.providerVerification = benchmarkMode === "single"
    ? validateRequiredWebGpuEvidence(
      provider,
      evidence,
      DECODE_STATE.sessions.filter((record) => !record.error).length
    )
    : evidence;
  DECODE_STATE.readyAtMs = performance.now();
  return {
    bundle_and_session_ms: DECODE_STATE.readyAtMs - loadStarted,
    provider: DECODE_STATE.providerVerification,
  };
}

function dtypeBytes(dtype) {
  const sizes = {
    float16: 2,
    float32: 4,
    int64: 8,
  };
  if (!sizes[dtype]) throw new Error(`Unsupported tensor dtype '${dtype}'.`);
  return sizes[dtype];
}

function tensorLogicalBytes(tensor) {
  if (!tensor || !Array.isArray(tensor.dims)) return 0;
  return tensor.dims.reduce((product, value) => product * value, 1) *
    dtypeBytes(tensor.type);
}

function reportedTensorLocation(tensor) {
  return typeof tensor?.location === "string" ? tensor.location : "unknown_not_exposed";
}

function assertTensorLocation(tensor, expected, label) {
  const observed = reportedTensorLocation(tensor);
  if (observed !== expected) {
    throw new Error(`${label} location is ${observed}; expected ${expected}.`);
  }
  return observed;
}

function validateNextTokenTensor(tensor, label, vocabSize) {
  if (!tensor || tensor.type !== "int64" || !arrayEquals(tensor.dims, [1])) {
    throw new Error(
      `${label} must be an int64 [1] next_token tensor; got ` +
      `${tensor?.type || "missing"} ${JSON.stringify(tensor?.dims || null)}.`
    );
  }
  assertTensorLocation(tensor, "cpu", label);
  if (!tensor.data || tensor.data.length !== 1) {
    throw new Error(`${label} CPU token data is unavailable.`);
  }
  const value = Number(tensor.data[0]);
  if (
    !Number.isSafeInteger(value) ||
    value < 0 ||
    !Number.isInteger(vocabSize) ||
    value >= vocabSize
  ) {
    throw new Error(`${label} contains an invalid token ID.`);
  }
  return value;
}

function float16BitsToNumber(bits) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0x1f) return fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function floatTensorValue(tensor, index) {
  const raw = tensor.data[index];
  if (
    tensor.type === "float16" &&
    tensor.data instanceof Uint16Array &&
    tensor.data.constructor?.name !== "Float16Array"
  ) {
    return float16BitsToNumber(raw);
  }
  return Number(raw);
}

function validateLogitsTensor(tensor, label, vocabSize, precision) {
  const expectedDtype = expectedCacheDtype(precision);
  if (
    !tensor ||
    tensor.type !== expectedDtype ||
    !arrayEquals(tensor.dims, [1, vocabSize])
  ) {
    throw new Error(
      `${label} must be a ${expectedDtype} [1, ${vocabSize}] logits tensor; got ` +
      `${tensor?.type || "missing"} ${JSON.stringify(tensor?.dims || null)}.`
    );
  }
  assertTensorLocation(tensor, "cpu", label);
  if (!tensor.data || tensor.data.length !== vocabSize) {
    throw new Error(`${label} CPU logits data is unavailable.`);
  }
  let bestToken = 0;
  let bestValue = floatTensorValue(tensor, 0);
  if (!Number.isFinite(bestValue)) {
    throw new Error(`${label} contains a non-finite logit.`);
  }
  for (let token = 1; token < vocabSize; token++) {
    const value = floatTensorValue(tensor, token);
    if (!Number.isFinite(value)) {
      throw new Error(`${label} contains a non-finite logit.`);
    }
    if (value > bestValue) {
      bestToken = token;
      bestValue = value;
    }
  }
  return bestToken;
}

function validateDecodeDecisionOutputs(
  outputs,
  label,
  vocabSize,
  precision,
  decisionAbi = DECODE_DECISION_ABI_LOGITS
) {
  const compatibilityToken = validateNextTokenTensor(
    outputs?.next_token,
    `${label}.next_token`,
    vocabSize
  );
  if (decisionAbi === DECODE_DECISION_ABI_LEGACY) {
    if (outputs?.logits != null) {
      throw new Error(`${label} legacy next-token ABI unexpectedly returned logits.`);
    }
    return compatibilityToken;
  }
  if (decisionAbi !== DECODE_DECISION_ABI_LOGITS) {
    throw new Error(`${label} uses unsupported decision ABI '${decisionAbi}'.`);
  }
  const selectedToken = validateLogitsTensor(
    outputs?.logits,
    `${label}.logits`,
    vocabSize,
    precision
  );
  if (compatibilityToken !== selectedToken) {
    throw new Error(
      `${label} next_token ${compatibilityToken} disagrees with logits argmax ${selectedToken}.`
    );
  }
  return selectedToken;
}

function expectedCacheDtype(precision) {
  if (precision === "fp16") return "float16";
  if (precision === "fp32") return "float32";
  throw new Error(`Unsupported graph precision '${precision}'.`);
}

function validateCacheTensor(
  tensor,
  slot,
  expectedAttentionSequence,
  config,
  provider,
  precision,
  name
) {
  const expectedDtype = expectedCacheDtype(precision);
  if (!tensor || tensor.type !== expectedDtype || !Array.isArray(tensor.dims)) {
    throw new Error(
      `${name} must be a ${expectedDtype} cache tensor; got ${tensor?.type || "missing"}.`
    );
  }
  const expectedDims = slot.kind === "attn"
    ? [1, config.n_kv_heads, expectedAttentionSequence, config.d_model / config.n_heads]
    : [1, config.d_model, config.conv_kernel - 1];
  if (!arrayEquals(tensor.dims, expectedDims)) {
    throw new Error(
      `${name} dims ${JSON.stringify(tensor.dims)} do not match ` +
      `${JSON.stringify(expectedDims)}.`
    );
  }
  const requestedLocation = provider === "webgpu" ? "gpu-buffer" : "cpu";
  const observedLocation = assertTensorLocation(tensor, requestedLocation, name);
  return {
    name,
    slot: slot.slot,
    kind: slot.kind,
    dtype: tensor.type,
    dims: [...tensor.dims],
    logical_bytes: tensorLogicalBytes(tensor),
    requested_location: requestedLocation,
    reported_location: observedLocation,
  };
}

function makeAllocationTracker() {
  return {
    cache_tensors_allocated: 0,
    next_token_tensors_allocated: 0,
    logits_tensors_allocated: 0,
    decode_input_tensors_allocated: 0,
    cache_dispose_attempted: 0,
    cache_dispose_succeeded: 0,
    cache_dispose_failed: 0,
    cache_dispose_api_unavailable: 0,
    next_token_dispose_attempted: 0,
    next_token_dispose_succeeded: 0,
    next_token_dispose_failed: 0,
    next_token_dispose_api_unavailable: 0,
    logits_dispose_attempted: 0,
    logits_dispose_succeeded: 0,
    logits_dispose_failed: 0,
    logits_dispose_api_unavailable: 0,
    decode_input_dispose_attempted: 0,
    decode_input_dispose_succeeded: 0,
    decode_input_dispose_failed: 0,
    decode_input_dispose_api_unavailable: 0,
    superseded_cache_tensors_released: 0,
    final_cache_tensors_released: 0,
  };
}

function disposeTracked(tensor, kind, tracker, disposed) {
  if (!tensor || disposed.has(tensor)) return false;
  disposed.add(tensor);
  const prefix = kind === "cache"
    ? "cache"
    : kind === "next_token"
      ? "next_token"
      : kind === "logits"
        ? "logits"
        : "decode_input";
  tracker[`${prefix}_dispose_attempted`] += 1;
  if (typeof tensor.dispose !== "function") {
    tracker[`${prefix}_dispose_api_unavailable`] += 1;
    return false;
  }
  try {
    tensor.dispose();
    tracker[`${prefix}_dispose_succeeded`] += 1;
    return true;
  } catch {
    tracker[`${prefix}_dispose_failed`] += 1;
    return false;
  }
}

function disposeCacheMap(cacheMap, tracker, disposed, releaseKind) {
  let released = 0;
  for (const tensor of cacheMap?.values?.() || []) {
    if (disposeTracked(tensor, "cache", tracker, disposed)) released += 1;
  }
  if (releaseKind === "superseded") {
    tracker.superseded_cache_tensors_released += released;
  } else if (releaseKind === "final") {
    tracker.final_cache_tensors_released += released;
  }
  cacheMap?.clear?.();
  return released;
}

function extractCacheOutputs(
  outputs,
  arm,
  expectedAttentionSequence,
  provider,
  tracker
) {
  const cacheMap = new Map();
  const metadata = [];
  for (const slot of arm.provenance.graph_contract.cache_slots) {
    for (const name of slot.present_outputs) {
      const tensor = outputs[name];
      metadata.push(validateCacheTensor(
        tensor,
        slot,
        expectedAttentionSequence,
        arm.config,
        provider,
        arm.precision,
        name
      ));
      cacheMap.set(name, tensor);
      tracker.cache_tensors_allocated += 1;
    }
  }
  return { cacheMap, metadata };
}

function cacheLogicalBytes(cacheMap) {
  let bytes = 0;
  for (const tensor of cacheMap.values()) bytes += tensorLogicalBytes(tensor);
  return bytes;
}

function cacheFeedsFromPresent(cacheMap, contract) {
  const feeds = {};
  for (const slot of contract.cache_slots) {
    for (let index = 0; index < slot.past_inputs.length; index++) {
      const pastName = slot.past_inputs[index];
      const presentName = slot.present_outputs[index];
      const tensor = cacheMap.get(presentName);
      if (!tensor) throw new Error(`Missing direct cache binding ${presentName} -> ${pastName}.`);
      // Deliberately bind the ORT tensor object itself; cache contents are never materialized.
      feeds[pastName] = tensor;
    }
  }
  return feeds;
}

function emptyConditionRecord(phase, condition, arm, provider, outputTokens, globalOrder) {
  const decodePasses = Math.max(0, outputTokens - 1);
  return {
    phase,
    benchmark_session_id: DECODE_STATE.benchmarkSessionId,
    run_id: DECODE_STATE.runId,
    run_challenge: DECODE_STATE.runChallenge,
    global_order_index: globalOrder,
    repetition: condition.repetition,
    order_index: condition.order_index,
    arm_id: condition.arm_id,
    pair_role: arm.role,
    input_tokens: condition.input_tokens,
    prompt_tokens_requested: condition.input_tokens,
    actual_input_tokens:
      DECODE_STATE.inputs.get(condition.input_tokens)?.actual_tensor_tokens ?? null,
    prompt_tokens_actual:
      DECODE_STATE.inputs.get(condition.input_tokens)?.actual_tensor_tokens ?? null,
    output_tokens_requested: outputTokens,
    actual_output_tokens: 0,
    generated_token_ids: [],
    generated_token_interpretation: DECODE_STATE.artifactMode === "trained"
      ? (
        "trained-weight logits argmax IDs used only to drive cached graph passes; not decoded, " +
        "quality-scored, or interpreted as actions"
      )
      : "meaningless deterministic-random-weight logits argmax IDs; not decoded or scored",
    decision_output_abi: arm.decision_output_abi,
    graph_pass_counts: {
      prefill: 0,
      decode: 0,
      prefill_attempted: 0,
      decode_attempted: 0,
      total: 0,
      total_attempted: 0,
      expected_prefill: 1,
      expected_decode: decodePasses,
      expected_total: outputTokens,
    },
    graph_files: {
      prefill: arm.prefill_file,
      decode: arm.decode_file,
    },
    graph_sha256: {
      prefill: arm.prefill_sha256,
      decode: arm.decode_sha256,
    },
    graph_bytes: {
      prefill: arm.prefill_bytes,
      decode: arm.decode_bytes,
    },
    ttft_ms: null,
    tpot_ms: null,
    decode_tokens_per_second: null,
    prefill_ms: null,
    decode_inference_ms: null,
    decode_wall_ms: null,
    generation_wall_ms: null,
    model_decode_tokens_per_second: null,
    decode_pass_records: [],
    cache: {
      enabled: true,
      dtype: expectedCacheDtype(arm.precision),
      requested_residency: provider === "webgpu" ? "gpu-buffer" : "cpu",
      next_token_residency: "cpu",
      logits_residency:
        arm.decision_output_abi === DECODE_DECISION_ABI_LOGITS ? "cpu" : null,
      token_selection_source:
        arm.decision_output_abi === DECODE_DECISION_ABI_LOGITS
          ? "validated_logits_argmax"
          : "legacy_exported_next_token",
      next_token_role:
        arm.decision_output_abi === DECODE_DECISION_ABI_LOGITS
          ? "compatibility_cross_check"
          : "legacy_primary_decision_output",
      update_strategy:
        "present_outputs_rebound_directly_as_past_inputs_without_cpu_materialization",
      cache_data_read_to_javascript: false,
      slot_count: arm.provenance.graph_contract.cache_slots.length,
      tensor_count:
        arm.provenance.graph_contract.cache_slots.reduce(
          (sum, slot) => sum + slot.present_outputs.length, 0
        ),
      slots: arm.provenance.graph_contract.cache_slots,
      prefill_tensors: [],
      prefill_logical_bytes: null,
      final_tensors: [],
      final_logical_bytes: null,
    },
    allocation_disposal: makeAllocationTracker(),
    disposal_contract_verified: null,
    provider_requested: provider,
    provider_actual: null,
    provider_actual_observation:
      "not exposed; exact provider request/session creation and cache tensor locations recorded",
    graph_wide_provider_verified: false,
    whole_session_provider_retry: false,
    per_node_placement_verified: false,
    per_node_fallback_status: "unknown",
    ...currentDecodeLabels(),
    run_ok: false,
    error: null,
  };
}

async function runDecodeCondition(
  phase,
  condition,
  provider,
  outputTokens,
  globalOrder
) {
  const arm = DECODE_STATE.arms.get(condition.arm_id);
  const input = DECODE_STATE.inputs.get(condition.input_tokens);
  if (!arm || !input) throw new Error("Unknown cached decode benchmark condition.");
  const record = emptyConditionRecord(
    phase, condition, arm, provider, outputTokens, globalOrder
  );
  const tracker = record.allocation_disposal;
  const disposed = new Set();
  let currentCache = new Map();
  let pendingOutputs = null;
  let currentDecodeInput = null;
  try {
    const prefillStarted = performance.now();
    record.graph_pass_counts.prefill_attempted += 1;
    pendingOutputs = await arm.prefill_session.run({ input_ids: input.tensor });
    const prefillResolved = performance.now();
    tracker.next_token_tensors_allocated += 1;
    if (pendingOutputs.logits) tracker.logits_tensors_allocated += 1;
    const firstToken = validateDecodeDecisionOutputs(
      pendingOutputs,
      `${arm.id}.prefill`,
      arm.config.vocab_size,
      arm.precision,
      arm.decision_output_abi
    );
    const firstTokenAvailable = performance.now();
    record.prefill_ms = prefillResolved - prefillStarted;
    record.ttft_ms = firstTokenAvailable - prefillStarted;
    record.generated_token_ids.push(firstToken);
    disposeTracked(pendingOutputs.next_token, "next_token", tracker, disposed);
    disposeTracked(pendingOutputs.logits, "logits", tracker, disposed);

    const prefillCaches = extractCacheOutputs(
      pendingOutputs,
      arm,
      condition.input_tokens,
      provider,
      tracker
    );
    currentCache = prefillCaches.cacheMap;
    record.cache.prefill_tensors = prefillCaches.metadata;
    record.cache.prefill_logical_bytes = cacheLogicalBytes(currentCache);
    record.graph_pass_counts.prefill += 1;
    pendingOutputs = null;

    const decodePasses = outputTokens - 1;
    let decodeInferenceMs = 0;
    const decodeWallStarted = firstTokenAvailable;
    let lastTokenAvailableAt = decodeWallStarted;
    for (let passIndex = 0; passIndex < decodePasses; passIndex++) {
      const inputToken = record.generated_token_ids.at(-1);
      currentDecodeInput = new ort.Tensor(
        "int64",
        BigInt64Array.of(BigInt(inputToken)),
        [1, 1]
      );
      tracker.decode_input_tensors_allocated += 1;
      const feeds = {
        input_ids: currentDecodeInput,
        ...cacheFeedsFromPresent(currentCache, arm.provenance.graph_contract),
      };
      const passStarted = performance.now();
      record.graph_pass_counts.decode_attempted += 1;
      pendingOutputs = await arm.decode_session.run(feeds);
      const passResolved = performance.now();
      tracker.next_token_tensors_allocated += 1;
      if (pendingOutputs.logits) tracker.logits_tensors_allocated += 1;
      const nextToken = validateDecodeDecisionOutputs(
        pendingOutputs,
        `${arm.id}.decode[${passIndex}]`,
        arm.config.vocab_size,
        arm.precision,
        arm.decision_output_abi
      );
      const nextTokenAvailable = performance.now();
      lastTokenAvailableAt = nextTokenAvailable;
      const inferenceMs = passResolved - passStarted;
      decodeInferenceMs += inferenceMs;

      const present = extractCacheOutputs(
        pendingOutputs,
        arm,
        condition.input_tokens + passIndex + 1,
        provider,
        tracker
      );
      const previousCacheBytes = cacheLogicalBytes(currentCache);
      disposeCacheMap(currentCache, tracker, disposed, "superseded");
      currentCache = present.cacheMap;
      record.graph_pass_counts.decode += 1;
      record.generated_token_ids.push(nextToken);
      disposeTracked(pendingOutputs.next_token, "next_token", tracker, disposed);
      disposeTracked(pendingOutputs.logits, "logits", tracker, disposed);
      disposeTracked(currentDecodeInput, "decode_input", tracker, disposed);
      currentDecodeInput = null;
      pendingOutputs = null;
      record.decode_pass_records.push({
        pass_index: passIndex,
        input_token_id: inputToken,
        output_token_id: nextToken,
        input_tokens: 1,
        output_tokens: 1,
        attention_cache_sequence_length: condition.input_tokens + passIndex + 1,
        inference_ms: inferenceMs,
        token_available_ms: nextTokenAvailable - passStarted,
        pass_started_offset_ms: passStarted - decodeWallStarted,
        pass_resolved_offset_ms: passResolved - decodeWallStarted,
        token_available_offset_ms: nextTokenAvailable - decodeWallStarted,
        cache_logical_bytes_before: previousCacheBytes,
        cache_logical_bytes_after: cacheLogicalBytes(currentCache),
        cache_tensor_count: present.metadata.length,
        cache_tensors: present.metadata,
        cache_reported_locations: [
          ...new Set(present.metadata.map((item) => item.reported_location)),
        ],
        cache_bound_directly_without_readback: true,
      });
    }
    record.decode_inference_ms = decodeInferenceMs;
    record.decode_wall_ms = lastTokenAvailableAt - decodeWallStarted;
    record.generation_wall_ms = lastTokenAvailableAt - prefillStarted;
    record.tpot_ms = record.decode_wall_ms / decodePasses;
    record.decode_tokens_per_second = 1000 / record.tpot_ms;
    record.model_decode_tokens_per_second = decodePasses * 1000 / decodeInferenceMs;
    record.actual_output_tokens = record.generated_token_ids.length;
    record.actual_graph_input_token_positions =
      condition.input_tokens + record.graph_pass_counts.decode;
    record.cache.final_tensors = record.decode_pass_records.at(-1)
      ? arm.provenance.graph_contract.cache_slots.flatMap((slot) =>
        slot.present_outputs.map((name) => ({
          name,
          dtype: currentCache.get(name)?.type || null,
          dims: currentCache.get(name)?.dims ? [...currentCache.get(name).dims] : null,
          logical_bytes: currentCache.get(name)
            ? tensorLogicalBytes(currentCache.get(name))
            : null,
          reported_location: currentCache.get(name)
            ? reportedTensorLocation(currentCache.get(name))
            : null,
        }))
      )
      : record.cache.prefill_tensors;
    record.cache.final_logical_bytes = cacheLogicalBytes(currentCache);
    record.graph_pass_counts.total =
      record.graph_pass_counts.prefill + record.graph_pass_counts.decode;
    record.graph_pass_counts.total_attempted =
      record.graph_pass_counts.prefill_attempted +
      record.graph_pass_counts.decode_attempted;
    record.run_ok =
      record.graph_pass_counts.prefill === 1 &&
      record.graph_pass_counts.decode === decodePasses &&
      record.actual_output_tokens === outputTokens;
    if (!record.run_ok) {
      throw new Error("Observed graph/token counts do not match the fixed decode contract.");
    }
    return record;
  } catch (error) {
    record.actual_output_tokens = record.generated_token_ids.length;
    record.actual_graph_input_token_positions =
      (record.graph_pass_counts.prefill_attempted ? condition.input_tokens : 0) +
      record.graph_pass_counts.decode_attempted;
    record.graph_pass_counts.total =
      record.graph_pass_counts.prefill + record.graph_pass_counts.decode;
    record.graph_pass_counts.total_attempted =
      record.graph_pass_counts.prefill_attempted +
      record.graph_pass_counts.decode_attempted;
    if (currentCache.size) record.cache.final_logical_bytes = cacheLogicalBytes(currentCache);
    record.error = errorDetail(error);
    return record;
  } finally {
    if (pendingOutputs) {
      const contract = arm.provenance.graph_contract;
      disposeTracked(pendingOutputs.next_token, "next_token", tracker, disposed);
      disposeTracked(pendingOutputs.logits, "logits", tracker, disposed);
      for (const name of contract.cache_slots.flatMap((slot) => slot.present_outputs)) {
        disposeTracked(pendingOutputs[name], "cache", tracker, disposed);
      }
    }
    disposeTracked(currentDecodeInput, "decode_input", tracker, disposed);
    disposeCacheMap(currentCache, tracker, disposed, "final");
  }
}

function validateAcceptanceDisposalRecord(
  record,
  acceptanceMode = DECODE_STATE.acceptanceMode
) {
  if (!acceptanceMode || !record.run_ok) return record;
  const tracker = record.allocation_disposal;
  const cacheTensorCount = record.cache.tensor_count;
  const outputTokens = DECODE_ACCEPTANCE_PROTOCOL.output_tokens_per_condition;
  const decodePasses = outputTokens - 1;
  const exactCounters = {
    cache_tensors_allocated: cacheTensorCount * outputTokens,
    cache_dispose_attempted: cacheTensorCount * outputTokens,
    cache_dispose_succeeded: cacheTensorCount * outputTokens,
    cache_dispose_failed: 0,
    cache_dispose_api_unavailable: 0,
    superseded_cache_tensors_released: cacheTensorCount * decodePasses,
    final_cache_tensors_released: cacheTensorCount,
    next_token_tensors_allocated: outputTokens,
    next_token_dispose_attempted: outputTokens,
    next_token_dispose_succeeded: outputTokens,
    next_token_dispose_failed: 0,
    next_token_dispose_api_unavailable: 0,
    logits_tensors_allocated: outputTokens,
    logits_dispose_attempted: outputTokens,
    logits_dispose_succeeded: outputTokens,
    logits_dispose_failed: 0,
    logits_dispose_api_unavailable: 0,
    decode_input_tensors_allocated: decodePasses,
    decode_input_dispose_attempted: decodePasses,
    decode_input_dispose_succeeded: decodePasses,
    decode_input_dispose_failed: 0,
    decode_input_dispose_api_unavailable: 0,
  };
  const valid =
    tracker &&
    Object.entries(exactCounters).every(([field, value]) => tracker[field] === value);
  record.disposal_contract_verified = valid;
  if (!valid) {
    record.run_ok = false;
    record.error = {
      name: "Error",
      message: "Acceptance cache/tensor disposal accounting is unavailable or incomplete.",
      stack: null,
    };
    throw new Error(record.error.message);
  }
  return record;
}

function applyDecodeArtifactModeUi(mode) {
  const trained = mode === "trained";
  const heading = decodeElement("decode-artifact-heading");
  const description = decodeElement("decode-artifact-description");
  const disclaimer = decodeElement("decode-artifact-disclaimer");
  const disclaimerDescription = decodeElement("decode-artifact-disclaimer-description");
  if (heading) {
    heading.textContent = trained
      ? DECODE_TRAINED_LABEL
      : "UNTRAINED RANDOM WEIGHTS — NO CAPABILITY OR QUALITY RESULT.";
  }
  if (description) {
    description.textContent = trained
      ? (
        " This page measures checkpoint-backed prefill and iterative cache-bearing ONNX graph " +
        "execution. Generated token IDs are not decoded, scored, or treated as actions."
      )
      : (
        " This page measures prefill and iterative cache-bearing ONNX graph execution. " +
        "Generated token IDs are meaningless random-model outputs and are retained only to " +
        "prove the exact autoregressive pass count."
      );
  }
  if (disclaimer) {
    disclaimer.textContent = trained
      ? DECODE_TRAINED_LABEL
      : "LATENCY ONLY · UNTRAINED RANDOM WEIGHTS";
  }
  if (disclaimerDescription) {
    disclaimerDescription.textContent = trained
      ? (
        "No accuracy, language quality, action success, tool use, or agent capability is " +
        "evaluated here. Checkpoint and tokenizer identities are pinned; quality belongs to " +
        "a separate scorecard."
      )
      : (
        "No accuracy, language quality, action success, or agent capability is evaluated. " +
        "For WebGPU, present caches are requested as GPU buffers and rebound without " +
        "JavaScript cache readback; physical residency and per-node placement are not exposed. " +
        "WASM requests CPU caches."
      );
  }
}

function setDecodeStatus(kind, text, provider = null) {
  const status = decodeElement("decode-status");
  if (status) status.className = `status ${kind}`;
  const label = decodeElement("decode-status-text");
  if (label) label.textContent = text;
  const badge = decodeElement("decode-backend-badge");
  if (badge && provider) {
    badge.textContent = provider;
    badge.hidden = false;
  }
}

function setDecodeProgress(text) {
  const progress = decodeElement("decode-progress");
  if (progress) progress.textContent = text;
}

function fixed(value, digits = 2) {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : "—";
}

function renderDecodeSummary(payload) {
  const output = decodeElement("decode-output");
  if (!output || !payload.summary) return;
  const rows = payload.summary.conditions.map((condition) => (
    `<tr><td>${condition.arm_id}</td><td>${condition.prompt_tokens}</td>` +
    `<td>${condition.output_tokens}</td>` +
    `<td>${condition.completed}/${condition.attempted}</td>` +
    `<td>${fixed(condition.ttft_ms.p50)}</td>` +
    `<td>${fixed(condition.tpot_ms.p50)}</td>` +
    `<td>${fixed(condition.tpot_ms.p95)}</td>` +
    `<td>${fixed(condition.decode_tokens_per_second.p50, 1)}</td>` +
    `<td>${fixed(condition.decode_tokens_per_second.p95, 1)}</td>` +
    `<td>${fixed(condition.model_decode_tokens_per_second.p50, 1)}</td>` +
    `<td>${fixed(condition.final_cache_logical_bytes.p50 / 1048576, 2)}</td></tr>`
  )).join("");
  const modelHeading = payload.trained_weights === true ? "Trained model" : "Random model";
  output.innerHTML = `
    <div class="metric-grid">
      <div class="metric"><span>Measured conditions</span><strong>${payload.summary.attempted}</strong></div>
      <div class="metric"><span>Completed</span><strong>${payload.summary.completed}</strong></div>
      <div class="metric"><span>Failed</span><strong>${payload.summary.failed}</strong></div>
      <div class="metric"><span>Quality metrics</span><strong>none</strong></div>
    </div>
    <table class="benchmark-table">
      <thead><tr><th>${modelHeading}</th><th>Prompt tokens</th><th>Output tokens</th>
      <th>Completed</th><th>p50 TTFT ms</th><th>p50 TPOT ms</th><th>p95 TPOT ms</th>
      <th>p50 wall tok/s</th><th>p95 wall tok/s</th>
      <th>p50 model tok/s</th><th>final cache MiB</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <details><summary>Non-capability protocol labels</summary><pre>${JSON.stringify({
      latency_only: payload.latency_only,
      untrained_random_weights: payload.untrained_random_weights,
      trained_weights: payload.trained_weights ?? null,
      benchmark_label: payload.benchmark_label ?? null,
      capability_artifact: payload.capability_artifact,
      action_capability_claimed: payload.action_capability_claimed ?? false,
      quality_evaluation: payload.quality_evaluation,
      quality_scored_separately: payload.quality_scored_separately ?? false,
      provider: payload.metadata?.provider,
      cache_contract: payload.metadata?.cache_contract,
    }, null, 2)}</pre></details>`;
}

function publicArmMetadata() {
  return Array.from(DECODE_STATE.arms.values()).map((arm) => ({
    id: arm.id,
    pair_role: arm.role,
    model_parameters: arm.provenance.model.full_model_parameters,
    precision: arm.precision,
    decision_output_abi: arm.decision_output_abi,
    config_file: arm.config_file,
    config_source_sha256: arm.config_source_sha256,
    config_canonical_sha256: arm.config_canonical_sha256,
    provenance_file: arm.provenance_file,
    provenance_sha256: arm.provenance_sha256,
    prefill_file: arm.prefill_file,
    prefill_sha256: arm.prefill_sha256,
    prefill_bytes: arm.prefill_bytes,
    decode_file: arm.decode_file,
    decode_sha256: arm.decode_sha256,
    decode_bytes: arm.decode_bytes,
    checkpoint_sha256: arm.provenance.weights?.checkpoint_sha256 || null,
    checkpoint_step: arm.provenance.weights?.checkpoint_step ?? null,
    tokens_seen: arm.provenance.weights?.tokens_seen ?? null,
    tokenizer_sha256: arm.provenance.tokenizer?.sha256 || null,
    cache_slot_count: arm.provenance.graph_contract.cache_slots.length,
    cache_tensor_count: arm.provenance.graph_contract.cache_slots.reduce(
      (sum, slot) => sum + slot.present_outputs.length, 0
    ),
    cache_slots: arm.provenance.graph_contract.cache_slots,
    config: arm.config,
    runtime_metadata: arm.runtime_metadata || null,
    training_lineage: arm.training_lineage || null,
    training_lineage_raw_text: arm.training_lineage_raw_text || null,
    provenance_raw_text: arm.provenance_raw_text || null,
    provenance_bytes: arm.provenance_bytes || null,
    provenance: arm.provenance,
    ...currentDecodeLabels(),
  }));
}

function publishDecodePayload(payload) {
  LAST_DECODE_BENCHMARK = payload;
  if (typeof window !== "undefined") {
    window.__localAgentDecodeBenchmarkResult = payload;
  }
  const resultNode = decodeElement("decode-result-json");
  if (resultNode) resultNode.textContent = JSON.stringify(payload);
  const download = decodeElement("download-decode-benchmark");
  if (download) download.disabled = false;
  return payload;
}

function safeProviderEvidence(provider) {
  if (provider !== "webgpu" && provider !== "wasm") {
    return {
      provider_requested: provider,
      provider_actual: null,
      provider_actual_observation: "invalid provider rejected",
      execution_provider_list: [],
      whole_session_provider_retry: false,
      per_node_placement_verified: false,
      per_node_placement_status: "unknown",
      per_node_fallback_status: "unknown",
    };
  }
  return DECODE_STATE.providerVerification ||
    providerEvidence(provider, DECODE_STATE.sessions.some((item) => !item.error));
}

function makeDecodePayload({
  status,
  provider,
  seed,
  warmups,
  repetitions,
  outputTokens,
  warmupRecords,
  records,
  errors,
}) {
  const evidence = safeProviderEvidence(provider);
  const trained = DECODE_STATE.artifactMode === "trained";
  const singleModel = DECODE_STATE.benchmarkMode === "single";
  const acceptanceMode = DECODE_STATE.acceptanceMode === true;
  const decisionAbis = new Set(
    Array.from(DECODE_STATE.arms.values()).map((arm) => arm.decision_output_abi)
  );
  const decisionAbi = decisionAbis.size === 1 ? [...decisionAbis][0] : null;
  return {
    schema_version: 1,
    benchmark: acceptanceMode
      ? "localagent_single_trained_cached_autoregressive_decode_acceptance_latency"
      : (
        singleModel
          ? "localagent_single_trained_cached_autoregressive_decode_latency"
          : "localagent_matched_cached_autoregressive_decode_latency"
      ),
    status,
    created_at: new Date().toISOString(),
    ...currentDecodeLabels(),
    warning: trained
      ? DECODE_TRAINED_LABEL
      : "UNTRAINED RANDOM WEIGHTS — LATENCY ONLY; NOT A CAPABILITY OR QUALITY ARTIFACT.",
    metadata: {
      protocol_version: "cached-decode-latency-0.2",
      benchmark_session_id: DECODE_STATE.benchmarkSessionId,
      run_id: DECODE_STATE.runId,
      run_challenge: DECODE_STATE.runChallenge,
      external_machine_condition_sha256:
        DECODE_STATE.externalMachineConditionSha256,
      evidence_scope: acceptanceMode ? { ...DECODE_EVIDENCE_SCOPE } : null,
      acceptance_acquisition_roots:
        DECODE_STATE.acceptanceAcquisitionRoots == null
          ? null
          : { ...DECODE_STATE.acceptanceAcquisitionRoots },
      harness_identity: DECODE_STATE.harnessIdentity,
      benchmark_mode: DECODE_STATE.benchmarkMode || "unverified",
      acceptance_mode: acceptanceMode,
      acceptance_wrapper_manifest_sha256:
        acceptanceMode ? DECODE_STATE.acceptanceRootSha256 : null,
      acceptance_protocol: acceptanceMode
        ? {
          ...DECODE_ACCEPTANCE_PROTOCOL,
          context_lengths: [...DECODE_ACCEPTANCE_PROTOCOL.context_lengths],
          exact: true,
        }
        : null,
      artifact_mode: DECODE_STATE.artifactMode || "unverified",
      benchmark_label: trained ? DECODE_TRAINED_LABEL : null,
      action_capability_evaluated: false,
      action_capability_claimed: false,
      estimand: "prefill_and_iterative_cache_bearing_graph_latency",
      ttft_boundary:
        decisionAbi === DECODE_DECISION_ABI_LEGACY
          ? "immediately before prefill session.run through validated legacy CPU next_token"
          : (
            "immediately before prefill session.run through validated CPU logits argmax "
            + "availability"
          ),
      tpot_boundary:
        decisionAbi === DECODE_DECISION_ABI_LEGACY
          ? (
            "wall time from the first validated legacy CPU next_token through the final "
            + "iterative legacy CPU next_token divided by output_tokens_minus_one"
          )
          : (
            "wall time from the first validated CPU logits argmax through the final iterative "
            + "validated CPU logits argmax divided by output_tokens_minus_one"
          ),
      prefill_ms_boundary:
        "immediately before prefill session.run through promise resolution",
      decode_inference_ms_boundary:
        "sum of immediately-before decode session.run through promise resolution",
      model_decode_tokens_per_second_definition:
        "(output_tokens_minus_one)*1000/summed_decode_inference_ms",
      excluded_from_latency:
        "manifest/config/provenance/graph fetch and hashing, session creation, deterministic " +
        "prompt construction, summary rendering, download serialization, and all quality evaluation",
      provider: evidence,
      required_webgpu_provider_verification: singleModel,
      ort_script_url:
        DECODE_STATE.harnessIdentity?.ort?.javascript?.url ||
        new URL(DECODE_ORT_SCRIPT_PATH, document.baseURI).href,
      ort_wasm_url:
        DECODE_STATE.harnessIdentity?.ort?.wasm?.url ||
        new URL(DECODE_ORT_WASM_PATH, document.baseURI).href,
      ort_version_pin: DECODE_ORT_VERSION,
      ...ortVersionEvidence(),
      cross_origin_isolated: globalThis.crossOriginIsolated ?? null,
      shared_array_buffer_available: typeof globalThis.SharedArrayBuffer !== "undefined",
      ort_wasm_num_threads: globalThis.ort?.env?.wasm?.numThreads ?? null,
      browser: browserRuntimeMetadata(),
      gpu: gpuRuntimeMetadata(),
      user_agent: globalThis.navigator?.userAgent || null,
      language: globalThis.navigator?.language || null,
      hardware_concurrency: globalThis.navigator?.hardwareConcurrency || null,
      device_memory_gb: globalThis.navigator?.deviceMemory || null,
      timer: "performance.now",
      concurrency: 1,
      tab_visibility_required: true,
      run_once_reload_required: true,
      manifest_url: DECODE_STATE.manifestUrl,
      manifest_sha256: DECODE_STATE.manifestSha256,
      manifest_raw_text: DECODE_STATE.manifestRawText,
      manifest: DECODE_STATE.manifest,
      verified_identities: {
        manifest_sha256: DECODE_STATE.manifestSha256,
        artifacts: DECODE_STATE.artifacts.map((artifact) => ({
          artifact_kind: artifact.artifact_kind,
          artifact_id: artifact.artifact_id,
          relative_path: artifact.relative_path,
          expected_sha256: artifact.expected_sha256 || null,
          actual_sha256: artifact.actual_sha256,
          expected_bytes: artifact.expected_bytes || null,
          bytes: artifact.bytes,
          hash_verified: artifact.hash_verified,
          bytes_verified: artifact.bytes_verified ?? null,
        })),
        arms: publicArmMetadata().map((arm) => ({
          id: arm.id,
          pair_role: arm.pair_role,
          config_source_sha256: arm.config_source_sha256,
          config_canonical_sha256: arm.config_canonical_sha256,
          provenance_sha256: arm.provenance_sha256,
          prefill_sha256: arm.prefill_sha256,
          decode_sha256: arm.decode_sha256,
          checkpoint_sha256:
            arm.provenance.weights?.checkpoint_sha256 || null,
          checkpoint_step:
            arm.provenance.weights?.checkpoint_step ?? null,
          tokens_seen:
            arm.provenance.weights?.tokens_seen ?? null,
          tokenizer_sha256:
            arm.provenance.tokenizer?.sha256 || null,
        })),
      },
      context_lengths: [...DECODE_CONTEXT_LENGTHS],
      prompt_lengths_tokens: [...DECODE_CONTEXT_LENGTHS],
      context_condition: "exact_prefill_input_tensor_sequence_length",
      input_semantics: "deterministic_pretokenized_ids",
      input_fixture_contract: "ids[i]=(131*i+17) mod vocab_size",
      tokenizer_asset: trained ? {
        loaded_by_benchmark: false,
        input_ids_are_pretokenized: true,
        provenance_pin: DECODE_STATE.manifest?.tokenizer || null,
      } : null,
      output_tokens_per_condition: outputTokens,
      reported_percentiles: ["p50", "p95"],
      decision_output_abi: decisionAbi,
      greedy_selection:
        decisionAbi === DECODE_DECISION_ABI_LEGACY
          ? "legacy_exported_next_token_argmax"
          : "validated_logits_argmax",
      graph_pass_contract: {
        prefill_per_condition: 1,
        decode_per_condition: Math.max(0, outputTokens - 1),
        total_per_condition: outputTokens,
        first_token_source:
          decisionAbi === DECODE_DECISION_ABI_LEGACY
            ? "legacy prefill.next_token"
            : "prefill.logits argmax; next_token compatibility cross-check",
        remaining_token_source:
          decisionAbi === DECODE_DECISION_ABI_LEGACY
            ? "legacy decode.next_token"
            : "decode.logits argmax; next_token compatibility cross-check",
      },
      cache_contract: {
        enabled: true,
        webgpu_cache_residency: "gpu-buffer",
        wasm_cache_residency: "cpu",
        next_token_residency: "cpu",
        logits_residency:
          decisionAbi === DECODE_DECISION_ABI_LEGACY ? null : "cpu",
        token_selection_source:
          decisionAbi === DECODE_DECISION_ABI_LEGACY
            ? "legacy_exported_next_token"
            : "validated_logits_argmax",
        update_strategy:
          "present_outputs_rebound_directly_as_past_inputs_without_cpu_materialization",
        cache_data_read_to_javascript: false,
        superseded_and_final_cache_disposal_attempted: true,
      },
      arm_count: DECODE_STATE.arms.size,
      arms: publicArmMetadata(),
      case_order_seed: seed,
      session_order_seed:
        DECODE_STATE.benchmarkMode === "single"
          ? `${DECODE_DEFAULT_SEED}:single:session-create`
          : `${DECODE_DEFAULT_SEED}:session-create`,
      warmups_per_condition: warmups,
      measured_repetitions_per_condition: repetitions,
      warmups_excluded_from_summary: true,
      page_to_ready_ms: DECODE_STATE.readyAtMs != null &&
        globalThis.window?.__localAgentDecodeBenchmarkStart != null
        ? DECODE_STATE.readyAtMs - window.__localAgentDecodeBenchmarkStart
        : null,
    },
    artifact_verification_records: [...DECODE_STATE.artifacts],
    session_records: [...DECODE_STATE.sessions],
    input_preparation_record: DECODE_STATE.inputPreparationRecord,
    inputs: Array.from(DECODE_STATE.inputs.values()).map((input) => ({
      ...input,
      tensor: undefined,
    })),
    warmup_records: warmupRecords,
    records,
    summary: records.length
      ? summarizeDecodeRecords(records, DECODE_STATE.artifactMode || "random")
      : null,
    failures: [
      ...warmupRecords.filter((record) => !record.run_ok),
      ...records.filter((record) => !record.run_ok),
    ].map((record) => ({
      phase: record.phase,
      arm_id: record.arm_id,
      input_tokens: record.input_tokens,
      repetition: record.repetition,
      graph_pass_counts: record.graph_pass_counts,
      actual_output_tokens: record.actual_output_tokens,
      error: record.error,
    })),
    errors,
  };
}

async function runDecodeBenchmark() {
  if (DECODE_RUN_STARTED) return LAST_DECODE_BENCHMARK;
  if (DECODE_STATE.runId == null) {
    DECODE_STATE.runId = newBenchmarkIdentity("benchmark run");
  }
  const runButton = decodeElement("start-decode-benchmark");
  let provider =
    window.__localAgentDecodeRequestedProvider ||
    new URLSearchParams(window.location.search).get("backend") ||
    "webgpu";
  const outputTokens = Number.parseInt(
    decodeElement("decode-output-tokens")?.value || `${DECODE_DEFAULT_OUTPUT_TOKENS}`,
    10
  );
  const warmups = Number.parseInt(decodeElement("decode-warmups")?.value || "3", 10);
  const repetitions = Number.parseInt(
    decodeElement("decode-repetitions")?.value || "30", 10
  );
  const seed = decodeElement("decode-seed")?.value.trim() || DECODE_DEFAULT_SEED;
  const warmupRecords = [];
  const records = [];
  const errors = [];
  try {
    provider = requireExplicitProvider(provider);
    validateDecodeProtocolSettings(
      { outputTokens, warmups, repetitions, seed },
      DECODE_STATE.acceptanceMode
    );
    if (
      Array.from(DECODE_STATE.arms.values()).some(
        (arm) => Math.max(...DECODE_CONTEXT_LENGTHS) + outputTokens > arm.config.max_seq_len
      )
    ) {
      throw new Error("Output-token setting would exceed a model's maximum sequence length.");
    }
    const armIds = Array.from(DECODE_STATE.arms.keys());
    const expectedArmCount = DECODE_STATE.benchmarkMode === "single" ? 1 : 2;
    if (armIds.length !== expectedArmCount) {
      throw new Error(
        `${expectedArmCount} verified prefill/decode session pair(s) must be ready.`
      );
    }
    if (
      DECODE_STATE.acceptanceMode &&
      Array.from(DECODE_STATE.arms.values()).some(
        (arm) => arm.decision_output_abi !== DECODE_DECISION_ABI_LOGITS
      )
    ) {
      throw new Error("Acceptance mode requires final logits plus next-token cross-check ABI.");
    }
    if (
      DECODE_STATE.benchmarkMode === "single" &&
      DECODE_STATE.providerVerification?.required_verification_passed !== true
    ) {
      throw new Error("Required single-model WebGPU provider verification is missing.");
    }
    if (document.visibilityState !== "visible") {
      throw new Error("Run cannot start while the benchmark tab is hidden.");
    }
    DECODE_RUN_STARTED = true;
    if (runButton) runButton.disabled = true;

    const warmSchedule = buildDecodeSchedule(
      armIds,
      DECODE_CONTEXT_LENGTHS,
      warmups,
      seed,
      "warmup",
      DECODE_STATE.benchmarkMode || "matched"
    );
    for (let index = 0; index < warmSchedule.length; index++) {
      if (document.visibilityState !== "visible") {
        throw new Error("Run stopped because the benchmark tab became hidden.");
      }
      const condition = warmSchedule[index];
      setDecodeProgress(
        `Warmup ${index + 1}/${warmSchedule.length}: ` +
        `${condition.arm_id} @ ${condition.input_tokens} + ${outputTokens} outputs`
      );
      const record = await runDecodeCondition(
        "warmup", condition, provider, outputTokens, index
      );
      warmupRecords.push(record);
      validateAcceptanceDisposalRecord(record);
      if (!record.run_ok) {
        throw new Error(
          `Warmup failed for ${condition.arm_id} @ ${condition.input_tokens}: ` +
          `${record.error?.message || "unknown error"}`
        );
      }
    }

    const measuredSchedule = buildDecodeSchedule(
      armIds,
      DECODE_CONTEXT_LENGTHS,
      repetitions,
      seed,
      "measured",
      DECODE_STATE.benchmarkMode || "matched"
    );
    for (let index = 0; index < measuredSchedule.length; index++) {
      if (document.visibilityState !== "visible") {
        throw new Error("Run stopped because the benchmark tab became hidden.");
      }
      const condition = measuredSchedule[index];
      setDecodeProgress(
        `Measured ${index + 1}/${measuredSchedule.length}: ` +
        `${condition.arm_id} @ ${condition.input_tokens} + ${outputTokens} outputs`
      );
      const record = await runDecodeCondition(
        "measured", condition, provider, outputTokens, index
      );
      records.push(record);
      validateAcceptanceDisposalRecord(record);
      if (!record.run_ok) errors.push(record.error || { message: "unknown condition failure" });
    }
    const status = errors.length ? "failed" : "complete";
    const payload = makeDecodePayload({
      status,
      provider,
      seed,
      warmups,
      repetitions,
      outputTokens,
      warmupRecords,
      records,
      errors,
    });
    publishDecodePayload(payload);
    renderDecodeSummary(payload);
    if (status === "complete") {
      setDecodeStatus("ready", "Cached-decode latency run complete.", provider);
      setDecodeProgress(
        `Complete: ${records.length} measured conditions; raw records are globally available.`
      );
    } else {
      setDecodeStatus(
        "error",
        `Run completed with ${errors.length} failed measured condition(s).`,
        provider
      );
      setDecodeProgress("Partial records and failure details are available for download.");
    }
    return payload;
  } catch (error) {
    errors.push(errorDetail(error));
    const payload = makeDecodePayload({
      status: "failed",
      provider,
      seed,
      warmups,
      repetitions,
      outputTokens,
      warmupRecords,
      records,
      errors,
    });
    publishDecodePayload(payload);
    renderDecodeSummary(payload);
    setDecodeStatus("error", `Benchmark failed: ${error.message}`, provider);
    setDecodeProgress(
      "Partial raw records and error details are available; reload before retrying."
    );
    return payload;
  }
}

function downloadDecodeBenchmark() {
  if (!LAST_DECODE_BENCHMARK) return;
  const blob = new Blob(
    [JSON.stringify(LAST_DECODE_BENCHMARK, null, 2)],
    { type: "application/json" }
  );
  const anchor = document.createElement("a");
  anchor.href = URL.createObjectURL(blob);
  anchor.download =
    `localagent-cached-decode-${LAST_DECODE_BENCHMARK.created_at.replace(/[:.]/g, "-")}.json`;
  anchor.click();
  URL.revokeObjectURL(anchor.href);
}

async function initializeDecodeBenchmark() {
  const runButton = decodeElement("start-decode-benchmark");
  const downloadButton = decodeElement("download-decode-benchmark");
  if (runButton) runButton.addEventListener("click", runDecodeBenchmark);
  if (downloadButton) downloadButton.addEventListener("click", downloadDecodeBenchmark);
  let provider =
    window.__localAgentDecodeRequestedProvider ||
    new URLSearchParams(window.location.search).get("backend") ||
    "webgpu";
  try {
    provider = requireExplicitProvider(provider);
    if (DECODE_STATE.benchmarkSessionId == null) {
      DECODE_STATE.benchmarkSessionId = newBenchmarkIdentity("benchmark session");
    }
    const benchmarkMode = requestedDecodeBenchmarkMode();
    const acceptanceMode = requestedDecodeAcceptanceMode();
    const acceptanceEvidence = requestedDecodeAcceptanceEvidence();
    DECODE_STATE.acceptanceMode = acceptanceMode;
    DECODE_STATE.runChallenge = acceptanceEvidence?.run_challenge || null;
    DECODE_STATE.externalMachineConditionSha256 =
      acceptanceEvidence?.machine_condition_sha256 || null;
    DECODE_STATE.acceptanceAcquisitionRoots = acceptanceEvidence
      ? {
        html_sha256: acceptanceEvidence.html_sha256,
        javascript_sha256: acceptanceEvidence.javascript_sha256,
        ort_javascript_sha256: acceptanceEvidence.ort_javascript_sha256,
        ort_wasm_sha256: acceptanceEvidence.ort_wasm_sha256,
      }
      : null;
    DECODE_STATE.harnessIdentity = await captureDecodeHarnessIdentity(
      DECODE_STATE.acceptanceAcquisitionRoots
    );
    configureAcceptanceProtocolSettings(acceptanceMode);
    const graphCount = benchmarkMode === "single" ? "two" : "four";
    setDecodeStatus(
      "loading",
      `Hashing configs, provenance, and ${graphCount} ONNX graphs…`,
      provider
    );
    setDecodeProgress(
      "No ONNX session will be created until every selected artifact passes byte and SHA-256 checks."
    );
    await loadDecodeBundle(provider);
    if (runButton) runButton.disabled = false;
    setDecodeStatus(
      "ready",
      benchmarkMode === "single"
        ? "Verified one trained prefill/decode graph pair and WebGPU provider. Ready to run."
        : "Verified two prefill/decode graph pairs. Ready for one latency-only run.",
      provider
    );
    setDecodeProgress(
      "Ready. Run once; reload the page before collecting another repetition set."
    );
  } catch (error) {
    const payload = makeDecodePayload({
      status: "failed_to_initialize",
      provider,
      seed: DECODE_DEFAULT_SEED,
      warmups: DECODE_MIN_WARMUPS,
      repetitions: DECODE_MIN_REPETITIONS,
      outputTokens: DECODE_DEFAULT_OUTPUT_TOKENS,
      warmupRecords: [],
      records: [],
      errors: [errorDetail(error)],
    });
    publishDecodePayload(payload);
    setDecodeStatus("error", `Initialization failed closed: ${error.message}`, provider);
    setDecodeProgress(
      "No inference ran. Verification/session evidence is available in the raw payload."
    );
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    DECODE_ACCEPTANCE_PROTOCOL,
    DECODE_CONTEXT_LENGTHS,
    DECODE_MIN_WARMUPS,
    DECODE_MIN_REPETITIONS,
    buildDecodeSchedule,
    captureDecodeHarnessIdentity,
    canonicalJson,
    decodeDecisionAbi,
    decodeLabelsForMode,
    decodeManifestMode,
    decodeSessionOptions,
    dtypeBytes,
    latencySummary,
    modelConfigCanonicalJson,
    normalizeDecodeAcceptanceMode,
    normalizeDecodeBenchmarkMode,
    outputLocationsForContract,
    parsePositiveInteger,
    requireExplicitProvider,
    requestedDecodeAcceptanceRootSha256,
    requestedDecodeAcceptanceEvidence,
    singleDecodeProvenanceContext,
    summarizeDecodeRecords,
    tensorLogicalBytes,
    validateCacheSlot,
    validateAcceptanceTrainingLineage,
    validateAcceptanceDisposalRecord,
    validateCachedRuntimeMetadata,
    validateDecodeDecisionOutputs,
    validateDecodeManifest,
    validateDecodeProvenance,
    validateDecodeProtocolSettings,
    validateLogitsTensor,
    validateRequiredWebGpuEvidence,
    validateSingleDecodeManifest,
    validateSingleDecodeProvenance,
    validateSingleDecodeQuery,
    verifyOrtVersionPin,
  };
}

if (
  typeof window !== "undefined" &&
  typeof document !== "undefined" &&
  !window.__localAgentSkipInit
) {
  window.addEventListener("DOMContentLoaded", initializeDecodeBenchmark);
}
