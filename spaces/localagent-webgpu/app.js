/* LocalAgent — in-browser tool calling on onnxruntime-web (WebGPU + explicit WASM control).
 *
 * The transformer forward pass runs as an ONNX graph emitting `hidden` (and optionally `logits`).
 * The GENERABLE
 * dispatch (route head -> dense two-tower selector -> pointer-copy args) is ported here from the
 * Python pipeline. Bundle contract (see localagent.inference.export):
 *   model.fp16.onnx      inputs: input_ids[int64, 1xT]  outputs: logits[1,T,V], hidden[1,T,d]
 *   dispatch_heads.json  { route_head:{weight:[5][d],bias:[5],routes:[5],stop_index},
 *                          dense_selector:{q_proj_weight:[p][d],q_proj_bias:[p],proj:p,
 *                                          tool_matrix:[N][p],tool_names:[N],normalize_query} }
 *                          retrieval_selector:{dim,tool_matrix:[N][dim],tool_names,tool_routes} }
 *   heads.json           { pointer_head:{arg_idx,arg_emb,start_W,end_W}, ... }   (args copy)
 *   meta.json            { model_file, action_model_file, d_model, markers, tools } (63 tools)
 *
 * Selection is NOT a fixed-N classifier: the dense selector scores every tool by its description
 * embedding, so adding/removing a tool is adding/removing a tool_matrix row.  An explicit
 * retrieval_selector ablations are available with ?selector=retrieval and
 * ?selector=retrieval_then_dense; both are reported separately from the default dense selector.
 */

const ACTION_POLICIES = Object.freeze({
  STRUCTURED: "structured_one_forward",
  RAW_AR: "raw_autoregressive_json",
  CONSTRAINED_AR: "grounded_candidate_trie_autoregressive",
});
let MODEL_URL = "model.fp16.onnx"; // Compatibility alias: structured/action graph.
let LOGITS_MODEL_URL = "model.fp16.onnx";
let SESSION = null; // Compatibility alias: structured/action session.
let LOGITS_SESSION = null;
let CACHED_PREFILL_SESSION = null;
let CACHED_DECODE_SESSION = null;
let CACHED_DECODE_BUNDLE = null;
let HEADS = null;
let META = null;
let DISPATCH = null;
let TOKENIZER = null;
let BUNDLE_MANIFEST = null;
let BACKEND = "wasm";
const BUNDLE_LOAD_TIMING = {};
const BUNDLE_ASSET_CACHE = new Map();
const BUNDLE_ASSET_EVIDENCE = new Map();
const MODEL_ARTIFACT_CACHE = new Map();
const MODEL_BYTE_EVIDENCE = new Map();
const OPENAI_FULL_CATALOG_V1 = "openai_full_catalog_v1";
const BPE_EOS_MARKER = "<|end|>";
const TOOL_CATALOG_OPEN = "<|tool_catalog|>";
const TOOL_CATALOG_CLOSE = "</|tool_catalog|>";
const CACHED_DECODE_STRATEGY = "prefill_then_kv_cached_decode";
const BENCHMARK_GRADE = window.__localAgentBenchmarkGrade === true ||
  Number.isFinite(window.__localAgentBenchmarkStart) ||
  Number.isFinite(window.__localAgentBrowserTasksStart);
const REQUESTED_BACKEND = window.__localAgentRequestedBackend ||
  new URLSearchParams(window.location.search).get("backend") || "auto";
const MOBILE_LEXICAL_GUARD = (() => {
  const value = new URLSearchParams(window.location.search).get("mobile_guard");
  return value !== "0" && value !== "false" && value !== "off";
})();
window.__localAgentMobileLexicalGuardEnabled = MOBILE_LEXICAL_GUARD;
// Diagnostic-only switch: production/demo URLs keep this guard enabled by default.  A native
// control run can disable it to measure the learned selector rather than the explicit URL safety
// adapter; no external navigation is ever executed by the capability harness.
const PRODUCTIVITY_LEXICAL_GUARD = (() => {
  const value = new URLSearchParams(window.location.search).get("productivity_guard");
  return value !== "0" && value !== "false" && value !== "off";
})();
window.__localAgentProductivityLexicalGuardEnabled = PRODUCTIVITY_LEXICAL_GUARD;
const URL_LEXICAL_GUARD = (() => {
  const value = new URLSearchParams(window.location.search).get("url_guard");
  return value !== "0" && value !== "false" && value !== "off";
})();
window.__localAgentUrlLexicalGuardEnabled = URL_LEXICAL_GUARD;
const REQUESTED_SELECTOR = new URLSearchParams(window.location.search).get("selector") || "dense";

function sessionOptions(provider) {
  if (provider !== "webgpu" && provider !== "wasm") {
    throw new Error(`Unknown execution provider '${provider}'.`);
  }
  return { executionProviders: [provider] };
}

function manifestArtifactFor(fileName, manifest = BUNDLE_MANIFEST) {
  if (!fileName || !manifest?.artifacts) return null;
  const matches = Object.values(manifest.artifacts)
    .filter((artifact) => artifact?.file === fileName);
  if (matches.length > 1) {
    throw new Error(`Bundle manifest contains duplicate entries for ${fileName}.`);
  }
  return matches[0] || null;
}

async function sha256Bytes(bytes) {
  if (!globalThis.crypto?.subtle) {
    throw new Error("Web Crypto SHA-256 is unavailable; model bytes cannot be verified.");
  }
  const digest = await globalThis.crypto.subtle.digest(
    "SHA-256",
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
  );
  return Array.from(
    new Uint8Array(digest),
    (value) => value.toString(16).padStart(2, "0")
  ).join("");
}

async function verifyArtifactBytesAgainstManifest(
  artifactUrl,
  bytes,
  manifest,
  manifestRequired = false
) {
  const sha256 = await sha256Bytes(bytes);
  const expected = manifestArtifactFor(artifactUrl, manifest);
  const manifestVerified = expected != null &&
    expected.sha256 === sha256 &&
    expected.bytes === bytes.byteLength;
  if (manifestRequired && !expected) {
    throw new Error(`Bundle manifest has no entry for fetched artifact ${artifactUrl}.`);
  }
  if (expected && expected.sha256 !== sha256) {
    throw new Error(
      `SHA-256 mismatch for ${artifactUrl}: fetched ${sha256}, manifest ${expected.sha256}.`
    );
  }
  if (expected && expected.bytes !== bytes.byteLength) {
    throw new Error(
      `Byte-length mismatch for ${artifactUrl}: fetched ${bytes.byteLength}, ` +
      `manifest ${expected.bytes}.`
    );
  }
  return Object.freeze({
    file: artifactUrl,
    bytes: bytes.byteLength,
    sha256,
    manifest_bytes: expected?.bytes ?? null,
    manifest_sha256: expected?.sha256 ?? null,
    manifest_verified: manifestVerified,
    fetch_contract: "browser_fetch_arraybuffer_sha256_before_parse_or_session",
    verification_scope: "exact_fetched_response_body_bytes",
  });
}

async function verifyModelBytesAgainstManifest(
  modelUrl,
  bytes,
  manifest,
  manifestRequired = false
) {
  const evidence = await verifyArtifactBytesAgainstManifest(
    modelUrl,
    bytes,
    manifest,
    manifestRequired
  );
  return Object.freeze({
    ...evidence,
    session_source: "in_memory_verified_bytes",
  });
}

async function verifyPinnedArtifactBytes(artifactUrl, bytes, expectedIdentity) {
  if (
    !Number.isInteger(expectedIdentity?.bytes) ||
    expectedIdentity.bytes < 1 ||
    !/^[0-9a-f]{64}$/.test(expectedIdentity?.sha256 || "")
  ) {
    throw new Error(`Pinned identity for ${artifactUrl} is invalid.`);
  }
  const sha256 = await sha256Bytes(bytes);
  if (bytes.byteLength !== expectedIdentity.bytes) {
    throw new Error(
      `Pinned byte-length mismatch for ${artifactUrl}: fetched ${bytes.byteLength}, ` +
      `expected ${expectedIdentity.bytes}.`
    );
  }
  if (sha256 !== expectedIdentity.sha256) {
    throw new Error(
      `Pinned SHA-256 mismatch for ${artifactUrl}: fetched ${sha256}, ` +
      `expected ${expectedIdentity.sha256}.`
    );
  }
  return Object.freeze({
    file: artifactUrl,
    bytes: bytes.byteLength,
    sha256,
    expected_bytes: expectedIdentity.bytes,
    expected_sha256: expectedIdentity.sha256,
    identity_source: expectedIdentity.identity_source || null,
    identity_verified: true,
    verification_scope: "exact_fetched_response_body_bytes_before_parse",
  });
}

function validateBenchmarkBundleContract(meta = META, manifest = BUNDLE_MANIFEST) {
  if (!manifest) {
    throw new Error("Benchmark-grade runs require bundle-manifest.json.");
  }
  if (
    !Number.isInteger(manifest.schema_version) ||
    manifest.schema_version < 3
  ) {
    throw new Error("Benchmark-grade runs require bundle manifest schema_version >= 3.");
  }
  if (
    manifest.parity_gate?.hard_gate !== true ||
    manifest.parity_gate?.passed !== true
  ) {
    throw new Error("Benchmark-grade runs require a passed hard export parity gate.");
  }
  if (!meta.model_file) {
    throw new Error("Benchmark-grade runs require meta.json model_file.");
  }
  if (!meta.action_model_file) {
    throw new Error(
      "Benchmark-grade structured runs require a hidden-only action_model_file; " +
      "re-export with action_only=True."
    );
  }
  if (meta.action_model_file === meta.model_file) {
    throw new Error("The benchmark action graph must be distinct from the logits graph.");
  }
  const graphContracts = [
    [meta.model_file, ["logits", "hidden"]],
    [meta.action_model_file, ["hidden"]],
  ];
  for (const [fileName, expectedOutputs] of graphContracts) {
    const artifact = manifestArtifactFor(fileName, manifest);
    if (!artifact) {
      throw new Error(`Bundle manifest does not bind required model artifact ${fileName}.`);
    }
    if (
      !Number.isInteger(artifact.bytes) ||
      artifact.bytes < 1 ||
      !/^[0-9a-f]{64}$/.test(artifact.sha256)
    ) {
      throw new Error(`Bundle manifest has invalid byte/hash evidence for ${fileName}.`);
    }
    const parity = manifest.parity_gate.results?.[fileName];
    if (parity?.passed !== true) {
      throw new Error(`Bundle manifest lacks passed graph parity for ${fileName}.`);
    }
    if (
      !Array.isArray(parity.expected_outputs) ||
      parity.expected_outputs.length !== expectedOutputs.length ||
      parity.expected_outputs.some((name, index) => name !== expectedOutputs[index])
    ) {
      throw new Error(`Bundle manifest has the wrong output contract for ${fileName}.`);
    }
    if (
      parity.artifact?.sha256 !== artifact.sha256 ||
      parity.artifact?.bytes !== artifact.bytes
    ) {
      throw new Error(`Bundle manifest parity is not content-bound to ${fileName}.`);
    }
    for (const outputName of expectedOutputs) {
      const delta = parity.max_abs_diff_by_output?.[outputName];
      const threshold = parity.threshold_max_abs_diff_by_output?.[outputName];
      if (
        !Number.isFinite(delta) ||
        !Number.isFinite(threshold) ||
        threshold < 0 ||
        delta > threshold
      ) {
        throw new Error(`Bundle manifest parity evidence is invalid for ${fileName}.`);
      }
    }
  }
  return true;
}

async function fetchBundleArtifactBytes(artifactUrl, manifestRequired = BENCHMARK_GRADE) {
  if (BUNDLE_ASSET_CACHE.has(artifactUrl)) return BUNDLE_ASSET_CACHE.get(artifactUrl);
  const pending = (async () => {
    const response = await fetch(artifactUrl);
    if (!response.ok) {
      throw new Error(`Missing deploy artifact ${artifactUrl} (HTTP ${response.status}).`);
    }
    const contentType = response.headers.get("content-type") || "";
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (manifestRequired && !globalThis.crypto?.subtle) {
      throw new Error(
        "Benchmark-grade runs require Web Crypto SHA-256 before parsing or session creation."
      );
    }
    const evidence = globalThis.crypto?.subtle
      ? await verifyArtifactBytesAgainstManifest(
        artifactUrl,
        bytes,
        BUNDLE_MANIFEST,
        manifestRequired
      )
      : Object.freeze({
        file: artifactUrl,
        bytes: bytes.byteLength,
        sha256: null,
        manifest_bytes: manifestArtifactFor(artifactUrl)?.bytes ?? null,
        manifest_sha256: manifestArtifactFor(artifactUrl)?.sha256 ?? null,
        manifest_verified: false,
        fetch_contract: "browser_fetch_arraybuffer_without_webcrypto_demo_only",
        verification_scope: "unverified_demo_only",
      });
    BUNDLE_ASSET_EVIDENCE.set(artifactUrl, evidence);
    return { bytes, contentType, evidence };
  })();
  BUNDLE_ASSET_CACHE.set(artifactUrl, pending);
  try {
    return await pending;
  } catch (error) {
    BUNDLE_ASSET_CACHE.delete(artifactUrl);
    throw error;
  }
}

async function fetchJsonArtifact(artifactUrl, manifestRequired = BENCHMARK_GRADE) {
  const artifact = await fetchBundleArtifactBytes(artifactUrl, manifestRequired);
  if (!artifact.contentType.includes("json")) {
    throw new Error(
      `Deploy artifact ${artifactUrl} returned ` +
      `${artifact.contentType || "an unknown content type"}, expected JSON.`
    );
  }
  const text = new TextDecoder("utf-8", { fatal: true }).decode(artifact.bytes);
  let value;
  try {
    value = JSON.parse(text);
  } catch (error) {
    throw new Error(`Deploy artifact ${artifactUrl} is not valid JSON: ${error.message}`);
  }
  return { ...artifact, text, value };
}

async function fetchPinnedJsonArtifact(artifactUrl, expectedIdentity) {
  const response = await fetch(artifactUrl);
  if (!response.ok) {
    throw new Error(`Missing pinned suite artifact ${artifactUrl} (HTTP ${response.status}).`);
  }
  const contentType = response.headers.get("content-type") || "";
  const bytes = new Uint8Array(await response.arrayBuffer());
  const evidence = await verifyPinnedArtifactBytes(artifactUrl, bytes, expectedIdentity);
  if (!contentType.includes("json")) {
    throw new Error(
      `Pinned suite artifact ${artifactUrl} returned ` +
      `${contentType || "an unknown content type"}, expected JSON.`
    );
  }
  const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  let value;
  try {
    value = JSON.parse(text);
  } catch (error) {
    throw new Error(`Pinned suite artifact ${artifactUrl} is not valid JSON: ${error.message}`);
  }
  return { bytes, contentType, evidence, text, value };
}

function bundleArtifactEvidence(artifactUrl) {
  return BUNDLE_ASSET_EVIDENCE.get(artifactUrl) || null;
}

function bundleManifestByteEvidence() {
  const evidence = bundleArtifactEvidence("bundle-manifest.json");
  if (!evidence) return null;
  return Object.freeze({
    ...evidence,
    role: "parsed_bundle_manifest_trust_anchor",
    external_expected_identity: null,
    external_verification_status: "not_available_self_describing_manifest",
  });
}

function cachedDecodeBundleEvidence() {
  return CACHED_DECODE_BUNDLE?.evidence || null;
}

function runtimeAssetEvidence() {
  const evidence = {
    heads_json: bundleArtifactEvidence("heads.json"),
    meta_json: bundleArtifactEvidence("meta.json"),
    dispatch_heads_json: bundleArtifactEvidence("dispatch_heads.json"),
    tokenizer: META?.tokenizer_file
      ? bundleArtifactEvidence(META.tokenizer_file)
      : null,
  };
  if (BENCHMARK_GRADE) {
    for (const [name, value] of Object.entries(evidence)) {
      if (name === "tokenizer" && !META?.tokenizer_file) continue;
      if (!value?.manifest_verified) {
        throw new Error(`Benchmark runtime asset ${name} lacks verified fetched-byte evidence.`);
      }
    }
  }
  return Object.freeze(evidence);
}

async function fetchModelArtifact(modelUrl) {
  if (MODEL_ARTIFACT_CACHE.has(modelUrl)) return MODEL_ARTIFACT_CACHE.get(modelUrl);
  const pending = (async () => {
    const artifact = await fetchBundleArtifactBytes(modelUrl, BENCHMARK_GRADE);
    const evidence = Object.freeze({
      ...artifact.evidence,
      session_source: artifact.evidence.manifest_verified
        ? "in_memory_verified_bytes"
        : "in_memory_unverified_bytes_demo_only",
    });
    MODEL_BYTE_EVIDENCE.set(modelUrl, evidence);
    return { bytes: artifact.bytes, evidence };
  })();
  MODEL_ARTIFACT_CACHE.set(modelUrl, pending);
  try {
    return await pending;
  } catch (error) {
    MODEL_ARTIFACT_CACHE.delete(modelUrl);
    throw error;
  }
}

function modelArtifactEvidence(modelUrl) {
  return MODEL_BYTE_EVIDENCE.get(modelUrl) || null;
}

function validateSessionOutputs(session, modelUrl, expectedOutputs) {
  if (!expectedOutputs) return;
  const actual = Array.from(session.outputNames || []);
  if (
    actual.length !== expectedOutputs.length ||
    actual.some((name, index) => name !== expectedOutputs[index])
  ) {
    throw new Error(
      `Model ${modelUrl} output contract mismatch: expected ` +
      `${JSON.stringify(expectedOutputs)}, got ${JSON.stringify(actual)}.`
    );
  }
}

function exactJsonEqual(left, right) {
  return canonicalActionJson(left) === canonicalActionJson(right);
}

function isLowerSha256(value) {
  return typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
}

function cachedPresentNames(contract) {
  return contract.cache_slots.flatMap((slot) => slot.present_outputs);
}

function cachedPastNames(contract) {
  return contract.cache_slots.flatMap((slot) => slot.past_inputs);
}

function expectedCachedDtype(precision) {
  if (precision === "fp16") return "float16";
  if (precision === "fp32") return "float32";
  throw new Error(`Unsupported cached graph precision '${precision}'.`);
}

function validateCachedIo(entries, names, cacheDtype, shapes, label) {
  if (!Array.isArray(entries) || entries.length !== names.length) {
    throw new Error(`${label} typed I/O does not match its declared names.`);
  }
  entries.forEach((entry, index) => {
    const name = names[index];
    const dtype = name === "input_ids" || name === "next_token"
      ? "int64"
      : cacheDtype;
    if (
      entry?.name !== name ||
      entry.dtype !== dtype ||
      !exactJsonEqual(entry.shape, shapes.get(name))
    ) {
      throw new Error(`${label} tensor ${name} has a stale dtype or shape.`);
    }
  });
}

function validateCachedGraphContract(metadata) {
  if (
    metadata?.artifact_type !== "localagent_cached_autoregressive_onnx" ||
    metadata.schema_version !== 1 ||
    !metadata.model?.config ||
    !Number.isInteger(metadata.vocab_size) ||
    metadata.vocab_size < 1 ||
    metadata.model.config.vocab_size !== metadata.vocab_size
  ) {
    throw new Error("Cached metadata is not a supported LocalAgent autoregressive bundle.");
  }
  const config = metadata.model.config;
  const contract = metadata.graph_contract;
  const precision = metadata.default_precision;
  const cacheDtype = expectedCachedDtype(precision);
  if (
    !contract ||
    !Array.isArray(contract.cache_slots) ||
    !contract.cache_slots.length ||
    contract.decode_token_axis_fixed_one !== true ||
    contract.cache_update_strategy !==
      "attention K/V append one token; short-conv state replaces its fixed-width tail" ||
    contract.prefill_projection !==
      "only the final normalized prompt feature is projected to vocabulary logits"
  ) {
    throw new Error("Cached metadata has an incomplete cache contract.");
  }
  if (
    !Number.isInteger(config.n_layers) ||
    !Number.isInteger(config.n_loops) ||
    !Number.isInteger(config.n_heads) ||
    !Number.isInteger(config.n_kv_heads) ||
    !Number.isInteger(config.d_model) ||
    config.d_model % config.n_heads !== 0 ||
    config.n_heads % config.n_kv_heads !== 0 ||
    !Array.isArray(config.layer_types) ||
    config.layer_types.length !== config.n_layers ||
    contract.cache_slots.length !== config.n_layers * config.n_loops
  ) {
    throw new Error("Cached model config cannot define the declared cache layout.");
  }

  const shapeByName = new Map();
  contract.cache_slots.forEach((slot, index) => {
    const layer = index % config.n_layers;
    const loop = Math.floor(index / config.n_layers);
    const kind = config.layer_types[layer];
    const expectedPast = kind === "attn"
      ? [`past_${index}_key`, `past_${index}_value`]
      : [`past_${index}_conv`];
    const expectedPresent = kind === "attn"
      ? [`present_${index}_key`, `present_${index}_value`]
      : [`present_${index}_conv`];
    const expectedShape = kind === "attn"
      ? ["batch", config.n_kv_heads, "cache_sequence", config.d_model / config.n_heads]
      : ["batch", config.d_model, config.conv_kernel - 1];
    const expectedUpdate = kind === "attn"
      ? "append_one_token_along_axis_2"
      : "replace_with_latest_fixed_width_tail";
    if (
      slot?.slot !== index ||
      slot.loop !== loop ||
      slot.layer !== layer ||
      slot.kind !== kind ||
      !exactJsonEqual(slot.past_inputs, expectedPast) ||
      !exactJsonEqual(slot.present_outputs, expectedPresent) ||
      !exactJsonEqual(slot.shape, expectedShape) ||
      slot.update !== expectedUpdate ||
      slot.dtype_by_precision?.[precision] !== cacheDtype
    ) {
      throw new Error(`Cached cache slot ${index} is stale or inconsistent with model config.`);
    }
    [...expectedPast, ...expectedPresent].forEach((name) => {
      if (shapeByName.has(name)) throw new Error(`Duplicate cached tensor name ${name}.`);
      shapeByName.set(name, expectedShape);
    });
  });

  const presentNames = cachedPresentNames(contract);
  const pastNames = cachedPastNames(contract);
  const graph = contract.graphs?.[precision];
  const prefillInputs = ["input_ids"];
  const outputs = ["next_token", "logits", ...presentNames];
  const decodeInputs = ["input_ids", ...pastNames];
  if (
    graph?.cache_dtype !== cacheDtype ||
    graph.prefill?.file !== `prefill.${precision}.onnx` ||
    graph.decode?.file !== `decode.${precision}.onnx` ||
    !exactJsonEqual(graph.prefill.input_names, prefillInputs) ||
    !exactJsonEqual(graph.prefill.output_names, outputs) ||
    !exactJsonEqual(graph.decode.input_names, decodeInputs) ||
    !exactJsonEqual(graph.decode.output_names, outputs)
  ) {
    throw new Error("Cached prefill/decode names do not match the production ABI.");
  }
  const prefillInputShapes = new Map([["input_ids", ["batch", "prompt_sequence"]]]);
  const decodeInputShapes = new Map([["input_ids", ["batch", 1]], ...shapeByName]);
  const outputShapes = new Map([
    ["next_token", ["batch"]],
    ["logits", ["batch", "vocab_size"]],
    ...presentNames.map((name) => [name, shapeByName.get(name)]),
  ]);
  validateCachedIo(
    graph.prefill.inputs,
    prefillInputs,
    cacheDtype,
    prefillInputShapes,
    `${precision}.prefill.inputs`
  );
  validateCachedIo(
    graph.prefill.outputs,
    outputs,
    cacheDtype,
    outputShapes,
    `${precision}.prefill.outputs`
  );
  validateCachedIo(
    graph.decode.inputs,
    decodeInputs,
    cacheDtype,
    decodeInputShapes,
    `${precision}.decode.inputs`
  );
  validateCachedIo(
    graph.decode.outputs,
    outputs,
    cacheDtype,
    outputShapes,
    `${precision}.decode.outputs`
  );
  if (
    contract.next_token?.name !== "next_token" ||
    contract.next_token?.dtype !== "int64" ||
    !exactJsonEqual(contract.next_token?.shape, ["batch"]) ||
    contract.next_token?.decode !==
      "compatibility argmax over the exported final-token logits" ||
    contract.logits?.name !== "logits" ||
    contract.logits?.description !==
      "unnormalized LM scores for the final input token only" ||
    contract.logits?.dtype_by_precision?.[precision] !== cacheDtype ||
    !exactJsonEqual(contract.logits?.shape, ["batch", metadata.vocab_size])
  ) {
    throw new Error("Cached token/logits metadata does not match the production ABI.");
  }
  const firstAttention = contract.cache_slots.find((slot) => slot.kind === "attn");
  if (
    !firstAttention ||
    contract.decode_position?.caller_position_input !== false ||
    contract.decode_position?.derived_from !== firstAttention.past_inputs[0] ||
    contract.decode_position?.rule !==
      "RoPE position = first attention past-key axis-2 length"
  ) {
    throw new Error("Cached decode position is not derived from its first attention cache.");
  }
  return Object.freeze({
    cacheDtype,
    contract,
    decodeFile: graph.decode.file,
    decodeInputs,
    outputs,
    pastNames,
    precision,
    prefillFile: graph.prefill.file,
    presentNames,
  });
}

function validateTrainingLineageExport(lineageExport, metadata) {
  const checkpoint = metadata.checkpoint;
  if (
    lineageExport?.kind !== "localagent_training_lineage_export" ||
    lineageExport.schema_version !== 1 ||
    lineageExport.stage !== "rl" ||
    lineageExport.checkpoint_sha256 !== checkpoint.sha256 ||
    lineageExport.conversation_prompt_contract !== OPENAI_FULL_CATALOG_V1 ||
    !exactJsonEqual(lineageExport.lineage, checkpoint.lineage) ||
    !Array.isArray(lineageExport.training_artifact_sha256) ||
    !lineageExport.training_artifact_sha256.length ||
    new Set(lineageExport.training_artifact_sha256).size !==
      lineageExport.training_artifact_sha256.length ||
    lineageExport.training_artifact_sha256.some((value) => !isLowerSha256(value))
  ) {
    throw new Error("Cached training-lineage sidecar does not identify the final RL checkpoint.");
  }
  return lineageExport;
}

function validateProductionCachedBundle(
  metadata,
  provenance,
  actionMetadata,
  tokenizerEvidence,
  lineageExport
) {
  const graph = validateCachedGraphContract(metadata);
  const checkpoint = metadata.checkpoint;
  const lineage = checkpoint?.lineage;
  const requiredLineageHashes = [
    "config_sha256",
    "data_sha256",
    "model_config_sha256",
    "tokenizer_sha256",
    "parent_checkpoint_sha256",
  ];
  if (
    checkpoint?.stage !== "rl" ||
    !Number.isInteger(checkpoint.step) ||
    checkpoint.step < 0 ||
    !isLowerSha256(checkpoint.sha256) ||
    checkpoint.conversation_prompt_contract !== OPENAI_FULL_CATALOG_V1 ||
    lineage?.version !== 1 ||
    lineage.stage !== "rl" ||
    requiredLineageHashes.some((field) => !isLowerSha256(lineage[field])) ||
    lineage.model_config_sha256 !== metadata.model.config_canonical_sha256 ||
    lineage.tokenizer_sha256 !== metadata.tokenizer?.sha256 ||
    typeof lineage.git?.dirty !== "boolean" ||
    !/^[0-9a-f]{40}$/.test(lineage.git?.commit || "") ||
    !isLowerSha256(lineage.git?.repository_sha256) ||
    !isLowerSha256(lineage.git?.worktree_sha256) ||
    checkpoint.lineage_export?.kind !== "localagent_training_lineage_export" ||
    checkpoint.lineage_export?.schema_version !== 1 ||
    typeof checkpoint.lineage_export?.file !== "string" ||
    !checkpoint.lineage_export.file
  ) {
    throw new Error("Cached metadata does not carry canonical final-RL lineage.");
  }
  if (
    metadata.encoding !== "bytelevel-bpe" ||
    metadata.tokenizer?.kind !== "bpe" ||
    metadata.tokenizer?.verified !== true ||
    metadata.tokenizer?.vocab_size !== metadata.vocab_size ||
    metadata.tokenizer?.file !== "tokenizer.json" ||
    !isLowerSha256(metadata.tokenizer.sha256) ||
    metadata.eos_id !== 0 ||
    metadata.pad_id !== 0
  ) {
    throw new Error("Final cached bundle must carry its exact verified BPE tokenizer identity.");
  }
  if (
    !actionMetadata ||
    actionMetadata.encoding !== metadata.encoding ||
    actionMetadata.vocab_size !== metadata.vocab_size ||
    actionMetadata.d_model !== metadata.d_model ||
    actionMetadata.max_seq_len !== metadata.max_seq_len ||
    actionMetadata.eos_id !== metadata.eos_id ||
    actionMetadata.pad_id !== metadata.pad_id ||
    !exactJsonEqual(actionMetadata.markers, metadata.markers) ||
    !exactJsonEqual(actionMetadata.tools, metadata.tools) ||
    tokenizerEvidence?.sha256 !== metadata.tokenizer.sha256
  ) {
    throw new Error(
      "Cached model tokenizer/catalog metadata disagrees with the structured browser bundle."
    );
  }
  if (
    provenance?.schema_version !== 1 ||
    provenance.artifact_type !== "trained_checkpoint_cached_decode_onnx" ||
    provenance.trained !== true ||
    provenance.weights?.source !== "strict_lineage_validated_lm_checkpoint" ||
    provenance.weights.checkpoint_sha256 !== checkpoint.sha256 ||
    provenance.weights.checkpoint_stage !== "rl" ||
    provenance.weights.checkpoint_step !== checkpoint.step ||
    !exactJsonEqual(provenance.checkpoint_lineage, lineage) ||
    !exactJsonEqual(provenance.graph_contract, metadata.graph_contract) ||
    !exactJsonEqual(provenance.tokenizer, metadata.tokenizer) ||
    provenance.auxiliary_heads?.available !== false ||
    provenance.auxiliary_heads?.exported !== false ||
    provenance.auxiliary_heads?.validated !== true
  ) {
    throw new Error("Cached provenance does not bind the final trained model metadata.");
  }
  const parity = provenance.parity?.results?.[graph.precision];
  if (
    provenance.parity?.hard_gate !== true ||
    parity?.hard_gate !== true ||
    parity.passed !== true ||
    parity.greedy_next_token_exact !== true ||
    parity.cache_dtype !== graph.cacheDtype ||
    !exactJsonEqual(
      parity.final_token_logits_shape,
      ["batch", metadata.vocab_size]
    ) ||
    !Number.isFinite(parity.max_logits_abs_diff) ||
    parity.max_logits_abs_diff < 0 ||
    parity.max_logits_abs_diff > parity.logits_atol ||
    parity.reference_independence?.onnx_logits_vs_pytorch_cached_path !== true ||
    parity.reference_independence
      ?.pytorch_cached_vs_fresh_full_context_logits !== true
  ) {
    throw new Error("Cached logits/cache parity is missing or failed.");
  }
  for (const [kind, file] of [
    ["prefill", graph.prefillFile],
    ["decode", graph.decodeFile],
  ]) {
    const artifact = provenance.artifacts?.[file];
    const parityArtifact = parity.artifacts?.[kind];
    if (
      artifact?.file !== file ||
      artifact.precision !== graph.precision ||
      !Number.isInteger(artifact.bytes) ||
      artifact.bytes < 1 ||
      !isLowerSha256(artifact.sha256) ||
      parityArtifact?.bytes !== artifact.bytes ||
      parityArtifact?.sha256 !== artifact.sha256
    ) {
      throw new Error(`Cached ${kind} graph is not content-bound to passed parity.`);
    }
  }
  validateTrainingLineageExport(lineageExport, metadata);
  return graph;
}

function cachedOutputLocations(provider, presentNames) {
  const cacheLocation = provider === "webgpu" ? "gpu-buffer" : "cpu";
  return Object.fromEntries([
    ["next_token", "cpu"],
    ["logits", "cpu"],
    ...presentNames.map((name) => [name, cacheLocation]),
  ]);
}

function cachedSessionOptions(provider, presentNames) {
  return {
    executionProviders: [provider],
    preferredOutputLocation: cachedOutputLocations(provider, presentNames),
  };
}

async function createBundleSession(modelUrl, provider, expectedOutputs = null) {
  const artifact = await fetchModelArtifact(modelUrl);
  const session = await ort.InferenceSession.create(
    artifact.bytes.slice(),
    sessionOptions(provider)
  );
  validateSessionOutputs(session, modelUrl, expectedOutputs);
  return session;
}

// ---- bundle loading -------------------------------------------------------
async function loadBundle() {
  const loadStart = performance.now();
  if (
    BENCHMARK_GRADE &&
    REQUESTED_BACKEND !== "webgpu" &&
    REQUESTED_BACKEND !== "wasm"
  ) {
    throw new Error(
      "Benchmark-grade runs require an explicit ?backend=webgpu or ?backend=wasm condition."
    );
  }
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/";
  const metadataStart = performance.now();
  let manifestDocument;
  try {
    manifestDocument = await fetchJsonArtifact("bundle-manifest.json", false);
  } catch (error) {
    if (BENCHMARK_GRADE) throw error;
    console.warn("Bundle manifest unavailable; continuing in demo-only mode:", error);
    manifestDocument = null;
  }
  BUNDLE_MANIFEST = manifestDocument?.value ?? null;
  const [headsDocument, metaDocument, dispatchDocument] = await Promise.all([
    fetchJsonArtifact("heads.json", BENCHMARK_GRADE),
    fetchJsonArtifact("meta.json", BENCHMARK_GRADE),
    fetchJsonArtifact("dispatch_heads.json", BENCHMARK_GRADE),
  ]);
  HEADS = headsDocument.value;
  META = metaDocument.value;
  DISPATCH = dispatchDocument.value;
  BUNDLE_LOAD_TIMING.metadata_ms = performance.now() - metadataStart;
  const tokenizerStart = performance.now();
  if (META.encoding === "bytelevel-bpe") {
    if (!META.tokenizer_file) {
      throw new Error("BPE metadata is missing tokenizer_file.");
    }
    const tokenizerDocument = await fetchJsonArtifact(
      META.tokenizer_file,
      BENCHMARK_GRADE
    );
    TOKENIZER = await LocalAgentTokenizer.fromMeta(META, async (requestedPath) => {
      if (requestedPath !== META.tokenizer_file) {
        throw new Error(`Unexpected tokenizer asset request ${requestedPath}.`);
      }
      return tokenizerDocument.value;
    });
  } else {
    TOKENIZER = await LocalAgentTokenizer.fromMeta(META, async () => {
      throw new Error("Byte-tokenizer bundles must not request a tokenizer asset.");
    });
  }
  BUNDLE_LOAD_TIMING.tokenizer_ms = performance.now() - tokenizerStart;
  LOGITS_MODEL_URL = META.model_file || LOGITS_MODEL_URL;
  if (BENCHMARK_GRADE && !META.action_model_file) {
    throw new Error(
      "Benchmark-grade runs cannot fall back from a missing action graph to the logits graph."
    );
  }
  MODEL_URL = META.action_model_file || LOGITS_MODEL_URL;
  if (BENCHMARK_GRADE) validateBenchmarkBundleContract();
  const actionOutputs = BENCHMARK_GRADE ? ["hidden"] : null;
  const sessionStart = performance.now();
  if (REQUESTED_BACKEND === "wasm") {
    SESSION = await createBundleSession(MODEL_URL, "wasm", actionOutputs);
    BACKEND = "wasm";
    BUNDLE_LOAD_TIMING.session_create_ms = performance.now() - sessionStart;
    BUNDLE_LOAD_TIMING.total_ms = performance.now() - loadStart;
    return;
  }
  if (REQUESTED_BACKEND === "webgpu") {
    SESSION = await createBundleSession(MODEL_URL, "webgpu", actionOutputs);
    BACKEND = "webgpu";
    BUNDLE_LOAD_TIMING.session_create_ms = performance.now() - sessionStart;
    BUNDLE_LOAD_TIMING.total_ms = performance.now() - loadStart;
    return;
  }
  if (REQUESTED_BACKEND !== "auto") {
    throw new Error(`Unknown backend '${REQUESTED_BACKEND}'; expected auto, webgpu, or wasm.`);
  }
  try {
    SESSION = await createBundleSession(MODEL_URL, "webgpu", actionOutputs);
    BACKEND = "webgpu";
  } catch (e) {
    console.warn("WebGPU unavailable, falling back to WASM:", e);
    SESSION = await createBundleSession(MODEL_URL, "wasm", actionOutputs);
    BACKEND = "wasm";
  }
  BUNDLE_LOAD_TIMING.session_create_ms = performance.now() - sessionStart;
  BUNDLE_LOAD_TIMING.total_ms = performance.now() - loadStart;
}

async function ensureLogitsSession() {
  if (LOGITS_SESSION) return LOGITS_SESSION;
  if (LOGITS_MODEL_URL === MODEL_URL) {
    if (BENCHMARK_GRADE) {
      throw new Error(
        "Benchmark-grade autoregressive controls require a distinct full logits graph."
      );
    }
    LOGITS_SESSION = SESSION;
    BUNDLE_LOAD_TIMING.logits_session_create_ms = 0;
    return LOGITS_SESSION;
  }
  const started = performance.now();
  // BACKEND was resolved while loading the action graph. The full graph must use that exact
  // provider; a failure is surfaced instead of silently changing the comparison backend.
  LOGITS_SESSION = await createBundleSession(
    LOGITS_MODEL_URL,
    BACKEND,
    BENCHMARK_GRADE ? ["logits", "hidden"] : null
  );
  BUNDLE_LOAD_TIMING.logits_session_create_ms = performance.now() - started;
  return LOGITS_SESSION;
}

function requestedCachedMetadataUrl(meta = META) {
  const query = new URLSearchParams(window.location?.search || "");
  return window.__localAgentCachedMetaUrl ||
    query.get("cached_meta") ||
    meta?.cached_decode_meta_file ||
    "cached/meta.json";
}

function cachedArtifactUrl(fileName, metadataUrl) {
  return new URL(fileName, new URL(metadataUrl, document.baseURI)).href;
}

async function fetchCachedArtifact(artifactUrl, expectedIdentity, label) {
  const response = await fetch(artifactUrl);
  if (!response.ok) throw new Error(`Missing ${label} ${artifactUrl} (HTTP ${response.status}).`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  const actualSha256 = await sha256Bytes(bytes);
  if (expectedIdentity) {
    if (
      !Number.isInteger(expectedIdentity.bytes) ||
      expectedIdentity.bytes < 1 ||
      !isLowerSha256(expectedIdentity.sha256) ||
      expectedIdentity.bytes !== bytes.byteLength ||
      expectedIdentity.sha256 !== actualSha256
    ) {
      throw new Error(`${label} bytes/hash do not match cached provenance.`);
    }
  }
  return {
    bytes,
    evidence: Object.freeze({
      bytes: bytes.byteLength,
      file: artifactUrl,
      sha256: actualSha256,
      verified: Boolean(expectedIdentity),
      verification_scope: "exact_fetched_response_body_before_parse_or_session",
    }),
  };
}

function parseCachedJson(artifact, label) {
  try {
    return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(artifact.bytes));
  } catch (error) {
    throw new Error(`${label} is not valid UTF-8 JSON: ${error.message}`);
  }
}

function validateCachedSessionContract(session, graph, kind) {
  const expectedInputs = graph[`${kind}Inputs`];
  const actualInputs = Array.from(session.inputNames || []);
  const actualOutputs = Array.from(session.outputNames || []);
  if (!exactJsonEqual(actualInputs, expectedInputs)) {
    throw new Error(
      `Cached ${kind} session inputs ${JSON.stringify(actualInputs)} do not match ` +
      `${JSON.stringify(expectedInputs)}.`
    );
  }
  if (!exactJsonEqual(actualOutputs, graph.outputs)) {
    throw new Error(
      `Cached ${kind} session outputs ${JSON.stringify(actualOutputs)} do not match ` +
      `${JSON.stringify(graph.outputs)}.`
    );
  }
}

function validateFullCatalogTokenizer(tokenizer = TOKENIZER) {
  if (
    !tokenizer ||
    !exactJsonEqual(tokenizer.encode(BPE_EOS_MARKER), [tokenizer.eosId])
  ) {
    throw new Error(
      "openai_full_catalog_v1 requires the canonical EOS marker to encode to eos_id."
    );
  }
  for (const suffix of [
    "<|system|>boundary-check",
    "<|user|>boundary-check",
    "<|tool|>boundary-check",
    "<|assistant|>boundary-check",
  ]) {
    if (
      !exactJsonEqual(
        tokenizer.encode(BPE_EOS_MARKER + suffix),
        [tokenizer.eosId, ...tokenizer.encode(suffix)]
      )
    ) {
      throw new Error(
        `openai_full_catalog_v1 tokenizer does not preserve the EOS boundary before ${suffix}.`
      );
    }
  }
}

async function ensureCachedDecodeSessions() {
  if (CACHED_DECODE_BUNDLE) return CACHED_DECODE_BUNDLE;
  const metadataRequest = requestedCachedMetadataUrl();
  const metadataUrl = new URL(metadataRequest, document.baseURI).href;
  const provenanceRequest = META.cached_decode_provenance_file ||
    `${metadataRequest.slice(0, metadataRequest.lastIndexOf("/") + 1)}provenance.json`;
  const provenanceUrl = new URL(provenanceRequest, document.baseURI).href;
  const manifestProvenancePin = manifestArtifactFor(provenanceRequest);
  if (BENCHMARK_GRADE && !manifestProvenancePin) {
    throw new Error(
      `Benchmark bundle manifest does not pin cached provenance request ${metadataRequest}.`
    );
  }
  const provenanceArtifact = await fetchCachedArtifact(
    provenanceUrl,
    manifestProvenancePin,
    "cached provenance"
  );
  const provenance = parseCachedJson(provenanceArtifact, "cached provenance");
  const metadataPin = provenance.artifacts?.["meta.json"];
  const metadataArtifact = await fetchCachedArtifact(
    metadataUrl,
    metadataPin,
    "cached metadata"
  );
  const metadata = parseCachedJson(metadataArtifact, "cached metadata");
  const lineageFile = metadata.checkpoint?.lineage_export?.file;
  const lineageArtifact = await fetchCachedArtifact(
    cachedArtifactUrl(lineageFile || "missing-training-lineage.json", metadataUrl),
    provenance.artifacts?.[lineageFile],
    "cached training lineage"
  );
  const lineageExport = parseCachedJson(lineageArtifact, "cached training lineage");
  validateFullCatalogTokenizer();
  const graph = validateProductionCachedBundle(
    metadata,
    provenance,
    META,
    bundleArtifactEvidence(META.tokenizer_file),
    lineageExport
  );

  const prefillArtifact = await fetchCachedArtifact(
    cachedArtifactUrl(graph.prefillFile, metadataUrl),
    provenance.artifacts[graph.prefillFile],
    "cached prefill graph"
  );
  const decodeArtifact = await fetchCachedArtifact(
    cachedArtifactUrl(graph.decodeFile, metadataUrl),
    provenance.artifacts[graph.decodeFile],
    "cached decode graph"
  );
  const options = cachedSessionOptions(BACKEND, graph.presentNames);
  const [prefillSession, decodeSession] = await Promise.all([
    ort.InferenceSession.create(prefillArtifact.bytes.slice(), options),
    ort.InferenceSession.create(decodeArtifact.bytes.slice(), options),
  ]);
  const sessionContract = {
    ...graph,
    prefillInputs: ["input_ids"],
    decodeInputs: graph.decodeInputs,
  };
  validateCachedSessionContract(prefillSession, sessionContract, "prefill");
  validateCachedSessionContract(decodeSession, sessionContract, "decode");
  CACHED_PREFILL_SESSION = prefillSession;
  CACHED_DECODE_SESSION = decodeSession;
  CACHED_DECODE_BUNDLE = Object.freeze({
    ...graph,
    decodeSession,
    evidence: Object.freeze({
      decode: decodeArtifact.evidence,
      lineage: lineageArtifact.evidence,
      metadata: metadataArtifact.evidence,
      prefill: prefillArtifact.evidence,
      provenance: provenanceArtifact.evidence,
    }),
    lineageExport,
    metadata,
    prefillSession,
    provenance,
  });
  return CACHED_DECODE_BUNDLE;
}

async function prepareActionPolicy(policy) {
  if (!Object.values(ACTION_POLICIES).includes(policy)) {
    throw new Error(`Unknown action policy '${policy}'.`);
  }
  if (policy !== ACTION_POLICIES.STRUCTURED) await ensureCachedDecodeSessions();
}

// ---- tokenizer ------------------------------------------------------------
// Markers are literal strings. The bundle metadata selects UTF-8 bytes or the exact trained
// ByteLevel BPE tokenizer, both matching localagent.model.tokenizer.
const enc = new TextEncoder();
function mark(name) { return META.markers[name].text; } // markers carry { text, ids }

const RESERVED_FULL_CATALOG_MARKERS = Object.freeze([
  BPE_EOS_MARKER,
  "<|system|>",
  "<|user|>",
  "<|assistant|>",
  "<|tool|>",
  "<tool_call>",
  "</tool_call>",
  "<tool_response>",
  "</tool_response>",
  TOOL_CATALOG_OPEN,
  TOOL_CATALOG_CLOSE,
]);

function assertNoPromptMarker(value, label) {
  if (typeof value !== "string") throw new Error(`${label} must be text.`);
  const marker = RESERVED_FULL_CATALOG_MARKERS.find((candidate) => value.includes(candidate));
  if (marker) throw new Error(`${label} contains reserved prompt marker ${marker}.`);
}

function renderFullFunctionCatalog(meta = META) {
  if (!Array.isArray(meta?.tools) || !meta.tools.length) {
    throw new Error("openai_full_catalog_v1 requires a non-empty complete tool catalog.");
  }
  const names = new Set();
  const tools = meta.tools.map((tool, index) => {
    if (
      typeof tool?.name !== "string" ||
      !tool.name ||
      typeof tool.description !== "string" ||
      !tool.schema ||
      typeof tool.schema !== "object" ||
      Array.isArray(tool.schema)
    ) {
      throw new Error(`Tool catalog entry ${index} is incomplete.`);
    }
    assertNoPromptMarker(tool.name, `tool ${index} name`);
    assertNoPromptMarker(tool.description, `tool ${tool.name} description`);
    if (names.has(tool.name)) throw new Error(`Tool catalog repeats ${tool.name}.`);
    names.add(tool.name);
    return {
      type: "function",
      function: {
        name: tool.name,
        description: tool.description,
        parameters: tool.schema,
      },
    };
  });
  return TOOL_CATALOG_OPEN + canonicalActionJson({ tools }) + TOOL_CATALOG_CLOSE;
}

function renderFullCatalogContextText(query, steps = [], meta = META) {
  assertNoPromptMarker(query, "user message");
  const marker = (name) => meta.markers[name].text;
  let history = marker("user") + query;
  for (const [index, step] of (steps || []).entries()) {
    if (typeof step?.tool !== "string" || !step.tool) {
      throw new Error(`Tool step ${index} has no tool name.`);
    }
    const response = step.response || "ok";
    assertNoPromptMarker(response, `tool step ${index} response`);
    history += marker("assistant") + canonicalToolCompletion({
      tool: step.tool,
      args: step.args,
    }, meta) + BPE_EOS_MARKER;
    history += marker("tool") + marker("tool_response_open") +
      response + marker("tool_response_close");
  }
  return renderFullFunctionCatalog(meta) + BPE_EOS_MARKER + history + marker("assistant");
}

// Render a user turn the way the model was trained.
function renderContextText(query, steps) {
  let s = mark("user") + query + mark("assistant");
  for (const st of steps || []) {
    s += mark("tool_call_open") + st.tool + "(" + JSON.stringify(st.args) + ")" + mark("tool_call_close");
    s += mark("tool") + mark("tool_response_open") + (st.response || "ok") + mark("tool_response_close");
    s += mark("assistant");
  }
  return s;
}
function renderContext(query, steps) {
  return TOKENIZER.encode(renderContextText(query, steps));
}

function padPromptIds(baseIds, assistantIds, whitespaceId, targetInputTokens) {
  if (targetInputTokens == null) return [...baseIds];
  const target = Number.parseInt(targetInputTokens, 10);
  if (!Number.isInteger(target) || target < 1) {
    throw new Error("targetInputTokens must be a positive integer.");
  }
  if (baseIds.length > target) {
    throw new Error(`Natural prompt has ${baseIds.length} tokens, above target ${target}.`);
  }
  if (!assistantIds.length || assistantIds.length > baseIds.length) {
    throw new Error("Assistant marker ids are missing from bundle metadata.");
  }
  const suffixStart = baseIds.length - assistantIds.length;
  if (!assistantIds.every((tokenId, index) => baseIds[suffixStart + index] === tokenId)) {
    throw new Error("Rendered prompt does not end with the declared assistant marker ids.");
  }
  return [
    ...baseIds.slice(0, suffixStart),
    ...Array(target - baseIds.length).fill(whitespaceId),
    ...assistantIds,
  ];
}

function padPromptIdsTrailing(baseIds, whitespaceId, targetInputTokens) {
  if (targetInputTokens == null) return [...baseIds];
  const target = Number.parseInt(targetInputTokens, 10);
  if (!Number.isInteger(target) || target < 1) {
    throw new Error("targetInputTokens must be a positive integer.");
  }
  if (baseIds.length > target) {
    throw new Error(`Natural prompt has ${baseIds.length} tokens, above target ${target}.`);
  }
  return [...baseIds, ...Array(target - baseIds.length).fill(whitespaceId)];
}

function actionPrompt(
  query,
  targetInputTokens = null,
  paddingPlacement = "pre_assistant"
) {
  const naturalText = renderContextText(query, []);
  const naturalIds = TOKENIZER.encode(naturalText);
  const assistantIds = META.markers?.assistant?.ids || TOKENIZER.encode(mark("assistant"));
  const whitespaceIds = TOKENIZER.encode(" ");
  if (whitespaceIds.length !== 1) {
    throw new Error("The exported tokenizer must encode one neutral space as exactly one token.");
  }
  let ids;
  let decisionInputTokens;
  let contextPaddingPlacement;
  if (paddingPlacement === "trailing_compute") {
    ids = padPromptIdsTrailing(naturalIds, whitespaceIds[0], targetInputTokens);
    decisionInputTokens = naturalIds.length;
    contextPaddingPlacement = targetInputTokens == null
      ? "none"
      : "after_natural_assistant_marker";
  } else if (paddingPlacement === "pre_assistant") {
    ids = padPromptIds(naturalIds, assistantIds, whitespaceIds[0], targetInputTokens);
    decisionInputTokens = ids.length;
    contextPaddingPlacement = targetInputTokens == null
      ? "none"
      : "before_assistant_marker";
  } else {
    throw new Error(`Unknown context padding placement '${paddingPlacement}'.`);
  }
  const materializedText = TOKENIZER.decode(ids, false);
  return {
    ids,
    inputBytes: enc.encode(materializedText).length,
    naturalInputTokens: naturalIds.length,
    paddingTokens: ids.length - naturalIds.length,
    contextPaddingPlacement,
    decisionInputTokens,
    decisionFeatureIndex: decisionInputTokens - 1,
  };
}

function cachedActionPrompt(query, targetInputTokens = null) {
  const naturalText = renderFullCatalogContextText(query);
  const naturalIds = TOKENIZER.encode(naturalText);
  const assistantIds = META.markers?.assistant?.ids || TOKENIZER.encode(mark("assistant"));
  const whitespaceIds = TOKENIZER.encode(" ");
  if (whitespaceIds.length !== 1) {
    throw new Error("The exported tokenizer must encode one neutral space as exactly one token.");
  }
  const ids = padPromptIds(
    naturalIds,
    assistantIds,
    whitespaceIds[0],
    targetInputTokens
  );
  return {
    ids,
    inputBytes: enc.encode(TOKENIZER.decode(ids, false)).length,
    naturalInputTokens: naturalIds.length,
    paddingTokens: ids.length - naturalIds.length,
    contextPaddingPlacement: targetInputTokens == null
      ? "none"
      : "before_assistant_marker",
    decisionInputTokens: ids.length,
    decisionFeatureIndex: ids.length - 1,
    promptContract: OPENAI_FULL_CATALOG_V1,
    toolCatalogSize: META.tools.length,
  };
}

// ---- model forward --------------------------------------------------------
async function forwardWithSession(ids, session) {
  const arr = BigInt64Array.from(ids.map((x) => BigInt(x)));
  const input = new ort.Tensor("int64", arr, [1, ids.length]);
  const out = await session.run({ input_ids: input });
  return out; // { logits, hidden }
}

async function forward(ids) {
  return forwardWithSession(ids, SESSION);
}

async function forwardLogits(ids) {
  const session = await ensureLogitsSession();
  const out = await forwardWithSession(ids, session);
  if (!out.logits) {
    throw new Error(
      `Full graph ${LOGITS_MODEL_URL} did not expose logits; this is not an autoregressive bundle.`
    );
  }
  return out.logits;
}

function float16BitsToNumber(bits) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0x1f) return fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function materializeCachedLogits(tensor, bundle, label) {
  const vocabSize = bundle.metadata.vocab_size;
  if (
    !tensor ||
    tensor.type !== bundle.cacheDtype ||
    !exactJsonEqual(tensor.dims, [1, vocabSize]) ||
    !tensor.data ||
    tensor.data.length !== vocabSize
  ) {
    throw new Error(
      `${label} must be ${bundle.cacheDtype} [1, ${vocabSize}] CPU logits.`
    );
  }
  if (tensor.location && tensor.location !== "cpu") {
    throw new Error(`${label} must be materialized on CPU for token selection.`);
  }
  const values = new Float32Array(vocabSize);
  const rawFloat16 = bundle.cacheDtype === "float16" &&
    tensor.data instanceof Uint16Array &&
    tensor.data.constructor?.name !== "Float16Array";
  for (let index = 0; index < vocabSize; index++) {
    const value = rawFloat16
      ? float16BitsToNumber(tensor.data[index])
      : Number(tensor.data[index]);
    if (!Number.isFinite(value)) throw new Error(`${label} contains a non-finite value.`);
    values[index] = value;
  }
  return values;
}

function cachedCompatibilityToken(tensor, vocabSize, label) {
  if (
    !tensor ||
    tensor.type !== "int64" ||
    !exactJsonEqual(tensor.dims, [1]) ||
    !tensor.data ||
    tensor.data.length !== 1
  ) {
    throw new Error(`${label} must be an int64 [1] CPU compatibility token.`);
  }
  if (tensor.location && tensor.location !== "cpu") {
    throw new Error(`${label} must be materialized on CPU.`);
  }
  const token = Number(tensor.data[0]);
  if (!Number.isSafeInteger(token) || token < 0 || token >= vocabSize) {
    throw new Error(`${label} contains an invalid token ID.`);
  }
  return token;
}

function validateCachedCacheTensor(tensor, slot, sequenceLength, bundle, name) {
  const config = bundle.metadata.model.config;
  const dims = slot.kind === "attn"
    ? [1, config.n_kv_heads, sequenceLength, config.d_model / config.n_heads]
    : [1, config.d_model, config.conv_kernel - 1];
  if (
    !tensor ||
    tensor.type !== bundle.cacheDtype ||
    !exactJsonEqual(tensor.dims, dims)
  ) {
    throw new Error(
      `${name} must be ${bundle.cacheDtype} ${JSON.stringify(dims)}; got ` +
      `${tensor?.type || "missing"} ${JSON.stringify(tensor?.dims || null)}.`
    );
  }
  const expectedLocation = BACKEND === "webgpu" ? "gpu-buffer" : "cpu";
  if (tensor.location && tensor.location !== expectedLocation) {
    throw new Error(`${name} cache location ${tensor.location} != ${expectedLocation}.`);
  }
}

function validateCachedStepOutputs(outputs, bundle, sequenceLength, label) {
  const logits = materializeCachedLogits(outputs?.logits, bundle, `${label}.logits`);
  const compatibilityToken = cachedCompatibilityToken(
    outputs?.next_token,
    bundle.metadata.vocab_size,
    `${label}.next_token`
  );
  const logitsArgmax = greedyToken(logits);
  if (compatibilityToken !== logitsArgmax) {
    throw new Error(
      `${label} next_token ${compatibilityToken} disagrees with logits argmax ${logitsArgmax}.`
    );
  }
  const caches = new Map();
  for (const slot of bundle.contract.cache_slots) {
    for (const name of slot.present_outputs) {
      validateCachedCacheTensor(outputs?.[name], slot, sequenceLength, bundle, name);
      caches.set(name, outputs[name]);
    }
  }
  return { caches, compatibilityToken, logits, logitsArgmax };
}

function disposeOrtTensor(tensor) {
  if (typeof tensor?.dispose === "function") tensor.dispose();
}

function disposeCachedOutputs(outputs, bundle, keepCaches = false) {
  disposeOrtTensor(outputs?.next_token);
  disposeOrtTensor(outputs?.logits);
  if (!keepCaches) {
    for (const name of bundle.presentNames) disposeOrtTensor(outputs?.[name]);
  }
}

function disposeCacheTensors(caches) {
  for (const tensor of caches?.values?.() || []) disposeOrtTensor(tensor);
  caches?.clear?.();
}

function cachedFeeds(caches, bundle) {
  const feeds = {};
  for (const slot of bundle.contract.cache_slots) {
    slot.past_inputs.forEach((pastName, index) => {
      const presentName = slot.present_outputs[index];
      const tensor = caches.get(presentName);
      if (!tensor) throw new Error(`Missing direct cache binding ${presentName} -> ${pastName}.`);
      feeds[pastName] = tensor;
    });
  }
  return feeds;
}

function createCachedAutoregressiveRunner(bundle, promptIds) {
  if (!bundle?.prefillSession || !bundle?.decodeSession) {
    throw new Error("Cached prefill/decode sessions are not ready.");
  }
  if (!Array.isArray(promptIds) || !promptIds.length) {
    throw new Error("Cached autoregressive runner requires a non-empty prompt.");
  }
  let caches = new Map();
  let sequenceLength = promptIds.length;
  let prefilled = false;
  let disposed = false;

  async function prefill() {
    if (disposed || prefilled) throw new Error("Cached prefill may run exactly once.");
    const input = new ort.Tensor(
      "int64",
      BigInt64Array.from(promptIds, (token) => BigInt(token)),
      [1, promptIds.length]
    );
    let outputs;
    try {
      outputs = await bundle.prefillSession.run({ input_ids: input });
      const decision = validateCachedStepOutputs(
        outputs,
        bundle,
        sequenceLength,
        "cached.prefill"
      );
      caches = decision.caches;
      prefilled = true;
      disposeCachedOutputs(outputs, bundle, true);
      return {
        compatibilityToken: decision.compatibilityToken,
        logits: decision.logits,
        logitsArgmax: decision.logitsArgmax,
      };
    } catch (error) {
      disposeCachedOutputs(outputs, bundle);
      throw error;
    } finally {
      disposeOrtTensor(input);
    }
  }

  async function decode(inputToken) {
    if (disposed || !prefilled) throw new Error("Cached decode requires one completed prefill.");
    if (
      !Number.isSafeInteger(inputToken) ||
      inputToken < 0 ||
      inputToken >= bundle.metadata.vocab_size
    ) {
      throw new Error("Cached decode input is outside the model vocabulary.");
    }
    const input = new ort.Tensor(
      "int64",
      BigInt64Array.of(BigInt(inputToken)),
      [1, 1]
    );
    let outputs;
    try {
      outputs = await bundle.decodeSession.run({
        input_ids: input,
        ...cachedFeeds(caches, bundle),
      });
      const decision = validateCachedStepOutputs(
        outputs,
        bundle,
        sequenceLength + 1,
        `cached.decode[${sequenceLength}]`
      );
      disposeCacheTensors(caches);
      caches = decision.caches;
      sequenceLength += 1;
      disposeCachedOutputs(outputs, bundle, true);
      return {
        compatibilityToken: decision.compatibilityToken,
        logits: decision.logits,
        logitsArgmax: decision.logitsArgmax,
      };
    } catch (error) {
      disposeCachedOutputs(outputs, bundle);
      throw error;
    } finally {
      disposeOrtTensor(input);
    }
  }

  function dispose() {
    if (disposed) return;
    disposed = true;
    disposeCacheTensors(caches);
  }

  return Object.freeze({ decode, dispose, prefill });
}

function seededTokenRandom(seedText) {
  let state = 2166136261;
  for (let index = 0; index < seedText.length; index++) {
    state ^= seedText.charCodeAt(index);
    state = Math.imul(state, 16777619);
  }
  return () => {
    state += 0x6D2B79F5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function selectTokenFromLogits(logits, options = {}, allowedTokenIds = null) {
  const candidates = allowedTokenIds == null
    ? Array.from({ length: logits.length }, (_, token) => token)
    : Array.from(allowedTokenIds);
  if (!candidates.length) throw new Error("Token selection has no allowed candidates.");
  if (
    candidates.some((token) =>
      !Number.isInteger(token) || token < 0 || token >= logits.length
    )
  ) {
    throw new Error("Token selection contains an out-of-vocabulary candidate.");
  }
  const temperature = Number(options.temperature ?? 0);
  if (!Number.isFinite(temperature) || temperature < 0) {
    throw new Error("temperature must be finite and non-negative.");
  }
  if (temperature === 0) return argmaxAllowed(logits, candidates);
  const requestedTopK = Number.parseInt(options.topK ?? candidates.length, 10);
  if (!Number.isInteger(requestedTopK) || requestedTopK < 1) {
    throw new Error("topK must be a positive integer.");
  }
  const ranked = candidates
    .map((token) => ({ token, value: logits[token] }))
    .sort((left, right) => right.value - left.value || left.token - right.token)
    .slice(0, Math.min(requestedTopK, candidates.length));
  const maximum = ranked[0].value;
  const weights = ranked.map((entry) => Math.exp((entry.value - maximum) / temperature));
  const total = weights.reduce((sum, value) => sum + value, 0);
  if (!Number.isFinite(total) || total <= 0) {
    throw new Error("Sampling probabilities are invalid.");
  }
  const random = options.random || seededTokenRandom(String(options.seed ?? "localagent"));
  let threshold = random() * total;
  for (let index = 0; index < ranked.length; index++) {
    threshold -= weights[index];
    if (threshold <= 0) return ranked[index].token;
  }
  return ranked.at(-1).token;
}

// ---- generable dispatch: route head -> dense two-tower selector ------------
function lastHidden(hiddenTensor, T) {
  const d = META?.d_model ?? hiddenTensor?.dims?.at(-1);
  if (!Number.isInteger(d) || d < 1) throw new Error("Hidden tensor has no valid model width.");
  const H = hiddenTensor.data, off = (T - 1) * d;
  return H.subarray ? H.subarray(off, off + d) : Array.from(H).slice(off, off + d);
}
function linrow(W, b, x) {                 // W[o][d] · x[d] + b[o] -> [o]
  const o = W.length, out = new Float32Array(o);
  for (let i = 0; i < o; i++) { const Wi = W[i]; let a = b ? b[i] : 0; for (let k = 0; k < x.length; k++) a += Wi[k] * x[k]; out[i] = a; }
  return out;
}
function argmax(v) { let bi = 0; for (let i = 1; i < v.length; i++) if (v[i] > v[bi]) bi = i; return bi; }
function greedyToken(logits) { return argmax(logits); }
function softmaxAt(v, i) { let m = -Infinity; for (const x of v) m = Math.max(m, x); let z = 0; for (const x of v) z += Math.exp(x - m); return Math.exp(v[i] - m) / z; }

function mobileLexicalSelect(query, dispatch = DISPATCH) {
  if (!MOBILE_LEXICAL_GUARD) return null;
  const names = new Set(dispatch?.dense_selector?.tool_names || []);
  if (!names.has("mobile_click") || typeof query !== "string") return null;
  // Stateful prompts include a goal and serialized state before the actionable instruction.
  // Only inspect the requested next action; otherwise words such as "fill" in a Gmail goal can
  // override the actual open-app action and turn a valid dense prediction into a wrong tool.
  const actionMatch = query.match(/(?:next required action|current step(?:\s+[^:]+)?|current action|instruction)\s*:\s*([\s\S]*)$/i);
  const low = (actionMatch ? actionMatch[1] : query).toLowerCase();
  const composeState = /"screen"\s*:\s*"compose"/i.test(query) &&
    /"focus"\s*:\s*"[^"]+"/i.test(query);
  // Require an explicitly mobile/handset cue. Generic browser prompts often mention a screen,
  // window, app, click, or scroll too; those must continue through the learned standard pool.
  const mobileCue = /\b(?:mobile|android|phone|touch|tap|swipe)\b/.test(low);
  if (!mobileCue && !composeState) return null;
  const choose = (name) => names.has(name) ? {
    name,
    route: "computer_use",
    conf: 1,
    isStop: false,
    selection_policy: "mobile_lexical_guard",
  } : null;
  if (composeState && /\b(?:send|submit|deliver|dispatch)\b/.test(low)) {
    // Prefer the full-field compose contract when the catalog exposes it.  A catalog with only
    // the legacy recipient-only contract still receives the generic productivity fallback below.
    if (names.has("email_send")) return choose("email_send");
    if (names.has("send_email")) return choose("send_email");
  }
  if (/\b(?:navigate|go|return|press)\b[\s\S]*\bhome\b/.test(low)) return choose("mobile_navigate_home");
  if (/\b(?:navigate|go|return|press)\b[\s\S]*\bback\b/.test(low)) return choose("mobile_navigate_back");
  if (/\b(?:press|hit|send)\b[\s\S]*\benter\b/.test(low)) return choose("mobile_press_enter");
  if (/\b(?:type|input|fill|enter)\b/.test(low)) return choose("mobile_input_text");
  if (/\b(?:long[ -]?press|hold)\b/.test(low)) return choose("mobile_long_press");
  if (/\bswipe\b/.test(low)) return choose("mobile_swipe");
  if (/\bscroll\b/.test(low)) return choose("mobile_scroll");
  if (/\b(?:open|launch|start|bring up)\b/.test(low) && !/https?:\/\//.test(low)) {
    return choose("mobile_open_app");
  }
  if (/\b(?:tap|click|touch|select)\b/.test(low)) return choose("mobile_click");
  if (/\b(?:wait|sleep)\b/.test(low)) return choose("mobile_wait");
  return null;
}

function compactDispatchQuery(query, marker = " instruction:") {
  if (typeof query !== "string") return "";
  const lower = query.toLowerCase();
  const markers = [String(marker).trim(), "next required action:"];
  let index = -1;
  let selected = marker;
  for (const candidate of markers) {
    const candidateIndex = lower.lastIndexOf(candidate.toLowerCase());
    if (candidateIndex > index) {
      index = candidateIndex;
      selected = candidate;
    }
  }
  return index >= 0 ? query.slice(index + selected.length).trim() : query;
}

function crc32Utf8(text) {
  const bytes = new TextEncoder().encode(text);
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit++) {
      crc = (crc >>> 1) ^ (0xedb88320 * (crc & 1));
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function retrievalEmbedding(text, dim) {
  const normalized = ` ${String(text).toLowerCase().replace(/[^a-z0-9]+/g, " ").trim()} `;
  const vector = new Float32Array(dim);
  for (const n of [3, 4, 5]) {
    for (let index = 0; index + n <= normalized.length; index++) {
      vector[crc32Utf8(normalized.slice(index, index + n)) % dim] += 1;
    }
  }
  let norm = 0;
  for (const value of vector) norm += value * value;
  norm = Math.sqrt(norm);
  if (norm > 0) for (let index = 0; index < vector.length; index++) vector[index] /= norm;
  return vector;
}

function retrievalScoresFromSidecar(query, R) {
  if (!R?.tool_matrix?.length || !Array.isArray(R.tool_names)) return null;
  const compact = compactDispatchQuery(query, R.compact_instruction_marker);
  const q = retrievalEmbedding(compact, Number(R.dim));
  const scores = [];
  for (let row = 0; row < R.tool_matrix.length; row++) {
    const values = R.tool_matrix[row];
    let score = 0;
    for (let index = 0; index < q.length; index++) score += q[index] * values[index];
    scores.push(score);
  }
  const order = Array.from({ length: scores.length }, (_, index) => index).sort(
    (left, right) => scores[right] - scores[left] || left - right,
  );
  return { scores, order };
}

function retrievalCandidatesFromSidecar(query, R, k = 10) {
  const scored = retrievalScoresFromSidecar(query, R);
  if (!scored) return [];
  const limit = Math.max(1, Math.min(Number(k) || 10, scored.order.length));
  return scored.order.slice(0, limit).map((index) => ({
    name: R.tool_names[index],
    score: scored.scores[index],
    route: R.tool_routes?.[index] || "text",
  }));
}

function retrievalSelectFromSidecar(query, R) {
  const candidates = retrievalCandidatesFromSidecar(query, R, 1);
  if (!candidates.length) return null;
  const best = candidates[0];
  const bestScore = best.score;
  return {
    name: best.name,
    route: best.route,
    conf: Math.max(0, Math.min(1, (bestScore + 1) / 2)),
    isStop: false,
    selection_policy: "retrieval_selector",
  };
}

function retrievalCandidateNames(query, R, k = 10) {
  return retrievalCandidatesFromSidecar(query, R, k).map((candidate) => candidate.name);
}

function retrievalCandidateIndexSet(query, R, toolNames, k = 10) {
  const names = new Set(retrievalCandidateNames(query, R, k));
  return toolNames.reduce((indices, name, index) => {
    if (names.has(name)) indices.push(index);
    return indices;
  }, []);
}

function denseCandidateIndices(query, S, dispatch, requestedSelector) {
  if (requestedSelector !== "retrieval_then_dense") {
    return { indices: Array.from({ length: S.tool_names.length }, (_, index) => index), policy: "dense_selector" };
  }
  const indices = retrievalCandidateIndexSet(query, dispatch?.retrieval_selector, S.tool_names, 10);
  if (!indices.length) {
    return { indices: Array.from({ length: S.tool_names.length }, (_, index) => index), policy: "dense_selector" };
  }
  return { indices, policy: "retrieval_then_dense" };
}

function retrievalSelect(query) {
  return retrievalSelectFromSidecar(query, DISPATCH?.retrieval_selector);
}

// ---- side-effect safety ---------------------------------------------------
// The static demo has no external credentials or tool servers, but its action contract is meant
// to be reusable by a real browser/mobile harness.  Keep the safety decision explicit at the
// boundary so an adapter cannot accidentally turn a good route prediction into an unconfirmed
// email, Notion write, deletion, or message.  This is a policy gate, not learned-quality evidence.
const SIDE_EFFECT_TOOLS = new Set([
  "send_email", "email_send", "notion_write", "notion_create_page", "calendar_event",
  "slack_send", "slack_post_message", "create_event", "update_event", "delete_event",
  "write_file", "edit_file", "delete_file", "move_file", "run_command", "shell_exec",
  "send_message", "post_message", "delete_message", "submit_form", "purchase",
]);
const INTERACTIVE_TOOLS = new Set([
  "click", "web_click", "browser_click", "type_text", "web_type", "browser_type",
  "mobile_click", "mobile_type", "tap", "press_key", "scroll", "open_app",
]);
const DESTRUCTIVE_TOOLS = new Set([
  "delete_event", "delete_file", "move_file", "run_command", "shell_exec", "delete_message",
  "purchase", "submit_form",
]);
// This is a deliberately small lexical safety layer for high-risk public mobile/browser cases.
// It is independent of the learned selector and does not claim semantic safety understanding.
const RISK_SENSITIVE_READ_TOOLS = new Set([
  "web_search", "open_url", "read_message", "read_file", "screenshot",
]);
const SAFETY_POLICY_VERSION = "side_effect_confirmation_v1";
const INTERACTION_POLICY_VERSION = "user_in_the_loop_v1";
const RISK_POLICY_VERSION = "text_harm_block_v1";

function safetyText(value) {
  return typeof value === "string" ? value : "";
}

function promptInjectionIndicators(text) {
  const low = safetyText(text).toLowerCase();
  const patterns = [
    /\b(?:ignore|disregard|override)\b[^.\n]{0,80}\b(?:previous|prior|system|developer|user)\b/,
    /\b(?:system|developer)\s+(?:message|instruction|prompt)\b[^.\n]{0,100}\b(?:send|delete|upload|share|forward|exfiltrat)/,
    /\b(?:exfiltrat|steal|leak)\w*\b|\b(?:password|credential|secret|api[ -]?key)\b[^.\n]{0,60}\b(?:send|upload|share|forward)/,
    /\b(?:do not tell|don't tell|hide from|without (?:asking|confirmation)|silently)\b/,
  ];
  return patterns
    .map((pattern, index) => (pattern.test(low) ? `pattern_${index + 1}` : null))
    .filter(Boolean);
}

function safetyRiskIndicators(text) {
  const low = safetyText(text).toLowerCase();
  const patterns = [
    /\b(?:i\s+(?:will|am going to)|want to)\s+(?:kill|murder|harm|hurt)\b/,
    /\b(?:ship|buy|sell|make|manufacture)\b[^.\n]{0,80}\b(?:drug|cocaine|weapon|katana|chemical|explosive|exit bag)\b/,
    /\b(?:jailbreak|malicious use|illegal free movie|123movies)\b/,
    /\b(?:google authentication code|verification code|one[- ]time password|api key|credential)\b/,
    /\badversarial prompt attack\b/,
    /\b(?:sexual orientation|pregnan(?:cy|t)|sex video)\b/,
    /\b(?:women|men|race|ethnic(?:ity)?)\b[^.\n]{0,50}\b(?:superior|inferior)\b/,
    /\b(?:drunk|intoxicated)\b/,
  ];
  return patterns
    .map((pattern, index) => (pattern.test(low) ? `risk_pattern_${index + 1}` : null))
    .filter(Boolean);
}

function actionToolSpec(actionName, options = {}) {
  if (options.toolSpec && typeof options.toolSpec === "object") return options.toolSpec;
  if (!Array.isArray(META?.tools)) return null;
  return META.tools.find((tool) => tool?.name === actionName) || null;
}

function missingRequiredActionArguments(actionName, actionOrTool, options = {}) {
  const hasActionArgs = actionOrTool && typeof actionOrTool === "object" &&
    Object.prototype.hasOwnProperty.call(actionOrTool, "args");
  const hasOptionArgs = Object.prototype.hasOwnProperty.call(options, "args");
  if (!hasActionArgs && !hasOptionArgs) return [];
  const spec = actionToolSpec(actionName, options);
  const required = Array.isArray(spec?.schema?.required) ? spec.schema.required : [];
  if (!required.length) return [];
  const args = hasOptionArgs ? options.args : actionOrTool.args;
  return required.filter((name) => {
    const value = args && typeof args === "object" ? args[name] : null;
    return value == null || (typeof value === "string" && !value.trim());
  });
}

function actionSafetyPolicy(actionOrTool, request = "", options = {}) {
  const tool = typeof actionOrTool === "string"
    ? actionOrTool
    : actionOrTool?.name || actionOrTool?.tool || actionOrTool?.action?.tool;
  const actionName = typeof tool === "string" ? tool : "";
  const context = [
    safetyText(request),
    safetyText(options.untrustedText),
    safetyText(options.observation),
  ].filter(Boolean).join("\n");
  const indicators = promptInjectionIndicators(context);
  const riskIndicators = safetyRiskIndicators(context);
  const sideEffect = SIDE_EFFECT_TOOLS.has(actionName) || INTERACTIVE_TOOLS.has(actionName);
  const riskSensitive = sideEffect || RISK_SENSITIVE_READ_TOOLS.has(actionName);
  const destructive = DESTRUCTIVE_TOOLS.has(actionName);
  const missingArguments = missingRequiredActionArguments(actionName, actionOrTool, options);
  if (indicators.length && sideEffect) {
    return {
      policy_version: SAFETY_POLICY_VERSION,
      status: "blocked",
      severity: "high",
      requires_confirmation: false,
      tool: actionName,
      reason: "prompt_injection_or_secret_exfiltration_signal",
      indicators,
    };
  }
  if (riskIndicators.length && riskSensitive) {
    return {
      policy_version: SAFETY_POLICY_VERSION,
      risk_policy_version: RISK_POLICY_VERSION,
      status: "blocked",
      severity: "high",
      requires_confirmation: false,
      tool: actionName,
      reason: "high_risk_safety_signal",
      indicators,
      risk_indicators: riskIndicators,
    };
  }
  if (missingArguments.length) {
    return {
      policy_version: SAFETY_POLICY_VERSION,
      interaction_policy_version: INTERACTION_POLICY_VERSION,
      status: "clarification_required",
      severity: "medium",
      requires_confirmation: false,
      tool: actionName,
      reason: "missing_required_action_arguments",
      missing_arguments: missingArguments,
      indicators,
    };
  }
  if (sideEffect) {
    return {
      policy_version: SAFETY_POLICY_VERSION,
      status: "confirmation_required",
      severity: destructive ? "high" : "medium",
      requires_confirmation: true,
      tool: actionName,
      reason: destructive
        ? "destructive_or_irreversible_side_effect"
        : SIDE_EFFECT_TOOLS.has(actionName) ? "external_state_write" : "interactive_external_action",
      indicators,
    };
  }
  return {
    policy_version: SAFETY_POLICY_VERSION,
    status: "allowed",
    severity: "none",
    requires_confirmation: false,
    tool: actionName || null,
    reason: "read_only_or_abstention_action",
    indicators,
  };
}

function productivityLexicalSelect(query, dispatch = DISPATCH) {
  if (typeof query !== "string") return null;
  const names = new Set(dispatch?.dense_selector?.tool_names || []);
  // Long-horizon prompts contain a goal and serialized observation before the current action.
  // Route only on the latest instruction when that boundary exists; otherwise preserve the
  // ordinary single-turn query unchanged.
  const low = compactDispatchQuery(query).toLowerCase();
  // These are deliberately narrow intent guards, not learned-quality evidence.  They keep an
  // obvious email/Notion side-effect request from being routed to an unrelated timer/URL tool
  // when the tiny model is out of distribution.  The caller still validates the selected schema
  // and the demo never touches an external account.
  // Prefer the canonical public schema when both it and a legacy mobile alias are present.  The
  // metadata intentionally carries both names for compatibility, but evaluation and downstream
  // adapters use `send_email`/`notion_write`; selecting the alias here changes the argument shape
  // (`to`/`subject`/`body` versus `recipient`) and makes an otherwise valid action look wrong.
  const emailTool = names.has("send_email") ? "send_email" : names.has("email_send") ? "email_send" : null;
  const explicitEmailWord = /\b(?:email|e-mail)\b/.test(low);
  const emailAction = /\b(?:send|compose|write|draft)\b/.test(low);
  const directEmailCommand = /^\s*(?:email|e-mail)\b/.test(low);
  const explicitMailAction = /\bmail\b/.test(low) && emailAction;
  if (emailTool && ((explicitEmailWord && (emailAction || directEmailCommand)) || explicitMailAction)) {
    return {
      name: emailTool,
      route: "app_action",
      conf: 1,
      isStop: false,
      selection_policy: "productivity_email_intent_guard",
    };
  }
  const notionIntent = /\bnotion\b/.test(low) ||
    (/\b(?:save|write|add)\b/.test(low) && /\b(?:note|page)\b/.test(low));
  if (notionIntent && (names.has("notion_create_page") || names.has("notion_write"))) {
    const createPage = /\b(?:create|new|make)\b/.test(low) && names.has("notion_create_page");
    const name = createPage
      ? "notion_create_page"
      : names.has("notion_write") ? "notion_write" : "notion_create_page";
    return {
      name,
      route: "app_action",
      conf: 1,
      isStop: false,
      selection_policy: "productivity_notion_intent_guard",
    };
  }
  return null;
}

// A tiny fail-closed boundary for unmistakable direct-answer requests.  This is intentionally
// lexical and narrow: it prevents an OOD definition/acknowledgement prompt from becoming a GUI
// click, while all tool-shaped requests continue through the learned route and selector heads.
function semanticTextLexicalSelect(query) {
  if (typeof query !== "string") return null;
  const low = query.toLowerCase().trim();
  if (!low || /\b(?:email|e-mail|send|search|look up|open|visit|click|tap|type|write|save|create|run|execute|schedule)\b/.test(low)) {
    return null;
  }
  const directAnswer = /\bwhat does\b[\s\S]*\bmean\b|\bdefine\b|\bwhat is\b[\s\S]*\?|\bexplain\b/.test(low);
  const acknowledgement = /\b(?:thanks|thank you)\b[\s\S]*\b(?:all|done|no more|finished)\b|\bno action is needed\b|\bjust acknowledge\b/.test(low);
  if (!directAnswer && !acknowledgement) return null;
  return {
    isStop: true,
    route: "text",
    conf: 1,
    selection_policy: "semantic_text_safety_guard",
  };
}

function statefulActionLexicalSelect(query, dispatch = DISPATCH) {
  if (typeof query !== "string" || !/current state json\s*:/i.test(query)) return null;
  const action = compactDispatchQuery(query).toLowerCase();
  if (!action || action === query.toLowerCase()) return null;
  const names = new Set(dispatch?.dense_selector?.tool_names || []);
  const choose = (name) => names.has(name) ? {
    name,
    route: "computer_use",
    conf: 1,
    isStop: false,
    selection_policy: "stateful_action_lexical_guard",
  } : null;
  if (/\b(?:press|hit)\s+(?:the\s+)?enter\b/.test(action)) return choose("key_press");
  if (/\b(?:type|input|fill)\b/.test(action)) return choose("type_text");
  if (/\b(?:click|select)\b/.test(action)) return choose("click");
  return null;
}

function dispatchSelect(
  hiddenTensor,
  T,
  query = "",
  dispatch = DISPATCH,
  requestedSelector = REQUESTED_SELECTOR,
) {
  const mobile = mobileLexicalSelect(query, dispatch);
  if (mobile) return mobile;
  // Explicit URL navigation is a stable browser contract, not a visual click.  Keep this small
  // lexical safety adapter ahead of the learned heads so an OOD URL cannot become a side-effecting
  // GUI click; the receipt records this policy separately from learned selector accuracy.
  const urlTool = dispatch?.dense_selector?.tool_names?.includes("open_url");
  if (URL_LEXICAL_GUARD && urlTool &&
      /\b(?:open|go to|navigate to|visit|pull up)\b[\s\S]{0,120}https?:\/\//i.test(String(query))) {
    return {
      name: "open_url",
      route: "web_search",
      conf: 1,
      isStop: false,
      selection_policy: "explicit_url_safety_guard",
    };
  }
  // For a compound request, the first explicit web-search clause is the only safe first step when
  // the planner checkbox is off.  Do not pretend this is multi-step completion; the UI still
  // returns one action and the receipt must score the follow-up separately.
  if (
    dispatch?.dense_selector?.tool_names?.includes("web_search") &&
    /\b(?:search the web|look up|find information)\b/i.test(String(query)) &&
    /\b(?:then|after that|and save|and note)\b/i.test(String(query))
  ) {
    return {
      name: "web_search",
      route: "web_search",
      conf: 1,
      isStop: false,
      selection_policy: "compound_search_first_step_guard",
    };
  }
  if (PRODUCTIVITY_LEXICAL_GUARD && requestedSelector === "dense") {
    const productivity = productivityLexicalSelect(query, dispatch);
    if (productivity) return productivity;
  }
  const semanticText = semanticTextLexicalSelect(query);
  if (semanticText) return semanticText;
  const statefulAction = statefulActionLexicalSelect(query, dispatch);
  if (statefulAction) return statefulAction;
  if (requestedSelector === "retrieval") {
    const retrieved = retrievalSelectFromSidecar(query, dispatch?.retrieval_selector);
    if (retrieved) return retrieved;
  }
  const last = lastHidden(hiddenTensor, T);
  // 1. route head (5-way modality gate); the `text` route (stop_index) = abstain / direct answer.
  const R = dispatch.route_head;
  const rl = linrow(R.weight, R.bias, last);
  const ri = argmax(rl);
  if (ri === R.stop_index) return { isStop: true, route: R.routes[ri], conf: softmaxAt(rl, ri), selection_policy: "dense_selector" };
  // 2. dense selector: q = normalize(q_proj(last)); score_j = q · tool_matrix[j]; argmax.
  const S = dispatch.dense_selector;
  const q = linrow(S.q_proj_weight, S.q_proj_bias, last);
  if (S.normalize_query) { let n = 0; for (const x of q) n += x * x; n = Math.sqrt(n) || 1; for (let i = 0; i < q.length; i++) q[i] /= n; }
  const candidates = denseCandidateIndices(query, S, dispatch, requestedSelector);
  let bi = candidates.indices[0] ?? 0, bs = -Infinity;
  for (const j of candidates.indices) {
    const Tj = S.tool_matrix[j]; let a = 0; for (let i = 0; i < S.proj; i++) a += Tj[i] * q[i];
    if (a > bs) { bs = a; bi = j; }
  }
  return {
    name: S.tool_names[bi],
    route: R.routes[ri],
    conf: (bs + 1) / 2,
    isStop: false,
    selection_policy: candidates.policy,
    candidate_count: candidates.indices.length,
  };
}

// ---- argument grounding: learned pointer head + schema-typed extraction -------
//   q = arg_emb[arg_idx[arg]];  qs = start_W·q;  qe = end_W·q
//   start = argmax_t hidden[t]·qs;  end = argmax_{t>=start} hidden[t]·qe;  value = bytes[start..end]
//
// The pointer head covers open-ended copy arguments. Enums, numbers, booleans, and string
// arguments without a learned pointer embedding are grounded deterministically from their JSON
// Schema, matching localagent.agent.schema_decode. A required value that cannot be grounded makes
// the action incomplete (`args: null`) instead of silently substituting an empty string.
function matvec(M, v) {
  const d = v.length, out = new Float32Array(d);
  for (let i = 0; i < d; i++) { const Mi = M[i]; let a = 0; for (let j = 0; j < d; j++) a += Mi[j] * v[j]; out[i] = a; }
  return out;
}
function dotAt(H, t, d, q) { const off = t * d; let a = 0; for (let k = 0; k < d; k++) a += H[off + k] * q[k]; return a; }
function pointerSpan(arg, ids, H, T) {
  const ph = HEADS.pointer_head, d = META.d_model;
  const ai = ph.arg_idx[arg];
  if (ai == null) return "";
  const qs = matvec(ph.start_W, ph.arg_emb[ai]);
  const qe = matvec(ph.end_W, ph.arg_emb[ai]);
  let s = 0, sb = -Infinity;
  for (let t = 0; t < T; t++) { const v = dotAt(H, t, d, qs); if (v > sb) { sb = v; s = t; } }
  let e = s, eb = -Infinity;
  for (let t = s; t < T; t++) { const v = dotAt(H, t, d, qe); if (v > eb) { eb = v; e = t; } }
  try { return TOKENIZER.decode(ids.slice(s, e + 1), false); } catch { return ""; }
}

// Browser observations contain a bounded, pipe-delimited candidate list.  A pointer span is
// still the learned signal, but it can legally span several adjacent candidates when the model
// has not yet learned the DOM protocol.  Before accepting that span, rank the observed candidates
// against the task text and use the highest-scoring *observed* backend id.  This is a safety
// adapter, not a hidden label: it only reads the natural-language request and the candidate
// attributes already supplied by the environment.  If no candidate shares a task token, the
// learned pointer remains the fallback (important for unlabeled/icon-only controls).
const DOM_CANDIDATE_STOPWORDS = new Set([
  "a", "an", "and", "are", "at", "be", "by", "do", "for", "from", "in", "into", "is",
  "it", "latest", "most", "of", "on", "or", "please", "results", "the", "this", "to", "use",
  "with", "your",
]);
const DOM_CANDIDATE_FIELDS = new Set([
  "target_id", "operation", "tag", "role", "id", "title", "aria_label", "placeholder", "value",
  "text",
]);

function domCandidateTokens(value) {
  return String(value).toLowerCase().match(/[a-z0-9]+/g) || [];
}

function browserCandidateTargetId(prompt) {
  const marker = "Browser DOM candidates:";
  const markerIndex = String(prompt).lastIndexOf(marker);
  if (markerIndex < 0) return null;
  let task = String(prompt).slice(0, markerIndex);
  // In a trajectory, earlier tool responses may contain their own snapshots.  They are state
  // history, not the current request; isolate the initial task before scoring the latest view.
  const earlierMarkerIndex = task.indexOf(marker);
  if (earlierMarkerIndex >= 0) task = task.slice(0, earlierMarkerIndex);
  const instruction = task.match(/(?:next required action|instruction|request)\s*:\s*([\s\S]*)$/i);
  if (instruction) task = instruction[1];
  const taskTokens = new Set(
    domCandidateTokens(task).filter((token) => token.length >= 3 && !DOM_CANDIDATE_STOPWORDS.has(token))
  );
  if (!taskTokens.size) return null;
  const segments = String(prompt).slice(markerIndex + marker.length).split(/\s*\|\s*/);
  let best = null;
  for (let index = 0; index < segments.length; index++) {
    const segment = segments[index];
    const match = segment.match(/(?:^|\s)target_id=([^\s|]+)/i);
    if (!match) continue;
    const targetId = match[1];
    const candidateText = segment.replace(/\b([a-z_]+)=([^\s|]+)/gi, (whole, key, value) => {
      return DOM_CANDIDATE_FIELDS.has(String(key).toLowerCase()) ? ` ${value} ` : whole;
    });
    const candidateTokens = new Set(domCandidateTokens(candidateText));
    let overlap = 0;
    for (const token of taskTokens) if (candidateTokens.has(token)) overlap += 1;
    if (!overlap) continue;
    const score = overlap / taskTokens.size;
    if (!best || score > best.score || (score === best.score && index < best.index)) {
      best = { targetId, score, index };
    }
  }
  return best?.targetId || null;
}

const PATH_HINTS = new Set([
  "path", "file", "filepath", "filename", "source", "src", "dest", "destination", "target",
]);
const URL_HINTS = new Set(["url", "link", "website", "site", "href", "address", "endpoint"]);
const ENTITY_HINTS = new Set([
  "name", "recipient", "person", "city", "location", "user", "author", "artist", "assignee",
  "owner", "contact", "to", "sender", "app", "app_name",
]);
const QUOTED_HINTS = new Set([
  "message", "subject", "title", "content", "body", "text", "note", "summary", "caption",
  "comment", "description", "label",
]);

function groundingPools(prompt) {
  // Stateful prompts carry a goal and JSON observation before the actionable instruction.  Those
  // JSON keys/values are not argument candidates (for example, `"app"` must not become an
  // app_name).  Keep the legacy whole-prompt behavior for ordinary single-turn requests.
  const actionMatch = prompt.match(/(?:next required action|current step(?:\s+[^:]+)?|current action|instruction)\s*:\s*([\s\S]*)$/i);
  const source = actionMatch ? actionMatch[1] : prompt;
  const body = source.trim().split(/\s+/).slice(1).join(" ");
  const arithmetic = source.match(/\d+\s*[-+*/]\s*\d+(?:\s*[-+*/]\s*\d+)*/);
  return {
    quoted: Array.from(source.matchAll(/'([^']+)'|"([^"]+)"/g), (m) => m[1] || m[2]),
    path: Array.from(
      source.matchAll(/[A-Za-z0-9_.\-/]+\/[A-Za-z0-9_.\-/]*|[A-Za-z0-9_.\-/]+\.[A-Za-z0-9]{1,5}\b/g),
      (m) => m[0].replace(/\.$/, "")
    ),
    url: Array.from(
      source.matchAll(/(?:https?:\/\/)?[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?:\/[\w./-]*)?/g),
      (m) => m[0].replace(/\.$/, "")
    ),
    email: Array.from(
      source.matchAll(/[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/g),
      (m) => m[0]
    ),
    caps: Array.from(body.matchAll(/(?:[A-Z][a-z]+)(?:\s+[A-Z][a-z]+)*/g), (m) => m[0]),
    number: Array.from(source.matchAll(/-?\d+(?:\.\d+)?/g), (m) => m[0]),
    arithmetic: arithmetic ? [arithmetic[0].replace(/\s+/g, "")] : [],
  };
}

function popGrounding(pool) {
  return pool.length ? pool.shift() : null;
}

function stripGrounding(value) {
  const cleaned = value
    .replace(/^[^A-Za-z0-9'"]+/, "")
    .replace(/\s*(online|please)?\s*[.?!]*$/i, "")
    .trim();
  return /^(['"])(.*)\1$/.test(cleaned) ? cleaned.slice(1, -1).trim() : cleaned;
}

function freeTextGrounding(prompt) {
  const actionMatch = prompt.match(/(?:next required action|current step(?:\s+[^:]+)?|current action|instruction)\s*:\s*([\s\S]*)$/i);
  const source = actionMatch ? actionMatch[1].trim() : prompt;
  const low = source.toLowerCase();
  const tails = [];
  for (const prep of ["for", "about", "to", "in", "on", "of", "with", "from"]) {
    const index = low.indexOf(` ${prep} `);
    if (index >= 0) {
      const tail = stripGrounding(source.slice(index + prep.length + 2));
      if (tail) tails.push(tail);
    }
  }
  if (tails.length) return tails.sort((a, b) => b.length - a.length)[0];
  const words = source.split(/\s+/);
  return words.length > 1 ? stripGrounding(words.slice(1).join(" ")) : null;
}

function cuePresent(prompt, cue) {
  const escaped = cue.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`\\b${escaped.replace(/\s+/g, "\\s+")}\\b`, "i").test(prompt);
}

function stateArgumentValue(prompt, name) {
  const match = String(prompt).match(
    /current state json\s*:\s*([\s\S]*?)(?=\s+(?:next required action|current step|current action|instruction)\s*:|$)/i
  );
  if (!match) return null;
  let state;
  try {
    state = JSON.parse(match[1].trim());
  } catch {
    return null;
  }
  const visit = (value) => {
    if (!value || typeof value !== "object") return null;
    if (Object.hasOwn(value, name) && value[name] != null) return value[name];
    for (const child of Object.values(value)) {
      const found = visit(child);
      if (found != null) return found;
    }
    return null;
  };
  const value = visit(state);
  return typeof value === "string" || typeof value === "number" || typeof value === "boolean"
    ? value
    : null;
}

function fillSchemaArg(prompt, name, schema, pools, required, pointerValue) {
  const type = schema.type || "string";
  const format = schema.format;
  const tail = () => required ? freeTextGrounding(prompt) : null;
  const statefulPrompt = /(?:next required action|current step(?:\s+[^:]+)?|current action|instruction)\s*:/i.test(prompt);
  const actionText = compactDispatchQuery(prompt);
  const action = actionText.toLowerCase();
  if (statefulPrompt && /\b(?:send|submit|dispatch)\b/.test(action)) {
    const stateValue = stateArgumentValue(prompt, name);
    if (stateValue != null) return stateValue;
  }

  if (name === "target") {
    const targetMatch = actionText.match(
      /(?:click|select|tap)\s+(?:on\s+)?((?:the\s+)?[A-Za-z][A-Za-z ]*?)(?=\s+(?:in|on|at)\s+|\s*[.!?]|$)/i
    );
    if (targetMatch?.[1]) {
      const target = stripGrounding(targetMatch[1]);
      return target.toLowerCase().startsWith("the ") ? target : `the ${target}`;
    }
  }

  if (name === "app_name") {
    const appMatch = prompt.match(
      /\b(?:open|launch|start|bring up)\s+(?:the\s+)?([A-Za-z][\w.-]*(?:\s+[A-Za-z][\w.-]*)*?)\s+(?:app|application)\b/i
    );
    if (appMatch) return appMatch[1].trim();
    const shortAppMatch = prompt.match(/\b(?:open|launch|start)\s+(?:the\s+)?([A-Z][\w.-]*)\b/);
    if (shortAppMatch) return shortAppMatch[1];
  }

  // Email recipients are not necessarily capitalized entities and often share the sentence with
  // a subject/body.  Prefer the explicit address, then stop a natural-language recipient before
  // the next email clause instead of trusting an under-trained pointer span.
  if (name === "recipient" || name === "to") {
    if (pools.email.length) return pools.email[0];
    const recipientMatch = prompt.match(
      /\b(?:email|send|write|compose|drop|shoot)\s+(?:an?\s+)?(?:email\s+)?(?:to\s+)?(.+?)(?=\s+(?:the|with|about|subject|body|saying)\b|[.!?]|$)/i
    );
    if (recipientMatch?.[1]) return stripGrounding(recipientMatch[1]);
  }

  // Notion/content prompts frequently use an unquoted value after an explicit cue.  Recover that
  // cue before the generic tail fallback, while still allowing quoted pools and learned pointers.
  if (name === "content") {
    const contentMatch = prompt.match(
      /\b(?:content|note|text|saying)\s+(?:is\s+)?(.+?)(?:[.!?]|$)/i
    );
    if (contentMatch?.[1]) return stripGrounding(contentMatch[1]);
    const notionSaveMatch = prompt.match(
      /\b(?:save|write|add)\s+(?:the\s+)?(.+?)\s+(?:to|in)\s+Notion\b/i
    );
    if (notionSaveMatch?.[1]) return stripGrounding(notionSaveMatch[1]);
  }

  if (schema.enum) {
    return schema.enum.find((value) => cuePresent(prompt, String(value))) ?? null;
  }
  if (type === "boolean") {
    if (["turn on", "enable", "yes", "true", "activate", "on"].some((cue) =>
      cuePresent(prompt, cue))) return true;
    if (["turn off", "disable", "no", "false", "deactivate", "off"].some((cue) =>
      cuePresent(prompt, cue))) return false;
    return null;
  }
  if (format === "arithmetic" || name.includes("express")) {
    return popGrounding(pools.arithmetic);
  }
  if (type === "integer" || type === "number") {
    const raw = popGrounding(pools.number);
    if (raw == null) return null;
    const value = type === "integer" ? Math.trunc(Number(raw)) : Number(raw);
    return Number.isFinite(value) ? value : null;
  }

  const copied = pointerValue && pointerValue.trim();
  // Pointer spans can include the serialized user marker and instruction prefix when the
  // action graph has not learned the exact URL span.  A URL argument must never carry that
  // protocol text into the browser tool; recover only the URL-shaped substring and keep the
  // normal schema pool as the fallback for non-URL strings.
  if (format === "url" && copied) {
    if (pools.url.length) return pools.url[0];
    const copiedUrl = copied.match(/https?:\/\/[^\s'"<>]+/i)?.[0]?.replace(/[.,!?]+$/, "");
    if (copiedUrl) return copiedUrl;
  }
  // A pointer span is only trusted for quoted fields when it exactly reproduces one quoted
  // candidate.  This keeps a pointer head trained on an older schema from stitching together
  // adjacent title/content phrases in new email/Notion contracts; the schema extractor can then
  // consume the deterministic quoted pool in field order.
  // App names are often unquoted in accessibility-style instructions.  The pointer head can
  // otherwise copy the generic word "app"; prefer the deterministic capitalized entity pool.
  const copiedIsCandidate = pools.quoted.includes(copied) || pools.path.includes(copied) ||
    pools.url.includes(copied);
  if (
    copied && name !== "app_name" &&
    (!statefulPrompt || copiedIsCandidate) &&
    (!QUOTED_HINTS.has(name) || pools.quoted.includes(copied))
  ) {
    return copied;
  }
  if (format === "quoted") return popGrounding(pools.quoted) || tail();
  if (format === "path") return popGrounding(pools.path);
  if (format === "url") return popGrounding(pools.url);
  if (PATH_HINTS.has(name)) return popGrounding(pools.path);
  if (URL_HINTS.has(name)) return popGrounding(pools.url);
  if (QUOTED_HINTS.has(name)) return popGrounding(pools.quoted) || tail();
  if (ENTITY_HINTS.has(name)) return popGrounding(pools.caps) || tail();
  return popGrounding(pools.quoted) || tail();
}

function groundedArgsValid(args, schema) {
  if (!args || typeof args !== "object" || Array.isArray(args)) return false;
  const properties = schema.properties || {};
  for (const name of schema.required || []) {
    if (!(name in args)) return false;
  }
  for (const [name, value] of Object.entries(args)) {
    const property = properties[name];
    if (!property) return false;
    if (property.enum && !property.enum.includes(value)) return false;
    if (property.type === "string" && typeof value !== "string") return false;
    if (property.type === "integer" && !Number.isInteger(value)) return false;
    if (property.type === "number" && (typeof value !== "number" || !Number.isFinite(value))) {
      return false;
    }
    if (property.type === "boolean" && typeof value !== "boolean") return false;
  }
  return true;
}

function groundFromSchema(prompt, schema, pointerValues = {}) {
  const properties = schema.properties || {};
  const required = new Set(schema.required || []);
  const effectivePointerValues = { ...pointerValues };
  const candidateTargetId = browserCandidateTargetId(prompt);
  if (candidateTargetId && properties.target_id) {
    effectivePointerValues.target_id = candidateTargetId;
  }
  const pools = groundingPools(prompt);
  const args = {};
  for (const [arg, argSchema] of Object.entries(properties)) {
    const value = fillSchemaArg(
      prompt, arg, argSchema, pools, required.has(arg), effectivePointerValues[arg] || null
    );
    if (value != null && value !== "") {
      args[arg] = value;
    } else if (required.has(arg)) {
      return null;
    }
  }
  return groundedArgsValid(args, schema) ? args : null;
}

function groundArgs(tool, prompt, ids, hiddenTensor, T) {
  const spec = (META.tools || []).find((t) => t.name === tool);
  if (!spec) return null;
  const schema = spec.schema || {};
  const H = hiddenTensor.data;
  const pointerValues = {};
  for (const arg of Object.keys(schema.properties || {})) {
    if (HEADS.pointer_head.arg_idx[arg] != null) {
      pointerValues[arg] = pointerSpan(arg, ids, H, T);
    }
  }
  return groundFromSchema(prompt, schema, pointerValues);
}

// ---- complete-action policy baselines ------------------------------------
function validateActionForMeta(action, meta = META) {
  if (!action || typeof action !== "object") {
    return { valid: false, error: "action_not_object" };
  }
  if (action.abstain === true) return { valid: true, error: null };
  const spec = (meta?.tools || []).find((tool) => tool.name === action.tool);
  if (!spec) return { valid: false, error: "unknown_tool" };
  if (!groundedArgsValid(action.args, spec.schema || {})) {
    return { valid: false, error: "invalid_arguments" };
  }
  return { valid: true, error: null };
}

function canonicalActionJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalActionJson).join(",")}]`;
  if (value && typeof value === "object") {
    const entries = Object.keys(value).sort().map(
      (key) => `${JSON.stringify(key)}:${canonicalActionJson(value[key])}`
    );
    return `{${entries.join(",")}}`;
  }
  return JSON.stringify(value);
}

function canonicalToolCompletion(action, meta = META) {
  const payload = { arguments: action.args, name: action.tool };
  const open = meta.markers.tool_call_open.text;
  const close = meta.markers.tool_call_close.text;
  return `${open}${canonicalActionJson(payload)}${close}`;
}

function invalidGeneratedAction(parseKind, generatedText, error) {
  return {
    action: { abstain: false, tool: null, args: null },
    generated_text: generatedText,
    parse_kind: parseKind,
    parse_failure: true,
    parse_error: error,
    schema_valid: false,
    validation_failure: false,
    validation_error: null,
  };
}

function parseGeneratedAction(generatedText, stopReason, meta = META) {
  const trimmed = generatedText.trim();
  const open = meta?.markers?.tool_call_open?.text || "<tool_call>";
  const close = meta?.markers?.tool_call_close?.text || "</tool_call>";
  let payloadText = trimmed;
  let parseKind = "raw_json";
  if (trimmed.startsWith(open) || trimmed.endsWith(close)) {
    if (!trimmed.startsWith(open) || !trimmed.endsWith(close)) {
      return invalidGeneratedAction("framed_json", generatedText, "incomplete_tool_call_frame");
    }
    payloadText = trimmed.slice(open.length, trimmed.length - close.length);
    parseKind = "framed_json";
  } else if (!trimmed.startsWith("{")) {
    // A model trained on mixed tool and text turns may emit ordinary answer text. Once EOS is
    // reached (or the constrained grammar observes a non-tool first token), that is a valid
    // no-tool decision, not JSON masquerading as a tool call.
    if (stopReason === "eos" || (stopReason === "candidate_terminal" && !trimmed)) {
      return {
        action: { abstain: true },
        generated_text: generatedText,
        parse_kind: "direct_text",
        parse_failure: false,
        parse_error: null,
        schema_valid: true,
        validation_failure: false,
        validation_error: null,
      };
    }
    return invalidGeneratedAction("unframed_text", generatedText, "completion_is_not_json");
  }

  let payload;
  try {
    payload = JSON.parse(payloadText);
  } catch (error) {
    return invalidGeneratedAction(parseKind, generatedText, `invalid_json: ${error.message}`);
  }
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return invalidGeneratedAction(parseKind, generatedText, "json_root_not_object");
  }

  let action;
  if (payload.abstain === true) {
    action = { abstain: true };
  } else if (typeof payload.name === "string" && "arguments" in payload) {
    action = { tool: payload.name, args: payload.arguments };
  } else if (typeof payload.tool === "string" && "args" in payload) {
    action = { tool: payload.tool, args: payload.args };
  } else {
    return invalidGeneratedAction(parseKind, generatedText, "json_action_shape_invalid");
  }
  const validation = validateActionForMeta(action, meta);
  return {
    action,
    generated_text: generatedText,
    parse_kind: parseKind,
    parse_failure: false,
    parse_error: null,
    schema_valid: validation.valid,
    validation_failure: !validation.valid,
    validation_error: validation.error,
  };
}

function buildCandidateTrie(candidates) {
  const root = { children: new Map(), terminal: null };
  for (const candidate of candidates) {
    let node = root;
    for (const tokenId of candidate.token_ids) {
      if (!node.children.has(tokenId)) {
        node.children.set(tokenId, { children: new Map(), terminal: null });
      }
      node = node.children.get(tokenId);
    }
    if (node.terminal && canonicalActionJson(node.terminal.action) !==
        canonicalActionJson(candidate.action)) {
      throw new Error("Two constrained actions encode to the same token sequence.");
    }
    node.terminal = candidate;
  }
  return root;
}

function groundedActionCandidates(query, meta = META, tokenizer = TOKENIZER) {
  // EOS is the canonical no-tool candidate. Every constrained next-token decision, including the
  // root, is therefore masked to an edge in this trie.
  const candidates = [{
    action: { abstain: true },
    completion: "",
    token_ids: [tokenizer.eosId],
  }];
  for (const spec of meta?.tools || []) {
    const args = groundFromSchema(query, spec.schema || {});
    if (args == null) continue;
    const action = { tool: spec.name, args };
    const completion = canonicalToolCompletion(action, meta);
    const tokenIds = tokenizer.encode(completion);
    if (!tokenIds.length) continue;
    candidates.push({ action, completion, token_ids: tokenIds });
  }
  return candidates;
}

function tensorLastTokenLogits(logitsTensor, sequenceLength) {
  const vocab = logitsTensor.dims?.[logitsTensor.dims.length - 1] || META.vocab_size;
  const offset = (sequenceLength - 1) * vocab;
  return logitsTensor.data.subarray
    ? logitsTensor.data.subarray(offset, offset + vocab)
    : Array.from(logitsTensor.data).slice(offset, offset + vocab);
}

function argmaxAllowed(logits, allowedTokenIds) {
  let bestToken = null;
  let bestLogit = -Infinity;
  for (const tokenId of allowedTokenIds) {
    const value = logits[tokenId];
    if (bestToken == null || value > bestLogit || (value === bestLogit && tokenId < bestToken)) {
      bestLogit = value;
      bestToken = tokenId;
    }
  }
  if (bestToken == null) throw new Error("Constrained decoder reached an empty trie node.");
  return bestToken;
}

function autoregressiveTiming(
  started,
  tokenizedAt,
  inferenceMs,
  decodeControlMs,
  parseValidateMs,
  tokenTimes
) {
  const completedAt = performance.now();
  return {
    tokenize_ms: tokenizedAt - started,
    inference_ms: inferenceMs,
    decode_control_ms: decodeControlMs,
    dispatch_ms: 0,
    parse_validate_ms: parseValidateMs,
    // TTFT begins when the tokenized request is submitted to inference. TTFA deliberately starts
    // earlier and includes tokenization plus parse/validation because it is user-visible latency.
    ttft_ms: tokenTimes.length ? tokenTimes[0] - tokenizedAt : null,
    tpot_ms: tokenTimes.length > 1
      ? (tokenTimes[tokenTimes.length - 1] - tokenTimes[0]) / (tokenTimes.length - 1)
      : null,
    ttfa_ms: completedAt - started,
  };
}

function generationLimit(promptIds, requestedMax) {
  const requested = Math.max(1, Number.parseInt(requestedMax, 10) || 192);
  if (!META.max_seq_len) return requested;
  const available = META.max_seq_len - promptIds.length;
  if (available <= 0) {
    throw new Error(
      `Prompt has ${promptIds.length} tokens but model max_seq_len is ${META.max_seq_len}.`
    );
  }
  return Math.min(requested, available);
}

async function rawAutoregressiveAction(query, options = {}) {
  const started = performance.now();
  const prompt = cachedActionPrompt(query, options.targetInputTokens);
  const promptIds = prompt.ids;
  const inputBytes = prompt.inputBytes;
  const tokenizedAt = performance.now();
  const maxOutputTokens = generationLimit(promptIds, options.maxOutputTokens);
  const bundle = await ensureCachedDecodeSessions();
  const runner = createCachedAutoregressiveRunner(bundle, promptIds);
  const selectionOptions = {
    ...options,
    random: options.random || seededTokenRandom(
      String(options.seed ?? `raw:${query}`)
    ),
  };
  const generated = [];
  const visible = [];
  const tokenTimes = [];
  let inferenceMs = 0;
  let decodeControlMs = 0;
  let stopReason = "max_tokens";

  try {
    for (let step = 0; step < maxOutputTokens; step++) {
      const inferenceStarted = performance.now();
      const decision = step === 0
        ? await runner.prefill()
        : await runner.decode(generated.at(-1));
      const inferenceEnded = performance.now();
      inferenceMs += inferenceEnded - inferenceStarted;
      const controlStarted = performance.now();
      const tokenId = selectTokenFromLogits(decision.logits, selectionOptions);
      generated.push(tokenId);
      if (tokenId !== TOKENIZER.eosId) visible.push(tokenId);
      tokenTimes.push(performance.now());
      decodeControlMs += performance.now() - controlStarted;
      if (tokenId === TOKENIZER.eosId) {
        stopReason = "eos";
        break;
      }
    }
  } finally {
    runner.dispose();
  }

  const generatedText = TOKENIZER.decode(visible, false);
  const parseStarted = performance.now();
  const parsed = parseGeneratedAction(generatedText, stopReason);
  const parseValidateMs = performance.now() - parseStarted;
  const timing = autoregressiveTiming(
    started, tokenizedAt, inferenceMs, decodeControlMs, parseValidateMs, tokenTimes
  );
  return {
    ...parsed.action,
    policy: ACTION_POLICIES.RAW_AR,
    generated_text: parsed.generated_text,
    parse_kind: parsed.parse_kind,
    parse_failure: parsed.parse_failure,
    parse_error: parsed.parse_error,
    schema_valid: parsed.schema_valid,
    validation_failure: parsed.validation_failure,
    validation_error: parsed.validation_error,
    stop_reason: stopReason,
    output_tokens: visible.length,
    decode_steps: generated.length,
    inference_passes: generated.length,
    prefill_passes: generated.length ? 1 : 0,
    cached_decode_passes: Math.max(0, generated.length - 1),
    decode_cache: true,
    decode_strategy: CACHED_DECODE_STRATEGY,
    token_selection: Number(options.temperature ?? 0) > 0
      ? "temperature_top_k_sampling_from_logits"
      : "deterministic_argmax_from_logits",
    next_token_role: "compatibility_argmax_cross_check",
    prompt_contract: prompt.promptContract,
    tool_catalog_size: prompt.toolCatalogSize,
    input_tokens: promptIds.length,
    input_bytes: inputBytes,
    natural_input_tokens: prompt.naturalInputTokens,
    context_padding_tokens: prompt.paddingTokens,
    context_padding_placement: prompt.contextPaddingPlacement,
    decision_input_tokens: prompt.decisionInputTokens,
    decision_feature_index: prompt.decisionFeatureIndex,
    timing,
    ms: timing.ttfa_ms,
  };
}

async function constrainedAutoregressiveAction(query, options = {}) {
  const started = performance.now();
  const prompt = cachedActionPrompt(query, options.targetInputTokens);
  const promptIds = prompt.ids;
  const inputBytes = prompt.inputBytes;
  const candidates = groundedActionCandidates(query);
  if (!candidates.length) {
    throw new Error("Prompt-grounded constrained decoder produced no schema-valid candidates.");
  }
  const root = buildCandidateTrie(candidates);
  let node = root;
  const tokenizedAt = performance.now();
  const maxOutputTokens = generationLimit(promptIds, options.maxOutputTokens);
  const bundle = await ensureCachedDecodeSessions();
  const runner = createCachedAutoregressiveRunner(bundle, promptIds);
  const selectionOptions = {
    ...options,
    random: options.random || seededTokenRandom(
      String(options.seed ?? `constrained:${query}`)
    ),
  };
  const generated = [];
  const tokenTimes = [];
  let inferenceMs = 0;
  let decodeControlMs = 0;
  let stopReason = "max_tokens";

  try {
    for (let step = 0; step < maxOutputTokens; step++) {
      const inferenceStarted = performance.now();
      const decision = step === 0
        ? await runner.prefill()
        : await runner.decode(generated.at(-1));
      const inferenceEnded = performance.now();
      inferenceMs += inferenceEnded - inferenceStarted;
      const controlStarted = performance.now();
      const tokenId = selectTokenFromLogits(
        decision.logits,
        selectionOptions,
        node.children.keys()
      );
      generated.push(tokenId);
      tokenTimes.push(performance.now());
      node = node.children.get(tokenId);
      decodeControlMs += performance.now() - controlStarted;
      if (node.terminal) {
        stopReason = "candidate_terminal";
        break;
      }
    }
  } finally {
    runner.dispose();
  }

  const visible = generated.filter((tokenId) => tokenId !== TOKENIZER.eosId);
  const generatedText = TOKENIZER.decode(visible, false);
  const parseStarted = performance.now();
  const parsed = parseGeneratedAction(generatedText, stopReason);
  const parseValidateMs = performance.now() - parseStarted;
  const timing = autoregressiveTiming(
    started, tokenizedAt, inferenceMs, decodeControlMs, parseValidateMs, tokenTimes
  );
  return {
    ...parsed.action,
    policy: ACTION_POLICIES.CONSTRAINED_AR,
    generated_text: parsed.generated_text,
    parse_kind: parsed.parse_kind,
    parse_failure: parsed.parse_failure,
    parse_error: parsed.parse_error,
    schema_valid: parsed.schema_valid,
    validation_failure: parsed.validation_failure,
    validation_error: parsed.validation_error,
    stop_reason: stopReason,
    output_tokens: visible.length,
    decode_steps: generated.length,
    inference_passes: generated.length,
    prefill_passes: generated.length ? 1 : 0,
    cached_decode_passes: Math.max(0, generated.length - 1),
    decode_cache: true,
    decode_strategy: CACHED_DECODE_STRATEGY,
    token_selection: Number(options.temperature ?? 0) > 0
      ? "temperature_top_k_sampling_from_allowed_logits"
      : "deterministic_argmax_from_allowed_logits",
    next_token_role: "unconstrained_compatibility_argmax_cross_check",
    prompt_contract: prompt.promptContract,
    tool_catalog_size: prompt.toolCatalogSize,
    candidate_count: candidates.length,
    input_tokens: promptIds.length,
    input_bytes: inputBytes,
    natural_input_tokens: prompt.naturalInputTokens,
    context_padding_tokens: prompt.paddingTokens,
    context_padding_placement: prompt.contextPaddingPlacement,
    decision_input_tokens: prompt.decisionInputTokens,
    decision_feature_index: prompt.decisionFeatureIndex,
    timing,
    ms: timing.ttfa_ms,
  };
}

// ---- single grounded call -------------------------------------------------
async function structuredAction(query, options = {}) {
  const t0 = performance.now();
  const prompt = actionPrompt(
    query,
    options.targetInputTokens,
    "trailing_compute"
  );
  const ids = prompt.ids;
  const inputBytes = prompt.inputBytes;
  const t1 = performance.now();
  const out = await forward(ids);
  const t2 = performance.now();
  const sel = dispatchSelect(out.hidden, prompt.decisionInputTokens, query);
  const result = sel.isStop
    ? { abstain: true, route: sel.route, conf: sel.conf }
    : {
        tool: sel.name,
        route: sel.route,
        args: groundArgs(
          sel.name,
          query,
          ids,
          out.hidden,
          prompt.decisionInputTokens
        ),
        conf: sel.conf,
        selection_policy: sel.selection_policy,
      };
  const t3 = performance.now();
  const validation = validateActionForMeta(result);
  const t4 = performance.now();
  const timing = {
    tokenize_ms: t1 - t0,
    inference_ms: t2 - t1,
    decode_control_ms: 0,
    dispatch_ms: t3 - t2,
    parse_validate_ms: t4 - t3,
    ttft_ms: null,
    tpot_ms: null,
    ttfa_ms: t4 - t0,
  };
  return {
    ...result,
    policy: ACTION_POLICIES.STRUCTURED,
    generated_text: null,
    parse_kind: "structured_heads",
    parse_failure: false,
    parse_error: null,
    schema_valid: validation.valid,
    validation_failure: !validation.valid,
    validation_error: validation.error,
    stop_reason: "one_forward",
    output_tokens: 0,
    decode_steps: 0,
    inference_passes: 1,
    decode_cache: null,
    decode_strategy: "one_forward_structured_heads",
    input_tokens: ids.length,
    input_bytes: inputBytes,
    natural_input_tokens: prompt.naturalInputTokens,
    context_padding_tokens: prompt.paddingTokens,
    context_padding_placement: prompt.contextPaddingPlacement,
    decision_input_tokens: prompt.decisionInputTokens,
    decision_feature_index: prompt.decisionFeatureIndex,
    timing,
    ms: timing.ttfa_ms,
  };
}

async function callPolicyOnce(query, policy = ACTION_POLICIES.STRUCTURED, options = {}) {
  await prepareActionPolicy(policy);
  if (policy === ACTION_POLICIES.STRUCTURED) return structuredAction(query, options);
  if (policy === ACTION_POLICIES.RAW_AR) return rawAutoregressiveAction(query, options);
  return constrainedAutoregressiveAction(query, options);
}

async function callOnce(query) {
  return callPolicyOnce(query, ACTION_POLICIES.STRUCTURED);
}

// ---- planner rollout ------------------------------------------------------
async function planRollout(query, maxSteps = 4) {
  const steps = [];
  const stepTimings = [];
  const t0 = performance.now();
  for (let i = 0; i < maxSteps; i++) {
    const s0 = performance.now();
    let ids = renderContext(query, steps);
    const s1 = performance.now();
    // Once the first search step has returned, expose the intended follow-up clause to the
    // selector instead of repeatedly reclassifying the original compound request.  This is a
    // bounded planner-context adapter; it does not claim that the browser has executed a real
    // search or written to Notion.
    let dispatchQuery = query;
    if (steps.length && steps.at(-1)?.tool === "web_search" && /\b(?:Notion|note|save)\b/i.test(query)) {
      dispatchQuery = "Save the search result to Notion.";
    }
    // The follow-up selector should read the normalized clause, not a long history whose original
    // imperative can dominate the frozen 10M parameter feature.  This extra forward is bounded to
    // planner follow-ups and remains a local simulated workflow.
    if (dispatchQuery !== query) ids = actionPrompt(dispatchQuery, undefined, "trailing_compute").ids;
    const out = await forward(ids);
    const s2 = performance.now();
    const sel = dispatchSelect(out.hidden, ids.length, dispatchQuery);
    if (sel.isStop) {
      const s3 = performance.now();
      stepTimings.push({
        step: i,
        input_tokens: ids.length,
        stopped: true,
        tokenize_ms: s1 - s0,
        inference_ms: s2 - s1,
        dispatch_ms: s3 - s2,
        ttfa_ms: s3 - s0,
      });
      break;
    }
    const groundingText = [query, ...steps.map((step) => step.response || "")].join(" ");
    let args = groundArgs(sel.name, groundingText, ids, out.hidden, ids.length);
    if (sel.name === "notion_write" && steps.at(-1)?.tool === "web_search") {
      const resultText = String(steps.at(-1)?.response || "").replace(/^result:\s*/i, "").trim();
      if (resultText) args = { ...(args || {}), content: resultText };
    }
    const safety = actionSafetyPolicy(sel.name, query, {
      args,
      toolSpec: (META?.tools || []).find((tool) => tool?.name === sel.name),
      untrustedText: steps.at(-1)?.response || "",
    });
    const s3 = performance.now();
    steps.push({ tool: sel.name, route: sel.route, args: safety.status === "blocked" ? null : args,
      conf: sel.conf, selection_policy: sel.selection_policy, safety,
      // A state-changing step is staged for confirmation; do not fabricate a successful tool
      // response that could make a later planner step believe the external write happened.
      response: safety.status === "allowed" ? simResponse(sel.name, args) : null });
    stepTimings.push({
      step: i,
      input_tokens: ids.length,
      stopped: false,
      tokenize_ms: s1 - s0,
      inference_ms: s2 - s1,
      dispatch_ms: s3 - s2,
      ttfa_ms: s3 - s0,
    });
    if (safety.status !== "allowed") break;
    // The public demo's search→Notion workflow has two explicit milestones.  Stop after the
    // second accepted tool instead of repeatedly replaying the first clause when the tiny model
    // has no learned STOP state for the simulated response.
    if (
      steps.length >= 2 &&
      steps.at(-2)?.tool === "web_search" &&
      steps.at(-1)?.tool === "notion_write" &&
      /\b(?:Notion|note|save)\b/i.test(query)
    ) break;
  }
  return { steps, stepTimings, ms: performance.now() - t0 };
}

// A compact simulated tool response so downstream steps have context.
function simResponse(tool, args) {
  if (/read_file|grep|list_dir|find/.test(tool)) return Object.values(args)[0] || "ok";
  if (/search|news|http|open_url|define/.test(tool)) return "result: " + (Object.values(args)[0] || "");
  return "ok";
}

// ---- UI -------------------------------------------------------------------
const $ = (id) => document.getElementById(id);

function setStatus(cls, text, backend) {
  const s = $("status");
  s.className = "status " + cls;
  $("status-text").textContent = text;
  const b = $("backend-badge");
  if (backend) { b.hidden = false; b.textContent = backend.toUpperCase(); }
}

function renderCall(step, idx) {
  const div = document.createElement("div");
  div.className = "call" + (step.abstain ? " abstain" : "");
  const conf = step.conf != null ? `<span class="conf">${(step.conf * 100).toFixed(0)}%</span>` : "";
  const route = step.route ? `<span class="route">${step.route}</span>` : "";
  if (step.abstain) {
    div.innerHTML = `${conf}${route}<span class="tool">— abstains (no tool needed)</span>`;
  } else {
    const ix = idx != null ? `<span class="step-index">${idx + 1}.</span>` : "";
    div.innerHTML = `${conf}${route}${ix}<span class="tool">${step.tool}</span>` +
      `<pre>${JSON.stringify(step.args, null, 2)}</pre>`;
  }
  if (step.safety) {
    const notice = document.createElement("div");
    notice.className = `safety-notice ${step.safety.status}`;
    const label = step.safety.status === "blocked"
      ? "Blocked by safety policy"
      : step.safety.status === "clarification_required"
        ? "Clarification required before tool call"
      : step.safety.status === "confirmation_required"
        ? "Confirmation required before external side effect"
        : "Read-only action";
    notice.textContent = `${label} · ${step.safety.reason}`;
    div.appendChild(notice);
  }
  return div;
}

async function run() {
  const query = $("prompt").value.trim();
  if (!query || !SESSION) return;
  $("run").disabled = true;
  const res = $("result");
  res.hidden = false;
  res.innerHTML = '<div class="call"><span class="tool">…thinking</span></div>';
  try {
    if ($("plan-mode").checked) {
      const { steps, ms } = await planRollout(query);
      res.innerHTML = "";
      if (!steps.length) res.appendChild(renderCall({ abstain: true }));
      steps.forEach((s, i) => res.appendChild(renderCall(s, i)));
      const t = document.createElement("div");
      t.className = "timing";
      t.textContent = `${steps.length} step(s) · ${ms.toFixed(0)} ms · ${BACKEND}`;
      res.appendChild(t);
    } else {
      const out = await callOnce(query);
      out.safety = actionSafetyPolicy(out, query, { args: out.args });
      if (out.safety.status === "blocked" || out.safety.status === "clarification_required") {
        out.args = null;
      }
      res.innerHTML = "";
      res.appendChild(renderCall(out));
      const t = document.createElement("div");
      t.className = "timing";
      t.textContent = `${out.ms.toFixed(0)} ms · ${BACKEND}`;
      res.appendChild(t);
    }
  } catch (e) {
    res.innerHTML = `<div class="call abstain"><span class="tool">error</span><pre>${e}</pre></div>`;
  } finally {
    $("run").disabled = false;
  }
}

function wireUI() {
  $("run").addEventListener("click", run);
  $("prompt").addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run();
  });
  document.querySelectorAll(".chip").forEach((c) => {
    c.addEventListener("click", () => {
      $("prompt").value = c.textContent;
      $("plan-mode").checked = c.dataset.plan === "1";
      run();
    });
  });
}

let LOCALAGENT_READY;
if (window.__localAgentSkipInit) {
  LOCALAGENT_READY = Promise.resolve({ backend: null });
} else {
  LOCALAGENT_READY = (async function main() {
    wireUI();
    try {
      setStatus("loading", "Loading model… (first load downloads & caches the weights)");
      await loadBundle();
      setStatus("ready", "Model ready — runs locally in your browser.", BACKEND);
      $("run").disabled = false;
      return { backend: BACKEND };
    } catch (e) {
      console.error(e);
      setStatus("error", "Failed to load the model bundle: " + e.message);
      throw e;
    }
  })();
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    ACTION_POLICIES,
    argmaxAllowed,
    bundleArtifactEvidence,
    bundleManifestByteEvidence,
    browserCandidateTargetId,
    buildCandidateTrie,
    cachedActionPrompt,
    cachedDecodeBundleEvidence,
    cachedFeeds,
    cachedOutputLocations,
    cachedSessionOptions,
    canonicalActionJson,
    compactDispatchQuery,
    createCachedAutoregressiveRunner,
    dispatchSelect,
    actionSafetyPolicy,
    greedyToken,
    groundedActionCandidates,
    materializeCachedLogits,
    padPromptIds,
    padPromptIdsTrailing,
    parseGeneratedAction,
    renderFullCatalogContextText,
    renderFullFunctionCatalog,
    runtimeAssetEvidence,
    fillSchemaArg,
    fetchPinnedJsonArtifact,
    groundFromSchema,
    groundedArgsValid,
    groundingPools,
    manifestArtifactFor,
    mobileLexicalSelect,
    statefulActionLexicalSelect,
    modelArtifactEvidence,
    retrievalEmbedding,
    retrievalCandidatesFromSidecar,
    retrievalSelectFromSidecar,
    sha256Bytes,
    selectTokenFromLogits,
    validateActionForMeta,
    validateBenchmarkBundleContract,
    validateCachedGraphContract,
    validateCachedStepOutputs,
    validateProductionCachedBundle,
    validateSessionOutputs,
    validateTrainingLineageExport,
    verifyArtifactBytesAgainstManifest,
    verifyModelBytesAgainstManifest,
    verifyPinnedArtifactBytes,
  };
}

// Optional integration hook for local stateful mobile/MCP harnesses. The normal demo UI remains
// unchanged; consumers must await the ready promise and use the returned schema-validated action.
if (typeof window !== "undefined") {
  window.__localAgentReady = LOCALAGENT_READY;
  window.__localAgentCallPolicyOnce = callPolicyOnce;
  window.__localAgentStructuredAction = structuredAction;
}
