/* Complete-action benchmark for the LocalAgent WebGPU bundle.
 *
 * The measured interval ends only after route selection and pointer argument grounding, when a
 * complete action can be executed.  It therefore reports TTFA rather than treating raw token
 * throughput as the user-visible result.
 */

const BENCHMARK_DEADLINES_MS = [100, 250, 500, 1000, 2000];
const BENCHMARK_ACTION_TIMEOUT_MS = 10_000;
const BENCHMARK_SUITE_IDENTITY = Object.freeze({
  file: "benchmark-cases.json",
  bytes: 4596,
  sha256: "f6e479c10b420edcd1630e99c43df3206a40431f1256d919d5ea1a57bc88142c",
  identity_source:
    "configs/data/pretrain-paper.yaml:evaluation_decontamination/local-realtime-actions",
});
let LAST_BENCHMARK = null;
let MODEL_READY_MS = null;

function benchmarkWithWatchdog(operation, timeoutMs = BENCHMARK_ACTION_TIMEOUT_MS) {
  let timeoutId = null;
  const timeout = new Promise((resolve, reject) => {
    timeoutId = setTimeout(() => {
      const error = new Error(
        `Policy call exceeded the ${timeoutMs} ms watchdog; aborting the entire page run.`
      );
      error.name = "BenchmarkActionTimeoutError";
      error.code = "benchmark_action_timeout";
      error.timeout_ms = timeoutMs;
      reject(error);
    }, timeoutMs);
  });
  const pending = Promise.resolve().then(operation);
  return Promise.race([pending, timeout]).finally(() => clearTimeout(timeoutId));
}

function isBenchmarkActionTimeout(error) {
  return error?.code === "benchmark_action_timeout";
}

function benchmarkPercentile(values, q) {
  const ordered = [...values].sort((a, b) => a - b);
  const position = (ordered.length - 1) * q;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return ordered[lower];
  return ordered[lower] * (upper - position) + ordered[upper] * (position - lower);
}

function benchmarkLatencySummary(values) {
  values = values.filter((value) => Number.isFinite(value));
  if (!values.length) {
    return { count: 0, min: null, mean: null, p50: null, p90: null, p95: null, p99: null, max: null };
  }
  const sum = values.reduce((acc, value) => acc + value, 0);
  return {
    count: values.length,
    min: Math.min(...values),
    mean: sum / values.length,
    p50: benchmarkPercentile(values, 0.50),
    p90: benchmarkPercentile(values, 0.90),
    p95: benchmarkPercentile(values, 0.95),
    p99: benchmarkPercentile(values, 0.99),
    max: Math.max(...values),
  };
}

function seededRandom(seedText) {
  let state = 2166136261;
  for (let i = 0; i < seedText.length; i++) {
    state ^= seedText.charCodeAt(i);
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

function shuffledCases(cases, seedText) {
  const ordered = [...cases];
  const random = seededRandom(seedText);
  for (let i = ordered.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [ordered[i], ordered[j]] = [ordered[j], ordered[i]];
  }
  return ordered;
}

async function sha256Hex(text) {
  if (!crypto.subtle) return null;
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return Array.from(new Uint8Array(digest), (value) => value.toString(16).padStart(2, "0")).join("");
}

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) =>
      `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function benchmarkJsonValue(value) {
  if (value === undefined) return null;
  try {
    return JSON.parse(JSON.stringify(value));
  } catch {
    return null;
  }
}

function normalizeBenchmarkAction(action) {
  if (action?.abstain === true) return { abstain: true };
  return {
    tool: typeof action?.tool === "string" ? action.tool : null,
    args: benchmarkJsonValue(action?.args),
  };
}

function benchmarkValueMatchesSchema(value, schema, path, errors) {
  if (schema.enum && !schema.enum.some(
    (candidate) => canonicalJson(candidate) === canonicalJson(value)
  )) {
    errors.push(`${path} is not in the declared enum.`);
  }
  if (Object.hasOwn(schema, "const") &&
      canonicalJson(value) !== canonicalJson(schema.const)) {
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
        benchmarkValueMatchesSchema(child, properties[key], `${path}.${key}`, errors);
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
        benchmarkValueMatchesSchema(item, schema.items, `${path}[${index}]`, errors)
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

function validateBenchmarkActionSchema(action) {
  const normalized = normalizeBenchmarkAction(action);
  if (normalized.abstain === true) {
    return {
      validator: "benchmark-json-schema-subset-v2",
      valid: true,
      errors: [],
      schema_tool: null,
      tool_schema: null,
    };
  }
  const spec = (META.tools || []).find((tool) => tool.name === normalized.tool);
  if (!spec) {
    return {
      validator: "benchmark-json-schema-subset-v2",
      valid: false,
      errors: [`Unknown tool ${JSON.stringify(normalized.tool)}.`],
      schema_tool: null,
      tool_schema: null,
    };
  }
  const schema = benchmarkJsonValue(spec.schema || {});
  const errors = [];
  benchmarkValueMatchesSchema(normalized.args, schema, "$.args", errors);
  return {
    validator: "benchmark-json-schema-subset-v2",
    valid: errors.length === 0,
    errors,
    schema_tool: spec.name,
    tool_schema: schema,
  };
}

function actionSchemaValid(action) {
  return validateBenchmarkActionSchema(action).valid;
}

function benchmarkParseEvidence(action, runtimeError = null) {
  return {
    policy: action?.policy ?? null,
    inference_passes: action?.inference_passes ?? null,
    parse_kind: action?.parse_kind ?? null,
    parse_failure: action?.parse_failure === true,
    parse_error: action?.parse_error ?? null,
    runtime_validation_failure: action?.validation_failure === true,
    runtime_validation_error: action?.validation_error ?? null,
    runtime_error: runtimeError,
  };
}

function scoreBenchmarkAction(
  action,
  benchmarkCase,
  schemaValid = validateBenchmarkActionSchema(action).valid
) {
  const normalized = normalizeBenchmarkAction(action);
  const expected = benchmarkCase.expected;
  if (expected.abstain) {
    const exactAction = normalized.abstain === true;
    return {
      exact_tool: exactAction,
      exact_args: exactAction,
      exact_action: exactAction,
      success: exactAction,
      schema_valid: schemaValid,
    };
  }
  const toolOk = normalized.abstain !== true && normalized.tool === expected.tool;
  const argsOk = toolOk && canonicalJson(normalized.args) === canonicalJson(expected.args);
  return {
    exact_tool: toolOk,
    exact_args: argsOk,
    exact_action: toolOk && argsOk,
    success: toolOk && argsOk,
    schema_valid: schemaValid,
  };
}

function benchmarkErrorRecord(error) {
  return {
    name: error?.name || "Error",
    message: error?.message || String(error),
  };
}

async function adapterInfo() {
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

function modelResourceTiming(modelUrl = MODEL_URL) {
  const entry = performance.getEntriesByType("resource")
    .find((item) => item.name.endsWith(`/${modelUrl}`));
  if (!entry) return null;
  return {
    duration_ms: entry.duration,
    transfer_bytes: entry.transferSize || null,
    encoded_bytes: entry.encodedBodySize || null,
    decoded_bytes: entry.decodedBodySize || null,
  };
}

function bundleArtifactFor(fileName) {
  if (!fileName || !BUNDLE_MANIFEST?.artifacts) return null;
  return Object.values(BUNDLE_MANIFEST.artifacts)
    .find((artifact) => artifact?.file === fileName) || null;
}

function benchmarkHarnessTtfa(record) {
  return record.harness_ttfa_ms ?? record.ttfa_ms;
}

function benchmarkContextAudit(records, targetInputTokens, policy) {
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
    const structured = policy === ACTION_POLICIES.STRUCTURED;
    const matchesFeatureContract = structured
      ? record.decision_input_tokens === record.natural_input_tokens &&
        record.decision_feature_index === record.natural_input_tokens - 1 &&
        record.context_padding_placement === (
          targetInputTokens == null ? "none" : "after_natural_assistant_marker"
        )
      : record.decision_input_tokens === record.input_tokens &&
        record.decision_feature_index === record.input_tokens - 1 &&
        record.context_padding_placement === (
          targetInputTokens == null ? "none" : "before_assistant_marker"
        );
    if (!matchesLength || !matchesFeatureContract) mismatched += 1;
  }
  return {
    requested_input_tokens: targetInputTokens,
    policy,
    verified_records: records.length - missing,
    missing_records: missing,
    mismatched_records: mismatched,
  };
}

function summarizeBenchmark(records) {
  if (!records.length) throw new Error("At least one measured action record is required.");
  const harnessLatencies = records.map(benchmarkHarnessTtfa);
  if (harnessLatencies.some((value) => !Number.isFinite(value) || value < 0)) {
    throw new Error(
      "Every measured opportunity must retain a finite non-negative harness TTFA."
    );
  }
  const timings = [
    "harness_ttfa_ms", "runtime_ttfa_ms", "independent_validate_ms", "ttft_ms",
    "tpot_ms", "tokenize_ms", "inference_ms", "decode_control_ms", "dispatch_ms",
    "parse_validate_ms",
  ];
  const latency = {};
  for (const key of timings) {
    const values = key === "harness_ttfa_ms"
      ? harnessLatencies
      : records.map((row) => row[key]);
    latency[key] = benchmarkLatencySummary(values);
  }
  // Preserve the legacy summary key for consumers of schema v1/v2 artifacts. It is an exact alias,
  // not another clock.
  latency.ttfa_ms = latency.harness_ttfa_ms;
  const exact = records.filter((row) => row.success).length;
  const valid = records.filter((row) => row.schema_valid).length;
  const parseFailures = records.filter((row) => row.parse_failure).length;
  const validationFailures = records.filter((row) => row.validation_failure).length;
  const outputTokens = records.reduce((sum, row) => sum + (row.output_tokens || 0), 0);
  const totalMs = harnessLatencies.reduce((sum, value) => sum + value, 0);
  const deadlines = {};
  for (const deadline of BENCHMARK_DEADLINES_MS) {
    const onTime = records.filter((row) => benchmarkHarnessTtfa(row) <= deadline);
    const useful = onTime.filter((row) => row.success && row.schema_valid);
    deadlines[deadline] = {
      opportunities: records.length,
      on_time: onTime.length,
      on_time_rate: onTime.length / records.length,
      useful: useful.length,
      useful_rate: useful.length / records.length,
      success_at_deadline: useful.length / records.length,
      useful_actions_per_minute: totalMs > 0 ? useful.length / (totalMs / 60000) : null,
    };
  }
  return {
    latency_ms: latency,
    exact_action_accuracy: exact / records.length,
    schema_valid_rate: valid / records.length,
    parse_failure_rate: parseFailures / records.length,
    validation_failure_rate: validationFailures / records.length,
    total_output_tokens: outputTokens,
    deadline_attainment_ms: deadlines,
  };
}

function fixed(value, digits = 1) {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : "—";
}

function renderBenchmarkSummary(payload) {
  const summary = payload.summary;
  const ttfa = summary.latency_ms.harness_ttfa_ms;
  const deadlineRows = Object.entries(summary.deadline_attainment_ms)
    .map(([deadline, values]) =>
      `<tr><td>${deadline} ms</td><td>${fixed(values.on_time_rate * 100)}%</td>` +
      `<td>${fixed(values.useful_rate * 100)}%</td>` +
      `<td>${fixed(values.useful_actions_per_minute)}</td></tr>`)
    .join("");
  $("benchmark-output").innerHTML = `
    <div class="metric-grid">
      <div class="metric"><span>Harness TTFA p50</span><strong>${fixed(ttfa.p50)} ms</strong></div>
      <div class="metric"><span>Harness TTFA p95</span><strong>${fixed(ttfa.p95)} ms</strong></div>
      <div class="metric"><span>Exact action</span><strong>${fixed(summary.exact_action_accuracy * 100)}%</strong></div>
      <div class="metric"><span>Schema valid</span><strong>${fixed(summary.schema_valid_rate * 100)}%</strong></div>
      <div class="metric"><span>Parse failures</span><strong>${fixed(summary.parse_failure_rate * 100)}%</strong></div>
      <div class="metric"><span>Validation failures</span><strong>${fixed(summary.validation_failure_rate * 100)}%</strong></div>
    </div>
    <table class="benchmark-table">
      <thead><tr><th>Deadline</th><th>On time</th><th>Correct + on time</th><th>Useful actions/min</th></tr></thead>
      <tbody>${deadlineRows}</tbody>
    </table>
    <details><summary>Full summary and metadata</summary><pre>${JSON.stringify({
      metadata: payload.metadata,
      summary: payload.summary,
    }, null, 2)}</pre></details>`;
}

function setBenchmarkProgress(text) {
  $("benchmark-progress").textContent = text;
}

async function runActionBenchmark() {
  const button = $("start-benchmark");
  const warmupRecords = [];
  const records = [];
  let activeOpportunity = null;
  button.disabled = true;
  $("download-benchmark").disabled = true;
  $("benchmark-output").innerHTML = "";
  try {
    if (BACKEND !== REQUESTED_BACKEND) {
      throw new Error(
        `Explicit ${REQUESTED_BACKEND} run initialized unexpected backend ${BACKEND}.`
      );
    }
    const suiteDocument = await fetchPinnedJsonArtifact(
      BENCHMARK_SUITE_IDENTITY.file,
      BENCHMARK_SUITE_IDENTITY
    );
    const suite = suiteDocument.value;
    const suiteByteEvidence = suiteDocument.evidence;
    if (!suiteByteEvidence.identity_verified) {
      throw new Error("Action benchmark suite lacks verified raw-byte identity evidence.");
    }
    const warmups = Math.max(0, Number.parseInt($("benchmark-warmups").value, 10) || 0);
    const repetitions = Math.max(1, Number.parseInt($("benchmark-repetitions").value, 10) || 1);
    const seed = $("benchmark-seed").value.trim() || "slmw2026-v1";
    const policy = $("benchmark-policy").value;
    const maxOutputTokens = Math.max(
      1, Number.parseInt($("benchmark-max-output-tokens").value, 10) || 192
    );
    const contextValue = $("benchmark-context-tokens").value;
    const targetInputTokens = contextValue ? Number.parseInt(contextValue, 10) : null;
    if (
      targetInputTokens != null &&
      policy !== ACTION_POLICIES.STRUCTURED &&
      META.max_seq_len &&
      targetInputTokens + maxOutputTokens > META.max_seq_len
    ) {
      throw new Error(
        `Target input ${targetInputTokens} + AR cap ${maxOutputTokens} exceeds ` +
        `model max_seq_len ${META.max_seq_len}.`
      );
    }
    setBenchmarkProgress(
      `Loading ${policy} with the single requested ${BACKEND} execution provider…`
    );
    await prepareActionPolicy(policy);
    const cachedEvidence = policy === ACTION_POLICIES.STRUCTURED
      ? null
      : cachedDecodeBundleEvidence();
    const policyModelUrl =
      policy === ACTION_POLICIES.STRUCTURED
        ? MODEL_URL
        : `${CACHED_DECODE_BUNDLE.prefillFile} + ${CACHED_DECODE_BUNDLE.decodeFile}`;
    const policyByteEvidence = policy === ACTION_POLICIES.STRUCTURED
      ? modelArtifactEvidence(policyModelUrl)
      : Object.freeze({
          bytes: cachedEvidence.prefill.bytes + cachedEvidence.decode.bytes,
          file: policyModelUrl,
          manifest_verified: cachedEvidence.provenance.verified,
          sha256: cachedEvidence.decode.sha256,
          session_source: "two_content_verified_in_memory_cached_graphs",
        });
    if (!policyByteEvidence?.manifest_verified) {
      throw new Error(
        `Benchmark model ${policyModelUrl} was not byte-verified against bundle-manifest.json.`
      );
    }
    const runtimeAssets = runtimeAssetEvidence();

    for (let i = 0; i < warmups; i++) {
      setBenchmarkProgress(`Warm-up ${i + 1}/${warmups}…`);
      const benchmarkCase = suite.cases[i % suite.cases.length];
      activeOpportunity = {
        phase: "warmup",
        index: i,
        case_id: benchmarkCase.id,
      };
      const action = await benchmarkWithWatchdog(
        () => callPolicyOnce(
          benchmarkCase.query,
          policy,
          { maxOutputTokens, targetInputTokens }
        )
      );
      const predictedAction = normalizeBenchmarkAction(action);
      const validationStarted = performance.now();
      const independentSchema = validateBenchmarkActionSchema(predictedAction);
      const independentValidateMs = performance.now() - validationStarted;
      const runtimeTtfaMs = action.timing.ttfa_ms;
      const harnessTtfaMs = runtimeTtfaMs + independentValidateMs;
      warmupRecords.push({
        index: i,
        phase: i === 0 ? "first_inference" : "warmup",
        case_id: benchmarkCase.id,
        policy,
        action_timeout_ms: BENCHMARK_ACTION_TIMEOUT_MS,
        watchdog_outcome: "completed_before_timeout",
        output_tokens: action.output_tokens,
        predicted_action: predictedAction,
        expected_action: benchmarkJsonValue(benchmarkCase.expected),
        predicted_tool: predictedAction.abstain ? null : predictedAction.tool,
        expected_tool: benchmarkCase.expected.tool || null,
        independent_schema: independentSchema,
        parse_evidence: benchmarkParseEvidence(action),
        parse_failure: action.parse_failure === true,
        validation_failure: !independentSchema.valid,
        ...scoreBenchmarkAction(predictedAction, benchmarkCase, independentSchema.valid),
        ...action.timing,
        runtime_ttfa_ms: runtimeTtfaMs,
        independent_validate_ms: independentValidateMs,
        harness_ttfa_ms: harnessTtfaMs,
        ttfa_ms: harnessTtfaMs,
      });
      activeOpportunity = null;
    }

    const total = repetitions * suite.cases.length;
    for (let repetition = 0; repetition < repetitions; repetition++) {
      const repetitionCases = shuffledCases(suite.cases, `${seed}:${repetition}`);
      for (let orderIndex = 0; orderIndex < repetitionCases.length; orderIndex++) {
        const benchmarkCase = repetitionCases[orderIndex];
        if (document.visibilityState !== "visible") {
          throw new Error("Run stopped because the benchmark tab became hidden.");
        }
        setBenchmarkProgress(
          `Measured action ${records.length + 1}/${total}: ${benchmarkCase.id}`
        );
        activeOpportunity = {
          phase: "measured",
          repetition,
          order_index: orderIndex,
          case_id: benchmarkCase.id,
        };
        const actionStarted = performance.now();
        let action;
        let runtimeError = null;
        try {
          action = await benchmarkWithWatchdog(
            () => callPolicyOnce(
              benchmarkCase.query, policy, { maxOutputTokens, targetInputTokens }
            )
          );
        } catch (error) {
          if (isBenchmarkActionTimeout(error)) throw error;
          runtimeError = benchmarkErrorRecord(error);
          const elapsed = performance.now() - actionStarted;
          action = {
            abstain: false,
            tool: null,
            args: null,
            output_tokens: 0,
            decode_steps: 0,
            inference_passes: 0,
            decode_cache: policy === ACTION_POLICIES.STRUCTURED ? null : true,
            decode_strategy: "runtime_exception",
            stop_reason: "runtime_exception",
            generated_text: null,
            parse_kind: "runtime_exception",
            parse_failure: true,
            parse_error: runtimeError.message,
            validation_failure: true,
            validation_error: "runtime_exception",
            timing: {
              tokenize_ms: null,
              inference_ms: null,
              decode_control_ms: null,
              dispatch_ms: null,
              parse_validate_ms: null,
              ttft_ms: null,
              tpot_ms: null,
              ttfa_ms: elapsed,
            },
          };
        }
        const predictedAction = normalizeBenchmarkAction(action);
        const expectedAction = benchmarkJsonValue(benchmarkCase.expected);
        const parseEvidence = benchmarkParseEvidence(action, runtimeError);
        const validationStarted = performance.now();
        const independentSchema = validateBenchmarkActionSchema(predictedAction);
        const independentValidateMs = performance.now() - validationStarted;
        const score = scoreBenchmarkAction(
          predictedAction, benchmarkCase, independentSchema.valid
        );
        const runtimeTtfaMs = action.timing.ttfa_ms;
        const harnessTtfaMs = runtimeTtfaMs + independentValidateMs;
        records.push({
          case_id: benchmarkCase.id,
          family: benchmarkCase.family,
          repetition,
          order_index: orderIndex,
          backend: BACKEND,
          policy,
          action_timeout_ms: BENCHMARK_ACTION_TIMEOUT_MS,
          watchdog_outcome: "completed_before_timeout",
          input_tokens: action.input_tokens,
          input_bytes: action.input_bytes,
          natural_input_tokens: action.natural_input_tokens,
          context_padding_tokens: action.context_padding_tokens,
          context_padding_placement: action.context_padding_placement,
          decision_input_tokens: action.decision_input_tokens,
          decision_feature_index: action.decision_feature_index,
          output_tokens: action.output_tokens,
          decode_steps: action.decode_steps,
          inference_passes: action.inference_passes,
          prefill_passes: action.prefill_passes ?? null,
          cached_decode_passes: action.cached_decode_passes ?? null,
          decode_cache: action.decode_cache,
          decode_strategy: action.decode_strategy,
          token_selection: action.token_selection ?? null,
          next_token_role: action.next_token_role ?? null,
          prompt_contract: action.prompt_contract ?? null,
          tool_catalog_size: action.tool_catalog_size ?? null,
          stop_reason: action.stop_reason,
          generated_text: action.generated_text,
          parse_kind: action.parse_kind,
          parse_failure: parseEvidence.parse_failure,
          parse_error: action.parse_error,
          validation_failure: !independentSchema.valid,
          validation_error: action.validation_error,
          predicted_action: predictedAction,
          expected_action: expectedAction,
          independent_schema: independentSchema,
          parse_evidence: parseEvidence,
          candidate_count: action.candidate_count ?? null,
          ...action.timing,
          runtime_ttfa_ms: runtimeTtfaMs,
          independent_validate_ms: independentValidateMs,
          harness_ttfa_ms: harnessTtfaMs,
          // Backward-compatible alias. metadata.latency_clock names the unambiguous primary field.
          ttfa_ms: harnessTtfaMs,
          runtime_error: runtimeError,
          ...score,
          predicted_tool: predictedAction.abstain ? null : predictedAction.tool,
          predicted_route: action.route ?? null,
          route_confidence: action.conf ?? null,
          expected_tool: expectedAction.tool || null,
        });
        activeOpportunity = null;
      }
    }

    const policyArtifact = policy === ACTION_POLICIES.STRUCTURED
      ? bundleArtifactFor(policyModelUrl)
      : null;
    const tokenizerArtifact = bundleArtifactFor(META.tokenizer_file);
    const gpuAdapter = await adapterInfo();
    const ortWebVersion = ort.env?.versions?.web || null;
    const contextAudit = benchmarkContextAudit(records, targetInputTokens, policy);
    if (contextAudit.mismatched_records > 0) {
      throw new Error(
        `${contextAudit.mismatched_records} measured records violate the context condition.`
      );
    }
    LAST_BENCHMARK = {
      schema_version: 3,
      benchmark: suite.name,
      created_at: new Date().toISOString(),
      metadata: {
        benchmark_version: "rtab-0.4",
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
        policy,
        max_output_tokens: maxOutputTokens,
        target_input_tokens: targetInputTokens,
        context_condition:
          targetInputTokens == null
            ? "natural"
            : policy === ACTION_POLICIES.STRUCTURED
              ? "fixed_compute_tokens_natural_decision_feature"
              : "fixed_final_tokenizer_tokens_pre_assistant_stress",
        context_padding:
          targetInputTokens == null
            ? "none"
            : policy === ACTION_POLICIES.STRUCTURED
              ? "single-token spaces appended after the natural assistant marker"
              : "single-token spaces inserted immediately before the assistant marker",
        decision_feature_contract:
          policy === ACTION_POLICIES.STRUCTURED
            ? "hidden[natural_input_tokens - 1]; pointer scan bounded to natural_input_tokens"
            : "last hidden state of the materialized autoregressive prefix",
        context_audit: contextAudit,
        latency_clock: "harness_ttfa_ms",
        latency_boundaries: {
          harness_ttfa:
            "immediately before prompt tokenization through independent schema validation",
          runtime_ttfa:
            "runtime prompt tokenization through runtime parse/schema validation",
          ttfa_ms:
            "backward-compatible exact alias of harness_ttfa_ms; not an additional clock",
          ttft: "runtime inference submission through first sampled token",
          exact_action_scoring: "excluded from TTFA",
        },
        timeout_ms: BENCHMARK_ACTION_TIMEOUT_MS,
        action_timeout_ms: BENCHMARK_ACTION_TIMEOUT_MS,
        watchdog_scope: "every warmup and measured policy call",
        timeout_contract:
          "a timeout aborts the entire page collection; ORT session.run is not cancellable, " +
          "so no subsequent policy call starts while timed-out inference may still be live",
        decode_cache: policy === ACTION_POLICIES.STRUCTURED ? null : true,
        decode_strategy: policy === ACTION_POLICIES.STRUCTURED
          ? "one_forward_structured_heads"
          : CACHED_DECODE_STRATEGY,
        model_url: MODEL_URL,
        action_model_url: MODEL_URL,
        logits_model_url: policy === ACTION_POLICIES.STRUCTURED
          ? LOGITS_MODEL_URL
          : null,
        cached_prefill_model_url: CACHED_DECODE_BUNDLE?.prefillFile ?? null,
        cached_decode_model_url: CACHED_DECODE_BUNDLE?.decodeFile ?? null,
        policy_model_url: policyModelUrl,
        precision: (
          policy === ACTION_POLICIES.STRUCTURED
            ? (MODEL_URL.includes("fp16") ? "fp16" : "fp32")
            : CACHED_DECODE_BUNDLE.precision
        ),
        ort_web_version: ortWebVersion,
        onnxruntime_version: ortWebVersion,
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
        page_to_model_ready_ms: MODEL_READY_MS,
        bundle_load_timing_ms: { ...BUNDLE_LOAD_TIMING },
        warmup_records: warmupRecords,
        action_model_resource: modelResourceTiming(MODEL_URL),
        logits_model_resource: modelResourceTiming(LOGITS_MODEL_URL),
        cached_decode_evidence: cachedEvidence,
        model_vocab_size: META.vocab_size,
        model_d_model: META.d_model,
        model_parameters: META.model_parameters ?? null,
        model_encoding: META.encoding,
        model_layers: META.n_layers ?? null,
        model_tool_count: (META.tools || []).length,
        git_commit: BUNDLE_MANIFEST?.git_commit ?? null,
        model_hash: policyByteEvidence.sha256,
        checkpoint_hash: BUNDLE_MANIFEST?.checkpoint_sha256 ?? null,
        tokenizer_hash: runtimeAssets.tokenizer?.sha256 ?? null,
        heads_hash: runtimeAssets.heads_json.sha256,
        dispatch_heads_hash: runtimeAssets.dispatch_heads_json.sha256,
        meta_file_hash: runtimeAssets.meta_json.sha256,
        runtime_asset_evidence: runtimeAssets,
        graph_hash: policyByteEvidence.sha256,
        model_bytes: policyByteEvidence.bytes,
        model_byte_evidence: policyByteEvidence,
        manifest_graph_hash: policyArtifact?.sha256 ??
          CACHED_DECODE_BUNDLE?.provenance?.artifacts?.[
            CACHED_DECODE_BUNDLE.decodeFile
          ]?.sha256 ?? null,
        manifest_model_bytes: policyArtifact?.bytes ??
          (
            CACHED_DECODE_BUNDLE
              ? CACHED_DECODE_BUNDLE.provenance.artifacts[
                  CACHED_DECODE_BUNDLE.prefillFile
                ].bytes +
                CACHED_DECODE_BUNDLE.provenance.artifacts[
                  CACHED_DECODE_BUNDLE.decodeFile
                ].bytes
              : null
          ),
        manifest_tokenizer_hash: tokenizerArtifact?.sha256 ?? null,
        artifact_hash_contract:
          "structured model, heads, metadata, dispatch, tokenizer, and cached provenance are " +
          "content-pinned; cached prefill/decode graphs are additionally bound to passed parity " +
          "before ORT receives verified in-memory bytes",
        model_meta_canonical_sha256: await sha256Hex(canonicalJson(META)),
        bundle_manifest: BUNDLE_MANIFEST,
        bundle_manifest_byte_evidence: bundleManifestByteEvidence(),
        bundle_manifest_canonical_sha256: BUNDLE_MANIFEST
          ? await sha256Hex(canonicalJson(BUNDLE_MANIFEST))
          : null,
        suite_schema_version: suite.schema_version,
        suite_sha256: suiteByteEvidence.sha256,
        suite_bytes: suiteByteEvidence.bytes,
        suite_byte_evidence: suiteByteEvidence,
        case_order_seed: seed,
        warmups,
        repetitions,
        cases: suite.cases.length,
        measured_records: records.length,
        concurrency: 1,
        timer: "performance.now",
        tab_visibility_required: true,
        schema_source: "loaded META.tools[].schema",
        schema_validator:
          "independent benchmark.js JSON Schema subset v2; runtime validation is diagnostic",
        raw_action_evidence_contract:
          "each row stores normalized predicted_action, full expected_action, parse_evidence, " +
          "and independent_schema including errors and the exact selected tool schema",
      },
      summary: summarizeBenchmark(records),
      records,
    };
    renderBenchmarkSummary(LAST_BENCHMARK);
    setBenchmarkProgress(`Complete: ${records.length} measured actions on ${BACKEND}.`);
    $("download-benchmark").disabled = false;
  } catch (error) {
    console.error(error);
    if (isBenchmarkActionTimeout(error)) {
      LAST_BENCHMARK = {
        schema_version: 3,
        benchmark: "localagent-held-out-action-latency",
        created_at: new Date().toISOString(),
        status: "aborted_incomplete",
        metadata: {
          benchmark_version: "rtab-0.4",
          action_timeout_ms: BENCHMARK_ACTION_TIMEOUT_MS,
          watchdog_scope: "every warmup and measured policy call",
          timeout_contract:
            "a timeout aborts the entire page collection; ORT session.run is not cancellable, " +
            "so no subsequent policy call starts while timed-out inference may still be live",
          completed_warmups: warmupRecords.length,
          completed_measured_records: records.length,
        },
        failure: {
          kind: "action_timeout",
          fatal_to_page_collection: true,
          inference_cancellation_supported: false,
          no_subsequent_policy_call_started: true,
          timeout_ms: error.timeout_ms ?? BENCHMARK_ACTION_TIMEOUT_MS,
          active_opportunity: activeOpportunity,
          error: benchmarkErrorRecord(error),
        },
        warmup_records: warmupRecords,
        records,
      };
      $("download-benchmark").disabled = false;
    }
    setBenchmarkProgress(`Benchmark failed: ${error.message}`);
  } finally {
    button.disabled = false;
  }
}

function downloadBenchmark() {
  if (!LAST_BENCHMARK) return;
  const blob = new Blob([JSON.stringify(LAST_BENCHMARK, null, 2)], { type: "application/json" });
  const anchor = document.createElement("a");
  anchor.href = URL.createObjectURL(blob);
  anchor.download = `localagent-webgpu-${Date.now()}.json`;
  anchor.click();
  URL.revokeObjectURL(anchor.href);
}

if (!(typeof window !== "undefined" && window.__localAgentSkipInit)) {
  (async function initBenchmark() {
    $("start-benchmark").disabled = true;
    try {
      await LOCALAGENT_READY;
      MODEL_READY_MS = performance.now() - window.__localAgentBenchmarkStart;
      $("start-benchmark").disabled = false;
      setBenchmarkProgress("Model ready. Measurements run at concurrency 1.");
    } catch {
      setBenchmarkProgress("The model bundle did not load; benchmark unavailable.");
    }
    $("start-benchmark").addEventListener("click", runActionBenchmark);
    $("download-benchmark").addEventListener("click", downloadBenchmark);
  })();
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    BENCHMARK_ACTION_TIMEOUT_MS,
    benchmarkWithWatchdog,
    isBenchmarkActionTimeout,
    normalizeBenchmarkAction,
    validateBenchmarkActionSchema,
    scoreBenchmarkAction,
    benchmarkLatencySummary,
    seededRandom,
    shuffledCases,
    summarizeBenchmark,
  };
}
