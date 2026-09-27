# Real-Time Action Benchmark (RTAB) specification

Version: 0.4 for corrected collection. The preserved negative action/DOM artifacts use historical
runner v0.2 and are never rewritten; v0.4 adds the corrected decision-feature contract,
independently rescorable normalized actions, and a fail-stop per-action watchdog.
Implementation:

- `src/localagent/eval/realtime.py`
- `src/localagent/eval/external_action_contract.py`
- `scripts/realtime_agent_benchmark.py`
- `scripts/fresh_action_eval.py`
- `spaces/localagent-webgpu/benchmark.html`
- `spaces/localagent-webgpu/benchmark.js`
- `spaces/localagent-webgpu/benchmark-cases.json`
- `spaces/localagent-webgpu/browser-tasks.js`
- `spaces/localagent-webgpu/browser-task-cases.json`
- `spaces/localagent-webgpu/backbone-benchmark.html`
- `spaces/localagent-webgpu/backbone-benchmark.js`
- `spaces/localagent-webgpu/decode-benchmark.html`
- `spaces/localagent-webgpu/decode-benchmark.js`

## Unit of evaluation

The unit is one complete action opportunity at concurrency one.

An action is complete when:

1. A route/tool or abstention decision exists.
2. All required arguments exist.
3. Argument types and enums satisfy the declared schema.
4. The action can be dispatched without generating another model token.

The first generated token is not an action unless it alone encodes the complete valid action.

## Timestamps

```text
t0            user event or new observation becomes available
t_ackpaint    visible busy/acknowledgment state is painted
t_call        benchmark/runtime call begins immediately before prompt tokenization
t_submit      model request is submitted
t_first       first autoregressive token becomes available
t_last        final required decode/terminal step becomes available
t_runtime_valid
              runtime parser/validator has produced a complete action
t_valid       independent harness validation passes; action is dispatch-ready
t_dispatch    action invocation begins, when the harness actually invokes it
t_tool_done   tool or browser action completes
t_nextpaint   resulting state is visibly painted
```

Derived measurements:

```text
acknowledgment latency = t_ackpaint - t0
TTFT                   = t_first - t_submit
TPOT                   = (t_last - t_first) / (N_decode - 1),
                         for N_decode >= 2
model TTFA             = t_runtime_valid - t_submit
runtime TTFA           = t_runtime_valid - t_call
harness TTFA           = t_valid - t_call
user TTFA              = t_valid - t0
dispatch latency       = t_dispatch - t_valid
tool latency           = t_tool_done - t_dispatch
closed-loop latency    = t_nextpaint - t0
```

TPOT is undefined, and must be exported as `null`, for a one-token action. Acknowledgment latency
is not itself an INP result. INP is a page-level field responsiveness metric derived from
qualifying interactions via the Event Timing API; the
[external "good" threshold of 200 ms](https://web.dev/articles/inp) applies at the 75th percentile
of page visits, not to complete model actions. RTAB may report an Event-Timing-derived interaction
duration separately when that instrumentation exists.

The complete-action page records:

```text
tokenize_ms
inference_ms         # total model execution time
decode_control_ms    # non-model autoregressive decode control
dispatch_ms          # legacy name: route/selector/grounding; no tool invocation
parse_validate_ms
runtime_ttfa_ms       # t_call through runtime parsing/schema validation
independent_validate_ms
harness_ttfa_ms       # t_call through independent validation; primary artifact clock
ttfa_ms              # backward-compatible exact alias of harness_ttfa_ms
```

The DOM microtask export uses the same flat action fields so the command-line analyzer can consume
it, and also retains them under `latency_ms` alongside tool, paint, and closed-loop timings.
`harness_ttfa_ms` includes tokenization and therefore is not model TTFA as defined above. It starts
at `t_call`, not at an instrumented user event, so it must not be presented as fully measured user
TTFA. Paper comparisons use `harness_ttfa_ms` unless a future artifact explicitly instruments and
declares `user_ttfa_ms`. `runtime_ttfa_ms` is a diagnostic clock and must not be mixed with the
harness clock.

For an invalid output, runtime exception, or timeout, `t_runtime_valid`, `t_valid`, and `t_dispatch`
may not exist. Such an opportunity still needs one record with `success: false`,
`schema_valid: false` when applicable, and a finite elapsed time to its terminal outcome. That
terminal-attempt latency participates in latency percentiles; it must not be dropped merely
because no valid action existed. The current pages convert returned model exceptions into finite
failure rows. They do **not** yet cancel a hung inference: `timeout_ms` is `null`, and a hang aborts
collection rather than yielding the required timeout row. This is a paper-grade blocker.

Use `performance.now()` as specified by
[W3C High Resolution Time](https://www.w3.org/TR/hr-time-3/) for monotonic browser timing. Do not
use wall-clock timestamps for intervals.

## Primary score

For deadline `B` and one declared latency clock:

```text
Success@B =
    count(exact_action and schema_valid and TTFA <= B)
    --------------------------------------------------
                 total action opportunities
```

The comparison is inclusive (`TTFA <= B`), and every measured repetition remains in the
denominator. This strict opportunity-level score is row-weighted. The task-bootstrap estimate
described below is a separately labeled case-macro score; the two are identical under the required
balanced-repetition design.

Report `B` in `{0.5 s, 1 s, 2 s}` and name the clock used. These are proposed RTAB reporting
points, not W3C, MLPerf, or universal human-factors standards, and a project SLO at one of these
deadlines must be labeled as proposed. Also report unconstrained exact action accuracy so a
latency failure can be separated from a policy failure.

For an autoregressive action requiring `N_decode >= 1` decode steps, where TTFT already includes
production of the first step:

```text
model TTFA =
    TTFT
    + 1000 * (N_decode - 1) / decode_tokens_per_second
    + runtime_postprocess_ms

runtime TTFA = tokenize_ms + model TTFA

harness TTFA = runtime TTFA + independent_validate_ms
```

`N_decode` includes any EOS or other terminal step that the implementation must observe before it
knows the action is complete. In the current browser artifact, use `decode_steps` for this quantity;
`output_tokens` excludes EOS and is not always sufficient. When `N_decode = 1`, no
iterative-decode interval exists: the minimum iterative decode rate is zero if the fixed costs fit
the deadline. When `N_decode > 1` and the fixed costs leave zero or negative budget, no finite
decode rate can meet the deadline. Inputs and computed finite results must reject NaN and infinity;
the infinity sentinel is reserved for the no-finite-rate deadline case. This equation does not
apply to the structured one-forward policy, which emits no autoregressive tokens.

Secondary Pareto metrics:

```text
useful actions/minute at B
Success@B per model MiB
episode successes/hour
```

Do not collapse latency, memory, and success into one hand-weighted score in the main paper.
Present the Pareto frontier.

## Workload layers

### A. Frozen function calling

- Exact AST/tool/schema match.
- Irrelevant-tool abstention.
- Typed strings, integers, numbers, booleans, and enums.
- Output-length buckets: 1–16, 17–32, 33–64, and more than 64 native tokens.
- Keep BFCL or equivalent official test sets frozen.
- Convert the real immutable external revision through an explicit adapter, then freeze/audit it
  with [`FRESH_EXTERNAL_EVAL_CONTRACT.md`](FRESH_EXTERNAL_EVAL_CONTRACT.md). No external source is
  currently supplied by this repository.

### B. Deterministic browser microactions

- The implemented harness covers click, double-click, type, key press, scroll, drag, move-cursor,
  and open-URL.
- Select, extract, and submit remain proposed extensions.
- Serve local versioned pages so network variance cannot hide model latency.
- Score exact action and final DOM state.
- Include missing/disabled/stale elements and confirmation-required actions.

### C. Browser task trajectories (not yet wired)

- Reproducible BrowserGym/WebArena/WorkArena subset.
- Final-state task success.
- Per-step TTFA and action count.
- Growing conversation history.
- Single request in flight.

### D. Context stress

Run the same action cases at 128, 512, 1,024, and 1,536 actual final-tokenizer input tokens. The
largest matched bucket reserves the declared autoregressive output cap inside the 2,048-token
sequence limit. Padding must be fixed and content-neutral by protocol; the implementation uses
single-token spaces immediately before the assistant marker. Record the requested target, actual
input tokens, natural input tokens, padding tokens, and UTF-8 bytes because byte and BPE tokens are
not comparable content units. A failed row may have `null` actual counts if tokenization itself
failed, but it still remains in the condition's denominator.

Fixed-length construction is an experimental treatment, not a neutral guarantee. The seed-2027
SFT pilot below showed that real, unmasked pre-assistant filler changes the feature seen by the
structured heads. Label this exact construction the **pre-assistant-padding stress condition**.
A corrected fixed-compute arm instead appends filler after the natural assistant marker, executes
the same target token count, dispatches from `hidden[natural_input_tokens - 1]`, and bounds pointer
scans to `natural_input_tokens`. Report both conditions by name; do not substitute natural-context
offline quality for either browser score.

### E. Visible response

Use a fixed-length grounded summary response to measure first meaningful visible chunk, TTFT, and
TPOT. Keep this separate from action dispatch; a text response and an executable action have
different completion semantics.

## Cold and warm phases

Report separately:

1. Bundle download and cache status.
2. Session creation.
3. Shader compilation/first inference.
4. Warm steady-state actions.
5. Sustained run for thermal or throttling effects.

Never mix the first inference into warm percentiles.

The complete-action and DOM in-page runners preserve raw warmup records and label the first one
`first_inference`; measured summaries exclude all warmups. This phase can include shader
compilation, but the browser runtime does not expose shader-compilation time as a separate
interval. Bundle/session timings are recorded separately. A run is valid only if no policy
inference occurred on the page before the recorded first-inference warmup.

The architecture-only control uses the separate `backbone-benchmark.html` runner and the
exporter-produced `matched-backbones.json` random-weight suite. It is explicitly latency-only:
there is no tokenizer asset, text, action, success, or quality score. Both hidden-only graphs
receive the exact same deterministic pre-tokenized IDs,
`ids[i] = (131 i + 17) mod vocabulary_size`, at actual tensor lengths 128, 512, 1,024, and 1,536.
Because dynamic input shapes can compile separately, it retains one randomized
`first_for_condition` record for all eight graph-by-length conditions; only the earliest such
record for each graph is also `first_ever_for_graph`. Those eight records, three subsequent
warmups per condition, and thirty randomized measured repetitions per condition remain separate
raw arrays. Graph outputs use ONNX Runtime's default CPU output contract, so WebGPU measurements
include hidden-state readback; every output tensor is released immediately after shape
validation.

The cache-bearing latency control uses the separate `decode-benchmark.html` runner and an
exporter-produced matched random-weight suite. It is not a complete-action or quality benchmark.
The export contains one dynamic-prompt prefill graph and a separate decode graph whose token axis
is fixed at `T=1`. Before the pair manifest is published, multi-length, multi-step parity requires
exact greedy next-token agreement between ONNX and the PyTorch cached path and between cached
PyTorch and fresh full-context PyTorch; floating caches must remain within the declared
precision-specific tolerance.

Each measured sample performs one prefill pass, returning the first token and initialized caches,
then 31 one-token decode passes. For WebGPU, `next_token` is requested on CPU and every
`present_*` cache is requested as `gpu-buffer`. Measured records show those presents rebound
directly as the corresponding next-call past inputs without cache-content readback to JavaScript,
with superseded/final tensors disposed. The graph nevertheless creates fresh presents on every
call: attention K/V uses append/concat and short-conv state uses a fresh fixed-width tail. This is
not in-place or paged cache management.

Tracked hybrid p50 wall-decode rates, reported as the median of three within-run p50s rather than
pooled samples, are:

| pair | 128 | 512 | 1,024 | 1,536 | 100 tok/s engineering reference |
|---|---:|---:|---:|---:|---|
| [34.2M](results/m5-webgpu-cached-decode-20260728.summary.json) | 74.05 | 64.54 | 57.53 | 47.15 | misses all |
| [15.6M](results/m5-webgpu-cached-decode-16m-20260728.summary.json) | 90.87 | 89.26 | 80.94 | 79.77 | misses all |
| [10.5M](results/m5-webgpu-cached-decode-10m-20260728.summary.json) | 159.23 | 160.46 | 143.49 | 127.57 | clears all |

The 34.2M hybrid is faster than its matched attention control at every context but remains under
100 tok/s. The reference is an engineering screen, not an RTAB action deadline or quality gate.
The [result index](results/README.md) links every summary, raw run, and matched config. These
random-weight graphs are not loaded by the trained complete-action runner.

## Observed seed-2027 SFT action pilot

The bounded pilot used the exact seed-2027 SFT checkpoint
`79387105de75d332413262e8d8ddb847b6cc13bc03f5e4df3c81663d9897aef1` and a parity-gated,
21,430,301-byte fp16 action graph. Three fresh page/session runs on one Apple Metal 3 adapter,
Chrome 150, and ONNX Runtime Web 1.27.0 requested exactly one WebGPU execution provider with no
whole-session retry. Per-node placement/fallback remains unknown.

Each run used 20 held-out cases, three warmups, 30 measured repetitions per case, concurrency one,
the `slmw2026-v1` order seed, and exactly 512 final tokenizer tokens. The point estimate is the
median of the three within-run percentiles. These preserved raw runs are the internally
prespecified pre-assistant-padding stress condition:

| Run | TTFA p50 (ms) | TTFA p95 (ms) | Exact action | Schema valid |
|---:|---:|---:|---:|---:|
| 1 | 24.75 | 34.405 | 5% | 100% |
| 2 | 24.55 | 34.30 | 5% | 100% |
| 3 | 25.20 | 34.80 | 5% | 100% |
| Median-of-runs | 24.75 | 34.405 | 5% | 100% |

The latency result is real; the policy result fails. All 20 unique cases had
`predicted_tool: null`, so capability was 1/20 overall, 0/19 tool-required, and 1/1 abstention.
Across timing repetitions this becomes 90/1,800 exact and 0/1,710 tool-required; those rows are
not independent capability tasks. Because every row completed before 100 ms, strict
`Success@100ms`,
`Success@250ms`, `Success@500ms`, `Success@1s`, and `Success@2s` all equal the same 5% exact rate.
Schema validity is 100% only because the function-calling suite treats abstention as a valid
output. It must not be described as correct schema-bearing tool use. The
[validated action summary](results/m5-webgpu-sft-action-pilot-seed2027.summary.json) binds the
checkpoint, graph, tokenizer, heads, suite, browser protocol, and three raw payloads.

The executable local-DOM follow-up used the same fixed-512 condition for eight tool-required
cases. Capability was 0/8 unique tasks. Across three runs and 720 repeated timing opportunities,
exact action, independent executable-
schema validity, final DOM state, state transition, and closed-loop success were all zero.
Pooled closed-loop latency was 33.30 ms p50 and 66.80 ms p95; these are failed-attempt timings,
not useful-action latency. The
[mechanically validated DOM summary](results/m5-webgpu-sft-dom-pilot-seed2027.summary.json)
binds the raw payloads:
[run 1](results/raw/m5-webgpu-sft-dom-pilot-seed2027-run1.json),
[run 2](results/raw/m5-webgpu-sft-dom-pilot-seed2027-run2.json), and
[run 3](results/raw/m5-webgpu-sft-dom-pilot-seed2027-run3.json).

An exploratory offline parity check diagnosed a feature-materialization shift. With natural
held-out prompts, the route was correct for 17/20 cases and the dense selector was top-1 correct
for 17/19 tool cases; zero cases routed to text, so the sole abstention was missed. On the
independent frozen eval decisions, natural route accuracy is 83/98, selector top-1 is 72/79, and
dispatched tool accuracy is 70/79. The pre-assistant-padding stress condition inserts real, unmasked
single-token spaces before the marker and reads `hidden[-1]`, unlike the natural SFT-probe feature
materialization; every case at 128 tokens and above routes to text. Native PyTorch, fp32/fp16 ONNX,
and exported JSON heads agree, so export or fp16 conversion is not causal. This is neither a
natural-prompt browser run nor evidence that the fixed-length score should be discounted: the
fixed-512 capability gate fails.

The corrected fixed-compute runner is implemented: it appends filler after the natural assistant
marker, executes 512 tokens, dispatches from `hidden[natural_input_tokens - 1]`, and bounds pointer
scans to the natural tokens. The
[offline audit](results/sft-structured-context-robustness-seed2027.summary.json) preserves natural
route/selector counts on both frozen suites. Runner versions `rtab-0.4` and `rtab-dom-0.4` persist
normalized predicted and expected actions, parse evidence, and the independent validator result;
offline aggregation recomputes tool, arguments, exact action, and schema validity and rejects
stored aliases that disagree. A 10,000 ms watchdog fail-stops the whole page collection and
exports an incomplete payload; no later inference begins because the timed-out ORT call itself is
not cancellable. Artifact identities, case-order seeds, and required record checks are frozen in
the [corrected protocol](results/webgpu-proxy-pilot-seed2027.corrected-browser.protocol.json), but
browser results are still missing and an external timestamp is still required. These suites were
inspected during diagnosis, so the corrected runs are deployment-parity re-evaluations rather
than untouched capability confirmation. A claim of genuine capability under the original
pre-marker 512-token materialization would require separate head training and evaluation on that
exact condition.

This pilot does not supply the planned raw-AR or candidate-trie comparison. Those controls
still need the same action-trained checkpoint with cache-bearing decode. Final
34M-screen/20-TPP training, BrowserGym/open-web tasks, cross-device replication, and
natural-prompt WebGPU quality also remain open.

## Provider protocol

- WebGPU and WASM are separate arms.
- Request exactly one provider for a benchmark run.
- A session created with fallback providers is not proof of WebGPU execution.
- Record failures and fallback events.
- Report ONNX Runtime version and graph/operator coverage where available.

The backbone runner passes exactly one entry in `executionProviders` and performs no
whole-session provider retry. This does not prove per-node placement: the artifact records
`per_node_placement_verified: false` and `per_node_fallback_status: "unknown"` unless a future
runtime exposes direct evidence. For WASM it also records cross-origin isolation,
`SharedArrayBuffer` availability, and `ort.env.wasm.numThreads`, since a server without COOP/COEP
may silently turn the CPU arm into a single-threaded condition.

The cache-bearing runner follows the same exact-one-provider rule. Its reported `gpu-buffer`
present-cache locations and no-JavaScript-readback bindings are runtime evidence about tensor
interfaces between calls, not proof of physical device residency or per-node execution.
`per_node_placement_verified` remains false and fallback status remains unknown.

The complete-action and DOM runners also request exactly one provider and perform no whole-session
retry in benchmark mode. Their `backend_requirement` labels say
`explicit-webgpu-no-whole-session-retry` or
`explicit-wasm-no-whole-session-retry-control`; they separately retain
`per_node_placement: "unknown"` and `per_node_fallback_status: "unknown"`. The browser benchmark
defaults to `benchmark.html?backend=webgpu`. Use `benchmark.html?backend=wasm` for the matched
control arm.

## Repetition and statistics

- Warm up with at least three unmeasured actions.
- Use at least 30 randomized measured repetitions per condition for paper results.
- Report count, min, mean, p50, p90, p95, p99, and max.
- Report strict opportunity-level `Success@B`.
- Average repetitions within each task, then bootstrap task IDs for a separately labeled
  case-macro `Success@B` 95% interval.
- Randomize case order with a recorded seed.
- Use paired bootstrap or paired permutation tests for identical case IDs and matched per-case
  opportunity counts across two systems.
- For the final external slice, average repetitions within `case_id`, then resample whole
  `task_cluster_id` clusters while retaining every case in a selected cluster. Use the same draw
  for both systems. `paired_clustered_exact_action_delta_ci` implements this exact estimand.
- Report every failed/invalid action; do not time only successful samples.

Both in-page runners default to three warmups and 30 repetitions per case. The 30-repetition rule
is an internally prespecified timing minimum, not a guarantee of capability power; preserve every
raw record, report interval width, and report all task-family results. Repetitions improve timing
precision but do not create new capability tasks. Lead capability results by unique-case
denominators, and apply a two-point quality margin only to a substantially larger frozen task set.

The reference implementation calls the two statistical estimands
`success_at_deadline` (strict opportunity-level) and `case_macro_success_at_deadline` (mean of the
per-case success rates). Its cluster interval belongs only to the latter. Balanced paper runs make
their point estimates equal; if they differ, report both and investigate the unbalanced run rather
than attaching the task-macro interval to the opportunity-level point estimate.

## Required metadata

```json
{
  "benchmark_version": "...",
  "git_commit": "...",
  "model_hash": "...",
  "checkpoint_hash": "...",
  "tokenizer_hash": "...",
  "heads_hash": "...",
  "dispatch_heads_hash": "...",
  "meta_file_hash": "...",
  "graph_hash": "...",
  "runtime_asset_evidence": {
    "heads_json": {"sha256": "...", "bytes": 1, "manifest_verified": true},
    "meta_json": {"sha256": "...", "bytes": 1, "manifest_verified": true},
    "dispatch_heads_json": {"sha256": "...", "bytes": 1, "manifest_verified": true},
    "tokenizer": null
  },
  "model_byte_evidence": {
    "sha256": "...",
    "bytes": 1,
    "manifest_verified": true
  },
  "suite_byte_evidence": {
    "sha256": "...",
    "bytes": 1,
    "identity_verified": true,
    "identity_source": "configs/data/pretrain-paper.yaml:evaluation_decontamination/..."
  },
  "bundle_manifest_byte_evidence": {
    "sha256": "...",
    "bytes": 1,
    "role": "parsed_bundle_manifest_trust_anchor",
    "external_expected_identity": null
  },
  "model_parameters": 1,
  "model_bytes": 1,
  "precision": "...",
  "backend": "webgpu|wasm",
  "backend_requirement": "explicit-*-no-whole-session-retry",
  "execution_provider_request": {
    "requested": "webgpu|wasm",
    "session_provider_count": 1,
    "whole_session_retry": false,
    "single_provider_session_creation_succeeded": true,
    "per_node_placement": "unknown",
    "per_node_fallback_status": "unknown"
  },
  "onnxruntime_version": "...",
  "browser": "...",
  "os": "...",
  "gpu_adapter": null,
  "power_mode": "...",
  "hardware_concurrency": null,
  "device_memory_gb": null,
  "latency_clock": "harness_ttfa_ms",
  "timeout_ms": 2000,
  "case_order_seed": "...",
  "cases": 1,
  "warmups": 3,
  "repetitions": 30,
  "concurrency": 1
}
```

Positive numeric values above illustrate the required type and protocol minima; they are not
benchmark claims. Unknown fields must be `null`, not guessed.

The current bundle manifest supplies the checkpoint digest and per-artifact byte counts and
SHA-256 values. Before parsing structured heads, metadata, dispatch heads, or a BPE tokenizer—and
before creating an ORT session—the benchmark pages fetch each runtime artifact as bytes, compute
SHA-256 in-browser, verify both digest and byte count against the manifest, and retain the observed
evidence in `runtime_asset_evidence` or `model_byte_evidence`. ORT receives the verified in-memory
model bytes. The exporter publishes the manifest atomically only after every declared graph passes
content-bound parity and every artifact is rechecked unchanged. The canonical hashes of parsed
`meta.json` and the parsed manifest remain separately labeled
`model_meta_canonical_sha256` and `bundle_manifest_canonical_sha256`; they are not raw-file hashes.
Each result also records the observed raw `bundle-manifest.json` digest and size as
`bundle_manifest_byte_evidence`, binding the preserved parsed trust anchor to the exact fetched
bytes; this self-describing manifest has no independent expected digest, so that limitation remains
explicit. The two local scoring suites are independently pinned by byte count and SHA-256 in the
tracked runner constants, matching `configs/data/pretrain-paper.yaml`, and are verified before JSON
parsing. Their observed immutable evidence is retained as `suite_byte_evidence`.
The exporter does not yet put a Git revision in the manifest, browser OS detection is left `null`,
power mode is `null`, and timeout is `null`. Those gaps must be closed or explicitly reported
before calling an artifact paper-grade.

## Action record

Minimum JSON record:

```json
{
  "case_id": "gui-click",
  "family": "computer_use",
  "repetition": 0,
  "backend": "webgpu",
  "input_tokens": 41,
  "input_bytes": 41,
  "output_tokens": 0,
  "tokenize_ms": 0.1,
  "inference_ms": 42.0,
  "decode_control_ms": 0.0,
  "dispatch_ms": 1.0,
  "parse_validate_ms": 0.5,
  "runtime_ttfa_ms": 43.5,
  "independent_validate_ms": 0.1,
  "harness_ttfa_ms": 43.6,
  "ttfa_ms": 43.6,
  "schema_valid": true,
  "success": true,
  "predicted_tool": "click",
  "expected_tool": "click"
}
```

`ttfa_ms` is retained only as an exact compatibility alias for `harness_ttfa_ms`; new analysis
selects the field named by `metadata.latency_clock`.

An autoregressive record additionally includes TTFT, TPOT, decode tokens/s, `decode_steps`,
visible output tokens, parse time, and the exact serialized output.

The current browser AR implementations are explicitly labeled
`decode_strategy: "full_context_recompute"` and `decode_cache: false`. The constrained arm masks
each next token to a finite trie containing one deterministically grounded action per tool plus an
EOS abstention terminal. These are reproducible implementation controls, not claims of a
cache-optimized decoder or a complete JSON-Schema grammar.
The standalone random-weight `decode-benchmark.html` graphs do not change this implementation
status and must not be combined with the trained runner's action quality or TTFA.

## Quality definitions

- **Exact action:** canonical tool name and canonical typed arguments match.
- **Schema valid:** an independent validator checks the supported required/type/enum/no-extra-key
  subset. General JSON Schema coverage is not claimed.
- **Abstention:** no tool is emitted for an irrelevant request.
- **First-step success:** first action agrees with a verified reference or reaches an allowed state.
- **Episode success:** the final environment state passes a deterministic evaluator.
- **Recovery:** a deliberately failed action is followed by a correct recovery within the step cap.

The model's own parser cannot be the only validator.

## Anti-patterns

Do not:

- Report only aggregate tokens/s.
- Compare byte tok/s directly with BPE tok/s.
- Stop timing before arguments are grounded and validated.
- Exclude invalid or slow samples from percentiles.
- Label simulated tool feedback as browser task success.
- Infer WebGPU use from successful session creation with fallback enabled.
- Report theoretical Q4 bytes as a measured quantized runtime.
- Tune on the frozen test pages or include their HTML in pretraining.

## Command-line analysis

Compute the rate required by output length:

```bash
PYTHONPATH=src python scripts/realtime_agent_benchmark.py requirements \
  --ttft-ms 200 --parse-ms 20 --output-tokens 16,32,64 --deadlines 500,1000
```

The retained `--output-tokens` flag name means required decode steps through the terminal step for
this calculation. The rate helper is a fixed-cost envelope: for model TTFA, pass TTFT as
`--ttft-ms` and runtime postprocessing as `--parse-ms`; for harness TTFA, pass tokenization plus
TTFT as the fixed predecode cost and runtime postprocessing plus independent validation as the
fixed postdecode cost. The aliases `--predecode-ms` and `--postdecode-ms` make
that latter interpretation explicit. In the illustrative command above, 200 ms and 20 ms are
harness-level fixed costs, not a claim that model-only TTFT was measured at 200 ms.

Summarize an exported browser result:

```bash
PYTHONPATH=src python scripts/realtime_agent_benchmark.py summarize \
  runs/bench-webgpu/example/metrics.json
```

The CLI recomputes all metrics from raw records. By default it selects
`metadata.latency_clock`, falling back to legacy `ttfa_ms` only when no clock is declared. When
metadata declares `cases` and `repetitions`, it rejects missing, duplicate, or unbalanced
case/repetition rows so failed trials cannot silently disappear. Nullable diagnostic stage fields
from runtime failures are counted as missing rather than causing the failed opportunity to be
dropped. Grouped reporting by `family` is on by default; pass an empty `--group-key` only for an
explicit ungrouped exploratory analysis. A successful CLI invocation does not by itself certify a
paper run: verify the frozen split, metadata and hashes, warmup/repetition minimums, provider,
single-request concurrency, finite timeout, and explicit TTFA clock from the raw artifact.
