---
title: LocalAgent Tool Calling (WebGPU)
emoji: 🛠️
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
license: mit
short_description: Sub-100M from-scratch tool-calling agent in the browser
---

# LocalAgent — tool calling in the browser (WebGPU)

A **sub-100M, pretrained-from-scratch** BPE agent that does **grounded tool
calling** and **multi-step planning** — running **entirely in your browser** on
[onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/) while requesting its **WebGPU**
execution provider (WASM is a separately labeled control, and the interactive demo can retry the
whole session on WASM). No server, no API key; the model is downloaded once and cached.

The staged model size is bound by `meta.json` and `bundle-manifest.json`; the currently verified
seed-2027 export is 10,524,544 parameters. Do not infer a parameter count from this static text.

The public Hugging Face model URL is intentionally omitted until authentication and upload are
verified. The current local paired-release receipt is tracked in
[`m336-hf-paired-release-local-current-63-tool-v1.json`](../../docs/paper/results/raw/m336-hf-paired-release-local-current-63-tool-v1.json).
Source: [LocalAgent](https://github.com/sangbumchoi/localagent).

## What it shows (generable dispatch — no fixed-N classifier)

- **Route gate** — a 5-way head (`web_search / computer_use / code / app_action / text`) on the ONNX
  `hidden` output; the `text` route is **abstention** (answer directly / no tool).
- **Tool selection** — a **dense two-tower selector**: the query tower projects `hidden`, scored by
  cosine against a precomputed per-tool description-embedding matrix over the **63-tool** release
  surface (50 standard tools plus 11 mobile and 2 productivity schemas)
  (`argmax_j q·tool_matrix[j]`). Adding/removing a tool is adding/removing a row — no retraining.
  The browser bundle also exposes `?selector=retrieval_then_dense`, which retrieves the top ten
  sidecar candidates before dense scoring; `?selector=retrieval` remains the retrieval-only
  ablation. Neither optional policy is presented as a quality promotion without a fresh held-out
  receipt.
- **Grounded arguments** — eligible string fields can use learned prompt-span pointers; other
  primitive fields use deterministic schema-aware grounding. Missing required values abort the
  action, and an independent validator checks the supported schema subset before dispatch.
- **Multi-step plans** — the rollout: pick a tool → ground it → feed back a simulated response →
  pick the next, until the route head emits `text`.

The current deployment repair and safety boundary are receipt-bound in
[`m336-hf-paired-release-local-current-63-tool-v1.json`](../../docs/paper/results/raw/m336-hf-paired-release-local-current-63-tool-v1.json).
The bundle binds the model and WebGPU selector matrices to the same 63-tool pool. Named, bounded
browser/mobile/productivity safety policies remain separate from learned-head accuracy and do not
execute external accounts.

## How it runs (honest version)

The structured path requests a single-provider ONNX Runtime session for the hidden-only
`action_model.fp16.onnx` graph. Autoregressive policies lazily load the final RL
`prefill.<precision>.onnx` / `decode.<precision>.onnx` pair, require its canonical lineage and
full-catalog BPE contract, run one prefill, and then rebind cache tensors for one-token decode.
ONNX Runtime Web does not expose per-node placement, so a WebGPU session request is
recorded as such and is not presented as proof that every node ran on the GPU.
The **route head**, **dense selector** (matmul + normalize + argmax over the precomputed tool
matrix), grounding, and planner loop are light JavaScript on top—a port of the Python
`routes` / `dense_selector` / `pointer_head` pipeline. Every emitted ONNX graph—including the
deployed fp16 hidden-only graph—is hard parity-gated against PyTorch before the exporter publishes
`bundle-manifest.json`. Legacy bundles remain usable by the interactive demo, but benchmark pages
reject them rather than silently timing the full logits graph as a structured action graph.

## Files
- `index.html` / `style.css` — the UI shell.
- `app.js` — byte tokenizer, onnxruntime-web session (WebGPU + WASM fallback), route+selector dispatch,
  grounding, and the planner rollout.
- `benchmark.html` / `benchmark.js` / `benchmark-cases.json` — a concurrency-1 held-out benchmark
  for complete-action latency (TTFA), p50/p90/p95/p99 tails, exact action quality, schema validity,
  parse/validation failures, and deadline-conditioned useful action rate. It exposes three
  separately labeled policies over the same cases and provider: (1) the current structured
  one-forward route/select/copy head, (2) raw autoregressive generation from exported final-token
  logits with EOS/token-cap stopping, and (3) grounded candidate-trie autoregressive
  decoding whose next tokens are masked by a trie of canonical, prompt-grounded schema-valid
  actions. Both AR controls require the lineage-bound final RL cache ABI, consume `[1,V]` logits,
  and report `decode_cache: true` / `prefill_then_kv_cached_decode`. Exported `next_token` is
  checked against the unrestricted logits argmax for compatibility. Structured output is never
  counted as generated tokens or labeled autoregressive.
  Downloaded JSON can be
  summarized with
  `python scripts/realtime_agent_benchmark.py summarize <result.json>`. The benchmark defaults to
  an explicit WebGPU-only session; use `benchmark.html?backend=wasm` for a provider-matched control.
- `backbone-benchmark.html` / `backbone-benchmark.js` — a separate latency-only harness for the
  exporter-produced matched 34M hybrid/all-attention random-weight pair. It accepts a
  `matched-backbones.json` URL through `?manifest=...`, verifies pair-manifest, provenance, graph,
  and copied-config hashes in the browser, rejects graphs with outputs other than `hidden`, and
  requests exactly one session provider. It sweeps 128/512/1,024/1,536 deterministic pre-tokenized
  IDs for both graphs, preserves eight randomized first-condition inferences, then three warmups
  and thirty randomized measured runs per graph/length. Hidden outputs use ORT's default CPU output
  contract and are released immediately after shape validation. The payload reports per-node
  placement/fallback as unknown because ORT Web does not expose it; WebGPU and WASM must be run as
  separate whole-session requests. Every payload is labeled `latency_only` and
  `untrained_random_weights`, contains no quality metric, and is available both as a JSON download,
  `window.__localAgentBackboneBenchmarkResult`, and the text of `#backbone-result-json`.

  When serving the repository root locally, the checked export can be loaded at:

  ```text
  /spaces/localagent-webgpu/backbone-benchmark.html?backend=webgpu&manifest=../../runs/webgpu/random-backbone-latency-seed-20260728/matched-backbones.json
  ```

  Replace `backend=webgpu` with `backend=wasm` for the explicit WASM arm. The result records
  cross-origin isolation, `SharedArrayBuffer` availability, and ORT's WASM thread setting; a plain
  `python -m http.server` result may therefore be a single-threaded CPU condition.
- `decode-benchmark.html` / `decode-benchmark.js` — a standalone latency-only harness for the
  exporter-produced matched random-weight cache-bearing pair. It verifies the pair manifest,
  parity-gated model provenance, copied config, and selected prefill/decode graph bytes before
  creating any ONNX Runtime session. The export has a dynamic-prompt prefill graph and a separate
  decode graph with a fixed `T=1` token axis. Its hard parity gate covers multiple prompt lengths
  and iterative steps: greedy token IDs must match exactly for ONNX versus cached PyTorch and for
  cached PyTorch versus fresh full-context PyTorch, while cache values must stay within the
  declared precision-specific tolerance.

  Each condition uses exactly one prefill pass for the first token and 31 one-token decode passes
  by default, at 128/512/1,024/1,536 input tokens. It retains at least three warmups and thirty
  measured repetitions per model/length under a seeded randomized schedule. WebGPU sessions
  request `next_token` on CPU and every present cache on `gpu-buffer`; measured present tensors are
  rebound directly into the next decode call without reading cache contents into JavaScript, and
  superseded/final tensors are disposed. This is interface-level evidence, not proof of physical
  residency or per-node placement. The graph returns fresh presents each step—attention uses
  append/concat and short-conv replaces a fixed-width tail—rather than an in-place or paged cache.
  The separately requested WASM condition requests and reports CPU cache outputs.

  Results include TTFT, TPOT, wall decode tok/s, summed decode-inference time, model-only decode
  tok/s, graph/pass counts, cache logical bytes/output locations, allocation/disposal accounting,
  failures, verified identities, and environment evidence. They contain no capability or quality
  metric and are exposed through `window.__localAgentDecodeBenchmarkResult`.

  The checked evidence bundle opens from a repository-root server at:

  ```text
  /spaces/localagent-webgpu/decode-benchmark.html?backend=webgpu&manifest=../../runs/webgpu/random-cached-decode-latency-seed-20260728-v2/matched-decode.json
  ```

  Use `backend=wasm` for a separate CPU-provider result. For a newly exported directory, replace
  the `manifest` query value with that directory's `matched-decode.json`.

  Across three Apple M5 Chrome/WebGPU page/session runs, median-of-three hybrid p50 wall-decode
  tok/s at 128/512/1,024/1,536 tokens were 74.05/64.54/57.53/47.15 for 34.2M,
  90.87/89.26/80.94/79.77 for 15.6M, and 159.23/160.46/143.49/127.57 for 10.5M. Only the 10.5M
  pair clears the 100 tok/s engineering reference at every context. See the tracked
  [summary/raw/config index](https://github.com/sangbumchoi/localagent/blob/main/docs/paper/results/README.md).
  These are random-weight latency results and are not production model results. `app.js` instead
  requires a separately exported final-RL cached bundle with matching tokenizer, catalog, parity,
  checkpoint lineage, and training-lineage sidecar.
- `model.fp16.onnx`, `action_model.fp16.onnx`, `heads.json`, `meta.json`,
  `dispatch_heads.json`, `bundle-manifest.json` — the exported inference bundle (**not in the
  source repo**; deploy artifacts). The hidden-only action graph is used for the structured policy,
  while a separately pinned final-RL cached bundle supplies production autoregressive decoding.
  Place that complete export under `cached/`; the primary manifest must pin
  `cached/provenance.json`, which recursively pins cached metadata, training lineage, tokenizer,
  config, and graphs.
  Benchmark pages require the manifest and action graph; hash the fetched graph, heads, metadata,
  dispatch heads, and tokenizer assets in-browser; compare byte counts and SHA-256 values with the
  manifest before parsing; and create the ORT session from verified in-memory model bytes.
  The action and DOM case files have separate tracked byte/SHA-256 pins matching the frozen-suite
  identities in `configs/data/pretrain-paper.yaml`; tampering stops the run before case parsing.
  Results also preserve the observed raw manifest bytes/hash as the self-describing trust anchor.

## Deploy
See **`DEPLOY.md`** for copy-paste build + push commands. In short: export the bundle from the
latest checkpoint and upload the static app plus the generated bundle files into a `sdk: static`
Space:

```bash
python -c "from localagent.inference.export.to_onnx import export_web; \
           export_web('runs/sft-webgpu-proxy-pilot-hybrid-seed2027/latest.pt', 'build/web', action_only=True)"
```

All generated bundle files must sit next to `index.html`. Export refuses to publish its manifest
unless every emitted fp32/fp16 graph passes the recorded numerical thresholds; do not hand-create
or omit that manifest. The graphs are standard opset-17. The interactive app's `auto` mode can
retry the whole session on WASM if WebGPU session creation fails. Benchmark pages require an
explicit WebGPU or WASM condition, perform no whole-session retry, surface provider failure, and
report per-node placement and per-node fallback as unknown.

## Single-step DOM closed loop

`browser-tasks.html` loads `browser-task-cases.json` and runs the production `callOnce` path before
dispatching the predicted action into a fresh versioned fixture. `browser-tasks.js` independently
checks the selected tool's exported JSON Schema, dispatches click, double-click, focused typing,
keyboard, scroll, drag/drop, synthetic cursor-move, or whitelisted local navigation events, waits
for a paint, and then checks the declared final DOM state. Case order is randomized from a recorded
seed. The downloadable JSON retains raw model outputs, pre/post assertions, schema diagnostics,
event details, harness TTFA, tool time, closed-loop time, environment metadata, and the exact order.

The page defaults to an explicit WebGPU execution provider and does not silently fall back; use
`browser-tasks.html?backend=wasm` only as a separately labeled control. This is deliberately a
single-step, text-only, same-page microbenchmark. Its events are synthetic and untrusted, the
cursor action does not move the physical pointer, and navigation stays inside a disposable local
iframe. It does **not** measure visual grounding, multi-step completion, trusted OS input, external
site navigation, or browser-wide automation.

## Screenshot-conditioned action probe

`visual-action.html` is an isolated screenshot-conditioned sidecar probe. It accepts a local
PNG/JPEG/WebP screenshot and task text, requests WebGPU (or an explicitly selected WASM control),
and emits the seven-action/pointer ABI. The graph is pinned by `visual-action-manifest.json` but is
deliberately not part of the text-only release manifest: it comes from the bounded m714
AndroidControl pilot and still has no Android emulator or official AndroidControl verifier.

## Stateful mobile and productivity trajectory gate

`mobile-trajectories.html` runs the v11 text-first policy against three local workflows (Gmail
compose/send, Notion capture, and a mail-page search/open flow). It feeds the current state JSON
into the next prompt and checks the shipped `meta.json` schema, exact arguments, preconditions,
state transitions, and complete-trajectory pass@1. The suite is a deployment diagnostic only:
it contacts no external service, does not send email or edit Notion, has no screenshot grounding or
trusted OS input, and must not be reported as AndroidWorld, BrowserGym, OSWorld, or MCP success.
The corrected v14 receipt records the current result: guarded dense dispatch reaches `4/13` exact
actions and `3/13` closed-loop transitions; no-guard dense reaches `1/13`. Complete-trajectory
pass@1 remains `0/3`, so this page is a failure gate rather than a publish-ready agent claim.
