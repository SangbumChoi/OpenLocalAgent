# From Tokens per Second to Time to First Action:
# Co-Designing Compact WebGPU Agents

> Anonymous working draft for the
> [1st Workshop on Small Language Models for Agentic Systems](https://slmw2026.github.io/#call-for-papers),
> NeurIPS 2026. Every bracketed field is deliberately unresolved; do not submit or remove the
> brackets until the corresponding artifact exists. This is a >6,000-word long-form evidence
> draft, not the required four-page submission. The compact
> [NeurIPS 2026 LaTeX source](slmw2026/main.tex) and visually verified
> [working PDF](../../output/pdf/slmw2026-compact-webgpu-agents-wip.pdf) now exist; conservative
> checklist answers deliberately expose the unfinished reproducibility and release work.
> Anonymous artifact packaging remains open.

The central thesis and reviewer claim ladder are maintained in
[`WORKSHOP_INSIGHT.md`](WORKSHOP_INSIGHT.md): feature-materialization parity is a deployment
interface, and a fast parity-correct graph is not evidence of a useful agent.

## Abstract

Browser language models are commonly described by decoded tokens per second, although an agent
cannot execute a partial JSON token. We study **time to first complete action** (TTFA): elapsed time
until a schema-valid, dispatch-ready action under an explicitly named clock. The intended user
clock starts when an observation is available; the current implementation reports harness TTFA
from prompt tokenization through independent validation and does not claim user TTFA. Coupled with
exact action success, TTFA prevents a fast invalid policy from winning. We implement a reproducible
concurrency-one WebGPU benchmark and specify comparisons among raw autoregressive JSON,
grounded candidate-trie autoregression, and a one-forward structured policy composed of a route
gate, dense tool selector, and grounded typed arguments. We also specify two
parameter-matched 34M backbones: all multi-query attention and a hybrid with periodic attention
among gated short-convolution layers. These counts exclude the separately learned action heads.
The matched runs will use the same 16K ByteLevel BPE
tokenizer, training tokens, sample order, and action heads. In a separate random-weight systems
gate on one Apple M5, a 10.5M four-layer hybrid sustained a median 127.57--160.46 decoded tokens/s
across 1,536--128-token prompts with cache-bearing fp16 WebGPU graphs; the matched all-attention
control fell below 100 tokens/s at the longest context. This is deployment evidence, not trained
capability evidence. In a separate observed exploratory seed-2026,
approximately-one-token-per-parameter MPS training proxy, the 10.525M hybrid reached held-out
BPB 2.0319 versus 2.0989 for the matched 10.547M all-attention arm after exactly 10,551,291 loss
tokens each. Ten thousand paired document bootstraps put the attention-minus-hybrid BPB difference
at 0.0669, 95% interval `[0.0634, 0.0707]`; this conditions on one architecture seed and
240 documents. A clean prospective confirmatory set at seeds 2027–2029, evaluated with CPU-fp32
scorecards, yielded attention-minus-hybrid BPB differences of 0.07084, 0.07353, and 0.07449
(mean 0.07295; model-based Student-t, df=2, 95% interval `[0.06825, 0.07765]`). Hybrid was favored
3/3, although the exact sign test remains underpowered (one-sided `p=0.125`, two-sided `p=0.25`).
Separately, the exact seed-2026 pretrain-only hybrid checkpoint sustained 116.476--137.711 median
p50 wall tokens/s across the same contexts in three M5/Chrome WebGPU runs and cleared 100 in every
run/context. It did not pass the joint tail gate: median p95 TPOT was at most 10 ms only at
512 tokens, and no context passed both thresholds in every run. This is neither confirmatory-set
latency nor agent capability. The provisional seed-2027 hybrid then completed a bounded
midtrain/SFT/offline-RL pilot. Midtraining and SFT improved their targeted held-out loss and token
accuracy, but midtraining slightly regressed general held-out loss and RL realized 12 optimizer
updates with zero held-out delta. The exact SFT action graph was fast at fixed 512-token inputs:
three WebGPU runs had TTFA p50 24.75/24.55/25.20 ms and p95 34.405/34.30/34.80 ms. Capability
failed. The unique-case result was 1/20 overall—0/19 tool-required and 1/1 abstention—with
0/8 executable local-DOM tasks. Repeated timing rows yield 90/1,800 exact action and 0/720 DOM
success, but are not independent capability trials. An exploratory offline check traces the
failure to a deployment
feature-materialization shift: natural prompts yielded 17/20 correct routes and 17/19 tool
selections but zero text predictions, missing the sole abstention; real, unmasked pre-assistant
filler routed every case to text. Native PyTorch, fp32/fp16 ONNX, and exported heads agree, so
export/precision mismatch is not causal. This
diagnostic is not natural-prompt WebGPU quality. The fixed-512 stress gate fails. A corrected
512-compute runner that dispatches from the natural marker position is implemented and its
protocol was superseded before an external pre-run timestamp or browser run; a new freeze against
the final cached-autoregressive bundle and current runner remains pending. These reused suites
were inspected during
diagnosis and therefore test deployment parity, not untouched capability. The 34M five-TPP screen,
its promoted 20-TPP/downstream comparison, a trained cached-action bundle and browser run,
BrowserGym, cross-device evaluation, and the full policy comparison remain
pending.

## 1. Introduction

An interactive browser agent must turn an observation into an executable decision quickly enough
to preserve user flow. “Tokens per second” is an incomplete requirement for this setting. It
depends on the tokenizer, excludes prompt processing and validation, and gives the same credit to
a token that completes an action and a token that merely opens a JSON object. The ambiguity is
especially severe when comparing byte and BPE models or autoregressive and structured policies.

Existing interactive inference standards separate time to first token (TTFT) from time per output
token (TPOT). [MLPerf's small-LLM interactive limit](https://mlcommons.org/2025/09/small-llm-inference-5-1/)
of 30 ms TPOT corresponds to 33.3 decoded tokens/s, while its
[newer reasoning workload](https://mlcommons.org/2026/03/mlperf-inference-gpt-oss/) uses 15 ms
TPOT, or 66.7 tokens/s. These are useful diagnostics, not evidence that every agent requires 200,
400, or 600 tokens/s. Computer-use latency also includes planning calls, excess steps, tool
execution, and page rendering; [OSWorld-Human](https://arxiv.org/abs/2506.16042), for example,
finds that planning/reflection calls dominate tested computer-use latency and that agents take
more steps than human reference trajectories.

We ask a narrower question: what model and action representation maximize the probability of a
correct browser action before an explicit latency budget on WebGPU? Our contributions are:

1. We define TTFA and `Success@B`, a deadline-conditioned exact-action metric.
2. We provide a WebGPU/WASM benchmark that exports raw per-action timing, correctness, schema
   validity, bundle hashes, runtime metadata, and cold/warm phases, plus an offline task-clustered
   confidence-interval protocol.
3. We implement and prespecify a comparison of a one-forward structured action policy with raw
   and grounded candidate-trie autoregressive serialization baselines. A fixed-512 structured
   pilot is measured and fails capability; the matched autoregressive comparison remains pending.
4. We specify a parameter-matched comparison between a 34.199M hybrid and 34.276M all-attention
   backbone instead of presuming that lower asymptotic attention cost is faster in a browser, and
   measure a random-weight cached-decode sizing frontier down to 10.5M parameters. Backbone and
   deployed-policy parameter counts are reported separately.

## 2. Complete-action latency

For an autoregressive action requiring `N_decode >= 1` model steps, including any EOS or terminal
step the runtime must observe, and decode rate `r`,

```text
TTFA_AR,harness = tokenize + TTFT + (N_decode - 1) / r + parse + validate.
```

The rate required by a deadline `B` is therefore

```text
r_min = (N_decode - 1) / (B - tokenize - TTFT - parse - validate).
```

For the illustrative rate calculation below we fold tokenization into a 200 ms fixed pre-decode
cost and reserve another 20 ms for postprocessing. A 32-step decode then needs 39.7 tokens/s to
finish within one second, but 110.7 tokens/s to finish within 500 ms. A 64-step decode needs 80.8
and 225 tokens/s respectively. Visible `output_tokens` may exclude EOS, so benchmark artifacts use
`decode_steps` for this equation. There is no action-independent 200/400/600 tokens/s threshold.

The requested candidate rates are now an executable counterfactual:

```bash
PYTHONPATH=src python scripts/realtime_agent_benchmark.py calibrate \
  --ttft-ms 200 --parse-ms 20 \
  --decode-steps 16,32,64,128 --decode-rates 200,400,600 \
  --deadlines 500,1000
```

Under exactly those fixed costs, the harness complete-action TTFA scenarios are:

| Decode steps through terminal | 200 tok/s | 400 tok/s | 600 tok/s |
|---:|---:|---:|---:|
| 16 | 295.0 ms | 257.5 ms | 245.0 ms |
| 32 | 375.0 ms | 297.5 ms | 271.7 ms |
| 64 | 535.0 ms (misses 500 ms) | 377.5 ms | 325.0 ms |
| 128 | 855.0 ms (misses 500 ms) | 537.5 ms (misses 500 ms) | 431.7 ms |

All listed scenarios meet one second. This table does not establish that 600 tokens/s is
universally necessary or sufficient: it excludes tool execution, page rendering, observation
capture, and later reasoning steps, and its action-length grid is not a sampled workload. The CLI
therefore labels the result non-empirical and deliberately emits no mean, percentile, or
confidence interval over the arbitrary grid. Confirmatory reporting substitutes observed
per-action `decode_steps` and stage timings, reports each action family, estimates `Success@B`
uncertainty by task-cluster rather than timing row, and summarizes tail TTFA across independent
browser runs.

Our structured policy emits no serialized action tokens:

```text
TTFA_structured =
    tokenize + one backbone forward + route/select/ground/validate.
```

For deadline `B`, the primary score is

```text
Success@B =
  count(exact action ∧ independently schema-valid ∧ TTFA ≤ B) / opportunities.
```

We report `B ∈ {0.5, 1, 2}` seconds as proposed comparison points, not universal human-factors
standards. We separately report visible acknowledgment latency and closed-loop time to the next
painted state.

## 3. Model and action policy

### 3.1 Matched browser backbones

Both candidates are dense causal decoders with a 16,384-token ByteLevel BPE vocabulary, tied
embeddings, width 448, 12 layers, seven query heads, one KV head, SwiGLU, RMSNorm, RoPE, QK
normalization, and a 2,048-token context. The all-attention control uses FFN width 1,328 and has
34.276M backbone parameters. The hybrid uses FFN width 1,152 and repeats
`[short convolution, short convolution, attention]` four times for 34.199M backbone parameters.
Parameter matching therefore changes both the mixer pattern and FFN allocation. The estimand is a
compound backbone-configuration treatment, not an isolated causal effect of replacing attention
with convolution.

The hybrid transfers only the testable principle behind Kimi K3's
[official 93-layer configuration](https://github.com/MoonshotAI/Kimi-K3/tree/7c5be9599120d7993748de66a76128614f15f210)—69 KDA and
24 Gated-MLA layers—as well as
[Kimi Linear](https://arxiv.org/abs/2510.26692), and
[Hymba](https://arxiv.org/abs/2411.13676)—periodic global attention among cheaper local mixers.
Its gated depthwise convolution is not Kimi Delta Attention. At 2K context and one KV head, the
hybrid saves only about 4.2 MB of fp16 inference state relative to the control, while its mixer
contains more projection parameters than MQA. WebGPU operator fusion, memory traffic, and kernel
launches therefore determine the winner. K3's technical report discloses its data domains,
optimizer/schedule, progressive-context curriculum, and SFT/RL/MOPD stages, but not exact source
identities/weights, training-token count, several selected hyperparameters, or a browser-portable
kernel contract. Those missing details are not inferred here.

### 3.2 Structured dispatch

The final normalized hidden state feeds a five-way diagnostic route head. Operationally, the
runtime uses it as a binary text-versus-tool gate: every non-text route enters one global dense
two-tower selector whose precomputed tool-description matrix supports changing the catalog without
a fixed `N`-class output head. A configured subset of string argument names uses learned start/end
pointer spans over the prompt; paths and URLs are included in that pointer subset. Enums, numbers,
booleans, and remaining primitive fields use deterministic schema-aware grounding. If a required
value cannot be grounded, the call is invalid rather than filled with an empty string.

The browser action graph returns hidden states only, avoiding the 16K vocabulary projection. The
export bundles the exact tokenizer, marker IDs, heads, and SHA-256/size manifest. The manifest is
published only after every emitted fp32/fp16 graph passes output-specific PyTorch parity
thresholds; benchmark-grade browser runs verify graph, tokenizer, metadata, and head bytes before
using them.

## 4. Training

We train from scratch because arbitrary frontier weights do not align with the 16K tokenizer,
width, or convolution layers. External teachers contribute filtered sequence/trajectory
distillation rather than unaligned tokenwise KL; teacher output is not an executed outcome unless
a separate verifier establishes it. Distillation-first is a compute allocation rather than a
claim that RL cannot work: the
[DeepSeek-R1 release](https://github.com/deepseek-ai/DeepSeek-R1) reports stronger distilled small
models than its compared small-model RL setup while also showing RL-induced reasoning at frontier
scale.

Pretraining targets approximately 20 tokens per parameter, with 5- and 10-token/parameter proxy
sweeps. The current 50/15/25/10 character-budget configuration uses pinned revisions of
[FineWeb-Edu-Dedup and Cosmopedia v2](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus),
permissively licensed
[CodeParrot Clean Python](https://huggingface.co/datasets/codeparrot/codeparrot-clean), and
[WebSight HTML](https://huggingface.co/datasets/HuggingFaceM4/WebSight); we will report realized
final-token shares rather than assuming they equal the download weights. Separately
licensed/provenanced tests, API documentation, multilingual code, and JSON/Schema/OpenAPI/YAML
remain proposed additions and are not claimed as configured data. We deduplicate before a stable
document split and train the tokenizer on training documents only. The base corpus publishes a
checksummed, content-bound split map; every general/code/structured repack must import it so a base
validation document cannot become midtraining data merely because its family is prepared
separately. The paper run will screen real, revision-frozen BFCL, Mind2Web, WebLINX, BrowserGym,
action-suite, DOM-suite, and synthetic agent-evaluation prompt exports before both operations. The
corpus-owned seven-suite requirement cannot be weakened by the supplied list-manifest. That
manifest pins every file's size and SHA-256, and the corpus manifest records both those artifacts
and the normalized denylist fingerprint. The four external exports are not shipped in this
repository and must not be treated as present until supplied. The bounded downloader records upstream
exact-duplicate drops as counts only; exact duplicates that reach local staging retain alias
provenance without a second text copy. Code-like documents use exact dedup only so case-sensitive
identifier changes are not erased by heuristic near-dedup.

Midtraining targets 70–100M supervised tokens. Its scheduled loss-token mixture moves from
45/25/15/15% general/code/structured/rule-audited-agent data to 30/25/15/30%. Canonical
`Conversation` rows apply loss only to assistant actions; text/code retains full-token loss. The
runner uses deterministic loss-token deficit scheduling and supervised-token-normalized gradient
accumulation; entitlement/served state is checkpointed so an interrupted run resumes exactly,
while a legacy draw-only checkpoint is rejected because its prior token entitlements are
unprovable. Both paper arms checkpoint every 100 steps and auto-resume only when optimizer,
Python/Torch/backend RNG, history, global/per-source accounting, scheduler observations, lineage,
and resolved execution identity validate. Bit-exact continuation is tested on one CPU runtime;
cross-backend exactness is not claimed. The first production planner replay rejected the former
2,500-step ceiling because it scheduled only 7,429,270 supervised tokens. The corrected matched
25,000-step configs each deterministically plan 85,536,552 input and 74,836,551 supervised tokens,
inside the 70–100M target. The general/code/structured split maps are an independently checked,
document-disjoint exact union of all 504,010 frozen parent records. These are plan and lineage
facts, not trained-model results.
Pretraining and midtraining checkpoints preserve optimizer/RNG state, realized input/loss-token
counts, and strict config/data/tokenizer/code lineage. Pretraining persists its validation history;
midtraining reports scheduled entitlements, deficits, row draws, realized per-source input/loss
shares, and deterministic same-draw held-out pre/post loss and token accuracy; independently
packed held-out roots must prove document-ID/content-hash disjointness through verified split
assignments. SFT reports per-file assistant-loss tokens, the same lineage, and held-out assistant
loss, token accuracy, and teacher-forced all-assistant-token exactness, but is not yet resumable.
These are preserved measurements; numeric promotion thresholds require baseline runs and are not
yet configured.
The implementation supports WSD and cosine schedules, but no WSD advantage is claimed without an
identical-token, identical-data-order cosine control. That estimates a schedule change at shared
hyperparameters; a separate performance-envelope comparison must tune peak learning rate and
batch for each schedule. Changing the decay-window data mix is a third factor rather than evidence
for the schedule itself.

The planned SFT corpus balances tool/argument binding, DOM actions, failures/recovery, abstention,
and confirmation. The current implementation jointly optimizes the LM with fixed-class tool and
pointer heads, then fits route and dense-selector probes on frozen backbone features for every
assistant decision, including marker-framed multi-turn histories; it does not yet include a
configured general-instruction replay source. The train config pins both local browser suites and
forbids canonical equality with any of their 21 unique queries across every training user turn;
shared enums, no-argument intents, and template vocabulary remain possible. The current RL control
is offline GRPO with exact
normalized tool-AST and exact text-match rewards, a schema-aware format bonus, and a truncation
penalty. It requires a separate held-out conversation artifact, rejects canonical-row and
rendered-prompt overlap, separately fingerprints the complete artifacts and exactly scored
single-turn projections, validates parent tokenizer/stage lineage and target length, and stores
greedy pre/post exact/reward/format metrics plus attempted-versus-realized update accounting. A
[BrowserGym](https://github.com/ServiceNow/BrowserGym) ablation with final-state, step-count,
restraint, and measured-latency rewards remains future work. Because this control updates only the
autoregressive LM, its checkpoint invalidates rather than silently reuses the SFT structured heads.
No learned judge supplies the primary reward.

The bounded seed-2027 run now supplies a first numeric stage audit. Midtraining improved the agent
holdout from 7.7869 to 2.6371 loss and 3.71% to 69.80% token accuracy, while general loss
regressed from 5.7064 to 5.7342. SFT improved held-out assistant loss from 2.7320 to 1.8146 and
token accuracy from 67.29% to 73.13%, but only 1/65 complete sequences was exact. Offline RL
realized 12 updates from six informative groups, yet all 53-row greedy held-out metrics were
unchanged. These are pilot measurements, not promotion thresholds or final-model results.

## 5. Experimental protocol

The planned paper protocol uses three layers; layer 3 is not yet wired. Protocols must be
externally timestamped before their corresponding outcomes are collected:

1. Frozen typed function calls and irrelevant-tool abstention.
2. Versioned local DOM microtasks with actual dispatch and final-state checks.
3. A held-out BrowserGym subset for multi-step final-state success.

Each system sees identical cases at 128, 512, 1,024, and 1,536 final-tokenizer input tokens; the
largest bucket reserves room inside the 2,048-token sequence limit for autoregressive output.
The preserved seed-2027 runs use an internally prespecified pre-assistant-padding stress
construction; no external timestamp is currently available, so this is not a formal
preregistration.
Because that materialization changes the structured feature, the frozen follow-up protocol uses
the implemented corrected fixed-compute arm: append filler after the natural assistant marker,
run the same token count, dispatch from `hidden[natural_input_tokens - 1]`, and bound pointer
scans to the natural span. The two arms are reported separately.
WebGPU and WASM are separate, single-provider arms at concurrency one. We report download, session
creation, first/shader-compiling inference, and warm steady state separately. Warm experiments use
at least 30 randomized repetitions per condition. The paper protocol includes returned failures,
invalid actions, and runtime exceptions in percentiles. Because an in-flight ONNX Runtime Web
inference is not cancellable, the v0.4 runner must abort the entire collection on a finite
per-action watchdog before paper-grade collection.
Action and DOM suite bytes are size/SHA-256 checked against immutable pins shared with the corpus
decontamination policy before JSON parsing; results retain both observed suite evidence and the raw
bundle-manifest trust-anchor identity.
`Success@B` confidence intervals resample task IDs, not repeated timing rows. Paired system
differences use the same task clusters.

### 5.1 Decision rules to freeze before final collection

The current 20-case action suite moves in five-point increments and cannot resolve a two-point
non-inferiority margin. It is diagnostic only. The final structured-versus-autoregressive
comparison requires at least 200 new unique tasks plus an external benchmark slice, paired by task
and clustered by task rather than timing repetition. Promote the structured policy only if the
task-clustered 95% interval for its exact-action difference has lower bound above `-0.02` **and**
its median-of-run p95 TTFA is at least 15% lower. These thresholds and the new task identities must
be externally timestamped before outcomes are inspected.
The implemented
[`fresh external evaluation contract`](FRESH_EXTERNAL_EVAL_CONTRACT.md) binds the source revision
and every declared training artifact, deterministically selects before auditing, fails on
prompt/shingle or labeled-action-template overlap, and independently AST-scores paired outputs.
No real external export is currently present or frozen; the final comparison therefore remains
pending.

The five-token-per-parameter 34M matrix is an architecture screen. It can promote a backbone
configuration to the planned 20-token-per-parameter/downstream comparison; it is not the final
quality selection. For autoregressive deployment, the selected trained model must also meet the
absolute 100 tokens/s and tail-TPOT gate. A larger model may instead qualify only for the
one-forward structured mode by meeting the absolute TTFA and `Success@B` gates.

- Do not report Q4 until a real quantized browser graph, rather than a byte estimate, runs.

## 6. Results

> Replace only from immutable raw benchmark artifacts. Report negative results.

### 6.1 Pre-training-free architecture latency gate

Before spending the matched training budget, we exported the 34.199M hybrid and 34.276M
all-attention backbones with identical deterministic random initialization and measured their
hidden-only fp16 graphs on a 32 GB Apple M5 MacBook Air in Chrome 150. ONNX Runtime Web 1.27.0 was
requested with `executionProviders: ["webgpu"]`; no whole-session retry was allowed. Timing starts
immediately before `session.run` and ends after the full fp32 hidden tensor is resolved at the
default CPU output location, so it includes output readback.

Three page/session runs each retain 30 warm measurements after three excluded warmups. The table
reports the median of the three within-run percentiles and the median paired p95 ratio.

| Input tokens | Attention p50 / p95 median (ms) | Hybrid p50 / p95 median (ms) | p95 speedup |
|---:|---:|---:|---:|
| 128 | 74.50 / 83.38 | 43.90 / 48.61 | 1.71× |
| 512 | 115.65 / 130.71 | 66.25 / 82.75 | 1.58× |
| 1,024 | 304.50 / 321.41 | 154.70 / 174.54 | 1.87× |
| 1,536 | 593.05 / 621.68 | 288.65 / 300.30 | 2.00× |

Order is paired, balanced, and seeded. All 720 measurements completed. The hybrid passes this
one-device latency gate, but random weights cannot establish quality or justify final architecture
selection. Absolute latency shifted materially after the first run, while the paired advantage
remained. Because the failed WASM attempt occurred between runs one and two, these are not clean
independent system-state repeats; the artifact preserves each run and does not pool away that
variation. ORT reported version 1.27.0 and exposed a WebGPU device, but did not expose adapter
identity or per-node placement. We report the requested WebGPU end-to-end condition with per-node
fallback unknown, not an all-node-GPU claim. The three immutable raw artifacts and their hashes
are tracked with a test that recomputes the summary statistics. Power reporting was also
inconsistent (AC Power while the battery
discharged), and thermal status was unavailable. A separate single-thread WASM control lost the
browser target during inference after both sessions loaded and yielded no valid latency
measurement.

### 6.2 Random-weight cached-decode sizing gate

The hidden-only gate does not measure iterative decoding. We therefore exported each random-weight
arm as two fp16 graphs: prompt prefill and fixed-`T=1` greedy decode. Before publication, the exact
graph bytes had to match PyTorch cached decoding at prompt lengths 1, 8, and 31 for four successive
decode steps, including exact greedy token IDs and cache tensors within the declared fp16
tolerance. In the browser, attention K/V and convolution state were requested and reported as
`gpu-buffer`; each graph's present tensors were rebound directly as the next decode graph's past
inputs without JavaScript cache readback. The implementation appends/concatenates into fresh
present tensors. It is neither an in-place nor a paged cache, and separate prefill/decode sessions
may duplicate weights.

Three clean foreground page/session runs per size each excluded three warmups and retained 30
seeded, balanced measurements for every arm and context. Each sample generated 32 tokens with one
prefill and 31 cached-decode graph calls. The table reports the median of the three within-run p50
wall-throughput estimates. Entries within a row follow contexts 128 / 512 / 1,024 / 1,536. Tail
latency is reported as p95 TPOT rather than as a high percentile of throughput.

| Matched size | Attention p50 wall tokens/s | Hybrid p50 wall tokens/s | Hybrid p95 TPOT (ms) | Hybrid ≥100 tokens/s? |
|---|---:|---:|---:|---:|
| 34.2M | 52.57 / 44.82 / 36.07 / 27.70 | 74.05 / 64.54 / 57.53 / 47.15 | 16.17 / 18.06 / 20.01 / 22.71 | No |
| 15.6M | 64.75 / 62.73 / 58.39 / 46.61 | 90.87 / 89.26 / 80.94 / 79.77 | 12.98 / 12.96 / 15.22 / 15.04 | No |
| 10.5M | 130.20 / 126.90 / 107.19 / 91.73 | 159.23 / 160.46 / 143.49 / 127.57 | 7.81 / 8.10 / 9.14 / 9.65 | Yes, all four contexts |

The 10.5M hybrid has width 384 and only four layers, with
`[convolution, attention, convolution, attention]`; it is parameter-matched within 0.214% to a
four-layer all-attention control. Its 10.525M parameters are a latency-selected deployment
candidate, not a replacement for the planned 34M quality screen. The 100 tokens/s line
is an engineering reference requested for this deployment study, not a universal realtime
standard. Cross-size rates are descriptive because the three size pairs were collected in
separate two-arm runs rather than one interleaved six-arm experiment. Every graph uses
deterministic random weights, so these results establish no language, code, tool-use, or agent
quality. The matched 34M five-tokens-per-parameter training matrix remains staged and unstarted.
ORT Web exposed a WebGPU device and accepted an exact one-provider session request with no
whole-session retry, but adapter identity and per-node placement/fallback remain unknown.
The tracked summaries for the
[34.2M](results/m5-webgpu-cached-decode-20260728.summary.json),
[15.6M](results/m5-webgpu-cached-decode-16m-20260728.summary.json), and
[10.5M](results/m5-webgpu-cached-decode-10m-20260728.summary.json) pairs bind the raw artifact
hashes and are recomputed by regression tests.

### 6.3 Observed exploratory seed-2026 one-TPP pretraining proxy

We separately trained the matched 10.5M pair on a bounded corpus. The raw mixture contains
120,014,016 accepted characters in 24,125 documents: 78,002,243 FineWeb-Edu-Dedup characters,
18,001,053 Cosmopedia-v2 characters, and 24,010,720 permissively licensed Python characters.
Its JSONL SHA-256 is
`22d4270ad6157a9701e86be8bfd73a4fc9c480dd2cfd82337a4d6a5218183e6c`. Quality filtering retained
24,004 documents. A fixed 23,764/240-document split contains 28,045,897 train tokens and 287,995
packed validation tokens; the full-document scorecard covers 287,615 source tokens. The packed
manifest SHA-256 is
`6a10cc606902a648258dc58ddb3ba19aa68c5b5ed6d812fcb2f06cbedfcaa9fd`, and the train-only 16K
tokenizer SHA-256 is
`8365405524329487aea3b087cc999db887d8276115e67e88ebfcf7901b15617c`.

Both arms used seed 2026, AdamW, the same WSD schedule and draws, 322 updates, and exactly
10,551,291 realized loss tokens. The all-attention arm has 10,547,072 parameters and the hybrid
10,524,544, a 0.214% difference. Because both arms use WSD, the experiment contains no
WSD-versus-cosine evidence.

| Validation slice | Attention CE / BPB / top-1 | Hybrid CE / BPB / top-1 | Hybrid change |
|---|---:|---:|---:|
| Aggregate, 240 docs | 6.0547 / 2.0989 / 15.24% | 5.8617 / 2.0319 / 17.28% | CE −0.1930; BPB −0.0669; +2.04 pp |
| General, 216 docs | 6.1901 / 2.0785 / 14.39% | 6.0043 / 2.0161 / 16.23% | CE −0.1859; BPB −0.0624; +1.84 pp |
| Code, 24 docs | 5.3951 / 2.2204 / 19.39% | 5.1673 / 2.1266 / 22.42% | CE −0.2278; BPB −0.0937; +3.03 pp |

Ten thousand paired nonparametric document bootstraps give aggregate 95% intervals for
attention-minus-hybrid of `[0.1835, 0.2034]` CE, `[0.0634, 0.0707]` BPB, and
`[-2.271, -1.834]` top-1 percentage points. The general and code intervals also exclude zero.
These intervals condition on one architecture seed and the same 240 held-out documents; they are
not multi-seed architecture uncertainty.

Checkpoint SHA-256 values are
`b86929f708b0294ff305fa9ffbfa5059e04a807facfc0c5c55d64c471215f4a9` (attention) and
`00dd2cf6651b0a27e18d707d287b464361e4f0636c7c787fafc7570682ab2e6d` (hybrid). The
[immutable-style summary](results/webgpu-proxy-1tpp-10m-seed2026.summary.json) binds exact source
revisions, licenses, configs, checkpoints, scorecards, and hashes.

This observed exploratory, one-seed, approximately-one-token-per-parameter result does not satisfy
the planned selection rule. Its bounded 4,854-entry denylist covers only three local prompt
exports and removed no documents; BFCL, Mind2Web, WebLINX, and BrowserGym exports were absent. The
corpus has no structured-data source. Consequently, these results support only an early
language-model optimization comparison, not structured output, tool use, browser-task, or agent
quality.

### 6.4 Clean prospective confirmatory seeds 2027–2029

The clean confirmatory set repeated the matched 10.5M comparison at three prospectively
designated training seeds. The primary difference is attention minus hybrid aggregate BPB, so a
positive estimate favors hybrid:

| Training seed | Attention BPB | Hybrid BPB | Attention − hybrid BPB |
|---:|---:|---:|---:|
| 2027 | 2.103014998958501 | 2.032179123685033 | +0.07083587527346794 |
| 2028 | 2.104777477117417 | 2.031249477749292 | +0.0735279993681252 |
| 2029 | 2.0996846011662638 | 2.0251960502934447 | +0.07448855087281897 |
| Mean | 2.102492359080727 | 2.029541550575923 | +0.07295080850480402 |

Hybrid was favored in all three seeds. A model-based Student-t 95% interval with two degrees of
freedom is `[0.06824707441091234, 0.07765454259869571]`; it assumes approximately normal seed
effects, which cannot be assessed with three seeds. The exact sign test is therefore the more
assumption-free but low-power check: one-sided `p=0.125`, two-sided `p=0.25`. Mean
attention-minus-hybrid differences are `+0.2104463773820182` CE and
`-0.021929778813112435` top-1 accuracy, equivalent to a 2.193 percentage-point hybrid advantage.
General and code BPB each favor hybrid 3/3, with mean differences
`+0.06813601214548147` and `+0.10169832878279628`.

Every confirmatory scorecard explicitly ran on CPU in fp32. The training configs used
`device: auto` and `dtype: auto`, but the runner did not persist the resolved training device.
The current sandbox reports MPS unavailable and resolves `auto` to CPU; that present-host
observation is not retrospective proof of the confirmatory training device. The
[confirmatory summary](results/webgpu-proxy-1tpp-10m-seeds2027-2029.summary.json) and
[raw bundle](results/raw/pretrain-proxy-seeds2027-2029/) bind exact configs/checkpoints,
per-document scorecards, and paired comparisons.

These results support only a provisional hybrid choice for a bounded post-training pilot. They do
not satisfy the 34M-class at-least-five-TPP architecture screen and, by themselves, contain no
structured-action, browser-task, agent-quality, or WebGPU-latency measurement. That screen can
only promote a treatment to the subsequent 20-TPP/downstream comparison.

### 6.5 Exploratory seed-2026 trained pretrain-only WebGPU cache result

The exact checkpoints in Section 6.3 were exported as parity-gated fp16 prefill and fixed-`T=1`
cache-bearing decode graphs. On one Apple M5 with Chrome 150 and ONNX Runtime Web 1.27.0, three
page/session runs retained 720 measurements after 72 warmups:

| Input tokens | Hybrid p50 wall tokens/s | Hybrid p95 TPOT (ms) |
|---:|---:|---:|
| 128 | 137.711 | 10.305 |
| 512 | 128.232 | 9.610 |
| 1,024 | 123.779 | 10.214 |
| 1,536 | 116.476 | 10.205 |

The hybrid exceeded 100 p50 wall tokens/s at every context in every page run. The joint gate
requires both p50 at least 100 tokens/s and p95 TPOT at most 10 ms. Only 512 tokens passed by the
median-of-run statistic, and no context passed both thresholds in every run, so the joint gate
failed. The
[trained-latency summary](results/m5-webgpu-cached-decode-10m-trained-proxy-20260728.summary.json)
binds the exact checkpoints and graphs to the tracked payloads
([run 1](results/raw/m5-webgpu-cached-decode-10m-trained-proxy-20260728-run1.json),
[run 2](results/raw/m5-webgpu-cached-decode-10m-trained-proxy-20260728-run2.json),
[run 3](results/raw/m5-webgpu-cached-decode-10m-trained-proxy-20260728-run3.json)).

This is a latency-only browser artifact for the exploratory seed-2026 pretrain-only checkpoints,
not confirmatory-set latency. Held-out language quality was evaluated separately and joined by
checkpoint hash. It contains no midtraining, SFT, RL, action, browser-task, or agent-capability
measurement. The exact WebGPU-only session request succeeded without whole-session retry, but
adapter identity and per-node placement/fallback remain unknown; cross-device and cross-browser
generalization is unmeasured.

### 6.6 Bounded seed-2027 post-training stages

The provisional seed-2027 hybrid completed a lineage-checked midtrain/SFT/offline-RL chain on
MPS/fp32. Midtraining strongly improved the small agent holdout but slightly regressed the much
larger general holdout. SFT improved assistant-token metrics, but teacher-forced
all-assistant-token exactness reached only one of 65 rows; this is not free-running generation.
RL had enough reward variation to perform 12 optimizer updates, yet every held-out metric was
unchanged:

| Stage and held-out metric | Before | After |
|---|---:|---:|
| Midtrain agent loss / token accuracy | 7.7869 / 3.71% | 2.6371 / 69.80% |
| Midtrain general loss / token accuracy | 5.7064 / 18.68% | 5.7342 / 18.30% |
| SFT assistant loss / token accuracy | 2.7320 / 67.29% | 1.8146 / 73.13% |
| SFT teacher-forced all-assistant-token exact | 0/65 | 1/65 |
| RL exact / tool exact / format valid | 1/53 / 0/51 / 13/51 | 1/53 / 0/51 / 13/51 |

The [stage summary](results/webgpu-proxy-pilot-seed2027.summary.json) binds checkpoints and
metrics. The RL result is a measured zero delta, not a no-op: 6/128 groups were informative,
12 updates were realized, and 26/32 steps had zero signal.

The pilot midtraining config scheduled agent-row draw weight from 25% to 50%, but those weights
were not token shares. Short agent conversations contributed 10,290/645,170 input tokens (1.60%)
and 6,673/641,553 loss tokens (1.04%); 2,048-token general rows dominated token accounting. The
gain is therefore directional evidence from about 6.7K supervised agent targets, not evidence for
a 25–50% agent-token mixture. Forward paper configs now define the treatment in realized loss
tokens through the checkpointed deficit scheduler described above and report row draws, input
tokens, and loss tokens separately. This is an implementation result, not a new training result;
the historical pilot config and measurements remain unchanged.

### 6.7 Pre-assistant-padding fixed-512 complete-action pilot

The exact SFT checkpoint was exported as a parity-gated 21,430,301-byte fp16 action graph. The
backbone contains 10,524,544 learned parameters; the route, dense-selector, and pointer heads used
by structured dispatch add 2,499,333, for 13,023,877 active learned policy parameters. The
serialized fixed-action payload—action graph, dispatch heads, pointer-head bundle, tokenizer, and
metadata—is 27,091,138 bytes, excluding page code and ONNX Runtime. The pointer-head bundle also
contains a 19,635-parameter fixed tool head that this dispatch path does not use. Three
fresh WebGPU page/session runs on one Apple Metal 3 adapter in Chrome 150 / ONNX Runtime Web
1.27.0 used 20 held-out cases, three warmups, 30 repetitions per case, concurrency one, and
exactly 512 final tokenizer tokens under the internally prespecified pre-assistant-padding stress
condition.

| Policy and condition | Opportunities | Exact action | Schema valid | TTFA p50 / p95 (ms) | Success@1s |
|---|---:|---:|---:|---:|---:|
| Structured hybrid, fixed 512, run 1 | 600 | 5% | 100% | 24.75 / 34.405 | 5% |
| Structured hybrid, fixed 512, run 2 | 600 | 5% | 100% | 24.55 / 34.30 | 5% |
| Structured hybrid, fixed 512, run 3 | 600 | 5% | 100% | 25.20 / 34.80 | 5% |
| Autoregressive JSON, same checkpoint | — | — | — | — | — |
| Grounded candidate-trie AR, same checkpoint | — | — | — | — | — |

All 20 unique cases abstained, so capability was 1/20 overall, 0/19 tool-required, and 1/1
abstention. Repeated timing rows yield 90/1,800 exact and 0/1,710 tool-required, not 1,800
independent capability trials. The 100% schema rate therefore means valid abstention, not valid
tool output. All rows met even the 100 ms deadline, so every reported `Success@B` from 100 ms to
2 s equals the same 5% exact rate. The
[action summary](results/m5-webgpu-sft-action-pilot-seed2027.summary.json) binds the checkpoint,
graph, tokenizer, heads, cases, environment, and raw payloads.

The historical `rtab-0.2` action rows omitted normalized predicted arguments and full expected
actions; the summary therefore labels exact-action and schema outcomes as browser-reported and
non-recomputable at argument level. The all-abstention tool-name failure remains directly
observable. The v0.4 corrected runner closes this evidence gap before any new collection.

On the executable local-DOM harness, all 8 unique tasks failed. Across 720 repeated fixed-512
timing opportunities, exact action, independent executable-schema validation, final DOM state,
and closed-loop success remained zero.
Pooled closed-loop latency was 33.30 ms p50 and 66.80 ms p95, but no useful action was produced.
The [DOM summary](results/m5-webgpu-sft-dom-pilot-seed2027.summary.json) binds all three raw runs.

An exploratory offline parity diagnostic found natural-prompt route correctness of 17/20 and
selector top-1 correctness of 17/19 tool cases. The internally prespecified stress condition
inserts real,
unmasked space tokens before the assistant marker and reads the final hidden state, unlike the
natural SFT-probe feature materialization; every case at 128 tokens and above routes to text.
Native PyTorch, fp32/fp16 ONNX, and exported JSON heads agree, so export/precision mismatch is not
causal. This is a feature-materialization diagnosis, not generic long-context evidence or a
natural-prompt WebGPU result. The fixed-512 capability gate fails.

A preserved
[full-stack parity gate](results/sft-structured-export-parity-seed2027.summary.json) then applies
the exact browser grounding and normalization code to the corrected 512-token inputs. Native
PyTorch, fp32 ONNX, and fp16 ONNX agree on 20/20 routes, selected tools, grounded arguments, and
normalized actions, including 11/11 learned pointer spans. Their shared offline diagnostic score
is 16/20 exact and 20/20 schema-valid. Because this is the already-inspected action suite, it is
deployment parity—not a new capability estimate or WebGPU result.

The corrected fixed-compute browser runner appends filler after the natural assistant marker,
still executes 512 tokens, dispatches from `hidden[natural_input_tokens - 1]`, and bounds pointer
scans to the natural span. The
[offline audit](results/sft-structured-context-robustness-seed2027.summary.json) preserves the
natural route/selector counts on both frozen suites, and the
[browser protocol](results/webgpu-proxy-pilot-seed2027.corrected-browser.protocol.json) locally
freezes runner and artifact identities; an external pre-run timestamp is still required. The
browser runs are still required. Because the action, DOM, and
65-row suites informed the diagnosis, this is a reused-suite deployment-parity re-evaluation,
not confirmatory capability. Genuine capability under the
original pre-marker materialization would require training and evaluation on that exact condition
before comparison with the still-missing cache-bearing AR controls.

## 7. Limitations and conclusion

The model is text/DOM grounded rather than vision-first. Local microtasks cannot substitute for
open-web, multi-step evaluation. The proposed deadlines are reporting points, not universal
perceptual limits. A 34M model is expected to produce bounded grounded summaries, not frontier
world knowledge. The current complete-action ONNX autoregressive controls still recompute the full
prefix at every token. Separately, the cache-bearing exporter and sizing benchmark measure
random-weight and exact pretrain-only checkpoints with fixed-one-token decode, but they are not
integrated with an action-trained complete-action baseline. The candidate trie contains one
deterministically grounded assignment per tool plus an EOS abstention terminal, not the complete
JSON-Schema language. Hardware coverage and full-budget training seeds may be limited by compute.
The quality evidence reported here consists of an observed exploratory seed-2026 pilot and a clean
prospective seed-2027–2029 set, all at roughly 10M parameters and one loss token per
parameter—not the 34M-class five-TPP screen or subsequent 20-TPP/downstream comparison. The
confirmatory scorecards are CPU fp32;
their training configs used `device: auto`, but the runner did not persist the resolved device.
The seed-2026 10,000-resample paired-document intervals condition on that seed and 240 documents,
while the three-seed Student-t interval assumes normal seed effects that cannot be assessed at
`n=3`; the exact sign test is not conventionally significant. Trained WebGPU evidence now
includes the seed-2026 pretrain-only cache result and the seed-2027 SFT structured action/DOM
pilot on one M5/Chrome/ORT configuration. The latter measures fast forwards but fails capability:
all predictions abstain, every tool-required action fails, and DOM closed-loop success is zero.
Its 100% action-suite schema validity reflects valid abstention. The pre-assistant-padding
feature materialization is itself a measured failure condition; offline natural-prompt
route/selector diagnostics cannot be reported as WebGPU quality. The in-page runner records its
first warmup as the first-inference phase, but its `ttfa_ms` clock begins at prompt tokenization
rather than observation availability. Runner v0.4 now fail-stops the entire collection after a
10,000 ms watchdog because the in-flight ORT call cannot be cancelled; it never continues timing
after a timeout. Fully instrumented user TTFA, an external protocol timestamp, a new untouched
capability set, natural-prompt browser evaluation, matched cache-bearing AR controls, anonymous
artifact publication, and per-node provider verification are still required before paper-grade
complete-action claims.

Our central claim is intentionally conditional: if exact complete actions are the unit an agent
executes, model design and evaluation should optimize successful actions before a deadline. This
pilot demonstrates why: a 25 ms forward with zero tool dispatch is not a useful realtime agent.
The remaining measurements must determine whether a repaired structured policy or either
autoregressive control belongs on the frontier.

## Artifact checklist before unblinding

- [ ] Every bracketed result replaced from a preserved JSON artifact.
- [ ] Git revision, model/tokenizer/graph/head hashes, data revisions, and device metadata recorded.
- [ ] Frozen task files excluded from pretraining, tokenizer training, SFT, and RL.
- [x] Three clean prospective 10M proxy seeds, per-seed estimates, small-`n` interval assumption,
      exact sign test, scorecard device, and raw artifacts preserved; this is not the 34M
      five-TPP screen or promoted 20-TPP/downstream selection.
- [x] One-seed/one-TPP proxy corpus, tokenizer, checkpoints, scorecards, hashes, and limitations
      preserved as the observed exploratory seed-2026 pilot.
- [x] Exact seed-2026 pretrain-only checkpoints measured with cache-bearing WebGPU graphs; the
      partial throughput/tail gate, separation from confirmatory latency, and absence of agent
      capability are explicit.
- [x] Exact WebGPU session request, ORT version/device evidence, and no whole-session retry
      recorded; per-node placement/fallback is explicitly unknown because ORT does not expose it.
- [x] Standalone random-weight cache-bearing prefill/decode graphs are parity-gated and measured.
- [x] Seed-2027 midtrain/SFT/offline-RL pilot preserved with resolved runtime, held-out pre/post
      metrics, and the zero RL delta reported.
- [x] Fixed-512 SFT structured action and executable local-DOM runs preserved; the abstention
      collapse, 1/20 unique-case action exactness, 0/8 unique-case DOM success, and failed
      capability gate are explicit.
- [x] Corrected fixed-compute dispatch-position runner implemented with independently rescorable
      v0.4 rows and a fail-stop watchdog; the protocol is locally frozen.
- [ ] Externally timestamp the final v0.4 protocol and full-stack 512-token parity artifact before
      any corrected browser collection.
- [ ] Run three corrected action and DOM browser sessions; train/evaluate the original pre-marker
      materialization separately if claiming genuine fixed-512 capability.
- [ ] Trained complete-action autoregressive baselines use the same backbone checkpoint and tool
      schema, integrate a cache-bearing decoder, and report their cache/recompute strategy.
- [ ] DOM claims require final-state success; BrowserGym claims require episode success.
- [ ] Q4 omitted unless measured.
- [x] Official NeurIPS 2026 double-blind-workshop LaTeX, bibliography, conservative checklist,
      four-page body/reference render, and visual/PDF-metadata QA completed.
- [ ] Anonymous repository/artifact contains no author metadata.
