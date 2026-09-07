# Research: prior art and the good parts we steal

> Goal recap: a **< 100M-parameter** LLM, **pretrained from scratch**, that works as an
> **agent** (tool calling + free-form text generation), built as a **minimal, hackable,
> pure-PyTorch ML pipeline** (nanochat-style) with **training + evaluation**, running on
> **CPU / GPU / NPU**, with **PyTorch / ONNX implemented** and **GGUF / ExecuTorch planned**, and
> improved over time by a **data flywheel** that learns from stored conversations.

This document surveys the projects worth learning from and records, for each, **only the parts
we adopt** and why. The design in [`ARCHITECTURE.md`](./ARCHITECTURE.md) is the synthesis of
these decisions.

---

## 1. Full-stack minimal trainers

### nanochat — Karpathy ([audited repository](https://github.com/karpathy/nanochat/tree/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd))
A single, dependency-light codebase covering the model lifecycle. The
[speedrun at audited commit `92d63d4`](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/runs/speedrun.sh)
executes tokenizer → pretrain/base evaluation → SFT/chat evaluation; it does not run RL or a
separate midtrain stage. LocalAgent retains explicit midtraining because its code/tool
distribution shift is large and must be measurable separately.
The 2026 speedrun now downloads the
[`karpathy/climbmix-400b-shuffle`](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle)
repack. That is useful evidence about nanochat's current recipe, but it is not a license shortcut:
the repack card says the content is unchanged from NVIDIA Nemotron-ClimbMix, whose upstream card
is CC BY-NC 4.0/research-only. LocalAgent therefore transfers the pipeline design but keeps
ClimbMix out of its default distributable corpus unless the rights question is resolved.

The model at that pinned revision is also more than a plain historical GPT-2 block: it combines RoPE,
QK-Norm, untied embedding/unembedding weights, ReLU-squared MLPs, an `SSSL` tiled
sliding/full-attention schedule, value embeddings, learned residual/input-embedding mixing,
Muon+AdamW, and optional FP8 training. Those are useful ablation candidates, not a package to
copy wholesale. In particular, the reference `base_train.py` currently instantiates
`n_kv_head == n_head` even though its module supports GQA, and its H100 Flash-Attention/FP8
choices do not establish a WebGPU optimum. LocalAgent keeps MQA/GQA, SwiGLU, and the matched
hybrid-versus-attention experiment as explicit browser-scale decisions.

**We adopt:**
- The **single cohesive codebase** philosophy — no giant config monsters, no framework. Every
  stage is a readable script you can fork.
- The **stage ordering**: tokenizer → pretrain → SFT → eval → serve, with LocalAgent's explicit
  domain midtrain and optional RL as separately measured extensions.
- A **`speedrun.sh`** that runs the whole pipeline start-to-finish at toy scale.
- A **built-in chat UI** as the demo surface.
- Its standalone
  [`chat_rl.py`](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/chat_rl.py) as a useful
  minimal outcome-reward reference, while preserving its own terminology: it puts “GRPO” in
  quotes and describes the update as closer to REINFORCE, without reference KL or PPO
  ratio/clipping and with DAPO-style token-level normalization. It is not conventional GRPO.

**We drop:** the speedrun's 8×H100 reference budget and any assumption that its current corpus can
be redistributed under the repo's MIT license. Our default target is a single small GPU *or* CPU,
at <100M params.

### Kimi K2 / K2.5 / K3, Kimi Linear, and Attention Residuals

Kimi K3's [official repository at audited commit `7c5be95`](https://github.com/MoonshotAI/Kimi-K3/tree/7c5be9599120d7993748de66a76128614f15f210) publishes a concrete
93-layer configuration: 69 KDA layers and 24 Gated-MLA layers, 2.8T total/104B active parameters,
16 of 896 routed experts, a 160K vocabulary, and a 1,048,576-token context. The
[official launch post](https://www.kimi.com/blog/kimi-k3) additionally describes Attention
Residuals and SFT-onward MXFP4-weight/MXFP8-activation quantization-aware training. The
[technical report](https://github.com/MoonshotAI/Kimi-K3/blob/7c5be9599120d7993748de66a76128614f15f210/k3_tech_report.pdf) also
discloses the high-level lifecycle: curated web/code/math/knowledge plus vision pretraining,
small-model data-mixture and scaling-law studies, Per-Head Muon with weight clipping, cosine
decay with 1% warmup, progressive context extension, SFT, domain/effort-specific RL, and
Multi-Teacher On-Policy Distillation. It does not publish exact corpus identities and weights,
training-token count, or all selected learning-rate/batch/TPP values, so those quantities remain
unknown rather than inferred.

Kimi K2.5's [official report](https://arxiv.org/abs/2602.02276) describes continual pretraining
atop K2-Base on approximately 15T mixed visual/text tokens, native multimodal post-training, and
learned parallel Agent Swarm orchestration. LocalAgent transfers explicit continual-stage
accounting and verified parallel-task construction only. It does not implement vision, MoE/MLA,
Agent Swarm, or the reported parallel-agent training as the deployment path: the bounded
Micro-MoE below is an independent active-matched experiment, not a K2.5 reproduction. It also
does not treat vendor swarm latency as an on-device browser measurement. The repository source was audited at
[`3e60763`](https://github.com/MoonshotAI/Kimi-K2.5/tree/3e60763b943e93c443287c383e0468ffe05b188f).

**We adopt:** the architectural *principles* of hybrid sequence mixers, periodic exact-retrieval
attention, stable Q/K dynamics, explicitly measured information flow across depth, small-model
mixture ablations, verified trajectory synthesis, and executable white-box agent environments.

**We drop from the deployment default:** literal KDA/MLA, Attention Residuals, and million-token
infrastructure until browser kernels make them beneficial. We now test one bounded Micro-MoE
transfer in PyTorch: actual top-k FFN dispatch, a total-budgeted 43.86M model, and a 17.30M
active-matched dense control. It remains unpromoted until specialization, quality, resident
memory, sparse export, and browser latency all pass. The current short-conv mixer is not labeled
or treated as KDA, and its theoretical Q4 byte count is not K3-style quantization-aware training.
Per-Head Muon, cosine/WSD tuning, MTP drafting, and QAT remain separate measured ablations rather
than changes justified by K3's frontier-scale result alone.

See [`TRAINING_SYSTEM.md`](./TRAINING_SYSTEM.md) for the 2026 source comparison and staged recipe.

### Upstage SOLAR-10.7B

[SOLAR-10.7B](https://arxiv.org/abs/2312.15166) applies depth up-scaling to a compatible
pretrained transformer: duplicate/expand its layers, then continue pretraining the expanded
checkpoint. The reusable idea is checkpoint growth without training the deeper topology from its
initial random state. It is not directly available to LocalAgent's from-scratch sub-100M recipe:
there is no larger compatible parent with the same tokenizer and blocks, so there is no inherited
knowledge to preserve. Layer duplication after training a smaller LocalAgent checkpoint remains
an optional continued-training ablation with matched added compute; it is neither a substitute for
pretraining data nor evidence of lower WebGPU latency. The repository now implements that narrow
ablation through `localagent grow-checkpoint`: it requires an explicit complete same-kind layer
map and an otherwise identical, same-tokenizer model schema; emits a self-hashed, target-state-
hashed checkpoint; rejects posttraining auxiliary heads; and enters continued pretraining as a
fresh-optimizer `init_from`. Repeated residual blocks change the function, so the implementation
does not claim Net2Net-style function preservation or a quality gain before a matched run.

### GLM-5.2 and Grok 4.5 — coding-agent systems, not tiny-model blueprints

Z.ai's [official GLM-5.2 report](https://z.ai/blog/glm-5.2) discloses IndexShare across groups of
four sparse-attention layers, a verifier-backed MTP acceptance study, extended coding-agent
midtraining, compaction-aware critic PPO for long trajectories, parallel on-policy distillation,
and an online anti-hack path for executable coding rewards. The same report states that lower
per-token FLOPs do not proportionally reduce its KV-cache requirement. At LocalAgent's 4K context
and 1–96M scale, the transferable parts are reward-integrity checks, compacted-trajectory
accounting, and the rule that speculative speed needs measured accepted drafts plus end-to-end
latency. Its 1M-context sparse index and serving stack are not browser defaults.

xAI's [official Grok 4.5 release](https://x.ai/news/grok-4-5) discloses strong deduplication,
quality/domain filtering, hundreds of thousands of technical RL tasks, automated/model grading,
and asynchronous rollouts that can last hours. It does not disclose a block-level architecture.
LocalAgent can test the data-curation and executable-rollout principles, but it cannot truthfully
copy a hidden Grok topology. The reported hosted 80 tokens/s is also not a WebGPU requirement: it
does not isolate tokenizer, hardware, prompt, output length, TTFT, or action correctness.

### GPT-2 / GPT-3 / GPT-4 — disclosed foundations and explicit boundaries

[GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
and [GPT-3](https://arxiv.org/abs/2005.14165) establish the decoder-only causal next-token
baseline, scaling studies, and task specification through prompts or in-context examples.
[GPT-4](https://arxiv.org/abs/2303.08774) adds evidence for predictable scaling and staged
post-training, but does not disclose a transferable block-level architecture or exact data
recipe. LocalAgent therefore adopts dense causal controls, frozen held-out evaluation, and
scaling curves without inventing GPT-4/4.5/5.x topology or treating dense all-attention as a
browser optimum.

### nanoGPT / modded-nanoGPT — Karpathy & community
Minimal GPT model + training loop; modded-nanoGPT is the speedrun playground (Muon optimizer,
architectural tweaks).

**We adopt:** the compact `model.py` + `train.py` skeleton, and Muon/AdamW as a tunable
optimizer choice for pretraining.

---

## 2. Tiny-model architecture & data

### SmolLM2 (135M/360M/1.7B) — HuggingFace ([paper](https://arxiv.org/abs/2502.02737))
Data-centric training of genuinely small, on-device models.

**We adopt:**
- **Grouped-Query Attention (GQA)** for cheap inference at small scale.
- **Depth over width** (more layers, narrower `d_model`) as a quality hypothesis for the <100M
  budget, not a browser-latency law. The measured random-weight WebGPU sweep found that a
  four-layer, width-384 10.5M hybrid outperformed a deeper/thinner 15.6M shape, so deployment
  geometry remains empirical.
- **Data-centric curation** over raw web scrapes — quality-filtered, structured corpora.
- The 135M point as a **reference size/recipe sanity check** for our from-scratch model.

Here “dense” means a no-MoE deployment baseline, not that dense all-attention is universally best.
[MobileLLM](https://arxiv.org/abs/2402.14905) supports a strong dense sub-billion baseline, while
[Kimi Linear](https://arxiv.org/abs/2510.26692) is direct primary-source evidence that a hybrid
can beat full attention under its own matched recipe. LocalAgent therefore compares dense
all-attention and dense hybrid backbones on the target browser rather than deciding by label.

The observed exploratory seed-2026 proxy preserves that distinction. At approximately one loss
token per parameter, the 10.525M hybrid reached aggregate BPB 2.0319 versus 2.0989 for the
10.547M attention control. Ten thousand paired document bootstraps put the
attention-minus-hybrid BPB difference at 0.0669 with a 95% interval of `[0.0634, 0.0707]`;
that interval conditions on the same architecture seed and 240 held-out documents, not multi-seed
uncertainty.

A clean, prospectively designated confirmatory set then repeated the comparison at seeds
2027–2029. The attention-minus-hybrid BPB estimates were `+0.07083587527346794`,
`+0.0735279993681252`, and `+0.07448855087281897`, with mean
`+0.07295080850480402`; hybrid was favored 3/3. The model-based Student-t interval with two
degrees of freedom is `[0.06824707441091234, 0.07765454259869571]`. That interval assumes
approximately normal seed effects, which cannot be assessed at `n=3`; the exact sign test gives
one-sided `p=0.125` and two-sided `p=0.25`. Mean attention-minus-hybrid gaps were
`+0.2104463773820182` CE and `-0.021929778813112435` top-1 accuracy, equivalent to a 2.193-point
hybrid accuracy advantage. General and code BPB favored hybrid in every seed as well.
Confirmatory scorecards were CPU fp32. The training configs used `device: auto`, but the runner
did not persist the resolved training device; this host currently reports MPS unavailable and
resolves `auto` to CPU, which is not retrospective proof. See the
[confirmatory summary](./paper/results/webgpu-proxy-1tpp-10m-seeds2027-2029.summary.json) and
[raw scorecards](./paper/results/raw/pretrain-proxy-seeds2027-2029/).

Separately, the exact seed-2026 pretrain-only hybrid checkpoint produced median-of-run p50 wall
rates of 137.711 / 128.232 / 123.779 / 116.476 tokens/s at
128 / 512 / 1,024 / 1,536 tokens on one M5/Chrome 150/ONNX Runtime Web 1.27.0 setup, clearing
100 in every page run and context. Its p95 TPOT values were
10.305 / 9.610 / 10.214 / 10.205 ms, so the combined throughput/tail gate failed at three
contexts by the median statistic and at every context under the every-run requirement. This is
seed-2026 pretraining quality plus latency—not confirmatory-set latency or agent capability. The
consistent three-seed quality result provisionally selects hybrid for a bounded post-training
pilot only; the 34M five-TPP screen and subsequent 20-TPP/downstream quality selection remain
unrun. See the
[seed-2026 quality summary](./paper/results/webgpu-proxy-1tpp-10m-seed2026.summary.json) and
[seed-2026 trained-latency summary](./paper/results/m5-webgpu-cached-decode-10m-trained-proxy-20260728.summary.json).

### Bounded seed-2027 post-training and WebGPU action pilot

The provisional seed-2027 hybrid was continued through a deliberately bounded, lineage-checked
post-training chain. Midtraining strongly improved the small agent holdout (loss 7.7869→2.6371,
token accuracy 3.71%→69.80%) but slightly regressed the general holdout
(loss 5.7064→5.7342, accuracy 18.68%→18.30%). SFT improved held-out assistant loss
2.7320→1.8146 and token accuracy 67.29%→73.13%, while teacher-forced
all-assistant-token exactness reached only 1/65. This is not a free-running generation metric.
Offline canonical-toolcall GRPO found six informative groups and realized 12 optimizer
updates, but its 53-row greedy holdout was identical before and after: 1/53 exact, 0/51 tool
exact, 13/51 tool-format valid, mean reward 0.0434. This makes RL a measured zero-delta control,
not a capability claim. The
[stage summary](./paper/results/webgpu-proxy-pilot-seed2027.summary.json) binds all configs,
data, checkpoints, metrics, and MPS/fp32 execution metadata.

The exact SFT checkpoint was also exported as a parity-gated fp16 action graph. Three
pre-assistant-padding fixed-512 WebGPU stress runs on one Apple M5/Chrome 150/ONNX Runtime Web
1.27.0 produced within-run TTFA
p50 values of 24.75/24.55/25.20 ms and p95 values of 34.405/34.30/34.80 ms. However, the policy
predicted abstention on every case. Capability was 1/20 unique cases overall—0/19 tool-required
and 1/1 abstention. The 30 timing repetitions across three sessions yield 90/1,800 exact and
0/1,710 tool-required, not 1,800 independent capability trials. The action
suite's 100% schema-valid rate means only that abstention is valid there. On the executable local
DOM suite, all 720 rows failed exact action, independent executable-schema validation, final DOM,
and closed-loop success; pooled closed-loop p50 was 33.3 ms. The
[DOM summary](./paper/results/m5-webgpu-sft-dom-pilot-seed2027.summary.json) reproduces all three
raw runs.

An exploratory parity diagnostic found natural-prompt route correctness of 17/20 and dense-
selector top-1 correctness of 17/19 tool cases. The internally prespecified stress condition
instead
materializes real, unmasked space tokens before the assistant marker and reads the final hidden
state; every case at 128 tokens and above routes to text. Native PyTorch, fp32/fp16 ONNX, and the
exported JSON heads agree, excluding export/precision mismatch as the cause. This is a deployment
feature-materialization shift, not generic long-context evidence or a natural-context WebGPU
quality score. The fixed-512 run fails the capability gate.

The corrected fixed-compute runner appends filler after the natural assistant marker, still
executes 512 tokens, dispatches from `hidden[natural_input_tokens - 1]`, and bounds pointer scans
to the natural span. The
[full-stack export-parity gate](./paper/results/sft-structured-export-parity-seed2027.summary.json)
gets exact native/fp32-ONNX/fp16-ONNX agreement on all 20 reused-suite routes, tools, grounded
arguments, and normalized actions; its shared 16/20 offline exact score is deployment diagnosis,
not a new capability estimate. The
[offline audit](./paper/results/sft-structured-context-robustness-seed2027.summary.json) preserves
the natural route/selector counts on both frozen suites, and the
[browser protocol](./paper/results/webgpu-proxy-pilot-seed2027.corrected-browser.protocol.json)
is locally frozen but still requires an external pre-run timestamp; browser runs remain pending.
Because all three evaluation sets informed the diagnosis,
the corrected run is a reused-suite deployment-parity re-evaluation, not confirmatory capability.
A genuine claim about the original pre-marker
fixed-512 materialization
would separately require training and evaluation on that exact condition. Cache-bearing
complete-action autoregressive controls, the 34M at-least-five-TPP screen and promoted
20-TPP/downstream training,
BrowserGym/open-web, and cross-device replication remain absent. See the
[action summary](./paper/results/m5-webgpu-sft-action-pilot-seed2027.summary.json) and
[artifact index](./paper/results/README.md).

---

## 3. Agent / tool-calling data generation

### APIGen / xLAM — Salesforce ([xLAM-function-calling-60k])
A verifiable, multi-stage synthetic data pipeline for function-calling, with format checking,
**actual execution checking**, and semantic verification; ships a 60k function-calling dataset
with a clean JSON schema.

**We adopt:** the **multi-stage verification** idea (format → execution → semantic), the
**function-calling JSON schema** as our on-disk format for agent samples, and a pinned offline
TRAIN adapter for the public 60K release. The adapter's rule verification does not upgrade source
records into a claim that LocalAgent executed them.

### Mind2Web, WebLINX, and public split policy

Mind2Web contributes crowdsourced, grounded multi-step web tasks. LocalAgent ingests only the
pinned public Mind2Web TRAIN shards and converts positive CLICK/TYPE/SELECT targets into canonical
browser trajectories; held-out benchmark material remains outside training. WebLINX is retained
for evaluation/non-default research because its CC-BY-NC-SA-4.0 terms do not fit the default
training mixture. BFCL remains external evaluation only. Every accepted file is bound by upstream
revision, license evidence, byte count, SHA-256, and exact-prompt decontamination.

See [`SPARSE_EXPERTS_REAL_DATA.md`](./SPARSE_EXPERTS_REAL_DATA.md) for exact revisions and the
frozen real-use evaluation contract.

### ToolACE ([paper](https://arxiv.org/abs/2409.00920))
An agentic **self-evolution** synthesis loop over a large API pool, a **complexity evaluator**
that targets samples at the model's level, and a **dual-layer (rule + model) verifier**.

**We adopt:** the **multi-agent synthesis loop**, the **complexity-targeting** of generated
samples, and the **rule-based + model-based dual verification** of every sample.

### Hammer ([repo](https://github.com/MadeAgents/Hammer))
On-device function-calling robustness via **function masking** and an
**irrelevance-augmented** dataset (examples where *no* tool should be called).

**We adopt:** **irrelevance / "no tool needed" negatives** as a first-class part of the dataset,
and **function masking** during training so the model generalizes across tool name/arg surface
forms instead of memorizing them.

---

## 4. Distillation (the "distillation part")

### MiniLLM ([paper](https://arxiv.org/abs/2306.08543)) & On-Policy Distillation (Thinking Machines, MiniPLM)
Reverse-KL instead of forward-KL for generative KD; **on-policy** distillation where the
*student* generates trajectories and the *teacher* scores them (fixes exposure bias).

**We adopt:** **reverse-KL** as the KD objective and an **on-policy loop** (student samples →
teacher logits/scores → update) as the advanced distillation mode. A simpler **off-policy
sequence-KD** mode (teacher generates, student imitates) is the default starter.

### Distilling LLM Agents into Small Models ([2505.17612])
Distill **full agentic trajectories** (with tool invocations) rather than just CoT; tricks like
**first-thought prefix** and **self-consistent action selection** let 0.5–3B students match the
next tier up.

**We adopt:** **trajectory-level distillation** (the teacher's whole tool-use rollout, including
reasoning + tool calls + tool responses, is the training signal) and **first-thought prefix**
prompting when generating teacher data.

Distillation-first is a compute allocation, not a claim that RL cannot work. The
[DeepSeek-R1 release](https://github.com/deepseek-ai/DeepSeek-R1) reports that its distilled small
models beat the reasoning patterns found by its compared small-model RL runs, while the same work
also demonstrates that RL can elicit reasoning in a sufficiently capable base model. For this
project, verified distillation supplies dense capability transfer first; outcome-reward RL remains
a measured last-mile stage for behavior that has executable rewards.

---

## 5. Memory & the data flywheel

### MemGPT / Letta ([docs](https://docs.letta.com/concepts/letta/))
LLM-as-OS: a **memory hierarchy** (in-context "core" memory vs out-of-context archival), the
model **self-edits** memory via tools, and a pager moves data in/out of the context window.

**We adopt:** a **two-tier memory** (core/in-context + archival/out-of-context), exposed to the
model **as tools** (`memory_append`, `memory_search`, …) so memory management is itself agent
behavior, plus a simple **paging/consolidation** policy.

### Agent-in-the-Loop data flywheel — Airbnb ([2510.06674])
A production flywheel that captures four annotation types — **pairwise preferences**, **adoption
decisions + rationale**, **knowledge-relevance checks**, **missing-knowledge flags** — and feeds
them back into model updates, cutting retrain cycles from months to weeks.

**We adopt:** the **flywheel schema** (every stored conversation can carry these feedback
signals) and the **loop**: log conversations → mine/verify into new training samples → schedule
a retrain/distill → evaluate → redeploy.

---

## 6. Evaluation

### BFCL — Berkeley Function Calling Leaderboard ([proceedings](https://proceedings.mlr.press/v267/patil25a.html))
The standard tool-use benchmark: **AST-based** evaluation of single/parallel/multiple calls,
multi-turn (v3+), and **irrelevance detection** (should the model abstain from calling?).

**We adopt:** an **AST-based tool-call evaluator** (compare parsed call trees, not strings),
plus **multi-turn** and **irrelevance** splits in our eval harness.

### tau-bench / ToolBench
Multi-turn tool use in realistic customer-service / real-API settings.

**We adopt:** a small **multi-turn task harness** with a simulated user + tool sandbox for
end-to-end agent eval beyond static call matching.

### Tool/function-calling benchmark landscape — survey ([BFCL_v4 survey](https://huggingface.co/datasets/tuandunghcmut/BFCL_v4_information))
A survey that maps the tool-calling eval space along three axes — **task complexity** (single →
parallel/sequential → agentic planning), **interaction model** (stateless single-turn → stateful
multi-turn), and **evaluation method** (AST match → execution → outcome → LLM-judge) — and names the
core tension: *more realistic benchmarks force less reliable scoring* (open-ended state pushes you
toward LLM-as-judge with its position/verbosity biases). It documents a **mechanics-vs-strategy**
split — e.g. GPT-4o ~84% on stateless BFCL but ~7% on a stateful variant — i.e. grounded call
*mechanics* and long-horizon *strategy* are distinct competencies. Beyond BFCL/ToolBench it catalogs
**API-Bank** (decomposes tool use into planning / retrieving / calling), **ToolSandbox** (stateful
multi-turn scored by a Milestone DAG), AgentBench, and MCP-AgentBench.

**We adopt:** the **AST/execution end** of that ladder (score grounded call trees against a
deterministic oracle, never an LLM judge); **API-Bank's planning/retrieving/calling decomposition**,
which maps cleanly onto our own split of *tool selection* (the head) vs *argument grounding*
(pointer/heuristics) vs *abstention*; and the **mechanics-vs-strategy** finding as direct support for
scoping the 28M model to mechanics and leaving strategy to the orchestrator (see the ALE note below).

### tool-calling-benchmark — Veerman ([repo](https://github.com/MikeVeerman/tool-calling-benchmark))
Exactly our regime: **small open-weight models (0.5–3.8B), CPU-only**, scored on *judgment*. 12
single-turn prompts mix actionable calls, **restraint** cases (where calling *any* tool is wrong,
e.g. "what tools do you have?"), and **misleading** prompts that bait keyword-triggered calls.
Scoring is a weighted **Agent Score = Action·0.4 + Restraint·0.3 + Wrong-Tool-Avoidance·0.3**,
aggregated by **majority vote over 20 runs** (3 runs proved unreliable). Top small model: qwen3:1.7b
at 0.960.

**We adopt:** **restraint/abstention as a first-class *scored* axis** (our irrelevance negatives
already train it — this says to weight and report it alongside action, not bury it); an explicit
**wrong-tool penalty** for same-shaped distractor tools (our known 22-way selector confusion); and
**multi-run majority-vote** for stable tiny-model eval on CPU, where single-run accuracy is noisy.

### Agents' Last Exam (ALE) ([abs](https://arxiv.org/abs/2606.05405) · [site](https://agents-last-exam.org)) — *contrast, not a target*
The frontier end of the spectrum: **1,000+ long-horizon, economically-valuable real-world tasks**
with **verifiable outcomes**, spanning 55 subfields across 13 industry clusters (U.S. occupational
taxonomy), built with 250+ industry experts. It is deliberately unsaturated — mainstream frontier
configs average just **~2.6% full pass** on the hardest tier. This is autonomous *professional
workflow* evaluation, the opposite pole from a <100M single-/parallel-call tool router.

**We don't target it** — a 28M on-device model is a **controller**, not an autonomous long-horizon
agent; sustained multi-step workflows are the *orchestrator's* job, with the tiny model supplying
grounded tool calls underneath. What we **do** take is the **verifiable-outcome** discipline (score
against an executable/AST oracle, never an LLM judge) and ALE as an explicit **scope boundary**: it
keeps us honest that our held-out tool-call accuracy measures the router, not end-to-end autonomy.

---

## 7. On-device inference / export ("runs on CPU/GPU/NPU")

| Runtime | Strength | Status / intended use |
|---|---|---|
| **PyTorch** (eager) | dev loop, ground truth | implemented reference + correctness oracle |
| **GGUF / llama.cpp** | CPU + Apple Silicon + Intel/AMD GPU/NPU (via OpenVINO upstream) | planned "runs everywhere" path; no exporter or parity claim yet |
| **ONNX Runtime** | one graph, Execution Providers for CPU/GPU/NPU (incl. AMD Ryzen AI) | implemented export/parity path, including cached WebGPU graphs |
| **ExecuTorch** | compact native PyTorch AOT runtime | planned mobile/NPU path; no exporter or parity claim yet |

**Implemented now:** PyTorch is the reference and ONNX exports have fixed-prompt and iterative
cached-decode parity checks. GGUF/Q4 and ExecuTorch are roadmap targets; they are not described as
implemented converters, measured runtimes, or completed parity paths.

---

## Summary table — what each project contributes

| Project | One thing we take |
|---|---|
| nanochat | single-codebase tokenizer → pretrain → SFT speedrun + chat UI; separate REINFORCE-like RL reference |
| Kimi K2/K2.5/K3 / Kimi Linear | disclosed agent-data, continual-stage, parallel-task, and hybrid-mixer lessons, transferred only through bounded browser experiments |
| Upstage SOLAR-10.7B | optional layer-duplication checkpoint-growth ablation; no inherited knowledge without a compatible pretrained parent |
| nanoGPT | compact model/train skeleton + optimizer choice |
| SmolLM2 | GQA + depth-over-width + data-centric curation @ <100M |
| APIGen/xLAM | multi-stage verification + function-calling JSON schema + pinned TRAIN adapter |
| Mind2Web | grounded public TRAIN web-action trajectories with held-out split protection |
| ToolACE | multi-agent synthesis + complexity targeting + dual verify |
| Hammer | irrelevance negatives + function masking |
| MiniLLM / OPD | reverse-KL + on-policy distillation |
| Agent-distill (2505.17612) | trajectory-level distillation + first-thought prefix |
| MemGPT/Letta | two-tier self-editing memory exposed as tools |
| Airbnb AITL | data-flywheel feedback schema + retrain loop |
| BFCL / tau-bench | AST tool eval + multi-turn + irrelevance |
| Tool-calling survey (BFCL_v4) | AST/execution-end scoring + planning/retrieving/calling split + mechanics-vs-strategy |
| tool-calling-benchmark (Veerman) | restraint as a scored axis + wrong-tool penalty + multi-run majority vote |
| Agents' Last Exam | verifiable-outcome discipline + a scope boundary (contrast, not a target) |
| PyTorch/ONNX; GGUF/ExecuTorch | PyTorch/ONNX implemented with parity; GGUF/Q4 and ExecuTorch planned |

---

## See also — computer-use & tool-calling agents

For the recent **computer-use / multi-agent-browser / tool-calling** wave (UI-TARS, OSWorld-G/Jedi,
WebVoyager, SeeAct, Mind2Web, WebArena, Gorilla, ToolLLM, xLAM/APIGen, Hammer, ToolACE, ToolAlpaca,
CodeAct, OctoTools, AutoGen, CoAct-1) — with each paper's good part, a two-persona **scale vs
structure** debate, and the verdict for a sub-100M on-device agent — see
[`COMPUTER_USE_DEBATE.md`](./COMPUTER_USE_DEBATE.md). That survey is what motivated
`ToolCaller.plan()` (planner over grounded calls).

For the **model architecture** itself — what the 2024–2026 frontier and small-model waves
(Qwen3, DeepSeek-V3/R1, Nemotron-H, Llama-4, LFM2, MobileLLM, MiniCPM, Phi, Gemma-3n, …) teach a
sub-100M on-device tool agent, staged as a multi-persona debate with adopt/skip verdicts for our
1M/30M tiers — see [`ARCHITECTURE_DEBATE.md`](./ARCHITECTURE_DEBATE.md). Headline: distillation
(30M→1M), a gated short-conv + GQA hybrid port, and an opt-in active-matched Micro-MoE experiment;
MLA/FP8/SSM/sparse-long-context remain excluded from the browser default.
