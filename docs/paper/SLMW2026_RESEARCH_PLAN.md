# SLM-Agents 2026 research and submission plan

Status: working research plan, 2026-07-29; official CFP last updated 2026-07-23  
Target: [1st Workshop on SLMs for Agentic Systems at NeurIPS 2026](https://slmw2026.github.io/#call-for-papers)  
Deadline: 2026-08-22 AoE (2026-08-23 20:59 KST, assuming the usual 23:59 AoE cutoff)

## Decision

Do not write a paper claiming that a browser agent needs 200, 400, or 600 tokens/s. No accepted
standard supports those thresholds, and tokenizer-dependent tokens/s is not comparable across a
byte model and a BPE model.

The submission should test this narrower, falsifiable thesis:

> Time to first complete valid action (TTFA), conditioned on action correctness, is the useful
> service objective for a local browser agent. A compact model with a one-forward structured
> action head can improve `Success@1s` over autoregressive JSON generation; the WebGPU backbone
> should be selected from a measured hybrid-versus-all-attention Pareto comparison. Backbone,
> action-head, total deployed-parameter, and bundle-byte counts must be reported separately.

The strongest currently supported insight sharpens that thesis:

> The hidden-state extraction point is part of the deployed model. A fixed-length WebGPU graph can
> be numerically parity-correct and still lose all tool capability when padding is inserted before
> the assistant decision marker; preserving the marker state restores the offline policy. The
> feature contract, not token rate alone, must therefore be frozen and evaluated.

Current evidence does not establish that hypothesis. The internally prespecified
pre-assistant-padding
fixed-512 structured arm is fast but collapses to abstention and fails the capability gate; the
matched autoregressive controls remain unmeasured. The corrected fixed-compute dispatch-position
arm described below was implemented and locally frozen, but was superseded before any external
pre-run timestamp or browser collection. The current cached-autoregressive runner needs a new
freeze against a final trained bundle; no cached complete-action browser result is claimed.

Working title:

> **From Tokens per Second to Time to First Action: Co-Designing Compact WebGPU Agents**

This is a better four-page workshop contribution than “a miniature Kimi/Grok”:

1. It defines and implements an agent-specific latency metric.
2. It couples latency to exact action quality so a fast invalid model cannot win.
3. It identifies feature-materialization parity as a deployment-critical model interface.
4. It compares a browser-oriented structured policy with an autoregressive tool-call baseline.
5. It tests, rather than assumes, whether a hybrid sequence mixer helps WebGPU.
6. It fits the workshop's architecture, training, agent, on-device, and evaluation themes.

The current official CFP offers either a four-page extended abstract plus references or an
optional eight-page full paper plus references, using the NeurIPS workshop template. Submit a
strong four-page extended abstract, not a rushed full paper. Submission is double-blind through
OpenReview, and the CFP says each paper receives three reviews. Original work in progress is
eligible and the proceedings are non-archival, but the work must not already have appeared at a
major ML/AI venue or be under review at or accepted to the NeurIPS 2026 main conference. As of
2026-07-28, the [official CFP](https://slmw2026.github.io/#call-for-papers) still labels the
submission portal “coming soon” and provides no portal link, despite its timeline saying
submissions open July 25; the
[official OpenReview group](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FSLM-Agents)
does resolve.

The official 2026 `dblblindworkshop` source now lives in
[`slmw2026/main.tex`](slmw2026/main.tex). Its verified working render uses four
body/reference pages plus the required checklist; it is a WIP artifact, not submission-ready,
because several checklist answers remain honestly `No`/`N/A` until the anonymous reproduction
package and final experiments exist.

The paper-v5 acquisition is complete and independently `--resume` verified: 507,082 documents and
2,200,005,290 accepted characters in a 2,450,305,820-byte raw mixture (SHA-256
`46650a9bc0ebbdacc6dbd6c87ca3191aee47152f0d179bb2f2e0475f1017094a`). Its 229,501-byte
download manifest has self-hash
`d0cae6e931738261a9481b3ec02c628bed189f27f24c8dfcae02e75bf7238d94`. The four external
prompt-only suites are also strictly replayed and composed. Paper-all preparation retained 504,010
documents and packed 523,358,082 training plus 5,311,528 validation tokens with a train-only 16K
tokenizer. Independent freeze verification passed with canonical SHA-256
`b005ef46b2fd3c8db91a725ff5dca894d97448f2a8e174f7d7f36811dcdd2aa9`. The supplied-denylist
audit screened 19,334 normalized prompts, performed 8,633,077 candidate checks, and removed 15
documents; it is explicitly non-exhaustive.

## What “real time” means

### Evidence-backed external anchors

There is no universal agentic tokens/s threshold. The best current anchors separate prompt and
decode latency:

- [MLPerf Inference 5.1 small-LLM](https://mlcommons.org/2025/09/small-llm-inference-5-1/)
  defines an interactive target of TTFT at most 0.5 seconds and TPOT at most 30 ms, equivalent to
  at least 33.3 decoded tokens/s after the first token.
- [MLPerf Inference 6.0 reasoning](https://mlcommons.org/2026/03/mlperf-inference-gpt-oss/)
  uses interactive TPOT at most 15 ms, or at least 66.7 tokens/s, with workload-specific TTFT
  limits of 1.5–2 seconds.
- [MLCommons Edge Agentic](https://mlcommons.org/2026/07/mlperf-inference-v61-edge-agentic/)
  uses a closed-loop, concurrency-one replay and reports per-turn TTFT, TPOT, end-to-end latency,
  and input/output lengths at tail percentiles. It explicitly avoids treating aggregate throughput
  as the main edge-agent result.
- Google's [Interaction to Next Paint guidance](https://web.dev/articles/optimize-inp) considers
  a next painted response within 200 ms at the 75th percentile good. That is an acknowledgment/UI
  requirement, not a claim that the model must finish a complex action in 200 ms.
- [OSWorld-Human](https://arxiv.org/abs/2506.16042) finds that planning and reflection model calls
  dominate computer-use latency and that strong agents take 1.4–2.7 times as many steps as human
  reference trajectories.
- The [WebLLM paper](https://arxiv.org/abs/2412.15803) reports 71.1 tokens/s for Phi-3.5-mini 3.8B
  and 41.1 tokens/s for Llama-3.1-8B on an M3 Max in Chrome Canary. This shows that the 33 tokens/s
  interactive envelope can be reached on high-end browser hardware; it is not evidence for ordinary
  laptops or for a 200–600 tokens/s requirement.

The proposed LocalAgent service levels are therefore:

| Measurement | Proposed target | Status |
|---|---:|---|
| Visible acknowledgment while inference runs | p75 at most 200 ms | External web UX anchor |
| Complete valid action, reactive tier | `Success@0.5s` | Proposed reporting point |
| Complete valid action, normal interactive tier | p90 at most 1 s | Proposed SLO |
| Local action plus next observable UI state | p90 at most 2 s | Proposed closed-loop SLO |
| Autoregressive decode diagnostic | p90 TPOT at most 30 ms | MLPerf interactive anchor |
| Stretch decode diagnostic | p99 TPOT at most 15 ms | MLPerf 6.0 anchor |

The 0.5/1/2-second points are a benchmark proposal, not an industry standard. The paper must label
them that way.

### Complete-action latency equation

For an autoregressive action requiring `N_decode >= 1` model steps, including any EOS or terminal
step the runtime must observe:

```text
TTFA_harness = tokenize + TTFT + (N_decode - 1) / decode_rate + parse_and_validation

minimum_decode_rate =
    (N_decode - 1) / (TTFA_budget - tokenize - TTFT - parse_and_validation)
```

With tokenization plus TTFT fixed at 200 ms and parse/validation fixed at 20 ms:

| Required decode steps | Needed for 500 ms TTFA | Needed for 1 s TTFA |
|---:|---:|---:|
| 16 | 53.6 tok/s | 19.2 tok/s |
| 32 | 110.7 tok/s | 39.7 tok/s |
| 64 | 225.0 tok/s | 80.8 tok/s |

The rate is conditional on required decode length and fixed latency. It is not an intrinsic
requirement. Moving from 200 to 600 tokens/s saves only 50 ms for a 16-step decode, about 103 ms for
32 steps, and 210 ms for 64 steps. The gain can disappear beneath a network tool call or page
render.

This exact 200/400/600 tokens/s counterfactual is executable rather than hand-calculated:

```bash
PYTHONPATH=src python scripts/realtime_agent_benchmark.py calibrate \
  --ttft-ms 200 --parse-ms 20 \
  --decode-steps 16,32,64,128 --decode-rates 200,400,600 \
  --deadlines 500,1000
```

At a 500 ms budget, 200 tokens/s admits the listed 16- and 32-step scenarios but not 64 or 128;
400 admits 16, 32, and 64 but not 128; and 600 admits all four. All four listed lengths meet one
second at every candidate rate. This is a fixed-cost harness sensitivity analysis, not evidence
of closed-loop browser compatibility. The implementation emits no aggregate percentile over the
equal-weight scenario grid. Final calibration must instead use the frozen external task
distribution, observed decode-step counts and stage costs, independent device runs, task-clustered
`Success@B` intervals, and median-of-run p95 TTFA.

For LocalAgent's structured path:

```text
TTFA_structured = tokenize + one_backbone_forward + route/select/copy/validate
```

It pays no iterative JSON decode term. That is the main treatment, not merely a faster decoder.

## Research questions and prespecified tests

### RQ1: Is TTFA more predictive than tokens/s?

Compare output length, TTFT, TPOT, TTFA, and closed-loop time across actions and contexts. Test
whether tokens/s rankings disagree with `Success@0.5s`, `Success@1s`, and `Success@2s`.

Falsifier: if tokens/s predicts the same system ranking at every tested action length, context, and
device, the new metric adds little.

### RQ2: Does a one-forward structured action head beat autoregressive JSON?

At an identical backbone checkpoint and tool schema, compare:

1. Autoregressive canonical JSON.
2. Grounded candidate-trie constrained autoregressive JSON.
3. Route plus dense selector plus pointer/typed argument heads.

Primary comparison: exact actions delivered within 1 second. Also report raw exact match, invalid
action rate, the declared harness TTFA, and total task time.

The complete-action browser controls now use a lineage-bound prompt-prefill and fixed-`T=1`
cache-bearing decode ABI. The browser validates the final tokenizer, catalog, checkpoint lineage,
parity evidence, metadata, and graph bytes before creating its two sessions, then rebinds cache
tensors without JavaScript readback. Random-weight sizing arms and exact 10.5M pretrain-only
checkpoints have been measured on WebGPU, but no final action-trained cached bundle has. A
headline structured-versus-autoregressive latency claim still requires that trained export and
browser collection.

Falsifier: on at least 200 new unique tasks plus an external benchmark slice, reject the
structured path unless the task-clustered 95% interval for its exact-action difference has lower
bound above `-0.02` and its median-of-run p95 TTFA is at least 15% lower. The current 20-case
diagnostic suite has five-point resolution and cannot adjudicate this rule.
The deterministic import, training-corpus audit, immutable identity, and paired cluster-bootstrap
mechanism is implemented in
[`FRESH_EXTERNAL_EVAL_CONTRACT.md`](FRESH_EXTERNAL_EVAL_CONTRACT.md). No labeled
chronologically-fresh evaluation export has yet been acquired or frozen; the completed four-suite
prompt-only exclusions are decontamination inputs, so this remains readiness evidence rather than
a result.

### RQ3: Does the hybrid backbone help on real WebGPU?

Treatment: `configs/model/webgpu-35m-hybrid.yaml`.

- 34.199M backbone parameters; learned action heads are additional.
- 16K BPE, tied embeddings.
- Width 448, 12 layers, SwiGLU, RMSNorm, RoPE, QK norm.
- One KV head.
- `[conv, conv, attention] x 4`.

Control: `configs/model/webgpu-35m-attn.yaml`.

- Same vocabulary, width, depth, context, heads, normalization, and data.
- Twelve attention layers.
- 34.276M backbone parameters; learned action heads are additional.
- FFN width is 1,328 rather than the hybrid's 1,152 to make the total backbone counts comparable,
  so this is a compound parameter-matched configuration treatment rather than an isolated mixer
  intervention.
- FFN 1328 to reach 34.276M parameters, only 0.22% above the treatment.

Do not presume the hybrid is faster. At 2K context, the absolute MQA KV-cache saving is only a few
MiB, while the current gated-convolution mixer has more projection weights than MQA. WebGPU kernel
launches and memory traffic can reverse a FLOP-based prediction.

Falsifier: use the all-MQA control if the hybrid does not improve p95 action latency at 512–1,536
input tokens without losing more than two points of exact action success.

## What to transfer from frontier models

The design rule is “transfer verified principles, not frontier-scale mechanisms.”

| Source | Publicly supported lesson | Adopt | Do not copy |
|---|---|---|---|
| [nanochat speedrun](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/runs/speedrun.sh) | At audited commit `92d63d4`, the path is tokenizer → pretrain/eval → SFT/eval; RL is not in the speedrun | Reproducible single-command stages, BPB, small proxy sweeps | Present a separate midtrain or RL stage as part of the audited reference speedrun |
| [nanochat RL script](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/chat_rl.py) | Its standalone “GRPO” script calls the update closer to REINFORCE, removes reference KL and PPO ratio/clipping, and uses DAPO-style token normalization | Minimal outcome-reward control as a separate ablation | Relabel the implementation as conventional GRPO |
| [GPT-3](https://arxiv.org/abs/2005.14165) and scaling work | Dense causal decoder and predictable scaling | Strong no-MoE dense baseline and learning curves | Treat dense all-attention as universally best, or current proprietary GPT architecture/training as known |
| [Kimi K3 technical report](https://github.com/MoonshotAI/Kimi-K3/blob/7c5be9599120d7993748de66a76128614f15f210/k3_tech_report.pdf) | 69 KDA + 24 Gated MLA layers; 2.8T total/104B active; curated web/code/math/knowledge plus vision data; small-model mixture/scaling studies; Per-Head Muon; cosine with 1% warmup; progressive context; SFT → nine RL experts → MOPD; MXFP4/MXFP8 QAT | Test hybrid mixing, mixture ablations, verified agent synthesis, and deployment-aware training | Call short convolution “KDA,” call a byte estimate “QAT,” or invent exact source weights/token counts and unpublished selected hyperparameters |
| [Kimi K2](https://arxiv.org/abs/2507.20534) | Agent training at scale and MuonClip/QK-logit clipping | Direct-response action mode and agent evaluation discipline; isolate optimizer/stability tests | Treat MuonClip/QK-Clip as QK-Norm, or copy MoE/MLA into a 35M browser model |
| [Kimi K2.5](https://arxiv.org/abs/2602.02276) | Continual pretraining atop K2-Base on approximately 15T mixed visual/text tokens, native multimodal post-training, and learned parallel Agent Swarm orchestration | Explicit continual-stage accounting and verified parallel-task construction | Vision, MoE/MLA, Agent Swarm, or vendor swarm latency as a browser result |
| [Kimi Linear](https://arxiv.org/abs/2510.26692) | Periodic global attention plus efficient local/linear mixers can help at very long context | Test periodic global attention as a hypothesis | KDA custom kernels or 1M-context claims in ONNX WebGPU |
| [Upstage SOLAR-10.7B](https://arxiv.org/abs/2312.15166) | Depth up-scaling duplicates/expands layers of a compatible pretrained parent, then continues pretraining | Optional checkpoint-growth ablation after a LocalAgent base exists, with matched added compute | Claim inherited knowledge in a from-scratch run, or assume more depth is lower latency |
| [GLM-4.5](https://arxiv.org/abs/2508.06471) | Deep/thin design, GQA, QK norm, direct/thinking modes, multi-token prediction | Deep/thin, MQA/GQA, QK norm; test concise action mode | Assume an MTP auxiliary objective supplies free speculative decoding |
| [GLM-5.2](https://z.ai/blog/glm-5.2) | IndexShare reuses one sparse-attention indexer for four layers; MTP is paired with an explicit verifier/acceptance study; long-horizon RL uses compaction-aware critic PPO and online anti-hack controls | Share only work whose browser kernel cost is measured; add anti-hack checks and preserve compacted-trajectory accounting | Import 1M-context sparse-attention infrastructure into a 2K model, or quote MTP speed without accepted-draft and end-to-end evidence |
| [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | MTP supplies an auxiliary training objective; its speed result uses speculative verification and accepted draft tokens | Train-only MTP ablation | Keep MTP at inference without a browser verifier, acceptance-rate, and latency win |
| [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) | Distilled small models beat its compared small-model RL runs, while frontier-scale RL also elicited reasoning | Distillation-first compute allocation, then executable-reward RL ablation | Conclude that RL cannot work at any scale |
| [Grok-1](https://github.com/xai-org/grok-1) | Public 314B model uses 8 experts, two active, GQA-like 48Q/8KV heads, RoPE | GQA is a credible efficiency baseline | Large vocabulary/MoE; xAI does not publish current coding-model architecture |
| [Grok 4.5](https://x.ai/news/grok-4-5) | The official release discloses aggressive deduplication/quality/domain curation and asynchronous long-running agent RL over verifiable and model-graded technical tasks, but not block-level topology | High-signal data filtering, executable software tasks, and asynchronous rollout orchestration as training-system hypotheses | Infer a hidden architecture, use its reported hosted 80 tokens/s as a WebGPU requirement, or import model-graded rewards as ground truth |
| [MobileLLM](https://arxiv.org/abs/2402.14905) | Deep/thin, GQA, embedding sharing improve sub-billion models | Primary small-model baseline principles | Extrapolate its mobile kernels to browser WebGPU without measurement |
| [OpenELM](https://arxiv.org/abs/2404.14619) | Layer-wise allocation can improve parameter efficiency | Later layer-scaling ablation | Add complexity before the core comparison is stable |
| [Hymba](https://arxiv.org/abs/2411.13676) | Sparse global attention among local/state-space layers works at 125M/350M | Motivation for periodic global layers | Assume SSM scan kernels are portable |
| [WebLLM](https://arxiv.org/abs/2412.15803) | Browser kernels, memory planning, workers, and caching materially affect realized speed | Treat runtime as part of the experiment | Infer device-wide performance from one high-end GPU |

As of 2026-07-28, Moonshot's official K3 repository gives a concrete 93-layer ratio—69 KDA and
24 Gated-MLA layers—along with 2.8T total/104B active scale, 16/896 routing, a 160K vocabulary,
1,048,576-token context, and the stated QAT precisions. The
[launch post](https://www.kimi.com/blog/kimi-k3) describes Attention Residuals, while the
[technical report](https://github.com/MoonshotAI/Kimi-K3/blob/7c5be9599120d7993748de66a76128614f15f210/k3_tech_report.pdf) gives
the data domains, Per-Head Muon plus clipping, cosine/1%-warmup schedule, progressive context
curriculum, and SFT/RL/MOPD stages. It still omits exact source identities and weights, total
training tokens, several selected hyperparameter values, and a browser-portable kernel contract.
Base this paper's implementable choices on those disclosed principles, Kimi Linear, Kimi K2/K2.5,
MobileLLM, and measured LocalAgent ablations.

## Data and training plan

Pretraining and the general/code/structured midtraining streams use checksummed packed shards
built from provenance-bearing documents with stable IDs. Agent midtraining, SFT, and RL use the
repository's canonical `Conversation` interchange. Tool targets use sorted-key compact JSON, and
declared synthetic train/evaluation slot pools and exact rendered prompts remain disjoint. Shared
schema enums, no-argument intents, and template vocabulary are explicitly not claimed disjoint.
The train synthesis config additionally pins the two tracked browser suites and excludes all 21
unique canonical suite queries from every training user turn using Unicode NFKC, case folding, and
Unicode-whitespace collapse. Deterministic template rows are
rule-audited for schema/tool/metadata consistency but are not called environment-verified: their
tool outcomes are not executed.

### Data hygiene gates

Before a full run:

1. Deduplicate documents before splitting. The preparation path now does exact normalized
   deduplication followed by bounded SimHash-LSH candidate search and sampled five-shingle Jaccard
   verification; the manifest labels the near-duplicate audit non-exhaustive.
2. Train the tokenizer on the training split only. `prepare_corpus.py` now computes one
   deterministic split assignment, trains BPE from that exact train set, reuses the assignment for
   packing, and records split/tokenizer document-set fingerprints. Packing also publishes a
   canonical, checksummed document-ID/content-to-split artifact. General/code/structured repacks
   must import that artifact from `paper-all`; they may not rank each subset into a new split.
3. Build and pass an evaluation denylist for BFCL, Mind2Web, WebLINX, BrowserGym tasks, every
   frozen local browser task, and the frozen synthetic agent-evaluation artifact. The corpus-owned
   seven-suite policy cannot be weakened by a denylist manifest. The paper CLI takes a
   provenance-bound list manifest for the four real external prompt-only exports and three direct
   local inputs whose expected byte sizes and SHA-256 hashes live in the corpus config. The
   external exports are not fabricated or vendored in this repository and contain no gold
   actions. Each external list row is linked to a self-hashed per-suite freezer provenance
   manifest, the source-specific adapter audit, and the exact prompt output. Exact-short and
   anchor-shingle containment screening is implemented, but its coverage is only as complete as
   the supplied prompt exports. This pretraining freeze establishes local holdout against known
   benchmark revisions, not chronological freshness. Any fresh capability claim must use a
   different post-training revision, hidden steward export, or newly frozen procedural seed set.
   Exact upstream pins, licenses, native scoring boundaries, privacy constraints, and per-suite
   adapter decisions are recorded in
   [`EXTERNAL_BENCHMARK_AUDIT.md`](EXTERNAL_BENCHMARK_AUDIT.md).
4. Record dataset revision, license, source URL, accepted/rejected counts, and hashes.
5. Keep model, tokenizer, data-order, and task-suite hashes in every benchmark artifact.

The acquisition implementation now makes step 4 a pre-network gate rather than a later prose
promise. It emits a deterministic self-hashed plan whose integer source allocations sum exactly to
the requested character count, verifies revision-bound dataset-card artifacts, enforces a
configured free-disk floor and raw-byte ceiling, and refuses underfilled sources. It checkpoints
only complete source spools. On interruption, verified completed sources can be reused while the
partial immutable stream is replayed; row-exact streaming resume is not claimed.

Paper-scale preparation uses a SQLite-backed multi-pass path rather than retaining every document
body in Python lists. It stages exact-dedup winners, applies the supplied denylist, runs the bounded
near-dedup index, freezes document-level splits, streams only the training split into BPE, and then
streams those same assignments into checksummed shards. The manifest records the raw stream,
upstream download-manifest, staging, filtered-corpus, split, and tokenizer-set fingerprints.
Resource claims remain bounded: one maximum-size document and one shard buffer still occupy memory,
the tokenizer trainer has internal state, and raw/staging/filtered/shard artifacts require several
times the source size in local scratch storage.

Derived midtraining families read the retained `paper-all/filtered.jsonl` and verify the
generation-scoped split assignment referenced by `paper-all/manifest.json`. The mapping binds
`sha256(document_id)` and the filtered-text SHA-256 to `train`/`val`. A missing ID, changed content
under the same ID, checksum mismatch, or split-policy mismatch aborts preparation instead of
silently assigning the row from the smaller family's rank order.
For a source with a fixed config-level license, `license_counts` is evidence of the declared
dataset-distribution license, not a document-level rights audit. Archive the exact upstream
dataset-card/license revision and report that limitation; only row-level license fields support
per-document filtering in the current downloader.

Do not substitute nanochat's MIT-tagged
[`karpathy/climbmix-400b-shuffle`](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle)
repack for licensed general text. Its card identifies a no-content-change redistribution of
NVIDIA
[`Nemotron-ClimbMix`](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix), whose upstream
card is CC BY-NC 4.0/research-only. Conflicting repack metadata is not a transfer of permission;
ClimbMix remains excluded from the default corpus.

### Stage 1: pretraining

Target: approximately 0.7B final-tokenizer tokens for the full 34M pair, subject to compute. Run
proxy sweeps first at 5, 10, and 20 tokens per parameter rather than assuming a single scaling rule
holds below 100M.

Proposed bounded-download mixture (weights allocate characters, not final tokens):

| Character-budget weight | Source family | Purpose |
|---:|---|---|
| 50% | FineWeb-Edu-Dedup from the [SmolLM corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | General language and factual explanations |
| 15% | Cosmopedia v2 from the same corpus | Sample-efficient textbook-style explanations |
| 25% | Proposed permissively licensed code, tests, docs, and API examples; current config is CodeParrot Clean Python only | Coding and executable structures |
| 10% | Proposed CC-BY-4.0 WebSight HTML plus separately sourced generated JSON, JSON Schema, OpenAPI, YAML, and structured UI text; current config is WebSight HTML only | Browser observations and tool schemas |

The existing 65/15/20 data config is a valid smoke mixture.
`configs/data/pretrain-paper.yaml` instantiates the 50/15/25/10 weights, but its configured 25%
arm is permissively licensed CodeParrot Clean Python and its configured 10% arm is WebSight HTML.
It does not yet include separate JSON/Schema/OpenAPI/YAML, tests/API-doc, or multilingual-code
sources, so those remain proposed additions rather than corpus claims. FineWeb's construction is
documented in its [paper](https://arxiv.org/abs/2406.17557); use only sources whose licenses and
revisions are frozen in the manifest. WebSight screenshots are excluded from this text-only model.
Report realized final-token shares from the packed artifacts; do not relabel the character-budget
weights as token shares.

Evaluate validation bits per byte in addition to token loss so byte and BPE experiments remain
comparable. nanochat uses the same vocabulary-invariant principle.

### Stage 2: agent midtraining

Target: 70–100M supervised loss tokens. The forward paper configs use
`data.mixture.unit: loss_tokens`: a deterministic largest-deficit scheduler tracks scheduled
per-source entitlements against measured supervised tokens, normalizes accumulated gradients by
that token mass, and checkpoints enough state for same-runtime exact interruption/resume. Both
paper arms enable resume from 100-step checkpoints, and reject missing optimizer/RNG/history or
inconsistent scheduler/global/per-source accounting, lineage, and resolved runtime identity.
Bit-exact continuation is tested on CPU, not across backends. Reports retain row draws and
input-token shares as separate diagnostics. The production planner made this distinction
operational: it rejected the former 2,500-update ceiling because that horizon contained only
7,429,270 supervised tokens. The corrected 25,000-update horizon schedules 85,536,552 input and
74,836,551 supervised tokens for each arm.

The production one-pass fanout is complete. `paper-general`, `paper-code`, and
`paper-structured` contain 320,850, 51,788, and 131,372 documents and their split-assignment
streams are a document-disjoint exact union of the frozen 504,010-document parent. Hybrid and
attention midtraining plan self-hashes are
`39e7dc0c17adc9cf46cb541f6eb5531195fede79a5bca47dce42fa9b55ca4a78` and
`f52d8446ee70c144b4c406ddbd0656df99deab7d82f112f18376c1057db79cb4`;
both independently replay-verified. No production midtraining result is claimed yet.

| Share | Material |
|---:|---|
| 30% | General pretraining replay |
| 25% | Code, tests, configs, and API documentation |
| 15% | DOM/accessibility trees, JSON, OpenAPI, and schemas |
| 30% | Rule-audited synthetic transitions plus separately executed environment traces |

Increase the agent-transition share late in the stage. Include invalid schemas, unavailable
elements, stale observations, tool errors, retries, irrelevant requests, and
permission/confirmation boundaries. Parsers and JSON Schema can rule-audit static call structure;
claims about task outcomes require unit tests or actual environment state. The current deterministic
generator supplies the former, not the latter. An LLM judge is not the scored source of truth.

### Stage 3: supervised fine-tuning

Target: 5–20M assistant-output tokens.

| Share | Capability |
|---:|---|
| 35% | Function choice and typed argument binding |
| 30% | DOM/accessibility observation to browser action |
| 20% | Error recovery and corrected trajectories |
| 15% | Abstention, irrelevance, and confirmation before irreversible actions |

Use executable synthetic `Conversation` episodes and audited tool datasets such as
[API-Bank](https://arxiv.org/abs/2304.08244) or
[ToolACE](https://arxiv.org/abs/2409.00920). The official Mind2Web training partition is only a
candidate after CC BY attribution, canary/test isolation, and transformation documentation.
WebLINX is held out of the default training recipe: its data are CC BY-NC-SA, trained-weight
share-alike implications need legal review, and the audit found a credential-bearing
demonstration that requires deterministic whole-demo exclusion. Keep
[BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
frozen for evaluation. Exact pins and handling rules are in
[`EXTERNAL_BENCHMARK_AUDIT.md`](EXTERNAL_BENCHMARK_AUDIT.md).

Teach concise latent decisions and exact actions. Do not distill long private chain-of-thought.

The production train-v2/eval-v1 binding described below yields identical deterministic schedules
for the hybrid and attention SFT configs: 10,000 updates, 160,000 draws, 11,629,065 input tokens,
and 7,012,269 loss tokens. The latter lies inside the 5–20M target. Both plan artifacts were
independently replay-verified exactly with `plan_stage_budget.py --verify`; that verifies
deterministic planning, not SFT execution or performance.

### Stage 4: RL simulation

Start in deterministic [BrowserGym](https://github.com/ServiceNow/BrowserGym) / MiniWoB++ tasks,
then evaluate on held-out WebArena/WorkArena-style tasks.

Exact reward components:

- Episode completion.
- Valid AST/schema.
- Correct tool and arguments.
- Penalty for invalid actions and unnecessary steps.
- Measured action-latency and step-count penalty.
- Refusal or confirmation before irreversible actions.

Train on procedural task families and evaluate on unseen templates, websites, schemas, and values.
The paper should report whether RL adds value beyond agent midtraining and SFT; do not assume it.

The production RL configs strictly load the same train-v2/eval-v1 artifacts. Longest gold outputs
are 62 and 56 tokens, respectively, against the frozen 256-token rollout cap, with no truncation.
No production rollout or reward outcome is claimed.

The current canonical-toolcall GRPO runner is an autoregressive control, not structured-policy RL.
It drops inherited structured heads after changing the backbone. Any structured RL claim requires
joint policy-head optimization or an explicit post-RL head recalibration and frozen evaluation.
It now requires a separate `eval_conversations` artifact, fails on canonical-row or rendered-prompt
overlap, and stores deterministic greedy pre/post exact/reward/schema-format metrics plus split
fingerprints. The bounded seed-2027 run supplies that offline measurement: six groups were
informative, 12 optimizer updates were realized, and every held-out metric had zero pre/post
delta. It is not BrowserGym environment RL and demonstrates no improvement.

### Preserving general capabilities such as summarization

Tool specialization must not replace the language distribution.

1. Keep 30% general replay during midtraining.
2. Keep 10–20% general instruction data during SFT, outside the scored agent mixture.
3. Generate sequence-level summaries from licensed pretraining documents with a stronger teacher,
   then filter for source coverage, length, and unsupported named entities.
4. Evaluate summary faithfulness and held-out BPB before and after every agent stage.
5. Reject a data mix that improves agent success but degrades general BPB by more than 2%.

At 34M parameters, “general summarizer” is a bounded capability. The honest goal is short
extractive/abstractive summaries grounded in the supplied document, not frontier factual recall.

## Reusing pretrained weights

Arbitrary public weights do not map directly into the custom hybrid:

- Widths and layer types differ.
- The 16K tokenizer differs.
- A convolutional block has no one-to-one attention weight mapping.
- API teachers generally expose neither logits nor aligned tokens.

Ranked options:

1. **Sequence and trajectory distillation** is the main external-teacher path. It is architecture
   and tokenizer agnostic. Combine verified hard actions with concise teacher outputs and on-policy
   student states. Relevant methods: [MiniLLM](https://arxiv.org/abs/2306.08543) and
   [GKD](https://arxiv.org/abs/2306.13649).
2. **Same-tokenizer LocalAgent teacher** is the cleanest logit-distillation path. Train a larger
   dense teacher with the exact 16K tokenizer, then use the repository's top-k KD during SFT.
3. **Structured pruning plus KD** can compress a compatible 125–160M dense teacher. The
   [Minitron](https://arxiv.org/abs/2407.14679) result is promising but must be revalidated at 35M.
4. **GQA/MQA uptraining** can group pretrained K/V heads, then continue training. The
   [GQA paper](https://arxiv.org/abs/2305.13245) reports conversion with a small fraction of
   original pretraining compute.
5. **Layer expansion** can duplicate compatible blocks before continued pretraining, following
   Upstage SOLAR/Net2Net. SOLAR starts from a pretrained parent and therefore inherits knowledge
   that a from-scratch LocalAgent run does not have. Here it is only a checkpoint-growth ablation
   after a compatible base exists, with matched added compute—not a browser-latency result.
   `localagent grow-checkpoint SOURCE TARGET_CONFIG OUT --layer-map TARGET:SOURCE,...` now
   implements the auditable transform. It requires a complete same-tokenizer config match outside
   explicit depth/layer-type changes, verifies every mapped block kind and target-state hash,
   rejects posttraining auxiliary heads, discards optimizer/RNG/progress state, and is explicitly
   non-function-preserving. A grown checkpoint enters pretraining through weight-only `init_from`
   with a fresh optimizer; it is not an exact resume.
6. **Tokenizer transfer** should use overlapping-token initialization such as
   [FOCUS](https://arxiv.org/abs/2305.14481) when tokenizers differ. Do not apply tokenwise KL
   across unaligned vocabularies.

This ordering is distillation-first, not “RL cannot work.” The
[DeepSeek-R1 release](https://github.com/deepseek-ai/DeepSeek-R1) reports stronger distilled small
models than its compared small-model RL setup while also demonstrating RL-induced reasoning at
frontier scale. LocalAgent therefore spends its scarce capability budget on verified distillation
first and still evaluates executable-reward RL as a distinct last-mile treatment.

For the August submission, prioritize from-scratch matched backbones plus sequence-level teacher
data. Building a robust arbitrary-weight importer is out of scope.

## Experiment matrix

### Required arms

| Axis | Arms |
|---|---|
| Action serialization | Raw AR JSON; grounded candidate-trie AR; structured one-forward |
| Backbone | 34.199M hybrid; 34.276M all-MQA |
| Runtime | WebGPU-only; WASM-only |
| Precision | FP32 and FP16 now; Q4 only after a real implementation |
| Context | 128, 512, 1,024, 1,536 final-tokenizer input tokens; reserve AR output within 2,048 |
| Training | Pretrain only; + agent midtrain; + SFT; + RL if ready |
| Agent share | 10%, 30%, 50% with fixed general replay |

Use identical tokenizer, document order, token count, optimizer, seed, and action heads for the
paired backbone comparison. The clean 10M proxy now has three prospective seeds; the 34M
five-TPP screen still requires at least three. If the promoted 20-TPP/downstream comparison cannot
support
three seeds, report that limitation and bootstrap task-level confidence intervals without
presenting task resampling as architecture-seed uncertainty.

### Secondary ablations

- One versus two KV heads.
- Full 448-dimensional versus factorized 256-dimensional embeddings.
- Twelve unique layers versus six blocks repeated twice.
- Attention placement: all, half, one-third, and first/middle/last.
- General pretraining versus rule-audited agent midtraining versus static SFT.
- Naive Q4 versus activation-aware Q4 only after browser export works.
- Multi-token prediction at inference only if retained draft heads plus a browser-side verifier
  exceed 1.25 accepted tokens/pass and improve end-to-end latency by at least 15% without
  exact-action loss; otherwise keep the auxiliary heads train-only and drop them for deployment.

### Metrics

- Validation bits per byte and token loss.
- Exact tool AST, exact arguments, typed argument accuracy, and schema validity.
- Abstention, unavailable-element, error-recovery, and confirmation accuracy.
- Browser first-step success and final DOM-state/episode success.
- Cold download, session creation, shader compilation, and first inference.
- Warm p50/p90/p95/p99 TTFA and stage breakdown.
- TTFT/TPOT/tokens/s only for autoregressive diagnostics.
- Model bytes, peak memory, input length, and output length.
- `Success@0.5s`, `Success@1s`, `Success@2s`.
- Success per second and success per MiB Pareto frontiers.

## Current implementation state

Already reusable:

- Deterministic `Conversation` data format, disjoint declared train/eval slot pools, zero exact
  rendered-prompt overlap, and exact exclusion of pinned browser-suite queries from training;
  shared schema/template vocabulary remains possible, and template outcomes are labeled
  rule-audited rather than environment-executed.
- Provenance/license fields, stable document IDs, exact deduplication, and checksummed shards.
- Resumable pretraining and midtraining, AMP, validation, atomic checkpoints, WSD/cosine
  schedules, realized input/loss tokens per stage and source, and strict resume-lineage checks.
  No WSD benefit is attributed until an identical-token, identical-data-order cosine control is
  run. That control estimates a same-hyperparameter schedule change; a separate tuned-envelope
  comparison must optimize peak learning rate and batch per schedule. Decay-window data changes
  are a third factor.
- Persisted pretraining validation history plus deterministic same-draw pre/post held-out
  loss/token-accuracy metrics for every configured midtraining source; independently packed
  train/eval roots require split-assignment document/content disjointness proof. SFT records
  held-out assistant loss, token accuracy, and teacher-forced all-assistant-token exactness across
  both main and decay training overlap checks; do not present the last metric as free-running
  generation.
- SFT tool selection, route, pointer heads, compact top-k KD, and exact-AST rewards.
- Explicit disjoint RL holdout input with separate full-artifact and exactly-scored projection
  fingerprints, row/prompt overlap rejection, deterministic greedy pre/post metrics, parent
  tokenizer/stage lineage validation, target-budget checks, and attempted-versus-realized update
  accounting; this remains offline reference-reward GRPO rather than environment RL.
- 28M byte browser demo with one-forward route/select/copy dispatch.
- Hidden-only ONNX action graph that avoids the vocabulary projection while preserving the legacy
  logits-plus-hidden export.
- Exact browser BPE parity with the Python ByteLevel tokenizer, including special markers and
  Unicode, plus validated tokenizer bundling for non-byte checkpoints.
- Fail-closed numerical parity for every emitted fp32/fp16 graph, followed by an exported bundle
  SHA-256/size manifest, checkpoint/config lineage hashes, and browser verification of the graph,
  tokenizer, metadata, and head bytes used by benchmark-grade runs.
- Browser-side typed grounding for strings, enums, integers, numbers, booleans, paths, and URLs.
- Explicit WebGPU-only and WASM-only provider arms with randomized, seeded case order.
- Raw action/DOM suite bytes are checked against immutable size/SHA-256 pins shared with the
  pretraining decontamination policy before parsing; result payloads preserve observed suite and
  bundle-manifest trust-anchor byte evidence.
- Tokenizer training restricted to the deterministic training split, with split and tokenizer
  document-set fingerprints plus a generation-scoped, content-bound assignment artifact in the
  corpus manifest; derived source families import rather than recompute that assignment.
- SQLite-backed corpus hygiene, split assignment, tokenizer iteration, and shard packing without
  retaining all document bodies in Python memory; disk and tokenizer-state limits stay explicit.
- Bounded, deterministic near-deduplication and evaluation-denylist screening before splitting or
  tokenizer training, with explicit non-exhaustive coverage metadata.
- New complete-action metrics in `src/localagent/eval/realtime.py`.
- New WebGPU action benchmark at `spaces/localagent-webgpu/benchmark.html`.
- Matched browser policy modes for one-forward structured dispatch, unrestricted greedy JSON, and
  grounded candidate-trie autoregression, including parse/validation failures and TTFT/TPOT/TTFA.
- Versioned eight-action local DOM harness at `spaces/localagent-webgpu/browser-tasks.html`, with
  independent schema validation, synthetic dispatch, paint barrier, and final-state scoring.

### Frozen synthetic agent-data pair, 2026-07-28

The paired configs now materialize 5,000 train and 5,000 evaluation `Conversation` rows. Each split
contains 4,000 single-turn and 1,000 multi-turn rows, including 750 restraint/no-tool rows. The
full JSONL SHA-256 values are
`161c4b8baab4adb75fe7f968f2bfc15b3e1828c5e376a16b494ee61072179a00` (train) and
`e2c2406865a076f21ca1dd8747187ae2c2a2af0c06888b207b702aa6f2ebfb07` (evaluation).
The RL split audit found zero canonical-row overlap and zero exact rendered single-turn-prompt
overlap. Its order-independent full-artifact fingerprints are
`e645c1d0884f27e6987e52228036c2f888fe4de6b7fdf664c26cc2278ce6870d` and
`1afab95b27bef8102cc5afcf85c5569166e3ac56e735c8b3288b199151a77df9`;
the exactly scored 4,000-row single-turn projections are separately fingerprinted as
`6bc2e16a29c9900bb7c253acb2511040e2f27ae7f905c503b7e652ee1784380f` and
`5d7eb680a55d3014af8602788f1d18e28c5e2936df48cce2b1f0a49042398b21`.
All 21 unique tracked action/DOM queries have zero overlap with every training user turn.
These are deterministic rule-audited templates, not executed environment traces; both manifests
set `environment_executed=false`.

### Production post-training data binding, 2026-07-29

The production lane retains the frozen v1 5,000-row evaluation set above and replaces only the
training side with the 50,000-row train-v2 artifact. The train JSONL is 526,494,339 bytes with
SHA-256 `233f4f2d796568097897c73d4547a0129e73a8509981a308600779e3cb4cc060`; its 6,341-byte sidecar
has file SHA-256 `9d415aef41a1557d4dd16339fcde94d6dff5fcf6fec121372e5cfe3f1875f383`
and self-hash `e5b9d66c7761fb6d9f731e4fab2aa5b316d4714911ab7d00d89a9cbe1bd36243`.
The 1,600-byte generator config SHA-256 is
`2f03b929507e49f7f73a50e125c144fdd09efa6989306ad0e3c0d03beabc6dbe`.
Strict loads found zero semantic-row overlap and zero rendered-prefix overlap across 93,504 train
and 7,963 evaluation prefixes.

One-pass 16K-BPE accounting found 3,633,959 input / 2,191,728 loss tokens in train-v2 and
280,949 / 179,607 in eval-v1. Maximum rendered lengths were 244 and 214 tokens with no truncation.
The hybrid SFT plan at
`data/provenance/paper/stage-budgets/sft-paper-hybrid.json` has file SHA-256/self-hash
`605c418c338c35c02e0947ae9d063dacf81ce80d0b5ce26c9b3979d5c88681e2` and
`f72ac3ff9e1e8866c5437e831f0b3b9340ee68e669bd3278917b21bc4960b286`;
the attention plan at `data/provenance/paper/stage-budgets/sft-paper-attn.json` has identities
`5b6594e70aad40400affcf9cdace07dc3e77863564a267060af1a34768a7aab3` and
`f8e27c4576ec1ed395804fc08cfa2fd6a5835203140710c5f8b58e38f76ff64c`.
Both plan artifacts independently replay-verified those identities exactly. The tested one-pass
derived-corpus implementation has now run on the production parent. Its three repacks preserve an
exact, disjoint union of all 504,010 parent document/split identities. The first measured plan
invalidated the old 2,500-update ceiling at only 7,429,270 loss tokens; corrected matched
25,000-update plans each contain 74,836,551 loss tokens and replay-verify self-hashes
`39e7dc0c17adc9cf46cb541f6eb5531195fede79a5bca47dce42fa9b55ca4a78` and
`f52d8446ee70c144b4c406ddbd0656df99deab7d82f112f18376c1057db79cb4`.
No production post-training run or score exists.

### Bounded matched-training smoke, 2026-07-28

One deterministic CPU-fp32 smoke run exercised the *full* 34.199M hybrid and 34.276M attention
configs through the same pretraining loop. Each arm consumed the same 12,800 next-token
predictions (batch 4, sequence 128, 25 updates, seed 2026) and the sampled-offset fingerprint was
identical:
`e7cad4fd8ff2928a316e768aa0eece8399c31cc96eb685c86d0efa189ca57a70`.

| Arm | Fixed validation CE, initial → final | CPU training time | Predictions/s |
|---|---:|---:|---:|
| Hybrid 34.199M | 9.7966 → 2.8056 | 26.900 s | 475.84 |
| Attention 34.276M | 9.6799 → 3.7855 | 10.540 s | 1,214.44 |

This proves only that both matched configs and the common data path train end to end. The attention
arm was 2.55× faster with the current PyTorch CPU kernels; this says nothing about ONNX WebGPU.
The loss gap is also unusable for model selection: the run is one seed, only 0.000374
predictions/parameter, uses a repetitive local synthetic corpus, fits its tokenizer across the
source before the later train/validation slice, realizes only 4,094 of 16,384 vocabulary IDs, and
has no capability evaluation. The raw results live in the ignored `runs/matched-proxy/` workspace
and must not enter a paper results table. Paper screening still requires the prespecified
multi-seed, held-out, at-least-5-tokens/parameter proxy.

That proxy is now frozen as six configs covering paired seeds 2026–2028. Each arm receives the
same 5,231 deterministic update draws and 171,409,408 scheduled loss-token opportunities; a new
runner gate rejects a packed paper corpus with fewer retained training tokens than that budget.
The exact matrix, execution order, and reporting constraints are in
[`MATCHED_5TPP_RUNBOOK.md`](MATCHED_5TPP_RUNBOOK.md). Raw acquisition and the four real external
prompt-only denylist exports are complete. The prepared manifest, train-only tokenizer, packed
shards, and content-addressed freeze now pass independent verification, but the entire matrix
remains unstarted and has no 34M training outcome.

### Observed exploratory seed-2026 one-TPP 10.5M proxy, 2026-07-28

A separate bounded proxy now supplies the first trained language-quality result, but it does not
replace the prespecified 34M matrix. Its raw mixture contains 120,014,016 accepted characters in
24,125 documents: 78,002,243 FineWeb-Edu-Dedup characters, 18,001,053 Cosmopedia-v2 characters,
and 24,010,720 permissively licensed Python characters. The raw JSONL SHA-256 is
`22d4270ad6157a9701e86be8bfd73a4fc9c480dd2cfd82337a4d6a5218183e6c`. Quality filtering retained
24,004 documents. The fixed 23,764/240-document train/validation split contains 28,045,897
training tokens and 287,995 packed validation tokens; the frozen scorecard covers 287,615 source
tokens. The packed-manifest SHA-256 is
`6a10cc606902a648258dc58ddb3ba19aa68c5b5ed6d812fcb2f06cbedfcaa9fd`, and the train-only
16,384-token BPE SHA-256 is
`8365405524329487aea3b087cc999db887d8276115e67e88ebfcf7901b15617c`.

Both arms used seed 2026, AdamW, the same WSD schedule and deterministic draws, 322 updates, and
exactly 10,551,291 realized loss tokens. The all-attention arm has 10,547,072 parameters and the
hybrid 10,524,544, a 0.214% difference. Because both arms use WSD, this run contains no
WSD-versus-cosine comparison. Evaluation uses every frozen validation document:

| Slice | Attention CE / BPB / top-1 | Hybrid CE / BPB / top-1 | Hybrid change |
|---|---:|---:|---:|
| Aggregate, 240 docs | 6.0547 / 2.0989 / 15.24% | 5.8617 / 2.0319 / 17.28% | CE −0.1930; BPB −0.0669; +2.04 pp |
| General, 216 docs | 6.1901 / 2.0785 / 14.39% | 6.0043 / 2.0161 / 16.23% | CE −0.1859; BPB −0.0624; +1.84 pp |
| Code, 24 docs | 5.3951 / 2.2204 / 19.39% | 5.1673 / 2.1266 / 22.42% | CE −0.2278; BPB −0.0937; +3.03 pp |

Ten thousand paired nonparametric document bootstraps give aggregate 95% intervals for
attention-minus-hybrid of `[0.1835, 0.2034]` CE, `[0.0634, 0.0707]` BPB, and
`[-2.271, -1.834]` top-1 percentage points. All three metric intervals also exclude zero within
the general and code slices. These intervals condition on one architecture seed and the same
240 held-out documents; they are not multi-seed architecture uncertainty.

Checkpoint SHA-256 values are
`b86929f708b0294ff305fa9ffbfa5059e04a807facfc0c5c55d64c471215f4a9` (attention) and
`00dd2cf6651b0a27e18d707d287b464361e4f0636c7c787fafc7570682ab2e6d` (hybrid). The immutable-style
[proxy summary](results/webgpu-proxy-1tpp-10m-seed2026.summary.json) binds exact source revisions,
licenses, config and artifact hashes, counts, scorecards, and deltas.

This is an observed exploratory seed and approximately one loss token per parameter, so it cannot
select an architecture. Its 4,854-entry denylist covers only the frozen synthetic-agent and two
local browser suites, is bounded/non-exhaustive, and removed zero documents. BFCL, Mind2Web,
WebLINX, and BrowserGym exports were not supplied, and no structured-data source was included. The
run measures neither tool use nor structured output, browser tasks, or agent quality.

### Clean prospective confirmatory seeds 2027–2029, 2026-07-28

The prospectively designated clean set repeated the matched 10.5M comparison for three new
training seeds. The primary estimand is attention minus hybrid aggregate BPB; positive differences
favor hybrid:

| Training seed | Attention BPB | Hybrid BPB | Attention − hybrid BPB |
|---:|---:|---:|---:|
| 2027 | 2.103014998958501 | 2.032179123685033 | +0.07083587527346794 |
| 2028 | 2.104777477117417 | 2.031249477749292 | +0.0735279993681252 |
| 2029 | 2.0996846011662638 | 2.0251960502934447 | +0.07448855087281897 |
| Mean | 2.102492359080727 | 2.029541550575923 | +0.07295080850480402 |

Hybrid was favored in all three seeds. The model-based Student-t 95% interval with two degrees of
freedom is `[0.06824707441091234, 0.07765454259869571]`; it assumes approximately normal seed
effects, which cannot be assessed with three observations. The exact sign test is low-power and
not conventionally significant: one-sided `p=0.125`, two-sided `p=0.25`. Secondary mean
attention-minus-hybrid differences are `+0.2104463773820182` CE and
`-0.021929778813112435` top-1 accuracy, equivalent to a 2.193 percentage-point hybrid advantage.
General and code BPB favor hybrid 3/3 as exploratory subgroup checks, with mean differences
`+0.06813601214548147` and `+0.10169832878279628`.

The confirmatory scorecards explicitly ran on CPU in fp32. The six training configs used
`device: auto` and `dtype: auto`, but the runner did not persist the resolved training device.
The current sandbox reports MPS unavailable and resolves `auto` to CPU; that is present-host
evidence, not retrospective proof of the confirmatory training device. The
[confirmatory summary](results/webgpu-proxy-1tpp-10m-seeds2027-2029.summary.json) and
[raw bundle](results/raw/pretrain-proxy-seeds2027-2029/) bind every config/checkpoint, per-document
scorecard, and paired comparison.

This clean set reproduces the direction of the bounded one-TPP language-quality effect across all
three seeds, subject to the small-seed inference limits above. It makes hybrid the provisional
choice for the bounded post-training pilot reported below. The pretraining comparison itself does
does not complete the 34M-class at-least-five-TPP screen, the promoted 20-TPP/downstream quality
comparison, or provide action, agent,
browser-task, or confirmatory WebGPU-latency evidence.

### Exploratory seed-2026 trained pretrain-only WebGPU cache result

The exact seed-2026 checkpoints from the exploratory subsection were exported as parity-gated
fp16 cache-bearing graphs and measured on one Apple M5 in Chrome 150 with ONNX Runtime Web 1.27.0.
Three page/session runs retained 720 measurements after 72 warmups:

| Input tokens | Hybrid p50 wall tokens/s | Hybrid p95 TPOT (ms) |
|---:|---:|---:|
| 128 | 137.711 | 10.305 |
| 512 | 128.232 | 9.610 |
| 1,024 | 123.779 | 10.214 |
| 1,536 | 116.476 | 10.205 |

The hybrid cleared 100 p50 wall tokens/s in every context and every page run. The joint gate of
p50 at least 100 tokens/s and p95 TPOT at most 10 ms nevertheless failed: only the 512-token
condition passed by the median-of-run statistic, and no condition passed both thresholds in every
run. The
[trained summary](results/m5-webgpu-cached-decode-10m-trained-proxy-20260728.summary.json) binds
the checkpoint, tokenizer, graphs, runtime, and the tracked payloads
([run 1](results/raw/m5-webgpu-cached-decode-10m-trained-proxy-20260728-run1.json),
[run 2](results/raw/m5-webgpu-cached-decode-10m-trained-proxy-20260728-run2.json),
[run 3](results/raw/m5-webgpu-cached-decode-10m-trained-proxy-20260728-run3.json)).

This is trained latency for the exploratory seed-2026 pretrain-only checkpoints, not an
agent/capability artifact or confirmatory-seed latency. Quality was scored separately and joined
by exact checkpoint hash; the browser payload contains no quality metric. One checkpoint seed,
one M5/browser/runtime, unavailable adapter/per-node placement, and no cross-device or
confirmatory-latency evidence remain binding limitations.

### Bounded seed-2027 post-training pilot, 2026-07-28

The provisional seed-2027 hybrid checkpoint was continued through 64 midtraining, 320 SFT, and
32 offline-RL steps. The
[mechanically validated stage summary](results/webgpu-proxy-pilot-seed2027.summary.json) binds
every config, parent/checkpoint hash, data and tokenizer identity, held-out contract, and runtime.
All three runs resolved `device: auto` to MPS/fp32 under PyTorch 2.13.0.

| Stage and frozen metric | Before | After | Interpretation |
|---|---:|---:|---|
| Midtrain agent loss / token accuracy | 7.7869 / 3.71% | 2.6371 / 69.80% | Strong directional agent adaptation |
| Midtrain general loss / token accuracy | 5.7064 / 18.68% | 5.7342 / 18.30% | Small general regression |
| SFT assistant loss / token accuracy | 2.7320 / 67.29% | 1.8146 / 73.13% | Directional improvement |
| SFT teacher-forced all-assistant-token exact | 0/65 | 1/65 | Weak teacher-forced consistency; not autonomous generation |
| RL greedy exact / tool exact | 1/53 / 0/51 | 1/53 / 0/51 | Zero held-out delta |

The RL run attempted 128 groups and 256 rollouts. Six groups were informative, producing
12 optimizer updates over two policy epochs each; 26 of 32 steps had zero signal. Tool-format
validity stayed 13/51 and mean reward stayed 0.0434. This is a realized-update negative result,
not a no-op run and not evidence of RL benefit. The RL artifact invalidates the inherited
structured heads because the control updates only the autoregressive LM, so the browser action
pilot uses the SFT checkpoint.

### Pre-assistant-padding fixed-512 SFT WebGPU action and DOM pilot, 2026-07-28

The exact SFT checkpoint
`79387105de75d332413262e8d8ddb847b6cc13bc03f5e4df3c81663d9897aef1` was exported as a
21,430,301-byte parity-gated fp16 action graph. Three fresh page/session runs on one Apple Metal 3
adapter in Chrome 150 / ONNX Runtime Web 1.27.0 each used 20 held-out cases, three warmups,
30 measured repetitions, concurrency one, seed `slmw2026-v1`, and exactly 512 final tokenizer
tokens under the internally prespecified pre-assistant-padding stress condition.

| Run | TTFA p50 (ms) | TTFA p95 (ms) | Exact action | Schema valid |
|---:|---:|---:|---:|---:|
| 1 | 24.75 | 34.405 | 5% | 100% |
| 2 | 24.55 | 34.30 | 5% | 100% |
| 3 | 25.20 | 34.80 | 5% | 100% |

All 20 unique cases abstained, so capability was 1/20 overall, 0/19 tool-required, and 1/1
abstention. Repeated timing rows yield 90/1,800 exact and 0/1,710 tool-required; those are not
independent capability tasks. Every row was on time by 100 ms, so `Success@100ms`,
`Success@250ms`,
`Success@500ms`, `Success@1s`, and `Success@2s` all equal the 5% exact rate. The 100% schema
rate means that abstention is a valid action-suite output, not that the policy emitted correct
tool schemas. The
[action summary](results/m5-webgpu-sft-action-pilot-seed2027.summary.json) binds the exact graph,
checkpoint, tokenizer, heads, suite, runtime, and three raw runs.

The executable eight-case DOM harness used the same fixed-512 condition. Capability was 0/8 unique
tasks. Across three 240-row timing runs, exact action, independent executable-schema validity,
final DOM state, state transition, and closed-loop success were all 0/720. Pooled closed-loop
latency was 33.30 ms p50 and 66.80 ms
p95, but no action was useful. The
[DOM summary](results/m5-webgpu-sft-dom-pilot-seed2027.summary.json) binds the raw payloads:
[run 1](results/raw/m5-webgpu-sft-dom-pilot-seed2027-run1.json),
[run 2](results/raw/m5-webgpu-sft-dom-pilot-seed2027-run2.json), and
[run 3](results/raw/m5-webgpu-sft-dom-pilot-seed2027-run3.json).

An exploratory offline parity diagnostic found 17/20 correct route decisions on natural held-out
prompts and 17/19 selector top-1 matches among tool cases; it produced no text route and therefore
missed the sole abstention. On the independent frozen eval decisions, natural route accuracy is
83/98, selector top-1 is 72/79, and dispatched tool accuracy is 70/79. The internally prespecified
pre-assistant-padding stress condition inserts real, unmasked single-token spaces before the
assistant marker and reads the final hidden state, unlike the natural SFT-probe feature
materialization; every case at 128 tokens and above routes to text. Native PyTorch, fp32/fp16 ONNX,
and exported JSON heads agree, so export or fp16 conversion is not causal. This is a
feature-materialization shift, not generic long-context evidence. It does not convert the offline
diagnostic into a natural-prompt WebGPU score or excuse the fixed-512 failure; the capability
gate fails.

The corrected fixed-compute runner appends filler after the natural assistant marker, still runs
the 512-token graph, dispatches from `hidden[natural_input_tokens - 1]`, and bounds pointer scans
to the natural tokens. The
[offline audit](results/sft-structured-context-robustness-seed2027.summary.json) preserves the
natural route/selector counts on both frozen suites. The
[full-stack parity gate](results/sft-structured-export-parity-seed2027.summary.json) has exact
native/fp32-ONNX/fp16-ONNX agreement on 20/20 reused-suite routes, tools, grounded arguments, and
normalized actions; its shared 16/20 offline exact score is not a new capability estimate. Runner
versions `rtab-0.4` and `rtab-dom-0.4`, artifact identities, case-order seeds, and required record
checks are locally frozen in the
[corrected browser protocol](results/webgpu-proxy-pilot-seed2027.corrected-browser.protocol.json);
an external timestamp plus three action and three DOM browser runs remain pending. Because the
action, DOM, and 65-row suites
informed the diagnosis, the corrected arm is a reused-suite deployment-parity re-evaluation, not
confirmatory capability. A genuine claim about the original
pre-marker fixed-512 materialization would separately require head training and evaluation on
that exact condition.

### One-device matched WebGPU latency gate, 2026-07-28

The two parameter-matched fp16 hidden-only graphs were then measured in Chrome 150 with ONNX
Runtime Web 1.27.0 on a 32 GB MacBook Air with an Apple M5. The runner requested exactly
`executionProviders: ["webgpu"]`, made no whole-session provider retry, stayed foreground at
concurrency one, and timed `session.run` through the resolved `[1, sequence, 448]` fp32 hidden
output. This includes the default CPU output readback. Three fresh page/session runs each excluded
three warmups and retained 30 balanced, seed-randomized repetitions per graph and exact tensor
length. The point estimates below are medians of the three within-run percentiles; the extra column
shows the complete run-to-run p95 range.

| Input tokens | Attention p50 / p95 median (ms) | Hybrid p50 / p95 median (ms) | p95 range, attention / hybrid (ms) | Median run ratio, p50 / p95 |
|---:|---:|---:|---:|---:|
| 128 | 74.50 / 83.38 | 43.90 / 48.61 | 57.51–85.32 / 33.71–52.95 | 1.70× / 1.71× |
| 512 | 115.65 / 130.71 | 66.25 / 82.75 | 114.83–132.79 / 70.46–93.02 | 1.66× / 1.58× |
| 1,024 | 304.50 / 321.41 | 154.70 / 174.54 | 284.90–335.21 / 151.56–178.90 | 1.97× / 1.87× |
| 1,536 | 593.05 / 621.68 | 288.65 / 300.30 | 529.57–627.25 / 266.28–313.43 | 2.05× / 2.00× |

All 720 measured forwards completed with zero runtime failures. The result passes the *latency*
half of the hybrid selection gate on this device, especially at long context; it does not pass the
quality half for the 35M-class pair. Both graphs have deterministic random weights, so final
architecture screening still requires the prespecified at-least-five-TPP 35M-class comparison
and trained exact-action evaluation. Absolute latency shifted materially after the first run,
while the paired hybrid advantage remained. The failed WASM attempt occurred between runs one and
two, so these are not clean independent repeats of one system state; this is why the result
reports run clusters rather than pooling or cherry-picking one page run. macOS reported AC Power
while the battery discharged from 62% to 56%, and its thermal/performance warning status was
unavailable; a paper-quality rerun must control and record those conditions.

The three raw collection artifacts remain under `runs/webgpu/results/` and are also copied
byte-for-byte into the tracked `docs/paper/results/raw/` directory. They have SHA-256 values
`b414a805d6d57f64ac76956daa862a31a988e48696a92d2f3d03f5289237b5af`,
`cdf83f0fbf9aa496ec2a9b5885cb4e6d0226f9923a50fc040873a02e2e834cac`, and
`bf234a694916a583989f5dd7585683c7677453baecdb337a916bf718c4c3ec7f`; the tracked compact record is
[`results/m5-webgpu-backbone-20260728.summary.json`](results/m5-webgpu-backbone-20260728.summary.json).
`tests/test_webgpu_result_artifacts.py` verifies every tracked byte count and hash and recomputes
the reported percentiles, ranges, and paired ratios from those raw records.
The graph hashes, canonical configs, and 0.223% parameter delta were verified before inference.
ORT reported version 1.27.0 and exposed a WebGPU device, but not adapter identity or per-node
placement, so the provider evidence remains an exact WebGPU session request with per-node fallback
status **unknown**, not proof that every ONNX node ran on the GPU. A separate single-thread WASM
control loaded both sessions and then lost the browser target during inference; it produced no
valid latency payload and is a failure condition, not a censored timing row.

### Cache-bearing random-weight deployment frontier, 2026-07-28

The hidden-only gate above is insufficient for decoded-token throughput. A second exporter now
emits two graphs per arm: prompt prefill and fixed-one-token greedy decode. It flattens
heterogeneous attention K/V and convolution state into a stable graph ABI, derives RoPE position
from the attention cache, and refuses to publish unless the exact graph bytes pass PyTorch
trajectory parity at distinct prompt lengths 1, 8, and 31 for four successive decode steps.
Greedy token IDs match exactly; fp16 cache tensors stay within a maximum allowed absolute
tolerance of `1e-1` (the observed maxima were below `8.4e-3`).

The browser requests cache outputs as `gpu-buffer`, binds every present tensor directly as the
next past input, never reads cache data into JavaScript, and disposes superseded/final tensors.
The first token is the CPU-resident prefill `next_token`. TPOT is the elapsed interval from first
to last CPU-available token divided by 31, while model-only throughput separately sums the 31
decode `session.run` intervals. The cache implementation uses append/concat and fresh present
tensors; it is not in-place or paged. Prefill and decode are separate sessions and may duplicate
weights.

Three foreground page/session runs per size retained 30 seeded, balanced measurements after three
warmups for both architectures at 128, 512, 1,024, and 1,536 prompt tokens. Each sample generated
32 tokens through one prefill plus 31 cache-bearing decode calls. The table uses the median of the
three within-run p50 wall-throughput estimates; p95 TPOT is used for the tail-latency view.
Values within a cell follow the four contexts in ascending order.

| Matched pair | Attention p50 wall tokens/s | Hybrid p50 wall tokens/s | Hybrid p95 TPOT (ms) | 100 tokens/s reference |
|---|---:|---:|---:|---|
| 34.276M / 34.199M | 52.57 / 44.82 / 36.07 / 27.70 | 74.05 / 64.54 / 57.53 / 47.15 | 16.17 / 18.06 / 20.01 / 22.71 | Missed |
| 15.618M / 15.638M | 64.75 / 62.73 / 58.39 / 46.61 | 90.87 / 89.26 / 80.94 / 79.77 | 12.98 / 12.96 / 15.22 / 15.04 | Missed |
| 10.547M / 10.525M | 130.20 / 126.90 / 107.19 / 91.73 | 159.23 / 160.46 / 143.49 / 127.57 | 7.81 / 8.10 / 9.14 / 9.65 | Hybrid passed all contexts |

The smallest hybrid uses four width-384 layers with
`[convolution, attention, convolution, attention]`, rather than the deeper/thinner nine-layer
15.6M shape. It is the only tested hybrid to clear the requested 100--300 tokens/s engineering
band at every context on this device. That line is a deployment reference, not a universal
realtime standard. The 34M hybrid still wins its matched latency comparison by 1.42--1.69x but
does not reach 100 tokens/s; reducing parameters without WebGPU-aware shape selection is also not
monotonic. Cross-size values are descriptive because the size pairs were measured in separate
two-arm experiments.

A leftover quick-flywheel child process was discovered during preliminary 15.6M attempts. Those
attempts were excluded before evidence was finalized, the exact process was terminated, and all
three tracked 15.6M raw files were recollected without that process. This does not affect the
earlier 34M collection, which preceded the 15.6M candidate work. The final compact summaries and
byte-for-byte raw records are tracked under [`results/`](results/), with tests that recompute
hashes, cache accounting, graph-pass counts, within-run percentiles, run ranges, and paired
ratios.

All three pairs use deterministic random weights. The measurements select a *latency candidate*
only: they establish no language modeling, coding, tool-use, exact-action, or agent quality, and
they do not change the prespecified 34M matched-training estimand. ORT Web exposed a WebGPU device
and accepted an exact one-provider request without a whole-session retry, but adapter identity and
per-node placement/fallback remain unknown.

P0 blockers:

1. The 10M SFT action graph is deployable and fast, but the pre-assistant-padding fixed-512
   stress condition collapses to abstention. The corrected fixed-compute dispatch-position arm is
   implemented and locally frozen but still needs an external timestamp plus three action and
   three DOM browser runs; the tracked 34M
   smoke checkpoint lacks trained route/pointer heads.
2. The interactive planner still simulates tool responses. The trained 10M DOM pilot ran all
   eight paths but achieved 0/8 unique-task closed-loop success (0/720 repeated timing
   opportunities); multi-step BrowserGym remains separate.
3. No quantized WebGPU graph exists; Q4 size numbers are theoretical.
4. The revision-frozen BFCL, Mind2Web, WebLINX, and BrowserGym prompt-only exports pass strict replay
   and are named by the private v3 aggregate manifest. The independently verified paper-all freeze
   binds them with the three config-hash-pinned local exclusions and a train-only tokenizer. Its
   19,334-prompt supplied-denylist audit is bounded and explicitly non-exhaustive; it is not a
   labeled fresh-evaluation slice or native benchmark score. Protected terms require private local
   retention, so a full paper corpus remains unreproducible from tracked files alone until an
   authorized archival package exists.
5. The complete-action browser AR controls are wired, but they recompute the full prefix. The
   standalone random-weight frontier and the exact pretrain-only trained cache graphs prove the
   cache-bearing runtime path, but neither includes action-trained weights, heads, or the schema
   used by the complete-action comparison. Candidate-trie AR also covers one deterministically
   grounded argument assignment per tool rather than a general JSON-Schema language.
6. Only one device has been measured, using three sequential random-backbone page/session runs
   whose system state was not fully controlled. The exact exploratory seed-2026 one-TPP 10.5M
   checkpoints have three trained WebGPU page/session runs, and a separate clean seed-2027–2029
   set reproduces the 10M language-quality direction on CPU-fp32 scorecards. The seed-2027 SFT
   checkpoint now also has three fixed-512 action and three DOM runs, all negative on
   tool-required capability. The prespecified multi-seed 35M-class ≥5-TPP comparison,
   confirmatory checkpoint latency, controlled independent repeats, natural-prompt agent
   evaluation, and cross-device coverage remain uncollected.
7. Runtime exceptions receive finite failure rows, but a hung browser inference is not cancellable;
   a paper-grade finite timeout/worker-isolation path is still required.
8. The historical 5,000-row training artifact remains the bounded pilot described above. Production
   now uses 50,000 train-v2 rows against the frozen 5,000-row eval-v1 set with zero semantic or
   rendered-prefix overlap, but its synthetic tool outcomes remain rule-audited rather than
   environment-executed and the SFT configs still have no general-instruction replay. Larger volume
   and a valid token budget are not evidence of broad agent coverage or performance.
9. Exact duplicates removed by the bounded upstream downloader retain aggregate skipped counts,
   not per-alias provenance. The richer alias table begins at local corpus staging.
10. Pretraining, midtraining, and SFT preserve held-out metrics, and the bounded run supplies the
    first numeric stage evidence, but no general promotion thresholds are configured. Baseline
    runs must establish defensible regression tolerances before automatic accept/reject gates.

The paper must not claim a trained 34M browser agent, multi-step browser success, a trained and
integrated best-case cached-AR complete-action comparison, useful capability from the failed
fixed-512 SFT pilot, or a Q4 result until those gates pass.

## Four-page paper structure

1. **Introduction (0.55 page)**  
   Tokens/s is ambiguous for an agent; complete valid actions are the executable unit. State TTFA
   and the compact WebGPU co-design question, with backbone and deployed-policy counts separated.

2. **Method (1.05 pages)**  
   Define TTFA/`Success@B`; describe the structured route/select/copy path; specify the matched
   hybrid and all-MQA backbones.

3. **Benchmark and protocol (0.65 page)**  
   Frozen function calls, deterministic browser microtasks, closed-loop tasks, context sweep,
   cold/warm separation, devices, and statistical procedure.

4. **Results (1.25 pages)**  
   One main Pareto plot, one latency-breakdown plot, and one compact ablation table. Report negative
   results honestly.

5. **Limitations/conclusion (0.5 page)**  
   Text/DOM rather than vision, limited task suite/hardware, small capacity, no claim that proposed
   deadlines are universal.

## Execution calendar

| KST date | Exit criterion |
|---|---|
| Jul 27–29 | Completed: TTFA definitions, benchmark JSON schema, matched config, benchmark implemented |
| Jul 30–Aug 2 | Partly completed early: BPE/action export and WebGPU arm; matched WASM/action controls remain |
| Aug 3–5 | Completed early as a negative pilot: executable local DOM scoring, 0/8 unique-task success |
| Aug 6–8 | Completed early: bounded 10M post-training pilot; fixed-512 capability failed and the 34M five-TPP screen plus 20-TPP follow-up remain open |
| Aug 9–13 | Train two finalists with frozen data/tokenizer/order; checkpoint every stage |
| Aug 14–16 | Run device matrix, repeated latency tests, action/task evaluation |
| Aug 17–18 | Statistics, figures, failure analysis, and contamination audit |
| Aug 19 | Complete anonymous four-page draft |
| Aug 20 | Internal review and reproduction from a clean checkout |
| Aug 21 | Final PDF, appendix/artifact audit, OpenReview metadata |
| Aug 22 AoE | Submit; do not use the translated KST edge as planned working time |

Go/no-go on August 8:

- **Go with a positive architecture paper** only if the five-TPP screen promotes a candidate, the
  promoted 20-TPP/downstream model passes on a new untouched capability set, and the same trained
  checkpoint satisfies its declared structured or autoregressive deployment gate. Success on the
  reused corrected suite alone is insufficient.
- **Go with benchmark/system paper** if the byte structured model and multiple baselines have real
  measurements but the final BPE training is incomplete.
- **Go with a WIP/negative-methodology paper** if the central evidence remains the TTFA/Success@B
  definition and deployment feature-contract failure, with every missing positive comparison
  explicit.
- **Do not submit performance claims** if results are simulated, providers are unverified, or raw
  artifacts cannot reproduce the tables.
