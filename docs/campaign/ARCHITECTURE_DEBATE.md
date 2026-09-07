# Architecture debate: what a sub-100M on-device tool agent should steal from the 2024–2026 model zoo

> Companion to [`COMPUTER_USE_DEBATE.md`](./COMPUTER_USE_DEBATE.md) (which argued *scale vs structure* for
> the agent loop) and [`RESEARCH.md`](./RESEARCH.md) (per-project adoptions). This one is about the
> **model itself** — we read the architecture/training reports of the recent frontier and small-model
> waves, extract the *fundamental philosophies*, stage them as a debate, and decide what actually ports
> to our **~1M** and **~30M** from-scratch, pure-PyTorch, browser-deployable tool-calling tiers.

## The central tension (state it before the debate)

Almost every celebrated model below is a **scale-era frontier model optimizing a *serving* constraint** —
KV-cache memory, long-context throughput, expert routing at hundreds of billions of params. **Our
constraint is the opposite: we are parameter- and compute-starved, on short contexts, under browser
download, memory, and kernel constraints.** Most of the famous tricks (MLA, MoE, FP8, million-token
sparse attention) are answers to problems we *don't have*. The genuinely portable ideas are a
minority — and they cluster in **training method**, not architecture. The debate exists to separate
the two.

## The cast

- **The Scaler** — capability comes from parameters, cleverly served. MoE, big teachers, RL. ("Just
  distill a giant into it.")
- **The Structuralist** — the *primitive* is the lever. Cheap mixers (short conv), depth over width,
  weight sharing. ("Spend your few params on the right operator.")
- **The Efficiency Hawk** — nothing counts unless it pays on the target device. Hardware-in-the-loop;
  latency/memory over benchmark vanity. ("Show me the p95 decode in the target browser.")
- **The Data/Distillation Camp** — architecture is nearly a rounding error; capability is information
  density per token + imitation of a teacher. (Phi, SmolLM2, the R1 distills.)

---

## The axes, the argument, the verdict

### 1. Scale vs structural efficiency
**Scaler:** Kimi K2 (1T/32B-active), DeepSeek-V3 (671B/37B), and Llama-4 MoE pursue frontier quality at
a fraction of *served* FLOPs. Kimi K3's
[official configuration](https://github.com/MoonshotAI/Kimi-K3/tree/7c5be9599120d7993748de66a76128614f15f210) is even more explicit: 2.8T total /
104B active parameters across 93 layers, split into 69 KDA and 24 Gated-MLA layers. Its complete
source inventory and mixture weights are not disclosed, but the
[technical report](https://github.com/MoonshotAI/Kimi-K3/blob/7c5be9599120d7993748de66a76128614f15f210/k3_tech_report.pdf) does
specify Per-Head Muon, cosine decay with 1% warmup, progressive context extension, and an
SFT→RL-experts→MOPD post-training pipeline. Those frontier-scale choices still require local
ablation rather than literal transfer. **Structuralist:** Nemotron-H replaces ~92% of attention with
Mamba-2 for 3× decode; LFM2 is conv-heavy; MiniMax/Qwen3-Next go 75–87% linear. The frontier itself is
drifting from "scale up" to "scale up **but make each served token cheaper**."
**Verdict (ours):** we have no scale to serve — only the *cheaper-primitive* half of the lesson applies.
Structuralist wins our tier outright.

### 2. Attention vs sub-quadratic mixing
**Structuralist:** most mixing should be cheap and local — Nemotron-H (~8% attention), Qwen3-Next (25%),
LFM2 (≈37% attention: 10 conv + 6 GQA), Gemma-3 (5 local : 1 global). **Scaler/Hawk rebuttal:** you still
need *some* softmax attention for in-context retrieval and **copying tool arguments verbatim** — the one
thing a tool agent cannot get wrong. Consensus number across the zoo: **~1 attention layer per 3–8
sub-quadratic layers.** The Hawk adds the decisive datum: Liquid's *hardware-in-the-loop* search found
that **Mamba/SSM/linear-attention did not beat a plain gated short-conv + GQA hybrid under CPU
latency/RSS budgets** — and often hurt them.
GLM-5.2's [official report](https://z.ai/blog/glm-5.2) is a useful boundary case: sharing one
dynamic-sparse-attention indexer across four layers reduces indexer work at one-million-token
context, yet the authors explicitly report that KV-cache size does not fall proportionally. That
solves a long-context serving problem, not LocalAgent's 2K-context browser bottleneck.
**Verdict (ours):** **gated short causal conv (k=3, double-gated, LIV-style) as the primary mixer + 1–2
global GQA layers** for argument-copying. **Do not reach for Mamba/SSM without a measured browser
kernel** — its fused-CUDA advantage is not portable, and a naive PyTorch scan can also be slower than
small windowed attention on the CPU fallback. (LFM2; Nemotron-H.)

### 3. Dense vs sparse (MoE)
**Scaler:** MoE decouples capacity from active FLOPs — the default above ~30B-active. **Everyone else:**
every MoE model in the corpus is *huge*; none is small, because MoE needs enough params-per-expert to be
worth the routing tax, and load-balancing is unstable with tiny batches.
**Verdict (ours):** ⚠️ **Dense deployment control plus an opt-in Micro-MoE experiment.** A dense no-MoE
model remains the deployment baseline because every expert increases browser download and resident
weight memory, while dynamic dispatch is poorly portable. The PyTorch research path now implements
actual top-k expert execution, honest total/active accounting, a training-only balance loss, and routing
diagnostics. Its 43.86M-total/17.32M-active treatment is compared with a topology-matched 17.30M dense
control. It cannot earn a WebGPU-speed claim until a sparse export, parity, and target-browser benchmark
exist. This tests capacity per active path rather than assuming that frontier MoE transfers.
([MobileLLM](https://arxiv.org/abs/2402.14905);
[Kimi Linear](https://arxiv.org/abs/2510.26692);
[OLMoE](https://arxiv.org/abs/2409.02060).)

### 4. Learn-from-scratch vs distill-from-teacher  ← *the load-bearing axis*
**Data/Distillation Camp:** the
[DeepSeek-R1 release](https://github.com/deepseek-ai/DeepSeek-R1) reports that its distilled small
models outperform the reasoning patterns found by its compared small-model RL runs. The same work,
however, also demonstrates that RL can elicit reasoning in a sufficiently capable base model.
Distillation-first is therefore a compute-allocation argument for this tier, not evidence that RL
cannot work. LFM2 goes furthest by making a decoupled tempered Top-K distillation objective a primary
training signal rather than a post-hoc compression step. **Scaler caveat:** gains depend on the
teacher–student capacity gap, and our teacher is only 30M rather than 7B, so expect modest rather than
miraculous lifts. **Sub-debate (which divergence):** forward KL is mode-covering, whereas reverse KL
is mode-seeking and can suit tool-calling, where one correct call beats a diffuse distribution
(MiniLLM). **On-policy** distillation on student-sampled trajectories also attacks exposure bias.
**Verdict (ours):** ✅ **30M → 1M distillation is the first capability-transfer experiment.** The tiers
share tokenizer and heads, so no vocabulary projection is required. Recipe: tempered Top-K KD, followed
by a measured on-policy pass over student-sampled tool-call trajectories. Outcome-reward RL remains a
separate, executable-reward ablation rather than being declared impossible.

### 5. Next-token vs richer objectives (MTP / RL)
**Scaler:** DeepSeek-V3, Qwen3-Next, and GLM add **Multi-Token Prediction** heads for an auxiliary
training signal; R1 uses RL for reasoning. **Hawk:** MTP is not free speculative decoding. The
[DeepSeek-V3 report](https://arxiv.org/abs/2412.19437) drops the MTP modules for ordinary inference;
its acceleration result retains draft predictions inside a speculative-decoding framework and
depends on verifier acceptance. A browser implementation therefore needs retained heads, a verifier,
an acceptance-rate measurement, and end-to-end latency evidence. GLM-5.2 makes that dependency
especially explicit: its seven-step MTP ablation reports accepted draft length, shares index/KV
state, and uses rejection sampling plus an end-to-end distributional loss. Its separate long-horizon
agent stage moves to critic-based PPO and deploys online anti-hack checks around tool calls. We adopt
the executable-verifier and anti-hack discipline, not its frontier rollout infrastructure.
**Verdict (ours):** **MTP is train-only by default and worth a controlled 30M ablation** as a denser
auxiliary signal; skip it at 1M. Promote it to inference only after the speculative path wins on the
browser. Use RL as a measured last-mile stage for behaviors with executable rewards, not as an
unverified primary source of sub-100M capability.

### 6. Depth + weight-sharing vs width (the small-model regime)
**Structuralist:** MobileLLM — "deeper is more crucial than wider"; a 30-layer 125M beats wide-shallow,
and since **embeddings are >20% of params at this scale**, embedding tying + **immediate block-wise weight
sharing** (reuse a block back-to-back, free in cache) add +0.7–0.8 pts. **Us:** our 1M already runs
**factorized embeddings + depth-recurrence** — i.e. MobileLLM's two levers in a *more aggressive* form
(recurrence = the limit of block sharing).
**Verdict (ours):** keep both; **but recurrence's weakness is "every layer is identical."** Two cheap
fixes from the zoo: (a) **partially untie** the recurrence (cycle 2–3 *distinct* shared blocks, MobileLLM-LS
style, rather than one block) ; (b) **Per-Layer Embeddings as capacity injection** (Gemma-3n's PLE idea,
*re-purposed* — not for VRAM offload, which is moot on CPU, but to give each recurrence step a learned
per-layer bias so identical weights still specialize).

### 7. Data density, schedule, and "train once, deploy many"
**Data Camp:** Phi (textbook/synthetic data), SmolLM2 (edu-filtered curation),
[MiniCPM](https://arxiv.org/abs/2404.06395) (**192× data:model ratio** — small models want *far* more
tokens than Chinchilla) — capability is **information density per token**. **MiniCPM's WSD schedule:**
long stable plateau + short exponential decay, with the **sharp
decay-phase loss drop**; fork the stable checkpoint to any budget, and **inject your cleanest data in the
decay window**. **Elastic camp:** Gemma-3n MatFormer nests submodels; merge-camp (Arcee/LFM2.5) combines
specialist runs in weight space instead of retraining.
**Verdict (ours):** ⚠️ **WSD is a promising schedule, not a free gain.** First compare it against
cosine with identical data order, tokens, optimizer, seed, validation, and shared hyperparameters
to estimate the controlled schedule change. Then tune peak learning rate and batch independently
for each schedule before comparing their achievable envelopes: Kimi K3 reports that shared
hyperparameters can favor the schedule they happen to fit. Clean-data injection during decay is a
third experimental factor and must not be conflated with either comparison. ✅ **Curriculum**
(LFM2 difficulty-ordering by empirical success
probability). ✅ **Model merging across flywheel rounds** (soup → TIES if interference appears) instead
of always retraining from union data. ⚠️ **MatFormer only *within* a tier** (sample FFN width per step
inside the 30M) — **not across tiers**, since our 1M differs from the 30M in depth/recurrence, not just
FFN width, so a 1M is *not* a clean top-left slice of the 30M.

### 8. Stability as a first-class design target (the quiet 8th axis)
[Qwen3](https://arxiv.org/abs/2505.09388) adds **QK-Norm** (RMSNorm on Q/K). Kimi K2's
[MuonClip](https://arxiv.org/abs/2507.20534) instead clips QK attention logits while using Muon.
These are distinct mechanisms: K2's QK-Clip result does not directly validate QK-Norm. Stability is
treated as architecture rather than an afterthought, and it matters at tiny width and with byte-level
inputs.
**Verdict (ours):** ✅ Keep **QK-Norm** as the cheap Qwen-derived baseline; test logit clipping and Muon
only as separately controlled optimizer/stability ablations.

---

## Verdict table — what the 1M/30M adopt

| Idea | Source | 1M | 30M | Note |
|---|---|---|---|---|
| **30M→1M distillation** (tempered Top-K → on-policy, reverse-KL) | LFM2, R1, Qwen3, MiniLLM | ✅ student | ✅ teacher | highest leverage; same tokenizer ⇒ no projection |
| **Gated short-conv (k=3) + 1–2 GQA hybrid** | LFM2, Nemotron-H | ✅ mostly-conv | ✅ ~3:1 conv:attn | keep ≥1 attn layer for arg-copying |
| **QK-Norm** | Qwen3 | ✅ | ✅ | distinct from K2 MuonClip/QK-Clip |
| **WSD schedule** | MiniCPM | ⚠️ compare | ⚠️ compare | matched cosine first; decay-data mix is a separate factor |
| **Curriculum (difficulty-ordered)** | LFM2 | ✅ | ✅ | reuse the loop's per-category accuracy |
| **Depth-recurrence + factorized embeddings** | MobileLLM (kept) | ✅ | — | partially untie recurrence (2–3 blocks); PLE as capacity injection |
| **Model merging across rounds** | Arcee, LFM2.5 | ✅ | ✅ | soup → TIES |
| **μP tiny-proxy LR transfer** | MiniCPM | proxy | ✅ | verify transferred LR with one short run (recurrence breaks clean μP) |
| **MTP auxiliary head (train-only)** | DeepSeek-V3 | ❌ | ⚠️ try | drop at inference |
| Micro-MoE FFN | Switch, DeepSeekMoE, OLMoE, Kimi | ❌ default | ⚠️ matched experiment | PyTorch sparse dispatch; WebGPU promotion requires measured export/runtime gates |
| MLA / frontier quantization infrastructure / sparse long context | DeepSeek, Llama-4, Kimi | ❌ | ❌ | deployment anti-patterns at our scale |
| RL as the primary capability source | DeepSeek-R1 comparison | ⚠️ unverified | ⚠️ unverified | distillation first; executable-reward RL remains an ablation |
| Mamba/SSM/linear-attention mixers | Nemotron-H, MiniMax | ❌ | ❌ | Liquid HW-search: lose to conv+GQA on CPU |

## Application roadmap (the "improve this model" half)

1. **Distillation (now, highest ROI):** the decoupled tempered Top-K KD objective lands in `distill.py`;
   run **30M → 1M**; then add an **on-policy** pass over 1M-sampled tool-call trajectories scored by the
   30M. Measure against the current 1M (single-turn 48%, planner free-rollout 2%).
2. **Hybrid backbone tier:** a new config with **k=3 double-gated short-conv blocks + 1–2 GQA layers +
   QK-Norm**, budget-guarded; A/B vs the current decoder at 30M (CPU decode tok/s + accuracy) and a
   mostly-conv 1M variant.
3. **Training schedule:** run a matched WSD-versus-cosine pretraining comparison first. Only after
   isolating schedule should a second ablation inject curated tool-call traces in the WSD decay
   window; difficulty-curriculum ordering remains a separate flywheel treatment.
4. **Flywheel merging:** soup/TIES-merge per-round specialists instead of retraining from union data.

Each step is measured against the live baselines, not assumed — per the project rule of reporting real
numbers.

## Apply-phase results (measured — real numbers, including the nulls)

What the roadmap above actually produced when run. The project rule is no faking, so the mixed and
inconclusive results are reported as such.

| Lever | Measured result | Verdict |
|---|---|---|
| **Planner→action (1M)**, learned `plan_rollout` over a 4-round flywheel | teacher-forced next-tool **78→81%**, grounded-call **70%**, single-turn **48%** — but free-rollout whole-plan stays **~2%** | **Mechanics strong, strategy capped.** The 1M nails next-tool-given-context and grounding; autonomous multi-step planning collapses under compounding error — the capacity ceiling, cleanly shown. |
| **30M→1M Top-K distillation, distill-*then*-SFT**, controlled A/B (only the distill stage differs) | T−C: single-turn **+5.4**, free-rollout whole-plan **+7.5**, teacher-forced **0.0**, grounded **−29** (small-n, noisy) | **Modest + mixed.** Confirms the debate's own Scaler caveat: a 71% 30M teacher is too weak to transfer crisp argument-copying — and warming the backbone then SFT-ing the heads *re-specializes it away* from copying, so grounding **regressed**. |
| **30M→1M Top-K distillation, distill-*throughout*-SFT** (concurrent KD term added to `sft`), 3-arm A/B vs control & distill-then | grounded **+9.7 vs distill-then** (regression *erased*: 62.5% vs 52.8%, and +2.5 vs control); single-turn **+12.4 vs distill-then**; TF step-acc +3.1; but **vs plain control**: single-turn ~flat (−2.0), free-rollout noisy 0% | **The schedule matters; the gap still caps the gain.** Concurrent KD is *strictly better than* distill-then (recovers grounding, lifts selection) — so **if you distill at this gap, do it throughout SFT, not before.** But it does *not* cleanly beat from-scratch control on generative metrics: at a 71%→1M gap the distillation benefit is marginal, exactly as the capacity-gap caveat predicts. |
| **Hybrid short-conv + GQA + QK-Norm (30M)** vs standard decoder, equal params (29.1M vs 28.3M) | **+20% prefill, +5% decode** tok/s on CPU; accuracy 6.9% vs 11.0% at a minimal SFT-only budget — but at `n=24` that 4-pt gap is **~1 example (inconclusive)** | **Speed win real, accuracy unresolved.** The conv's O(L·k) edge over O(L²) is modest at short tool-agent contexts, as predicted; a fair accuracy ranking needs a pretrained, larger-eval A/B (currently blocked by the sandbox SIGKILLing the `batch=64` pretrain — addressable with a smaller pretrain batch). |
| **10.5M hybrid vs attention, one-TPP matched pretraining + exact-checkpoint WebGPU** | Hybrid aggregate BPB **2.0319 vs 2.0989**; 10,000 paired-document bootstraps give attention-minus-hybrid BPB **0.0669**, 95% `[0.0634, 0.0707]`. The trained hybrid clears **100 p50 wall tok/s in every run/context** (116.476–137.711), but p95 TPOT is ≤10 ms by median only at 512 tokens. | **Aligned early quality and throughput, partial tail failure.** The interval conditions on one architecture seed and 240 documents, not multi-seed uncertainty. The checkpoints are pretrain-only, the joint latency gate fails in every context under the every-run rule, and there is no agent/action evidence. ([artifacts](paper/results/m5-webgpu-cached-decode-10m-trained-proxy-20260728.summary.json)) |
| **WSD schedule** (MiniCPM), 1M, 50-step proxy, LM-only | train loss **−0.005**, single-turn **+1.4pp** | **No schedule conclusion.** The proxy is short and lacks a matched cosine arm, so neither delta can be attributed to WSD. The implementation is opt-in (`lr_schedule="wsd"` + optional `decay_samples`); a longer identical-token WSD/cosine comparison must precede any payoff claim. |
| **Curriculum** (LFM2 difficulty-ordering), 1M | single-turn **0/0 (floored)**, TF step −8.3, grounded −20 | **Within noise + a real mechanism finding.** Strict ordered passes **starve the hard tail unless `steps×batch ≥ pool size`** — the curriculum arm reached only the easiest ~25% of data in 50 steps and collapsed. Difficulty score validated (range 0.09→5.67); the lever is sound, the *budget* was too small for a full easy→hard sweep. |
| **On-policy reverse-KL** (1M←28M), 20-step smoke | reverse-KL **12.15→5.60** (first-third 11.82 → last-third 6.25); ~13.8 s/step | **Mechanism works cleanly.** Token-level full reverse-KL on the student's *own* rollouts moves it toward the teacher; exactly 0 when student==teacher. The free-rollout payoff needs a full pass + eval-harness measurement on a faster host (student sampling is ~14 s/step on this CPU). Implemented opt-in with CE/entropy/off-policy-mix stabilizers. |
| **Flywheel merging** (model-soup + TIES) | 17 tests pass; merged real 1M checkpoint loads into model+heads and evals (59.2%, sanity) | **Implemented + verified to load/eval.** Training-free weight-space combine of per-round specialists; a measured merge-vs-retrain A/B needs the round checkpoints + the slow eval harness. |

**Cross-cutting takeaway.** The architecture lever now has a coherent early result: the 10.5M hybrid
improves pretraining CE/BPB/top-1 on one matched seed and clears 100 p50 wall tokens/s across every
tested context and page run. That does not resolve architecture selection: paired-document intervals
condition on one seed/240 documents, the p95 tail gate fails, and no agent quality was measured. The
earlier 30M minimal-SFT accuracy comparison also remains inconclusive. Training levers such as
distillation and planner rollout still move the measured capability metrics most (for better and,
on grounding, for worse). Distillation at our own 30M→1M gap is *weakly* positive rather than the
frontier-scale story, exactly the capacity-gap caveat. The crisp reusable training result remains:
**distill-*throughout*-SFT strictly beats distill-*then*-SFT** (+9.7 grounding, +12.4 single-turn),
because concurrent KD avoids the backbone re-specialization caused by the two-stage schedule.

**Measurement caveat (load-bearing).** The dominant limit on these results is *this sandbox's compute*,
not the ideas: head-SFT runs ~37 s/step and greedy eval ~17 min/arm on a contended CPU, which forces
budgets down to where single-turn accuracy floors and most A/Bs land *within noise*. So the honest
deliverable of the apply phase is **the levers implemented as tested, opt-in, default-off code in the
pipeline** (Top-K KD, distill-throughout, hybrid backbone, WSD, curriculum) plus the **robust engineering
findings** (distill-throughout > distill-then; the one-seed 10.5M hybrid improves held-out
pretraining quality and clears the p50 throughput component but misses the tail gate; curriculum
needs a full-sweep budget; on-policy reverse-KL moves cleanly on the student's own rollouts) —
*not* multi-seed or agent-capability deltas, which need real budgets. **All eight adopt-verdict levers
are now implemented as tested, opt-in, default-off pipeline code** (Top-K KD, distill-throughout, hybrid
backbone + QK-Norm, WSD, curriculum, on-policy reverse-KL, model-merging) — the toolkit is complete; what
remains is measurement at real budgets on a faster host.

## Sources
Qwen3 [2505.09388](https://arxiv.org/abs/2505.09388) · DeepSeek-V3 [2412.19437](https://arxiv.org/abs/2412.19437) ·
DeepSeek-R1 [release](https://github.com/deepseek-ai/DeepSeek-R1) · Kimi-K3 [pinned official repository](https://github.com/MoonshotAI/Kimi-K3/tree/7c5be9599120d7993748de66a76128614f15f210) ·
GLM-5.2 [official report](https://z.ai/blog/glm-5.2) ·
Nemotron-H [adlr/nemotronh](https://research.nvidia.com/labs/adlr/nemotronh/) ·
Llama-4 [ai.meta.com](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) · Qwen3-Next [blog.vllm.ai](https://blog.vllm.ai/2025/09/11/qwen3-next.html) ·
MiniMax-01/M1 [2501.08313](https://arxiv.org/abs/2501.08313) / [2506.13585](https://arxiv.org/abs/2506.13585) ·
Kimi-K2 [github](https://github.com/MoonshotAI/Kimi-K2) · Mistral [2310.06825](https://arxiv.org/abs/2310.06825) ·
LFM2 [2511.23404](https://arxiv.org/html/2511.23404v1) · MobileLLM [2402.14905](https://arxiv.org/abs/2402.14905) ·
MiniCPM [2404.06395](https://arxiv.org/html/2404.06395v1) · Phi [2306.11644](https://arxiv.org/abs/2306.11644) ·
SmolLM2/SmolVLM [smollm](https://github.com/huggingface/smollm) / [2504.05299](https://arxiv.org/html/2504.05299v1) ·
Gemma-3 [2503.19786](https://arxiv.org/html/2503.19786v1) / Gemma-3n [MatFormer](https://arxiv.org/abs/2310.07707) ·
MiniLLM [2306.08543](https://arxiv.org/abs/2306.08543) · mergekit [2403.13257](https://arxiv.org/html/2403.13257v2).

*Closed models (GPT-4.5/5.x, Claude Opus, Cursor Composer, and current Grok coding models) are
deliberately omitted from architectural verdicts: no published architecture means any claimed
block-level transfer would be speculation. [Grok 4.5's official
release](https://x.ai/news/grok-4-5) does disclose aggressive curation and asynchronous
verifiable/model-graded agent RL, which informs data and evaluation discipline but not this model
topology. Where a disclosed training direction is relevant, it is already represented by
reproducible open-model evidence above.*
