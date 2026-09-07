# WebGPU training system: research decisions and reproducible recipe

This document is the implementation contract for the pretrain → midtrain → SFT → RL pipeline.
It separates ideas that transfer to a sub-100M browser model from frontier-scale infrastructure
that does not.

## 1. Architecture decision

| System | Publicly supported idea | LocalAgent decision |
|---|---|---|
| [Kimi K3](https://github.com/MoonshotAI/Kimi-K3/blob/7c5be9599120d7993748de66a76128614f15f210/k3_tech_report.pdf) | The pinned report discloses 69 KDA + 24 Gated MLA layers; 2.8T total/104B active parameters; 16/896 experts; curated web/code/math/knowledge plus vision data; Per-Head Muon; cosine with 1% warmup; progressive context; SFT → RL experts → MOPD; and MXFP4/MXFP8 QAT | Keep the *hybrid sequence-mixing*, small-model mixture-ablation, verified-agent-data, and deployment-aware-training principles. Do not call short-conv “KDA” or estimated Q4 “QAT”; exact corpus identities/weights/token count and several chosen hyperparameters remain unpublished. |
| [Kimi Linear](https://arxiv.org/abs/2510.26692) | KDA/MLA layerwise hybrid reduces KV memory at extreme context | Retain periodic full attention for verbatim copying and cheap mixers elsewhere. At the current 4K context, one KV head already makes cache cost small. |
| [Attention Residuals](https://arxiv.org/abs/2603.15031) | content-dependent aggregation across depth mitigates PreNorm dilution | Track as an ablation, not a default. It adds depth-state traffic and needs a measured browser kernel before adoption. |
| [Kimi K2](https://arxiv.org/abs/2507.20534) | MuonClip adds QK-logit clipping to Muon; large agent synthesis and joint RL use environment feedback | Treat MuonClip/QK-Clip and QK-Norm as distinct mechanisms. Keep LocalAgent's QK-Norm as an independently motivated baseline; test clipping or Muon only in isolated ablations. |
| [Kimi K2.5](https://arxiv.org/abs/2602.02276) | Continual pretraining atop K2-Base on approximately 15T mixed visual/text tokens, followed by native multimodal post-training and learned parallel Agent Swarm orchestration | Transfer explicit continual-stage accounting and verified parallel-task construction only. LocalAgent does not reproduce K2.5 vision, frontier MoE/MLA, Agent Swarm, or its parallel-agent training; the bounded Micro-MoE is an independent matched experiment, and vendor swarm latency is not a browser measurement. |
| [GLM-4.5](https://arxiv.org/abs/2508.06471) | unified agent/reasoning/code data, multistage training, environment RL | A scheduled general/code/agent midtraining mixture and executable agent tasks. |
| [GLM-5.2](https://z.ai/blog/glm-5.2) | The official release describes 1M context, one IndexShare indexer per four sparse-attention layers, MTP IndexShare/KVShare, a 128K long-context midtraining stage, and critic-assisted PPO for compacted long trajectories with anti-reward-hacking work | Transfer long-horizon data, executable evaluation, and anti-hacking controls. Do not import million-token sparse attention into a 4K browser model. Add critic-assisted online PPO only after a real long-horizon environment exists; the current bounded exact-reward stage remains explicitly offline. |
| [Grok-1](https://github.com/xai-org/grok-1) | RoPE, strongly grouped KV heads, sparse MoE | Adopt RoPE/GQA. Keep dense as the deployment control, but test an opt-in Micro-MoE with honest total/active accounting; all experts still consume browser memory and WebGPU needs a measured sparse dispatch. |
| [Grok 4.5](https://x.ai/news/grok-4-5) | xAI reports coding/science/engineering/math data, hundreds of thousands of multi-step software tasks, automated plus model grading, and asynchronous multi-hour RL; the release reports served throughput but does not publish a transferable architecture recipe | Transfer executable multi-step task construction and asynchronous rollout as later environment work. Do not infer architecture or equate frontier served throughput with a local WebGPU decode measurement. |
| [Upstage SOLAR-10.7B](https://arxiv.org/abs/2312.15166) | Depth up-scaling expands a compatible pretrained transformer by duplicating layers, then continues pretraining | The reported method starts from a larger pretrained parent and inherits its knowledge; LocalAgent trains from scratch and has no compatible parent. Treat layer duplication plus continued training only as a checkpoint-growth ablation, with matched added compute—not as knowledge transfer or a WebGPU-latency claim. |
| [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | MTP is an auxiliary training objective; its reported decode gain uses a speculative-decoding framework and accepted-token verification | MTP is optional and train-only by default. It is not free speculative decoding: keep a head only after a browser verifier, acceptance, and end-to-end latency ablation. |
| [MobileLLM](https://arxiv.org/abs/2402.14905) | deep/thin, embedding sharing, and GQA form a strong dense sub-billion baseline | Use no-MoE dense parameterization as the deployment control, not as a claim that all-attention is universally best. Compare dense attention with dense hybrid mixers. |
| [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) / [GPT-3](https://arxiv.org/abs/2005.14165) | decoder-only causal next-token pretraining, scaling, and in-context task specification | Use the dense causal decoder and frozen-prompt evaluation as baselines. Scaling evidence does not make frontier width, corpus size, or dense all-attention optimal for WebGPU. |
| [GPT-4 report](https://arxiv.org/abs/2303.08774) | predictable scaling and staged post-training; architecture/data details are not public | Do not invent unpublished GPT-4/4.5/5.x architecture details. Adopt held-out loss/eval gates and scaling experiments. |
| [nanochat speedrun](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/runs/speedrun.sh) | at the audited commit, the reference path is tokenizer → pretrain/eval → SFT/eval | Implement the same operational properties in pure PyTorch. LocalAgent retains explicit domain midtrain and optional RL because they are separate experimental questions. |
| [nanochat model/train](https://github.com/karpathy/nanochat/tree/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd) | the pinned `nanochat/gpt.py` and `scripts/base_train.py` test RoPE, QK-Norm, tiled sliding/full attention, value/residual mixing, ReLU², Muon+AdamW, and optional FP8; the audited builder uses one KV head per query head | Treat these as an experiment inventory, not a transferable bundle. Keep the simpler MQA/GQA browser controls and adopt only changes that win matched quality, memory, and exported-latency gates. |
| [nanochat RL script](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/chat_rl.py) | standalone “GRPO” is described as closer to REINFORCE, with no reference KL or PPO ratio/clipping and DAPO-style token normalization | Use it only as a minimal outcome-reward reference; do not relabel it as conventional GRPO or as part of the speedrun. |

The `webgpu-35m-hybrid` configuration is the prespecified quality-treatment candidate: roughly
34M dense parameters, 16K BPE, 12 narrow blocks, 8 short-conv mixers, 4 full GQA blocks, one KV
head, and QK-Norm. Its estimated Q4 weights are under 24 MiB; actual tokens/second must be measured
in the target browser/runtime and is never inferred from parameter count.

### Opt-in sparse-capacity pair

`webgpu-44m-moe` stores 43,862,464 parameters and selects a nominal 17,320,384-parameter path
(eight 512-wide experts, top-2). `webgpu-17m-dense-moe-control` has 17,297,344 total/active
parameters with the same width, tokenizer, nine-layer mixer topology, and 1,024 active FFN units.
The hard budget always uses the total count. The active count is not a checkpoint, memory, prefill,
or throughput claim. See [`SPARSE_EXPERTS_REAL_DATA.md`](./SPARSE_EXPERTS_REAL_DATA.md) for the
matched data, router, quality, and target-browser gates.

### Shared-paper deployment tiers

The explicit 1M/10M/near-100M experiment uses the exact paper 16K tokenizer, 2,048-token packed
training rows, and a 4,096-token model context for every tier. This choice gives up some 1M
parameter efficiency, but it removes tokenizer and validation-split confounds from the cross-tier
comparison and leaves room for schema-conditioned agent prompts:

| tier | exact LM parameters | architecture | scheduled pilot | evidence boundary |
|---|---:|---|---:|---|
| `webgpu-1m-bpe-router` | 980,480 | factorized 32→128 embeddings; two unique `[conv, attn]` blocks looped 3×; MQA | 19,628,032 tokens (20.019 TPP) | real pretrain → midtrain → base SFT + parent-anchor continuation complete; retained step 12 failed the bound development greedy gate |
| `webgpu-10m-hybrid-4k` | 10,524,544 | four unique blocks in a `[conv, attn, conv, attn]` pattern; QK-Norm; MQA | 52,756,480 tokens (5.013 TPP) | architecture-matched to the historical 2K proxy arm; this 4K paper run is new |
| `webgpu-96m-hybrid` | 95,320,448 | eighteen layers, 12 conv + 6 attention; QK-Norm; MQA | 95,322,112 tokens (1.000 TPP) | untrained feasibility arm; no browser-throughput evidence |

The 10M and 96M controls are all-attention models with parameter deltas of 0.214% and 0.023%.
Every paired arm uses the same optimizer updates, packed-row draws, tokenizer, shards, seed, and
validation schedule. The 96M pilot uses `micro_batch_size: 1` and `grad_accum_steps: 16` because
MPS currently trains in FP32 and activation checkpointing is not implemented; even that setting
must pass a one-update memory/throughput preflight before a long run.

All five backbone variants have config-complete `pretrain → midtrain → SFT → RL` lineage. The
midtrain horizons mirror the tier pilots—599 updates at 1M, 1,610 at 10M, and 2,909 at 96M—and
schedule source weights by measured input tokens. The base SFT horizons are 348/936/1,690
optimizer updates for 1M/10M/96M; the 1M lane additionally executed a separate 372-update
parent-anchor child. Exact-reward RL uses 18/48/86 rollout steps. Those horizons are
compute-scaled rather than copied from the rejected legacy 10,000-SFT/300-RL settings. Both
stages use deterministic quota-stratified, no-replacement decision order, a separately
stratified 512-conversation held-out slice, and periodic exact-resume checkpoints. Config
completeness is not execution evidence: only the 1M treatment has completed the real paper
pretrain, midtrain, base SFT, and continuation stages; each 96M arm remains gated on a measured
one-update memory/throughput preflight for every stage, and no 96M post-training checkpoint or
score is claimed.

The internal full-catalog agent scorecard renders all 50 tool names, descriptions, and complete
JSON schemas before system/history turns. The frozen paper tokenizer's longest evaluation prompt
is 3,590 tokens; with a 96-token decode allowance it requires 3,686 tokens and leaves 410 tokens
of headroom at 4K. The historical proxy tokenizer requires 3,799. All new tiers therefore use 4K
context. The legacy byte 1M, historical 2K 10M, and 2K 35M checkpoints fail this preflight
honestly and are not assigned a fabricated full-catalog score.

The [m378 one-microbatch preflight](paper/results/raw/m378-pretrain-96m-hybrid-one-microbatch-v1.json)
now exercises the exact `webgpu-96m-hybrid` model shape (`95,320,448` parameters), paper-all
packed corpus, BPE tokenizer, AdamW/WSD path, checkpoint writer, and lineage checks on CPU. It
completes one isolated update over 2,048 tokens with loss `9.799`, taking `250.27 s` and peaking at
`3.73 GiB` RSS (`8.18 input tokens/s`). This receipt uses the tracked preflight derivative with
`grad_accum_steps: 1`; production remains `grad_accum_steps: 16`, so the result proves a bounded
architecture/update path only—not production throughput, 4K WebGPU performance, or full 96M
training quality. The 10M tier remains the only deployment candidate with browser evidence.

### Executed 1M paper lane, 2026-07-29 through 2026-07-30

The 1M treatment completed 599 pretraining updates over 19,628,032 input tokens. Its fixed
validation loss fell from 9.704137 at step 0 to 5.701670 at step 598. The child midtraining stage
completed its exact 599-update plan over 21,680,586 input / 16,813,239 supervised tokens; the
same-draw aggregate held-out loss improved from 5.019692 to 4.758946 and token accuracy from
23.31% to 27.07%, with every configured source improving.

The base SFT completed its exact 348-update, 5,568-decision plan over 19,595,660 input / 164,136
assistant-loss tokens. On the sealed 512-conversation / 820-decision holdout, teacher-forced loss
improved from 6.406627 to 2.637803 and token accuracy from 4.85% to 63.30%. Sequence exactness
remained zero. Its subsequent serial greedy scorecard produced 0/820 exact actions, 0/820
format-valid outputs, 0/611 tool-name exact cases, and only 153/820 EOS terminations. This is an
internal Conversation-schema BFCL-style benchmark, not an official BFCL submission.

The base checkpoint's isolated one-update RL preflight preserved the production group size,
256-token generation budget, and two policy epochs while capping held-out evaluation to the exact
62-row
mandatory-strata coverage set. Across 32 real rollouts it observed only reward 0.0, six complete
parser-valid generations, zero strict-format-valid generations, and zero informative groups.
Consequently it performed zero optimizer updates and wrote a signed **failed** receipt. Production
RL was intentionally blocked at that parent: outcome-only group-relative optimization had no
signal.

The separate recovery child has now executed. Its MPS run completed 372 updates—348 exact
parent-replay updates plus 24 interleaved format pulses—over 5,952 decisions, 20,932,540 input
tokens, and 174,074 assistant-loss tokens. It published 31 immutable archives at 12-update
intervals. The production receipt proves integrity and accounting, not quality. A frozen
teacher-forced sweep evaluated every archive with loss-increase tolerance `1e-4` and zero
token/sequence-accuracy drop. Two of 31 archives were eligible; the fixed ranking selected step
12 (`e1cf203368f6f19a8a46f0cd2a297bd61373e44fbf24155e3d68b5137db430c7`) at loss
2.636792 and 11,687/18,460 correct assistant tokens. Step 372 lowered loss to 2.627632 but fell to
11,646 correct tokens and was ineligible. Step 24 was eligible but was not a fallback.

Candidate preparation independently replayed the sweep and sealed both development and
confirmatory scorecard configs before observing generation. Step 12 then failed the bound
development scorecard: 170/820 EOS completions, 650/820 truncations, and zero successes for
complete format, strict tool format, schema validity, case-exact tool name, whole-call exactness,
and structural abstention. The independently replayed decision is
`development_gate_failed`; confirmatory evaluation was not supplied, fallback was forbidden,
and neither candidate-bound RL nor candidate-specific WebGPU export was authorized. The earlier
readiness summary and RL receipt remain explicitly scoped to the base checkpoint.

Theoretical Q4 estimates are not deployment results. The current browser exporter is FP16-only,
so the 96M model is about 181.8 MiB of weights per graph and separate prefill/decode sessions can
duplicate that storage. Existing 34M cached-decode evidence is already below 100 tok/s; therefore
the 96M tier is not described as meeting the 100–300 tok/s target until an actual export and
browser run says so. The measured 10M tier remains the autoregressive deployment candidate.

“Kimi-like” is therefore a falsifiable transfer hypothesis, not a scale model. The treatment keeps
periodic global retrieval among cheaper mixers; the matched control is all attention. It omits
KDA, Attention Residuals, MLA, SiTU, and K3's quantization-aware recipe until each has an
exportable WebGPU implementation and a measured benefit at 4K context. Micro-MoE is now an
opt-in PyTorch experiment, not part of the deployed hybrid treatment or a browser-speed claim.
The official K3 repository
now gives the 69-KDA/24-Gated-MLA layer composition, major dimensions, data domains, optimizer,
schedule, context curriculum, and SFT/RL/MOPD sequence. The exact source inventory, mixture
weights, token count, selected peak learning rate/batch/TPP values, and a directly runnable
end-to-end recipe are still not public; LocalAgent does not fill those narrower gaps with
third-party inference.

The default no-MoE implementation remains “dense” for deployment simplicity, while both the mixer
and sparse-capacity choice are experimental axes. Dense all-attention is the mixer control rather
than a universal winner; the 17.30M dense FFN is the active-matched control for the 43.86M-total
Micro-MoE. Neither comparison substitutes parameter arithmetic for measured target-device results.

The first matched hardware gate now supports keeping the hybrid as the treatment: across three
Apple M5 Chrome 150 page/session runs, its random-weight hidden-only graph reduced warm p95
`session.run` latency versus the 0.223%-larger all-attention control by a median paired 1.58× at
512 tokens, 1.87× at 1,024, and 2.00× at 1,536. Absolute latency varied materially across runs, so
the artifact reports run medians and ranges rather than pooling them. This is a latency result, not
a capability result. ONNX Runtime Web 1.27.0 accepted an exact WebGPU-only session request and
exposed a WebGPU device, but did not expose adapter identity or per-node placement; fallback status
is therefore unknown. See the
[tracked result summary](paper/results/m5-webgpu-backbone-20260728.summary.json). The clean 10M
quality replication supported the bounded hybrid post-training pilot below, whose internally
prespecified
pre-assistant-padding action gate failed. The 34M five-TPP architecture screen remains unrun; it
can only promote a candidate to the subsequent 20-TPP/downstream quality comparison, and a
corrected trained-action gate must also pass.

The cache-bearing latency gate is also complete for three untrained matched pairs. Its exporter
produces separate prefill and fixed-`T=1` decode ONNX graphs, then hard-gates each precision over
multiple prompt lengths and iterative steps. Greedy token IDs agree exactly for ONNX versus
PyTorch cached execution and for PyTorch cached execution versus a fresh full-context reference;
cache values must stay within the declared fp16/fp32 tolerance. The browser runs use one prefill
pass plus 31 iterative decode calls per sample.

Median-of-three within-run p50 wall-decode rates for the hybrid arms were:

| pair | 128 | 512 | 1,024 | 1,536 | 100 tok/s reference |
|---|---:|---:|---:|---:|---|
| [34.2M](paper/results/m5-webgpu-cached-decode-20260728.summary.json) | 74.05 | 64.54 | 57.53 | 47.15 | misses all |
| [15.6M](paper/results/m5-webgpu-cached-decode-16m-20260728.summary.json) | 90.87 | 89.26 | 80.94 | 79.77 | misses all |
| [10.5M](paper/results/m5-webgpu-cached-decode-10m-20260728.summary.json) | 159.23 | 160.46 | 143.49 | 127.57 | clears all |

The 34.2M hybrid was faster than its matched attention control at all four contexts but remained
below the engineering reference. The 10.5M result clears that reference, but the reference is only
a latency screen—not a quality threshold or complete-action SLO. Configs and all raw runs are
linked from the [result index](paper/results/README.md).

The deployment tracks therefore separate. For autoregressive generation, a 10M-class full-budget
model is the current latency-feasible candidate. A 34M model can remain in the scientific screen
or qualify for one-forward structured actions only if its trained complete policy meets the
absolute TTFA and `Success@B` gates; relative speed over attention is insufficient.

For WebGPU, the runner requested and observed `gpu-buffer` present-cache tensors, rebound them as
the next call's past inputs without reading cache data into JavaScript, and disposed superseded
tensors. This does not establish physical GPU placement: adapter identity, per-node placement, and
per-node fallback remain unknown. The graphs return fresh presents on every step—attention grows
K/V through append/concat and short-conv returns a fresh fixed-width tail—so this is not in-place
or paged cache management.

The primary browser runner now has a lazy, hash- and lineage-validated consumer for the same
prefill plus fixed-one-token cache ABI. Its two autoregressive controls report
`decode_cache: true` and `prefill_then_kv_cached_decode`, rebind present tensors as the next
decode inputs, and keep the structured one-forward policy separately labeled. This is an
implemented deployment path, not a completed browser result. The original 348-update 1M SFT
checkpoint—not the retained continuation step 12—has been exported into a deterministic
single-model fp32/fp16 cached bundle whose iterative PyTorch/ONNX token IDs agree exactly and
whose cache/logit errors remain within the declared tolerances. The browser harness has not yet
produced a fresh-session WebGPU measurement for that bundle, and its separate capability score is
zero. Step 12 failed the development gate and was therefore not exported or benchmarked. No
100–300 tok/s or full-catalog cached-action WebGPU claim is made.

## 2. Dataset contract

Pretraining is self-supervised next-token prediction over broad text and code. It should teach the
base model how language, programs, and structured documents work before the model is asked to act
like an assistant. Instruction dialogues, canonical tool calls, environment trajectories, and
preference rewards are therefore not substitutes for a base corpus; they are concentrated in
midtraining, SFT, and RL after the base model has useful priors.

Every raw document must retain a stable `doc_id`, source URI/path, license, and free-form metadata.
`scripts/prepare_corpus.py` produces:

1. `filtered.jsonl` — readable accepted documents and provenance;
2. `corpus-staging.sqlite3` — the disk-backed hygiene, near-dedup, and split index;
3. `generations/<uuid>/train-*.npy` and `val-*.npy` — immutable memory-mappable fixed rows;
4. matching `*.lengths.npy` — exact masks so padding never contributes to loss;
5. atomically committed `manifest.json` — active generation, split, vocabulary, counts, sizes,
   SHA-256 checksums, and shard names.

Filtering is deterministic. It rejects bad lengths, control-character corruption, replacement
character corruption, repetitive boilerplate, and exact normalized duplicates. It intentionally
does **not** use an English alphabetic-ratio filter because source code, JSON schemas, and tool
traces are core data.

Train/validation assignment happens at document level after deduplication. It uses a stable seeded
hash, so a document cannot appear in both splits and re-running preparation produces the same
assignment.

The CLI uses SQLite staging by default. Exact-dedup winners, denylist decisions, bounded
SimHash-LSH buckets, and split membership stay on disk; BPE training reads only the staged training
split, then packing streams the same frozen assignments. The manifest fingerprints the raw
document stream, staging database, filtered JSONL, optional upstream download manifests, split
assignments, and tokenizer-training document set. This removes the former full-corpus Python lists,
but it is not a constant-resource claim: memory still includes one quality-bounded document, the
denylist, tokenizer trainer state, the bounded near-dedup cache/candidate set, and one shard buffer.
Plan local scratch space for the raw JSONL, SQLite text/indexes, filtered JSONL, and token shards;
peak disk can be several times the source size. Near-dedup remains explicitly non-exhaustive.
The staging v2 schema retains removed exact-duplicate provenance in an aliases-only SQLite table
(source, license, metadata, occurrence count, and raw-text hash without another text copy).
Case-sensitive code-like rows bypass heuristic near-dedup after exact normalized dedup so
identifier case changes are not collapsed. Local JSONL/plain ingestion rejects records above the
configurable 8 MiB raw guard before decoding; already-materialized upstream dataset rows still
depend on their source/transport limits. Immutable old shard generations are retained until a
separate garbage-collection policy is implemented.

### Recommended capability mix

Start with a measured ablation rather than treating percentages as universal constants:

| stream | pretrain | midtrain start → end | purpose |
|---|---:|---:|---|
| curated general/educational text | 45–55% | 55% → 35% | language and broad priors |
| permissively licensed code + tests + docs | 25–35% | 30% → 35% | syntax, repository vocabulary, verification habits |
| structured data/API schemas/configs | 10–15% | included with code | exact structure and key/value binding |
| rule-audited synthetic agent conversations/trajectories | 5–15% | 15% → 30% | tools, abstention, multi-turn state, recovery |

For a 34M model, 20 tokens per parameter is a reasonable first full pretraining run (~680M
tokens), followed by a 5–15% midtraining continuation. This is a starting point, not a claim of
compute optimality: run token-budget sweeps and select on held-out bits-per-byte plus agent/code
evaluations.

### Practical Colab mixture

`configs/data/pretrain-colab.yaml` is a reproducible first-run preset, not a universal optimum:

The weights below allocate the bounded download by UTF-8 character count. They are not guaranteed
token shares; the packed manifest must report realized final-token counts by source for the paper.

| source | character-budget weight | why it is present |
|---|---:|---|
| [SmolLM FineWeb-Edu-Dedup](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | 65% | filtered, near-deduplicated educational web text; ODC-By corpus |
| [Cosmopedia v2](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | 15% | synthetic textbook-style explanations that are sample-efficient for small models |
| [CodeParrot Clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean) | 20% | deduplicated Python; the downloader accepts only configured permissive per-file licenses |

The downloader streams a bounded sample rather than mirroring the full corpora:

```bash
python scripts/download_pretrain_mixture.py configs/data/pretrain-colab.yaml \
  --out data/raw/colab --target-chars 120000000
```

It writes `mixture.jsonl` plus `download_manifest.json`, including per-source requested/accepted
sizes, skipped-license counts, provenance, and the mixture-config checksum. Character count is only
a download/storage bound; `prepare_corpus.py` records the exact token count after tokenization.
Before opening any dataset stream, the downloader constructs a path-independent, self-hashed
acquisition plan. Integer source budgets use deterministic largest-remainder apportionment and sum
to the exact requested character total. Duplicate source names, floating revisions, ambiguous
fixed-vs-row-level licensing, empty license allow-lists, and unknown licenses fail before network
access.

The paper config additionally requires local, checksum-matching copies of three revision-bound
dataset cards. A dry run opens no dataset streams and reports missing evidence or disk headroom:

```bash
PYTHONPATH=src python scripts/download_pretrain_mixture.py configs/data/pretrain-paper.yaml \
  --out data/raw/paper-v5 --dry-run \
  --license-evidence smollm-card=data/provenance/paper/smollm-card.md \
  --license-evidence codeparrot-card=data/provenance/paper/codeparrot-card.md \
  --license-evidence websight-card=data/provenance/paper/websight-card.md \
  --plan-out data/provenance/paper/acquisition-preflight-v5.json
```

The expected evidence IDs are `smollm-card`, `codeparrot-card`, and `websight-card`. Archive the
exact `license_evidence.url` bytes from the config under durable experiment storage, then supply
each as `--license-evidence ID=PATH` to both dry-run and acquisition commands. The pinned byte
sizes and SHA-256 values are checked locally. Passing a current dataset card or an unverified URL
is insufficient.

Paper acquisition fails before streaming unless at least 30.8 GB is free on the output
filesystem, caps the combined raw JSONL at 13.2 GB, and requires every configured source to fill
its exact apportioned character budget. These are conservative acquisition guards, not estimates
of training-token count. Completed sources are atomically spooled under `download_state/`.
After interruption, rerun the identical command with `--resume`: checksum-verified completed
sources are reused, but the partial source is replayed from its pinned revision. This is
source-boundary recovery, not a false claim of row-exact Hugging Face `IterableDataset` resume.
Changing the config, target, source order, revision, license policy, or evidence changes the plan
hash and invalidates old state.

For sources with a fixed YAML `license` rather than a row-level `license_field`, that manifest
records the dataset-distribution license asserted by the frozen config and upstream dataset card;
it is not a document-level audit of underlying web-content rights. A paper run must archive the
exact dataset-card/license evidence and revision alongside the manifest and describe this boundary.

NVIDIA ClimbMix is attractive for research comparisons, but the official release is GPT-2-tokenized
and CC BY-NC 4.0. It is not the default here because LocalAgent needs a trainable 16K tokenizer and
a path that does not quietly impose noncommercial-only use. The Stack v2/Python-Edu path is also
valuable at scale, but its content retrieval and governance workflow are poorly matched to a short
Colab session. These can be later controlled ablations, with their terms followed explicitly.
In particular, nanochat's raw-text
[`karpathy/climbmix-400b-shuffle`](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle)
repack currently carries an MIT card tag, while it describes itself as a no-content-change
redistribution of NVIDIA
[`Nemotron-ClimbMix`](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix), whose official
terms are CC BY-NC 4.0/research-only. The repack metadata does not erase upstream restrictions.
Treat this as a license conflict, not permission laundering; neither source enters the default
trainable corpus.

### Observed exploratory seed-2026 10.5M one-TPP proxy

The bounded proxy was materialized on 2026-07-28, then used for an observed exploratory matched
seed. It is not the paper corpus. The raw mixture contains 120,014,016 accepted characters in
24,125 documents
(`mixture.jsonl` SHA-256
`22d4270ad6157a9701e86be8bfd73a4fc9c480dd2cfd82337a4d6a5218183e6c`): 78,002,243
FineWeb-Edu-Dedup characters, 18,001,053 Cosmopedia-v2 characters, and 24,010,720 permissively
licensed Python characters. Local quality filtering retained 24,004 documents. A train-only 16K
BPE tokenizer (SHA-256
`8365405524329487aea3b087cc999db887d8276115e67e88ebfcf7901b15617c`) produced 28,045,897
training tokens and 287,995 packed validation tokens from a fixed 23,764/240-document split.

The local denylist contained 4,854 entries from the frozen synthetic-agent and two local browser
suites and removed no documents. Its matching is bounded and non-exhaustive. BFCL, Mind2Web,
WebLINX, and BrowserGym exports were not supplied, and the mixture contains no structured-data
source, so this corpus cannot satisfy the paper contamination or capability contract.

Both 10.5M arms used seed 2026, AdamW, the same WSD schedule and draws, 322 updates, and exactly
10,551,291 loss tokens. The all-attention model has 10,547,072 parameters and the hybrid has
10,524,544 (0.214% fewer). Because both arms use WSD, this experiment provides no WSD-versus-cosine
evidence. Frozen all-validation-document scorecards are:

| Slice | Attention CE / BPB / top-1 | Hybrid CE / BPB / top-1 | Hybrid change |
|---|---:|---:|---:|
| Aggregate, 240 docs | 6.0547 / 2.0989 / 15.24% | 5.8617 / 2.0319 / 17.28% | CE −0.1930; BPB −0.0669; +2.04 pp |
| General, 216 docs | 6.1901 / 2.0785 / 14.39% | 6.0043 / 2.0161 / 16.23% | CE −0.1859; BPB −0.0624; +1.84 pp |
| Code, 24 docs | 5.3951 / 2.2204 / 19.39% | 5.1673 / 2.1266 / 22.42% | CE −0.2278; BPB −0.0937; +3.03 pp |

Ten thousand paired nonparametric document bootstraps give aggregate 95% intervals for
attention-minus-hybrid of `[0.1835, 0.2034]` CE, `[0.0634, 0.0707]` BPB, and
`[-2.271, -1.834]` top-1 percentage points. The intervals exclude zero for aggregate, general,
and code, but they condition on this single architecture seed and these 240 held-out documents.
They are not multi-seed architecture uncertainty.

Checkpoint SHA-256 values are
`b86929f708b0294ff305fa9ffbfa5059e04a807facfc0c5c55d64c471215f4a9` (attention) and
`00dd2cf6651b0a27e18d707d287b464361e4f0636c7c787fafc7570682ab2e6d` (hybrid). The
[tracked proxy summary](paper/results/webgpu-proxy-1tpp-10m-seed2026.summary.json) binds the
corpus, configs, checkpoints, scorecards, and limitations.

### Clean prospective confirmatory seeds 2027–2029

Three prospectively designated training seeds repeated the matched 10.5M comparison against the
same frozen validation-document set. The primary difference is attention minus hybrid BPB, so a
positive value favors hybrid:

| Training seed | Attention BPB | Hybrid BPB | Attention − hybrid BPB |
|---:|---:|---:|---:|
| 2027 | 2.103014998958501 | 2.032179123685033 | +0.07083587527346794 |
| 2028 | 2.104777477117417 | 2.031249477749292 | +0.0735279993681252 |
| 2029 | 2.0996846011662638 | 2.0251960502934447 | +0.07448855087281897 |
| Mean | 2.102492359080727 | 2.029541550575923 | +0.07295080850480402 |

Hybrid was favored in all three seeds. A model-based Student-t interval with two degrees of
freedom is `[0.06824707441091234, 0.07765454259869571]`; it assumes approximately normal seed
effects, an assumption that cannot be assessed with three seeds. The assumption-free exact sign
test is correspondingly low-power: one-sided `p=0.125`, two-sided `p=0.25`. The mean
attention-minus-hybrid CE gap is `+0.2104463773820182`, and the mean top-1 accuracy gap is
`-0.021929778813112435`, equivalent to a 2.193 percentage-point hybrid advantage. General and
code BPB also favor hybrid 3/3, with mean gaps of `+0.06813601214548147` and
`+0.10169832878279628`.

All confirmatory scorecards explicitly ran on CPU in fp32. The six training configs used
`device: auto` and `dtype: auto`, while the runner did not persist the resolved training device.
The current sandbox reports MPS unavailable and resolves `auto` to CPU, but that present-host
observation is not retrospective proof of the confirmatory training device. The
[confirmatory summary](paper/results/webgpu-proxy-1tpp-10m-seeds2027-2029.summary.json) and
[raw scorecard bundle](paper/results/raw/pretrain-proxy-seeds2027-2029/) preserve every seed
estimate, checkpoint/config binding, per-document scorecard, and paired comparison.

This clean set reproduces the direction of the bounded 10M pretraining-quality effect across all
three seeds, subject to the small-seed inference limits above. It provisionally selects the hybrid
for a bounded post-training pilot only; it does not complete the 34M
at-least-five-token-per-parameter screen or subsequent 20-TPP/downstream selection and, by itself,
provides no action, agent, or
browser-task result.

### Exploratory seed-2026 trained pretrain-only WebGPU cache result

The exact seed-2026 pretrain-only checkpoints above were exported as parity-gated fp16
cache-bearing graphs and measured on one Apple M5 in Chrome 150 with ONNX Runtime Web 1.27.0.
Three fresh page/session runs retained 720 measurements after 72 warmups:

| Input tokens | Hybrid p50 wall tokens/s | Hybrid p95 TPOT (ms) |
|---:|---:|---:|
| 128 | 137.711 | 10.305 |
| 512 | 128.232 | 9.610 |
| 1,024 | 123.779 | 10.214 |
| 1,536 | 116.476 | 10.205 |

The hybrid exceeded 100 p50 wall tokens/s at every context in every page run. The joint engineering
gate—p50 at least 100 tokens/s **and** p95 TPOT at most 10 ms—still failed: only 512 tokens passed
by the median-of-run statistic, and no context passed both thresholds in every run. The
[trained-latency summary](paper/results/m5-webgpu-cached-decode-10m-trained-proxy-20260728.summary.json)
binds the exact checkpoints, graphs, runtime, and the three raw payloads
([run 1](paper/results/raw/m5-webgpu-cached-decode-10m-trained-proxy-20260728-run1.json),
[run 2](paper/results/raw/m5-webgpu-cached-decode-10m-trained-proxy-20260728-run2.json),
[run 3](paper/results/raw/m5-webgpu-cached-decode-10m-trained-proxy-20260728-run3.json)).

This browser result remains a seed-2026, approximately-one-token-per-parameter checkpoint result,
not confirmatory seed-set latency or an architecture selection. The browser payload is
latency-only; held-out language quality is linked separately by checkpoint hash. There is no
midtraining, SFT, RL, structured-data source, tool-use, complete-action, browser-task, or
agent-quality evidence. Coverage is one M5/browser/runtime, and the exact WebGPU session request
does not reveal adapter identity or per-node placement/fallback.

## 3. Stage gates

### Pretraining

- Full next-token loss.
- BPE or byte tokenizer trained only on the training corpus.
- AdamW baseline first; compare Muon only in an isolated optimizer ablation.
- Gradient accumulation, document-held-out validation, token counts, and resume checkpoints.
- Gate: validation loss decreases without spikes; no train/validation document overlap.

The optional SOLAR-style depth-growth ablation is explicit and narrow:

```bash
uv run --frozen localagent grow-checkpoint runs/base/latest.pt configs/model/deeper.yaml \
  runs/growth/deeper-init.pt --layer-map 0:0,1:0,2:1
```

Every target layer must name one source layer of the same block kind. The resolved configs must
match on every field except name, depth, and the explicit layer-type sequence; both checkpoints
must use the same content-bound tokenizer. The command verifies a complete config schema, source
lineage, source/target tensor contracts, and a self-hashed growth manifest. It rejects structured
SFT/RL auxiliary state rather than silently carrying or dropping stale heads, publishes
deterministic checkpoint bytes without clobbering a concurrent creator, and records every
discarded top-level payload key. Repeating residual blocks is **not function-preserving**.
Top-level pretraining `init_from` therefore loads only the verified target-shaped weights and
starts a fresh optimizer, while `runtime.resume` requires the complete optimizer, progress,
accounting, Python/Torch/device RNG, and applicable scaler state. Exact pretraining resume is
supported on CPU, CUDA, and MPS; unsupported accelerator backends still fail closed. On MPS, the
runner synchronizes queued optimizer writes before checkpointing, captures/restores the backend
RNG state, and rejects execution-identity or backend-state drift. A live PyTorch-2.13/M5
regression deliberately advances MPS RNG after the periodic checkpoint, interrupts the hybrid
conv--attention run, and obtains bit-identical resumed loss history and parameters. This is
same-backend evidence, not a claim that checkpoint continuations are identical across devices or
PyTorch builds. Depth growth and exact resume remain different operations; neither the
implementation nor this document claims that depth growth improves quality or browser latency
before a matched continued-pretraining ablation is run.

### Midtraining

- Continue from the base checkpoint.
- Mix general/code shards with canonical `Conversation` rows.
- Conversation rows use assistant-only masks; text/code rows use full-token loss.
- Change source weights gradually. Compare WSD with a same-hyperparameter cosine control before
  attributing a local schedule effect, then tune peak learning rate and batch independently for
  each schedule before choosing a production envelope. Only then test whether putting the cleanest
  domain mix in WSD's decay window helps.
- Gate: agent/code scores rise without an unacceptable regression in general validation loss.

The one-pass production fanout is complete. Its three split-assignment streams form an independently
verified, document-disjoint exact union of all 504,010 parent records while preserving every
document/content/split binding. General, code, and structured views contain 320,850, 51,788, and
131,372 documents; their manifest SHA-256 values are
`9eec204d170812021e43e8094de4d107fbbdcb50b67a08a32479f60c75b3867c`,
`39537e463d82785be2e703f76915d36625d72747c5191f4c0c537973a65900e2`, and
`43663ebe357aab0bbc56fed49eac1a5b6574d7a96382e1049b6d5b6087dd8d1f`.

For the historical 35M legacy-prompt pair, the first real planner replay rejected the old
2,500-update ceiling: it contained only 7,429,270 supervised tokens. Both corrected 25,000-update
plans now replay-verify 85,536,552 input and 74,836,551 loss tokens, inside the 70–100M target.
Hybrid file/self hashes are
`41a1bc6500e820b4589da3d3a63cea530d650f09d25cfee24231fba107f9af2c` /
`39e7dc0c17adc9cf46cb541f6eb5531195fede79a5bca47dce42fa9b55ca4a78`;
attention hashes are
`ae94548b7b482be8f5c8ef284624ff76514149acebe7ea837037b9ef52471512` /
`f52d8446ee70c144b4c406ddbd0656df99deab7d82f112f18376c1057db79cb4`.
These are deterministic plans, not trained-model results.

### SFT

- Use rule-audited targets for static SFT and executed outcomes for environment-derived targets.
- Balance single, parallel, sequential, multi-turn, abstention, wrong-tool distractors, malformed
  tool results, and recovery.
- Keep declared train/eval slot pools disjoint and fail on exact rendered-prompt overlap. Shared
  schema enums, no-argument intents, and template vocabulary must be reported rather than called
  value- or template-disjoint.
- Train the language backbone with tool-selection and pointer/copy heads when used.
- Gate: exact AST/execution score, abstention, wrong-tool avoidance, and whole-episode success.

`localagent synth configs/data/agent_synth.yaml` and its frozen
`agent_synth_eval.yaml` companion write canonical `Conversation` JSONL plus manifests. Their
generation contracts match while seeds, outputs, declared slot pools, and exact rendered prompts
are disjoint. The training config also pins the action and DOM suite bytes/hashes and excludes
their 21 unique canonical queries from every training user turn. Canonical equality applies Unicode
NFKC, case folding, and Unicode-whitespace collapse. This is prompt-equality exclusion, not a claim
that shared primitive values or template vocabulary are absent.
The deterministic baseline enforces the configured multi-turn and irrelevance quotas;
the regular single-turn/parallel mix is registry-driven and reported as measured structural counts,
not quota-controlled. It validates every tool name and required typed argument against its schema
and refuses to claim model-based verification unless a real verifier adapter exists. Its failure
and success tool responses are template literals, not executed results: row provenance explicitly
sets `verified=false`, `rule_verified=true`, and `environment_executed=false`.

The production paper lane preserves the frozen v1 5,000-row evaluation artifact but replaces the
legacy 5,000-row training artifact with train-v2: 50,000 rule-audited `Conversation` rows in
`data/synth/agent_sft_paper_train_v2.jsonl`. The JSONL is 526,494,339 bytes with SHA-256
`233f4f2d796568097897c73d4547a0129e73a8509981a308600779e3cb4cc060`; its 6,341-byte sidecar has
file SHA-256 `9d415aef41a1557d4dd16339fcde94d6dff5fcf6fec121372e5cfe3f1875f383`
and self-hash `e5b9d66c7761fb6d9f731e4fab2aa5b316d4714911ab7d00d89a9cbe1bd36243`.
The 1,600-byte generator config SHA-256 is
`2f03b929507e49f7f73a50e125c144fdd09efa6989306ad0e3c0d03beabc6dbe`.
The frozen eval-v1 JSONL remains 50,859,313 bytes with SHA-256
`e2c2406865a076f21ca1dd8747187ae2c2a2af0c06888b207b702aa6f2ebfb07`;
its 3,834-byte sidecar file SHA-256/self-hash are
`299152ffa32e7b358e59de37fbf127461d31dd3e1b0e782b4a1fca9600553bdd` and
`3e923355316dc222ab5fcee46e205faef388fab245827fad8444cfb9d43dd10b`.

Strict loading found zero semantic-row and zero rendered-prefix overlap between train-v2 and
eval-v1. The verified loader reads each JSONL exactly once through a descriptor-bound regular
file, checks its declared byte count, row count, and SHA-256 while parsing, interns identical tool
catalogs, and returns a recursively read-only `Conversation` graph. It does not retain a raw-file
copy or use `splitlines()`. On the 526,494,339-byte train-v2 artifact, the measured load retained
one shared 50-tool catalog and peaked at 389,742,592 bytes RSS; this is observed host evidence, not
a constant-memory claim.

The new 4K tier uses `openai_full_catalog_v1`: canonical OpenAI-style function definitions for all
50 tools, a literal EOS boundary, the exact system/user/tool/prior-assistant history, and the
current assistant marker. It emits one training row per assistant decision, masks the complete
prompt, learns only that decision's body plus EOS, and forbids truncation. Eleven framing markers
are rejected recursively from user-controlled message text, tool responses, names, descriptions,
arguments, and JSON-schema keys/values, preventing a row from injecting a false role, catalog,
tool-result, or early-EOS boundary.

The exact full-contract audit has 93,504 training decisions and 7,963 held-out decisions with zero
semantic-row and zero rendered-prompt overlap. The one shared train catalog plus EOS is 3,441
tokens (3,390 for the frozen eval catalog). Training prompt lengths are min 3,446 / p95 3,623 /
max 3,664, and the longest prompt-plus-gold row is 3,685 tokens. Held-out prompt lengths are
min 3,396 / p95 3,536 / max 3,590, and the longest complete row is 3,604. A 256-token RL reserve
therefore requires at most 3,920 tokens and fits the 4,096-token models without truncation.

For comparison, the historical 35M configs retain the legacy conversation prompt contract. Their
one-pass 16K-BPE audit counted 3,633,959 input / 2,191,728 loss tokens for train and 280,949 /
179,607 for eval; maximum rendered lengths were 244 and 214 tokens. Their two deterministic
10,000-step SFT plans each schedule 160,000 draws, 11,629,065 input tokens, and 7,012,269 loss
tokens. The loss-token total lies within the prespecified 5–20M assistant-output-token target. The
legacy hybrid plan file SHA-256/self-hash are
`605c418c338c35c02e0947ae9d063dacf81ce80d0b5ce26c9b3979d5c88681e2` and
`f72ac3ff9e1e8866c5437e831f0b3b9340ee68e669bd3278917b21bc4960b286`; the attention identities
are `5b6594e70aad40400affcf9cdace07dc3e77863564a267060af1a34768a7aab3` and
`f8e27c4576ec1ed395804fc08cfa2fd6a5835203140710c5f8b58e38f76ff64c`. Both artifacts were
independently replay-verified exactly with `plan_stage_budget.py --verify`. This establishes
deterministic plan reproduction, not SFT execution or performance.

### RL simulation

- Roll out in deterministic sandboxes.
- Reward task outcome/AST correctness, not matching one reference trajectory.
- Use group-relative advantages, clipped old-policy ratios, reference KL, and a small valid-format
  shaping reward. Drop shaping once exact rewards become dense enough.
- Include tool failures, stale state, retries, partial completion, and irreversible-action
  abstention in the environments.
- Gate: improvement over SFT on frozen environments with no regression on safety/restraint sets.

The currently wired `canonical_toolcalls` GRPO stage is narrower: it optimizes only the
autoregressive LM with exact normalized tool-AST rewards and exact text-match rewards. Because
changing the backbone invalidates route/selector/pointer calibration, its checkpoint deliberately
drops inherited structured heads instead of silently carrying stale tensors. Use the SFT
checkpoint for the structured WebGPU arm; retrain/recalibrate heads after RL before any future
structured-RL export. The runner requires an explicit `data.eval_conversations` input, rejects
canonical-row or rendered-prompt overlap with RL training data, and records deterministic greedy
pre/post exact-match, reward, schema-format, and split-fingerprint metrics. Full conversation
artifacts and the exactly scored single-turn row/prompt projections receive separate fingerprints.
The runner validates the parent tokenizer and complete stage lineage, rejects gold outputs that do
not fit the configured decoding budget, and reports attempted rollouts separately from informative
groups and realized optimizer updates. These gates make an offline RL-over-SFT comparison
auditable; they do not turn reference-output training into environment RL or establish improvement
before a real run.

The existing production RL configs bind the same train-v2/eval-v1 pair and the rejected base SFT
checkpoint. Their strict load found longest gold outputs of 62 training and 56 evaluation tokens
against `max_new_tokens: 256`, so no gold target is truncated. This is historical input-contract
evidence only. No candidate-bound RL config, preflight, rollout, or reward result exists because
step 12 failed before authorization.

Distillation-first does not mean RL is impossible. The
[DeepSeek-R1 release](https://github.com/deepseek-ai/DeepSeek-R1) reports stronger small-model
reasoning from distillation than from its compared small-model RL setup, while also demonstrating
RL-induced reasoning at frontier scale. LocalAgent therefore prioritizes verified sequence and
trajectory transfer under a small compute budget, then keeps executable-reward RL as a separately
gated last-mile experiment.

Pretraining, midtraining, and SFT now write realized input-token and supervised/loss-token
accounting rather than inferring budgets from update counts. Midtraining additionally records
per-source realized tokens and can resume optimizer/RNG/accounting state from periodic atomic
checkpoints. Pretraining persists its complete validation history and evaluation RNG state.
Midtraining evaluates frozen per-source inputs on identical deterministic draws before and after
the stage, reporting teacher-forced loss and token accuracy. Independently packed train/eval roots
must prove disjoint document IDs and content hashes through verified split-assignment artifacts or
the run fails closed. SFT likewise reports held-out assistant loss, token accuracy, and
teacher-forced all-assistant-token exactness before and after training; the last metric is not
free-running generation. Its overlap audit covers both the main and optional decay training
corpora. Stage lineage
hashes the canonical config, model config, data artifacts, tokenizer, parent checkpoint, Git
commit, and code worktree; every implemented stage rejects mismatched resume lineage. SFT and RL
write atomic checkpoints only after complete optimizer/rollout boundaries and restore the model,
optimizer, optional scaler, backend RNG, deterministic data/prompt schedule, accounting, history,
and applicable SFT heads or frozen RL reference policy. Resume is an exact continuation of one
fixed training contract; it rejects missing or drifted sealed state rather than restarting.
Held-out metrics are auditable stage evidence, not automatic promotion. The parent-anchor
continuation froze its teacher-forced thresholds before training: mean-loss increase at most
`1e-4` and zero token/sequence-accuracy drop. Its bound greedy promotion gate separately required
at least 90% EOS completion, at most 10% truncation, and both one success and 5% accuracy for each
required format/tool/schema/abstention metric. These thresholds are specific to this lane; they do
not retroactively promote older checkpoints.

## 4. Reproducible commands

```bash
# Public-domain smoke corpus, byte tokenizer
python scripts/prepare_corpus.py --sample --out data/shards/sample \
  --seq-len 128 --rows-per-shard 64 --val-fraction 0.1
uv run --frozen localagent train pretrain configs/train/pretrain-speedrun.yaml

# Browser-tier BPE corpus
python scripts/prepare_corpus.py data/raw/general data/raw/code \
  --tokenizer bpe --vocab-size 16384 --tokenizer-path data/tokenizer-16k.json \
  --out data/shards/webgpu --seq-len 2048
uv run --frozen localagent model-info configs/model/webgpu-35m-hybrid.yaml

# Before a production tier run, execute exactly one isolated optimizer update on the intended
# device. The derived config disables resume/evaluation, writes only under /private/tmp, and the
# self-hashed receipt proves that the production checkpoint and source configs stayed untouched.
python scripts/preflight_training_update.py \
  configs/train/pretrain-paper-tier-1m.yaml \
  --device mps \
  --work-dir /private/tmp/localagent-preflight-pretrain-tier-1m-20260729 \
  --out data/provenance/paper/preflights/pretrain-paper-tier-1m-mps.json

# Freeze rule-audited synthetic train/eval artifacts first: the eval artifact is one of the
# config-owned pretraining/tokenizer denylist inputs. Outcomes are not environment-executed.
uv run --frozen localagent synth configs/data/agent_synth.yaml
uv run --frozen localagent synth configs/data/agent_synth_eval.yaml
# Production post-training train-v2; eval-v1 above stays frozen.
uv run --frozen localagent synth configs/data/agent_synth_paper_train_v2.yaml

# Paper-scale mixture: download first, then stage locally with frozen eval exclusions.
# PAPER_EVAL_DENYLIST_MANIFEST must be the provenance-bound four-external-suite manifest described
# below. The three local suites are supplied directly and checked against config-owned hashes.
: "${PAPER_EVAL_DENYLIST_MANIFEST:?set the verified paper denylist manifest path}"
# Archive the three revision-bound cards named by `license_evidence` in the data config.
# This dry run opens no corpus streams and writes timestampable plan/readiness evidence.
PYTHONPATH=src python scripts/download_pretrain_mixture.py configs/data/pretrain-paper.yaml \
  --out data/raw/paper-v5 --dry-run \
  --license-evidence smollm-card=data/provenance/paper/smollm-card.md \
  --license-evidence codeparrot-card=data/provenance/paper/codeparrot-card.md \
  --license-evidence websight-card=data/provenance/paper/websight-card.md \
  --plan-out data/provenance/paper/acquisition-preflight-v5.json
PYTHONPATH=src python scripts/download_pretrain_mixture.py configs/data/pretrain-paper.yaml \
  --out data/raw/paper-v5 \
  --license-evidence smollm-card=data/provenance/paper/smollm-card.md \
  --license-evidence codeparrot-card=data/provenance/paper/codeparrot-card.md \
  --license-evidence websight-card=data/provenance/paper/websight-card.md
# If interrupted, repeat the acquisition command with `--resume`. Completed source spools are
# verified and reused; the partial source is replayed from its immutable revision.
PYTHONPATH=src python scripts/prepare_corpus.py data/raw/paper-v5/mixture.jsonl \
  --source-manifest data/raw/paper-v5/download_manifest.json \
  --eval-denylist-manifest "$PAPER_EVAL_DENYLIST_MANIFEST" \
  --eval-denylist local-realtime-actions=spaces/localagent-webgpu/benchmark-cases.json \
  --eval-denylist local-browser-tasks=spaces/localagent-webgpu/browser-task-cases.json \
  --eval-denylist local-agent-eval=data/synth/agent_eval.jsonl \
  --tokenizer bpe --vocab-size 16384 --tokenizer-path data/tokenizer-paper-16k.json \
  --max-chars 1000000 --out data/shards/paper-all --seq-len 2048 \
  --seed 2026 --val-fraction 0.01

# Freeze only after the full audit passes. This streams every shard hash, recomputes packed-token
# counts from length arrays, proves train/validation identity and content disjointness from the
# split artifact, verifies downloader/config/denylist provenance, binds the train-only tokenizer,
# and checks all six 5-TPP plus both possible 20-TPP consumer configs. It emits no freeze on a
# failed gate.
PYTHONPATH=src python scripts/freeze_corpus.py configs/data/pretrain-paper-freeze.yaml \
  --out data/shards/paper-all/freeze.json
PYTHONPATH=src python scripts/freeze_corpus.py configs/data/pretrain-paper-freeze.yaml \
  --verify data/shards/paper-all/freeze.json

# All eight paper pretraining configs require this path. `uv run --frozen localagent train pretrain`
# rebuilds the
# audit before model construction and stores its self-hash in lineage, so the CLI gate is not
# optional.

# Reproduce the completed one-pass derivation. Existing byte-identical destinations are accepted
# as deterministic replay; drifted destinations fail without overwrite.
PYTHONPATH=src python scripts/prepare_derived_corpora.py \
  --freeze data/shards/paper-all/freeze.json \
  --freeze-spec configs/data/pretrain-paper-freeze.yaml \
  --parent-filtered-jsonl data/shards/paper-all/filtered.jsonl \
  --parent-manifest data/shards/paper-all/manifest.json \
  --tokenizer data/tokenizer-paper-16k.json \
  --group data/shards/paper-general=fineweb_edu_dedup+cosmopedia_v2 \
  --group data/shards/paper-code=permissive_python \
  --group data/shards/paper-structured=structured_html

# Replay-verify the two exact 74,836,551-loss-token midtraining plans.
PYTHONPATH=src python scripts/plan_stage_budget.py \
  --verify data/provenance/paper/stage-budgets/midtrain-paper-hybrid.json
PYTHONPATH=src python scripts/plan_stage_budget.py \
  --verify data/provenance/paper/stage-budgets/midtrain-paper-attn.json

# Run the matched 5-tokens/parameter 34M architecture screen before promoting either treatment to
# the 20-TPP/downstream comparison. The provisional 10M hybrid pilot is separate.
# Repeat the paired commands for seeds 2027 and 2028 using the frozen configs.
uv run --frozen localagent train pretrain \
  configs/train/pretrain-paper-5tpp-hybrid-seed2026.yaml
uv run --frozen localagent train pretrain \
  configs/train/pretrain-paper-5tpp-attn-seed2026.yaml

# Domain continuation once the configured sources exist
uv run --frozen localagent train midtrain configs/train/midtrain-paper-hybrid.yaml

# Create and independently replay-verify deterministic no-model SFT budget plans before any SFT run.
PYTHONPATH=src python scripts/plan_stage_budget.py configs/train/sft-paper-hybrid.yaml \
  --out data/provenance/paper/stage-budgets/sft-paper-hybrid.json
PYTHONPATH=src python scripts/plan_stage_budget.py configs/train/sft-paper-attn.yaml \
  --out data/provenance/paper/stage-budgets/sft-paper-attn.json
PYTHONPATH=src python scripts/plan_stage_budget.py \
  --verify data/provenance/paper/stage-budgets/sft-paper-hybrid.json
PYTHONPATH=src python scripts/plan_stage_budget.py \
  --verify data/provenance/paper/stage-budgets/sft-paper-attn.json

# Masked assistant training + one-forward route/select/copy heads
uv run --frozen localagent train sft configs/train/sft-paper-hybrid.yaml

# Executed 1M parent-anchor continuation and fail-closed promotion lane
uv run --frozen localagent train sft \
  configs/train/sft-paper-tier-1m-parent-anchor-pulse-pilot.yaml
PYTHONPATH=src python -c \
  'from localagent.eval.sft_production_receipt import verify_sft_production_receipt_against_artifacts as verify; verify("data/provenance/paper/production/sft-paper-tier-1m-parent-anchor-pulse-pilot.json", expected_receipt_file_sha256="ab4a0d34b9165a7cf6fbad24b8fc7b16a49342faab074fbd67be9963be2b6b01")'
PYTHONPATH=src python scripts/sweep_sft_checkpoints.py \
  configs/eval/paper-tier-1m-parent-anchor-pulse-sft-sweep.yaml \
  --output runs/eval/paper-tier-1m-parent-anchor-pulse-sft-sweep-20260730-v1.json
PYTHONPATH=src python scripts/sft_candidate_promotion.py prepare \
  --sweep-result runs/eval/paper-tier-1m-parent-anchor-pulse-sft-sweep-20260730-v1.json \
  --development-config-out \
    configs/eval/paper-tier-1m-parent-anchor-pulse-selected-dev.yaml \
  --confirmatory-config-out \
    configs/eval/paper-tier-1m-parent-anchor-pulse-selected-confirmatory.yaml \
  --binding-out \
    data/provenance/paper/sft-candidate-parent-anchor-pulse-selected.json
PYTHONPATH=src python -m localagent.eval.agent_scorecard \
  configs/eval/paper-tier-1m-parent-anchor-pulse-selected-dev.yaml \
  --out runs/eval/paper-tier-1m-parent-anchor-pulse-selected-dev-scorecard-20260730-v1.json
# Expected exit 1 after atomically writing the development_gate_failed receipt.
PYTHONPATH=src python scripts/sft_candidate_promotion.py verify \
  --binding data/provenance/paper/sft-candidate-parent-anchor-pulse-selected.json \
  --development-scorecard \
    runs/eval/paper-tier-1m-parent-anchor-pulse-selected-dev-scorecard-20260730-v1.json \
  --decision-out \
    data/provenance/paper/sft-candidate-parent-anchor-pulse-development-decision.json

# Optional exact-AST tool-call GRPO (offline deterministic reward, not BrowserGym)
uv run --frozen localagent train rl configs/train/rl-paper-hybrid.yaml
```

The completed bounded seed-2027 pilot used the separate proxy configs:

```bash
uv run --frozen localagent train midtrain \
  configs/train/midtrain-webgpu-proxy-pilot-hybrid.yaml
uv run --frozen localagent train sft \
  configs/train/sft-webgpu-proxy-pilot-hybrid.yaml
uv run --frozen localagent train rl \
  configs/train/rl-webgpu-proxy-pilot-hybrid.yaml
PYTHONPATH=src python scripts/summarize_stage_pilot.py \
  --midtrain-config configs/train/midtrain-webgpu-proxy-pilot-hybrid.yaml \
  --midtrain-metrics runs/midtrain-webgpu-proxy-pilot-hybrid-seed2027/metrics.json \
  --midtrain-checkpoint runs/midtrain-webgpu-proxy-pilot-hybrid-seed2027/latest.pt \
  --sft-config configs/train/sft-webgpu-proxy-pilot-hybrid.yaml \
  --sft-metrics runs/sft-webgpu-proxy-pilot-hybrid-seed2027/metrics.json \
  --sft-checkpoint runs/sft-webgpu-proxy-pilot-hybrid-seed2027/latest.pt \
  --rl-config configs/train/rl-webgpu-proxy-pilot-hybrid.yaml \
  --rl-metrics runs/rl-webgpu-proxy-pilot-hybrid-seed2027/metrics.json \
  --rl-checkpoint runs/rl-webgpu-proxy-pilot-hybrid-seed2027/latest.pt \
  --output docs/paper/results/webgpu-proxy-pilot-seed2027.summary.json
```

The [validated summary](paper/results/webgpu-proxy-pilot-seed2027.summary.json) records that all
three stages resolved `auto` to MPS/fp32 under PyTorch 2.13.0. Midtraining improved the agent
holdout from 7.7869 to 2.6371 loss and 3.71% to 69.80% token accuracy, while the general holdout
regressed slightly from 5.7064 to 5.7342 loss and 18.68% to 18.30% accuracy. SFT improved
held-out assistant loss from 2.7320 to 1.8146 and token accuracy from 67.29% to 73.13%, but
teacher-forced all-assistant-token exactness reached only 1/65. Offline GRPO realized 12 optimizer
updates from six
informative groups; every 53-row held-out metric had zero pre/post delta. Since the RL policy
updates only the autoregressive LM, that artifact invalidates the SFT structured heads and is not
the checkpoint used by the structured browser runner.

The pilot's scheduled agent-row weight ramped from 25% to 50%, but realized shares were only
10,290/645,170 input tokens (1.60%) and 6,673/641,553 loss tokens (1.04%) because agent
conversations were much shorter than 2,048-token general rows. Treat the observed gain as
directional evidence from about 6.7K supervised agent targets, not as evidence for a 25–50%
agent-token mixture. The historical 2K paper configs therefore selected
`data.mixture.unit: loss_tokens`. That is not a safe transfer to
`openai_full_catalog_v1`: assistant targets are short while every row carries a masked
~3.5K-token 50-tool prefix. The first exact 10M tier plan exposed the consequence—192,051 of
200,000 microbatches came from agent rows and the schedule consumed 1,386,267,522 input tokens for
only 41,570,856 supervised tokens. The 4K tier configs now use `input_tokens` and match their
midtrain update horizons to their 20/5/1-TPP pretrain pilots (599/1,610/2,909 updates). Their
deterministic deficit scheduler selects the most under-served source by measured input-token
entitlement, still normalizes accumulated gradients by supervised-token count, checkpoints the
entitlement/served state, and reports draw, input-token, and loss-token shares separately. An old
draw-only or loss-token checkpoint cannot resume into this mode because it cannot prove the same
entitlements. No new full-budget run has been collected. Both matched paper arms checkpoint every
100 steps and enable resume. Resume fails closed unless the optimizer, scaler when used,
Python/Torch/backend RNG, history, global/per-source accounting, scheduler observations, lineage,
and resolved execution identity agree; the bit-exact regression test is on one CPU runtime, so
cross-backend exactness is not claimed.

The exact SFT checkpoint was exported as a parity-gated fp16 action graph and run in three fresh
WebGPU page sessions under the internally prespecified pre-assistant-padding stress condition at
fixed 512-token inputs. Its median-of-three within-run TTFA was 24.75 ms p50 and 34.405 ms p95,
but it predicted abstention on every case. Capability was 1/20 unique action cases overall—0/19
tool-required and 1/1 abstention—and 0/8 unique DOM tasks. Repetition for timing yields 90/1,800
action exactness and 0/720 DOM success; those are not independent capability sample sizes. The
[action summary](paper/results/m5-webgpu-sft-action-pilot-seed2027.summary.json)
and [DOM summary](paper/results/m5-webgpu-sft-dom-pilot-seed2027.summary.json) preserve the result.

An offline parity diagnostic found natural-prompt route correctness of 17/20 and selector top-1
correctness of 17/19 tool cases. The internally prespecified stress condition materializes real,
unmasked
space tokens before the assistant marker and reads the final hidden state; every case at 128 tokens
and above routes to text. Native PyTorch, fp32/fp16 ONNX, and exported JSON heads agree, so this is
not an export or precision failure. Treat it as a feature-materialization shift, not generic
long-context evidence or natural-context WebGPU quality. The fixed-512 condition fails the
capability gate.

The corrected fixed-compute runner now appends filler after the natural assistant marker, still
executes the 512-token graph, dispatches from `hidden[natural_input_tokens - 1]`, and restricts
pointer scans to the natural span. The
[offline audit](paper/results/sft-structured-context-robustness-seed2027.summary.json) preserves
natural route/selector counts on both frozen suites, and the
[full-stack export-parity gate](paper/results/sft-structured-export-parity-seed2027.summary.json)
shows exact native/fp32-ONNX/fp16-ONNX agreement on 20/20 reused action-suite routes, tools,
grounded arguments, and normalized actions at the corrected decision index. Its shared 16/20
offline exact-action score is diagnostic reuse, not browser or independent capability evidence.
The
[browser protocol](paper/results/webgpu-proxy-pilot-seed2027.corrected-browser.protocol.json)
drew no external timestamp and no browser result before the runner moved to the
lineage-bound cached-autoregressive ABI. It is preserved as superseded evidence rather than
rewritten. Any future collection needs a new freeze against the current runner and final trained
bundle. The action, DOM, and 65-row suites were inspected
during diagnosis, so the corrected measurements are reused-suite deployment-parity
re-evaluations, not an untouched capability test. If the target is genuine capability under
the original pre-marker materialization, train and evaluate the heads on that exact condition
instead.

Future SFT runs now persist `heldout_structured_eval` in both checkpoint and metrics whenever an
explicit disjoint evaluation corpus and route/dense heads exist. It reports natural and
trailing-fixed-compute route, selector, and dispatched-tool metrics plus compact casewise mismatch
counts; restored selector embedding width is inferred from the checkpoint. This makes the
training-to-browser decision-feature contract a routine gate rather than a post-hoc diagnosis.

The exact six-run matrix, corpus-size gate, counterbalanced order, evidence rules, and current
compute estimate are in
[`docs/paper/MATCHED_5TPP_RUNBOOK.md`](paper/MATCHED_5TPP_RUNBOOK.md).
As of 2026-07-28, all six 34M five-tokens-per-parameter configs are staged, but no matrix run has
started. There is therefore no quality-trained 34M result to combine with the random-weight latency
evidence.

The repository does not vendor BFCL, Mind2Web, WebLINX, or BrowserGym payloads. Their private,
prompt-only exports have nevertheless been produced from the exact revisions below, replay-frozen
with the v3 suite contracts, and consumed by the paper-all decontamination run. The aggregate
manifest covers 19,334 normalized external prompts; together with the three config-hash-pinned
local suites it caused 8,633,077 bounded candidate checks and removed 15 documents. Protected
payloads and prompt rows remain ignored/private. These artifacts contain no current-step gold
actions or outcomes and establish revision-bound local holdout, not chronological freshness; an
old public benchmark does not become fresh because its cases are hash-selected after training.
The aggregate manifest shape is:

```json
{
  "kind": "localagent_evaluation_denylist_manifest",
  "schema_version": 1,
  "required_suites": [
    "bfcl",
    "browsergym",
    "mind2web",
    "weblinx"
  ],
  "suites": [
    {
      "name": "bfcl",
      "path": "<path to the real exported prompt file>",
      "bytes": 123456,
      "sha256": "<actual lowercase SHA-256>",
      "provenance": {
        "path": "<path to the frozen per-suite provenance manifest>",
        "bytes": 2345,
        "sha256": "<actual lowercase SHA-256>",
        "manifest_self_sha256": "<self-hash declared inside that manifest>"
      }
    }
  ],
  "manifest_self_sha256": "<self-hash over the preceding canonical object>"
}
```

The shown suite row is a schema example, not a supplied artifact; the composer writes one real,
sorted row for every supplied external provenance manifest and makes `required_suites` equal that
set. Paths are relative to the list manifest. The three direct local files are
`spaces/localagent-webgpu/benchmark-cases.json`,
`spaces/localagent-webgpu/browser-task-cases.json`, and `data/synth/agent_eval.jsonl`; their
expected sizes and hashes come from the checksummed corpus config, so regeneration requires an
intentional policy update. `prepare_corpus.py` refuses a legacy unprovenanced list row, missing
required names, drifted per-suite or list self-hashes, prompt-output byte/hash mismatches,
duplicate names, and duplicate resolved paths. It records every named input artifact and the
normalized prompt-set fingerprint. Direct `--eval-denylist [NAME=]PATH` remains available; in the
paper lane, config hashes bind the three local direct inputs.

Suite files must first be converted to one of the reader's explicit prompt-only formats: one
prompt per line, JSONL records with a prompt/query/instruction/text/content or user messages field,
or a versioned JSON object with a top-level `cases` array. Hashes cover those exact exports; this
screen does not infer prompts from arbitrary upstream benchmark schemas, expected actions, DOM
snapshots, or remote environments. Protected benchmark exports remain private when their terms
forbid redistribution; publish their hashes and adapter contract, not their contents.

The later, post-training external capability slice has a stricter evaluation-only freeze and
comparison contract in
[`docs/paper/FRESH_EXTERNAL_EVAL_CONTRACT.md`](paper/FRESH_EXTERNAL_EVAL_CONTRACT.md). It binds
every declared pretrain/midtrain/SFT text artifact, selects cases before overlap inspection,
screens normalized/shingle prompt overlap plus derived action-template overlap on labeled
`Conversation` rows, and resamples task clusters for the paired exact-action interval. No real
label-bearing external evaluation slice or native benchmark score is currently present, and any
future expected-action slice must never be used as training input. A genuinely fresh claim
requires a new post-training revision, hidden steward export, or procedurally generated seed set
frozen before either system's outcomes are inspected. The prompt-only pretraining denylist and
this future post-training source are deliberately different artifacts.
The exact BFCL, BrowserGym/MiniWoB, Mind2Web, and WebLINX revisions, licenses, native metrics,
privacy rules, and adapter limitations are audited in
[`docs/paper/EXTERNAL_BENCHMARK_AUDIT.md`](paper/EXTERNAL_BENCHMARK_AUDIT.md).

The production Mind2Web DOM ranker is deliberately runtime-bound to CPython 3.12 and Unicode
15.0, matching `.python-version` and its frozen config. A different Python/Unicode runtime fails
closed rather than silently producing new prompt bytes. Core model code supports Python 3.10+,
but full paper-provenance replay and its corresponding tests must use the frozen 3.12 runtime.

Every packed corpus now publishes a generation-scoped, canonical
`split-assignment.jsonl`, content-bound to both `sha256(document_id)` and the filtered UTF-8 text
hash. The base manifest pins its byte size, SHA-256, and assignment fingerprint. Derived paper
corpora verify that artifact and preserve each overlapping document's held-out membership; they do
not recompute an exact-size validation slice within each smaller family.

For an actual GPU-backed, Drive-persisted run, upload and execute
`notebooks/localagent_pretraining_colab.ipynb`. It first performs a 10-update smoke run, then can be
switched to the resumable 6,000-update preset. Data preparation and memory-mapped shard reads stay
on `/content`; only manifests, configs, tokenizer, receipts, and atomic `latest.pt` checkpoints are
mirrored to `MyDrive/LocalAgent/pretraining/`. This follows Colab's recommendation to avoid many
small reads and writes through a mounted Drive filesystem. If Drive authorization is unavailable,
the notebook creates a downloadable artifact bundle instead of losing the run with the VM.

### Measured Colab smoke (2026-07-25)

The browser-tier path was executed—not estimated—on a Colab Tesla T4 with 14.6 GiB:

- model: `webgpu-35m-hybrid`, 16K BPE, 12 layers, hybrid conv/GQA, QK-Norm;
- streamed corpus: 429 accepted documents and 442,967 packed tokens;
- licenses: ODC-By plus allowed Apache-2.0, BSD-2/3-Clause, ISC, and MIT code;
- optimization: 10 updates / 10,240 consumed tokens in 15.9 seconds;
- loss: 9.773 → 8.844;
- verified weights SHA-256:
  `ce06a9b838bb048a4bc95301ebbc55b5ec3e663035e85ea7f2a09a7f85055fea`.

This smoke proves the corpus → tokenizer → shards → mixed-precision CUDA → checkpoint path. Ten
updates do not produce a useful model and must not be reported as trained capability.

The sample dataset is a plumbing check, not a training recommendation. Full runs must record the
manifest, exact config, git revision, tokenizer checksum, checkpoint, and frozen evaluation report.
