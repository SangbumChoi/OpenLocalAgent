# Sparse experts and public real-use data

This document defines LocalAgent's opt-in small-model sparse-expert experiment and the public-data
contract used to test it. It is an implementation and evaluation plan, not evidence that the
sparse model is already trained, faster in WebGPU, or better than the dense baseline.

## 1. What transfers from frontier MoE

[Switch Transformers](https://arxiv.org/abs/2101.03961),
[DeepSeekMoE](https://arxiv.org/abs/2401.06066),
[OLMoE](https://arxiv.org/abs/2409.02060), and
[Kimi K3](https://github.com/MoonshotAI/Kimi-K3/blob/7c5be9599120d7993748de66a76128614f15f210/k3_tech_report.pdf)
separate stored capacity from the parameters selected for a token. LocalAgent tests the same
principle at browser scale without copying the frontier serving stack:

- every block has an independent, bias-free router and a bank of SwiGLU experts;
- stable token-choice top-k routing makes exact ties deterministic;
- the PyTorch reference invokes only selected token/expert pairs, rather than evaluating every
  expert and masking the results;
- a Switch-style load-balancing term is added only to training objectives; reported validation
  language-model loss remains pure cross-entropy; and
- routing telemetry is detached and JSON-safe: assignment counts, load, token fraction, mean
  router probability, entropy, dead experts, coefficient of variation, and total/active counts.

The dense default is unchanged. `ffn_num_experts: 1` allocates no router, preserves the legacy
SwiGLU module and state-dict keys, and makes total and active counts identical.

## 2. Honest parameter and runtime accounting

`ModelConfig.estimate_params()` counts every stored expert and remains the source of truth for the
strict `<100M` budget. `estimate_active_params()` counts shared parameters, every router, and only
the selected top-k expert banks once. The latter is a nominal single-token path count, not
checkpoint size, download size, peak resident memory, or sequence-prefill FLOPs. Different tokens
in one prompt can collectively select every expert.

The first matched pair is:

| arm | stored parameters | nominal active parameters | FFN | role |
|---|---:|---:|---|---|
| `webgpu-44m-moe` | 43,862,464 | 17,320,384 | 8 × 512 experts, top-2 | capacity treatment |
| `webgpu-17m-dense-moe-control` | 17,297,344 | 17,297,344 | one dense 1,024-wide FFN | active-compute control |

Both use the same 16K vocabulary, width 320, nine-layer
`[conv, conv, attn] × 3` topology, GQA geometry, context, tokenizer, data order, seed, update
horizon, and evaluation schedule. Their nominal active counts differ by less than 0.14%.

Inspect the counts without constructing a checkpoint:

```bash
openlocalagent model-info configs/model/webgpu-44m-moe.yaml
openlocalagent model-info configs/model/webgpu-17m-dense-moe-control.yaml
```

The current ONNX exporter does not lower dynamic expert routing into a proven sparse WebGPU
dispatch. Exporting a graph that evaluates every expert would erase the intended compute saving.
Therefore the candidate remains a PyTorch research path until all of these are measured:

1. the exported graph executes only selected experts;
2. parity passes against the exact PyTorch checkpoint;
3. browser download and resident memory include all stored experts honestly; and
4. target-device prefill, decode p50/p95, and tokens/second beat the matched dense control.

The published 10.5M WebGPU SFT pilot is a separate dense checkpoint. It must not be relabeled as
the sparse candidate.

## 3. Training contract

The sparse fields are model-config values rather than command-line overrides:

```yaml
ffn_num_experts: 8
ffn_top_k: 2
router_aux_loss_coef: 0.01
```

Pretraining, midtraining, and SFT optimize:

```text
optimization_loss = pure_language_model_CE + coefficient × router_aux_loss
```

Each stage stores the optimization, language-model, unweighted-router, and weighted-router loss
histories separately. Validation and held-out reports use pure CE. Router balance is necessary but
not sufficient: a perfectly uniform router can still learn no useful specialization.

The matched experiment advances only if the sparse arm:

- stays below the total stored-parameter budget;
- has no dead experts on the frozen real-use suite;
- clears declared utilization and normalized-entropy floors;
- exhibits non-zero category-conditioned routing divergence;
- improves held-out language and strict action quality at fixed training data and updates; and
- passes measured deployment latency and memory gates on the target browser.

If it only improves quality by using more stored weights, report that as a capacity result. If it
only balances experts, report router health rather than specialization.

## 4. Public-data boundary

Public data is acquired separately and then ingested offline. Every local snapshot is pinned by
dataset, subset, upstream revision, public HTTPS URL, license and evidence URL, byte count,
SHA-256, and local path. The importer performs no network access and fails closed on drift.

| source | pinned policy | allowed use |
|---|---|---|
| [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | revision `26d14ebfe18b1f7b524bd39b404b50af5dc97866`, CC-BY-4.0, gated access | training; verified synthetic API/function-calling records |
| [Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web) | revision `17ece8eb89862368edc0cc806acee6fca5163474`, CC-BY-4.0, `data/train/train_*.json` only | training; grounded multi-step web actions |
| [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) | upstream benchmark protocol | external evaluation only; never training |
| [WebLINX](https://huggingface.co/datasets/McGill-NLP/WebLINX) | CC-BY-NC-SA-4.0 | evaluation/non-default research only; never the default training mixture |

`openlocalagent.data.adapters.public_agent.build_public_agent_dataset()` converts pinned inputs to the one
`Conversation` interchange format. Supported adapters are `xlam_v1`, `mind2web_v1`, and an
explicit `openlocalagent_v1` audit interchange. Build with:

```bash
python scripts/normalize_public_agent.py configs/data/<public-mixture>.yaml
```

The YAML declares `outputs.train`, `outputs.eval`, a self-hashed `manifest`, exact-prompt
holdouts, and `sources[]`. Source records are converted to canonical sorted-key tool calls,
schema-checked, deduplicated, and deterministically enriched. Known BFCL/WebLINX training inputs,
Mind2Web non-train shards, unexpected licenses/revisions, hash drift, and exact held-out prompt
collisions are rejected.

Public provenance remains on every `Conversation.meta` record. `rule_verified: true` means only
that the schema, catalog, arguments, sequence, and split-slot contract passed. It does not mean a
model verified semantics or that actions were executed in a live environment.

## 5. Realistic action coverage and evaluation

Real-use coverage must contain more than one-call happy paths:

- action, abstention, and irrelevant-request behavior;
- single, parallel, sequential, and multi-turn decisions;
- retrieval followed by grounded action;
- browser click/type/select traces;
- safe refusal or wait-for-approval paths;
- distractor tools and wrong-tool avoidance; and
- failures, corrections, and recovery where the public source supplies a verifiable target.

`openlocalagent.eval.real_use` audits a frozen public evaluation snapshot, then wraps the existing
strict whole-output AST/schema and teacher-forced multi-turn scorer. Its thresholds are
caller-declared—there is no hidden universal production cutoff. The report includes dataset,
license, category, behavior, capability, multi-action, and source-revision coverage plus a
self-hashed case-set identity.

Sparse promotion additionally consumes `model.routing_diagnostics()` for every prompt. Dense or
missing telemetry earns no diversity credit. Category-conditioned Jensen–Shannon divergence is
reported as evidence of different routing distributions, not proof that experts encode named
skills.

This remains an offline gold-history evaluation. BrowserGym/MiniWoB-style environment execution
and final-state rewards are still required before claiming end-to-end task success.

## 6. Stage-by-stage recipe

1. **Pretrain:** use the same frozen general/code corpus, tokenizer, packed rows, and update
   schedule for both arms. Track pure validation CE and router health.
2. **Midtrain:** mix general replay, code, structured data, xLAM TRAIN, and Mind2Web TRAIN with
   measured input/supervised-token accounting. Preserve general-validation floors.
3. **SFT:** train canonical current-decision prompts, action/abstention/irrelevance balance,
   parallel/sequential calls, and route/select/copy heads. Keep the public eval snapshot excluded
   by exact rendered-prompt hashes.
4. **Distill:** use verified teacher trajectories only after the source and verifier identities
   are bound. Do not treat teacher confidence as execution.
5. **RL:** start only when free-running outputs provide informative exact-reward groups. Prefer
   executable environment/final-state rewards; stop rather than optimize a constant reward.
6. **Promote:** require quality, router, parity, browser latency, memory, and artifact-lineage
   gates together. A failure in any axis remains visible.
