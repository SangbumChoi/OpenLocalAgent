# LocalAgent — Architecture

A minimal, hackable, **pure-PyTorch** pipeline for an LLM that acts as a local **agent** (tool
calling + text generation).

The ladder now spans **10M to 700M** parameters; the "< 100M" framing below predates the 150M/300M/
700M tiers and survives only where it describes the on-device tiers specifically. Each model config
declares the `param_budget` it is held to — see [CONVENTIONS.md](CONVENTIONS.md).

Design rationale is in [`campaign/RESEARCH.md`](campaign/RESEARCH.md) and the original build order
in [`campaign/ROADMAP.md`](campaign/ROADMAP.md); both are campaign-era and describe the repository
as it was. For what the stages do now, read [STAGES.md](STAGES.md).

---

## 1. System at a glance

```
                         ┌───────────────────────────────────────────────┐
                         │                  PIPELINE                      │
                         │  (openlocalagent.pipeline — orchestrates)     │
                         └───────────────────────────────────────────────┘
                                            │
   DATA                  TRAIN                          EVAL              SERVE
 ┌────────────┐   ┌──────────────────────┐      ┌──────────────┐   ┌──────────────┐
 │ pretrain   │   │ pretrain (from scratch)│      │ tool_eval    │   │ agent.runtime│
 │ corpus     │──▶│ domain midtrain       │─────▶│ (AST/schema) │──▶│ + tools      │
 │ agent_synth│   │ SFT / distill         │      │ text_eval    │   │ + memory     │
 │ flywheel   │   │ offline RL (optional) │      │ multi-turn   │   │ demos/ (UI)  │
 └────────────┘   └──────────────────────┘      └──────────────┘   └──────┬───────┘
       ▲                                                                   │
       │                                                                   ▼
       │                       ┌──────────────────────────────┐    ┌──────────────┐
       └───────────────────────│      DATA FLYWHEEL            │◀───│ conversation │
         mined + verified      │ ingest → mine → verify →      │    │   store      │
         training samples      │ schedule retrain → eval →     │    │ (+feedback)  │
                               │ redeploy                      │    └──────────────┘
                               └──────────────────────────────┘
                                            │
                      EXPORT: PyTorch ▸ ONNX/WebGPU today
                              GGUF / ExecuTorch remain honest stubs
```

Everything is a small Python module under `src/openlocalagent/`, driven by YAML configs in
`configs/` and a single CLI (`openlocalagent ...`). No training framework; just PyTorch + a BPE
tokenizer.

---

## 2. The model (`openlocalagent.model`)

A compact decoder, sized to stay strictly **< 100M params**, favoring depth over width and a
small KV-head count. Sequence mixing is an experimental axis: periodic full causal
attention is compared against cheap causal short-convolution, rather than calling the latter
Kimi Delta Attention.

- **Blocks:** pre-norm RMSNorm → MQA/GQA attention with RoPE and optional QK-Norm, or gated
  causal short-convolution → SwiGLU MLP, with residuals.
- **Optional sparse FFN:** `ffn_num_experts > 1` replaces each dense SwiGLU with a stable top-k
  routed expert bank. The hard budget counts every stored expert; a separate nominal active count
  includes the router and selected experts. Dense configs allocate no router and preserve legacy
  state keys.
- **Tied** input/output embeddings (the embedding table dominates param count at this scale).
- **Inference state:** attention stores K/V per token; short-convolution stores a fixed tail.
  Recurrent passes share weights but retain separate cache slots, preserving exact cached/fresh
  parity.
- **Optional screenshot bridge:** `vision_enabled` adds a compact 8-bit RGB patch encoder and
  prepended visual-token prefix (`webgpu-10m-vision`: 10,612,224 parameters, 36 visual tokens).
  It is budget-counted and trainable in PyTorch; legacy text checkpoints leave it disabled and
  unchanged.  Cache-bearing visual decode and ONNX/WebGPU export remain unimplemented until the
  preprocessing and native visual evaluator are frozen.
- **Shared-paper tiers:** one frozen 16K BPE tokenizer and corpus split, 2K packed pretraining
  rows, and a 4K model limit make loss and prompt-capacity comparisons meaningful.

  | config | d_model / embed | unique blocks × loops | heads / kv | ffn | exact params | job |
  |---|---|---|---|---|---:|---|
  | `webgpu-1m-bpe-router` | 128 / 32 | 2 × 3, `[conv, attn]` | 2 / 1 | 432 | 980,480 | recurrent router/planner |
  | `webgpu-10m-hybrid-4k` | 384 / 384 | 4 × 1, `[conv, attn, conv, attn]` | 6 / 1 | 512 | 10,524,544 | latency-feasible AR candidate |
  | `webgpu-10m-attn-4k` | 384 / 384 | 4 × 1, all attention | 6 / 1 | 624 | 10,547,072 | matched 10M control |
  | `webgpu-10m-vision` | 384 / 384 | 4 × 1, `[conv, attn, conv, attn]` + 36 visual tokens | 6 / 1 | 512 | 10,612,224 | experimental screenshot bridge |
  | `webgpu-96m-hybrid` | 640 / 640 | 18 × 1, 12 conv + 6 attention | 10 / 1 | 1728 | 95,320,448 | near-budget quality/feasibility |
  | `webgpu-96m-attn` | 640 / 640 | 18 × 1, all attention | 10 / 1 | 1984 | 95,298,944 | matched 96M control |
  | `webgpu-44m-moe` | 320 / 320 | 9 × 1, 6 conv + 3 attention | 5 / 1 | 8×512, top-2 | 43,862,464 total / 17,320,384 active | opt-in capacity treatment |
  | `webgpu-17m-dense-moe-control` | 320 / 320 | 9 × 1, 6 conv + 3 attention | 5 / 1 | 1024 | 17,297,344 | active-matched dense control |

  The 1M tier pays for a common BPE vocabulary through factorized embeddings and recovers
  effective depth six through shared-weight recurrence. The 10M and 96M FFN widths deliberately
  compensate for mixer parameter differences; each comparison estimates the complete matched
  package, not an isolated mixer substitution. Legacy byte 1M and 30M/90M configs remain useful
  smoke/reference models but are not the shared-paper tiers.

**Tokenizer** (`model/tokenizer.py`): a byte-level BPE (trained with the `tokenizers` lib) with
**agent special tokens** so tool use is in-vocabulary:
`<|end|>`, `<|user|>`, `<|assistant|>`, `<|tool|>`, `<tool_call>` / `</tool_call>`, and
`<tool_response>` / `</tool_response>`. The full-catalog contract additionally reserves
`<|system|>` and `<|tool_catalog|>` framing and rejects those literal boundaries recursively
from user-controlled data.

---

## 3. Agent format & runtime (`openlocalagent.agent`)

### Wire format (ChatML-ish, tool-native)
```
<|tool_catalog|>{"tools":[{"type":"function","function":{...}}]}</|tool_catalog|><|end|>
<|system|>You are LocalAgent.
<|user|>
What's the weather in Paris?
<|assistant|>
<tool_call>{"arguments":{"city":"Paris"},"name":"get_weather"}</tool_call><|end|>
<|tool|>
<tool_response>{"temp_c": 19, "cond": "cloudy"}</tool_response>
<|assistant|>
It's 19°C and cloudy in Paris.<|end|>
```
The model decides between **emitting text** and **emitting a `<tool_call>`** (or abstaining — the
irrelevance case from Hammer/BFCL).

### Runtime loop (`agent/runtime.py`)
```
build prompt(system+tools+history+memory) → generate →
  parser.extract_tool_calls():
    if tool_call  → tools.dispatch() → append <tool_response> → loop
    else (text)   → return assistant text
```
- `agent/tools.py` — a **tool registry** (`@tool` decorator → name, JSON schema, callable),
  with a sandbox boundary and a few built-ins (calc, web stub, memory tools).
- `agent/parser.py` — tolerant extraction/repair for legacy interactive runtime use. Training,
  RL rewards, and scorecards use a separate strict whole-output parser so repaired or
  mixed-content generations cannot earn exact credit.
- `agent/memory.py` — **two-tier MemGPT-style memory**: `core` (always in context) + `archival`
  (out-of-context, searchable), exposed to the model as tools (`memory_append`,
  `memory_search`, `memory_replace`) with a paging/consolidation policy.

---

## 4. Data (`openlocalagent.data`)

- `schema.py` — canonical dataclasses: `Message`, `ToolSpec`, `ToolCall`, `Conversation`,
  `Sample`. One JSONL line = one `Conversation`. This is the single interchange format across
  synth, flywheel, SFT, and eval (APIGen/xLAM-style).
- `pretrain_corpus.py` — descriptor-bound streaming, disk-backed quality filtering,
  exact/near-dedup, document-level splitting, train-only tokenizer fitting, and immutable
  memory-mapped packed shards. The production paper corpus has 504,010 retained documents and
  528,669,610 verified BPE tokens.
- `conversation_artifact.py` — one-pass JSONL verification against manifest byte/hash/count
  declarations, catalog interning, recursively immutable verified rows, and semantic plus
  exact-rendered-prompt contamination audits.
- `prompt_contract.py` — one exact full OpenAI-style function catalog and role-history
  materialization shared by midtraining, SFT, RL, and evaluation. It emits one masked row per
  assistant decision and forbids truncation.
- `agent_synth.py` — the **synthetic agent-data generator** (ToolACE + APIGen + Hammer):
  1. sample tools from an API pool, 2. multi-agent dialog synthesis at a **target complexity**,
  3. inject **irrelevance negatives**, 4. **dual verification** (rule: schema/AST valid; model:
  semantic), 5. emit verified `Conversation`s. Pluggable "teacher" backend (any OpenAI-style
  endpoint or a local model).
- `public_agent.py` — offline, hash-pinned xLAM/Mind2Web/local audit adapters. It canonicalizes
  public action records into `Conversation`, preserves per-row provenance, rejects benchmark and
  license/split boundary violations, checks exact prompt holdouts, and writes a self-hashed
  mixture manifest.
- `visual.py` — dependency-free decoding of bounded 8-bit RGB/RGBA PNG screenshots into tensors for
  the opt-in visual bridge; unsupported/interlaced formats fail closed rather than being silently
  admitted to training.
- `flywheel.py` — ingest canonical conversation exports, **mine** positive/provenance-bound
  training rows, **verify** split/schema/provenance policy, and idempotently append to the
  SFT/distill pool. Production SQLite feedback ingestion remains Phase 8.

## 5. Training (`openlocalagent.train`)

All stages share `train/device.py` (one **device abstraction**: CUDA / MPS / CPU / XPU / other
accelerators, autocast + dtype policy) so the *same* loop runs on GPU, CPU, or NPU-ish backends.

| stage | file | objective |
|---|---|---|
| Pretrain | `pretrain.py` | next-token CE on packed corpus shards; AdamW, cosine/WSD, exact accounting, periodic resumable checkpoints |
| Midtrain | `midtrain.py` | scheduled general/code/structured/agent continuation with token-faithful mixture accounting and resumable state |
| SFT | `sft.py` | exact full-catalog decision prompts, loss masked to the current assistant body + EOS, plus optional route/select/copy heads |
| Distill | `distill.py` | **off-policy seq-KD** (default) and **on-policy reverse-KL** (MiniLLM/OPD) from a teacher; trajectory-level |
| RL | `rl.py` | optional offline group-relative clipped-ratio update with reference KL and strict exact tool/text rewards |

Checkpoints are plain PyTorch mappings with model/config/tokenizer/data/code lineage. Pretraining
and midtraining restore optimizer, RNG, sampler, accounting, and validation state. SFT and RL also
write periodic atomic checkpoints and restore their optimizer, backend RNG, deterministic
decision/prompt schedule, accounting, history, and stage-specific auxiliary state. Every resume
path fails closed on lineage, execution, or sealed-state drift.

## 6. Evaluation (`openlocalagent.eval`)

- `tool_eval.py` — strict whole-output AST/schema scoring for single, parallel, sequential,
  multi-turn, text, and abstention decisions. It is an internal **BFCL-style** scorecard, not an
  official BFCL result; official BFCL uses its own released generator/checker.
- `agent_scorecard.py` — binds the model, checkpoint, tokenizer, training contract, cases,
  evaluator source, 4K prompt/decode budget, and deterministic generation into a self-hashed
  result. It fails closed on prompt-contract or lineage drift.
- `text_eval.py` — perplexity + a few task accuracies (small ARC/GSM-style sets) for the
  text-generation side.
- `harness.py` — runs a config'd suite, writes a JSON report; also drives a **multi-turn**
  simulated-user + tool-sandbox eval (tau-bench-lite).
- `real_use.py` — fail-closed audit and promotion gates for frozen public action suites, including
  dataset/category/behavior/capability coverage and optional sparse-router utilization, entropy,
  dead-expert, and category-divergence evidence.
- Exported models are checked for **parity** vs the PyTorch reference here.

## 7. Conversation store + data flywheel (`openlocalagent.agent.memory`)

Every served conversation is persisted (SQLite to start) with optional **feedback signals**
(Airbnb AITL): pairwise preference, adoption decision + rationale, knowledge-relevance,
missing-knowledge. The flywheel job:

```
log conversations ─▶ mine candidates ─▶ dual-verify ─▶ append to train pool
        ▲                                                        │
        └──────────── redeploy ◀── eval ◀── distill/SFT ◀────────┘
```
This is what makes the system improve from real local usage instead of staying static.

## 8. Inference & export (`openlocalagent.inference`)

- `generate.py` — KV-cached sampling (greedy / temperature / top-p) for PyTorch runtime/eval.
- `export/to_onnx.py` — PyTorch/ONNX parity plus separate cache-bearing prefill and single-token
  decode graphs used by the WebGPU latency harness. The matched random-weight 10.5M hybrid clears
  100 tok/s at 128–1,536 cached contexts on one M5/Chrome setup; that is a systems result, not
  learned capability.
- The browser agent now consumes lineage-bound post-training prefill/decode exports for both
  unrestricted and schema-constrained autoregressive controls. It lazily loads one prefill and one
  fixed-`T=1` decode graph, rebinds cache tensors without JavaScript readback, and reports the
  cache strategy explicitly. A final trained 4K bundle and real 3.6K-prompt WebGPU score remain
  required; integration is not performance evidence.
- `to_gguf.py` and `to_executorch.py` remain explicit `NotImplementedError` stubs. Theoretical
  Q4 byte counts are not presented as an implemented browser quantizer.

## 9. Demos (`demos/`)

- The static WebGPU demo and benchmark surfaces visualize tool/action paths and measured runtime
  evidence. The CLI `chat` command remains an honest Phase-7 stub.

## 10. Orchestration (`openlocalagent.pipeline` + `scripts/speedrun.sh`)

`flow.py` dispatches config-bound stages (`pretrain → midtrain → SFT → optional distill/RL →
eval → export`) with lineage-bearing artifacts. `speedrun.sh` remains the toy CPU-oriented
nanochat-style smoke path; the real paper recipe is intentionally a separately frozen,
resource-gated sequence in [`TRAINING_SYSTEM.md`](campaign/TRAINING_SYSTEM.md).

---

## Conventions
- **Configs over flags:** every stage takes a YAML in `configs/`; the CLI just selects one.
- **One interchange format:** the `Conversation` schema flows through synth → flywheel → train →
  eval. Add a field there, everything downstream sees it.
- **Stubs are honest:** unimplemented functions raise `NotImplementedError` with a `TODO(phase-N)`
  pointing at the roadmap. The README status table tracks what's real.
