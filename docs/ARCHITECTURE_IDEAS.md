# Architecture ideas — going past the open-source playbook

> We don't have to copy what the small-LLM repos do. This doc collects original structural ideas
> for LocalAgent, says which we **ship now** vs **propose**, and gives a recommendation. The
> central bet: at the sizes we care about, the *agent-as-controller* framing changes the
> architecture, and the *vocabulary* is the dominant design lever.

## 0. The reframe: at ~1M params, build a controller, not a chatbot

A 1M-param model cannot store the world. So stop trying. An agent's job is **routing** — given a
user turn + tool specs + memory, decide: *answer directly, call tool X with args, or abstain*.
That decision is mostly **pattern + control flow**, not knowledge. Knowledge is offloaded to
**tools** and the **memory/retrieval store** (which the bigger pipeline already has).

This gives each tier a distinct job rather than "the same model, smaller":

| Tier | ~params | Job | Vocab | Realistic ask |
|---|---|---|---|---|
| **ultra-tiny** | ~1M | **tool router / planner** | byte (256) | pick the right tool + args, abstain when no tool fits; little free-form text |
| **tiny** | ~30M | **agent** | BPE 32k | tool calls + short grounded answers + memory ops |
| **small** | ~90M | **capable local agent** | BPE 32k | multi-turn tool use + coherent text |

The eval that matters for ultra-tiny is **BFCL AST accuracy + irrelevance**, *not* perplexity.

## 1. The vocabulary tax (why ultra-tiny needs a different structure)

At 1M params a 32k×d embedding table is impossible — with d=128 it's already 4M params, 4× the
whole budget. So the embedding *is* the architecture problem at this scale. Three levers, all now
wired into `ModelConfig`:

- **Byte-level vocab (256).** Embedding table becomes negligible (256×embed_dim). Sequences get
  longer, but tool-routing turns are short. Tokenizer-free, trivially robust to any input.
- **Factorized embeddings (ALBERT-style).** Learn `vocab×embed_dim` then up-project
  `embed_dim→d_model`. Decouples vocab cost from model width. (`embed_dim` knob.)
- **Depth-recurrence / weight sharing (Universal Transformer + ALBERT).** Run `n_layers` blocks
  `n_loops` times with shared weights → *effective depth = n_layers × n_loops* at the param cost
  of `n_layers`. A 1M model gets depth-12 reasoning from 2 blocks. (`n_loops` knob, with a small
  per-loop embedding so a block knows its iteration.)

`ultra-tiny-1m` = byte(256) + embed_dim 64 + 2 blocks × 6 loops ≈ **0.98M params, effective depth 12.**
These three are **implemented and tested today.**

## 2. Proposed structural ideas (designed, not yet built)

### 2a. Dual output head: a **tool head** + grounded args — **IMPLEMENTED**
Instead of forcing the tiny model to emit valid JSON token-by-token, split the job:
- a **tool head** = a classifier over available tools + a "text" class, reading the prompt's final
  hidden state (`agent/tool_head.py`);
- **arguments** grounded in the prompt by the schema-driven decoder (`agent/constrained.py`):
  proper-noun / preposition-tail spans for strings, regex for numbers/arithmetic, enum members.

This turns "generate valid function calls" (hard) into "classify the tool + extract a span" (easy).
**The key result is that the tool head must be trained *jointly* with the model** (auxiliary
classification loss during SFT, `train.sft(joint_tool_head=True)`), not as a post-hoc probe:

| tool head | tool_call | web_search | planner | text | overall (held-out, level 1) |
|---|---|---|---|---|---|
| frozen linear probe | 0.65 | 0.67 | 0.50 | 0.82 | ~0.63 |
| **jointly trained (SFT aux loss)** | **0.98** | **0.92** | **0.75** | **1.00** | **~0.94** |

A frozen probe plateaus because the raw features don't separate planner/web_search; the auxiliary
loss *shapes* the representation so they do. This is fully **trigger-free** — unlike the earlier
template decoder whose per-tool phrases secretly did tool selection.

**Pointer/copy argument head (`agent/pointer_head.py`) — implemented.** Conditioned on which
argument it is filling, the head predicts a `(start, end)` span over the prompt's byte positions;
the value is the copied span. Trained jointly with the model (SFT aux loss). It is the *general*
grounding mechanism — no per-arg regex/preposition heuristics — and is what lets a follow-up arg be
grounded in an earlier **tool response** (multi-turn). Honest ablation: on the clean templated
single-turn eval the tuned deterministic extractors still edge it out (they're near-perfect,
0/787), so the deployed single-turn path uses heuristics and the pointer head is used for
multi-turn; closing the single-turn gap wants more joint training / a bigger tier.

**Multi-turn coding episodes (B).** The dataset includes Claude Code / Codex-style trajectories
(`read_file`→response→`run_tests`→…; `grep_search`→response→`read_file` where the path comes from
the response). `eval.multi_turn_eval` replays each episode and AST-matches the action at every
tool step over the full history. This is the frontier: it needs the pointer head (heuristics can't
reach into a tool response) and benefits from training the heads on multi-turn contexts.

### 2b. Grounded / grammar-constrained decoding — **IMPLEMENTED, and it works**
Tool-call structure is decoded through the tool schema; **string arguments are grounded in spans
of the prompt** rather than free-generated. The model only has to *rank* candidate calls
(teacher-forced, one forward each — no autoregression), so correctness is a property of the
decoder, not something the tiny model must byte-copy. See `agent/constrained.py`.

**Empirical result (the reason this idea exists).** Training the ~1M byte model from scratch
(pretrain → SFT) on held-out slot values:

| decoder | tool_call | web_search | planner | text | overall |
|---|---|---|---|---|---|
| free-gen (raw autoregressive bytes) | ~0% | ~0% | ~0% | ~0% | ~0% |
| **grounded constrained (same model)** | **100%** | **100%** | **100%** | **100%** | **100%** |

The raw model learns call *structure* perfectly by a few hundred steps but substitutes a
*memorized* slot value instead of copying the unseen one (e.g. asked about held-out "Boston" it
emits a training city like "Dublin"). Grounding the arguments in the prompt closes the gap
entirely. This is the empirical case for "controller, not chatbot": let the tiny model select +
rank, let the decoder guarantee valid, grounded arguments. (Current span extractors are
template-aware heuristics; a general typed n-gram enumerator is the next step.)

### 2c. KV-free backbone option (SSM / linear attention)
A `backbone: attn | ssm` switch. For multi-turn agents on edge/NPU, an SSM (Mamba/RWKV-style)
gives **constant memory per step** (no growing KV cache) — attractive for long tool traces and
the memory module. Attention stays the default; SSM is the edge/long-context variant. Reserved as
a config knob.

### 2d. Parameters = skills, external store = knowledge
Lean fully into retrieval: the weights encode *control/skills*, a kNN/RETRO-lite store holds
*facts*. This is already half-present via `agent/memory.py` + the data flywheel; the structural
add is a retrieval *head* that conditions generation on fetched snippets, so the tiny model never
needs to memorize.

### 2e. Micro-MoE for the `small` tier — **IMPLEMENTED AS AN OPT-IN EXPERIMENT**
Each block can now replace its dense SwiGLU with a deterministically routed expert bank. The
PyTorch reference executes only selected token/expert pairs and exposes load, entropy, dead-expert,
and total/active-parameter diagnostics. The first treatment stores 43.86M parameters while
selecting a nominal 17.32M per-token path; its dense control has 17.30M total/active parameters.

This is not the deployment default. Every expert still increases download and resident memory,
and the ONNX/WebGPU exporter has no proven sparse-dispatch lowering. Promotion requires matched
quality, routing-diversity, memory, and measured browser-latency gates. See
[`SPARSE_EXPERTS_REAL_DATA.md`](./SPARSE_EXPERTS_REAL_DATA.md).

## 3. Recommendation

1. **Ship now (done):** the three vocab/structure levers + the three-tier scheme. They are what
   make a sub-1M agent even plausible.
2. **Build next (Phase 4 alongside SFT):** **2a + 2b** — the dual tool-head + grammar-constrained
   decoding. This is where a small, original design beats "shrink a Llama," because it converts the
   hardest tiny-model task (valid tool calls) into easy ones (classify + copy + masked decode).
3. **Keep as switches:** **2c** (SSM backbone) for the edge/NPU export story, **2d** retrieval head
   for knowledge offload. **2e** is implemented but remains an opt-in research candidate.

Everything here is additive to the existing pipeline — same `Conversation` schema, same
train/eval/export stages, same data flywheel. See `docs/ROADMAP.md` for where each lands.
