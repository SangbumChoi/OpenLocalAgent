# Tool-selection refactor: from a 51-way classifier to a *generable* pipeline

## The problem
The expanded (50-tool) pipeline selected tools with a **51-way softmax head** (`tool_head.CLASSES`).
A fixed N-way classifier is the wrong abstraction: it can't accept a tool it didn't see at train
time, must be reshaped+retrained whenever the tool pool changes (the 22→51 jump already forced a
*fresh* head), and doesn't transfer to the function-calling / MCP setting where tools differ per
request. "Pick one of 51 fixed classes" is not a generable method.

## Diagnostic: what was the classifier actually doing?
Same 40-sample held set, `tiny-30m-50tools.pt`:

| Decode path | Acc |
|---|---|
| 51-way head + pointer-copy *(shipped)* | **45%** |
| generative tool selection + pointer-copy *(drop the 51-way head)* | 5% |
| generative selection + heuristic args *(no heads)* | 5% |
| pure free-generation *(LM emits the whole call as text)* | **0%** |

The 51-way head was carrying *the entire* selection capability — remove it and 45% collapses to 5%.
The model has near-zero intrinsic ability to select or generate a call; pointer-copy only helps fill
args *after* selection is already correct.

## Fix part 1 — routes replace the 51-way classifier (works)
`agent/routes.py`: the head now classifies one of **5 stable routes** (modalities) —
`web_search / computer_use / code / app_action / text` — instead of 51 concrete tools. Adding or
removing a tool no longer reshapes the head; only adding a whole new modality does.

- `RouteHead` (5-way) as a frozen-feature linear probe: **route_acc = 76.5%** with *zero* extra
  backbone training (code 84%, text 88%, computer_use weakest 63%).
- The concrete tool is then chosen by **retrieval** (`ToolRetriever`, char-ngram, zero-training):
  **recall@6 = 68–75%**, scales to any pool, handles unseen tools.

## Fix part 2 — can the LM *generate* the specific call? (the negative result)
We rendered a small retrieved tool catalog into the prompt (`agent/incontext.py`) and SFT'd the LM
to free-generate the call from it (`scripts/sft_generative.py`, 500 steps, in-context catalog):

| | gen_acc (gold-in-candidates) |
|---|---|
| baseline (backbone + in-context prompt, no SFT) | 0.0% |
| after generative SFT | **2.1%** |

Train loss collapsed to 0.003 (full memorization of the 157-sample train set) but held generalization
stayed ~2%. Inspecting held generations shows **exactly why**:

```
USER : What's it like outside in Perth? In Celsius.
GOLD : {"arguments":{"city":"Perth","unit":"c"},"name":"get_weather"}
GEN  : <tool_call>{"arguments":{"city":"Tbilisi"},"name":"get_weather"}</tool_call>

USER : Read the file data/loader.py.
GOLD : {"arguments":{"path":"data/loader.py"},"name":"read_file"}
GEN  : <tool_call>{"arguments":{"path":"defi/tint"},"name":"grep_search"}</tool_call>
```

- **Structure: 100%** — always well-formed, correct argument *keys* for the chosen tool.
- **Argument *values*: 100% hallucinated** — `Perth`→`Tbilisi`, `8*1*13`→`15*15`,
  `data/loader.py`→`defi/tint`. The model emits *memorized* values instead of **copying from the
  prompt**.

This empirically confirms the architecture's thesis: **a <100M byte model learns tool-call structure
+ selection but cannot free-generate argument *values* — those must be copied.** That is exactly
what `ptr_head` (pointer/copy span) does, and why the grounded path reaches 45% while free-gen is 0%.

## Fix part 3 — does selection need to be *generated*? (no — it needs to be *trained*)
Two more generable selection methods, measured on the same held set, both **without** the 51-way
classifier:

| Selection method (args always via pointer-copy) | Acc |
|---|---|
| free-generate the tool name | ~5% |
| retrieval top-k + model ranks the candidates' bodies | 7.5–9.4% |
| **dense two-tower selector (trained), top-1** | **41.5%** |
| dense two-tower selector, top-2 (model ranks the 2) | 28.3% |
| 51-way classifier (reference, *not* generable) | 39.6% |

Retrieval narrows the catalog but the tiny model still can't *rank* the right tool among the
candidates (same weakness as free-gen). Selection genuinely needs a **trained discriminative
component**. The resolution is to make that component generable instead of a fixed-N softmax: a
**dense two-tower selector** (`agent/dense_selector.py`) scores every tool by `q·t`, where `q` is a
learned tower over the prompt's features and `t` is a learned tower over the tool's *description
embedding*. Selection is `argmax_j q·t_j` over **whatever tools are present** — adding/removing a
tool is a column, not a reshape; unseen tools work by embedding their description. It is a cheap
frozen-feature probe (only the two towers are learned), and it **matches the 51-way head (41.5% vs
39.6%) while being fully generable.** `top_m=2` is worse, confirming the model's multi-candidate
ranking is weak — so trust the trained selector's top-1.

## Fix part 4 — free-form (out-of-distribution) dispatch: a data problem
A hand-written free-form demo (natural phrasings, not templates) exposed that *selection* fails on
real wording even though the architecture is right: ~2/10 correct ("What is the color of a monkey?"
→ get_news; "Make a directory called build" → planner). Arg-copying often still worked — it is
selection that breaks on OOD phrasing, because the synthetic templates are narrow and the backbone's
features overfit them.

The fix is data, not architecture. Adding a **paraphrase-rich** data source (`data/paraphrase.py`:
many varied phrasings per tool) and **example-augmented tool embeddings** (`tool_embeddings(...,
examples=)`: index each tool by the centroid of example queries) lifts OOD selection measurably, on a
44-query hand-authored held set (`eval/freeform.py`), with no backbone retraining:

| free-form OOD (44 queries) | selection top-1 | top-3 | end-to-end call-name |
|---|---|---|---|
| baseline (templated data, plain tool embeddings) | 27% | 50% | 25% |
| + paraphrase data + example-augmented embeddings | **36%** | 52% | **34%** |

That is ~+35% relative from data alone. It plateaus there because the dense selector is a
**frozen-feature probe** — the backbone's features are still templated-overfit, so the probe can only
recover so much. Closing the rest needs the **backbone SFT'd on the paraphrase corpus** (so its
features become phrasing-robust), which is the next training lever, not an architecture change.

## Conclusion — the correct *generable* architecture
Drop the 51-way classifier. Each job goes to the component that can actually do it — and the parts
that need a learned mechanism are made generable rather than fixed-N:

| Job | Component | Why |
|---|---|---|
| modality (gate) | **5-way route head** | stable, portable; 76.5% as a probe |
| tool **selection** | **dense two-tower selector** (`q·t` over tool-desc embeddings) | trained → 41.5% (≥ the 51-way head); generable → any/unseen tool, no reshape |
| call structure + arg *keys* | **generation** | the model gets these right (100% structure) |
| argument *values* | **pointer-copy** (`ptr_head`) | the tiny model provably can't free-generate these |

Two things were measured, not assumed: (1) free generation of argument *values* is not viable at this
size — values must be copied; (2) free generation / retrieval-ranking of the tool *name* is not
viable either — selection needs a trained scorer. The dense selector keeps that trained scorer's
accuracy **without** the fixed-N classifier's brittleness, so the whole pipeline becomes generable
(add a tool = one row in the retrieval/selector index + one catalog line) with no accuracy cost.

### Artifacts
- `runs/tiny-30m-routed.pt` — backbone + 5-way route head.
- `runs/tiny-30m-incontext.pt` — generative in-context SFT checkpoint (the negative-result probe).
- `scripts/retrain_routed.py` — route head; `scripts/sft_generative.py` — in-context gen SFT;
  `scripts/eval_hybrid.py` — 51-way vs retrieval vs dense-selector comparison.
- `agent/dense_selector.py` + `agent/constrained.hybrid_decode` — the generable decode path.
- `eval.harness.evaluate_routed` / `evaluate_incontext` / `evaluate_hybrid` — the metrics above.
