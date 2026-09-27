# Reliable tool calling (`ToolCaller`)

The developer-facing core: turn a natural-language turn into a **schema-valid, grounded** tool
call on *any* JSON-schema tools — or abstain. No model, no fine-tuning required.

## API

```python
from openlocalagent import ToolCaller
from openlocalagent.data.schema import ToolSpec

caller = ToolCaller(tools, retrieve_k=12, examples=None, min_score=0.0)
caller.call(query)        # -> ToolCall(name, arguments)  |  None  (abstain)
caller.candidates(query)  # -> [(ToolSpec, score), ...]   ranked candidates
caller.explain(query)     # -> {candidates, call}         debug view
```

- `tools` — a list of `ToolSpec(name, description, parameters)` where `parameters` is a JSON
  schema (`type/properties/required`, per-property `type`/`format`/`enum`).
- `examples` — optional `{tool_name: [example phrasings]}`; indexing tools by example usages
  bridges the paraphrase gap ("reserve a flight" → `book_flight`). Strongly recommended.
- `min_score` — abstain if the top retrieved tool's similarity is below this (tune per catalog;
  ~0.12 declined 75% of irrelevant queries in the benchmark).

## How it works (and why it's reliable)

1. **Select** by retrieval (`agent/retriever.py`): a char-n-gram embedding ranks all tools by
   similarity to the query. O(top-k) downstream, scales to thousands of tools, and **new tools work
   with zero retraining** (just add them).
2. **Ground** by schema-guided constrained decoding (`agent/schema_decode.py`): we *construct* the
   argument dict from the schema instead of free-generating JSON, so it's always valid. Each
   property is filled by a typed, arg-aware slot-filler:
   - `enum` → the member mentioned · `integer`/`number` → a number · `boolean` → on/off cues
   - `format: path` → a file path · `url` → a URL · `quoted` → a quoted span · `arithmetic` → an expr
   - entity-ish names → a proper-noun span · free text → a quoted span or the descriptive tail
   - **multi-argument**: same-typed values are pulled from one pool *in schema order*
     (`move_file(source, dest)` over two paths → source, dest).
3. **Validate** against the schema; if a required arg can't be grounded, that tool doesn't fill and
   the caller tries the next candidate, or **abstains**.

## Benchmark

`python scripts/eval_toolcall.py [--scale N] [--min-score S]` — 18 realistic multi-arg tools
(`eval/toolcall_bench.py`), paraphrased held-out queries (verb synonyms), disjoint train/eval
slot values, plus irrelevant queries for abstention.

| catalog | full-call | tool@1 | args-exact | abstention |
|---|---|---|---|---|
| 18 tools | 72% | 86% | 84% | — |
| 18 tools, `min_score 0.12` | 69% | 83% | 83% | 75% |
| 18 + 1,000 distractors | 58% | 72% | 81% | — |

*full-call* = correct tool **and** every argument exact. The honest gaps: (1) verb paraphrases the
retriever can't bridge unless example usages cover them (so always pass `examples`), and (2)
free-text multi-word arguments (a learned **pointer head** is the fix — see
`docs/ARCHITECTURE_IDEAS.md`).

## When to add a model

`ToolCaller` needs no model. To re-rank the retrieved candidates with the trained byte model (or
to use the tool/pointer heads on a small fixed toolset), pass them through `agent/constrained.py`'s
`grounded_decode`. For a large catalog, retrieval + this constrained decoder is the recommended,
reliable, training-free path.
