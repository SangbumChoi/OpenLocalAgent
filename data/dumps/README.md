---
license: mit
language: [en]
task_categories: [text-generation]
pretty_name: LocalAgent Dispatch Data
tags: [tool-calling, function-calling, agent, dispatch, synthetic, routing]
configs:
- config_name: paraphrase
  data_files:
  - {split: train, path: paraphrase_train.jsonl}
  - {split: eval,  path: paraphrase_eval.jsonl}
- config_name: contextual
  data_files:
  - {split: train, path: contextual_train.jsonl}
  - {split: eval,  path: contextual_eval.jsonl}
- config_name: scenarios_single
  data_files:
  - {split: train, path: scenarios_single_train.jsonl}
  - {split: eval,  path: scenarios_single_eval.jsonl}
- config_name: scenarios_episodes
  data_files:
  - {split: train, path: scenarios_episodes_train.jsonl}
  - {split: eval,  path: scenarios_episodes_eval.jsonl}
- config_name: freeform
  data_files:
  - {split: train, path: freeform_train.jsonl}
  - {split: eval,  path: freeform_eval.jsonl}
---

# LocalAgent Dispatch Data

Synthetic data for training/evaluating a **generable tool-dispatch** model over a 50-tool surface
(route head → dense selector → pointer-copy). A static snapshot of the deterministic generators in
[LocalAgent](https://github.com/sangbumchoi/localagent) (`src/localagent/data/`). Train/eval are
**disjoint in both phrasing and slot values**. Companion model + demo:
[danelcsb/localagent-tiny-30m-byte](https://huggingface.co/danelcsb/localagent-tiny-30m-byte) ·
[Space](https://huggingface.co/spaces/danelcsb/localagent-webgpu).

## Configs

| config | rows (train/eval) | what it is |
|---|---|---|
| `paraphrase` | 1000 / 1000 | many natural phrasings per tool (all 50 tools), single-turn |
| `contextual` | 230 / 230 | **referent-conditioned**: same instruction → different tool by referent ("Open status.host.io"→`open_url`, "Open 'Chrome'"→`open_app`, "Open docs/intro.md"→`read_file`) |
| `scenarios_single` | 60 / 40 | clarify / **abstain** (over-trigger negatives, no tool) / parallel (≥2 calls) |
| `scenarios_episodes` | 60 / 60 | multi-turn episodes: workflow / chained / error-recovery |
| `freeform` | 86 / 45 | hand-written natural queries — **train** (86) + out-of-distribution **eval** (45) |

## Schema

Single-turn (`paraphrase`, `contextual`, `scenarios_single`):
```json
{"prompt": "...", "kind": "tool|text", "category": "...", "tool": "<name|>",
 "target": "{\"arguments\":{...},\"name\":\"...\"}", "calls": [{...}]?}
```
- `kind="text"` (clarify/abstain) → no tool; `target` is the natural-language reply.
- `calls` present for parallel turns (≥2 tool calls). Tool-call arg values are **literal substrings
  of `prompt`** (so a copy/pointer mechanism can ground them).

Episodes (`scenarios_episodes`): `{"category": "...", "turns": [{"role", "content", "tool_calls":
[{"name","arguments"}], "tool_response"}]}`. Copy-args in follow-up calls appear verbatim in a prior
`tool_response`.

`freeform`: `{"prompt": "...", "tool": "<gold tool>"}`.

## Usage
```python
from datasets import load_dataset
ds = load_dataset("danelcsb/localagent-dispatch-data", "paraphrase", split="train")
```

Regenerate deterministically from source: `python scripts/export_dataset.py` in the LocalAgent repo.
