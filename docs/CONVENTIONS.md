# Conventions: finding things, and adding things

Two jobs. The first half tells you where anything lives and what to grep for. The second is a set
of recipes: for each kind of change, every file you have to touch, in order.

---

## Finding things

### One directory, one job

| Path | Holds | Does **not** hold |
|---|---|---|
| `src/openlocalagent/model/` | decoder, KV cache, `ModelConfig`, tokenizer | anything that reads a dataset |
| `src/openlocalagent/data/` | the interchange contract only — `schema`, `prompt_contract`, `render`, `conversation_artifact`, `tool_catalog` | anything source-specific |
| `src/openlocalagent/data/corpus/` | building and freezing pretraining corpora | agent data |
| `src/openlocalagent/data/synth/` | generating synthetic agent data | reading external sources |
| `src/openlocalagent/data/adapters/` | one module per external source | anything the whole pipeline depends on |
| `src/openlocalagent/data/decontam/` | denylists, eval-split isolation | training |
| `src/openlocalagent/train/` | the three stages, loss curve, loop utilities | scoring |
| `src/openlocalagent/eval/` | `suite.py` (ten benchmarks), `harness.py` (synthetic), tool scoring | training |
| `src/openlocalagent/agent/` | runtime loop, tools, parser, retriever | model internals |
| `src/openlocalagent/inference/` | KV-cache generation, ONNX/GGUF/ExecuTorch export | training |
| `src/openlocalagent/stages.py` | the `Stage` vocabulary | anything else |
| `src/openlocalagent/pipeline.py` | runs one stage | stage implementations |
| `configs/model/` | the ladder, one file per tier | training settings |
| `configs/{pretrain,midtrain,posttrain}/` | one file per tier per stage | model shape |
| `scripts/` | corpus acquisition, launchers, analysis | anything that produces a reported number |
| `demos/` | pages and CLIs a human looks at | library code |
| `notebooks/` | teaching walkthroughs | anything the pipeline imports |
| `docs/` | current documentation | campaign history — that is `docs/campaign/` |

**Anything that produces a number this project reports belongs in `src/`, not `scripts/`.**
`docs/CAMPAIGN.md` records what it cost to learn that.

### Names you can grep

Naming is regular on purpose, so one `grep` answers "where is X".

| Pattern | Means | Example |
|---|---|---|
| `configs/model/la-<size>.yaml` | a ladder tier | `la-93m`, `la-300m-moe` |
| `configs/<stage>/la-<size>.yaml` | one tier's one stage | `configs/midtrain/la-93m.yaml` |
| `configs/<stage>/la-<size>-<arm>.yaml` | an ablation arm, generated from its base | `la-93m-s2`, `la-93m-matrix` |
| `runs/<stage>-la-<size>/` | that run's output | `results/runs/posttrain-la-93m/` |
| `runs/<stage>-la-<size>/curve.jsonl` | its per-step loss curve | always this filename |
| `results/runs/archive/<label>/` | a retired run, with its receipt | `scripts/archive_run.sh` |
| `results/evalsuite-full/<model>.json` | a ten-benchmark receipt | `la-93m.json` |
| `results/evalsuite-full/<model>-<label>.json` | a superseded receipt | `la-45m-recipe-v1.json` |
| `explog/<stage>-la-<size>.log` | that job's stdout on the box | `explog/midtrain-la-45m.log` |
| `scripts/<verb>_*` | see the verb table below | `fetch_fineweb_xl.sh` |

So: *which tiers have a midtrain config?* `ls configs/midtrain/la-*`. *What did la-93m's posttrain
do?* `results/runs/posttrain-la-93m/curve.jsonl`. *What produced this number?* the receipt's `location`
field names the checkpoint.

### Script names are a closed verb set

Every file in `scripts/` starts with one of eight verbs, so `ls scripts/<verb>_*` answers "where is
the thing that does X". `tests/test_scripts_naming.py` enforces it — a ninth prefix fails the suite
rather than starting the next twenty.

| Verb | Does | Example |
|---|---|---|
| `fetch_` | acquire bytes from an external source | `fetch_fineweb_xl.sh` |
| `normalize_` | turn one external source into `Conversation` rows | `normalize_tau2.py` |
| `build_` | derive a dataset or artifact from data already local | `build_corpus.py` |
| `train_` | run or launch training | `train_queue.sh` |
| `eval_` | score a model | `eval_toolcall.py` |
| `analyze_` | produce a figure or report from results | `analyze_throughput.py` |
| `export_` | emit an artifact for somewhere else | `export_to_hf.py` |
| `archive_` | retire a run and its receipt | `archive_run.sh` |

`speedrun.sh` is the one exception: a one-command entry point, named for what it is.

The same test also requires a docstring or header comment in the first six lines, so `head -3` is
enough to tell whether a script is the one you want.

### Configs that are not ladder tiers

`configs/model/` also holds names outside the `la-*` ladder — `webgpu-*`, `tiny-30m*`, `la64k-*`.
Most are not dead: they are the **provenance of published receipts**, and deleting one leaves a
number in `docs/campaign/` that nothing explains. `tests/test_config_hygiene.py` draws that line
automatically — a model config referenced by no code, no test, no doc *and* no receipt fails the
suite, and everything else stays.

The check is transitive, which matters more than it sounds: removing `webgpu-10m-convheavy-4k`
orphaned `webgpu-10m-convheavy`, whose only reference had been that file's name as a substring.

### Shell scripts

Every `.sh` in `scripts/` starts `#!/usr/bin/env bash`, sets `-euo pipefail`, resolves the
repository root from `BASH_SOURCE` rather than assuming a working directory, and takes data roots
from an environment variable with a default. `tests/test_scripts_naming.py` enforces all four.

`set -u` alone is the one worth spelling out: it lets a failed download scroll past and the script
exit 0. `fetch_midtrain_open.sh` ran that way, several `curl`s 404'd, and the run reported success
with files missing.

### Where the box tooling went

`tools/` is git-ignored. It holds operator scripts for driving the training box — kernel execution,
file push, the restart supervisor — with machine-specific paths and credentials from the
environment. Useful to whoever is driving the GPUs, not part of the pipeline, and nothing under
`src/` imports it.

### The three places that must agree

A change that touches one usually has to touch all three. They are the contracts:

1. **`Conversation` / `ToolSpec`** (`src/openlocalagent/data/schema.py`) — the one interchange format.
   Data flows source → train → eval through it. Add a field there, never in an ad-hoc dict.
2. **`Stage`** (`src/openlocalagent/stages.py`) — the stage vocabulary. Each member carries its own
   entry point and resume capability; there is no separate lookup table to keep in sync.
3. **The prompt contract** (`src/openlocalagent/data/prompt_contract.py`) — `openai_full_catalog_v1` is
   what the eval suite renders. Training under a different one scores the model on a prompt shape
   it never saw, and reads as a capability failure. Pin it in every training config.

---

## Adding things

### A new model tier

1. `configs/model/la-<size>.yaml` — declare `param_budget` in the file; there is deliberately no
   environment override.
2. `tests/test_model_ladder.py` — add `(name, exact_params, budget)` to `LADDER`. The count is
   written out, not recomputed, so a config edit that changes capacity fails here instead of
   silently re-labelling a published result.
3. `configs/{pretrain,midtrain,posttrain}/la-<size>.yaml` — keep the budget ordering
   **pretrain > midtrain > posttrain**.
4. `README.md` ladder table — include **active** parameters if they differ from total.

```bash
openlocalagent model-info configs/model/la-<size>.yaml   # params vs declared budget
pytest -q tests/test_model_ladder.py
```

### A new eval suite

1. Add the row file under `data/public/` and a `scripts/normalize_<suite>.py` that produces it. A
   suite you cannot rebuild is a suite you cannot check.
2. Register it in `SUITES` in `src/openlocalagent/eval/suite.py`, with a comment stating the **claim
   boundary** — whether the metric is the published one or a projection of it.
3. Document it in `docs/BENCHMARKS.md` with its row count and that boundary.
4. If it is evaluation-only, say so in the comment and keep it out of every training config.

An unknown `--suites` name is a hard error on purpose. Do not soften it back to a skip.

### A new training data source

1. Acquire it with a `scripts/fetch_*.sh`, recording the licence **before** anything downloads.
   Non-commercial terms (CC-BY-NC, CC-BY-NC-SA) are excluded — see the exclusions in
   `fetch_midtrain_open.sh`.
2. Normalise to `Conversation` rows with `scripts/normalize_<source>.py`.
3. Decide which side of the line it falls on, by one question: *does it correspond to a suite the
   harness scores?* Yes → posttrain. No → midtrain. They are disjoint by construction.
4. Verify it satisfies the prompt contract before adding it to a mixture — a corpus that cannot
   will fail the whole stage on its first offending row.
5. Add it to the mixture and update `docs/DATASETS.md` with rows, bytes and licence.

### A new training stage

1. Add the member to `Stage` in `src/openlocalagent/stages.py`. Its `entry_point` and
   `supports_exact_resume` are properties on the member. Both dispatches declare a return type, so
   forgetting a branch fails type-check rather than falling through.
2. Implement `run(config_path, *, resume=False)` in `src/openlocalagent/train/`.
3. Write `<out_dir>/curve.jsonl` from it — one record per step, while it trains, not at the end.
4. Add `configs/<stage>/`.

### A new figure

`from openlocalagent.figs import savefig; savefig(fig, "name")` puts it in `docs/figures/`. Do not
write plot files by hand — the gallery is one directory so the docs can reference it stably.

---

## Before you commit

- `pytest -q` green, `ruff check src tests scripts demos` clean.
- Structural and behavioural changes in **separate commits** — a mixed diff cannot be reviewed or
  reverted cleanly.
- If you changed a model config, paste the `openlocalagent model-info` line.
- If you retired a run, use `scripts/archive_run.sh` so its receipt goes with it. A published
  number needs the weights that produced it.
- Artifacts (`runs/`, checkpoints, `*.png` outside `docs/figures/`) are git-ignored.
