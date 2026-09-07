# AGENTS.md — guide for coding agents (Codex, Cursor, Claude Code, …)

The cross-tool instructions file. Claude Code also reads `CLAUDE.md`, which defers to this one and
adds sub-agent routing. Cursor rules live in `.cursor/rules/`. Keep all three in sync.

## What this project is

LocalAgent trains a tool-calling agent **from scratch** in **pure PyTorch** on **open data**, in
seven sizes from `la-10m` to `la-700m` — including a sparse-expert arm with two matched dense
controls — through three stages, scored on ten public agent benchmarks.

Read **[`docs/CONVENTIONS.md`](docs/CONVENTIONS.md)** first. It maps the tree, gives the grep-able
naming patterns, and lists exactly which files to touch to add a model tier, an eval suite, a data
source or a training stage. Then [`README.md`](README.md), [`docs/STAGES.md`](docs/STAGES.md),
[`docs/DATASETS.md`](docs/DATASETS.md), [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).
[`docs/README.md`](docs/README.md) indexes the rest.

## Setup / build / test

```bash
pip install -e ".[dev]"
pytest -q                           # 921 passed, 7 skipped — keep it green
ruff check src tests scripts demos  # tools/ is git-ignored and out of scope
```

## Run the pipeline

```bash
openlocalagent model-info configs/model/la-93m.yaml       # params vs declared budget
bash scripts/speedrun.sh                              # toy end-to-end smoke, CPU OK

openlocalagent train pretrain  configs/pretrain/la-93m.yaml
openlocalagent train midtrain  configs/midtrain/la-93m.yaml
openlocalagent train posttrain configs/posttrain/la-93m.yaml

openlocalagent-eval-suite --model openlocalagent:results/runs/posttrain-la-93m/latest.pt \
                      --out results/evalsuite-full/la-93m.json --device cuda --batch-size 32
```

On the training box, one command runs a tier's whole spine and scores it:

```bash
scripts/train_queue.sh 0 pretrain:93m midtrain:93m posttrain:93m evalsuite:93m
```

`notebooks/training_101.ipynb` walks one full session end to end at toy scale.

## Map of the code

| Area | Path |
|---|---|
| Model — decoder, KV cache, `ModelConfig` + budget, tokenizer | `src/openlocalagent/model/` |
| **Data contract** — `schema`, `prompt_contract`, `render`, `conversation_artifact`, `tool_catalog` | `src/openlocalagent/data/` |
| Per-source adapters — one module per external dataset | `src/openlocalagent/data/adapters/` |
| Corpus building and freezing | `src/openlocalagent/data/corpus/` |
| Synthetic agent data | `src/openlocalagent/data/synth/` |
| Denylists and eval-split isolation | `src/openlocalagent/data/decontam/` |
| Training — the three stages, loss curve, loop utilities | `src/openlocalagent/train/` |
| Eval — `suite.py` (the ten), `harness.py` (synthetic), tool scoring | `src/openlocalagent/eval/` |
| Agent — runtime loop, tools, parser, retriever | `src/openlocalagent/agent/` |
| Inference — KV-cache generate, ONNX/GGUF/ExecuTorch export | `src/openlocalagent/inference/` |
| Stage vocabulary | `src/openlocalagent/stages.py` |
| Runs one stage | `src/openlocalagent/pipeline.py` |
| Corpus acquisition, launchers, analysis | `scripts/` |
| Browser demo, status page, CLIs | `demos/` |
| Current documentation (campaign history in `docs/campaign/`) | `docs/` |

`data/` holds only what the whole pipeline depends on. Anything specific to one external source
belongs in `data/adapters/`, so the contract is never buried among them.

## Conventions

### The contracts — a change to one usually touches all three

- **One interchange format:** `openlocalagent.data.schema.Conversation` / `ToolSpec`. Data flows
  source → train → eval through it. Add a field there, never in an ad-hoc dict.
- **Stage names come from `openlocalagent.stages.Stage`** — `pretrain`, `midtrain`, `posttrain`, plus
  `distill` / `rl` / `eval`. Never reintroduce a bare stage string; the enum carries the entry
  point and the resume capability. `Stage.parse` accepts `"sft"` as the older spelling, and
  `Stage.POSTTRAIN.checkpoint_stage` is still `"sft"` because existing checkpoints and lineage
  receipts carry that value.
- **Pin the prompt contract in every training config.** `openai_full_catalog_v1` is what the eval
  suite renders. Leaving the key absent silently selects `legacy`, and a model trained on one
  prompt shape and scored on the other reads as a capability failure rather than a configuration
  one. This has already cost a full round of runs.

### Naming, so grep works

| Pattern | Means |
|---|---|
| `configs/model/la-<size>.yaml` | a ladder tier |
| `configs/<stage>/la-<size>.yaml` | one tier's one stage |
| `runs/<stage>-la-<size>/curve.jsonl` | that run's per-step loss curve |
| `results/evalsuite-full/<model>.json` | a ten-benchmark receipt |
| `results/evalsuite-full/<model>-<label>.json` | a superseded receipt, kept |
| `scripts/<verb>_*` | one of eight verbs, below |

**Scripts start with one of eight verbs** — `fetch_`, `normalize_`, `build_`, `train_`, `eval_`,
`analyze_`, `export_`, `archive_` (`speedrun.sh` is the one entry-point exception).
`tests/test_scripts_naming.py` enforces the set and requires a docstring or header comment in the
first six lines, so a ninth prefix fails the suite rather than starting the next twenty.

**Shell scripts** declare `#!/usr/bin/env bash`, `set -euo pipefail`, resolve the repository root
from `BASH_SOURCE` rather than assuming a working directory, and take data roots from an
environment variable with a default. The same test enforces all four. `set -u` alone is the trap
worth naming: it lets a failed download scroll past and the script exit 0.

### Everything else

- **Pure PyTorch only.** No `transformers` / `trl` / DeepSpeed in the training path.
- **Configs over flags.** Model, stage and data settings live in `configs/*.yaml`, one directory
  per stage. No hardcoded endpoints, paths or constants.
- **Midtrain and posttrain are disjoint by construction.** One question decides which side a corpus
  is on: does it correspond to a suite the harness scores? Yes is posttrain, no is midtrain. That is
  what makes midtrain a measurement rather than a rehearsal.
- **Budget ordering is pretrain > midtrain > posttrain.** Posttrain makes roughly one pass over the
  benchmark splits — fitting them, not drilling them.
- **Budget guard:** every model config declares `param_budget` and must pass
  `ModelConfig.assert_within_budget()`. Raise it in the config if a tier genuinely needs more —
  there is deliberately no environment override.
- **Every training stage writes `<out_dir>/curve.jsonl`**, one record per step, *while it runs*, not
  at the end. A new stage should too.
- **Figures go to `docs/figures/`** via `openlocalagent.figs.savefig`, never written by hand.
- **`tools/` is git-ignored** operator tooling for the training box. Nothing under `src/` imports it.
- **Honest numbers.** Unimplemented code raises `NotImplementedError`. Don't fake results; if an
  eval isn't 100%, report the real number, and say when a metric is a projection of a published one
  rather than the published metric.
- **Determinism in data:** synthetic targets are canonical (sorted-key compact JSON) and train/eval
  slot pools are disjoint. Both are what make exact-match eval meaningful.
- Style: ruff, line length 100, type hints on public functions, match the surrounding code.

## Things that will bite you

Each of these cost a real debugging round.

- **`--resume` is the strict form.** It requires an existing `latest.pt` and fails without one.
  Launchers rely on `runtime.resume: true` in the config, which resumes if present and starts clean
  if not.
- **The training box does not stay up.** It stops roughly every three to four hours regardless of
  load — a lifetime cap, not an idle culler, so touching the kernel does not prevent it. Run
  `tools/kfsupervise.py` alongside a long campaign; it restarts the notebook and relaunches the
  queues. Every stage resumes, so nothing is lost but wall clock.
- **Verify a sync landed, and that it *converged*.** Two separate failures. A push that fails on an
  expired cookie leaves the box running the old file while you reason about the new one — three
  identical failures in a row against a config already fixed locally. And extract-over never
  deletes, so a module the refactor removed keeps living on the box; a stale *package* then
  outranks the module that replaced it. `src/openlocalagent/pipeline/` shadowed `pipeline.py` that way
  and killed every queued job on an `ImportError` while the supervisor cheerfully relaunched them
  every three minutes. The sync now ships a manifest of local `src/**/*.py` and prunes the rest,
  bytecode included.
- **Test the precision the box actually trains in.** The whole suite runs fp32 on CPU, so a dtype
  bug that only exists under autocast passes locally and dies on the first H100 step — the MoE
  accumulator did exactly that. `torch.autocast("cpu", dtype=torch.bfloat16)` reproduces it on a
  laptop; a new layer that allocates its own buffer deserves that test.
- **Don't put code that produces a reported number in `scripts/` only.** The eval suite used to live
  there and a sync silently destroyed two receipts; see `docs/CAMPAIGN.md`.
- **An unrecognised `--suites` name is a hard error**, on purpose. Don't soften it back to a skip.
- **The contamination guard is usually right.** When it rejects a stage, the holdout genuinely
  overlaps the training sources. Find a clean split or report no holdout — do not disable it.
- **`openai_full_catalog_v1` refuses to truncate.** A conversation whose rendered row exceeds
  `seq_len` aborts the stage, because a row missing part of its own catalog has an unreachable
  answer. Filter with `scripts/build_context_fitted.py`; it reports what it dropped.
- **Retire a run with `scripts/archive_run.sh`**, which moves it and its receipt together. A
  published number needs the weights that produced it — renaming the receipt and deleting the run
  leaves a score nobody can re-score.
- **Artifacts** (`runs/`, checkpoints, `*.png` outside `docs/figures/`) are git-ignored. Surface
  them rather than committing them.

## Before you commit

- `pytest -q` green, `ruff check src tests scripts demos` clean.
- If you changed a model config, paste the `openlocalagent model-info` line in the PR/commit.
- Keep structural and behavioural changes in separate commits — a mixed diff cannot be reviewed or
  reverted cleanly.
- Deleted campaign code is recoverable from the `pre-refactor` tag; recover from there rather than
  rewriting it from memory.
