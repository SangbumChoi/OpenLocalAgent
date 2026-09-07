# CLAUDE.md

Project guidance for Claude Code. **Read [`AGENTS.md`](AGENTS.md) first** — it has the setup,
build/test commands and conventions, and is the source of truth shared with Cursor and Codex. This
file adds only Claude-specific sub-agent routing.

## TL;DR

Pure-PyTorch tool-calling agent trained from scratch on open data. Seven tiers from `la-10m` to
`la-700m`, including a sparse-expert arm with two matched dense controls, through
`pretrain → midtrain → posttrain`, scored on ten public agent benchmarks. Keep `pytest -q` green
and `ruff check src tests scripts demos` clean. Never bypass a model config's declared
`param_budget`. Report real eval numbers.

## Sub-agents (delegate by area)

Specialized sub-agents live in `.claude/agents/`. Prefer delegating focused work so each keeps a
tight context.

| When the task is about… | Use sub-agent |
|---|---|
| corpora, the `Conversation` schema, synthesis, source adapters | `data-engineer` |
| model architecture, the three training stages, KV cache, optimizers | `model-trainer` |
| the ten-benchmark suite, tool scoring, regression checks | `evaluator` |
| GGUF/ONNX/ExecuTorch export, quantization, parity, on-device perf | `exporter` |

The main thread stays the orchestrator: it plans, wires stages in `src/openlocalagent/pipeline.py`, and
integrates results. Sub-agents own *their* module and must leave the shared contracts intact — the
`Conversation` schema, the `Stage` enum, config YAMLs, the budget guard. They control their own
part, not the cross-cutting interfaces.

## Guardrails

- Do not add heavy ML frameworks (`transformers`, `trl`, `deepspeed`, …) to the training path.
- Do not edit the `Conversation` schema, `openlocalagent/stages.py`, the prompt contract, or a model
  config without saying so explicitly — they are the cross-cutting contracts, and other sub-agents
  depend on them.
- Scripts start with one of eight verbs and shell scripts `set -euo pipefail`; both are enforced by
  `tests/test_scripts_naming.py`.
- Long training runs go on the GPU box, detached, via `scripts/train_queue.sh`, supervised by
  `tools/kfsupervise.py` — the box stops every few hours regardless of load. Use
  `scripts/speedrun.sh` for a CPU smoke check.
- Artifacts (`runs/`, checkpoints, `*.png` outside `docs/figures/`) are git-ignored — surface
  them, don't commit them.
- Deleted campaign code lives under the `pre-refactor` tag. Recover from there rather than
  rewriting it from memory; see `docs/CAMPAIGN.md`.

## Quickest signal

`pytest -q`, then `bash scripts/speedrun.sh`.
