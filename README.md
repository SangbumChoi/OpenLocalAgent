# LocalAgent

A small tool-calling agent trained **from scratch**, in **pure PyTorch**, on **open data**.

Seven model sizes, three training stages, one evaluation suite of ten public agent benchmarks. No
training framework, no `transformers` in the training path, no closed corpora.

```
pretrain  ──▶  midtrain   ──▶  posttrain  ──▶  eval
open text      open agentic    curated        ten public
corpora        four axes       exactness      benchmarks
```

Midtrain is about **capability**, posttrain about **exactness**. Keeping them apart is what lets
midtrain read broad, messy, real agentic data without that messiness reaching the output format.

---

## The ladder

| Tier | Total params | Active params | d_model | layers | heads | FFN |
|---|---|---|---|---|---|---|
| `la-10m` | 10,524,544 | 10,524,544 | 384 | 4 | 6 | 512 |
| `la-45m` | 45,114,368 | 45,114,368 | 512 | 12 | 8 | 1,408 |
| `la-93m` | 93,108,608 | 93,108,608 | 640 | 18 | 10 | 1,664 |
| `la-150m` | 149,294,784 | 149,294,784 | 704 | 24 | 11 | 1,920 |
| `la-300m` | 298,535,808 | 298,535,808 | 1,152 | 18 | 18 | 3,200 |
| `la-300m-moe` | 298,701,696 | **149,402,496** | 1,152 | 18 | 18 | 400 × 8, top-2 |
| `la-700m` | 700,884,352 | 700,884,352 | 1,280 | 27 | 20 | 5,120 |

Every tier is the same hybrid backbone — two gated short-convolution mixers per attention layer,
QK-Norm, RoPE, SwiGLU, MQA — so the ladder varies capacity and not architecture. Twelve of
eighteen mixers in `la-93m` are constant-state convolutions, which is what holds the decode state
down at a 4K context.

### The sparse arm

`la-300m-moe` is the one place the ladder varies architecture, and it does so against a control.
An agent's competence at this scale looks **memory-bound rather than compute-bound**: what it has
to hold is a large, mostly-inert body of tool and API knowledge, and only a small slice of that is
relevant to any one request. A sparse routed FFN is the shape that matches — keep the knowledge
resident, pay for a fraction of it per token.

It has **two** controls, because "matched" means two things:

- `la-150m` matches its **active** parameters (−0.07%) — *does the extra stored capacity earn
  anything at equal compute?* This is the primary control, and the convention
  [docs/SPARSE_EXPERTS_REAL_DATA.md](docs/SPARSE_EXPERTS_REAL_DATA.md) set for this repository.
- `la-300m` matches its **total** parameters (+0.06%) — *at equal stored parameters, does routing
  beat spending them densely?*

Everything but the FFN is identical across all three. One honest limit: the ONNX exporter does not
lower dynamic expert routing, so a result here is a PyTorch finding about capacity versus compute,
**not** a demonstration that this tier runs on device.

Each config declares the `param_budget` it is held to. The three small tiers declare 100M, the
300M pair 350M, `la-700m` 750M — in version control, where a reviewer sees it.

```bash
openlocalagent model-info configs/model/la-93m.yaml
```

## Where it stands

The headline number is the **mean step-success rate over ten public agent benchmarks** —
AndroidControl, Mind2Web, ToolACE, xLAM, BFCL, AgentNet, ToolBench, ToolSandbox, MCP-Atlas,
MobileActions. Full definitions and the claim boundary on each in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

| Model | Params | Mean step-success |
|---|---|---|
| `ft-qwen3-06b` (open, fine-tuned) | 596M | 34.83 |
| `ft-lfm2-700m` (open, fine-tuned) | 742M | 33.14 |
| `ft-granite-h-350m` (open, fine-tuned) | 340M | 26.40 |
| **`la-45m`** (ours, from scratch) | **45.1M** | **25.61** |
| `fcall-96m` (previous best from-scratch) | 95.3M | 23.07 |
| `ft-danube3-500m` (open, fine-tuned) | 514M | 22.90 |
| `ft-mamba-790m` (open, fine-tuned) | 793M | 22.58 |

The `ft-*` rows all received the same fine-tuning — full parameters, 600 steps, lr 2e-5, the same
union corpus — so the table varies the model and not the recipe. **`la-45m` passes the previous
best from-scratch result at less than half its size**, and passes fine-tuned open models up to
793M. Full per-suite numbers and the claim boundaries in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Quickstart

```bash
pip install -e ".[dev]"
pytest -q                          # 894 passed, 7 skipped
ruff check src tests scripts demos

# toy end-to-end run, CPU is fine
bash scripts/speedrun.sh
```

Train one stage:

```bash
openlocalagent train pretrain configs/pretrain/la-93m.yaml
```

Evaluate against all ten suites:

```bash
openlocalagent-eval-suite --model openlocalagent:results/runs/posttrain-la-93m/latest.pt \
                      --out results/evalsuite-full/la-93m.json --device cuda --batch-size 32
```

Every stage writes a per-step loss curve to `<out_dir>/curve.jsonl` **as it trains**, so a live run
and a crashed one both have one:

```python
from openlocalagent.train.curve import read_curve
points = read_curve("results/runs/pretrain-la-93m")
print(points[-1].step, points[-1].loss)
```

## The three stages

All three optimise the **same objective** — next-token cross-entropy with `ignore_index=-100`,
implemented once in `model/transformer.py`. What separates them is **which tokens are targets**.
Nothing else about the loss changes.

| Stage | Mask | Tokens scored | Learns |
|---|---|---|---|
| pretrain | none | 100% | language |
| midtrain | per source | text 100%, conversations assistant-only | domain shift |
| posttrain | assistant body + EOS | few | what to emit |

Budget ordering is **pretrain > midtrain > posttrain**, held at every tier. Global batch is fixed
at 131,072 tokens/step across the whole ladder, so tiers differ in capacity and step count, never
in batch. Schedule is WSD; AdamW, `weight_decay 0.1` (posttrain 0.0), `grad_clip 1.0`,
`seq_len 2048`. Peak LR falls with size (6.0e-4 at 10M → 2.0e-4 at 700M); midtrain and posttrain
share a rate about a third of pretrain's.

> `curve.jsonl` records `loss` under the same name in all three stages, but the **denominators
> differ**. Posttrain's 1.63 is not "better" than pretrain's 2.42 — it is a smaller, far more
> predictable set of tokens. Compare within a stage, never across.

### 1 · pretrain — open text, no mask

`data/shards/pt-big`, 460,691,205 tokens packed at `seq_len` 2048 against a 16,384 BPE tokenizer:
FineWeb-Edu dedup (260.1M, ODC-BY-1.0), Cosmopedia v2 (69.9M, Apache-2.0), permissive Python
(135.5M, MIT/Apache/BSD/ISC).

That is Chinchilla-matched for the small tiers and **not** for the large ones. Measured:

| Tier | tokens | tok/param | epochs of pt-big | vs Chinchilla |
|---|---|---|---|---|
| la-93m | 2.10 B | 22.5 | 4.6 | 1.13× |
| la-150m | 2.10 B | 14.0 | 4.6 | 0.70× |
| la-300m | 3.15 B | 10.5 | 6.8 | 0.53× |
| la-700m | 4.19 B | 6.0 | 9.1 | **0.30×** |

Two things worsen together as tiers grow: tokens per parameter fall while repetition of the same
corpus rises. `scripts/fetch_pretrain_4b.sh` builds the ~4.4B mixture (adds C4, Wikipedia,
FineMath) that separates those two variables.

### 2 · midtrain — agentic data the harness does *not* score

Continued pretraining, not a second SFT. Text and code sources keep the full next-token loss;
canonical `Conversation` sources keep the assistant-only mask. 62,891 rows / 62.9M tokens:
`distill_reasoning` 40.7M, `free_episodes` 8.8M, `r0b0t_tools` 7.7M, `toucan` 4.3M, `device` 1.4M
— plus a **0.45-weight replay stream of pt-big**, which is what stops the base model collapsing
while the mixture shifts.

Capability axes are weighted by **sqrt(axis rows)**, not equally. Equal weighting gave web
control's 3,028 rows the same share as GUI control's 91,636 — a ~30× oversample that cost xlam
14.5 points and mobileactions 21.0.

### 3 · posttrain — the benchmarks' own train splits

199,383 rows / 210.0M tokens: androidcontrol 84.6M, agentnet 61.3M, xlam 21.5M, merged-v2 17.6M,
toolace 9.0M, mobileactions 6.8M, toolbench 4.1M, mind2web 3.9M, toolsandbox 1.2M. Roughly one
pass — fitting the surfaces, not drilling them.

**The line between stages is one question**: *does this corpus correspond to a suite the harness
scores?* Yes → posttrain. No → midtrain. Disjoint by construction, which is what makes midtrain a
measurement rather than a rehearsal. `Agent-FLAN` is Apache-2.0 and still excluded: it derives
from ToolBench, which is scored.

**BFCL and MCP-Atlas enter no training stage at all**, so those scores are pure transfer.
Posttrain has **no in-training holdout** — `merged-v2/eval` overlaps its own training pool by
1,834 rows — so the stage reports only its loss curve and evaluation is the ten-suite harness.

Nothing here is redistributed from this repository: corpora are built on the training box by
`scripts/`, and only manifests, hashes and counts are committed. Full tables in
[docs/DATASETS.md](docs/DATASETS.md); every suite's example, expected output and claim boundary in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Repo layout

| Path | What |
|---|---|
| `src/openlocalagent/model/` | decoder, KV cache, config + budget guard, tokenizer |
| `src/openlocalagent/data/` | `Conversation` schema, corpus building, synthesis |
| `src/openlocalagent/train/` | `pretrain.py`, `midtrain.py`, `sft.py`, loss curve, loop utilities |
| `src/openlocalagent/eval/` | `suite.py` (the ten benchmarks), `harness.py` (synthetic) |
| `src/openlocalagent/agent/` | runtime loop, tools, parser, retriever |
| `src/openlocalagent/inference/` | KV-cache generation, ONNX / GGUF / ExecuTorch export |
| `src/openlocalagent/stages.py` | the `Stage` vocabulary every surface uses |
| `configs/{pretrain,midtrain,posttrain}/` | one directory per stage |
| `configs/model/` | the ladder |
| `scripts/` | corpus building, training launchers, analysis — one of eight verb prefixes |
| `demos/` | `webgpu/` browser demo · `ladder/` status page · `chat_cli.py` · `web/` |
| `notebooks/` | [`training_101.ipynb`](notebooks/training_101.ipynb) — one full session end to end |
| `docs/` | [index](docs/README.md) · [conventions](docs/CONVENTIONS.md) · [stages](docs/STAGES.md) · [datasets](docs/DATASETS.md) · [benchmarks](docs/BENCHMARKS.md) |
| `docs/figures/` | the committed plot gallery — everything `openlocalagent.figs.savefig` writes |
| `docs/campaign/` | pre-refactor research notes and paper material, kept for provenance |

## Conventions

Where everything lives and what to touch to add a tier, suite, data source or stage:
**[docs/CONVENTIONS.md](docs/CONVENTIONS.md)**.

- **Pure PyTorch.** No `transformers` / `trl` / DeepSpeed in the training path. A BPE tokenizer
  library is the only model-adjacent dependency, and the byte-level tier avoids even that.
- **One interchange format**: `openlocalagent.data.schema.Conversation`. Data flows synth → train →
  eval through it. Add a field there, not in an ad-hoc dict.
- **Configs over flags.** Model, stage and data settings live in `configs/*.yaml`.
- **Honest numbers.** Unimplemented code raises `NotImplementedError`; an eval that is not 100%
  reports what it is; a metric that is a projection of a published one says so.

## History

This repository carried a long experiment campaign — roughly 250k lines at its peak — that has
been cut back to the maintained pipeline. Everything removed is recoverable from the
`pre-refactor` git tag; see [docs/CAMPAIGN.md](docs/CAMPAIGN.md) for what went and why.

## License

MIT. See [LICENSE](LICENSE).
