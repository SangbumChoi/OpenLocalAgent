# Data, by stage

Everything the ladder trains on is openly licensed and publicly available. Nothing here is
redistributed from this repository: the corpora are built on the training box by the ingestion
scripts in `scripts/`, and only manifests, hashes and counts are committed.

## Stage 1 — pretrain

`data/shards/pt-big`, packed at `seq_len` 2048 against `data/tokenizer-h100-16k.json` (16,384 BPE).

| Source | Train tokens | License |
|---|---|---|
| FineWeb-Edu (deduplicated) | 260,131,126 | ODC-BY-1.0 |
| Cosmopedia v2 | 69,893,410 | Apache-2.0 |
| Permissive Python | 135,474,879 | MIT / Apache-2.0 / BSD-2 / BSD-3 / ISC |
| **Total** | **460,691,205** | |

That is enough for the three smaller tiers at roughly Chinchilla-matched budgets and **not** enough
for `la-300m` upward.

`data/shards/pt-xl` extends it: eight FineWeb-Edu-dedup parquet files at roughly 0.94B tokens each,
~7.5B tokens in total, built by `scripts/fetch_fineweb_xl.sh`. The measurement that makes this
practical, and that is not guessable: through the Nexus HuggingFace mirror a **whole-file GET runs
at ~302 MB/s** (2.4GB in 7.9s) while a **range read of the same file runs at ~61 KB/s**. Never
range-read that mirror. Its `/api/**` is still 400, so filenames come from the release pattern
rather than a tree listing.

There are other shard pools on the box (`pool-general`, `pool-16k`, `h100-mix`) built from chat and
distillation mixtures rather than open web text. They are not what the ladder pretrains on.

## The line between midtrain and posttrain

One question decides which stage a corpus belongs to: **does it correspond to a suite the
ten-benchmark harness scores?**

- **Yes** → posttrain. The model is meant to learn those surfaces.
- **No** → midtrain. Whatever it buys has to appear as transfer, which makes the stage a
  measurement rather than a rehearsal.

The two are therefore disjoint by construction.

## Stage 2 — midtrain

Agentic data the harness does not score. Measured sizes, not estimates:

| Source | Rows | Bytes | Tokens | Licence |
|---|---|---|---|---|
| distill_reasoning | 40,812 | 127.0 MB | 40.7 M | derived, internal |
| free_episodes | 9,072 | 27.0 MB | 8.8 M | derived, internal |
| r0b0t_tools | 4,845 | 24.6 MB | 7.7 M | permissive |
| toucan | 5,911 | 13.1 MB | 4.3 M | permissive |
| device | 2,251 | 4.8 MB | 1.4 M | permissive |
| **Total** | **62,891** | **196.5 MB** | **62.9 M** | |

That is small for the middle stage, and it is being grown from permissively-licensed open sources
that are not benchmark splits — `scripts/fetch_midtrain_open.sh`:

| Source | Licence | Axis |
|---|---|---|
| `microsoft/orca-agentinstruct-1M-v1` — `tool_use`, `webagent_flow`, `code_`, `follow_up`, `rag`, … | CDLA-Permissive-2.0 | tool, **web**, python, task comprehension |
| `xingyaoww/code-act` | Apache-2.0 | python control |
| `nvidia/When2Call` | CC-BY-4.0 | when *not* to call a tool |
| `interstellarninja/hermes_reasoning_tool_use` | Apache-2.0 | tool use with reasoning |

Measured 2026-09-03 and normalized by `scripts/normalize_midtrain_open.py`: orca-agentinstruct
tool_use 50,000 · webagent_flow 25,000 · code 99,984 · follow_up 99,046 · task 125,000, code-act
7,139 rows. **When2Call and hermes are refused**: 2,243 of When2Call's 3,215 tool names are in
the xlam eval catalog, and hermes is built from When2Call, ToolAce and xLAM. They belong to the
scored side of the line. The widened mixture is `configs/midtrain/la-93m-matrix-openmid.yaml`;
see `docs/DATA_MIXTURE.md`.

**Excluded on purpose**, each for a stated reason:

- `Salesforce/APIGen-MT-5k`, `osunlp/UGround` — CC-BY-NC. Same call as WebLINX.
- `internlm/Agent-FLAN` — Apache-2.0, but derived from ToolBench, which the harness scores. It
  belongs on posttrain's side of the line.

## Stage 3 — posttrain

The benchmarks' own train splits and the curated and synthetic sets built from them:

| Source | Rows | Bytes | Tokens |
|---|---|---|---|
| androidcontrol | 82,944 | 269.2 MB | 84.6 M |
| agentnet (catalog-repaired) | 49,309 | 194.3 MB | 61.3 M |
| xlam | 24,000 | 66.1 MB | 21.5 M |
| merged-v2 | 16,300 | 56.1 MB | 17.6 M |
| toolace | 8,044 | 28.6 MB | 9.0 M |
| mobileactions | 8,692 | 22.6 MB | 6.8 M |
| toolbench | 6,000 | 13.4 MB | 4.1 M |
| mind2web | 3,028 | 11.3 MB | 3.9 M |
| toolsandbox (synthetic) | 1,066 | 4.2 MB | 1.2 M |
| **Total** | **199,383** | **665.9 MB** | **210.0 M** |

**Posttrain has no in-training holdout.** `merged-v2/eval` was drawn from the same public pools
this stage trains on — 1,834 overlapping rows, 3,214 overlapping rendered prompts — and every other
candidate sits in either this mixture or midtrain's. Rather than quote a number from a contaminated
split, the stage reports only its loss curve and evaluation is the ten-benchmark suite.

**BFCL and MCP-Atlas appear in no training stage at all** — they are evaluation-only by repository policy and have no training split here, so their
scores are pure transfer.

AgentNet enters through `data/public/agentnet-train-catalog.jsonl`: the train split with its own
eight-action desktop space attached, because it ships `tools: []` upstream and so cannot satisfy
`openai_full_catalog_v1`. `scripts/build_agentnet_catalog.py` builds it — 49,309 rows, 0 rejected.
**Only the train split is repaired.** The eval split keeps its empty catalog, since that is what
the open baselines were scored against and changing it would inflate our numbers against theirs.

## Budgets

Compute per stage follows **pretrain > midtrain > posttrain**:

| Tier | pretrain | midtrain | posttrain | posttrain passes |
|---|---|---|---|---|
| la-45m | 1.00 B | 328 M | 210 M | 1.00x |
| la-93m | 2.10 B | 393 M | 262 M | 1.25x |
| la-300m | 3.15 B | 524 M | 328 M | 1.56x |
| la-700m | 4.19 B | 655 M | 393 M | 1.87x |

Posttrain makes roughly one pass over the benchmark splits — fitting them once, rather than
drilling data drawn from the suites being scored.

## Contamination

Two habits keep the ten-benchmark number meaningful:

- **Train and eval splits are disjoint by construction**, and the Mind2Web eval rows are held out
  from the pinned *train* file by rendered-prompt hash, since repository policy keeps the official
  test split out of the checkout.
- **AndroidControl is the known exception.** Dropping the screenshot collapses distinct screens
  onto identical instruction strings, so roughly a quarter of its test rows have an exact
  (instruction, action, arguments) twin in training. This affects every model trained on these
  corpora — ours and the fine-tuned open baselines alike — so the comparison stays symmetric, but
  `scripts/eval_androidcontrol_clean.py` scores the clean subset and that is the honest number to quote.
- **Tool-name overlap is measured, not assumed.** `scripts/analyze_name_coverage.py` reports, per
  suite, how much of the gold tool vocabulary a training corpus already contains. A suite whose
  gold names all appear in training measures recall of a memorised mapping; one whose names are
  unseen measures reading the catalog. `xlam_opaque` / `xlam_shuffled` (see
  [BENCHMARKS.md](BENCHMARKS.md)) separate the two directly.

## Rebuilding

```bash
python scripts/fetch_pretrain_mixture.py    # fetch open corpora
python scripts/build_corpus.py               # filter, dedup, pack to shards
python scripts/normalize_public_agent.py configs/data/public-agent-training.example.yaml
```

Each writes a manifest carrying source token counts, license counts and a split-assignment hash, so
a corpus can be checked against the one a result was produced on.
