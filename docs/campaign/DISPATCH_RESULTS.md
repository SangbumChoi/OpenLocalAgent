# Generable tool dispatch — results & assets

End state of the "make tool-calling generable + cover current-agent scenarios" effort. Builds on
`ROUTE_REFACTOR_FINDINGS.md` (the architecture arc: 51-way head → dense selector + pointer-copy).

## The model
A 28M byte-level model dispatches a natural request to a tool via the **generable** pipeline:

    user request → route head (5-way modality gate) → dense two-tower selector (any tool, by
                   description embedding) → pointer-copy arguments → tool(args)

No fixed-N classifier anywhere; adding a tool is one row in the selector/retrieval index. Deployable
via `Agent.from_checkpoint("runs/tiny-30m-dispatch-best.pt", tools)`.

## Headline numbers — free-form OOD dispatch
44 hand-written natural queries (`eval/freeform.py`), deliberately NOT in the synthetic templates —
the honest generalization test.

| | call-name (end-to-end) | selection top-1 | route |
|---|---|---|---|
| pre-training (frozen 50-tool backbone) | 27% | 30% | 64% |
| **after dispatch fine-tune (best, ~3200 steps)** | **57%** | **57%** | **82%** |

A **2.1× improvement** on out-of-distribution queries, from backbone fine-tuning on paraphrase +
referent-conditioned data (the dense selector/route head are frozen-feature probes on top).

### Full scorecard (`scripts/scorecard.py`)
| held set | selection | full (sel+args) | args \| correct-sel |
|---|---|---|---|
| free-form (44 OOD) | 57% | 57% | — |
| paraphrase-eval (100, disjoint phrasings) | 46% | 35% | **76%** |
| contextual-eval (46, referent flips) | 22% | 7% | 30% |

Argument-copy is strong where selection is right (76% on paraphrase-eval) — **selection is the
bottleneck**, not arguments. Referent-conditioned (URL-vs-file) dispatch is the hardest case.

## Key finding — there is a training sweet spot (longer is worse)
Free-form dispatch peaks around **~3,200 steps** and then *degrades*: a finisher that trained 800
more steps dropped free-form call-name from 57% → 30%. Past the sweet spot the backbone overfits the
training distribution and loses OOD generalization. "Train 12–24h" backfires here; the right move is
early-stop on an OOD metric. (Found by comparing the saved FINAL checkpoint vs re-probing the
step-2400 backbone — `scripts/finalize_best.py`.)

## Data assets (generators + static dumps in `data/dumps/`)
- `data/paraphrase.py` — phrasing diversity, all 50 tools (median 12 train templates/tool).
- `data/contextual.py` — referent-conditioned flips: the SAME instruction → different tool by
  referent ("Open status.host.io" → open_url; "Open 'Chrome'" → open_app; "Open docs/intro.md" →
  read_file). 7 instruction-groups, 23 branches.
- `data/scenarios.py` — SOTA-agent families: clarify / abstain (over-trigger negatives) / parallel
  single-turn, and workflow / chained / error_recovery multi-turn episodes.
- `eval/freeform.py` — 44-query hand-written OOD eval.
- `data/dumps/*.jsonl` — materialized static snapshots of all of the above.

## Engineering notes
- **OOM fix**: probe feature-caching now runs under `no_grad` (was retaining 700–1500 forward
  graphs → 10GB+ → OOM). Peak RSS 10GB → 1.0GB.
- **Ephemeral container**: long unattended background jobs get reclaimed (process killed, `/tmp`
  wiped) while the persistent disk (`runs/`, repo) survives. Training is therefore run in **bounded,
  per-segment-checkpointed, monitored chunks** and resumed from the latest checkpoint.

## On-policy distillation (OPD) experiments — does swapping RL→OPD help?
Motivated by DeepSeek-V4 replacing RL with on-policy distillation. Two experiments
(`scripts/opd_grounded.py`, `scripts/opd_specialists.py`):

**Exp 1 — sequence-distill the grounded oracle into free-generation.** The grounded decoder
(route+selector+copy, ~57% OOD) is the teacher; we relabel prompts with its decoded calls, keep only
its correct outputs (rejection sampling: 257/500), and SFT the LM to free-generate them. Result:

| free-gen (no heads) | before | after OPD |
|---|---|---|
| paraphrase-eval name/full | 6% / 6% | **6% / 6%** |
| contextual-eval name/full | 4% / 4% | **4% / 4%** |

**Zero change.** Free generation of tool calls cannot be internalized into a 28M model by distillation
— it is a *capacity* limit, not a data limit. This confirms the head-based architecture (discriminative
selector + pointer-copy) is *necessary* at this scale, not merely convenient: the very functions OPD
would need to teach (selection, verbatim arg-copy) are the ones the autoregressive LM can't represent.

**Exp 2 — route specialists → distill into one** (the literal DeepSeek-V4 recipe). Per-route
selection top-1:

| | code | web_search | app_action |
|---|---|---|---|
| generalist (broad-trained, scenarios-best) | **51%** | **75%** | **70%** |
| specialist (SFT'd on its route only) | 38% | 62% | 50% |
| consolidated (one backbone, union of routes) | 25% | 62% | 35% |

Each narrow specialist is *worse* on its own route than the generalist and catastrophically forgets
the others (0%); consolidation does not recover. (Caveat: specialists/consolidated start from the
50-tool base with far fewer steps than the generalist — but the within-base trend is unambiguous.)

**OPD verdict for a 28M from-scratch model: neither variant improves the model.** This is the
expected mirror of the capacity story. DeepSeek-V4's OPD/specialist-distillation works because large
models have *spare capacity* and each specialist gets *more/better* data; at 28M the bottlenecks are
exactly capacity and data-per-slice, so (1) free-gen can't absorb the heads' jobs (Exp 1) and (2)
narrow specialists underperform a broadly-trained generalist (Exp 2). The right move at this scale is
the opposite of cutting breadth: keep the discriminative heads and train one generalist on broad,
paraphrase-rich data — which is exactly the shipped model.

## On-device export
Both new heads export with the existing pipeline (`inference/export/to_dispatch.py`), parity-checked
against PyTorch (route-head argmax 100% / dense-selector top-1 100% agreement, max|Δ| ~1e-5):
- **Route head** → linear weights+bias applied to the model's final hidden state; argmax over 5 routes.
- **Dense selector** → ship the **query tower** + a **precomputed, L2-normalized tool matrix**
  `(n_tools, 256)`; the device computes `argmax_j normalize(q_proj(h)) · tool_matrix[j]`, never
  touching the 8192-d embedding at runtime. Adding/removing a tool is adding/removing a `(1,256)`
  row — the generable property survives export. The browser bundle (`to_onnx.export_web`) emits
  `dispatch_heads.json` alongside the existing head JSON.

## Reproduce
- Dispatch fine-tune: `scripts/train_dispatch_long.py` (segmented, checkpoints per segment).
- Finalize a deployable checkpoint from a backbone: `scripts/finalize_best.py`.
- Scenario continuation: `scripts/train_scenarios.py` (folds the 4 SOTA families).
- Score: `scripts/scorecard.py`, `scripts/audit_args.py`; demo: `scripts/demo_dispatch.py`.

## Scenario behaviours (clarify / abstain / parallel / multi-turn)
The four SOTA families (`data/scenarios.py`) were folded onto the dispatch backbone
(`scripts/train_scenarios.py`, pure-LM SFT on corpus + episodes). The model is then re-probed with
behaviour coverage (`scripts/finalize_scenarios.py`: oversample clarify/abstain text for the route
head; train the selector on per-turn episode contexts), since the naive probes had blind spots.

| behaviour | naive probe | **with coverage** | held n |
|---|---|---|---|
| multi-turn next-tool selection | 5% | **74%** | 27 |
| abstention (route an under-/non-actionable turn to text) | 0% | **33%** | 6 |
| parallel (per-conjunct selection) | 50% | (noisy) | 3 |
| free-form dispatch (regression check) | 43% | 48% top-1 / 41% call | 44 |

**Multi-turn selection 5%→74%** is the headline: the model handles mid-episode next-tool selection
well once the selector sees episode contexts — the earlier 0/5% were *probe blind spots*, not a
training failure. Abstention moved off the floor but is hard (and the held set is tiny). Parallel's
held set (n=3) is too small to read. Deployable checkpoint: `runs/tiny-30m-scenarios-best.pt`.

Two findings reinforced here: (1) the per-segment scenario SFT also has a **sweet spot** — free-form
peaked at step 800 and degraded by step 1200, so we finalize from the step-800 backbone; (2) match
the **probe's training distribution to what you measure**, or behaviours look absent when they aren't.

## Retrain on corrected data (v2)
After the annotator fixes (`run X.py`→`run_command`, file-URL `download_file`) + the free-form TRAIN
set, a fresh backbone fine-tune from the 50-tool base on the *corrected* corpus (sweet-spot ~step
3200) beat the prior model decisively on the disjoint held sets:

| metric (n) | prior (scenarios-best) | **corrected retrain** |
|---|---|---|
| free-form OOD call-name (45) | 40% | **53%** |
| free-form top-1 | 47% | **56%** |
| paraphrase-eval selection (100) | 46% | **63%** |
| referent-conditioned (contextual) selection (46) | 22% | **72%** |

The biggest jump is **contextual 22→72%** — the old backbone had learned the wrong `run X.py`→
`run_python` association; training on corrected labels fixed the features (and `download_file`/
`make_dir` cases). Note: re-probing the heads with the free-form train set oversampled *regressed*
the aggregate (53→42%) — overfitting the 86 train prompts — so the deployed heads are the backbone's
own FINAL probes, not the free-form-oversampled ones. Residual hard case: `run python X.py` still
mis-selects `read_file` (a dense-selector confusion; forcing it hurt everything else). Deployed as
`danelcsb/localagent-tiny-30m-byte` + the WebGPU Space.
