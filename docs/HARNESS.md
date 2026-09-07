# The harness, examined from first principles

Every number in this document was computed from the suite files the runner loads, by
`scripts/analyze_harness.py`, on 2026-09-02. Rerun it and the numbers move with the data.

The question asked of each suite is not "what does the model score" but three prior ones:

1. **What does a predictor that never reads the observation score?** If a constant answer gets
   most of the credit, the metric is measuring the label distribution, not the model.
2. **Is the gold answer a function of the input at all?** If identical observations carry
   different golds, or the gold argument is not in the observation, no predictor can reach 100%.
3. **Is the eval split separated from training by the unit the model can memorise?** Exact-prompt
   overlap is the wrong unit when tasks span several steps.

---

## The audit

| suite | rows | tools | majority type_match | constant step | no-arg | arg derivable | reachable | dup rows (ambiguous) | catalog |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| xlam | 2,941 | 1,313 | 1.7% | 0.0% | 1.2% | 78.1% | 100% | 35 (2) | 2.8 |
| toolace | 949 | 929 | 0.2% | 0.0% | 6.9% | 65.6% | 100% | 10 (0) | 3.4 |
| bfcl | 204 | 152 | 3.9% | 0.0% | 2.5% | 78.3% | 100% | 4 (0) | 2.1 |
| androidcontrol | 904 | 8 | 13.8% | 0.0% | 30.9% | 28.9% | 97.8% | 224 (47) | 11.0 |
| mobileactions | 961 | 7 | 22.3% | 0.0% | 29.4% | 84.8% | 100% | 0 | 7.0 |
| toolbench | 687 | 197 | 1.9% | 1.0% | 47.4% | 54.6% | 100% | **553 (0)** | 5.9 |
| mcpatlas | 495 | 43 | 8.5% | 0.0% | 28.3% | 30.6% | 100% | 0 | 15.2 |
| agentnet | 12,482 | 8 | **72.9%** | 0.0% | 0.0% | 8.4% | 100% | 0 | 8.0 |
| mind2web | 252 | 3 | **80.2%** | 0.0% | 0.0% | 15.6% | 100% | 0 | 3.0 |
| toolsandbox | **17** | 14 | 17.6% | 17.6% | 100.0% | – | 100% | 0 | 33.0 |
| tau2 | 402 | 22 | 23.9% | 0.0% | 0.5% | 48.8% | **87.3%** | 150 (**128**) | 15.6 |

*majority type_match* = a predictor that always emits the most frequent gold tool. *constant
step* = that predictor with empty arguments. *reachable* = the ceiling once identical observations
with different golds are counted. *dup rows* = rows whose observation appears more than once.

Eval/train overlap, for the suites that have a posttrain split (`--train-overlap`):

| suite | eval rows | exact | first-60-chars |
|---|---:|---:|---:|
| xlam | 2,941 | **90** | 233 |
| toolace | 949 | 0 | 153 |
| toolbench | 687 | 0 | 82 |
| mind2web | 252 | 0 | **214** |
| androidcontrol | 904 | 0 | 904 |
| mobileactions | 961 | 0 | 961 |

*exact* = the eval observation appears verbatim in the training split. *first-60-chars* = a
training row opens with the same 60 characters. The audit also counts *gold pair in train*,
the eval (tool, arguments) appearing verbatim as a training label: 252 of 252 on mind2web. The prefix test is only meaningful where an
observation opens with task-specific text: androidcontrol and mobileactions open with a shared
template (mobileactions' 332-char date/time preamble), so 100% there says nothing, and toolace's
153 are generic openings. mind2web's 214 looked like the finding; the receipt showed the leak
is its label space instead (below). xlam's 90 exact matches (3.1%) are a small, measured leak in the suite the headline
should rest on; they are listed in the receipt and should be removed from the eval split (plan
item 5).

And what the models actually did, as parse / type / step:

| suite | la-10m-matrix | la-93m-matrix | la-150m | la-300m |
|---|---|---|---|---|
| xlam | 93.5 / 69.4 / 36.0 | 95.9 / 92.8 / 62.6 | 97.5 / 93.0 / 63.8 | 97.6 / 93.6 / 64.2 |
| toolace | 81.3 / 36.6 / 12.5 | 79.2 / 64.8 / 31.7 | 84.8 / 71.4 / 36.2 | 85.0 / 68.2 / 31.7 |
| bfcl | 88.7 / 60.3 / 8.8 | 93.1 / 79.4 / 23.5 | 96.1 / 80.9 / 23.0 | 93.1 / 68.6 / 23.5 |
| androidcontrol | 99.0 / 78.4 / 46.3 | 100 / 80.6 / 54.9 | 100 / 80.9 / 55.5 | 99.9 / 81.1 / 57.3 |
| mobileactions | 70.4 / 69.3 / 53.8 | 69.9 / 69.8 / 61.0 | 83.0 / 82.6 / 72.4 | 83.1 / 82.8 / 71.5 |
| toolbench | 91.4 / 28.1 / 13.1 | 96.7 / 39.6 / 25.8 | 99.1 / 38.6 / 24.3 | 95.9 / 38.6 / 24.2 |
| mcpatlas | 96.2 / 3.4 / 1.4 | 99.4 / 2.8 / 0.4 | 99.2 / 3.2 / 0.8 | 97.6 / 3.8 / 1.2 |
| agentnet | 81.7 / 8.7 / 1.8 | 98.7 / **74.3** / 3.0 | 97.5 / 73.1 / 4.9 | 95.7 / 49.6 / 5.7 |
| mind2web | 100 / 80.2 / 2.0 | 100 / 86.1 / 22.6 | 100 / 84.5 / 24.2 | 100 / 90.1 / **64.3** |
| toolsandbox | 0 / 0 / 0 | 100 / 5.9 / 5.9 | 100 / 5.9 / 5.9 | 100 / 0 / 0 |

---

## What each suite actually measures

### Sound single-action tool calling — `xlam`, `toolace`, `bfcl`

Near-unique tools per row (1,313 / 929 / 152 distinct), constant floors at 0–4%, arguments 66–78%
copyable from the observation, no ambiguity. xlam carries 90 eval rows (3.1%) whose observation
is verbatim in its train split - small, but it is the only exact leak measured, and in the suite
that matters most. These measure exactly what the harness claims:
choose the tool, lift the arguments from the text. Scores rise monotonically with size and the
parse → type → step funnel is informative at every stage. **The headline number should be built
on these.**

### Sound, but projected — `androidcontrol`, `mobileactions`

`androidcontrol`: constant floor 13.8%, model type_match 80% — real selection. Step success is
scored with the published 14%-of-screen-width grounding radius. It is a **text-only projection**
of a multimodal benchmark (the screenshot is omitted), so it is not comparable to the published
number, and 47 rows are ambiguous (identical observation, different gold).

`mobileactions`: the one suite where **parse_rate is the bottleneck** (70–83% across tiers).
Arguments are 85% copyable and multi-field; the observation carries a 332-char date/time preamble.
Failures are format, not selection. The step number understates selection ability by the parse
gap.

### Repetition-weighted — `toolbench`

553 of 687 rows are exact duplicates of another row's observation (137 groups), all with
consistent golds and **0 overlap with the train split**. The suite is ~271 unique prompts counted
with repetition; 47% of golds take no arguments. It measures selection among ~200 RapidAPI names,
weighted by however many times each prompt was sampled.

### Constant-floor type_match, grounding without pixels — `agentnet`

Always-`click` scores **72.9% type_match; la-93m-matrix scores 74.3%**. The type metric is the
label distribution. Step success requires a click coordinate inferred from a text description of
the screen (argument derivability 8.4%); it is a grounding task with the image removed, and every
tier lands at 2–6%. At 12,482 rows it is the largest file and, at 1/10 suite weight, contributes
one number that is almost entirely floor.

### Label-leaked — `mind2web`

Three checks that each rule out the obvious story:

- gold `target_id` has 156 distinct values, top-1 share 2.4% — **not** guessable by popularity
- exact observation overlap with posttrain is **0** — **not** prompt memorisation
- the observation is the task sentence alone (109 chars, **0 rows with DOM markup**, `target_id`
  present in the observation **once**) — the answer is **not derivable from the input**

The first explanation this audit reached was task-level: 214 of 252 eval observations share
their first 60 characters with posttrain rows (other steps of the same Mind2Web task). The
receipt then falsified it: on la-300m-matrix the 38 rows *outside* that overlap score **57.9%**
against 58.7% on the full split. Shared tasks are not the mechanism.

The mechanism is the label space. **All 252 eval golds — `web_click` with an element id such as
`'107'` — appear verbatim among the 4,508 posttrain golds.** The model has learned which ids are
answers, and their distribution, from a training split that shares every page with the eval
split. That is why the score rises with capacity the way memorisation does (2 → 23 → 27 → 59)
while derivability says it cannot. **No clean subset exists**; the leak index records the whole
split, and the receipt's `clean` block says so rather than scoring an empty set. The constant
floor for type_match is 80%.

### Zero-shot wall — `mcpatlas`

Held out from every training stage, by policy. Parse 96–99%, **type 3–4%, step 0.4–1.4% at every
tier from 10M to 300M.** The model emits well-formed calls to the wrong one of 15 MCP tools it has
never seen named. It is a wall, not a gradient: it will say nothing about the ladder until some
model clears it.

### Noise — `toolsandbox`

17 rows, all argument-free. One row is 5.88 points on the suite and **0.59 points on the
ten-suite mean** — wider than the 0.13 that separated the sparse arm from its dense control.
`1/17` and `0/17` are adjacent outcomes, not a regression.

### Not a function of its input — `tau2` (blocked)

99% of observations are the user-simulator's second-person brief, and **128 rows carry identical
briefs with different gold actions**, capping any predictor at 87.3% before the persona problem is
counted. Blocked in `interaction.py` with the evidence; scoring it needs a stateful, dialogue-grade
harness.

---

## What the ten-suite mean is averaging

| category | suites | share of the mean |
|---|---|---|
| sound tool calling | xlam, toolace, bfcl | 30% |
| sound but projected | androidcontrol, mobileactions | 20% |
| repetition-weighted | toolbench | 10% |
| constant-floor / no pixels | agentnet | 10% |
| label-leaked | mind2web | 10% |
| zero-shot wall | mcpatlas | 10% |
| noise | toolsandbox | 10% |

Forty percent of the headline number is floor, leakage, a wall, or noise. The mean is still
useful for continuity across the campaign — every published receipt uses it — but it is not the
number to reason from.

---

## The matrix ladder, reported by category

`scripts/analyze_receipts.py results/evalsuite-full/la-*matrix*.json` on 2026-09-03. *sound* is
the mean step_success over xlam, toolace and bfcl; *proj* over androidcontrol and mobileactions;
the rest are single suites read against their floor; *legacy10* is the old ten-suite mean.

| receipt | sound | proj | mind2web | agentnet | toolbench | toolsandbox | mcpatlas | legacy10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| la-10m-matrix | 19.1 | 50.1 | 2.0 | 1.8 | 13.1 | 0.0 | 1.4 | 17.6 |
| la-10m-matrix-s2 | 18.3 | 48.6 | 0.8 | 2.2 | 10.0 | 0.0 | 0.2 | 16.5 |
| la-10m-matrix-s3 | 17.2 | 49.4 | 2.0 | 2.6 | 11.1 | 0.0 | 2.4 | 16.8 |
| la-45m-matrix | 31.2 | 55.8 | 7.5 | 3.9 | 15.6 | 0.0 | 2.6 | 23.5 |
| la-93m-matrix | 39.3 | 57.9 | 22.6 | 3.0 | 25.8 | 5.9 | 0.4 | 29.1 |
| la-150m-matrix | 41.1 | 57.2 | 26.6 | 4.0 | 21.0 | 0.0 | 2.6 | 29.2 |
| la-300m-matrix | 44.6 | 59.2 | 58.7 | 5.8 | 30.0 | 0.0 | 1.8 | 34.9 |
| la-300m-moe-matrix | 44.3 | 59.5 | 56.3 | 5.2 | 28.7 | 0.0 | 2.2 | 34.4 |

la-300m-matrix is the first receipt scored with the leak index: xlam's 90 verbatim rows barely
matter (clean 66.5 vs 67.3 full), and mind2web's clean subset is what falsified the task-prefix
story above.

Read this way the ladder says something the legacy mean hid: 93m → 150m is +1.8 on sound tool
calling and flat on the legacy mean (29.1 → 29.2), because the legacy mean is diluted by four
suites that do not move. The three 10m seeds put the seed noise on *sound* at about ±1 point.

---

## The harness to construct

Each item below follows from a measurement above. Items 1–8 and 11 are done; 9 and 10 need
data and simulators this repository does not yet have.

1. **Every receipt carries its floors.** `Floors.measure(tasks)` is written beside each suite's
   score: majority type_match, constant step_success, no-argument share, argument derivability,
   reachable ceiling. A score is read against its floor, not against zero. *(done)*
2. **Suite terms are data.** `SuiteSpec` in `interaction.py` carries interaction mode, claim
   boundary, minimum rows and blocked reason; `audit()` is written into the receipt. *(done)*
3. **The audit is a script**, not a session. `scripts/analyze_harness.py` reproduces this whole
   document from the data, including train overlap by exact and by 60-char prefix. *(done)*
4. **Report by category, not only by mean.** Every `SuiteSpec` carries a `Category`; the
   receipt's `summary` block averages only within *sound* and *projected*, reports the rest
   singly with floor and ceiling, and keeps the ten-mean labelled `legacy_ten_suite_mean`.
   `scripts/analyze_receipts.py` prints the table above for any receipts. *(done)*
5. **Episode-disjoint and exact-disjoint splits.** mind2web's eval split must be re-cut by task,
   not by step, before its number means anything; the 60-char-prefix overlap is the test that a
   re-cut passes. `analyze_harness.py --train-overlap` writes the leaked rows to
   `results/analysis/leaked-eval-rows.json` (xlam 90 exact, mind2web 252 by gold label) and
   the runner scores every receipt's non-leaked subset as a `clean` block beside the full number.
   The full number stays, because every earlier receipt is on it. *(done; the re-cut of the
   split itself is deliberately not applied)*
6. **Deduplicate or weight.** Receipts carry `unique_prompts` and `unique_prompt_step_success`
   beside the row-weighted number. *(done)*
7. **Split step_success into name and arguments-given-name.** Receipts carry
   `arguments_given_name`; floors carry `fully_derivable_share`, the step_success a perfect
   copier could reach, so "cannot be derived" and "was not derived" are different numbers. *(done)*
8. **Parse failures are their own axis.** Receipts carry `step_success_given_parse`. *(done)*
9. **Grounding suites need pixels.** agentnet and androidcontrol as text-only projections belong
   in a separate multimodal track, not averaged with tool calling.
10. **Stateful and dialogue suites need environments.** `Interaction.STATEFUL` / `DIALOGUE`,
    an `Environment` per domain, and — only where a suite truly needs it — a `UserSimulator`
    marked `EXTERNAL_MODEL`. That is what unblocks τ²; nothing less does.
11. **Minimum rows enforced.** `summarize` excludes a suite under its `min_rows_for_claim`
    from every category mean and lists it under `excluded_from_means`. *(done)*

Items 9 and 10 change what the project trains on and how it scores, and remain proposals.
