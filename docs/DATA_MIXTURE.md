# The data mixture, by capability

The three-stage spine (`docs/STAGES.md`) already separates *language* (pretrain), *structured
interaction* (midtrain) and *the scored surfaces* (posttrain). This page maps the capability-first
proposal the campaign was measured against onto that spine, names what was added on 2026-09-03,
and records, for every dataset the proposal lists, whether it is actually reachable from the
training box and whether it may be used at all. "Reachable" was measured against the Nexus
HuggingFace mirror (`scripts/fetch_pretrain_4b.sh`), not assumed; "allowed" is the midtrain line
(does it correspond to a suite the harness scores?) plus the licence gate.

## What the proposal asks for, and where it lands

| proposal | in the repo | status |
|---|---|---|
| base pretraining on web + code + knowledge | five-source matrix (FineWeb-Edu, C4, Wikipedia, Cosmopedia, FineMath) | done; **code was missing** — `la-93m-matrix-code` adds permissive Python (166M train tokens) as a sixth source |
| midtrain that exposes structured interaction before SFT | midtrain mixture of non-benchmark agent data with general replay | done; widened by the six open sets below in `la-93m-matrix-openmid` |
| instruction SFT on tool / agent trajectories | posttrain on the benchmarks' train splits | done |
| Model A / B / C comparison (no midtrain / tool midtrain / wider midtrain) | `la-93m-matrix-nomid` / `la-93m-matrix` / `la-93m-matrix-openmid` | queued behind the 300m spine on GPU 1 |
| evaluation split into selection, arguments, planning, completion | `type_match`, `arguments_given_name`, `step_success_given_parse`, per-category summary | done (`docs/HARNESS.md`) |
| canonical tool-call representation across sources | the `Conversation` schema + `openai_full_catalog_v1` contract | done since the start |
| general instruction following (OpenHermes, UltraChat) | not a training axis | not added: the harness scores no instruction-following suite, so it would be an unmeasured change |

## Every dataset the proposal names

| dataset | licence | reachable from the box | allowed | stage | decision |
|---|---|---|---|---|---|
| HuggingFaceFW/fineweb-edu | ODC-BY-1.0 | yes | yes | pretrain | in the matrix |
| allenai/c4 | ODC-BY | yes | yes | pretrain | in the matrix |
| tiiuae/falcon-refinedweb | ODC-BY-1.0 | **404** (gated upstream) | – | pretrain | unreachable |
| allenai/dolma | ODC-BY | **404** (gated upstream) | – | pretrain | unreachable |
| bigcode/the-stack-v2, starcoderdata | ToS-gated | **404** | – | pretrain | unreachable; The Stack v2 ships ids only, content needs SWH credentials |
| codeparrot/codeparrot-clean, row-level permissive licences only | MIT/Apache/BSD/ISC | yes | yes | pretrain | **added** as `python` (`data/shards/src/python`) |
| wikimedia/wikipedia | CC-BY-SA-4.0 | yes | yes | pretrain | in the matrix |
| manu/project_gutenberg | public domain | **404** | – | pretrain | unreachable |
| HuggingFaceTB/finemath, cosmopedia | ODC-BY / Apache-2.0 | yes | yes | pretrain | in the matrix |
| ToolBench | Apache-2.0 | yes | posttrain only | posttrain | its train split is already there; the harness scores it |
| Gorilla APIBench | Apache-2.0 | yes | **no** | – | BFCL is Gorilla's evaluation and is held out here; training on APIBench would contaminate it |
| AppWorld | Apache-2.0 | environment, not a corpus | – | – | trajectories need the environment run; a candidate for the stateful track (`docs/HARNESS.md` item 10) |
| WebArena | Apache-2.0 | environment, not a corpus | – | – | same |
| microsoft/orca-agentinstruct-1M-v1 | CDLA-Permissive-2.0 | yes | yes (synthetic) | midtrain | **added**: tool_use 50,000 · webagent_flow 25,000 · code 99,984 · follow_up 99,046 · task 125,000 rows |
| xingyaoww/code-act | Apache-2.0 | yes | yes | midtrain | **added**: 7,139 rows |
| nvidia/When2Call | CC-BY-4.0 | yes | **no** | – | 2,243 of its 3,215 tool names are in the xlam eval catalog |
| interstellarninja/hermes_reasoning_tool_use | Apache-2.0 | yes | **no** | – | built from When2Call, ToolAce and xLAM |
| OpenMathInstruct, NuminaMath | permissive | yes | yes | – | not added: FineMath already carries math in pretraining; reasoning SFT is off the harness |
| OpenHermes, UltraChat | mixed / MIT | yes | licence review needed | – | not added (see above) |

The refusals are enforced in code, not only listed: `scripts/normalize_midtrain_open.py` refuses
When2Call and hermes by file name with the measured reason.

## The mid-training question, as an experiment

> Does tool-use capability emerge from continued pretraining on tool-interaction corpora, before
> instruction tuning?

The harness can answer it only if the training data for midtrain and posttrain stay disjoint
(they are, by the line above) and the score is read by category (it is, since `docs/HARNESS.md`).
Three spines at la-93m, same pretrain checkpoint, same posttrain budget:

| arm | midtrain | what a gap means |
|---|---|---|
| A `la-93m-matrix-nomid` | none | the sound-tool-calling mean posttrain alone reaches |
| B `la-93m-matrix` | 5 agent sources + replay (current) | what the existing midtrain buys |
| C `la-93m-matrix-openmid` | B + orca tool_use/webagent/code/follow_up/task + code-act | whether *more* structured interaction, none of it benchmark-derived, transfers |

plus `la-93m-matrix-code`, the same spine with Python in pretraining, for the proposal's claim that
code is where structured generation comes from. Seed noise on the sound mean is about ±1 point
(three la-10m seeds), so a gap under 2 points is not a result.

### Result at la-93m (2026-09-04, one seed each)

| arm | sound | proj | mind2web | toolbench | legacy10 |
|---|---:|---:|---:|---:|---:|
| A `la-93m-matrix-nomid` (no midtrain) | 39.9 | 57.7 | 12.7 | 24.0 | 27.8 |
| B `la-93m-matrix` (current midtrain) | 39.3 | 57.9 | 22.6 | 25.8 | 29.1 |
| C `la-93m-matrix-openmid` (+ six open sets) | 39.9 | 57.8 | 13.9 | 23.1 | 27.9 |
| `la-93m-matrix-code` (B + Python in pretrain) | 40.0 | 57.8 | 25.0 | 25.8 | 29.4 |

Seed replicate and the 300m arms (2026-09-04 evening):

| arm | sound | proj | mind2web | legacy10 |
|---|---:|---:|---:|---:|
| B `la-93m-matrix-s2` (seed 7) | 40.8 | 57.7 | 21.4 | 28.4 |
| A `la-300m-matrix-nomid` | 43.9 | 59.8 | 61.9 | 34.8 |
| B `la-300m-matrix` | 44.6 | 59.2 | 58.7 | 34.9 |
| C `la-300m-matrix-openmid` | 44.8 | 59.6 | 60.7 | 35.1 |

The two 93m seeds of the control differ by 1.5 points on *sound* (39.3 / 40.8), so the noise at
this scale is about ±0.8; A and C at 93m (39.9 / 39.9) and at 300m (43.9 / 44.8 against 44.6) are
inside it. **The null holds at both scales.**

On the suites that measure tool calling, midtrain moves nothing at this scale: A, B and C are
inside the ±1 seed noise measured at 10m, and posttrain alone reaches the same sound mean. The
one suite midtrain lifts, mind2web (12.7 → 22.6), is the label-leaked one, so what it lifts is
memorised label structure, not selection. Code in pretraining is also inside noise (bfcl +4.0,
toolace −1.4). The claim "structured interaction in mid-training forms agentic capability" is
therefore **not supported at 93m** on this harness; seed replicates at 93m and the same A/C arms at
300m are queued to say whether the null holds.

Per-source losses are logged every step for every stage (`curve.jsonl` → `sources`), so the C arm
also shows which of the new sources the model actually fits.
