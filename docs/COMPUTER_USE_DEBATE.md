# Computer-use & tool-calling agents: a survey, a debate, and what we applied

> Companion to [`RESEARCH.md`](./RESEARCH.md). That file surveys *trainers*; this one surveys the
> recent wave of **agent** papers — computer-use, multi-agent-browser, and tool-calling — extracts
> the good part of each, then settles the central design tension with a debate between two personas
> and a verdict **specific to a sub-100M, byte-level, on-device agent**.

The point of reading these together is not to copy any one of them. It is to locate the *axis* they
disagree on, decide where this project should sit on that axis, and apply the one or two ideas that
actually move a tiny model. We applied **planner/decomposition** (`ToolCaller.plan()`); everything
else is recorded as adopted-in-spirit or deliberately deferred, with the why.

---

## Part 1 — The papers (good part + the "why")

Grouped by what they actually argue, not by venue.

### A. Native computer-use, end-to-end from pixels

**UI-TARS** (ByteDance, 2025). A single vision-language model that perceives a screenshot and emits
the next GUI action in a **unified action space** (click/type/scroll/hotkey), trained on a huge
corpus of GUI trajectories and improved by an "agents-are-data" flywheel (deploy → log trajectories
→ reflect → retrain).
- **Good part:** the flywheel framing — *the deployed agent is your best data generator* — and a
  single unified action vocabulary instead of bespoke per-app glue.
- **Why it works:** at scale, a model can learn perception+grounding+control jointly; removing the
  hand-built accessibility layer removes a brittle dependency.

**WebVoyager** (2024). An end-to-end *multimodal* web agent that drives a real browser from
screenshots + the DOM, with a GPT-4V-as-judge eval on live websites.
- **Good part:** evaluate on *live* sites, and judge task success with a model rather than brittle
  string matches.
- **Why:** static benchmarks overstate real reliability; live eval is the honest signal.

**OSWorld / OSWorld-G + Jedi** (2024–25). A real-OS benchmark (execution-based, not multiple-choice)
plus a finding that **GUI grounding** is the dominant failure mode, fixed by *decomposing* the UI
and *synthesizing* grounding data (Jedi).
- **Good part:** grounding is the bottleneck, and you fix it with **targeted synthetic data**, not
  more parameters. This is the same lesson our figures 12/14 reach from the text side.
- **Why:** execution-based eval can't be gamed; it exposes that "knowing what to do" ≠ "being able
  to point at the right element."

### B. Plan textually, act through structure (don't generate pixels)

**SeeAct** (2024). Reformulates web action as **ground a *textual* plan** onto concrete page
elements — the model says what it wants, a grounding stage maps it to an element id.
- **Good part:** the *plan / ground* split. Reasoning happens in language; commitment happens
  through a constrained grounding step.
- **Why:** lets a weaker actor be reliable — it never has to emit exact coordinates or exact JSON.

**Mind2Web** (2023). Generalist web-agent benchmark; the **candidate-ranking** recipe — a small
model *filters* the DOM to a handful of candidate elements before the big model picks — became the
template everyone reuses.
- **Good part:** **rank-then-select**. Don't make the policy choose among thousands of elements;
  retrieve a short list first.
- **Why:** this is exactly our retrieval-over-classifier-head argument (fig 12), discovered
  independently for web elements.

**WebArena** (2023). A reproducible, self-hostable web environment with **functional-correctness**
rewards (did the booking actually happen?), not similarity to a reference trajectory.
- **Good part:** reward the *outcome*, not the trajectory. Lets many valid paths win.
- **Why:** trajectory-matching punishes correct-but-different behavior; outcome rewards are what RL
  actually needs.

### C. Tool/API calling at catalog scale

**Gorilla** (2023). An LLM for **1,600+ ML APIs**; a **retriever** supplies the relevant API docs at
inference, and *retrieval-aware training* cuts hallucinated/outdated calls.
- **Good part:** **retrieval is the selection architecture** once the catalog is large; train the
  model to *use* the retrieved docs.
- **Why:** you can't memorize or N-way-classify thousands of evolving APIs; you look them up.

**ToolLLM / ToolBench** (2023). **16,000+ real REST APIs**; introduces **DFSDT** (depth-first search
over the decision tree) so the agent can *back out of* a bad tool choice instead of committing
greedily.
- **Good part:** treat multi-step tool use as **search with backtracking**, not a single forward
  pass.
- **Why:** real tasks branch; a greedy chain dies on the first wrong call.

**xLAM / APIGen** (Salesforce, 2024). A **verifiable** synthetic data pipeline: every generated
call is checked by format → execution → semantic stages before it enters training; topped BFCL.
- **Good part:** **verify synthetic data before training on it** (format, executability, semantics).
- **Why:** unverified synthetic tool data teaches plausible-but-wrong calls; a verifier is cheaper
  than the bug it prevents.

**Hammer** (2024). On-device function calling; **function masking** + deliberately injected
**irrelevance/abstention** negatives make a small model robust to distractors and to "no tool fits."
- **Good part:** **train abstention explicitly** with irrelevant cases and masked functions —
  directly relevant to a tiny on-device model.
- **Why:** small models over-trigger; the ability to say "none of these" is a learned skill, not a
  freebie.

**ToolACE** (2024). Self-evolving synthesis: an agentic pipeline grows an **API pool**, escalates
**complexity**, and **dual-verifies** (rule + model) the data.
- **Good part:** **complexity curriculum** + dual verification; coverage and difficulty are dialed,
  not hoped for.
- **Why:** static datasets plateau; a generator that targets the model's current frontier doesn't.

**ToolAlpaca** (2023). Shows a **small** model can learn *generalized* tool use from only ~3.9k
simulated cases across diverse tools.
- **Good part:** evidence that **breadth of tools beats depth of examples** for small models — a
  little data over *many* tools generalizes.
- **Why:** matches our fig 15 (per-tool examples saturate ~16); spend the budget on tool variety.

### D. The action *interface*: JSON vs code vs roles

**CodeAct** (2024). Let the agent emit **executable Python** as its action instead of a JSON blob;
+20% success on multi-step tasks because code **composes** (loops, variables, control flow, reusing
a prior result) and runs in a real interpreter.
- **Good part:** **executable code as the action space** — composition and intermediate results come
  for free from the language.
- **Why:** JSON tool-calls can't express "do this for each result of that"; code can.

**OctoTools** (Stanford, 2025). A training-free framework with **tool cards** (capability metadata)
and an explicit **planner ⟂ executor** separation; beat AutoGen by ~10.6% on a 16-task suite.
- **Good part:** **separate the planner from the executor**, and describe tools with structured
  *cards* the planner reasons over.
- **Why:** one model doing both plan and act muddles the two failure modes; splitting them makes
  each debuggable and lets a *small* executor pair with a *smarter* planner.

**AutoGen / AG2** (Microsoft, 2023–24). Multi-agent **conversation** framework: specialized agents
(planner, coder, critic, executor) collaborate via GroupChat.
- **Good part:** the **planner→executor→critic** decomposition as a reusable pattern.
- **Why:** roles localize errors and allow review before commit — but at the cost of many model
  calls.

**CoAct-1** (2025). A computer-using **multi-agent** system that mixes GUI actions with direct code
execution, routing each subtask to whichever modality is cheaper/more reliable.
- **Good part:** **route the subtask to the cheapest reliable modality** (code when you can, GUI
  when you must).
- **Why:** clicking through a UI to do what one shell command does is wasteful and fragile.

---

## Part 2 — The debate (two Claude personas)

The papers split cleanly along one axis: **does reliability come from *scale + an open-ended action
space* (pixels/code, end-to-end, learned), or from *structure + constraint* (retrieve → plan →
ground, decoded against a schema)?** I argued it out as two personas.

### Persona **SCALE** — "the bitter lesson wins; get out of the model's way"
> UI-TARS and WebVoyager perceive raw pixels and act in one unified space; CodeAct hands the model a
> *Turing-complete* action space and gains 20 points because composition is free. The throughline is
> the bitter lesson: **every hand-built scaffold you add is a ceiling you'll later remove.** Your
> retriever, your schema-decoder, your slot pools — those are 2014-era feature engineering. The
> right move is the biggest model the budget allows, an expressive action space (code, not JSON),
> trained end-to-end on trajectories from its own deployment (UI-TARS' flywheel). Constraints make a
> weak model *look* reliable on a benchmark while capping what it can ever do. Don't pre-decide the
> tool's arguments — let the model write `edit_file(p, find_replace(read(p), a, b))` if that's what
> the task needs. Structure is a crutch you'll throw away at the next scale.

### Persona **STRUCTURE** — "constraint is what makes a *small* model trustworthy"
> Every one of those wins is *purchased with scale you do not have.* UI-TARS is billions of params
> and a data center of trajectories; CodeAct's executable actions assume a model that rarely writes
> broken code and a sandbox you trust to run it. You have **<100M params, byte-level, on a CPU.** At
> that size free generation of an unseen slot value is ~1% correct — we measured it (fig 01) — and
> grounded constrained decoding is **100%** on the *same* weights. The papers themselves keep
> rediscovering structure under pressure: OSWorld-G says grounding is the bottleneck and fixes it
> with *synthesized grounding data*; Mind2Web *ranks candidates* before the policy chooses; Gorilla
> and ToolLLM *retrieve* because you can't classify 16k APIs; Hammer *trains abstention* explicitly.
> That is not feature engineering — it is **putting the reliable parts where the model is weak**.
> CodeAct's real lesson isn't "emit code," it's "let actions **compose**." I can get composition
> from a *planner* that sequences grounded calls — without handing a 30M model an interpreter and
> praying. Constraint isn't a crutch; for a tiny on-device model it is the *entire* value
> proposition: predictable, auditable, no sandbox, runs offline.

### Where they actually agree (the synthesis, not a coin-flip)
Strip the rhetoric and both personas concede three things:

1. **Selection, not grounding, is the bottleneck at scale** — SCALE via "perception is hard"
   (OSWorld-G), STRUCTURE via "retrieve before you classify" (Mind2Web/Gorilla). We already sit
   here (figs 12, 14).
2. **Composition is the real prize of CodeAct**, and composition does *not* require an interpreter —
   it requires *sequencing*. A planner delivers it.
3. **The agent is your best data source** (UI-TARS flywheel = our failure-driven flywheel, fig 07).

They only truly diverge on **the action space**: open-ended code vs schema-constrained calls. And
*that* divergence is decided entirely by **model size and deployment**.

---

## Part 3 — Verdict (for *this* project)

**STRUCTURE wins — because the deciding variable is fixed against SCALE.** The bitter lesson is a
statement about the *asymptote*; this project lives at the opposite end on purpose: <100M params,
byte-level, CPU/NPU, offline, auditable. At that operating point the empirical facts are ours, not
hypothetical: free-gen ≈1% vs grounded 100% (fig 01); retrieval scales selection to 1,350 tools
(fig 12); abstention is a learned skill (Hammer, our `min_score`). SCALE's prescription — hand the
model an interpreter and end-to-end pixels — is *correct at a billion params and wrong at thirty
million.* You don't give a 30M byte model a Python sandbox as its action space.

But SCALE lands one blow that STRUCTURE must answer: **composition.** A single grounded call can't
express "read the file, then run the tests, then commit." STRUCTURE's honest response is not to
reach for code — it's to **add a planner that sequences grounded calls.** That captures CodeAct's
*actual* win (multi-step composition, +20%) while keeping every property that makes a tiny model
worth shipping: no interpreter, no sandbox trust, every step still schema-valid and auditable.

So the verdict converts directly into one applied change and a ranked backlog.

### Applied now
- **`ToolCaller.plan(query)`** — planner/decomposition (AutoGen → OctoTools → CodeAct, via SeeAct's
  plan/ground split). Splits a multi-step request on connectives (`then` / `and then` / `after that`
  / `and` / `;`), grounds each clause with the existing reliable single-call path, and returns an
  **ordered list of schema-valid `ToolCall`s**. Composition without an interpreter. Tested in
  `tests/test_caller.py` (`test_plan_*`). This is the one idea that both personas endorse for a tiny
  model.

### Adopted in spirit already (the survey confirmed our direction)
- **Retrieve-then-select** (Mind2Web/Gorilla/ToolLLM) → `ToolRetriever` (fig 12).
- **Explicit abstention** (Hammer) → `min_score` threshold + IRRELEVANT eval.
- **Plan/ground split** (SeeAct) → grounded constrained decoding decoupled from selection.
- **Agent-as-data flywheel** (UI-TARS) → failure-driven flywheel (fig 07).
- **Verify synthetic data** (xLAM/APIGen) → schema `validate()` gate on generated calls.

### Deferred, with the why
- **Executable-code action space (CodeAct).** *Why not:* needs a trusted sandbox and a model that
  reliably writes correct code — neither holds at 30M/byte-level/offline. The planner captures the
  composition win without the risk. Revisit only if we ship a "small" tier with a sandboxed runtime.
- **DFSDT search-with-backtracking (ToolLLM).** *Why not yet:* `plan()` is currently a greedy
  forward decomposition. Backtracking pays off when steps can fail at runtime — worth adding once
  `plan()` is wired to *execute* and observe tool responses (the multi-turn path), not before.
- **Multi-agent GroupChat (AutoGen).** *Why not:* N collaborating model calls is the opposite of a
  single tiny offline model's value proposition. We take the *decomposition pattern*, not the
  *multi-process* implementation.
- **End-to-end pixel perception (UI-TARS/WebVoyager).** *Why not:* out of scope — this is a
  tool/text agent, not a vision GUI agent; pixels demand vision-scale models.

### The one-line thesis
> The literature's reliability gains come from **either** scale **or** structure; at <100M params on
> a CPU, structure is the only one you can afford — so buy the *composition* that scale was selling
> with a **planner over grounded calls**, not an interpreter.

---

## Follow-ups

- **Why is planning a different operation from acting at all?** Answered mechanistically (in terms
  of the model's logit distribution / entropy regimes) in
  [`PLANNING_VS_ACTING.md`](./PLANNING_VS_ACTING.md).
- **Why does grounded decoding beat free byte-generation?** Measured on the 28M checkpoint in
  experiment **17** of [`EXPERIMENTS.md`](./EXPERIMENTS.md) (`figures/17_*`): structural bytes 0.18
  bits / 96.6% top-1 vs slot-value bytes 3.28 bits / 23.7% top-1, with 74.9% of slot probability
  mass on bytes present in the prompt.
