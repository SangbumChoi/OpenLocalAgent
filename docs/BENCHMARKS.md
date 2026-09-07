# The evaluation matrix

Ten public agent benchmarks, one metric: **`step_success_rate`** — the predicted call's function
name **and every argument** match the gold call. The headline number is the unweighted mean over
the ten.

Every example below is a real row printed from the file the runner loads, not an illustration.

---

## How to read a suite

Each suite carries its own terms in `src/openlocalagent/eval/interaction.py`, and
`tests/test_eval_interaction.py` keeps them from drifting from the data:

| Field | Means |
|---|---|
| `interaction` | `SINGLE_ACTION` / `STATEFUL` / `DIALOGUE` — how many turns before the gold answer is determined |
| `reproducibility` | `SELF_CONTAINED`, or `EXTERNAL_MODEL` if the score depends on a model this repo does not train |
| `claim_boundary` | How this number differs from the upstream published metric |
| `min_rows_for_claim` | Below this, one row moves the mean more than differences we call results |
| `blocked_reason` | Set when the suite must not be scored as-is |

```python
from openlocalagent.eval.interaction import spec_for
spec_for("toolsandbox").audit(rows=17)
# ['17 rows is below 100: one row moves the ten-suite mean by 0.59 points', ...]
```

---

## The ten

### `xlam` — 2,941 rows · tool_use

Plain API selection from a small catalog (mean 2.9 tools). The cleanest suite in the set.

```
observation  What information can be obtained about the Maine Coon cat breed?
expected     get_breed_information({"breed": "Maine Coon"})
```

### `toolace` — 949 rows · tool_use

Longer requests where arguments must be lifted verbatim out of the text.

```
observation  Can you help me download a video from YouTube? Here's the link:
             https://www.youtube.com/watch?v=dQw4w9WgXcQ. Please save it to my desktop.
expected     S_YTD({"video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "output_path": "desktop"})
```

### `bfcl` — 204 rows · tool_use · **held out**

Berkeley Function Calling Leaderboard. Dotted names, typed numeric arguments.

```
observation  Calculate the area of a triangle, given the lengths of its three sides: 3, 4, and 5.
expected     math.triangle_area_heron({"side1": 3, "side2": 4, "side3": 5})
```

> **Claim boundary** — appears in no training stage. Pure transfer.

### `mcpatlas` — 495 rows · tool_use · **held out**

MCP server tools, 15-tool catalogs, multi-hop phrasing.

```
observation  I've been looking into contributions in open source videogames and I'm curious
             about the longevity of this one open source project. Can you get me the year
             difference between...
expected     github_search_repositories({"query": "assaultcube"})
```

> **Claim boundary** — appears in no training stage. Pure transfer.

### `toolbench` — 687 rows · tool_use

RapidAPI-style names; many golds take no arguments, so it mostly tests **selection**.

```
observation  I'm a sports enthusiast and I'm interested in attending a greyhound race.
             Can you give me the race schedule for this week?
expected     racecards_for_greyhound_racing_uk({})
```

### `androidcontrol` — 904 rows · gui_control

Android UI steps. Scored with **grounding accuracy** as well: a tap counts if it lands within
14% of screen width of the gold point.

```
observation  You are a mobile action parser. The screenshot is omitted; emit only the action
             JSON. ... Instruction: ...
expected     mobile_click({"x": 561, "y": 535})
```

> **Claim boundary** — text-only. The screenshot is omitted, so this sits below the published
> multimodal number and is **not comparable to it**.

### `mobileactions` — 961 rows · gui_control

Phone assistant actions with real multi-field arguments — the hardest arguments in the set.

```
observation  Current date and time given in YYYY-MM-DDTHH:MM:SS format: 2024-09-15T17:35:19
             Day of week is Sunday ...
expected     create_contact({"first_name": "Lena", "last_name": "Petrova",
                             "phone_number": "+359 888 123 456",
                             "email": "lena.petrova.design@webmail.com"})
```

All four fields must be exact. One wrong digit scores zero.

### `agentnet` — 12,482 rows · gui_control

Desktop control. The action space is one `click` tool; the difficulty is **where**.

```
observation  Task: Open the Pikachu picture on the desktop using GIMP, and then select the
             "Venetian Blinds" filter in the animation.
             Observation: The current screen shows a desktop ...
expected     click({"target": "button=left;x=0.018000;y=0.508000"})
```

> **Claim boundary** — the eval split keeps its **empty upstream catalog**. Only the train split
> was catalog-repaired, because changing the eval split would inflate our numbers against the open
> baselines that were scored on the original.

### `mind2web` — 334 rows · web_control

Web navigation; the gold is a DOM element id.

```
observation  Find all campgrounds located in California.
expected     web_click({"target_id": "107"})
```

> **Claim boundary** — drawn from the `merged-v2` pool that posttrain also trains on: **1,834 rows
> and 3,214 rendered prompts overlap**. Not a clean holdout.

### `toolsandbox` — **17 rows** · tool_use

Stateful device control, flattened to one turn. The flattening holds here — the user turns are
real requests and the golds take no arguments — which is exactly why it scores where τ² cannot.

```
observation  Turn off cellular
expected     set_cellular_service_status({})
```

> ⚠️ **17 rows.** At 1/10 weight, **one row moves the ten-suite mean by 0.59 points**. That is
> wider than differences this project has reported as results — `la-300m-moe` beat `la-300m` by
> 0.13. Do not read a toolsandbox change as a signal; `1/17 = 5.88%` and `0/17 = 0.00%` are
> adjacent, not a regression.

---

## Blocked: `tau2`

τ²-bench is in the repo and **is not scored**. This is a measurement, not a preference.

| Evidence | tau2 | xlam |
|---|---|---|
| Observations that are second-person persona script | **401 / 402 (99.8%)** | 0.2% |
| Opaque-id gold arguments absent from the observation | **62 / 83 (75%)** | 7 / 30 |
| Gold argument values derivable from the prompt | **48.8%** | 78.7% |

The observation is the **user simulator's brief**, not a request to the agent:

```
observation  You recently spoke on the phone with a customer support representative that told
             you that a service agent will be able to help you cancel your reservation.
             The trip you want to cancel is the one from Philadelphia to LaGuardia.
             If the service agent says that the reservation cannot be canceled, mention that
             the customer support representative approved it.
expected     get_user_details({"user_id": "raj_sanchez_7340"})
```

`raj_sanchez_7340` appears nowhere in the observation. In τ²-bench the agent obtains it **by
asking the user** — obtaining the information *is* the task. Flattened to one turn, the answer is
not a function of the input.

At 48.8% per-argument derivability and ~3.6 arguments per call, a perfect model caps near
**0.488³·⁶ ≈ 15%**. Measured: **0.0% at every tier from 10M to 300M**, with `parse_rate` steady
near 60% — the model emits well-formed calls and cannot know which. Size-independent zero with
healthy parse rate is the signature of a harness fault, not a capability limit.

Scoring it needs `Interaction.DIALOGUE`: an `Environment` that executes calls against airline and
telecom state, and a `UserSimulator`. The simulator is another language model, which is why the
spec marks τ² `EXTERNAL_MODEL` — a score from it would be conditional on a model this repository
does not train, and that breaks the project's reproducibility claim.

---

## Extending the harness

`interaction.py` makes single-turn the degenerate case of a general loop, so the ten keep their
exact numbers while stateful suites become expressible:

```
SINGLE_ACTION   observation → action → compare to gold          (model only)
STATEFUL        observation → action → env.step → observation → ... → env.goal_reached()
DIALOGUE        STATEFUL + user.reply() supplies what the agent must elicit
```

`Interaction.metrics` refuses the category error directly: `step_success_rate` is offered only in
`SINGLE_ACTION`. For stateful modes the metrics are `task_success_rate`, `pass_k`, `mean_steps`,
`invalid_call_rate` — because the "gold first action" is an artifact of flattening, not the
benchmark's definition of success.

To add a stateful suite:

1. Implement `Environment` — `reset`, `step(ToolCall) -> observation`, `goal_reached`, `snapshot`.
   An invalid call returns an observation ("no such tool"), never an exception: recovering from a
   bad call is behaviour a single-action harness cannot see.
2. Register a `SuiteSpec` with the right `interaction` and an honest `claim_boundary`.
3. Add the row file and a `scripts/normalize_<suite>.py` that rebuilds it. A suite you cannot
   rebuild is a suite you cannot check.
4. If it needs a `UserSimulator`, mark it `EXTERNAL_MODEL` and say which model in the receipt.

An unrecognised `--suites` name is a hard error on purpose. Do not soften it back to a skip.
