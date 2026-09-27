# Realistic agent evaluation and data policy

This project targets a text-first WebGPU agent with a small (<100M) decoder.  Real deployments
still need to be tested against the environments people use: mobile UI control, browser actions,
desktop workflows, and stateful API/MCP tools.  The source-linked inventory is
[`configs/data/realistic-agent-eval.catalog.yaml`](../configs/data/realistic-agent-eval.catalog.yaml).
It is deliberately a catalog, not a downloader: every acquired byte must be recorded in a local
provenance manifest with an upstream revision, byte count, and SHA-256.

The current canonical catalog contains 42 source-linked rows (four train-eligible and 38 evaluation
or restricted) and has a canonical SHA-256 fingerprint recorded by the preflight command below.
The separate 27-row public evaluation matrix marks five research train candidates, adding the
supplemental ToolACE adapter; its current byte and canonical hashes are bound by the [m359 matrix
audit](paper/results/raw/m359-public-realistic-eval-matrix-audit-v1.json). These are deliberately
different layers: the canonical catalog controls acquisition configs, while the matrix records the
broader source-linked research plan.
The fingerprint is generated from the canonical catalog by the preflight command below.

Current release status (2026-08-08): the exact 10,524,544-parameter BPE checkpoint is
`runs/sft-mind2web-public-continuation-20260805/latest.pt` with SHA-256
`6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45`.  The complete current
checkpoint-bound MobileGym and BrowserGym/MiniWoB receipts pass their native contracts, and the
static 63-tool WebGPU bundle plus HF-compatible model export pass local parity.  The fail-closed
publication gate remains `ready=false`: nine additional native benchmark receipts and authenticated
public Hub model/Space URLs are still missing.  See the current [m513 gate](paper/results/raw/m513-workshop-gate-current-native-browsergym-mobilegym-v1.json),
[m515 bundle verification](paper/results/raw/m515-current-webgpu-deploy-verify-v1.json), and
[m516 local HF preparation](paper/results/raw/m516-hf-release-local-prepare-v1.json).  The refreshed
[m523 gate](paper/results/raw/m523-workshop-gate-current-m522-v2.json) now passes the explicit
transfer/no-transfer weight requirement using the current checkpoint, but still reports the nine
missing native receipts, the unbound current-checkpoint RL preflight, and unauthenticated public
artifacts.  The latest [m524 WebGPU rerun](paper/results/raw/m524-webgpu-current-bundle-rerun-v1.json)
passes the native hardware check on Apple Metal-3 with 3/3 exact local structured actions at
`674.7` p50 tokens/s; [m525](paper/results/raw/m525-workshop-gate-current-m524-v1.json) binds that
receipt into the same fail-closed gate.  The local actions intentionally execute no real email,
URL navigation, or Notion side effect.  Older sections
below are retained as historical experiment records.

The latest stateful transfer bridge is recorded in the [m583 receipt](paper/results/raw/m583-stateful-head-transfer-toolsandbox-v1.json).
Head-only fitting raises local selector top-1 from `46.67%` to `53.33%`, but warm and matched-random
closed-loop completion are both `68.75%` (task completion `40%`). The m553 parent and m581 child
both pass the bounded single-step ToolSandbox smoke (`3/3`) and both fail the bounded multi-turn
stress (`0/3`). This is a negative transfer diagnostic, not an official ToolSandbox or real-account
result.

The [m584 public-source snapshot audit](paper/results/raw/m584-public-dataset-snapshot-audit-v1.json)
now binds eight current Hugging Face revisions to their original source repositories and records
the train/evaluation boundary. It confirms that AndroidControl, AgentNet, Mind2Web, and xLAM are
the only rows in this snapshot eligible for text-first continuation; Computer Agent Arena, OSWorld
2.0/Verified trajectories, and EnterpriseOps-Gym stay evaluation/provenance-only. The audit is
metadata-only and does not claim a native benchmark result.

The [m585 continuation bridge](paper/results/raw/m585-public-multisurface-transfer-webgpu-v1.json)
repeats the warm/random protocol with 202 public train-projection rows and 53 source-local held-out
rows across AndroidControl, AgentNet, Mind2Web, and MCPMark. Warm accuracy is `68.39%` versus
`46.74%` for random, but exact sequence accuracy remains zero. The warm child is checkpoint-bound
to a parity-verified WebGPU bundle: `3/3` structured probes, `1,367.5` tok/s p50, `7.4 ms` p50,
and `13/13` local email/Notion/browser state transitions. These are text/accessibility and
resettable local-fixture results, not official native benchmark or real-account scores.

The [m586 BrowserGym receipt](paper/results/raw/m586-browsergym-m585-warm-full-v1.json) now binds
that warm child to a fresh native run over the verified official 240-task MiniWoB plan. It passes
`5/240` tasks (`2.08%`) with `219` grounded steps and zero action errors; four successes are
`miniwob.click-button` and one is `miniwob.sign-agreement`. The runner used accessibility-tree text
only, with no coordinate or semantic fallback and no external side effects. The `1,960`
noop/ungrounded steps and absent visual grounding keep this diagnostic rather than a general
browser/computer-use result, and the AndroidWorld, iOS, desktop, official stateful-tool, and
authenticated-Hub gates remain open.
The receipt is emitted in the canonical native schema, and the fail-closed gate independently
recognizes `native:browsergym_miniwob` as `pass` when supplied; this does not waive the remaining
requirements.

The [m588 MobileGym receipt](paper/results/raw/m588-m585-mobilegym-native-full-v1.json) upgrades the
current warm child from a text-only continuation to a complete native MobileGym test-split run:
`1/256` tasks pass (`0.39%`), matching the m553 parent exactly. It is now consumable by the gate as
`native:mobilegym=pass`, but the unchanged score shows that representation transfer alone did not
improve closed-loop mobile control.

The [m589 ToolSandbox smoke](paper/results/raw/m589-m585-toolsandbox-native-smoke-v1.json) also
passes `3/3` bounded scenarios at similarity `1.0` on the same warm child. It is explicitly
non-official (`official_split_verified=false`, no model-based user simulator) and does not change
the ToolSandbox publication blocker; it only confirms current-checkpoint one-turn tool-contract
preservation.

The [m590 matched Computer Agent Arena control](paper/results/raw/m590-computer-agent-arena-m585-warm-parent-control-v1.json)
uses 256 unique tasks from the pinned public JSONL and runs the m553 parent and m585 warm child
under the identical instruction-only evaluator. Both arms produce `1.56%` tool exactness,
`1.95%` action-family exactness, `100%` route accuracy, and `96.88%` abstention. This is a
checkpoint-bound desktop action-prior result, not AgentNetBench, OSWorld, visual grounding, or
native desktop success; the image-backed and runtime gates remain open.

The [m591 AgentNet control](paper/results/raw/m591-agentnet-m585-warm-parent-control-v1.json)
adds the normalized public trajectory holdout: 16 parent tasks and 257 projected actions, matched
between m553 and m585. Both arms have `0%` exact trajectories, `75%` first-action type accuracy,
and byte-identical predictions. This is an evaluation-only text projection—not AgentNetBench,
OSWorld, visual grounding, or native desktop success—and it confirms that the current weight
transfer has not solved computer-use trajectory execution.

The [m592 email control](paper/results/raw/m592-enterpriseopsgym-m585-warm-parent-email-control-v1.json)
replays all 67 public EnterpriseOps-Gym email rows with the current m553 parent and m585 warm child.
Warm top-1 tool retrieval improves from `43.28%` to `56.72%` (`+13.43 pp`), while top-3 and top-5
remain `76.12%` and `91.04%`. The result is explicitly out-of-domain name-only retrieval: no MCP
server, SQL verifier, email account, or stateful execution ran, so the official enterprise gate
remains open.

The [m593 MCPMark router control](paper/results/raw/m593-mcpmark-m585-warm-parent-router-control-v1.json)
adds the pinned standard/easy task-description suites. Parent and warm results are identical:
`25/169` standard and `10/70` easy, with `0/28` Notion routes on standard and `25/25` Playwright
routes. This confirms that the m585 transfer did not improve Notion service selection; official
MCP execution, verifier success, and account-backed state transitions remain untested.

The latest public continuation [m526 comparison](paper/results/raw/m526-realistic-cross-surface-warm-random-8step-comparison-v1.json)
uses reacquired AndroidControl, OpenCUA-AgentNet, Mind2Web-train, and redacted MCPMark
trajectory-log slices. MCPMark contributes filesystem, Notion, GitHub, Playwright, and Postgres
traces; the 16-row train/8-row held-out cap is source-disjoint. Warm initialization scores
`50.92%` held-out assistant-token accuracy against `0%` for the matched random control, but exact
sequence accuracy is `0%` on both arms. This supports checkpoint reuse as a transfer candidate,
not a native benchmark or live-account claim. The [m528 gate](paper/results/raw/m528-workshop-gate-current-m526-v1.json)
passes the weight ablation and WebGPU checks but remains `ready=false` until the required native
mobile, browser, desktop, stateful-tool, RL, and public-artifact evidence is complete.

The stronger [m530 transfer receipt](paper/results/raw/m530-realistic-cross-surface-warm-random-64step-comparison-v1.json)
extends the same source-disjoint protocol to 58 train and 29 held-out rows. Warm-start accuracy is
`55.03%` versus `24.58%` for random, with warm ahead on AndroidControl, AgentNet, Mind2Web, and
MCPMark; exact sequence accuracy is still `0%`. This is the best current representation-transfer
evidence, but it remains text/accessibility-only and cannot substitute for native computer-use or
live email/Notion verification. The [m531 gate](paper/results/raw/m531-workshop-gate-current-m530-v1.json)
binds it and correctly passes both official BrowserGym/MiniWoB and MobileGym receipt checks.

The MCPMark acquisition itself is now deterministic and auditable: run
PYTHONPATH=.:src python scripts/download_mcpmark_trajectory.py --output <dir>
with the pinned revision, then pass the selected files through the redacting normalizer. The
[m532 manifest](paper/results/raw/m532-mcpmark-trajectory-acquisition-v1.json) records the
surface-balanced 10/5 train/eval selection and every downloaded file hash; no raw trajectory
archive is committed to the repository.
The [m533 receipt](paper/results/raw/m533-mcpmark-acquisition-normalization-reproducibility-v1.json)
binds those normalized bytes to the same acquisition manifest used by the transfer experiment.
The dated [m534 source audit](paper/results/raw/m534-official-realistic-source-integrity-audit-v1.json)
keeps the external benchmark contracts and release pins reviewable alongside the local receipts.

The current-checkpoint [m535 ToolSandbox smoke](paper/results/raw/m535-toolsandbox-native-current-v1.json)
executes the pinned upstream simulator in an isolated environment: the three bounded one-step
scenarios pass the milestone verifier (`3/3`, similarity `1.0`) and make no external API calls.
Because this is still a scripted-user smoke rather than the official split with the model-based user
simulator and full scenario matrix, the [m536 gate](paper/results/raw/m536-workshop-gate-current-m535-v1.json)
records the evidence but remains blocked on `official_split_not_verified`.

The [m537 release-preparation receipt](paper/results/raw/m537-local-hf-webgpu-release-prepare-v1.json)
binds the current checkpoint to a freshly generated HF model bundle and static WebGPU Space staging
directory. The hard ONNX/PyTorch parity gate passes for logits, hidden, and action graphs, with 63
dispatch tools. No upload or public URL is claimed: HF authentication and the anonymous post-upload
current-checkpoint audit are still required.

The [m538 candidate audit](paper/results/raw/m538-warm-realistic-candidate-webgpu-toolsandbox-v1.json)
binds the warm realistic continuation child to both deployment and native-runtime smoke evidence.
It preserves the source-disjoint `55.03%` held-out token result, passes all three local WebGPU
structured probes at `1,311.5` p50 input tok/s, and passes three bounded ToolSandbox milestone
checks. This child is not promoted to the current release because the ToolSandbox official split,
live-account side effects, and public Hub upload remain unverified.
On the repository's resettable in-memory trajectory suite, the same child reaches `13/13` exact
actions and state transitions across Gmail, Notion, and browser flows (`3/3` trajectories pass@1).
This strengthens the local deployment case but does not upgrade the result to real-account or
official benchmark success.
The candidate also passes the isolated RL preflight with two realized updates and nonzero policy
movement; mean held-out reward rises `0.00→0.10625`, but exact tool match remains `0%`. This is RL
simulation evidence only and is not a reason to promote the child to the public release.

The [m540 promotion audit](paper/results/raw/m540-head-preserved-rl-webgpu-promotion-audit-v1.json)
also tests exportability. Raw RL output is intentionally non-deployable because its structured
heads are invalidated; preserving the five frozen deployment-head groups restores WebGPU export,
without changing action-head weights. The resulting child still passes the bounded local suite,
but remains a candidate rather than a public release.
The pinned BrowserGym runtime also executes cleanly for a 16-episode canary, but success is `0/16`
with both grounding fallbacks disabled. The canary is diagnostic only; the 240-episode official
split remains unfulfilled for this child.

The current BrowserGym diagnosis is split explicitly by grounding contract.  The official
one-episode diagnostic [m510](paper/results/raw/m510-browsergym-current-canary-v1.json) keeps the
live accessibility-only adapter and remains `0/1`: the model routes to `click`, but MiniWoW's SVG
number controls have no actionable accessibility role, so fail-closed grounding produces `noop`.
The non-official [m521 coordinate/semantic canary](paper/results/raw/m521-browsergym-current-coordinate-semantic-canary-v1.json)
replays the same pinned task with DOM geometry as a sidecar and reaches `1/1` (five grounded
clicks).  This is evidence for the deployment bridge and its tests, not an official BrowserGym
score or visual-agent claim; the full 240-episode gate still requires the accessibility contract
without either fallback.
The current warm/random weight audit [m522](paper/results/raw/m522-realistic-cross-surface-warm-random-2step-comparison-v1.json)
also binds the same checkpoint to four public source projections.  It is intentionally a bounded
training canary, with `8` train and `4` held-out rows, and must not be substituted for the missing
full native Android, browser, desktop, and MCP receipts.
The post-freeze public-source audit is kept separately in
[`configs/data/realistic-agent-eval.supplemental.yaml`](../configs/data/realistic-agent-eval.supplemental.yaml).
It now adds twenty-four high-value sources—Computer Agent Arena, CUA-Lite AgentNet, OSWorld 2.0
trajectories, EnterpriseOps-Gym, MCPMark, ToolSandbox, AndroidWorld, BrowserGym, WebBench, and
BU Bench V1, plus their mobile and desktop companion rows—with explicit license, split, runtime,
and WebGPU-projection policies. These entries are catalog-only until an exact revision and
acquisition receipt are frozen; they do not silently become training data.

The newest safety additions are [VPI-Bench](https://github.com/cua-framework/agents), which
measures attempted and successful visual prompt-injection actions across browser and computer-use
platforms including Email, and [AgentCIBench](https://github.com/UKPLab/arxiv2026-agentcibench),
which scores contextual-integrity disclosure over `must_share`/`must_not_share` information. Both
are evaluation-only; their code revisions are pinned, while attack pages, scenario pools, judge
prompts, and traces remain excluded from SFT and public WebGPU bundles.

The latest source audit adds four deployment-relevant references. [KnowU-Bench](https://github.com/ZJU-REAL/KnowU-Bench)
is an interactive, personalized, and proactive mobile benchmark (192 registered tasks, 23 apps,
hidden profiles, exposed behavioral logs, and an online user simulator). [AppAgent](https://github.com/TencentQQGYLab/AppAgent)
provides a smartphone-agent evaluation release built around tap/swipe actions and Android
emulators. [GroundCUA](https://github.com/ServiceNow/GroundCUA) contributes dense desktop element
grounding annotations (screenshots and boxes), while [UI-TARS](https://github.com/bytedance/UI-TARS)
defines a useful computer/mobile/grounding action contract. All four are evaluation or protocol
references here: their screenshot assets, hidden profiles, app packages, and benchmark task text
are not admitted to the text-only WebGPU SFT bundle.

The current browser/desktop audit also records [Online-Mind2Web](https://github.com/OSU-NLP-Group/Online-Mind2Web)
and [OSWorld-Human](https://github.com/WukLab/osworld-human). Online-Mind2Web is a live-web
submission benchmark with a 2026 v2 schema and site-drift corrections; OSWorld-Human measures
steps and efficiency against human reference trajectories and explicitly warns against training on
its repository. Both are supplemental evaluation-only rows, not static SFT data. The OSWorld-V2
row is tied to the upstream `v2026.06.24` code tag and `osworld-v2-2026.06.24` release; its public
asset mirror is incomplete, so gated assets must be pinned and hashed before a native run.

The browser audit now includes [WebBench](https://github.com/Halluminate/WebBench), which covers
realistic live-site READ/CREATE/UPDATE/DELETE/file workflows, and [BU Bench V1](https://github.com/browser-use/benchmark),
which combines 100 encrypted tasks from WebBench, Mind2Web 2, BrowseComp, GAIA, and custom
challenges. Both remain evaluation-only here: live credentials and encrypted/decrypted task text
are excluded from SFT, model bundles, and public task artifacts. The WebGPU projection is limited
to DOM/action safety and routing diagnostics, not a live-browser or leaderboard claim.

The pinned [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) snapshot is also registered
as a supplemental public source.  Its strict first-action, multi-turn, and action-history
projections are useful for tool-call continuation, but remain offline diagnostics rather than
native MCP/BFCL or side-effectful productivity evaluation.

Run the read-only readiness report before acquiring or evaluating anything:

```bash
PYTHONPATH=src python scripts/realistic_agent_preflight.py
```

The report currently identifies the four local text-first adapters as runnable and all 38
environment/evaluation rows as blocked by their pending integration status (for example, no
`adb`, Docker, VM, or upstream BrowserGym checkout).  `--strict` intentionally exits non-zero
until those external runners are installed and pinned; this is a readiness gate, not a benchmark
score.

The current inventory also tracks recent realism upgrades: AndroidDaily (daily-use closed-source
mobile tasks), Windows Agent Arena (Windows desktop workflows), WebChoreArena (memory-heavy
browser chores), and TimeWarp (cross-version web UI drift).  They are evaluation-only until their
runtime dependencies, task revisions, and licensing terms are pinned.

### Workshop/publication gate

The catalog preflight is necessary but not sufficient for a workshop claim.  The stricter
[`scripts/workshop_gate.py`](../scripts/workshop_gate.py) requires explicit native receipts for
AndroidWorld, MobileGym, MobileSafetyBench, iOSWorld, BrowserGym/MiniWoB, OSWorld, AgentNet,
ToolSandbox, MCPMark, and EnterpriseOps-Gym; it also requires a hardware-WebGPU capability/latency
receipt, both transfer and no-transfer weight reports, a successful current-checkpoint-bound RL
preflight, and a public model/demo manifest.  It never
treats a protocol bridge, a synthetic state loop, SwiftShader, or a local checkpoint path as a pass:

```bash
PYTHONPATH=src python scripts/workshop_gate.py --strict
```

### Bounded multi-hop runtime contract

The deployed Python runtime now honors `Agent.chat(max_tool_hops=...)` for model-backed agents:
after each dispatch it appends the serialized `TOOL_RESULT` to the next prompt, stops at the hop
budget, and refuses to dispatch an identical tool/argument pair twice. This is the control-loop
primitive needed for workflows such as browser search followed by a staged Notion write. Tool
registries remain responsible for confirmation and authorization, and this unit-tested loop is
runtime evidence—not a native email, Notion, MCP, browser, or mobile benchmark score.

The command without supplied receipts exits non-zero with fifteen blocking requirements.  Supplying
the verified native WebGPU, full BrowserGym, MobileGym, and public-artifact receipts reduces this
to ten when the current checkpoint is also bound (nine native requirements plus the stale-public-
artifact binding):

```bash
PYTHONPATH=src python scripts/workshop_gate.py --strict \
  --webgpu-receipt docs/paper/results/raw/m265-webgpu-native-current-browser-context-v1.json \
  --native-receipt browsergym_miniwob=docs/paper/results/raw/m282-browsergym-current-checkpoint-official-v1.json \
  --native-receipt mobilegym=docs/paper/results/raw/m262-mobilegym-native-current-browser-context-v1.json \
  --weight-report docs/paper/results/raw/m25-weight-transfer-ablation-v1.json \
  --rl-preflight-receipt docs/paper/results/raw/m321-current-stateful-rl-preflight-v1.json \
  --public-artifact-manifest docs/paper/results/raw/m305-public-hf-legacy-current-audit-v1.json \
  --current-checkpoint runs/sft-webgpu-browser-context-adapter-20260802/latest.pt
```

The nine remaining native benchmark receipts are still absent.  The public artifact manifest now
verifies the already-live 28.32M-parameter byte model and static WebGPU Space, but it does not
bind the current checkpoint, so the current-checkpoint-aware gate marks it blocked until the
10.52M BPE model and its matching demo are uploaded and the manifest records the exact checkpoint
SHA-256.  It also does not prove native OS/emulator/MCP control or task success.  That is intentional:
four train adapters and the tracked offline receipts are useful progress, but they do not prove
those native task outcomes.  Once a native runner produces a receipt, it
can be supplied as `--native-receipt BENCHMARK_ID=PATH`; the receipt contract requires explicit
environment execution, official split verification, task count, and success rate.  The gate is
therefore an auditable workshop decision, not a claim that the current model has passed.

The current canonical receipt is joined in [`m218`](paper/results/raw/m218-workshop-gate-current-canonical-native-v1.json):
MobileGym and BrowserGym/MiniWoB are accepted as official-split native evidence, while the
ToolSandbox diagnostic remains blocked for missing official-split verification.  The latest
source-corrected current-checkpoint-bound report is [`m318`](paper/results/raw/m318-workshop-gate-source-corrected-v1.json):
it is `ready=false` with ten blockers (nine absent native receipts plus
`current_checkpoint_not_bound`). Every supplied native receipt must identify the same current
checkpoint SHA-256; older gate receipts are retained only for historical comparison.  The failed
canonical-toolcall RL preflight is recorded in [`m319`](paper/results/raw/m319-current-rl-preflight-v1.json),
while the current-checkpoint stateful email/Notion/browser fixture preflight in [`m321`](paper/results/raw/m321-current-stateful-rl-preflight-v1.json)
passes with reward diversity and real policy updates.  The latest RL-bound gate is [`m322`](paper/results/raw/m322-workshop-gate-stateful-rl-v1.json):
RL now passes, but the nine native receipts and current public-artifact binding remain blocked.

The [m323 continuation control](paper/results/raw/m323-current-stateful-long-continuation-v1.json)
also rejects a longer local curriculum: 64 SFT plus 8 RL steps increase reward diversity but
reduce held-out exactness and leave resettable email/Notion/browser completion at `0/5`.  Its
[weight audit](paper/results/raw/m323-current-stateful-long-weight-v1.json) shows large shared-body
movement with frozen action heads, so the child is not promoted or exported.

The follow-up [m326 low-rate transfer ablation](paper/results/raw/m326-current-stateful-lowrate-transfer-v1.json)
keeps the current checkpoint bound and compares frozen, low-rate-unfrozen, and matched-random
backbones on the same disjoint local fixture.  Low-rate updates improve shaped reward (`0.2031`
frozen → `0.2344`) and exact tool selection (`25%` → `37.5%`) while keeping backbone movement
below `0.2%`, but productive closed-loop completion remains `0/4` and total completion remains
`1/5` because only abstention succeeds.  It is therefore a weight-safety control, not evidence
for email, Notion, browser, MCP, or native WebGPU capability.

The [m329 current Mind2Web pointer transfer](paper/results/raw/m329-current-mind2web-pointer-transfer-v1.json)
uses the public train-only DOM/action projection with a deterministic seed and an explicit
17-row legacy pointer-vocabulary migration.  Pointer span exactness improves from `0/63` to
`6/63` (`9.52%`), while the shared embedding/attention/FFN move only `0.23%`/`0.27%`/`0.29%`.
The result is still a teacher-forced DOM transfer diagnostic: native BrowserGym replay, visual
grounding, and live browser success remain unproven, so the child is not exported.

The [m330 HF-format export](paper/results/raw/m330-hf-local-export-current-v1.json) rebuilds the
current `10,524,544`-parameter BPE bundle from the exact checkpoint after the pointer migration.
The local safetensors, agent heads, tokenizer, and config all have recorded byte hashes; the
config binds checkpoint SHA `bc1aca20…` and tokenizer SHA `83654055…`.  This closes the local
export reproducibility step, but `published=false` remains correct because Hugging Face
authentication and anonymous re-fetch verification have not occurred.

The new [`publish_hf_release.py`](../scripts/publish_hf_release.py) closes the operational gap
between local export and publication: it prepares the model and static Space together, refuses
missing tokens or missing audit output, uploads only after the parity verifier passes, and exits
without a current-release claim unless the anonymous Hub audit matches the checkpoint.
Its local-only end-to-end preparation is recorded in [m331](paper/results/raw/m331-hf-paired-release-local-v1.json):
the exact current checkpoint produced a verified model bundle and Space staging directory, while
publication remains false.

The [m334 paired receipt](paper/results/raw/m334-hf-paired-release-local-current-v1.json) is a
historical local preparation after the safety-policy update.  The model bundle and all four ONNX
graphs pass hard parity from the exact `bc1aca20…` checkpoint, and that staged Space recorded the
then-current `app.js` hash `a2d8fed1…`; later release edits intentionally make the hash differ.
This is still local-only evidence: HF authentication, upload, anonymous re-fetch, and a public URL
remain absent.

The follow-up [m336 aligned release receipt](paper/results/raw/m336-hf-paired-release-local-current-63-tool-v1.json)
fixes a concrete deployment contract mismatch: the HF config carried 63 inferred dispatch tools,
but the WebGPU metadata and selector sidecars defaulted to 50.  The publisher now binds one
63-name pool—50 standard tools plus 11 mobile and 2 productivity schemas—to the model config,
`meta.json`, dense selector, and retrieval selector.  The regenerated local bundle passes all four
ONNX parity checks and the shared tool-name hash is `fd3c5d30…`; this is deployment-integrity
evidence only, not native browser/mobile/MCP capability or a public release.

The [m337 MobileGym reconciliation](paper/results/raw/m337-mobilegym-current-catalog-reconciliation-v1.json)
updates the source-linked matrix to the already completed pinned native run: revision
`093a3292…`, 28 apps, 416 templates, and the complete official 256-task test split.  The current
checkpoint passes `1/256` (`0.39%`) using text/DOM state only.  This is native simulator/state-diff
evidence but not screenshot or visual-grounding evidence; MobileGym data remains evaluation-only
and no training rows were admitted.

The [m338 runtime-capability audit](paper/results/raw/m338-realistic-agent-runtime-capability-audit-v1.json)
records the current host's read-only dependency probes against all 40 catalog rows: four text-first
adapters are runnable and 36 environment/evaluation rows remain blocked.  The missing `adb`,
Docker/QEMU, BrowserGym/MCPMark modules, and unavailable CoreSimulator service are recorded as
infrastructure blockers, not benchmark failures.  The receipt downloads no benchmark payloads,
starts no external service, admits no evaluation rows to training, and makes no native-score claim.

The [m339 current CUA-Gym surface probe](paper/results/raw/m339-cua-gym-current-surface-probe-v1.json)
re-runs the public CC-BY-4.0 metadata-only transfer control from the exact current BPE checkpoint.
The backbone is frozen and the same 300-step linear surface head is trained for the warm and matched
random arms over 3,005 task-ID-disjoint rows.  Warm held-out accuracy is `77.55%` versus `77.87%`
random (balanced `77.36%` versus `78.50%`), so the measured transfer delta is negative and the
checkpoint is explicitly rejected for deployment on this probe.  No setup/reward artifacts,
screenshots, action traces, or native desktop/browser outcomes were used.

The [m340 current AgentNet projection](paper/results/raw/m340-agentnet-current-text-projection-v1.json)
runs the exact current checkpoint over all eight retained Ubuntu evaluation parents (133 projected
actions).  It produces complete predictions but `0%` first-action type accuracy, `0%` action score,
and `0/8` exact trajectories.  This is an offline text/action regression diagnostic only: screenshots,
desktop state, OS effects, and the upstream AgentNetBench runtime remain unconsumed.

The [m342 static browser smoke](paper/results/raw/m342-webgpu-static-browser-smoke-v1.json) catches
and fixes a release-surface mismatch in the visible demo banner: it now identifies the current BPE
63-tool bundle rather than the stale 50-tool byte-model text. The regenerated staging Space returns
HTTP 200 with no page errors and all static controls present; this headless host has no GPU adapter,
so the observed WASM fallback is explicitly not a hardware WebGPU claim.
The deployment verifier now also accepts the checkpoint SHA and expected tool count, preventing an
internally self-consistent but stale 50-tool bundle from passing the local release preflight.

The [m348 ToolSandbox native smoke](paper/results/raw/m348-toolsandbox-native-current-smoke-v1.json)
boots the pinned simulator and milestone verifier with the exact current checkpoint. The bounded
single-step protocol passes `2/3` scenarios, but the interactive message continuation passes `0/1`.
Because the official split and model-based user simulator were not run, this is a native diagnostic,
not an official ToolSandbox score or evidence of email/Notion/MCP completion.

The [m349 transfer control](paper/results/raw/m349-toolsandbox-weight-transfer-native-control-v1.json)
compares warm and matched-random frozen-head adaptation over 107 public projection rows with 20
source-disjoint parent scenarios. Warm selector top-1 is `85%` versus `80%` random, but native
single-step verifier success is identical at `2/3` for both arms; the transfer is rejected for
deployment.

The [m350 MCPMark native filesystem receipt](paper/results/raw/m350-mcpmark-native-filesystem-current-v1.json)
is the first current-checkpoint run through a real MCP stdio server and an independent public
verifier. On the pinned easy `structure_analysis` task, the model failed all three bounded turns
(`verifier_exit_code=1`) after selecting the wrong tools and malformed paths. The result is useful
as a concrete MCP tool/schema failure, but it is not an official MCPMark split score: the user
simulator and remaining Notion/GitHub/Postgres/Playwright services were not executed.

The [m351 native transfer control](paper/results/raw/m351-mcpmark-public-state-transfer-native-control-v1.json)
then trains the exact current parent on eight public state-summary trajectories and compares it with
a deterministic random-backbone control. Warm held-out token accuracy reaches `31.97%` versus
`8.97%` random, but both children fail the native filesystem verifier (`0/1`, exit code `1`).
This separates representation transfer from executable MCP competence: the warm child is rejected
for deployment and no native MCPMark score is claimed.

The [m352 strict gate re-audit](paper/results/raw/m352-workshop-gate-current-v1.json) binds the
MCPMark receipt to the exact current checkpoint and records the final publication decision:
`ready=false`. Catalog coverage, current MobileGym, BrowserGym/MiniWOb, native WebGPU capability,
weight ablations, and the stateful RL preflight pass. The gate still blocks publication for the
seven missing official native receipts (AndroidWorld, MobileSafetyBench, iOSWorld, OSWorld,
OSWorld-V2, AgentNet, and EnterpriseOps-Gym), non-official ToolSandbox/MCPMark splits, and the
missing anonymous current-checkpoint model/demo manifest.

The [m353 three-surface transfer control](paper/results/raw/m353-three-surface-transfer-current-v1.json)
repeats the current-parent versus matched-random protocol across public AndroidControl, AgentNet,
and train-only Mind2Web projections. Warm held-out token accuracy rises from `59.86%` to `67.58%`,
while random rises from `0%` to `9.41%`; the warm child is ahead by `58.17` points on the aggregate
and on each surface. The warm embedding, mixer, FFN, and normalization movements are only
`0.394%`, `0.204%`, `0.244%`, and `0.008%`, respectively, but exact sequence accuracy remains
`0%`. This is stronger lineage evidence for reusing the parent body as an initialization, not
evidence of native mobile, browser, or desktop success; the child remains unexported.

The [m356 browser-head control](paper/results/raw/m356-current-mind2web-browser-head-transfer-v1.json)
isolates the current browser route/selector heads on 219 public Mind2Web TRAIN decisions and 63
parent/typed-slot-disjoint held-out decisions. Warm frozen-head transfer improves selector top-1
from `19.05%` to `85.71%` and top-3 from `85.71%` to `100%`; a matched random head reaches the
same post-training scores. Warm head movement is much smaller (`0.232` route and `0.216` selector
relative L2 versus `1.154`/`1.030` random), which supports low-rate head adaptation as a safer
initialization experiment, not browser competence. Native BrowserGym, visual grounding, and
external side effects were not executed, so the child is not promoted.

The [m357 current WebGPU browser smoke](paper/results/raw/m357-current-webgpu-browser-realistic-actions-v1.json)
exercises the exact hash-verified 63-tool bundle in the in-app browser at a local HTTP origin.
The page loads with no request failures or page errors, reports `WEBGPU`, and exposes the expected
10.52M-parameter BPE/63-tool banner. A realistic email request routes to `send_email` with the
recipient grounded as `Dana`, and a Notion request routes to `notion_write` with the content
grounded as `quarterly report summary`; both stop at the explicit confirmation boundary.
Read-only URL navigation routes to `open_url` without a side effect. With the planner checkbox
enabled, the compound prompt produces a two-step `web_search → notion_write` plan; the second step
is staged for confirmation, so the Notion write is selected but not executed. This is local
routing and safety evidence only: it is not a hardware-adapter receipt, official BrowserGym or
MCPMark score, or public Hub release, and no external side effect occurred.

The [m360 current bundle verification](paper/results/raw/m360-current-webgpu-deploy-verification-v1.json)
rechecks the workspace artifacts against the current checkpoint rather than reusing the historical
m357 browser receipt. All eight ONNX/head/tokenizer artifacts are present, manifest-bound, and
parity-gated; the current bundle identity is `e0e19a8a…a296ed`, with no local deployment blockers.
This establishes a reproducible upload candidate, not a public Hub URL, physical adapter result,
throughput claim, or browser task-success score.

### Current paired HF + Space preparation rebuild (m364)

The [m364 receipt](paper/results/raw/m364-hf-paired-release-local-current-v1.json) reruns
`publish_hf_release.py` from the exact `10,524,544`-parameter checkpoint. The local model bundle,
all eight WebGPU artifacts, aligned 63-tool metadata, and copied static Space verify successfully;
the bundle identity remains `e0e19a8a…a296ed` and all ONNX parity gates pass. `hf auth whoami` is
still unauthenticated, so publication, anonymous re-fetch, and public model/Space URLs remain
false rather than being inferred from local staging.

### Current interactive ToolSandbox stress run (m368)

The [m368 receipt](paper/results/raw/m368-toolsandbox-native-current-interactive-v1.json) reruns
the exact current checkpoint in the pinned ToolSandbox simulator/verifier with bounded model
continuation after tool results (up to four agent turns). All three selected scenarios execute,
but none reaches exact completion (`0/3`, with milestone similarities `0.5`, `0.5`, and `0.0`).
This is a stronger stateful-runtime stress diagnostic than the one-step smoke, not an official
ToolSandbox score: the upstream official split and model-based user simulator were not run, so the
publication blocker remains open.

### Strict workshop gate after interactive ToolSandbox run (m369)

The [m369 gate receipt](paper/results/raw/m369-workshop-gate-current-toolsandbox-interactive-v1.json)
re-runs the strict publication contract with m368, the current native WebGPU receipt, BrowserGym/
MiniWoB, MobileGym, weight-transfer, RL-preflight, and public-artifact inputs. It remains
`ready: false`: ToolSandbox is blocked by `official_split_not_verified`; AndroidWorld,
MobileSafetyBench, iOSWorld, OSWorld, OSWorld-V2, AgentNet, MCPMark, and EnterpriseOps-Gym have no
supplied official native receipts; and the legacy public artifact is not bound to the current
checkpoint. The result is fail-closed and does not treat local packaging or protocol bridges as
publication evidence.

### Six-source public continuation at 32 steps (m372)

The [m372 receipt](paper/results/raw/m372-all-public-candidate-transfer-32step-v1.json) extends
the matched AndroidControl, AITW, AgentNet, Mind2Web, ToolACE, and xLAM public-train control to
`32` optimizer updates over `86` train and `80` source-disjoint held-out rows at a `512`-token
context. The warm current-parent arm improves aggregate held-out token accuracy from `51.31%` to
`55.35%`; the deterministic-random arm moves from `0%` to `7.43%`, so warm remains ahead by
`47.92` points and wins on every surface. Exact sequence accuracy is `0%` for both arms. Warm
embedding/attention/FFN movement is `0.405%/0.188%/0.225%` relative L2 (normalization `0.009%`),
while the random control moves those groups by `0.78–1.20×`; action heads remain unchanged.
This supports parent-weight reuse as an initialization candidate only—no child is exported or
promoted without native verifier-backed checks.

### Six-source public continuation at 64 steps (m377)

The [m377 receipt](paper/results/raw/m377-all-public-candidate-transfer-64step-v1.json) repeats
the same six-source, `86`-train/`80`-eval, `512`-token protocol for `64` optimizer updates. The
split validator now checks parent IDs and typed slots within each named source, while allowing
generic slot values to recur across unrelated datasets. Warm held-out token accuracy rises from
`51.31%` to `56.18%`; the matched random backbone rises from `0%` to `31.26%`, leaving a `24.91`
point warm advantage and a warm win on all six surfaces. Exact sequence accuracy remains `0%`.
Warm movement is largest in the embedding (`0.837%`), followed by FFN (`0.397%`) and
attention/mixer (`0.329%`); random movement is `0.78–1.20×`. The result supports parent geometry
as an initialization candidate only; no child is exported, and no official/native benchmark or
email/Notion side effect is claimed.

### Current publication gate with unified transfer evidence (m379)

The [m379 gate receipt](paper/results/raw/m379-workshop-gate-current-v1.json) now accepts the
nested warm/random compatibility report from m377 and verifies its parent SHA against the exact
current checkpoint. This removes a schema-version ambiguity in the weight gate without changing
the claim boundary: `ready` remains `false` because AndroidWorld, MobileSafetyBench, iOSWorld,
OSWorld, OSWorld-V2, AgentNet, MCPMark, and EnterpriseOps-Gym still lack official native receipts,
ToolSandbox is not official-split verified, and the public legacy model/demo manifest is stale.

The [m380 stateful runtime receipt](paper/results/raw/m380-current-stateful-runtime-v1.json) then
replayed five resettable email/Notion/browser/recovery/abstention tasks against the exact current
checkpoint. The oracle reaches `5/5` tasks and `16/16` accepted steps, validating the verifier; the
model reaches `0/5` tasks and `0/48` accepted attempts despite three retries per step. This is a
useful negative checkpoint-in-loop result: the runtime and reward contract work, but the current
model is not yet a productive closed-loop policy, and no native-account claim follows.

The [m381 receipt](paper/results/raw/m381-current-stateful-lowrate-80-v1.json) records a bounded
80-step continuation from that same parent with a `1e-5` backbone learning rate and `5e-3` head
learning rate, plus matched frozen and random-backbone controls. Low-rate training raises the
held-out selector top-1 from `53.33%` to `60.00%` and mean shaped reward from `0.2656` to `0.2813`,
while moving the backbone by only `0.242%`. The strict three-retry replay still completes only
`1/5` tasks and accepts `1/16` steps; the frozen and random controls also complete `1/5`. This is
therefore a useful weight-transfer diagnostic, not evidence for native mobile, browser, desktop,
MCP, real email/Notion, public-benchmark, or WebGPU capability, and the child is not promoted.

### ToolACE free-run parent/child control (m375)

The [m375 receipt](paper/results/raw/m375-toolace-parent-warm-free-run-v1.json) runs the parent
checkpoint and the m370 warm child through the same 16-row, 30-step ToolACE action-history probe.
The parent reaches `26.67%` tool exactness and `80%` schema validity; the warm child falls to
`23.33%` tool exactness with unchanged `0%` argument, step, and episode exactness. No tools are
dispatched and no external accounts are touched. This is negative free-run transfer evidence, so
the child is explicitly rejected for full-policy/WebGPU promotion.

### Current native ToolSandbox smoke (m366)

The [m366 receipt](paper/results/raw/m366-toolsandbox-native-current-v1.json) runs the current
checkpoint inside the pinned Apple ToolSandbox simulator and milestone verifier. Two of three
single-step scenarios are exact (`cellular_off` and `wifi_off`); the phone-message scenario reaches
`0.425` milestone similarity. The run executed the simulator/verifier with no external API and
binds checkpoint `bc1aca20…c16361`, but it deliberately did not run the official split or the
model-based user simulator. It therefore adds genuine native smoke evidence without clearing the
ToolSandbox publication blocker. The isolated Python 3.12 environment used compatible `ccy 1.4.0`
because the source-declared `ccy 1.3.1` has no Python 3.12 release.

### Unified public-candidate continuation and weight control (m361)

The [m361 receipt](paper/results/raw/m361-all-public-candidate-transfer-v1.json) joins six
publicly referenced projections—AndroidControl, Android-in-the-Wild (AITW), AgentNet, Mind2Web,
ToolACE, and xLAM—under one matched eight-step continuation. Each source contributes four train
and four held-out rows, with source-record or whole-parent separation preserved by the normalized
inputs. The warm arm starts from the current `bc1aca20…c16361` BPE checkpoint; the control starts
from a deterministic random backbone under the same optimizer, seed, batch, and sequence budget.

Warm held-out token accuracy is `53.72%` versus `0%` for the random control (`+53.72` points),
and the warm arm is ahead on all six surfaces. This is teacher-forced continuation, not exact
trajectory success: both arms are `0%` exact sequence accuracy. The AITW evaluation is a local
four-row holdout derived from public AITW train parent IDs, not the official AITW test split;
AndroidControl rows omit screenshots, and none of the six projections ran a native emulator,
browser, desktop VM, MCP server, verifier, or external account.

The tensor audit shows the warm shared-body movement stays below `0.1%` relative L2 for embedding,
mixer, FFN, and normalization, while the random control moves those groups by roughly `0.78–1.20×`.
Action heads are unchanged in both arms. The resulting decision is to retain the parent geometry as
an initialization candidate while keeping surface-specific adapters and all official/native
publication gates open; the child is not exported to WebGPU.

### Scaled six-source continuation control (m362)

The [m362 receipt](paper/results/raw/m362-all-public-candidate-transfer-scaled-v1.json) repeats
the same six-source protocol at `86` train and `80` held-out rows, using up to 16 rows per source,
an 8-step matched warm/random continuation, and a 512-token context. Warm held-out token accuracy
is `52.59%` versus `0%` for the deterministic-random control (`+52.59` points), and the warm arm
remains ahead on all six surfaces. Exact sequence accuracy is still `0%`, and Mind2Web/ToolACE
show small warm token regressions despite the aggregate advantage, so this is stronger lineage
evidence rather than a universal adapter claim.

The shared warm body remains compatible with the parent and moves by only `0.003–0.100%` relative
L2 across the major groups; the random body moves by `0.086–119.721%`. This larger control does not
change the deployment decision: no child export, native capability, official split score, visual
grounding, or email/Notion/MCP side effect is claimed.

The [m358 ToolSandbox function-masking transfer pilot](paper/results/raw/m358-toolsandbox-function-masking-transfer-v1.json)
adds a bounded public-data transfer experiment for the tool-catalog path. It uses the pinned
Apple ToolSandbox scenario projection and the `openai_full_catalog_v1` prompt contract, so a
deterministic opaque tool-name alias remains visible in the catalog and can be copied by the
decoder. The raw projection contains `107` train and `20` eval rows; `78` and `15` are schema
valid, while `29` and `5` are retained as invalid-coverage counts and excluded from the strict
renderer because required arguments are incomplete. Function masking doubles the valid pools to
`156` and `30` rows without changing descriptions, JSON schemas, arguments, user text, or tool
responses.

After a matched `16`-step CPU LM-only continuation, the warm child moves the shared body by
`0.1646%` relative L2 (embedding `0.1907%`, mixer `0.1919%`, FFN `0.2388%`) and improves the
bounded canonical-eval token accuracy from `55.84%` to `60.41%`; the matched-random child moves
`0.2518%` and reaches `10.15%` from `0%`. These are teacher-forced transfer and weight-movement
diagnostics, not tool execution, native simulator success, or a leaderboard score. The receipt
explicitly records that no ToolSandbox simulator, verifier, user simulator, external API, mobile
runtime, or desktop environment ran, so the official ToolSandbox gate remains closed.

The [m332 MobileSafetyBench text projection](paper/results/raw/m332-mobilesafety-text-policy-v1.json)
then exercises the deployed `side_effect_confirmation_v1` boundary over 90 public task rows and
3 QA rows without retaining task text in Git or executing Android actions.  The added narrow
`text_harm_block_v1` layer leaves all 45 helpful rows unblocked and produces `45` confirmations,
`23` allowed decisions, and `22` risk blocks on the public task projection (plus one QA block).
This is useful safety coverage for the WebGPU boundary, but it is explicitly not the benchmark's
Appium/ADB, screenshot, device-state, helpfulness, or safety score.

The [m335 MCPMark source audit](paper/results/raw/m335-mcpmark-current-source-audit-v1.json)
refreshes the catalog against the current public `MCPMark Verified` commit
`cd45b7f57923b9b3985467f5139927575f83141c`.  The pinned README reports `127` standard tasks and
`50` easy tasks (`177` total) across five MCP services; the Git tree contains six task-root
variants because `playwright_webarena` is represented separately.  The audit records `354`
metadata/description paths, Apache-2.0 licensing, and zero retained task payloads, runtime
executions, scores, or training rows.  Earlier `239`-row receipts are explicitly historical and
must not be combined with the current tree.

The current public computer-use result is [`m220`](paper/results/raw/m220-agentnet-current-text-action-evaluation-v1.json).
It evaluates 133 source-disjoint AgentNet text-action rows across eight unseen Ubuntu trajectories
against a matched random-backbone control.  Warm first-action type coverage is 100% versus 12.5%
random, but both arms have 0% exact trajectories and effectively zero coordinate/text action score;
screenshots and desktop state were not consumed.  This is evidence for retaining the compatible
pretrained body as a candidate initialization, not a native desktop or visual-grounding result.

The current-checkpoint [m327 Computer Agent Arena probe](paper/results/raw/m327-computer-agent-arena-current-v1.json)
extends the desktop audit to 256 unique public trajectories.  The instruction-only route gate is
100%, but first-action tool exactness is only 3.91% and pointer-action exactness is 0%; the model
mostly emits screenshot or abstention actions when a click/move is required.  This confirms the
deployment bottleneck is action grounding rather than computer-use route selection.  Screenshots,
accessibility state, later trajectory steps, and arguments were intentionally excluded, so this is
not a Computer Agent Arena, AgentNetBench, visual-grounding, or native desktop score.

The follow-up [`m221 continuation`](paper/results/raw/m221-agentnet-public-continuation-transfer-v1.json)
trains the same public AgentNet train projection for 32 steps from the warm m180 child and a
matched random backbone.  Warm held-out token accuracy rises 55.80% → 67.00% and route accuracy
18.80% → 83.46%, while the random control reaches 21.22% token accuracy and 0% route accuracy.
Neither child reaches an exact trajectory or meaningful coordinate/text action score, so the
result supports warm-weight reuse for future visual/action-grounded training only.

The cross-surface [`m222 bridge`](paper/results/raw/m222-agentnet-continuation-stateful-productivity-bridge-v1.json)
then runs the warm and random children through the local resettable email/Notion/browser/recovery
fixture.  Warm accepts 5/16 state transitions versus 1/16 random, but both complete only the
abstention task (1/5); email and Notion completion remain zero.  The oracle reaches 16/16 and 5/5,
so this is a valid fixture regression signal, not real-account or native benchmark evidence.

The [`m223 MCPMark bridge`](paper/results/raw/m223-agentnet-continuation-mcpmark-routing-bridge-v1.json)
reruns both m221 children on the pinned public MCPMark task-description manifests: 169 standard
rows and 70 easy rows covering filesystem, GitHub, Notion, Playwright, and Postgres.  Warm routing
is 19.53% standard / 22.86% easy versus 22.49% / 18.57% for random.  Warm standard routing is
concentrated in Notion (35.71%) and Playwright (92%), with filesystem, GitHub, and Postgres all at
0%; warm selector top-1 is 0%.  This is a public service-family routing proxy only.  No MCP server,
credential, verifier, pass@k aggregation, browser state, email/Notion side effect, or official
MCPMark score is claimed, and the task text was not used for training.

The actual SFT checkpoint has also been exercised through the local deployment path with
[`scripts/deploy_smoke.py`](../scripts/deploy_smoke.py).  The pinned
[`m24 receipt`](paper/results/raw/m24-local-deployment-smoke-v1.json) records 10 realistic prompts
covering browser, computer, calculator, text, and email intent.  It selected the expected tool on
4/10 prompts (browser 1/3, computer 1/4, productivity 1/1); the text prompt incorrectly emitted a
tool.  The handlers only echo calls, so this is a useful regression smoke and a concrete failure
sample, not browser/OS/email capability evidence.

The HF-format exporter was corrected before any publication attempt.  A local export of the same
checkpoint now includes the recorded 16K BPE tokenizer, checkpoint/tokenizer hashes, safetensors,
and `tool_head`, `ptr_head`, `route_head`, `dense_selector`, and `selector_proj`.  The
[`m26 receipt`](paper/results/raw/m26-hf-local-export-v2.json) binds all five file hashes and marks
`published: false`: this environment is not authenticated to Hugging Face, so no public model URL
is claimed.  That receipt describes the newer local 10.52M BPE export.  Separately, the older
public `danelcsb/localagent-tiny-30m-byte` model and `danelcsb/localagent-webgpu` Space are live
and hash-bound by [`public-model-demo-manifest-v1`](paper/results/raw/public-model-demo-manifest-v1.json).
The public deployment is a 28.32M-parameter byte model; it must not be described as the m46 BPE
continuation or as evidence of native Android, desktop-VM, MCP-server, email, or Notion task
success.

The current stateful-child checkpoint was exported again in the [`m164 receipt`](paper/results/raw/m164-hf-local-export-current-v1.json).
This bundle is `10,524,544` parameters with a `42.10 MB` safetensors file, the recorded 16K BPE
tokenizer, all five action-head groups, `63` inferred runtime tools, and `23` pointer arguments.
The inference is schema-guarded: it is enabled only for a legacy checkpoint whose tool head has
the known 51-class width (50 standard tools plus abstention); unknown head widths remain
metadata-free.  The bundle is locally verified but still unpublished because Hugging Face
authentication is absent.

The static WebGPU bundle was then exercised in the Codex in-app browser with the explicit
`?backend=webgpu` path.  The [`m165 receipt`](paper/results/raw/m165-webgpu-browser-smoke-v1.json)
binds the fetched graph, tokenizer, heads, metadata, checkpoint, manifest, and parity hashes before
the session is created.  The single-provider session was created successfully; the eight-case
synthetic DOM smoke measured model p50 `10.25 ms`, closed-loop p50 `33.3 ms`, schema validity
`87.5%`, and end-to-end DOM success `2/8` (`25%`).  This is a local deployment diagnostic only:
it is not the current m164 child, an official BrowserGym/MiniWob score, visual grounding, trusted
browser/OS control, or a public Hugging Face upload.  ORT Web does not expose per-node placement,
so the receipt records placement and fallback as unknown.

The currently public legacy Space was separately checked as a black box in [`m224`](paper/results/raw/m224-public-space-black-box-realistic-prompts-v1.json).
It is HTTP 200, running, and WebGPU-labelled, but its 10,986-byte `app.js` is an older byte-level
artifact rather than the current 10,524,544-parameter BPE bundle.  URL opening routed to
`open_url` (`1/1`), while `Email Dana the quarterly report` routed to `set_reminder` (`0/1`);
Search→Notion emitted a single `notion_write` proxy without executing or verifying either step.
No credentials or side effects were used.  This is a regression/publication-boundary receipt, not
an official browser, email, Notion, or MCP score, and the current bundle still requires an
authenticated Hugging Face upload and public re-fetch verification.

The current action boundary also has a separate local safety receipt, [`m227`](paper/results/raw/m227-webgpu-side-effect-safety-policy-v1.json).
It requires confirmation for email/Notion and other state-changing tools and blocks prompt-injection
or secret-exfiltration indicators before a real harness could execute them.  This policy is
deterministic and unlearned; it does not replace native safety benchmarks or prove external task
success.

The first native checkpoint-in-loop browser probe is now recorded in the [`m29 receipt`](paper/results/raw/m29-browsergym-native-model-eval-v1.json).
It executed the pinned BrowserGym/MiniWoB environment for all 240 episodes (60 task variants,
four fixed seeds) with live accessibility-tree observations.  This is a real environment result,
but only a one-step diagnostic: the current SFT checkpoint abstained on all 240 episodes, produced
zero grounded actions, and achieved 0/240 success.  It is explicitly not an official multi-step
BrowserGym score and does not support visual, WebArena, mobile-emulator, desktop-VM, or real-account
email/Notion claims.

### Browser-context contract adaptation (m30)

The native probe exposed a concrete distribution mismatch before any browser reward could be
learned: the SFT checkpoint was trained mostly on short action prompts, while the runner supplied a
goal followed by a live accessibility tree.  The tracked
[`train_browser_context_adapter.py`](../scripts/train_browser_context_adapter.py) addresses only
that interface contract.  It filters single-turn synthetic computer-use rows, adds deterministic
quoted element names and an abstention instruction, then performs 300 low-rate backbone updates and
1,000 route/selector probe updates.  It never reads BrowserGym goals, screenshots, task plans, or
labels.

On ten held-out projected rows (one each for click, double-click, drag, key press, move, open-app,
screenshot, scroll, type, and wait), route accuracy moved from `40%` to `100%`, tool accuracy from
`30%` to `100%`, and exact argument accuracy from `20%` to `60%`.  The complete hash-bound receipt is
[`m30`](paper/results/raw/m30-browser-context-adapter-v1.json).  This is evidence that the
observation-contract adapter is learnable, not a native BrowserGym result; the remaining argument
errors and the 240-episode m29 abstention result still block any browser-agent claim.  The native
runner now emits quoted accessibility names so a future run evaluates the same contract.

The replayed train-only adapter is recorded in [`m42`](paper/results/raw/m42-browser-context-adapter-replay-v1.json).
It reused 589 synthetic computer-use rows for 300 low-rate backbone updates plus 1,000 route and
selector-head updates.  On ten disjoint projected rows, route accuracy improved `40% → 100%`,
tool accuracy `30% → 100%`, and exact arguments `20% → 60%`.  This is a contract-transfer result,
not BrowserGym training; no BrowserGym goals, screenshots, task plans, or labels were read.

The complete pinned native split is now recorded in the [`m43 receipt`](paper/results/raw/m43-browsergym-native-adapter-full-model-eval-v1.json).
It ran the exact 240-episode plan (60 task variants, four fixed seeds, ten model steps per
episode) with BrowserGym `0.14.3`, MiniWoB++ at the pinned revision, Playwright `1.44.0`, and
Chromium `125.0.6422.26`; `official_split_verified` is true.  The train-only accessibility
contract adapter produced `5/240` success (`2.08%`): two `click-button` and three
`stock-market` episodes.  This is a reproducible native text/accessibility-tree ablation, not a
visual-agent, WebArena, real-account, or leaderboard score; the compact tracked receipt retains
per-task and per-seed aggregates and hashes the full local case log.  The unchanged base
checkpoint remains the m41 `0/240` baseline.

### Public Mind2Web trajectory continuation (m44)

To validate the public-data path independently of the older bounded `train_10` shard, the Hugging
Face dataset-server `rows` endpoint was queried at the pinned Mind2Web revision with a bounded
request for eight `train` records. Five records passed the positive-grounding and operation
allowlist, yielding 20 normalized Conversation variants. Four parent records (16 rows) were used
for continuation and one unseen parent (four rows) was held out; parent IDs and typed slot values
were disjoint. The acquisition, filtered-source, and normalized-output hashes are in the
[`m44 acquisition receipt`](paper/results/raw/m44-mind2web-acquisition-v1.json), while the
training metrics and child checkpoint hash are in the [`m44 continuation report`](paper/results/raw/m44-mind2web-trajectory-continuation-v1.json).

Starting from the 10.52M BPE WebGPU parent, 32 low-rate SFT updates plus 128 route/selector probe
updates reduced held-out mean loss `2.2349 → 1.6979` and raised assistant-token accuracy
`66.14% → 74.21%`. Sequence exactness remained `0/4`; the selector reached `7/11` tool
decisions (`63.64%`) and route accuracy reached `34/34`. These are text-first held-out
train-record metrics, not an official Mind2Web test score.

The paired [`m46 weight-transfer analysis`](paper/results/raw/m46-weight-transfer-analysis-v1.json)
shows why the pretrained backbone remains deployable: configuration, tokenizer, and all 51 shared
tensors are compatible; the attention/mixer, FFN, and embedding groups moved only `0.37%`, `0.45%`,
and `0.93%` in relative L2, respectively, while action heads moved `69.19%`. This supports a
small-learning-rate backbone plus larger head-rate recipe, but does not prove transfer is optimal;
the existing matched no-transfer ablation remains the required quality control.

The child was also exercised in a ten-episode pinned BrowserGym/MiniWoB canary using the explicit
`realistic_browser` pool (`web_click`, `web_type`, `web_select`). The [`m45 receipt`](paper/results/raw/m45-browsergym-native-realistic-toolpool-canary-v1.json)
records `0/10` success. This is a diagnostic only: Mind2Web backend-node IDs are not MiniWoB's
live DOM IDs, so the canary does not measure cross-site grounding or justify an official BrowserGym
score. The native runner defaults to the legacy `standard` pool and now exposes the realistic pool
explicitly to prevent silent vocabulary mismatch.

### Larger bounded public Mind2Web continuation (m46)

The next dataset-server request covered 16 additional `train` records. Twelve passed the same
fail-closed grounding rule after truncating each sequence to 12 actions, producing 48 normalized
rows. Nine parent records (36 rows) were used for training and three unseen parents (12 rows) for
evaluation; the acquisition receipt records zero parent and typed-slot overlap. Starting again from
the same 10.52M parent, 64 backbone updates and 256 frozen-feature route/selector updates raised
held-out assistant-token accuracy `65.75% → 78.73%` and reduced mean loss `2.2328 → 1.4992`.
Route accuracy reached `66/66`, selector top-1 reached `55/63` (`87.30%`), and sequence exactness
remained `0/12`. This is stronger public-train evidence than m44, but it is still not an official
Mind2Web test score, native browser success, or screenshot-grounding result.

### Bounded public AgentNet computer-use continuation (m47)

The public [OpenCUA AgentNet](https://huggingface.co/datasets/xlangai/AgentNet) Ubuntu trajectory
file was acquired at revision `d76ee50a63fad81cfdbe576416757d7c2091ed50` using an explicit
`0..8,388,607` byte range. The bounded prefix contains 40 complete parent tasks; a deterministic
parent-disjoint split uses 32 tasks for training and eight for evaluation. The new
[`ingest_agentnet_text.py`](../scripts/ingest_agentnet_text.py) adapter keeps task text and a
bounded textual screen observation, maps click/double-click/drag/keyboard/type/scroll/cursor/wait
actions into existing computer-use tools, and drops screenshots plus termination/triple-click
markers. The acquisition and projection identities are in the
[`m47 acquisition receipt`](paper/results/raw/m47-agentnet-acquisition-v1.json) and
[`projection metadata`](paper/results/raw/m47-agentnet-projection-metadata-v1.json).

Starting from the 10.52M BPE mobile-dispatch parent, 32 SFT updates on 513 projected train rows
reduced the source-disjoint 133-row teacher-forced mean loss `3.7804 → 2.6121` and raised token
accuracy `47.09% → 58.89%`. Sequence exactness remained `0/133`; route accuracy was `99.25% →
100%`, and the frozen-feature selector top-1 diagnostic moved `5.26% → 12.03%`. This is a
text-only desktop-action continuation, not an official AgentNet validation score, AgentNetBench,
OSWorld, screenshot-grounding, or native desktop result. The paired
[`m47 weight-transfer report`](paper/results/raw/m47-agentnet-weight-transfer-v1.json) found equal
configuration/tokenizer and 51 compatible tensors: attention/mixer, FFN, and embedding movement
was `0.29%`, `0.35%`, and `0.49%`, with action heads unchanged because this run intentionally
isolated backbone SFT. The result supports warm-start compatibility, not a claim that transfer
is optimal.

The matched [`m48 transfer ablation`](paper/results/raw/m48-agentnet-weight-transfer-ablation-v1.json)
uses the same rows, tokenizer, configuration, seed, 32 updates, and learning rate with a fresh
backbone control. After continuation, the warm-start arm reached `58.89%` held-out token
accuracy and mean loss `2.6121`, versus `21.19%` and `8.5930` for the fresh-backbone arm; both
arms remained at `0/133` sequence exactness. This isolates a strong initialization effect in this
bounded pilot, while leaving the no-transfer and native-runtime limitations explicit.

### Mixed public desktop + browser continuation and weight audit (m68)

To test the requested deployment mixture rather than another single-source arm, the same 10.5M
BPE WebGPU parent was continued on two source-record-disjoint public training projections:
513 Ubuntu AgentNet desktop rows (screenshots removed, textual observations retained) and 20
grounded Mind2Web browser rows.  The held-out set contains the corresponding eight AgentNet
parents plus one separate Mind2Web train parent, for 533 training and 137 evaluation rows total.
The complete input identities and pinned revisions are in the [`m68 receipt`](paper/results/raw/m68-mixed-public-agent-continuation-v1.json).

After 32 SFT updates at `1e-5`, held-out teacher-forced token accuracy improved from `51.67%` to
`60.05%` and mean loss from `3.5609` to `2.5574`.  The route head diagnostic improved from
`0.60%` to `14.37%`, but selector exactness remained `0/166` tool rows and sequence exactness
remained `0/137`.  This identifies the current WebGPU bottleneck precisely: shared language
adaptation moves, while grounded action/tool selection does not.

The paired [`m68 weight report`](paper/results/raw/m68-mixed-public-agent-weight-transfer-v1.json)
finds matching configuration and tokenizer across 51 tensors.  Relative movement is `0.490%`
for embeddings, `0.277%` for attention/mixer, `0.337%` for FFN, and `0.011%` for normalization;
the route, tool, pointer, and selector heads were unchanged because this was a backbone-only SFT
continuation.  These figures support reusing the parent with a smaller backbone learning rate,
but do not prove transfer is better than a matched random control.  The run is not an official
AgentNetBench or Mind2Web score, and it uses no screenshots, browser/desktop runtime, MCP server,
emulator, or external account.

### Mixed public action-head adaptation (m69)

The [`m69 receipt`](paper/results/raw/m69-mixed-public-agent-head-adaptation-v1.json) isolates
the action-head bottleneck found by m68.  It reuses the m68 mixed public continuation, performs
one warm-up SFT step whose learning rate is zero at the first schedule position, then trains only
the frozen-feature route and dense-selector probes for 800 steps at `5e-3`.  The public rows,
source revisions, and held-out IDs are unchanged from m68.

This controlled head update raises held-out route accuracy from `14.37%` to `99.40%` and dense
selector top-1 from `0%` to `57.83%` over 166 tool decisions, while teacher-forced language
metrics remain unchanged (`60.05%` token accuracy; `0/137` sequence exact).  The paired
[`m69 weight report`](paper/results/raw/m69-mixed-public-agent-head-weight-transfer-v1.json)
confirms `0%` movement in embeddings, attention/mixer, FFN, and normalization; movement is
concentrated in the route/action-head group.  This supports a two-rate recipe—small backbone
learning rate plus a separately trained candidate-conditioned selector—but it does not establish
native closed-loop success or that head transfer beats a matched random head control.  Screenshots,
OS/browser runtimes, MCP services, emulators, and external accounts were not used.

### Matched random-head control (m70)

The [`m70 receipt`](paper/results/raw/m70-mixed-public-agent-random-head-control-v1.json) repeats
the m69 head schedule from the identical m68 backbone, public rows, seed, and 800-step budget, but
initializes both route and dense-selector heads randomly.  It reaches the same held-out route
accuracy (`99.40%`) and selector top-1 (`57.83%`) as the parent-head m69 arm; the after-training
deltas are exactly zero.  The [`m70 weight report`](paper/results/raw/m70-mixed-public-agent-random-head-weight-transfer-v1.json)
again shows zero movement in the shared backbone and movement only in the action-head group.

This is the adoption insight: the compatible pretrained backbone is useful for the language
continuation, but inherited action-head weights do not provide a measurable advantage after a
matched frozen-feature retraining schedule.  The recommended WebGPU recipe is therefore a
low-rate backbone continuation plus independently initialized, candidate-conditioned route and
selector heads, followed by a native closed-loop validation.  m70 remains a bounded public-data
diagnostic rather than an official benchmark or end-to-end browser/desktop result.

### Public adaptation in the resettable productivity runtime (m71)

The [`m71 comparison receipt`](paper/results/raw/m71-public-adaptation-stateful-runtime-comparison-v1.json)
is the required closed-loop follow-up to the m69/m70 public projection diagnostics.  The evaluator
first runs an oracle through the deterministic local email, Notion, browser-search, recovery, and
abstention tasks, then runs both adapted checkpoints with three bounded retries per step.  A legacy
WebGPU checkpoint carries a 17-argument pointer vocabulary; the evaluator now migrates those shared
rows into the current 23-argument stateful vocabulary before decoding, so the comparison is not
silently invalidated by a tensor-shape mismatch.

The oracle completes `5/5` workflows and accepts `16/16` steps.  Both m69 (inherited heads) and m70
(random heads) accept `0/16` steps and complete `0/5` workflows, with the same deterministic model
event hash.  Thus the strong public text-projection selector result (`57.83%` top-1) does not
transfer to the 62-tool state-conditioned contract.  This is a useful negative result: the next
training unit must include state-conditioned trajectories, the stateful tool vocabulary, and
closed-loop/recovery rewards; retraining projected route/selector heads alone is insufficient.
The runtime is resettable and in-memory only, and this receipt is not an AndroidWorld, BrowserGym,
OSWorld, MCPMark, ToolSandbox, email, Notion, or native WebGPU benchmark score.

### Stateful closed-loop adoption from the public checkpoint (m89)

The [`m89 adoption receipt`](paper/results/raw/m89-public-to-stateful-closed-loop-adoption-v1.json)
records the next training and runtime decision.  Starting from the m70 AgentNet+Mind2Web public
continuation, the stateful probe adds 512 low-rate updates over the disjoint local productivity
contract plus one action-mismatch recovery view per decision.  The matched ablation moves the
backbone by only `0.422%` relative L2 (embedding `0.133%`, mixer `0.884%`, FFN `0.936%`,
normalization `0.0197%`) and improves the fixed-step selector diagnostic to `80%` top-1.

The deployment-shaped runtime then uses a bounded top-5 selector-first candidate set, remembers
exact rejected decoder outputs per episode, grounds labelled numeric/text spans, and exposes the
recovery URLs in the current-step observation.  The [`m89 runtime receipt`](paper/results/raw/m89-stateful-runtime-public-adaptation-v2.json)
reaches `5/5` workflows and `16/16` accepted steps after 20 attempts; the oracle is also `5/5`
and `16/16`.  The earlier m71 public-only arms were `0/5` and `0/16`, so this is evidence that
state-conditioned adaptation and execution-aware decoding—not public action-head projection alone—
are the useful adoption path.  The result remains a deterministic in-memory local diagnostic and
does not satisfy any official native benchmark gate.

### ToolSandbox native stateful-adaptation stress smoke (m90)

The current stateful child was also run inside the pinned Apple ToolSandbox simulator and
milestone verifier, rather than only through the local productivity state machine.  The five
selected scenarios cover state dependency, multiple tool calls, canonicalization, insufficient
information, and a multiple-user-turn task.  The runner executes the real upstream tools and
verifier, but its scripted user terminates after the first agent response; multi-tool and
multi-turn scenarios are therefore intentionally truncated.  This makes the result a native
execution stress signal, not an official ToolSandbox score.

The m70 public-only control and m75 stateful/error-recovery child both complete the
insufficient-information case but fail the four truncated multi-step cases: `1/5` (`20%`)
for each arm.  The matched result is useful because it rejects a false adoption claim: adding
stateful views improved the resettable local runtime, but did not yet transfer to ToolSandbox's
tool names, multi-step conversation contract, or user-simulator protocol.  Receipts are
[`m90 m75`](paper/results/raw/m90-toolsandbox-native-stateful-m75-v1.json) and
[`m90 m70 control`](paper/results/raw/m90-toolsandbox-native-public-m70-v1.json).  The official
ToolSandbox split, model-based user simulator, full scenario matrix, and optional RapidAPI tools
remain unexecuted, so the workshop gate stays blocked.

### ToolSandbox bounded multi-step transfer (m91)

The runner's `--interactive` protocol now lets the checkpoint receive each execution-environment
result and issue another bounded tool call, with a scripted user ending only after final text or
eight agent turns.  This is closer to the requested stateful computer/tool loop than m90, while
remaining distinct from ToolSandbox's model-based user simulator.  The run also hardens the
WebGPU path: overlong histories are suffix-truncated for feature extraction and grounded
candidates that cannot fit the 2,048-token model window are rejected rather than sent to RoPE.

On the same five ToolSandbox scenarios, the m70 public-only control, the m75 stateful/error-recovery
child, and the m59 ToolSandbox-projection child all score `1/5` (`20%`): only the
insufficient-information abstention passes, while state-dependent, canonicalization, and
multi-tool/multi-turn tasks fail.  This matched result shows that neither local productivity
adaptation nor static ToolSandbox projection training has transferred to the native tool vocabulary
or stateful conversational contract.  The interactive receipts are
[`m91 m75`](paper/results/raw/m91-toolsandbox-native-interactive-stateful-m75-v1.json) and
[`m91 m70 control`](paper/results/raw/m91-toolsandbox-native-interactive-public-m70-v1.json), plus
the [`m91 m59 projection arm`](paper/results/raw/m91-toolsandbox-native-interactive-public-projection-m59-v1.json).
They still do not satisfy the official-split/user-simulator gate.

### ToolSandbox native adapter hardening (m92)

The next matched rerun fixed two adapter-level grounding defects exposed by the m91 trace: system
policy prose was removed from the model-facing prompt so generic entity extraction cannot copy
the word `Don` from `Don't`, and phone extraction now prefers explicit `+...` numbers over UUID
digits in tool results.  The native loop also records each exact executed body in retry memory so
an error or successful lookup cannot repeat the identical candidate forever.  These changes are
runtime/adapter safeguards, not new training data.

The m92 m75, m70, and m59 arms remain `1/5` (`20%`) end-to-end success, but matched milestone
similarities improve on the stateful and multi-turn cases (m75/m70/m59: `0.25`, `0.3333`, `1.0`,
`0.0`, `0.5` in the fixed scenario order).  The result is a useful diagnosis—safe retries and
slot extraction remove avoidable failures, while planning the correct multi-tool sequence remains
unresolved.  Receipts are [`m92 m75`](paper/results/raw/m92-toolsandbox-native-interactive-stateful-m75-v1.json),
[`m92 m70 control`](paper/results/raw/m92-toolsandbox-native-interactive-public-m70-v1.json), and
[`m92 m59 projection`](paper/results/raw/m92-toolsandbox-native-interactive-public-projection-m59-v1.json).
They are still bounded scripted-user stress runs, not the official split, user simulator, or a
workshop-ready ToolSandbox score.

### AgentNet text-observation/action projection evaluation (m62)

The retained eight-parent evaluation projection was then run through the actual LocalAgent
constrained decoder and the repository's AgentNetBench-compatible scorer.  This is a stronger
checkpoint-in-the-loop test than teacher forcing: each of the 133 retained action rows is decoded
from its task plus bounded textual observation, grouped back into its parent trajectory, and
scored for action type, coordinates, text, keyboard, scroll, and sequence alignment.  The
[`m62 receipt`](paper/results/raw/m62-agentnet-text-projection-eval-v1.json) verifies all eight
parent IDs and records the exact source/model/projection hashes.

The warm-start SFT child gets `75%` first-action-type agreement, but mean trajectory score is
approximately zero, exact trajectory success is `0/8`, and coordinate/text/keyboard/scroll
families receive no useful credit.  The matched random-backbone child gets `0%` first-action-type
agreement and zero score.  This separates two facts: pretrained initialization transfers a
coarse action prior, while screenshot-dependent grounding and long-horizon state alignment do
not transfer to text-only WebGPU inference.  The run is explicitly not official AgentNetBench,
native desktop, OSWorld, or screenshot-grounding evidence; termination and triple-click source
actions dropped by the text projection are outside the score.

### Computer Agent Arena metadata audit (m49)

The public [Computer Agent Arena](https://huggingface.co/datasets/xlangai/computer-agent-arena)
metadata file was downloaded at revision
`897b9f45287c516a44f9e79879b14bc3c1bc5b0a` and hash-bound in the
[`m49 receipt`](paper/results/raw/m49-computer-agent-arena-metadata-v1.json).  The snapshot has
4,641 unique trajectories and 99,765 steps; 76,942 steps reference screenshots, 83,007 contain
textual thought, and 89,144 contain a conservatively recognized computer-use primitive.  The
metadata records a 57.90% human-evaluation correctness rate across the published trajectories.
The 50,609,777-byte JSONL was read, but no screenshot archive was downloaded or opened.

This is a source and modality-coverage audit, not a Computer Agent Arena score: the benchmark is
evaluation-only, screenshots are omitted from the text-first WebGPU path, and the rows were not
used for SFT.  The result makes the visual-grounding gap measurable rather than silently treating
thought text as an observation.  A future visual model must use the pinned image archive and a
task-disjoint protocol before any image-grounded or native desktop claim is made.

### Computer Agent Arena instruction-only checkpoint probe (m50)

The current 10.52M-parameter BPE WebGPU checkpoint was then evaluated on 128 deterministic,
source-disjoint first-action cases using only each public task instruction.  The probe is recorded
in the [`m50 receipt`](paper/results/raw/m50-computer-agent-arena-instruction-probe-v1.json) and
implemented by [`evaluate_computer_agent_arena_text.py`](../scripts/evaluate_computer_agent_arena_text.py).
The route head correctly identified `computer_use` on `128/128` cases, but exact local-tool
selection was only `5.47%` and action-family accuracy was `7.03%`; the model abstained on `9.38%`
of cases.  It over-produced `screenshot` for pointer tasks, which is the expected failure mode when
the model is asked to ground a coordinate action without an image or accessibility tree.

This is a deliberately leakage-safe action-prior diagnostic.  It does not use the published
thought traces, screenshots, later trajectory state, or action arguments, and it is not an official
Computer Agent Arena, AgentNetBench, visual-grounding, or native desktop result.  The result is
useful for model design: the route gate transfers, while grounding requires a visual/accessibility
encoder and a stateful action loop before further SFT or RL can be expected to improve real tasks.

## Bounded public Mind2Web training continuation

The public [Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web) training split was streamed
at revision `17ece8eb89862368edc0cc806acee6fca5163474` without placing raw data in Git.  Of the
first eight streamed records, five passed the adapter's fail-closed positive-grounding rule; they
normalized to 10 deterministic Conversation rows, 116 grounded tool calls, and the
`web_click`/`web_type`/`web_select` capability set.  The complete byte/hash/training receipt is
[`mind2web-public-train-sample-v1.json`](paper/results/raw/mind2web-public-train-sample-v1.json).

Sixteen BPE-tokenizer SFT updates from the corrected 10.5M mobile parent reduced mean loss from
`2.2790` to `1.7595` and raised assistant-token accuracy from `63.97%` to `74.29%`; exact
assistant trajectories remained `0/10`.  The transfer audit found 51 shared tensors, equal model
configuration and tokenizer identity, no shape additions/removals, and frozen action heads.  This
is legitimate public-data adaptation evidence, not a native Mind2Web score: the protected test
archive and browser environment were not executed.

### Held-out public Mind2Web continuation (m31)

The next public-data run downloaded only the pinned `data/train/train_10.json` shard, retained six
records whose complete action sequences had positive grounded backend-node IDs, and normalized 24
Conversation rows. Five source records (20 enriched rows) were used for training; the sixth source
record (four rows) was held out, with both parent-record IDs and typed slot values disjoint. Raw
Mind2Web bytes remain outside Git; the exact shard, filtered snapshot, normalized output, split
hashes, and manifest are bound in the [`m31 receipt`](paper/results/raw/m31-mind2web-public-train10-continuation-v1.json).

This run also closes a tool-catalog mismatch without changing the legacy 50-tool checkpoint shape:
[`REALISTIC_BROWSER_TOOLS`](../src/localagent/agent/toolset.py) extends the dense two-tower pool with
`web_click`, `web_type`, and `web_select`, and all three map to the stable `computer_use` route.
After 32 low-rate backbone updates and 256 frozen-feature head updates, held-out teacher-forced
token accuracy rose from `69.1%` to `80.0%`; the browser selector rose from `0/6` to `6/6` top-1
tool decisions. A generative replay was weaker—parseable calls `2/6` to `3/6`, exact tool names
`0/6` to `3/6`—because multi-turn context and argument copying still fail. This is public-train
generalization evidence, not an official Mind2Web test score or native browser execution result. The
extended 53-tool dispatch export also reproduced PyTorch route argmax and selector top-1 exactly
(`1.0` agreement; max score drift `1.26e-6`), proving that the new browser columns can be shipped
to a device without changing the backbone graph.

The adapted child was then exported through the browser bundle path.  The fp32/fp16 logits and
hidden graphs plus the distinct hidden-only action graph passed the hard parity gate; fp32 logits
drift was `6.91e-06`, fp32 hidden drift `5.66e-06`, and fp16 logits argmax agreement was `1.0`.
In the in-app browser, an explicit WebGPU session loaded the verified 10.5M bundle and the local
9-step mobile/email/Notion state suite produced schema validity `6/9` and closed-loop success
`4/9` (mobile `2/7`, productivity `2/2`).  The WASM control had the same quality result.  The
versioned receipt is [`m11-webgpu-mind2web-bpe-child.json`](paper/results/raw/m11-webgpu-mind2web-bpe-child.json);
no hardware throughput claim is made because timing was not collected in this run.

### Realistic browser-tool WebGPU bundle (m32)

The same held-out Mind2Web child was exported with the 53-tool
[`REALISTIC_BROWSER_TOOLS`](../src/localagent/agent/toolset.py) pool, including `web_click`,
`web_type`, and `web_select` while preserving the 10.5M backbone and the existing pointer-head
shapes.  The model graph, hidden-only action graph, serialized heads, tokenizer, and tool metadata
are hash-bound in the [`m32 receipt`](paper/results/raw/m32-webgpu-realistic-browser-tool-pool-v1.json).
The fp32 and fp16 export gates passed; the JavaScript grounding bridge and native PyTorch agreed
exactly with both ONNX variants on all 20 reused action cases (route, selected tool, grounded
arguments, and normalized action).

This closes an export-contract gap, not the capability gap.  The reused diagnostic suite has only
`0/19` tool-required exact actions and `0%` tool-selection accuracy, and no hardware WebGPU timing,
official benchmark score, browser environment, or real account was executed.  The bundle is
therefore suitable for a controlled demo artifact, but it is not a publication-quality browser-agent
result until the native and hardware gates in the workshop checklist pass.

### Grounded DOM pointer continuation and WebGPU check (m33)

The m32 result exposed a specific training-contract error: Mind2Web supplies positive/negative DOM
Candidates and backend node IDs, but the text-first normalized rows did not carry that observation
into the model context.  [`export_mind2web_grounded_rows.py`](../scripts/export_mind2web_grounded_rows.py)
now emits a bounded, deterministic 12-candidate snapshot before each action, preserving the public
revision, source-record split, and slot hashes.  The pointer head now supports a backward-compatible
browser vocabulary (`target_id`, `value`) without changing legacy checkpoint shapes; public browser
names are mapped only for auxiliary 51-way head supervision (`web_click` → `click`, etc.).

 A 32-step continuation warm-started the m31 10.5M model and retained its route/dense heads.  On six
held-out public-train decisions, exact pointer spans reached `3/6` (`50%`), but the fresh four-case
explicit-Metal-3 WebGPU action suite remained `1/4` exact (`25%`): tool selection was `3/3`, the
cancellation abstention was `1/1`, and all three web arguments copied a long candidate span starting
at `target_id=2932` instead of the gold `target_id=170`.  Schema validity was `4/4`; harness TTFA was
`18.65 ms` p50 and `19.385 ms` p95, with every case under 100 ms.  The full hash-bound receipt is
[`m33`](paper/results/raw/m33-mind2web-grounded-dom-webgpu-v1.json).

This is the desired realistic failure signal, not a success claim: compact DOM text alone does not
yet teach a 10.5M model to rank the correct candidate under a held-out website.  The next required
experiment is larger official-train coverage plus candidate-ranking/state-transition supervision,
followed by native browser execution; the four-case train-derived suite must not be promoted to an
official Mind2Web score.

### Gold-independent DOM candidate safety adapter (m34)

The m33 failure was specific: the model selected `web_click` correctly but its pointer span crossed
several serialized candidates and returned the wrong backend node.  The WebGPU runtime now has an
explicit safety adapter that ranks only the observed candidate attributes against task-token
overlap.  It selects the highest-scoring observed `target_id`; if no candidate matches, it preserves
the learned pointer output for unlabeled/icon-only controls.  The adapter does not read positive or
negative labels and is therefore a deployment policy, not a hidden test oracle.

On the unchanged four-case public-train held-out suite, the adapter improved complete structured
actions from `1/4` to `4/4`: three candidate-ranked web clicks and one cancellation abstention.
The explicit in-app WebGPU adapter was Apple Metal-3 (non-fallback), with p50 harness TTFA
`37.20 ms` and p95 `38.65 ms`.  The underlying learned pointer metric remains `3/6` exact spans;
the 4/4 result must not be described as pointer-head-only generalization or official Mind2Web
browser success.  See [`m34`](paper/results/raw/m34-mind2web-grounded-dom-webgpu-candidate-safety-v1.json).

### Native WebGPU action capability receipt (m40)

The [`webgpu-capability.html`](../spaces/localagent-webgpu/webgpu-capability.html) harness loads
the hash-bound 10,524,544-parameter action graph with an exact WebGPU provider request, queries
`navigator.gpu.requestAdapter()`, and records the exposed Apple Metal-3 adapter identity
(`vendor=apple; architecture=metal-3`). The expanded receipt covers three realistic local
dispatch cases—email, browser navigation, and a Notion write—with three warmups and 30 measured
repetitions per case. All three selected the expected tool and schema-valid argument on all 30
repetitions. The action graph's p50 input-throughput estimate was `1,355.9 tokens/s`, p50
end-to-end dispatch latency was `7.9 ms`, and the conservative graph-plus-host-tensor memory
estimate was `20.46 MB`.

This is now sufficient for the workshop gate's native WebGPU capability/latency requirement, but
it is not an external side-effect test: no email account, browser navigation, or Notion account
was touched, so `closed_loop_success` is explicitly `0`. The memory figure is an allocation
estimate because WebGPU/ORT exposes no driver VRAM counter. The two-case predecessor remains
[`m39`](paper/results/raw/m39-webgpu-native-capability-v1.json); the full three-case hash-bound
receipt is [`m40`](paper/results/raw/m40-webgpu-native-capability-notion-v1.json).

### Offline normalized mobile action protocol

[`src/localagent/eval/mobile.py`](../src/localagent/eval/mobile.py) now provides the common
prediction contract for AndroidControl and AITW rows after normalization. It extracts the
expected `mobile_*` calls from `localagent_v1`, accepts either `{name, arguments}` or
`{tool, args}` predictions, and reports tool accuracy, exact action accuracy, trajectory exactness,
optional coordinate proximity, and optional evaluator-provided target-box grounding. It validates
finite coordinates and malformed streams before scoring. This is an offline action diagnostic; it
does not simulate Android state or substitute for AndroidWorld's emulator reward. The emulator
runner should reuse this per-step contract while publishing device reward separately.

The scorer was replayed against the bounded normalized public training slices: 16 AndroidControl
records and 10 AITW records (130 tool calls total). Ground-truth replay is exact for both sources;
the sanity receipt is [`m22-mobile-action-score-replay-v1.json`](paper/results/raw/m22-mobile-action-score-replay-v1.json).
This validates the interchange and scorer only, not model quality.

### AndroidWorld emulator-result bridge

The official [AndroidWorld checkpointer](https://github.com/google-research/android_world/blob/main/android_world/checkpointer.py)
writes one `task_template_instance_id.pkl.gz` file per completed task instance under a timestamped
`run_...` directory. The upstream episode record contains the goal, task template, instance ID,
binary `is_successful` reward, episode length, runtime, exception information, and the full
step-level episode payload. [`src/localagent/eval/androidworld.py`](../src/localagent/eval/androidworld.py)
consumes that contract without importing AndroidWorld or starting `adb`: it hashes every result
file, rejects symlinks and malformed/duplicate instances, and reports per-task and overall
success rates. Supplying the exact expected task list and `n_task_combinations` is required for
`completeness.verified`; otherwise the receipt is intentionally marked `incomplete`.

Because upstream uses Python pickle, the parser's default loader accepts builtin-only fixtures.
Loading a full trusted emulator checkpoint requires the explicit
`--allow-unsafe-pickle` acknowledgement in [`scripts/aggregate_androidworld.py`](../scripts/aggregate_androidworld.py).
This is a local result-protocol bridge, not a live runner or an official AndroidWorld leaderboard
score. No AndroidWorld device result is published yet: the current environment lacks the
upstream emulator/`adb` prerequisites.

### AndroidWorld public task inventory (m51)

The public [`android_world/task_metadata.json`](https://github.com/google-research/android_world/blob/3e50888527ef9f29b9157ecd537e408008bb1c85/android_world/task_metadata.json)
was acquired at commit `3e50888527ef9f29b9157ecd537e408008bb1c85` and summarized in the
[`m51 receipt`](paper/results/raw/m51-androidworld-metadata-v1.json).  It contains 116 unique
task templates: 61 easy, 36 medium, and 19 hard; the median optimal budget is six steps, 96
templates are tagged parameterized, 24 require screen reading, and eight span multiple apps.
The task families cover messaging, calendar, contacts, browser, files, notes, recipes, settings,
media, and information retrieval—closely matching the requested mobile/email/productivity surface.

This receipt is a metadata inventory, not an AndroidWorld score.  It did not import AndroidWorld,
install APKs, invoke `adb`, launch an emulator, read screenshots, or feed task templates to SFT/RL.
The upstream environment reward and generated task instances remain required for a native mobile
claim; the inventory instead gives the WebGPU adapter a pinned, auditable task taxonomy and step
budget for future emulator runs.

### EnterpriseOps-Gym public card/API inventory (m52)

The pinned [EnterpriseOps-Gym dataset card](https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym)
and Hugging Face API metadata were acquired at revision `c8e538eae8a6205294f0a86675fefdc1fac408f6`.
The card/API profile is frozen in the [`m52 receipt`](paper/results/raw/m52-enterpriseopsgym-metadata-v1.json)
and records the source hashes, schema, split sizes, and tool-set modes.  It makes an important
source discrepancy explicit: the card's About paragraph says **1,150** expert-curated tasks,
while its domain table sums to **1,115**; the API's downloadable row counts are 649 (`oracle`) and
637 for each `plus_5_tools`, `plus_10_tools`, and `plus_15_tools` configuration.  The domain table
also reports 104 email tasks, 79 email tools, eight enterprise domains, 512 tools overall, and a
9.15-step average (maximum 34).

This is metadata-only evidence.  The profile did not download parquet rows, read task verifiers or
server configuration, start Docker/MCP services, execute a state transition, or add benchmark text
to training.  A native EnterpriseOps claim still requires the resettable containerized servers and
SQL final-state verifiers; the existing 67-row email retrieval receipt remains a name-only selector
diagnostic, not task success.

## What can be trained

Only the following public demonstrations are currently training candidates:

| Source | What it teaches | WebGPU representation | Policy |
| --- | --- | --- | --- |
| [AndroidControl](https://github.com/google-research/google-research/tree/master/android_control) | Human mobile tasks over accessibility trees, screenshots, goals, and JSON actions; 15,283 demonstrations over 833 apps | Goal + compact accessibility tree → normalized action JSON | Train only on the official training partition; keep app/task generalization holdouts separate |
| [Android in the Wild](https://github.com/google-research/google-research/tree/master/android_in_the_wild) | Large-scale gesture and multi-step mobile behavior (715k episodes) | Instruction + screen description → gesture/action JSON; screenshots remain native-only | Train only after verifying the official archive and attribution; never mix evaluation episodes |
| [xLAM Function Calling 60K](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | Tool selection and canonical argument filling | Compact tool schema + request → canonical JSON tool call | Existing `xlam_v1` adapter; exact denylist protects external evaluations |
| [Mind2Web train](https://huggingface.co/datasets/osunlp/Mind2Web) | Grounded browser navigation and forms | Cleaned DOM/accessibility tree + task → CLICK/TYPE/SELECT | Existing `mind2web_v1` adapter; only official train files |

AndroidControl is particularly useful for a first realistic converter because its official format
contains high- and low-level instructions, accessibility trees, screenshots, and JSON actions.
The converter should discard pixels for the text-only model, preserve a deterministic compact tree,
and retain the original episode/app/task identifiers in provenance.  A visual WebGPU policy can be
added later as a separate model; it must not silently change the text model's input contract.

## What must stay evaluation-only

The following are important reality checks, not extra SFT rows:

- [AndroidWorld](https://github.com/google-research/android_world): 116 hand-crafted tasks in 20
  apps with dynamically generated variations and live-emulator rewards.
- [WebArena](https://github.com/web-arena-x/webarena): 812 self-hosted realistic web tasks.
- [BrowserGym/MiniWoB++](https://github.com/ServiceNow/BrowserGym): reproducible browser primitives
  for smoke tests, not a substitute for enterprise workflows.
- [OSWorld](https://github.com/xlang-ai/osworld) and [OSWorld-Human](https://github.com/xlang-ai/OSWorld-Human):
  desktop multi-application workflows and human trajectories; use a VM runner and do not claim an
  official leaderboard score from a local run.
- [AgentNet / OpenCUA](https://github.com/xlang-ai/OpenCUA): 22.6K human-annotated computer-use
  tasks across Windows, macOS, and Ubuntu, with an offline AgentNetBench trajectory evaluator.
  Its action-reduction and state-action matching protocol is especially relevant to a compact
  state-conditioned WebGPU policy; keep screenshots, task labels, and official OS/app holdouts out
  of training until the dataset terms and split hashes are verified.

- [AppWorld](https://github.com/StonyBrookNLP/appworld): protected task/app/API-specific bundles;
  never unpack those into plaintext training data.
- [tau-bench](https://github.com/sierra-research/tau-bench) / tau3-bench: interactive user-agent-tool
  conversations and pass^k stability.
- [Apple ToolSandbox](https://github.com/apple/ToolSandbox): state dependencies, user simulation,
  intermediate milestones, and final state checks.
- [Toolathlon](https://github.com/hkust-nlp/Toolathlon): 32 applications, 604 tools, 108 long-horizon
  tasks, including Notion and email-like productivity workflows.  The repository terms need to be
  checked before any redistribution, so it remains evaluation-only.
- [MCPMark Verified](https://github.com/eval-sys/mcpmark): pinned Notion, GitHub, filesystem, Postgres,
  and Playwright environments with deterministic verifiers.
- [EnterpriseOps-Gym](https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym): the pinned
  card reports about 1,150 expert-curated tasks in its About paragraph, while its domain table
  totals 1,115 (including 104 email tasks), across eight domains and 512 tools.  The API exposes
  649 rows in `oracle` and 637 rows in each distractor configuration; these are configuration
  views, not additional task claims.  The official card describes containerized execution with
  SQL state verifiers and reports Apache-2.0; tasks/verifiers stay evaluation-only to prevent
  benchmark memorization.  The pinned HF revision is `c8e538eae8a6205294f0a86675fefdc1fac408f6`.
  The card/API reconciliation is frozen in the [`m52 metadata receipt`](paper/results/raw/m52-enterpriseopsgym-metadata-v1.json).
  The card-level inventory (104 email tasks) is not the same as a downloadable config row count:
  the bounded oracle and `plus_15_tools` email configs used below expose 67 matching task IDs, so
  the retrieval receipt reports 67 and retains 104 only as inventory context.
- [MCP-Persona](https://github.com/wwh0411/MCP-Persona): 173 personalized tool-calling tasks that
  include Notion and email MCP servers.  Its repository terms must be checked before redistribution;
  keep the simulated account state and checkpoint labels evaluation-only.
- [TUA-Bench](https://github.com/facebookresearch/TUA-Bench): 120 execution-based terminal tasks,
  including document editing, email management, and live-web information seeking.  It is CC-BY-NC
  and explicitly a benchmark, so it is not a training corpus.

The repository now has a fail-closed `agentnet_v1` adapter in
[`src/localagent/data/agentnet.py`](../src/localagent/data/agentnet.py) and a bounded normalizer in
[`scripts/normalize_agentnet.py`](../scripts/normalize_agentnet.py). It accepts both the official
`steps`/`ground_truth_actions` sample shape and the Hugging Face `traj`/`value.code` JSONL shape,
parses only literal `pyautogui`/`computer` calls, and requires a textual observation. Coordinate
actions remain `agentnet_*` tools for offline scoring; they are intentionally not relabeled as the
text-grounded WebGPU `click` or `type_text` tools. The one-sample integration receipt is
[`m17-agentnet-offline-adapter-sample-v1.json`](paper/results/raw/m17-agentnet-offline-adapter-sample-v1.json).
[`src/localagent/eval/agentnet.py`](../src/localagent/eval/agentnet.py) now supplies a dependency-free
AgentNetBench-compatible proxy for coordinate, text, keyboard, scroll, termination, and action-count
scoring. A ground-truth replay of the official sample scores `1.0` across 12 actions, which is only
a scorer sanity check; the pinned receipt is
[`m18-agentnet-scoring-proxy-sample-v1.json`](paper/results/raw/m18-agentnet-scoring-proxy-sample-v1.json).

A pinned metadata-only snapshot of the public Hugging Face export is also profiled in
[`m19-agentnet-metadata-profile-v1.json`](paper/results/raw/m19-agentnet-metadata-profile-v1.json):
22,532 unique task rows across Windows (12,364), macOS/Darwin (5,168), and Ubuntu (5,000), with
2–131 steps and 13 recorded low-level action types. The source JSONL is 18,840,344 bytes with
SHA-256 `9bb101e8373cd8cd1316f29d53c938b378f96aae1f09776a32bcc27454a0184d`; no screenshots or
trajectory payloads were consumed. This is an inventory receipt, not training data or an
AgentNetBench score.

The same boundary is now applied to a pinned [Toolathlon-GYM](https://github.com/eigent-ai/toolathlon_gym)
checkout. The configuration-only profile in
[`m20-toolathlon-gym-config-profile-v1.json`](paper/results/raw/m20-toolathlon-gym-config-profile-v1.json)
covers all 503 finalpool tasks and 25 MCP servers without opening task prompts, workspaces,
preprocessors, evaluators, or ground-truth outputs. It records 258 email tasks, 118 Notion tasks,
63 Playwright/browser tasks, 162 calendar tasks, and 419 filesystem tasks; the most common
multi-server pair is `emails+filesystem` (193 tasks). This gives the WebGPU tool bridge a concrete
long-horizon productivity gate while keeping the benchmark evaluation-only.

The pinned [MCPMark](https://github.com/eval-sys/mcpmark) metadata profile at commit
`cd45b7f57923b9b3985467f5139927575f83141c` is recorded in
[`m21-mcpmark-metadata-profile-v1.json`](paper/results/raw/m21-mcpmark-metadata-profile-v1.json).
The checkout contains 239 metadata rows: 169 `standard` L3 tasks and 70 `easy` L1 smoke tasks,
covering 38 Notion, 35 browser/Playwright, 40 filesystem, 33 GitHub, and 93 database tasks.
State types are text (133), URL (71), and video (33); the profile retains neither descriptions nor
state assets. MCPMark's own runner supports repeated runs and pass@k/pass^k aggregation, so it is
the next appropriate stability gate once the local MCP/browser dependencies are installed.

The current upstream [tau2-bench](https://github.com/sierra-research/tau2-bench) repository is the
maintenance line for the original tau-bench paper and now points new work to tau3-bench.  Its
base split must be used for a complete evaluation, and the leaderboard protocol requires repeated
trials for Pass^k; a single trajectory replay or a historical tau-bench number is not a current
deployment score.

[`src/localagent/eval/mcpmark.py`](../src/localagent/eval/mcpmark.py) now provides a fail-closed
local result aggregator: it requires every expected task for every run before reporting
pass@1/pass@k/pass^k, mean turns, latency, and token use. No result has been produced here because
Docker/MCP services are not installed; the implementation is a protocol bridge, not a score.

### MCPMark task-router proxy (m37)

To quantify the model's current productivity/tool gap without pretending to execute external
accounts, [`eval_mcpmark_task_router.py`](../scripts/eval_mcpmark_task_router.py) consumes only the
public `meta.json` and `description.md` files from the pinned MCPMark checkout. It evaluates both
the 169-task standard suite and the 70-task easy suite against service-level tool families:
Notion, Playwright/browser, filesystem, GitHub, and Postgres. Task text is used transiently for
feature extraction and is not retained in the receipt; verifiers and MCP servers are never run.

The frozen 62-tool AndroidControl dispatch child achieved only `15.90%` combined route accuracy,
`0%` selector top-1, and `0.42%` selector top-3 on 239 tasks. This is a useful transfer failure:
mobile action adaptation does not transfer to stateful Notion/email/browser/database workflows.
The result is a service-routing proxy, not MCPMark task success, pass@k, or leaderboard evidence;
the complete source and checkpoint hashes are in [`m37`](paper/results/raw/m37-mcpmark-task-router-proxy-v1.json).

### MCP service-contract transfer probe (m38)

The first routing failure is now separated from the language-model representation. The
[`train_mcp_service_probe.py`](../scripts/train_mcp_service_probe.py) experiment creates 28 short
service/tool-contract examples for filesystem, GitHub, Notion, Playwright, and PostgreSQL. It
trains only the five-way route head and the description-based dense selector; the 10.5M language
model backbone is copied unchanged (aggregate backbone delta `0.0`). No MCPMark description,
state fixture, verifier, or server is used as training input.

On the same untouched 169-task standard and 70-task easy MCPMark descriptions, the child reaches
`46.86%` combined route accuracy, `37.24%` selector top-1, and `69.87%` selector top-3, compared
with m37's `15.90%`, `0%`, and `0.42%`. Standard is `50.89% / 37.87% / 66.27%`; easy is
`37.14% / 35.71% / 78.57%`. This is a meaningful head-transfer result, not an execution score:
the official MCPMark runner, stateful MCP servers, and verifiers are still required before any
claim of task success. The complete hash-bound receipt is
[`m38`](paper/results/raw/m38-mcp-service-contract-probe-v1.json).

### MCP service-contract matched no-transfer control (m53)

To test whether the m38 selector result came from the pretrained representation rather than only
head fitting, the same 28 generated service/tool contracts, tokenizer, architecture, 800 route
updates, 800 selector updates, and seed were run with a fresh random backbone.  The control was
evaluated against the same 239 public MCPMark task descriptions; task text was transient, and no
MCP server or verifier was executed.  The [`m53 receipt`](paper/results/raw/m53-mcp-service-contract-no-transfer-v1.json)
records the child hash and self-hash.

| Condition | Route accuracy | Selector top-1 | Selector top-3 |
| --- | ---: | ---: | ---: |
| Pretrained backbone frozen (m38) | 46.86% | 37.24% | 69.87% |
| Fresh random backbone (m53) | 47.28% | 23.43% | 64.44% |
| Random − pretrained | +0.42 pp | −13.81 pp | −5.44 pp |

This is bounded transfer evidence: the pretrained representation materially improves tool-family
selection on unseen Notion/browser/database descriptions, while route classification is unchanged.
It is not MCPMark task success, pass@k, verifier success, or evidence that live accounts can be
controlled; the official runner remains a workshop-gate requirement.

### Mind2Web→MCPMark matched transfer probe (m93)

The same 28 service/tool-contract rows were then trained with the frozen m46 checkpoint, whose
backbone had been continued on public Mind2Web browser trajectories.  A second arm used the same
architecture, parent checkpoint, rows, seed, and 800+800 head-update budget but replaced the
backbone with deterministic random weights.  Neither arm used MCPMark task text during training.
On the untouched pinned 169 standard plus 70 easy descriptions, the Mind2Web transfer arm scored
`26.36%` route accuracy, `22.18%` selector top-1, and `65.27%` top-3.  The matched random arm
scored `47.28%`, `17.15%`, and `65.27%`, respectively: transfer improves exact tool selection by
`+5.02` percentage points but loses `20.92` points on service routing and does not change top-3.

The transfer receipt records zero backbone movement; the dense selector and route-head movement is
`92.49` and `4.82` relative units, so this is a head-only diagnostic rather than a learned MCP
representation.  It is deliberately negative evidence against adopting the Mind2Web backbone as
a universal MCP router.  The receipts are [`m93 transfer`](paper/results/raw/m93-mind2web-mcp-service-contract-v1.json)
and [`m94 matched random control`](paper/results/raw/m94-mind2web-mcp-service-random-matched-v1.json); MCP servers,
state fixtures, verifiers, and official MCPMark scoring were not executed.

The required third arm is now recorded in [`m96`](paper/results/raw/m96-mind2web-mcp-service-lowrate-matched-v1.json):
the same m46 backbone was unfrozen at `1e-5` for 200 updates while route/selector heads used
`5e-3`.  Low-rate unfreezing raises route accuracy to `48.12%`, just `+0.84` points over random,
but lowers selector top-1/top-3 to `15.90%/55.23%`.  Relative backbone movement is `0.297%`
(embedding `0.059%`, mixer `0.602%`, FFN `0.697%`, normalization `0.013%`).  Across all three
arms, the decision is therefore mixed: retain the m46 initialization for browser work, but do not
promote low-rate MCP adaptation or frozen transfer as a general tool-use weight policy.

The three service-contract children were then run through the same five native scripted-user
scenarios.  [`m97 low-rate`](paper/results/raw/m97-toolsandbox-native-interactive-mcp-lowrate-v1.json),
[`m98 frozen`](paper/results/raw/m98-toolsandbox-native-interactive-mcp-frozen-v1.json), and
[`m99 random`](paper/results/raw/m99-toolsandbox-native-interactive-mcp-random-v1.json) all complete
only `1/5` (`20%`), with identical pinned ToolSandbox source and runner hashes.  This agrees with
the proxy evidence: neither frozen transfer nor low-rate unfreezing improves stateful native
ToolSandbox completion.  These receipts use a bounded scripted user; the official split,
model-based user simulator, full scenario matrix, and optional RapidAPI tools were not run.

### Cross-surface public continuation and weight audit (m100)

To test whether one small WebGPU backbone can absorb the requested mobile, browser, and desktop
surfaces together, the same 10.52M-parameter BPE parent was continued for 32 low-rate updates over
4,629 public-train projection rows: 4,096 AndroidControl mirror rows, 513 AgentNet Ubuntu rows,
and 20 Mind2Web train records.  The 1,041-row evaluation is source-disjoint: 904 AndroidControl
test rows, 133 AgentNet held-out parents, and four held-out Mind2Web train records.  The catalog
links AndroidControl's primary Google source; this continuation's input is the explicit
Apache-2.0 OfficerChul mirror, alongside AgentNet and Mind2Web.  The receipt records those public
references, file hashes, split metadata, and visual omission.

Aggregate teacher-forced token accuracy rises from `57.67%` to `62.58%`, driven by mobile
`59.28%` → `65.45%`; desktop changes `49.70%` → `48.87%`, and the four-row browser slice changes
`69.09%` → `67.27%`.  Relative movement is small—embedding `0.494%`, mixer `0.205%`, FFN
`0.250%`, normalization `0.010%`—with action heads unchanged.  This is useful evidence that the
parent is compatible with all three text-first interfaces, but it is also a negative adoption
signal: a single low-rate mixed continuation does not improve every surface and is not a native
Android, browser, desktop, email, Notion, or MCP score.  The reproducible receipt is
[`m100`](paper/results/raw/m100-cross-surface-public-continuation-v1.json), and the runner is
[`train_cross_surface_continuation.py`](../scripts/train_cross_surface_continuation.py).

### Cross-surface matched random-backbone ablation (m101–m102)

The missing weight-adoption control uses the identical m100 data hashes, parent, BPE tokenizer,
32-step schedule, batch size, learning rate, and optimizer seed, but replaces the model backbone
with deterministic random weights.  The [`m101 random receipt`](paper/results/raw/m101-cross-surface-random-continuation-v1.json)
and [`m102 paired comparison`](paper/results/raw/m102-cross-surface-transfer-ablation-v1.json)
bind both arms to the same 4,629 train and 1,041 evaluation rows.

After training, the warm-start arm beats random by `+30.76` percentage points aggregate token
accuracy (`62.58%` vs `31.82%`), with advantages on mobile `+27.09` points (`65.45%` vs `38.36%`),
desktop `+47.11` (`48.87%` vs `1.76%`), and the four-row browser slice `+60.00` (`67.27%` vs
`7.27%`).  This is stronger evidence that the pretrained representation is useful than m100
alone, but it remains a teacher-forced text/accessibility comparison: both arms retain the same
action heads, no native environment executes, and no real email, Notion, or MCP side effect is
claimed.  The adoption decision is to reuse the verified BPE parent as initialization while
keeping surface-specific adapters and requiring native closed-loop evidence before merging them.

### Cross-surface dispatch-head adaptation ablation (m103–m105)

The m100/m101 continuation children were then given the same frozen-backbone route and dense
selector probe: 200 cached-feature updates, batch size 128, learning rate `5e-3`, and head seed
2029.  The probe expands the 1,041 evaluation conversations to 1,044 tool-routing decisions
(904 mobile, 133 desktop, and seven browser).  The backbone and language-model action logits are
unchanged; only the route/selector heads move, with action-head relative movement `0.709%` in
both arms and zero movement in embedding, mixer, FFN, and normalization groups.

Warm and random heads both reach `100%` route accuracy on mobile and desktop.  On the desktop
tool rows they tie at `73.68%` selector top-1; on the seven-row browser slice, warm reaches
`100%` selector top-1 versus random `33.33%`, while route accuracy is `85.71%` versus `100%`.
Across all tool decisions, warm selector top-1 is `74.82%` versus random `71.94%` (`+2.88` pp),
but aggregate route accuracy is `99.90%` versus `100%`.  This is therefore a surface-specific
head result—not evidence that the pretrained backbone universally improves dispatch, and not an
end-to-end mobile, browser, desktop, email, Notion, or MCP success result.  The reproducible
receipts are [`m103 warm`](paper/results/raw/m103-cross-surface-warm-head-v1.json),
[`m104 random`](paper/results/raw/m104-cross-surface-random-head-v1.json), and the matched
[`m105 comparison`](paper/results/raw/m105-cross-surface-head-ablation-v1.json); the runners are
[`train_cross_surface_dispatch_heads.py`](../scripts/train_cross_surface_dispatch_heads.py) and
[`compare_cross_surface_heads.py`](../scripts/compare_cross_surface_heads.py).  Keep separate
surface adapters/heads and require native closed-loop verification before publication.

### Current native ToolSandbox head ablation (m106–m108)

To connect the public cross-surface children to a real stateful tool environment, the warm m103
and random m104 checkpoints were run through the pinned ToolSandbox simulator and milestone
verifier on the same five multi-turn scenarios used by the earlier MCP transfer triad.  Both runs
executed the environment and verifier with the bounded scripted user, completing `1/5` scenarios
(`20%`) and reaching identical partial milestone scores on four scenarios.  The only difference
was the ambiguous contact-removal case, where random reached `0.333` similarity and warm `0.0`.

The matched [`m108 comparison`](paper/results/raw/m108-toolsandbox-native-head-ablation-v1.json)
therefore finds no warm native advantage (`0.0` percentage-point success difference).  The
[`m106 warm`](paper/results/raw/m106-toolsandbox-native-warm-v1.json) and
[`m107 random`](paper/results/raw/m107-toolsandbox-native-random-v1.json) receipts bind the
checkpoint hashes, ToolSandbox revision, runner hash, scenarios, and complete per-scenario
verifier outputs.  This is stronger than a text-only projection, but it is still not the official
ToolSandbox split: the upstream model-based user simulator, full scenario matrix, and optional
RapidAPI tools were not executed.  It is negative evidence against promoting the dispatch-head
transfer as a stateful capability.

### Current warm-head WebGPU bundle (m109)

The warm m103 checkpoint was exported as a fresh browser bundle with the 16K BPE tokenizer,
full-logits and hidden-only action graphs, serialized heads, and dispatch metadata.  The exporter
hard parity gate passed for fp32 and fp16 graphs (fp32 maximum hidden difference `5.47e-6`, fp16
`4.50e-3`, argmax agreement `1.0`), and a clean temporary copy of the static app verified every
manifest byte count and SHA-256.  The [`m109 receipt`](paper/results/raw/m109-webgpu-warm-head-deploy-v1.json)
binds the bundle to checkpoint `d81771f6…`, the 10,524,544-parameter stage, and tokenizer
`83654055…`.  This proves a reproducible local deploy artifact; it does not prove a hardware
WebGPU adapter, throughput, native task success, or a public Hub/Space upload.

### Current warm-head browser probe (m110)

The clean m109 staging copy was exercised in the Codex in-app Browser with four realistic
requests: two emails, a Notion page creation, and a search→Notion workflow.  The page reported
`Model ready` and `webgpu`, and each request returned in 11–43 ms without external side effects;
however, all four selected `computer_use.click` rather than the expected email, search, or Notion
tool.  The [`m110 receipt`](paper/results/raw/m110-webgpu-warm-head-browser-probe-v1.json) records
this as `0/4` exact dispatches and a current-bundle regression.  It is direct browser evidence
against publication of this checkpoint, not a native WebGPU adapter or benchmark score.

### Deployment dispatch repair and browser verification (m111)

The m110 failure was traced to a head-data mismatch: the m103 probe covered AndroidControl,
AgentNet, and Mind2Web surface rows but had no deployment-specific email/Notion adapter examples.
The [`m111 trainer`](../scripts/train_deployment_dispatch_repair.py) therefore freezes the 10.52M
backbone, mixes the same 4,731 public-train decisions with 1,030 deterministic productivity/browser
adapter rows, and trains matched warm/random route and dense-selector arms.  The warm arm reaches
5/6 canonical route decisions and 4/6 canonical tool decisions offline; the matched random arm
reaches 4/6 and 4/6, so this is not evidence of a universal warm-start selector gain.

The repaired bundle passes the same ONNX parity/hash gates and produces 5/5 exact single-step
browser tool names (email twice, Notion, web search, and URL open) with corrected recipient/content/
URL grounding.  With planner mode enabled, the bounded local workflow returns exactly
`web_search → notion_write` in two steps.  URL/search selection and planner normalization are
explicit safety adapters, not hidden benchmark labels.  The [`m111 receipt`](paper/results/raw/m111-deployment-dispatch-repair-v1.json) keeps those policies separate from learned-head metrics and
continues to disclaim native adapters, external side effects, and official benchmark success.

### AgentNet surface-selector follow-up (m112)

The bounded AgentNet text projection exposed a separate contract issue: the public rows retain
low-level `agentnet_*` tool names, while the browser repair trains generic `click`/`type_text`
names.  The [`m112 receipt`](paper/results/raw/m112-agentnet-surface-selector-repair-v1.json)
adds a surface-specific selector to the same checkpoint, freezes the 10.52M backbone, and compares
warm and random selector arms on the same 513/133 projected rows.  Warm selector top-1 is 70.68%
versus 71.43% random; the end-to-end projection reaches 1.0 first-action type but 0/8 complete
trajectories because screenshots, coordinates, and desktop execution are absent.  A legacy
list-wrapped AgentNet argument form was also normalized in the evaluator.  This is explicit
negative transfer evidence, not native AgentNetBench or OSWorld success.

### ToolSandbox static projection replay (m113)

The same m111 warm checkpoint was also run through the pinned 20-row ToolSandbox static projection.
With each row's candidate list preserved, the decoder produced 30% exact tool names and 100%
schema-valid calls; the global 50-tool selector control produced 0% exact names and 5% schema-valid
calls.  Canonicalization rows are only 21.43% exact and state-dependent rows are 0%, which warns
that a valid JSON call is not stateful task success.  The [`m113 receipt`](paper/results/raw/m113-toolsandbox-text-projection-current-checkpoint-v1.json)
records the pinned source revision and explicitly excludes the simulator, user model, verifiers,
official split, and external services.

### EnterpriseOps-Gym public email retrieval ablation (m114)

The m111 warm head was evaluated on 67 public EnterpriseOps-Gym email rows with 15-tool distractor
pools, after dropping server configuration and SQL verifiers.  Warm retrieval reaches hit@1/3/5 of
20.90/59.70/86.57%, while the matched random checkpoint reaches 13.43/47.76/76.12%.  The warm arm
therefore beats random on all three retrieval depths, but trails the older m14 diagnostic at hit@1
(26.87%).  The [`m114 receipt`](paper/results/raw/m114-enterpriseopsgym-email-retrieval-m111-ablation-v1.json)
keeps this as an out-of-domain name-only diagnostic, not EnterpriseOps-Gym task success or real
email execution.

### xLAM-derived function-calling transfer (m115)

The official Salesforce xLAM repository is gated in the current environment, so the evaluator uses
128 rows from a hash-pinned public Apache-2.0 derivative test shard and records the original link
separately.  Row-local retrieval reaches 50% first-tool exactness and 100% schema validity, but
argument exactness is 0%.  The global selector reaches only 0.78% first-tool exactness for warm
weights versus 0% for the matched random control.  The [`m115 receipt`](paper/results/raw/m115-xlam-derived-function-calling-transfer-v1.json)
therefore supports retrieval/schema diagnostics, not official xLAM, multi-call, or live API success.

### MCPMark service-routing transfer (m116)

The pinned public [MCPMark](https://github.com/eval-sys/mcpmark) checkout contributes 169 standard
and 70 easy task descriptions spanning Playwright/browser, Notion, filesystem, GitHub, and Postgres
services.  The m111 warm checkpoint and matched random control both reach 14.79% route/top-1 on
standard and 14.29% on easy.  Playwright is 100% in both arms, while the other four service
families are 0%.  The [m116 receipt](paper/results/raw/m116-mcpmark-routing-transfer-v1.json) keeps
this explicitly as a description-level routing proxy: no MCP server, state transition, verifier,
official leaderboard score, or training artifact is claimed.

### MCPMark trajectory-log provenance (m117)

The public MIT-licensed [MCPMark trajectory-log dataset](https://huggingface.co/datasets/Jakumetsu/mcpmark-trajectory-log)
provides real multi-turn tool traces.  The [m117 receipt](paper/results/raw/m117-mcpmark-trajectory-metadata-v1.json)
profiles one hash-pinned filesystem trace with 45 events and 21 paired calls across five tools.
Only metadata is retained: raw prompts, arguments, assistant text, and tool outputs are excluded,
so this is acquisition evidence rather than an evaluation score or training artifact.

### MCPMark redacted trajectory SFT transfer (m118)

The [m118 receipt](paper/results/raw/m118-mcpmark-redacted-trajectory-sft-transfer-v1.json) records
a content-audited continuation experiment over three public MIT-licensed trajectory files from the
same dataset: filesystem and Notion are training rows, and Playwright is a held-out service.  The
normalizer ([`normalize_mcpmark_trajectory.py`](../scripts/normalize_mcpmark_trajectory.py)) retains
user requests and structured tool arguments, redacts all tool outputs and assistant free text with
fixed markers, and rewrites absolute paths to short workspace suffixes.  The resulting 67 paired
calls are valid canonical `Conversation` rows and are bound by source/output hashes.

With 16 CPU SFT steps, both the warm and matched-random parents produce identical teacher-forced
metrics: training loss falls 4.9972→4.5967 and the held-out Playwright loss falls 5.6167→5.5599,
but held-out sequence accuracy remains 0%.  The warm/random parent and child backbones are
bitwise identical, so this run shows no body-level transfer advantage.  A constrained first-action
probe on the unseen Playwright row predicts `browser_type`/`browser_click` instead of the target
`browser_navigate` in all four arms.  This is a redacted SFT and decoder diagnostic only—not an
official MCPMark result, native browser/MCP execution, verifier success, or evidence that the tiny
sample is sufficient for deployment.

### MCPMark dynamic selector transfer (m119)

The [m119 receipt](paper/results/raw/m119-mcpmark-dynamic-selector-transfer-v1.json) trains a
22-tool two-tower selector on the 51 assistant decisions in the redacted filesystem and Notion
rows, then evaluates the 19 tool decisions in the held-out Playwright row.  The global catalog is
constructed from the public row schemas, while tool outputs and assistant free text remain fixed
redaction markers.  The backbone is frozen during selector training, and a deterministic random
backbone with the same configuration, tokenizer, seeds, and optimizer is the control.

| Arm | Train top-1 | Held-out Playwright top-1 | Held-out top-3 | Held-out top-5 | Held-out top-10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Warm WebGPU body | 95.83% | 0% | 0% | 10.53% | 84.21% |
| Random body | 97.92% | 0% | 0% | 0% | 0% |

This is a useful retrieval-side transfer result: the pretrained body ranks many unseen browser
tools in its top ten, but exact top-1 selection and argument grounding still fail.  The evaluator
does not start MCP services, launch Playwright, run a verifier, or report an official MCPMark score;
the global catalog includes held-out tool schemas, so this must not be read as vocabulary discovery
or native browser competence.

### MCPMark broad redacted SFT transfer (m120)

The [m120 receipt](paper/results/raw/m120-mcpmark-broad-redacted-sft-transfer-v1.json) broadens the
public trajectory experiment to ten hash-pinned MIT traces from the [MCPMark trajectory dataset](https://huggingface.co/datasets/Jakumetsu/mcpmark-trajectory-log): two filesystem, two Notion, two GitHub, two Postgres, and two Playwright tasks.  Filesystem/Notion/GitHub/Postgres rows provide 107 training calls; both Playwright rows (24 calls) are held out.  Tool outputs and assistant free text are fixed redaction markers, and no service is executed.

| Arm | Held-out loss before→after | Held-out token accuracy before→after | Exact sequence after |
| --- | ---: | ---: | ---: |
| Warm pretrained body | 4.5228 → 4.3138 | 39.68% → 42.27% | 0% |
| Matched random body | 9.8292 → 9.3376 | 0.69% → 3.12% | 0% |

The warm arm is +39.15 percentage points over random after identical 32-step SFT.  Its aggregate
backbone movement is 0.345% relative L2, compared with 102.06% for the random arm; action heads
were fixed.  This supports the adoption policy “reuse the pretrained body, use low-rate backbone
updates, and train surface-specific heads,” but it is not an official MCPMark score, native
Playwright/MCP execution, verifier success, or real GitHub/Notion/Postgres activity.

### MobileGym + OSWorld-V2 source audit (m121)

The [`m121 receipt`](paper/results/raw/m121-mobilegym-osworld-source-audit-v1.json) adds a
hash-pinned provenance pass for the two most relevant next native gates.  [MobileGym](https://github.com/Purewhiter/mobilegym)
provides 28 simulated mobile apps, 416 parameterized templates, a 256-task held-out test split,
structured reset/fork/diff state, and deterministic judges.  Its code is Apache-2.0 but its
released benchmark data is CC-BY-NC-4.0, so the data archive is not copied into SFT or public
artifacts.  [OSWorld-V2](https://github.com/xlang-ai/OSWorld-V2) is bound to the active
`osworld-v2-2026.06.24` release with 108 tasks, gated task/assets, and a release-matched VM/image
contract.  Neither native runner was available in this environment; the receipt is therefore a
source/protocol audit with zero training rows and zero official scores, not a mobile or desktop
capability result.

### MobileGym official split profile (m122)

The [`m122 receipt`](paper/results/raw/m122-mobilegym-source-split-profile-v1.json) extracts only
the official split metadata from the pinned MobileGym source archive.  It verifies 160 train task
IDs and 256 test task IDs with zero train/test overlap (416 unique tasks); the seven payment and
fourteen high-risk IDs are recorded as overlapping safety subsets.  No task text, simulator state,
or benchmark content enters the repository, and the profiler reports zero training rows and zero
simulator/native scores.  The next valid experiment is a release-compliant simulator run using
the test split, not SFT on the benchmark tasks.

### MobileGym native runtime smoke (m123)

The [`m123 receipt`](paper/results/raw/m123-mobilegym-native-runtime-smoke-v1.json) proves that
the pinned MobileGym frontend builds and serves locally, Chromium can load its 360×800 simulator,
and the official task registry resolves every train/test ID.  The structured state bridge is
available, but repeated reset snapshots differ at timestamp fields; this is recorded as a runtime
reproducibility issue rather than hidden.  No model action, task judge, or official success rate
was produced, so the strict gate now requires a separate release-compliant `native:mobilegym`
receipt.  The same gate also requires `native:osworld_v2` for the dated desktop release.

### MobileGym text-only model probe (m124)

The [`m124 receipt`](paper/results/raw/m124-mobilegym-model-probe-v1.json) is a bounded native
episode using the current 10.52M checkpoint and one official MobileGym test ID.  It uses the
documented `mobile_*` action bridge with a compact DOM-text projection and no screenshot input.
The model produced two `mobile_input_text` actions and the state judge passed `0/1`.  The receipt
stores only hashes and aggregate judge fields, is marked `native_receipt_eligible: false`, and
therefore cannot satisfy the publication gate; it is a direct negative result for the current
text-first mobile adapter.

### MobileGym answer-tool adaptation and transfer control (m125)

The [`m125 receipt`](paper/results/raw/m125-mobilegym-answer-adaptation-v1.json) records a bounded
surface adaptation rather than silently turning the m124 failure into a benchmark claim.  The
training input is the public AndroidControl/AITW train projection (4,096 rows) with its disjoint
held-out file (904 rows), plus the existing synthetic mobile rows and eight generic answer-action
examples.  No MobileGym task text, answer value, simulator state, or judge output entered SFT.
The additive `mobile_submit_answer(message)` schema is translated to MobileGym's native answer
action only during evaluation.

Two 300-step, frozen-backbone dispatch-head arms are matched on data and budget.  The pretrained
head warm start reaches `42.92%` held-out selector top-1, while the random-head control reaches
`47.46%`; both route at `100%`.  Weight inspection finds 51 same-shape tensors with matching
configuration and tokenizer.  The backbone remains unchanged (`0.0%` relative ΔL2) in both
arms; action-head movement is `44.83%` for warm and `93.72%` for random.  This supports retaining
the compatible pretrained body while treating the surface head as an ablation variable, not as
a capability guarantee.

On the official `notes.ReadTodoText` test task, each arm executes two native calls and scores
`0/1`.  The warm arm does select the new answer action, but its copied `message` is empty, so all
three answer fields fail.  The result is explicitly `native_receipt_eligible: false`: it is a
useful grounding/argument failure record, not a MobileGym score, screenshot-grounding result,
or publication-gate pass.  The next mobile milestone is answer-span/pointer training plus a
larger multi-task native run, not promotion of this adapter.

### MobileGym pointer/copy adaptation control (m126)

The [`m126 receipt`](paper/results/raw/m126-mobilegym-pointer-adaptation-v1.json) follows that
failure with the smallest targeted intervention: the backbone stays frozen, while the existing
copy head receives 300 updates from 4,225 train-side rows containing literal string arguments.
This includes the public AndroidControl/AITW train projection, generic mobile answer examples,
and state-conditioned productivity rows; the 904-row public evaluation file remains held out.
The synthetic answer rows now carry explicit `message` spans, so pointer supervision is real
span supervision rather than an empty-argument selector label.

The warm and matched-random arms both reach `25%` exact pointer spans on four held-out string
rows and both emit non-empty arguments on the native probe.  Neither reaches the answer task:
both execute two `mobile_submit_answer` calls and score `0/1`, with all three answer fields still
wrong.  Configuration/tokenizer compatibility remains exact, backbone movement remains `0%`,
and action-head movement is `64.84%` for warm versus `104.77%` for random.  The intervention
removes the empty-span symptom but does not solve multi-answer state grounding, so it is retained
as a diagnostic and not promoted to a MobileGym or publication result.

### MobileGym official Notes native sweep (m127)

The [`m127 receipt`](paper/results/raw/m127-mobilegym-notes-sweep-v1.json) runs the pointer-
adapted warm checkpoint against all five official Notes IDs in MobileGym's test split:
create-folder/move, create-with-reminder, delete-completed, delete-one, and read-todo-text.
The environment and deterministic state judges execute natively for every task, but the probe is
intentionally capped at two model steps, so this is not the full benchmark protocol and remains
non-gating.

The result is `0/5` success.  Every task is routed to `mobile_submit_answer`, including the
create/delete workflows that require app navigation and state transitions.  The two delete tasks
reach judge progress `0.5` before the repeated answer action, while the create and read tasks make
no judged progress.  This sweep establishes a broader routing failure: pointer supervision fixes
the empty argument symptom, but the model still lacks state-conditioned mobile action selection.

### MobileGym state-balanced routing and reranking ablation (m130)

The [`m130 receipt`](paper/results/raw/m130-mobilegym-state-routing-rerank-v1.json) tests two
deployment-side changes without touching the language-model backbone.  First, the dispatch
continuation adds generic Notes-like state/action rows (no MobileGym task text or state values)
and balances navigation, tap, typing, back, and wait examples.  Second, the native probe exposes
the existing selector beam as a bounded `--selector-top-m` option so grounded body reranking can
compare several candidates.

The warm held-out public selector rises from `42.92%` to `47.12%`, with exact configuration and
tokenizer compatibility and `0%` backbone movement.  On the native create-folder task, top-1
still selects `mobile_submit_answer` and scores `0/1`; top-8 reranking changes the action to
`mobile_wait` but also scores `0/1`.  The matched random arm selects different low-level actions
under top-8 and also scores `0/1`.  This is a useful routing/reranking failure analysis, not
evidence of mobile capability or a publication-gate pass.

### MCPMark current-checkpoint routing proxy (m60)

The same untouched pinned MCPMark descriptions were rerun with the m56 stateful child and the m59
ToolSandbox-SFT child.  The [`m60 receipt`](paper/results/raw/m60-mcpmark-current-checkpoint-router-proxy-v1.json)
binds all 169 standard and 70 easy rows, both checkpoint hashes, and the source manifests.

| Child | Standard route / selector top-1 / top-3 | Easy route / selector top-1 / top-3 |
| --- | --- | --- |
| m56 | 17.75% / 3.55% / 13.61% | 12.86% / 2.86% / 12.86% |
| m59 | 20.71% / 2.37% / 14.79% | 22.86% / 1.43% / 14.29% |

ToolSandbox SFT therefore transfers some service-family routing but does not improve exact MCP
tool selection on this untouched proxy.  The task descriptions, state fixtures, servers, and
verifiers remain evaluation-only; this is not MCPMark task success or a leaderboard result.  The
next meaningful training input must be a permitted schema-conditioned MCP corpus, followed by the
official stateful server/verifier run.

### ToolSandbox public scenario metadata profile (m54)

The pinned [Apple ToolSandbox](https://github.com/apple/ToolSandbox) source is now inventory-bound
at commit `165848b9a78cead7ca7fe7c89c688b58e6501219`.  The dependency-free
[`profile_toolsandbox_metadata.py`](../scripts/profile_toolsandbox_metadata.py) parses only the
four public scenario modules with Python's AST; it does not import ToolSandbox or retain user
prompts.  The receipt reports 129 unique base scenarios: 19 single-tool, 54 multiple-tool, 28
multiple-user-turn, and 28 insufficient-information scenarios.  Literal source metadata names 32
tools, 59 canonicalization tags, and 24 state-dependency tags.  The upstream `named_scenarios`
augmentation policy expands each base definition into four distraction levels plus four schema
scramble variants, or 1,032 source-level variants before runtime tool-similarity ranking.

This is a public-source inventory, not a model score or training set.  The Apple Software License
and `ACKNOWLEDGEMENTS` are preserved in the source manifest and require a separate redistribution
review.  Native evaluation still requires the pinned simulator, state databases, user simulator,
milestone verifiers, and complete upstream result coverage; no external API, tool, or account was
used here.  The complete hash-bound receipt is
[`m54`](paper/results/raw/m54-toolsandbox-metadata-v1.json).

The public [Apple ToolSandbox](https://github.com/apple/ToolSandbox) is the next stateful
productivity gate. Its scenarios model implicit tool dependencies, user simulation, messaging,
canonicalization, distraction tools, and insufficient-information abstention; the upstream CLI
writes one JSON result record per scenario with milestone similarity, minefield similarity,
turn count, exceptions, and a milestone-to-turn mapping. The dependency-free
[`src/localagent/eval/toolsandbox.py`](../src/localagent/eval/toolsandbox.py) bridge consumes that
official `result_summary.json` shape, hashes the source, mirrors the upstream category rule for
distraction augmentations, and fails closed unless the expected scenario list is present.
[`scripts/aggregate_toolsandbox.py`](../scripts/aggregate_toolsandbox.py) writes a receipt without
executing tools or importing the ToolSandbox package. This is a local result-summary aggregation
bridge; a live ToolSandbox run and any score remain pending.

The official [BrowserGym environment](https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/core/src/browsergym/core/env.py)
uses the Gymnasium episode contract: each browser action is passed to `step(action)` and returns
observation, reward, terminated, truncated, and info, with task-specific validation supplying the
reward. [`src/localagent/eval/browsergym.py`](../src/localagent/eval/browsergym.py) defines the
portable JSONL projection of that contract (`task_id`, `seed`, ordered action/reward steps, and
terminal flags), hashes the log, checks exact task/seed coverage, and reports per-task reward,
success, step count, and action errors. [`scripts/aggregate_browsergym.py`](../scripts/aggregate_browsergym.py)
consumes the projection without Playwright or a browser. It is a local BrowserGym/WebArena/
WorkArena/MiniWoB protocol bridge; no live browser score is claimed until the pinned runtime is
installed and the runner produces a complete receipt.

### AgentNet/OpenCUA desktop-action result bridge

The public [AgentNet dataset](https://huggingface.co/datasets/xlangai/AgentNet) contains
human-annotated desktop computer-use trajectories across Windows, macOS, and Ubuntu. The
[OpenCUA repository](https://github.com/xlang-ai/OpenCUA) publishes AgentNetBench as an offline
low-level action evaluator. Its coordinate, text, keyboard, scroll, and termination signals are
useful for a WebGPU policy, but the source trajectories are screenshot-grounded; a text-only
checkpoint must not convert them into training rows without an accessibility/vision bridge.

The existing [`src/localagent/eval/agentnet.py`](../src/localagent/eval/agentnet.py) mirrors the
per-trajectory action protocol without opening screenshots. The new
[`src/localagent/eval/agentnet_results.py`](../src/localagent/eval/agentnet_results.py) joins a
ground-truth JSONL export with a prediction JSONL stream, rejects duplicate or mismatched task
IDs, hashes both inputs, and reports mean action score, exact trajectory rate, action-count
penalties, action-type scores, and platform breakdowns. The
[`scripts/aggregate_agentnet.py`](../scripts/aggregate_agentnet.py) CLI requires an explicit
expected-ID list for a complete receipt. This remains an offline AgentNetBench-compatible proxy,
not a native AgentNetBench leaderboard score or an OS/VM run.

### tau2-bench interactive tool-result bridge

The public [tau2-bench](https://github.com/sierra-research/tau2-bench) runner evaluates a
half-duplex user simulator and agent against stateful airline, retail, telecom, and
banking-knowledge tool environments. Its official [task-schema/evaluation guide](https://github.com/sierra-research/tau2-bench/blob/main/docs/evaluation.md)
is important for interpretation: the default reward is an outcome product (database state plus
required communication), while `evaluation_criteria.actions` is usually one reference
trajectory, not a required script. The official [leaderboard guide](https://github.com/sierra-research/tau2-bench/blob/main/docs/leaderboard-submission.md)
requires the complete base split, consistent configuration, and at least four trials for a
publication-quality Pass^k submission.

[`src/localagent/eval/tau2.py`](../src/localagent/eval/tau2.py) consumes the upstream
`Results` JSON contract in both supported storage forms: one monolithic `simulations[]` file,
or `results.json` plus individual `simulations/*.json` files. It validates task/trial IDs,
reward, termination, duration, and metadata; hashes every source file; excludes
`infrastructure_error` runs as the upstream metrics do; and computes the upstream combinatorial
Pass^k estimator per task (not a misleading all-trials product). The
[`scripts/aggregate_tau2.py`](../scripts/aggregate_tau2.py) CLI accepts expected cases in the
stable `domain/task_id@trial` form and marks a receipt incomplete until exact coverage and the
requested trial count are present. This is a dependency-free result bridge: it does not invoke
tau2, user simulation, or any email/retail tool, and no tau2 score has been produced in this
environment.

The catalog also binds the newer realistic environments that should be part of the workshop-grade
evaluation matrix:

- [MobileGym](https://github.com/Purewhiter/mobilegym): 28 simulated mobile apps and 416 reusable
  task templates with deterministic judges.  The code is Apache-2.0, while the released task/data
  terms are CC-BY-NC-4.0, so it is evaluation-only here.
- [MobileWorld](https://github.com/Tongyi-MAI/MobileWorld): 201 long-horizon tasks across 20 mobile
  apps, including agent-user interaction and MCP-augmented workflows.
- [MemGUI-Bench](https://github.com/lgy0404/MemGUI-Bench): 128 GUI tasks (with a 40-task quick
  subset) that test interaction memory across browser, office, and file workflows.
- [WorkArena](https://github.com/ServiceNow/WorkArena): enterprise ServiceNow browser tasks,
  integrated through [BrowserGym](https://github.com/ServiceNow/BrowserGym); upstream task terms
  require review before redistribution.
- [WebLINX](https://github.com/mcgill-nlp/weblinx): roughly 100k grounded browser interactions
  across 2,300 demonstrations with dialogue context.  The official dataset terms remain binding.
- [OSWorld-V2](https://github.com/xlang-ai/OSWorld-V2): the dated 2026-06-24 release of the
  long-horizon desktop benchmark; gated task files/assets remain outside training and public
  checkpoints.
- [Toolathlon-GYM](https://github.com/eigent-ai/toolathlon_gym): 503 tasks over 25 local MCP servers,
  including reproducible email and Notion-style workflows without live external accounts.
- [GUIOdyssey](https://github.com/OpenGVLab/GUI-Odyssey): 8,834 cross-app mobile episodes over
  212 apps and six devices; the latest public snapshot provides explicit app/device/task splits.
- [MobileAgentBench](https://github.com/MobileAgentBench/mobile-agent-bench): 100 Android tasks
  over 10 open-source apps, designed as a small reproducible emulator gate before larger suites.
- [AgentBench Function Calling](https://github.com/THUDM/AgentBench): the current containerized
  release covers five interactive environments; the original benchmark spans eight environments.
- [VisualWebArena](https://github.com/web-arena-x/visualwebarena): 910 visually grounded WebArena
  tasks plus released human trajectories; a text-only run is explicitly a modality ablation.
- [OmniACT](https://huggingface.co/datasets/Writer/omniact): 9,799 desktop/web instruction-image
  pairs with PyAutoGUI scripts; it is a vision benchmark, not a text-first score.
- [WorldGUI](https://showlab.github.io/WorldGUI/): 315 dynamic desktop tasks over 10 applications
  with varied initial states and pre-action sequences.
- [macOSWorld](https://github.com/showlab/macosworld): 202 multilingual tasks over 30 macOS
  applications, including a safety subset.
- [ASSISTGUI](https://github.com/showlab/assistgui): 100 Windows productivity tasks over nine
  applications with project-file state.

The existing BFCL, WebLINX, and protected Mind2Web/browser captures in `data/private/` remain
evaluation-only under the same rule.  A public license does not make a benchmark safe to train on:
task prompts, verifier code, and gold state can directly inflate the score.

### Capability-to-benchmark map

| Capability | Representative public evaluations | What must be measured | Current WebGPU bridge |
| --- | --- | --- | --- |
| Mobile single-step and long-horizon control | AndroidWorld, MobileGym, MobileWorld, GUIOdyssey, MobileAgentBench, PhoneWorld | grounded action validity, milestone reward, episode success, app/device generalization | accessibility-tree text plus goal; no screenshot grounding yet |
| Browser navigation and visual grounding | BrowserGym/MiniWoB++, WebArena, WorkArena, WebLINX, VisualWebArena | element accuracy, operation F1, DOM/state transitions, task success | cleaned DOM/A11y text; visual suites are modality ablations |
| Desktop/computer use | AgentNet/AgentNetBench, OSWorld/OSWorld-V2, MemGUI-Bench, WorldGUI, macOSWorld, ASSISTGUI | VM task success, offline action/trajectory accuracy, long-horizon recovery, latency, safety | AgentNet offline parser is ready; coordinate-to-vision bridge and VM runners are pending |
| Stateful tools and productivity | Toolathlon-GYM, MCPMark, EnterpriseOpsGym, AgentBench FC, AppWorld, tau-bench | exact calls, schema validity, state delta, pass^k, abstention | local email/Notion mocks and retrieval sidecar; no external accounts |

This map is intentionally asymmetric: a text-first 10.5M model can produce a credible structured
tool/DOM bridge, but it cannot claim screenshot-level grounding or native OS control.  Those are
separate acceptance gates, not hidden quality assumptions.

## Pipeline for a WebGPU model

1. **Acquire and bind.** Download only a catalog row with `train_policy: train`; record the source
   URL, revision, license evidence, bytes, and SHA-256.  Run
   `python scripts/validate_realistic_agent_catalog.py` before ingestion.
2. **Normalize.** Convert AndroidControl/AITW records into the canonical `Conversation` schema.  Use
   an explicit decoder-produced intermediate row and
   [`normalize_realistic_agent_jsonl.py`](../scripts/normalize_realistic_agent_jsonl.py), which
   emits the existing `localagent_v1` format.  The normalizer rejects screenshot-only rows and
   unknown action arguments, so a converter that cannot prove the official split or text projection
   fails closed.
3. **Pre-train and mid-train.** Keep broad text/code pretraining separate from the UI/tool mixture;
   then mix verified mobile/browser/tool demonstrations with a fixed, documented ratio.  Preserve
   disjoint app/task/domain slots and the existing exact prompt denylists.
4. **SFT.** Train canonical tool calls and browser actions on the training candidates plus
   operator-reviewed synthetic examples.  Generate email/Notion-like workflows against local mock
   tools; do not copy Toolathlon, MCPMark, AppWorld, or EnterpriseOpsGym task text.
5. **RL simulation.** Use local stateful mocks with checkpoint rewards: tool selection, argument
   validity, state delta, user-facing completion, and safe abstention.  Re-run the same seed suite
   for pass@1/pass^k and keep the external benchmark tasks untouched.
6. **Evaluation.** Report capability by family, not one blended number: mobile action/episode
   success, browser element/task success, desktop task success, tool-call exact match, state-delta
   success, abstention, TTFA, tokens/s, and peak WebGPU memory.  Every score must name the exact
   upstream revision and whether it came from a local runner or an official leaderboard protocol.

## WebGPU observation contract

The first deployable bridge should use accessibility/DOM text and compact state deltas.  This keeps
the model under the WebGPU token and memory budget while still testing realistic control.  Screenshots
are retained in provenance for AndroidControl/AITW but are not fed to the text-only checkpoint.
Native screenshots and pixel gestures belong to a future multimodal branch, with a separate model
config and evaluation receipt.

## Reproducible AndroidControl bridge pilot

One bounded acquisition has now been exercised end to end without placing source payloads in Git:

- Official object: `gresearch/android_control/android_control-00000-of-00020`, generation
  `1717537205165914`, 2,492,987,829 bytes, upstream MD5 `QuhP+X5aiP+iD5amlNM6tA==`.
- Local range: bytes `0..33,554,431`, SHA-256
  `380b58842079ae52cf7b2e92e74940ec1c571c3eb3955785a2d8556ebb4e17cd`; the range is explicitly
  marked as a truncated-GZIP acquisition and contains complete records through the selected bound.
- Split binding: official `splits.json`, SHA-256
  `aa0943b0b4eb354796f3daf4a6e36836918c71254db2d4060b45ae98213ad2b6`, with only `train` episode
  IDs selected.  Eight records (episode IDs `0,20,40,60,80,100,160,220`) were normalized to 16
  deterministic Conversation rows after level-1 enrichment.
- Normalized JSONL: 1,511,467 bytes, SHA-256
  `dce5c7738fc3702f16ad5dabddbcafa8f67edad2fe1ad50d937df4dc7456e7f7`.  The decoder projects
  the official accessibility protobuf to bounded text and rejects screenshot-only records.

A second bounded train source now exercises the larger [AITW release](https://github.com/google-research/google-research/tree/master/android_in_the_wild):

- Official object: `gresearch/android-in-the-wild/general/general-00001-of-00321`, generation
  `1686095863743117`, 3,249,449 bytes, upstream MD5 `o86d8TEjui+XzNjKrxywxA==`.
- Official `standard.json` split binding is SHA-256
  `324ca94dbbb0778cee7fd1f2bd8dcba20cce592fbef6aec41ee475ed47ae7681`; all five complete
  episodes in this shard are in the official `train` list.
- The AITW adapter reconstructs step-level TFRecords into complete episodes, converts normalized
  dual-point gestures to pixel-space click/swipe actions, preserves type/home/back/enter actions,
  and projects UI annotation text/bounds without copying screenshots.  Five episodes became 10
  enriched Conversation rows (canonical JSONL SHA-256
  `31bdd0320ddcd75345aca73fd7ca81c05d11b6831091da94faca57b841df5ce6`).

The v2 projection used for the corrected continuation keeps each step instruction after the
observation: AndroidControl normalized JSONL SHA-256
`705bf99465b5feb2b24f03e5c81cc2c1560ac8f28200eb83ed8c921626c995fa` (16 rows) and AITW normalized
JSONL SHA-256 `1f0a08fac5684f21d62a10855fc05171b4d7ba699f8aae51f062d01f7b52031` (10 rows). The
source manifests remain bound to the same official train-only episode selections.

An earlier 8-step continuation over the 26-row AndroidControl+AITW mixture reported mean
normalized-row loss `8.6239` → `7.9730`, but it loaded the default byte tokenizer against the
16K-BPE parent checkpoint.  That receipt is retained only as a historical invalidation example;
its token accuracy and checkpoint must not be used for model-selection claims.

The corrected BPE-lineage continuation uses the tokenizer recorded in the parent checkpoint and
the same train-only rows.  It reduced mean loss `5.4868` → `4.7944` and assistant-token accuracy
rose `25.68%` → `31.69%`; exact trajectory accuracy remained `0/26`.  The child checkpoint is
kept outside Git (`cc12da27808d251df46d1beb649622e896cb32233dc2b6317deb917dec93c08a`) and is a
language-model bridge measurement only, not evidence of emulator success.  The tokenizer
compatibility gate is now covered by `tests/test_train_androidcontrol_pilot.py` and the pilot
refuses a missing or vocabulary-mismatched BPE artifact.

### Public AndroidControl 84K text-action continuation (m35)

The public [OfficerChul/Android-Control-84k](https://huggingface.co/datasets/OfficerChul/Android-Control-84k)
mirror exposes 82,944 train rows and a deliberately balanced 904-row test set derived from
Google's Apache-2.0 AndroidControl release.  [`ingest_androidcontrol_json.py`](../scripts/ingest_androidcontrol_json.py)
converts the LLaMA-Factory JSON shape into the canonical `Conversation` schema and records both
raw-file and normalized-output hashes.  The train/test files are never mixed.

This first scale-up is intentionally text-action only: the mirror supplies screenshot paths, but
the current checkpoint has no vision encoder, so the adapter marks every row
`visual_input_omitted=true` and `grounding_evaluable=false`.  A deterministic stratified 4,096-row
train slice (all 42 long-press examples and balanced coverage of the other train actions) was
continued for 32 BPE SFT steps from the 10.52M parent.  On all 904 held-out test rows, teacher-forced
assistant-token accuracy rose `59.28% → 65.47%` and mean loss fell `2.789 → 2.049`; teacher-forced
exact assistant sequences stayed `0/904`.

A second 300-step frozen-feature probe kept the adapted backbone fixed and trained only route and
dense dispatch heads over the 62-tool pool.  Held-out route accuracy was `100%`, while selector
top-1 was `46.24%` overall: navigate-back/open-app `74.4%`, scroll `66.4%`, wait `51.2%`, input
text `48.0%`, click `20.0%`, and long-press/home `0%`.  The transfer audit found matching config,
shapes, and tokenizer (`51` shared tensors); LM continuation moved embedding/attention/FFN norms
by `0.52%/0.22%/0.28%/0.01%` relative L2, while the frozen-head probe moved only dispatch heads
(`107.3%` relative L2).  This supports freezing the transferred backbone before dispatch-head
specialization, but it is not evidence that transfer improves emulator reward.

The complete source, split, training, selector, and weight-audit identities are in the
[`m35 receipt`](paper/results/raw/m35-androidcontrol-84k-text-action-transfer-v1.json).  Because
pixels were omitted, m35 must not be reported as AndroidControl visual grounding, AndroidWorld
success, or WebGPU hardware throughput; a workshop claim still requires a vision-capable adapter
and a native emulator run on the official evaluation split.

### AndroidControl dispatch balance ablation (m36)

The m35 split exposes a measurable train/test support gap: the 4,096-row train slice contains only
42 `mobile_long_press` rows and no `mobile_navigate_home` rows, while the public test contains 125
long-press and 29 home rows.  The dispatch runner now supports an explicit, hash-bound
`--focus-tool`/`--focus-repeat` ablation that adds synthetic and state-conditioned views without
touching the 904-row evaluation file.

Oversampling both missing/rare actions by 32 repeats did not solve the gap: held-out selector
top-1 fell from `46.24%` to `33.63%`, while both long-press and home remained `0%`.  The route head
stayed at `100%`, and the frozen backbone did not move.  This negative result is retained because
it rules out a tempting but overfit weighting fix; the next valid intervention is a licensed train
split containing those actions or a separately hash-bound real-device continuation.  The complete
comparison and head-movement audit are in the [`m36 receipt`](paper/results/raw/m36-androidcontrol-dispatch-balance-ablation-v1.json).

### Browser runtime receipt (local compatibility smoke)

The exported fp32 bundle was then exercised in the existing browser harness with an explicit
single-provider WebGPU session.  This is a provider/dispatch receipt, not a benchmark score for
AndroidWorld, WebArena, OSWorld, MCPMark, email, or Notion: the suite has eight one-step local DOM
fixtures, text-only observations, semantic targets, no screenshots, no physical cursor, and no
external navigation.

- Bundle: `sft_realistic_mobile_pilot`, 10,524,544 parameters; checkpoint SHA-256
  `268cd21d8f6a49e4e63c001ef73a26c67820d33407e5bb611a03382966700d1f`; bundle-manifest SHA-256
  `57cedfe061174fddf44cdae1d280bf1da9baef286f82655fdf17a78f5cbf139c`.
- Requested provider: `webgpu` with one ORT Web session; the observed adapter was Chrome
  SwiftShader (`google`), so this run demonstrates software WebGPU compatibility rather than a
  hardware-GPU throughput claim.  The model artifact was fetched, size-checked, and SHA-256
  verified in-browser before session creation.
- Result: 8/8 independent action schemas valid; exact action 6/8 (75%); final DOM/state
  transition 6/8 (75%); closed-loop success 6/8 (75%).  Click, double-click, type, scroll, drag,
  and cursor-move cases passed; the single key-press and local-navigation cases failed.
- Timing: harness TTFA p50 `157.6 ms`, closed-loop p50 `165.6 ms`, tool dispatch p50 `0.35 ms`.
  The slowest drag/scroll cases make the p90 closed-loop latency `4.28 s`; only 3/8 cases both
  completed and succeeded within a 250 ms deadline.  Treat p50 and deadline attainment separately.
- Versioned receipt JSON: [`m6-webgpu-realistic-mobile-mixture-browser-result.json`](paper/results/raw/m6-webgpu-realistic-mobile-mixture-browser-result.json), SHA-256
  `7b3872cd6e153b340b2fe9bc0a47d4b339635264c7ae7f57f0b2aeb21ffb6c26`.

The same eight-fixture control run under WASM (fp16 bundle) completed at roughly `33.3 ms` p50
with the same 75% exact/final-DOM/end-to-end rate.  Explicit WebGPU with that fp16 artifact did
not create a usable session because the selected device rejected `f16` gather kernels; this is a
runtime capability failure, not a model-quality score.  The fp32 export is therefore the current
compatibility fallback for WebGPU devices without shader-f16 support.  Neither receipt is evidence
that the model can yet control a real Android emulator, browser account, email system, Notion
workspace, or MCP server; those evaluations remain required before publication.

### Text-first mobile/productivity closed-loop pilot (custom 60-tool bundle)

The corrected mobile projection keeps the action instruction at the end of each observation, so a
left-truncated WebGPU context cannot discard the command after a long accessibility-tree dump. A
separate additive dispatch bundle was exported from the 10.5M-parameter mobile-dispatch child:

- Parent checkpoint: `sft-realistic-mobile-mixture-pilot.pt`, SHA-256
  `268cd21d8f6a49e4e63c001ef73a26c67820d33407e5bb611a03382966700d1f`.
- Child checkpoint: `sft-realistic-mobile-dispatch-pilot-sft200-v2.pt`, SHA-256
  `3606aa01952de1f006c039fef9f64f6184cc611ba320452674a71bbfc8ac137f`.
- Exported pool: 60 schemas (the stable standard 50 plus 10 `mobile_*` schemas), with fp32
  `model.onnx`/`action_model.onnx` parity max drift `9.86e-06`/`6.59e-06`.
- Bundle manifest SHA-256 `e6d56e433f7f6f9bcc712db241a21cf0cfbc1a3a36c388d2415182cfc721e427`;
  model and action graph SHA-256 values are pinned in the raw receipt.
- Local browser receipt: [`m7-webgpu-mobile-productivity-pilot.json`](paper/results/raw/m7-webgpu-mobile-productivity-pilot.json),
  SHA-256 `7887eaebd7512c360c4a99e9292e41634bce7f9f0e4899700a4ed7253015c507`.

The receipt runs nine deterministic steps against an in-memory Android/Gmail-style state plus
local Notion and email records. It uses explicit WebGPU on Chrome SwiftShader (software WebGPU),
text-only observations, no screenshots, no trusted OS input, and no external accounts. All nine
outputs were independently schema-valid. Mobile actions were 7/7 exact and closed-loop, but the
learned dense selector chose the wrong standard tool for both productivity tasks (0/2 exact), so
the aggregate is 7/9 (77.8%) and must not be presented as a real-device or real-account score.
The seven mobile successes use the explicitly reported `mobile_lexical_guard`; they are a runtime
contract check, not evidence that the dense selector has solved mobile dispatch. The held-out
selector result remains 40% top-1 on ten mobile rows, and the next acceptance gate is a no-guard
mobile ablation plus emulator/real-environment evaluation.

That 40% held-selector number is now historical only.  The pilot script had accidentally loaded the
default byte tokenizer while fine-tuning a 16K BPE checkpoint, so its selector metric was not a
valid learned-policy comparison.  The corrected BPE-tokenizer run below is the authoritative
dispatch measurement.

### Corrected BPE-tokenizer productivity dispatch and WebGPU receipt

The corrected continuation starts from the same 10.5M-parameter SFT parent but loads the tokenizer
recorded in the checkpoint (`data/tokenizer-webgpu-proxy-16k.json`) instead of the byte-level
default.  This removes the Python/browser feature mismatch that invalidated the earlier selector
diagnostic.

- Child checkpoint: `sft-realistic-mobile-dispatch-productivity-v6.pt`, SHA-256
  `be4f1216c88bfc6b12554e5d6324aa581e20409d6b213a3e47eaf5a3cc9f1583`.
- Training report: 3,000 steps; productivity train selector `12/12` (100%), productivity held
  selector `3/4` (75%), route accuracy `100%` on both splits.  The broader mobile held selector
  remains `4/10` (40%), so no-guard mobile control is not yet publication-ready.
- Export: 62 schemas, 10,524,544 parameters, fp32 ONNX parity max drift `8.58e-06` for logits and
  `6.14e-06` for hidden states.  Bundle-manifest SHA-256
  `5b711ff9768db66ef7a4d855bcb339244dc88ec1e6f6b5bf6eb5180b62f5644d`.
- Browser receipt: [`m9-webgpu-mobile-productivity-pilot-v6.json`](paper/results/raw/m9-webgpu-mobile-productivity-pilot-v6.json),
  SHA-256 `8fee312a92e897591c82f2327b6aa04693d412baa15ad14f03d234001bd1ad6c`.  On Chrome SwiftShader
  software WebGPU, the nine-step local state suite achieved schema validity `9/9`, exact tool
  selection `9/9`, exact arguments/action `8/9`, state transitions `8/9`, and closed-loop success
  `8/9`.  The seven mobile rows still use the explicit lexical guard; the dense selector passed
  the email row and missed the Notion title/content arguments.  Closed-loop latency ranged from
  `121 ms` to `7.44 s` with a `148 ms` median, so this is compatibility evidence rather than a
  100–300 tokens/s hardware-WebGPU claim.

The corrected receipt is therefore a reproducible local WebGPU gate, not evidence of AndroidWorld,
OSWorld-V2, WorkArena, WebLINX, MCPMark, Toolathlon-GYM, real email, or real Notion access.  Those
environment runners and official task splits remain required before a workshop or public model
claim.

### Retrieval sidecar ablation (v9r)

The v9r bundle adds an explicit 256-dimensional CRC32 character n-gram retrieval sidecar.  It is
not trained weights and is reported separately from the dense neural selector; its purpose is to
provide an auditable open-tool fallback for small WebGPU catalogs without reintroducing the
mobile lexical guard.  The sidecar uses clean synthetic action examples rather than inherited
accessibility dumps, and compacts a trailing `instruction:` field when present.

- Checkpoint: `sft-realistic-mobile-dispatch-productivity-v9r.pt`, SHA-256
  `523946f03fd8439e0c085ad9cbcab3bc43bcf3d28ef9385d9c42666a6f9eb0ec`.
- Bundle manifest SHA-256 `0785de8e1f84efb0ca68082bbb56e5b366ccf5212679f249091ed05cfc106af5`;
  dispatch sidecar SHA-256 `da09acd9df9e7073555543c99f83c579a52c3f4de4f20f67310877c0e26bdbba`.
- Receipt: [`m10-webgpu-mobile-productivity-v9r.json`](paper/results/raw/m10-webgpu-mobile-productivity-v9r.json),
  SHA-256 `9b69b0ff5e8b20f3756de4ece767883a1aa42a1820312e04400b5fca49973f3c`.
- With `mobile_guard=0&selector=retrieval`, Chrome SwiftShader WebGPU achieved 9/9 exact tools,
  9/9 exact arguments, 9/9 state transitions, and 9/9 closed-loop success across seven mobile
  actions plus email and Notion.  The same receipt's learned `dense_selector` no-guard ablation
  achieved 4/9 overall (2/7 mobile, 2/2 productivity).  This separation prevents a retrieval
  baseline from being misreported as neural tool-selection accuracy.

The retrieval sidecar is a deployment fallback, not a substitute for improving the learned
selector: the external mobile held-out neural selector is 6/10 (60%) after balanced synthetic
augmentation.  Workshop acceptance still requires a stronger no-guard neural result, an official
runtime benchmark, and hardware-WebGPU performance measurements.

### Stateful mobile/productivity trajectory gate (v11)

The next continuation adds 11 state-conditioned examples to the v10 parent-head probe while
keeping the 10.5M-parameter backbone frozen.  The local browser harness feeds the resulting state
JSON back into each next prompt and independently checks the bundle's `meta.json` schemas, exact
tool/argument equality, a preconditioned state transition, and complete-trajectory pass@1.  The
suite has three workflows and 13 ordered steps: Gmail compose/send (6), Notion capture (2), and a
local mail-page search/open flow (5).  It is deliberately text-first and uses no screenshots,
real accounts, MCP servers, or trusted OS input.

- Child checkpoint: `sft-realistic-mobile-dispatch-productivity-v11.pt`, SHA-256
  `4cccca1139699776c555876bb282783276f6e84b1a5e7e949dd83d5a2e8140ed`.
- The child keeps the v10 held-out selector results: mobile `6/10` (60%), productivity `3/4`
  (75%), and route accuracy `100%` on both; train selector is `86.21%` after 3,000 steps.
- The exported 62-tool bundle passes fp32/fp16 parity (fp32 logits drift `8.58e-06`; fp16
  logits argmax agreement `1.0`) and was requested through the in-app browser's WebGPU provider.
- The stateful run is a negative result: schema validity `13/13`, exact actions `0/13`, state
  transitions `0/13`, closed-loop success `0/13`, and complete-trajectory pass@1 `0/3`.  The first
  Gmail action emitted `mobile_input_text({text: "app"})`; the Notion and browser workflows both
  started with `mobile_open_app({app_name: "app"})`, so the validator failed closed before applying
  any state transition.

The complete identity, parity, browser condition, and first-failure records are in the
[`m15 receipt`](paper/results/raw/m15-webgpu-mobile-productivity-trajectories-v11.json).  This
diagnostic's SHA-256 is `a12f7c4328cc21f429056ab94ca2ee575af7b67f7db39032db83646860ce315b`.
It shows that adding state-conditioned rows did not yet produce robust sequential control;
it is not an AndroidWorld, AITW, BrowserGym, OSWorld, AppWorld, MCPMark, EnterpriseOps-Gym,
real-email/Notion, screenshot-grounding, or hardware-throughput score.

### Corrected stateful trajectory gate and weight audit (v14)

The v14 continuation adds 23 balanced state-conditioned rows, including browser open/click/type/
Enter variants, and preserves the v11 parent-head initialization.  A code audit found three
deployment/harness issues that are now fixed: lexical mobile selection was reading the goal instead
of only the next action, state JSON polluted quoted argument candidates, and the trajectory scorer
compared runtime metadata to the expected `{tool,args}` object.  The independent validator now
projects runtime responses to `{tool,args}` and keeps the state-transition check separate.

The weight audit is consistent with probe transfer: all 40 shared backbone tensors are unchanged
(relative L2 `0.0`), while the route and dense-selector probes move by relative L2 `0.2548` and
`0.2580`.  The v14 train selector is `89.66%`; held-out mobile and productivity selector scores
remain `60%` and `75%`, so the added rows have not improved the external holdouts.

The corrected in-app WebGPU run reports:

| Condition | Schema-valid | Exact `{tool,args}` | Closed-loop | Complete pass@1 |
| --- | ---: | ---: | ---: | ---: |
| Guarded dense | 12/13 | 4/13 | 3/13 | 0/3 |
| No-guard dense | 11/13 | 1/13 | 1/13 | 0/3 |
| No-guard retrieval sidecar | 8/13 | 2/13 | 1/13 | 0/3 |

The guarded condition improves because the lexical selector is now scoped to the requested action,
and the corrected grounder recovers app names such as `Gmail` and `Notion` instead of state keys
such as `app`.  The learned dense selector still confuses browser open/click and later mobile
actions; the retrieval sidecar is an ablation, not neural accuracy.  The hash-bound v14 checkpoint,
bundle, provider condition, first failures, and weight audit are in the [`m16 receipt`](paper/results/raw/m16-webgpu-mobile-productivity-trajectories-v14.json),
SHA-256 `9ad33d0d5d1613cb7c4f024792412dd61d4a6b165c3039163cd1e83902976dc8`.

This is stronger local evidence than the earlier M15 receipt, but it remains a negative workshop
gate: no official Android emulator, AgentNetBench, BrowserGym/OSWorld VM, MCP server, screenshot
grounding, real account, or hardware-throughput run has been completed.

### Stateful email/Notion/browser transfer probe (m55)

The reusable [`stateful_productivity.py`](../src/localagent/data/stateful_productivity.py) contract
now gives SFT, pointer supervision, selector probes, and future RL one canonical local state
machine.  It contains disjoint train/evaluation slots and phrasing for five workflows: complete
Gmail compose/send, Notion page creation, browser search, browser 404 recovery, and no-tool
abstention.  The evaluation split has 16 ordered decisions over a 62-tool WebGPU catalog.  Every
step is scored independently for schema validity, tool/argument exactness, state transition, and
closed-loop completion; final task state and recovery are scored separately.

The [`m55 receipt`](paper/results/raw/m55-stateful-productivity-transfer-v1.json) compares the
frozen 10.5M-parameter BPE backbone with a seed-matched random backbone.  Both arms receive the
same 160 route/selector updates and pointer-copy budget.  The pretrained arm reaches selector
top-1 `53.33%` and top-3 `80.00%`, versus `46.67%` and `86.67%` for the random arm.  Closed-loop
success is only `1/16` (`6.25%`) for both arms, with `0/5` complete workflows and `0/1` recovery;
the only successful episode is abstention (`1/1`).  This is a useful negative result: better
teacher-forced selection does not yet imply stateful execution, so the next training target is
argument/state grounding and recovery rather than more route-head fitting.

The frozen shaped-reward projection (schema `0.10`, tool `0.25`, arguments `0.25`, transition
`0.25`, terminal `0.15`) averages `0.2500` for the pretrained arm and `0.2656` for the random
control, another warning that partial local signals can move independently of task completion.

The probe is synthetic and local by design: it uses no public benchmark task text, screenshots,
emulators, MCP servers, external APIs, real email, or real Notion account.  It is a training and
diagnostic contract, not an AndroidWorld, BrowserGym, MCPMark, or publication score.

### Stateful productivity low-rate transfer ablation (m56)

The [`m56 receipt`](paper/results/raw/m56-stateful-productivity-transfer-ablation-v1.json) adds the
missing weight-adoption control.  It keeps the same five disjoint workflows, 16 evaluation
decisions, 62-tool catalog, seed, pointer vocabulary, and closed-loop evaluator while comparing:

| Arm | Selector top-1 | Selector top-3 | Closed-loop | Complete workflows | Backbone relative ΔL2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pretrained, frozen | 53.33% | 73.33% | 1/16 | 1/5 | 0 |
| Pretrained, low-rate unfrozen (`1e-5`) | 73.33% | 80.00% | 1/16 | 1/5 | 0.188% |
| Matched random backbone | 33.33% | 80.00% | 1/16 | 1/5 | not applicable |

The low-rate arm improves held-out selector top-1 by `20.00` percentage points over the frozen
arm and `40.00` points over random, while changing the mixer/FFN/embedding groups by only
`0.41%`/`0.41%`/`0.046%` respectively.  It does **not** improve closed-loop execution, recovery
(`0/1`), or complete workflows; the only completed task remains abstention.  The appropriate
weight policy is therefore: retain the verified BPE backbone as a low-rate initialization for
future native adaptation, but do not promote it as evidence of stateful agent capability.  The
next decisive experiment is a larger public-train continuation with native BrowserGym/MobileGym
or MCPMark execution, not more selector-only tuning on this five-task mock.

This remains a synthetic local transfer ablation.  It reads no public benchmark task text,
screenshots, emulator, MCP server, real email, or Notion account and must not be reported as an
official benchmark score.

### Current-checkpoint stateful transfer rerun (m161)

To avoid treating the older m56 parent as the deployment baseline, the same five-workflow,
16-decision state machine was rerun from the current `runs/sft-webgpu-proxy-pilot-hybrid-seed2027`
checkpoint.  The tracked [`m161 receipt`](paper/results/raw/m161-stateful-productivity-current-transfer-v1.json)
binds the current parent and child hashes, train/eval task hashes, all three transfer arms, and
the report identity.  Frozen and low-rate-unfrozen pretrained arms each reach `5/16` closed-loop
steps (`31.25%`), while the matched random arm reaches `1/16` (`6.25%`).  No email, Notion, or
browser workflow completes end-to-end; the result is a transfer diagnostic only.  The low-rate
arm moves the backbone by relative L2 `0.159%` overall (`0.0447%` embedding, `0.350%` FFN,
`0.338%` mixer), supporting conservative backbone reuse but not capability promotion.

### Exported current-child WebGPU replay (m162)

The m161 low-rate child was exported into a clean temporary WebGPU bundle and checked by the
exporter’s hash-bound fp32/fp16 graph parity gate.  The fp32 graph stayed within `7.63e-6` logits
and `6.08e-6` hidden max error; the fp16 graph stayed within `5.19e-3` logits and `4.14e-3`
hidden error, with argmax agreement `1.0`.  The deployment verifier found every required artifact
and no static-bundle blocker.  In the in-app browser with an explicit `backend=webgpu`, the same
three local synthetic trajectories produced `5/13` exact actions (`38.46%`), `12/13` schema-valid
actions (`92.31%`), and `0/3` complete workflows; the email path failed at body entry, Notion at
page creation, and browser at the first URL action.  This is bound in the [`m162 receipt`](paper/results/raw/m162-stateful-export-browser-v1.json).

The reused 20-case offline structured-action parity diagnostic also exposed a publication blocker:
fp16 pointer-score drift peaked at `1.16097`, above the hard `1.0` threshold, even though route,
tool, grounded-argument, and final normalized-action decisions matched the fp32 reference.  We
retain this as a diagnostic failure rather than relaxing the threshold.  The result is not a native
AndroidWorld/BrowserGym/OSWorld/MCPMark score and does not establish real email, Notion, or browser
control.

### Public mobile/GUI source audit (m163)

The [`m163 receipt`](paper/results/raw/m163-mobile-grounding-source-audit-v1.json) hash-pins the
current public source revisions for four deployment-relevant references without copying their
benchmark payloads.  KnowU-Bench reports `192` registered tasks over `23` apps, split into `42`
general, `86` personalized, and `64` proactive tasks, with hidden profiles, exposed behavioral
logs, and an online user simulator.  AppAgent's public test manifest contains `45` smartphone
tasks over `9` Android apps.  GroundCUA reports `56K` screenshots and at least `3.56M` human-
verified element annotations across `87` desktop applications.  UI-TARS documents separate
computer-use, mobile-use, and grounding modes, including `long_press`, `open_app`, `press_home`,
and `press_back`.

The receipt retains only source README/license/manifest hashes and these aggregate contract fields.
It records `0` training rows, `0` native runs, `0` WebGPU runs, and `0` official scores.  The
appropriate next step is native, release-matched evaluation with Android/desktop runtimes and
visual encoders where required; converting these source summaries into SFT would violate the
evaluation-only boundary.

### Mobile safety and personal-intelligence source audit (m178)

The [`m178 receipt`](paper/results/raw/m178-mobile-safety-personalization-source-audit-v1.json)
hash-pins two public benchmarks that cover the safety and personal-state surfaces missing from a
text-only browser smoke test.  MobileSafetyBench defines `100` Android-emulator tasks: `50`
helpfulness tasks, `42` safety-scored tasks, and `8` indirect-prompt-injection tasks across
messaging, web navigation, social, finance, device/data management, and utility apps.  Its
rule-based evaluator inspects action history and device state, and its Appium/ADB runtime must be
treated as part of the benchmark rather than reduced to a prompt-only classification.

The same receipt binds iOSWorld's `26` seeded SwiftUI apps and `133` tasks: `27` single-app,
`60` multi-app, and `46` memory/personalization tasks.  The persistent cross-app identity and
optional accessibility-XML/MCP modes make this a separate state-and-consent gate, not a generic
mobile action score.  Both source audits retain only README/license hashes and aggregate contract
fields (and records the missing MobileSafetyBench license file); they add `0` training rows, `0` native runs, and `0` official scores.  A publishable result
still requires the release-matched Android emulator or iOS Simulator, seeded state, action logs,
and official verifier.  In particular, the current WebGPU model must not claim safe messaging,
finance, or personal-data handling from this metadata-only audit.

### Public mobile evaluation-manifest acquisition audit (m179)

The [`m179 receipt`](paper/results/raw/m179-mobile-evaluation-manifest-audit-v1.json) moves the
two references one step beyond README metadata while preserving the evaluation boundary.  The
pinned iOSWorld `tasks.json` is `192,020` bytes with `133` rows, reconciled as `27` single-app,
`60` multi-app, and `46` memory/personalization tasks over `26` apps.  MobileSafetyBench's pinned
source task table is `69,821` bytes with `90` rows across six operational areas, and its separate
QA analysis file is `1,778` bytes with `3` rows.  The paper's `100`-task suite count is retained as
the authoritative contract; the two public files are not silently summed because the upstream
runner's assembly rules are not available here.

Only byte counts, SHA-256 values, and aggregate category counts are retained.  The manifests were
fetched to a temporary audit directory, their task text was not copied into Git or SFT/RL, and no
simulator, seeded profile, APK, or official verifier was executed.  This is stronger split and
provenance evidence, not a mobile benchmark score.

### Current-child mixed mobile/computer continuation and matched random control (m180)

The [`m180 receipt`](paper/results/raw/m180-current-child-mixed-public-continuation-v1.json)
records a bounded continuation from the current AndroidControl-adapted child over source-pinned
public AgentNet and Mind2Web projections: `549` train rows and `145` held-out rows, with
source-disjoint AgentNet parent records and Mind2Web tasks/typed pointer slots.  After `32` CPU
updates at learning rate `1e-5`, held-out assistant-token accuracy improves from `51.99%` to
`58.06%` and mean loss falls from `3.438` to `2.662`; route accuracy rises from `9.55%` to
`35.18%`.  A matched random-backbone control reaches only `12.93%` token accuracy and `1.01%`
route accuracy after the same updates, so the warm-start gap is real for this teacher-forced
projection rather than an artifact of the optimizer alone.

The result remains diagnostic.  Both arms have `0/145` sequence exactness, action heads are
unchanged, and the fixed browser tool catalog does not cover the AgentNet/Mind2Web action names,
making the reported selector top-1 values uninterpretable.  Weight audits find compatible
configuration/tokenizer and `51` shared tensors, with warm-start relative movement of `0.449%`
embedding, `0.223%` attention/mixer, and `0.273%` FFN (random control: `0.554%`, `0.210%`, and
`0.278%`).  This supports retaining the transferred representation for further surface-specific
experiments, not publishing a universal transfer claim or asserting native browser, desktop,
mobile, email, Notion, MCP, or WebGPU control.

### Current AgentNet surface-selector ablation (m181)

The [`m181 receipt`](paper/results/raw/m181-agentnet-current-surface-selector-ablation-v1.json)
retests the current m180 child with the correct 14-action `agentnet_*` catalog instead of the
browser catalog used by the mixed-continuation diagnostic.  The backbone is frozen and the
surface-specific selector is trained for 400 updates on 513 public-train projection rows, with
133 source-held-out rows.  Across three seeds, warm-start held-out selector top-1 averages
`69.67%`, while the matched random selector averages `70.68%`; warm is better on zero seeds,
equal on one, and worse on two.  Selector movement is therefore head adaptation, not evidence
that the inherited language representation improves desktop action choice.

The strongest replay uses the seed-2044 warm selector and covers all eight held-out parent
trajectories: first-action type is `100%`, but exact trajectory and task success are both `0%`.
Screenshots, coordinates as visual input, the official split, and an Ubuntu desktop runtime were
not used.  The correct conclusion is to keep the surface-specific catalog plumbing while
rejecting selector-transfer adoption until screenshot grounding and native AgentNet/OSWorld
verification are available.

### Current-child MobileGym native canary (m182)

The [`m182 receipt`](paper/results/raw/m182-current-child-mobilegym-native-canary-v1.json) runs
the current m180 child inside the pinned MobileGym simulator at revision
`093a3292d13fc4186e279af4ef1b005ac8e4d2b7`.  The runner verifies the official `160`/`256`
train/test split, launches Chromium, executes the state-diff judge, and completes one official
test task with zero runtime or judge errors.  Under the same two-step text projection used by the
existing m146 result, the current child scores `0/1`; the task reaches no judged progress and the
model emits no translated mobile action.

This is genuine native environment evidence for the current checkpoint, but deliberately only a
canary: it is not the complete 256-task MobileGym result, visual grounding, or Android emulator
evidence.  The next required comparison is a complete split run with the current child and the
same release-matched runtime.

### Current m180 Hugging Face-format export (m183)

The [`m183 receipt`](paper/results/raw/m183-hf-local-export-current-m180-v1.json) binds a fresh
local export of the current m180 child: `10,524,544` parameters, a `42,101,904`-byte
`model.safetensors`, a `1,134,224`-byte BPE tokenizer, and serialized 63-tool/23-pointer dispatch
heads.  The export is locally hash-verified and self-contained, but it is not a Hub repository or
download URL: `hf auth whoami` is unauthenticated and no upload is claimed.  This bundle keeps the
generic deployment catalog; the AgentNet surface selector remains a separate diagnostic artifact.

### Current m180 WebGPU export and clean bundle verification (m184)

The [`m184 receipt`](paper/results/raw/m184-current-m180-webgpu-export-v1.json) exports the same
checkpoint to full-logits and hidden-only ONNX graphs, fp16 counterparts, tokenizer, generic
dispatch heads, and metadata.  All four graphs pass the hard CPU parity gate; fp32 logits drift is
`8.11e-6`, fp16 logits drift is `3.96e-3`, and fp16 argmax agreement is `100%`.  The clean static
app copy verifies every artifact hash and the manifest with no blockers.

This establishes a reproducible WebGPU-shaped artifact for the current child, not physical GPU
placement, useful browser-agent quality, native benchmark success, or a public Hub URL.  The model
and Space remain unpublished until authenticated upload and independent hosted verification.

### ToolSandbox public-source projection and transfer probe (m59)

ToolSandbox is the most direct public stress test for the requested stateful productivity surface:
its scenarios cover state dependency, canonicalization, multiple tool calls, multi-turn
clarification, and insufficient information.  The [`m59 receipt`](paper/results/raw/m59-toolsandbox-public-projection-transfer-v1.json)
comes from a pinned upstream checkout and a static-AST adapter, producing 107 train and 20
held-out canonical `Conversation` rows over 32 candidate tools.  The adapter retains the public
scenario request and candidate list but reads no verifier, imports no simulator, executes no tool,
and calls no external API.

Starting from the transferred m56 child, 32 low-rate SFT updates improve held-out teacher-forced
token accuracy from `65.31%` to `71.53%` and reduce mean loss from `2.766` to `2.139`; sequence
exactness remains `0/20`.  A candidate-list dense-selector probe raises top-1 from `45%` for the
inherited selector to `75%` after retraining, but a matched random backbone also reaches `75%`.
The result supports the data adapter and selector retraining protocol, not representation transfer
or stateful execution; the adoption decision is explicitly `do_not_adopt_as_representation_evidence`.

### ToolSandbox schema-conditioned transfer probe (m61)

The next pass keeps the same task-disjoint split but enriches each candidate tool with a static
JSON schema: function signatures, primitive annotations, required arguments, and the first
docstring summary are parsed from the pinned `tool_sandbox/tools/*.py` files with Python's AST.
The adapter never imports ToolSandbox, resolves decorators, reads verifiers, executes tools, or
contacts an API.  This is the right interface for the requested email/Notion-style deployment:
the model sees the task plus the actual candidate schema instead of a name-only placeholder.

The [`m61 receipt`](paper/results/raw/m61-toolsandbox-schema-conditioned-transfer-v1.json) binds
38 statically extracted functions, 107 train rows, and 20 held-out rows.  The transferred
backbone's teacher-forced metrics are unchanged from m59 (`65.31% → 71.53%` token accuracy;
sequence exactness `0/20`) because this SFT path predicts the assistant call rather than training
the schema text itself.  The schema-aware candidate selector improves top-3 coverage from `85%`
to `100%`, while top-1 is `75%`; the matched-random backbone also reaches `75%` top-1.  Therefore
schema conditioning is adopted as a deployment interface improvement, but pretrained
representation reuse is still not promoted as a quality claim.  Native ToolSandbox/MCPMark
execution and state verifiers remain required for any real side-effect or productivity score.  On
the untouched pinned MCPMark descriptions, the same child remains at `20.71%` standard routing,
`2.37%` selector top-1, and `14.79%` top-3 (`22.86%`, `1.43%`, and `14.29%` on easy), exactly
matching m60's m59 arm.  This negative transfer is important: local schemas improve candidate
coverage, but do not solve cross-service Notion/filesystem/Postgres schema retrieval.

### ToolSandbox checkpoint-in-loop text decoder evaluation (m63)

The selector-only probe was insufficient to answer whether the deployed WebGPU dispatch path can
produce a usable call.  The [`m63 receipt`](paper/results/raw/m63-toolsandbox-text-decoder-eval-v1.json)
now runs the checkpoint-backed constrained decoder on all 20 held-out rows.  Each row supplies its
retained public candidate list and static JSON schemas; the evaluator runs route, character
retrieval, pointer grounding, parsing, and recursive schema validation, then compares the call to
the AST-extracted target.

With the schema-child checkpoint, restricting retrieval to the row's candidate list gives `100%`
route coverage, `60%` tool exactness, `30%` exact tool-and-argument matches, and `100%` schema
validity.  Canonicalization is `64.29%` tool exact / `35.71%` argument exact; state-dependency rows
fall to `25%` / `25%`.  Exposing the checkpoint's inherited global selector instead of the row
candidate contract collapses to `10%` tool exactness and `10%` schema validity.  The matched
no-schema child ties the row-retrieval arm exactly, so schema extraction is not yet evidence that
the pretrained representation was adopted; it is an interface and safety contract only.

This is a stronger decoder-in-the-loop diagnostic than m59/m61 teacher forcing, but it remains a
static text projection: no ToolSandbox simulator, user simulator, milestone verifier, MCP service,
official split, native browser, or external side effect ran.  The practical deployment decision is
to require task-scoped candidate retrieval and schema validation before dispatch, while treating
stateful execution and exact argument grounding as unresolved training targets.

### ToolSandbox native simulator/verifier smoke (m64)

The [`m64 receipt`](paper/results/raw/m64-toolsandbox-native-smoke-v1.json) runs the same transferred
checkpoint inside the pinned ToolSandbox execution context and upstream milestone evaluator for
`cellular_off`, `wifi_off`, and `send_message_with_phone_number_and_content`.  The generic decoder
now emits typed booleans and separates phone/message spans; all three tool mutations reached
`milestone_similarity = 1.0` in the resettable smoke.

This is native simulator/verifier evidence, not an official benchmark score: the smoke uses a
scripted user, a deterministic post-tool confirmation template, three single-turn scenarios, and
no model-based user simulator or RapidAPI.  The official split remains unverified, so m64 cannot
clear the native ToolSandbox publication gate.  The result does, however, turn the m63 argument
grounding failure into a reproducible training/runtime target instead of a projection-only claim.

### ToolSandbox transfer movement audit (m65)

The [`m65 receipt`](paper/results/raw/m65-toolsandbox-weight-transfer-v1.json) compares the exact
stateful-productivity low-rate parent and the ToolSandbox continuation child.  Config and tokenizer
hashes match, with no shape mismatch.  Relative movement is `0.476%` in the embedding, `0.196%` in
attention/mixer, `0.243%` in FFN, and `0.0097%` in normalization; the inherited route, pointer,
dense-selector, and fixed tool heads were unchanged in this child.

That pattern supports reusing the pretrained backbone with a smaller backbone learning rate and
separate head initialization, but it does not establish that transfer is better than a matched
random control.  The adoption decision therefore remains compatibility-positive and quality-claim
negative until the same native task set is run with frozen, low-rate, and matched-random arms.

### Resettable local productivity runtime (m66)

The [`m66 receipt`](paper/results/raw/m66-stateful-runtime-evaluation-v1.json) closes a gap in the
earlier fixed-step stateful probe.  [`StatefulRuntime`](../src/localagent/data/stateful_productivity.py)
now keeps a current action index, rejects malformed or out-of-order calls without advancing the
episode, exposes the rejection as the next observation, and records an append-only event log.  The
receipt runs five held-out workflows—email send, Notion page creation, browser search, browser
recovery, and abstention—through that retry-capable runtime.

The oracle reaches `5/5` complete workflows and `16/16` accepted steps, validating the verifier and
reset contract.  The 10.52M low-rate child accepts only `4/16` steps over 46 bounded attempts and
completes `1/5` workflows (abstention only); it can open Gmail and navigate/click in the browser,
but cannot finish the email, Notion, search, or recovery trajectories.  This is a useful negative
deployment result: transfer and schema validity alone are not enough, and the dominant remaining
failure is route/argument grounding under a long state-conditioned prompt.  The runtime is
deterministic and in-memory, so m66 is not a public benchmark, native browser/emulator run, real
email/Notion operation, or WebGPU hardware result.

### Stateful productivity GRPO simulation (m67)

The [`m67 receipt`](paper/results/raw/m67-stateful-productivity-grpo-v1.json) records the first
actual RL update against the canonical local email, Notion, browser-search, recovery, and
abstention surface.  The runner first performs a bounded 32-update state-conditioned SFT warm
start, then executes four rollout steps with two samples per prompt.  The opt-in local reward is
strictly shaped: a complete tool envelope contributes `0.10`, schema validity `0.10`, exact tool
name `0.20`, exact arguments `0.20`, and the state-transition terms `0.25 + 0.15`.  The normal
canonical-toolcall environment is unchanged; this shaping exists only to avoid a zero-gradient
diagnostic when a small warm-start model has not yet learned the tool vocabulary.

The run produced five informative groups out of 16 and four realized optimizer updates.  Its
rollout reward distribution was 25 zero-reward and 7 strict-envelope-only rollouts (`0.1`); no
exact rollout succeeded, and greedy held-out exact tool/text accuracy remained `0/16`.  This is
therefore evidence that the RL plumbing and checkpoint lineage are live, not evidence of learned
productivity capability.  The rows are deterministic local fixtures with in-memory side effects;
no public benchmark text, emulator, browser, MCP service, real account, or native WebGPU runtime
was used.  The next valid experiment is to acquire and hash an official split, then compare this
shaped arm against canonical-reward and matched-random controls under the same task IDs.

### Local WebGPU and Hugging Face export receipts (m57–m58)

The same m56 child was exported to a clean static-demo bundle and an independent Hugging Face
format bundle.  The [`m57 receipt`](paper/results/raw/m57-stateful-webgpu-deploy-verification-v1.json)
binds the 10,524,544-parameter checkpoint, tokenizer, eight generated inference artifacts, and
their exporter manifest.  The fp32 and fp16 ONNX graphs pass the hard CPU parity gate (maximum
fp32 logit difference `6.68e-6`; fp16 logit difference `7.36e-3`; fp16 argmax agreement `1.0`),
and the clean static app contains all required files.  The [`m58 receipt`](paper/results/raw/m58-stateful-hf-local-export-v1.json)
binds the local `model.safetensors`, serialized heads, tokenizer, config, and README hashes.

These are local artifact and deployment checks, not native WebGPU capability measurements.  The
Hub is not authenticated in this environment, so neither receipt claims a public model/Space URL
or upload; a user-provided write token and repository namespace are still required.  No external
email, Notion, browser, MCP, or desktop side effect was executed.

The [`m131 receipt`](paper/results/raw/m131-hf-local-export-m129-mobile-v1.json) repeats the
export with the m129 mobile state-routing checkpoint and fixes an important deployment contract:
the HF bundle now carries the complete 63-tool dispatch catalog, 17 pointer argument names, and
selector/retrieval examples alongside the numeric heads.  The model card reports the actual
10.52M parameter and 63-tool configuration instead of a stale fixed tool count.  The bundle is
hash-verified locally, but remains unpublished until Hugging Face authentication and a repository
namespace are supplied.

The [`m132 receipt`](paper/results/raw/m132-webgpu-m129-deploy-v1.json) stages the same checkpoint
through the browser exporter and copies the verified bundle beside `spaces/localagent-webgpu`.
All eight required artifacts are present and hash-match the exporter manifest; fp32 logits parity
is `8.58e-6`, fp32 hidden parity `5.30e-6`, and fp16 logits drift `5.85e-3`, with all hard gates
passing.  This proves a locally deployable static bundle, not a hosted Space, hardware-WebGPU
latency result, or successful email/Notion/mobile task execution.

### MobileGym complete official test evaluation (m133)

The [`m133 receipt`](paper/results/raw/m133-mobilegym-native-text-eval-v1.json) is the first
complete official MobileGym test execution for the WebGPU checkpoint.  The runner uses the
hash-pinned `test.txt` whitelist (256 IDs), the pinned simulator revision, the upstream task
`setup()` lifecycle, the native action handlers, and the upstream state-diff judge.  It records
only public task IDs, action names, hashes, judge field names, and aggregate outcomes; task
parameters, DOM text, screenshots, and raw arguments are not retained.

With the m129 warm checkpoint, selector top-1, and a fixed two-step cap, all `256/256` official
test tasks executed without environment or judge errors.  Success was `13/256` (`5.08%`).  The
best suite slices were weather `4/9`, clock `3/9`, SMS `1/3`, and Tencent Meeting `1/9`; Notes
remained `0/5`.  The model emitted `mobile_submit_answer` on 200 tasks and opened an app once,
which makes the failure mode unambiguous: the simulator and judge are functioning, but the
current text-first dispatch policy collapses to answer emission instead of state-conditioned UI
actions.  This is a native MobileGym simulator result over the official split, not a visual
mobile-agent score, Android emulator result, or screenshot-grounding claim.  The fixed two-step
cap is disclosed and should be increased in a later long-horizon run before making a leaderboard
comparison.

The follow-up [`m134 workshop-gate report`](paper/results/raw/m134-workshop-gate-mobilegym-v1.json)
accepts m133 as a native MobileGym receipt.  With MobileGym added, the strict publication gate
passes the catalog, BrowserGym/MiniWoB, WebGPU capability, transfer ablation, and public-artifact
checks, while seven independent native requirements remain absent: AndroidWorld, OSWorld,
OSWorld-V2, AgentNet, ToolSandbox, MCPMark, and EnterpriseOps-Gym.  The report therefore remains
`ready: false`; the MobileGym result must not be used to imply completion of those other surfaces.

### Focused mobile-action warm-start canary (m135)

The [`m135 canary receipt`](paper/results/raw/m135-mobile-action-focus-canary-v1.json) records a
train-only corrective arm after m133 exposed answer-tool collapse.  It starts from the m129
state-routing checkpoint, repeats the seven low-level mobile actions 32 times, and adds 31
generic state-conditioned trajectory rows.  The public AndroidControl/AITW eval file remains
held out; no MobileGym task text, DOM state, answer, or judge output enters training.  The arm
raises the held-out selector top-1 from the parent report's `47.12%` to `50.00%`, while pointer
exactness is `25%` and route accuracy is `100%`.

That offline movement does not transfer to the native canary: on the same first 20 official
MobileGym test IDs, m135 also passes `1/20` (`5%`) and emits `mobile_submit_answer` 14 times.
The child is therefore not promoted as a production checkpoint or a full benchmark score.  The
receipt is useful negative evidence: action-head oversampling can improve a held-out selector
without fixing state-conditioned UI grounding, and the backbone remains a frozen warm-start
transfer rather than a newly pretrained representation.

### Four-source public continuation with xLAM-derived tool data (m137)

The [`m137 transfer receipt`](paper/results/raw/m137-cross-surface-xlam-derivative-transfer-v1.json)
is the first explicit four-source continuation for this deployment family.  It combines 8,540
public train rows from AndroidControl, AgentNet, Mind2Web, and the public Apache-2.0
[`xLAM-derived dataset`](https://huggingface.co/datasets/product-science/xlam-function-calling-60k-raw),
then evaluates 1,541 source-disjoint, globally slot-disjoint rows.  The original Salesforce xLAM
dataset remains gated; the derivative is therefore bound to its own revision and is never called
an official Salesforce split.

The derivative normalizer rejects rather than coerces malformed generated schemas: 89/4,000
train rows and 18/1,000 held-out rows were excluded, and a further 482 held-out rows with
overlapping declared argument slots were removed to preserve the repository's exact-match split
contract.  These exclusions are recorded in the receipt and are not hidden data cleaning.

Starting from m129, warm continuation raises aggregate held-out token accuracy from `57.65%` to
`58.81%` (Android `65.47% → 67.20%`, desktop `47.27% → 48.95%`, derivative tools `52.84% →
53.45%`).  The matched random-backbone control reaches only `11.22%`.  Warm backbone movement is
small (embedding `0.48%`, mixer `0.15%`, FFN `0.18%`, normalization `0.007%`), while the random
control moves those groups by `123.6%`, `77.9%`, `87.8%`, and `7.9%`.  This supports reusing the
pretrained backbone with conservative updates and separate head learning rates.

It does not establish a production tool-use win: on a 128-row derivative xLAM first-call probe,
row-local tool exactness changes `50.78% → 49.22%`, argument exactness remains `0%`, and global
63-tool selection remains `0%`.  The warm child is consequently not promoted.  The experiment
supports weight lineage and initialization choice, not official xLAM, native MCP/API, or WebGPU
throughput claims.

### Dynamic xLAM tool-catalog selector transfer (m138)

The [`m138 receipt`](paper/results/raw/m138-xlam-dynamic-selector-transfer-v1.json) keeps the
backbone frozen and trains the two-tower selector against the public derivative's row-local tool
catalog.  This is necessary because the derivative contains 2,702 distinct training tool names
and 2,840 names in the combined candidate union, rather than the fixed 63-tool browser/productivity
pool used by the deployed bundle.  The 3,911 training rows and 500 eval rows are the normalized,
globally slot-disjoint files from m137; 482 held-out rows with overlapping argument slots remain
excluded.

With a fixed 64-step, 256-dimensional selector probe, row-local closed-world tool top-1 improves
from `55.8%` to `60.4%` on the held-out rows.  The global catalog top-1 is only `0%` to `2%`,
which is the more deployment-relevant warning: a selector cannot reliably choose among thousands
of arbitrary APIs from text alone.  The union also contains 194 train-time and 209 combined
row-schema name conflicts, which are retained in the receipt rather than silently canonicalized.
This is useful evidence for candidate retrieval and schema-conditioned dispatch, but it is not
official Salesforce xLAM, generated argument exactness, live API execution, or a workshop-gate
native score; the child is explicitly not promoted.

### Deployment-shaped retrieval before selector scoring (m140)

The m138 result exposed a real runtime bug: when a dense selector was present, `Agent.chat` passed
the entire catalog to the selector and silently bypassed the documented top-`k` retrieval path.
The fix restricts `BoundSelector.rank` to the retrieved candidate names and makes the runtime pass
the retrieved `ToolSpec` list into `hybrid_decode`.  The [`m140 receipt`](paper/results/raw/m140-xlam-runtime-retrieval-selector-v1.json)
replays the public 128-row derivative shard with the exact deployment-shaped path (`k=10`).

Global first-tool exactness improves from `0%` with unrestricted dense selection to `9.375%` with
retrieval followed by selector scoring; schema validity improves from `0.78%` to `14.84%`.  The
row-local upper-bound diagnostic remains `50.78%`, and first-argument exactness remains `0%`, so
this is an efficiency and safety correction—not a quality promotion.  The checkpoint remains
unpromoted, and the result is not official xLAM, live API execution, or native WebGPU evidence.

### Five-surface public continuation and transfer audit (m142)

The [`m142 receipt`](paper/results/raw/m142-five-surface-public-continuation-v1.json) extends the
same 10.5M BPE parent across five public projections: AndroidControl, AgentNet, Mind2Web,
ToolSandbox source metadata, and redacted MCPMark traces.  It uses 4,746 train rows and 1,064
source-disjoint evaluation rows for 48 low-rate CPU updates.  Aggregate held-out assistant-token
accuracy rises `61.15% → 68.65%` and mean loss falls `2.468 → 2.157`; the per-source deltas are
`+9.51` Android, `+0.80` AgentNet, `+0.61` Mind2Web, `+0.24` ToolSandbox, and `−0.16` MCPMark
percentage points.  Exact assistant-sequence accuracy remains `0%` on every source.

The transfer audit finds identical configuration/tokenizer and 51 shared tensors.  Relative
backbone movement is `0.255%` attention/mixer, `0.325%` FFN, `0.742%` embedding, and `0.016%`
normalization; action heads remain unchanged.  This supports low-rate reuse as a controlled
initialization experiment: the matched random-backbone control reaches `39.68%` aggregate token
accuracy, so the warm-minus-random gap is `+28.97` points under the same schedule.  The MCPMark
regression and zero exact sequences still keep the child unpromoted.  The rows are
text/accessibility projections only: no native emulator, desktop VM,
BrowserGym, MCP server, screenshot grounding, or external account was executed.

On the separate 128-row public xLAM-derived first-call diagnostic, the child reaches `7.03%`
first-tool exactness and `12.50%` schema validity through the deployment-shaped retrieval path,
versus `0.78%`/`1.56%` for unrestricted dense selection; row-local retrieval remains a `50.78%`
upper-bound diagnostic and exact arguments remain `0%`.  This confirms the retrieval safety fix but
also shows that the five-surface continuation did not improve general function-calling quality.

### ToolSandbox native base-scenario transfer audit (m143)

The [`m143 receipt`](paper/results/raw/m143-toolsandbox-native-base-transfer-audit-v1.json)
executes the current five-surface warm child and a matched random-backbone control inside the
pinned Apple ToolSandbox simulator and upstream milestone verifier.  It covers all `129` public
base/no-distraction scenarios (the source expands these to `1,032` augmented variants) in a fixed
sorted order.  The warm child and random control each complete all `129` simulator runs without
environment exceptions and both reach `28/129` exact verifier successes (`21.71%`); the aggregate
native difference is therefore `0` points, with six per-scenario outcome changes.

This is stronger native evidence than the earlier three- or five-task smokes, but it remains a
diagnostic rather than an official ToolSandbox result.  The adapter uses a deterministic one-step
scripted user, so multi-tool and multi-user-turn scenarios are intentionally truncated; the
upstream model-based user simulator, official split declaration, full augmented matrix, optional
RapidAPI tools, MCP servers, and external accounts were not executed.  The receipt therefore keeps
`official_split_verified: false` and `native_official_gate_eligible: false`, and the checkpoint is
not promoted.  The matched random result also cautions against reading this one-step score as a
weight-transfer quality win; the separate m142 held-out token audit is the evidence for warm-start
initialization, not this native base-scenario aggregate.

### Current m142 native BrowserGym/MiniWoB evaluation (m144)

The [`m144 receipt`](paper/results/raw/m144-browsergym-native-current-checkpoint-v1.json) runs the
current five-surface m142 child through the complete pinned BrowserGym/MiniWoB plan: `240`
episodes, `60` task variants, four fixed seeds, and a ten-step budget.  Chromium, BrowserGym, and
MiniWoB all executed natively with zero environment or action errors.  The model nevertheless
achieves `0/240` success, produces `0` grounded actions, and emits `2,400/2,400` `noop(0)` actions
(`1,400` explicit abstentions).

This is an official-split **negative** result for the current checkpoint, not a deployment failure:
the native browser gate is technically eligible, but the child is not promoted.  The earlier m43
adapter checkpoint reached `5/240` (`2.08%`) on the same pinned plan, so m144 is a regression in
native UI grounding after the five-surface continuation.  The receipt retains only per-episode
aggregates in Git and hashes the raw trace; it does not claim visual grounding, WebArena, real
email/Notion accounts, Android control, or WebGPU hardware quality.

### Current m142 native MobileGym evaluation (m146)

The [`m146 receipt`](paper/results/raw/m146-mobilegym-native-current-checkpoint-v1.json) runs the
same current five-surface m142 child over all `256` official MobileGym test tasks at the pinned
simulator revision.  Every task completed with zero runtime or judge errors, but only `13/256`
(`5.08%`) passed under the two-step, selector-top-1 protocol.  Weather (`4/9`), clock (`3/9`),
SMS (`1/3`), and Tencent Meeting (`1/9`) are the only non-zero suites; account, notes, map, and
the cross-app suites remain at zero.  The model emitted `mobile_submit_answer` on `215` tasks and
returned no tool on the remaining `41`; all 13 successes came from the answer tool, making the
remaining state-conditioned action-collapse failure explicit.

This is an official-split native simulator result with a bounded **text projection** only:
screenshots and visual encoders were not used, so it is not an Android emulator score, visual
grounding result, or long-horizon mobile-agent claim.  The result matches the earlier m133
`13/256` score but uses the current m142 checkpoint; it therefore does not justify replacing the
deployment checkpoint or claiming a transfer win.  The strict gate accepts the receipt contract,
while the seven independent Android/desktop/tool-server requirements remain open.

### Train-only browser-context adapter from m142 (m148)

The [`m148 adapter report`](paper/results/raw/m148-browser-context-adapter-m142-v1.json) tests
whether the current five-surface child can recover the live accessibility-tree contract without
reading BrowserGym data.  It reuses only the 589 synthetic computer-use rows in
`data/synth/agent_sft.jsonl`, projects quoted element names deterministically, and evaluates ten
disjoint synthetic rows.  Route accuracy improves `70% → 100%`, tool accuracy `10% → 70%`, and
exact arguments `10% → 20%` after 300 low-rate backbone updates plus 1,000 frozen-feature head
updates.

The native [`m148 canary receipt`](paper/results/raw/m148-browsergym-native-adapter-canary-v1.json)
then runs that child in ten pinned BrowserGym/MiniWoB episodes.  It reaches `0/10`, with zero
grounded actions and 100 no-op steps; model outputs alternate among `move_cursor`, `open_app`, and
abstention, but none match the live accessibility candidates.  This is not an official split
score and the full run was intentionally not promoted after the bounded canary failed.

The paired [`m148 weight audit`](paper/results/raw/m148-browser-context-m142-weight-transfer-v1.json)
finds all 51 tensors compatible and tokenizer-identical, but relative movement of 24.76% in the
embedding group, 3.20% in attention/mixer, 4.51% in FFN, and 78.11% in action heads.  The result
supports the diagnosis that offline contract fitting alone does not restore native grounding;
the m142 child and m148 adapter remain unpromoted.

### Optional DOM-coordinate grounding bridge (m151/m152)

The m148 failure exposed a concrete modality gap: several live MiniWoB controls are generic/SVG
nodes that are absent from BrowserGym's accessibility tree.  The optional `--coordinate-fallback`
diagnostic therefore reads only the native DOM snapshot's `isClickable` indices, text-backed child
nodes, and layout bounds, then converts CSS geometry using the page device-pixel ratio and
BrowserGym screenshot scale.  It does not read screenshots, task verifiers, hidden labels, or
BrowserGym task plans, and the model prompt remains the original trained accessibility contract;
the geometry is a runtime sidecar used only when a high-level click cannot resolve an accessibility
bid.

The ten-episode [`m151 canary receipt`](paper/results/raw/m151-browsergym-native-coordinate-canary-v1.json)
solves all four `ascending-numbers` seeds (`4/10`, `40%`) with grounded coordinate clicks.  The
complete [`m152 receipt`](paper/results/raw/m152-browsergym-native-coordinate-full-m148-v1.json)
solves `4/240` (`1.67%`), with all successes still confined to that suite; the remaining tasks
continue to expose the adapter's fill/selector collapse.  Because this bridge changes the native
action path and is not the official BrowserGym protocol, both receipts set `official_split_verified:
false` and cannot replace the m144 gate result (`0/240`, no coordinate fallback).  The gain is
runtime grounding evidence, not evidence that m148's pretrained weights transferred successfully.

### EnterpriseOps-Gym public email retrieval diagnostic (v10)

To probe realistic enterprise tool breadth without contaminating training or claiming an execution
score, the frozen v10 dense selector was evaluated on the 67 email task IDs exposed in both the
public `oracle` and `plus_15_tools` configs at HF revision
`c8e538eae8a6205294f0a86675fefdc1fac408f6`.  The adapter consumes only task/domain/system/user
text and selected tool names.  It drops `verifiers` and `gym_servers_config`, deduplicates repeated
candidate names in memory, generates name-only descriptions, and never executes MCP servers or
state transitions.  The two raw JSON payloads remain outside Git; their sizes and SHA-256 hashes
are bound in the [`m14 receipt`](paper/results/raw/m14-enterpriseopsgym-email-retrieval-v10.json),
whose tracked SHA-256 is `b4fc14edc5e87f6bec14db52b21f90eee0ab8beb34ab4e6f215444c68429d669`.

| Metric | Result |
| --- | ---: |
| Records | 67 email tasks |
| Oracle tools (mean) | 5.69 |
| Candidate tools (mean) | 20.00 |
| Retrieval hit@1 | 26.87% (18/67) |
| Retrieval hit@3 | 56.72% (38/67) |
| Retrieval hit@5 | 79.10% (53/67) |

The dominant false positives were generic `update_label` (21), `get_user_profile` (12), and
`send_message` (5), showing that the current dense selector is biased toward frequent tool names
when descriptions contain no schemas or examples.  This is useful failure evidence for the next
training pass—schema-conditioned descriptions, calibrated abstention, and stateful local MCP
execution—but it is explicitly not an EnterpriseOps task-success, verifier, or leaderboard score.

### v10 no-guard dense WebGPU continuation

The next continuation starts from the v9r retrieval-sidecar child and keeps the same 10.5M
parameter BPE model, tokenizer, and 62-tool catalog.  Its 3,000-step report covers 114 normalized
rows (86 repeated synthetic mobile rows plus bounded AndroidControl/AITW rows) and holds out two
mobile episodes and four productivity rows.  Route accuracy is `100%`; the held-out mobile dense
selector is `6/10` (`60%`) and the held-out productivity selector is `3/4` (`75%`).  A weight audit
found 51 shared tensors, no model or tokenizer mismatch, zero backbone movement, and an action-head
relative delta L2 of `0.5294`.  This establishes compatible pretrained-weight reuse and head
movement, not a no-transfer improvement claim.

The fp32/fp16 ONNX graphs and serialized dispatch heads pass the export parity gate (fp32 logits
drift `8.58e-06`, fp32 hidden drift `6.14e-06`, fp16 argmax agreement `1.0`).  The explicit in-app
browser WebGPU run with `mobile_guard=0&selector=dense` reports `4/9` exact tools, `4/9` exact
arguments/actions, `6/9` schema-valid actions, and `4/9` closed-loop successes: mobile `2/7`,
productivity `2/2`.  Timing was not collected.  The full identity and summary hash are in the
[`m12-v10 receipt`](paper/results/raw/m12-webgpu-mobile-productivity-v10.json).

This result is a clearer no-guard reproduction than the retrieval ablation, not a higher-accuracy
result: the learned dense selector remains the limiting component.  It is still synthetic,
text-first, single-concurrency state evaluation with no screenshot grounding, trusted OS input,
real accounts, official environment runner, or hardware throughput measurement.  The retrieval
sidecar's earlier `9/9` result must remain reported separately from this learned policy.

### Pretrained-head transfer control

The dispatch runner now exposes an explicit `--probe-init` switch so transfer is tested rather
than assumed.  A matched 3,000-step replay from the v9r parent compared the inherited route and
dense-selector heads with seeded-random probe initialization; the backbone, tokenizer, data bytes,
holdout IDs, optimizer settings, and seed were held constant.  Parent-head reuse raised training
selector accuracy from `75.86%` to `86.21%` and kept action-head movement smaller (`0.4426` vs
`1.0753` relative delta L2), but held-out selector accuracy was identical: mobile `6/10` and
productivity `3/4` for both arms.  The result supports lineage and optimization stability, not a
quality gain.  The complete hash-bound control is [`m13 transfer ablation`](paper/results/raw/m13-mobile-dispatch-transfer-ablation.json).

For comparison, an earlier AndroidControl-only baseline continued the WebGPU-tier parent
(`webgpu-10m-hybrid`, 10,524,544 parameters) for eight CPU
SFT updates at `1e-5` and `max_seq_len=2048`.  On the same normalized rows, mean assistant loss
  changed from `8.6985` to `7.9501`; assistant-token accuracy was `2.33%` and exact trajectory
  accuracy was `0%`.  These are language-model bridge metrics, not emulator task success.  The
  baseline child checkpoint SHA-256 is
  `9ea68807ac758c564200a5649f9196ecde3bccb8a66edae04eec8c6c14813eae`; its ONNX bundle manifest
  SHA-256 is `13e5549d69ab9286ab41e2c75901ee67412327487997f4740dd6c31f7f726940`.  The bundle passed
  the hard PyTorch parity gate for fp32 and fp16 logits/hidden graphs (fp32 maximum absolute drift
  `7.39e-06`, fp16 `6.08e-03`) and contains a 10.5M-parameter action graph suitable for the
  existing static WebGPU demo.  The child checkpoint and bundle are deliberately kept outside Git
  until a full held-out/runtime evaluation receipt exists.

### Public Mind2Web DOM-grounded pointer transfer (m157)

The [`m157 receipt`](paper/results/raw/m157-mind2web-grounded-transfer-v1.json) is the first
action-level browser continuation in this refresh that keeps the original public DOM candidates
instead of only task text.  It uses the CC-BY-4.0 Mind2Web train split at revision
`17ece8eb89862368edc0cc806acee6fca5163474`, with 9 parent records / 219 actions for training and
3 disjoint parent records / 63 actions for an in-source holdout; typed slot values are disjoint as
well.  The [grounding adapter](../scripts/export_mind2web_grounded_rows.py) caps each candidate
snapshot at 12 elements and the [trainer](../scripts/train_grounded_mind2web.py) expands the
pointer vocabulary from 17 to 19 rows.

The warm m142 arm starts at `0/63` exact spans and reaches `15/63` (`23.81%`) after 64 updates;
the matched random arm remains `0/63`.  The weight audit records warm relative movement of `0.90%`
embedding, `0.50%` attention/mixer, `0.55%` FFN, `0.019%` normalization, and `46.40%` action
heads.  This is useful evidence for the requested adoption recipe—reuse the body, give the new
grounding head the larger rate—but the holdout is small and in-source.  It is not the official
Mind2Web test score, BrowserGym/WebArena success, screenshot-grounded control, or a real browser,
email, Notion, MCP, or external-account side effect.

### Current-parent mobile dispatch and pointer adaptation (m158)

The [`m158 receipt`](paper/results/raw/m158-mobile-dispatch-transfer-v1.json) repeats the mobile
adaptation from the current m142 five-surface parent rather than the older m129 parent.  It uses the
public AndroidControl/AITW projection for 4,096 train rows and keeps the disjoint 904-row public
evaluation file outside optimization.  Focused oversampling targets `mobile_long_press`,
`mobile_navigate_home`, and `mobile_submit_answer`; the pointer pass uses 300 updates and four
held-out literal spans.  The source screenshots are deliberately omitted, so these are text and
accessibility action diagnostics rather than visual Android control.

On the held-out projection, route accuracy is `100%` and selector top-1 is `41.26%`; pointer exact
is `1/4` (`25%`).  The weight audit finds exact tokenizer/config compatibility, zero movement in
the inherited embedding, mixer, FFN, and normalization tensors, and `124.77%` relative action-head
movement.  This is consistent with a frozen-body / high-rate-head deployment recipe, but it is not
evidence that the transferred body is optimal.

The child was then run in the pinned MobileGym simulator over the first 20 official test IDs with
the state-diff judge, a five-step cap, and a DOM/text observation projection.  It passes `1/20`
(`5%`) with no runtime or judge errors, exactly matching the parent m135 canary on the same range;
the child emits `mobile_submit_answer` on 13 tasks.  This is a bounded native canary, not the full
MobileGym score, an Android emulator result, screenshot grounding, or a publication-gate pass.

### Current-child AndroidControl/AITW continuation and weight audit (m166)

The [`m166 receipt`](paper/results/raw/m166-androidcontrol-current-transfer-v1.json) continues the
current `10,524,544`-parameter BPE child for 64 CPU updates on 4,096 rows from the public
AndroidControl/AITW text-and-accessibility train projection.  The 904-row public test projection
was held out from optimization; both projections omit screenshots by construction.  Assistant-token
accuracy on the held-out projection rises from `59.38%` to `69.67%` and mean loss falls from
`2.849` to `2.041`, while exact sequence accuracy remains `0%`.

The paired weight audit in the [`m166 receipt`](paper/results/raw/m166-androidcontrol-current-transfer-v1.json)
finds no config, shape, or tokenizer mismatch across 51 shared tensors.  Relative movement is
`0.95%` in embeddings, `0.30%` in attention/mixer, `0.39%` in FFN, `0.018%` in normalization, and
`0%` in the inherited action heads.  This supports reusing the compatible backbone with a smaller
learning rate for transfer, but it does not establish optimality, visual grounding, emulator reward,
AndroidWorld/MobileGym success, or a public leaderboard score.

### Current-child MCPMark redacted-trajectory transfer (m167)

The [`m167 receipt`](paper/results/raw/m167-mcpmark-current-transfer-v1.json) continues the current
m166 AndroidControl-adapted child on the public MIT-licensed
[MCPMark trajectory-log](https://huggingface.co/datasets/Jakumetsu/mcpmark-trajectory-log) at
revision `e50578f0ab904d8e6a7c576c387c1e76ae482c89`.  Eight parent records train and two
source-disjoint parent records remain held out.  Tool outputs and assistant free text are fixed
redaction markers, and visual input is omitted; neither an MCP server nor a verifier is started.

With 32 matched updates, the warm current-child arm improves held-out assistant-token accuracy
from `41.28%` to `43.18%`; the random-backbone arm improves from `0.69%` to `4.72%`.  The warm
minus random after-training gap is `+38.46` percentage points, while exact sequence accuracy is
`0%` for both arms.  Warm relative movement is approximately `0.52%` embedding, `0.26%`
attention/mixer, `0.30%` FFN, and `0.011%` normalization, versus roughly `123.53%`, `77.88%`,
`87.81%`, and `7.91%` for the random arm.  The matched compatibility audit finds 51 shared
tensors with equal tokenizer and no config or shape mismatches.  This supports the low-rate
verified-backbone transfer recipe, but it is diagnostic only—not an official MCPMark score, live
MCP/server/verifier execution, native browser or desktop success, screenshot grounding, or a real
Notion/email side effect.

### Current-child four-surface public continuation (m168)

The [`m168 receipt`](paper/results/raw/m168-current-cross-surface-transfer-v1.json) continues the
current m167 child over four public projections: AndroidControl text/accessibility actions,
AgentNet Ubuntu desktop actions, grounded Mind2Web browser actions, and redacted MCPMark
trajectories.  The source references are [Android-Control-84k](https://huggingface.co/datasets/OfficerChul/Android-Control-84k),
[AgentNet](https://huggingface.co/datasets/xlangai/AgentNet),
[Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web), and
[MCPMark trajectory-log](https://huggingface.co/datasets/Jakumetsu/mcpmark-trajectory-log).
The mixture contains `4,637` public train rows and `1,043` held-out rows; the receipt binds each
input hash, revision, source-disjoint parent-record audit, and public URL.  AndroidControl rows
omit screenshots, and MCPMark tool outputs/free text remain redacted.

After 32 CPU updates at learning rate `1e-5`, the warm current-child arm reaches `70.11%`
aggregate held-out assistant-token accuracy versus `30.80%` for the matched random-backbone arm,
a `+39.31` percentage-point gap.  Warm-start remains better after training on every held-out
surface: AndroidControl `75.39%` vs `38.37%`, AgentNet `49.60%` vs `1.74%`, Mind2Web `61.93%`
vs `0%`, and MCPMark `33.26%` vs `0.89%`.  Exact sequence accuracy is `0%` for both arms.

The compatibility audit finds 51 shared tensors with equal tokenizers and no config/shape
mismatches.  Warm relative movement is `0.50%` embedding, `0.19%` attention/mixer, `0.24%` FFN,
`0.012%` normalization, and `0%` action-head movement; the matched random control moves those
groups by approximately `123.44%`, `77.88%`, `87.81%`, and `7.90%`.  This strengthens the
low-rate transferred-body recommendation, but it remains a teacher-forced diagnostic—not an
official AgentNet, Mind2Web, AndroidControl, or MCPMark score, native Android/desktop/browser/MCP
execution, screenshot grounding, or a real email/Notion side effect.

### Current-child WebGPU browser smoke (m169)

The [`m169 receipt`](paper/results/raw/m169-current-child-webgpu-browser-smoke-v1.json)
exercises the m168 warm child after a fresh export through the static browser harness.  The
bundle verifier finds every generated artifact, the manifest hashes match, and the fp32/fp16
PyTorch-to-ONNX parity gates pass (fp32 logits max error `8.58e-6`; fp16 logits max error
`5.42e-3`).  The action-only fp16 graph is `21.43 MB`; the model remains `10,524,544`
parameters and the checkpoint is pinned by SHA-256.

In the Codex in-app Browser, an explicit single-provider WebGPU session loads and completes
eight one-step semantic DOM loops.  Schema validity is `8/8` and exact action/argument,
state-transition, and closed-loop success are each `2/8` (`25%`).  Harness TTFA p50 is
`8.4 ms` and closed-loop p50 is `33.3 ms`; `type_text` and `open_url` are the two successful
cases.  This is a deployment and runtime smoke diagnostic only: the fixture uses text-only
prompts, synthetic untrusted events, local semantic targets, no screenshots, no multi-step
planning, and no external navigation.  It is not a BrowserGym/MiniWoB score, native
Android/desktop/MCP execution, trusted browser control, or real email/Notion side effect.
The bundle is not uploaded because Hugging Face authentication is still not configured.

### Current-child ToolACE first-action transfer (m171)

The [`m171 receipt`](paper/results/raw/m171-current-child-toolace-transfer-v1.json) adds the
public Apache-2.0 [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) snapshot
at revision `6bda777c88d21e5a204703c1ee45597a8fa4f734`.  The raw `37,154,735` bytes are
hash-bound.  The importer accepts only strict bracketed calls such as
`[email_send(to="user@example.com")]`, projects the first canonical assistant action, and
keeps tool responses and later turns out of the WebGPU SFT projection.  It accepts `8,993` of
`11,300` rows; `2,293` rows have no strict first action and `14` fail schema/projection checks.
The resulting source-record/prompt-disjoint projection contains `8,044` train and `949` held-out
rows.  This is a reproducible adapter boundary, not an official ToolACE split.

For a CPU-bounded continuation, `1,024` train rows and `256` held-out rows were sampled from the
frozen projection and matched against a fresh random-backbone control for 32 updates at
learning rate `1e-5`.  The warm current child improves held-out assistant-token accuracy
`39.35% → 42.23%` while the random control reaches `8.00%`; the warm-minus-random gap after
training is `+34.23` percentage points.  Both arms remain at `0%` exact sequence accuracy.
Warm relative movement is only `0.47%` embedding, `0.16%` mixer, `0.19%` FFN, and `0.007%`
normalization, versus `123.40%`, `77.79%`, `87.84%`, and `7.91%` for the random body; action
heads are unchanged.  The 51-tensor compatibility and equal-tokenizer audit supports the same
low-rate transferred-body recipe, but this bounded first-action projection is diagnostic only:
it is not BFCL, official ToolACE, multi-turn execution, native browser/mobile/desktop/MCP,
email/Notion side effects, or a public Hub upload.

### Current-child ToolACE multi-turn transfer (m172)

The [`m172 receipt`](paper/results/raw/m172-current-child-toolace-multiturn-transfer-v1.json)
reuses the same byte-pinned ToolACE source but preserves every user turn, assistant free-text turn,
strict assistant action, and tool response in the canonical `Conversation` history.  The full
projection accepts `8,992` of `11,300` rows (`8,043` train / `949` held out); `1,187` tool-response
messages are retained in train and `170` in held out.  One additional row is rejected by the
schema/prompt-marker guard, so this is a stricter stateful projection than m171 rather than a
silent rewrite of its first-action data.

On a matched CPU arm (`256` train / `64` held out, 16 updates, batch 2, learning rate `1e-5`),
the warm child improves held-out assistant-token accuracy `17.58% → 18.56%`; the random control
improves `0.40% → 1.19%`, leaving a `+17.37` percentage-point warm-minus-random gap.  Both arms
remain at `0%` exact sequence accuracy.  Warm body movement is `0.236%` embedding, `0.124%`
mixer, `0.153%` FFN, and `0.006%` normalization; action heads are unchanged.  This supports
preserving tool history for future stateful SFT, but remains a bounded teacher-forced diagnostic:
it is not official ToolACE/BFCL scoring, tool execution, email/Notion side effects, or native
browser/mobile/desktop/MCP evidence.

### Current-child ToolACE action-history transfer (m173)

The [`m173 receipt`](paper/results/raw/m173-current-child-toolace-action-history-transfer-v1.json)
uses the same source-disjoint multi-turn data but removes non-action assistant prose from the
history.  User turns, strict assistant calls, and tool responses remain; `1,806` assistant prose
turns are explicitly counted as omitted.  The full projection still contains `8,992` accepted
rows, `9,679` action turns, `1,357` tool responses, and `21,164` retained messages.

With the same bounded `256/64` split and matched 16-update warm/random arm, held-out assistant-token
accuracy is `48.95% → 49.99%` for the warm child versus `0.02% → 0.51%` for random, a `+49.48`
percentage-point gap.  Exact sequence accuracy remains `0%`, but the action-history target is far
more WebGPU-aligned than the full-history `18.56%` result because the loss is concentrated on
action decisions rather than assistant prose.  This is still a teacher-forced diagnostic, not
official ToolACE/BFCL scoring or executed email/Notion/browser/MCP behavior.

### Current-child ToolACE action-history free-run probe (m174)

The [`m174 receipt`](paper/results/raw/m174-current-child-toolace-action-history-free-run-v1.json)
replays the WebGPU-shaped catalog-plus-history decode path over `16` held-out conversations and
`30` action steps without dispatching any tool.  The corrected decoder separates serialized
function-catalog context from argument-grounding text, preventing catalog JSON from being copied
into arguments.  The warm m173 child reaches only `10.0%` tool-name exactness, `3.33%` full
tool-and-argument exactness, `60.0%` schema-valid output, and `0%` exact whole episodes.
This is the required warning against promoting teacher-forced `49.99%` action-token accuracy to
autonomous capability: selector/routing and grounded argument behavior still require substantial
work before email, Notion, browser, or MCP execution can be claimed.

### Current-child ToolACE catalog-aware selector transfer (m176)

The [`m176 selector receipt`](paper/results/raw/m176-current-child-toolace-action-history-selector-transfer-v1.json)
trains the dense two-tower selector on the exact catalog-plus-action-history context used by the
WebGPU path.  It evaluates `113` held-out actions from the same source-disjoint `256/64` ToolACE
projection, with `4.52` candidates per action on average.  The inherited selector is `21.24%`
top-1 and `73.45%` top-3; warm selector retraining reaches `29.20%` top-1 and `76.11%` top-3,
but the matched random-backbone selector reaches `32.74%` top-1 and `76.11%` top-3.  The warm
selector moves substantially (`124.05%` query tower and `204.11%` tool tower), so movement alone
is not evidence of useful transfer.

The paired [`m176 free-run receipt`](paper/results/raw/m176-current-child-toolace-action-history-selector-free-run-v1.json)
improves tool-name exactness from `10.0%` to `16.67%` and whole-episode exactness from `0%` to
`6.25%`, but full action exactness remains `3.33%` and schema-validity falls to `53.33%`.  Because
the warm selector does not beat the matched random control on candidate ranking, it is not adopted
as representation evidence; the free-run increase is retained as a bounded dispatch diagnostic,
not an official ToolACE/BFCL result or native browser/email/Notion/MCP success.

### Current-child ToolACE dynamic pointer transfer (m177)

The [`m177 pointer receipt`](paper/results/raw/m177-current-child-toolace-action-history-pointer-transfer-v1.json)
expands the pointer vocabulary to `96` argument names observed in the training projection and
trains only on `182` train / `63` held-out string spans that occur verbatim in the catalog/history
context.  Warm decoded-value exactness is `9.68%` on the covered held-out spans, above the inherited
`0%` but below the matched random-backbone `19.35%`; covered-span rate is only `49.21%`, so this is
not a general argument solution.

The paired [`m177 free-run receipt`](paper/results/raw/m177-current-child-toolace-action-history-pointer-free-run-v1.json)
remains at `3.33%` full action exactness and `3.33%` argument exactness (`13.33%` tool-name exact).
During this probe the pointer path was also hardened so a serialized catalog cannot be copied as an
argument candidate; the allowed pointer interval is now the grounding suffix, while tool-result
history remains available for stateful copying.  The warm pointer is not adopted: it loses to the
matched random control on direct span transfer and does not improve the free-run score.

### Current-parent MCPMark redacted-trajectory transfer (m159)

The [`m159 receipt`](paper/results/raw/m159-mcpmark-current-parent-transfer-v1.json) repeats the
public [MCPMark trajectory-log](https://huggingface.co/datasets/Jakumetsu/mcpmark-trajectory-log)
comparison from the current m142 parent.  The input is pinned to revision
`e50578f0ab904d8e6a7c576c387c1e76ae482c89`; eight rows train and two parent records remain held
out.  Tool outputs and assistant free text are replaced with fixed redaction markers, so no live
service state or verifier result is admitted to SFT.

The warm child improves held-out assistant-token accuracy from `39.38%` to `43.18%`; the matched
random-backbone arm reaches `3.12%` after the same 32 updates, a `+40.06` point warm-start gap.
Both arms remain at `0/2` exact sequences.  Warm relative movement is `0.46%` embedding,
`0.18%` mixer, `0.21%` FFN, and `0.008%` normalization, while the random arm moves the backbone
by roughly `79%–123%` per group.  This strengthens the weight-adoption recommendation—reuse the
verified body with a low rate and isolate service/action adapters—but is still not native MCPMark,
Notion, email, browser, or external-account success.

### Current browser-context child WebGPU export (m261)

The [`m261 receipt`](paper/results/raw/m261-webgpu-export-current-browser-context-v1.json)
exports the same checkpoint into four ONNX graphs (`model`, `model.fp16`, `action_model`, and
`action_model.fp16`) plus tokenizer, heads, dispatch metadata, and a schema-3 bundle manifest.
All four graphs pass the exporter’s hard CPU parity gate: fp32 logits max absolute error is
`7.15e-6`, fp16 logits max absolute error is `6.05e-3`, and fp16 logits argmax agreement is
`100%`; the action-only hidden graph also passes (`4.35e-6` fp32 / `4.59e-3` fp16).

This is the exact current-checkpoint WebGPU artifact identity, not a performance or quality claim.
The receipt keeps `native_webgpu_provider_verified` and `public_space_uploaded` false: no browser
hardware session, throughput measurement, public Space revision, or external email/Notion/browser
side effect has been attached to this export.  The older checked-in static bundle and public
legacy Space manifest must not be conflated with this current browser-context child.

### Current browser-context child MobileGym official evaluation (m262)

The [`m262 receipt`](paper/results/raw/m262-mobilegym-native-current-browser-context-v1.json)
is the first current-checkpoint run over the complete pinned MobileGym public test split.  The
unchanged simulator, task lifecycle, native action handlers, and upstream state-diff judge execute
all `256/256` test IDs with zero environment or judge errors.  The source revision, train/test
split hashes, task-ID hash, checkpoint SHA-256 (`bc1aca…`), and a SHA-256 of the full task-level
receipt are recorded; only aggregate/task-result hashes are committed.

At the disclosed two-step cap, the current text/DOM-projection policy passes `1/256` (`0.39%`).
The only non-zero suite is `crossapp_life` (`1/29`); account, notes, browser-like app flows, and
the remaining suites are `0`.  The dominant trace pattern is `mobile_press_enter` (`110` calls)
and `mobile_input_text` (`146` calls), showing that the current policy collapses toward generic
input/submit actions instead of state-specific navigation.  This is valid native MobileGym
evidence, but it is not visual screenshot grounding, Android-emulator performance, or evidence
that email/Notion/browser actions are ready for deployment.

### Workshop gate after current MobileGym + BrowserGym receipts (m264)

The [`m264 gate`](paper/results/raw/m264-workshop-gate-current-mobile-browser-v1.json) adds the
existing full BrowserGym/MiniWoB receipt to m263.  Its checkpoint hash is the same current
browser-context child (`bc1aca…`), and the pinned 240-episode, 60-variant, four-seed plan is
official-split verified.  The gate therefore passes both browser and mobile native requirements,
along with the catalog, transfer/no-transfer, and public-artifact checks.

`ready` remains false with ten blockers: AndroidWorld, MobileSafetyBench, iOSWorld, OSWorld,
OSWorld-V2, AgentNet, ToolSandbox, MCPMark, EnterpriseOps-Gym, and current native WebGPU
capability/latency.  The BrowserGym receipt reports `5/240` success (`2.08%`) for the current
checkpoint; it is still text/accessibility-tree evidence, not visual grounding or real-account
email/Notion control.

### Current browser-context child native WebGPU capability (m265)

The [`m265 receipt`](paper/results/raw/m265-webgpu-native-current-browser-context-v1.json)
closes the current-checkpoint hardware-runtime evidence gap.  A real Chromium page was launched
with an explicit `webgpu` provider request and no WASM retry; `navigator.gpu.requestAdapter()`
reported `vendor=apple; architecture=metal-3`, and the action graph was the manifest-bound
`bc1aca…` checkpoint export (`action_model.fp16.onnx`, SHA-256 `8856dca…`).  The three bounded
email, URL, and Notion cases each produced a schema-valid exact action on all `30/30` measured
repetitions after three warmups.

The p50 end-to-end dispatch latency is `7.4 ms`, the action-input throughput estimate is about
`1,391 tokens/s`, and the conservative graph-plus-host-tensor estimate is `20.46 MB`.  This is
native WebGPU capability/performance evidence only: the two productivity routes are explicit
intent guards, the calls are local predictions, no email/browser/Notion account was touched, and
`closed_loop_success` is `0`.  It must not be presented as learned benchmark accuracy or a
cross-device throughput guarantee.

### Workshop gate after current native WebGPU receipt (m266)

The [`m266 gate`](paper/results/raw/m266-workshop-gate-current-mobile-browser-webgpu-v1.json)
replaces the m261 export placeholder with the manifest-bound m265 native Chromium receipt.  The
gate now passes `webgpu:native_capability_and_latency` together with the current MobileGym and
BrowserGym/MiniWoB receipts, catalog, transfer/no-transfer ablation, and public-artifact manifest.
`ready` remains `false` with nine independent native requirements blocked: AndroidWorld,
MobileSafetyBench, iOSWorld, OSWorld, OSWorld-V2, AgentNet, ToolSandbox, MCPMark, and
EnterpriseOps-Gym.  These require their real environments, official splits, and task rewards;
the local WebGPU capability receipt cannot substitute for them.

### Current browser-context child xLAM public derivative evaluation (m267)

The [`m267 receipt`](paper/results/raw/m267-xlam-current-browser-context-v1.json) adds a
current-checkpoint, public-data tool-use control.  It evaluates 128 source-disjoint rows from a
1,000-row Apache-2.0 derivative shard of xLAM Function Calling, with the exact checkpoint hash,
source JSONL hash, and evaluator hash recorded.  The original Salesforce dataset is gated and its
official split is not authenticated or verified here, so this is explicitly a derivative result.

Row-local retrieval reaches `50.78%` first-tool exactness, but only `1.56%` first-argument
exactness.  The deployment-shaped runtime retrieval-plus-selector path falls to `11.72%` tool
exactness and `14.06%` schema validity; global selector exactness is `0.78%`.  This isolates the
same failure seen in the native browser/mobile runs: candidate retrieval can route a tool, while
argument grounding and state-conditioned selection are not yet reliable.  It is first-call-only,
with no live API calls, side effects, official xLAM score, or native-environment claim.

### Native learned-selector control with guards disabled (m268)

The [`m268 receipt`](paper/results/raw/m268-webgpu-native-current-unguarded-retrieval-v1.json)
uses the same current checkpoint and Apple Metal-3 WebGPU runtime, but disables both lexical
productivity and URL guards while selecting through the retrieval sidecar.  This is the more
honest learned-control measurement: URL and Notion are exact on `30/30` repetitions each, while
the email request is routed to `type_text` with `{text: "Dana the quarterly report"}` on all
`30/30` repetitions instead of `send_email` with a recipient.  The aggregate is `2/3` local
structured actions, p50 `7.5 ms`, and about `1,379 tokens/s`; no external action was executed.
The result confirms that the current WebGPU demo's email success depends on the explicit guard,
and that email argument/schema grounding remains a release blocker.

### Tokenizer-bound native WebGPU rerun (m270)

The [`m270 receipt`](paper/results/raw/m270-webgpu-native-tokenizer-bound-v1.json) re-exports the
repaired child with the BPE tokenizer recorded by the checkpoint (`836540…`) and exercises the
same Chromium/WebGPU/Apple Metal-3 harness.  The exporter now fails closed when a supplied BPE
file has the wrong SHA-256, preventing a same-vocabulary tokenizer from silently changing token
IDs.  With the matching tokenizer, the learned dense selector produces exact email, URL, and
Notion structured actions on `90/90` local repetitions; p50 latency is `7.85 ms` and the
conservative action-graph estimate is `20.46 MB`.

This is a corrected local deployment receipt, not a replacement for the official MobileGym or
BrowserGym results: no external side effect ran, `closed_loop_success` is `0`, and the nine
native publication blockers listed by m266 remain unresolved.

### Current checkpoint ToolSandbox projection control (m272)

The [`m272 receipt`](paper/results/raw/m272-toolsandbox-current-checkpoint-text-projection-v1.json)
runs the current `10,524,544`-parameter browser-context checkpoint against a pinned, public
Apple ToolSandbox AST/text projection.  On `20` rows and `25` candidate tools, row-local
retrieval routes `55%` of calls exactly, grounds all arguments exactly on `30%`, and produces a
schema-valid call on `95%` of rows.  The category breakdown exposes the stateful weakness:
`STATE_DEPENDENCY` tool exactness is only `25%`, versus `64.29%` for the larger
`CANONICALIZATION` slice.

This is an offline projection diagnostic: the upstream ToolSandbox simulator, model-based user,
milestone verifiers, official split, and external services were not executed.  It is therefore
not an official ToolSandbox score, an MCPMark score, or native WebGPU capability evidence, and it
does not close the ToolSandbox publication blocker.

### Public mobile/desktop continuation and warm-vs-random weight audit (m273)

The [`m273 receipt`](paper/results/raw/m273-cross-surface-public-weight-transfer-v1.json) records
a source-bound continuation over the public [AndroidControl](https://github.com/google-research/google-research/tree/master/android_control)
mirror and the public [OpenCUA AgentNet](https://github.com/xlang-ai/OpenCUA) text projection.
The deterministic bounded slice uses `512` training rows per source and `32` held-out rows per
source, with the AndroidControl official train/test boundary preserved and AgentNet parent
trajectories kept source-disjoint.

Starting from the current tokenizer-compatible checkpoint, 16 SFT updates improve held-out
teacher-forced token accuracy from `51.81%` to `55.44%` (`3.772` to `3.077` mean loss); AndroidControl
rises `60.91%`→`65.44%`, and AgentNet rises `45.88%`→`48.92%`.  The matched random-backbone control
reaches only `0.53%` held-out token accuracy after the same updates.  Warm shared-backbone movement
is small (`0.11%` attention/mixer, `0.14%` FFN, `0.18%` embedding relative L2), while the random
control moves by `77.88%`, `87.79%`, and `119.72%` respectively.  This supports adopting compatible
pretrained weights with lower backbone learning rates, but it is not proof that transfer is
optimal.

The bounded four-action AgentNet control remains `0/1` successful trajectories for both warm and
random children.  Screenshots were not used, and neither Android emulators nor desktop runtimes
were launched; this is training/weight-transfer evidence, not an official AndroidControl,
AgentNetBench, OSWorld, or native WebGPU score.

### Extended public weight-transfer continuation (m286)

The [m286 receipt](paper/results/raw/m286-cross-surface-public-weight-transfer-v1.json) extends
the same source-disjoint AndroidControl/AgentNet protocol to `32` low-rate updates (`1,024`
training rows and `64` held-out rows).  The warm parent arm raises held-out token accuracy from
`51.81%` to `57.04%` and lowers mean loss from `3.772` to `2.716`; AndroidControl rises from
`60.91%` to `66.67%`, and AgentNet rises from `45.88%` to `50.76%`.  The matched random arm
reaches only `8.71%` held-out token accuracy after the same budget.

The warm shared-backbone movement remains small (`0.008%` normalization to `0.374%` embedding
relative L2), while the random arm moves `7.8%`–`119.7%` by group.  This strengthens the
pretrained-initialization and smaller-backbone-learning-rate policy.  It does not promote the
child: sequence exactness is still `0/64`, and no emulator, browser, desktop VM, screenshot
grounding, MCP server, or external account was executed.

### ToolSandbox runtime-head transfer control (m288)

The [m288 receipt](paper/results/raw/m288-toolsandbox-runtime-head-transfer-v1.json) trains only
the route and dense candidate-selector heads against the pinned public ToolSandbox AST projection:
`107` source-disjoint training rows and `20` held-out rows, with the `10,524,544`-parameter body
frozen.  The warm parent improves row-local selector top-1 from `45%` to `80%`, selector top-3
from `90%` to `95%`, and app-action routing from `0%` to `100%`.  A matched random-body control
also reaches `80%`/`95%`/`100%`, which shows that this small projection mostly measures head
capacity and candidate-list regularity rather than transferable backbone knowledge.

The warm and random head movements are large (`1.31`–`1.43` and `1.27`–`1.49` relative L2), so
neither child is promoted.  This is a public schema-adapter diagnostic only: ToolSandbox's
simulator, model-based user, milestone verifiers, external APIs, and native environment were not
executed.  The native ToolSandbox publication blocker therefore remains unchanged.

### Public Mind2Web grounding continuation and BrowserGym canary (m289)

The [m289 receipt](paper/results/raw/m289-mind2web-browsergym-transfer-v1.json) continues the
current `10,524,544`-parameter BPE checkpoint on a public Mind2Web **train-only** DOM/action
projection: `36` train conversations (`219` decisions) and `12` parent/typed-slot-disjoint held-out
conversations (`63` decisions).  The browser pointer vocabulary adds `target_id` and `value`, but
held-out exact pointer-span accuracy remains `0/63` before and after 64 low-rate updates.

The child moves the shared backbone only `0.286%`–`0.309%` by relative L2 across attention and FFN
groups (`0.293%` for embeddings), preserving configuration and tokenizer compatibility.  In a
separate native, accessibility-only canary over the pinned BrowserGym/MiniWoW environment, it
achieves `0/20` successes.  This is a reproducible negative transfer result: no official Mind2Web
test data, BrowserGym full-plan claim, screenshot grounding, email/Notion account, or external side
effect was used, and the child is not promoted.

### Mind2Web browser-head and pointer transfer audit (m290–m291)

The [m290–m291 receipt](paper/results/raw/m290-mind2web-browser-head-pointer-v1.json) separates
browser tool selection from argument grounding on the same public Mind2Web train-only projection.
With the body frozen, a warm route/dense-selector adapter raises held-out selector top-1 from
`19.05%` to `92.06%` (`85.71%`→`100%` top-3); a matched random-head control reaches `85.71%`
top-1.  Warm head movement is smaller (`0.674` route, `0.820` selector relative L2) than random
(`1.253`, `1.133`), supporting compatible initialization for the proxy only.

Mapping public `web_click`/`web_type`/`web_select` into deployed `click`/`type_text` schemas and
training `target`/`text` pointer rows improves held-out pointer spans only `0/63`→`3/63`
(`4.76%`).  Three native 20-episode BrowserGym/MiniWoW canaries remain `0/20`: the head adapter
repeatedly chooses `type_text` with a `Submit` placeholder, while the pointer child copies a prompt
fragment.  The constrained decoder now gives learned pointer spans precedence over quoted-string
heuristics when that argument is present, making this failure explicit.  No child is promoted and
the native browser gate remains blocked.

### Current MCPMark service-routing control (m274)

The [`m274 receipt`](paper/results/raw/m274-mcpmark-current-service-routing-v1.json) evaluates the
current checkpoint and both m273 children against the pinned public MCPMark standard descriptions
(`169` rows) plus the current checkpoint's easy suite (`70` rows).  The current checkpoint routes
`25/169` standard rows (`14.79%`) and `10/70` easy rows (`14.29%`).  Its service profile is sharply
surface-specific: Playwright routes `25/25`, while filesystem, GitHub, Notion, and Postgres all
route `0` rows; the easy Notion slice is also `0/10`.

The warm m273 child preserves `25/169` routing and the random control falls to `10/169` (`5.92%`),
but neither child improves Notion routing.  This isolates service-family/schema retrieval as a
deployment blocker for the requested Notion and stateful-tool workflows.  MCP servers, isolated
state, verifiers, user simulation, credentials, and external side effects were not run, so this
remains a routing proxy rather than an official MCPMark result or native WebGPU capability claim.

### Current EnterpriseOps-Gym email retrieval control (m275)

The [`m275 receipt`](paper/results/raw/m275-enterpriseopsgym-current-email-retrieval-v1.json)
evaluates `67` public EnterpriseOps-Gym email rows with an average `20`-tool distractor catalog.
The current checkpoint reaches `20.90%` hit@1, `53.73%` hit@3, and `76.12%` hit@5.  The warm m273
child reaches `25.37%`, `55.22%`, and `74.63%`; the matched random-backbone control reaches
`35.82%`, `80.60%`, and `94.03%`.

The counterintuitive random-control advantage is important: the current warm checkpoint has learned
strong local tool-name priors but transfers poorly to unseen EnterpriseOps email schemas.  This is
negative-transfer evidence, not a reason to ship the random child; it means the next training stage
needs an in-domain schema adapter, explicit name/description diversity, and a no-transfer control.
No EnterpriseOps container, SQL verifier, email account, or external side effect was used.

### Workshop gate after current MobileGym receipt (m263)

The [`m263 gate`](paper/results/raw/m263-workshop-gate-current-mobilegym-v1.json) joins the
current m262 native result with the current m261 export, the existing transfer/no-transfer
ablation, the public artifact manifest, and the 40-row realistic-evaluation catalog.  It now
accepts MobileGym as a current official native receipt, and the catalog, weight, and artifact
checks pass as well.  The gate remains `ready: false` with eleven explicit blockers: AndroidWorld,
MobileSafetyBench, iOSWorld, BrowserGym/MiniWob, OSWorld, OSWorld-V2, AgentNet, ToolSandbox,
MCPMark, EnterpriseOps-Gym, and a native WebGPU hardware capability/latency receipt for the
current checkpoint.  This is the current publication decision; no synthetic or older-checkpoint
result is silently substituted for those requirements.

### Current browser-context child Hugging Face-format export (m260)

The [`m260 receipt`](paper/results/raw/m260-hf-local-export-current-browser-context-v1.json)
binds a fresh export of `runs/sft-webgpu-browser-context-adapter-20260802/latest.pt` (SHA-256
`bc1aca…`, `10,524,544` parameters).  The bundle contains `model.safetensors` (`42,101,904`
bytes), a BPE tokenizer (`1,134,224` bytes), `config.json`, the model card, and serialized agent
heads for the full `63`-tool catalog.  Reconstructing `LocalAgentLM` from `config.json` and loading
the safetensors succeeds with `40` tensors and no missing or unexpected keys.  This verifies the
local interchange bundle, not its behavior.

`hf auth whoami` reports no login, so `published`, `uploaded`, and `hub_url` remain false/null.
The receipt deliberately records that the checkpoint's pointer-argument metadata is empty and
that no Hub URL, hosted inference endpoint, native benchmark score, or external side effect is
claimed.  A maintainer must authenticate and choose a repository namespace before this exact
bundle can be uploaded; after upload, the Hub files and revision still need independent hash
verification before calling the model public.

### Stateful productivity GRPO continuation and closed-loop reality check (m276)

The [`m276 GRPO receipt`](paper/results/raw/m276-stateful-productivity-grpo-v1.json) is a fresh,
bounded pure-PyTorch RL simulation from the current 10.52M BPE browser-context checkpoint. It uses
only the repository's deterministic, source-disjoint local email, Notion, browser-search, recovery,
and abstention state machine; no public benchmark task text, emulator, MCP server, or external
account enters training. A 16-step SFT warm-up plus four GRPO rollout steps raises the held-out
local shaped reward from `0.000` to `0.09375`, but exact action/text accuracy remains `0%`.

The [`m276 runtime receipt`](paper/results/raw/m276-stateful-productivity-runtime-v1.json) keeps
the verifier boundary explicit: the oracle reaches `5/5` complete tasks, while the checkpoint
completes only the abstention task (`1/5`, `20%`). Email, Notion, browser search, and recovery all
remain `0/1`; this is stronger than a teacher-forced metric but is still a local simulation, not
AndroidWorld, BrowserGym, MCPMark, EnterpriseOps-Gym, or real-account evidence.

The paired [`m276 weight audit`](paper/results/raw/m276-stateful-productivity-grpo-weight-v1.json)
confirms identical configuration/tokenizer/shapes and frozen deployment heads. Relative backbone
movement is small but non-zero (embedding `1.89%`, attention/mixer `0.87%`, FFN `1.04%`,
normalization `0.035%`). The adoption decision is unchanged: retain the compatible body only as
an initialization candidate, and do not ship this RL child until service-schema grounding and
native closed-loop success improve against a matched no-transfer control.

### Current ToolACE action-history continuation and free-run control (m277)

The [`m277 transfer receipt`](paper/results/raw/m277-toolace-action-history-transfer-v1.json)
extends the current checkpoint audit to the pinned public [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE)
action-history projection. The run uses `256` source-record-disjoint training rows and `64`
held-out rows from revision `6bda777c88d21e5a204703c1ee45597a8fa4f734`, with the tool-response and
non-action prose boundary recorded by the adapter manifest. Warm continuation preserves compatible
weights and lowers held-out mean loss from `4.655` to `4.250`; random-backbone control remains near
chance (`9.781` to `9.474`). Warm held-out token accuracy is `47.30%` versus `0.17%` for random,
but both arms remain at `0%` teacher-forced sequence exactness.

The stricter [`warm free-run receipt`](paper/results/raw/m277-toolace-action-history-warm-v1.json)
finds `20%` tool-name exactness, `0%` argument exactness, and `0%` complete action-history episodes
over 16 held-out rows. The matched [`random free-run receipt`](paper/results/raw/m277-toolace-action-history-random-v1.json)
finds `16.67%` tool-name exactness and `3.33%` step exactness, so the random control wins this tiny
free-run slice despite its much worse teacher-forced loss. This is a deployment warning: backbone
reuse is useful representation-learning evidence, but it is not sufficient to adopt the current
tool heads or claim reliable multi-turn tool execution.

### Current AppWorld native baseline and action-step transfer (m278)

The [`m278 current-checkpoint receipt`](paper/results/raw/m278-appworld-current-checkpoint-native-v1.json)
is the first current-checkpoint AppWorld native baseline in this evaluation chain.  AppWorld's own
contract verifier passes, six public tasks reset independently in isolated databases, and the native
runtime executes.  The model emits no replayable AppWorld action, so native success is `0/6`, with
zero action replays and zero API calls.  The receipt explicitly excludes AppWorld leaderboard,
AppWorld-UL, and real email/SMS/Spotify claims.

The [`m278 public continuation report`](paper/results/raw/m278-appworld-action-step-sft-v1.json)
uses a train-only 24-row first-action projection and a disjoint 12-row dev projection from the
public AppWorld `0.2.0` data release.  Held-out assistant-token accuracy improves `43.28%`→`58.70%`
and mean loss `3.989`→`2.596`, but sequence exactness remains `0/12`; this is teacher-forced
evidence only.  The [`m278 weight report`](paper/results/raw/m278-appworld-action-step-weight-v1.json)
confirms tokenizer/config compatibility, frozen deployment heads, and low-rate shared-backbone
movement (embedding `0.434%`, attention/mixer `0.248%`, FFN `0.306%`, normalization `0.010%`).

The paired [`m278 native action-step receipt`](paper/results/raw/m278-appworld-action-step-native-v1.json)
uses strict one-call AST replay plus schema grounding on the 12 disjoint dev tasks.  The child
still produces no replayable API call: `0/12` native successes, `0` action replays, and `0` native
API calls, despite the teacher-forced gain.  This is a reproducible negative transfer-to-control
result and keeps the AppWorld/native closed-loop requirement blocked; it is not an official
AppWorld score or a substitute for AndroidWorld, OSWorld, BrowserGym, MCPMark, or real-account
evaluation.

### AppWorld route/selector head-only repair and native schema grounding (m279)

The [`m279 head adapter report`](paper/results/raw/m279-appworld-head-adapter-v1.json) keeps the
backbone effectively frozen while training only the route and dense-selector heads on 24 public
AppWorld train rows.  On 12 source-disjoint dev rows, route accuracy and selector top-1 both rise
from `0%` to `100%`.  The [`m279 weight report`](paper/results/raw/m279-appworld-head-weight-v1.json)
confirms zero movement in the shared body and `65.93%` relative movement in the action-head group.

With selector-first inference, the [`m279 native receipt`](paper/results/raw/m279-appworld-head-native-v1.json)
replays `9/12` API calls inside resettable AppWorld fixtures, including `48` total native requests
and `27` credential/bootstrap requests.  None of the 12 task verifiers pass.  This cleanly separates
three requirements: head routing is learnable, native API execution is live, and API/schema grounding
is still missing.  It is not an official AppWorld score, an external-account result, or evidence that
the model is ready for WebGPU productivity control.

### Longer AppWorld action-code SFT and free-run/native controls (m280)

The [`m280 SFT report`](paper/results/raw/m280-appworld-long-sft-v1.json) trains 256 updates on
24 public AppWorld train rows and evaluates 12 disjoint dev rows.  Held-out token accuracy rises to
`96.64%` and teacher-forced sequence exactness reaches `9/12` (`75%`).  However, the unconstrained
[`run_python native receipt`](paper/results/raw/m280-appworld-long-native-runpython-v1.json) emits
no replayable action on the same dev prompts.  This is a direct warning that assistant-token loss
and sequence accuracy do not establish a deployable tool policy.

For a controlled routing comparison, the m280 body is paired with the m279 selector heads.  The
[`combined native receipt`](paper/results/raw/m280-appworld-long-heads-native-v1.json) replays
`12/12` bounded API calls in isolated AppWorld fixtures, but full verifier success remains `0/12`.
The independent [`first-action exactness receipt`](paper/results/raw/m280-appworld-first-action-exactness-v1.json)
finds `0/12` exact translated API-code matches against the public dev projection.  The paired body
and combined weight reports show low-rate body movement (embedding `3.579%`, attention/mixer
`1.405%`, FFN `1.814%`, normalization `0.108%`) and the expected `65.93%` action-head movement.
This closes the experiment's diagnosis: routing and native execution are measurable, but exact API
schema grounding and multi-step stateful success are not yet learned.

### Public-train AppWorld schema retriever control (m281)

The [m281 learned API-head report](paper/results/raw/m281-appworld-api-head-training-v1.json)
is a matched frozen-feature control: it reaches 6/12 held-out API labels and is not promoted.
The [m281 lexical retriever report](paper/results/raw/m281-appworld-api-retriever-v1.json)
uses only public train prompt examples, has learned_weights=false, and reaches 12/12 API-label
accuracy on the disjoint 12-row dev projection.

The existing native runner now accepts this sidecar only to restrict schema candidates and derive
argument fields observed in train traces. The [m281 native receipt](paper/results/raw/m281-appworld-retriever-native-v1.json)
records 12/12 resettable native API calls (48 total requests, 36 bootstrap) and the
[m281 first-action exactness receipt](paper/results/raw/m281-appworld-first-action-exactness-v1.json)
records 12/12 exact translated-code hashes. The independent AppWorld verifiers still report 0/12,
because the adapter executes exactly one action per task; this is first-action schema and execution
evidence, not a complete trajectory, leaderboard result, or external-account claim.

### Current-checkpoint native BrowserGym canary (m282)

The [m282 comparison receipt](paper/results/raw/m282-browsergym-current-checkpoint-canary-v1.json)
executes 20 pinned BrowserGym/MiniWoB test episodes against the current checkpoint.  With the
accessibility-only action path, native success is `0/20`, reward is `0.0`, and only `10/200` steps
are grounded; most outputs are `open_app`, `move_cursor`, or absent tool calls, so they become
`noop(0)` in the browser adapter.

The matched coordinate fallback is a non-official control: it reaches `4/20` by solving only
ascending-numbers (`4/4`), while the other four families remain `0/16`.  This is evidence of a
policy/tool-vocabulary and grounding gap, not evidence of visual-agent quality or a complete
BrowserGym score.  The complete plan is recorded separately below.

### Current-checkpoint official BrowserGym/MiniWoB plan (m282)

The [m282 official-plan receipt](paper/results/raw/m282-browsergym-current-checkpoint-official-v1.json)
executes the complete pinned `240`-episode test plan (`60` task families × `4` fixed seeds) with
`official_split_verified=true`.  The current checkpoint scores `0/240` native successes, `0.0`
reward, and `320/2400` grounded steps; `2080` actions become `noop(0)`.  All 60 task families are
`0/4`, including the email-inbox families.  This is a complete official-split native negative
result, not visual-agent success, WebArena success, or real-account email/Notion access.

### Current export and native WebGPU deployment audit (m283)

The [m283 audit receipt](paper/results/raw/m283-current-export-deployment-audit-v1.json) rebuilds
the current checkpoint's local HF-format bundle and WebGPU bundle.  Reloading the HF weights gives
exact source parity (`max_abs_diff=0`, argmax agreement `1.0`); the ONNX export passes all hard
fp32/fp16 parity checks; and a clean static demo directory verifies the complete manifest-bound
artifact set.

The external publication and native-runtime gates remain open.  Hugging Face authentication is
absent, so `published=false` and no Hub URL is provided.  Both headed and headless Chromium
WebGPU probes fail closed with “no non-empty hardware identity”; this is a host capability
failure, not evidence of model throughput or successful browser/email/Notion control.

### Current workshop/publication gate re-audit (m284)

The [m284 gate receipt](paper/results/raw/m284-workshop-gate-current-v1.json) is the current
fail-closed decision for this checkpoint.  It accepts the catalog coverage, the official-split
MobileGym receipt (`256` tasks, `0.39%` success), the complete official BrowserGym/MiniWoB
receipt (`240` tasks, `0/240` success), the earlier hardware-backed WebGPU receipt, and the
transfer/no-transfer ablation as explicit evidence.  It reports `ready=false`: native receipts
are still missing for AndroidWorld, MobileSafetyBench, iOSWorld, OSWorld, OSWorld-V2, AgentNet,
ToolSandbox, MCPMark, and EnterpriseOps-Gym.

The public-artifact check is intentionally interpreted narrowly.  The verified public manifest
points to the older `28,322,304`-parameter byte model and its Space; it does not publish the
current `10,524,544`-parameter BPE checkpoint.  The m283 local HF/WebGPU export is hash-verified
but unauthenticated and unpublished, and its fresh headed/headless host probes did not expose a
usable adapter identity.  Therefore no workshop or public-deployment approval is claimed.

### Anonymous public Hub release audit (m305)

The [m305 receipt](paper/results/raw/m305-public-hf-legacy-current-audit-v1.json) independently
fetches the public model and Space without credentials.  Both endpoints return HTTP `200` at
revisions `d15db7c…` and `3c07f7d…`; the downloaded legacy model weight is
`a0043408…c1d0c2` and the Space's `model.fp16.onnx` is `59f06336…5d3d34c`.  The remote model
config exposes no checkpoint binding, so `current_checkpoint_match=false` against the current
`bc1aca20…c16361` BPE checkpoint.  This confirms the public URLs are live but still legacy; the
current model/demo upload and a matching manifest remain outstanding.

### Current parent → stateful RL weight audit (m306)

The [m306 receipt](paper/results/raw/m306-current-parent-stateful-rl-weight-transfer-v1.json)
compares the current `bc1aca20…c16361` parent with the m276 stateful-productivity RL child at the
tensor level. All `51` shared tensors have compatible shapes/configuration and the tokenizer hash
matches. The child leaves action heads unchanged (`0%` movement) but moves the shared embedding,
attention/mixer, and FFN groups by `1.89%`, `0.87%`, and `1.04%`; this is materially larger than
the earlier low-rate transfer controls. The adoption decision remains conservative: reuse the
compatible parent initialization, use a smaller body learning rate and larger task-head rate, and
retain a matched no-transfer arm before any deployment promotion. This is weight-lineage evidence,
not a native email, Notion, browser, or MCP success claim.

The source audit also binds OSWorld 2.0 to the upstream `osworld-v2-2026.06.24` release: code,
task classes, gated assets, and mocked websites must come from the same release.  The required
gated task/asset snapshots and a desktop VM are not present locally, so OSWorld-V2 remains a
receipt blocker rather than a text-projection score.

### AndroidControl mobile dispatch transfer and native MobileGym canary (m292)

The [m292 receipt](paper/results/raw/m292-mobile-dispatch-transfer-native-v1.json) adds a
source-disjoint mobile dispatch experiment using the public AndroidControl mirror: `4,096`
train rows and a separate balanced `904`-row test file covering click, text input, long press,
navigate-back, navigate-home, open-app, scroll, and wait.  The mirror is explicitly text-first;
screenshots were not loaded, so this is action/dispatch evidence rather than visual grounding.

With the backbone frozen, warm route/selector continuation reaches `100%` route accuracy and
`42.81%` selector top-1 on the held-out rows.  The matched random-head control reaches the same
synthetic route score and `38.94%` selector top-1.  Warm gains are uneven: open-app is `89.6%`,
navigate-back `58.4%`, wait `59.2%`, scroll `50.4%`, input `44.0%`, click `8.0%`, and both
long-press and navigate-home are `0%`.  This is a modest parent-geometry signal, not evidence
that the model grounds screen state.

The warm child is then executed in the pinned MobileGym simulator (`093a329…`) on a 20-task
official-test canary with the independent state-diff judge.  Native success is `0/20`; the child
emits `mobile_press_enter` on 19 episodes and `mobile_submit_answer` on one.  The parent’s first
20 tasks in the complete m262 official run were also `0/20` (with 19 `mobile_press_enter` and one
`mobile_input_text`; its run used a two-step limit), so the dispatch transfer does not improve
closed-loop mobile behavior.  The child is not promoted, and no Android emulator, screenshot, or
real-device claim is made.

### Selector-first MobileGym control ablation (m293)

The [m293 receipt](paper/results/raw/m293-mobilegym-selector-first-canary-v1.json) adds an
implementation control to the m292 child: the runner can now emit the top learned selector
candidate directly, without letting the language model re-rank it.  This mode is opt-in and does
not change the default runner.  On the same pinned 20-task MobileGym canary, selector-first also
scores `0/20` and emits the identical `19` `mobile_press_enter` plus one `mobile_submit_answer`
pattern.  The negative control localizes the failure to selector generalization on MobileGym
task-state prompts, rather than only to candidate-body ranking; the mode is not promoted as a
default policy.

### Current-checkpoint native ToolSandbox smoke and interactive control (m297)

The [m297 receipt](paper/results/raw/m297-toolsandbox-current-native-smoke-v1.json) executes the
current 10.52M checkpoint inside the pinned Apple ToolSandbox simulator and milestone verifier.
The bounded single-step smoke passes `2/3` scenarios: `cellular_off` and `wifi_off` reach exact
milestones, while `send_message_with_phone_number_and_content` reaches only `0.425` milestone
similarity.  No external API or account is touched.

The interactive scripted-user control (four agent turns) falls to `0/3`: the two settings tasks
reach `0.5` milestone similarity and the message task reaches `0.0`.  This is genuine native
simulator/verifier evidence, but the official split and model-based user simulator were not run,
so it does not satisfy the ToolSandbox workshop gate or establish MCP/email/Notion readiness.

The [m298 gate re-audit](paper/results/raw/m298-workshop-gate-current-toolsandbox-v1.json) includes
this receipt explicitly and remains `ready=false`; ToolSandbox is now blocked for the precise reason
`official_split_not_verified`, while the other missing native benchmark receipts remain unchanged.
