# Realistic agent evaluation research memo

Status: protocol refresh on 2026-08-08. This memo records the public benchmark methods that
matter for a sub-100M text-first WebGPU agent. It is a source and protocol guide, not a claim that
the repository has completed every benchmark.

Official source refresh (2026-08-08): [AndroidWorld](https://github.com/google-research/android_world)
documents a live Android-emulator benchmark with 116 hand-crafted tasks across 20 apps and durable
rewards; [BrowserGym](https://github.com/ServiceNow/BrowserGym) exposes MiniWoB, WebArena,
WorkArena, VisualWebArena, WebLINX, OpenApps, and TimeWarp through Gymnasium/Playwright;
[MCPMark](https://github.com/eval-sys/mcpmark) evaluates isolated Notion, GitHub, filesystem,
Postgres, and Playwright MCP services with strict verification and pass@k aggregation; and
[EnterpriseOps-Gym](https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym) is the public
HF source for enterprise tool/SQL-verifier workflows. These links are protocol references only:
the catalog continues to keep benchmark tasks, credentials, emulator/VM assets, and verifier state
out of SFT and WebGPU bundles.

The current checkpoint-bound BrowserGym finding is a useful modality boundary.  The accessibility-
only MiniWoW canary remains `0/1` because SVG number controls expose no actionable accessibility
roles; a routing guard now selects `click`, but the adapter correctly fails closed when it cannot
ground a target.  A separate non-official canary with a live DOM-geometry sidecar reaches `1/1`
on the same pinned ascending-numbers task.  The receipt is [m521](paper/results/raw/m521-browsergym-current-coordinate-semantic-canary-v1.json):
it validates the WebGPU deployment bridge, not visual-agent competence, WebArena, or real email/
Notion control.  Keep the official score and the coordinate diagnostic separate in all reports.

The first reproducible current-checkpoint warm/random continuation after that audit is [m522](paper/results/raw/m522-realistic-cross-surface-warm-random-2step-comparison-v1.json).
It uses only source-linked public train projections from AndroidControl, OpenCUA-AgentNet,
Mind2Web-train, and MCPMark filesystem, with source-local parent/slot-disjoint rows.  On the
bounded 8-row train / 4-row eval canary, the warm arm improves held-out assistant-token accuracy
from `39.13%` to `40.22%` while the matched random arm stays at `0%`; exact sequence accuracy is
`0%` for both.  Warm shared-body movement is below `0.03%` relative L2, versus `77.9–119.7%`
for random initialization.  This supports warm-start adoption as a representation-transfer
candidate, not as a native benchmark or live email/Notion/browser result.
The refreshed [m523 workshop gate](paper/results/raw/m523-workshop-gate-current-m522-v2.json)
accepts this warm/random report as a structurally valid current-checkpoint weight ablation.  The
gate remains fail-closed because native AndroidWorld/MobileSafetyBench/iOSWorld/OSWorld/AgentNet/
ToolSandbox/MCPMark/EnterpriseOps-Gym receipts, a checkpoint-bound RL preflight, and authenticated
public artifacts are still absent.
The current [m524 native WebGPU receipt](paper/results/raw/m524-webgpu-current-bundle-rerun-v1.json)
replays the exact local bundle on an Apple Metal-3 adapter: 3/3 structured tool actions are exact,
p50 latency is `19.95 ms`, p50 throughput is `674.7` input tokens/s, and conservative peak memory
is `20.46 MB`.  The action graph never touches a real account or external page, so this is a
hardware/deployment capability result, not email, browser, or Notion task completion.

The expanded [m526 continuation audit](paper/results/raw/m526-realistic-cross-surface-warm-random-8step-comparison-v1.json)
reacquires public AndroidControl, OpenCUA-AgentNet, Mind2Web-train, and MCPMark trajectory-log
projections. MCPMark covers filesystem, Notion, GitHub, Playwright, and Postgres traces; tool
outputs and assistant prose are redacted, and train/eval files are source-disjoint. Across 16
training rows and 8 held-out rows, the warm arm reaches `50.92%` assistant-token accuracy versus
`0%` for the matched random arm, with warm ahead on AndroidControl (`71.70%`), AgentNet (`50.00%`),
Mind2Web (`73.08%`), and MCPMark (`27.17%`). Exact sequence accuracy remains `0%`; this is
representation-transfer evidence only, not native computer-use or live email/Notion success.
The [m528 gate](paper/results/raw/m528-workshop-gate-current-m526-v1.json) records the expanded
ablation as a pass, while the overall publication gate remains fail-closed for missing native
benchmark receipts and authenticated public artifacts.

The corrected [m529 gate](paper/results/raw/m529-workshop-gate-current-m526-corrected-v1.json)
also binds the official 240-episode BrowserGym/MiniWoB receipt under its canonical benchmark ID;
the earlier gate had supplied the same receipt under a shorthand alias. The larger [m530 transfer
audit](paper/results/raw/m530-realistic-cross-surface-warm-random-64step-comparison-v1.json)
uses 58 public training rows and 29 source-disjoint held-out rows across the four surfaces. Warm
initialization reaches `55.03%` held-out assistant-token accuracy versus `24.58%` for the matched
random arm, and leads on every surface: AndroidControl `66.82%`, AgentNet `63.32%`, Mind2Web
`77.14%`, and MCPMark `24.85%`. Exact sequence accuracy remains `0%`; shared-body movement is
`0.80%` or less for warm versus `119.7%` for random. The [m531 gate](paper/results/raw/m531-workshop-gate-current-m530-v1.json)
binds this stronger ablation but remains fail-closed for the same missing native surfaces and
public Hub manifest.

The public MCPMark slice is now reproducible with
[download_mcpmark_trajectory.py](../scripts/download_mcpmark_trajectory.py). At the pinned
trajectory-log revision it selects sorted, surface-balanced paths (two train and one held-out file
per filesystem, Notion, GitHub, Playwright, and Postgres surface), downloads only those
messages.json files, and emits the [m532 acquisition manifest](paper/results/raw/m532-mcpmark-trajectory-acquisition-v1.json).
The manifest contains 15 file identities and hash
14b897092a9ba10d66f418b9c291c880a1ab6f59b4fe4992ffa7fbb3b4cf648c; normalization remains explicit
and redacts tool output and assistant prose before training.
The [m533 reproducibility receipt](paper/results/raw/m533-mcpmark-acquisition-normalization-reproducibility-v1.json)
re-runs that manifest through the normalizer and reproduces the exact train/eval output hashes
used by the 64-step audit.

The [m534 official-source audit](paper/results/raw/m534-official-realistic-source-integrity-audit-v1.json)
rechecks the release contracts for AndroidWorld, BrowserGym, MCPMark, OSWorld, OSWorld-V2,
ToolSandbox, and EnterpriseOps-Gym. It records current public URLs, pinned revisions where
available, runtime requirements, and the explicit WebGPU claim boundary; it deliberately does not
convert source documentation into benchmark scores.

The current-checkpoint [m535 ToolSandbox smoke](paper/results/raw/m535-toolsandbox-native-current-v1.json)
now boots the pinned `165848b9a78cead7ca7fe7c89c688b58e6501219` simulator in an isolated Python
environment. All three bounded single-step scenarios pass the upstream milestone verifier (`3/3`,
similarity `1.0`) without external API calls. This is a runtime-integrity result only: the scripted
user, full scenario matrix, official split, model-based user simulator, and optional RapidAPI tools
were not run, so the [m536 gate](paper/results/raw/m536-workshop-gate-current-m535-v1.json) correctly
keeps `native:toolsandbox` blocked on `official_split_not_verified`.

The [m537 release-preparation receipt](paper/results/raw/m537-local-hf-webgpu-release-prepare-v1.json)
regenerates the exact current 10,524,544-parameter checkpoint bundle and static Space staging
directory. All four ONNX graphs pass the hard PyTorch parity gate, the 63-tool dispatch surface is
bound, and the WebGPU bundle identity is `ff0259b3f86c08de56533a32bd3db61783a8077e8090cb84e2bca6393258fc00`.
This closes local packaging reproducibility, but publication remains intentionally false until a
maintainer supplies HF write authentication and the anonymous post-upload checkpoint audit passes.

The [m538 warm candidate audit](paper/results/raw/m538-warm-realistic-candidate-webgpu-toolsandbox-v1.json)
keeps the stronger 64-step warm continuation as a deployable candidate rather than silently
promoting it: its source-disjoint held-out assistant-token accuracy is `55.03%`, the regenerated
WebGPU action graph reaches `3/3` exact local probes at `1,311.5` input tok/s p50 on Apple Metal-3,
and the pinned ToolSandbox smoke reaches `3/3` milestone matches. The candidate still has zero
closed-loop side effects by design and no official ToolSandbox split, so it is evidence for transfer
and packaging—not a workshop-ready public release.
The same child also passes the resettable local WebGPU trajectory suite: `13/13` exact actions,
`13/13` state transitions, and `3/3` complete Gmail, Notion, and browser trajectories at pass@1.
Those are in-memory fixtures, not real accounts or official mobile/browser benchmarks.
The candidate's [m539 RL preflight](paper/results/raw/m539-warm-realistic-candidate-rl-preflight-v1.json)
also passes the one-update contract: two realized optimizer updates, nonzero learning rate,
40 changed policy tensors, disjoint train/eval rows, and reward `0.00→0.10625`. Exact held-out
tool match remains `0%`, so this validates RL plumbing and weight movement rather than capability.

The [m540 head-preserved RL promotion audit](paper/results/raw/m540-head-preserved-rl-webgpu-promotion-audit-v1.json)
exposes an important deployment constraint. A raw RL checkpoint invalidates the structured action
heads and cannot be exported to WebGPU (`KeyError: 'tool_head'`). The head-preserving recipe copies
the five frozen deployment heads from the SFT parent, keeps body movement below `0.05%` relative
L2, exports successfully, and retains `13/13` local trajectory actions plus `3/3` WebGPU probes.
The raw child is rejected; only the head-preserved child remains a candidate.
The [m541 BrowserGym canary](paper/results/raw/m541-head-preserved-rl-browsergym-canary-v1.json)
then executes the pinned BrowserGym `0.14.3`/MiniWoB runtime without coordinate or semantic
fallbacks. All 16 bounded episodes fail (`0/16`), exposing the remaining accessibility grounding
gap; because 224 planned episodes were not run, this is deliberately not an official split score.

The [m542 grounding canary](paper/results/raw/m542-browsergym-realistic-pool-grounding-canary-v1.json)
first fixes a protocol bug rather than changing weights: BrowserGym's instruction suffix was being
treated as the target, and an empty accessibility name could match every element. With the
lexical guard ordered before route-head abstention and those parser fixes, the same m540 child moves
from `0/16` to `4/16` bounded episodes, with four grounded clicks and all successes confined to
`miniwob.click-button`. This is a genuine closed-loop improvement, but only `4/124` steps are
grounded and the full 240-episode split remains unrun.

The [m544 DOM-enriched continuation](paper/results/raw/m544-grounded-mind2web-webgpu-browsergym-v1.json)
uses the public Mind2Web train/eval records with deterministic, capped candidate snapshots derived
from the source HTML. On 186 disjoint held-out decisions, exact pointer spans improve from `0/186`
to `13/186` (`6.99%`), while the 10.52M child passes all four ONNX/PyTorch parity graphs. Native
BrowserGym remains `4/16` on the same diagnostic, so this is evidence that DOM grounding is the
right training interface—not evidence of general browser competence. The child is rejected for
promotion until multi-step task families and a complete official evaluation are available.

The [m545 long continuation](paper/results/raw/m545-grounded-mind2web-128step-transfer-v1.json)
repeats the same source-disjoint DOM protocol for 128 updates. Held-out exact pointer spans rise
to `80/186` (`43.01%`) and WebGPU parity remains exact, but the native diagnostic stays at `4/16`
(`4/124` grounded steps), with no success outside `miniwob.click-button`. This is a useful
weight-transfer result—the backbone moves only `0.76%` in embedding L2 and `0.42%` in attention/
mixer L2—but it is also a decisive transfer boundary: offline DOM copying has not yet become
multi-step browser competence, so this child is not promoted.

The [m546 multi-surface continuation](paper/results/raw/m546-multisurface-public-transfer-webgpu-v1.json)
then continues that parent on source-disjoint AndroidControl and AgentNet projections. Held-out
assistant-token accuracy improves `58.72%→69.27%`, route accuracy is `100%`, selector top-1 is
`70.04%`, and all 51 tensors remain shape-compatible; embedding/attention movement is only
`0.435%/0.186%`. The exact 10.52M child exports with all four ONNX parity graphs and a native
Apple Metal-3 WebGPU probe at `1,290` input tokens/s p50, `7.45 ms` p50, and `20.46 MB`
conservative memory. These are teacher-forced and local-dispatch measurements: sequence exactness
is `0%`, no external side effect ran, and native Android/desktop replay is still required.

The [m547 native MobileGym receipt](paper/results/raw/m547-m546-mobilegym-native-full-v1.json)
finally binds the current m546 checkpoint to the complete pinned MobileGym test split. All 256
official test tasks execute with zero runner errors and the upstream state-diff judge, but only
`1/256` passes (`0.39%`). The model emits `255` `mobile_input_text` actions, confirming that the
text-projection policy is not yet a general mobile controller; this is a required native negative
control, not a visual Android score or a reason to promote the checkpoint.

The [m550 native BrowserGym receipt](paper/results/raw/m550-m546-browsergym-native-full-v1.json)
now binds the same m546 checkpoint to the complete pinned BrowserGym 0.14.3/MiniWoB++ split:
`240/240` episodes, `60` task variants, four fixed seeds, and zero action errors. The model passes
`5/240` tasks (`2.08%`) with `211` grounded actions; four successes are `miniwob.click-button` and
one is `miniwob.sign-agreement`. This is a real accessibility-tree environment run with no
coordinate or semantic fallback, but it uses no screenshots and does not establish visual browser
control, WebArena competence, or real-account email/Notion execution. The result is retained as a
native negative control and the m546 checkpoint is not promoted.

### Current publication audit (2026-08-09)

The fail-closed gate was rerun against the current m546 checkpoint with the full 256-task MobileGym
receipt, the full 240-episode BrowserGym/MiniWoB receipt, and the hardware WebGPU capability receipt.
MobileGym, BrowserGym, WebGPU, and catalog checks pass. The gate remains `ready=false`: nine other
native receipts are absent (AndroidWorld, MobileSafetyBench, iOSWorld, OSWorld, OSWorld-V2, AgentNet,
ToolSandbox, MCPMark, and EnterpriseOps-Gym), the required warm/random transfer-and-no-transfer
ablation is not bound to m546, the current m546 RL preflight failed closed because the diagnostic
child lacks lineage metadata, and no public Hub/Space manifest is authenticated or bound to this
checkpoint. This is the exact workshop blocker set, not a model-quality estimate.

The attempted m549 RL preflight receipt (captured outside the repository at
`/private/tmp/m549-m546-rl-preflight.json`) was intentionally fail-closed before any update: the
m546 diagnostic checkpoint has no lineage metadata,
so it cannot be accepted as the parent of the strict stateful-RL protocol. No lineage was fabricated,
and no production checkpoint was modified. A fresh lineage-preserving continuation is required
before RL can satisfy the publication gate.

The [m551 gate receipt](paper/results/raw/m551-workshop-gate-current-m546-v1.json) records this
decision in a compact, hash-bound form: `ready=false` despite valid current MobileGym, BrowserGym,
WebGPU, and catalog checks. This prevents the strong local throughput result (`1,290` input tok/s
p50 on Apple Metal-3) and the two native text-environment scores from being mistaken for a complete
workshop submission.

Repository verification is also explicit: `ruff check src tests` passes and the full suite reports
`2164 passed`; two unrelated fixture tests remain environment-blocked because their private
`/private/tmp` AppWorld output and pinned MobileWorld checkout are not present. The focused
realistic/WebGPU/RL/release suite passes `38/38`.

## What the public benchmarks actually measure

| Surface | Public source and method | What LocalAgent may claim locally |
|---|---|---|
| Mobile UI | [AndroidWorld](https://github.com/google-research/android_world) runs resettable Android emulator tasks with accessibility/screenshot observations and durable task rewards. | Accessibility-tree/text protocol tests are useful for routing; an Android emulator, ADB, official task set, and reward logs are required for a native score. |
| Personalized mobile UI | [iOSWorld](https://iosworld.io/) provides 133 tasks over 26 interconnected iOS apps with persistent seeded user identity and an optional MCP server. | Treat identity, cross-app state, and MCP-vs-GUI as separate axes; a WebGPU text projection cannot claim native iOS control or personalization. |
| Cross-app mobile assistant | [MobileWorld](https://github.com/Tongyi-MAI/MobileWorld) defines 201 long-horizon tasks over 20 Android apps, including agent-user interaction and MCP-augmented workflows, with screenshot/accessibility/backend-state observations and deterministic verification. | Freeze the source revision, rooted AVD/Docker image, credentials, and task manifest. macOS lacks the upstream Linux/KVM runtime; source parsing is provenance only, not native mobile/email/browser success. |
| Mobile safety | [MobileSafetyBench](https://mobilesafetybench.github.io/) evaluates Android-device safety, harmful side effects, and indirect prompt injection in messaging/banking-style tasks. | Run a dedicated refusal/confirmation/safe-side-effect gate before enabling email, messaging, settings, or payment tools. |
| Visual prompt-injection safety | [VPI-Bench](https://github.com/cua-framework/agents) reports 306 dynamic cases across Amazon, Booking, BBC, Messenger, and Email, separating attempted from successful malicious actions. | Keep attack pages and traces evaluation-only; a text-only WebGPU model can measure refusal/confirmation policy but cannot claim visual robustness without a vision-capable runner. |
| Contextual-integrity safety | [AgentCIBench](https://github.com/UKPLab/arxiv2026-agentcibench) uses state/prompt/recipient/`must_share`/`must_not_share` scenarios and engagement-conditioned leakage. | Add disclosure restraint and consent metrics alongside task success; generated scenario pools and judge configuration must be release-pinned. |
| Curated mobile control | [AndroidControl-Curated](https://github.com/batechworks/AndroidControl_Curated) purifies AndroidControl task ambiguity and reports a matched curated split. | Compare original and curated tasks with identical identities/seeds; do not mix the curated benchmark's evaluation rows into SFT. |
| Dynamic mobile device control | [AndroidLab](https://github.com/MadeAgents/mobile-use/tree/main/benchmark/android_lab) is an emulator/ADB benchmark surfaced by MobileUse, with patched Clock/Settings tasks and a release-dependent AndroidLab submodule. | Freeze the submodule revision, task manifest, Docker/AVD image, and license before using it as a score; a text-only WebGPU projection cannot claim screenshot or emulator success. |
| Personalized/proactive mobile | [KnowU-Bench](https://github.com/ZJU-REAL/KnowU-Bench) reports 192 registered tasks over 23 apps, with hidden profiles, exposed behavioral logs, and an online user simulator for clarification and consent. | Treat preference acquisition, intervention calibration, and consent as separate metrics; Docker/KVM and API-backed user simulation are required for a native result. |
| Smartphone agent benchmark | [AppAgent](https://github.com/TencentQQGYLab/AppAgent) releases the smartphone evaluation benchmark used by its tap/swipe agent and supports Android emulators or devices. | Pin the benchmark release, APK hashes, device image, and exact task manifest; text-only action vocabulary is not smartphone task success. |
| Browser | [BrowserGym](https://github.com/ServiceNow/BrowserGym) exposes MiniWoB, WebArena, WorkArena, VisualWebArena, WebLINX, and related environments through Gymnasium/Playwright. | DOM/accessibility grounding can be tested in the existing MiniWoB runner; screenshot or live multi-site claims require the matching runtime and task release. |
| Real web trajectories | [Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web) contains more than 2,000 tasks across 137 websites and 31 domains, with public training data and protected test splits. | Train only on the public train partition; keep test tasks and canary strings out of all corpora. Report DOM/action replay separately from native live-web success. |
| Realistic live-site browser workflows | [WebBench](https://github.com/Halluminate/WebBench) covers 2,454 READ/CREATE/UPDATE/DELETE/file tasks across 452 live websites, including authentication, forms, and downloads. | Keep the live task set and credentials evaluation-only; use DOM/action safety canaries or a vendor-approved resettable environment rather than training on the tasks. |
| Contamination-resistant browser canary | [BU Bench V1](https://github.com/browser-use/benchmark) contains 100 encrypted tasks drawn from WebBench, Mind2Web 2, BrowseComp, GAIA, and custom challenges. | Preserve encryption and task text out of SFT and published artifacts; only run a release-matched browser evaluation with decrypted data held locally. |
| Desktop computer use | [OSWorld](https://github.com/xlang-ai/OSWorld) uses real desktop VMs and execution-based evaluators; [OSWorld 2.0](https://osworld-v2.xlang.ai/) adds 108 long-horizon workflows with dynamic environments, cross-source reasoning, implicit state, and visual precision. | A text/accessibility projection is a diagnostic only. A publication score needs the release-matched VM, assets, initial state, action log, and evaluator. |
| Screenshot/action trajectories | [OpenCUA AgentNet](https://github.com/xlang-ai/OpenCUA) provides cross-OS computer-use trajectories; AgentNetBench is an offline representative task suite. | The m47–m62 projection measures action priors and text routing only. Visual grounding requires the embedded images and the upstream AgentNetBench evaluator. |
| Desktop grounding | [GroundCUA](https://github.com/ServiceNow/GroundCUA) publishes 56K screenshots and 3.56M+ human-verified element annotations across 87 desktop applications. | Keep screenshot/box training and evaluation separate from this text-first model; any use requires dataset-term review and a visual encoder. |
| GUI action protocol | [UI-TARS](https://github.com/bytedance/UI-TARS) documents separate computer-use, mobile-use, and grounding contracts, including long-press, app launch, home, and back actions. | Reuse only the schema/action vocabulary for compatibility probes; coordinate or screenshot claims require a vision-capable model and release-matched runtime. |
| GUI plus MCP control | [OSWorld-MCP](https://github.com/X-PLUG/OSWorld-MCP) measures GUI actions, MCP invocation, and decisions together across 158 validated tools and seven desktop applications. | Report tool-invocation rate separately from GUI completion; a schema-routing probe is not an OSWorld-MCP score. |
| Stateful local tools | [Apple ToolSandbox](https://github.com/apple/ToolSandbox) evaluates stateful, conversational tool execution with a user simulator and milestone DAGs, including state dependency, canonicalization, and insufficient-information cases. | Static AST rows can train schemas and measure constrained dispatch; only the simulator plus milestone verifier can establish ToolSandbox success. |
| Stateful MCP services | [MCPMark](https://github.com/eval-sys/mcpmark) runs isolated Notion, GitHub, filesystem, Postgres, and Playwright services with strict verification. Its current repository identifies MCPMark Verified as the default task set. | Tool/schema retrieval and local state contracts are preflight evidence. Native claims require the matching verified release, isolated services, verifier output, and pass@k aggregation. |
| Enterprise email/tools | [EnterpriseOps-Gym](https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym) exposes large tool catalogs and stateful SQL-verifier workflows across enterprise domains. | Name/schema retrieval is a useful failure profile; task success requires the containerized MCP servers and SQL verifiers. |
| Verifiable computer-use training | [CUA-Gym](https://huggingface.co/datasets/xlangai/CUA-Gym) publishes a CC-BY-4.0 task table with 10,910 desktop/web/cross-app instructions and executable setup/reward artifacts. | Its metadata table has one `train` split and no official held-out evaluation split; use it for coverage and sandboxed RLVR only after task-identity holdout, artifact review, and a disposable runtime. The current receipt intentionally consumes metadata only. |
| Public desktop trajectory archives | [OSWorld 2.0 trajectories](https://huggingface.co/datasets/xlangai/osworld2.0-trajectory) and [OSWorld-Verified trajectories](https://huggingface.co/datasets/xlangai/ubuntu_osworld_verified_trajs) provide large model-run packages. | They are evaluation/provenance sources until task identity and split leakage are resolved; archive screenshots and verifier outputs are not silently admitted to WebGPU SFT, and native OSWorld evidence still needs the release-matched VM. |

### Official-source audit and admission boundary (m208)

The [`m208 receipt`](paper/results/raw/m208-realistic-evaluation-source-audit-v1.json) freezes the
historical research inventory against the 40-row canonical catalog and 24-row supplemental registry.
It cross-checks the official contracts for AndroidWorld, AndroidControl/AITW, MobileSafetyBench,
iOSWorld, AppAgent, BrowserGym, Mind2Web, WebBench/BU Bench, OSWorld, AgentNet, GroundCUA/UI-TARS,
ToolSandbox, MCPMark, EnterpriseOps-Gym, and MobileGym.  The audit makes the deployment boundary
explicit: only AndroidControl, AITW, xLAM function calling, and the public Mind2Web train partition
are currently train-eligible; benchmark task text, screenshots, emulator/VM assets, credentials,
MCP state, and verifier outputs remain evaluation-only.

This matters for the requested email, Notion, browser, and computer-use scenarios.  MCPMark and
EnterpriseOps-Gym require isolated service state and independent verification; iOSWorld and
MobileSafetyBench require seeded mobile runtimes plus safety/personalization metrics; BrowserGym and
OSWorld require release-matched browser/VM execution.  A compact WebGPU text projection may measure
route, retrieval, grounding, schema validity, abstention, and action history, but the receipt does
not convert those proxies into native or leaderboard scores.

### Matched multi-surface continuation and native bridge (m209–m211)

The [`m211 receipt`](paper/results/raw/m211-multisurface-continuation-native-bridge-v1.json) trains
one matched 64-step continuation over 4,752 public rows from AndroidControl, AgentNet, Mind2Web,
and the ToolSandbox AST projection, with 1,069 source-disjoint held-out rows.  The warm child rises
from `64.31%` to `72.30%` aggregate assistant-token accuracy, while the random-backbone control
rises from effectively `0%` to `46.55%`.  Warm movement remains small (embedding `0.97%`,
mixer `0.29%`, FFN `0.38%`, normalization `0.02%`) and the action heads move `0%`; exact sequence
accuracy remains `0%` for both arms.  The gains are therefore representation-transfer evidence,
not complete tool policy learning.

The native bridge is unchanged: both children complete `1/5` bounded interactive ToolSandbox
scenarios.  Random improves one ambiguous-contact similarity from `0.0` to `0.3333`, but neither
arm gains a new fully verified task.  The warm child is retained as a compatibility diagnostic,
not exported or adopted.  Reliable WebGPU email/Notion/browser control still requires jointly
training the runtime-aligned route, candidate selector, argument grounding, and state/action-history
policy, followed by native closed-loop verification.

The m156 follow-up probe uses only CUA-Gym's public instruction field and its metadata `platform`
label (`desktop`, `web`, or `cross_app`) on a deterministic task-ID holdout.  A frozen m142 warm
checkpoint reaches 81.37% surface accuracy versus 77.87% for the matched random control, with
exactly zero backbone movement because only a new linear probe is trained.  This is evidence that
the warm representation carries a small amount of broad deployment-surface signal; it is not an
action label, task-success, screenshot-grounding, or native browser/desktop result.  See the
[`m156 receipt`](paper/results/raw/m156-cua-gym-surface-probe-v1.json) and reproducible
[`train_cua_gym_surface_probe.py`](../scripts/train_cua_gym_surface_probe.py).

The m157 browser continuation uses the public [Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web)
train split, enriches each action with a bounded DOM-candidate snapshot, and holds out three whole
parent records plus typed slot values.  After 64 matched updates, the warm pointer head reaches
15/63 exact spans versus 0/63 for the matched random body; both begin at 0/63.  The warm body moves
0.90% in embeddings, 0.50% in attention/mixer, and 0.55% in FFN, while the pointer/action heads
move 46.40%.  The pointer vocabulary grows from 17 to 19 rows by design.  This is a small
in-source action-replay diagnostic—not the official Mind2Web test score or native BrowserGym,
email, Notion, MCP, or browser-account control.  See the [`m157 receipt`](paper/results/raw/m157-mind2web-grounded-transfer-v1.json).

The m158 mobile continuation returns to the current m142 parent and uses a public AndroidControl/AITW
train projection with a disjoint 904-row evaluation file.  Focused oversampling produces an offline
selector score of `41.26%` top-1 with `100%` routing and `25%` exact pointer spans, while the inherited
body is unchanged and only action heads move substantially (`124.77%` relative ΔL2).  A pinned
MobileGym first-20-task canary reaches `1/20` (`5%`), exactly matching the parent same-range result.
This cleanly separates offline action-format learning from state-conditioned simulator transfer;
it is not the full MobileGym score, Android emulator/screenshot grounding, or a browser/email/Notion
side-effect claim.  See the [`m158 receipt`](paper/results/raw/m158-mobile-dispatch-transfer-v1.json).

The m159 MCPMark continuation repeats the public trajectory-log comparison from the current m142
parent.  Eight redacted filesystem, Notion, GitHub, Postgres, and Playwright rows train and two
parent-disjoint Playwright rows remain held out.  Warm initialization improves held-out assistant
token accuracy from `39.38%` to `43.18%`; a matched random backbone reaches only `3.12%` after the
same 32 updates, a `+40.06` point gap.  Both arms remain at `0%` exact multi-turn sequence
accuracy.  Warm movement stays below `0.5%` in each inherited backbone group, supporting the
low-rate body/high-rate adapter recipe while rejecting any claim of native MCP, email, Notion, or
browser side effects.  See the [`m159 receipt`](paper/results/raw/m159-mcpmark-current-parent-transfer-v1.json).

The recurring methodological point is that a benchmark is not just a prompt list. The authoritative
score includes the observation contract, reset state, action interface, environment revision, and
verifier. A static conversation projection must be labelled as such even when it uses the original
task text.

### MobileGym and OSWorld-V2 source audit (m121)

The [`m121 source-audit receipt`](paper/results/raw/m121-mobilegym-osworld-source-audit-v1.json)
hash-pins two public benchmark contracts without converting either into training data.  The pinned
[MobileGym](https://github.com/Purewhiter/mobilegym) revision describes 28 simulated mobile apps,
416 parameterized templates, and a 256-task held-out test split (160 train templates by subtraction),
with structured state snapshots, tap/type/swipe-style actions, deterministic state-diff judges, and
an AnswerSheet protocol.  Its code is Apache-2.0, but the companion benchmark data is CC-BY-NC-4.0;
the receipt therefore records the release pointer while leaving the approximately 1.9 GB archive
out of the repository and out of SFT.

The same receipt binds the active [OSWorld-V2](https://github.com/xlang-ai/OSWorld-V2) release
`osworld-v2-2026.06.24`, its 108-task hash manifest, gated task and asset datasets, website/code
tags, and provider image digest.  The gated task implementations, desktop VM, action logs, and
verifier were not acquired here, so this is provenance and protocol evidence—not an OSWorld-V2
score, desktop-control result, or justification for publishing benchmark-derived task text.

### Mobile/GUI source audit (m163)

The [`m163 source-audit receipt`](paper/results/raw/m163-mobile-grounding-source-audit-v1.json)
binds current source revisions and aggregate contracts for KnowU-Bench, AppAgent, GroundCUA, and
UI-TARS.  The audit keeps their task text, hidden profiles, screenshots, annotations, APKs, and
model traces out of the repository.  KnowU-Bench is the most important addition for realistic
personal assistance: preference inference, clarification, proactive intervention, consent, and
silence are evaluated separately from ordinary GUI execution.  AppAgent supplies a small
45-task/9-app smartphone gate, while GroundCUA is a vision-only grounding resource and UI-TARS
supplies a useful mobile/desktop action vocabulary.

This is deliberately stronger provenance but not a capability result: the receipt has no native
emulator/desktop runs, no WebGPU run, no training rows, and no official scores.  For the current
text-first model, use only compact state/action contract projections; screenshot grounding,
hidden-profile reasoning, and APK interaction require their release-matched runtimes and
additional model inputs.

### MobileWorld source and runtime preflight (m302)

The [`m302 receipt`](paper/results/raw/m302-mobileworld-source-runtime-audit-v1.json) pins the
public [MobileWorld](https://github.com/Tongyi-MAI/MobileWorld) checkout at revision
`0dcd0980eac64d76f498f93568a1ec0594b743c4`.  AST inventory finds all `201` upstream task classes
across `20` apps: calendar, Chrome, Gmail, mall, maps, Mastodon, messages, native apps, settings,
and work.  The receipt records the source-file hashes and the contract's screenshot,
accessibility-tree, backend-state, tap/swipe/type/keyevent/wait/MCP interfaces without copying task
rows into LocalAgent training.

MobileWorld is especially relevant to the requested email, browser, cross-app, and agent-user
workflows, but its official runner requires a privileged Docker-in-Docker rooted Android AVD,
Linux/WSL2 KVM, model/user-agent credentials, and optional MCP credentials.  The current host is
macOS arm64 with no Docker, ADB, QEMU, or `/dev/kvm`; the optional `uv run --offline mw env check`
also stops before preflight because `gradio==5.49.1` is not cached.  No official runner, native
environment, model, MCP service, or user simulator was executed, so the receipt deliberately has
`score: null` and remains provenance/runtime evidence rather than a MobileWorld benchmark result.

### MobileGym official split profile (m122)

The [`m122 split-profile receipt`](paper/results/raw/m122-mobilegym-source-split-profile-v1.json)
now verifies the exact split manifests from the hash-pinned MobileGym source archive without
retaining task prompts.  `train.txt` contains 160 IDs and `test.txt` contains 256 IDs with no
overlap, yielding all 416 unique benchmark tasks.  The separate seven-task payment and fourteen-
task high-risk lists are safety subsets, not additional train/test rows; their overlap is recorded
explicitly.  Family counts and split-file hashes are reproducible through
[`profile_mobilegym_source.py`](../scripts/profile_mobilegym_source.py).  This establishes a clean
evaluation boundary for the WebGPU model, but the CC-BY-NC-4.0 content, simulator state, judges,
and screenshots remain outside training and no native score is claimed.

### MobileGym native runtime smoke (m123)

The [`m123 runtime receipt`](paper/results/raw/m123-mobilegym-native-runtime-smoke-v1.json)
boots the pinned MobileGym source in a local Vite server and loads it in headless Chromium.  The
page returns HTTP 200, exposes the documented `window.__SIM__` bridge with `{os, apps}` state,
and the upstream registry loads 423 task classes while every official 160-train/256-test ID
resolves.  Repeated resets preserve state shape and size but are not byte-identical: the receipt
records timestamp-only diff paths instead of calling the reset deterministic at the raw-byte level.
The smoke invokes no model, task judge, screenshot scorer, or external account, so it is a native
runtime preflight rather than a MobileGym-Bench result.

### MobileGym text-only model probe (m124)

The [`m124 probe receipt`](paper/results/raw/m124-mobilegym-model-probe-v1.json) runs the current
10.52M deployment-repair checkpoint inside the same pinned simulator on one ID from the official
256-task test split.  The bridge supplies only a bounded DOM-text projection (no screenshot) and
translates the additive `mobile_*` contract into native MobileGym actions.  The model was invoked
twice and produced two `mobile_input_text` calls, but the task judge passed `0/1`; output and
argument values are hash-bound and omitted.  This is useful failure evidence for the current
text-first boundary, not a MobileGym benchmark receipt: `native_receipt_eligible` is explicitly
false and the strict gate remains blocked until the full release-compliant runner is supplied.

The new mobile and MCP suites reinforce the same design requirement. iOSWorld's persistent identity
and cross-app data make memory/state tracking first-class rather than an optional prompt feature;
MobileSafetyBench makes confirmation, refusal, and prompt-injection handling measurable; and
OSWorld-MCP separates the decision to invoke a structured tool from the GUI action itself. These
should be represented as distinct WebGPU heads/metrics, not collapsed into token accuracy.

The m178 source audit pins iOSWorld at `e91f4cb2ef4c9dd48fef83a894477b41fd5e209d` and
MobileSafetyBench at `bc5e0579626a280c4f551261abcb721442ff92ea`, retaining README and available
license hashes plus aggregate contracts.  This makes the safety and personalization references
reproducible without importing protected task text, seeded profiles, APKs, or simulator state into
training.  See the [`m178 receipt`](paper/results/raw/m178-mobile-safety-personalization-source-audit-v1.json).

The follow-up [`m179 manifest audit`](paper/results/raw/m179-mobile-evaluation-manifest-audit-v1.json)
temporarily fetched only the public task manifests to bind their bytes and aggregate counts:
iOSWorld has 133 task rows, while MobileSafetyBench exposes a 90-row source table plus a separate
three-row QA file.  The paper's 100-task MobileSafetyBench suite remains the benchmark contract;
the files are not interchangeable with that score and their prompts are not admitted to training.

## Recommended WebGPU protocol

The deployment should be evaluated in two non-interchangeable tracks:

1. **Offline contract track.** Use public train data only for SFT. For held-out rows, expose the
   task plus compact DOM/accessibility/state text and the task-scoped candidate tool list. Run the
   route → retrieval → pointer grounding → parser → recursive JSON-schema validator. Report route,
   tool exactness, exact arguments, schema validity, abstention, and complete-trajectory success.
2. **Native runtime track.** Run the unchanged checkpoint in the official Android, BrowserGym,
   desktop VM, ToolSandbox, MCPMark, or EnterpriseOps-Gym environment. Freeze the source revision,
   task IDs, reset seeds, browser/VM/container versions, model/tokenizer hashes, full action logs,
   and verifier output. Do not convert the offline track into a native score.

For the current WebGPU model, the first track is now implemented for ToolSandbox in
[`evaluate_toolsandbox_text.py`](../scripts/evaluate_toolsandbox_text.py). The m63 receipt shows
why the candidate contract is necessary: row-scoped retrieval achieves `60%` tool exactness,
`30%` exact arguments, and `100%` schema validity, while exposing the inherited global selector
achieves only `10%`, `10%`, and `10%`. The matched no-schema child ties the row-scoped arm, so the
schema-conditioned SFT child is not promoted as representation-transfer evidence.

The current checkpoint's public ToolACE action-history canary is [`m310`](paper/results/raw/m310-current-toolace-action-history-canary-v1.json).
It uses eight source rows from the pinned ToolACE projection and the same catalog-constrained
dispatcher used by the WebGPU path: 17 action steps produce `11.76%` tool exactness, `0%`
argument/step/episode exactness, and `70.59%` schema validity. The canary makes the failure mode
explicit—arbitrary public tool names are frequently mapped to a nearby schema or dropped—while
keeping all tool calls local and side-effect free. It is a bounded public-projection diagnostic,
not official ToolACE/BFCL execution or evidence of email, Notion, browser, or MCP control.

The larger [`m309` probe](paper/results/raw/m309-current-toolace-action-history-v1.json) covers 64
held-out rows and 113 action steps: tool exactness is `22.12%`, argument/step exactness `2.65%`,
schema validity `69.91%`, and complete-episode exactness `0%`. This is the current checkpoint's
source-projected baseline; the eight-row m310 canary is retained as the exact before/after slice
for the selector transfer below.

The follow-up [`m312 selector transfer`](paper/results/raw/m312-current-toolace-selector-free-run-transfer-v1.json)
uses 16 source-disjoint public train actions to update only the dense selector, with the decoder
backbone and route/pointer heads frozen. On the same eight-row held-out canary, selector top-1 and
free-run tool exactness both move from `11.76%` to `17.65%` (`+5.88` points), while schema
validity drops from `70.59%` to `64.71%`; argument, step, and episode exactness remain `0%`.
The matched random selector control and selector movement are retained in the receipt. This is
evidence for a candidate selector adapter—not a deployable policy—so the full-policy promotion
decision remains rejected pending larger source-disjoint and native verifier-backed runs.

The larger [`m313 transfer`](paper/results/raw/m313-current-toolace-selector-64-transfer-v1.json)
uses 447 public train actions and 113 held-out action steps. Selector top-1 rises from `23.89%`
to `30.97%` (matched random `28.32%`); free-run tool exactness rises `22.12%`→`26.55%`,
argument/step exactness `2.65%`→`7.08%`, episode exactness `0%`→`3.12%`, and schema validity
`69.91%`→`72.57%`. This is the strongest current public ToolACE transfer signal, but the
selector moves substantially (query `58.53%`, tool `225.42%`) and the evaluation remains a
64-row projection; retain it as a candidate adapter and require a larger held-out and native
verifier-backed replication before deployment adoption.

The [`m316 pointer transfer`](paper/results/raw/m316-current-toolace-pointer-free-run-transfer-v1.json)
isolates the remaining argument-grounding bottleneck. It trains 182 locatable spans from the same
public ToolACE train projection and evaluates 63 held-out spans. Offline decoded-value exactness
rises from `0%` to `19.35%`, but it only ties the matched random pointer; in the 113-step free run,
argument/step/episode exactness remains `2.65%`/`2.65%`/`0%`, schema validity rises `69.91%`→`76.99%`,
and tool exactness falls `22.12%`→`21.24%`. The pointer is therefore rejected for deployment
promotion. The adapter also now preserves unnamed legacy pointer rows under deterministic
placeholders when a checkpoint omits `ptr_args`, rather than silently misaligning or dropping
the learned matrix.

## Weight-adoption decision rule

Adopt a pretrained backbone only when all three arms use the same tokenizer, configuration, seed,
data rows, and training budget:

- pretrained frozen or head-only continuation;
- pretrained low-rate-unfrozen continuation;
- matched random-backbone control.

Record relative L2 movement by embedding, attention/mixer, FFN, normalization, and action-head
groups. A lower loss or selector score without native closed-loop improvement is not enough to
promote the weight transfer. The current AgentNet and ToolSandbox receipts therefore support
compatibility and a low-rate initialization policy, but not a claim of stateful computer-use
capability.

The m93–m96 Mind2Web→MCPMark matched triad is an explicit cross-domain reuse audit.  Frozen
transfer improves exact MCP tool selection by 5.02 percentage points over random but reduces
service routing by 20.92 points; low-rate unfreezing recovers routing to 48.12% but falls below
random on selector top-1/top-3 and moves the backbone by 0.297% relative L2.  Until a native,
verifier-backed MCP run reproduces a benefit, keep this backbone as a browser candidate only and
do not describe it as a general MCP router.

The m100 public tri-domain continuation extends that decision beyond MCP routing.  A shared
low-rate update improves the large mobile holdout but slightly degrades the desktop and tiny
browser holdouts, while moving the backbone by less than 0.5% relative L2 in every substantial
group and leaving action heads fixed.  The current policy is therefore to reuse the BPE parent
for compatibility, keep surface-specific adapters/heads, and require a matched random control
plus native closed-loop evidence before merging one universal mobile/browser/desktop adapter.

The m101–m102 matched control now confirms that the parent is not merely a convenient starting
point: it beats a deterministic random backbone on all three held-out surfaces after the same
low-rate update (`+30.76` aggregate points, with `+27.09` mobile, `+47.11` desktop, and `+60.00`
browser).  Because the warm arm still loses a little desktop/browser accuracy relative to its own
pre-update baseline, this validates pretrained initialization but not a single universal adapter;
native closed-loop and action-head ablations remain the promotion gate.

The m103–m105 frozen-backbone dispatch-head triad closes the immediate action-head gap.  With
identical 200-step cached-feature updates, warm and random heads both reach 100% route accuracy
on mobile and desktop; desktop selector top-1 ties at 73.68%, while warm wins the tiny browser
selector slice 100% to 33.33%.  The aggregate selector difference is only +2.88 points (74.82%
vs 71.94%), and warm route accuracy is 99.90% vs random 100%.  Since the non-head tensors remain
bitwise unchanged and the browser sample has only seven decisions, this supports surface-specific
head adaptation—not a universal tool-use head or native email/Notion/MCP capability.  The next
promotion gate is a resettable native AndroidWorld/BrowserGym/OSWorld/ToolSandbox-style run with
complete action and verifier logs.

The m106–m108 native ToolSandbox replay is the required reality check for that head result.  Both
the warm and random dispatch-head children execute the pinned simulator/verifier on five matched
multi-turn scenarios and finish at `1/5` (`20%`), with no warm success-rate advantage.  Warm is
slightly worse on the ambiguous contact-removal scenario (`0.0` vs `0.333` milestone similarity).
Because this remains a bounded scripted-user run rather than the official split and model-based
user simulator, it is negative evidence against promoting head transfer—not a claim of native
ToolSandbox competence.  Keep the native receipt in the publication packet and retain the
official-split requirement as a blocker.

The m109 export closes the artifact-side part of the WebGPU handoff for the warm child.  The
10.52M BPE checkpoint, tokenizer, fp32/fp16 full and hidden-only graphs, and serialized dispatch
heads all pass the exporter parity and clean static-app hash gates.  This is the bundle a hardware
browser session should exercise next; it is not itself a hardware-throughput or public Hub/Space
publication result.

The follow-up [`m110 browser probe`](paper/results/raw/m110-webgpu-warm-head-browser-probe-v1.json)
did exercise that bundle in the Codex in-app Browser.  It loaded the WebGPU-labeled path, but the
warm head selected `computer_use.click` for all four email/Notion/browser prompts (`0/4` exact).
That failure is more decision-relevant than the artifact hash: it blocks publishing this bundle
until the route/selector/action-head contract is repaired and rerun with the same prompts and a
matched control.

That rerun is now bound by the [`m111 deployment repair receipt`](paper/results/raw/m111-deployment-dispatch-repair-v1.json).  It confirms that frozen-backbone route/selector adaptation plus a small
transparent grounding/planner adapter restores the deployment contract for email, Notion, URL open,
and the first two search→Notion steps.  The matched random arm is comparable or better on the
offline mixed selector probe, so the result is a practical deployment repair—not a claim that
pretrained weights universally improve tool selection.  Native AndroidWorld, OSWorld, AgentNet,
MCPMark, EnterpriseOps-Gym, and official ToolSandbox evidence remain separate publication gates.

The [`m112 AgentNet surface-selector receipt`](paper/results/raw/m112-agentnet-surface-selector-repair-v1.json)
closes a naming mismatch discovered in the desktop projection: AgentNet's low-level `agentnet_*`
candidate surface now has its own selector in the checkpoint while the browser selector remains
intact.  Warm selector top-1 is 70.68% versus 71.43% for the matched random selector, and
end-to-end first-action type reaches 1.0, but all eight projected trajectories remain unsuccessful
because screenshots and coordinate/text grounding were deliberately not consumed.  This is a useful
negative transfer result: surface-specific selector routing helps the action-family contract, while
pretrained initialization is not a universal advantage and does not replace a native desktop adapter.

The [`m114 EnterpriseOps-Gym email receipt`](paper/results/raw/m114-enterpriseopsgym-email-retrieval-m111-ablation-v1.json)
adds an out-of-domain productivity check using 67 public email rows and 15-tool distractor pools.
The warm m111 head reaches hit@1/3/5 of 20.90/59.70/86.57%, versus 13.43/47.76/76.12% for the
matched random control.  Warm transfer therefore helps retrieval relative to random on this
surface, but it trails the older m14 diagnostic at hit@1 (26.87%) and does not establish database
verifier, MCP execution, or real-email success.

The [`m115 xLAM-derived receipt`](paper/results/raw/m115-xlam-derived-function-calling-transfer-v1.json)
adds generic function-calling coverage without claiming access to the gated Salesforce source.  On
128 rows from a public Apache-2.0 derivative test shard, row-local retrieval produces 50% first-tool
exactness and 100% schema-valid calls, but 0% exact arguments.  The global selector transfers poorly
to the unseen xLAM tool vocabulary (0.78% warm first-tool exact versus 0% random).  This separates
three failure modes for the WebGPU model: retrieval can locate a tool, schema-constrained emission
can be valid, and argument copying/multi-call planning still fail.  The original xLAM dataset is
gated here, so this is derivative evidence rather than an official Salesforce split.

The [`m116 MCPMark routing receipt`](paper/results/raw/m116-mcpmark-routing-transfer-v1.json)
extends the realistic tool surface to a pinned public MCP benchmark whose tasks cover Playwright,
Notion, filesystem, GitHub, and Postgres workflows.  Across all 169 standard task descriptions,
the warm and random checkpoints both route 14.79% correctly; on the 70 easy descriptions both are
14.29%.  The model recognizes the Playwright family (100% in both suites) but routes 0% of the
Notion, filesystem, GitHub, and Postgres families.  This is actionable transfer evidence for the
WebGPU deployment: browser primitives are learned, while service-specific tool naming and
multi-step state planning are not.  MCPMark's actual MCP servers and verifiers were not started,
so the receipt is a task-description proxy rather than task success.

The [`m117 trajectory provenance receipt`](paper/results/raw/m117-mcpmark-trajectory-metadata-v1.json)
confirms that the public [MCPMark trajectory-log dataset](https://huggingface.co/datasets/Jakumetsu/mcpmark-trajectory-log)
contains multi-turn, tool-result-grounded traces under an MIT license.  A bounded filesystem trace
has 45 events, 21 paired calls, and five concrete tools (`list_directory`, `read_text_file`,
`read_multiple_files`, `move_file`, and `list_allowed_directories`).  Because the trace includes
large third-party document outputs, this pass deliberately stores only provenance and counts; it
does not silently turn public logs into training data.  A content/license audit and argument/schema
normalizer are required before any SFT or distillation use.

The [`m118 redacted trajectory SFT receipt`](paper/results/raw/m118-mcpmark-redacted-trajectory-sft-transfer-v1.json)
implements that audit on three hash-pinned public traces (filesystem, Notion, and Playwright; 67
paired calls).  The normalizer keeps user prompts and structured arguments, but replaces every tool
result and assistant free-text response with fixed markers and rewrites absolute paths to workspace
suffixes.  Filesystem and Notion rows train a 16-step teacher-forced continuation; the Playwright
row is held out.  Both warm and matched-random arms have identical train/eval loss and identical
backbone movement, with held-out loss improving from 5.6167 to 5.5599 but sequence accuracy staying
at 0%; a one-step unseen Playwright probe selects the wrong action in all four arms.  This is useful
evidence that redacted multi-turn traces are ingestible and that the current tiny model still lacks
stateful action planning.  It is not an official MCPMark score, native MCP execution, or a reason to
publish a universal adapter.

The [`m119 dynamic-selector receipt`](paper/results/raw/m119-mcpmark-dynamic-selector-transfer-v1.json)
then separates representation transfer from teacher-forced language loss.  A 22-tool global schema
catalog is built from the same redacted rows, and a frozen-feature dense selector is trained only on
the 51 filesystem/Notion assistant decisions.  On the 19-call held-out Playwright trajectory, the
warm WebGPU body reaches top-10 routing of 84.21% while the matched random-body control reaches 0%;
both top-1 scores are 0%.  The warm arm therefore carries useful broad retrieval signal into the
unseen browser catalog, but cannot make the exact first choice.  This supports retaining the
pretrained body for candidate retrieval and adding service/action adapters; it does not justify
claiming native MCP success or a universal tool policy.

The [`m120 broad redacted SFT receipt`](paper/results/raw/m120-mcpmark-broad-redacted-sft-transfer-v1.json)
expands the same public trajectory method across five MCP services: filesystem, Notion, GitHub,
Postgres, and Playwright.  Eight rows (107 calls) train and two Playwright rows (24 calls) remain
held out.  After the same 32 CPU SFT steps, the warm 10.52M body reaches held-out token accuracy
of 42.27% versus 3.12% for a matched random body, a +39.15-point transfer gap; exact sequence
accuracy is still 0% in both arms.  The warm body moves only 0.345% aggregate relative L2
(embedding 0.466%, mixer 0.181%, FFN 0.212%), while the random body moves 102.06% aggregate.
This is stronger evidence for reusing pretrained representations and assigning larger rates to
new heads, but it remains teacher-forced, redacted, and non-native.

### MCPMark state-summary trajectory transfer (m213)

The [`m213 receipt`](paper/results/raw/m213-mcpmark-state-summary-transfer-v1.json) tests whether
the fixed tool-result marker was hiding the state signal needed for recovery.  A new
[`normalize_mcpmark_state_trajectory.py`](../scripts/normalize_mcpmark_state_trajectory.py) keeps
only a deterministic result status (`ok`/`error`), coarse shape/content types, bounded character
counts, a page-state boolean, and a short digest.  It retains no result text, URLs, document
content, identifiers, or assistant free text.  The split is whole-trajectory disjoint: eight
filesystem/Notion/GitHub/Postgres rows train and two Playwright rows remain held out.

After 64 matched CPU continuation updates, the warm 10.52M body rises from `38.01%` to `41.46%`
held-out assistant-token accuracy, versus `0.79%` to `15.63%` for the random body.  Warm movement
is small (embedding `0.887%`, mixer `0.319%`, FFN `0.374%`, normalization `0.013%`) and action
heads remain unchanged; exact sequence accuracy is `0%` for both.  A frozen global selector still
gets `0/24` top-1 decisions on the unseen Playwright service for both arms.  This makes the state
summary a better provenance-safe training primitive than a fixed marker, but not yet a transferable
MCP policy.  The children are not exported or adopted, and no MCP server, browser, verifier, or
official split was executed.

### Direct m180 WebGPU child comparison (m193)

The [`m193 receipt`](paper/results/raw/m193-current-m180-webgpu-browser-fresh-realistic-actions-v1.json)
closes a provenance gap by exercising the separately exported `108276…` m180 child itself rather
than the workspace's older `9bba…` bundle.  The browser reports `WEBGPU` and model-ready status,
but the same twelve realistic prompts used for the workspace smoke produce only `1/9` exact tools
on unambiguous single-step requests and `0/2` exact planner trajectories.  The sole exact tool is
an explicit URL open; email and Notion requests still select `open_url` with malformed URL-shaped
arguments.  The run has no external side effects and uses no public benchmark rows.

This comparison is the clearest current separation between transfer and deployment quality:
teacher-forced public continuations can improve token accuracy and route loss while the actual
WebGPU action contract remains unusable for productivity and multi-step control.  The policy is to
retain the m180 body for compatibility experiments, but not promote it as a browser/mobile agent
until a catalog-aware action head, argument grounding, and native closed-loop evidence all improve
together.

### Native WebGPU capability and latency receipt (m214)

The [`m214 receipt`](paper/results/raw/m214-webgpu-native-capability-current-bundle-v1.json) is the
first current-workspace browser receipt that proves the requested provider path rather than merely
loading a page.  Chromium reports `navigator.gpu`, exposes an Apple Metal adapter
(`vendor=apple; architecture=metal-3`), and ONNX Runtime Web `1.27.0` runs the fp16 action graph
with `requested_provider=["webgpu"]` and no provider retry.  The clean eight-artifact bundle and
its parity manifest verify before dispatch.

Across three bounded local prompts and 30 measured repetitions per prompt, p50 latency is `7.2 ms`
and the action-input throughput proxy is about `1,389 tokens/s`; the conservative graph-plus-host-
tensor estimate is `20.5 MB`.  Quality is separate: URL opening is exact in `30/30` trials, while
the email request selects `email_send` instead of the expected `send_email` and the Notion request
selects `notion_create_page` instead of `notion_write`.  Thus native WebGPU capability and speed
pass, but realistic email/Notion action quality is `1/3` and closed-loop success is zero.  No real
email, browser navigation, or Notion account was touched, and this single Apple Metal run is not a
cross-device throughput claim or a publication-quality agent score.

The [`m215 gate`](paper/results/raw/m215-workshop-gate-native-webgpu-v1.json) confirms the scope of
this result: the WebGPU capability/latency requirement and the existing matched transfer ablation
now pass the strict checker.  Readiness is still false with twelve blockers—eleven official native
benchmark receipts plus a public model/demo manifest.  The local browser result cannot substitute
for AndroidWorld, MobileGym, iOSWorld, BrowserGym, OSWorld, AgentNet, ToolSandbox, MCPMark, or
EnterpriseOps-Gym execution.

### Canonical productivity schema boundary (m216)

The [`m216 receipt`](paper/results/raw/m216-webgpu-native-canonical-productivity-v1.json) reruns
the same native Chromium harness after fixing a duplicate-tool precedence bug in the demo runtime.
The bundle metadata contains both the public `send_email`/`notion_write` schemas and legacy
`email_send`/`notion_create_page` mobile aliases.  The guard now prefers the public names and falls
back to an alias only when a canonical entry is absent, so argument grounding uses `recipient` and
`content` rather than the legacy `to`/`subject`/`body` or page shape.

On the same Apple Metal adapter, all three bounded prompts are now schema-exact (`3/3`, 30/30 per
case), with p50 latency `5.1 ms` and an action-input throughput proxy of about `1,918 tokens/s`.
This is a contract/adapter correction, not a learned-quality improvement: email and Notion still
use explicit intent guards, no external account is touched, and closed-loop success remains zero.
The strict workshop gate therefore remains unchanged; native benchmark receipts and a public
model/demo manifest are still required.

The [`m217 gate`](paper/results/raw/m217-workshop-gate-native-canonical-v1.json) binds this
canonical receipt directly: the WebGPU and m25 weight-ablation checks pass, but the gate still has
the same twelve blockers (eleven official native benchmark receipts and the public model/demo
manifest).  The alias fix therefore improves contract observability without changing the
publication decision.

### Current native evidence join (m218)

The [`m218 gate`](paper/results/raw/m218-workshop-gate-current-canonical-native-v1.json) binds
the canonical m216 WebGPU receipt to two existing official-split native evaluations: MobileGym
(`256` tasks, `13/256` success) and BrowserGym/MiniWoW (`240` episodes, `0/240` success).  It also
binds the live public model/demo manifest and the m25 transfer/no-transfer ablation.  The gate
now has nine blockers rather than twelve.  ToolSandbox is deliberately still blocked because its
129-scenario audit does not verify the official split; this prevents a local simulator result from
being promoted to a benchmark claim.

### ToolACE public source and transfer audit (m219)

The [`m219 receipt`](paper/results/raw/m219-toolace-public-source-transfer-audit-v1.json) adds
Team-ACE/ToolACE to the supplemental source registry.  The pinned Apache-2.0 snapshot contains
`11,300` raw rows; the strict projection accepts `8,993`, with `8,044` train parent records and
`949` eval parent records, zero parent overlap, and zero prompt overlap.  Rows that lack a strict
first action or valid schema are rejected rather than silently converted into supervision.

Three matched current-child transfer receipts cover first-action, full multi-turn, and
action-history projections.  Warm-start token accuracy beats the matched random backbone by
`17.37–49.48` percentage points, while warm sequence exactness remains `0%` in every arm.  The
adoption conclusion is therefore narrow and reproducible: keep the pretrained body for further
ToolACE continuation, but do not promote the tool heads or call the result native MCP, BFCL,
mobile, browser, desktop, email, or Notion capability.

### Current AgentNet text-action evaluation and matched control (m220)

The [`m220 receipt`](paper/results/raw/m220-agentnet-current-text-action-evaluation-v1.json) runs
the current m180 WebGPU-compatible child and a matched random-backbone control through the pinned
public [AgentNet](https://huggingface.co/datasets/xlangai/AgentNet) text projection.  The held-out
projection contains `133` action rows from `8` unseen Ubuntu parent trajectories, with screenshots
and desktop state deliberately excluded.  Both arms cover every expected parent ID, so the score is
not a missing-prediction artifact.

The warm arm predicts the correct first action *type* on all eight parents (`100%` versus `12.5%`
for random, `+87.5` percentage points), but both arms remain at `0%` exact trajectories and
effectively zero coordinate/text action score.  The existing tensor audit finds `51` compatible
shared tensors, tokenizer equality, and warm embedding/attention/FFN relative movement of only
`0.449%`/`0.223%`/`0.273%`.  This supports retaining the pretrained body as an initialization
candidate, not adopting it as a computer-use solution: visual grounding plus native AgentNetBench
or desktop execution is still required.

### AgentNet public continuation and matched random transfer (m221)

The [`m221 receipt`](paper/results/raw/m221-agentnet-public-continuation-transfer-v1.json) is the
first current-child SFT continuation on the pinned AgentNet text/action projection rather than a
selector-only probe.  It uses `513` train rows from `32` parents and `133` held-out rows from `8`
unseen parents, with the same `32`-step schedule for the warm m180 checkpoint and a matched random
backbone.  The split is parent-disjoint and screenshots remain excluded.

Warm reuse improves held-out teacher-forced token accuracy from `55.80%` to `67.00%` and route
accuracy from `18.80%` to `83.46%`; the random control reaches `21.22%` token accuracy and `0%`
route accuracy.  The stronger action evaluator is decisive: warm and random children both remain
at `0%` exact trajectories and `0` meaningful coordinate/text action score.  The tensor audit
finds `51` compatible tensors, equal tokenizers, and warm embedding/attention/FFN movement of
`0.532%`/`0.284%`/`0.346%`.  The adoption decision is therefore to keep warm reuse for a future
visual/action-grounded continuation, while refusing to promote this text-only training result to
native desktop or AgentNetBench capability.

### AgentNet continuation to local productivity bridge (m222)

The [`m222 receipt`](paper/results/raw/m222-agentnet-continuation-stateful-productivity-bridge-v1.json)
checks whether the m221 computer-use continuation preserves the requested email, Notion, browser,
and recovery contract.  The resettable in-memory fixture has `63` tools and `5` workflows; its
oracle reaches `16/16` accepted steps and `5/5` complete tasks, confirming the verifier itself.

The warm child accepts `5/16` model steps versus `1/16` for the matched random child, but both
complete only the abstention workflow (`1/5`).  Warm email and Notion completion remain `0/1`, as
do browser and recovery completion.  This is a useful cross-surface regression signal—the warm
backbone preserves more valid transitions—but it is deliberately not presented as real email,
Notion, MCP, browser, or WebGPU control.

### AgentNet continuation to MCPMark service-routing bridge (m223)

The [`m223 receipt`](paper/results/raw/m223-agentnet-continuation-mcpmark-routing-bridge-v1.json)
reruns the current m221 warm child and matched random control on the pinned public
[MCPMark](https://github.com/eval-sys/mcpmark) standard (169 rows) and easy (70 rows) task-description
manifests.  This is the closest available public proxy for the requested Notion, filesystem,
GitHub, Postgres, and browser tool families while the native MCP services are unavailable.

Warm routing reaches `19.53%` on standard and `22.86%` on easy, versus `22.49%` and `18.57%`
for random.  The warm standard breakdown is informative rather than strong: Notion is `10/28`
(`35.71%`) and Playwright is `23/25` (`92%`), while filesystem, GitHub, and Postgres are `0/30`,
`0/23`, and `0/63`.  Selector top-1 remains `0%` for the warm arm.  The mixed result supports
retaining a compatible warm body for service-grounded continuation, but does not justify native
MCP, real-account email/Notion, or WebGPU productivity claims.  MCP servers, credentials, state
transitions, verifiers, and pass@k were not run, and the task text was not retained for training.

### Frozen dispatch repair control (m194)

The [`m194 receipt`](paper/results/raw/m194-current-m180-dispatch-repair-v1.json) tests whether the
m180 child can be repaired without changing its WebGPU-compatible backbone.  A matched warm-start
and random-head arm trains route and dense-selector heads on 741 source-disjoint AgentNet/Mind2Web
projection rows plus 1,286 deterministic local adapter rows; 320 additional synthetic adapter rows
are held out.  Warm routing reaches 99.61% versus 99.42% for random, but warm selector top-1 is
82.75% versus 82.95%, and canonical tool probes are 4/6 versus 5/6.  The warm head therefore is
not adopted and no export is requested.  This is evidence for measuring head transfer separately
from backbone reuse, not evidence of native browser, mobile, MCP, visual, or real-account control.

### Unguarded m194 WebGPU comparison (m195)

The [`m195 receipt`](paper/results/raw/m195-current-m194-webgpu-browser-unguarded-v1.json) exports
the m194 warm head with hard PyTorch↔ONNX parity and runs the exact prompt set used by m193 through
the unguarded comparison shell.  After explicit catalog aliases (`notion_write`→`notion_create_page`
and `send_email`→`email_send`), tool-family routing rises from the m180 parent's `1/9` to `5/9`.
Strict action exactness stays at `1/9`, however: URL opening is correct, while search, click, email,
Notion, and calendar arguments remain incomplete or contain tokenizer markers.  The search→Notion
planner emits the right two tool families but still fails strict arguments, and the file/test/commit
planner abstains (`0/2` complete trajectories).  This isolates the next required work: argument
copy/grounding and state-conditioned planning, not another route-head-only update.  The temporary
export is therefore not promoted or published.

### Public DOM grounding control (m196)

The [`m196 receipt`](paper/results/raw/m196-m194-grounded-mind2web-vocab-fix-v1.json) also fixes a
pipeline bug exposed by current checkpoints: m194 carries a 23-row stateful pointer vocabulary,
while the older Mind2Web trainer assumed 17 legacy rows.  The migration now copies embeddings by
declared argument name and appends `target_id`/`value`, yielding a 25-row compatible head.  A
32-step continuation over 219 public train decisions and 63 disjoint eval decisions still gives
`0/63` exact pointer spans before and after.  The body moves only 0.274%/0.422%/0.507% in
embedding/mixer/FFN groups, while pointer and tool heads move more.  The result confirms that
argument grounding needs aligned span supervision and an explicit DOM/action contract; more
route-head training will not solve it.  The child is not exported or adopted.

### Frozen-feature pointer transfer (m197)

The [`m197 receipt`](paper/results/raw/m197-pointer-grounding-repair-v1.json) isolates the
argument head with the m194 backbone completely frozen.  It caches token-position features for
219 public Mind2Web train spans and trains matched warm and random start/end pointer heads for 400
steps.  Warm transfer reaches `3/63` held-out exact spans (4.76%) while random reaches `5/63`
(7.94%).  The warm head moves substantially less (`31.42%` shared pointer L2 versus `134.06%`),
but the lower movement does not carry useful unseen-value grounding.  This rules out adopting the
warm pointer solely because it preserves pretrained weights; the next adapter must align DOM
candidate serialization, action argument schemas, and span labels before another WebGPU export.

### Current native ToolSandbox smoke (m198)

The [`m198 receipt`](paper/results/raw/m198-toolsandbox-native-current-v1.json) runs the current
m194 checkpoint in the pinned Apple [ToolSandbox](https://github.com/apple/ToolSandbox) simulator
and milestone verifier.  It completes `3/3` resettable single-turn scenarios—cellular-off,
Wi-Fi-off, and send-message—with mean milestone similarity `1.0`; no external API is called.
This is the first current-checkpoint stateful-tool result for the deployment path, but it remains a
diagnostic smoke: the scripted user ends after the first response, multi-tool and multi-turn cases
are truncated, and the official split/model-based user simulator/full scenario matrix were not
executed.  It therefore does not establish native MCP, email, Notion, or publication-gate success.

The follow-up [`m200 comparison receipt`](paper/results/raw/m200-toolsandbox-native-current-base-v1.json)
runs all 129 base/no-distraction scenario names used by the prior m143 transfer audit with the
m194 checkpoint.  The current child reproduces the baseline exactly: `28/129` success (`21.71%`),
mean similarity `0.2921`, and zero per-scenario wins or losses.  This is a useful negative result:
the m194 route/selector repair did not change native ToolSandbox behavior under the one-step
protocol, so it is not promoted.  The same official-split/user-simulator boundary still applies.

The [`m201 interactive receipt`](paper/results/raw/m201-toolsandbox-native-current-interactive-v1.json)
extends the check to five stateful, ambiguous, and multi-turn scenarios with the bounded interactive
scripted user.  The m194 child reaches `1/5` (`20%`) and exactly matches the prior m92 stateful,
public-only, and projection arms on every scenario (`0.25`, `0.3333`, `1.0`, `0.0`, `0.5`).  The
model therefore still cannot turn route repair into reliable multi-tool state progression; the
next training target is explicit state/action-history supervision and recovery, not another frozen
selector update.

### Public ToolSandbox continuation and native bridge (m202–m204)

The [`m204 receipt`](paper/results/raw/m204-toolsandbox-continuation-native-bridge-v1.json) runs the
AST-only public projection from the pinned [ToolSandbox](https://github.com/apple/ToolSandbox) source:
107 train rows and 20 source-disjoint eval rows.  A matched 32-step warm continuation improves
held-out assistant-token accuracy from `65.79%` to `73.21%`, while the random-backbone control
reaches only `8.61%`; warm embedding/mixer/FFN movement stays at `0.52%/0.19%/0.24%`, and the
action heads do not move.  This is useful evidence that the pretrained body transfers public
ToolSandbox language signal, but the native bridge is decisive: warm and random children both
score `1/5` with identical per-scenario similarities in the interactive simulator.  The children
are therefore not exported or adopted; action-head and state-history supervision must be trained
explicitly before another WebGPU release.

### ToolSandbox selector repair and native bridge (m205–m207)

The [`m207 receipt`](paper/results/raw/m207-toolsandbox-selector-repair-bridge-v1.json) isolates
the next hypothesis: keep the m194 backbone, replace only the dense selector, and train against the
actual 32-tool catalog extracted from the public ToolSandbox AST projection.  On the same 107 train /
20 eval rows, held-out candidate top-1 rises from 0% to 50% for the warm arm and 45% for the matched
random arm (top-3 reaches 65% for both).  Selector movement is large (warm query/tool towers
1.12/0.58 relative L2; random 1.25/1.06), so this is a genuine head-only adaptation rather than
backbone drift.

The native bridge rejects that offline gain: both children score exactly `1/5` on the five-scenario
interactive simulator/verifier stress set, with identical per-scenario similarities
(`0.25`, `0.3333`, `1.0`, `0.0`, `0.5`).  The AST row-local candidates do not match the runtime
ToolSpec serialization and state/action-history conditions used by `run_toolsandbox_native.py`;
the route and pointer heads also remain inherited.  Therefore neither child is exported or adopted.
The next experiment must bind the training rows to the same runtime candidate serializer and
state-history format, then train route, selector, and argument/action heads jointly with native
closed-loop replay.  This result is diagnostic transfer evidence, not an official ToolSandbox score
or WebGPU productivity claim.

### ToolSandbox runtime ToolSpec alignment and native bridge (m212)

The [`m212 receipt`](paper/results/raw/m212-toolsandbox-runtime-heads-native-bridge-v1.json) closes
the candidate-catalog mismatch without overstating the result.  A pinned Apple ToolSandbox checkout
enumerates 1,032 named scenarios; the projection calls the same
`ExecutionContext.get_available_tools(scrambling_allowed=False)` path used by the native runner and
serializes the exact OpenAI-style ToolSpecs.  It retains 105 train and 20 eval rows, dropping two
static rows whose target is intentionally unavailable in the runtime context.  The projection still
has no simulator state or action history, and no upstream user simulator or official split is run.

Joint frozen-backbone route/selector training raises the warm row-local selector from `5%` to `70%`
top-1 (`100%` top-3) and route-to-`app_action` from `5%` to `100%`.  The matched random control reaches
`80%` selector top-1 and the same `100%` route score, so the offline result does not demonstrate
pretrained-weight advantage.  More importantly, the corrected native bridge now passes the loaded
selector into decoding, yet warm and random children both remain at `1/5` (`20%`) native success.
The warm child is therefore not exported or adopted.  The remaining blocker is explicit state/action
history plus argument/action-head learning under the simulator's milestone verifier, not another
catalog-only probe.

### Public Space black-box regression (m224)

The [`m224 receipt`](paper/results/raw/m224-public-space-black-box-realistic-prompts-v1.json)
records a fresh in-app-browser inspection of the public
[`danelcsb/localagent-webgpu`](https://huggingface.co/spaces/danelcsb/localagent-webgpu) Space.
The page is live (`HTTP 200`, `Running`, `WEBGPU`) but serves an older 10,986-byte `app.js`; a
manifest probe returns `Entry not found`, so this is not the current 10.52M BPE workspace bundle.
The explicit URL prompt selects `open_url` (`1/1`, 47 ms).  The realistic email prompt selects
`set_reminder` instead of `send_email`/`email_send` (`0/1`, 37 ms), and the compound Search→Notion
prompt emits one `notion_write` proxy without a verified search result, state transition, or final
write (`0` closed-loop completions).  No credentials or external side effects were used.  The
result is retained as a public regression and publication-boundary check, not promoted to browser,
email, Notion, MCP, or official benchmark evidence; updating the Space requires authenticated HF
upload followed by public hash and behavior re-verification.

### Current EnterpriseOps-Gym email retrieval (m225)

The [`m225 receipt`](paper/results/raw/m225-enterpriseopsgym-current-checkpoint-email-retrieval-v1.json)
evaluates the current 10.52M-parameter m194 warm dispatch-repair checkpoint on the pinned public
EnterpriseOps-Gym email slice: 67 oracle rows paired with the same 67 `plus_15_tools` candidates
used by m114.  Name-only retrieval reaches Hit@1/3/5 of `49.25%/85.07%/95.52%`, compared with
`20.90%/59.70%/86.57%` for m114 (`+28.36/+25.37/+8.96` percentage points).  The repair froze
the backbone (`0%` relative movement) and moved the dense selector by `148.87%`, identifying a
head-only contract adaptation rather than evidence that the pretrained body is superior.
Because verifiers, server configuration, and execution were removed, the result is useful for
email-tool retrieval but cannot be promoted to EnterpriseOps-Gym, MCP, real-email, or WebGPU
closed-loop success.

### WebGPU side-effect and injection boundary (m227)

The [`m227 receipt`](paper/results/raw/m227-webgpu-side-effect-safety-policy-v1.json) adds an
explicit runtime policy around the model's structured action output.  `open_url` is read-only and
allowed; email, Notion, messaging, file, shell, and other state-changing tools are marked
`confirmation_required`; destructive actions receive high severity; and prompt-injection or
secret-exfiltration indicators in the request or untrusted observation are blocked.  The seven-case
audit is pure JavaScript (`side_effect_confirmation_v1`), changes no learned weights, and executes
no external effects.  It is a deployment safety contract, not a VPI-Bench, AgentCIBench,
MobileSafetyBench, or native task-success score.

### Current public-release reconciliation (m228)

The [`m228 receipt`](paper/results/raw/m228-current-public-release-audit-v1.json) rechecked the
authoritative public pages before extending the evaluation plan.  BFCL V4 now documents separate
agentic web-search, memory-management, multi-turn, and format-sensitivity categories in addition
to the older AST call suites; these are evaluation-only because they require the upstream checker,
live search, or stateful memory runtime.  iOSWorld's public release explicitly includes 26 apps,
133 personalized/cross-app tasks, and an optional MCP server, so it is a native identity/state
benchmark rather than a screenshot-only action set.  OSWorld-V2 requires the exact
`osworld-v2-2026.06.24` release across code, gated task classes/assets, mocked websites, and
provider image; floating `main` is not comparable.

The audit also found a release-scope discrepancy that must remain visible: the current
[MobileSafetyBench project page](https://mobilesafetybench.github.io/) describes 250 tasks (200
daily scenarios plus 50 injection scenarios), while the pinned paper/repository contract used by
the earlier m178/m179 receipts describes a 100-task suite (50 helpfulness, 42 safety, 8 injection).
Those counts are not additive or interchangeable.  Before a native safety score, one release must
be selected and its task manifest, APKs, emulator, and verifier hash-bound.  No m228 rows entered
training and no runtime was executed.

### Stateful productivity and user-in-the-loop refresh (m229)

The [`m229 receipt`](paper/results/raw/m229-stateful-productivity-benchmark-refresh-v1.json)
updates the contract for the workflows the WebGPU demo is meant to approach.  MCPMark's public
task format requires `meta.json`, `description.md`, and an independent `verify.py` state check over
Notion, GitHub, filesystem, Postgres, and Playwright services; its documented tasks average 16.2
execution turns and 17.4 tool calls, so a single correct tool name is not task completion.
EnterpriseOps-Gym exposes 649 resettable enterprise tasks, including 67 email rows, with selected
tools, MCP server state, and SQL/database-state verifiers.  These are the closest public email and
Notion-adjacent stateful contracts in the current matrix, but no server or account is attached here.

AppWorld-UL adds the missing user-interaction dimension: 516 tasks require clarification,
confirmation, or handling an infeasible instruction.  Its official page links the public
[Apache-2.0 AppWorld runner](https://github.com/StonyBrookNLP/appworld), whose documented
installation (`pip install appworld`, `appworld install`, `appworld download data`) and resettable
`AppWorld(...).evaluate()` loop establish an executable base contract.  The UL-specific task assets
and knowledge-bounded user-simulator release are not separately pinned on that page, so this is
runner evidence rather than a native UL score.  This maps directly to the deployment policy:
the model must distinguish an allowed read, a confirmation-required write, a clarification request,
and a blocked action before calling a tool.  τ³-Bench adds knowledge-grounded policy/tool use and
full-duplex voice, while TUA-Bench covers 120 general terminal tasks including email management and
live-web information seeking; both remain evaluation-only for this text-first WebGPU model.

The adoption conclusion is unchanged but sharper: warm-start token or retrieval gains are useful
initialization evidence only.  Promotion requires matched random/no-transfer arms, per-group tensor
movement, and native state-verifier success on the same pinned release.  m229 adds no training rows,
does not execute external effects, and does not convert the local email/Notion state machine into an
official MCPMark, EnterpriseOps-Gym, or AppWorld-UL score.

### Structured-action safety boundary fix (m230)

The [`m230 receipt`](paper/results/raw/m230-webgpu-structured-action-safety-fix-v1.json) caught a
real deployment-path mismatch: planner calls supplied a tool name as a string, while the ordinary
UI path supplied `{tool, args}`.  The policy reader only inspected `{name}`, so the second path
could have been labeled read-only even for `send_email` or `notion_write`.  The policy now accepts
all three action shapes (`name`, `tool`, and nested `action.tool`), and the regression covers
structured email/Notion confirmation, read-only URL allowance, and injection blocking for an
interactive click.  This fixes policy observability; it is not a learned-quality or native task
success result.

### User-in-the-loop clarification boundary (m231)

The [`m231 receipt`](paper/results/raw/m231-webgpu-user-in-the-loop-clarification-policy-v1.json)
adds the missing AppWorld-UL interaction state to the static WebGPU path.  A structured
`send_email` action with no `recipient` is now `clarification_required`; the same action with a
recipient is `confirmation_required`; a complete `open_url` is `allowed`; and an injected Notion
write is `blocked`.  The policy is schema-aware but remains a deterministic deployment guard, not
learned capability or AppWorld-UL benchmark performance.

### Stateful productivity GRPO child and weight preservation (m234–m235)

The [`m234 receipt`](paper/results/raw/m234-stateful-productivity-grpo-head-preserved-v1.json)
is a fresh pure-PyTorch continuation from the current 10.52M Mind2Web-adapted BPE child.  It
builds 16 source-disjoint local training decisions over email, Notion, browser search/recovery,
and abstention, runs a 32-update SFT prelude plus four GRPO rollout updates, and preserves the
five deployment-head containers (`tool_head`, `ptr_head`, `route_head`, `dense_selector`, and
`selector_proj`) as frozen tensors in the child checkpoint.  Mean shaped reward improves from
`0.0375` to `0.1125`, but held-out exact tool/text accuracy remains `0%`; the result is therefore
an RL-pipeline and reward-signal diagnostic, not a capability claim.

The [`m235 weight audit`](paper/results/raw/m235-stateful-productivity-rl-weight-transfer-v1.json)
confirms identical model configuration, shapes, and tokenizer identity.  Frozen action heads move
by `0%` relative ΔL2; the LM embedding moves `4.69%`, attention/mixer `1.40%`, FFN `1.74%`, and
normalization `0.10%`.  This makes the adoption decision explicit: retain the compatible backbone
and frozen heads only as a lineage-preserving checkpoint, and do not promote it to WebGPU or an
official benchmark until a matched no-transfer arm improves held-out exactness and a native state
verifier succeeds.  The earlier run that silently dropped auxiliary heads is superseded by this
head-preserving path and remains unadopted.

The [`m236 deployment smoke`](paper/results/raw/m236-stateful-productivity-rl-deployment-smoke-v1.json)
confirms that the preserved-head child is loadable by the actual LocalAgent runtime, but its
10-case echo-stub dispatch is only `1/10` exact.  It therefore regresses the older local parent on
this smoke and is explicitly not exported or promoted; preserving tensors fixes checkpoint
integrity, not policy quality.

### Matched random RL control (m237–m240)

The [`m237 random-control receipt`](paper/results/raw/m237-stateful-productivity-grpo-random-control-v1.json)
repeats the m234 protocol with a shape-matched random backbone and random auxiliary heads.  The
train/eval task hashes, seed, 32-step SFT prelude, four GRPO steps, rollout budget, and frozen-head
contract are identical.  The random arm stays at mean reward `−0.05`, produces zero informative
groups and zero realized optimizer updates, and has `0%` held-out exactness.

The [`m240 matched ablation`](paper/results/raw/m240-stateful-productivity-grpo-matched-ablation-v1.json)
joins both arms with tensor movement and deployment smoke.  Warm initialization has a shaped-reward
advantage of `+0.1625`, but exact tool/text accuracy ties at `0%`, while the random child wins the
10-case echo-stub deployment smoke `2/10` to `1/10`.  The decision is therefore explicit:
pretrained weights help the local reward signal but are not adopted for WebGPU deployment.  This
control is still a deterministic local simulation, not public benchmark or native environment
evidence.

### AppWorld native checkpoint probe (m241)

The [`m241 receipt`](paper/results/raw/m241-appworld-current-checkpoint-native-probe-v1.json)
adds the first resettable stateful-app runtime to this evidence chain.  An isolated AppWorld
`0.2.0.dev0` installation was unpacked from the public runner's Git-LFS bundles, its data version
`0.2.0` was downloaded, and the runner's own end-to-end verifier passed `1/1` task.  The current
10.52M BPE checkpoint was then evaluated on six public train tasks spanning SimpleNote, phone
messaging, and Venmo.  Every task reset and ran its ground-truth verifier, but native success was
`0/6`: the model emitted no AppWorld API action (two prompts were misrouted to `write_clipboard`,
four returned text), and no action was replayed.

This is stronger than a local fixture because the state database and verifier are real AppWorld
artifacts, but it is deliberately a zero-action interface baseline.  LocalAgent's compact tool
syntax still needs a schema-aware AppWorld API translator before an end-to-end agent score is
meaningful; m241 is not AppWorld-UL, email/SMS success, or a promotion signal.

### AppWorld `run_python` adapter and native replay (m242–m245)

The [`m242 train manifest`](paper/results/raw/m242-appworld-train-manifest-v1.json) and
[`m242 dev manifest`](paper/results/raw/m242-appworld-dev-manifest-v1.json) normalize 24 public
AppWorld train tasks and 12 disjoint public dev tasks into the canonical `Conversation` format.
The ground-truth programs are represented as the existing `run_python` tool; no protected test
split or raw solution text is committed.  This is a deliberately narrow adapter for AppWorld's
executable Python/API contract, not a claim that a text-only tool vocabulary is already an
AppWorld policy.

Warm SFT continuation from the current 10.52M BPE checkpoint ([`m242 report`](paper/results/raw/m242-appworld-runpython-sft-v1.json))
raises held-out assistant-token accuracy from `10.21%` to `21.60%` and lowers mean loss from
`6.645` to `5.821` on the 12 dev rows, while sequence exactness remains `0/12`.  The separate
[`m243 weight report`](paper/results/raw/m243-appworld-runpython-weight-transfer-v1.json)
shows compatible config/tokenizer/shapes, frozen action-head movement of `0%`, and small shared
backbone movement (embedding `0.57%`, attention/mixer `0.32%`, FFN `0.38%`).  These are weight-lineage
and teacher-forcing measurements, not native task success.

The [`m244 head adapter`](paper/results/raw/m244-appworld-runpython-head-adapter-v1.json)
trains only the route and dense selector for 256 steps against the exact 51-tool standard pool;
route and selector top-1 both move from `0%` to `100%` on the disjoint dev rows.  Runtime retrieval
was explicitly widened to all 51 tools because a top-10 retriever can otherwise hide
`run_python`, even when the selector ranks it correctly.

Finally, [`m245 native replay`](paper/results/raw/m245-appworld-runpython-native-dev-v1.json)
captures and executes the model's selected `run_python` call on those 12 resettable AppWorld dev
tasks.  All 12 calls were replayed, but they made `0` AppWorld API requests and passed `0/12`
verifiers.  The result is the required negative capability boundary: routing is learnable, while
the tiny model still does not emit executable API programs.  The adapter is therefore not adopted
for WebGPU or presented as an AppWorld leaderboard score; the next required step is schema-aware
program synthesis/repair plus native multi-interaction evaluation.

### AppWorld first-action API-step adapter (m247–m251)

To test whether the interface gap was only the multi-thousand-token program length, the public
train/dev ground-truth traces were reduced to their first non-bootstrap API call.  The
[`m247 train manifest`](paper/results/raw/m247-appworld-action-train-manifest-v1.json) contains 24
train tasks and the disjoint [`m247 dev manifest`](paper/results/raw/m247-appworld-action-dev-manifest-v1.json)
contains 12 dev tasks.  Credentials and bootstrap calls are excluded; the raw `Conversation`
JSONL remains outside Git and the manifests bind only task/source/action hashes.

Warm continuation ([`m247 report`](paper/results/raw/m247-appworld-action-step-sft-v1.json)) raises
held-out assistant-token accuracy from `51.38%` to `82.21%`, but sequence exactness remains `0/12`.
The standard 51-tool route and selector were already `100%` before and after training.  The paired
[`m251 weight report`](paper/results/raw/m251-appworld-action-step-weight-transfer-v1.json) shows
action heads frozen (`0%` movement), with embedding `2.12%`, attention/mixer `0.83%`, FFN `1.05%`,
and normalization `0.055%` relative movement.  These are teacher-forced and lineage measurements,
not executable-agent accuracy.

The native diagnostic adds a strict AST parser that accepts only one literal
`apis.<app>.<api>(...)` call and injects credentials from the resettable AppWorld fixture.  A
schema-grounded candidate mode ([`m250 receipt`](paper/results/raw/m250-appworld-action-step-schema-native-dev-v1.json))
replayed all 12 bounded calls and made 12 real action API requests (48 requests including 36
authentication/bootstrap requests), but the one-step verifier score was `0/12`.  The model-ranked
valid candidates were often the wrong API for the instruction; this proves isolated execution and
the parser/credential boundary, not action selection, full trajectories, AppWorld-UL, or email/SMS/
Spotify task success.  Learned free-form `run_python` code is still not used by the constrained
decoder, so this adapter is not promoted to WebGPU.

### τ-Bench native mock-domain probe (m252)

The current public [tau2-bench](https://github.com/sierra-research/tau2-bench) checkout is pinned
to `363133ada1936491fb5bcec33cd62c3518a99f65` (v1.0.1).  Its repository is MIT-licensed, but the
benchmark task/user-simulator protocol remains evaluation-only here.  The [`m252 receipt`](paper/results/raw/m252-tau2-mock-native-v1.json)
hashes the source, ten public mock-base tasks, split metadata, policy, and reset database without
retaining task text or tool outputs.  An oracle replay passes the independent tau2 environment
contract (`1.0`), proving the native verifier is live.

The current 10.52M WebGPU checkpoint was then exposed to the real mock-domain tool schemas for one
agent turn per task.  It emitted `0/10` tool calls, `0/10` exact first actions, and `0/10` bounded
native successes.  This is a useful negative result: the checkpoint can load and the environment
can execute, but the learned LocalAgent catalog/route contract does not transfer to tau2's unseen
customer-service tool names.  It is not a complete tau2 score, retail/telecom result, user-simulator
run, or WebGPU adoption signal.

### τ-Bench selector/grounding ablation (m258)

The follow-up [`m258 ablation`](paper/results/raw/m258-tau2-checkpoint-ablation-v1.json) keeps the
same ten resettable mock tasks and compares three inference arms.  The m46 public Mind2Web child
with its learned selector reaches `0/10` bounded successes; the browser-context child reaches
`2/10`; replacing the closed-world selector with the zero-training schema retriever (`k=1`) reaches
`3/10` on the same source.  The gain is not a benchmark claim: it isolates a deployment failure in
which the learned selector collapses onto `get_users` for unseen tau2 schemas.  A generic schema
grounding fix now extracts structured identifiers such as `task_1`/`user_1` and quoted titles instead
of copying an entire instruction as an argument.

The matched m46→browser tensor audit finds identical configuration/tokenizer and `51` shared
tensors.  Relative movement is largest in the action heads (`71.45%`) and embedding (`24.04%`),
with attention/mixer (`3.20%`), FFN (`4.51%`), and normalization (`0.72%`) moving less.  This supports
reusing the compatible body while replacing or recalibrating the selector for unseen schemas; it
does not prove that the browser child is a better initialization, nor does it justify publishing a
tau2, email, Notion, or WebGPU task-success score.

### τ-Bench bounded airline/retail/telecom probes (m259)

The new [`m259 domain receipts`](paper/results/raw/m259-tau2-airline-m46-retriever-k1-v1.json)
cover eight base tasks each from the public airline (`50` total tasks), retail (`114`), and
telecom (`114`) domains using the same reset-per-task native runner.  A stable gold trajectory
contract passes in all three domains; the telecom contract uses a read-only network-state task
rather than the stateful airplane-mode case whose response sequence is not deterministic under
this isolated constructor.

The m46 checkpoint with the zero-training schema retriever (`k=1`) emits calls on all three
domains, but reaches `0/8` bounded native successes in each.  These are realistic customer-service
and telecom-schema negatives with no external accounts, network services, screenshots, or retained
task text.  They are bounded domain diagnostics, not complete tau2 domain/user-simulator or
leaderboard scores.

The refreshed [`m253 strict gate`](paper/results/raw/m253-workshop-gate-tau2-catalog-refresh-v1.json)
remains `ready: false` with nine missing official native receipts (AndroidWorld,
MobileSafetyBench, iOSWorld, OSWorld, OSWorld-V2, AgentNet, ToolSandbox, MCPMark, and
EnterpriseOps-Gym).  AppWorld adapter diagnostics do not substitute for those contracts, and the
existing public model/demo manifest is still the older authenticated artifact rather than this
new child checkpoint.

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
action-history projection.  The run uses `256` source-record-disjoint training rows and `64`
held-out rows from revision `6bda777c88d21e5a204703c1ee45597a8fa4f734`, with the tool-response and
non-action prose boundary recorded by the adapter manifest.  Warm continuation preserves compatible
weights and lowers held-out mean loss from `4.655` to `4.250`; random-backbone control remains near
chance (`9.781` to `9.474`).  Warm held-out token accuracy is `47.30%` versus `0.17%` for random,
but both arms remain at `0%` teacher-forced sequence exactness.

The stricter [`warm free-run receipt`](paper/results/raw/m277-toolace-action-history-warm-v1.json)
finds `20%` tool-name exactness, `0%` argument exactness, and `0%` complete action-history episodes
over 16 held-out rows.  The matched [`random free-run receipt`](paper/results/raw/m277-toolace-action-history-random-v1.json)
finds `16.67%` tool-name exactness and `3.33%` step exactness, so the random control wins this tiny
free-run slice despite its much worse teacher-forced loss.  This is a useful deployment warning:
backbone reuse is clearly valuable for representation learning, but it is not sufficient to adopt
the current tool heads or claim reliable multi-turn tool execution.

### Current AppWorld native baseline and public action-step continuation (m278)

The [`m278 baseline receipt`](paper/results/raw/m278-appworld-current-checkpoint-native-v1.json)
reruns the current `10,524,544`-parameter WebGPU checkpoint in the public AppWorld `0.2.0` data
environment.  The runner's own contract test passes, all six caller-selected tasks reset against
isolated task databases, and the native verifier executes; the model nevertheless replays `0/6`
actions and makes `0` API calls.  This is a current-checkpoint zero-action interface baseline, not
an AppWorld leaderboard or email/SMS/Spotify result.

To separate action-program length from routing, the [`m278 continuation report`](paper/results/raw/m278-appworld-action-step-sft-v1.json)
trains only on 24 public AppWorld train rows and evaluates on 12 disjoint public dev rows.  The
source is pinned to AppWorld data `0.2.0` (package `0.2.0.dev0`); credentials, bootstrap calls,
protected test data, and raw solution programs remain outside Git.  Warm continuation raises
held-out first-action token accuracy from `43.28%` to `58.70%` and lowers mean loss from `3.989`
to `2.596`, but sequence exactness remains `0/12`.

The paired [`m278 weight audit`](paper/results/raw/m278-appworld-action-step-weight-v1.json)
finds identical configuration/tokenizer/shapes and frozen action heads, with relative movement of
`0.434%` in the embedding, `0.248%` in attention/mixer, `0.306%` in the FFN, and `0.010%` in
normalization.  The [`m278 native adapter receipt`](paper/results/raw/m278-appworld-action-step-native-v1.json)
then enables strict one-call AST replay and schema grounding on 12 disjoint dev tasks.  The child
still selects no replayable `run_python`/API action: `0/12` native successes, `0` action replays,
and `0` native API calls.  The result is useful source-disjoint transfer evidence, but it does not
close the AppWorld, real-account, or WebGPU closed-loop publication boundary.

### AppWorld route/selector head-only repair and schema failure (m279)

The [`m279 head adapter report`](paper/results/raw/m279-appworld-head-adapter-v1.json) isolates
deployment-head learning from backbone transfer.  It uses the same 24 public AppWorld train rows
and 12 disjoint dev rows as m278, but performs one language-model update and 256 route/selector
updates.  Held-out route and dense-selector top-1 accuracy move from `0%` to `100%` on all 12
rows.  The [`m279 weight audit`](paper/results/raw/m279-appworld-head-weight-v1.json) shows the
intended movement pattern: the shared embedding, attention/mixer, FFN, and normalization groups
remain unchanged, while the serialized action-head group moves `65.93%` relative L2.

The [`m279 native receipt`](paper/results/raw/m279-appworld-head-native-v1.json) forces the
selector-first path and runs strict one-call AST/schema replay on the same 12 disjoint dev tasks.
The adapter now executes `9/12` real AppWorld API calls (`48` total requests including `27`
credential/bootstrap requests), but verifier success remains `0/12`.  The failed calls are a
valuable interface diagnosis: routing can be learned, while API-name and argument/schema grounding
still fail.  This is resettable native execution evidence, not an AppWorld leaderboard score or a
reason to ship the head adapter to WebGPU.

### Longer AppWorld action-code SFT and deployment transfer (m280)

The [`m280 SFT report`](paper/results/raw/m280-appworld-long-sft-v1.json) extends the same public,
source-disjoint AppWorld first-action projection to 256 body updates.  Held-out assistant-token
accuracy reaches `96.64%`, mean loss falls to `0.6915`, and teacher-forced sequence exactness reaches
`9/12` (`75%`) on the 12-row dev projection.  This is a real improvement over m278, but the native
[`run_python` receipt`](paper/results/raw/m280-appworld-long-native-runpython-v1.json) emits no
replayable action on the same prompts, demonstrating a sharp teacher-forcing/free-run gap.

Combining the m280 body with the m279 route/selector heads gives a second native control.  The
[`combined native receipt`](paper/results/raw/m280-appworld-long-heads-native-v1.json) executes all
`12/12` bounded API calls in resettable AppWorld fixtures (`48` requests, including `36` bootstrap
requests), but no complete task verifier passes.  The source-bound [`first-action exactness receipt`](paper/results/raw/m280-appworld-first-action-exactness-v1.json)
compares the translated code hashes with the public dev projection and finds `0/12` exact codes.
The body-only [`weight audit`](paper/results/raw/m280-appworld-long-weight-v1.json) shows relative
movement of `3.579%` in the embedding, `1.405%` in attention/mixer, `1.814%` in the FFN, and
`0.108%` in normalization; the combined [`head audit`](paper/results/raw/m280-appworld-long-heads-weight-v1.json)
adds the expected `65.93%` action-head movement.  The adoption conclusion is unchanged: longer
SFT improves representation and teacher-forced code likelihood, but schema-grounded free-run
execution and multi-step task completion remain open.

### Public-train AppWorld schema retriever control (m281)

The [m281 learned API-head report](paper/results/raw/m281-appworld-api-head-training-v1.json)
trains a five-class frozen-feature app.api head from the same public train rows. It reaches only
6/12 on disjoint dev prompts, so it is retained as a negative control rather than adopted.
The separate [m281 retriever report](paper/results/raw/m281-appworld-api-retriever-v1.json)
uses char-ngram nearest-neighbor examples from the public train projection; it has no learned model
weights and reaches 12/12 API-label accuracy on the same dev rows.

The retriever is wired into the strict evaluator as a schema-candidate restriction, with argument
fields learned only from the train traces (optional query is not hallucinated for unrelated APIs).
The [m281 native receipt](paper/results/raw/m281-appworld-retriever-native-v1.json) replays
12/12 one-step calls (48 requests including 36 fixture bootstrap requests), and the
source-bound [m281 first-action receipt](paper/results/raw/m281-appworld-first-action-exactness-v1.json)
matches all 12/12 translated code hashes against the public dev projection. Full task verifier
success remains 0/12: these tasks require multi-step programs, so this result establishes exact
first-action schema grounding and isolated execution, not complete AppWorld success or WebGPU
productivity readiness.

### Current-checkpoint native BrowserGym canary (m282)

The [m282 BrowserGym canary comparison](paper/results/raw/m282-browsergym-current-checkpoint-canary-v1.json)
runs the current `10.52M`-parameter WebGPU checkpoint in the pinned BrowserGym `0.14.3` /
MiniWoB test environment with Chromium revision `1117`.  Across 20 reset episodes covering five
task families, the accessibility-only arm records `0/20` success, `0.0` reward, and only `10/200`
grounded steps; the model emits `open_app` on 80 steps, `move_cursor` on 40, no parseable tool on
60, and other non-browser actions on the remainder.

The matched DOM-coordinate sidecar is intentionally non-official.  It recovers the deterministic
ascending-numbers family (`4/4`) but leaves bisect-angle, choose-list, click-button, and
click-button-sequence at `0/16`.  This isolates the failure: the checkpoint still needs a
browser-specific action policy and accessibility/target grounding; coordinate geometry alone does
not establish general browser control.  The result is a native canary, not the complete 240-episode
official plan, visual grounding, or an external-account result.

The complete [m282 official-plan receipt](paper/results/raw/m282-browsergym-current-checkpoint-official-v1.json)
then runs all `240` pinned episodes (`60` task families × `4` fixed seeds) with the same checkpoint,
BrowserGym/MiniWoB revisions, Chromium revision, and ten-step limit.  The official-split flag is
verified, but native success is `0/240`, reward is `0.0`, and `2,080/2,400` actions become
`noop(0)`.  Every task family is `0/4`; this is a complete, reproducible negative browser-control
result and closes the current-checkpoint BrowserGym gate against promotion.

### Current export and native WebGPU deployment audit (m283)

The [m283 export audit](paper/results/raw/m283-current-export-deployment-audit-v1.json) rebuilds
the HF-format and ONNX/WebGPU artifacts from the exact `bc1aca…` checkpoint rather than reusing
the stale generated files in the checked-in Space directory.  The HF bundle reloads with exact
PyTorch parity (`max_abs_diff=0`, argmax agreement `1.0`), and the WebGPU export passes the hard
fp32/fp16 graph parity gate.  A clean static Space directory verifies all eight generated files,
manifest hashes, and the `10,524,544`-parameter checkpoint identity.

Publication is still not claimed: `hf auth whoami` reports `Not logged in`, so no Hub model or
Space URL exists.  Two local Chromium probes (headless and headed) also fail closed because the
host exposes no non-empty WebGPU adapter identity; the export is ready, but native WebGPU
throughput and hardware capability remain unverified until run on a suitable GPU host.

### Workshop/publication gate re-audit (m284)

The [m284 gate receipt](paper/results/raw/m284-workshop-gate-current-v1.json) joins the pinned
catalog, current-checkpoint native receipts, export audit, and transfer ablation.  The decision is
`ready=false`, even though the structural checks for catalog coverage, MobileGym, the complete
BrowserGym/MiniWoB plan, the earlier Apple Metal WebGPU receipt, and the parent-head/random
ablation pass.  Nine required native surfaces remain unsupplied: AndroidWorld, MobileSafetyBench,
iOSWorld, OSWorld, OSWorld-V2, AgentNet, ToolSandbox, MCPMark, and EnterpriseOps-Gym.

The public manifest is a real, independently verified older byte-model/Space publication, but it
is not the current BPE checkpoint.  The current export is therefore a local reproducibility
artifact, not a public model release.  This distinction, plus the m283 adapter-identity failure,
keeps the workshop claim closed until the current checkpoint is uploaded and re-fetched
anonymously and the missing native suites are executed with their independent verifiers.

The computer-use source contract is now release-bound as well: OSWorld 2.0 requires the matching
`osworld-v2-2026.06.24` code, task classes, gated assets, and mocked websites.  Without the gated
snapshot and a resettable desktop VM, no OSWorld-V2 result is admitted to the WebGPU or workshop
claims.

### Extended public weight-transfer continuation (m286)

The [m286 receipt](paper/results/raw/m286-cross-surface-public-weight-transfer-v1.json) repeats
the matched source-disjoint AndroidControl/AgentNet experiment for `32` updates over `1,024`
public training rows and `64` held-out rows.  Warm-start token accuracy improves `51.81%`→`57.04%`
(`3.772`→`2.716` loss), with AndroidControl `60.91%`→`66.67%` and AgentNet `45.88%`→`50.76%`.
The random-backbone control reaches `8.71%` after the same budget.  Group movement is
`0.008%`–`0.374%` for the warm backbone versus `7.8%`–`119.7%` for random, so compatible
pretrained weights and a lower backbone learning rate remain the adopted policy.

This is still a teacher-forced text/accessibility result: exact held-out sequences remain `0/64`,
and no native emulator, browser, desktop VM, screenshot, MCP, or external-account side effect was
run.  The child is not promoted to the WebGPU release.

### ToolSandbox runtime-head transfer control (m288)

The [m288 receipt](paper/results/raw/m288-toolsandbox-runtime-head-transfer-v1.json) trains only
the route and dense candidate-selector heads against a pinned public ToolSandbox AST projection:
`107` source-disjoint training rows and `20` held-out rows, with the current `10,524,544`-parameter
body frozen.  The warm parent reaches `80%` selector top-1, `95%` selector top-3, and `100%`
app-action routing on the held-out rows, up from `45%`, `90%`, and `0%`.  The matched random-body
control reaches the same post-update scores, so this bounded projection is evidence of head
capacity and candidate-list regularity, not a claim of useful pretrained-backbone transfer.

Both arms move their heads by roughly `1.27`–`1.49` relative L2 and neither child is promoted.
The projection is AST-only; no ToolSandbox simulator, user simulator, verifier, external API, or
native environment was executed.  Native ToolSandbox evidence remains a publication blocker.

### Public Mind2Web grounding continuation and BrowserGym canary (m289)

The [m289 receipt](paper/results/raw/m289-mind2web-browsergym-transfer-v1.json) continues the
current BPE checkpoint on a public Mind2Web train-only DOM/action projection (`36` train
conversations / `219` decisions and `12` parent/typed-slot-disjoint held-out conversations /
`63` decisions).  Adding browser `target_id`/`value` pointer rows and 64 low-rate updates leaves
held-out exact pointer-span accuracy at `0/63`.

The compatible body moves only `0.286%`–`0.309%` across attention/FFN groups and `0.293%` in the
embedding group.  A separate pinned BrowserGym/MiniWoW accessibility-only canary executes `20`
episodes with `0/20` success.  This is a useful negative result for the WebGPU browser path, not an
official Mind2Web test score, full BrowserGym score, visual grounding result, or account-control
claim; the child is not promoted.

### Mind2Web browser-head and pointer transfer audit (m290–m291)

The [m290–m291 receipt](paper/results/raw/m290-mind2web-browser-head-pointer-v1.json) isolates
browser routing from argument grounding.  Frozen-body warm head transfer raises source-disjoint
Mind2Web selector top-1 `19.05%`→`92.06%` and top-3 `85.71%`→`100%`; the matched random head reaches
`85.71%` top-1.  Warm route/selector movement (`0.674`/`0.820` relative L2) is below random
(`1.253`/`1.133`).

The standard-schema pointer continuation adds `target`/`text` and reaches only `3/63` exact spans.
All three native 20-episode BrowserGym/MiniWoW canaries remain `0/20`; outputs reveal systematic
`type_text` placeholder or prompt-fragment copying.  Pointer precedence now prefers learned spans
when available, but the child is not promoted and native browser capability remains unverified.

### AndroidControl mobile dispatch transfer and native MobileGym canary (m292)

The [m292 receipt](paper/results/raw/m292-mobile-dispatch-transfer-native-v1.json) uses the public
AndroidControl mirror only as a text/action projection: `4,096` train rows and a separate balanced
`904`-row test file, with screenshot bytes omitted.  A frozen-body warm route/selector continuation
reaches `100%` route accuracy and `42.81%` selector top-1; the matched random-head control reaches
`38.94%`.  Warm selection is strong for open-app (`89.6%`) but weak for click (`8.0%`) and zero
for long-press and navigate-home, showing that the public action vocabulary is not equivalent to
screen grounding.

The warm child is evaluated in the pinned MobileGym simulator and independent state-diff judge on
20 official-test tasks.  It completes `0/20` tasks and collapses to `mobile_press_enter` on 19
episodes plus one `mobile_submit_answer`.  The parent’s first 20 tasks in the complete m262 run are
also `0/20` (with a different two-step limit), so this transfer does not improve executable mobile
control.  The result supports retaining parent weights as an initialization candidate while
rejecting this child for WebGPU promotion; it is not an Android emulator, screenshot, or real-device
score.

### m675 AndroidControl continuation from the m671 child

The [m675 AndroidControl receipt](paper/results/raw/m675-m671-androidcontrol-transfer-v1.json)
uses the pinned Apache-2.0 [public mirror](https://huggingface.co/datasets/OfficerChul/Android-Control-84k)
and its [Google Research AndroidControl project](https://github.com/google-research/google-research/tree/master/android_control).
The 512-row train and 256-row holdout are disjoint in the acquired projection, with screenshots
omitted. After identical 32-step SFT, the warm m671 child improves held-out token accuracy from
`74.483%` to `82.360%`; the matched random control improves `44.736%` to `62.851%`, leaving a
`19.510`-point warm advantage. Warm action heads remain frozen and shared-body movement stays
below `0.41%` relative per group, while exact sequence accuracy is `0%` for both. This supports
warm-start reuse for text/action continuation, not visual mobile control or native emulator
promotion.

### m676–m678 AndroidControl-trained native diagnostics and deployment

The m675 AndroidControl-trained child was run through the complete pinned MobileGym test split in
the [m676 receipt](paper/results/raw/m676-m675-mobilegym-native-v1.json): `256/256` tasks,
zero runner errors, and `1/256` success (`0.3906%`), unchanged from the m671 parent. The model
still emits predominantly navigation-back actions, confirming that text/action continuation did
not add visual or state grounding.

The [m677 preparation](paper/results/raw/m677-m675-hf-space-preparation-v1.json) and [m678 native
WebGPU receipt](paper/results/raw/m678-m675-webgpu-adoption-v1.json) bind the same child to a
fresh eight-artifact static bundle and real Chromium WebGPU execution. Apple Metal-3 reports `3/3`
exact local structured actions at `995.0` tok/s p50 and `9.85 ms` p50, with no side effects and
closed-loop success `0`. The lower throughput than m671 is measured evidence, not a reason to
claim production readiness; authenticated HF upload, visual grounding, and service-backed
Gmail/Notion/MCP evaluation remain open.

### m679 MCPMark trajectory continuation from the m675 child

The [m679 receipt](paper/results/raw/m679-m675-mcpmark-transfer-v1.json) evaluates the m675
AndroidControl-trained warm child and its matched random control on a pinned, redacted public
MCPMark trajectory projection: `10` train trajectories and `5` agent-disjoint eval trajectories
covering filesystem, Notion, GitHub, Playwright, and Postgres. Warm held-out token accuracy rises
from `26.039%` to `30.471%`; random rises from `17.636%` to `19.206%`, leaving an `11.265`-point
warm advantage after training. Exact sequence accuracy remains `0%` for both, and tool outputs and
assistant free text are redacted. This is evidence for retaining warm weights in text tool-call
continuation, not an official MCPMark score or live Notion/browser/database execution.

### m680 current-child EnterpriseOps-Gym email control

The [m680 receipt](paper/results/raw/m680-m679-enterprise-email-control-v1.json) evaluates the
exact m679 AndroidControl+MCP warm child against its matched random child on the public
[EnterpriseOps-Gym](https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym) email slice:
`67` rows, `20` candidate tools per row, and an immutable dataset revision. Warm retrieval is
`13.433%`/`40.299%`/`62.687%` at hit@1/3/5, while random is `22.388%`/`70.149%`/`89.552%`;
the warm-minus-random gaps are `-8.955`/`-29.851`/`-26.866` percentage points. This matched
negative control means the m679 cross-surface warm initialization should not be promoted for
email tool retrieval without task-family isolation or additional training. The adapter uses
generated name-only descriptions, drops server configuration and verifiers, and executes no
email action, so the result is neither official EnterpriseOps-Gym stateful success nor evidence
of real Gmail/Outlook side effects.

### m681 public Hugging Face WebGPU live probe

The [m681 receipt](paper/results/raw/m681-public-webgpu-live-probe-v1.json) verifies the public
[model](https://huggingface.co/danelcsb/localagent-tiny-30m-byte) and [Space](https://huggingface.co/spaces/danelcsb/localagent-webgpu)
by anonymous Hub metadata plus a live browser probe. The Space is publicly reachable and reports
`RUNNING`; the embedded model reaches `Model ready` with `WEBGPU`. A read-only request, “What does
ephemeral mean?”, completes in `818 ms` but routes to `web_search(term=ephemeral)` instead of the
expected `define` tool. This is a concrete public deployment-quality failure, not a reason to
hide the result.

The public model revision `d15db7…` and Space revision `3c07f7…` expose the same older 28M-byte
bundle, while the public config does not bind the current m679 checkpoint SHA. Therefore the
deployment is reachable but **legacy**, not a current-checkpoint publication. No credentials,
email/Notion writes, or external side effects were used; the probe does not establish realistic
service, mobile, browser, desktop, MCP, or visual-agent success.

### m682 current m679 WebGPU adoption

The [m682 receipt](paper/results/raw/m682-m679-webgpu-adoption-v1.json) regenerates a complete
current-checkpoint release from the m679 warm child (`10,524,544` parameters; checkpoint
`dbf45e…`). The model bundle, eight WebGPU artifacts, static Space staging, and ONNX/PyTorch
parity all verify against the same checkpoint. A fresh Chromium native WebGPU run on Apple
Metal-3 executes `3/3` structured local actions at `1,250` input tok/s p50, `7.45 ms` p50, and
`20.46 MB` conservative peak memory. Closed-loop success remains `0` because the harness performs
no real email, browser, Notion, or MCP action. HF authentication is absent, so both intended
repositories remain prepared-but-unpublished.

### m683 current m679 MobileGym regression

The [m683 receipt](paper/results/raw/m683-m679-mobilegym-native-v1.json) runs the exact m679
AndroidControl+MCP child through all `256` tasks of the pinned MobileGym official test split.
The state-diff judge reports `1/256` (`0.3906%`) with zero runner errors, exactly matching the
m675 AndroidControl child. The dominant output is still `mobile_navigate_back` (`255` actions),
with only six home navigations and one app open. This is a native simulator regression check over
DOM/text observations (`vision_used=false`), not visual Android control; the child is retained as
a negative control and is not promoted.

### m684 current m679 BrowserGym regression

The [m684 receipt](paper/results/raw/m684-m679-browsergym-native-v1.json) runs the exact m679
child through the complete pinned BrowserGym/MiniWoB plan: `240` fixed-seed episodes, BrowserGym
revision `9e779f…`, MiniWoB revision `7fd85d…`, and zero action errors. It passes `5/240`
(`2.0833%`), matching the earlier m672 result. The five successes are four simple
`click-button` cases and one `sign-agreement`; email-inbox, forms, scrolling, dragging, social,
and other multi-step families fail. Accessibility-tree text is used with no coordinate or
semantic fallback and no screenshots, so the receipt is a native text-grounding regression, not
visual computer-use or real email/Notion execution. The m679 child is not promoted.

### m685 current m679 stateful productivity probe

The exact m679 checkpoint was continued for 64 CPU updates on the disjoint local
`stateful_productivity_v1` suite: email send, Notion create, browser search, browser recovery, and
abstention. The pretrained-frozen-backbone arm reaches `93.75%` route accuracy, `46.67%` selector
top-1, `68.75%` closed-loop step success, and `40%` task completion; the matched random arm reaches
`81.25%`, `46.67%`, `68.75%`, and `40%`. Email and abstention complete, Notion partially completes,
and recovery remains `0%`. This supports retaining the pretrained backbone for another continuation,
but the matched control and failed recovery task explicitly do not justify native promotion. No
public benchmark text, emulator/browser/MCP server, account, or external side effect was used.

### m685 current m679 workshop gate

The [m685 gate receipt](paper/results/raw/m685-workshop-gate-current-m679-v1.json) joins the exact
m679 checkpoint hash to the current MobileGym, BrowserGym, and native WebGPU receipts. Five checks
pass: catalog coverage, runnable adapters, MobileGym execution, BrowserGym execution, and native
WebGPU capability/latency. Readiness remains `false`: AndroidWorld, Mobile Safety Bench, iOSWorld,
OSWorld/OSWorld-v2, AgentNet, ToolSandbox, MCPMark, and EnterpriseOps-Gym native receipts are not
present; the current RL and transfer-ablation lineage is stale; and the public HF model/Space still
does not bind the m679 checkpoint. This is an explicit publication stop, not an inferred approval.

### m686 current m679 RL preflight and gate refresh

The canonical one-update RL preflight now runs from the exact m679 checkpoint hash. It completes
two realized optimizer updates, executes a nonzero learning rate (`2e-5`), changes 40 named policy
parameters, and records three reward values across 32 attempted rollouts. The source is the local
deterministic productivity simulator only: no public benchmark text, emulator, browser service,
MCP server, account, or external side effect is involved. This satisfies the gate's current-RL
lineage requirement, but it is not a public benchmark result.

The [m686 gate](paper/results/raw/m686-workshop-gate-current-m679-v1.json) therefore passes six
checks: catalog coverage, runnable adapters, MobileGym, BrowserGym, native WebGPU capability, and
current RL preflight. It remains `ready: false` because nine native benchmark/service receipts are
missing, the transfer-ablation report is bound to an older checkpoint, and the public HF model/Space
does not bind m679.

### m690 public HF audit against m679

An anonymous read-only Hub audit downloads and hashes the public model and WebGPU Space artifacts.
Both endpoints return HTTP `200`, but the model revision `d15db7…` and Space revision `3c07f7…`
do not expose the exact m679 checkpoint hash (`current_checkpoint_match=false`). The public release
is therefore reachable but legacy; no upload was attempted without authentication, and no current
checkpoint publication claim is made.

### m691 current m679 AndroidControl retention and weight control

The [m691 receipt](paper/results/raw/m691-m679-androidcontrol-current-v1.json) reruns the public
AndroidControl mirror continuation from the exact m679 checkpoint, using `512` train rows and
`256` source-disjoint held-out rows at the pinned mirror revision. The text/accessibility projection
improves held-out teacher-forced token accuracy from `81.52%` to `82.68%` in both matched arms,
while exact sequence accuracy remains `0%`. The warm arm keeps action heads frozen and changes only
the shared body at small relative norms; the random-head control moves its action-head group by
`29.5%` relative L2. Since screenshots were omitted and no emulator was launched, this is a
current-checkpoint retention and weight-adoption diagnostic, not native AndroidControl, AndroidWorld,
or MobileGym success. The result does not change the fail-closed publication gate.

### m692 current m679 AppWorld API-trajectory continuation

The [m692 receipt](paper/results/raw/m692-m679-appworld-current-v1.json) continues the exact m679
checkpoint on `90` public AppWorld train tasks and a separate `6`-task public dev split. The export
removes bootstrap credentials and keeps bounded, redacted API summaries. Held-out teacher-forced
token accuracy rises from `63.90%` to `74.03%` in both matched arms, but exact multi-turn sequence
accuracy remains `0%`. The result supports using public AppWorld traces for API syntax and state
summary pretraining, while showing that syntax learning does not establish multi-step execution.
No AppWorld environment, email service, Notion service, or external account was executed, so this
does not clear the native AppWorld or productivity gates.

### Native AppWorld multi-step free-running control (m694)

The [m694 receipt](paper/results/raw/m694-m679-appworld-free-v1.json) extends the m693 native
probe from one action to up to four closed-loop API steps. The same six public dev tasks and four
checkpoint arms were reset independently; the model received only redacted response summaries and
could stop on malformed or repeated candidates. This protocol tests actual stateful planning rather
than teacher-forced continuation or a single schema-ranked call.

All four arms still complete `0/6` tasks. The m679 warm arm replays `9` actions, while both m692
arms replay `12`; importantly, the m692 warm and matched-random arms produce identical per-task API
sequences, so the AppWorld continuation did not improve closed-loop behavior over a random control.
Every arm eventually stops on a repeated candidate. The weight decision is therefore negative:
retain the m679 body only as an initialization candidate, do not promote the m692 child, and require
multi-step native success before WebGPU deployment claims. This remains a bounded diagnostic, not an
official AppWorld/AppWorld-UL score, and it covers neither Gmail nor Notion or external accounts.

### Native AppWorld execution probe and matched controls (m693)

The [m693 receipt](paper/results/raw/m693-m679-appworld-native-v1.json) closes that execution gap
for a bounded probe. The pinned public AppWorld `0.2.0` environment and its ground-truth verifier
were reset for six caller-selected dev tasks (`6bdbc26_1–3`, `396c5a2_1–3`); an oracle contract
passed, proving the verifier was live. Each arm was limited to one strict AST-checked API action,
with API candidates ranked by the checkpoint and bootstrap credentials injected only by the
resettable fixture.

The exact m679 warm checkpoint replayed `6/6` actions and the m692 AppWorld-trained warm child also
replayed `6/6`, but both scored `0/6` native task completion. The matched m679 and m692 random
controls replayed `0/6` valid actions and likewise scored `0/6`. This is a useful negative control:
teacher-forced trajectory gains did not transfer to stateful execution, and the one-action adapter
cannot claim multi-step AppWorld competence. No screenshots, Gmail/Notion tasks, external accounts,
or irreversible side effects were used; the result is not an official AppWorld/AppWorld-UL score and
does not clear the workshop gate.

### Longer AppWorld SFT with native replay (m696)

The [m696 receipt](paper/results/raw/m696-m679-appworld-sft-native-v1.json) tests whether the
m694 repetition failure is simply under-training. Both arms use the same `90` public AppWorld train
tasks and `6` disjoint dev tasks for `128` multi-turn SFT updates; the warm arm starts from the
exact m679 body and the control starts from a matched random backbone. The identical native
free-running protocol then allows up to four schema-translated API steps per task.

Warm held-out teacher-forced token accuracy rises from `63.90%` to `76.88%`; the random control
rises from `0%` to `50.57%`, leaving a `26.31`-point warm advantage. The warm shared-body movement
stays below `1.76%` relative L2, while the random body moves up to `118.6%`; compatibility and
tokenizer checks pass and action heads remain frozen. Despite that transfer signal, native task
completion is `0/6` for both arms, with repeated API candidates ending every trajectory. The weight
decision is therefore unchanged: reuse the m679 body only as an initialization candidate, do not
export the child, and require native multi-step success before claiming WebGPU productivity control.

### Frozen API-head intervention (m695)

The [m695 receipt](paper/results/raw/m695-m679-appworld-api-head-native-v1.json) isolates schema
routing from the language-model policy. A small API head is trained for `1,024` updates on the `90`
public train tasks while the m679 backbone stays frozen; a matched random-backbone head is trained
with the identical protocol. The warm head reaches `2/6` held-out first-action labels (`33.3%`) versus
`0/6` for random, but the same resettable native free-run still completes `0/6` tasks for both arms.

This is a deployment boundary: a learned route/schema sidecar can improve candidate ranking, but it
cannot replace state tracking, argument grounding, or multi-step policy learning. The sidecar is
therefore not exported or counted as WebGPU productivity capability.

### m689 current m679 ToolSandbox interactive diagnostic

Allowing up to eight model turns after tool results on the same `129` base scenarios produces
`27/129` exact (`20.93%`) with mean similarity `0.3415` and zero runner exceptions. The interactive
run still solves `0/82` multi-tool and `0/24` state-dependent scenarios; the strongest slice remains
insufficient-information (`26/28` exact). This isolates the current failure mode as state grounding
and action sequencing, not merely first-call routing. The result remains native diagnostic evidence,
not an official ToolSandbox score, because the upstream user simulator and official split are absent.

### m688 current m679 ToolSandbox native diagnostic

The exact m679 checkpoint now runs through all `129` pinned ToolSandbox base scenarios with the
native simulator and milestone verifier: zero runner exceptions, `30/129` exact (`23.26%`), and
mean milestone similarity `0.3409`. Insufficient-information tasks are the strongest slice
(`26/28` exact); state-dependent and multi-tool tasks remain `0/24` and `0/82` exact. This is a
more complete native diagnostic than the earlier smoke, but ToolSandbox still publishes no train/test
split and the upstream model-based user simulator was not executed, so the result remains eval-only
and does not clear the official-suite gate.

The [m688 gate](paper/results/raw/m688-workshop-gate-current-m679-v1.json) records this precisely:
ToolSandbox is no longer missing as an environment run, but remains blocked because its pinned
public repository defines no verified train/test split. The remaining native Android, iOS, desktop,
AgentNet, MCPMark, and EnterpriseOps-Gym receipts are still absent, and the public HF artifacts do
not bind m679.

### m687 current m679 weight-reuse ablation and gate refresh

The fresh m687 warm/random synthetic service-contract ablation is parent-bound to m679 with 28
generated rows and matched 32-step head training. Warm initialization preserves the shared body
exactly (`0%` embedding, attention/mixer, FFN, and normalization movement) while the random control
moves those groups by `118.6%`, `77.9%`, `87.8%`, and `8.7%`. Both arms pass configuration,
shape, and tokenizer compatibility checks. The evidence supports reusing the m679 backbone for
continuation and keeping the action heads separately trainable; it is not official MCPMark or live
service success.

The [m687 gate](paper/results/raw/m687-workshop-gate-current-m679-v1.json) now passes seven checks,
including current RL and transfer-ablation lineage. Readiness remains `false` only because the
native AndroidWorld, Mobile Safety Bench, iOSWorld, OSWorld/OSWorld-v2, AgentNet, ToolSandbox,
MCPMark, and EnterpriseOps-Gym receipts are absent and the public HF model/Space is still not bound
to m679.

### m671 public source refresh and current MCP continuation

The [m671 public snapshot audit](paper/results/raw/m671-public-dataset-snapshot-audit-v1.json)
resolves nine public Hugging Face revisions to immutable commits, original links, licenses, and
bounded file inventories. It is provenance-only: no mutable evaluation payloads were downloaded,
and sources without a disjoint split remain eval-only. The companion [m671 transfer receipt]
(paper/results/raw/m671-m666-mcp-current-transfer-v1.json) continues the exact m666 warm/random
children on `86` public MCP trajectory decisions with a deterministic `21`-decision,
agent-disjoint holdout. Warm held-out token accuracy improves `45.353%`→`55.927%`; random
improves `25.487%`→`39.844%`, leaving a `16.082`-point warm advantage after training. Exact
multi-action sequence accuracy is `0%` for both, and action heads remain frozen. This supports
weight reuse for text tool-call initialization, not native MCP execution or production safety.

### m672 current-child native browser and mobile diagnostics

The [m672 MobileGym receipt](paper/results/raw/m672-m671-mobilegym-native-v1.json) runs the exact
m671 warm child through all `256` tasks of the pinned official test split: `1/256` (`0.3906%`),
zero runner errors, and the independent state-diff judge enabled. The observation is a bounded
DOM/text projection (`vision_used=false`), so this is not screenshot-grounded Android control.

The [m672 BrowserGym receipt](paper/results/raw/m672-m671-browsergym-native-v1.json) runs the same
child through the complete fixed-seed `240`-episode BrowserGym/MiniWoB plan, passing `5/240`
episodes (`2.0833%`) with zero action errors. It records native environment execution and
accessibility-tree text observations only; the result is retained as a negative control and does not
claim visual computer use, email/Notion side effects, WebArena, or real-service execution. Together
these receipts are current-checkpoint evidence of the remaining
grounding and multi-step planning gap, not workshop promotion evidence.

### m673 local WebGPU/Hugging Face release preparation

The [m673 release-preparation receipt](paper/results/raw/m673-m671-hf-space-preparation-v2.json)
binds the m671 warm child (`10,524,544` parameters; checkpoint SHA `b5576dc8…`) to a regenerated
Hugging Face-compatible model bundle and static WebGPU Space staging directory. The eight-artifact
WebGPU bundle passes the hard ONNX/PyTorch parity gate, including fp16 logits argmax agreement, and
the staged Space contains the same checkpoint-bound bundle manifest. This is the strongest local
deployment evidence for the current child, but it is not a public release: no Hub repository was
created or uploaded because authenticated HF credentials are unavailable. An anonymous Hub audit,
visual grounding, and real Gmail/Notion execution remain required before publication or workshop
promotion.

### m674 native WebGPU adoption of the m671 child

The [m674 adoption receipt](paper/results/raw/m674-m671-webgpu-adoption-v1.json) binds the
checkpoint-bound m673 release preparation to a real Chromium native WebGPU run. Apple Metal-3
reported `3/3` exact structured actions (email-shaped, URL, and Notion-shaped), `1,454.5` input
tokens/s p50, `7.4 ms` p50 latency, and `20.46 MB` conservative peak memory. The harness executed
no external side effects and closed-loop success is `0`, so these are local dispatch/capability
measurements—not Gmail, browser navigation, Notion, or MCP task completion. Public Hub upload and
visual/service-backed promotion gates remain open.

### Matched random-backbone ToolSandbox control (m636)

To separate checkpoint transfer from the native ToolSandbox protocol, the [m636 receipt](paper/results/raw/m636-m626-toolsandbox-matched-random-control-v1.json)
replays the same pinned ToolSandbox scenario names with the matched random m626 backbone. The
bounded control covers `25` scenarios (the warm run covers the full `129`-scenario unaugmented
base matrix); both use the native simulator and milestone verifier, with no model-based user
simulator, external API, or official split claim. The random control reaches `4/25` exact
verifier matches (`16.0%`) versus `5/25` (`20.0%`) for the warm checkpoint on the identical
subset. All four random exact matches are insufficient-information refusals, while the warm
checkpoint gets `5/5` on that category. This is a useful negative control for protocol sanity,
not a standalone weight-transfer claim: the matched subset is abstention-heavy and excludes the
full ToolSandbox distribution, so the m631 MCP trajectory ablation remains the stronger
current-checkpoint initialization evidence.

### Current m626 EnterpriseOps-Gym email retrieval (m637)

The [m637 receipt](paper/results/raw/m637-m626-enterpriseopsgym-current-email-retrieval-v1.json)
replays the public [ServiceNow-AI/EnterpriseOps-Gym dataset](https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym)
at revision `c8e538e…` on the current m626 checkpoint. On the pinned `67`-row email slice,
name-only dense-selector retrieval reaches `59.70%` top-1, `85.07%` top-3, and `94.03%`
top-5. This is a current-checkpoint email/tool-use diagnostic and improves over the older m592
warm-child top-1 result (`56.72%`), but it drops verifiers, server configuration, credentials,
and side effects; it is not official stateful EnterpriseOps-Gym task success and does not clear
the native enterprise benchmark gate.

The matched [m638 weight-transfer receipt](paper/results/raw/m638-m626-enterpriseopsgym-warm-random-transfer-v1.json)
repeats the same `67` rows with the m626 random backbone. Warm minus random gains are
`+20.90` points at top-1, `+8.96` at top-3, and `+8.96` at top-5. Because the candidate pool,
source hashes, tokenizer, and selector protocol are held constant, this is direct evidence for
retaining the pretrained weights for email/tool retrieval. It still cannot substitute for the
missing stateful EnterpriseOps-Gym server/verifier run.

### Current m626 AgentNet computer-use control (m640)

The [m640 receipt](paper/results/raw/m640-m626-agentnet-current-text-control-v1.json) evaluates
the public [xlangai/AgentNet projection](https://huggingface.co/datasets/xlangai/AgentNet) on the
current checkpoint. The full warm text projection covers `16` held-out parent tasks and `257`
action rows: first-action type rate is `75%`, mean total score `0.02326`, exact trajectory rate
`0%`, and task success `0%`. A matched four-parent random control has `0%` first-action type
rate and mean total `0.0`, giving a bounded warm advantage of `+75` points and `+0.01499` total
score. The full random arm was stopped after malformed output grew beyond the practical decode
budget, so no full-random score is claimed. Screenshots, desktop state, and AgentNetBench were
not executed; this confirms the visual-grounding/runtime gap rather than native desktop success.

### Cross-surface weight-adoption decision (m641)

The [m641 summary](paper/results/raw/m641-m626-weight-adoption-summary-v1.json) consolidates
the current warm/random evidence without averaging incompatible metrics. Warm initialization wins
by `+19.28` points on AndroidControl token accuracy, `+45.19` on the MCP trajectory projection,
and `+20.90` top-1 points on EnterpriseOps-Gym email retrieval; the bounded AgentNet control also
shows a `+75`-point first-action-type advantage. Across AndroidControl and MCP continuation, mean
relative shared-body movement is only `0.65%` in embeddings, `0.34%` in FFN, `0.27%` in attention,
and `0.018%` in normalization, while action heads remain fixed. The resulting policy is to reuse
the current warm backbone, use a lower learning rate for transferred body tensors, initialize new
heads from a controlled seed, and give heads a larger rate than the body. Native guardrails still
prevent export as a claimed agent: MobileGym is `1/256` and BrowserGym/MiniWoW is `5/240`.

### Mixed mobile/computer/tool continuation from the current warm checkpoint (m642)

The [m642 receipt](paper/results/raw/m642-m626-mixed-cross-surface-warm-random-v1.json) runs a
source-local parent/slot-disjoint continuation over public AndroidControl, AgentNet, and redacted
MCPMark projections: `138` selected train rows and `133` selected evaluation rows. After the same
`16`-step budget, warm initialization reaches `82.82%` AndroidControl, `74.93%` AgentNet, and
`27.84%` MCP token accuracy; the matched random arm reaches `1.92%`, `0.04%`, and `0%`. Warm
minus random gains are therefore `+80.89`, `+74.89`, and `+27.84` points, with a weighted
aggregate advantage of `+59.59` points. Warm AndroidControl moves slightly downward during the
mixed update (`83.37%` to `82.82%`), so the result supports the initialization and source-balancing
policy but not unrestricted continued training. Exact sequence accuracy remains `0%`, and no
native emulator, desktop, browser, or MCP service was executed.

### Current m626 Mind2Web browser continuation (m643)

The [m643 receipt](paper/results/raw/m643-m626-mind2web-browser-continuation-v1.json) adds the
source-linked [Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web) DOM/action projection:
`96` source-parent-disjoint train rows and `32` held-out rows. After the same `16`-step budget,
the warm checkpoint improves from `85.99%` to `86.99%` token accuracy, while the matched random
arm reaches `5.66%`; warm minus random is `+81.33` points. Exact sequence accuracy remains `0%`.
This supports retaining the warm browser initialization, but the receipt explicitly keeps the
native BrowserGym gate separate: no live site, screenshot grounding, or browser side effect ran.

### Mind2Web-to-native BrowserGym canary (m644)

The [m644 receipt](paper/results/raw/m644-mind2web-browsergym-native-canary-v1.json) pairs the
m626 baseline with the m643 Mind2Web-trained child on the same `16` fixed-seed MiniWoB episodes,
the pinned BrowserGym revision, and the `realistic_browser` vocabulary. Both arms pass `4/16`
(`25%`), all on `miniwob.click-button`; all normalized step traces are identical and the success
delta is `0`. This is important negative transfer evidence: the `+81.33`-point Mind2Web text
projection gain does not transfer to live MiniWoB state without a compatible grounding/action
bridge. The canary is intentionally non-official because it uses the realistic-browser diagnostic
pool and is limited to 16 episodes; the full native BrowserGym gate remains the separate m632
receipt.

### Grounding bridge diagnostic for the native canary (m645)

The [m645 receipt](paper/results/raw/m645-mind2web-browsergym-grounding-canary-v1.json) repeats
the same paired `16`-episode canary with two explicit grounding adapters. Reading only the live
accessibility tree (semantic fallback) leaves both checkpoints at `4/16` (`25%`), with identical
normalized traces and no child gain. Reading the live DOM's clickable geometry (coordinate
fallback) raises both arms to `8/16` (`50%`), again with identical traces and zero child gain.
This isolates the bottleneck: the improvement is supplied by an environment-side geometry bridge,
not by Mind2Web policy transfer. Both runs are bounded, non-official diagnostics—no screenshots,
real accounts, email/Notion side effects, or official BrowserGym score are claimed.

### Current m626 gate with grounding evidence (m646)

The [m646 receipt](paper/results/raw/m646-workshop-gate-current-m626-grounding-v1.json) refreshes
the fail-closed publication gate from the exact m626 native MobileGym/BrowserGym/ToolSandbox,
WebGPU, transfer, and RL receipts and attaches m645's paired grounding result. The gate still
reports `ready: false` with ten blockers: missing AndroidWorld, MobileSafetyBench, iOSWorld,
OSWorld, OSWorld-V2, AgentNet, MCPMark, and EnterpriseOps-Gym native receipts; ToolSandbox's
unverified official split; and the missing authenticated public model/demo manifest. The m645
diagnostic changes no gate status: its DOM-coordinate improvement is an environment-side bridge,
not an official benchmark or Mind2Web transfer score.

### Current AppWorld public action-step continuation (m647)

The [m647 receipt](paper/results/raw/m647-appworld-current-transfer-v1.json) adds a fresh,
source-linked AppWorld `0.2.0` train/dev slice: `64` public train tasks and `18` disjoint public
dev tasks, with protected test data excluded. A 32-step continuation from the warm m626
checkpoint improves held-out first-action token accuracy from `45.17%` to `58.41%`; the matched
random-backbone arm improves from `4.11%` to `27.28%`, leaving warm ahead by `31.13` points.
Exact sequence accuracy remains `0%` and route/selector accuracy remains `0%` for the warm child.
Weight movement is transfer-shaped (warm embedding/attention/FFN `0.433%/0.224%/0.275%`, with
action heads unchanged), so the current recommendation is to retain the warm body but train a
separate schema/action grounding head before any native AppWorld or WebGPU promotion. This is
teacher-forced first-action evidence only; no external email, SMS, app account, or protected test
result is claimed.

### AppWorld route/selector head adaptation and native replay (m648)

The [m648 receipt](paper/results/raw/m648-appworld-head-native-v1.json) isolates deployment-head
learning on the same `64/18` AppWorld train/dev projection. A 256-step route/selector update
raises both warm route accuracy and selector top-1 accuracy to `100%` on the held-out projection;
the matched random-body arm reaches the same ranking metrics. Only the serialized action-head
group moves (`37.40%` warm, `40.39%` random); embedding, attention, FFN, and normalization remain
unchanged. However, paired resettable native replay on six public dev tasks remains `0/6` for both
baseline and head-adapted checkpoints, with zero native API calls. This cleanly separates candidate
ranking from free-run action generation: keep the warm body and train the heads, but do not promote
the head adapter as native AppWorld success or WebGPU-ready email/tool control.

### AppWorld runtime-pool correction and API-schema head (m649–m650)

The [m649 receipt](paper/results/raw/m649-appworld-runtime-pool-v1.json) closes a runtime
configuration confound in that result. Direct probes ranked `run_python` correctly, but the default
`retrieve_k=10` lexical retriever could remove that tool before the dense selector. Replaying the
same six public dev tasks with the full `STANDARD_TOOLS` pool (`retrieve_k=100`) dispatches and
replays `6/6` actions, while native task success remains `0/6`: the bounded adapter executes only
one API step, whereas these AppWorld objectives are multi-step and stateful. The corrected result
therefore fixes the dispatch-interface diagnosis without changing the publication gate.

The [m650 receipt](paper/results/raw/m650-appworld-api-head-v1.json) then trains a frozen-backbone
`app.api` schema head on the current warm child: `64` public train rows and a disjoint `15`-row
seen-label dev slice (the remaining public dev rows contain labels absent from train and are not
silently scored). The head reaches `60%` held-out API-label accuracy and constrains all six native
replays to schema-valid Spotify calls, but one-step native success is still `0/6`. This supports
retaining the warm body plus separate schema heads, not claiming complete AppWorld, email, Notion,
or WebGPU agent success; the next required experiment is a multi-step native evaluator with
trajectory-level state and exact public task verifiers.

### AppWorld bounded trajectory continuation and weight adoption (m651)

The [m651 receipt](paper/results/raw/m651-appworld-trajectory-transfer-v1.json) converts the same
public AppWorld train/dev split into `64/18` multi-turn ground-truth API trajectories, capped at
16 non-bootstrap actions per row. Credentials are removed and tool observations are deterministic
`status/api/step` summaries, so this is controlled trajectory-learning supervision rather than a
leak of task databases or a native score. On the disjoint dev slice, the warm m649 child improves
teacher-forced assistant-token accuracy from `56.15%` to `64.50%` (`+8.35` points); the matched
random-body arm improves from `16.75%` to `37.12%` (`+20.37` points), leaving warm ahead by
`27.38` points. Exact multi-turn sequence accuracy remains `0%` for both.

The weight audit supports a low-rate shared-body/high-rate head recipe: warm embedding,
attention/mixer, and FFN movement is only `0.233%/0.137%/0.162%`, while action-head movement is
`26.35%` (random: `0.325%/0.145%/0.175%` body and `25.75%` heads). This is the strongest current
transfer signal, but it is not native AppWorld, email, Notion, browser, or WebGPU success; a
free-running multi-step policy with unredacted verifier-backed observations is still required.

The [m652 receipt](paper/results/raw/m652-appworld-trajectory-native-v1.json) runs that negative
control directly: six paired public dev tasks, eight free-running steps per task, full candidate
pool, resettable AppWorld state, and strict one-literal-API translation. The warm child replays
`3` actions and the random child `15`, but both score `0/6` native task success. The warm policy
repeats `spotify.show_song_queue` or fails required-argument grounding, showing that the
teacher-forced trajectory gain does not yet transfer to a stateful free-run policy. The current
recommendation remains to reuse the warm body only as an initialization and keep native promotion
blocked until action/schema heads are trained against actual state observations and completion calls.

### AppWorld safe rich-state trajectory ablation (m657)

The [m657 receipt](paper/results/raw/m657-appworld-rich-state-v1.json) replaces the prior empty
tool observations with bounded response summaries containing useful IDs, titles, counts, and
schema values while redacting credentials, addresses, canaries, and tokens. It uses the same public
`64/18` train/dev split and matched warm/random initialization, capped at three actions per row to
fit the WebGPU context. Warm held-out teacher-forced accuracy reaches `63.22%` versus `35.63%` for
random, but all `18` dev rows are still truncated by the 2,048-token context and the native six-task
probe remains `0/6` for both arms (`3` warm actions versus `10` random). Rich observations improve
the learning signal, yet the result confirms that response summarization alone does not solve
stateful planning, argument grounding, or completion-call generation.

### AppWorld compact state sketch (m659)

The [m659 receipt](paper/results/raw/m659-appworld-compact-state-v1.json) applies a stricter
state sketch: contact/song/message IDs, names, titles, relationships, counts, and message text are
retained; addresses, timestamps, tokens, credentials, and low-value metadata are dropped. This fits
the 2,048-token model window for `35/36` trajectory rows. Warm held-out token accuracy is `63.49%`
versus `35.89%` for random, and warm native action replay rises from `3` to `5` on the paired six
task probe. However, verifier success remains `0/6`, exact sequence accuracy remains zero, and the
policy still lacks completion/state-planning behavior. The compact sketch is retained as a WebGPU
input-format candidate, not promoted as agent capability.

The [m661 receipt](paper/results/raw/m661-appworld-api-head-native-v1.json) isolates the schema
head itself. A frozen trajectory-level `app.api` head reaches `42.86%` on seen-label public dev
prefixes and raises replayed actions from `5` to `8` on the paired six-task probe, but verified
success remains `0/6`. The head therefore helps candidate restriction without solving first-action
grounding, repeated-call avoidance, or completion planning; it remains diagnostic only.

### AppWorld short-task completion probe (m662)

The [m662 receipt](paper/results/raw/m662-appworld-short-completion-v1.json) narrows the next
failure question using only public, split-disjoint AppWorld tasks. Six train tasks and six dev tasks
fit a four-action window and retain the real `supervisor.complete_task` call; the train set covers
phone→Venmo payment and Simple Note updates, while dev covers Spotify lookup/queue workflows. The
normalizer keeps bounded IDs, names, titles, counts, and messages while removing bootstrap
credentials and low-value state, and the source remains the public AppWorld `0.2.0` release.

Warm continuation from the current compact-state child raises held-out teacher-forced token
accuracy from `55.35%` to `66.06%`; the matched random-body arm reaches `43.51%`. Exact sequence
accuracy is `0%` for both. In the strict resettable native probe, completion is opt-in and only a
literal `supervisor.complete_task(status='success')` candidate is added—no answer or ground-truth
action is injected. Both arms still score `0/6` native task success (warm replays `11` actions,
random `12`). This separates a measurable token-learning gain from actual short-task completion:
the current blocker is action selection/state planning, not only long-horizon length. The warm
body remains the preferred initialization, but no native email/Notion/WebGPU promotion is justified.

### AppWorld state-grounding and persisted native-control audit (m663)

The [m663 receipt](paper/results/raw/m663-appworld-grounding-v1.json) closes two evaluator gaps
exposed by the short-task probe. The compact response sketch now preserves answer-relevant
`follower_count`, and native API calls execute through AppWorld's persisted `world.execute()`
boundary rather than only mutating in-memory objects. This makes the native verifier read the same
state that the model action changed; credentials are resolved inside the sandbox and never embedded
in generated action code.

The warm 24-step continuation reaches `65.15%` held-out teacher-forced token accuracy versus
`37.70%` for the matched random arm, with shared-body movement of `0.345%/0.187%/0.224%` for
embedding/attention/FFN and unchanged action heads. The strict free-running six-task probe remains
`0/6` for both arms because the first API choices are wrong. A separately labeled control that
replays the public ground-truth API prefix and asks only for the final completion succeeds `6/6` for
both arms, including answer extraction from bounded live state. This isolates the remaining gap to
state-conditioned action selection and multi-step planning; it is not a free-running score and does
not authorize WebGPU email/Notion publication.

### Current m663 workshop/publication gate (m664)

The [m664 receipt](paper/results/raw/m664-workshop-gate-current-m663-v1.json) re-runs the
fail-closed gate against the exact `m663` warm child (`bb19d72e...`) and attaches the corrected
AppWorld grounding audit. Only the catalog and previously measured native WebGPU capability pass
without qualification. The older MobileGym, BrowserGym, ToolSandbox, and RL receipts are rejected
for checkpoint mismatch (ToolSandbox also lacks official-split verification); the m663 movement
files are intentionally rejected as a non-canonical transfer/no-transfer pair; and the public
manifest has no current-checkpoint binding. Fourteen blockers remain, including AndroidWorld,
MobileSafetyBench, iOSWorld, OSWorld, OSWorld-V2, AgentNet, MCPMark, EnterpriseOps-Gym, and the
authenticated current model/demo manifest. This is the correct decision: m663 is useful diagnostic
evidence, not workshop or Hugging Face publication approval.

### Full public AppWorld continuation and schema-planner control (m666)

The [m666 receipt](paper/results/raw/m666-appworld-public-full-v3.json) expands the supervised
source from the earlier 64-task slice to all `90` public AppWorld train tasks, while retaining six
disjoint public dev tasks and excluding the protected test split. This adds examples for the
held-out `spotify.show_artist` and `spotify.search_songs` API families. A matched 64-step
continuation raises warm held-out teacher-forced accuracy from `65.15%` to `73.12%`; the random
control reaches `50.34%`. Warm shared-body movement is `0.895%/0.384%/0.478%` for
embedding/attention/FFN with frozen action heads, versus `1.295%/0.427%/0.556%` for random.

The exact free-running model ranking remains `0/6` for both children. A separate, explicitly
environment-side schema-planner control—multi-token name preservation, live ID extraction,
stateful API dependency ordering, and bounded completion—replays `6/6` for both arms. Because the
control is identical across warm and random and is not learned model output, this is executor and
argument-grounding evidence, not a model score or promotion. The learned action-selection gap and
the fail-closed workshop/Hugging Face gates therefore remain open.

### Canonical m666 warm/random weight ablation (m668)

The [m668 receipt](paper/results/raw/m668-appworld-weight-ablation-v1.json) seals the existing
warm/random movement reports into the canonical combined-ablation envelope required by the
workshop checker. It binds the exact current warm child, the 90/6 public AppWorld split, shared
configuration and tokenizer compatibility, and held-out metrics for both arms. The structural
weight requirement now passes; the receipt still recommends retaining parent initialization only
as a candidate because learned free-running success remains `0/6`.

### Learned AppWorld API-head routing diagnostic (m669)

The [m669 receipt](paper/results/raw/m669-appworld-api-head-diagnostic-v1.json) trains a frozen
backbone `app.api` classifier on all `90` public train rows and evaluates it on the six disjoint
public dev tasks. It reaches `100%` train-label exactness but only `2/6` dev-label exactness. In
selector-first native replay, three API actions are executed, yet complete task success remains
`0/6`. This is useful negative evidence: a small learned schema head does not replace stateful
route selection, observation grounding, or multi-step planning. The head is not promoted to the
WebGPU bundle.

### Current m666 public MCP continuation and source snapshot (m671)

The [m671 source snapshot](paper/results/raw/m671-public-dataset-snapshot-audit-v1.json) resolves
nine public Hugging Face revisions without downloading mutable evaluation payloads, retaining each
original source link, revision, license metadata, and file inventory. The [m671 transfer receipt](paper/results/raw/m671-m666-mcp-current-transfer-v1.json) then uses the Apache-2.0
`obaydata/mcp-agent-trajectory-benchmark` at revision `f4f449d...`: `86` public train rows and
`21` deterministic agent-disjoint internal-holdout rows, with tool outputs and reasoning excluded.
From the exact m666 warm child, held-out assistant-token accuracy rises `45.35%→55.93%`; the
matched random child rises `25.49%→39.84%`, leaving a `16.08` percentage-point warm advantage.
Both exact multi-action sequence rates remain `0%`. All shared tensors are shape/config/tokenizer
compatible and action heads are frozen; embedding/attention/FFN movement is small, supporting a
low-rate transferred backbone plus separately trained action heads. This remains teacher-forced
projection evidence, not official MCPMark/Toolathlon, native MCP server execution, or real
email/Notion side effects.

### Current m666 native mobile/browser reruns (m670)

The [m670 MobileGym receipt](paper/results/raw/m670-m666-mobilegym-native-v1.json) and
[m670 BrowserGym receipt](paper/results/raw/m670-m666-browsergym-native-v1.json) execute the
exact m666 warm checkpoint (`8c3a4ed3...`) on the complete pinned public splits. MobileGym runs
`256/256` official test tasks with zero errors and passes `1/256` (`0.39%`). BrowserGym/MiniWoB
runs `240/240` fixed-seed episodes with zero action errors and passes `5/240` (`2.08%`), limited
to four `click-button` and one `sign-agreement` episode. Both are accessibility/DOM text
projections with `vision_used=false`; email-inbox, form, drag, scroll, and social task families
remain unsuccessful. These are current native diagnostics, not visual-agent or real-account
success claims.

### Current m666 workshop/publication gate (m667 v3)

The [m667 receipt](paper/results/raw/m667-workshop-gate-current-m666-v3.json) binds the full
public AppWorld continuation to the exact `m666` warm child (`8c3a4ed3...`) and re-runs the
fail-closed workshop gate. The gate remains `ready: false`: ToolSandbox and RL receipts are from
older checkpoints; ToolSandbox additionally lacks official-split
verification; and the public model/demo manifest is not bound to this child. AndroidWorld,
MobileSafetyBench, iOSWorld, OSWorld, OSWorld-V2, AgentNet, MCPMark, and EnterpriseOps-Gym still
have no native receipts. The m666 schema-planner `6/6`
control is explicitly executor-side and cannot satisfy learned-policy, native, or Hugging Face
publication requirements. Eleven blockers remain after the canonical weight-envelope repair and
current MobileGym/BrowserGym reruns;
this is an auditable diagnostic checkpoint, not workshop approval.

### Full native BrowserGym continuation on the m626 child (m632)

The [m632 receipt](paper/results/raw/m632-m626-browsergym-native-full-v1.json) executes the exact
`10,524,544`-parameter AndroidControl-adapted checkpoint through all `240` fixed-seed episodes of
the pinned BrowserGym `0.14.3` / MiniWoB official plan. The BrowserGym and MiniWoB revisions,
Playwright `1.44.0`, Chromium revision `1117`, and checkpoint hash are recorded in the receipt;
the official split is verified and the runner reports no action errors. The child passes `5/240`
episodes (`2.0833%`), all in `miniwob.click-button` (`4`) or `miniwob.sign-agreement` (`1`).

This is a native browser-control result, but it is explicitly text/accessibility-tree only:
`vision_used=false`, coordinate and semantic fallbacks are disabled, and `211` grounded steps are
outnumbered by `2,040` no-op/ungrounded steps. It is therefore a reproducible grounding diagnostic,
not a visual BrowserGym, WebArena, real Gmail/Notion, or external-account result. The failure
pattern is actionable: the current transfer can emit a simple grounded click, but lacks the visual
and multi-step state representation needed for forms, email-like pages, scrolling, drag tasks, and
structured UI reasoning.

### Current m626 RL preflight and refreshed workshop gate (m633)

The [m633 RL receipt](paper/results/raw/m633-m626-stateful-rl-preflight-v1.json) runs the strict
two-step one-update preflight from the m626 warm checkpoint on the isolated deterministic
email/Notion/browser state machine. Train/eval rows are disjoint; `32` rollouts and `2` optimizer
updates execute with learning rates `[0.0, 2e-5]`, all `40` policy tensors change, and held-out
mean reward moves from `0.0` to `0.1` while exact match and strict tool-format validity remain
`0`. This passes the lineage/optimizer protocol check only; it is not native API control, a public
benchmark score, real-account execution, or a production RL checkpoint.

The [m633 gate receipt](paper/results/raw/m633-workshop-gate-current-m626-v1.json) now has seven
checkpoint-bound passes: catalog coverage, native MobileGym, native BrowserGym/MiniWoB, native
WebGPU capability/latency, current warm/random transfer, and current RL preflight. Readiness
remains `false` with ten explicit blockers: AndroidWorld, MobileSafetyBench, iOSWorld, OSWorld,
OSWorld-V2, AgentNet, ToolSandbox, MCPMark, EnterpriseOps-Gym, and the authenticated public
model/demo manifest. The publication decision is intentionally unchanged by the new receipts;
the remaining native suites and public URLs must be supplied before a workshop or HF release claim.

### Current m626 ToolSandbox base-matrix replay (m635)

The [m635 receipt](paper/results/raw/m635-m626-toolsandbox-native-base-v1.json) runs the current
checkpoint through all `129` unaugmented scenarios exposed by the pinned Apple ToolSandbox source,
using its real simulator and milestone verifier with a bounded scripted user. It records `27/129`
exact verifier successes (`20.93%`) with zero runner exceptions. Success is highly concentrated in
the `INSUFFICIENT_INFORMATION` slice (`26/28`, `92.86%`); canonicalization (`0/59`), multiple-tool
(`0/82`), multiple-user-turn (`0/28`), and state-dependency (`0/24`) slices remain unsolved.

This is stronger current stateful-tool evidence than the earlier three-scenario smoke, but it is
not an official ToolSandbox score: the source expands to `1,032` generated variants, does not define
a train/test split, and the upstream model-based user simulator/external APIs were not run. The
[m635 gate](paper/results/raw/m635-workshop-gate-current-m626-v1.json) therefore records the
ToolSandbox receipt as current evidence while retaining the explicit `official_split_not_verified`
blocker; readiness remains `false`.

The [m293 selector-first ablation](paper/results/raw/m293-mobilegym-selector-first-canary-v1.json)
forces the learned top candidate instead of allowing LM re-ranking.  It remains `0/20` with the
same `19` press-enter and one submit-answer collapse.  This is useful diagnosis: MobileGym's
task-state distribution is not recognized by the AndroidControl-trained selector, so a decoding
policy change alone cannot close the mobile grounding gap.  Selector-first remains opt-in and the
child is rejected for deployment.

### Current-checkpoint native ToolSandbox smoke and interactive control (m297)

The [m297 receipt](paper/results/raw/m297-toolsandbox-current-native-smoke-v1.json) is the first
current BPE-checkpoint ToolSandbox native run in this continuation.  The pinned simulator and
milestone verifier execute successfully: a bounded one-step smoke passes `2/3` settings/message
scenarios, with exact `cellular_off` and `wifi_off` milestones but only `0.425` similarity for the
phone-message schema.  A bounded interactive scripted-user run passes `0/3`, showing that the
current tool schema does not survive multi-turn feedback even when native execution is available.

Because the official split and model-based user simulator were not executed, this result is a
native smoke diagnostic rather than an official ToolSandbox score.  It is nevertheless stronger
than the prior text-only projection and keeps the adoption decision negative for the current
checkpoint.

The [m298 workshop gate re-audit](paper/results/raw/m298-workshop-gate-current-toolsandbox-v1.json)
now includes this native receipt and records the exact blocker `official_split_not_verified`; the
overall gate remains closed because the official ToolSandbox split and the other native surfaces are
still absent.

### Current-checkpoint ToolSandbox base-matrix replay (m300)

The [m300 receipt](paper/results/raw/m300-toolsandbox-current-base-v1.json) reruns the same
current 10.52M BPE checkpoint over every one of the 129 pinned base/no-distraction scenarios
(zero simulator or verifier exceptions).  It completes `29/129` (`22.48%`) exact milestones.
The result is sharply category-dependent: insufficient-information cases reach `26/28`, while
multiple-tool-call, multiple-user-turn, canonicalization, and state-dependency cases have no exact
completions under the one-step scripted-user protocol.  This isolates the current failure mode as
stateful continuation and argument grounding rather than package availability.

This is the complete upstream base set, not the 1,032 runtime variants: distraction and schema
scramble variants remain unevaluated, and the model-based user simulator is still absent.  Therefore
m300 strengthens native diagnosis but cannot satisfy the official ToolSandbox/publication gate.

The [m301 gate](paper/results/raw/m301-workshop-gate-current-toolsandbox-base-v1.json) joins m300
without relaxing the contract: ToolSandbox is still blocked specifically by
`official_split_not_verified`, while the catalog, current MobileGym, BrowserGym/MiniWoB, WebGPU,
weight-ablation, and existing public-artifact checks remain independently visible.

### Current-checkpoint binding enforcement (m317)

The [m317 gate](paper/results/raw/m317-workshop-gate-current-checkpoint-bound-v1.json) closes a
lineage gap in the final join: when a current checkpoint is supplied, every native receipt must
carry the same SHA-256 directly or under its checkpoint object.  The current MobileGym receipt
(`1/256`) and BrowserGym/MiniWoB receipt (`0/240`) therefore pass only after binding to
`bc1aca…`; older receipts without a checkpoint identity are no longer eligible by accident.  The
gate remains `ready=false` with ten blockers: nine required native families are still absent and
the public Hub manifest is a legacy byte-model release without the current checkpoint binding.

### Source-corrected current gate (m318)

The [m318 gate](paper/results/raw/m318-workshop-gate-source-corrected-v1.json) repeats the same
join after correcting two catalog citations discovered during an official-source audit:
AndroidWorld now points to its actual `2405.14573` paper, and BrowserGym points to the official
2025 ecosystem paper (`openreview.net/forum?id=5298fKGmv3`) rather than unrelated arXiv records.
The catalog fingerprint is `b1b170b2…`; all native and public-release blockers remain explicit, and
the gate is still `ready=false`.

### Current-checkpoint RL preflight (m319)

The [m319 receipt](paper/results/raw/m319-current-rl-preflight-v1.json) tests the actual
`sft_browser_context` checkpoint rather than the older pilot parent.  The lineage rule now
accepts this specialized SFT stage because its checkpoint lineage explicitly remains `stage: sft`.
The isolated two-step RL prefix still fails closed: every sampled rollout receives zero reward,
there is no reward diversity, no nonzero-LR policy update changes a model tensor, and held-out
tool exactness remains `0%`.  This is a useful training diagnosis—not a promotion: the current
checkpoint needs a schema-valid, learnable reward curriculum before RL can be claimed or exported.

### Current-checkpoint stateful RL preflight (m321)

The [m321 receipt](paper/results/raw/m321-current-stateful-rl-preflight-v1.json) supplies the
learnable curriculum without weakening the canonical failure: it uses only the repository’s
resettable email/Notion/browser fixture and shaped schema/tool/argument/transition rewards.  The
same current checkpoint and tokenizer produce two reward values, change 40 policy tensors, and
realize two optimizer updates in the isolated prefix.  No benchmark text, emulator, account, or
external side effect is used, so this is a valid RL-simulation pass—not a native quality claim.

### RL-bound workshop gates (m320–m322)

The [m320 gate](paper/results/raw/m320-workshop-gate-rl-bound-v1.json) made that diagnosis part
of the publication contract.  The successful [m321 receipt](paper/results/raw/m321-current-stateful-rl-preflight-v1.json)
then satisfied the RL requirement without changing the canonical m319 result.  The [m322 gate](paper/results/raw/m322-workshop-gate-stateful-rl-v1.json)
therefore has ten blockers rather than eleven: RL now passes, while nine native benchmark
families and the current-checkpoint public artifact remain unresolved.

### Longer stateful continuation and weight audit (m323)

The [m323 control](paper/results/raw/m323-current-stateful-long-continuation-v1.json) extends the
same current parent with 64 stateful SFT updates and 8 shaped-reward RL steps.  It produces a
healthy six-value reward distribution and raises mean reward slightly (`0.3313→0.3500`), but
held-out exactness falls from `25%` to `12.5%`; the deployment-shaped runtime remains `0/5`
completed tasks.  The paired [weight report](paper/results/raw/m323-current-stateful-long-weight-v1.json)
finds compatible configuration/tokenizer/shapes, zero action-head movement, and much larger body
movement—embedding `7.59%`, attention/mixer `1.97%`, FFN `2.52%`.  This is a negative control,
not a promotion: longer local RL is not evidence of better realistic-agent behavior, and the
adoption recipe remains a verified body with low-rate updates plus separately controlled heads.

### Current-checkpoint low-rate transfer ablation (m326)

The [m326 receipt](paper/results/raw/m326-current-stateful-lowrate-transfer-v1.json) reruns the
current parent with matched frozen, low-rate-unfrozen, and random-backbone arms on the same
source-disjoint local email/Notion/browser/recovery fixture.  The low-rate arm keeps backbone
movement below `0.2%` and raises mean shaped reward to `0.2344` and exact tool selection to `37.5%`,
but closed-loop completion is unchanged at `1/5` tasks (`0/4` productive workflows); the frozen
arm reaches `25%` exact tools and the random arm `12.5%`.  Selector top-1 is not improved over the
frozen arm, so the receipt explicitly rejects capability adoption.  The result strengthens the
weight policy—small body updates are safer than m323's larger continuation—but still does not
establish native mobile, browser, desktop, MCP, email, Notion, or WebGPU task success.

### Current Computer Agent Arena desktop action-prior probe (m327)

The [m327 receipt](paper/results/raw/m327-computer-agent-arena-current-v1.json) evaluates the
current `bc1aca20…c16361` checkpoint on 256 unique trajectories from the pinned public
`xlangai/computer-agent-arena` revision.  Because this text-first model has no visual encoder, the
probe uses only each instruction and its first parseable action.  Route accuracy is `100%`, but
tool exactness is `3.91%`, with `0%` exactness on the 222 pointer rows; observation/screenshot is
correct on `6/7` observation rows.  The result is a precise action-grounding failure diagnosis,
not a native desktop or visual benchmark score, and it does not justify adding Arena rows to SFT.

### Current Mind2Web DOM pointer transfer (m329)

The [m329 receipt](paper/results/raw/m329-current-mind2web-pointer-transfer-v1.json) reruns the
browser pointer adapter from the exact current parent, using `36` train conversations and `12`
source-disjoint evaluation conversations from the pinned `osunlp/Mind2Web` revision.  The trainer
now accepts the parent's known 17-row pointer head when explicit metadata is absent, appends only
`target` and `text`, and records a deterministic seed.  Exact literal-span grounding rises from
`0/63` to `6/63` (`9.52%`), with low shared-body movement; the paired [weight audit](paper/results/raw/m329-current-mind2web-pointer-weight-v1.json)
still shows an expanded pointer embedding shape and no proof of native browser success.  This
supports the migration and low-rate policy, not deployment adoption.

### Current-checkpoint HF-format export (m330)

The [m330 receipt](paper/results/raw/m330-hf-local-export-current-v1.json) regenerates the
self-contained Hugging Face-format bundle from the exact current WebGPU checkpoint.  It binds
the `10,524,544`-parameter model, BPE tokenizer, `63`-tool dispatch pool, `23` pointer arguments,
and every local bundle file by byte count and SHA-256.  The model config itself carries the
checkpoint SHA, so the future public audit can reject a stale model even when the Hub URL is
reachable.  HF upload and anonymous verification remain intentionally pending until an
authenticated maintainer supplies a model repository and Space.

The paired [`scripts/publish_hf_release.py`](../scripts/publish_hf_release.py) now makes that
handoff reproducible: local-only preparation is the default, `--publish` is explicit, both model
and static Space folders are verified before upload, and an anonymous Hub audit is mandatory for
a successful current-release exit.

The [m334 paired receipt](paper/results/raw/m334-hf-paired-release-local-current-v1.json) is a
historical local preparation after the WebGPU safety-policy update.  It binds the exact
`10,524,544`-parameter checkpoint, hard-parity-passed model/action graphs, and the then-current
`app.js` hash `a2d8fed1…` in its staged Space; later release edits intentionally make that hash
different.  Publication remains false until a maintainer supplies HF write authentication and
the post-upload anonymous audit.

The [m336 aligned release receipt](paper/results/raw/m336-hf-paired-release-local-current-63-tool-v1.json)
closes a deployment metadata mismatch discovered during the release audit.  The HF config had a
63-name inferred dispatch pool while the WebGPU `meta.json` and selector matrices still defaulted
to the legacy 50-tool catalog.  The publisher now resolves one checkpoint-bound pool—50 standard,
11 mobile, and 2 productivity schemas—and passes it to both exports.  Fresh local staging reports
63 names in the model config, browser metadata, dense selector, and retrieval selector, with the
same name-set hash `fd3c5d30…`; all four ONNX parity checks pass.  This improves deploy correctness,
but does not create native mobile/browser/MCP success or a public HF release.

### Current MobileGym catalog/native reconciliation (m337)

The [m337 receipt](paper/results/raw/m337-mobilegym-current-catalog-reconciliation-v1.json)
corrects a registry lag discovered during the workshop-gate audit.  The pinned MobileGym source
`093a3292…` has 28 simulated apps, 416 templates, and a disjoint 160-template/256-task contract.
The current checkpoint's existing native receipt covers the complete official test split and passes
`1/256` (`0.39%`) under a compact text/DOM observation projection.  The score is native simulator
and state-diff evidence, but not visual mobile grounding: screenshots were not used.  The benchmark
data remains CC-BY-NC-4.0 and evaluation-only, with zero training rows admitted.

The [m338 runtime-capability audit](paper/results/raw/m338-realistic-agent-runtime-capability-audit-v1.json)
freezes the host-side boundary for the full 40-row realistic-agent catalog.  `adb`, Docker, QEMU,
BrowserGym, MCPMark, and the iOS CoreSimulator service are unavailable or uninstalled here, so
only four text-first adapters are runnable and 36 environment/evaluation rows remain blocked.  This
is a reproducibility receipt for infrastructure readiness; it intentionally performs no downloads,
service starts, benchmark executions, training admission, or native score claims.

The [m339 CUA-Gym surface probe](paper/results/raw/m339-cua-gym-current-surface-probe-v1.json)
provides a current-checkpoint warm/random control without overclaiming what metadata can show.
After 300 frozen-head steps on the task-ID-disjoint CUA-Gym table, the warm arm is `77.55%` versus
`77.87%` random on the 628-row holdout; balanced accuracy is `77.36%` versus `78.50%`.  The
negative deltas reject this warm initialization for deployment on the surface probe.  CUA-Gym
instructions are used only for platform labels; setup files, reward code, screenshots, action
traces, and native desktop/browser execution remain excluded.

The [m340 AgentNet projection](paper/results/raw/m340-agentnet-current-text-projection-v1.json)
updates the desktop text/action evidence to the exact current checkpoint.  All eight Ubuntu parents
and 133 projected actions are present, but first-action type accuracy, action score, and exact
trajectory rate are all `0`.  Compared with the older mixed-surface child, this is a current
regression signal rather than a promotion candidate; screenshots, OS state, and AgentNetBench
execution remain outside the projection.

The [m341 cross-surface weight-adoption decision](paper/results/raw/m341-cross-surface-weight-adoption-decision-v1.json)
now joins the current Mind2Web, ToolACE, CUA-Gym, AgentNet, and stateful-productivity controls.
It rejects exporting a child checkpoint: retain the pretrained BPE body only as an initialization
candidate, adapt action heads per surface, and require frozen/low-rate/matched-random arms plus a
release-matched native verifier before promotion. This prevents a positive offline selector or
pointer metric from being mistaken for reliable email, Notion, browser, mobile, or desktop control.

The [m342 browser smoke](paper/results/raw/m342-webgpu-static-browser-smoke-v1.json) verifies the
regenerated 63-tool staging Space at HTTP 200 with zero page errors and no failed asset requests.
It also found and corrected the stale 50-tool/byte-level banner in `index.html` and the app contract
comment. Chromium had no adapter and fell back to WASM, so this is static boot evidence only; the
native Apple Metal WebGPU receipt remains separate. The deployment verifier now supports an
explicit checkpoint SHA and expected tool-count binding, so a legacy 50-tool bundle cannot pass
pre-upload verification merely by matching its own manifest.

### Current ToolSandbox native smoke (m348)

The [m348 receipt](paper/results/raw/m348-toolsandbox-native-current-smoke-v1.json) finally boots
the pinned Apple ToolSandbox simulator and milestone verifier against the exact current checkpoint
in an isolated compatibility environment. The bounded single-step smoke passes `2/3` scenarios
(cellular and Wi-Fi state changes; message similarity `0.425`), while a bounded interactive message
continuation passes `0/1`. The upstream model-based user simulator, official split, full scenario
matrix, and RapidAPI tools were not executed, so this is native verifier diagnostic evidence only;
it does not satisfy the official ToolSandbox gate or establish email, Notion, MCP, or productivity
completion.

The [m349 transfer control](paper/results/raw/m349-toolsandbox-weight-transfer-native-control-v1.json)
trains matched frozen route/selector heads on the 107-row ToolSandbox projection and evaluates 20
held-out parent scenarios. Warm initialization reaches `85%` selector top-1 versus `80%` for the
matched random arm, route reaches `100%` for both, and native single-step verifier outcomes are
identical (`2/3`, with message similarity `0.425`). The result rejects deployment transfer despite
the offline gain and keeps the parent-only initialization policy.

### Current MCPMark native filesystem task (m350)

The [m350 receipt](paper/results/raw/m350-mcpmark-native-filesystem-current-v1.json) runs one
public MCPMark Verified filesystem task from the pinned `cd45b7f…` checkout against the exact
current checkpoint. A real `@modelcontextprotocol/server-filesystem@2025.12.18` stdio server,
the downloaded public state archive, and the task's independent verifier all executed. The model
did not complete the task: it selected incorrect tools and produced paths outside the allowed
directory, so the verifier exited `1`. This is a native failure diagnostic, not an official
MCPMark score: the full split, user simulator, and other MCP services were not run, and no task
result was admitted to training.

### Current public MCPMark state transfer with native control (m351)

The [m351 receipt](paper/results/raw/m351-mcpmark-public-state-transfer-native-control-v1.json)
repeats the transfer experiment from the exact current BPE parent using eight source-disjoint
public MCPMark state-summary trajectories for training and two held-out trajectories for evaluation.
The warm child improves held-out token accuracy from `30.21%` to `31.97%`; the matched random
backbone reaches `8.97%` after the same 16 updates, a `23.00`-point warm advantage. Weight movement
is small for the warm child (embedding `0.215%`, mixer `0.133%`, FFN `0.156%`) and large for the
random child. However, both children fail the same native filesystem MCPMark verifier (`0/1`,
exit code `1`) with malformed tool/path actions. The native control therefore rejects transfer
for deployment despite the teacher-forced gain; neither child is exported.

### Current-checkpoint workshop/publication gate re-audit (m352)

The [m352 gate receipt](paper/results/raw/m352-workshop-gate-current-v1.json) re-runs the strict
publication join after adding the current-checkpoint binding to m351. The gate remains
`ready=false`: catalog coverage, MobileGym, BrowserGym/MiniWob, native WebGPU capability/latency,
weight transfer, and stateful RL preflight pass, while AndroidWorld, MobileSafetyBench, iOSWorld,
OSWorld, OSWorld-V2, AgentNet, and EnterpriseOps-Gym have no native receipts. ToolSandbox and
MCPMark are explicitly blocked by `official_split_not_verified`, and the current public model/demo
manifest is still absent. This is the reproducible no-promotion decision for the present
checkpoint, not a claim that any missing native benchmark was run.

### Current three-surface public transfer control (m353)

The [m353 receipt](paper/results/raw/m353-three-surface-transfer-current-v1.json) trains a matched
warm/random continuation on source-disjoint public AndroidControl, AgentNet, and train-only
Mind2Web projections. The warm arm improves held-out token accuracy from `59.86%` to `67.58%`,
versus `0%` to `9.41%` for the random arm, giving a `58.17`-point warm advantage on all three
surfaces. Warm body movement remains below `0.40%` by group while random initialization moves the
embedding by `119.7%` and the mixer/FFN by `77.8%`/`87.9%`. Because exact sequence accuracy is
`0%` and no native replay was executed, the result supports parent-body reuse plus surface-specific
heads as a training hypothesis only; the child is not promoted or exported.

### Current Mind2Web browser-head matched control (m356)

The [m356 receipt](paper/results/raw/m356-current-mind2web-browser-head-transfer-v1.json) reruns
the browser route/selector adaptation against the exact current BPE parent, using 219 train and 63
source-disjoint held-out decisions from the pinned public Mind2Web TRAIN split. A frozen-body warm
head reaches `85.71%` selector top-1 and `100%` top-3 from `19.05%`/`85.71%`; the matched random
head reaches the same `85.71%`/`100%` after the same 32 updates. Warm route/selector movement is
`0.232`/`0.216` relative L2 versus `1.154`/`1.030` for random, so the evidence favors retaining
the parent representation as an initialization but does not show a warm quality advantage. No
BrowserGym replay, screenshot grounding, official test split, or external browser side effect was
run; both children remain diagnostic-only.

### Current WebGPU browser realistic-action smoke (m357)

The [m357 receipt](paper/results/raw/m357-current-webgpu-browser-realistic-actions-v1.json)
binds a headed in-app-browser smoke to the current checkpoint and the hash-verified 63-tool
bundle. The model loads at the local HTTP origin with the expected BPE banner and `WEBGPU`
provider label. Email and Notion requests select the intended action schemas and arguments but
are held at the confirmation boundary; a direct URL request is read-only. With planner mode
enabled, the compound request selects `web_search` followed by `notion_write`, with the latter
staged for confirmation rather than executed. This validates planner state propagation and the
demo safety boundary, while still stopping short of external multi-step productivity success. No
hardware adapter, external side effect, official benchmark score, or public model release is
claimed.

### Current paired HF + Space preparation rebuild (m364)

The [m364 receipt](paper/results/raw/m364-hf-paired-release-local-current-v1.json) rebuilds the
current Hugging Face-format model and static Space staging directories from the exact BPE parent.
The 63-tool pool is aligned across config, metadata, selectors, tokenizer, and ONNX graphs; the
bundle is parity-gated and locally verified. This closes the local packaging loop only. HF remains
unauthenticated, so no upload, anonymous download audit, public URL, hardware adapter result, or
benchmark claim is made.

### Current interactive ToolSandbox stress run (m368)

The [m368 receipt](paper/results/raw/m368-toolsandbox-native-current-interactive-v1.json) runs the
current BPE checkpoint inside the pinned Apple ToolSandbox simulator/verifier while allowing
bounded model continuation after each tool result. All three selected scenarios execute, but the
agent reaches `0/3` exact completions (`0.5`, `0.5`, and `0.0` milestone similarity), with no
external API call. This is native resettable runtime evidence, not an official ToolSandbox result:
the upstream split and model-based user simulator remain unexecuted, and the dependency
substitution (`ccy 1.4.0` for the unavailable Python-3.12 `ccy 1.3.1`) is recorded in the receipt.

### Strict workshop gate after interactive ToolSandbox run (m369)

The [m369 gate receipt](paper/results/raw/m369-workshop-gate-current-toolsandbox-interactive-v1.json)
joins the interactive native receipt with current WebGPU, BrowserGym/MiniWoB, MobileGym, weight,
RL-preflight, catalog, and artifact checks. It remains `ready: false`: ToolSandbox is explicitly
blocked because the official split was not verified; AndroidWorld, MobileSafetyBench, iOSWorld,
OSWorld, OSWorld-V2, AgentNet, MCPMark, and EnterpriseOps-Gym lack official native receipts; and
the supplied public artifact is not bound to the current checkpoint. This is the intended
fail-closed workshop decision, not a claim that the missing environments were evaluated.

### Six-source public continuation at 32 steps (m372)

The [m372 receipt](paper/results/raw/m372-all-public-candidate-transfer-32step-v1.json) is a
matched warm-parent/random-backbone continuation across six public sources: AndroidControl, AITW,
AgentNet, Mind2Web, ToolACE, and xLAM. It uses `86` train and `80` source-disjoint held-out rows,
`32` optimizer updates, and a `512`-token context. Warm held-out token accuracy rises `51.31% →
55.35%`, while random reaches `7.43%` from `0%`; warm wins all six surfaces by `22.22–64.88`
points, but exact sequence accuracy remains `0%` for both. Warm shared-body movement stays below
`0.5%` relative L2 while random movement is `0.78–1.20×`; action heads are unchanged. The
adoption decision is therefore “reuse the parent as an initialization candidate, keep native and
official-split gates open,” not a WebGPU policy promotion.

### Six-source public continuation at 64 steps (m377)

The [m377 receipt](paper/results/raw/m377-all-public-candidate-transfer-64step-v1.json) extends
the same six-source public-train control to `64` updates while retaining `86` train and `80`
source-local parent/slot-disjoint held-out rows at `512` tokens. Warm held-out token accuracy is
`56.18%` versus `31.26%` for the matched random backbone (`+24.91` points), and warm wins every
surface. Exact sequence accuracy remains `0%`. The warm child moves embedding/attention/FFN by
`0.837%/0.329%/0.397%` relative L2, while the random body moves `0.78–1.20×`; action heads are
unchanged. This is stronger initialization-lineage evidence, not an official score or policy
promotion, and the child remains unexported.

### ToolACE free-run parent/child control (m375)

The [m375 receipt](paper/results/raw/m375-toolace-parent-warm-free-run-v1.json) evaluates the
current parent and the 32-step warm continuation child on identical public ToolACE action-history
rows. Parent tool exactness is `8/30` (`26.67%`), while the child is `7/30` (`23.33%`); both arms
remain `0%` on argument, step, and episode exactness and `80%` schema-valid. The transfer therefore
does not survive deployment-shaped free-run decoding. Keep the parent as a representation
initialization candidate, but reject the child for policy export and require native verifier-backed
controls before adoption.

### Current native ToolSandbox smoke (m366)

The [m366 receipt](paper/results/raw/m366-toolsandbox-native-current-v1.json) executes three
scenarios in the pinned ToolSandbox simulator with its milestone verifier and the current BPE
checkpoint. `cellular_off` and `wifi_off` pass exactly; the message scenario reaches `0.425`
similarity. This is native resettable tool-state evidence, not an official ToolSandbox score: the
official split and model-based user simulator remain unexecuted, and the run made no external API
call. The Python-3.12 dependency substitution is recorded in the receipt rather than hidden.

### Unified six-source public transfer control (m361)

The [m361 receipt](paper/results/raw/m361-all-public-candidate-transfer-v1.json) performs a
matched continuation across six public source projections: AndroidControl, AITW, AgentNet,
Mind2Web, ToolACE, and xLAM. It uses four train and four source-disjoint held-out rows per source,
the current BPE parent for the warm arm, and a deterministic random backbone for the control. Warm
token accuracy reaches `53.72%` versus `0%` random after eight CPU updates and wins on every
surface, but exact sequence accuracy remains `0%` for both arms.

This is a lineage and representation-transfer diagnostic, not six official benchmark scores. The
AITW eval is a local whole-parent holdout from public train, AndroidControl screenshots are
omitted, and no native mobile/browser/desktop/MCP runtime or external account was touched. Warm
shared-body movement is below `0.1%` relative L2 while the random body moves by `0.78–1.20×`; the
adoption decision is therefore “parent initialization candidate only,” with no WebGPU child export
or workshop promotion.

### Scaled six-source transfer control (m362)

The [m362 receipt](paper/results/raw/m362-all-public-candidate-transfer-scaled-v1.json) expands
the same public-source control to `86` train and `80` held-out rows at a 512-token context. Warm
token accuracy is `52.59%` versus `0%` random after eight updates, and the warm arm still wins on
all six source projections. Exact sequence accuracy remains `0%`; Mind2Web and ToolACE do not
improve in this short horizon, which is why the result is treated as a representation/lineage
control rather than a universal policy adapter.

The current parent and tokenizer remain shape/hash-compatible. Warm body movement stays below
`0.1%` relative L2, while the random body moves by `0.086–119.721%`. AITW is still a local
whole-parent holdout from public train, screenshots are omitted for AndroidControl, and no native
runtime or external account was used.

### Public MobileSafetyBench policy projection (m332)

The [m332 receipt](paper/results/raw/m332-mobilesafety-text-policy-v1.json) binds the public
MobileSafetyBench revision and local task/QA file hashes, then runs the WebGPU
`side_effect_confirmation_v1` policy plus the narrow `text_harm_block_v1` lexical layer on a
canonical action-family projection.  The 90-row public table contains 45 helpful and 45 safety
scenarios; all helpful rows remain unblocked, while the projection emits 45 confirmation decisions,
23 allowed decisions, and 22 risk blocks (with one additional QA block).  The committed receipt
deliberately contains no task instructions, and the result is a safety-boundary diagnostic only:
no Android emulator, Appium, ADB, screenshot, device-state verifier, helpfulness score, or official
safety score was run.

### Current MCPMark Verified source refresh (m335)

The [m335 source audit](paper/results/raw/m335-mcpmark-current-source-audit-v1.json) corrects a
catalog snapshot drift without changing any model or training data.  At the pinned public commit
`cd45b7f57923b9b3985467f5139927575f83141c`, MCPMark Verified has `127` standard and `50` easy
tasks (`177` total) across the five README-level MCP services: Notion, GitHub, Filesystem,
Postgres, and Playwright.  The repository tree has six task-root variants because the
`playwright_webarena` subset is separate; its per-root counts sum to the same `177` tasks, with
`354` metadata/description files.

This is a source inventory only.  No task text, verifier, service state, credentials, runtime, or
score was admitted to training or executed.  Historical metadata receipts that report `239`
rows (`169` standard, `70` easy) remain useful only as historical records and are not current
MCPMark Verified evidence.

### Current action-tail and grounded-argument continuation (m393)

The [m393 receipt](paper/results/raw/m393-current-stateful-action-tail-lexical-grounding-v1.json)
records the first bounded fix after the stateful-runtime trace audit.  The dense selector keeps the
full state/history embedding but also queries the short `Next required action:` tail, while
schema-grounded extraction takes precedence over a stale learned pointer for email, URL, app, and
field values.  Training adds only deterministic, train-side UI paraphrases (`Select`/`Tap`,
`Send`/`Submit`, and similar) and leaves evaluation prompts unchanged.

On five disjoint local resettable tasks (email, Notion, browser, recovery, and abstention), the
oracle completes `5/5` and the current child completes `4/5` (`80%`, `11/16` accepted steps after
bounded retries).  Browser, Notion, recovery, and abstention complete; the email task remains a
failure.  This is closed-loop synthetic state-machine evidence only: it is not AndroidWorld,
BrowserGym, OSWorld, MCPMark, real email, Notion, or native WebGPU success, and the child was not
exported to the deployed checkpoint.

### Current EnterpriseOps-Gym email retrieval audit (m400)

The [m400 receipt](paper/results/raw/m400-enterpriseopsgym-current-email-retrieval-v1.json)
re-runs the frozen current `10,524,544`-parameter BPE parent over `67` public email rows with
matched plus-15-tool candidates at the pinned ServiceNow-AI revision.  Name-only dense retrieval
reaches hit@1/3/5 of `20.90%/53.73%/76.12%`; server configuration, SQL verifiers, and external
execution are dropped, and the receipt is independently self-hashed.  This is retrieval evidence,
not enterprise workflow completion or a training corpus.

The current Hugging Face site also exposes a separate `EnterpriseAgents/EnterpriseOpsGym` mirror
with `649` rows across eight domains (including `67` email rows).  It is not silently substituted
for the pinned ServiceNow-AI revision: the two source identities, files, and licenses must remain
separate until an upstream maintainer confirms equivalence.  The mirror is useful for research
discovery, but its task/server/verifier fields remain evaluation-only here.

### Generic mobile/action guard continuation (m402)

The [m402 receipt](paper/results/raw/m402-current-stateful-mobile-lexical-guard-v1.json) keeps the
m393 child and adds a catalog-driven action adapter in the constrained decoder.  It only fires for
explicit handset/mobile language, or a serialized focused compose screen, then derives the email
send choice from the available tool schemas (`to`/`subject`/`body`) rather than a task ID or Gmail
string.  Browser-focused fields are covered by a regression negative.

With the same five disjoint local workflows and at most three retries per step, the oracle and model
both complete `5/5`; the model accepts `16/16` steps in `17` attempts (`94.12%` attempt success).
This is a policy/grounding improvement over m393's `4/5` local result, not a learned-weight gain:
the child checkpoint is unchanged, and no public benchmark payload, emulator, browser account,
MCP server, or external email side effect is involved.  The receipt therefore cannot satisfy the
native AndroidWorld/BrowserGym or workshop-publication gates by itself.

### Authoritative realistic-source refresh (m404)

The [m404 receipt](paper/results/raw/m404-authoritative-realistic-source-refresh-v1.json) records
the current upstream contracts used by the evaluation plan.  AndroidWorld's official page describes
`116` parameterized tasks across `20` Android apps; iOSWorld exposes `133` tasks across `26`
persistent-identity iOS apps and an MCP option; MobileWorld describes `201` tasks across `20` apps
with agent-user and MCP-augmented workflows; OSWorld 2.0 is a separate `108`-workflow release; and
MCPMark requires version-pinned MCP services plus isolated state and verification.

The refresh also catches a reproducibility hazard: the pinned AgentNet revision
`d76ee50a63fad81cfdbe576416757d7c2091ed50` is still discoverable on Hugging Face, but the live
viewer currently reports a `DatasetGenerationCastError` because merged files expose inconsistent
columns.  LocalAgent therefore treats the raw pinned archive/normalized JSONL projection as the
reproducible input and never treats the viewer's apparent row count as a complete dataset.  These
are source-contract observations, not LocalAgent scores or native execution results.

### Upstream realism refresh (2026-08-05)

The current source pages reinforce the release-specific evaluation policy.  [iOSWorld](https://iosworld.io/)
reports `133` tasks across `26` persistent-identity iOS apps; [MobileWorld](https://tongyi-mai.github.io/MobileWorld/)
reports `201` tasks across `20` Android applications and explicitly separates agent-user and
MCP-augmented workflows; [AgentNet](https://huggingface.co/datasets/xlangai/AgentNet) now describes
`22.6K` cross-platform desktop tasks; and [WebBench](https://www.webbench.ai/) currently advertises
`5,750` tasks across `452` websites while the repository-era card used a smaller release count.
These counts are source/release metadata, not LocalAgent scores.  We retain the exact revision,
split, and native-run requirement for every benchmark rather than mixing counts across releases.

### Public source and host-capability refresh (m417)

The [m417 receipt](paper/results/raw/m417-public-realistic-source-refresh-v1.json) adds a
release-aware computer-use check to the catalog.  The upstream AgentNet card describes `22.6K`
human-annotated Windows/macOS/Ubuntu tasks, while the CUA-Lite AgentNet card exposes a much smaller
`4,900`-trajectory train split and `92`-trajectory validation split with `82,171` images.  Those
figures describe different releases/projections; they are not interchangeable training counts.
The receipt keeps the visual/coordinate rows outside this text-first checkpoint until the raw
revision, license, and split are hash-pinned.

The same refresh records the current BrowserGym family list (including WebArenaVerified and
VisualWebArena), AndroidWorld's `116` tasks across `20` apps, MobileWorld's `201` tasks across
`20` apps, iOSWorld's `133` tasks across `26` apps, MCPMark's current `127` standard plus `50`
easy tasks over five services, and the local ToolSandbox AST profile.  Host preflight confirms
that this Mac has Node, Playwright, ONNX Runtime, and `xcrun`, but no `adb`, Docker, or QEMU
Android/desktop runner.  Therefore the only new training admission is zero: these sources remain
evaluation-only or source-audit-only until their native environments and official splits are
available.  This makes the WebGPU trajectory result useful for local dispatch/grounding, while
preventing it from being mislabeled as AndroidWorld, BrowserGym, AgentNet, MCPMark, or ToolSandbox
task success.

### Four-source public continuation and weight-transfer control (m405)

The [m405 receipt](paper/results/raw/m405-four-source-public-continuation-v1.json) is a fresh,
source-local-disjoint continuation experiment over four public projections: AndroidControl, AITW,
Mind2Web, and xLAM.  It trains `170` rows for `16` updates at `max_seq_len=128` from the current
`10,524,544`-parameter BPE parent and compares that child with an identical-protocol,
shape-matched random-backbone arm.  The receipt binds every normalized JSONL input, source
revision/reference, parent and child checkpoint hash, and the warm/random comparator.

The parent-initialized arm reaches `57.05%` aggregate teacher-forced assistant-token accuracy
versus `0.76%` for the random control (`+56.29` percentage points) and is higher on each of the
four source projections.  Source-level warm-minus-random gaps are `64.09` pp (AndroidControl),
`22.22` pp (AITW), `62.59` pp (Mind2Web), and `52.73` pp (xLAM).  The warm child moves shared
embedding/attention/FFN relative L2 by `0.202%/0.115%/0.139%`, normalization by `0.006%`, and
leaves action heads unchanged.  This supports a compatible low-rate backbone/high-rate-head
recipe and a useful transfer baseline; it does not establish optimality, executable action
success, or native benchmark performance.

The claim boundary is intentionally narrow: AndroidControl rows omit screenshots, AITW uses a
small local train holdout, Mind2Web is a grounded-DOM projection, and xLAM is a function-calling
derivative.  No official benchmark split, Android emulator, BrowserGym/desktop VM, screenshot
grounding, MCP server, real email/Notion side effect, or external account was run.  The experiment
therefore improves the reproducible training/transfer evidence while leaving the native and
publication gates closed.

### Bounded public Mind2Web acquisition and weight-transfer continuation (m420)

The [m420 receipt](paper/results/raw/m420-mind2web-public-continuation-v1.json) is the first
current-checkpoint continuation in this thread backed by an acquired public source file rather than
only a pre-existing projection.  The pinned `osunlp/Mind2Web` TRAIN shard is `616,000,823` bytes
at revision `17ece8e…`; its SHA-256 and the derived subset hash are recorded.  The adapter inspected
83 source tasks, rejected 19 with no valid positive grounded target, and retained the first 64
tasks with supported CLICK/TYPE/SELECT operations (maximum eight actions).  The resulting 128
Conversation rows are protected by the existing BFCL/Mind2Web/WebLINX prompt denylist.

The 64 parents were split into 48 train and 16 evaluation parents with zero parent or typed-slot
overlap.  Starting from the exact deployed `10,524,544`-parameter checkpoint, eight CPU SFT
updates plus eight parent-initialized route/selector-head updates raise held-out teacher-forced
token accuracy from `60.60%` to `69.65%`.  Selector top-1 rises from `0.54%` to `84.95%`, while
embedding/attention/FFN relative movement stays at `0.157%/0.142%/0.161%` and the dense selector
moves `97.799%`; route accuracy was already `100%`.  Exact sequence accuracy remains `0%` before
and after.  This supports reusing the pretrained backbone and allocating adaptation capacity to
schema/grounding heads, but it does not establish Mind2Web test, BrowserGym, live-site, visual, or
native WebGPU task success.

### m420 child local Hugging Face/WebGPU release and trajectory (m421)

The [m421 receipt](paper/results/raw/m421-mind2web-child-webgpu-release-v1.json) binds the
`6a6520…` m420 child checkpoint to a local Hugging Face-format model bundle and a static WebGPU
Space bundle.  The export retains the exact `10,524,544`-parameter `webgpu-10m-hybrid` config;
fp32 model/action parity is below `8e-6`, fp16 parity is below `0.00432`, and fp16 argmax
agreement is `1.0`.  This is a reproducible release candidate, not a public upload: HF
authentication was unavailable, so both `model_uploaded` and `space_uploaded` remain false.

The same child was executed by Chromium with a WebGPU backend on the resettable in-memory
fixture.  It produced `13/13` exact schema actions, `13/13` state transitions, and `3/3`
complete trajectories (`gmail_compose_send`, `notion_capture`, and `browser_search_open`) at
`21.6 ms` p50 action latency.  The cold start was `2,521.5 ms`; page errors were zero.  These
figures demonstrate export/runtime integrity and local closed-loop behavior only.  They do not
replace Mind2Web test, BrowserGym, AndroidWorld, MCP, or real email/Notion/browser-account
evaluation, and no external side effect was attempted.

### Current m420 child ToolSandbox native stress (m422)

The [m422 receipt](paper/results/raw/m422-toolsandbox-native-child-v1.json) runs the exact m420
child inside the pinned Apple ToolSandbox simulator and milestone verifier.  The three bounded
single-step scenarios (`cellular_off`, `wifi_off`, and `send_message_with_phone_number_and_content`)
all pass (`3/3`, similarity `1.0`).  A separate state-dependent multi-turn scenario reaches
`0/1`; the failure is retained as a negative control rather than hidden by the easy smoke set.

This is native simulator/verifier evidence with no external API call, but it is not the official
ToolSandbox split: the model-based user simulator, full 1,032-scenario matrix, and RapidAPI tools
were not executed.  The current child therefore has a reproducible native diagnostic, while the
publication gate correctly remains blocked by `official_split_not_verified`.

### Current m420 child MCPMark filesystem native task (m423)

The [m423 receipt](paper/results/raw/m423-mcpmark-native-child-v1.json) executes the exact child
against the pinned MCPMark `cd45b7f…` filesystem task, a real
`@modelcontextprotocol/server-filesystem@2025.12.18` stdio server, a hash-bound resettable archive,
and the task's official verifier.  The child makes three mis-grounded `create_directory` calls;
the verifier exits nonzero and completion is `0/1`.  This negative result is important for the
deployment decision: a real MCP transport is reachable, but the current tiny model still needs
better path/argument grounding for stateful computer-use tasks.

The official MCPMark split, user simulator, remaining services, and leaderboard protocol were not
run.  No external API or account side effect occurred, so this receipt is a native failure
diagnostic rather than a public score or a promotion signal.

### Fail-closed current-child workshop gate (m424)

The [m424 gate](paper/results/raw/m424-workshop-gate-current-child-v1.json) re-evaluates
publication readiness against the exact m420 child rather than the older `bc1aca…` parent.  The
new ToolSandbox and MCPMark receipts are recognized and checkpoint-bound, but each remains blocked
by its explicit `official_split_not_verified` field.  The gate also refuses to reuse the m405
parent-bound transfer ablation or m321 parent-bound RL preflight, and rejects the local WebGPU
manifest because it has no public model/demo URLs.

This is the intended publication behavior: adding realistic native diagnostics improves the audit
trail without turning a three-task smoke, one failed MCP task, or a local-only export into a
workshop-passing result.  The remaining blockers are now named against the child checkpoint and
are actionable rather than hidden by stale parent evidence.

### Current-child matched Mind2Web transfer ablation (m425)

The [m425 receipt](paper/results/raw/m425-mind2web-child-transfer-ablation-v1.json) repeats the
same parent-disjoint m420 Mind2Web continuation with a matched random-backbone control, now
binding both arms to the exact `6a6520…` m420 child.  Eight CPU updates move held-out teacher-forced
token accuracy from `69.65%` to `76.23%` for the warm arm, while the random arm moves from `0%` to
`1.65%`; the warm-minus-random gap is `74.59` percentage points.  Warm embedding/attention/FFN
movement is `0.183%/0.122%/0.139%` with unchanged action heads; the random arm moves those groups
by `1.197×/0.779×/0.878×`.  Both sequence-exact rates remain zero.

This is the strongest child-specific weight-transfer evidence so far: the BPE body is compatible
and useful as an initialization, while surface-specific adaptation should remain concentrated in
heads or low-rate body updates.  It still does not establish executable browser success, visual
grounding, official Mind2Web test performance, or optimal hyperparameters.

### Current-child stateful productivity RL preflight (m426)

The [m426 receipt](paper/results/raw/m426-mind2web-child-rl-preflight-v1.json) runs the exact child
through the bounded stateful productivity RL prefix.  The split audit is clean (`prompt_overlap=0`,
`row_overlap=0`), the rollout produces two reward values (`0` and `0.1`) across two informative
groups, and the nonzero learning-rate step executes.  However, no policy model tensor changes
(`0/40`) and the runner fails with `isolated RL nonzero-LR prefix changed no policy model tensor`.

The child therefore cannot inherit the parent’s RL readiness.  The correct next training work is
to repair reward-to-gradient connectivity or the child’s action/route heads, then rerun this exact
preflight; no RL child or deployment promotion is claimed from this failed receipt.

### Current-child workshop gate after transfer and RL controls (m427)

The [m427 gate](paper/results/raw/m427-workshop-gate-current-child-v1.json) is the first gate
re-run that binds the transfer decision and all available diagnostics to the exact m420 child.  The
m425 warm/random Mind2Web ablation now passes the weight-transfer requirement; the m426 RL receipt
is correctly rejected because its nonzero-LR prefix changed no policy tensor.  ToolSandbox and
MCPMark remain native diagnostic receipts only because their official splits/user simulators were
not executed, and the local WebGPU bundle is not a public model/demo manifest.

The gate therefore narrows the child-specific risk: pretrained-weight reuse is supported, but RL
readiness and workshop publication are not.  This prevents a strong teacher-forced transfer number
from being mistaken for end-to-end agent capability.

### Current-child full MobileGym official test (m428)

The [m428 receipt](paper/results/raw/m428-mobilegym-native-child-full-v1.json) closes the missing
child checkpoint binding for the pinned MobileGym release.  The exact `6a6520…` m420 child ran all
`256/256` official test tasks in the simulator and state-diff judge with no runtime errors, reaching
`1/256` (`0.39%`) success.  This exactly matches the earlier parent-bound m262 result, so the
Mind2Web continuation preserved the parent's bounded text/DOM projection rate rather than
improving mobile task completion.

The result is native simulator evidence and is now eligible for the official-split gate, but it is
not visual Android control: `vision_used=false`, screenshots were not consumed, and no Android
emulator or external account was touched.  The dominant failure mode is action collapse toward
`mobile_input_text` (`255` calls), which points to a concrete next training target: action-family
and argument grounding, followed by longer-horizon state transitions, not more teacher-forced
Mind2Web token updates alone.

### Current-child full BrowserGym/MiniWoB official test (m431)

The [m431 receipt](paper/results/raw/m431-browsergym-native-child-full-v1.json) runs the exact
`6a6520…` child through all `240` fixed-seed episodes of the pinned BrowserGym `0.14.3` /
MiniWoB plan.  The simulator and independent rewards complete without errors, but success is
`0/240` and total reward is zero.  Relative to the parent-bound m282 run, grounded steps rise from
`320/2400` to `500/2400` (`+7.5` percentage points) and no-op actions fall from `2080` to `1900`,
yet no task reaches completion.

This separates a real transfer signal from end-to-end competence: the Mind2Web child is more often
producing syntactically grounded DOM actions, but it still fills stale or semantically wrong
targets and cannot solve even the bounded email/form tasks.  The next training target is therefore
live-DOM identity/argument grounding plus action verification and recovery, not another blind
teacher-forced continuation.  The run used `coordinate_fallback=false` and no screenshots, so it
does not establish visual computer use, WebArena success, or real email/Notion access.

### Current-child gate after complete mobile and browser runs (m432)

The [m432 gate](paper/results/raw/m432-workshop-gate-current-child-browsergym-v1.json) now admits
both the complete official MobileGym and BrowserGym/MiniWoB receipts for the same checkpoint.  This
removes the stale-parent and missing-receipt ambiguity from the two most directly relevant public
mobile/browser contracts.  The gate remains closed for AndroidWorld, MobileSafetyBench, iOSWorld,
OSWorld, OSWorld-V2, AgentNet, official ToolSandbox/MCPMark, EnterpriseOps-Gym, successful child RL,
and public HF/demo URLs.

### Current-child MCPMark filesystem easy service run (m433)

The [m433 receipt](paper/results/raw/m433-mcpmark-filesystem-easy-native-child-v1.json) expands
the earlier one-task MCPMark smoke to all ten public `filesystem/easy` fixtures at the pinned
MCPMark revision. Each task used its public state archive, a fresh root, the real
`@modelcontextprotocol/server-filesystem@2025.12.18` stdio server, and the repository's own
verifier. The exact `6a6520…` child produced zero workspace changes and passed `0/10` verifiers;
there were no MCP runtime crashes. The dominant failure is actionable: malformed path arguments
and repeated `create_directory` calls instead of reading state, grounding paths, and performing
the requested write/rename operation.

This is the strongest current MCP service diagnostic, but it is still not an official MCPMark
score: the MCPMark user simulator, standard/verified split, and Notion/GitHub/Postgres/Playwright
services were not executed. The result therefore blocks promotion of the current child for
general MCP tool use and points training toward tool-state observation, argument grounding,
write verification, and recovery from MCP errors.

### MCPMark state/argument transfer intervention (m434)

The [m434 receipt](paper/results/raw/m434-mcpmark-filesystem-transfer-native-holdout-v1.json)
tests that diagnosis with eight source-disjoint oracle decision rows from the file-context and
file-property families, holding out ten decisions from folder-structure, legal-document, papers,
and student-database tasks. A 24-step warm continuation improves held-out teacher-forced token
accuracy `38.18% → 52.52%` (`+14.34` points), while shared-body movement stays small (embedding
`0.696%`, attention/mixer `0.375%`, FFN `0.447%`, normalization `0.015%`; action heads unchanged).
The stronger test is native: the warm child still passes `0/5` held-out MCP filesystem verifiers,
changes `0/5` workspaces, and has zero MCP runtime errors.

This is negative promotion evidence. The random arm did not produce a valid report under local CPU
memory pressure, so no matched-random claim is made; the transfer is retained only as a training
diagnostic, not exported to WebGPU or counted toward official MCPMark readiness.

### Faithful MCP closed-loop dispatch/grounding audit (m439)

The [m439 receipt](paper/results/raw/m439-mcpmark-faithful-dispatch-grounding-v1.json) corrects the
native harness so it continues after `write_file`/`move_file`, returns each MCP result to the model,
and gives the independent verifier the final workspace. On five source-disjoint holdout tasks the
exact child still passes `0/5`, changes `0/5` workspaces, and has zero MCP runtime errors. A top-3
candidate probe is informative: the model selects the semantically correct `write_file`, but the
path/content arguments are malformed. Route/selector-head adaptation improves route accuracy but
reduces selector top-1 to `0%`; pointer SFT has no valid span gain because the public task prompts
do not contain the full target strings.

This moves the repair target from “more ordinary SFT” to a grounded action ABI: explicit state and
argument spans, relative-path normalization at the MCP boundary, verifier-aware retries, and a
matched-random control. No current child is promoted or exported.

### Current-child MCPMark Playwright native run and redacted transfer (m440, superseded)

The [m440 receipt](paper/results/raw/m440-mcpmark-playwright-native-child-and-transfer-v1.json)
extends the same exact `6a6520…` child to all four pinned public MCPMark Playwright standard
fixtures (table extraction, Turnstile authentication, an X-profile birth-year search, and a
DeepSeek R1 arXiv search). Its temporary runner discovered 22 tools but read the wrong MCP SDK
schema key (`inputSchema` instead of the Python SDK's `input_schema`). The receipt is therefore
superseded for native capability claims; its redacted teacher-forced transfer remains usable.

The same receipt records a deliberately redacted public trajectory continuation using one train
row and two source-disjoint Playwright evaluation rows from
[`Jakumetsu/mcpmark-trajectory-log`](https://huggingface.co/datasets/Jakumetsu/mcpmark-trajectory-log).
Tool outputs and assistant free text are fixed-marker redacted, and absolute paths are suffix
redacted. Eight warm updates from the current child improve held-out teacher-forced token accuracy
from `31.88%` to `33.15%`; the matched random-backbone arm moves from `0.73%` to `0.73%`, leaving a
`32.42`-point warm advantage after training. Exact multi-turn sequence accuracy remains `0%` for
both arms. Warm backbone movement stays below `0.25%` per large group (embedding `0.248%`,
attention `0.185%`, FFN `0.215%`), while the random embedding moves `119.71%`. This supports
reusing the compatible pretrained body with low-rate updates, but it does not repair grounded
tool selection or establish live browser/account competence; no child is promoted or exported.

### Schema-corrected MCPMark Playwright ABI guard (m445)

The [m445 receipt](paper/results/raw/m445-mcpmark-playwright-schema-corrected-abi-guard-v1.json)
reruns the four pinned Playwright fixtures with the corrected `input_schema` bridge and the
exact current `6a6520…` child, plus a selector-only warm child. Both real stdio-service runs
execute all four verifiers with zero runtime errors and score `0/4`; the official split and user
simulator remain unexecuted. The failure mode is now observable: the model can be forced to emit
`browser_navigate` with the explicit URL and then `browser_snapshot`, but it abstains before
grounding a table or page result. The new narrow adapter refuses to invent Playwright `ref` IDs or
JavaScript `function` bodies; this is a deterministic ABI/safety guard, not learned benchmark
performance. The selector child is not promoted, and m440's native section is explicitly marked
invalid because of the schema-key bridge bug.

The current [m447 workshop gate](paper/results/raw/m447-workshop-gate-current-child-mcpmark-schema-corrected-v1.json)
recognizes this receipt's native contract and reduces the MCPMark blocker to the honest
`official_split_not_verified` condition. That does not make the local four-task diagnostic an
official score or make the model ready for public release: the gate still blocks on missing native
mobile/desktop/enterprise families, child-bound RL, and a public artifact manifest.

### Public Mind2Web realistic SFT and weight audit (m448)

The [m448 receipt](paper/results/raw/m448-mind2web-realistic-sft-weight-v1.json) runs a bounded
12-step continuation from the exact `6a6520…` child using 96 public Mind2Web train rows and 32
source-disjoint eval rows from the pinned `17ece8…` snapshot. Held-out teacher-forced token
accuracy improves from `72.72%` to `79.02%`, but exact multi-turn sequence accuracy remains `0%`.
The group-wise transfer audit shows small movement—embedding `0.160%`, attention/mixer `0.102%`,
FFN `0.120%`, normalization `0.004%`—with action heads unchanged. A real Playwright extraction
canary reaches valid `browser_navigate` and `browser_snapshot` calls, then fails the verifier
`0/1`; this separates protocol grounding progress from task completion. The child remains a
low-rate initialization candidate only and is not exported to WebGPU.

### Current-child local HF/WebGPU release verification (m449)

The [m449 receipt](paper/results/raw/m449-hf-webgpu-local-release-v1.json) rebuilds the exact
`6a6520…` checkpoint into a 10,524,544-parameter HF-format model and a 63-tool static WebGPU
bundle. All generated files, byte hashes, checkpoint bindings, and fp32/fp16 ONNX parity checks
pass; the local bundle identity is `ff0259b3…`. The release is intentionally not public:
`hf auth whoami` reports no login, so publication, anonymous re-fetch, and public model/Space URLs
remain unproven. The unpromoted m448 child is not used for the bundle.

### Current-checkpoint WebGPU local trajectory rerun (m450)

The [m450 receipt](paper/results/raw/m450-webgpu-current-local-trajectory-v1.json) reruns the
resettable three-trajectory browser/email/Notion fixture against the exact `6a6520…` bundle after
the release rebuild. Chromium returned HTTP 200 with zero page errors; all 13 actions were
schema-valid, exact, and state-transition-valid, and all three trajectories completed (`pass@1 =
1.0`). This is deployment-shaped local evidence only: the adapter is explicit, the state machine
is in-memory, and no real email account, browser session, Notion API, MCP server, mobile emulator,
or official benchmark split was contacted. The receipt supersedes the older `bc1aca…`-bound local
trajectory artifacts for current-checkpoint binding, but it does not satisfy the publication gate's
native benchmark requirements.

### Current-checkpoint native WebGPU capability rerun (m451)

The [m451 receipt](paper/results/raw/m451-webgpu-native-current-rerun-v1.json) is a fresh
elevated Chromium run against the same `6a6520…` bundle. It identifies an Apple Metal-3 adapter,
uses the requested WebGPU provider without a WASM/CPU retry, and records 3/3 exact local action
probes, `457.52` input-tokens/s p50, `28.45` ms wall-latency p50, and a `20.46` MB conservative
host allocation estimate. The harness intentionally executes no side effect, so closed-loop
success is `0`; this is hardware/export evidence and not email, Notion, BrowserGym, MCPMark, or
agent-task success.

### Current-checkpoint workshop/publication gate refresh (m452)

The [m452 gate](paper/results/raw/m452-workshop-gate-current-v1.json) joins the current
`6a6520…` checkpoint, m451 native WebGPU capability, m425 warm/random weight audit, the corrected
m445 MCPMark ABI receipt, the RL preflight, and the existing public-manifest record. It remains
`ready: false` with 13 blocking requirements. WebGPU capability and the weight ablation pass; the
gate still requires native AndroidWorld/MobileGym/MobileSafetyBench/iOSWorld/BrowserGym/OSWorld/
AgentNet/ToolSandbox/EnterpriseOps-Gym execution, MCPMark's official split, a successful
current-checkpoint RL preflight, and public model/demo URLs bound to this checkpoint. This is the
publication decision artifact: the local demo and synthetic trajectory are not silently promoted
to workshop evidence.

### Public realistic-source refresh: OSWorld email, world models, and cross-platform GUI data (m453)

The [m453 receipt](paper/results/raw/m453-public-realistic-source-refresh-v1.json) adds four
source-linked rows to the supplemental registry without changing the canonical gate or SFT inputs:
[Computer Use Trajectories](https://huggingface.co/datasets/markov-ai/computer-use) (160 successful
OSWorld trajectories, including Thunderbird email), [AgentWorldBench](https://huggingface.co/datasets/Qwen/AgentWorldBench)
(2,170 grounded MCP/Android/Web/OS/terminal/SWE/search turn records), [ScaleCUA-Data](https://huggingface.co/datasets/OpenGVLab/ScaleCUA-Data)
(six-platform GUI trajectories with a 1.07 TB image release), and [GUI-World](https://huggingface.co/datasets/ONE-Lab/GUI-World)
(12,379 videos and roughly 100,000 GUI queries across desktop, mobile, web, and XR). The rows are
evaluation/catalog-only because licenses, benchmark contamination boundaries, visual modalities,
and release-matched runtimes are not yet frozen. They provide realistic email, MCP state, browser,
Android, desktop, and temporal GUI coverage for the next modality-specific training/evaluation
stage; none is claimed as a current score or admitted to SFT.

### Current-checkpoint strict stateful RL preflight (m457)

The [m457 receipt](paper/results/raw/m457-current-checkpoint-rl-preflight-v1.json) runs the
repository's strict one-update RL contract from the exact `6a6520…` checkpoint over disjoint
deterministic Conversation rows for email, Notion, browser search/recovery, and abstention. The
isolated prefix executes two rollout steps and 32 rollouts, observes three reward values, realizes
two optimizer updates, and changes all 40 named LM policy tensors after the first nonzero learning
rate. Held-out mean reward is `0 → 0.10625`; exact tool/trajectory success remains `0`, as expected
for this small diagnostic. The source is an in-memory local state machine with no public benchmark
payload, emulator, browser service, MCP server, or external account, so this proves RL plumbing and
weight movement—not native productivity capability.

### Current-checkpoint gate after strict RL preflight (m459)

The [m459 gate](paper/results/raw/m459-workshop-gate-current-rl-pass-v1.json) replaces the failed
historical RL evidence with m457 and binds it to the current checkpoint. The gate now passes the
RL-preflight, native WebGPU capability, and warm/random weight requirements. It remains
`ready: false` by design: official Android/iOS/desktop/MCP/enterprise benchmark receipts and a
public Hugging Face model/Space manifest bound to `6a6520…` are still absent.

### Current gate with nested official native receipts (m460)

The [m460 gate](paper/results/raw/m460-workshop-gate-current-native-nested-v1.json) fixes a
schema mismatch in the unified publication gate: complete native receipts produced by the MobileGym
and BrowserGym runners store execution and result fields under `environment` and `result`, while
older probes used a flat layout. The gate now validates both layouts, verifies the exact current
checkpoint, and admits the official current results: MobileGym `1/256` (`0.39%`) and BrowserGym /
MiniWoB `0/240` (`0%`). The same gate records current ToolSandbox and corrected MCPMark Playwright
receipts as executed diagnostics but keeps both blocked by `official_split_not_verified`. This
reduces the current publication blockers without converting text/DOM native results into visual
computer-use or real-account email/Notion claims.

### Longer current-parent Mind2Web SFT and weight audit (m461)

The [m461 receipt](paper/results/raw/m461-mind2web-long-sft-weight-v1.json) extends the exact
current `6a6520…` parent by 24 low-rate (`1e-5`) backbone updates on the pinned public Mind2Web
train shard: 96 train rows and 32 source-disjoint eval rows. Held-out teacher-forced token
accuracy improves `72.72% → 78.98%` and mean loss falls `1.830 → 1.310`, but exact multi-turn
sequence accuracy remains `0%`; route/selector heads are intentionally frozen. The compatibility
audit confirms identical configuration, shapes, and tokenizer. Relative movement is embedding
`0.328%`, attention/mixer `0.173%`, FFN `0.206%`, and normalization `0.0075%`. Because the longer
run moves the embedding/FFN groups more than the earlier low-rate candidate and has no native
replay, its decision is to retain it as an initialization candidate only, require a matched
no-transfer control, and forbid WebGPU export/promotion.

### Matched current-parent Mind2Web warm/random continuation (m462)

The [m462 receipt](paper/results/raw/m462-mind2web-long-warm-random-v1.json) closes the matched
no-transfer control requested by m461. Both arms use the same pinned `osunlp/Mind2Web` revision,
96 train rows, 32 source-disjoint eval rows, tokenizer/config, current `6a6520…` parent, seed,
batch size, `1e-5` learning rate, 512-token horizon, and 24 optimizer updates. The warm arm is
the m461 continuation; the control starts a fresh shape-matched random backbone with seed 2028.

Held-out teacher-forced token accuracy is `78.98%` warm versus `13.53%` random, a `65.45`-point
advantage. Warm improves from `72.72%` while random improves from `0%`; exact sequence accuracy
is `0%` for both. The result supports retaining the warm checkpoint as a low-rate initialization
candidate and rejects any claim that transfer alone yields executable browser competence. Native
replay, official benchmark receipts, and the public HF/WebGPU manifest remain required; the warm
child is not exported.

### Bounded public xLAM-derived warm/random continuation (m463)

The [m463 receipt](paper/results/raw/m463-xlam-derived-warm-random-v1.json) materializes one
pinned train shard and one pinned test shard from the Apache-2.0
[public derivative](https://huggingface.co/datasets/product-science/xlam-function-calling-60k-raw)
at revision `dfbd3c…`: 256 normalized train rows and 128 evaluation rows. Normalization rejected
three train rows and one eval row whose arguments violated the canonical tool schemas; no repair
or coercion was applied. The derivative is explicitly not the gated Salesforce xLAM split, and
generic slot values overlap across its train/test shards, so the split is recorded as a derivative
directory boundary rather than a contamination-free slot-disjoint benchmark.

Using the exact current `6a6520…` parent, 24 updates at `1e-5`, and a matched random backbone,
held-out teacher-forced token accuracy is `51.30%` warm versus `0.49%` random (`+50.81` points);
exact sequence accuracy is `0%` for both. Warm movement is embedding `0.260%`, attention/mixer
`0.109%`, FFN `0.135%`, and normalization `0.0066%`; the random body moves `119.70%/77.88%/87.79%`
for embedding/attention/FFN. This is useful initialization evidence for tool-call syntax, but
not an official xLAM/BFCL score, multi-call score, native MCP result, or live API capability.
The child remains unexported pending native replay and a full tool-use evaluation.

### Current-parent xLAM constrained-decoder free-run canary (m464)

The [m464 receipt](paper/results/raw/m464-xlam-current-free-run-row-canary-v1.json) replays the
same public-derived test shard through the actual constrained decoder after adding a bounded
candidate-mode selector to `scripts/evaluate_xlam_public.py`. On eight rows under the explicit
per-row candidate upper bound, both the current parent and m463 warm child route `4/8` tool names
exactly, match `0/8` argument objects exactly, and produce `8/8` schema-valid calls. The warm
child therefore shows no free-run improvement over the parent in this canary, despite its
teacher-forced token gain. This is negative promotion evidence: the child remains a low-rate
initialization candidate only, and the canary is not a global retriever, native MCP, or live
browser/email/Notion score.

### Matched native BrowserGym canary for xLAM warm transfer (m465)

The [m465 receipt](paper/results/raw/m465-browsergym-xlam-warm-parent-canary-v1.json) runs the
current parent and m463 xLAM warm child through the pinned BrowserGym/MiniWoB environment with the
same Chromium, accessibility/DOM observation path, four fixed seeds, and ten-step horizon. The
environment executes all four episodes for each arm, but both score `0/4`, emit `40/40` noop
actions, and ground `0` steps. This confirms the xLAM continuation does not transfer to native
browser task completion. It is a limit-4 diagnostic, not the official 240-episode score, visual
computer-use result, WebArena result, or live email/Notion account execution.

### Matched native MobileGym canary for xLAM warm transfer (m466)

The [m466 receipt](paper/results/raw/m466-mobilegym-xlam-warm-parent-canary-v1.json) runs the
current parent and m463 xLAM warm child on the same four pinned public MobileGym test IDs with
the same two-step budget, selector policy, and bounded DOM/text observation projection. The
native simulator and state-diff judge execute cleanly for both arms, but both score `0/4`, make
`0.0` aggregate progress, and emit only `mobile_input_text` on all four tasks. The warm child
therefore shows no transfer to mobile task completion. This is a limit-4 diagnostic, not the
official 256-task score, visual Android result, screenshot-grounding result, or real-account run.

### Current-checkpoint AgentNet warm/random transfer control (m467)

The [m467 receipt](paper/results/raw/m467-agentnet-current-warm-random-v1.json) replays the pinned
`xlangai/AgentNet` Ubuntu projection against the exact current `6a6520…` parent: 32 training rows
from four parent records, eight held-out rows from two disjoint parent records, eight low-rate
updates, and a matched deterministic-random backbone. Warm held-out token accuracy rises from
`47.81%` to `52.53%`, while random remains `0%`, a `52.53`-point post-update gap; exact sequence
accuracy is `0%` for both. Warm embedding/attention/FFN movement is `0.109%/0.076%/0.091%`, while
random movement is `119.70%/77.88%/87.79%`. Images were dropped, and no native Ubuntu VM or OSWorld
execution ran, so this is initialization-lineage evidence only and the child is not exported.

### AgentNet warm child native BrowserGym replay (m468)

The [m468 receipt](paper/results/raw/m468-browsergym-agentnet-warm-parent-canary-v1.json) replays
the m467 AgentNet warm child and the exact current parent on four fixed MiniWoB seeds in the pinned
BrowserGym/Chromium accessibility/DOM environment. Both arms execute all four episodes, but both
score `0/4`, ground `0` steps, and emit `40/40` no-op actions. The offline AgentNet token gain
therefore does not transfer to browser interaction; this remains a limit-4 diagnostic, not the
official 240-episode score or a live email/Notion result.

### AgentNet warm child native MobileGym replay (m469)

The [m469 receipt](paper/results/raw/m469-mobilegym-agentnet-warm-parent-canary-v1.json) runs the
same m467 child and current parent on four pinned MobileGym test IDs with the two-step text-first
protocol and official state-diff judge. Both arms score `0/4`, make `0.0` aggregate progress, and
emit only `mobile_input_text`; the desktop projection transfer does not improve mobile state
completion. This is a limit-4 diagnostic, not the complete 256-task score or visual Android result.

### Current-checkpoint workshop gate after AgentNet native replays (m471)

The [m471 gate](paper/results/raw/m471-workshop-gate-current-agentnet-v1.json) joins the current
`6a6520…` checkpoint with the m467 warm/random weight audit, m451 hardware-WebGPU receipt, m457 RL
preflight, current official-split MobileGym/BrowserGym full receipts, and the non-official
ToolSandbox/MCPMark diagnostics. Seven checks pass, including catalog coverage, WebGPU capability,
weights, and RL; the gate remains `ready: false` with ten blockers (nine native requirements plus
the public artifact): AndroidWorld, MobileSafetyBench,
iOSWorld, OSWorld, OSWorld 2.0, native AgentNet, EnterpriseOps-Gym, official ToolSandbox, official
MCPMark, plus the missing public model/demo manifest. The m468/m469 limit-4 replays are intentionally
not substituted for official native receipts.

### Current WebGPU MobileSafetyBench policy projection (m472)

The [m472 receipt](paper/results/raw/m472-mobilesafety-current-policy-v1.json) reruns the pinned
MobileSafetyBench text/context rows against the current checked-in WebGPU `actionSafetyPolicy`
(`c8e90a…` app hash), covering `90` task rows and `3` QA rows. The policy classifies `22` rows as
blocked, `45` as confirmation-required, and `23` as allowed; one row contains prompt-injection
indicators, and QA yields one block plus two confirmations. This is useful evidence for email,
messaging, URL, deletion, and calendar side-effect boundaries, but it is not Android/Appium
execution, screenshot grounding, helpfulness, or the official MobileSafetyBench score.

### Android-Control whole-episode transfer control (m475)

The [m475 transfer receipt](paper/results/raw/m475-androidcontrol-current-warm-random-v1.json)
uses the public `google/androidcontrol` source lineage through the pinned
`OfficerChul/Android-Control-84k@train4096` mirror. The source has 4,096 text-only action rows
and 3,483 recoverable episode ids. A reproducible SHA-256 episode bucket assigns complete
episodes to 3,269 train rows/2,767 episodes and 827 eval rows/716 episodes; the split has zero
episode overlap. On the bounded 32-row/8-row continuation, the exact current parent reaches
63.39% held-out teacher-forced token accuracy after eight warm updates versus 0% for a matched
random backbone (`+63.39` points). Exact assistant sequence accuracy remains 0% for both arms.
Warm embedding/attention/FFN movement is only `0.099%/0.064%/0.075%`, so this supports parent
initialization compatibility, not a claim that the child is a better mobile policy. Screenshots
were omitted by the mirror projection and no official AndroidControl score was computed.

### Android-Control child native MobileGym replay (m475)

The [m475 MobileGym receipt](paper/results/raw/m475-androidcontrol-mobilegym-replay-v1.json)
replays the m475 warm child and the exact current parent on the same four pinned official-test
MobileGym ids, using the native simulator/state-diff judge for two steps per task. Both arms
execute cleanly but score `0/4`, make zero progress, and emit only `mobile_input_text`. The
offline Android-Control token gain therefore does not transfer to native mobile state completion;
the child is retained as an initialization diagnostic only. This is a limit-4 text-projection
replay, not the complete 256-task MobileGym score, visual Android grounding, or real-account
execution.

### Realistic-agent catalog refresh: iOSWorld and MobileSafetyBench (m476)

The [m476 catalog receipt](paper/results/raw/m476-realistic-agent-catalog-refresh-v1.json)
refreshes the source-linked inventory to `42` entries and binds two newly verified mobile
evaluation contracts. [iOSWorld](https://iosworld.io/) releases `26` SwiftUI apps, a persistent
cross-app user identity, `133` tasks (`27` single-app, `60` multi-app, `46` memory), rubrics, an
optional MCP server, and an AWS Mac runner. [MobileSafetyBench](https://mobilesafetybench.github.io/)
defines `250` Android-emulator tasks: `200` daily safety/helpfulness scenarios and `50` indirect
prompt-injection scenarios across messaging, web, social, calendar, and finance. Their official
projects also specify the missing native requirements—iOSWorld needs Xcode 26+/iOS 26 Simulator;
MobileSafetyBench needs Android emulators, ADB/Appium, and versioned APK assets.

Both rows remain `eval_only`; the catalog still permits exactly four training sources
(AndroidControl, AITW, xLAM, and Mind2Web train). Task prompts, seeded identity/state, screenshots,
APK assets, rubrics, and safety labels do not enter SFT or tokenizer training. The current
m472 MobileSafetyBench text policy projection is therefore a safety-boundary diagnostic, not the
official 250-task score, and no iOSWorld native run is claimed on this host.

### Realistic-agent runtime preflight after catalog refresh (m477)

The [m477 preflight receipt](paper/results/raw/m477-realistic-agent-preflight-v1.json) probes the
workspace without downloading or launching benchmark environments. It binds the m476 catalog
fingerprint and reports `4` runnable source adapters (AndroidControl, AITW, xLAM, Mind2Web train)
and `38` blocked evaluation/runtime rows. The probes find no `adb`, Docker, QEMU, BrowserGym,
Gymnasium, MCPMark, or OSWorld runtime; `osascript`, `datasets`, `huggingface_hub`, and Playwright
being present does not make a native benchmark runnable. This is a fail-closed readiness snapshot,
not a benchmark score, and it confirms that text projections must remain labelled as diagnostics.

### Workshop gate after catalog refresh (m478)

The [m478 gate receipt](paper/results/raw/m478-workshop-gate-current-catalog-refresh-v1.json)
rejoins the refreshed 42-entry catalog with the current checkpoint, native WebGPU receipt, full
MobileGym/BrowserGym receipts, m467 warm/random weight ablation, and RL preflight. The gate remains
`ready: false`: ten blockers persist—AndroidWorld, MobileSafetyBench, iOSWorld, OSWorld, OSWorld
2.0, native AgentNet, official ToolSandbox, official MCPMark, EnterpriseOps-Gym, and the current
public model/demo manifest. The new iOSWorld and MobileSafetyBench rows are therefore visible in
the same fail-closed publication contract rather than being counted as unexecuted proxies.

### Public MCPMark productivity transfer: Notion + Playwright (m479)

The [m479 receipt](paper/results/raw/m479-mcpmark-productivity-transfer-v1.json) turns the
public [MCPMark trajectory-log dataset](https://huggingface.co/datasets/Jakumetsu/mcpmark-trajectory-log)
at revision `e50578f0ab904d8e6a7c576c387c1e76ae482c89` into a source-bound productivity slice. Two
redacted trajectories (one Notion workflow and one Playwright workflow) are used for continuation;
two different Playwright trajectories (table extraction and a multi-step forum/wiki workflow) are
held out. Tool outputs and assistant free text are replaced by markers, so no account state or
hidden answer is copied into training.

The eight-step matched ablation starts from the exact `6a6520…` parent. Warm initialization ends at
`32.97%` held-out teacher-forced token accuracy versus `0.73%` for the random backbone (`+32.24`
points), and wins on both holdout surfaces (`+21.56/+36.91` points). Exact multi-step sequence
accuracy is `0%` in every arm. Warm attention/embedding/FFN movement is only
`0.077%/0.101%/0.089%`, while random movement is `77.9%/119.7%/87.8%`; this supports reusing the
shared parent body with a small transfer rate, not claiming that the child is a capable browser or
Notion agent.

A one-task real MCP stdio/Playwright replay was attempted with the warm child and the pinned local
verifier. It failed closed before tool execution because `@playwright/mcp@0.0.68` could not be
fetched (`npm ENOTFOUND`); the official MCPMark split remains unverified. This is therefore a
provenance and weight-transfer result, not an official MCPMark score or live email/Notion side
effect. EnterpriseOps-Gym remains evaluation-only: its public email tasks are used for retrieval
audits, not SFT admission.

### Native MCPMark Playwright replay after installing the pinned MCP runtime (m480)

The [m480 receipt](paper/results/raw/m480-mcpmark-native-playwright-replay-v1.json) removes the
earlier infrastructure-only failure by running `@playwright/mcp@0.0.68` as a real stdio server,
Chromium, and the independent MCPMark verifier for `eval_web/extraction_table`. The warm m479 child
discovers `22` tools and executes `browser_navigate` plus `browser_snapshot` without a runtime error,
but emits an abstention instead of the required CSV; verifier success is `0/1`. The exact parent
baseline times out before tool discovery and also scores `0/1`, so the child is not promoted.

This is stronger evidence than offline token accuracy because the MCP server and verifier actually
ran, but it remains a one-task diagnostic: the official split is not verified, the task uses a public
web fixture rather than an email/Notion account, and no external account side effect is claimed.

### Warm-child local HF/WebGPU release candidate (m481)

The [m481 receipt](paper/results/raw/m481-warm-child-hf-webgpu-release-candidate-v1.json) exports
the exact m479 warm child (`10,524,544` parameters, checkpoint
`1fc2d401…`) to a local Hugging Face-format model bundle and a static WebGPU Space candidate. All
four ONNX graphs pass the hard PyTorch parity gate; the bundle contains the 63-tool dispatch pool.
Native Chromium reports an Apple Metal-3 WebGPU adapter, `607.90` input tokens/s p50,
`16.75 ms` wall-latency p50, and `20.46 MB` conservative peak memory across three structured
cases (`3/3` exact local actions). Closed-loop success remains `0` because the fixture intentionally
executes no real side effects.

Hugging Face publication is still fail-closed: this environment is not authenticated, so there is
no public model URL, Space URL, or current public-artifact manifest. The m481 candidate is therefore
deployable locally but not presented as a published release or as evidence of email/Notion/browser
task completion.

### Larger MCPMark cross-surface warm/random transfer (m482)

The [m482 receipt](paper/results/raw/m482-mcpmark-cross-surface-transfer-v1.json) expands the
trajectory experiment beyond the two-row m479 slice: three redacted public MCPMark trajectory-log
records (Notion, Playwright, and filesystem/messages) and eight MCPMark filesystem rows train the
continuation, while twelve source-local parent/slot-disjoint rows are held out.  The same
10,524,544-parameter parent, tokenizer, 16-step schedule, and learning rate are used for a warm
child and a matched random-backbone control.  Warm held-out token accuracy is `38.87%` versus
`0.38%` random (`+38.50` points); filesystem is `44.77%`, Playwright extraction `21.56%`, and
travel `38.48%`.  Exact sequence accuracy is `0%` on both arms.  The paired movement audit shows
the warm child changing the embedding/attention/FFN groups by only `0.181%`/`0.105%`/`0.125%`
relative L2, while the random control moves them by `119.70%`/`77.88%`/`87.79%`.  This supports
warm initialization for the bounded text-and-tool-sequence projection, not official MCPMark or
native browser/productivity success.

### Larger MCPMark child native replay and WebGPU evidence (m483)

The [m483 receipt](paper/results/raw/m483-mcpmark-cross-surface-webgpu-evidence-v1.json) binds the
exact m482 warm child to a real `@playwright/mcp@0.0.68` stdio server, Chromium, and the independent
MCPMark verifier for `eval_web/extraction_table`.  The server runs cleanly (`0` runtime errors,
`22` tools), but the child produces no CSV and the verifier remains `0/1`; the official split is
not verified.  The same checkpoint exports to a local HF-format/WebGPU bundle with all four ONNX
graphs passing hard parity.  Native Chromium reports Apple Metal-3, `573.07` input tokens/s p50,
`24.30 ms` p50 latency, and `20.46 MB` conservative peak memory; three resettable local structured
cases are exact, but closed-loop success is `0` because they execute no side effects.  HF/Space
publication is still fail-closed without authentication, so this is a local release candidate,
not a public model/demo or email/Notion account result.

### Official realistic-agent source/method audit (m484)

The [m484 receipt](paper/results/raw/m484-realistic-source-method-audit-v1.json) records the
evaluation method from the current public source pages rather than treating every JSONL projection
as an official score.  The training/evaluation boundary is now explicit:

- AndroidControl and AITW are public demonstration sources for training only on their official
  training partitions; app/task/platform holdouts remain untouched.
- AndroidWorld, MobileSafetyBench, and iOSWorld require native Android/iOS simulator state and
  independent success/safety rubrics.  iOSWorld adds persistent identity, cross-app workflows, and
  an optional MCP server, so a text-only or WebGPU dispatch result cannot satisfy its score.
- Mind2Web and AgentNet provide browser/desktop action trajectories, but DOM/coordinate projections
  are training or offline diagnostics; native BrowserGym and desktop/VM replay are separate gates.
- OSWorld and OSWorld 2.0 require release-matched VMs, screenshots/accessibility/terminal state,
  final-state evaluators, and (for 2.0) gated assets and long-horizon completion.  MCPMark requires
  an isolated service workspace and its independent task verifier across Notion, GitHub, filesystem,
  Postgres, and Playwright.

This method audit is deliberately source evidence, not a benchmark result.  The official pages
support the protocol choices: [AndroidWorld](https://google-research.github.io/android_world/)
uses dynamic parameterized tasks and durable system-state rewards; [iOSWorld](https://iosworld.io/)
uses 26 apps and 133 persistent-identity tasks with an MCP tool-use option; [OSWorld](https://github.com/xlang-ai/OSWorld)
uses real VMs and execution-based evaluators; [OSWorld 2.0](https://github.com/xlang-ai/OSWorld-V2)
pins code/tasks/assets to a release; [MobileSafetyBench](https://github.com/jylee425/mobilesafetybench)
requires Android emulator/ADB/Appium state; [AgentNet](https://huggingface.co/datasets/xlangai/AgentNet)
is a 22.6K-task desktop trajectory source; and [MCPMark](https://mcpmark.ai/docs/introduction)
isolates service environments and verifies final state independently.

### Balanced public cross-surface continuation canary (m485)

The [m485 receipt](paper/results/raw/m485-realistic-cross-surface-transfer-v1.json) is the next
matched warm/random experiment after the source-method audit.  It uses the exact current 10.52M
parent (`6a6520…`), the same tokenizer, two optimizer updates, and a balanced cap of 16 train rows
per source and four held-out rows per source.  The train side contains 59 rows from AndroidControl,
AgentNet, Mind2Web, and MCPMark; the held-out side contains 18 rows across AndroidControl, AgentNet,
Mind2Web, MCPMark filesystem, and two Playwright trajectory cases.  The split contract is source-local
parent/slot disjoint, and MCPMark's official split remains explicitly unverified.

Warm held-out token accuracy is `43.50%` versus `0%` for the matched random backbone (`+43.50`
points), with warm ahead on all six held-out surfaces: AndroidControl `60.91%`, AgentNet `50.34%`,
Mind2Web `66.67%`, filesystem `37.07%`, Playwright extraction `11.82%`, and travel `40.79%`.
Exact sequence accuracy is `0%` for both arms.  The warm child moves embedding/attention/FFN by
`0.025%`/`0.019%`/`0.022%` relative L2, while the random arm moves them by about
`119.70%`/`77.88%`/`87.79%`.  This supports reusing the pretrained backbone with a low-rate
continuation and separately controlled heads, but the cap and two-step horizon make it a canary,
not a final model-selection result or a native mobile/browser/desktop/MCP score.

### Longer matched realistic-agent transfer canary (m486)

The [m486 receipt](paper/results/raw/m486-realistic-cross-surface-transfer-v1.json) repeats the
same source-local parent/slot-disjoint contract with 16 low-rate updates, keeping the current
10.52M parent (`6a6520…`), tokenizer, 59 training rows, and 18 held-out rows fixed between a
warm-start and random-backbone arm.  This is deliberately a bounded canary: AndroidControl
screenshots are omitted, AgentNet and Mind2Web are text/accessibility projections, and the MCPMark
rows are not the official split or live service state.

Warm held-out assistant-token accuracy rises from `43.36%` to `46.61%`, while the matched random
arm remains `0%` after training (`+46.61` points).  Warm is ahead on every surface: AgentNet
`53.06%`, AndroidControl `64.55%`, MCPMark filesystem `41.46%`, Playwright extraction `11.82%`,
Playwright travel `43.42%`, and Mind2Web `71.11%`.  Exact sequence accuracy remains `0%`.  The
warm embedding/attention/FFN/normalization movement is only `0.190%`/`0.115%`/`0.136%`/`0.0042%`,
versus random `119.70%`/`77.88%`/`87.79%`/`8.57%`.  The evidence supports a low-rate transferred
backbone plus separately controlled action heads as the next training configuration; it does not
establish native completion, screenshot grounding, real email/Notion side effects, or workshop
publication readiness.

### Native MCPMark filesystem head adaptation (m489)

The [m489 receipt](paper/results/raw/m489-mcpmark-filesystem-native-head-adaptation-v1.json)
connects the offline and native layers that the earlier canaries deliberately kept separate.  The
m486 warm child received 64 additional low-rate updates on eight public MCPMark filesystem train
rows, improving the ten-row source-disjoint teacher-forced token accuracy from `41.09%` to
`57.75%`.  A second pass froze every backbone tensor and adapted only the route/selector heads;
the route probe moved from `0%` to `100%` on the held-out rows, while selector top-1 remained a
weak `10%`, so the head child is not promoted on offline metrics alone.

The important result is the isolated native replay: the same head child ran a real pinned
`@modelcontextprotocol/server-filesystem@2025.12.18` stdio server, issued `directory_tree` with
the allowed workspace root, then issued `write_file` with the grounded `structure_analysis.txt`
path and the count `1`.  The independent MCPMark verifier exited `0`.  This validates one
stateful side effect in a public easy fixture, not the official MCPMark split; the user simulator,
other MCP services, email/Notion accounts, and leaderboard matrix remain unexecuted.  The generic
path/operation/count guards are retained as runtime safety/grounding adapters, not as evidence that
the tiny backbone has learned unrestricted filesystem planning.

### Workshop gate after native MCPMark filesystem replay (m490)

The [m490 gate](paper/results/raw/m490-workshop-gate-after-mcpmark-filesystem-v1.json) re-runs the
fail-closed publication checklist with the m489 native receipt joined to the exact current
checkpoint, WebGPU capability, transfer ablation, RL preflight, catalog, MobileGym, and BrowserGym
evidence.  The result remains `ready: false`.  MCPMark is now represented by a native diagnostic,
but the gate correctly keeps it blocked because the receipt is not bound to the current checkpoint
and the official MCPMark split is still unverified.  The other missing native receipts are
AndroidWorld, MobileSafetyBench, iOSWorld, OSWorld, OSWorld 2.0, AgentNet, and EnterpriseOps-Gym;
ToolSandbox is still non-official-split, and the public HF/Space model-demo manifest is absent.

### Current-checkpoint MCPMark easy replay after generic grounding repair (m494)

The [m494 receipt](paper/results/raw/m494-mcpmark-current-filesystem-easy-grounded-v1.json) is a
separate current-parent replay, not the m489 adapted child.  It runs ten isolated public MCPMark
filesystem/easy fixtures through the pinned local `@modelcontextprotocol/server-filesystem@2025.12.18`
stdio server and each task's independent verifier.  The exact checkpoint is bound by SHA-256
(`6a6520…`): `folder_structure` and `papers` pass (`2/10`, `20%`), while the remaining eight
tasks fail through operation sequencing or content/argument grounding.  The generic repair
correctly handles backtick-delimited extensions and joins named directory targets to the workspace
root; it does not make the model a general filesystem planner.

The [m495 gate](paper/results/raw/m495-workshop-gate-after-current-mcpmark-easy-v1.json) now
recognizes MCPMark as a current-checkpoint native receipt and leaves only
`official_split_not_verified` for that requirement.  The result is still `ready: false`: the
official MCPMark split, user simulator, standard tasks, other services, and email/Notion workflows
remain unexecuted.

### Current MCPMark Verified source profile (m496)

The [m496 profile](paper/results/raw/m496-mcpmark-current-source-profile-v1.json) re-reads only
`meta.json` from the pinned public checkout `cd45b7f57923b9b3985467f5139927575f83141c`.  It freezes
`239` metadata rows: `169` standard and `70` easy.  The service memberships are filesystem `40`,
Notion `38`, browser (`playwright` plus `playwright_webarena`) `35`, GitHub `33`, and database
(`postgres`, `supabase`, `insforge`) `93`.  No prompt text, state asset, verifier source, trajectory,
or training row is retained.  This supersedes older source-audit prose that reported `127` standard
and `50` easy tasks for an earlier tree interpretation.

The source README calls the standard suite MCPMark Verified, but a source profile alone is not a
native score.  The current native receipt covers only the filesystem easy/standard boundary that
has actually been executed; Notion, GitHub, Postgres, Playwright, user simulation, and external
accounts remain separate runtime requirements.

### Current checkpoint-bound workshop gate refresh (m406)

The [m406 gate receipt](paper/results/raw/m406-workshop-gate-current-evidence-v1.json) recomputes the
publication checklist against the exact `bc1aca…` checkpoint rather than relying on an older gate.
It recognizes seven requirements as passing: realistic-family catalog coverage, no pending train
adapter, official-split MobileGym, official-split BrowserGym/MiniWoB, native WebGPU capability and
latency, the unified m405 warm/random weight ablation, and the current-checkpoint RL preflight.

The gate remains fail-closed (`ready: false`) with ten blockers: AndroidWorld, MobileSafetyBench,
iOSWorld, OSWorld, OSWorld 2.0, AgentNet native replay, ToolSandbox, MCPMark, EnterpriseOps-Gym,
and the current public model/demo manifest. The m265 WebGPU receipt reports `3/3` exact local
structured actions and `0` external side effects; it is capability/latency evidence, not real email,
Notion, browser-account, or cross-device task success. The m406 result narrows the remaining
publication work without converting local diagnostics into official leaderboard claims.

### Fresh native WebGPU rerun (m407/m409)

The [m407 receipt](paper/results/raw/m407-webgpu-native-current-rerun-v1.json) is a new elevated
Chromium run against the current local bundle, not a reused timing payload.  The page reported an
Apple Metal-3 adapter, WebGPU provider execution, exact dispatch for all `3/3` structured cases,
`546.45` input-tokens/s p50, `19.35` ms wall-latency p50, and `20.46` MB conservative peak memory.
The runner observed no page errors and executed no real email, browser, or Notion side effect;
closed-loop success is therefore `0` by design.

The [m409 gate](paper/results/raw/m409-workshop-gate-current-webgpu-rerun-v1.json) binds that fresh
receipt to the current checkpoint and still reports `ready: false` with ten non-WebGPU blockers.
This strengthens deployment evidence for the exact artifact while preserving the distinction
between local structured capability and real-account/browser-agent completion.

### Native WebGPU local browser/email/Notion trajectory (m412)

The [m412 receipt](paper/results/raw/m412-webgpu-local-trajectory-v1.json) drives the exact current
bundle through three resettable in-memory trajectories: Gmail compose/send (six steps), Notion
capture (two), and browser mail search/open (five).  The independent fixture validator accepts all
13 outputs as schema-valid, but only `4/13` actions are exact and `3/13` state transitions complete;
trajectory pass@1 is `0/3` and closed-loop success is `23.08%`.  The failure pattern is informative:
the email trajectory selects `send_email` where a text-entry action is required, while Notion and
browser steps often collapse to the same email guard.

This is a real Chromium/WebGPU execution with the current checkpoint, but it is deliberately a
local state-machine diagnostic.  No account, browser navigation, Notion API, MCP service, or native
mobile emulator was contacted.  The result therefore exposes the remaining multi-step routing and
argument-grounding gap instead of being reported as a productivity benchmark score.

### State-aware WebGPU trajectory repair (m416)

The [m416 receipt](paper/results/raw/m416-webgpu-local-trajectory-v1.json) reruns the unchanged
13-step fixture after a generic runtime repair.  Dispatch now isolates the latest action from the
long-horizon goal, recognizes serialized focused-state transitions, routes explicit browser verbs
to the matching schemas, and grounds send actions from state fields.  The same current checkpoint
then reaches `13/13` exact actions, `13/13` schema-valid actions, `13/13` state transitions, and
`3/3` complete trajectories (`pass@1 = 1.0`) on native WebGPU.

This is evidence that the deployment adapter can preserve ordered local state transitions; it is
not a learned-quality or public-benchmark gain.  The routing and normalization layer is explicit,
the fixture is resettable and in-memory, and no real email, browser account, Notion API, MCP
service, Android/iOS emulator, or official benchmark environment was contacted.  The m412 result
remains the pre-repair negative control, so the improvement is auditable rather than silently
overwriting a failure.

### Current workshop gate with trajectory companion (m419)

The [m419 gate](paper/results/raw/m419-workshop-gate-current-webgpu-plus-trajectory-v1.json)
rejoins the exact `bc1aca…` checkpoint with the fresh m407 hardware-WebGPU receipt and the m416
trajectory receipt.  Native WebGPU capability/latency, official-split MobileGym and
BrowserGym/MiniWoB, the m405 warm/random transfer audit, current-checkpoint RL preflight, and
catalog coverage pass.  The gate remains correctly fail-closed with ten blockers: AndroidWorld,
MobileSafetyBench, iOSWorld, OSWorld, OSWorld 2.0, native AgentNet, native ToolSandbox, native
MCPMark, native EnterpriseOps-Gym, and a public model/demo manifest bound to this checkpoint.
The local trajectory is included as a companion diagnostic, not substituted for any of those
native requirements.

## Publication checklist

- [ ] Official source revision, license, split, task IDs, and byte/hash receipt.
- [ ] Train/eval parent and typed-slot disjointness; benchmark canaries absent from training.
- [ ] Native environment executed with complete action/termination logs and independent verifier.
- [ ] WebGPU hardware adapter, TTFA/throughput, memory, export parity, and closed-loop action
      success measured separately from token accuracy.
- [ ] Public model/demo manifest binds the exact current checkpoint SHA-256; an older public
      artifact must not satisfy the current-release gate.
- [ ] Warm-start, low-rate, and matched-random transfer arms with per-group weight movement.
- [ ] No real email, Notion, GitHub, or browser side effect without an isolated resettable fixture.

### Current public-evaluation and WebGPU audit (m500/m501)

The current public-evaluation matrix audit contains 27 source-linked rows across mobile, browser,
computer, terminal, and tool-API families.
Only five rows are explicitly train-policy rows; 18 are evaluation-only, three have no static data,
and one is restricted.  The audit is metadata-only: it does not silently download evaluation text,
screenshots, credentials, VM assets, or verifier state.

The [m500 gate receipt](paper/results/raw/m500-workshop-gate-current-audit-v1.json) binds the exact
current checkpoint (`6a6520…`) and reports 12 blockers: the eleven required native benchmark receipts
and a public model/demo manifest.  The static bundle itself is internally consistent and its parity
gate passes, but it is not a public Hub/Space release.

The [m501 receipt](paper/results/raw/m501-webgpu-local-trajectory-current-v1.json) records a fresh
Chrome WebGPU run of the resettable local productivity fixture: Gmail compose/send (6 steps),
Notion capture (2), and browser search/open (5).  All 13 actions were exact, schema-valid, and
state-transition-valid (`3/3` trajectories, pass@1 `1.0`).  This is stronger deployment-shaped
evidence than the earlier negative controls, but remains a local fixture diagnostic—not native
AndroidWorld, BrowserGym, OSWorld, MCPMark, or real-account success.

The [m502 staging receipt](paper/results/raw/m502-local-hf-space-stage-current-v1.json) then rebuilt
the exact checkpoint into a 10,524,544-parameter HF-compatible model directory and a static Space
directory.  ONNX fp32/fp16 and action-model parity all passed, and the regenerated WebGPU bundle
bound the same `6a6520…` checkpoint.  The upload step was intentionally not run because Hugging Face
authentication is absent; the receipt therefore records `published: false` rather than inventing
public URLs.

### Current warm/random continuation control (m503)

The [m503 receipt](paper/results/raw/m503-current-mind2web-warm-random-v1.json) runs the same
four-update continuation from the exact `6a6520…` checkpoint twice on a source-disjoint Mind2Web
slice.  The parent-initialized arm improves held-out teacher-forced token accuracy from `72.33%` to
`77.75%`; the matched random-backbone arm remains at `0%`.  Parent embedding, mixer, and FFN groups
move by roughly `0.05%` relative L2, while the random arm moves by approximately `0.78–1.20x`.
This supports warm initialization and differential learning rates as the current engineering choice.
The larger matched cross-surface canary is recorded separately in the m486 receipt above; both
receipts remain text/accessibility transfer diagnostics until native browser/mobile/tool runtimes
and independent verifiers are supplied.

### Refreshed publication gate (m505)

The [m505 gate receipt](paper/results/raw/m505-workshop-gate-current-m486-v1.json) re-runs the
fail-closed checklist against the exact current checkpoint, the current native WebGPU capability
receipt, the current warm/random transfer ablation, the RL preflight, and the public-artifact
manifest.  Catalog coverage, WebGPU capability/performance, transfer compatibility, and RL
preflight pass.  The gate remains `ready: false` because the eleven native benchmark receipts
(AndroidWorld, MobileGym, MobileSafetyBench, iOSWorld, BrowserGym/MiniWoB, OSWorld, OSWorld 2.0,
AgentNet, ToolSandbox, MCPMark, and EnterpriseOps-Gym) are not supplied and the HF model/Space
manifest is not public or bound to an uploaded URL.  This is the release decision, not a claim that
the local WebGPU fixture or offline projections are native benchmark results.

### Current environment preflight (m506)

The [m506 preflight receipt](paper/results/raw/m506-current-realistic-agent-preflight-v1.json)
rechecks the exact 42-entry catalog in the `.venv` used for the current model runs.  Four adapters
remain runnable (`androidcontrol`, `android_in_the_wild`, `xlam_function_calling`, and
`mind2web_train`); 38 rows are blocked or evaluation-only.  The decisive missing capabilities are
ADB, Docker, QEMU, BrowserGym, Gymnasium, MCPMark, OSWorld, and Playwright in this runtime.  This
is an environment fact, not a dataset omission, and it is why native benchmark receipts remain
blocked while the public text/accessibility projections and local WebGPU fixture remain usable.

### Training acceleration preflight (m509)

The [m509 receipt](paper/results/raw/m509-acceleration-preflight-v1.json) records the local
PyTorch backend used for continuation runs: PyTorch `2.13.0` is MPS-built, but MPS and CUDA are
unavailable, so the reproducible training path is CPU-only.  This explains the bounded continuation
caps and does not change the separate native WebGPU measurement, which runs in Chromium on the
Apple Metal adapter.

### Native BrowserGym environment refresh (m510)

The [m510 receipt](paper/results/raw/m510-browsergym-current-canary-v1.json) re-runs one pinned
MiniWoB episode against the exact current checkpoint (`6a6520…`) after installing BrowserGym
`0.14.3`, Gymnasium `1.3.0`, and Playwright `1.44.0` in the project environment.  The live
accessibility/DOM environment executed ten model steps without a runtime exception, but task success
was `0/1`.  This is a bounded diagnostic (`official_split_verified=false`), not a replacement for
the complete `240`-episode official receipt in m431; it verifies that the current environment can
launch the native adapter and exposes the remaining MiniWoB action/argument grounding gap.

### Current runtime dependency preflight (m511)

The [m511 preflight](paper/results/raw/m511-current-realistic-agent-preflight-v1.json) updates the
catalog probe after that install.  BrowserGym, Gymnasium, Playwright, Hugging Face Hub, and
`datasets` now import successfully, but the catalog remains `4` runnable / `38` blocked because the
remaining rows require ADB, Docker/QEMU, iOS/desktop VMs, MCP services, or restricted/evaluation-only
assets.  This narrows the blocker from missing Python packages to native environment and access
contracts; it does not admit evaluation data into training.

### Current native mobile/browser workshop gate (m513)

The [m513 gate](paper/results/raw/m513-workshop-gate-current-native-browsergym-mobilegym-v1.json)
joins the current checkpoint with the complete official-split MobileGym (m428) and BrowserGym/
MiniWoB (m431) receipts, native WebGPU capability (m451), warm/random transfer (m425), and RL
preflight (m457).  Those checks now pass together.  The gate is still intentionally `ready=false`:
AndroidWorld, MobileSafetyBench, iOSWorld, OSWorld/OSWorld 2.0, AgentNet, ToolSandbox, MCPMark,
EnterpriseOps-Gym, and a public HF model/demo manifest are still absent.  Passing the browser/mobile
pair therefore means the publication evidence is more complete, not that every realistic computer-use
surface or public artifact is ready.

### Current WebGPU bundle verification (m515)

The [m515 receipt](paper/results/raw/m515-current-webgpu-deploy-verify-v1.json) verifies the tracked
static bundle against the exact `6a6520…` checkpoint.  All eight generated artifacts match their
manifest byte counts and SHA-256 values, the 63-tool metadata contract is present, and the hard
PyTorch/export parity gate passes.  This makes the local WebGPU package deployable once copied to an
authenticated Hub Space; it does not create a public URL or turn the local fixture into native
email/Notion/browser-account success.

### Current HF/WebGPU release preparation (m516)

The [m516 receipt](paper/results/raw/m516-hf-release-local-prepare-v1.json) rebuilds the
Hugging-Face-compatible safetensors model and static Space from the same `6a6520…` checkpoint in
local-only mode.  The `10,524,544`-parameter model, 63-tool WebGPU metadata, fp32/fp16 ONNX graphs,
and action graph all export and pass parity; the preparation reports `published=false` because no
Hub token is present.  This closes the local release-reproducibility step without claiming a public
model or demo URL.

### Official-source integrity refresh (m508)

The [m508 source audit](paper/results/raw/m508-official-source-integrity-audit-v1.json) rechecks
the original benchmark pages.  AndroidWorld currently describes live-emulator evaluation with 116
tasks across 20 apps; BrowserGym lists MiniWoB, WebArena/Verified, VisualWebArena, WorkArena,
AssistantBench, WebLINX, OpenApps, and TimeWarp; MCPMark documents isolated Notion, GitHub,
filesystem, Postgres, and Playwright services with pinned server versions; iOSWorld documents 26
iOS apps and 133 persistent-identity tasks; MobileSafetyBench requires Android emulator, ADB, and
Appium; and the official EnterpriseOps-Gym card exposes 649 oracle rows across eight domains and
512 tools.  These source facts reinforce the catalog’s training/evaluation boundary: they are
native-runtime or verifier-backed evaluations, not static SFT text to ingest by default.

### Current checkpoint lineage, transfer, and native-device audit (m552–m570)

The fresh continuation chain is now metadata-preserving: verified m540 RL → grounded m552 SFT →
multisurface m553 SFT, with exact tokenizer, parent-checkpoint, and public train/eval hashes in
the [lineage receipt](paper/results/raw/m563-m553-lineage-training-v1.json). Legacy checkpoints
without lineage are rejected rather than repaired. The matched warm/random [transfer ablation]
(paper/results/raw/m565-m553-transfer-ablation-v1.json) binds to m553 and reports a +62.13-point
teacher-forced token-accuracy advantage for warm initialization across AndroidControl and AgentNet;
this is transfer evidence, not native task success.

The strict [RL preflight](paper/results/raw/m564-m553-stateful-rl-preflight-v1.json) now passes its
gate-visible lineage contract and performs two real optimizer updates, while remaining a local
simulation with zero exact-success rollouts. On a real Apple WebGPU page, the m553 export reaches
1,311.5 tokens/s p50, 7.4 ms latency p50, and 20.5 MB peak memory for structured probes; no email,
browser, or Notion side effect ran ([receipt](paper/results/raw/m566-m553-webgpu-capability-v1.json)).

The complete pinned native splits are also recorded: MobileGym 1/256 and BrowserGym/MiniWoB 5/240,
both with zero runtime errors and no coordinate/semantic fallback ([MobileGym](paper/results/raw/m567-m553-mobilegym-native-full-v1.json),
[BrowserGym](paper/results/raw/m568-m553-browsergym-native-full-v1.json)). They are deliberately
reported as negative controls, not broad computer-use capability claims.

The current [m570 workshop gate](paper/results/raw/m570-workshop-gate-current-m553-v1.json) passes
catalog coverage, MobileGym, BrowserGym/MiniWoB, WebGPU capability, transfer, and RL preflight.
It remains `ready: false`: nine additional native benchmark receipts (AndroidWorld,
MobileSafetyBench, iOSWorld, OSWorld/OSWorld 2.0, AgentNet, ToolSandbox, MCPMark, and
EnterpriseOps-Gym) and a public Hugging Face model/demo manifest are still missing. No public URL,
leaderboard score, or workshop-ready claim is made until those blockers are resolved.

### Public realistic-evaluation matrix refresh (m571)

The [m571 source audit](paper/results/raw/m571-public-realistic-eval-matrix-audit-v1.json) validates
the separate [public evaluation matrix](../configs/data/realistic-agent-public-eval-matrix.v1.json):
28 source-linked entries span mobile (9), browser (4), computer (4), tool/API (10), and terminal (1)
families. Only six rows are train-eligible—AndroidControl, Android in the Wild, Mind2Web, AgentNet,
xLAM Function Calling, and ToolACE—and all other rows are explicitly evaluation-only, runtime-only, restricted, or
metadata-only. This prevents benchmark prompts, screenshots, credentials, VM images, MCP service
state, and verifier outputs from entering SFT/RL by accident.

The audit re-resolves current upstream heads and records the operational contracts that matter for a
WebGPU agent: AndroidWorld is a live emulator benchmark with 116 tasks across 20 apps; BrowserGym
unifies MiniWoB, WebArena, VisualWebArena, WorkArena, AssistantBench, WebLINX, OpenApps, and
TimeWarp; OpenCUA's AgentNet supplies 22.6K cross-OS human tasks with AgentNetBench; ToolSandbox and
MCPMark require stateful simulators/services and verifiers; MobileSafetyBench requires Android,
ADB, and Appium; MobileWorld covers 201 tasks across 20 apps with agent-user and MCP workflows; and
OSWorld requires a release-matched desktop VM. These are source/protocol facts, not scores.

The refreshed [m573 gate](paper/results/raw/m573-workshop-gate-current-m553-matrix-refresh-v1.json)
joins that 28-entry matrix with the 42-entry executable catalog. It still passes the current
checkpoint's measured evidence and remains `ready: false` for the same nine native benchmark and
public-artifact blockers; adding a source link never promotes a runtime benchmark to a score.

### Canonical plus supplemental source-registry reconciliation (m606)

The [m606 audit](paper/results/raw/m606-realistic-source-registry-audit-v1.json) reconciles the
two public-source inventories that had previously been described with historical counts. The
current canonical evaluation matrix contains `28` rows and the supplemental catalog contains `30`
catalog-only rows. They overlap on `9` IDs, yielding `49` unique source IDs across mobile (`13`),
browser (`7`), computer (`17`), tool/API (`11`), and terminal (`1`) families. The audit finds five
metadata conflicts that must be resolved before acquisition (BrowserGym revision, iOSWorld and
MCPMark license wording, MobileSafetyBench URL, and ToolSandbox license wording); it does not
silently choose a winner. Exactly six canonical rows remain train-eligible, while all 30
supplemental rows remain catalog-only. No benchmark payloads, screenshots, credentials, VM/APK
images, MCP state, or verifier traces entered the repository. This is the authoritative inventory
for selecting the next hash-bound public acquisition and native evaluation run.

### xLAM derivative training and weight-transfer control (m574)

The gated original Salesforce xLAM snapshot was unavailable without Hub authentication, so the
experiment uses the publicly downloadable Apache-2.0
[product-science derivative](https://huggingface.co/datasets/product-science/xlam-function-calling-60k-raw),
while retaining the original Salesforce URL and explicitly marking the split as non-official. The
[m574 receipt](paper/results/raw/m574-xlam-derived-warm-random-transfer-v1.json) binds two upstream
Parquet shards, deterministic normalization (256 train / 128 test rows; four schema-invalid rows
recorded and skipped), and a slot-disjoint 96/64 continuation from m553.

Warm initialization improves held-out teacher-forced token accuracy from `52.18%` to `54.35%`; the
matched random-backbone control reaches `23.03%`, so warm transfer leads by `31.32` percentage
points. Warm shared-backbone movement is small (embedding `0.80%`, attention `0.22%`, FFN `0.28%`)
versus random movement near `119.48%`, `77.88%`, and `87.79%`. Sequence exactness remains zero,
and the result is not an official xLAM/BFCL score or live API execution.

The [m575 gate](paper/results/raw/m575-workshop-gate-current-m553-xlam-v1.json) accepts this second
transfer receipt alongside the AndroidControl/AgentNet ablation; the overall publication decision is
unchanged because native AndroidWorld, MobileSafetyBench, iOSWorld, OSWorld variants, AgentNet,
ToolSandbox, MCPMark, EnterpriseOps-Gym, and the public HF/demo manifest are still absent.

### Public deployment audit and m553 release candidate (m576)

The [m576 audit](paper/results/raw/m576-public-release-audit-v1.json) anonymously rechecks the
existing public [model](https://huggingface.co/danelcsb/localagent-tiny-30m-byte) and
[WebGPU Space](https://huggingface.co/spaces/danelcsb/localagent-webgpu). Both are reachable, but
their pinned revisions expose the older byte-level release and do not match the current m553
checkpoint. A fresh m553 export is now locally reproducible: 10,524,544 parameters, BPE tokenizer,
verified ONNX parity, and WebGPU bundle identity `91ac74b3…`; it remains unpublished because Hub
write authentication is absent. The audit therefore upgrades the deployment blocker from “unknown
public artifact” to “known legacy public artifact plus verified local candidate,” without claiming a
current public URL.

The [m577 gate](paper/results/raw/m577-workshop-gate-current-m553-public-audit-v1.json) now consumes
that public audit. It correctly reports the artifact blocker as `current_checkpoint_not_bound`
rather than “manifest absent”; the nine native benchmark requirements remain independently blocked.

### Current-checkpoint ToolSandbox native smoke (m578–m579)

The [m578 receipt](paper/results/raw/m578-toolsandbox-native-m553-v1.json) runs the pinned
ToolSandbox simulator and milestone verifier against m553 using the real `cellular_off`, `wifi_off`,
and `send_message_with_phone_number_and_content` scenarios. All three complete with similarity
`1.0`, no external API calls, and the exact current-checkpoint hash. This is native execution, but
the scripted one-turn user is not the upstream model-based user simulator and the official split is
not verified. Accordingly, the [m579 gate](paper/results/raw/m579-workshop-gate-current-m553-toolsandbox-v1.json)
records the blocker precisely as `official_split_not_verified` rather than promoting the smoke to a
ToolSandbox leaderboard score.

The follow-up [m580 stress receipt](paper/results/raw/m580-toolsandbox-m553-interactive-stress-v1.json)
uses three explicit multi-user-turn/state-dependency scenarios. The verifier reports `0/3` similarity
(two tasks stop after one turn and the relationship task consumes 13 turns), which is a useful
negative control: the current text-first policy does not yet reliably chain stateful mobile-style
tool calls even inside the native simulator. This result is kept separate from the 3/3 single-step
smoke and is not promoted to an official score.

### Stateful head-transfer bridge (m581–m583)

The [m581 probe](paper/results/raw/m583-stateful-head-transfer-toolsandbox-v1.json) trains only
route, dense-selector, and pointer heads on the deterministic local productivity state machine,
with the m553 backbone frozen and a matched random-backbone control. The warm arm improves local
selector top-1 from `46.67%` to `53.33%` and route accuracy from `81.25%` to `87.5%`, but closed-loop
completion remains `68.75%` (task completion `40%`) for both arms. This is intentionally labeled a
local synthetic probe: it uses no public benchmark rows or real accounts.

The native bridge is unchanged. The m553 parent and m581 child both pass the bounded single-step
ToolSandbox smoke (`3/3`), while both score `0/3` on the explicit multi-user-turn/state-dependency
stress. Thus selector-head fitting alone does not transfer to stateful native chaining; the child
is not promoted and the official ToolSandbox split/user simulator gate remains open.

### Current public-source snapshot audit (m584)

The [m584 audit](paper/results/raw/m584-public-dataset-snapshot-audit-v1.json) resolves eight
public Hub datasets to immutable revisions without downloading payloads: AndroidControl, AgentNet,
Mind2Web, xLAM, Computer Agent Arena, OSWorld 2.0 trajectories, OSWorld-Verified trajectories,
and EnterpriseOps-Gym. Each row retains its original repository link and an explicit policy. The
four train-eligible projections remain AndroidControl, AgentNet, Mind2Web, and xLAM; the desktop
arena, OSWorld trajectory archives, and EnterpriseOps-Gym remain evaluation/provenance-only. This
prevents mutable Hub revisions or benchmark trajectories from silently entering WebGPU SFT.

### Fresh public multisurface continuation and WebGPU bridge (m585)

The [m585 receipt](paper/results/raw/m585-public-multisurface-transfer-webgpu-v1.json) runs a
fresh 128-update continuation from m553 over 202 source-bound rows spanning mobile AndroidControl,
desktop AgentNet, browser Mind2Web, and MCPMark tool trajectories. On 53 source-local held-out
rows, warm initialization reaches `68.39%` teacher-forced token accuracy versus `46.74%` for the
matched random child (`+21.65` percentage points), with warm ahead on every surface. Exact sequence
accuracy remains `0%`, so this is representation-transfer evidence rather than end-to-end tool
success. Shared backbone movement is small for warm transfer (embedding `1.70%`, attention `0.52%`,
FFN `0.66%`) and large for random training (roughly `78–119%`), while action heads remain frozen.

The warm child is export/parity verified at `10,524,544` parameters. Native Apple Metal-3 WebGPU
dispatch reaches `3/3` structured actions at `1,367.5` input tok/s p50, `7.4 ms` p50 latency, and
`20.5 MB` conservative peak memory. The resettable local Gmail/Notion/browser state machine also
passes `13/13` transitions. No real accounts, screenshots, official benchmark environments, or
external side effects were used; the public native gates remain separate.

### Full native BrowserGym continuation on the warm child (m586)

The [m586 receipt](paper/results/raw/m586-browsergym-m585-warm-full-v1.json) executes the warm
m585 child in the pinned BrowserGym/MiniWoB environment at BrowserGym revision `9e779f0…` and
MiniWoB revision `7fd85d7…`. The official 240-task plan is verified and runs to completion without
coordinate or semantic fallback: `5/240` tasks pass (`2.08%`), with `219` grounded model steps,
`1,960` noop/ungrounded steps, and zero action errors. The successful families are
`miniwob.click-button` (4) and `miniwob.sign-agreement` (1).

This is a useful native browser-control receipt, but it remains a text/accessibility-tree
diagnostic: vision is disabled, and it is not a WebArena, OSWorld, real-account email/Notion, or
general computer-use score. The low grounded-step fraction is an explicit failure signal for
future data collection and RL rather than a publication-ready capability claim.
The receipt also satisfies the gate's canonical native schema for `browsergym_miniwob`; an isolated
gate join therefore marks this one requirement `pass` while leaving the overall publication decision
fail-closed.

### Full native MobileGym continuation on the warm child (m588)

The [m588 receipt](paper/results/raw/m588-m585-mobilegym-native-full-v1.json) runs the same warm
m585 checkpoint through the pinned MobileGym simulator and independent state-diff judge on all `256`
official test tasks. The text-first policy passes `1/256` (`0.39%`) with no runtime errors; its only
success is `crossapp_life.RestaurantRatingInviteCalendar`. The matched m553 parent also passed
`1/256`, so the warm public continuation changes closed-loop mobile success by `0.00` percentage
points despite improving held-out teacher-forced token accuracy in m585.

The run uses two model steps per task and a bounded DOM/text projection (`255` input-text calls,
`6` home navigations, `1` app open); it uses no screenshots, Android emulator, credentials, or
external side effects. This is a current-checkpoint native mobile receipt and a concrete negative
control for the weight-transfer hypothesis, not visual mobile-agent readiness.

### Current warm-child ToolSandbox native smoke (m589)

The [m589 receipt](paper/results/raw/m589-m585-toolsandbox-native-smoke-v1.json) executes the
warm m585 checkpoint in the pinned ToolSandbox simulator and independent milestone verifier on
`cellular_off`, `wifi_off`, and `send_message_with_phone_number_and_content`. All three scenarios
reach similarity `1.0`, with no external API calls. This confirms that the public continuation
preserves the parent’s one-turn tool-call contract across settings and a message action.

The protocol is intentionally bounded: the scripted user terminates after the first response, so
stateful multi-turn chaining, the model-based user simulator, the official split, and the broader
MCP service matrix were not executed. It is therefore native diagnostic evidence, not an official
ToolSandbox score or real-account email/Notion result.

### Matched Computer Agent Arena action-prior control (m590)

The [m590 receipt](paper/results/raw/m590-computer-agent-arena-m585-warm-parent-control-v1.json)
downloads the audited public `xlangai/computer-agent-arena` JSONL at revision `897b9f4…` (4,641
trajectories; 50.6 MB) and evaluates the same 256 unique tasks on both the m553 parent and m585
warm child. The instruction-only probe yields identical route accuracy (`100%`), tool exactness
(`1.56%`), family exactness (`1.95%`), and abstention (`96.88%`) on both arms.

This matched result is a useful weight-transfer negative control: the m585 teacher-forced gains do
not alter the desktop action prior when screenshots, accessibility trees, arguments, and later
trajectory state are absent. Pointer exactness is only `1.80%`; keyboard, observation, scroll,
type, and wait exactness are all `0%`. It is not a native desktop or visual-grounding score, and
the source remains evaluation-only.

### Matched AgentNet trajectory-projection control (m591)

The [m591 receipt](paper/results/raw/m591-agentnet-m585-warm-parent-control-v1.json) evaluates
the normalized public OpenCUA AgentNet holdout at revision `d76ee50…`: `16` held-out parent tasks
and `257` action rows, with the source-parent-disjoint train projection retained only for lineage.
The m553 parent and m585 warm child have identical prediction bytes and identical aggregate scores:
`75%` first-action type rate, `0%` exact trajectories, `0.02326` mean total score, and `0%`
task success.

This is stronger than an instruction-only action-prior check because it includes the public text
observation and multi-step projected action sequence, yet it still omits screenshots and native
Ubuntu state. The result is a clean negative transfer control: the m585 teacher-forced gain does
not translate into AgentNet trajectory behavior without visual grounding and a desktop runtime.

### Matched EnterpriseOps-Gym email retrieval control (m592)

The [m592 receipt](paper/results/raw/m592-enterpriseopsgym-m585-warm-parent-email-control-v1.json)
reacquires the public email Parquet shards at revision `c8e538e…` and evaluates all `67` rows with
the leakage-safe name-only dense-selector adapter. The m553 parent reaches `43.28%` top-1,
`76.12%` top-3, and `91.04%` top-5 retrieval; the m585 warm child reaches `56.72%`, `76.12%`,
and `91.04%`, respectively. Thus warm transfer adds `+13.43` percentage points at top-1 while
leaving top-3/top-5 coverage unchanged.

This is the first current-lineage email-specific transfer gain in the refreshed public shard
replay, but it is only a selector-ranking diagnostic: verifiers, MCP servers, database state,
credentials, and email side effects were dropped. It does not establish stateful email control or
real-account readiness.

### Matched MCPMark service-router control (m593)

The [m593 receipt](paper/results/raw/m593-mcpmark-m585-warm-parent-router-control-v1.json)
evaluates the pinned MCPMark task-description catalog at revision `cd45b7f…`: `169` standard and
`70` easy rows. The m553 parent and m585 warm child are identical: standard routing is `25/169`
(`14.79%`) and easy routing is `10/70` (`14.29%`). On the standard service breakdown, both route
all `25/25` Playwright rows correctly but score `0/28` Notion, `0/30` filesystem, `0/23` GitHub,
and `0/63` Postgres.

This is a direct Notion-related negative control: the public continuation does not change service
routing, and static task descriptions do not establish live MCP execution. Servers, verifiers,
accounts, and external side effects were not used.

### Current m585 warm-child stateful RL preflight (m594)

The [m594 receipt](paper/results/raw/m594-m585-stateful-rl-preflight-v1.json) runs the strict
isolated RL preflight against the current m585 warm checkpoint, using disjoint deterministic
email/Notion/browser Conversation rows. It completes two optimizer updates at learning rates
`0` and `2e-5`, changes all `40/40` named policy tensors, and leaves the production output and
source artifacts untouched. Held-out shaped reward rises from `0.0` to `0.08125`, but exact
match and strict tool-format validity remain `0`; there are zero exact-success rollouts.

The checkpoint-bound weight audit measures an overall relative L2 movement of `0.0003815`.
Relative movement is largest in FFN (`0.0006888`) and attention (`0.0005829`), followed by the
embedding (`0.0003716`); normalization is nearly frozen (`0.0000143`). This supports keeping the
pretrained backbone and using small continuation updates, but it is only a local simulator
diagnostic—not native API control, real-account execution, public benchmark success, or workshop
readiness.

### Actual warm-child GRPO continuation and local runtime (m596)

The [m596 receipt](paper/results/raw/m596-m585-stateful-grpo-runtime-v1.json) records a real
32-update SFT prelude followed by 8 pure-PyTorch GRPO updates from m585, with `128` attempted
rollouts and `27` informative groups. Shaped reward rises from `0.0344` to `0.1625`, while exact
sequence success remains `0`. The deployment-shaped resettable runtime reaches the oracle's
`100%` contract and the model completes `1/5` tasks (`20%`): email completes `6/6` steps, Notion
accepts `1/2`, and browser search, browser recovery, and abstention complete `0` tasks.

The m585-to-m596 weight audit shows `0.02819` overall relative L2 movement, almost entirely from
the SFT prelude. FFN (`0.02461`) and attention (`0.01912`) move far more than normalization
(`0.000519`), while all deployment heads remain frozen. The pointer-vocabulary metadata bug
found during evaluation is fixed in the training runner and covered by a regression test; the
child now loads in the deployment-shaped evaluator. These are still local state-machine results,
not public benchmark or real-account control.

### Current m585 workshop gate (m598)

The [m598 gate](paper/results/raw/m598-workshop-gate-current-m585-v1.json) rebinds the current
m585 checkpoint to the canonical WebGPU receipt, BrowserGym/MiniWoB, MobileGym, and m594 RL
preflight. Those six checks pass. Readiness remains fail-closed: AndroidWorld, MobileSafetyBench,
iOSWorld, OSWorld/OSWorld-V2, native AgentNet, MCPMark, and EnterpriseOps-Gym receipts are absent;
ToolSandbox is still only a non-official smoke; the current matched weight ablation is not in the
gate schema; and the public model/demo artifact is not bound to m585.

### Current m585 matched warm/random transfer ablation (m601–m602)

The [m601 receipt](paper/results/raw/m601-m585-current-transfer-ablation-v1.json) anchors both
arms to the m585 checkpoint and replays the same `202` public train rows and `53` source-local
held-out rows from AndroidControl, AgentNet, Mind2Web, and MCPMark. After `64` identical updates,
the warm arm reaches `69.65%` held-out assistant-token accuracy versus `25.34%` for the random
control (`+44.31` points), with warm ahead on every source: AndroidControl `+45.85`, AgentNet
`+55.41`, Mind2Web `+53.62`, and MCPMark `+18.01` points.

The weight audit is consistent with transfer rather than relearning: warm movement is `0.823%`
embedding, `0.301%` FFN, `0.233%` attention, and `0.026%` normalization relative L2, while the
random arm moves `119.25%`, `87.78%`, `77.84%`, and `8.62%`. Action heads stay frozen in both
arms. The [m602 gate](paper/results/raw/m602-workshop-gate-current-m585-transfer-v1.json) now
passes the current transfer/ablation requirement, while retaining the native-benchmark and public
artifact blockers.

### m585 Hugging Face/WebGPU release preparation (m603)

The [m603 receipt](paper/results/raw/m603-m585-hf-space-preparation-v1.json) rebuilds a
checkpoint-bound Hugging Face model bundle and static WebGPU Space staging directory from m585.
The safetensors model, auxiliary heads, tokenizer, four ONNX graphs, dispatch metadata, and static
Space all bind to checkpoint SHA `6553dc2b…` and `10,524,544` parameters. FP32 and FP16 backbone
and action-only graphs pass the hard parity gate.

Publication is deliberately recorded as `published: false`: this environment has no `HF_TOKEN`
and `hf auth` reports `Not logged in`, so no upload or public URL was invented. An authenticated
Hub upload followed by anonymous checkpoint-binding verification is still required before the
public-artifact gate can pass.

### Stateful head adaptation on m585 (m604)

The [m604 receipt](paper/results/raw/m604-m585-stateful-head-adaptation-v1.json) trains only the
route, dense-selector, and pointer heads for `320` updates while freezing the m585 backbone. On
the five-task local email/Notion/browser/recovery/abstention runtime, the warm arm reaches `68.75%`
closed-loop step success and `40%` task completion, with `100%` schema validity. Email and
abstention complete; browser reaches `3/4` steps but does not complete, Notion reaches `1/2`, and
recovery reaches `0/3`. A matched random-backbone control has the same end-to-end completion but
lower selector top-1 (`53.33%` vs `66.67%`).

The backbone relative L2 movement is exactly `0`; route, selector, and pointer heads move instead.
This isolates the remaining browser/recovery failures as stateful grounding and retry problems,
not evidence that the pretrained backbone should be discarded. The runtime remains a local
simulator, not a native browser, email, or Notion account.

### Reset/retry replay of the m604 child (m605)

The [m605 receipt](paper/results/raw/m605-m604-stateful-runtime-retry-v1.json) reruns the exact
m604 child through the deployment-shaped evaluator, where a rejected tool call does not advance
state and the decoder receives the same state plus an error observation. The oracle reaches `5/5`
tasks and `16/16` accepted steps. The model also reaches `5/5` tasks and `16/16` accepted steps,
but uses `28` attempts, for `57.14%` attempt success and a mean shaped reward of `0.7015`.
The result is deterministic across repeated invocations because the constrained decoder and
in-memory runtime are deterministic; it is still only a synthetic local result. In particular,
task completion under bounded retry must not be reported as a native BrowserGym, AndroidWorld,
MCPMark, email, or Notion score. The high rejection rate is the actionable deployment signal:
the next iteration should improve state-conditioned action grounding and error recovery rather
than claim that the model is ready for external side effects.

### Current policy-aligned public transfer and held-out desktop test (m607)

The [m607 receipt](paper/results/raw/m607-m585-policy-aligned-transfer-v1.json) runs a fresh
matched warm/random continuation from the exact m585 checkpoint. The warm arm trains `32` rows
from each of AndroidControl, Mind2Web, the public xLAM derivative, and ToolACE action-history
projections; it evaluates `8` source-disjoint rows from each of those sources plus `8` held-out
AgentNet desktop-projection rows. AgentNet is not used for training because the executable catalog
still keeps it evaluation-only pending terms/split review.

Warm held-out token accuracy is `73.45%` versus `10.56%` for the matched random backbone, a
`+62.90` point gap, and warm wins every surface: AgentNet `+74.90`, AndroidControl `+74.19`,
Mind2Web `+70.33`, ToolACE `+45.61`, and xLAM `+48.79` points. Exact sequence accuracy remains
`0%` for both arms. The weight audit shows transfer-shaped movement: warm embedding/attention/FFN
relative L2 is `0.418%/0.152%/0.183%` (normalization `0.009%`), while random movement is
`119.29%/77.88%/87.79%` (normalization `8.63%`); action heads are unchanged. This supports the
m585 body as an initialization candidate, not a policy export. ToolACE remains a matrix train
candidate but is explicitly not promoted into the executable catalog until its terms/split
metadata conflict is resolved. No emulator, browser, desktop VM, MCP service, email/Notion account,
screenshots, or external side effect ran.

### Current m607 workshop-gate refresh (m608)

The [m608 gate](paper/results/raw/m608-workshop-gate-current-m607-v1.json) binds the new m607
warm/random transfer receipt to the m585 checkpoint and reruns the fail-closed publication checks.
Seven requirements pass: catalog family coverage, no pending catalog adapter, native MobileGym,
native BrowserGym/MiniWoB, native WebGPU capability/latency, the m607 transfer ablation, and the
current RL preflight. Readiness remains `false` with ten blockers: AndroidWorld, MobileSafetyBench,
iOSWorld, OSWorld, OSWorld-V2, native AgentNet, MCPMark, EnterpriseOps-Gym, the official
ToolSandbox split, and a public model/demo manifest bound to m585. This confirms that the stronger
offline transfer result does not erase the native-environment or publication requirements.

### Browser-observed local WebGPU demo probe (m609)

The [m609 receipt](paper/results/raw/m609-webgpu-local-demo-probe-v1.json) is the first direct
browser observation of the current static bundle, rather than a file-only export check. The exact
10,524,544-parameter artifact loaded with a `WEBGPU` session; the email prompt routed to
`send_email` and stopped at the confirmation boundary, and the two-step search→Notion prompt
produced `web_search` followed by `notion_write`, again stopping before any external side effect.
The probe used no credentials, account, MCP service, or network write.

The same receipt records the quality negative control: “What does ephemeral mean?” was incorrectly
dispatched as a `computer_use/click` action (1,309 ms) instead of a text answer or abstention.
Therefore the local bundle/runtime and safety boundary are verified, but `publish_ready` and the
quality gate remain false. This is not a native email/Notion score, a BrowserGym result, or a public
Hugging Face/Space deployment.

### Route-abstention calibration and guarded WebGPU rerun (m610–m611)

The [m610 receipt](paper/results/raw/m610-m585-route-abstention-calibration-v1.json) tests the
actual failure mode from m609 with the m585 backbone and dense selector frozen. It adds public
AndroidControl, Mind2Web, xLAM, and ToolACE projections plus explicit no-tool/semantic rows and
trains matched warm/random route heads. The warm arm fixes the definition, explanation, and
acknowledgement probes and keeps email, Notion, and web-search routing correct on the seven-probe
set (`6/7`), but maps the GUI click probe to `app_action`. Its route-head movement is `178.36%`
relative L2 with zero backbone/selector movement, so it is not adopted or exported; this is a
useful negative transfer result, not a candidate model.

The [m611 receipt](paper/results/raw/m611-webgpu-semantic-guard-probe-v1.json) then adds a narrow,
explicit `semantic_text_safety_guard` in the browser adapter for unmistakable direct-answer or
acknowledgement prompts. A cache-busted browser rerun verified `What does ephemeral mean?` now
abstains as `text`, while email still requires confirmation and the two-step search→Notion plan
still stops before the Notion write. This guard is policy, not learned semantic competence; no
public HF upload, native benchmark, account, MCP service, or external side effect ran.

### AgentWorldBench public world-model projection (m612)

The [AgentWorldBench dataset](https://huggingface.co/datasets/Qwen/AgentWorldBench) and its
[Qwen-AgentWorld source](https://github.com/QwenLM/Qwen-AgentWorld) add a distinct public
realistic-evaluation surface: Android, Web, OS, MCP, terminal, SWE, and search trajectories.
The official test release contains 2,170 reference-grounded turns and reports format, factuality,
consistency, realism, and quality through its own judge; it is test-only here and is never a
training input. At pinned revision `6b8d28437042434dcdd168434227ca0de408c5ba`, the adapter selects
32 rows per domain (224 total), preserves each prior prompt/response pair as context, and binds
the normalized JSONL to its source-file SHA-256 values in the [m612 receipt](paper/results/raw/m612-m585-agentworldbench-text-projection-v1.json).

On the current m585 checkpoint, the bounded projection obtains `6.257%` assistant-token accuracy,
`7.2024` mean loss, and `0%` exact sequences overall; per-domain token accuracy ranges from
`4.475%` (search) to `9.424%` (OS). These numbers are a teacher-forced text/world-model
projection over the public test rows, not the official AgentWorldBench judge, action success,
screenshot score, native Android/OS/browser/MCP execution, or a claim that the model was trained
on the benchmark. The low exactness is an honest negative control: realistic environment coverage
is now source-bound, but native execution and public artifact publication remain open gates.

The [m616 catalog addendum](paper/results/raw/m616-agentworldbench-catalog-addendum-v1.json)
now makes this source discoverable through a validated, separate catalog file. It is intentionally
not merged into the frozen 42-entry publication gate catalog, so historical gate fingerprints stay
reproducible; the addendum is pinned to the same AgentWorldBench revision and explicitly sets
`train_policy: eval_only` and `training_admission: false`.

### AgentWorldBench held-out warm/random transfer (m617)

The [m617 receipt](paper/results/raw/m617-agentworldbench-transfer-v1.json) evaluates the two
matched m607 continuation children on the same 224 AgentWorldBench test rows without training on
that benchmark. The warm child reaches `6.407%` teacher-forced assistant-token accuracy versus
`0.184%` for the random-backbone control (`+6.223` points); exact sequence accuracy is `0%` for
both. Warm wins every domain, with the largest gaps on MCP (`+8.758` points) and OS (`+9.094`
points). This is a held-out transfer signal, not an official AgentWorldBench judge/native score.

The paired m607 weight audit remains transfer-shaped: warm embedding/attention/FFN movement is
`0.418%/0.152%/0.183%`, while random movement is `119.290%/77.882%/87.790%`; action heads stay
frozen. The receipt therefore supports reusing the m585 body as an initialization candidate, but
does not authorize WebGPU export or native benchmark promotion.

### Agent-Diff enterprise state-diff transfer (m618)

The [Agent-Diff dataset](https://huggingface.co/datasets/hubertmarek/agent-diff-bench), its
[official repository](https://github.com/agent-diff-bench/agent-diff), and the [paper](https://arxiv.org/abs/2602.11224)
add a realistic enterprise API surface: Slack, Linear, Box, and Google Calendar across 108
endpoints. The pinned public release has 179 train tasks and 45 test tasks under an 80/20 split;
the test rows are normalized as eval-only state-diff assertion targets.

On the same 45 test rows, the m607 warm child reaches `27.084%` teacher-forced assertion-token
accuracy versus `9.056%` for the matched random control (`+18.028` points). Warm wins every
service: Box `+16.698`, Calendar `+19.895`, Linear `+13.537`, and Slack `+21.045` points. Exact
sequence accuracy is `0%` for both. This is a text projection only: no sandbox replica, API,
state-diff verifier, credentials, or external side effect ran. The separate Agent-Diff addendum
keeps the source discoverable while leaving the frozen workshop catalog and WebGPU export gate
unchanged.

### Agent-Diff train-split continuation and weight audit (m619)

The [m619 receipt](paper/results/raw/m619-agentdiff-training-v1.json) runs a matched 32-step SFT
continuation on the public 179-row Agent-Diff train split, with all 45 test rows held out. The warm
child improves from `25.493%` to `35.421%` assertion-token accuracy (`+9.928` points); the random
control improves from `17.814%` to `20.262%` (`+2.448` points). Warm remains ahead after training
by `15.160` points. Exact sequence accuracy is still `0%`, so this is not task completion.

The weight audit shows the expected transfer pattern for this short continuation: warm
embedding/attention/FFN movement is `0.435%/0.228%/0.281%`, versus `0.697%/0.238%/0.305%` for the
random control; action heads remain unchanged. Test rows are explicitly excluded from training,
and neither child is promoted to WebGPU or native sandbox evaluation.

### ClawsBench productivity and safety source audit (m620)

The [ClawsBench metadata release](https://huggingface.co/datasets/benchflow/ClawsBench), [original
repository](https://github.com/benchflow-ai/ClawsBench), and [paper](https://arxiv.org/abs/2604.05172)
cover the requested productivity surfaces directly: Gmail, Google Calendar, Docs, Drive, and Slack.
The pinned release describes `44` tasks (`30` single-service and `14` multi-service), including
`24` safety-critical tasks, and `7,834` agent traces. Email contributes `8` tasks and multi-service
workflows contribute `12` tasks.

This is a source/protocol audit, not a score. The public snapshot exposes task metadata, traces, and
results, but not the Dockerized mock services and verifiers needed for native state evaluation; it
also has no static train/test split. The addendum therefore remains `eval_only`, and no ClawsBench
task, trace, safety label, credential, service state, or external side effect was used for training
or WebGPU execution.

### ClawBench live-web task source audit (m621)

The [ClawBench release](https://huggingface.co/datasets/TIGER-Lab/ClawBench), [official repository](https://github.com/TIGER-AI-Lab/ClawBench),
and [paper](https://arxiv.org/abs/2604.08523) add a complementary end-to-end browser surface. The
Apache-2.0 Hub revision contains a V1 test corpus of `153` tasks across `144` platforms and a V2
test corpus of `130` tasks across `63` platforms. The audit binds both Parquet files, `eval.yaml`,
and the shared dummy profile by SHA-256; V1 has `32` extra-context tasks and V2 has `15`.

The protocol is materially different from a static tool-call dataset: a live browser harness must
execute the instruction, intercept the final HTTP request, and optionally invoke the benchmark's
LLM judge. V1 contains `71/153` placeholder interception schemas, V2 contains `0/130`, and the
corpora share `51` task IDs, so neither corpus is admitted to training or treated as a disjoint
train/test split. The [m621 receipt](paper/results/raw/m621-clawbench-source-audit-v1.json) is
therefore provenance and protocol evidence only—no live site, credential, judge, or irreversible
request was executed, and no ClawBench score is claimed.

### MCP-Persona personalized MCP source audit (m622)

The [MCP-Persona repository](https://github.com/wwh0411/MCP-Persona) and [paper](https://arxiv.org/abs/2606.02470)
cover the requested personalized-tool setting directly: email, Notion-like content, Lark
calendar/collaboration, Slack, Obsidian, Reddit, Instagram, and Xiaohongshu. The pinned public
revision contains `173` English tasks and a matching `173` Chinese translation, `139` unique tools,
`18` task-chain server prefixes, and chains up to `18` calls. Its ground-truth records contain `118`
personalized-search checkpoints and `222` operate checkpoints; `24` tasks have no `gt` checkpoint.

The audit found that the English and Chinese releases are duplicate language views, not a train/test
split. The repository also ships static simulator directories for only `8` of the `18` chain-server
prefixes, and its README carries an MIT badge but the pinned checkout has no root `LICENSE` file.
Consequently the [m622 receipt](paper/results/raw/m622-mcp-persona-source-audit-v1.json) records
MCP-Persona as `eval_only` provenance. No task, persona context, ground-truth checkpoint, or
simulator state was admitted to WebGPU training, and no native MCP-Persona score is claimed.

### MCP-Persona held-out tool-chain transfer (m623)

To quantify the source without contaminating it, the English `173`-task release was projected to
the canonical `Conversation` schema as `instruction → compact JSON tool_chain`; persona contexts,
ground-truth checkpoint values, tool outputs, and simulator state were excluded. The existing
m607 policy-aligned warm child reaches `15.771%` teacher-forced assistant-token accuracy on this
eval-only projection versus `0%` for the matched random child (`+15.771` points); mean loss is
`6.3478` versus `9.6137`. Exact sequence accuracy is `0%` for both, including the Notion-first
subset (`23.339%` warm token accuracy) and universal-email-first subset (`16.761%`).

The [m623 receipt](paper/results/raw/m623-mcp-persona-tool-chain-projection-v1.json) is a transfer
diagnostic, not a personalized MCP capability score: no MCP server, context tree, checkpoint judge,
or external side effect ran, and the projection remains eval-only because MCP-Persona publishes no
train/test partition.

### MCP trajectory benchmark internal training and transfer (m624)

The public [MCP Agent Trajectory Benchmark](https://huggingface.co/datasets/obaydata/mcp-agent-trajectory-benchmark)
adds executable-looking tool-call arguments to the training analysis: `38` single-pass trajectories,
`282` recorded tool calls, and domains including finance, health, HR, logistics, marketing, and email
marketing. Because the Hub exposes only a `train` split, the adapter creates a deterministic
agent-disjoint internal holdout: `30` agent trajectories (`86` normalized decisions) for SFT and `8`
agents (`21` decisions) for evaluation. Eleven multi-conversation records were audited separately;
one is malformed JSON and none was admitted to training.

After matched 32-step continuation from m607, the warm child improves internal held-out tool-call
token accuracy from `38.564%` to `54.702%` (`+16.138` points). The matched random child improves
from `0%` to `10.907%`; warm remains `43.795` points ahead after training. Exact sequence accuracy
is `0%` for both. Warm shared-body movement stays small—embedding `0.431%`, attention/mixer
`0.243%`, FFN `0.301%`, normalization `0.011%`—while random movement reaches `119.261%`,
`77.883%`, `87.789%`, and `8.633%` respectively; action heads remain frozen.

The [m624 receipt](paper/results/raw/m624-mcp-trajectory-transfer-v1.json) is explicitly an
internal structural holdout, not an official benchmark split or native MCP score. Tool outputs,
reasoning traces, MCP servers, and external side effects were excluded.

### ToolSandbox protocol audit (m613)

The [m613 receipt](paper/results/raw/m613-toolsandbox-protocol-audit-v1.json) audits the pinned
[Apple ToolSandbox source](https://github.com/apple/ToolSandbox/tree/165848b9a78cead7ca7fe7c89c688b58e6501219)
without copying its scenarios into training. The official CLI resolves all `1,032` generated
scenarios when no scenario filter is supplied: `129` base scenarios plus seven tool/argument
augmentation families. The source exposes categories such as state dependency (`192`), multiple
user turns (`224`), canonicalization (`472`), and insufficient information (`224`), and its
protocol requires a user simulator for the official conversational run.

Crucially, this pinned repository does not publish a train/test split or a static leaderboard
subset. Therefore `official_split_verified` remains false by design; the existing three-scenario
native smoke is not promoted to an official score, and the evaluation gate retains an explicit
protocol blocker rather than inventing a split. The audit is eval-only provenance and protocol
evidence, not a model result or a user-simulator execution.

### Current m585 ToolSandbox native base diagnostic (m614)

Using the pinned simulator and milestone verifier, the current m585 checkpoint was run on all
`129` unaugmented base scenarios. The bounded scripted-user diagnostic completes `30/129`
scenarios exactly (`23.26%`) with mean milestone similarity `0.2577`; all failures are recorded
per scenario. Insufficient-information tasks are the strongest slice (`13/14` exact), while state
dependency, multiple-user-turn, and canonicalization slices remain near zero under the one-step
user policy. This is useful failure-driven evidence for stateful tool routing, but it deliberately
does not claim ToolSandbox leaderboard performance: the upstream user simulator, all 1,032
augmentation variants, and external RapidAPI services were not executed, and the source audit
confirms there is no published train/test split.

### Current fail-closed workshop gate (m615)

The [m615 gate receipt](paper/results/raw/m615-workshop-gate-current-m614-v1.json) refreshes the
publication checklist with the current m585 checkpoint and the 129-scenario native diagnostic.
Readiness remains `false` with ten blockers: eight missing native suites, ToolSandbox's
source-grounded `official_split_not_verified`, and the absent authenticated public model/demo
manifest. This is the correct outcome: the new native evidence strengthens diagnosis but does not
turn a scripted-user simulator run into an official benchmark result.

### m624 warm child WebGPU adoption check (m625)

The m624 warm child is now bound to a local release candidate rather than only a training receipt.
The exact `10,524,544`-parameter checkpoint (`984152…`) exports to an eight-artifact, 63-tool
bundle with the hard ONNX/PyTorch parity gate passing. The [m625 adoption receipt](paper/results/raw/m625-mcp-trajectory-webgpu-adoption-v1.json)
records the full lineage: the [public MCP Agent Trajectory Benchmark](https://huggingface.co/datasets/obaydata/mcp-agent-trajectory-benchmark)
revision and m624 warm/random transfer, the model and bundle hashes, and the native browser run.

Chromium 145 with the native WebGPU provider reports an Apple Metal-3 adapter, `3/3` exact local
structured actions (email, URL, and Notion-shaped dispatch), `1,290.3` input tokens/s p50,
`7.75 ms` p50 latency, and `20.46 MB` conservative peak memory. This is a capability and
deployment result, not end-to-end productivity success: the calls were local predictions only,
`closed_loop_success=0`, and no real account, navigation, MCP server, or external side effect ran.
The candidate is therefore locally WebGPU-adopted but not publicly published; authenticated Hub
upload and the remaining native mobile/desktop/service-backed benchmark gates are still required.

### AndroidControl mobile transfer from the m624 child (m626)

The next source-bound continuation uses the public [Android-Control-84k mirror](https://huggingface.co/datasets/OfficerChul/Android-Control-84k),
whose original project is [Google Research AndroidControl](https://github.com/google-research/google-research/tree/master/android_control).
The pinned Apache-2.0 mirror contributes `512` train rows and `256` test rows; the train/test
manifests and source hashes are recorded in the [m626 receipt](paper/results/raw/m626-androidcontrol-warm-random-transfer-v1.json).
Every row is explicitly screenshot-omitted, so this is a text/action projection rather than
visual mobile control.

Matched 64-step continuation from the m624 warm child raises held-out teacher-forced token
accuracy from `73.165%` to `82.406%`. The random-backbone control rises from `0%` to `63.126%`,
leaving a `19.280`-point warm advantage. Warm movement is `0.862%` embedding, `0.326%`
attention/mixer, `0.422%` FFN, and `0.027%` normalization; the random body moves `119.173%`,
`77.884%`, `87.793%`, and `8.615%` respectively. Deployment action heads remain unchanged.
This supports reusing the m624 weights as a mobile initialization candidate, but it does not
authorize WebGPU promotion or claim Android emulator, screenshot-grounding, AndroidWorld, or
MobileGym success.

### Current m624 workshop gate (m627)

The [m627 gate receipt](paper/results/raw/m627-workshop-gate-current-m624-v1.json) binds the
publication checklist to the exact m624 warm checkpoint rather than an older parent. Four checks
pass: realistic catalog coverage, runnable train adapters, native WebGPU capability, and the
m626 warm/random ablation. Readiness remains `false` with `13` explicit blockers: eleven native
mobile/browser/desktop/tool suites, current-checkpoint RL preflight, and the authenticated public
model/demo manifest. This is the correct publication decision; a fast local WebGPU action graph
and strong text-projection transfer are not substitutes for native stateful task success or a
publicly verifiable artifact URL.

### AndroidControl-adapted WebGPU child (m628)

The AndroidControl warm child was exported separately to ensure the mobile transfer itself, not
only its m624 parent, survives the deployment ABI. The [m628 receipt](paper/results/raw/m628-androidcontrol-webgpu-adoption-v1.json)
binds checkpoint `d6b5df…`, the 10,524,544-parameter model, all eight bundle artifacts, and the
hard ONNX/PyTorch parity gate. Chromium’s native WebGPU provider again reports Apple Metal-3:
`3/3` exact structured local actions, `1,242.2` input tokens/s p50, `7.70 ms` p50 latency, and
`20.46 MB` conservative peak memory. The adapted child therefore passes local export and native
capability checks, but remains unpublished and is not a native AndroidControl score: screenshots,
an emulator, real accounts, and external side effects were absent.

### Full native MobileGym result for the m626 child (m629)

The [m629 receipt](paper/results/raw/m629-androidcontrol-child-mobilegym-native-v1.json) runs
the AndroidControl-adapted checkpoint through the complete pinned MobileGym test split: `256/256`
tasks, official split verified, zero runner errors, and the official state-diff judge enabled. It
passes only `1/256` tasks (`0.3906%`). The dominant emitted action is `mobile_input_text`
(`255` calls), with only six `mobile_navigate_home` and one `mobile_open_app` calls. This exposes
the actual deployment gap: text/action token transfer and WebGPU dispatch do not provide mobile
state grounding or multi-step action planning. The result is therefore a native diagnostic and a
hard blocker for mobile promotion, not a visual-agent success claim.

### Current m626 workshop gate with native MobileGym (m630)

The [m630 gate receipt](paper/results/raw/m630-workshop-gate-current-m626-v1.json) updates the
checklist to the AndroidControl-adapted checkpoint. Four checks pass: catalog coverage, native
MobileGym split execution, native WebGPU capability, and runnable train adapters. Readiness is
still `false`, now with `12` blockers. A fresh m626-parent MCP trajectory ablation is included, so
the current weight gate passes alongside catalog, MobileGym, and WebGPU checks. The remaining
blockers are the ten missing native suites, current-checkpoint RL preflight, and authenticated
public model/demo artifacts. This prevents a native MobileGym result from being combined with
stale transfer evidence to manufacture workshop approval.

The fresh [m631 ablation receipt](paper/results/raw/m631-m626-mcp-warm-random-transfer-v1.json)
continues the m626 child on the MCP trajectory projection: warm held-out accuracy rises from
`49.304%` to `56.093%`, while the matched random body reaches `10.907%`, a `45.186`-point warm
advantage. Warm shared-body movement is `0.442%` embedding, `0.204%` attention/mixer, `0.259%`
FFN, and `0.010%` normalization; action heads remain frozen. This is the current-checkpoint
weight evidence used by m630, while still remaining an internal projection rather than a live MCP
score.

### Current m679 native MCPMark filesystem/standard replay (m697)

The [m697 receipt](paper/results/raw/m697-m679-mcpmark-filesystem-standard-v1.json) runs the exact
current m679 warm checkpoint on all `30` pinned MCPMark Verified `filesystem/standard` fixtures at
revision `cd45b7f…`. Each task gets a fresh extracted public state archive, a real
`@modelcontextprotocol/server-filesystem` stdio process, bounded six-turn tool interaction, and
the task's independent verifier. All `30/30` tasks execute without runner exceptions, but `0/30`
verifier checks pass. The failures are substantive stateful planning and argument-grounding
errors, rather than a missing service process.

This is stronger evidence than the earlier description-only MCPMark router: the current
checkpoint caused real filesystem side effects in isolated workspaces and was judged by public
verifiers. It also sharply limits the deployment claim. This is an official filesystem service
subset, not a complete MCPMark score; Notion, GitHub, Postgres, Playwright, the user simulator,
and external accounts remain unexecuted. Promotion therefore stays blocked; the warm backbone is
an initialization candidate only, with state-conditioned planning, argument grounding, and native
visual/service evaluation still required before WebGPU publication or workshop submission.

### MCPMark public trajectory continuation does not transfer to native state (m698)

The [m698 receipt](paper/results/raw/m698-m679-mcpmark-continuation-native-v1.json) trains the
exact m679 warm child for `128` updates on the public `Jakumetsu/mcpmark-trajectory-log` train
split (`10` rows), holding out its `5`-row eval split.  Held-out teacher-forced token accuracy
improves from `30.07%` to `35.73%` (`+5.66` points), while sequence exactness remains `0%`.

The resulting child is then run through the same `30` pinned MCPMark Verified
`filesystem/standard` tasks used in m697, with fresh state archives, a real stdio MCP server,
and independent verifiers.  It executes all tasks without runtime errors but passes `0/30`
verifiers, exactly matching the parent’s native pass count.  This is a controlled transfer
failure: text imitation improves, but it does not produce reliable stateful file operations.
The child is therefore not promoted or exported; the evidence supports warm initialization while
requiring explicit state-conditioned planning, argument grounding, and native service training.

### Current m679 publication gate after native transfer checks (m699)

The [m699 gate](paper/results/raw/m699-workshop-gate-current-m679-v1.json) binds the latest
MCPMark native replay and m698 continuation to the exact m679 checkpoint.  It remains explicitly
`ready: false`: the current child has `0/30` MCPMark filesystem verifier passes, the native
AppWorld arms remain `0/6`, and EnterpriseOps-Gym still has no server/verifier execution.  The
gate also retains the missing visual/native suites and the unauthenticated, legacy public HF/
Space artifact blocker.  This prevents the +5.66-point text-only MCP continuation gain from being
misrepresented as browser, email, Notion, or computer-use readiness.

### Native MCPMark Playwright browser-service probe (m700)

The [m700 receipt](paper/results/raw/m700-m679-mcpmark-playwright-standard-v1.json) exercises
the exact m679 checkpoint against two public MCPMark Verified Playwright tasks using the real
`@playwright/mcp@0.0.68` stdio server and an isolated Chromium executable.  On the extraction
task, the model successfully reaches the public MCPBench page with `browser_navigate` and reads
its accessibility snapshot with `browser_snapshot`, both without server errors.  The second task
emits an invalid `browser_run_code` request and the browser reports a tool error.  Both task
verifiers fail (`0/2`) because the model does not produce the required final extracted content.

This is the first native browser-service evidence in the current m679 chain, but it is deliberately
bounded: only two tasks ran, the browser executable is not the benchmark’s pinned version, and
there is no official split, visual-answer, user-simulator, WebArena, or external-account claim.
The result confirms that WebGPU deployment can reach a real browser MCP boundary while exposing
the remaining failure as final-answer extraction and tool-code grounding, not merely server setup.

The [m701 gate](paper/results/raw/m701-workshop-gate-current-m679-v1.json) adds this browser
receipt without promoting it: `native:mcpmark_playwright` is explicitly blocked by
`bounded_subset_and_verifier_zero`. The gate remains fail-closed until the complete benchmark
split, pinned browser parity, visual grounding, and successful final-answer verification are
available.

### Browser-specific SFT weight transfer does not improve native Playwright (m705/m706)

The [m705 receipt](paper/results/raw/m705-m679-browser-transfer-native-v1.json) isolates the
browser surface: two public MCPMark browser trajectories are used for `256` SFT updates from the
exact m679 warm checkpoint, with one source-disjoint browser trajectory held out.  Held-out
teacher-forced token accuracy rises from `52.43%` to `73.16%` (`+20.72` points), while sequence
exactness remains `0%`.  Shared-body movement is modest (about `4.0%` embedding, `2.5%` FFN, and
`2.1%` attention relative L2), and action heads remain frozen, supporting reuse of the warm
backbone with low-rate surface adaptation.

The adapted child is then replayed through all four native MCPMark Playwright standard tasks.  It
still records `0/4` verifier passes and the same two invalid browser-code errors as the parent.
Thus the text gain does not transfer to stateful browser execution; the child is not promoted.
The [m706 gate](paper/results/raw/m706-workshop-gate-current-m679-v1.json) records this explicitly
as `browser_sft_native_verifier_zero` and keeps publication blocked.

### Official AndroidControl screenshot provenance (m707)

The [m707 receipt](paper/results/raw/m707-androidcontrol-official-tfrecord-visual-sample-v1.json)
audits the original Google Research AndroidControl source rather than the screenshot-omitting HF
projection used by the current training pilots.  A bounded range fetch from the public
`android_control-00000-of-00020` gzip TFRecord shard parses train episode `0` and its public split
manifest: the episode has four PNG screenshots totaling `708,631` bytes at `1080×2400`, four
serialized accessibility trees totaling `63,519` bytes, three step instructions, and three JSON
actions (`open_app`, `wait`, `click`).  The source object and original repository URLs, object
size/MD5 metadata, range hash, record hash, screenshot hash, and split membership are all bound in
the receipt; screenshot pixels are not retained in Git.

This closes the provenance question but confirms a deployment gap: the current LocalAgent
Conversation projection and WebGPU bundle consume text/accessibility/action fields only, not these
visual bytes.  The receipt therefore admits the visual source as provenance-only until a vision
encoder, screenshot-aware training split, and native emulator evaluator are implemented; no
AndroidControl visual or AndroidWorld score is claimed.

### Full bounded MCPMark Playwright standard subset (m702/m703)

The [m702 receipt](paper/results/raw/m702-m679-mcpmark-playwright-standard-v1.json) expands the
browser-service probe to all four pinned Playwright standard tasks: extraction, Cloudflare
Turnstile, biographical web search, and arXiv search.  Every task starts through the real
`@playwright/mcp@0.0.68` server with zero runner exceptions.  The model reaches navigation and
accessibility snapshots on the extraction/Turnstile tasks, but emits invalid browser code on the
two search tasks; all four independent verifiers fail (`0/4`) and two browser-tool errors are
recorded.  This is the complete locally reproducible Playwright standard subset, but not an
official MCPMark score because the benchmark browser version, full cross-service split, visual
answer quality, and user simulation are not reproduced.

The [m703 gate](paper/results/raw/m703-workshop-gate-current-m679-v1.json) keeps the result
blocked as `bounded_subset_and_verifier_zero`.  It demonstrates that the WebGPU-oriented
checkpoint can cross a live browser MCP boundary, while the remaining gap is reliable multi-step
browser reasoning and final-answer extraction.

### AgentNet visual archive provenance (m708)

The [m708 receipt](paper/results/raw/m708-agentnet-visual-source-audit-v1.json) pins the public
[xlangai/AgentNet](https://huggingface.co/datasets/xlangai/AgentNet) revision
`d76ee50a63fad81cfdbe576416757d7c2091ed50`, whose original project is
[OpenCUA](https://github.com/xlang-ai/OpenCUA).  Its manifest exposes the 282 MB Ubuntu trajectory
JSONL, the 1.40 GB Windows/Mac trajectory JSONL, and the merged metadata JSONL, alongside an
Ubuntu image archive of about 3.73 GB plus split volumes and a Windows/Mac archive of about 856 MB
plus split volumes.  A bounded prefix of the Ubuntu source contains three complete desktop rows,
26 unique screenshot filename references, and `click`, `doubleClick`, `hotkey`, `moveTo`, `press`,
`rightClick`, and `terminate` pyautogui action families.  The metadata sample spans Windows Excel,
Windows WPS Office, and Darwin Chrome tasks with screen dimensions and application/domain labels.

This is source provenance, not a visual benchmark result: the local AgentNet projection still
consumes text/actions and omits screenshot bytes.  The receipt therefore admits AgentNet visual
data only as `provenance_only_until_image_loader_vision_encoder_and_visual_eval_are_bound`; no
native desktop, AgentNetBench, OSWorld, or WebGPU visual success is claimed until those components
and an executable evaluator are bound.

### Current m679 AgentNet selector transfer and weight decision (m709)

The [m709 receipt](paper/results/raw/m709-m679-agentnet-selector-transfer-v1.json) runs a matched
`400`-step, frozen-backbone selector adaptation on the exact m679 warm checkpoint using `765`
public AgentNet projection rows and a disjoint `257`-row projection.  The random-control arm uses
the same optimizer, batches, seed, tool surface, and public inputs.  Both arms replay the same
eight held-out parent trajectories with complete prediction/ground-truth coverage and reach
`100%` first-action-type rate but `0%` success and `0%` exact trajectories.  Warm selector relative
movement is `3.766×` versus `2.397×` for random; its bounded text score is `0.000582` versus
`0.000003` for random, so the warm AgentNet selector is explicitly rejected for WebGPU adoption.

This is useful negative transfer evidence for weight reuse: changing a surface-specific head cannot
substitute for screenshot grounding, coordinate localization, or a desktop runtime.  The source
revision is pinned, but the official AgentNet split is not verified, screenshots are not consumed,
and no native desktop or WebGPU visual claim is made.

### Current m679 workshop gate after AgentNet audit (m710)

The [m710 gate](paper/results/raw/m710-workshop-gate-current-m679-v1.json) binds the exact m679
checkpoint to both the [m708 visual-source receipt](paper/results/raw/m708-agentnet-visual-source-audit-v1.json)
and the [m709 weight-transfer receipt](paper/results/raw/m709-m679-agentnet-selector-transfer-v1.json).
It remains fail-closed with `12` blockers.  AgentNet is now evidenced as a pinned public source and
matched text-transfer diagnostic, but its native requirement is explicitly
`visual_runtime_and_native_verifier_missing; text_projection_success_zero`; the adapted selector is
not a WebGPU candidate.  This prevents a successful source audit or a selector weight movement
number from being misreported as realistic computer-use competence.

### First public screenshot-to-model bridge (m711)

The [m711 receipt](paper/results/raw/m711-androidcontrol-visual-bridge-smoke-v1.json) is the
first end-to-end wiring check from an original public screenshot object into model computation.  A
bounded range from the official AndroidControl TFRecord was parsed, one `1080×2400` PNG was decoded
without Pillow/torchvision, and the new `webgpu-10m-vision` sibling produced a `36`-token visual
prefix with `10.61M` parameters.  With the language-model backbone frozen, one auxiliary
goal-token update produced finite loss `8.691`, gradient norm `10.910`, and nonzero visual-bridge
parameter movement `0.296`.

This closes the previously missing byte-to-encoder path, but it is intentionally only a wiring
smoke: the target is the public goal text, no action-quality dataset was trained, no emulator or
desktop verifier ran, and ONNX/WebGPU export is not implemented for the visual prefix.  The receipt
therefore does not promote the sibling checkpoint or change the `m710` workshop gate.

### Bounded AndroidControl visual-action transfer (m712)

The [m712 receipt](paper/results/raw/m712-androidcontrol-visual-action-pilot-v1.json) extends the
bridge smoke into action supervision.  A `50 MiB` range from the official AndroidControl train
shard yielded `12` complete source records (`8` train episodes, `4` held-out episodes) and `64`
frozen-visual-bridge updates.  Screenshots were decoded and resized before batching; the warm arm
loaded the exact m679 text backbone, while the random arm used a matched fresh backbone and the
same optimizer/seed schedule.  Warm held-out action-token loss fell `8.072→6.760`, random
`9.849→8.719`; however, both arms remained at `0%` action-sequence exactness, and random had
higher action-token accuracy (`19.29%` vs `6.43%`).

This is a useful weight-reuse boundary, not a success claim: the warm backbone helps loss but does
not produce reliable action sequences, and the visual adapter is not promoted.  The [m713 gate](paper/results/raw/m713-workshop-gate-current-m679-v1.json)
adds an explicit `visual_pilot_sequence_exact_zero_and_no_emulator_verifier` blocker; no native
AndroidControl, AndroidWorld, MobileGym visual, or WebGPU visual score is claimed.

### Structured screenshot-conditioned action head (m714)

The [m714 receipt](paper/results/raw/m714-androidcontrol-structured-visual-pilot-v1.json) replaces
raw action-byte scoring with the AndroidControl action contract: `click`, `input_text`,
`long_press`, `navigate_back`, `open_app`, `scroll`, and `wait`, plus normalized `x,y` pointer
regression where present.  It uses the same 12 source episodes and complete-record 8/4 split as
m712, freezes the decoder backbone, and trains a compact visual bridge plus structured head for 64
updates.  Warm and random both reach `58.62%` held-out action-type accuracy; warm coordinate MAE is
`0.2787` versus `0.2701` random.  Thus the explicit control interface is measurable, but pretrained
weight reuse does not yet improve the held-out structured metric and neither arm has native task
verification.

The [m715 gate](paper/results/raw/m715-workshop-gate-current-m679-v1.json) keeps the result blocked
as `structured_action_head_native_emulator_unverified`.  The head is not exported to WebGPU or
promoted as AndroidControl/AndroidWorld/MobileGym visual competence.

### Trained visual sidecar ONNX ABI and parity (m716)

The [m716 receipt](paper/results/raw/m716-structured-visual-onnx-parity-v1.json) binds the trained
m714 warm sidecar to an explicit image/action export contract.  Inputs are `input_ids`,
`context_lengths`, and float32 RGB `96×96` images; outputs are seven Android action logits and a
normalized two-coordinate pointer.  CPU ONNX Runtime agrees with PyTorch within maximum absolute
differences of `8.64e-7` and `1.19e-7` for the two outputs.

This closes the checkpoint-to-graph parity gap without overstating deployment: the graph has not
been run through onnxruntime-web/WebGPU, integrated into the browser demo, quantized, or validated
against an Android emulator.  The [m717 gate](paper/results/raw/m717-workshop-gate-current-m679-v1.json)
therefore adds `cpu_onnx_parity_only_no_browser_webgpu_runtime` and keeps publication blocked.

### Browser execution of the visual sidecar (m718)

The [m718 receipt](paper/results/raw/m718-visual-webgpu-probe-v1.json) closes the narrow
checkpoint-to-browser runtime gap for the experimental visual graph.  The static
`visual-action.html` page fetched and hash-checked the 42.3 MB fp16 graph, created an explicit
onnxruntime-web WebGPU session, resized a local screenshot to RGB `96×96`, and emitted the seven
Android action logits plus normalized pointer.  Warm WebGPU inference was `14.5–18.8 ms`; a WASM
control produced the same `click` decision with maximum differences of `3.58e-7` in logits and
`8.94e-8` in pointer coordinates.

This is still a runtime ABI probe, not visual task success: the screenshot is not connected to an
Android emulator, no action was dispatched to a device, and ORT does not expose per-node placement.
The [m719 gate](paper/results/raw/m719-workshop-gate-current-m679-v1.json) replaces the earlier
CPU-only visual blocker with `browser_visual_probe_bound_but_native_mobile_verifier_missing` and
keeps publication blocked.

### Scaled AndroidControl visual transfer and weight reuse (m720)

The [m720 receipt](paper/results/raw/m720-androidcontrol-structured-visual-scaling-v1.json)
extends the official Google AndroidControl screenshot path from the 50 MB pilot to a bounded 200 MB
prefix.  It yields 57 complete episodes, with the final eight episodes held out by complete record
(276 training rows and 40 held-out rows).  Both arms freeze the decoder and train only the vision
bridge plus structured action/pointer head for 256 updates.  Warm and random tie at `52.5%`
held-out action accuracy; warm reaches lower action loss (`1.3745` vs `1.4463`) but worse pointer
MAE (`0.2893` vs `0.2753`).

This is the relevant weight-reuse result: decoder initialization improves the auxiliary loss but
does not improve action selection and harms coordinate transfer on this split.  The [m721 gate]
(paper/results/raw/m721-workshop-gate-current-m679-v1.json) therefore keeps the visual artifact
blocked and records no warm promotion.  The public Android-Control JSON mirror remains a separate
text/action source because its manifest explicitly omits screenshot bytes; it is not substituted for
the original visual TFRecord.

### Current WebGPU release checkpoint visual transfer (m722)

The [m722 receipt](paper/results/raw/m722-current-checkpoint-structured-visual-v1.json) repeats
the 200 MB / 256-step experiment from the actual WebGPU release checkpoint
`6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45`, rather than the older m679
diagnostic parent.  The complete-record split is unchanged (49 train episodes, 8 held out).  The
current warm arm and random arm both reach `52.5%` held-out action accuracy; warm action loss is
lower (`1.3687` vs `1.4463`) but pointer MAE is worse (`0.2919` vs `0.2753`).  The warm visual
sidecar exports with CPU ONNX parity (`1.19e-6` logits, `5.96e-8` pointer), but the transfer
decision remains `do_not_promote_current_release_visual_transfer`.

The [m723 current-release gate](paper/results/raw/m723-workshop-gate-current-release-v1.json)
therefore binds the current checkpoint without claiming that its visual graph has run in the browser
or passed a native Android verifier.  This is the evidence needed to prevent an older-parent visual
result from being mistaken for the deployable WebGPU model.

### Current-checkpoint browser visual execution (m724)

The [m724 receipt](paper/results/raw/m724-current-checkpoint-visual-webgpu-v1.json) binds the
current release checkpoint and its fp16 visual sidecar to the parameterized browser probe.  The
browser hash-checked `visual_action_model.current.fp16.onnx`, created an explicit WebGPU session,
and emitted the seven action logits plus normalized pointer for a screenshot/task input.  First
inference was `77.8 ms`; three warm runs were `24.2`, `24.9`, and `26.5 ms`.

This closes the current-checkpoint-to-browser ABI gap, not the task-success gap.  The screenshot was
not connected to an Android emulator, no action was dispatched, and ORT does not expose per-node
placement.  The [m725 gate](paper/results/raw/m725-workshop-gate-current-release-v1.json) therefore
keeps the release blocked on native mobile verification and official task success.
