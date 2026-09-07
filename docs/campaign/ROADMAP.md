# Roadmap

Phased build order. Each phase is independently runnable and leaves the repo in a working state.
Stubs reference these phase numbers in their `TODO(phase-N)` markers.

## Phase 0 — Scaffold (this PR)
- [x] Research synthesis ([RESEARCH.md](./RESEARCH.md)) and architecture ([ARCHITECTURE.md](./ARCHITECTURE.md)).
- [x] Repo skeleton: package layout, configs, CLI, demos, tests, `pyproject.toml`.
- [x] Canonical `Conversation` schema and tool registry interface.
- [x] Model config + param-budget assertion; tiny Llama-style block implemented.
- Outcome: `pip install -e .`, `localagent --help`, `pytest` (schema/model/config tests) pass.

## Phase 1 — Model & tokenizer
- [x] Finish the decoder (GQA + RoPE + SwiGLU + KV cache), forward/generate.
- [x] Train BPE tokenizer with agent special tokens; round-trip tests.
- [x] `tiny-30m` constructs and runs a forward pass on CPU.
- Exit: untrained model generates tokens; param count < 100M asserted.

## Phase 2 — Pretrain
- [x] Corpus download + quality filter + document-level split + shard packing.
- [x] `pretrain.py` loop (AdamW, cosine/WSD LR, grad accum, ckpt/resume) on CPU & GPU.
- [x] `speedrun.sh` pretrains a toy model end-to-end.
- [x] Preserve the observed exploratory seed-2026 10.5M matched proxy, then complete the clean
      prospective seed-2027–2029 confirmation with exact held-out scorecards
      ([summary](./paper/results/webgpu-proxy-1tpp-10m-seeds2027-2029.summary.json);
      [raw bundle](./paper/results/raw/pretrain-proxy-seeds2027-2029/)).
- [ ] Externally timestamp, then complete the prespecified 35M,
      at-least-five-token-per-parameter, multi-seed architecture screen before promotion to the
      20-TPP/downstream comparison.
- [ ] Train a latency-feasible 10M-class candidate at the full token budget for the autoregressive
      deployment track. The measured 34M hybrid misses 100 tokens/s at every context and remains
      only a structured-action/scientific candidate without runtime gains.
- [ ] Isolated Muon/MuonClip optimizer ablation with stability metrics.
- Exit: loss decreases; coherent-ish completions at toy scale.

## Phase 2.5 — Domain midtrain
- [x] Scheduled general/code/agent mixture with masked canonical conversations.
- [x] WSD continuation, resumable checkpoints, and deterministic per-source held-out pre/post
      metrics.
- [ ] Run token-budget/mixture ablations on the WebGPU tier.
- [x] Run the confirmatory seed-2027 10M hybrid through a bounded midtrain/SFT/offline-RL pilot
      with frozen held-out metrics and artifact lineage. Midtrain and SFT improved their targeted
      held-out metrics, but midtrain slightly regressed general replay and RL produced zero
      held-out delta; this does not complete the 34M five-TPP screen or the subsequent
      20-TPP/downstream quality selection.
- [x] Implement and freeze the corrected fixed-compute browser arm: append filler after the
      natural assistant marker, dispatch from that marker's hidden position, and bound pointer
      scans to natural tokens.
- [ ] Externally timestamp v0.4, then run three corrected action and DOM browser sessions.
      Train/evaluate the prespecified
      pre-marker-padding materialization separately only if claiming genuine fixed-512-context
      capability.
- [x] Add an opt-in, budget-counted screenshot patch bridge and decode one official AndroidControl
      PNG through a frozen-backbone PyTorch wiring smoke; no visual quality claim is admitted.
- Exit: agent/code scores rise without unacceptable general-validation regression.

## Phase 3 — Agent data
- [x] `agent_synth.py`: deterministic synthesis → irrelevance negatives → schema verification.
- [x] Export rule-audited, non-executed `Conversation`s as JSONL with a dataset manifest, frozen
      train/eval split, and pinned exact-prompt holdouts.
- [x] Offline, identity-pinned xLAM and Mind2Web TRAIN importers with canonical `Conversation`
      output, deterministic enrichment, provenance, license/split policy, exact-prompt
      decontamination, and self-hashed manifests.
- [ ] External teacher/verifier adapters.
- [x] ToolACE-format importer: byte-pinned Apache-2.0 snapshot, strict first-action and opt-in
      multi-turn/action-history projections, explicit prose-omission accounting,
      prompt/parent-disjoint held-out splits, and matched warm/random transfer receipts.
- [x] ToolACE WebGPU-shaped dispatch probes: catalog-aware selector and dynamic pointer transfer
      arms, matched random controls, free-run replay, and catalog-bound pointer grounding.
- Exit: a few-thousand-sample rule-audited agent dataset on disk, with executed-environment
  verification required before any stronger claim.

## Phase 4 — SFT
- [x] `sft.py`: config-driven, loss-masked chat+tool training with route/select/copy heads.
- [x] Export and measure the seed-2027 SFT structured action graph in three WebGPU sessions.
      Fixed-512 TTFA was fast, but only 1/20 cases was exact and it was the abstention case.
- [x] Run the 1M parent-anchor continuation with immutable archives and frozen retention gates.
      Step 12 was the strict teacher-forced winner, but its bound development greedy scorecard
      had 170/820 EOS completions and zero valid protocol outputs; confirmatory evaluation,
      fallback selection, RL, and candidate WebGPU promotion were correctly withheld.
- [x] Schema-preserving function masking / tool-renaming augmentation, with deterministic aliases,
      full-catalog prompt support, and a matched ToolSandbox projection transfer receipt.
- [ ] Test explicit autoregressive protocol distillation or the full-budget 10M tier; additional
      teacher-forced loss reduction alone is not a justified continuation of the failed 1M lane.
- [x] Tool format wired through tokenizer + parser.
- Exit: model emits valid `<tool_call>`s and abstains on irrelevant queries.

## Phase 5 — Eval harness
- [x] AST call matching and single/parallel/irrelevance benchmark primitives.
- [x] Multi-turn simulated-tool replay and deterministic browser microaction harnesses.
- [x] Preserve three fixed-512 complete-action runs and three executable local-DOM runs for the
      SFT pilot. The capability gate failed: all tool-required actions abstained and DOM success
      was 0/8 unique tasks (0/720 repeated timing opportunities).
- [ ] Config-driven frozen-suite runner plus general-text evaluation.
- [x] Public real-use snapshot audit and explicit quality/coverage/router-diversity promotion
      gates layered over strict AST and teacher-forced multi-turn scoring.
- Exit: a reproducible scorecard for any checkpoint.

## Phase 6 — Distillation
- [x] `distill.py`: off-policy top-k/logit KD and on-policy reverse-KL experiments.
- [ ] General teacher adapter and trajectory-level distillation with concise decision prefixes.
- Exit: distilled student beats SFT-only baseline on the tool-eval scorecard.

## Phase 7 — Agent runtime + memory + demos
- [x] `agent/runtime.py` checkpoint-backed tool loop with tokenizer-aware loading.
- [ ] `agent/memory.py` two-tier memory as tools.
- [ ] Finish `chat_cli.py`; the web demo and WebGPU benchmark surfaces are implemented.
- Exit: hold a multi-turn, tool-using, memory-backed conversation locally.

## Phase 8 — Data flywheel
- [ ] Conversation store (SQLite) + feedback schema (AITL).
- [x] Deterministic synth → train → evaluate → failure-reweight research loop.
- [ ] Production ingest → mine → verify → append → retrain → evaluate → redeploy loop.
- Exit: a logged session produces verified new training samples and a retrain run.

## Phase 9 — Export (CPU/GPU/NPU)
- [x] ONNX/WebGPU and Hugging Face export with tokenizer/head metadata, fail-closed graph parity,
      and browser-side artifact-byte verification for benchmark runs.
- [x] Separate prompt-prefill/fixed-one-token cache-bearing ONNX graphs with multi-step trajectory
      parity and GPU-buffer cache rebinding in the standalone browser latency harness.
- [x] Bind the exploratory seed-2026 pretrain checkpoints to parity-gated graphs and three
      preserved browser runs; keep held-out quality separate and fail the joint gate when p95
      TPOT exceeds its threshold.
- [x] Export the seed-2027 SFT checkpoint as a parity-gated fp16 one-forward action graph and bind
      its exact checkpoint/tokenizer/head/graph identities to preserved browser payloads.
- [ ] Integrate cache-bearing decode with the trained complete-action autoregressive controls.
- [ ] Add screenshot-prefix ONNX/WebGPU graphs, browser preprocessing parity, and a native visual
      evaluator; the current `webgpu-10m-vision` config is intentionally not exportable.
- [ ] GGUF, ExecuTorch, and real Q4 converters with runtime parity tests.
- Exit: same weights run via PyTorch, llama.cpp, ONNX Runtime, and ExecuTorch with matching outputs.

## Phase 10 — RL (optional)
- [x] Config-driven autoregressive clipped GRPO with exact normalized tool-AST/text rewards, format
      shaping, reference KL, and explicit structured-head invalidation after backbone updates.
- [x] Run a bounded seed-2027 offline-RL pilot: 6/128 groups were informative, 12 optimizer
      updates were realized, and every held-out metric was unchanged.
- [x] Enforce candidate-bound readiness: the retained 1M step-12 candidate failed development, so
      no confirmatory, RL-preflight, production-RL, or reward artifact was created for it.
- [ ] BrowserGym/MiniWoB-style multi-step environments and final-state task rewards.
- Exit: measurable lift on multi-turn task success.
