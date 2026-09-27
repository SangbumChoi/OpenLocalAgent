# Workshop insight: fast is not an action

Status: submission-thesis candidate, 2026-08-01
Target: [SLM-Agents at NeurIPS 2026](https://slmw2026.github.io/#call-for-papers)

## The one-sentence contribution

Compact browser agents have a third failure axis between model capability and runtime parity:
**train--serve feature-materialization parity**. A structured graph can be fast and numerically
identical to its PyTorch reference, yet produce no tool action when fixed-length padding changes the
hidden-state position consumed by the action heads.

This is the insight to submit. It is narrower and more defensible than claiming a miniature
frontier model, a universal tokens-per-second requirement, or a positive WebGPU agent result.

The novelty is the measurement and interface contract, not a claim that padding itself is a new
neural operator. The paper should say so plainly: the contribution is a reproducible way to expose
and prevent a deployment-only capability failure.

## The falsifiable claim

For a structured agent whose decision is read from a particular hidden state, the tuple

    (tokenizer, assistant-marker position, padding direction, decision index,
     pointer span, grounding and validation code)

is part of the model interface. If training and serving materialize that tuple differently, the
deployment can fail capability while latency and numerical export parity remain intact.

The claim is causal only in the limited, testable sense supported by the frozen audit: keep the
checkpoint and heads fixed, change only where length-fixing tokens are inserted, and observe whether
the action policy changes. It is not a claim that all padding schemes fail, or that the model has
general browser competence.

## Evidence already bound in the repository

| Frozen condition | Route result | Tool result | What it establishes |
|---|---:|---:|---|
| Natural marker, 20-case action suite | 17/20 | 17/19 selected tools | The SFT checkpoint contains a usable offline structured signal. |
| Pre-assistant padding, 512 tokens | 1/20 (all 20 predicted text) | 0/19 required tools | The original fixed-length serving contract collapses the policy. |
| Trailing padding after the marker, 512 tokens | 17/20 | 17/19 selected tools | Holding compute length fixed while preserving the marker state restores the offline signal. |
| Full-stack native PyTorch vs fp32 ONNX vs fp16 ONNX | 20/20 runtime action agreements | 20/20 tool/argument agreements | Export, precision, serialized heads, and grounding are not the observed cause. |
| Paired original WebGPU fixed-512 pilots | 24.8 ms TTFA p50; 34.4 ms p95 | 0/19 required tools; 0/8 DOM successes | A fast forward pass is not a useful agent. |

The primary sources are the [context robustness audit](results/sft-structured-context-robustness-seed2027.summary.json),
[full-stack parity audit](results/sft-structured-export-parity-seed2027.summary.json),
[action pilot](results/m5-webgpu-sft-action-pilot-seed2027.summary.json), and
[DOM pilot](results/m5-webgpu-sft-dom-pilot-seed2027.summary.json). Repeated timing rows are
not independent capability trials.

## Why this is a workshop result

The workshop explicitly covers SLM architecture/training/inference, on-device deployment, and
reproducible evaluation. The result connects all three without requiring an unearned claim of
state-of-the-art quality:

1. **Architecture:** a one-forward route/select/copy policy exposes the decision index directly.
2. **Training:** heads are trained at the natural assistant-marker state.
3. **Deployment:** a fixed 512-token WebGPU graph changes the feature view if padding is inserted
   before that marker.
4. **Evaluation:** TTFA, exact action, schema validity, and final DOM state are reported
   separately, so latency cannot hide capability failure.

The useful abstraction is therefore not just a model checkpoint. It is a
**checkpoint + tokenizer + feature contract + grounding runtime + evaluator** bundle.

## Claim ladder for reviewers

### Safe to claim now

- TTFA and Success@B are better service metrics than decoded token rate for complete actions.
- The measured 10.5M hybrid proxy improved BPB in three small prospective seeds; this is a
  promotion signal, not a final architecture result.
- The exact pretrain-only checkpoint cleared 100 median decoded tokens/s on one Apple M5, with
  the declared tail-gate limitation.
- The structured graph had roughly 25 ms median harness TTFA on the tested device.
- The fixed-512 pilot failed capability, and the frozen offline audit localizes the failure to
  pre-marker feature materialization rather than export or precision drift.
- The result is a negative deployment-methodology finding, not a positive browser-agent score.

### Do not claim yet

- That structured dispatch beats autoregressive JSON: the trained cache-bearing AR controls are
  not collected.
- That the corrected trailing-padding runner has WebGPU capability: its browser rerun and an
  externally timestamped freeze are still required.
- That 100, 200, 400, or 600 tokens/s is a universal agent requirement.
- That the hybrid mixer is better at 34M or at full training compute.
- That one M5/Chrome/ONNX Runtime Web configuration represents ordinary laptops or all WebGPU
  implementations.
- That the 20-case/8-DOM reused suites are fresh capability evidence.
- That nominal active parameters make the opt-in Micro-MoE candidate smaller to download or faster
  in WebGPU; resident memory and sparse-export parity are still unmeasured.

## The decisive next experiment

The insight becomes a complete workshop result if the following preregistered sequence is run:

1. Externally timestamp the final runner, bundle, tokenizer, and case hashes.
2. Run three fresh WebGPU action sessions and three DOM sessions for the corrected trailing-padding
   contract; report run-level medians and tails, not pooled timing rows as task evidence.
3. Evaluate the original pre-marker contract separately, making the representation change the only
   treatment.
4. Add trained cache-bearing autoregressive and grounded-trie controls using the same checkpoint,
   tool catalog, tokenizer, and held-out task IDs.
5. Repeat on a fresh, task-disjoint set (the paper protocol requires at least 200 task IDs before
   making a capability comparison), then package anonymous artifacts and rerun from a clean clone.

The result is falsifiable. If corrected materialization remains at zero tool success, the proposed
cause is incomplete and the paper must downgrade it to a correlation. If it restores capability on
fresh cases, the paper has a replicated feature-contract effect. If the autoregressive controls
win, the conclusion becomes a representation trade-off rather than a structured-policy win.

## Reviewer-ready answer

> **What is new?** We show, with a fixed checkpoint and cross-runtime parity, that the hidden-state
> position used by a compact structured agent is a deployment-critical interface. Moving padding
> across the assistant marker changes a 25 ms policy from useful offline routing to zero required
> tool calls, while ONNX/fp16 parity remains exact. We therefore evaluate complete actions under a
> deadline and publish the feature contract alongside the model; token rate or graph parity alone
> is insufficient.

## Publication gate

The current artifact is **not yet honest to label submission-ready**. The four-page source compiles
and the negative result is strong, but the checklist remains red until the externally timestamped
corrected browser runs, trained AR controls, fresh task set, and anonymous reproduction package
exist. The appropriate submission decision today is:

- submit as a **negative-methodology/work-in-progress** paper only if the missing gates remain
  explicitly marked and the abstract says “not an agent”;
- upgrade to a positive systems paper only after corrected capability and matched AR comparisons
  clear their predeclared gates;
- never convert the offline parity audit into a WebGPU capability claim.

There is no honest 100% acceptance guarantee. This gate makes the contribution reviewable and
prevents a reviewer from having to discover the central limitation independently.
