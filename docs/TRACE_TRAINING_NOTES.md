# Recipes from `burtenshaw/training-agents` (SFT-on-traces) — what to adopt

Notes from studying [github.com/burtenshaw/training-agents](https://github.com/burtenshaw/training-agents)
(the "Codex builder-agent SFTs a 2B Gemma coding agent on `badlogicgames/pi-mono` traces" project).
Same thesis as LocalAgent — *specialize a small model by SFT on real agent traces* — but at 2B+LoRA
with TRL/HF-Jobs. This is a **techniques extraction**, mapped to our modules. No training implied.

Their stack (TRL/PEFT/HF-Jobs/Trackio/Inspect-AI) is exactly the "heavy framework" set our
`AGENTS.md` rules out, so we adopt the *ideas*, not the dependencies.

## Priority picks (quick wins, pure-PyTorch-friendly)

1. **Honest eval separation — the big one.** They never report the training metric as the benchmark
   score: training tracks *held-out prompt/completion loss + token accuracy*; code quality is a
   **separate** Inspect-AI HumanEval/MBPP run on truly held-out data. Their `grpo-agent-rewards` doc
   lists "training on held-out eval tasks" as a reward-hacking failure, and an `integrity-reviewer`
   agent exists solely to check leakage/eval-validity/claims. → This is the principled counter to our
   `scripts/train_benchmarks.py` contamination path: keep AIME/GPQA/etc. as *held-out eval*, and use
   HumanEval/MBPP (no Docker needed) as the code-eval harness. *Action:* add an eval-only benchmark
   harness; demote `train_benchmarks.py` to a clearly-fenced experiment.

2. **Render-and-assert completion split.** `render_prompt_completion()` renders the prompt
   (`add_generation_prompt=True`) and the full `context+assistant`, then asserts
   `full.startswith(prompt)` and takes `completion = full[len(prompt):]`. A template/EOS drift becomes
   a hard error instead of silently mislabeled loss spans. → `data/render.py::render_sft` masks at the
   token level; add the same prefix-assertion + a tokenization smoke check so masking can't silently
   drift. *Quick win.*

3. **Log the conversion funnel.** Their README records the exact funnel: `raw files 627 → message
   events 33,879 → assistant examples 15,251 → kept@4k 6,727 → train/eval 6,471/256 → reasoning
   ignored 12,451 → images omitted 48` (a `ConversionStats` dataclass + `filter_examples_by_length`).
   → `train_benchmarks.py` already logs per-source counts; extend it to a `ConversionStats`-style
   funnel (loaded → fit-context → dropped, with reasons). *Quick win.*

4. **Token accuracy during training, not just loss.** TRL logs mean next-token argmax accuracy on the
   completion span (their best run: eval loss 0.55, token acc 0.86). It's a cheaper, more legible
   signal than loss alone. → add an eval pass in `train/sft.py` (or `eval/`) that reports completion
   token-accuracy on a held-out split every `eval_steps`. *Quick win.*

5. **GRPO: log reward *components*, not one scalar.** `grpo-agent-rewards` insists on per-component
   logging (exact-answer, unit-test, JSON-valid, tool-arg-valid, terminal-success) for debugging. →
   `train/rl.py` logs only `mean_reward`; have `eval/harness._correct` expose its sub-checks
   (tool-name match / args match / schema valid) and log them separately. *Quick win.*

## Bigger bets (proposals — touch guarded contracts)

6. **One trace record → SFT / DPO / GRPO.** Their `trace-schema` stores task id, split, tools, action,
   observation, reward components, verifier output, judge critique, accepted/rejected. From it: SFT =
   accepted assistant actions; **DPO = accepted(chosen) vs failed(rejected)**; GRPO = prompt-only +
   verifier. LocalAgent has SFT+GRPO but **no preference/DPO path**. → Proposal to extend the
   `Conversation` schema (a guarded contract — discuss first) with accepted/rejected + reward metadata,
   unlocking a DPO stage in `train/`.

7. **Distillation loop with a promotion gate.** `distillation-loop`: sample → rollouts → deterministic
   verify → keep successes + representative failures → convert to SFT/preference → retrain → eval
   held-out, and **only promote a distilled dataset when held-out behavior improves**. → our flywheel
   (`scripts/flywheel.py`, `analyze_loop.py`) enriches every round unconditionally; add an explicit
   promotion rule (keep new data only if held-out accuracy goes up).

8. **Tool-calling data hygiene + recovery behavior.** `tool-calling-sft`: verify tool names match
   schemas, args are valid JSON, observations don't leak held-out labels, and include **both success
   and recovery-after-failed-command** examples; eval tool-call *validity* separately from final-answer
   quality. → `data/agent_synth.py` has little "failed tool → recover" coverage; add a recovery
   category, and split our eval into validity vs correctness.

9. **`integrity-reviewer` sub-agent.** They dedicate a Codex agent to leakage / eval-validity / result
   claims. Given our contamination episode, a parallel `.claude/agents/integrity-reviewer` (audits
   train/eval disjointness, flags contaminated checkpoints, checks reported numbers) would be a strong
   fit next to `data-engineer`/`model-trainer`/`evaluator`/`exporter`.

## Noted, low priority
- **Trackio** hosted dashboard vs our PNG+JSONL logging — pip-installable optional backend; nice-to-have.
- **LoRA target precision** lesson: broad LoRA suffixes hit unsupported wrapper modules; they fixed it
  by targeting `model.language_model.layers.*` only. N/A to our full-finetune from-scratch models, but
  a good reminder if we ever add adapters.
- **PEP 723 / `uv run` single-file scripts** for portable Jobs — our `scripts/*.py` are already
  single-file and argparse-driven; same spirit.

## One-line takeaway
The single most valuable thing to copy is their **discipline about eval validity** (held-out loss +
token-acc for training; a *separate* unit-test benchmark for quality; a reviewer that polices leakage)
— precisely the guardrail our `train_benchmarks.py` deliberately bent.
