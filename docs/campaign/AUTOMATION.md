# Automation: scheduled self-improvement loop → PR

The dispatch self-improvement loop wired as a hands-off **scheduled automation** (loop engineering's
"Automations" + "Connectors" blocks). The human leaves the inner loop and only reviews a PR.

## Flow
`.github/workflows/dispatch-auto-improve.yml` (daily cron + manual `workflow_dispatch`):
1. checkout → setup Python → `pip install -e .`
2. `python scripts/auto_improve.py --run` →
   - downloads the base checkpoint (`model.pt`) from the Hub if absent,
   - runs the closed loop `scripts/dispatch_loop.py` (VERIFY→DISCOVER→PLAN→EXECUTE→ITERATE, CPU-only,
     **zero LLM tokens**),
   - compares the best round vs the baseline (round 0) and writes `docs/dispatch_improvement.json`
     + a GITHUB_OUTPUT `improved=true/false`.
3. **Only if `improved`**, `create-pull-request` opens a PR with the log + metrics — the
   human-approval hand-off. No gain ⇒ no PR (cost discipline; the loop halts itself on patience).

## Why this shape
- **Closed, bounded**: the loop has a target + patience stop, so it never churns.
- **Maker ≠ checker**: the loop trains (maker); `auto_improve.py` verifies + gates (checker).
- **Token-free**: a 28M byte model makes the whole loop a cheap CPU job — no token bill, the
  constraint that kills most agent loops.
- **Memory**: `docs/DISPATCH_LOOP_LOG.md` accumulates every round across runs.

Run locally: `python scripts/auto_improve.py --run --margin 2`.
