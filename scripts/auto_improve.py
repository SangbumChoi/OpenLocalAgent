"""Automation glue for the dispatch self-improvement loop — the decision step of a scheduled loop.

Ensures the base checkpoint (downloads from the Hub if absent), runs the closed loop
(`dispatch_loop.py`), reads `docs/DISPATCH_LOOP_LOG.md`, and decides whether the best round BEAT the
baseline (round 0) by a margin. Emits a machine-readable decision to GITHUB_OUTPUT + docs/dispatch_improvement.json
so a CI workflow can open a PR ONLY when the work passed — the human-approval hand-off.

  python scripts/auto_improve.py --run --margin 2           # run the loop, then decide
  python scripts/auto_improve.py                            # just decide from the existing log
"""
import argparse
import json
import os
import re
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--run", action="store_true", help="run dispatch_loop.py before deciding")
ap.add_argument("--margin", type=float, default=2.0, help="min %-point gain over baseline to count")
ap.add_argument("--base-repo", default="danelcsb/localagent-tiny-30m-byte")
ap.add_argument("--base-ckpt", default="runs/tiny-30m-dispatch-long.pt")
args = ap.parse_args()

# 1. ensure the base checkpoint (CI starts clean; pull it from the Hub)
if not os.path.exists(args.base_ckpt):
    import shutil

    from huggingface_hub import hf_hub_download
    print(f"base checkpoint missing -> downloading model.pt from {args.base_repo}", flush=True)
    os.makedirs(os.path.dirname(args.base_ckpt), exist_ok=True)
    shutil.copy(hf_hub_download(args.base_repo, "model.pt"), args.base_ckpt)

# 2. run the loop
if args.run:
    print("running dispatch_loop.py ...", flush=True)
    subprocess.run([sys.executable, "scripts/dispatch_loop.py"], check=True)

# 3. parse the loop log: baseline = round 0, best = max over rounds
log = open("docs/DISPATCH_LOOP_LOG.md").read()
rows = re.findall(r"^\|\s*(\d+)\s*\|\s*(\d+)%", log, re.M)
if not rows:
    print("no rounds in log", flush=True)
    sys.exit(1)
scores = {int(r): int(s) for r, s in rows}
baseline = scores[0]
best = max(scores.values())
improved = (best - baseline) >= args.margin

decision = {"baseline_top1": baseline, "best_top1": best, "gain": best - baseline,
            "margin": args.margin, "improved": improved}
json.dump(decision, open("docs/dispatch_improvement.json", "w"), indent=2)
print(f"baseline={baseline}%  best={best}%  gain={best-baseline:+d}pp  improved={improved}", flush=True)

# 4. emit a CI output so the workflow can gate the PR on a real improvement
gh_out = os.environ.get("GITHUB_OUTPUT")
if gh_out:
    with open(gh_out, "a") as f:
        f.write(f"improved={'true' if improved else 'false'}\n")
        f.write(f"best={best}\nbaseline={baseline}\n")
print("AUTO_IMPROVE_DONE", flush=True)
