#!/usr/bin/env python
"""Score AndroidControl on only the rows whose (instruction, action, arguments) never appears in
the training corpus.

Dropping the screenshot collapses distinct screens into identical instruction strings, so a
quarter of the test rows have an exact twin in training. That inflates the suite for every model
trained on the union corpus — ours and the fine-tuned baselines alike — and the clean subset is
the number that should be quoted.
"""

import argparse
import json
from pathlib import Path

from openlocalagent.eval.suite import (
    SUITES,
    CatalogAdapter,
    DispatchAdapter,
    HuggingFaceAdapter,
    LocalAgentAdapter,
    LoraAdapter,
    build_tasks,
    score,
)

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cuda")
ap.add_argument("--rows", type=int, default=200)
args = ap.parse_args()

train = build_tasks(Path("data/merged-v2/train.jsonl"), 20000)
seen = {(task.observation.strip()[-400:], task.gold_name, json.dumps(task.gold_arguments,
                                                                    sort_keys=True))
        for task in train}
tasks = [task for task in build_tasks(SUITES["androidcontrol"], args.rows)
         if (task.observation.strip()[-400:], task.gold_name,
             json.dumps(task.gold_arguments, sort_keys=True)) not in seen]

kind, _, location = args.model.partition(":")
adapters = {"openlocalagent": LocalAgentAdapter, "hf": HuggingFaceAdapter, "lora": LoraAdapter,
            "dispatch": DispatchAdapter, "catalog": CatalogAdapter}
adapter = adapters[kind](location, args.device)
report = {"model": adapter.name, "kind": kind, "rows_kept": len(tasks),
          "rows_dropped": args.rows - len(tasks),
          "suites": {"eval_androidcontrol_clean": score(adapter, tasks, 64)}}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
row = report["suites"]["eval_androidcontrol_clean"]
print(f"{adapter.name}: kept={len(tasks)} type={row['type_match']*100:.1f} "
      f"exact={row['step_success_rate']*100:.1f}", flush=True)
print("CLEAN_DONE " + args.out, flush=True)
