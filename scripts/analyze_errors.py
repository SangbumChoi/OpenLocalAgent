#!/usr/bin/env python
"""Decompose a model's errors on the open-catalog suites into their mechanism.

Raising a score honestly starts with knowing which failure is being paid for: a parse failure is
a format problem, an emitted tool outside the row's catalog is a reading problem, and a wrong
choice within the catalog is a discrimination problem. Each points at a different lever.

  python scripts/analyze_errors.py --model catalog:results/runs/sft-distill-96m/latest.pt \
      --suites toolace,xlam,bfcl,toolbench --out results/runs/analysis/errors-distill-96m.json
"""

from __future__ import annotations

import argparse
import json
from enum import StrEnum
from pathlib import Path

from openlocalagent.eval.suite import (
    SUITES,
    CatalogAdapter,
    HuggingFaceAdapter,
    LoraAdapter,
    build_tasks,
    parse_call,
)


class Bucket(StrEnum):
    correct_name = "correct_name"
    wrong_in_catalog = "wrong_in_catalog"
    out_of_catalog = "out_of_catalog"
    parse_fail = "parse_fail"



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--suites", default="toolace,xlam,bfcl,toolbench")
    ap.add_argument("--rows", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # Reuse the harness's own adapter construction so the decomposition scores exactly what the
    # dashboard scores.
    spec_kind, location = args.model.split(":", 1)
    if spec_kind == "catalog":
        adapter = CatalogAdapter(location, args.device)
    elif spec_kind == "lora":
        adapter = LoraAdapter(location, args.device)
    else:
        adapter = HuggingFaceAdapter(location, args.device)

    report = {"model": args.model, "suites": {}}
    for suite in args.suites.split(","):
        path = SUITES[suite]
        tasks = build_tasks(path, args.rows)
        counts = {bucket: 0 for bucket in Bucket} | {"rows": len(tasks)}
        samples: list[dict] = []
        for task in tasks:
            prediction = parse_call(adapter.predict(task, args.max_new_tokens))
            catalog = {tool["name"] for tool in task.tools} if task.tools else set()
            if prediction is None:
                counts[Bucket.parse_fail] += 1
                bucket = Bucket.parse_fail
                predicted_name = None
            else:
                predicted_name = prediction[0]
                if predicted_name == task.gold_name:
                    counts[Bucket.correct_name] += 1
                    bucket = Bucket.correct_name
                elif catalog and predicted_name not in catalog:
                    counts[Bucket.out_of_catalog] += 1
                    bucket = Bucket.out_of_catalog
                else:
                    counts[Bucket.wrong_in_catalog] += 1
                    bucket = Bucket.wrong_in_catalog
            if bucket != Bucket.correct_name and len(samples) < 12:
                samples.append({"bucket": bucket, "gold": task.gold_name,
                                "predicted": predicted_name,
                                "catalog_size": len(catalog)})
        counts["shares"] = {bucket: round(100 * counts[bucket] / max(counts["rows"], 1), 1)
                            for bucket in Bucket}
        report["suites"][suite] = {**counts, "samples": samples}
        print(suite, json.dumps(counts["shares"]), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print("ERRORS_DONE " + str(out), flush=True)


if __name__ == "__main__":
    main()
