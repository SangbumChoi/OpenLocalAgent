"""Two floors, not one.

The majority-class floor answers 'always name the single most frequent tool'. But each task
presents its own small catalog, so a model choosing uniformly at random *from the row's own
candidates* is the floor that actually applies to a catalog-conditioned agent. Reporting only the
first one flatters every model on suites whose per-task catalogs are tiny.
"""
import json
import sys
from collections import Counter
from statistics import mean
sys.path.insert(0, "scripts")
from eval_suite import SUITES, build_tasks

report = {}
for suite, path in SUITES.items():
    if not path.exists():
        continue
    tasks = build_tasks(path, 200)
    sizes = [max(len(t.tools), 1) for t in tasks]
    counts = Counter(t.gold_name for t in tasks)
    name, hits = counts.most_common(1)[0]
    report[suite] = {
        "rows": len(tasks),
        "distinct_gold_tools": len(counts),
        "catalog_per_task_mean": mean(sizes),
        "catalog_per_task_min": min(sizes),
        "catalog_per_task_max": max(sizes),
        "majority_tool": name,
        "majority_rate": hits / len(tasks),
        "random_within_catalog": mean(1.0 / s for s in sizes),
    }
    r = report[suite]
    print(f"{suite:16s} rows={r['rows']:4d} catalog/task={r['catalog_per_task_mean']:5.1f} "
          f"(min {r['catalog_per_task_min']}, max {r['catalog_per_task_max']})  "
          f"majority={r['majority_rate']*100:5.1f}%  random-in-catalog="
          f"{r['random_within_catalog']*100:5.1f}%", flush=True)
json.dump(report, open("runs/evalsuite/chance-baseline.json", "w"), indent=2)
print("CHANCE_DONE")
