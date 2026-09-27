"""The floor every score should be read against: always answer the suite's most frequent tool."""
import json
import sys
from collections import Counter

sys.path.insert(0, "scripts")
from eval_suite import SUITES, _with_suite_catalog, build_tasks

report = {}
for suite, path in SUITES.items():
    tasks = _with_suite_catalog(build_tasks(path, 200))
    counts = Counter(task.gold_name for task in tasks)
    name, hits = counts.most_common(1)[0]
    report[suite] = {"rows": len(tasks), "distinct_tools": len(counts),
                     "majority_tool": name, "majority_rate": hits / len(tasks)}
    print(f"{suite:16s} rows={len(tasks)} tools={len(counts):4d} "
          f"majority={name} {hits / len(tasks) * 100:.1f}%")
json.dump(report, open("runs/evalsuite/majority-baseline.json", "w"), indent=1)
print("MAJORITY_DONE")
