"""Does the added training split share tool *names* with the evaluation split?

If it does, the gain from widening is partly memorisation of specific tools rather than a better
ability to read an unfamiliar catalog, and the paper has to say which.
"""
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
from eval_suite import SUITES, build_tasks

def gold_and_catalog(path, limit):
    tasks = build_tasks(Path(path), limit)
    gold = {t.gold_name for t in tasks}
    catalog = {tool["name"] for t in tasks for tool in t.tools}
    return gold, catalog, len(tasks)

train_gold, train_catalog, n_train = gold_and_catalog("data/public/xlam-train.jsonl", 24000)
print(f"xlam-train: rows={n_train} distinct gold={len(train_gold)} distinct catalog={len(train_catalog)}")
for suite in ("xlam", "toolace"):
    gold, catalog, rows = gold_and_catalog(SUITES[suite], 200)
    seen_gold = len(gold & train_catalog) / max(len(gold), 1)
    print(f"{suite:8s} rows={rows} distinct gold={len(gold):4d}  "
          f"gold names also in xlam-train catalog: {seen_gold*100:5.1f}%")
print("OVERLAP_DONE")
