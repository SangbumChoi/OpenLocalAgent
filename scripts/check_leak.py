"""Do the fine-tuning rows overlap the evaluation suites? The comparison is only fair if not."""
import sys
sys.path.insert(0, "scripts")
from eval_suite import SUITES, build_tasks
from pathlib import Path

train = build_tasks(Path("data/merged-v2/train.jsonl"), 20000)
train_keys = {(task.observation.strip()[-400:], task.gold_name) for task in train}
print(f"train rows: {len(train)}")
for suite, path in SUITES.items():
    if not path.exists():
        continue
    tasks = build_tasks(path, 200)
    overlap = sum((t.observation.strip()[-400:], t.gold_name) in train_keys for t in tasks)
    print(f"{suite:16s} rows={len(tasks):4d} overlapping_with_train={overlap}")
print("LEAK_CHECK_DONE")
