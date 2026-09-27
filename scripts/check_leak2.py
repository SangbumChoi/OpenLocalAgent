"""Are the AndroidControl overlaps true leakage, or common instructions repeated across episodes?"""
import sys
from collections import Counter
sys.path.insert(0, "scripts")
from eval_suite import SUITES, build_tasks
from pathlib import Path

train = build_tasks(Path("data/merged-v2/train.jsonl"), 20000)
index = {}
for task in train:
    index.setdefault((task.observation.strip()[-400:], task.gold_name), []).append(task)

tasks = build_tasks(SUITES["androidcontrol"], 200)
shown = 0
identical_args = 0
for task in tasks:
    matches = index.get((task.observation.strip()[-400:], task.gold_name))
    if not matches:
        continue
    same = any(m.gold_arguments == task.gold_arguments for m in matches)
    identical_args += same
    if shown < 6:
        print(f"--- eval obs: {task.observation.strip()[-90:]!r}")
        print(f"    gold: {task.gold_name} {task.gold_arguments}")
        print(f"    train match args: {[m.gold_arguments for m in matches][:2]} same={same}")
        shown += 1
print(f"\noverlapping rows with IDENTICAL arguments too: {identical_args}")
counts = Counter(t.observation.strip()[-400:] for t in train)
print("most repeated training instruction:", counts.most_common(3)[:3])
print("LEAK2_DONE")
