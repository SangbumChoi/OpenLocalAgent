"""What do the gold arguments look like on the suites whose exact match is ~0 for everyone?"""
import json
import sys
from collections import Counter
sys.path.insert(0, "scripts")
from eval_suite import SUITES, build_tasks

for suite in ("mind2web", "agentnet", "androidcontrol"):
    tasks = build_tasks(SUITES[suite], 200)
    keys = Counter(tuple(sorted(t.gold_arguments)) for t in tasks)
    print(f"=== {suite}: argument key-sets {keys.most_common(4)}")
    for task in tasks[:4]:
        print(f"    gold {task.gold_name} {json.dumps(task.gold_arguments)[:180]}")
print("ARGS_DONE")
