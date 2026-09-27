"""Is the information the gold argument demands actually present in the observation?

If it is not, exact match on that suite is unachievable and should not be reported as a capability.
"""
import sys
sys.path.insert(0, "scripts")
from eval_suite import SUITES, build_tasks

for suite in ("mind2web", "agentnet"):
    tasks = build_tasks(SUITES[suite], 200)
    visible = 0
    for task in tasks:
        gold = task.gold_arguments
        needle = str(gold.get("target_id") or gold.get("target") or "")
        if suite == "agentnet" and "x=" in needle:
            needle = needle.split("x=")[1].split(";")[0][:6]   # the x coordinate as written
        if needle and needle in task.observation:
            visible += 1
    print(f"{suite:12s} gold value literally present in the observation: "
          f"{visible}/{len(tasks)} ({visible/len(tasks)*100:.1f}%)")
    print(f"    sample observation tail: {tasks[0].observation[-260:]!r}")
print("OBS_DONE")
