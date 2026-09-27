"""How often is the gold tool inside the retriever's top-k?

This bounds a retrieve-then-decide agent: the language model can only choose correctly from a
shortlist that contains the answer, so top-k recall is the ceiling of the whole design.
"""
import sys
sys.path.insert(0, "scripts")
from eval_suite import SUITES, build_tasks

from localagent.agent.caller import ToolCaller
from localagent.data.schema import ToolSpec

KS = (1, 4, 8, 16, 32)
for suite in ("toolace", "xlam", "androidcontrol", "agentnet"):
    tasks = build_tasks(SUITES[suite], 200)
    hits = dict.fromkeys(KS, 0)
    sizes = []
    for task in tasks:
        specs = [ToolSpec(name=t["name"], description=t["description"] or t["name"],
                          parameters=t["parameters"] or {"type": "object", "properties": {}})
                 for t in task.tools]
        sizes.append(len(specs))
        caller = ToolCaller(specs, retrieve_k=max(KS))
        ranked = [spec.name for spec, _ in caller.candidates(task.observation)]
        for k in KS:
            hits[k] += task.gold_name in ranked[:k]
    line = "  ".join(f"top{k}={hits[k] / len(tasks) * 100:5.1f}" for k in KS)
    print(f"{suite:16s} catalog={sum(sizes)//len(sizes):4d}  {line}", flush=True)
print("TOPK_DONE")
