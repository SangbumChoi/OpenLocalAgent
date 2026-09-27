"""How often does a suite's rendered prompt exceed the student's context?

A catalog-conditioned model that cannot fit the catalog emits nothing, which the scorer counts as
a parse failure. That is a context-length limit, not a decision failure, and the two should not be
reported as the same thing.
"""
import sys
sys.path.insert(0, "scripts")
from eval_suite import SUITES, build_tasks

from localagent.data.prompt_contract import render_agent_decode_prompt
from localagent.data.schema import Message, Role, ToolSpec
from localagent.model.tokenizer import load_tokenizer

tok = load_tokenizer("bpe", "data/tokenizer-h100-16k.json")
MAX, NEW = 2048, 64
for suite, path in SUITES.items():
    if not path.exists():
        continue
    tasks = build_tasks(path, 200)
    lengths, failed = [], 0
    for task in tasks:
        tools = [ToolSpec(name=t["name"], description=t["description"] or t["name"],
                          parameters=t["parameters"] or {"type": "object", "properties": {}})
                 for t in task.tools]
        try:
            prompt = render_agent_decode_prompt(
                [Message(role=Role.user, content=task.observation[:4000])], tools)
        except (ValueError, KeyError):
            failed += 1
            continue
        lengths.append(len(tok.encode(prompt)))
    over = sum(1 for n in lengths if n > MAX - NEW - 4)
    mean = sum(lengths) / max(len(lengths), 1)
    print(f"{suite:16s} mean_tokens={mean:7.0f} max={max(lengths, default=0):6d} "
          f"over_context={over/max(len(tasks),1)*100:5.1f}%  render_failed={failed}", flush=True)
print("FIT_DONE")
