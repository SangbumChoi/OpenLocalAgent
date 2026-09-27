"""Materialize the synthetic generators to static JSONL snapshots (for inspection / external upload).

The datasets are generators (deterministic from a seed); this dumps a fixed snapshot to data/dumps/.
Pure data generation, no model — safe to run anytime.
"""
import json
import pathlib

from localagent.data.contextual import contextual_samples
from localagent.data.paraphrase import paraphrase_samples
from localagent.data.scenarios import scenario_episodes, scenario_samples
from localagent.eval.freeform import FREEFORM_EVAL, FREEFORM_TRAIN

OUT = pathlib.Path("data/dumps")
OUT.mkdir(parents=True, exist_ok=True)


def dump_samples(name, samples):
    p = OUT / f"{name}.jsonl"
    with p.open("w") as f:
        for s in samples:
            row = {"prompt": s.prompt, "kind": s.kind, "category": s.category,
                   "tool": s.ref_name, "target": s.target}
            if getattr(s, "calls", None):
                row["calls"] = s.calls
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{p}  ({len(samples)} rows)")


def dump_episodes(name, episodes):
    p = OUT / f"{name}.jsonl"
    with p.open("w") as f:
        for c in episodes:
            turns = [{"role": m.role.value, "content": m.content,
                      "tool_calls": [{"name": tc.name, "arguments": tc.arguments}
                                     for tc in (m.tool_calls or [])],
                      "tool_response": m.tool_response} for m in c.messages]
            f.write(json.dumps({"category": (c.meta or {}).get("category"), "turns": turns},
                               ensure_ascii=False) + "\n")
    print(f"{p}  ({len(episodes)} episodes)")


for split in ("train", "eval"):
    dump_samples(f"paraphrase_{split}", paraphrase_samples(20, seed=0, split=split))
    dump_samples(f"contextual_{split}", contextual_samples(10, seed=0, split=split))
    dump_samples(f"scenarios_single_{split}", scenario_samples(20, seed=0, split=split))
    dump_episodes(f"scenarios_episodes_{split}", scenario_episodes(20, seed=0, split=split))

def dump_freeform(name, pairs):
    p = OUT / f"{name}.jsonl"
    with p.open("w") as f:
        for q, gold in pairs:
            f.write(json.dumps({"prompt": q, "tool": gold}) + "\n")
    print(f"{p}  ({len(pairs)} rows)")


dump_freeform("freeform_train", FREEFORM_TRAIN)
dump_freeform("freeform_eval", FREEFORM_EVAL)
print("DUMP_DONE")
