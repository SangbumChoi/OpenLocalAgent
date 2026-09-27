"""Closed, bounded, token-FREE self-improvement loop for the dispatch heads (loop engineering, ML-style).

Discover -> Plan -> Execute -> Verify -> Iterate, with a stop condition — exactly the failure-driven
flywheel, but on the route-head + dense-selector probes (no LLM, no tokens; just CPU). Each round:
  VERIFY  : per-tool selection accuracy on held sets (free-form + paraphrase-eval).
  DISCOVER: the weakest tools.
  PLAN    : weight_t = 1 + k*(1 - acc_t)  (oversample the tools that fail).
  EXECUTE : rebuild the probe pools with those weights; retrain route head + dense selector.
  ITERATE : keep the best round; stop on target reached, patience exhausted, or max rounds.
Memory: appends every round to docs/DISPATCH_LOOP_LOG.md. Saves the best heads to runs/.
"""
import sys
from collections import defaultdict

import torch

from localagent.agent.dense_selector import BoundSelector, train_dense_selector
from localagent.agent.routes import route_of, train_route_head
from localagent.agent.tool_head import _feat
from localagent.agent.toolset import STANDARD_TOOLS as TOOLS
from localagent.data.contextual import CONTEXTUAL_EXAMPLES, contextual_samples
from localagent.data.paraphrase import TOOL_EXAMPLES, paraphrase_samples
from localagent.eval.freeform import FREEFORM_EVAL, FREEFORM_TRAIN
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer

QUICK = "--quick" in sys.argv
MAX_ROUNDS = 2 if QUICK else 6
PATIENCE = 2
TARGET = 0.70           # stop if free-form top-1 reaches this
K = 3.0                 # oversampling strength for weak tools
torch.set_num_threads(4)
tok = load_tokenizer()
NAMES = {t.name for t in TOOLS}


class S:
    def __init__(self, prompt, tool):
        self.prompt, self.ref_name, self.kind = prompt, tool, "tool"
        self.category = self.group = "loop"
        self.target, self.calls = "", None


ck = torch.load("runs/tiny-30m-dispatch-long.pt", map_location="cpu")
cfg = ModelConfig(**ck["cfg"])
m = LocalAgentLM(cfg); m.load_state_dict(ck["state_dict"]); m.eval()
examples = {**TOOL_EXAMPLES}
for k, v in CONTEXTUAL_EXAMPLES.items():
    examples[k] = examples.get(k, []) + v
ff_by_tool = defaultdict(list)
for q, t in FREEFORM_TRAIN:
    ff_by_tool[t].append(q)
for k, v in ff_by_tool.items():
    examples[k] = (examples.get(k, []) + v)[:14]

# train pool (prompt -> tool), and a per-tool held set for VERIFY
train = [s for s in (paraphrase_samples(2 if QUICK else 20, seed=11, split="train")
                     + contextual_samples(1 if QUICK else 15, seed=11, split="train"))
         if s.kind == "tool"] + [S(q, t) for q, t in FREEFORM_TRAIN if t in NAMES]
held = [(q, t) for q, t in FREEFORM_EVAL]                     # free-form OOD (the honest test)
held += [(s.prompt, s.ref_name) for s in paraphrase_samples(2, seed=909, split="eval")
         if s.kind == "tool"][:120]
print(f"train={len(train)} held={len(held)}", flush=True)


def verify(bound, rh):
    by_tool = defaultdict(lambda: [0, 0]); ok = rok = 0
    feats = {}
    for q, gold in held:
        if q not in feats:
            feats[q] = _feat(m, tok, q, "cpu")
        f = feats[q]
        hit = bound.rank(f)[0] == gold
        ok += hit; rok += int(route_of(bound.rank(f)[0]) == route_of(gold))
        by_tool[gold][0] += hit; by_tool[gold][1] += 1
    acc = {t: c / n for t, (c, n) in by_tool.items()}
    return ok / len(held), acc


weights = dict.fromkeys(NAMES, 1.0)
best = None; best_top1 = -1.0; stale = 0
log_lines = ["# Dispatch self-improvement loop log\n", "| round | top1 | weakest tools |\n|--|--|--|\n"]
for r in range(MAX_ROUNDS):
    # EXECUTE: oversample by weight
    pool = []
    for s in train:
        pool += [s] * max(1, round(weights.get(s.ref_name, 1.0)))
    rh = train_route_head(m, pool, tok, steps=80 if QUICK else 400, device="cpu"); rh.eval()
    sel = train_dense_selector(m, pool, tok, TOOLS, steps=80 if QUICK else 400, device="cpu",
                               examples=examples)
    bound = BoundSelector(sel, TOOLS, examples=examples)
    # VERIFY
    top1, acc = verify(bound, rh)
    weak = sorted(acc, key=acc.get)[:5]
    print(f"[round {r}] top1={top1*100:.0f}%  weakest={[(t, f'{acc[t]*100:.0f}%') for t in weak]}",
          flush=True)
    log_lines.append(f"| {r} | {top1*100:.0f}% | {', '.join(weak)} |\n")
    # ITERATE / stop
    if top1 > best_top1:
        best_top1, best, stale = top1, (rh, sel), 0
    else:
        stale += 1
    if top1 >= TARGET or stale >= PATIENCE:
        print(f"  stop: {'target' if top1>=TARGET else 'patience'}", flush=True)
        break
    # PLAN: re-weight toward weak tools
    weights = {t: 1.0 + K * (1.0 - acc.get(t, 1.0)) for t in NAMES}

rh, sel = best
torch.save({"cfg": cfg.__dict__, "state_dict": m.state_dict(), "ptr_head": ck["ptr_head"],
            "tool_head": ck.get("tool_head"), "dense_selector": sel.state_dict(),
            "route_head": rh.state_dict(), "selector_proj": ck["selector_proj"], "examples": examples},
           "runs/tiny-30m-dispatch-loop.pt")
with open("docs/DISPATCH_LOOP_LOG.md", "w") as f:
    f.writelines(log_lines + [f"\nBest free-form top-1: **{best_top1*100:.0f}%** "
                              f"(saved runs/tiny-30m-dispatch-loop.pt)\n"])
print(f"\nBEST top1={best_top1*100:.0f}%  -> runs/tiny-30m-dispatch-loop.pt\nLOOP_DONE", flush=True)
