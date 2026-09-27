"""Retrain the dispatch heads on the CORRECTED + EXPANDED data (incl. the new free-form TRAIN set).

The dense selector + route head are (prompt -> tool) frozen-feature probes, so the highest-leverage
use of the new data is to train them ON the free-form distribution the eval measures, and to enrich
the selector's tool-tower embeddings with free-form example phrasings. Backbone stays at its proven
sweet spot (scenarios-best) — we showed more backbone SFT overfits. Compares free-form OOD before
(current heads) vs after (heads retrained with FREEFORM_TRAIN) and saves a new deployable checkpoint.
"""
import sys
from collections import defaultdict

import torch

from localagent.agent.constrained import hybrid_decode
from localagent.agent.dense_selector import (
    BoundSelector, DenseToolSelector, tool_embeddings, train_dense_selector,
)
from localagent.agent.parser import extract_tool_calls
from localagent.agent.pointer_head import PointerHead
from localagent.agent.routes import ROUTES, RouteHead, route_of, train_route_head
from localagent.agent.tool_head import _feat
from localagent.agent.toolset import STANDARD_TOOLS as TOOLS
from localagent.data.contextual import CONTEXTUAL_EXAMPLES, contextual_samples
from localagent.data.paraphrase import TOOL_EXAMPLES, paraphrase_samples
from localagent.data.scenarios import scenario_samples
from localagent.eval.freeform import FREEFORM_EVAL, FREEFORM_TRAIN
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer

QUICK = "--quick" in sys.argv
torch.set_num_threads(4)
tok = load_tokenizer()
NAMES = {t.name for t in TOOLS}


class FF:  # lightweight (prompt -> tool) sample for the probes (no args needed)
    def __init__(self, prompt, tool):
        self.prompt, self.ref_name, self.kind = prompt, tool, "tool"
        self.category, self.group, self.target, self.calls = "freeform", "freeform", "", None


import os
BASE=os.environ.get("BASE_CKPT","runs/tiny-30m-dispatch-long.pt")
ck = torch.load(BASE, map_location="cpu")
cfg = ModelConfig(**ck["cfg"])
m = LocalAgentLM(cfg); m.load_state_dict(ck["state_dict"]); m.eval()
ptr = PointerHead(cfg.d_model); ptr.load_state_dict(ck["ptr_head"]); ptr.eval()

# examples for the tool tower: tool descriptions + contextual + FREE-FORM phrasings grouped by tool
examples = {**TOOL_EXAMPLES}
for k, v in CONTEXTUAL_EXAMPLES.items():
    examples[k] = (examples.get(k, []) + v)
ff_by_tool = defaultdict(list)
for q, t in FREEFORM_TRAIN:
    ff_by_tool[t].append(q)
for k, v in ff_by_tool.items():
    examples[k] = (examples.get(k, []) + v)[:14]

# probe pools (corrected data) + the free-form train set
para = paraphrase_samples(2 if QUICK else 30, seed=11, split="train")
ctx = contextual_samples(1 if QUICK else 20, seed=11, split="train")
scen = scenario_samples(2 if QUICK else 30, seed=11, split="train")
ff_train = [FF(q, t) for q, t in FREEFORM_TRAIN if t in NAMES]
tool_pool = [s for s in (para + ctx) if s.kind == "tool"]
import random
random.Random(0).shuffle(tool_pool)
tool_pool = tool_pool[: 200 if QUICK else 700]
# selector: corrected tool data + parallel + FREE-FORM (oversampled x4 so OOD register is well-seen)
sel_pool = tool_pool + [s for s in scen if s.kind == "tool"] + ff_train * 4
# route head: + clarify/abstain (text) oversampled + FREE-FORM
route_pool = tool_pool + [s for s in scen if s.kind == "text"] * 3 + ff_train * 4
print(f"sel_pool={len(sel_pool)} route_pool={len(route_pool)} ff_train={len(ff_train)} "
      f"examples_tools={len(examples)}", flush=True)


def score(rh, bound, label):
    rt = t1 = cn = 0
    for q, gold in FREEFORM_EVAL:
        f = _feat(m, tok, q, "cpu")
        rt += ROUTES[int(rh(f).argmax(-1))] == route_of(gold)
        t1 += bound.rank(f)[0] == gold
        c = extract_tool_calls(hybrid_decode(m, tok, q, TOOLS, selector=bound, top_m=1,
                                             route_head=rh, ptr_head=ptr))
        cn += bool(c) and c[0].name == gold
    n = len(FREEFORM_EVAL)
    print(f"[{label}] FREE-FORM route={rt/n*100:.0f}% top1={t1/n*100:.0f}% call={cn/n*100:.0f}% (n={n})",
          flush=True)
    return cn / n


# BEFORE — the currently deployed heads
rh0 = RouteHead(cfg.d_model); rh0.load_state_dict(ck["route_head"]); rh0.eval()
sel0 = DenseToolSelector(cfg.d_model, emb_dim=tool_embeddings(TOOLS[:1]).shape[1], proj=ck["selector_proj"])
sel0.load_state_dict(ck["dense_selector"])
bound0 = BoundSelector(sel0, TOOLS, examples=ck.get("examples", {}))
print("\n--- BEFORE (deployed heads) ---", flush=True)
score(rh0, bound0, "BEFORE")

# AFTER — retrain heads with FREEFORM_TRAIN + corrected data + enriched examples
steps = 150 if QUICK else 500
rh = train_route_head(m, route_pool, tok, steps=steps, device="cpu"); rh.eval()
sel = train_dense_selector(m, sel_pool, tok, TOOLS, steps=steps, device="cpu", examples=examples)
bound = BoundSelector(sel, TOOLS, examples=examples)
print("\n--- AFTER (heads retrained on corrected + free-form data) ---", flush=True)
score(rh, bound, "AFTER")

torch.save({"cfg": cfg.__dict__, "state_dict": m.state_dict(), "ptr_head": ck["ptr_head"],
            "tool_head": ck.get("tool_head"), "dense_selector": sel.state_dict(),
            "route_head": rh.state_dict(), "selector_proj": ck["selector_proj"], "examples": examples},
           "runs/tiny-30m-dispatch-v2.pt")
print("\nSAVED runs/tiny-30m-dispatch-v2.pt\nRETRAIN_DONE", flush=True)
