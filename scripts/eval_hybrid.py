"""Measure the *generable* hybrid decoder vs the 51-way-classifier baseline.

Hybrid = retrieval-based tool selection + pointer-copy arguments (+ optional 5-way route gate), with
NO fixed-N classifier. Reuses the already-trained ptr_head from tiny-30m-50tools.pt — no retraining.
Compares, on the same held set, against the shipped (51-way head + ptr) path and the no-head path.
"""
import sys

import torch

from localagent.agent.dense_selector import BoundSelector, train_dense_selector
from localagent.agent.pointer_head import PointerHead
from localagent.agent.retriever import ToolRetriever
from localagent.agent.routes import RouteHead
from localagent.agent.tool_head import ToolHead
from localagent.agent.toolset import STANDARD_TOOLS as TOOLS
from localagent.data.agent_synth import Generator
from localagent.eval.harness import evaluate_grounded, evaluate_hybrid
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer

QUICK = "--quick" in sys.argv
torch.set_num_threads(4)
tok = load_tokenizer()
retr = ToolRetriever(TOOLS)

ck = torch.load("runs/tiny-30m-50tools.pt", map_location="cpu")
cfg = ModelConfig(**ck["cfg"])
m = LocalAgentLM(cfg); m.load_state_dict(ck["state_dict"]); m.eval()
head = ToolHead(cfg.d_model); head.load_state_dict(ck["tool_head"])
ptr = PointerHead(cfg.d_model); ptr.load_state_dict(ck["ptr_head"])

# optional route head from the routed checkpoint (gate text vs tool); skip if absent
route_head = None
try:
    rk = torch.load("runs/tiny-30m-routed.pt", map_location="cpu")
    route_head = RouteHead(cfg.d_model); route_head.load_state_dict(rk["route_head"]); route_head.eval()
except Exception as e:  # noqa: BLE001
    print("(no route head:", e, ")", flush=True)

# train the dense two-tower selector (cheap frozen-feature probe) on disjoint train samples
sel_steps = 200 if QUICK else 500
train = Generator(level=3, seed=11).generate_balanced(1 if QUICK else 3)
sel_model = train_dense_selector(m, train, tok, TOOLS, steps=sel_steps, device="cpu", log=print)
selector = BoundSelector(sel_model, TOOLS, device="cpu")

n = 40 if QUICK else 120
held = Generator(level=3, seed=909, split="eval").generate_balanced(1)[:n]
print(f"held={len(held)}", flush=True)

base = evaluate_grounded(m, held, tok, TOOLS, tool_head=head, ptr_head=ptr)
retr_h = evaluate_hybrid(m, held, tok, TOOLS, retriever=retr, ptr_head=ptr, k=8)
dense = evaluate_hybrid(m, held, tok, TOOLS, selector=selector, top_m=1, ptr_head=ptr)
dense2 = evaluate_hybrid(m, held, tok, TOOLS, selector=selector, top_m=2, ptr_head=ptr)
print(f"\n51-way head + ptr        (shipped, not generable) = {base['overall']*100:5.1f}%", flush=True)
print(f"HYBRID retrieval + ptr   (no trained selector)    = {retr_h['overall']*100:5.1f}%", flush=True)
print(f"HYBRID dense-selector+ptr(trained, GENERABLE)     = {dense['overall']*100:5.1f}%  (top_m=1)",
      flush=True)
print(f"HYBRID dense top_m=2 +ptr(trained, GENERABLE)     = {dense2['overall']*100:5.1f}%  (top_m=2)",
      flush=True)
print(f"\n{'route':14}{'dense':>8}{'n':>5}", flush=True)
for r, v in dense["by_route"].items():
    print(f"{r:14}{v['acc']*100:7.0f}%{v['n']:5d}", flush=True)
print("HYBRID_DONE", flush=True)
