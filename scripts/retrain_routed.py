"""Route-based tool-calling pipeline (replaces the 51-way classifier).

Selection is now: a small **5-way route head** (web_search / computer_use / code / app_action / text)
picks the modality, and the LM **free-generates the specific `tool(args)` as text** (AST-scored).
Adding a concrete tool never reshapes the head — only adding a whole new modality does.

Reuses the already-SFT'd 50-tool backbone (runs/tiny-30m-50tools.pt) as the generative LM, trains
just the cheap route-head probe on top, and reports route_acc + generative tool-call accuracy per
route on the same held eval set the old pipeline used. `--quick` does a fast smoke.
"""
import sys
import time

import torch

from localagent.agent.routes import ROUTES, train_route_head
from localagent.data.agent_synth import Generator
from localagent.eval.harness import evaluate_routed
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer

QUICK = "--quick" in sys.argv
torch.set_num_threads(4)
tok = load_tokenizer()

ck = torch.load("runs/tiny-30m-50tools.pt", map_location="cpu")  # SFT'd generative LM (v1)
cfg = ModelConfig(**ck["cfg"])
model = LocalAgentLM(cfg)
model.load_state_dict(ck["state_dict"])
model.eval()

n_train = 120 if QUICK else 2400
n_eval = 16 if QUICK else 80
steps = 120 if QUICK else 400
train = Generator(level=3, seed=7).generate_balanced(n_train)
held = Generator(level=3, seed=909, split="eval").generate_balanced(n_eval)

t0 = time.time()
route_head = train_route_head(model, train, tok, steps=steps, device="cpu", log=print)
print(f"route-head trained in {time.time()-t0:.1f}s", flush=True)

r = evaluate_routed(model, held, tok, route_head, device="cpu")
print(f"\n[ROUTED] route_acc(head, 5-way) = {r['route_acc']*100:5.1f}%", flush=True)
print(f"[ROUTED] gen_acc (LM free-gen tool calls) = {r['gen_acc']*100:5.1f}%   <- portable headline",
      flush=True)
print(f"\n{'route':14} {'route_acc':>10} {'gen_acc':>9} {'n':>4}", flush=True)
for route in ROUTES:
    v = r["by_route"].get(route)
    if v:
        print(f"{route:14} {v['route']*100:9.0f}% {v['gen']*100:8.0f}% {v['n']:4d}", flush=True)

torch.save({"cfg": cfg.__dict__, "state_dict": model.state_dict(),
            "route_head": route_head.state_dict(), "routes": ROUTES},
           "runs/tiny-30m-routed.pt")
print("\nSAVED runs/tiny-30m-routed.pt", flush=True)
print("ROUTED_DONE", flush=True)
