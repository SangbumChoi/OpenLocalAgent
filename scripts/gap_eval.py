"""Measure the gap between the two tool-call paths on the SAME held eval set:
  - pure generative : the LM free-generates tool(args) as text -> AST-parsed (NO heads)
  - grounded        : LM + 51-way tool_head (selection) + ptr_head (arg copy)
Loads the retrained 50-tool checkpoint and reports overall + per-category for both,
sorted by the per-category gap (grounded - puregen) so we can see where the heads carry
the model vs where the LM already stands on its own."""
import torch

from localagent.agent.pointer_head import PointerHead
from localagent.agent.tool_head import ToolHead
from localagent.agent.toolset import STANDARD_TOOLS as TOOLS
from localagent.data.agent_synth import Generator
from localagent.eval.harness import evaluate, evaluate_grounded
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer

torch.set_num_threads(4)
tok = load_tokenizer()
ck = torch.load("runs/tiny-30m-50tools.pt", map_location="cpu")
cfg = ModelConfig(**ck["cfg"])
model = LocalAgentLM(cfg)
model.load_state_dict(ck["state_dict"])
model.eval()
head = ToolHead(cfg.d_model)
head.load_state_dict(ck["tool_head"])
ptr = PointerHead(cfg.d_model)
ptr.load_state_dict(ck["ptr_head"])

# SAME held set the retrain reported on
held = Generator(level=3, seed=909, split="eval").generate_balanced(80)

print(f"eval on {len(held)} held samples (seed=909)\n", flush=True)
g = evaluate_grounded(model, held, tok, TOOLS, device="cpu", tool_head=head, ptr_head=ptr)
p = evaluate(model, held, tok, device="cpu")

print(f"[GROUNDED  LM+heads ] overall = {g['overall']*100:5.1f}%", flush=True)
print(f"[PURE-GEN  LM only  ] overall = {p['overall']*100:5.1f}%", flush=True)
print(f"  => head lift = {(g['overall']-p['overall'])*100:+.1f} pts\n", flush=True)

# per-category gap, biggest head-dependence first
gc, pc = g["categories"], p["categories"]
rows = sorted(((k, gc.get(k, 0.0), pc.get(k, 0.0)) for k in gc),
             key=lambda r: (r[1] - r[2]), reverse=True)
print(f"{'category':22} {'grounded':>9} {'pure-gen':>9} {'gap':>7}", flush=True)
for k, gv, pv in rows:
    print(f"{k:22} {gv*100:8.0f}% {pv*100:8.0f}% {(gv-pv)*100:+6.0f}", flush=True)
print("GAP_EVAL_DONE", flush=True)
