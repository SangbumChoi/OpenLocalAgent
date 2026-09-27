"""Close the free-form (OOD) dispatch gap with paraphrase-rich data + example-augmented selection.

Trains, on the SAME frozen backbone, two dense selectors and a route head:
  BASELINE  — selector on templated data only, plain (name+desc) tool embeddings.
  IMPROVED  — selector on templated+paraphrase data, example-augmented tool embeddings.
Scores both on the hand-authored free-form held set (eval/freeform.py) — the honest OOD test — by
tool SELECTION (route head + selector top-1 / top-3 + full hybrid_decode call name). Reports the lift.
"""
import sys

import torch

from localagent.agent.constrained import hybrid_decode
from localagent.agent.dense_selector import BoundSelector, train_dense_selector
from localagent.agent.parser import extract_tool_calls
from localagent.agent.pointer_head import PointerHead
from localagent.agent.routes import ROUTES, RouteHead, route_of, train_route_head
from localagent.agent.tool_head import _feat
from localagent.agent.toolset import STANDARD_TOOLS as TOOLS
from localagent.data.agent_synth import Generator
from localagent.data.paraphrase import TOOL_EXAMPLES, paraphrase_samples
from localagent.eval.freeform import FREEFORM_EVAL
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer

QUICK = "--quick" in sys.argv
torch.set_num_threads(4)
tok = load_tokenizer()

ck = torch.load("runs/tiny-30m-50tools.pt", map_location="cpu")
cfg = ModelConfig(**ck["cfg"])
m = LocalAgentLM(cfg); m.load_state_dict(ck["state_dict"]); m.eval()
ptr = PointerHead(cfg.d_model); ptr.load_state_dict(ck["ptr_head"])

steps = 200 if QUICK else 600
templated = Generator(level=3, seed=11).generate_balanced(1 if QUICK else 4)
para = paraphrase_samples(3 if QUICK else 20, seed=11, split="train")  # n per tool (x50)
combined = templated + para
print(f"templated={len(templated)} paraphrase={len(para)} combined={len(combined)}", flush=True)

# BASELINE: templated only, plain tool embeddings
sel_base = train_dense_selector(m, templated, tok, TOOLS, steps=steps, device="cpu")
bound_base = BoundSelector(sel_base, TOOLS, device="cpu")
# IMPROVED: templated+paraphrase, example-augmented tool embeddings
sel_imp = train_dense_selector(m, combined, tok, TOOLS, steps=steps, device="cpu",
                               examples=TOOL_EXAMPLES, log=print)
bound_imp = BoundSelector(sel_imp, TOOLS, device="cpu", examples=TOOL_EXAMPLES)
# route head on combined data
route_head = train_route_head(m, combined, tok, steps=steps, device="cpu")
route_head.eval()


def score(bound, label):
    rt = top1 = top3 = call_ok = 0
    n = len(FREEFORM_EVAL)
    for q, gold in FREEFORM_EVAL:
        with torch.no_grad():
            feat = _feat(m, tok, q, "cpu")
            pred_route = ROUTES[int(route_head(feat).argmax(-1))]
            ranked = bound.rank(feat)
        rt += pred_route == route_of(gold)
        top1 += ranked[0] == gold
        top3 += gold in ranked[:3]
        out = hybrid_decode(m, tok, q, TOOLS, selector=bound, top_m=1, ptr_head=ptr)
        calls = extract_tool_calls(out)
        call_ok += bool(calls) and calls[0].name == gold
    print(f"[{label:8}] route_acc={rt/n*100:4.0f}%  sel_top1={top1/n*100:4.0f}%  "
          f"sel_top3={top3/n*100:4.0f}%  call_name_acc={call_ok/n*100:4.0f}%  (n={n})", flush=True)


print(f"\nFree-form OOD dispatch ({len(FREEFORM_EVAL)} hand-written queries):", flush=True)
score(bound_base, "BASELINE")
score(bound_imp, "IMPROVED")

torch.save({"cfg": cfg.__dict__, "state_dict": m.state_dict(),
            "dense_selector": sel_imp.state_dict(), "route_head": route_head.state_dict(),
            "selector_proj": 256, "examples": TOOL_EXAMPLES},
           "runs/tiny-30m-dispatch.pt")
print("\nSAVED runs/tiny-30m-dispatch.pt\nDISPATCH_DONE", flush=True)
