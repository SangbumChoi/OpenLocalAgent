"""Backbone paraphrase-SFT — the lever that makes the model's FEATURES phrasing-robust.

The selector/route head are frozen-feature probes, so OOD dispatch plateaued at the backbone's
templated-overfit features (free-form call-name ~34%). Here we SFT the backbone itself on the
combined templated + paraphrase corpus (plain LM loss), then retrain the route head + dense selector
on the new backbone and re-measure the SAME free-form held set. Compares against the frozen-backbone
numbers to show the lift from making features phrasing-robust.

`--quick` smoke. Saves runs/tiny-30m-dispatch-sft.pt.
"""
import sys
import time

import torch

from localagent.agent.constrained import hybrid_decode
from localagent.agent.dense_selector import BoundSelector, train_dense_selector
from localagent.agent.parser import extract_tool_calls
from localagent.agent.pointer_head import PointerHead
from localagent.agent.routes import ROUTES, route_of, train_route_head
from localagent.agent.tool_head import _feat
from localagent.agent.toolset import STANDARD_TOOLS as TOOLS
from localagent.data.agent_synth import Generator
from localagent.data.paraphrase import TOOL_EXAMPLES, paraphrase_samples
from localagent.eval.freeform import FREEFORM_EVAL
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from localagent.train.sft import sft

QUICK = "--quick" in sys.argv
torch.set_num_threads(4)
tok = load_tokenizer()

ck = torch.load("runs/tiny-30m-50tools.pt", map_location="cpu")
cfg = ModelConfig(**ck["cfg"])
m = LocalAgentLM(cfg); m.load_state_dict(ck["state_dict"])
ptr = PointerHead(cfg.d_model); ptr.load_state_dict(ck["ptr_head"]); ptr.eval()

sft_steps = 60 if QUICK else 400
probe_steps = 150 if QUICK else 500
templated = Generator(level=3, seed=11).generate_balanced(1 if QUICK else 6)
para = paraphrase_samples(3 if QUICK else 40, seed=11, split="train")   # n per tool (x50)
combined = templated + para
import random as _rnd
_rnd.Random(0).shuffle(combined)
probe_data = combined if QUICK else combined[:1500]   # cap probe feature-caching cost
print(f"templated={len(templated)} paraphrase={len(para)} combined={len(combined)} "
      f"probe_data={len(probe_data)} sft_steps={sft_steps}", flush=True)


def score(m, route_head, bound, label):
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
    print(f"[{label:16}] route_acc={rt/n*100:4.0f}%  sel_top1={top1/n*100:4.0f}%  "
          f"sel_top3={top3/n*100:4.0f}%  call_name_acc={call_ok/n*100:4.0f}%  (n={n})", flush=True)


def probes_and_score(label):
    rh = train_route_head(m, probe_data, tok, steps=probe_steps, device="cpu"); rh.eval()
    sel = train_dense_selector(m, probe_data, tok, TOOLS, steps=probe_steps, device="cpu",
                               examples=TOOL_EXAMPLES)
    bound = BoundSelector(sel, TOOLS, device="cpu", examples=TOOL_EXAMPLES)
    score(m, rh, bound, label)
    return rh, sel, bound


print("\n--- BEFORE backbone SFT (frozen 50tools features) ---", flush=True)
probes_and_score("frozen-backbone")

print(f"\n--- SFT backbone on combined corpus ({sft_steps} steps) ---", flush=True)
t0 = time.time()
sft(m, combined, tok, steps=sft_steps, batch_size=32, lr=5e-4, device="cpu", joint_tool_head=False,
    log=print)
print(f"backbone SFT done in {(time.time()-t0)/60:.1f} min", flush=True)
m.eval()

print("\n--- AFTER backbone SFT (phrasing-robust features) ---", flush=True)
rh, sel, bound = probes_and_score("sft-backbone")

torch.save({"cfg": cfg.__dict__, "state_dict": m.state_dict(), "dense_selector": sel.state_dict(),
            "route_head": rh.state_dict(), "selector_proj": 256, "examples": TOOL_EXAMPLES},
           "runs/tiny-30m-dispatch-sft.pt")
print("\nSAVED runs/tiny-30m-dispatch-sft.pt\nSFT_PARA_DONE", flush=True)
