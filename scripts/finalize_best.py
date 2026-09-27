"""Finalize a deployable dispatch checkpoint from a backbone-only mid-run checkpoint.

Trains the route head + dense selector probes on the given backbone (which mid-run probing showed
strong) and scores free-form, to resolve the mid-run-vs-FINAL discrepancy and save a clean
deployable checkpoint with all heads. Eval + cheap probes only (no backbone SFT).
"""
import argparse

import torch

from localagent.agent.constrained import hybrid_decode
from localagent.agent.dense_selector import BoundSelector, train_dense_selector
from localagent.agent.parser import extract_tool_calls
from localagent.agent.pointer_head import PointerHead
from localagent.agent.routes import ROUTES, route_of, train_route_head
from localagent.agent.tool_head import _feat
from localagent.agent.toolset import STANDARD_TOOLS as TOOLS
from localagent.data.agent_synth import Generator
from localagent.data.contextual import CONTEXTUAL_EXAMPLES, contextual_samples
from localagent.data.paraphrase import TOOL_EXAMPLES, paraphrase_samples
from localagent.eval.freeform import FREEFORM_EVAL
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--init", default="runs/dispatch-long-step2400.pt")
ap.add_argument("--out", default="runs/tiny-30m-dispatch-best.pt")
ap.add_argument("--pool", type=int, default=700)
ap.add_argument("--steps", type=int, default=300)
args = ap.parse_args()
torch.set_num_threads(4)
tok = load_tokenizer()

examples = {**TOOL_EXAMPLES}
for k, v in CONTEXTUAL_EXAMPLES.items():
    examples[k] = (examples.get(k, []) + v)[:10]
import random
corpus = (Generator(level=3, seed=11).generate_balanced(8)
          + paraphrase_samples(60, seed=11, split="train")
          + contextual_samples(30, seed=11, split="train"))
random.Random(0).shuffle(corpus)
pool = corpus[:args.pool]

ck = torch.load(args.init, map_location="cpu")
cfg = ModelConfig(**ck["cfg"])
m = LocalAgentLM(cfg); m.load_state_dict(ck["state_dict"]); m.eval()
ptr = PointerHead(cfg.d_model); ptr.load_state_dict(ck["ptr_head"]); ptr.eval()
print(f"finalize {args.init}: pool={len(pool)} steps={args.steps}", flush=True)

rh = train_route_head(m, pool, tok, steps=args.steps, device="cpu"); rh.eval()
sel = train_dense_selector(m, pool, tok, TOOLS, steps=args.steps, device="cpu", examples=examples)
bound = BoundSelector(sel, TOOLS, examples=examples)

rt = t1 = t3 = cn = 0
for q, gold in FREEFORM_EVAL:
    feat = _feat(m, tok, q, "cpu")
    rt += ROUTES[int(rh(feat).argmax(-1))] == route_of(gold)
    r = bound.rank(feat)
    t1 += r[0] == gold
    t3 += gold in r[:3]
    c = extract_tool_calls(hybrid_decode(m, tok, q, TOOLS, selector=bound, top_m=1,
                                         route_head=rh, ptr_head=ptr))
    cn += bool(c) and c[0].name == gold
n = len(FREEFORM_EVAL)
print(f"FREE-FORM: route={rt/n*100:.0f}% top1={t1/n*100:.0f}% top3={t3/n*100:.0f}% "
      f"call_name={cn/n*100:.0f}%", flush=True)

torch.save({"cfg": cfg.__dict__, "state_dict": m.state_dict(), "ptr_head": ck["ptr_head"],
            "tool_head": ck.get("tool_head"), "dense_selector": sel.state_dict(),
            "route_head": rh.state_dict(), "selector_proj": 256, "examples": examples}, args.out)
print(f"SAVED {args.out}\nFINALIZE_DONE", flush=True)
