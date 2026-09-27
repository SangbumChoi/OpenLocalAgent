"""Full dispatch scorecard for a checkpoint: free-form OOD + paraphrase-eval + contextual-eval,
with selection vs argument-value breakdown. Eval only (no training)."""
import argparse
import json

import torch

from localagent.agent.constrained import hybrid_decode
from localagent.agent.dense_selector import BoundSelector, DenseToolSelector, tool_embeddings
from localagent.agent.parser import extract_tool_calls
from localagent.agent.pointer_head import PointerHead
from localagent.agent.routes import ROUTES, RouteHead, route_of
from localagent.agent.tool_head import _feat
from localagent.agent.toolset import STANDARD_TOOLS as TOOLS
from localagent.data.contextual import contextual_samples
from localagent.data.paraphrase import paraphrase_samples
from localagent.data.schema import ToolCall
from localagent.eval.freeform import FREEFORM_EVAL
from localagent.eval.tool_eval import match_calls
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="runs/tiny-30m-dispatch-long.pt")
args = ap.parse_args()
torch.set_num_threads(4)
tok = load_tokenizer()
ck = torch.load(args.ckpt, map_location="cpu")
cfg = ModelConfig(**ck["cfg"])
m = LocalAgentLM(cfg); m.load_state_dict(ck["state_dict"]); m.eval()
ptr = PointerHead(cfg.d_model); ptr.load_state_dict(ck["ptr_head"]); ptr.eval()
examples = ck.get("examples", {})
emb_dim = tool_embeddings(TOOLS[:1]).shape[1]
sel = DenseToolSelector(cfg.d_model, emb_dim=emb_dim, proj=ck.get("selector_proj", 256))
sel.load_state_dict(ck["dense_selector"])
bound = BoundSelector(sel, TOOLS, examples=examples)
rh = RouteHead(cfg.d_model); rh.load_state_dict(ck["route_head"]); rh.eval()
print(f"scorecard for {args.ckpt}", flush=True)


def decode(prompt):
    return extract_tool_calls(hybrid_decode(m, tok, prompt, TOOLS, selector=bound, top_m=1,
                                            route_head=rh, ptr_head=ptr))


# free-form OOD
rt = t1 = cn = 0
for query, gold in FREEFORM_EVAL:
    feat = _feat(m, tok, query, "cpu")
    rt += ROUTES[int(rh(feat).argmax(-1))] == route_of(gold)
    t1 += bound.rank(feat)[0] == gold
    c = decode(query)
    cn += bool(c) and c[0].name == gold
n = len(FREEFORM_EVAL)
print(f"FREE-FORM (44 OOD): route={rt/n*100:.0f}%  sel_top1={t1/n*100:.0f}%  "
      f"call_name={cn/n*100:.0f}%", flush=True)


def audit(samples, label):
    tot = sel_ok = full = 0
    for s in samples:
        if s.kind != "tool":
            continue
        tot += 1
        c = decode(s.prompt)
        if c and c[0].name == s.ref_name:
            sel_ok += 1
            gold = ToolCall(**(json.loads(s.target) if not s.calls else s.calls[0]))
            full += match_calls([c[0]], [gold])
    print(f"{label}: tool_n={tot}  selection={sel_ok/max(1,tot)*100:.0f}%  "
          f"full(sel+args)={full/max(1,tot)*100:.0f}%  args|sel={full/max(1,sel_ok)*100:.0f}%",
          flush=True)


audit(paraphrase_samples(2, seed=909, split="eval"), "PARAPHRASE-eval")
audit(contextual_samples(2, seed=909, split="eval"), "CONTEXTUAL-eval")
print("SCORECARD_DONE", flush=True)
