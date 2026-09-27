"""Long full fine-tune (backbone UNFROZEN) for free-form + referent-conditioned dispatch.

The frozen-feature probes plateaued (~34% free-form call-name) because the backbone's features are
template-overfit. This is the definitive run the user authorized (hours-long): SFT the WHOLE backbone
on the combined corpus — templated + paraphrase-rich + referent-conditioned (contextual) — jointly
retraining the pointer head (features shift, so ptr must follow). Trains in SEGMENTS with a
checkpoint + cheap free-form probe after each, so progress is never lost and the trajectory is
visible. Final: full probes (route head + dense selector) + free-form / paraphrase-eval /
contextual-eval scores.

  python scripts/train_dispatch_long.py [--steps 4000] [--seg 400] [--init runs/ckpt.pt] [--quick]

Saves runs/dispatch-long-step{N}.pt per segment and runs/tiny-30m-dispatch-long.pt at the end.
"""
import argparse
import gc
import random
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

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--seg", type=int, default=400)
ap.add_argument("--lr", type=float, default=7e-4)
ap.add_argument("--init", default="runs/tiny-30m-50tools.pt")
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
if args.quick:
    args.steps, args.seg = 40, 20

torch.set_num_threads(4)
tok = load_tokenizer()

# ---- corpus: templated + paraphrase + referent-conditioned (contextual) ----
templated = Generator(level=3, seed=11).generate_balanced(1 if args.quick else 8)
para = paraphrase_samples(2 if args.quick else 60, seed=11, split="train")
examples = dict(TOOL_EXAMPLES)
try:
    from localagent.data.contextual import CONTEXTUAL_EXAMPLES, contextual_samples
    ctx = contextual_samples(1 if args.quick else 30, seed=11, split="train")
    for k, v in CONTEXTUAL_EXAMPLES.items():            # merge ambiguous-case examples
        examples[k] = (examples.get(k, []) + v)[:10]
    ctx_eval = contextual_samples(2, seed=909, split="eval")
except ImportError:
    print("WARNING: data.contextual not available — running without referent-conditioned data",
          flush=True)
    ctx, ctx_eval = [], []
corpus = templated + para + ctx
random.Random(0).shuffle(corpus)
para_eval = paraphrase_samples(2, seed=909, split="eval")
print(f"corpus: templated={len(templated)} paraphrase={len(para)} contextual={len(ctx)} "
      f"total={len(corpus)} | steps={args.steps} seg={args.seg} lr={args.lr}", flush=True)

ck = torch.load(args.init, map_location="cpu")
cfg = ModelConfig(**ck["cfg"])
m = LocalAgentLM(cfg)
m.load_state_dict(ck["state_dict"])
init_ptr = ck.get("ptr_head")
init_tool = ck.get("tool_head")


def freeform_score(m, route_head, bound, ptr, label):
    rt = top1 = top3 = call_ok = 0
    n = len(FREEFORM_EVAL)
    for q, gold in FREEFORM_EVAL:
        with torch.no_grad():
            feat = _feat(m, tok, q, "cpu")
            rt += ROUTES[int(route_head(feat).argmax(-1))] == route_of(gold)
            ranked = bound.rank(feat)
        top1 += ranked[0] == gold
        top3 += gold in ranked[:3]
        out = hybrid_decode(m, tok, q, TOOLS, selector=bound, top_m=1, ptr_head=ptr)
        calls = extract_tool_calls(out)
        call_ok += bool(calls) and calls[0].name == gold
    print(f"[{label}] FREE-FORM route={rt/n*100:.0f}% sel_top1={top1/n*100:.0f}% "
          f"sel_top3={top3/n*100:.0f}% call_name={call_ok/n*100:.0f}% (n={n})", flush=True)
    return call_ok / n


def split_score(m, bound, samples, label):
    """Selection top-1 on a held Sample split (paraphrase-eval / contextual-eval)."""
    ok = 0
    for s in samples:
        with torch.no_grad():
            ok += bound.rank(_feat(m, tok, s.prompt, "cpu"))[0] == s.ref_name
    print(f"[{label}] sel_top1={ok/max(1,len(samples))*100:.0f}% (n={len(samples)})", flush=True)


def probes(m, data, steps):
    rh = train_route_head(m, data, tok, steps=steps, device="cpu")
    rh.eval()
    sel = train_dense_selector(m, data, tok, TOOLS, steps=steps, device="cpu", examples=examples)
    return rh, sel, BoundSelector(sel, TOOLS, device="cpu", examples=examples)


probe_pool = corpus[:600 if not args.quick else 120]
ptr = PointerHead(cfg.d_model)
if init_ptr:
    ptr.load_state_dict(init_ptr)
ptr.eval()
rh0, _, bound0 = probes(m, probe_pool, 250)
freeform_score(m, rh0, bound0, ptr, "seg0/pre")

done = 0
seg_i = 0
t_start = time.time()
while done < args.steps:
    seg_i += 1
    seg = min(args.seg, args.steps - done)
    lr = args.lr * (0.88 ** (seg_i - 1))          # gentle decay across segments
    print(f"\n=== segment {seg_i}: steps {done}->{done+seg} lr={lr:.1e} "
          f"({(time.time()-t_start)/60:.0f} min elapsed) ===", flush=True)
    m.train()
    _, tool_head, ptr_head = sft(m, corpus, tok, steps=seg, batch_size=16, lr=lr, warmup=20,
                                 device="cpu", joint_tool_head=True, init_tool_head=init_tool,
                                 init_ptr_head=init_ptr, log=print)
    init_tool = tool_head.state_dict()
    init_ptr = ptr_head.state_dict()
    ptr = ptr_head
    ptr.eval()
    m.eval()
    done += seg
    torch.save({"cfg": cfg.__dict__, "state_dict": m.state_dict(), "ptr_head": init_ptr,
                "tool_head": init_tool, "steps_done": done},
               f"runs/dispatch-long-step{done}.pt")
    rh, _, bound = probes(m, probe_pool, 250)     # cheap mid-run probe
    freeform_score(m, rh, bound, ptr, f"seg{seg_i}/{done}st")
    gc.collect()

print("\n=== FINAL: full probes + all held evals ===", flush=True)
rh, sel, bound = probes(m, corpus[:1500], 600)
freeform_score(m, rh, bound, ptr, "FINAL")
split_score(m, bound, para_eval, "FINAL paraphrase-eval")
if ctx_eval:
    split_score(m, bound, ctx_eval, "FINAL contextual-eval")
torch.save({"cfg": cfg.__dict__, "state_dict": m.state_dict(), "ptr_head": init_ptr,
            "tool_head": init_tool, "dense_selector": sel.state_dict(),
            "route_head": rh.state_dict(), "selector_proj": 256, "examples": examples},
           "runs/tiny-30m-dispatch-long.pt")
print("\nSAVED runs/tiny-30m-dispatch-long.pt\nLONG_DONE", flush=True)
