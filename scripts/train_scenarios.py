"""Scenario continuation fine-tune — folds SOTA-agent behaviours onto the dispatch model.

Continues from a dispatch checkpoint (default the long run's output) and SFTs on the full corpus PLUS
the scenario data: clarify/abstain/parallel single-turn samples and multi-turn episodes (workflow /
chained / error_recovery, fed via sft `conversations=`). Backbone unfrozen, pointer head co-trained.
Segmented with checkpoints. Scores free-form dispatch + abstention (don't over-trigger) + parallel
selection + teacher-forced multi-turn next-tool selection.

  python scripts/train_scenarios.py [--steps 2000] [--seg 400] [--init runs/tiny-30m-dispatch-long.pt]
"""
import argparse
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
from localagent.data.contextual import CONTEXTUAL_EXAMPLES, contextual_samples
from localagent.data.paraphrase import TOOL_EXAMPLES, paraphrase_samples
from localagent.data.render import history_text
from localagent.data.scenarios import scenario_episodes, scenario_samples
from localagent.data.schema import Role
from localagent.eval.freeform import FREEFORM_EVAL
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import ASSISTANT, load_tokenizer
from localagent.train.sft import sft

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--seg", type=int, default=400)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--init", default="runs/tiny-30m-dispatch-long.pt")
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
if args.quick:
    args.steps, args.seg = 40, 20

torch.set_num_threads(4)
tok = load_tokenizer()

q = args.quick
templated = Generator(level=3, seed=11).generate_balanced(1 if q else 6)
para = paraphrase_samples(2 if q else 50, seed=11, split="train")
ctx = contextual_samples(1 if q else 30, seed=11, split="train")
scen = scenario_samples(2 if q else 30, seed=11, split="train")          # clarify/abstain/parallel
episodes = scenario_episodes(1 if q else 30, seed=11, split="train")      # multi-turn
corpus = templated + para + ctx + scen
random.Random(0).shuffle(corpus)
examples = {**TOOL_EXAMPLES}
for k, v in CONTEXTUAL_EXAMPLES.items():
    examples[k] = (examples.get(k, []) + v)[:10]

# held sets
ff = FREEFORM_EVAL
abstain_held = [s for s in scenario_samples(2, seed=909, split="eval")
                if s.category in ("clarify", "abstain")]
parallel_held = [s for s in scenario_samples(2, seed=909, split="eval") if s.category == "parallel"]
ep_held = scenario_episodes(2, seed=909, split="eval")
para_eval = paraphrase_samples(2, seed=909, split="eval")
ctx_eval = contextual_samples(2, seed=909, split="eval")
print(f"corpus={len(corpus)} episodes={len(episodes)} | held: abstain={len(abstain_held)} "
      f"parallel={len(parallel_held)} ep={len(ep_held)} para={len(para_eval)} ctx={len(ctx_eval)}",
      flush=True)

ck = torch.load(args.init, map_location="cpu")
cfg = ModelConfig(**ck["cfg"])
m = LocalAgentLM(cfg); m.load_state_dict(ck["state_dict"])
init_ptr = ck.get("ptr_head")
init_tool = ck.get("tool_head")


@torch.no_grad()
def ctx_feat(context: str):
    ids = tok.encode(context)
    _, h = m(torch.tensor([ids]), return_hidden=True)
    return h[0, -1]


def evals(rh, bound, ptr, label):
    # free-form dispatch
    rt = t1 = cn = 0
    for query, gold in ff:
        feat = _feat(m, tok, query, "cpu")
        rt += ROUTES[int(rh(feat).argmax(-1))] == route_of(gold)
        t1 += bound.rank(feat)[0] == gold
        calls = extract_tool_calls(hybrid_decode(m, tok, query, TOOLS, selector=bound, top_m=1,
                                                 route_head=rh, ptr_head=ptr))
        cn += bool(calls) and calls[0].name == gold
    n = len(ff)
    # abstention: clarify/abstain must route to text (no tool fired)
    ab = sum(ROUTES[int(rh(_feat(m, tok, s.prompt, "cpu")).argmax(-1))] == "text"
             for s in abstain_held)
    # parallel: per-conjunct selection, set of predicted names == set of gold names
    pp = 0
    for s in parallel_held:
        gold_names = {c["name"] for c in (s.calls or [])}
        parts = [p.strip() for p in s.prompt.replace(",", " and ").split(" and ") if p.strip()]
        pred = {bound.rank(_feat(m, tok, p, "cpu"))[0] for p in parts}
        pp += pred == gold_names
    # multi-turn: teacher-forced next-tool selection at each assistant tool-call turn
    ms = mt = 0
    for conv in ep_held:
        for i, msg in enumerate(conv.messages):
            if msg.role == Role.assistant and msg.tool_calls:
                feat = ctx_feat(history_text(conv.messages[:i]) + ASSISTANT)
                ms += bound.rank(feat)[0] == msg.tool_calls[0].name
                mt += 1
    print(f"[{label}] free-form: route={rt/n*100:.0f}% top1={t1/n*100:.0f}% call={cn/n*100:.0f}% | "
          f"abstain={ab/max(1,len(abstain_held))*100:.0f}% parallel={pp/max(1,len(parallel_held))*100:.0f}% "
          f"multiturn_sel={ms/max(1,mt)*100:.0f}% (mt={mt})", flush=True)


def probes(data, steps):
    rh = train_route_head(m, data, tok, steps=steps, device="cpu"); rh.eval()
    sel = train_dense_selector(m, data, tok, TOOLS, steps=steps, device="cpu", examples=examples)
    return rh, BoundSelector(sel, TOOLS, device="cpu", examples=examples), sel


probe_pool = corpus[:120 if q else 500]
ptr = PointerHead(cfg.d_model)
if init_ptr:
    ptr.load_state_dict(init_ptr)
ptr.eval()
rh, bound, _ = probes(probe_pool, 200)
evals(rh, bound, ptr, "pre")

done = seg_i = 0
t0 = time.time()
while done < args.steps:
    seg_i += 1
    seg = min(args.seg, args.steps - done)
    print(f"\n=== segment {seg_i}: {done}->{done+seg} ({(time.time()-t0)/60:.0f} min) ===", flush=True)
    m.train()
    # pure-LM SFT on corpus + episodes: the backbone learns clarify/abstain/parallel/episode
    # patterns (read by the fresh route-head/selector probes). joint_tool_head=False skips the
    # UNUSED 51-way head + the expensive multi-turn head training (we select via the dense selector).
    sft(m, corpus, tok, steps=seg, batch_size=8, lr=args.lr, warmup=20, device="cpu",
        joint_tool_head=False, conversations=episodes, log=print)
    m.eval()
    done += seg
    torch.save({"cfg": cfg.__dict__, "state_dict": m.state_dict(), "ptr_head": init_ptr,
                "tool_head": init_tool, "steps_done": done}, f"runs/scenarios-step{done}.pt")
    rh, bound, sel = probes(probe_pool, 200)
    evals(rh, bound, ptr, f"seg{seg_i}/{done}")

print("\n=== FINAL ===", flush=True)
rh, bound, sel = probes(corpus[:1500], 500)
evals(rh, bound, ptr, "FINAL")
torch.save({"cfg": cfg.__dict__, "state_dict": m.state_dict(), "ptr_head": init_ptr,
            "tool_head": init_tool, "dense_selector": sel.state_dict(), "route_head": rh.state_dict(),
            "selector_proj": 256, "examples": examples}, "runs/tiny-30m-scenarios.pt")
print("SAVED runs/tiny-30m-scenarios.pt\nSCEN_DONE", flush=True)
