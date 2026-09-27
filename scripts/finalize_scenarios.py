"""Finalize the scenario model with PROBE COVERAGE for the SOTA behaviours.

The per-segment scenario evals showed abstain/multiturn stuck because the route head / dense selector
probes barely saw abstain-text examples or multi-turn episode contexts. This re-probes with proper
coverage on the (already scenario-SFT'd) backbone:
  - route head: oversample clarify/abstain (text route) so it learns to NOT fire a tool.
  - dense selector: train on single-turn tool samples PLUS per-turn episode contexts (so it ranks the
    next tool mid-episode).
Then scores free-form + abstain + parallel + multiturn and saves a deployable checkpoint.
"""
import argparse
import random

import torch
import torch.nn.functional as F

from localagent.agent.constrained import hybrid_decode
from localagent.agent.dense_selector import BoundSelector, DenseToolSelector, tool_embeddings
from localagent.agent.parser import extract_tool_calls
from localagent.agent.pointer_head import PointerHead
from localagent.agent.routes import ROUTE_INDEX, ROUTES, RouteHead, route_of, route_of_sample
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

ap = argparse.ArgumentParser()
ap.add_argument("--init", default="runs/tiny-30m-scenarios.pt")
ap.add_argument("--out", default="runs/tiny-30m-scenarios-best.pt")
ap.add_argument("--steps", type=int, default=400)
args = ap.parse_args()
torch.set_num_threads(4)
tok = load_tokenizer()

examples = {**TOOL_EXAMPLES}
for k, v in CONTEXTUAL_EXAMPLES.items():
    examples[k] = (examples.get(k, []) + v)[:10]
name_idx = {t.name: i for i, t in enumerate(TOOLS)}

ck = torch.load(args.init, map_location="cpu")
cfg = ModelConfig(**ck["cfg"])
m = LocalAgentLM(cfg); m.load_state_dict(ck["state_dict"]); m.eval()
ptr = PointerHead(cfg.d_model); ptr.load_state_dict(ck["ptr_head"]); ptr.eval()


@torch.no_grad()
def feat_prompt(p):
    return _feat(m, tok, p, "cpu")


@torch.no_grad()
def feat_ctx(ctx):
    ids = tok.encode(ctx)
    _, h = m(torch.tensor([ids]), return_hidden=True)
    return h[0, -1]


# ---- training pools with scenario coverage ----
tool_pool = (Generator(level=3, seed=11).generate_balanced(6)
             + paraphrase_samples(40, seed=11, split="train")
             + contextual_samples(30, seed=11, split="train"))
random.Random(0).shuffle(tool_pool)
tool_pool = tool_pool[:600]
scen = scenario_samples(40, seed=11, split="train")            # clarify/abstain/parallel
episodes = scenario_episodes(40, seed=11, split="train")

# route-head pool: tool samples + ALL scenario text (oversampled x3 so 'text' route is well-seen)
route_pool = tool_pool + [s for s in scen if s.kind == "text"] * 3
print(f"route_pool={len(route_pool)} (text={sum(1 for s in route_pool if s.kind=='text')})  "
      f"tool_pool={len(tool_pool)} episodes={len(episodes)}", flush=True)

# selector training rows: (feature, tool_idx) from single-turn tool samples + episode tool-turns
sel_rows = []
with torch.no_grad():
    for s in tool_pool:
        if s.kind == "tool" and s.ref_name in name_idx:
            sel_rows.append((feat_prompt(s.prompt), name_idx[s.ref_name]))
    for conv in episodes:                                       # per-turn episode contexts
        for i, msg in enumerate(conv.messages):
            if msg.role == Role.assistant and msg.tool_calls and msg.tool_calls[0].name in name_idx:
                ctx = history_text(conv.messages[:i]) + ASSISTANT
                sel_rows.append((feat_ctx(ctx), name_idx[msg.tool_calls[0].name]))
print(f"selector rows={len(sel_rows)} (incl. episode contexts)", flush=True)

# ---- train route head (frozen-feature probe) ----
with torch.no_grad():
    rfeats = torch.stack([feat_prompt(s.prompt) for s in route_pool])
rlabels = torch.tensor([ROUTE_INDEX[route_of_sample(s)] for s in route_pool])
rh = RouteHead(cfg.d_model)
opt = torch.optim.AdamW(rh.parameters(), lr=5e-3)
rng = random.Random(0)
for step in range(args.steps):
    idx = torch.tensor([rng.randrange(len(route_pool)) for _ in range(64)])
    loss = F.cross_entropy(rh(rfeats[idx]), rlabels[idx])
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
rh.eval()

# ---- train dense selector on combined rows ----
sfeats = torch.stack([f for f, _ in sel_rows])
slabels = torch.tensor([j for _, j in sel_rows])
embs = tool_embeddings(TOOLS, examples=examples)
sel = DenseToolSelector(cfg.d_model, emb_dim=embs.shape[1], proj=256)
opt = torch.optim.AdamW(sel.parameters(), lr=5e-3)
for step in range(args.steps):
    idx = torch.tensor([rng.randrange(len(sel_rows)) for _ in range(64)])
    loss = F.cross_entropy(sel(sfeats[idx], embs), slabels[idx])
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
bound = BoundSelector(sel, TOOLS, examples=examples)

# ---- eval ----
ff = FREEFORM_EVAL
abst = [s for s in scenario_samples(3, seed=909, split="eval") if s.category in ("clarify", "abstain")]
par = [s for s in scenario_samples(3, seed=909, split="eval") if s.category == "parallel"]
eps = scenario_episodes(3, seed=909, split="eval")

rt = t1 = cn = 0
for q, gold in ff:
    f = feat_prompt(q)
    rt += ROUTES[int(rh(f).argmax(-1))] == route_of(gold)
    t1 += bound.rank(f)[0] == gold
    c = extract_tool_calls(hybrid_decode(m, tok, q, TOOLS, selector=bound, top_m=1, route_head=rh,
                                         ptr_head=ptr))
    cn += bool(c) and c[0].name == gold
n = len(ff)
ab = sum(ROUTES[int(rh(feat_prompt(s.prompt)).argmax(-1))] == "text" for s in abst)
pp = 0
for s in par:
    gold = {c["name"] for c in (s.calls or [])}
    parts = [p.strip() for p in s.prompt.replace(",", " and ").split(" and ") if p.strip()]
    pp += {bound.rank(feat_prompt(p))[0] for p in parts} == gold
ms = mt = 0
for conv in eps:
    for i, msg in enumerate(conv.messages):
        if msg.role == Role.assistant and msg.tool_calls:
            ms += bound.rank(feat_ctx(history_text(conv.messages[:i]) + ASSISTANT))[0] == \
                msg.tool_calls[0].name
            mt += 1
print(f"\n[SCENARIO-FINAL] free-form: route={rt/n*100:.0f}% top1={t1/n*100:.0f}% call={cn/n*100:.0f}%"
      f" | abstain={ab/max(1,len(abst))*100:.0f}%({len(abst)}) "
      f"parallel={pp/max(1,len(par))*100:.0f}%({len(par)}) "
      f"multiturn_sel={ms/max(1,mt)*100:.0f}%({mt})", flush=True)

torch.save({"cfg": cfg.__dict__, "state_dict": m.state_dict(), "ptr_head": ck["ptr_head"],
            "tool_head": ck.get("tool_head"), "dense_selector": sel.state_dict(),
            "route_head": rh.state_dict(), "selector_proj": 256, "examples": examples}, args.out)
print(f"SAVED {args.out}\nSCENFINAL_DONE", flush=True)
