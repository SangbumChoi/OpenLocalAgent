"""OPD experiment 2: route SPECIALISTS -> distill into one (the DeepSeek-V4 recipe, our setting).

Hypothesis: a 28M model has limited capacity, so a backbone fine-tuned on ONE route should dispatch
that route better than the generalist (less interference); then sequence-distilling the specialists
into a single consolidated model should keep most of those per-route gains in one model.

For each route we: SFT a specialist from the 50-tool backbone on that route's data, probe a dense
selector on it, and measure that route's selection top-1. Then we consolidate (SFT one backbone on
the union of the specialists' route data) and measure every route. Compared against the generalist
(scenarios-best). Bounded + checkpointed.
"""
import random
import sys
import time

import torch

from localagent.agent.dense_selector import BoundSelector, train_dense_selector
from localagent.agent.routes import route_of
from localagent.agent.tool_head import _feat
from localagent.agent.toolset import STANDARD_TOOLS as TOOLS
from localagent.data.contextual import CONTEXTUAL_EXAMPLES, contextual_samples
from localagent.data.paraphrase import TOOL_EXAMPLES, paraphrase_samples
from localagent.data.render import IGNORE, assistant_body
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import ASSISTANT, USER, load_tokenizer
from localagent.train.loop import cosine_lr, pad_batch, set_lr

QUICK = "--quick" in sys.argv
torch.set_num_threads(4)
tok = load_tokenizer()
ROUTES_TESTED = ["code", "web_search", "app_action"]
examples = {**TOOL_EXAMPLES}
for k, v in CONTEXTUAL_EXAMPLES.items():
    examples[k] = (examples.get(k, []) + v)[:10]


def base_model():
    ck = torch.load("runs/tiny-30m-50tools.pt", map_location="cpu")
    cfg = ModelConfig(**ck["cfg"])
    mm = LocalAgentLM(cfg); mm.load_state_dict(ck["state_dict"])
    return mm, cfg


def route_data(split, n):
    d = paraphrase_samples(n, seed=5, split=split) + contextual_samples(n, seed=5, split=split)
    return [s for s in d if s.kind == "tool"]


train_all = route_data("train", 2 if QUICK else 30)
eval_all = route_data("eval", 2)
by_route_eval = {r: [s for s in eval_all if route_of(s.ref_name) == r] for r in ROUTES_TESTED}


def sft_backbone(mm, cfg, samples, steps):
    rows = []
    for s in samples:
        p = tok.encode(f"{USER}{s.prompt}{ASSISTANT}")
        b = tok.encode(assistant_body(s)) + [tok.eos_id]
        rows.append((p + b, [IGNORE] * len(p) + b))
    rng = random.Random(0)
    mm.train()
    opt = torch.optim.AdamW(mm.parameters(), lr=5e-4, betas=(0.9, 0.95))
    for step in range(steps):
        set_lr(opt, cosine_lr(step, steps, 5e-4, 10, 0.1))
        batch = [rows[rng.randrange(len(rows))] for _ in range(16)]
        x, y = pad_batch(batch, tok.pad_id, "cpu")
        _, loss = mm(x, targets=y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(mm.parameters(), 1.0); opt.step()
    mm.eval()
    return mm


def route_sel_acc(mm, sel_train, label):
    """Train a selector probe on this backbone and report per-route selection top-1."""
    sel = train_dense_selector(mm, sel_train, tok, TOOLS, steps=40 if QUICK else 300, device="cpu",
                               examples=examples)
    bound = BoundSelector(sel, TOOLS, examples=examples)
    out = {}
    for r in ROUTES_TESTED:
        held = by_route_eval[r]
        ok = sum(bound.rank(_feat(mm, tok, s.prompt, "cpu"))[0] == s.ref_name for s in held)
        out[r] = (ok, len(held))
    print(f"[{label}] " + "  ".join(f"{r}={out[r][0]/max(1,out[r][1])*100:.0f}%({out[r][1]})"
                                    for r in ROUTES_TESTED), flush=True)
    return out


steps_spec = 40 if QUICK else 250
t0 = time.time()

# ---- generalist baseline (scenarios-best backbone) ----
gck = torch.load("runs/tiny-30m-scenarios-best.pt", map_location="cpu")
gen = LocalAgentLM(ModelConfig(**gck["cfg"])); gen.load_state_dict(gck["state_dict"]); gen.eval()
print("=== generalist (scenarios-best) per-route selection ===", flush=True)
route_sel_acc(gen, train_all, "GENERALIST")

# ---- per-route specialists ----
print("\n=== route specialists (each SFT on its own route only) ===", flush=True)
spec_data = {}
for r in ROUTES_TESTED:
    data_r = [s for s in train_all if route_of(s.ref_name) == r]
    spec_data[r] = data_r
    mm, cfg = base_model()
    mm = sft_backbone(mm, cfg, data_r, steps_spec)
    # selector trained on the SAME route data so it matches the specialist's competence
    res = route_sel_acc(mm, data_r, f"SPECIALIST[{r}]  (train n={len(data_r)})")
    print(f"    -> specialist {r} on its OWN route: "
          f"{res[r][0]/max(1,res[r][1])*100:.0f}% ({(time.time()-t0)/60:.0f} min)", flush=True)

# ---- consolidate: SFT one backbone on the union, then measure all routes ----
print("\n=== consolidated (one backbone, union of route data) ===", flush=True)
mm, cfg = base_model()
mm = sft_backbone(mm, cfg, train_all, 40 if QUICK else 400)
route_sel_acc(mm, train_all, "CONSOLIDATED")
torch.save({"cfg": cfg.__dict__, "state_dict": mm.state_dict(), "ptr_head": gck["ptr_head"],
            "tool_head": gck.get("tool_head"), "examples": examples}, "runs/tiny-30m-consolidated.pt")
print(f"\nSAVED runs/tiny-30m-consolidated.pt ({(time.time()-t0)/60:.0f} min)\nSPEC_DONE", flush=True)
