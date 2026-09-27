"""OPD experiment 1: on-policy / sequence distillation of the GROUNDED decoder (oracle teacher) into
free-generation.

The grounded decoder (route + dense selector + pointer-copy) produces correct tool calls (~57% OOD)
but free-generation does not (~0-5%). Here the grounded decoder is the *teacher*: we relabel a pool of
prompts with its decoded tool call, keep only the ones it gets right (rejection sampling — distill the
oracle's SUCCESSES, no error propagation), and SFT the LM to free-generate those targets. Then we
compare FREE-GENERATION (no heads at all) before vs after, to see if the selection+copy behaviour can
be internalised into the raw LM. Bounded + checkpointed.
"""
import json
import random
import sys
import time

import torch

from localagent.agent.constrained import hybrid_decode
from localagent.agent.dense_selector import BoundSelector, DenseToolSelector, tool_embeddings
from localagent.agent.parser import extract_tool_calls
from localagent.agent.pointer_head import PointerHead
from localagent.agent.routes import RouteHead
from localagent.agent.toolset import STANDARD_TOOLS as TOOLS
from localagent.data.contextual import contextual_samples
from localagent.data.paraphrase import paraphrase_samples
from localagent.data.render import IGNORE
from localagent.data.schema import ToolCall
from localagent.eval.tool_eval import match_calls
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import ASSISTANT, USER, load_tokenizer
from localagent.train.loop import cosine_lr, pad_batch, set_lr

QUICK = "--quick" in sys.argv
torch.set_num_threads(4)
tok = load_tokenizer()

ck = torch.load("runs/tiny-30m-scenarios-best.pt", map_location="cpu")
cfg = ModelConfig(**ck["cfg"])
m = LocalAgentLM(cfg); m.load_state_dict(ck["state_dict"]); m.eval()
ptr = PointerHead(cfg.d_model); ptr.load_state_dict(ck["ptr_head"]); ptr.eval()
emb_dim = tool_embeddings(TOOLS[:1]).shape[1]
sel = DenseToolSelector(cfg.d_model, emb_dim=emb_dim, proj=ck["selector_proj"])
sel.load_state_dict(ck["dense_selector"])
examples = ck.get("examples", {})
bound = BoundSelector(sel, TOOLS, examples=examples)
rh = RouteHead(cfg.d_model); rh.load_state_dict(ck["route_head"]); rh.eval()


def freegen(model, prompt, max_new=64):
    from localagent.inference.generate import generate
    out, _ = generate(model, tok, f"{USER}{prompt}{ASSISTANT}", max_new_tokens=max_new, temperature=0.0)
    return extract_tool_calls(out)


def eval_freegen(model, samples, label):
    name_ok = full_ok = tot = 0
    for s in samples:
        if s.kind != "tool":
            continue
        tot += 1
        c = freegen(model, s.prompt)
        if c and c[0].name == s.ref_name:
            name_ok += 1
            gold = ToolCall(**(json.loads(s.target) if not s.calls else s.calls[0]))
            full_ok += match_calls([c[0]], [gold])
    print(f"[{label}] FREE-GEN name={name_ok/max(1,tot)*100:.0f}% full={full_ok/max(1,tot)*100:.0f}%"
          f" (n={tot})", flush=True)


# ---- 1. relabel a prompt pool with the grounded teacher; keep teacher-correct (rejection) ----
pool = (paraphrase_samples(2 if QUICK else 30, seed=5, split="train")
        + contextual_samples(1 if QUICK else 20, seed=5, split="train"))
pool = [s for s in pool if s.kind == "tool"]
pool = pool[: 60 if QUICK else 500]
t0 = time.time()
distilled = []
kept = 0
for i, s in enumerate(pool):
    out = hybrid_decode(m, tok, s.prompt, TOOLS, selector=bound, top_m=1, route_head=rh, ptr_head=ptr)
    c = extract_tool_calls(out)
    gold = ToolCall(**(json.loads(s.target) if not s.calls else s.calls[0]))
    if c and match_calls([c[0]], [gold]):       # rejection: only distil the teacher's SUCCESSES
        distilled.append((s.prompt, out))
        kept += 1
    if i % 200 == 0:
        print(f"  relabel {i}/{len(pool)} kept={kept} ({time.time()-t0:.0f}s)", flush=True)
print(f"teacher relabel: kept {kept}/{len(pool)} correct pairs in {(time.time()-t0)/60:.1f} min",
      flush=True)

# eval sets (free-gen, has gold)
para_eval = paraphrase_samples(2, seed=909, split="eval")
ctx_eval = contextual_samples(2, seed=909, split="eval")
print("\n--- BEFORE OPD (free-gen on the scenarios-best backbone) ---", flush=True)
eval_freegen(m, para_eval, "BEFORE para")
eval_freegen(m, ctx_eval, "BEFORE ctx")

# ---- 2. SFT the LM to free-generate the teacher targets (pure LM, mask prompt) ----
rows = []
for prompt, target_body in distilled:
    p = tok.encode(f"{USER}{prompt}{ASSISTANT}")
    b = tok.encode(target_body) + [tok.eos_id]
    rows.append((p + b, [IGNORE] * len(p) + b))
steps = 40 if QUICK else 400
rng = random.Random(0)
m.train()
opt = torch.optim.AdamW(m.parameters(), lr=5e-4, betas=(0.9, 0.95))
t0 = time.time()
for step in range(steps):
    set_lr(opt, cosine_lr(step, steps, 5e-4, 20, 0.1))
    batch = [rows[rng.randrange(len(rows))] for _ in range(16)]
    x, y = pad_batch(batch, tok.pad_id, "cpu")
    _, loss = m(x, targets=y)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    if step % max(1, steps // 8) == 0:
        print(f"  [opd-sft] step {step}/{steps} loss {loss.item():.3f} "
              f"({(time.time()-t0)/max(1,step+1)*1000:.0f}ms/step)", flush=True)
m.eval()

print("\n--- AFTER OPD (free-gen, same model, no heads) ---", flush=True)
eval_freegen(m, para_eval, "AFTER para")
eval_freegen(m, ctx_eval, "AFTER ctx")
torch.save({"cfg": cfg.__dict__, "state_dict": m.state_dict(), "ptr_head": ck["ptr_head"],
            "tool_head": ck.get("tool_head"), "dense_selector": ck["dense_selector"],
            "route_head": ck["route_head"], "selector_proj": ck["selector_proj"],
            "examples": examples}, "runs/tiny-30m-opd.pt")
print("SAVED runs/tiny-30m-opd.pt\nOPD_DONE", flush=True)
