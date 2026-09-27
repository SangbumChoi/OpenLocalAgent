"""Generative in-context tool-calling SFT — makes tool calls *generable*.

The current 50-tool checkpoint can't free-generate a call (0% pure-gen; the 51-way head carries it).
Here we instead render a small *retrieved* tool catalog into the prompt and SFT the LM to free-
generate `<tool_call>{json}</tool_call>` from that context — no tool head, no pointer copy. Adding a
tool becomes one catalog line with zero retraining; a 5-way route head can ride along as a gate.

Warm-starts from runs/tiny-30m-50tools.pt (knows the JSON body shape), adapts it to read the catalog.
`--quick` runs a fast smoke. Saves runs/tiny-30m-incontext.pt.
"""
import random
import sys
import time

import torch

from localagent.agent.incontext import build_candidates, grounded_prompt
from localagent.agent.retriever import ToolRetriever
from localagent.agent.toolset import STANDARD_TOOLS as TOOLS
from localagent.data.agent_synth import Generator
from localagent.data.render import IGNORE, assistant_body
from localagent.eval.harness import evaluate_incontext
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from localagent.train.loop import cosine_lr, pad_batch, set_lr

QUICK = "--quick" in sys.argv
torch.set_num_threads(4)
tok = load_tokenizer()
by_name = {t.name: t for t in TOOLS}
retr = ToolRetriever(TOOLS)

K = 6
n_train = 200 if QUICK else 2400
n_eval = 16 if QUICK else 48
steps = 60 if QUICK else 500
bs = 16

ck = torch.load("runs/tiny-30m-50tools.pt", map_location="cpu")   # tool-body-aware backbone
cfg = ModelConfig(**ck["cfg"])
model = LocalAgentLM(cfg)
model.load_state_dict(ck["state_dict"])

train = Generator(level=3, seed=11).generate_balanced(1)[:n_train] if QUICK \
    else Generator(level=3, seed=11).generate_balanced(3)[:n_train]
held = Generator(level=3, seed=909, split="eval").generate_balanced(1)[:n_eval]
print(f"train={len(train)} held={len(held)} K={K} steps={steps}", flush=True)


def report(model, samples, tag):
    model.eval()
    r = evaluate_incontext(model, samples, tok, TOOLS, k=K, retriever=retr)
    print(f"[{tag}] gen_acc(gold-in)={r['gen_acc']*100:.1f}%  "
          f"gen_acc_e2e={r['gen_acc_e2e']*100:.1f}%  recall@{K}={r['recall_at_k']*100:.1f}%  "
          f"(n={r['n']})", flush=True)
    return r


def render_row(s, idx):
    """(input_ids, labels): in-context catalog + user prompt masked; learn the assistant body+EOS."""
    gold = s.ref_name if s.kind == "tool" else "text"
    cand = build_candidates(s.prompt, gold, retr, by_name, k=K, include_gold=True,
                            rng=random.Random(1000 + idx))
    p = tok.encode(grounded_prompt(s.prompt, cand))
    b = tok.encode(assistant_body(s)) + [tok.eos_id]
    return p + b, [IGNORE] * len(p) + b


t0 = time.time()
rows = [render_row(s, i) for i, s in enumerate(train)]
print(f"rendered {len(rows)} rows in {time.time()-t0:.1f}s "
      f"(avg len {sum(len(r[0]) for r in rows)//len(rows)} bytes)", flush=True)

# baseline: the warm-start backbone read the same in-context prompt, BEFORE adaptation
report(model, held[:16], "BASELINE pre-SFT")

model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0)
rng = random.Random(0)
t0 = time.time()
for step in range(steps):
    set_lr(opt, cosine_lr(step, steps, 1e-3, max(2, steps // 25), 0.1))
    batch = [rows[rng.randrange(len(rows))] for _ in range(bs)]
    x, y = pad_batch(batch, tok.pad_id, "cpu")
    _, loss = model(x, targets=y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % max(1, steps // 10) == 0 or step == steps - 1:
        print(f"  [gen-sft] step {step:4d}/{steps} loss {loss.item():.3f} "
              f"({(time.time()-t0)/max(1,step+1)*1000:.0f}ms/step)", flush=True)

print("", flush=True)
r = report(model, held, "AFTER gen-SFT")
print(f"{'route':14}{'gen':>7}{'e2e':>7}{'n':>6}", flush=True)
for route, v in r["by_route"].items():
    print(f"{route:14}{v['gen']*100:6.0f}%{v['e2e']*100:6.0f}%{v['n']:6d}", flush=True)

torch.save({"cfg": cfg.__dict__, "state_dict": model.state_dict(), "k": K,
            "format": "in-context-catalog"}, "runs/tiny-30m-incontext.pt")
print("\nSAVED runs/tiny-30m-incontext.pt\nGEN_SFT_DONE", flush=True)
