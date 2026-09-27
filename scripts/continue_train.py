"""Continue-train the 28M on the augmented dataset (new implicit-question->web_search +
disambiguation coverage), warm-starting backbone AND heads from the published checkpoint so we
keep the learned selection/grounding and adapt to the new patterns. Saves an updated checkpoint
and reports before/after accuracy on the lookup categories that were failing."""
import time
import torch

from localagent.agent.pointer_head import PointerHead
from localagent.agent.tool_head import ToolHead
from localagent.agent.toolset import STANDARD_TOOLS as TOOLS
from localagent.data.agent_synth import REALISTIC_WEIGHTS, Generator
from localagent.eval.harness import evaluate_grounded
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from localagent.train.sft import sft

torch.set_num_threads(4)
tok = load_tokenizer()
ck = torch.load("runs/tiny-30m-byte-best.pt", map_location="cpu")
cfg = ModelConfig(**ck["cfg"])
LOOK = ["web_search", "define", "get_news", "calculator"]
held = Generator(level=3, seed=909, split="eval").generate_balanced(32)


def build():
    m = LocalAgentLM(cfg)
    m.load_state_dict(ck["state_dict"])
    return m


def report(model, head, ptr, tag):
    r = evaluate_grounded(model, held, tok, TOOLS, device="cpu", tool_head=head, ptr_head=ptr)
    c = r["categories"]
    print(f"[{tag}] overall={r['overall']*100:.1f}% | "
          + " ".join(f"{k}={c.get(k, 0)*100:.0f}%" for k in LOOK), flush=True)
    return r


# BEFORE — the published heads
h0 = ToolHead(cfg.d_model); h0.load_state_dict(ck["tool_head"])
p0 = PointerHead(cfg.d_model); p0.load_state_dict(ck["ptr_head"])
report(build(), h0, p0, "BEFORE")

# CONTINUE-TRAIN — warm backbone + warm heads. Use the NATURAL augmented mix (the new
# web_search/define coverage is already ~2.8x there) so no category is starved; gentler lr
# to adapt without drifting the backbone (the x2 up-weight + lr 8e-4 regressed overall).
train = Generator(level=3, seed=7, split="train").generate(8000)
t0 = time.time()
model = build()
_, head, ptr = sft(model, train, tok, steps=160, batch_size=8, lr=4e-4, device="cpu",
                   joint_tool_head=True, log=print,
                   init_tool_head=ck["tool_head"], init_ptr_head=ck["ptr_head"])
print(f"train done in {(time.time()-t0)/60:.1f} min", flush=True)
report(model, head, ptr, "AFTER")
torch.save({"cfg": cfg.__dict__, "state_dict": model.state_dict(),
            "tool_head": head.state_dict(), "ptr_head": ptr.state_dict()},
           "runs/tiny-30m-byte-updated.pt")
print("SAVED runs/tiny-30m-byte-updated.pt", flush=True)
print("CONTINUE_TRAIN_DONE", flush=True)
