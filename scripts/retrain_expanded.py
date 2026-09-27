"""Retrain on the expanded 50-tool space (computer-use + modern agentic tools). Warm-start the
backbone from the 28M checkpoint, but the tool head is FRESH (CLASSES grew 22->51, so the old
22-class head can't transfer). Reports overall + computer-use accuracy and spot-checks the exact
computer-use prompts."""
import time
import torch

from localagent.agent.tool_head import CLASSES
from localagent.agent.toolset import STANDARD_TOOLS as TOOLS
from localagent.data.agent_synth import Generator
from localagent.eval.harness import evaluate_grounded
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from localagent.train.sft import sft

torch.set_num_threads(4)
tok = load_tokenizer()
ck = torch.load("runs/tiny-30m-byte-best.pt", map_location="cpu")
cfg = ModelConfig(**ck["cfg"])
CU = ["screenshot", "click", "double_click", "type_text", "key_press",
      "scroll", "drag", "wait", "move_cursor", "open_app"]
held = Generator(level=3, seed=909, split="eval").generate_balanced(80)


def predict(m, h, p):
    ids = tok.encode("<|user|>" + p + "<|assistant|>")
    with torch.no_grad():
        out = m(torch.tensor([ids]), return_hidden=True)
        hid = out[1] if isinstance(out, (tuple, list)) else out
        return CLASSES[int(h(hid[:, -1]).argmax(-1))]


# warm backbone, FRESH heads (sft builds a 51-class tool head from CLASSES)
model = LocalAgentLM(cfg)
model.load_state_dict(ck["state_dict"])
train = Generator(level=3, seed=7, split="train").generate(18000)
t0 = time.time()
_, head, ptr = sft(model, train, tok, steps=1500, batch_size=8, lr=1e-3, device="cpu",
                   joint_tool_head=True, log=print)
print(f"train {(time.time()-t0)/60:.1f} min", flush=True)

r = evaluate_grounded(model, held, tok, TOOLS, device="cpu", tool_head=head, ptr_head=ptr)
c = r["categories"]
cu = [c[t] for t in CU if t in c]
print(f"[AFTER] overall={r['overall']*100:.1f}% | "
      f"computer-use avg={sum(cu)/max(1, len(cu))*100:.0f}% ({len(cu)} CU tools in eval)", flush=True)
print("=== computer-use spot-check ===", flush=True)
for p in ['Take a screenshot.', 'Click "the Submit button".', 'Type "hello world".',
          'Press the Enter key.', 'Scroll down.', 'Run "math.sqrt(2)" in Python.',
          'Make an HTTP request to openai.com.', 'How tall is Mount Everest?']:
    print(f"  {p:40} -> {predict(model, head, p)}", flush=True)

torch.save({"cfg": cfg.__dict__, "state_dict": model.state_dict(),
            "tool_head": head.state_dict(), "ptr_head": ptr.state_dict()},
           "runs/tiny-30m-50tools.pt")
print("SAVED runs/tiny-30m-50tools.pt", flush=True)
print("RETRAIN_DONE", flush=True)
