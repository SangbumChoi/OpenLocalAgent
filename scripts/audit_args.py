"""Argument-copy audit — isolate selection vs argument-value accuracy on held sets.

For samples where the selector picks the RIGHT tool, what fraction also get the argument VALUES right
(full AST match)? This separates the two sub-skills so we know whether selection or arg-copy is the
end-to-end bottleneck. Runs on a given dispatch checkpoint.

  python scripts/audit_args.py [--ckpt runs/tiny-30m-dispatch-long.pt]
"""
import argparse
import json

import torch

from localagent.agent.constrained import hybrid_decode
from localagent.agent.dense_selector import BoundSelector, DenseToolSelector, tool_embeddings
from localagent.agent.parser import extract_tool_calls
from localagent.agent.pointer_head import PointerHead
from localagent.agent.tool_head import _feat
from localagent.agent.toolset import STANDARD_TOOLS as TOOLS
from localagent.data.contextual import contextual_samples
from localagent.data.paraphrase import paraphrase_samples
from localagent.data.schema import ToolCall
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


def audit(samples, label):
    sel_ok = both_ok = args_given_sel = 0
    for s in samples:
        if s.kind != "tool":
            continue
        out = hybrid_decode(m, tok, s.prompt, TOOLS, selector=bound, top_m=1, ptr_head=ptr)
        calls = extract_tool_calls(out)
        if calls and calls[0].name == s.ref_name:
            sel_ok += 1
            gold = ToolCall(**json.loads(s.target)) if not s.calls else ToolCall(**s.calls[0])
            if match_calls([calls[0]], [gold]):
                both_ok += 1
                args_given_sel += 1
            # args_given_sel counts correct-args among correct-selection
    tot = sum(1 for s in samples if s.kind == "tool")
    arg_acc = args_given_sel / max(1, sel_ok)
    print(f"[{label}] tool_n={tot}  selection={sel_ok/max(1,tot)*100:.0f}%  "
          f"full(sel+args)={both_ok/max(1,tot)*100:.0f}%  args|correct-sel={arg_acc*100:.0f}%",
          flush=True)


audit(paraphrase_samples(2, seed=909, split="eval"), "paraphrase-eval")
audit(contextual_samples(2, seed=909, split="eval"), "contextual-eval")
print("AUDIT_DONE", flush=True)
