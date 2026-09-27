"""Demo: natural question -> auto tool dispatch (route head -> dense selector -> grounded call).
Loads the improved dispatch checkpoint (runs/tiny-30m-dispatch.pt: paraphrase-trained selector +
example-augmented tool embeddings + route head) if present; else trains inline."""
import torch

from localagent.agent.constrained import hybrid_decode
from localagent.agent.dense_selector import BoundSelector, DenseToolSelector
from localagent.agent.pointer_head import PointerHead
from localagent.agent.routes import ROUTES, RouteHead
from localagent.agent.tool_head import _feat
from localagent.agent.toolset import STANDARD_TOOLS as TOOLS
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer

torch.set_num_threads(4)
tok = load_tokenizer()
ck = torch.load("runs/tiny-30m-50tools.pt", map_location="cpu")
cfg = ModelConfig(**ck["cfg"])
m = LocalAgentLM(cfg); m.load_state_dict(ck["state_dict"]); m.eval()
ptr = PointerHead(cfg.d_model); ptr.load_state_dict(ck["ptr_head"])

dk = torch.load("runs/tiny-30m-dispatch.pt", map_location="cpu")
examples = dk["examples"]
from localagent.agent.dense_selector import tool_embeddings
emb_dim = tool_embeddings(TOOLS[:1]).shape[1]
sel = DenseToolSelector(cfg.d_model, emb_dim=emb_dim, proj=dk["selector_proj"])
sel.load_state_dict(dk["dense_selector"])
selector = BoundSelector(sel, TOOLS, examples=examples)
route_head = RouteHead(cfg.d_model); route_head.load_state_dict(dk["route_head"]); route_head.eval()

QUERIES = [
    "What is the color of a monkey?",
    "Look up who invented the telephone.",
    "Open https://github.com/pytorch/pytorch in the browser",
    "Download the dataset from https://example.com/data.zip",
    "List the files in the src directory",
    "Search the codebase for the function train_step",
    "Make a directory called build",
    "Run the test suite",
    "What is 18 * 24?",
    "Email Dana the quarterly report",
]
print(f"\n{'query':50}{'route':>13}{'selected tool':>16}", flush=True)
print("-" * 95, flush=True)
for q in QUERIES:
    with torch.no_grad():
        feat = _feat(m, tok, q, "cpu")
        pred_route = ROUTES[int(route_head(feat).argmax(-1))]
        top = selector.rank(feat)[0]
    call = hybrid_decode(m, tok, q, TOOLS, selector=selector, top_m=1, ptr_head=ptr)
    print(f"{q[:48]:50}{pred_route:>13}{top:>16}", flush=True)
    print(f"  -> {call}", flush=True)
print("\nDEMO_DONE", flush=True)
