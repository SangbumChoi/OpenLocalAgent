#!/usr/bin/env python
"""Which regions of a public pretrained agent are worth loading, and which data teaches control?

One student architecture (the donor's own `tiny-30m-byte`), one SFT recipe, two knobs:

  --region  which parameter regions are copied from the public donor before training
  --data    which corpora the SFT stage sees

Every arm is scored on the same suite: generative tool calling and multi-turn trajectories on the
repository's held-out synthetic split, plus teacher-forced assistant accuracy on each public
held-out set (ToolACE function calls, Mind2Web browser control, AndroidControl mobile control,
AgentNet desktop control).

  python scripts/region_transfer.py --region attn --data union --out runs/region/attn-union
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from safetensors.torch import load_file

from localagent.agent.pointer_head import PointerHead
from localagent.agent.tool_head import ToolHead
from localagent.agent.toolset import STANDARD_TOOLS as TOOLS
from localagent.data.agent_synth import Generator
from localagent.eval.harness import evaluate, evaluate_grounded, multi_turn_eval
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from localagent.train.sft import _evaluate_conversations, sft
from localagent.train.stage_data import read_conversations

DONOR_DIR = Path("data/hf-campaign/localagent_tiny_model")
PUBLIC = Path("data/public")

PUBLIC_TRAIN = {
    "toolace": PUBLIC / "toolace-train.jsonl",
    "mind2web": PUBLIC / "mind2web-train.jsonl",
    "androidcontrol": PUBLIC / "androidcontrol-train.jsonl",
}
PUBLIC_EVAL = {
    "toolace": PUBLIC / "toolace-eval.jsonl",
    "androidcontrol": PUBLIC / "androidcontrol-test.jsonl",
    "agentnet": PUBLIC / "agentnet-eval.jsonl",
}


def region_keys(region: str, donor: dict[str, torch.Tensor], n_layers: int) -> list[str]:
    """Donor keys to copy for a named region. `blocks.N.` prefixes carry the layer index."""

    def layer_of(key: str) -> int | None:
        parts = key.split(".")
        return int(parts[1]) if parts[0] == "blocks" else None

    half = n_layers // 2
    rules = {
        "scratch": lambda key: False,
        "embed": lambda key: key.startswith("embed"),
        "norms": lambda key: "norm" in key,
        "attn": lambda key: ".attn." in key,
        "ffn": lambda key: ".ffn." in key,
        "early": lambda key: (index := layer_of(key)) is not None and index < half,
        "late": lambda key: (index := layer_of(key)) is not None and index >= half,
        "attn_embed": lambda key: ".attn." in key or key.startswith("embed"),
        "no_embed": lambda key: not key.startswith("embed"),
        "full": lambda key: True,
        "full_heads": lambda key: True,
    }
    if region not in rules:
        raise ValueError(f"unknown region {region!r}")
    return [key for key in donor if rules[region](key)]


def load_regions(model: LocalAgentLM, region: str) -> dict[str, object]:
    donor = load_file(str(DONOR_DIR / "model.safetensors"))
    keys = region_keys(region, donor, model.cfg.n_layers)
    state = model.state_dict()
    copied, copied_params = [], 0
    for key in keys:
        if key in state and state[key].shape == donor[key].shape:
            state[key] = donor[key].clone()
            copied.append(key)
            copied_params += donor[key].numel()
    model.load_state_dict(state)
    total = sum(tensor.numel() for tensor in state.values())
    return {
        "region": region,
        "tensors_copied": len(copied),
        "params_copied": copied_params,
        "params_total": total,
        "fraction_copied": copied_params / total,
    }


def load_donor_heads(cfg, device) -> tuple[ToolHead | None, PointerHead | None]:
    payload = json.loads((DONOR_DIR / "heads.json").read_text())
    tool_head = ptr_head = None
    raw_tool = payload.get("tool_head")
    if raw_tool:
        weight = torch.tensor(raw_tool["weight"])
        tool_head = ToolHead(cfg.d_model, classes=[f"c{i}" for i in range(weight.shape[0])])
        tool_state = {"fc.weight": weight}
        if "bias" in raw_tool:
            tool_state["fc.bias"] = torch.tensor(raw_tool["bias"])
        tool_head.load_state_dict(tool_state, strict=False)
        tool_head = tool_head.to(device)
    return tool_head, ptr_head


def build_training_data(data: str, n_synth: int, n_public: int, seed: int):
    samples, conversations, used = [], [], []
    if data in ("synthetic", "union"):
        samples = Generator(level=3, seed=seed, split="train").generate(n_synth)
        conversations += list(Generator(level=3, seed=5000 + seed, split="train").episodes(120))
        used.append("synthetic")
    for name, path in PUBLIC_TRAIN.items():
        if data in (name, "union") and path.exists():
            rows = read_conversations(path)
            random.Random(seed).shuffle(rows)
            conversations += rows[:n_public]
            used.append(f"{name}:{min(len(rows), n_public)}")
    return samples, conversations, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="full")
    ap.add_argument("--data", default="union")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-synth", type=int, default=2500)
    ap.add_argument("--n-public", type=int, default=4000)
    ap.add_argument("--mt-weight", type=float, default=1.0,
                    help="weight on the multi-turn head; >1 protects trajectory ability")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    tok = load_tokenizer("byte")
    cfg = ModelConfig.from_yaml("configs/model/tiny-30m-byte.yaml")
    model = LocalAgentLM(cfg).to(device)
    transfer = load_regions(model, args.region)
    print(f"[{args.region}] copied {transfer['tensors_copied']} tensors "
          f"({transfer['fraction_copied']*100:.1f}% of parameters)", flush=True)

    init_tool_head = None
    if args.region == "full_heads":
        donor_tool, _ = load_donor_heads(cfg, device)
        if donor_tool is not None and donor_tool.fc.weight.shape[0] == len(ToolHead(cfg.d_model).classes):
            init_tool_head = donor_tool.state_dict()

    samples, conversations, used = build_training_data(args.data, args.n_synth, args.n_public,
                                                       args.seed)
    print(f"[data:{args.data}] {len(samples)} single-turn + {len(conversations)} conversations "
          f"({', '.join(used)})", flush=True)

    if not samples:
        samples = Generator(level=3, seed=args.seed, split="train").generate(64)

    _, head, ptr = sft(model, samples, tok, steps=args.steps, batch_size=args.batch, lr=args.lr,
                       device=device, log=lambda *a: None, joint_tool_head=True,
                       conversations=conversations, seed=args.seed,
                       init_tool_head=init_tool_head, mt_weight=args.mt_weight)

    held = Generator(level=3, seed=1003, split="eval").generate_balanced(30)
    held_ep = Generator(level=3, seed=6003, split="eval").episodes(30)
    report = {
        "config": vars(args),
        "transfer": transfer,
        "data_used": used,
        "train_rows": {"single_turn": len(samples), "conversations": len(conversations)},
        "synthetic_grounded": evaluate_grounded(model, held, tok, TOOLS, device=device,
                                                tool_head=head, ptr_head=ptr),
        "synthetic_freegen": evaluate(model, held, tok, device=device),
        "synthetic_multi_turn": multi_turn_eval(model, held_ep, tok, TOOLS, device=device,
                                                tool_head=head, ptr_head=ptr),
        "public_eval": {},
    }
    for name, path in PUBLIC_EVAL.items():
        if not path.exists():
            continue
        rows = read_conversations(path)[:400]
        report["public_eval"][name] = _evaluate_conversations(
            model, rows, tok, max_seq_len=min(1024, cfg.max_seq_len), batch_size=8,
            device=str(device))
        accuracy = report["public_eval"][name]["assistant_token_accuracy"]
        print(f"  {name}: token_acc={accuracy*100:.1f}% "
              f"seq_acc={report['public_eval'][name]['assistant_sequence_accuracy']*100:.1f}%",
              flush=True)

    print(f"synthetic grounded={report['synthetic_grounded']['overall']*100:.1f}% "
          f"multi-turn={report['synthetic_multi_turn']['step_acc']*100:.1f}%", flush=True)
    json.dump(report, open(out / "report.json", "w"), indent=2)
    torch.save({"cfg": cfg.__dict__, "state_dict": model.state_dict(),
                "tool_head": head.state_dict() if head is not None else None,
                "ptr_head": ptr.state_dict() if ptr is not None else None}, out / "model.pt")
    print("REGION_TRANSFER_DONE " + str(out), flush=True)


if __name__ == "__main__":
    main()
