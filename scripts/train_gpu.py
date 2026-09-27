#!/usr/bin/env python
"""Single-GPU from-scratch training: pretrain -> SFT -> GRPO, with mixed precision.

The CPU-friendly path is `scripts/flywheel.py`; this is its GPU sibling. It auto-detects the
device (CUDA / MPS / Intel-XPU, else CPU), turns on **TF32 + AMP** (bf16 on Ampere+, fp16 on older
GPUs such as a Colab T4, or `--dtype` to force one), runs the three from-scratch stages on the
in-repo synthetic agent data, and checkpoints each stage to ``runs/gpu/``.

  python scripts/train_gpu.py --size tiny --quick     # smoke test (runs on CPU too)
  python scripts/train_gpu.py --size tiny             # real run on your GPU (~28M, byte-level)
  python scripts/train_gpu.py --size ultra-tiny       # ~1M router/planner
  python scripts/train_gpu.py --size tiny --no-amp    # force the fp32 path

For real public datasets (FineWeb-edu + Hermes/xLAM) and Hub upload, use `scripts/train_job.py`;
for the ultra-tiny enrichment loop with held-out eval, use `scripts/flywheel.py`.
"""
from __future__ import annotations

import argparse
import os
import time

import torch

from localagent.data.agent_synth import Generator
from localagent.data.render import build_pretrain_stream
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from localagent.train.device import enable_tf32, resolve_device, resolve_dtype
from localagent.train.pretrain import pretrain
from localagent.train.rl import grpo
from localagent.train.sft import sft

SIZES = {
    "ultra-tiny": "configs/model/ultra-tiny-1m.yaml",
    "tiny": "configs/model/tiny-30m-byte.yaml",
    "small": "configs/model/small-90m.yaml",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", choices=SIZES, default="tiny")
    ap.add_argument("--device", default="auto", help="auto|cuda|mps|xpu|cpu")
    ap.add_argument("--dtype", default="auto", help="auto|bf16|fp16|fp32 (autocast dtype)")
    ap.add_argument("--no-amp", action="store_true", help="disable mixed precision (force fp32)")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the dense pretrain/SFT forward (experimental; GPU win)")
    ap.add_argument("--quick", action="store_true", help="tiny smoke run (CPU OK)")
    ap.add_argument("--stages", default="pretrain,sft,grpo")
    ap.add_argument("--out", default="runs/gpu")
    # full-run budgets (--quick shrinks them); override any explicitly.
    ap.add_argument("--pretrain-steps", type=int, default=4000)
    ap.add_argument("--sft-steps", type=int, default=2500)
    ap.add_argument("--grpo-steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--train-samples", type=int, default=4000)
    args = ap.parse_args()

    if args.quick:
        args.pretrain_steps, args.sft_steps, args.grpo_steps = 40, 40, 6
        args.batch, args.seq_len, args.train_samples = 16, 128, 200

    enable_tf32()
    device = resolve_device(args.device)
    amp = not args.no_amp and device.type != "cpu"          # AMP only helps on accelerators
    dtype = resolve_dtype(device, args.dtype) if amp else torch.float32
    stages = set(args.stages.split(","))
    os.makedirs(args.out, exist_ok=True)

    cfg = ModelConfig.from_yaml(SIZES[args.size])
    tok = load_tokenizer("byte" if cfg.vocab_size == 256 else "bpe")
    model = LocalAgentLM(cfg).to(device)                    # raw handle: used for save / KV-cache RL
    # torch.compile only wraps the dense pretrain/SFT forward; it shares params with `model`, so the
    # raw module still owns the weights (save + GRPO rollouts run on `model` to avoid KV recompiles).
    fwd = torch.compile(model) if args.compile else model
    seq_len = min(args.seq_len, cfg.max_seq_len)
    print(f"device={device} dtype={str(dtype).split('.')[-1]} amp={amp} compile={args.compile} | "
          f"cfg={cfg.name} params~{model.num_params()/1e6:.1f}M | stages={sorted(stages)}", flush=True)

    def save(tag: str) -> str:
        p = f"{args.out}/{cfg.name}-{tag}.pt"
        torch.save({"cfg": cfg.__dict__, "state_dict": model.state_dict()}, p)
        print(f"  saved {p}", flush=True)
        return p

    samples = Generator(level=3, seed=7).generate_balanced(args.train_samples)
    t0 = time.time()

    if "pretrain" in stages:
        print("\n=== PRETRAIN (next-byte LM) ===", flush=True)
        stream = build_pretrain_stream(samples, tok)
        tic = time.time()
        pretrain(fwd, stream, tok, steps=args.pretrain_steps, batch_size=args.batch,
                 seq_len=seq_len, device=device, lr_schedule="wsd", amp=amp, amp_dtype=args.dtype)
        toks = args.pretrain_steps * args.batch * seq_len
        print(f"  ~{toks/(time.time()-tic)/1e3:.1f}k tok/s", flush=True)
        save("pretrain")

    if "sft" in stages:
        print("\n=== SFT (tool-call instruction tuning + heads) ===", flush=True)
        sft(fwd, samples, tok, steps=args.sft_steps, batch_size=max(8, args.batch // 2),
            device=device, joint_tool_head=True, amp=amp, amp_dtype=args.dtype)
        save("sft")

    if "grpo" in stages:
        print("\n=== GRPO (verifiable tool-call reward) ===", flush=True)
        rl_data = [s for s in samples if s.kind == "tool"]
        grpo(model, rl_data, tok, steps=args.grpo_steps, device=device,
             prompts_per_step=4 if args.quick else 8, amp=amp, amp_dtype=args.dtype)
        save("grpo")

    print(f"\nDONE in {(time.time()-t0)/60:.1f} min -> {save('final')}", flush=True)


if __name__ == "__main__":
    main()
