#!/usr/bin/env python
"""Load public EVAL BENCHMARKS as TRAINING data and SFT the from-scratch model on them.

⚠️  CONTAMINATION WARNING (read this): LiveCodeBench, GPQA-Diamond, AIME, BigCodeBench and MTBench
are *evaluation* benchmarks. This script trains ON them at the user's explicit request, so any score
you later get on these same benchmarks is **contaminated** and must NOT be reported as an honest
held-out number. Use it to study the training process, not to claim benchmark results.

What actually becomes SFT data (see src/localagent/data/hf_datasets.py for the honest details):
  AIME          Problem            -> Answer (or full Solution)
  BigCodeBench  instruct_prompt    -> code_prompt + canonical_solution
  MTBench       turn-1 prompt      -> reference answer   (open-ended judge rows have none -> skipped)
  GPQA-Diamond  Question + options -> correct letter     (GATED: needs HF_TOKEN + accepted terms)
  LiveCodeBench (no public reference solutions + script loader removed in datasets>=4 -> skipped)

Rich logging: per-source load counts + an example each, per-step SFT loss (--log-every), elapsed +
tok/s, and a JSONL trace + loss PNG under runs/benchmarks/.

  python scripts/train_benchmarks.py --quick                 # fast smoke (small caps, few steps)
  python scripts/train_benchmarks.py --size tiny --per-source 300 --sft-steps 3000
  HF_TOKEN=hf_xxx python scripts/train_benchmarks.py --which aime,gpqa   # include gated GPQA
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from localagent.data.hf_datasets import benchmark_sft_samples
from localagent.data.render import build_pretrain_stream, render_sft
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from localagent.train.device import enable_tf32, resolve_device, resolve_dtype
from localagent.train.pretrain import pretrain
from localagent.train.sft import sft

SIZES = {
    "ultra-tiny": "configs/model/ultra-tiny-1m.yaml",
    "tiny": "configs/model/tiny-30m-byte.yaml",
    "small": "configs/model/small-90m.yaml",
}
OUT = "runs/benchmarks"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", choices=SIZES, default="tiny")
    ap.add_argument("--which", default="all",
                    help="comma list of: aime,bigcodebench,mtbench,gpqa,livecodebench (or 'all')")
    ap.add_argument("--per-source", type=int, default=300, help="max rows per benchmark")
    ap.add_argument("--pretrain-steps", type=int, default=300, help="byte-LM warmup on the rows (0=skip)")
    ap.add_argument("--sft-steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--log-every", type=int, default=20, help="print loss every N steps")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    if args.quick:
        args.per_source, args.pretrain_steps, args.sft_steps = 20, 30, 40
        args.batch, args.seq_len, args.log_every = 8, 128, 5

    os.makedirs(args.out, exist_ok=True)
    logfile = open(f"{args.out}/run.log", "w")
    t_start = time.time()

    def log(*a):
        msg = " ".join(str(x) for x in a)
        line = f"[{time.time()-t_start:7.1f}s] {msg}"
        print(line, flush=True)
        logfile.write(line + "\n")
        logfile.flush()

    enable_tf32()
    device = resolve_device(args.device)
    amp = not args.no_amp and device.type != "cpu"
    dtype = resolve_dtype(device, args.dtype) if amp else torch.float32

    cfg = ModelConfig.from_yaml(SIZES[args.size])
    tok = load_tokenizer("byte" if cfg.vocab_size == 256 else "bpe")
    model = LocalAgentLM(cfg).to(device)
    fwd = torch.compile(model) if args.compile else model
    seq_len = min(args.seq_len, cfg.max_seq_len)

    log("=" * 78)
    log("⚠️  CONTAMINATION: training ON eval benchmarks — scores on them are NOT honest held-out.")
    log(f"device={device} dtype={str(dtype).split('.')[-1]} amp={amp} compile={args.compile}")
    log(f"model={cfg.name} params~{model.num_params()/1e6:.1f}M  seq_len={seq_len}")
    log("=" * 78)

    # ---- 1. LOAD BENCHMARKS AS TRAINING DATA ----
    which = list(benchmark_which(args.which))
    log(f"loading benchmarks as TRAINING data: {which}  (per_source={args.per_source})")
    rows, counts = benchmark_sft_samples(tok, which=which, per_source=args.per_source, log=log)
    if not rows:
        log("no SFT rows loaded (all sources unavailable?) — nothing to train on. exiting.")
        return

    # one example per source, so you can eyeball what is being learned
    seen = set()
    for r in rows:
        if r.category in seen:
            continue
        seen.add(r.category)
        log(f"  e.g. [{r.category}] PROMPT: {r.prompt[:90]!r}")
        log(f"           -> TARGET: {r.target[:90]!r}")

    # drop rows whose rendered length overflows the context window (real prompts can be long)
    fit = [s for s in rows if len(render_sft(s, tok)[0]) <= cfg.max_seq_len]
    log(f"{len(fit)}/{len(rows)} rows fit the {cfg.max_seq_len}-token context (dropped {len(rows)-len(fit)})")
    if not fit:
        log("no rows fit the context window. exiting.")
        return
    json.dump({"counts": counts, "fit": len(fit), "config": cfg.name},
              open(f"{args.out}/sources.json", "w"), indent=2)

    # ---- 2. (optional) BYTE-LM WARMUP so a from-scratch model isn't starting from noise ----
    pre_hist = []
    if args.pretrain_steps > 0:
        log(f"\n=== PRETRAIN warmup ({args.pretrain_steps} steps, next-byte LM on the rows) ===")
        stream = build_pretrain_stream(fit, tok)
        pre_hist = pretrain(fwd, stream, tok, steps=args.pretrain_steps, batch_size=args.batch,
                            seq_len=seq_len, device=device, amp=amp, amp_dtype=args.dtype,
                            log=log, log_every=args.log_every)

    # ---- 3. SFT on the benchmark rows ----
    log(f"\n=== SFT ({args.sft_steps} steps on {len(fit)} benchmark rows) ===")
    tic = time.time()
    sft_hist, _, _ = sft(fwd, fit, tok, steps=args.sft_steps, batch_size=args.batch, device=device,
                         joint_tool_head=False, amp=amp, amp_dtype=args.dtype, log=log,
                         log_every=args.log_every)
    toks = args.sft_steps * args.batch * seq_len
    log(f"SFT done: last loss {sft_hist[-1]:.3f}  (~{toks/(time.time()-tic)/1e3:.1f}k tok/s)")

    # ---- 4. PERSIST checkpoint + traces ----
    ckpt = f"{args.out}/{cfg.name}-benchmarks.pt"
    torch.save({"cfg": cfg.__dict__, "state_dict": model.state_dict(),
                "sources": counts, "contaminated": True}, ckpt)
    json.dump({"pretrain_loss": pre_hist, "sft_loss": sft_hist, "sources": counts},
              open(f"{args.out}/trainlog.jsonl", "w"))
    _plot_loss(pre_hist, sft_hist, args.out, log)
    log(f"\nDONE in {(time.time()-t_start)/60:.1f} min -> {ckpt}")
    log("reminder: do NOT report scores on these 5 benchmarks as honest eval — the model trained on them.")
    logfile.close()


def benchmark_which(spec: str):
    from localagent.data.hf_datasets import BENCH_LOADERS
    if spec.strip().lower() == "all":
        return list(BENCH_LOADERS)
    return [s.strip() for s in spec.split(",") if s.strip()]


def _plot_loss(pre, sft_hist, out, log) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(pre or [], color="tab:gray")
    axes[0].set_title("pretrain warmup (next-byte CE)")
    axes[1].plot(sft_hist, color="tab:blue")
    axes[1].set_title("SFT on benchmarks (masked LM)")
    for ax in axes:
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(f"{out}/loss.png", dpi=120)
    plt.close(fig)
    log(f"  wrote {out}/loss.png")


if __name__ == "__main__":
    main()
