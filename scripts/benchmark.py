#!/usr/bin/env python
"""Throughput + memory benchmark across the three tiers, with KV-cache vs no-cache decode.

Measures, per tier (ultra-tiny / tiny / small):
  - prefill tokens/sec
  - decode tokens/sec WITH the KV cache (prefill-then-decode)
  - decode tokens/sec WITHOUT a cache (recompute the prefix each step) -> shows the KV-cache win
  - parameter memory (MB) and KV-cache memory (MB) at a 256-token context
  - peak process RSS (MB)

Outputs (runs/bench/): throughput.png, memory.png, metrics.json
Usage:  python scripts/benchmark.py [--prompt-len 64] [--decode 64]
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time

import torch

from localagent.model import LocalAgentLM, ModelConfig
from localagent.train.device import resolve_device

TIERS = ["ultra-tiny-1m", "tiny-30m", "small-90m"]
OUT = "runs/bench"


def peak_rss_mb() -> float:
    # ru_maxrss is KB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


@torch.no_grad()
def bench_tier(name, prompt_len, decode, device):
    cfg = ModelConfig.from_yaml(f"configs/model/{name}.yaml")
    model = LocalAgentLM(cfg).to(device).eval()
    V = cfg.vocab_size
    prompt = torch.randint(0, V, (1, prompt_len), device=device)

    # warmup
    model(prompt)

    # prefill
    t0 = time.perf_counter()
    logits, _, caches = model(prompt, pos=0, caches=[None] * model.n_cache_slots())
    prefill_s = time.perf_counter() - t0

    # cached decode
    pos = prompt_len
    nxt = torch.randint(0, V, (1, 1), device=device)
    t1 = time.perf_counter()
    for _ in range(decode):
        logits, _, caches = model(nxt, pos=pos, caches=caches)
        nxt = logits[:, -1:].argmax(-1)
        pos += 1
    cached_s = time.perf_counter() - t1

    # uncached decode (recompute the whole growing prefix each step)
    seq = prompt.clone()
    t2 = time.perf_counter()
    for _ in range(decode):
        logits, _ = model(seq)
        nxt = logits[:, -1:].argmax(-1)
        seq = torch.cat([seq, nxt], dim=1)
    uncached_s = time.perf_counter() - t2

    param_bytes = model.num_params() * 4  # fp32
    ctx = 256
    kv_bytes = 2 * model.n_cache_slots() * cfg.n_kv_heads * cfg.head_dim * ctx * 4
    return {
        "params_M": round(model.num_params() / 1e6, 3),
        "prefill_tok_s": round(prompt_len / prefill_s, 1),
        "decode_cached_tok_s": round(decode / cached_s, 1),
        "decode_uncached_tok_s": round(decode / uncached_s, 1),
        "kv_cache_speedup": round((decode / cached_s) / (decode / uncached_s), 2),
        "param_mb": round(param_bytes / 1e6, 2),
        "kv_cache_mb_at_256": round(kv_bytes / 1e6, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--decode", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = resolve_device("auto")

    results = {}
    for name in TIERS:
        r = bench_tier(name, args.prompt_len, args.decode, device)
        results[name] = r
        print(f"{name:14s} {r['params_M']:>6.3f}M  prefill {r['prefill_tok_s']:>7.1f} tok/s  "
              f"decode(cache) {r['decode_cached_tok_s']:>6.1f}  decode(no-cache) "
              f"{r['decode_uncached_tok_s']:>6.1f}  KV speedup x{r['kv_cache_speedup']}  "
              f"params {r['param_mb']}MB")
    results["_peak_rss_mb"] = round(peak_rss_mb(), 1)
    results["_device"] = str(device)
    json.dump(results, open(f"{OUT}/metrics.json", "w"), indent=2)
    _plot(results)
    print(f"\nDevice {device}, peak RSS {results['_peak_rss_mb']} MB. "
          f"Charts in {OUT}/ (throughput.png, memory.png)")


def _plot(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:  # pragma: no cover
        print(f"(matplotlib unavailable: {e} — skipping plots)")
        return
    tiers = [t for t in TIERS]
    x = np.arange(len(tiers))
    w = 0.25

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w, [results[t]["prefill_tok_s"] for t in tiers], w, label="prefill")
    ax.bar(x, [results[t]["decode_cached_tok_s"] for t in tiers], w, label="decode (KV cache)")
    ax.bar(x + w, [results[t]["decode_uncached_tok_s"] for t in tiers], w, label="decode (no cache)")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([f"{t}\n{results[t]['params_M']}M" for t in tiers])
    ax.set_ylabel("tokens / sec (log)")
    ax.set_title(f"Throughput by tier ({results['_device']}) — KV cache vs recompute")
    for i, t in enumerate(tiers):
        ax.text(i, results[t]["decode_cached_tok_s"], f"x{results[t]['kv_cache_speedup']}",
                ha="center", va="bottom", fontsize=8)
    ax.grid(alpha=0.3, axis="y"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/throughput.png", dpi=120)

    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    ax2.bar(x - w / 2, [results[t]["param_mb"] for t in tiers], w, label="params (fp32)")
    ax2.bar(x + w / 2, [results[t]["kv_cache_mb_at_256"] for t in tiers], w,
            label="KV cache @256 ctx")
    ax2.set_xticks(x); ax2.set_xticklabels([f"{t}\n{results[t]['params_M']}M" for t in tiers])
    ax2.set_ylabel("memory (MB)")
    ax2.set_title("Memory footprint by tier")
    ax2.grid(alpha=0.3, axis="y"); ax2.legend()
    fig2.tight_layout(); fig2.savefig(f"{OUT}/memory.png", dpi=120)


if __name__ == "__main__":
    main()
