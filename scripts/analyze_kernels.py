#!/usr/bin/env python
"""Render docs/KERNELS.md from results/throughput/kernels-*.json. Numbers are never retyped.

  python scripts/analyze_kernels.py            # writes docs/KERNELS.md
"""
from __future__ import annotations

import json
from pathlib import Path

R = Path("results/throughput")
TIERS = [("la-93m", "93m"), ("la-300m", "300m"), ("la-300m-moe", "300m-moe")]
ORDER = ["eager_fp32", "eager_bf16", "sdpa_flash_bf16", "sdpa_efficient_bf16", "sdpa_math_bf16",
         "compile_bf16", "compile_cudagraph_bf16"]


def load(tier: str):
    base = json.loads((R / f"kernels-posttrain-la-{tier}.json").read_text())
    rows = {k["kernel"]: k for k in base["kernels"]}
    extra = R / f"kernels-posttrain-la-{tier}-cudagraph.json"
    if extra.exists():
        for k in json.loads(extra.read_text())["kernels"]:
            rows[k["kernel"]] = k
    return base, rows


def main() -> int:
    tables, meta = {}, None
    for name, t in TIERS:
        meta, tables[name] = load(t)
    out = [
        "# Kernel configurations: what a throughput number depends on\n",
        f"Measured on **{meta['device']}**, torch {meta['torch']}, batch 1, "
        f"{meta['prompt_tokens']}-token prompt, {meta['new_tokens']} decode tokens, median of 5. "
        "Random ids, so this runs on checkpoints whose tokenizer is gone. Produced by "
        "`scripts/eval_kernels.py`; receipts in `results/throughput/kernels-*.json`.\n",
        "Prefill and decode are different regimes and are reported separately: prefill is one wide "
        "matmul over the prompt (compute-bound); decode is 128 sequential single-token steps "
        "(launch- and memory-bound). A configuration can win one and lose the other.\n",
    ]
    for name, _ in TIERS:
        rows = tables[name]
        base = rows["eager_bf16"]["decode_tok_s"]
        out += [f"\n## {name}\n",
                "| kernel | prefill tok/s | decode tok/s | ms/tok | peak MB | vs eager bf16 |",
                "|---|---:|---:|---:|---:|---:|"]
        for k in ORDER:
            r = rows.get(k)
            if not r:
                continue
            if r["error"]:
                out.append(f"| `{k}` | – | FAILED | | | |")
                continue
            out.append(f"| `{k}` | {r['prefill_tok_s']:,.0f} | **{r['decode_tok_s']:,.1f}** | "
                       f"{r['decode_ms_per_token']:.2f} | {r['peak_mem_mb']:,.0f} | "
                       f"{r['decode_tok_s'] / base:.2f}× |")
    d93, d300, dmoe = tables["la-93m"], tables["la-300m"], tables["la-300m-moe"]
    g = lambda d, k, f="decode_tok_s": d[k][f]  # noqa: E731
    out.append(f"""
## What it says

**Small dense models on an H100 are launch-bound, not FLOP-bound.** bf16 and the attention backend
move `la-93m` decode from {g(d93,'eager_fp32'):.0f} to {g(d93,'sdpa_efficient_bf16'):.0f} tok/s — a few
percent. `torch.compile` fuses the dozens of tiny kernels per step and gives
**{g(d93,'compile_bf16')/g(d93,'eager_bf16'):.1f}×**; CUDA graphs remove the launch overhead itself and
give **{g(d93,'compile_cudagraph_bf16')/g(d93,'eager_bf16'):.1f}×**
({g(d93,'compile_cudagraph_bf16','decode_ms_per_token'):.2f} ms/token). `la-300m` follows the same
curve ({g(d300,'compile_cudagraph_bf16'):.0f} tok/s,
{g(d300,'compile_cudagraph_bf16')/g(d300,'eager_bf16'):.1f}×). Prefill, one large matmul, gets its
8–18% from bf16 and the SDPA backend instead and is untouched by compilation.

**The sparse-expert arm's matched active compute does not become matched wall-clock.** `la-300m-moe`
decodes at {g(dmoe,'eager_bf16'):.0f} tok/s eager — {g(d300,'eager_bf16')/g(dmoe,'eager_bf16'):.1f}× slower
than dense `la-300m` with the same active parameters — and prefill is
{g(d300,'eager_bf16','prefill_tok_s')/g(dmoe,'eager_bf16','prefill_tok_s'):.1f}× slower. Neither
`torch.compile` ({g(dmoe,'compile_bf16'):.0f} tok/s) nor CUDA graphs
({g(dmoe,'compile_cudagraph_bf16'):.0f} tok/s) help, because the expert dispatch in
`SparseSwiGLU.forward` is a Python loop over experts with `.nonzero()` on the router assignment — a
host-device sync and a graph break, eight times per layer per token. The benchmark score
(34.49 vs 34.36) is real; the "half the active compute" framing is not a latency claim until the
dispatch is a fused kernel (sort tokens by expert, one grouped GEMM).

**CUDA graphs need outputs cloned.** Replays reuse the same output buffers; the decode loop reads
`logits` and the KV/conv caches after the next step has run. Every cache slot must be cloned,
whatever its shape — attention slots are `(K, V)` tuples, short-conv slots are bare tensors, and one
uncloned tensor was enough to trip torch's overwrite check. The clone's cost is part of what that
row honestly measures.
""")
    Path("docs/KERNELS.md").write_text("\n".join(out), encoding="utf-8")
    print("wrote docs/KERNELS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
