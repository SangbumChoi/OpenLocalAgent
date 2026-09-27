# Kernel configurations: what a throughput number depends on

Measured on **NVIDIA H100 NVL**, torch 2.10.0, batch 1, 512-token prompt, 128 decode tokens, median of 5. Random ids, so this runs on checkpoints whose tokenizer is gone. Produced by `scripts/eval_kernels.py`; receipts in `results/throughput/kernels-*.json`.

Prefill and decode are different regimes and are reported separately: prefill is one wide matmul over the prompt (compute-bound); decode is 128 sequential single-token steps (launch- and memory-bound). A configuration can win one and lose the other.


## la-93m

| kernel | prefill tok/s | decode tok/s | ms/tok | peak MB | vs eager bf16 |
|---|---:|---:|---:|---:|---:|
| `eager_fp32` | 78,346 | **172.9** | 5.78 | 448 | 0.99× |
| `eager_bf16` | 84,946 | **175.4** | 5.70 | 240 | 1.00× |
| `sdpa_flash_bf16` | 89,852 | **184.7** | 5.41 | 240 | 1.05× |
| `sdpa_efficient_bf16` | 92,937 | **186.1** | 5.37 | 240 | 1.06× |
| `sdpa_math_bf16` | 78,215 | **166.9** | 5.99 | 255 | 0.95× |
| `compile_bf16` | 80,068 | **378.1** | 2.65 | 302 | 2.16× |
| `compile_cudagraph_bf16` | 79,141 | **1,122.5** | 0.89 | 240 | 6.40× |

## la-300m

| kernel | prefill tok/s | decode tok/s | ms/tok | peak MB | vs eager bf16 |
|---|---:|---:|---:|---:|---:|
| `eager_fp32` | 85,838 | **185.0** | 5.41 | 1,287 | 1.00× |
| `eager_bf16` | 92,837 | **184.5** | 5.42 | 657 | 1.00× |
| `sdpa_flash_bf16` | 94,263 | **188.3** | 5.31 | 657 | 1.02× |
| `sdpa_efficient_bf16` | 97,141 | **190.0** | 5.26 | 656 | 1.03× |
| `sdpa_math_bf16` | 82,149 | **172.6** | 5.79 | 697 | 0.94× |
| `compile_bf16` | 86,980 | **395.7** | 2.53 | 718 | 2.14× |
| `compile_cudagraph_bf16` | 81,579 | **927.7** | 1.08 | 718 | 5.03× |

## la-300m-moe

| kernel | prefill tok/s | decode tok/s | ms/tok | peak MB | vs eager bf16 |
|---|---:|---:|---:|---:|---:|
| `eager_fp32` | 17,383 | **60.7** | 16.48 | 1,308 | 1.17× |
| `eager_bf16` | 14,960 | **51.8** | 19.30 | 652 | 1.00× |
| `sdpa_flash_bf16` | 17,138 | **56.3** | 17.75 | 652 | 1.09× |
| `sdpa_efficient_bf16` | 17,234 | **56.9** | 17.58 | 652 | 1.10× |
| `sdpa_math_bf16` | 13,148 | **43.6** | 22.96 | 691 | 0.84× |
| `compile_bf16` | 16,264 | **46.8** | 21.38 | 713 | 0.90× |
| `compile_cudagraph_bf16` | 14,999 | **46.3** | 21.58 | 713 | 0.89× |

## What it says

**Small dense models on an H100 are launch-bound, not FLOP-bound.** bf16 and the attention backend
move `la-93m` decode from 173 to 186 tok/s — a few
percent. `torch.compile` fuses the dozens of tiny kernels per step and gives
**2.2×**; CUDA graphs remove the launch overhead itself and
give **6.4×**
(0.89 ms/token). `la-300m` follows the same
curve (928 tok/s,
5.0×). Prefill, one large matmul, gets its
8–18% from bf16 and the SDPA backend instead and is untouched by compilation.

**The sparse-expert arm's matched active compute does not become matched wall-clock.** `la-300m-moe`
decodes at 52 tok/s eager — 3.6× slower
than dense `la-300m` with the same active parameters — and prefill is
6.2× slower. Neither
`torch.compile` (47 tok/s) nor CUDA graphs
(46 tok/s) help, because the expert dispatch in
`SparseSwiGLU.forward` is a Python loop over experts with `.nonzero()` on the router assignment — a
host-device sync and a graph break, eight times per layer per token. The benchmark score
(34.49 vs 34.36) is real; the "half the active compute" framing is not a latency claim until the
dispatch is a fused kernel (sort tokens by expert, one grouped GEMM).

**CUDA graphs need outputs cloned.** Replays reuse the same output buffers; the decode loop reads
`logits` and the KV/conv caches after the next step has run. Every cache slot must be cloned,
whatever its shape — attention slots are `(K, V)` tuples, short-conv slots are bare tensors, and one
uncloned tensor was enough to trip torch's overwrite check. The clone's cost is part of what that
row honestly measures.
