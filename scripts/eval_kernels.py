#!/usr/bin/env python
"""Decode and prefill throughput of a checkpoint under each kernel configuration, on one GPU.

"Is the model fast" has no answer without saying which kernels ran. The same weights decode at
very different rates depending on dtype, which scaled-dot-product-attention backend PyTorch
chose, and whether the per-token step was compiled - and the default is whatever the dispatcher
picked that day. This pins each choice and measures it, so a throughput number carries its
kernel configuration the way a benchmark score carries its prompt contract.

No tokenizer is needed: throughput depends on shapes, not on which tokens, so the prompt is random
ids. That also means this runs on a checkpoint whose tokenizer file is gone.

Measured separately because they are different regimes: prefill is one wide matmul over the whole
prompt (compute-bound), decode is 128 sequential single-token steps (latency- and memory-bound).
A configuration that wins one can lose the other.

  python scripts/eval_kernels.py --checkpoint results/runs/posttrain-la-93m/latest.pt
  python scripts/eval_kernels.py --checkpoint ... --only eager_bf16,compile_bf16
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from openlocalagent.model import LocalAgentLM, ModelConfig  # noqa: E402


class Kernel(StrEnum):
    """One named kernel configuration. The name is the axis label in the results table."""

    EAGER_FP32 = "eager_fp32"
    EAGER_BF16 = "eager_bf16"
    SDPA_FLASH_BF16 = "sdpa_flash_bf16"
    SDPA_EFFICIENT_BF16 = "sdpa_efficient_bf16"
    SDPA_MATH_BF16 = "sdpa_math_bf16"
    COMPILE_BF16 = "compile_bf16"
    COMPILE_CUDAGRAPH_BF16 = "compile_cudagraph_bf16"

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32 if self is Kernel.EAGER_FP32 else torch.bfloat16

    @property
    def sdpa_backend(self):
        """Pin one attention backend, or None to let the dispatcher choose."""
        from torch.nn.attention import SDPBackend

        match self:
            case Kernel.SDPA_FLASH_BF16:
                return SDPBackend.FLASH_ATTENTION
            case Kernel.SDPA_EFFICIENT_BF16:
                return SDPBackend.EFFICIENT_ATTENTION
            case Kernel.SDPA_MATH_BF16:
                return SDPBackend.MATH
            case _:
                return None

    @property
    def compile_mode(self) -> str | None:
        match self:
            case Kernel.COMPILE_BF16:
                return "default"
            case Kernel.COMPILE_CUDAGRAPH_BF16:
                return "reduce-overhead"
            case _:
                return None

    def context(self):
        """The `with` block a measurement runs inside."""
        from torch.nn.attention import sdpa_kernel

        backend = self.sdpa_backend
        return sdpa_kernel(backend) if backend is not None else contextlib.nullcontext()


@dataclass(frozen=True)
class Measurement:
    kernel: str
    prompt_tokens: int
    new_tokens: int
    prefill_tok_s: float
    decode_tok_s: float
    decode_ms_per_token: float
    peak_mem_mb: float
    repeats: int
    error: str | None = None


def load(checkpoint: Path, device: str, dtype: torch.dtype) -> LocalAgentLM:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**(payload.get("cfg") or payload.get("config")))
    model = LocalAgentLM(cfg)
    model.load_state_dict(payload.get("state_dict") or payload.get("model"))
    return model.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def one_run(model, prompt: torch.Tensor, new_tokens: int, step_fn) -> tuple[float, float]:
    """Returns (prefill_s, decode_s). step_fn is the possibly-compiled single-token forward."""
    caches = [None] * model.n_cache_slots()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits, _, caches = model(prompt, pos=0, caches=caches)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0

    pos = prompt.shape[1]
    nxt = logits[0, -1].argmax().view(1, 1)
    t1 = time.perf_counter()
    for _ in range(new_tokens):
        logits, _, caches = step_fn(nxt, pos, caches)
        nxt = logits[0, -1].argmax().view(1, 1)
        pos += 1
    torch.cuda.synchronize()
    return prefill_s, time.perf_counter() - t1


def measure(kernel: Kernel, checkpoint: Path, device: str, prompt_len: int,
            new_tokens: int, repeats: int) -> Measurement:
    model = load(checkpoint, device, kernel.dtype)
    prompt = torch.randint(0, model.cfg.vocab_size, (1, prompt_len), device=device)

    def step(ids, pos, caches):
        return model(ids, pos=pos, caches=caches)

    step_fn = step
    if kernel.compile_mode:
        # dynamic=True: the KV cache grows every token, so a static-shape compile would
        # recompile 128 times and measure the compiler, not the model.
        compiled = torch.compile(step, mode=kernel.compile_mode, dynamic=True)
        if kernel is Kernel.COMPILE_CUDAGRAPH_BF16:
            # CUDA graphs replay into the SAME output buffers. The decode loop reads `logits`
            # after the next step has run, which is exactly the overwrite torch refuses:
            # "accessing tensor output of CUDAGraphs that has been overwritten by a subsequent
            # run". Cloning is the documented contract, and its cost is part of what this
            # configuration honestly measures.
            def detach_from_graph(value):
                # Every cache slot must be cloned, whatever its shape: attention slots are
                # (K, V) tuples but the short-conv slots are bare tensors, and one uncloned
                # tensor read after the next replay is enough to trip the overwrite check.
                if torch.is_tensor(value):
                    return value.clone()
                if isinstance(value, (tuple, list)):
                    return type(value)(detach_from_graph(v) for v in value)
                return value

            def step_fn(ids, pos, caches):
                logits, loss, new_caches = compiled(ids, pos, caches)
                return logits.clone(), loss, detach_from_graph(new_caches)
        else:
            step_fn = compiled

    torch.cuda.reset_peak_memory_stats()
    try:
        with kernel.context():
            one_run(model, prompt, 8, step_fn)                       # warm-up / compile
            samples = [one_run(model, prompt, new_tokens, step_fn) for _ in range(repeats)]
    except Exception as failure:                                      # a backend may refuse
        return Measurement(kernel.value, prompt_len, new_tokens, 0.0, 0.0, 0.0, 0.0, repeats,
                           error=f"{type(failure).__name__}: {str(failure)[:160]}")
    prefill = statistics.median(s[0] for s in samples)
    decode = statistics.median(s[1] for s in samples)
    return Measurement(
        kernel=kernel.value, prompt_tokens=prompt_len, new_tokens=new_tokens,
        prefill_tok_s=round(prompt_len / prefill, 1),
        decode_tok_s=round(new_tokens / decode, 1),
        decode_ms_per_token=round(1000 * decode / new_tokens, 3),
        peak_mem_mb=round(torch.cuda.max_memory_allocated() / 1e6, 1),
        repeats=repeats,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", help="JSON receipt; default results/throughput/kernels-<run>.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--only", help="comma-separated kernel names")
    args = parser.parse_args()

    kernels = list(Kernel)
    if args.only:
        wanted = {k.strip() for k in args.only.split(",")}
        unknown = wanted - {k.value for k in kernels}
        if unknown:
            print(f"unknown kernels {sorted(unknown)}; known {[k.value for k in kernels]}",
                  file=sys.stderr)
            return 1
        kernels = [k for k in kernels if k.value in wanted]

    checkpoint = Path(args.checkpoint)
    results = []
    for kernel in kernels:
        m = measure(kernel, checkpoint, args.device, args.prompt_len, args.new_tokens,
                    args.repeats)
        results.append(m)
        if m.error:
            print(f"  {kernel.value:<24} FAILED  {m.error}", flush=True)
        else:
            print(f"  {kernel.value:<24} prefill {m.prefill_tok_s:>9.1f} tok/s   "
                  f"decode {m.decode_tok_s:>7.1f} tok/s  ({m.decode_ms_per_token:.2f} ms/tok)  "
                  f"peak {m.peak_mem_mb:.0f} MB", flush=True)
        torch.cuda.empty_cache()

    out = Path(args.out) if args.out else \
        Path("results/throughput") / f"kernels-{checkpoint.parent.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": str(checkpoint),
        "device": torch.cuda.get_device_name(0) if args.device == "cuda" else args.device,
        "torch": torch.__version__,
        "prompt_tokens": args.prompt_len, "new_tokens": args.new_tokens,
        "kernels": [asdict(m) for m in results],
    }, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
