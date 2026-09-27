#!/usr/bin/env python3
"""Benchmark a matched set of LocalAgent model configurations on one device.

This is the portable architecture/latency comparison for the GPU machine.  It constructs fresh
random-weight models, so the result measures architecture, cache, dtype, and device differences;
it is not a quality or agent-success score.  Use ``run_gpu_campaign.py`` to combine this with
weight-transfer and public-evaluation receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from functools import partial
from pathlib import Path
from typing import Any, Iterable

import torch

from localagent.model import LocalAgentLM, ModelConfig
from localagent.train.device import resolve_device

DEFAULT_MODELS = (
    "webgpu-10m-attn",
    "webgpu-10m-hybrid",
    "webgpu-10m-vision",
    "webgpu-16m-attn",
    "webgpu-16m-hybrid",
    "webgpu-35m-attn",
    "webgpu-35m-hybrid",
    "webgpu-96m-attn",
    "webgpu-96m-hybrid",
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name != "auto":
        raise ValueError(f"unsupported dtype: {name}")
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def _identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _timed(fn, device: torch.device, repeats: int) -> tuple[float, list[float]]:
    values: list[float] = []
    for _ in range(repeats):
        _sync(device)
        start = time.perf_counter()
        fn()
        _sync(device)
        values.append(time.perf_counter() - start)
    return min(values), values


def _prefill(lm: LocalAgentLM, prompt: torch.Tensor) -> None:
    lm(prompt)


def _cached_decode(
    lm: LocalAgentLM,
    prompt: torch.Tensor,
    first_token: torch.Tensor,
    prompt_len: int,
    decode: int,
) -> None:
    _, _, caches = lm(prompt, pos=0, caches=[None] * lm.n_cache_slots())
    token = first_token
    for position in range(prompt_len, prompt_len + decode):
        logits, _, caches = lm(token, pos=position, caches=caches)
        token = logits[:, -1:].argmax(-1)


def _uncached_decode(lm: LocalAgentLM, prompt: torch.Tensor, decode: int) -> None:
    sequence = prompt
    for _ in range(decode):
        logits, _ = lm(sequence)
        sequence = torch.cat([sequence, logits[:, -1:].argmax(-1)], dim=1)


@torch.inference_mode()
def benchmark_one(
    model_name: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    prompt_len: int,
    decode: int,
    repeats: int,
    uncached: bool,
) -> dict[str, Any]:
    config_path = Path("configs/model") / f"{model_name}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing model config: {config_path}")
    cfg = ModelConfig.from_yaml(config_path)
    cfg.assert_within_budget()
    if prompt_len < 1 or decode < 1 or prompt_len + decode > cfg.max_seq_len:
        raise ValueError(
            f"invalid prompt/decode for {model_name}: {prompt_len}+{decode} > {cfg.max_seq_len}"
        )
    lm = LocalAgentLM(cfg).to(device=device, dtype=dtype).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, prompt_len), device=device)
    nxt = torch.randint(0, cfg.vocab_size, (1, 1), device=device)
    for _ in range(2):
        lm(prompt)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    _, prefill_samples = _timed(partial(_prefill, lm, prompt), device, repeats)
    prefill_best = min(prefill_samples)

    _, cached_samples = _timed(
        partial(_cached_decode, lm, prompt, nxt, prompt_len, decode), device, repeats
    )
    uncached_samples: list[float] = []
    if uncached:
        _, uncached_samples = _timed(partial(_uncached_decode, lm, prompt, decode), device, repeats)

    result: dict[str, Any] = {
        "model": model_name,
        "config": _identity(config_path),
        "parameters": lm.num_params(),
        "active_parameters": lm.active_num_params(),
        "dtype": str(dtype).removeprefix("torch."),
        "prompt_len": prompt_len,
        "decode_tokens": decode,
        "repeats": repeats,
        "prefill_tok_s": prompt_len / prefill_best,
        "cached_decode_tok_s": decode / min(cached_samples),
        "prefill_seconds": prefill_samples,
        "cached_decode_seconds": cached_samples,
        "uncached_decode_seconds": uncached_samples,
        "kv_cache_bytes_estimate": cfg.estimate_cache_bytes(prompt_len, torch.finfo(dtype).bits // 8),
        "weight_bytes_estimate": lm.num_params() * torch.finfo(dtype).bits // 8,
    }
    if uncached_samples:
        result["uncached_decode_tok_s"] = decode / min(uncached_samples)
        result["cache_speedup"] = result["cached_decode_tok_s"] / result["uncached_decode_tok_s"]
    if device.type == "cuda":
        result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    del lm
    return result


def run_benchmark(
    models: Iterable[str],
    *,
    device_name: str,
    dtype_name: str,
    prompt_len: int,
    decode: int,
    repeats: int,
    uncached: bool,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    dtype = _dtype(dtype_name, device)
    results = [
        benchmark_one(
            model,
            device=device,
            dtype=dtype,
            prompt_len=prompt_len,
            decode=decode,
            repeats=repeats,
            uncached=uncached,
        )
        for model in models
    ]
    return {
        "kind": "localagent_model_config_benchmark",
        "schema_version": 1,
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "host": {"platform": platform.platform(), "torch": torch.__version__},
        "protocol": {
            "prompt_len": prompt_len,
            "decode_tokens": decode,
            "repeats": repeats,
            "uncached_control": uncached,
            "random_weights": True,
        },
        "results": results,
        "claim_boundary": (
            "Random-weight architecture/device/cache benchmark only. It does not establish model "
            "quality, tool-call accuracy, native browser/mobile success, or WebGPU browser parity."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", dest="models", help="model config stem; repeat")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--decode", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-uncached", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    report = run_benchmark(
        args.models or DEFAULT_MODELS,
        device_name=args.device,
        dtype_name=args.dtype,
        prompt_len=args.prompt_len,
        decode=args.decode,
        repeats=args.repeats,
        uncached=not args.no_uncached,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
