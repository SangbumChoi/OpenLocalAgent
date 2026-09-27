#!/usr/bin/env python
"""Decode throughput in tokens per second, on the device a person would actually deploy to.

Seconds per decision conflates prompt length with model speed. This measures the two separately:
prefill over a fixed 512-token prompt, then sustained decode of 64 new tokens, reported as tokens
per second at a fixed thread count so the numbers are comparable across families.

  python scripts/throughput.py --model hf:data/baselines/SmolLM2-360M-Instruct --threads 4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def measure_hf(path: str, prompt_tokens: int, new_tokens: int, repeats: int) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    ids = torch.randint(0, min(tokenizer.vocab_size, 20000), (1, prompt_tokens))
    with torch.no_grad():
        model.generate(ids, max_new_tokens=4, do_sample=False,
                       pad_token_id=tokenizer.eos_token_id)  # warm up kernels and allocator
        # Hybrid-attention families raise from the KV cache on this transformers version; the
        # cacheless path is what the evaluation harness uses for them, so time that instead.
        cache = True
        try:
            model.generate(ids, max_new_tokens=2, do_sample=False,
                           pad_token_id=tokenizer.eos_token_id)
        except ValueError:
            cache = False
        prefills, decodes = [], []
        for _ in range(repeats):
            started = time.perf_counter()
            model(ids)
            prefills.append(time.perf_counter() - started)
            started = time.perf_counter()
            model.generate(ids, max_new_tokens=new_tokens, min_new_tokens=new_tokens,
                           do_sample=False, use_cache=cache,
                           pad_token_id=tokenizer.eos_token_id)
            decodes.append(time.perf_counter() - started)
    parameters = sum(p.numel() for p in model.parameters())
    return summarise(parameters, prompt_tokens, new_tokens, prefills, decodes)


def measure_catalog(checkpoint: str, prompt_tokens: int, new_tokens: int, repeats: int) -> dict:
    from localagent.model import LocalAgentLM, ModelConfig
    from localagent.model.tokenizer import load_tokenizer

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    fields = {key: value for key, value in payload["cfg"].items()
              if key in ModelConfig.__dataclass_fields__}
    model = LocalAgentLM(ModelConfig(**fields))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    tokenizer = load_tokenizer("bpe", "data/tokenizer-h100-16k.json")
    from localagent.inference.generate import generate

    prompt = " ".join(["the tool catalog and the request"] * 40)
    encoded = len(tokenizer.encode(prompt))
    with torch.no_grad():
        generate(model, tokenizer, prompt, max_new_tokens=4)
        prefills, decodes = [], []
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
        for _ in range(repeats):
            started = time.perf_counter()
            model(ids, pos=0, caches=[None] * model.n_cache_slots())
            prefills.append(time.perf_counter() - started)
            started = time.perf_counter()
            generate(model, tokenizer, prompt, max_new_tokens=new_tokens)
            decodes.append(time.perf_counter() - started)
    return summarise(model.num_params(), encoded, new_tokens, prefills, decodes)


def summarise(parameters: int, prompt_tokens: int, new_tokens: int,
              prefills: list[float], decodes: list[float]) -> dict:
    prefill = min(prefills)
    decode = min(decodes)
    return {
        "parameters": parameters,
        "prompt_tokens": prompt_tokens,
        "new_tokens": new_tokens,
        "prefill_seconds": prefill,
        "prefill_tokens_per_second": prompt_tokens / prefill,
        "decode_seconds": decode,
        # The generate() call includes one prefill; charge decode only for the tokens it produced.
        "decode_tokens_per_second": new_tokens / max(decode - prefill, 1e-6),
        "seconds_per_decision": decode,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="hf:<path> or catalog:<checkpoint>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    kind, _, location = args.model.partition(":")
    if kind == "hf":
        report = measure_hf(location, args.prompt_tokens, args.new_tokens, args.repeats)
    elif kind == "catalog":
        report = measure_catalog(location, args.prompt_tokens, args.new_tokens, args.repeats)
    else:
        raise SystemExit(f"unknown kind {kind!r}")
    name = Path(location).parent.name if location.endswith(".pt") else Path(location).name
    report |= {"model": name, "kind": kind, "threads": args.threads}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['model']}: decode {report['decode_tokens_per_second']:.1f} tok/s, "
          f"prefill {report['prefill_tokens_per_second']:.0f} tok/s", flush=True)
    print("THROUGHPUT_DONE " + args.out, flush=True)


if __name__ == "__main__":
    main()
