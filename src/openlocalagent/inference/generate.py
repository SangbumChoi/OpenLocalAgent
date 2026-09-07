"""KV-cached generation: prefill the prompt once, then decode one token at a time (Phase 1/7).

Returns the decoded text plus timing/throughput stats (prefill vs decode tokens/sec) used by
the benchmark + the agent runtime + eval. Greedy by default (deterministic → exact-match eval);
temperature/top-p for sampling (used by GRPO rollouts).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class GenStats:
    prompt_tokens: int
    new_tokens: int
    prefill_s: float
    decode_s: float

    @property
    def prefill_tok_s(self) -> float:
        return self.prompt_tokens / self.prefill_s if self.prefill_s > 0 else 0.0

    @property
    def decode_tok_s(self) -> float:
        return self.new_tokens / self.decode_s if self.decode_s > 0 else 0.0


def _sample(logits, temperature: float, top_p: float) -> int:
    if temperature <= 0:
        return int(logits.argmax(-1))
    probs = F.softmax(logits / temperature, dim=-1)
    if 0 < top_p < 1:
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        keep = cum <= top_p
        keep[0] = True
        sp = sp * keep
        sp = sp / sp.sum()
        return int(si[torch.multinomial(sp, 1)])
    return int(torch.multinomial(probs, 1))


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 96,
             temperature: float = 0.0, top_p: float = 0.95):
    model.eval()
    device = next(model.parameters()).device
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    caches = [None] * model.n_cache_slots()

    t0 = time.perf_counter()
    logits, _, caches = model(ids, pos=ids.shape[1] and 0, caches=caches)
    prefill_s = time.perf_counter() - t0
    pos = ids.shape[1]

    out_ids: list[int] = []
    t1 = time.perf_counter()
    for _ in range(max_new_tokens):
        nxt = _sample(logits[0, -1], temperature, top_p)
        if nxt == tokenizer.eos_id:
            break
        out_ids.append(nxt)
        step = torch.tensor([[nxt]], dtype=torch.long, device=device)
        logits, _, caches = model(step, pos=pos, caches=caches)
        pos += 1
    decode_s = time.perf_counter() - t1

    stats = GenStats(ids.shape[1], len(out_ids), prefill_s, decode_s)
    return tokenizer.decode(out_ids), stats
