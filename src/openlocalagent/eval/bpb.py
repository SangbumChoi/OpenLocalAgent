"""Bits-per-byte (BPB) — the standard, scale-appropriate pretrain metric for a byte-level LM.

OLMES / olmo-eval's multiple-choice suites (MMLU/ARC/HellaSwag) are near-random for a 28M model and
need a transformers/vLLM model; BPB on held-out text is the meaningful, comparable number here
(olmo-eval lists bits-per-byte as a metric too). BPB = mean next-byte NLL (nats) / ln(2).
"""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def bits_per_byte(model, stream, *, seq_len=512, max_windows=200, device="cpu") -> float:
    """Mean bits-per-byte over non-overlapping windows of a held-out byte-id `stream`."""
    model.eval()
    data = torch.tensor(stream, dtype=torch.long)
    n = (len(data) - 1) // seq_len
    if n == 0:
        return float("nan")
    step = max(1, n // max_windows)
    tot_nats = tot_bytes = 0
    for w in range(0, n, step):
        s = w * seq_len
        x = data[s:s + seq_len].unsqueeze(0).to(device)
        y = data[s + 1:s + 1 + seq_len].unsqueeze(0).to(device)
        logits, _ = model(x)
        nll = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        tot_nats += nll.item()
        tot_bytes += y.numel()
    return (tot_nats / tot_bytes) / math.log(2)
