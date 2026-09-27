"""Shared training utilities: padding/collation, cosine LR, one AdamW step.

Kept tiny and dependency-free so pretrain/sft/grpo all reuse it (Phase 2/4/10).
"""

from __future__ import annotations

import math
from typing import Any

import torch

from localagent.data.render import IGNORE


def validate_pad_to_input_tokens(
    value: Any,
    *,
    label: str = "pad_to_input_tokens",
) -> int | None:
    """Validate an optional fixed post-shift language-model input width."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def pad_batch(
    rows: list[tuple[list[int], list[int]]],
    pad_id: int,
    device,
    pad_to_input_tokens: int | None = None,
):
    """Return next-token tensors, optionally padded to an exact post-shift input width.

    ``pad_to_input_tokens`` describes the width returned to the language model, not the raw
    rendered row width.  The raw rows are therefore padded to one additional token before the
    shared next-token shift.  Rows are never truncated.
    """

    fixed_input_width = validate_pad_to_input_tokens(pad_to_input_tokens)
    maxlen = max(len(r[0]) for r in rows)
    if fixed_input_width is not None:
        required_input_width = maxlen - 1
        if required_input_width > fixed_input_width:
            raise ValueError(
                "row requires more input tokens than pad_to_input_tokens: "
                f"required={required_input_width}, configured={fixed_input_width}"
            )
        maxlen = fixed_input_width + 1
    xs, ys = [], []
    for ids, labels in rows:
        n = maxlen - len(ids)
        xs.append(ids + [pad_id] * n)
        ys.append(labels + [IGNORE] * n)
    x = torch.tensor(xs, dtype=torch.long, device=device)
    y = torch.tensor(ys, dtype=torch.long, device=device)
    # next-token shift: predict y[t+1] from x[t]
    return x[:, :-1], y[:, 1:]


def cosine_lr(step: int, total: int, peak: float, warmup: int, min_ratio: float) -> float:
    if step < warmup:
        return peak * step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


def wsd_lr(step: int, total: int, peak: float, warmup: int, decay_frac: float,
           min_ratio: float = 0.0) -> float:
    """Warmup-Stable-Decay LR (MiniCPM, 2404.06395).

    Three phases over `total` steps:
      - warmup  : linear 0 -> peak over the first `warmup` steps.
      - stable  : constant `peak` plateau (the reusable/forkable checkpoint lives here).
      - decay   : last `decay_frac` of steps, exponential `peak * 0.5^((s-S)/T)` where S is the
                  first decay step and T is the decay window length; the sharp loss drop happens
                  here. Clamped at `peak * min_ratio` (default 0 -> decays toward 0).

    With `decay_frac=0` this reduces to warmup + flat plateau (no decay).
    """
    if step < warmup:
        return peak * step / max(1, warmup)
    decay_steps = int(round(total * decay_frac))
    decay_start = total - decay_steps                      # S: first step of the decay window
    if step < decay_start or decay_steps <= 0:
        return peak                                        # stable plateau
    T = decay_steps                                        # decay time-constant = window length
    lr = peak * 0.5 ** ((step - decay_start) / max(1, T))
    return max(lr, peak * min_ratio)


def in_decay_window(step: int, total: int, decay_frac: float) -> bool:
    """True iff `step` falls in the WSD decay window (used to swap in curated `decay_samples`)."""
    decay_steps = int(round(total * decay_frac))
    return decay_steps > 0 and step >= total - decay_steps


def set_lr(opt, lr: float) -> None:
    for g in opt.param_groups:
        g["lr"] = lr


def router_loss_terms(
    model,
    lm_loss: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine an opt-in router auxiliary objective with a pure language-model loss.

    Returns ``(optimization_loss, router_aux_loss, router_weighted_loss)``. The model's normal
    target loss remains pure cross-entropy, so held-out evaluation is never contaminated by the
    balancing objective. Dense models return exact scalar zeros for both router terms.
    """

    coefficient = float(getattr(model.cfg, "router_aux_loss_coef", 0.0))
    router_aux_fn = getattr(model, "routing_aux_loss", None)
    router_aux = router_aux_fn() if callable(router_aux_fn) else None
    if router_aux is None:
        zero = lm_loss.new_zeros(())
        return lm_loss, zero, zero
    router_weighted = router_aux * coefficient
    return lm_loss + router_weighted, router_aux, router_weighted
