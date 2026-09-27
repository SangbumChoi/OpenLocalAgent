"""Flywheel model merging in weight space — model soup + TIES.

The flywheel (``scripts/analyze_loop.py``) trains a *specialist* per round and saves the best to
``runs/analyze_*/model.pt`` as ``{cfg, state_dict, tool_head, ptr_head}``. Instead of always
retraining from the union of all rounds' data (expensive on this compute-bound sandbox), we can
combine those specialists directly in weight space — cheap, training-free weight arithmetic. This
is the "merge vs retrain" lever (axis 7) from ``docs/ARCHITECTURE_DEBATE.md``, the LFM2.5 / Arcee
approach.

Two methods, both pure-PyTorch and deterministic:

- ``model_soup``  — uniform (or weighted) average of matching parameters across checkpoints. The
  classic "model soup": simple, robust when the checkpoints are fine-tunes of a common ancestor.
- ``ties_merge``  — TIES-Merging (Yadav et al. 2023). Builds *task vectors* (ckpt - base), trims
  each to its top-``density`` magnitude entries, elects the dominant sign per entry across
  checkpoints, averages only the entries that agree with the elected sign, then adds the result
  back onto the base. Resolves sign interference between specialists.

Both operate on the agent heads (``tool_head`` / ``ptr_head``) the same way as the decoder
``state_dict``, since they are just more ``{name: tensor}`` mappings.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping, Sequence

import torch

StateDict = Mapping[str, torch.Tensor]


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------
def _assert_aligned(state_dicts: Sequence[StateDict], *, what: str = "state_dict") -> list[str]:
    """Assert every state_dict has identical keys and per-key shapes/dtypes. Returns the key list."""
    if not state_dicts:
        raise ValueError(f"{what}: need at least one state_dict to merge")
    ref = state_dicts[0]
    ref_keys = list(ref.keys())
    ref_set = set(ref_keys)
    for i, sd in enumerate(state_dicts[1:], start=1):
        keys = set(sd.keys())
        if keys != ref_set:
            missing = ref_set - keys
            extra = keys - ref_set
            raise ValueError(
                f"{what} #{i} has mismatched keys (missing={sorted(missing)[:4]}, "
                f"extra={sorted(extra)[:4]})")
        for k in ref_keys:
            if sd[k].shape != ref[k].shape:
                raise ValueError(
                    f"{what} #{i} key '{k}' shape {tuple(sd[k].shape)} != {tuple(ref[k].shape)}")
    return ref_keys


def _normalize_weights(weights: Sequence[float] | None, n: int) -> torch.Tensor:
    """Return an (n,) float tensor of mixing weights summing to 1 (uniform if ``weights`` is None)."""
    if weights is None:
        return torch.full((n,), 1.0 / n, dtype=torch.float64)
    if len(weights) != n:
        raise ValueError(f"weights has length {len(weights)} but got {n} state_dicts")
    w = torch.tensor(list(weights), dtype=torch.float64)
    total = w.sum()
    if total <= 0:
        raise ValueError(f"weights must sum to a positive value, got {total.item()}")
    return w / total


# --------------------------------------------------------------------------------------------------
# model soup
# --------------------------------------------------------------------------------------------------
def model_soup(state_dicts: Sequence[StateDict],
               weights: Sequence[float] | None = None) -> "OrderedDict[str, torch.Tensor]":
    """Average matching parameters across checkpoints (the "model soup").

    Args:
        state_dicts: checkpoints to merge; must share identical keys and shapes.
        weights:     optional per-checkpoint mixing weights (normalized to sum to 1). ``None`` →
                     uniform average.

    Returns:
        A new state_dict (same keys/shapes/dtypes as the inputs) holding the weighted mean.

    Soup of a model with itself (or N identical copies) is the identity. Integer/bool buffers
    (e.g. cached masks) are passed through from the first checkpoint rather than averaged.
    """
    keys = _assert_aligned(state_dicts)
    w = _normalize_weights(weights, len(state_dicts))
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k in keys:
        ref = state_dicts[0][k]
        if not ref.is_floating_point():
            # Non-float tensors (indices, bool masks) are not meaningfully averaged — carry the
            # first checkpoint's value through unchanged.
            out[k] = ref.clone()
            continue
        acc = torch.zeros(ref.shape, dtype=torch.float64)
        for wi, sd in zip(w, state_dicts):
            acc += wi * sd[k].to(torch.float64)
        out[k] = acc.to(ref.dtype)
    return out


# --------------------------------------------------------------------------------------------------
# TIES merge
# --------------------------------------------------------------------------------------------------
def _trim(task_vec: torch.Tensor, density: float) -> torch.Tensor:
    """Keep the top-``density`` fraction of entries by magnitude (per tensor); zero the rest."""
    if density >= 1.0:
        return task_vec
    flat = task_vec.reshape(-1)
    n = flat.numel()
    k = max(1, int(round(density * n))) if n > 0 else 0
    if k >= n:
        return task_vec
    # threshold = magnitude of the k-th largest entry; keep entries at/above it.
    thresh = torch.topk(flat.abs(), k, largest=True, sorted=True).values[-1]
    mask = task_vec.abs() >= thresh
    return task_vec * mask


def ties_merge(state_dicts: Sequence[StateDict],
               base_state: StateDict,
               density: float = 0.2) -> "OrderedDict[str, torch.Tensor]":
    """TIES-Merging of fine-tuned checkpoints over a common ``base_state``.

    Steps (per float tensor):
      1. **Task vectors**: ``tau_i = ckpt_i - base``.
      2. **Trim**: keep only the top-``density`` magnitude entries of each ``tau_i`` (zero rest).
      3. **Elect sign**: per entry, the sign with the larger summed magnitude across checkpoints.
      4. **Disjoint merge**: average only the entries whose sign agrees with the elected sign.
      5. **Add back**: ``base + merged_task_vector``.

    Args:
        state_dicts: fine-tuned checkpoints (all aligned with each other and with ``base_state``).
        base_state:  the common ancestor the task vectors are taken relative to.
        density:     fraction of entries to retain per tensor during trim (``1.0`` → no trimming,
                     which reduces to the mean of the (sign-agreeing) task vectors + base).

    Returns:
        A new merged state_dict (same keys/shapes/dtypes as the inputs).

    With ``density=1.0`` and all task vectors sharing a sign on every entry, every checkpoint
    agrees with the elected sign, so the result is exactly ``base + mean_i(tau_i)``.
    """
    if not (0.0 < density <= 1.0):
        raise ValueError(f"density must be in (0, 1], got {density}")
    keys = _assert_aligned(state_dicts)
    _assert_aligned([state_dicts[0], base_state], what="base vs ckpt state_dict")

    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k in keys:
        ref = state_dicts[0][k]
        base = base_state[k]
        if not ref.is_floating_point():
            out[k] = base.clone()
            continue
        base_f = base.to(torch.float64)
        # 1 + 2: task vectors, trimmed. Stack as (n_ckpt, *shape).
        taus = torch.stack([_trim(sd[k].to(torch.float64) - base_f, density) for sd in state_dicts])
        # 3: elect the dominant sign per entry (sign of the summed task vector).
        elected = torch.sign(taus.sum(dim=0))   # in {-1, 0, +1}
        # 4: keep only entries agreeing with the elected sign, then average over those.
        agree = (torch.sign(taus) == elected) & (elected != 0)
        kept = taus * agree
        count = agree.sum(dim=0)                 # how many checkpoints contributed per entry
        merged = torch.where(count > 0, kept.sum(dim=0) / count.clamp(min=1), torch.zeros_like(base_f))
        # 5: add back onto the base.
        out[k] = (base_f + merged).to(ref.dtype)
    return out


# --------------------------------------------------------------------------------------------------
# checkpoint-level convenience (operates on the {cfg, state_dict, tool_head, ptr_head} dict format)
# --------------------------------------------------------------------------------------------------
def _check_cfg_compat(ckpts: Sequence[dict]) -> dict:
    """Assert all checkpoints share the same architecture cfg; return the (first) cfg dict."""
    def as_dict(c):
        return c if isinstance(c, dict) else c.__dict__
    ref = as_dict(ckpts[0]["cfg"])
    arch_keys = ("vocab_size", "d_model", "embed_dim", "n_layers", "n_loops",
                 "n_heads", "n_kv_heads", "ffn_hidden", "max_seq_len")
    for i, c in enumerate(ckpts[1:], start=1):
        cd = as_dict(c["cfg"])
        for key in arch_keys:
            if cd.get(key) != ref.get(key):
                raise ValueError(
                    f"checkpoint #{i} cfg.{key}={cd.get(key)} != {ref.get(key)} (arch mismatch)")
    return ref


def _merge_head(ckpts: Sequence[dict], name: str, method: str, base: dict | None,
                density: float, weights) -> dict | None:
    """Merge an optional head (``tool_head`` / ``ptr_head``) across checkpoints, or None if absent."""
    heads = [c.get(name) for c in ckpts]
    present = [h for h in heads if h is not None]
    if not present:
        return None
    if len(present) != len(heads):
        raise ValueError(f"'{name}' present in some checkpoints but not all — cannot merge")
    if method == "soup":
        return model_soup(heads, weights=weights)
    base_head = (base or {}).get(name)
    if base_head is None:
        raise ValueError(f"ties merge needs a base '{name}' but the base checkpoint lacks it")
    return ties_merge(heads, base_head, density=density)


def merge_checkpoints(ckpts: Sequence[dict], method: str = "soup",
                      base: dict | None = None, density: float = 0.2,
                      weights: Sequence[float] | None = None) -> dict:
    """Merge a list of ``{cfg, state_dict, tool_head, ptr_head}`` checkpoints into one of the same
    format. ``method`` is ``"soup"`` or ``"ties"`` (TIES requires ``base``)."""
    if not ckpts:
        raise ValueError("no checkpoints to merge")
    cfg = _check_cfg_compat(ckpts)
    sds = [c["state_dict"] for c in ckpts]
    if method == "soup":
        merged_sd = model_soup(sds, weights=weights)
    elif method == "ties":
        if base is None:
            raise ValueError("method='ties' requires a base checkpoint (--base)")
        merged_sd = ties_merge(sds, base["state_dict"], density=density)
    else:
        raise ValueError(f"unknown method {method!r} (expected 'soup' or 'ties')")
    return {
        "cfg": cfg,
        "state_dict": merged_sd,
        "tool_head": _merge_head(ckpts, "tool_head", method, base, density, weights),
        "ptr_head": _merge_head(ckpts, "ptr_head", method, base, density, weights),
    }
