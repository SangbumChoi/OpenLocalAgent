#!/usr/bin/env python
"""Merge flywheel specialist checkpoints in weight space (model soup / TIES).

Combines per-round specialists saved by ``scripts/analyze_loop.py``
(``runs/analyze_*/model.pt`` → ``{cfg, state_dict, tool_head, ptr_head}``) into a single
checkpoint of the same format — training-free, just weight arithmetic. The "merge vs retrain"
lever from docs/ARCHITECTURE_DEBATE.md.

  # uniform model soup of two rounds:
  python scripts/merge.py --checkpoints a.pt b.pt --out merged.pt

  # weighted soup:
  python scripts/merge.py --checkpoints a.pt b.pt --weights 0.7 0.3 --out merged.pt

  # TIES (needs a common base — e.g. the post-pretrain / round-0 checkpoint):
  python scripts/merge.py --checkpoints a.pt b.pt --method ties --base base.pt --out merged.pt
"""

from __future__ import annotations

import argparse
import os

import torch

from localagent.train.merge import merge_checkpoints


def _load(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="2+ checkpoint .pt files ({cfg, state_dict, tool_head, ptr_head})")
    ap.add_argument("--method", choices=["soup", "ties"], default="soup")
    ap.add_argument("--base", default=None,
                    help="base checkpoint for TIES task vectors (required for --method ties)")
    ap.add_argument("--density", type=float, default=0.2,
                    help="TIES: fraction of top-magnitude entries kept per tensor (default 0.2)")
    ap.add_argument("--weights", nargs="+", type=float, default=None,
                    help="soup: per-checkpoint mixing weights (normalized; default uniform)")
    ap.add_argument("--out", required=True, help="output merged checkpoint path")
    args = ap.parse_args()

    if len(args.checkpoints) < 2:
        raise SystemExit("need at least 2 checkpoints to merge")
    if args.method == "ties" and not args.base:
        raise SystemExit("--method ties requires --base <checkpoint.pt>")
    if args.weights is not None and len(args.weights) != len(args.checkpoints):
        raise SystemExit(f"--weights ({len(args.weights)}) must match "
                         f"--checkpoints ({len(args.checkpoints)})")

    ckpts = [_load(p) for p in args.checkpoints]
    base = _load(args.base) if args.base else None
    merged = merge_checkpoints(ckpts, method=args.method, base=base,
                               density=args.density, weights=args.weights)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.save(merged, args.out)

    desc = f"{args.method}"
    if args.method == "ties":
        desc += f" (density={args.density}, base={args.base})"
    elif args.weights:
        desc += f" (weights={args.weights})"
    print(f"merged {len(args.checkpoints)} checkpoints via {desc} -> {args.out}")
    print(f"  tensors={len(merged['state_dict'])}  "
          f"tool_head={'yes' if merged['tool_head'] else 'no'}  "
          f"ptr_head={'yes' if merged['ptr_head'] else 'no'}")


if __name__ == "__main__":
    main()
