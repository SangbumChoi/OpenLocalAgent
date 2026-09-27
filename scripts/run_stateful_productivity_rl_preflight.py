#!/usr/bin/env python3
"""Run the strict one-update RL preflight on the local productivity simulator.

This is the current-checkpoint companion to ``train_stateful_productivity_rl.py``.  It creates
the deterministic email/Notion/browser Conversation rows in an isolated source directory, then
routes the first nonzero-learning-rate prefix through ``run_one_update_rl_preflight``.  The
resulting receipt is suitable for the fail-closed workshop gate, but it remains an RL simulation:
there is no emulator, browser service, MCP server, public benchmark payload, or external account.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from localagent.train.update_preflight import run_one_update_rl_preflight

from scripts.train_stateful_productivity_rl import _write_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--prompts-per-step", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2036)
    args = parser.parse_args()

    if args.root.exists() or args.root.is_symlink():
        raise SystemExit(f"refusing to overwrite root: {args.root}")
    if args.receipt.exists() or args.receipt.is_symlink():
        raise SystemExit(f"refusing to overwrite receipt: {args.receipt}")
    if not args.parent.is_file():
        raise SystemExit(f"missing parent checkpoint: {args.parent}")
    for value, label in (
        (args.steps, "steps"),
        (args.prompts_per_step, "prompts-per-step"),
        (args.group_size, "group-size"),
        (args.max_new_tokens, "max-new-tokens"),
    ):
        if value < 1:
            raise SystemExit(f"{label} must be positive")

    source_root = args.root / "source"
    source_root.mkdir(parents=True)
    train_path = source_root / "stateful-productivity-train.jsonl"
    eval_path = source_root / "stateful-productivity-eval.jsonl"
    _write_rows(train_path, "train")
    _write_rows(eval_path, "eval")

    production_out = args.root / "production-output"
    config_path = source_root / "rl.yaml"
    config = {
        "stage": "rl",
        "model_config": "configs/model/webgpu-10m-hybrid.yaml",
        "init_from": str(args.parent),
        "data": {
            "strict_conversation_artifacts": False,
            "conversations": [str(train_path)],
            "eval_conversations": [str(eval_path)],
            "tokenizer": {"kind": "bpe", "path": "data/tokenizer-webgpu-proxy-16k.json"},
        },
        "environment": {"name": "stateful_productivity", "learned_judge": False},
        "rollout": {
            "prompts_per_step": args.prompts_per_step,
            "group_size": args.group_size,
            "max_new_tokens": args.max_new_tokens,
            "temperature": 1.0,
        },
        "policy": {"clip_ratio": 0.2, "kl_beta": 0.02, "epochs_per_rollout": 1},
        "reward": {"format_weight": 0.1, "truncation_penalty": 0.05},
        "optim": {"name": "adamw", "lr": 2.0e-5},
        "schedule": {"total_steps": args.steps, "warmup_steps": 1},
        "runtime": {"device": "cpu", "dtype": "float32", "seed": args.seed, "resume": False},
        "log": {"out_dir": str(production_out), "ckpt_every": args.steps},
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    isolated_work = args.root / "isolated"
    try:
        receipt = run_one_update_rl_preflight(
            config_path,
            work_dir=isolated_work,
            receipt_path=args.receipt,
            device="cpu",
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        # The preflight function seals failed receipts before raising.  Keep the CLI useful while
        # preserving the fail-closed status for callers and the workshop gate.
        if args.receipt.is_file():
            print(json.dumps(json.loads(args.receipt.read_text(encoding="utf-8")), indent=2))
        raise SystemExit(str(error)) from error
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
