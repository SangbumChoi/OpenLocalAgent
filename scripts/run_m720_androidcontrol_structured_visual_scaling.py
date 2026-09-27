#!/usr/bin/env python3
"""Scale the bounded official AndroidControl screenshot pilot with a disjoint episode holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from run_m714_androidcontrol_structured_visual_pilot import (
    _download,
    _load_samples,
    _run_arm,
    OBJECT_URL,
)
from localagent.model import ModelConfig
from localagent.model.vision import ANDROID_ACTIONS, VisualActionHead


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--prefix-bytes", type=int, default=200 * 1024 * 1024)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--warm-checkpoint", type=Path)
    parser.add_argument("--random-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.prefix_bytes < 1 or args.steps < 1 or args.eval_episodes < 1:
        raise ValueError("prefix-bytes, steps, and eval-episodes must be positive")

    raw = _download(OBJECT_URL, 0, args.prefix_bytes - 1)
    cfg = ModelConfig.from_yaml("configs/model/webgpu-10m-vision.yaml")
    samples, episodes = _load_samples(
        raw,
        image_size=cfg.vision_image_size,
        max_seq_len=cfg.max_seq_len - cfg.vision_tokens,
    )
    if len(episodes) <= args.eval_episodes:
        raise ValueError(f"need more than {args.eval_episodes} complete episodes; got {len(episodes)}")
    holdout = min(args.eval_episodes, len(episodes) - 1)
    train_episodes = len(episodes) - holdout
    train = [index for index, sample in enumerate(samples) if sample["episode_index"] < train_episodes]
    eval_rows = [index for index, sample in enumerate(samples) if sample["episode_index"] >= train_episodes]

    parent = torch.load(args.parent, map_location="cpu", weights_only=False)
    parent_state = parent.get("state_dict")
    if not isinstance(parent_state, dict):
        raise ValueError("parent checkpoint has no state_dict")
    torch.manual_seed(720)
    head_state = VisualActionHead(cfg.d_model).state_dict()
    warm, warm_model, warm_head = _run_arm(
        parent_state,
        cfg,
        samples,
        train,
        eval_rows,
        warm=True,
        steps=args.steps,
        seed=720,
        head_state=head_state,
    )
    random, random_model, random_head = _run_arm(
        None,
        cfg,
        samples,
        train,
        eval_rows,
        warm=False,
        steps=args.steps,
        seed=721,
        head_state=head_state,
    )
    for path, model, head, arm in (
        (args.warm_checkpoint, warm_model, warm_head, "warm"),
        (args.random_checkpoint, random_model, random_head, "random"),
    ):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "cfg": cfg.__dict__,
                    "state_dict": model.state_dict(),
                    "head_state": head.state_dict(),
                    "action_names": list(ANDROID_ACTIONS),
                    "arm": arm,
                },
                path,
            )

    payload: dict[str, Any] = {
        "kind": "localagent_m720_androidcontrol_structured_visual_scaling",
        "schema_version": 1,
        "source": {
            "dataset": "AndroidControl",
            "object_url": OBJECT_URL,
            "range_bytes": len(raw),
            "range_sha256": hashlib.sha256(raw).hexdigest(),
            "complete_episodes": len(episodes),
            "train_episodes": train_episodes,
            "heldout_episodes": holdout,
            "train_rows": len(train),
            "heldout_rows": len(eval_rows),
            "split": "complete-record disjoint; final heldout episodes",
        },
        "model": {
            "config": "configs/model/webgpu-10m-vision.yaml",
            "parameters": sum(parameter.numel() for parameter in warm_model.parameters()),
            "backbone_frozen": True,
            "trainable": ["vision", "structured_action_head"],
            "action_names": list(ANDROID_ACTIONS),
            "steps": args.steps,
        },
        "warm": warm,
        "random": random,
        "weight_transfer": {
            "warm_parent_sha256": hashlib.sha256(args.parent.read_bytes()).hexdigest(),
            "comparison": "warm frozen text backbone versus random frozen text backbone",
            "action_accuracy_delta_warm_minus_random": warm["after"]["action_accuracy"] - random["after"]["action_accuracy"],
            "coordinate_mae_delta_warm_minus_random": warm["after"]["coordinate_mae"] - random["after"]["coordinate_mae"],
        },
        "claim_boundary": (
            "Official AndroidControl screenshot/action scaling diagnostic with a complete-record "
            "holdout and matched warm/random frozen text backbones. It is not an official leaderboard "
            "score, Android emulator result, or WebGPU task-completion claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"episodes": len(episodes), "train_rows": len(train), "heldout_rows": len(eval_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
