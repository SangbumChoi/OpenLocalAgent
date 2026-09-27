#!/usr/bin/env python3
"""Train/evaluate a bounded screenshot-to-action pilot on official AndroidControl train rows.

The pilot keeps the decoder backbone frozen and updates only the screenshot bridge.  Episodes are
split by source record (not by individual steps), with matched warm/random backbone controls.  It
uses raw UTF-8 bytes for the tiny action-language target so no tokenizer artifact is silently
introduced.  This is a reproducible transfer diagnostic, not an official AndroidControl score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import zlib
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from audit_androidcontrol_tfrecord_sample import (  # noqa: E402
    OBJECT_URL,
    _bytes_list,
    _download,
    _feature_map,
    _int64_list,
)
from localagent.data.visual import decode_png_rgb
from localagent.model import LocalAgentLM, ModelConfig


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _records(gzip_prefix: bytes) -> list[bytes]:
    decompressed = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(gzip_prefix)
    result: list[bytes] = []
    offset = 0
    while offset + 12 <= len(decompressed):
        length = struct.unpack("<Q", decompressed[offset : offset + 8])[0]
        end = offset + 12 + length + 4  # length + length CRC + payload + payload CRC
        if end > len(decompressed):
            break
        result.append(decompressed[offset + 12 : offset + 12 + length])
        offset = end
    return result


def _canonical_action(raw: bytes) -> bytes:
    parsed = json.loads(raw.decode("utf-8"))
    return json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _source_samples(prefix: bytes, *, image_size: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = _records(prefix)
    samples: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        features = _feature_map(record)
        screenshots = _bytes_list(features["screenshots"])
        actions = _bytes_list(features["actions"])
        goals = _bytes_list(features["goal"])
        if not screenshots or not actions or not goals:
            continue
        goal = goals[0].decode("utf-8", errors="replace").strip()
        episode_ids = _int64_list(features["episode_id"]) if "episode_id" in features else []
        # AndroidControl stores one initial screenshot followed by one screenshot per transition.
        usable = min(len(actions), len(screenshots))
        episode = {
            "record_index": record_index,
            "record_sha256": _sha256(record),
            "record_bytes": len(record),
            "goal": goal,
            "episode_id": episode_ids[0] if episode_ids else None,
            "screenshots": len(screenshots),
            "actions": len(actions),
            "used_steps": usable,
            "screenshot_sha256": [_sha256(item) for item in screenshots[:usable]],
        }
        episodes.append(episode)
        for step_index in range(usable):
            image = decode_png_rgb(screenshots[step_index]).unsqueeze(0)
            image = F.interpolate(image, size=(image_size, image_size), mode="bilinear", align_corners=False)
            context = f"Task: {goal}\nAction: ".encode("utf-8")
            action = _canonical_action(actions[step_index])
            full = context + action
            if len(full) < 2:
                continue
            samples.append(
                {
                    "episode_index": len(episodes) - 1,
                    "record_index": record_index,
                    "step_index": step_index,
                    "image": image.squeeze(0),
                    "input": torch.tensor(list(full[:-1]), dtype=torch.long),
                    "target": torch.tensor(list(full[1:]), dtype=torch.long),
                    "mask_start": max(0, len(context) - 1),
                    "action_bytes": action,
                }
            )
    if len(episodes) < 4:
        raise ValueError(f"bounded source produced only {len(episodes)} complete episodes; need >=4")
    return samples, {"records_seen": len(records), "episodes": episodes}


def _batch(samples: list[dict[str, Any]], indices: list[int], *, max_len: int) -> tuple[torch.Tensor, ...]:
    chosen = [samples[index] for index in indices]
    width = min(max(len(item["input"]) for item in chosen), max_len)
    images = torch.stack([item["image"] for item in chosen])
    inputs = torch.zeros((len(chosen), width), dtype=torch.long)
    targets = torch.full((len(chosen), width), -100, dtype=torch.long)
    masks: list[tuple[int, int]] = []
    for row, item in enumerate(chosen):
        length = min(len(item["input"]), width)
        inputs[row, :length] = item["input"][:length]
        target_length = min(len(item["target"]), width)
        targets[row, :target_length] = item["target"][:target_length]
        start = min(int(item["mask_start"]), target_length)
        if start:
            targets[row, :start] = -100
        masks.append((start, target_length))
    return images, inputs, targets, masks


def _metrics(model: LocalAgentLM, samples: list[dict[str, Any]], indices: list[int], *, max_len: int) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    correct = total = 0
    exact = 0
    with torch.no_grad():
        for start in range(0, len(indices), 4):
            images, inputs, targets, masks = _batch(samples, indices[start : start + 4], max_len=max_len)
            logits, _, _, _ = model.forward_multimodal(inputs, images, targets, return_hidden=True)
            valid = targets.ne(-100)
            losses.append(float(F.cross_entropy(logits[valid], targets[valid])))
            predicted = logits.argmax(-1)
            for row, (mask_start, target_length) in enumerate(masks):
                if target_length <= mask_start:
                    continue
                expected = targets[row, mask_start:target_length]
                got = predicted[row, mask_start:target_length]
                correct += int(got.eq(expected).sum())
                total += int(expected.numel())
                exact += int(got.equal(expected))
    return {
        "samples": len(indices),
        "mean_action_token_loss": sum(losses) / max(1, len(losses)),
        "action_token_accuracy": correct / max(1, total),
        "action_sequence_exact": exact / max(1, len(indices)),
    }


def _train(model: LocalAgentLM, samples: list[dict[str, Any]], indices: list[int], *, steps: int, seed: int, max_len: int) -> tuple[dict[str, Any], dict[str, Any]]:
    if model.vision is None:
        raise ValueError("visual model is disabled")
    initial = {name: value.detach().clone() for name, value in model.vision.named_parameters()}
    optimizer = torch.optim.AdamW(model.vision.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(seed)
    model.train()
    losses: list[float] = []
    for _ in range(steps):
        selected = torch.randint(0, len(indices), (min(4, len(indices)),), generator=generator).tolist()
        batch_indices = [indices[index] for index in selected]
        images, inputs, targets, _ = _batch(samples, batch_indices, max_len=max_len)
        _, loss = model.forward_multimodal(inputs, images, targets)
        if loss is None or not torch.isfinite(loss):
            raise ValueError("visual action pilot produced a non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.vision.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    movement = torch.sqrt(
        sum((value.detach() - initial[name]).float().pow(2).sum() for name, value in model.vision.named_parameters())
    )
    return {"steps": steps, "mean_train_loss": sum(losses) / max(1, len(losses)), "vision_update_l2": float(movement)}, initial


def _arm(parent_state: dict[str, Any] | None, cfg: ModelConfig, samples: list[dict[str, Any]], train_indices: list[int], eval_indices: list[int], *, warm: bool, steps: int, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = LocalAgentLM(cfg)
    if warm:
        missing, unexpected = model.load_state_dict(parent_state or {}, strict=False)
        if unexpected or any(not name.startswith("vision.") for name in missing):
            raise ValueError(f"warm checkpoint compatibility failure: missing={missing}, unexpected={unexpected}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.vision.parameters():
        parameter.requires_grad = True
    before = _metrics(model, samples, eval_indices, max_len=cfg.max_seq_len - cfg.vision_tokens)
    training, _ = _train(model, samples, train_indices, steps=steps, seed=seed, max_len=cfg.max_seq_len - cfg.vision_tokens)
    after = _metrics(model, samples, eval_indices, max_len=cfg.max_seq_len - cfg.vision_tokens)
    return {"warm": warm, "seed": seed, "before": before, "training": training, "after": after}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--prefix-bytes", type=int, default=50 * 1024 * 1024)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.prefix_bytes < 1 or args.steps < 1:
        raise ValueError("prefix_bytes and steps must be positive")
    raw_prefix = _download(OBJECT_URL, 0, args.prefix_bytes - 1)
    cfg = ModelConfig.from_yaml("configs/model/webgpu-10m-vision.yaml")
    samples, source_profile = _source_samples(raw_prefix, image_size=cfg.vision_image_size)
    episode_count = len(source_profile["episodes"])
    train_episode_count = max(1, episode_count - 4)
    train_indices = [i for i, sample in enumerate(samples) if sample["episode_index"] < train_episode_count]
    eval_indices = [i for i, sample in enumerate(samples) if sample["episode_index"] >= train_episode_count]
    if not train_indices or not eval_indices:
        raise ValueError("source split did not produce both train and eval samples")
    parent = torch.load(args.parent, map_location="cpu", weights_only=False)
    parent_state = parent.get("state_dict")
    if not isinstance(parent_state, dict):
        raise ValueError("parent checkpoint has no state_dict")
    warm = _arm(parent_state, cfg, samples, train_indices, eval_indices, warm=True, steps=args.steps, seed=712)
    random = _arm(None, cfg, samples, train_indices, eval_indices, warm=False, steps=args.steps, seed=713)
    payload = {
        "kind": "localagent_m712_androidcontrol_visual_action_pilot",
        "schema_version": 1,
        "source": {
            "dataset": "AndroidControl",
            "object_url": OBJECT_URL,
            "range_bytes": len(raw_prefix),
            "range_sha256": _sha256(raw_prefix),
            "records": source_profile,
            "official_split": "train source records; held-out by complete record",
            "screenshot_bytes_consumed": True,
        },
        "parent": {
            "path": str(args.parent),
            "sha256": _sha256(args.parent.read_bytes()),
            "parameters": int(sum(value.numel() for value in parent_state.values() if torch.is_tensor(value))),
        },
        "model": {
            "config": "configs/model/webgpu-10m-vision.yaml",
            "parameters": LocalAgentLM(cfg).num_params(),
            "vision_tokens": cfg.vision_tokens,
            "backbone_frozen": True,
        },
        "split": {
            "episodes_total": episode_count,
            "train_episodes": train_episode_count,
            "eval_episodes": episode_count - train_episode_count,
            "train_samples": len(train_indices),
            "eval_samples": len(eval_indices),
            "parent_disjoint": True,
        },
        "warm": warm,
        "random": random,
        "weight_analysis": {
            "warm_eval_loss_delta": warm["after"]["mean_action_token_loss"] - warm["before"]["mean_action_token_loss"],
            "random_eval_loss_delta": random["after"]["mean_action_token_loss"] - random["before"]["mean_action_token_loss"],
            "warm_minus_random_after_loss": warm["after"]["mean_action_token_loss"] - random["after"]["mean_action_token_loss"],
            "adoption_decision": "retain_warm_backbone_only_until_visual_action_quality_is_replicated",
        },
        "claim_boundary": (
            "Bounded AndroidControl screenshot/action transfer diagnostic only. The source records "
            "are public train data, the complete-record holdout is local, and the decoder backbone "
            "is frozen. This is not an official AndroidControl score, emulator result, native mobile "
            "success, or WebGPU visual deployment claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(payload["weight_analysis"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
