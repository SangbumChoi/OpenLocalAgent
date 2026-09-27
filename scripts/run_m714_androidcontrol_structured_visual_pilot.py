#!/usr/bin/env python3
"""Train a structured screenshot-conditioned Android action head on a bounded public split."""

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
from localagent.model.vision import ANDROID_ACTIONS, VisualActionHead


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _records(gzip_prefix: bytes) -> list[bytes]:
    decompressed = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(gzip_prefix)
    records: list[bytes] = []
    offset = 0
    while offset + 12 <= len(decompressed):
        length = struct.unpack("<Q", decompressed[offset : offset + 8])[0]
        end = offset + 12 + length + 4
        if end > len(decompressed):
            break
        records.append(decompressed[offset + 12 : offset + 12 + length])
        offset = end
    return records


def _load_samples(prefix: bytes, *, image_size: int, max_seq_len: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    for record_index, record in enumerate(_records(prefix)):
        features = _feature_map(record)
        screenshots = _bytes_list(features.get("screenshots", b""))
        actions = _bytes_list(features.get("actions", b""))
        goals = _bytes_list(features.get("goal", b""))
        widths = _int64_list(features.get("screenshot_widths", b""))
        heights = _int64_list(features.get("screenshot_heights", b""))
        if not screenshots or not actions or not goals:
            continue
        goal = goals[0].decode("utf-8", errors="replace").strip()
        usable = min(len(actions), len(screenshots))
        episode = {
            "record_index": record_index,
            "record_sha256": _hash(record),
            "record_bytes": len(record),
            "episode_id": (_int64_list(features["episode_id"]) or [None])[0],
            "goal": goal,
            "steps": usable,
            "screenshot_sha256": [_hash(item) for item in screenshots[:usable]],
        }
        episodes.append(episode)
        for step_index in range(usable):
            action = json.loads(actions[step_index].decode("utf-8"))
            action_type = str(action.get("action_type", ""))
            if action_type not in ANDROID_ACTIONS:
                continue
            image = decode_png_rgb(screenshots[step_index]).unsqueeze(0)
            image = F.interpolate(image, size=(image_size, image_size), mode="bilinear", align_corners=False).squeeze(0)
            context = f"Task: {goal}\nAction: ".encode("utf-8")
            ids = torch.tensor(list(context[: max_seq_len - 1]), dtype=torch.long)
            width = float(widths[step_index]) if step_index < len(widths) and widths[step_index] else 1080.0
            height = float(heights[step_index]) if step_index < len(heights) and heights[step_index] else 2400.0
            coordinate = None
            if "x" in action and "y" in action:
                coordinate = [float(action["x"]) / width, float(action["y"]) / height]
            samples.append(
                {
                    "episode_index": len(episodes) - 1,
                    "record_index": record_index,
                    "step_index": step_index,
                    "image": image,
                    "ids": ids,
                    "action": ANDROID_ACTIONS.index(action_type),
                    "action_type": action_type,
                    "coordinate": coordinate,
                }
            )
    if len(episodes) < 4:
        raise ValueError("bounded source produced fewer than four complete episodes")
    return samples, episodes


def _batch(samples: list[dict[str, Any]], indices: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[float] | None]]:
    chosen = [samples[index] for index in indices]
    width = max(len(item["ids"]) for item in chosen)
    ids = torch.zeros((len(chosen), width), dtype=torch.long)
    for row, item in enumerate(chosen):
        ids[row, : len(item["ids"])] = item["ids"]
    images = torch.stack([item["image"] for item in chosen])
    labels = torch.tensor([item["action"] for item in chosen], dtype=torch.long)
    coordinates = [item["coordinate"] for item in chosen]
    return images, ids, labels, coordinates


def _features(model: LocalAgentLM, head: VisualActionHead, samples: list[dict[str, Any]], indices: list[int]) -> tuple[torch.Tensor, torch.Tensor, list[list[float] | None]]:
    model.eval()
    head.eval()
    actions: list[torch.Tensor] = []
    pointers: list[torch.Tensor] = []
    coords: list[list[float] | None] = []
    with torch.no_grad():
        for start in range(0, len(indices), 4):
            images, ids, _, batch_coords = _batch(samples, indices[start : start + 4])
            _, _, text_hidden, visual = model.forward_multimodal(ids, images, return_hidden=True)
            context_position = torch.tensor(
                [min(len(samples[index]["ids"]) - 1, text_hidden.shape[1] - 1) for index in indices[start : start + 4]],
                dtype=torch.long,
            )
            row = torch.arange(text_hidden.shape[0])
            action, pointer = head(text_hidden[row, context_position], visual)
            actions.append(action.cpu())
            pointers.append(pointer.cpu())
            coords.extend(batch_coords)
    return torch.cat(actions), torch.cat(pointers), coords


def _metrics(model: LocalAgentLM, head: VisualActionHead, samples: list[dict[str, Any]], indices: list[int]) -> dict[str, Any]:
    action_logits, pointer, coords = _features(model, head, samples, indices)
    labels = torch.tensor([samples[index]["action"] for index in indices])
    action_loss = float(F.cross_entropy(action_logits, labels))
    correct = int(action_logits.argmax(-1).eq(labels).sum())
    pointer_errors = [
        float(torch.abs(pointer[row] - torch.tensor(coord)).mean())
        for row, coord in enumerate(coords)
        if coord is not None
    ]
    return {
        "samples": len(indices),
        "action_loss": action_loss,
        "action_accuracy": correct / max(1, len(indices)),
        "coordinate_mae": sum(pointer_errors) / max(1, len(pointer_errors)),
        "coordinate_rows": len(pointer_errors),
    }


def _run_arm(parent_state: dict[str, Any] | None, cfg: ModelConfig, samples: list[dict[str, Any]], train: list[int], eval_rows: list[int], *, warm: bool, steps: int, seed: int, head_state: dict[str, Any]) -> tuple[dict[str, Any], LocalAgentLM, VisualActionHead]:
    torch.manual_seed(seed)
    model = LocalAgentLM(cfg)
    if warm:
        missing, unexpected = model.load_state_dict(parent_state or {}, strict=False)
        if unexpected or any(not name.startswith("vision.") for name in missing):
            raise ValueError(f"warm backbone mismatch: missing={missing}, unexpected={unexpected}")
    head = VisualActionHead(cfg.d_model)
    head.load_state_dict(head_state)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.vision.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW([*model.vision.parameters(), *head.parameters()], lr=1e-3)
    generator = torch.Generator().manual_seed(seed)
    before = _metrics(model, head, samples, eval_rows)
    losses: list[float] = []
    model.train()
    head.train()
    for _ in range(steps):
        chosen = torch.randint(0, len(train), (min(4, len(train)),), generator=generator).tolist()
        images, ids, labels, coords = _batch(samples, [train[index] for index in chosen])
        _, _, text_hidden, visual = model.forward_multimodal(ids, images, return_hidden=True)
        row = torch.arange(text_hidden.shape[0])
        context_position = torch.tensor([len(samples[train[index]]["ids"]) - 1 for index in chosen])
        action_logits, pointer = head(text_hidden[row, context_position], visual)
        loss = F.cross_entropy(action_logits, labels)
        coordinate_rows = [i for i, coord in enumerate(coords) if coord is not None]
        if coordinate_rows:
            target = torch.tensor([coords[i] for i in coordinate_rows], dtype=pointer.dtype)
            loss = loss + F.mse_loss(pointer[coordinate_rows], target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*model.vision.parameters(), *head.parameters()], 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    after = _metrics(model, head, samples, eval_rows)
    report = {
        "warm": warm,
        "seed": seed,
        "training": {"steps": steps, "mean_loss": sum(losses) / max(1, len(losses))},
        "before": before,
        "after": after,
    }
    return report, model, head


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--prefix-bytes", type=int, default=50 * 1024 * 1024)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--warm-checkpoint", type=Path)
    parser.add_argument("--random-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = _download(OBJECT_URL, 0, args.prefix_bytes - 1)
    cfg = ModelConfig.from_yaml("configs/model/webgpu-10m-vision.yaml")
    samples, episodes = _load_samples(raw, image_size=cfg.vision_image_size, max_seq_len=cfg.max_seq_len - cfg.vision_tokens)
    train_episodes = max(1, len(episodes) - 4)
    train = [i for i, sample in enumerate(samples) if sample["episode_index"] < train_episodes]
    eval_rows = [i for i, sample in enumerate(samples) if sample["episode_index"] >= train_episodes]
    parent = torch.load(args.parent, map_location="cpu", weights_only=False)
    parent_state = parent.get("state_dict")
    if not isinstance(parent_state, dict):
        raise ValueError("parent checkpoint has no state_dict")
    torch.manual_seed(714)
    head_state = VisualActionHead(cfg.d_model).state_dict()
    warm, warm_model, warm_head = _run_arm(parent_state, cfg, samples, train, eval_rows, warm=True, steps=args.steps, seed=714, head_state=head_state)
    random, random_model, random_head = _run_arm(None, cfg, samples, train, eval_rows, warm=False, steps=args.steps, seed=715, head_state=head_state)
    for path, model, head, arm in (
        (args.warm_checkpoint, warm_model, warm_head, "warm"),
        (args.random_checkpoint, random_model, random_head, "random"),
    ):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"cfg": cfg.__dict__, "state_dict": model.state_dict(), "head_state": head.state_dict(), "action_names": list(ANDROID_ACTIONS), "arm": arm}, path)
    payload = {
        "kind": "localagent_m714_androidcontrol_structured_visual_pilot",
        "schema_version": 1,
        "source": {
            "dataset": "AndroidControl",
            "object_url": OBJECT_URL,
            "range_bytes": len(raw),
            "range_sha256": _hash(raw),
            "episodes": episodes,
            "train_episodes": train_episodes,
            "eval_episodes": len(episodes) - train_episodes,
            "screenshot_bytes_consumed": True,
        },
        "parent": {"path": str(args.parent), "sha256": _hash(args.parent.read_bytes())},
        "model": {"config": "configs/model/webgpu-10m-vision.yaml", "parameters": LocalAgentLM(cfg).num_params(), "backbone_frozen": True, "action_names": list(ANDROID_ACTIONS)},
        "split": {"train_samples": len(train), "eval_samples": len(eval_rows), "parent_disjoint": True},
        "warm": warm,
        "random": random,
        "weight_analysis": {
            "warm_minus_random_after_action_accuracy": warm["after"]["action_accuracy"] - random["after"]["action_accuracy"],
            "warm_minus_random_after_coordinate_mae": warm["after"]["coordinate_mae"] - random["after"]["coordinate_mae"],
            "adoption_decision": "do_not_promote_structured_visual_head_until_native_emulator_validation",
        },
        "claim_boundary": (
            "Structured AndroidControl action-type/coordinate transfer diagnostic only. The source "
            "holdout is complete-record disjoint, but no emulator, native verifier, official score, "
            "or WebGPU visual export was run."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["weight_analysis"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
