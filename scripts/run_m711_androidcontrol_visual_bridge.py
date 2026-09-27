#!/usr/bin/env python3
"""Run one public AndroidControl screenshot through the optional visual bridge.

This is a wiring/training smoke, not a quality result: it fetches only the first bounded TFRecord
range, decodes one screenshot, performs one frozen-backbone visual-prefix update, and stores hashes
and tensor statistics without retaining image pixels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent))
from audit_androidcontrol_tfrecord_sample import (  # noqa: E402
    OBJECT_URL,
    _bytes_list,
    _download,
    _feature_map,
    _first_tfrecord,
)
from localagent.data.visual import decode_png_rgb
from localagent.model import LocalAgentLM, ModelConfig


def identity(data: bytes) -> dict[str, Any]:
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-bytes", type=int, default=10 * 1024 * 1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.prefix_bytes < 1:
        raise ValueError("prefix_bytes must be positive")
    raw_prefix = _download(OBJECT_URL, 0, args.prefix_bytes - 1)
    record = _first_tfrecord(raw_prefix)
    features = _feature_map(record)
    screenshot = _bytes_list(features["screenshots"])[0]
    goal = _bytes_list(features["goal"])[0].decode("utf-8", errors="replace")
    image = decode_png_rgb(screenshot).unsqueeze(0)
    cfg = ModelConfig.from_yaml("configs/model/webgpu-10m-vision.yaml")
    torch.manual_seed(711)
    model = LocalAgentLM(cfg)
    assert model.vision is not None
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.vision.parameters():
        parameter.requires_grad = True
    token_values = list(goal.encode("utf-8"))[: cfg.max_seq_len - cfg.vision_tokens]
    idx = torch.tensor([token_values], dtype=torch.long)
    optimizer = torch.optim.AdamW(model.vision.parameters(), lr=1e-3)
    model.train()
    logits, loss, _, visual = model.forward_multimodal(idx, image, idx, return_hidden=True)
    if loss is None or not torch.isfinite(loss):
        raise ValueError("visual bridge loss is not finite")
    loss.backward()
    grad_norm = float(
        torch.sqrt(
            sum(
                parameter.grad.detach().float().pow(2).sum()
                for parameter in model.vision.parameters()
                if parameter.grad is not None
            )
        )
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.vision.named_parameters()
    }
    optimizer.step()
    movement = torch.sqrt(
        sum(
            (parameter.detach() - before[name]).float().pow(2).sum()
            for name, parameter in model.vision.named_parameters()
        )
    )
    payload = {
        "kind": "localagent_m711_androidcontrol_visual_bridge_smoke",
        "schema_version": 1,
        "source": {
            "dataset": "AndroidControl",
            "object_url": OBJECT_URL,
            "range_start": 0,
            "range_bytes": len(raw_prefix),
            "range_sha256": hashlib.sha256(raw_prefix).hexdigest(),
            "record": identity(record),
            "screenshot": identity(screenshot),
        },
        "model": {
            "config": "configs/model/webgpu-10m-vision.yaml",
            "parameters": model.num_params(),
            "vision_tokens": cfg.vision_tokens,
            "image_tensor_shape": list(image.shape),
            "visual_token_shape": list(visual.shape),
            "logit_shape": list(logits.shape),
            "pixel_min": float(image.min()),
            "pixel_max": float(image.max()),
        },
        "training": {
            "backbone_frozen": True,
            "target_tokens": len(token_values),
            "one_update_loss": float(loss.detach()),
            "vision_grad_norm": grad_norm,
            "vision_parameter_update_l2": float(movement),
        },
        "pipeline_boundary": {
            "screenshot_bytes_consumed": True,
            "vision_bridge_forward": True,
            "vision_bridge_update": True,
            "native_emulator_executed": False,
            "webgpu_exported": False,
            "quality_training_admitted": False,
        },
        "claim_boundary": (
            "This receipt proves that one bounded public AndroidControl PNG can be decoded and "
            "condition a budget-compliant visual prefix with a nonzero frozen-backbone update. It "
            "does not claim visual action accuracy, Android emulator success, AndroidWorld/MobileGym "
            "performance, WebGPU export, or a trained visual checkpoint."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["training"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
