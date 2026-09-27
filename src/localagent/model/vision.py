"""Small screenshot-to-token bridge for optional multimodal experiments.

The bridge is deliberately independent of a pretrained vision tower: it is a compact patch
encoder that stays inside the model budget and can be trained from public screenshot/action rows.
It is disabled in all legacy configs unless ``vision_enabled`` is set explicitly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from localagent.model.config import ModelConfig


ANDROID_ACTIONS = (
    "click",
    "input_text",
    "long_press",
    "navigate_back",
    "open_app",
    "scroll",
    "wait",
)


class VisualPatchEncoder(nn.Module):
    """Encode ``[0, 1]`` RGB screenshots into a fixed visual-token prefix."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if not cfg.vision_enabled:
            raise ValueError("VisualPatchEncoder requires vision_enabled=True")
        self.image_size = cfg.vision_image_size
        self.patch = nn.Conv2d(
            3,
            cfg.vision_width,
            kernel_size=cfg.vision_patch_size,
            stride=cfg.vision_patch_size,
            bias=False,
        )
        self.norm = nn.LayerNorm(cfg.vision_width, eps=cfg.norm_eps)
        self.proj = nn.Linear(cfg.vision_width, cfg.d_model, bias=False)
        self.position = nn.Parameter(torch.zeros(cfg.vision_tokens, cfg.d_model))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [batch, 3, height, width]")
        if not images.is_floating_point():
            raise TypeError("images must be floating-point tensors in [0, 1]")
        if not torch.isfinite(images).all():
            raise ValueError("images contain non-finite values")
        if images.numel() and (float(images.min()) < 0.0 or float(images.max()) > 1.0):
            raise ValueError("images must be in [0, 1]")
        resized = F.interpolate(
            images,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        patches = self.patch(resized).flatten(2).transpose(1, 2)
        tokens = self.proj(self.norm(patches))
        return tokens + self.position.unsqueeze(0).to(tokens)


class VisualActionHead(nn.Module):
    """Structured mobile action head over one text-context feature and visual tokens.

    The action vocabulary mirrors AndroidControl's JSON action contract.  ``pointer`` predicts
    normalized ``x,y`` coordinates for click/long-press rows; callers should ignore it for actions
    without coordinates.  This sidecar is intentionally separate from the legacy text heads until
    its native emulator/export contract is verified.
    """

    def __init__(self, d_model: int, action_names: tuple[str, ...] = ANDROID_ACTIONS):
        super().__init__()
        self.action_names = tuple(action_names)
        if not self.action_names or len(set(self.action_names)) != len(self.action_names):
            raise ValueError("visual action names must be unique and non-empty")
        self.fuse = nn.Sequential(
            nn.Linear(2 * d_model, d_model, bias=False),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.action = nn.Linear(d_model, len(self.action_names))
        self.pointer = nn.Linear(d_model, 2)

    def forward(
        self,
        text_feature: torch.Tensor,
        visual_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if text_feature.ndim != 2 or visual_tokens.ndim != 3:
            raise ValueError("visual action inputs must be [batch, d_model] and [batch, tokens, d_model]")
        if text_feature.shape[0] != visual_tokens.shape[0] or text_feature.shape[1] * 2 != self.fuse[0].in_features:
            raise ValueError("visual action feature widths do not match")
        pooled = visual_tokens.mean(dim=1)
        fused = self.fuse(torch.cat([text_feature, pooled], dim=-1))
        return self.action(fused), self.pointer(fused).sigmoid()
