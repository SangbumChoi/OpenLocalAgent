"""ONNX export/parity helpers for the optional screenshot-conditioned action sidecar.

This ABI is deliberately separate from the text-only WebGPU bundle: inputs are tokenized context,
context lengths, and normalized RGB images; outputs are Android action logits and normalized pointer
coordinates.  It does not claim a browser runtime until a JavaScript consumer binds these names.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.vision import VisualActionHead


class VisualActionExport(nn.Module):
    """ONNX wrapper with explicit image/context/action tensor names."""

    def __init__(self, model: LocalAgentLM, head: VisualActionHead):
        super().__init__()
        if model.vision is None:
            raise ValueError("visual action export requires vision_enabled=True")
        self.model = model
        self.head = head

    def forward(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor,
        context_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, text_hidden, visual = self.model.forward_multimodal(input_ids, images, return_hidden=True)
        positions = (context_lengths - 1).clamp(min=0, max=text_hidden.shape[1] - 1)
        rows = torch.arange(text_hidden.shape[0], device=text_hidden.device)
        return self.head(text_hidden[rows, positions], visual)


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def load_sidecar(path: str | Path) -> tuple[LocalAgentLM, VisualActionHead, dict[str, Any]]:
    """Load a sidecar saved by the structured visual pilot."""

    source = Path(path)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**payload["cfg"])
    model = LocalAgentLM(cfg)
    model.load_state_dict(payload["state_dict"], strict=True)
    head = VisualActionHead(cfg.d_model, tuple(payload.get("action_names", ())))
    head.load_state_dict(payload["head_state"], strict=True)
    model.eval()
    head.eval()
    return model, head, {"sidecar": _identity(source), "cfg": cfg.__dict__, "action_names": head.action_names}


def export_visual_action_onnx(
    sidecar: str | Path,
    output: str | Path,
    *,
    sequence_length: int = 128,
    image_size: int | None = None,
    opset: int = 17,
    check: bool = True,
) -> dict[str, Any]:
    """Export and parity-check one visual action sidecar."""

    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    model, head, metadata = load_sidecar(sidecar)
    cfg = model.cfg
    size = image_size or cfg.vision_image_size
    if size != cfg.vision_image_size:
        raise ValueError("image_size must equal the checkpoint visual image size")
    wrapper = VisualActionExport(model, head).eval()
    ids = torch.zeros((1, sequence_length), dtype=torch.long)
    images = torch.zeros((1, 3, size, size), dtype=torch.float32)
    lengths = torch.tensor([min(sequence_length, 32)], dtype=torch.long)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (ids, images, lengths),
        out,
        opset_version=opset,
        dynamo=False,
        input_names=["input_ids", "images", "context_lengths"],
        output_names=["action_logits", "pointer_xy"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "images": {0: "batch"},
            "context_lengths": {0: "batch"},
            "action_logits": {0: "batch"},
            "pointer_xy": {0: "batch"},
        },
    )
    result: dict[str, Any] = {
        "artifact": _identity(out),
        "inputs": {"input_ids": list(ids.shape), "images": list(images.shape), "context_lengths": list(lengths.shape)},
        "outputs": {"action_logits": [1, len(head.action_names)], "pointer_xy": [1, 2]},
        "metadata": metadata,
        "parity": {"checked": False},
    }
    if check:
        import numpy as np
        import onnxruntime as ort

        with torch.no_grad():
            expected = wrapper(ids, images, lengths)
        session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        got = session.run(
            ["action_logits", "pointer_xy"],
            {"input_ids": ids.numpy(), "images": images.numpy(), "context_lengths": lengths.numpy()},
        )
        diffs = [float(np.max(np.abs(reference.numpy() - observed))) for reference, observed in zip(expected, got)]
        result["parity"] = {"checked": True, "max_abs_diffs": diffs, "passed": all(diff < 1e-3 for diff in diffs)}
        if not result["parity"]["passed"]:
            raise ValueError(f"visual action ONNX parity failed: {diffs}")
    return result
