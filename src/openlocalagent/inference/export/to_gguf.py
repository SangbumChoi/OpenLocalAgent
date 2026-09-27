"""Export to GGUF for llama.cpp (Phase 9) — primary cross-platform CPU/GPU/NPU path.

Map our Llama-style weights to GGUF tensors/metadata; default Q4_0 quantization. Runs on
CPU + Apple Silicon + Intel/AMD GPU/NPU (OpenVINO is upstream in llama.cpp).
"""

from __future__ import annotations


def export(checkpoint: str, out_path: str, quant: str = "q4_0") -> None:
    raise NotImplementedError("TODO(phase-9): write GGUF (metadata + quantized tensors)")
