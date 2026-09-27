"""Model configuration + the <100M-param budget guard.

Beyond a vanilla decoder, the config exposes two structural levers that make the
**ultra-tiny (~1M)** tier actually feasible (see docs/ARCHITECTURE_IDEAS.md):

  * ``embed_dim``  — factorized embeddings (ALBERT-style). At tiny scale the vocab×d_model
    table dominates params; we instead learn vocab×embed_dim and up-project embed_dim→d_model.
    Set ``embed_dim = d_model`` (or null in YAML) to disable factorization.
  * ``n_loops``    — depth-recurrence (Universal Transformer / ALBERT cross-layer sharing). The
    block stack is run ``n_loops`` times with shared weights, so *effective depth =
    n_layers × n_loops* at the param cost of ``n_layers`` blocks.

A byte-level vocab (``vocab_size: 256``) is the third lever — it removes the embedding tax
entirely, which is what lets a 1M model spend its budget on computation instead of the vocab.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import yaml

PARAM_BUDGET = 100_000_000  # exclusive hard ceiling — configs must stay below this


@dataclass
class ModelConfig:
    name: str = "tiny-30m"
    vocab_size: int = 32000
    d_model: int = 384
    embed_dim: int | None = None  # None -> = d_model (no factorization)
    n_layers: int = 12  # number of blocks that hold parameters
    n_loops: int = 1  # depth-recurrence: run the stack this many times (shared weights)
    n_heads: int = 6
    n_kv_heads: int = 2
    ffn_hidden: int = 1024
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    dropout: float = 0.0
    # --- Hybrid backbone levers (LFM2/Nemotron-H + Qwen3 stability) ---
    # Defaults preserve the existing all-attention, QK-Norm-off model EXACTLY.
    qk_norm: bool = False  # RMSNorm on Q/K per-head before RoPE (Qwen3-style)
    conv_kernel: int = 3  # depthwise causal short-conv kernel (LFM2 LIV)
    layer_types: list[str] | None = None  # per-layer block kind: "attn"|"conv"; None => all "attn"
    # --- Sparse routed FFN (small-model MoE) ---
    # One expert is the legacy dense SwiGLU path: no router is allocated and state-dict keys stay
    # unchanged. With >1 experts, every block owns an independent router and expert bank.
    ffn_num_experts: int = 1
    ffn_top_k: int = 1
    router_aux_loss_coef: float = 0.0
    # --- Optional screenshot bridge ---
    # Disabled by default so existing text/WebGPU checkpoints keep the exact legacy state dict.
    # When enabled, a tiny patch encoder prepends visual tokens to the decoder context.
    vision_enabled: bool = False
    vision_image_size: int = 96
    vision_patch_size: int = 16
    vision_width: int = 64

    def __post_init__(self) -> None:
        if self.embed_dim is None:
            self.embed_dim = self.d_model
        assert self.d_model % self.n_heads == 0, "d_model must divide n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
        assert self.n_loops >= 1, "n_loops must be >= 1"
        if self.layer_types is not None:
            assert len(self.layer_types) == self.n_layers, (
                f"layer_types has {len(self.layer_types)} entries, expected n_layers={self.n_layers}"
            )
            assert all(t in ("attn", "conv") for t in self.layer_types), (
                "layer_types entries must be 'attn' or 'conv'"
            )
            assert any(t == "attn" for t in self.layer_types), (
                "keep >=1 attention layer for verbatim argument-copying"
            )
        assert isinstance(self.ffn_num_experts, int) and not isinstance(
            self.ffn_num_experts, bool
        ), "ffn_num_experts must be an integer"
        assert self.ffn_num_experts >= 1, "ffn_num_experts must be >= 1"
        assert isinstance(self.ffn_top_k, int) and not isinstance(self.ffn_top_k, bool), (
            "ffn_top_k must be an integer"
        )
        assert 1 <= self.ffn_top_k <= self.ffn_num_experts, (
            "ffn_top_k must be in [1, ffn_num_experts]"
        )
        assert math.isfinite(self.router_aux_loss_coef) and self.router_aux_loss_coef >= 0.0, (
            "router_aux_loss_coef must be finite and non-negative"
        )
        if self.vision_enabled:
            assert self.vision_image_size > 0, "vision_image_size must be positive"
            assert self.vision_patch_size > 0, "vision_patch_size must be positive"
            assert self.vision_image_size % self.vision_patch_size == 0, (
                "vision_image_size must be divisible by vision_patch_size"
            )
            assert self.vision_width > 0, "vision_width must be positive"
            assert self.max_seq_len > self.vision_tokens, (
                "max_seq_len must leave room for visual prefix tokens"
            )

    def block_types(self) -> list[str]:
        """Resolved per-layer block kinds; None => all attention (legacy behavior)."""
        return self.layer_types if self.layer_types is not None else ["attn"] * self.n_layers

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def factorized(self) -> bool:
        return self.embed_dim != self.d_model

    @property
    def effective_depth(self) -> int:
        return self.n_layers * self.n_loops

    @property
    def sparse_ffn(self) -> bool:
        """Whether blocks use routed expert banks instead of the legacy dense SwiGLU."""

        return self.ffn_num_experts > 1

    @property
    def vision_grid(self) -> int:
        return self.vision_image_size // self.vision_patch_size

    @property
    def vision_tokens(self) -> int:
        return self.vision_grid * self.vision_grid if self.vision_enabled else 0

    @classmethod
    def from_yaml(cls, path: str) -> ModelConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        fields = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        return cls(**fields)

    def _estimate_params(self, *, active_only: bool) -> int:
        d, h, kv, hd, k = self.d_model, self.n_heads, self.n_kv_heads, self.head_dim, self.embed_dim
        embed_table = self.vocab_size * k  # vocab × embed_dim (tied/shared)
        in_proj = k * d if self.factorized else 0
        out_proj = d * k if self.factorized else 0
        head = 0 if self.tie_embeddings else self.vocab_size * k
        loop_embed = self.n_loops * d if self.n_loops > 1 else 0
        expert = 3 * d * self.ffn_hidden  # SwiGLU: gate, up, down
        if self.sparse_ffn:
            # The d×E router scores every expert and is therefore always active. The total budget
            # owns E expert banks per block; a nominal token executes only top-k of those banks.
            routed_experts = self.ffn_top_k if active_only else self.ffn_num_experts
            ffn = d * self.ffn_num_experts + routed_experts * expert
        else:
            ffn = expert

        def attn_mixer() -> int:
            qk = 2 * hd if self.qk_norm else 0  # RMSNorm(head_dim) gains on Q and K (shared/head)
            return (
                d * (h * hd)  # q proj
                + 2 * d * (kv * hd)  # k, v proj (GQA → smaller)
                + (h * hd) * d  # out proj
                + qk
            )

        def conv_mixer() -> int:
            # 3 in-projections (B, C, h̃) of width d, depthwise causal conv (k weights/channel),
            # and an out-projection. No bias.
            return 3 * d * d + d * self.conv_kernel + d * d

        total_blocks = 0
        for kind in self.block_types():
            mixer = conv_mixer() if kind == "conv" else attn_mixer()
            total_blocks += mixer + ffn + 2 * d  # + two RMSNorm gains (pre-mixer, pre-ffn)
        final = d  # final norm
        vision = 0
        if self.vision_enabled:
            # Conv2d patch weights, LayerNorm gain/bias, visual-to-model projection, and positions.
            vision = (
                3 * self.vision_width * self.vision_patch_size * self.vision_patch_size
                + 2 * self.vision_width
                + self.vision_width * d
                + self.vision_tokens * d
            )
        return embed_table + in_proj + out_proj + head + loop_embed + total_blocks + final + vision

    def estimate_params(self) -> int:
        """Total stored parameters, including every routed expert.

        This is the count enforced by the hard budget. It intentionally does not substitute an
        "activated parameters" figure for the actual checkpoint size.
        """

        return self._estimate_params(active_only=False)

    def estimate_active_params(self) -> int:
        """Nominal parameters on one token's routed path.

        The count includes all shared/dense parameters, the complete router in every parameter
        block, and ``ffn_top_k`` expert banks per block. Tied and recurrently shared weights are
        counted once, matching :meth:`estimate_params`. A batch or recurrent execution may route
        different tokens to every expert, so this is not a peak-memory or checkpoint-size claim.
        """

        return self._estimate_params(active_only=True)

    def assert_within_budget(self) -> None:
        n = self.estimate_params()
        # LOCALAGENT_PARAM_BUDGET raises the ceiling for explicitly over-budget ablations (the
        # 64k-vocabulary variants); the default keeps the sub-100M contract for everything else.
        import os
        budget = int(os.environ.get("LOCALAGENT_PARAM_BUDGET", PARAM_BUDGET))
        if n >= budget:
            raise ValueError(
                f"Model '{self.name}' is ~{n / 1e6:.1f}M params; "
                f"the budget requires fewer than {budget / 1e6:.0f}M."
            )

    def estimate_cache_bytes(self, context_len: int, dtype_bytes: int = 2) -> int:
        """Inference-state footprint for a batch of one at a given context length.

        Attention layers store K and V for every context token. Short-conv layers only retain
        ``kernel-1`` states. The estimate includes every recurrent loop pass.
        """

        if not 0 <= context_len <= self.max_seq_len:
            raise ValueError(f"context_len must be in [0, {self.max_seq_len}]")
        if dtype_bytes < 1:
            raise ValueError("dtype_bytes must be positive")
        per_loop = 0
        for kind in self.block_types():
            if kind == "attn":
                per_loop += 2 * self.n_kv_heads * self.head_dim * context_len * dtype_bytes
            else:
                per_loop += self.d_model * max(0, self.conv_kernel - 1) * dtype_bytes
        return per_loop * self.n_loops

    def estimate_weight_bytes(self, bits: int = 16) -> int:
        """Packed weight footprint (runtime overhead and alignment excluded)."""

        if bits <= 0:
            raise ValueError("bits must be positive")
        return (self.estimate_params() * bits + 7) // 8

    def estimate_active_weight_bytes(self, bits: int = 16) -> int:
        """Nominal routed-path weight bytes; not checkpoint size or peak runtime memory."""

        if bits <= 0:
            raise ValueError("bits must be positive")
        return (self.estimate_active_params() * bits + 7) // 8
