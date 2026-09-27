"""The tiny decoder: pre-norm RMSNorm + GQA(RoPE) + SwiGLU, tied factorized embeddings,
optional depth-recurrence, and a real KV cache for prefill + incremental decode.

Training/prefill path: full sequence, causal SDPA, pos=0, cache=None  (numerically vanilla).
Decode path:           one token at a time, attends to the cached K/V (state-of-the-art-ish
                       prefill-then-decode). With depth-recurrence each (loop, layer) pass keeps
                       its own cache slot.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from localagent.model.config import ModelConfig
from localagent.model.vision import VisualPatchEncoder

# One cache slot per loop×layer pass: attention stores (k, v), short-conv stores its input tail.
KVCache = list


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm.type_as(x) * self.weight


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim/2)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, n_heads, T, head_dim); cos/sin: (T, head_dim/2) already sliced to abs positions
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack((rx1, rx2), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.rep = cfg.n_heads // cfg.n_kv_heads
        self.q = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        # QK-Norm (Qwen3-style): per-head RMSNorm on Q and K before RoPE. Off by default so
        # legacy models are byte-identical (the gains are not even allocated when disabled).
        if cfg.qk_norm:
            self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
            self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, cos, sin, cache=None):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv_heads, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv_heads, self.hd).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            pk, pv = cache
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_cache = (k, v)
        k = k.repeat_interleave(self.rep, dim=1)  # GQA expand
        v = v.repeat_interleave(self.rep, dim=1)
        # full-sequence prefill/training (no cache) is causal; a cached single-step decode query
        # attends to all cached keys, so no mask. (Python bool -> ONNX-exportable.)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=cache is None)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.o(out), new_cache


class GatedShortConv(nn.Module):
    """LFM2 LIV-style double-gated depthwise short-conv mixer (a cheap, CPU-friendly
    sub-quadratic alternative to attention).

        B, C, h̃ = Linear(x)              # three width-d projections
        y       = B ⊙ h̃                  # input gate
        z       = DepthwiseCausalConv1d(y, k)   # per-channel causal conv (left-pad k-1)
        o       = Linear_out(C ⊙ z)      # output gate

    Decode: a short causal conv only needs the last k-1 inputs per channel, so the cache slot
    holds a (B, d, k-1) ring of prior `y` values → O(1) single-token decode.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, self.k = cfg.d_model, cfg.conv_kernel
        self.in_proj = nn.Linear(d, 3 * d, bias=False)   # -> B, C, h̃
        # depthwise (groups=d) causal conv; we left-pad manually so decode-state is exact.
        self.conv = nn.Conv1d(d, d, kernel_size=self.k, groups=d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin, cache=None, use_cache: bool = False):
        B, T, d = x.shape
        gb, gc, h = self.in_proj(x).chunk(3, dim=-1)
        y = (gb * h).transpose(1, 2)                     # (B, d, T)
        if cache is None:
            pad = F.pad(y, (self.k - 1, 0))              # causal left-pad
        else:
            # cache holds the prior (k-1) `y` columns; prepend them before the new inputs.
            pad = torch.cat([cache, y], dim=2)
        cache_enabled = use_cache or cache is not None
        if cache_enabled:
            # Slicing the padded prefill also retains leading zeros when T < k-1.
            new_cache = (
                pad[:, :, -(self.k - 1):].contiguous()
                if self.k > 1
                else pad[:, :, :0].contiguous()
            )
        else:
            new_cache = None
        z = self.conv(pad).transpose(1, 2)               # (B, T, d), causal
        o = self.out_proj(gc * z)
        return o, new_cache


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class SparseSwiGLU(nn.Module):
    """Token-routed bank of SwiGLU experts with deterministic sparse dispatch.

    Router logits are ranked with a stable descending sort, so exact ties prefer the lower expert
    index. Only selected token/expert pairs execute an expert; this PyTorch path deliberately does
    not compute every expert and mask afterward. The dynamic index dispatch is not currently a
    promise that LocalAgent's ONNX/WebGPU exporter will retain sparse compute -- runtimes need a
    dedicated sparse-dispatch lowering before making that performance claim.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_experts = cfg.ffn_num_experts
        self.top_k = cfg.ffn_top_k
        self.router = nn.Linear(cfg.d_model, self.num_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLU(cfg) for _ in range(self.num_experts))
        self._last_aux_loss: torch.Tensor | None = None
        self._last_routing_record: dict[str, object] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        router_logits = self.router(flat)
        # Keep probability/auxiliary-loss arithmetic in fp32 even when experts run in fp16/bf16.
        router_probs = F.softmax(router_logits.float(), dim=-1)
        # Stable ordering makes tie behavior explicit: expert 0 precedes expert 1, and so on.
        ranked = torch.argsort(router_logits.float(), dim=-1, descending=True, stable=True)
        selected = ranked[:, : self.top_k]
        selected_probs = router_probs.gather(-1, selected)
        if self.top_k == 1:
            # A raw Switch-style gate keeps a task-loss gradient into a top-1 router.
            gates = selected_probs
        else:
            gates = selected_probs / selected_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        output = torch.zeros_like(flat)
        for expert_index, expert in enumerate(self.experts):
            token_and_slot = (selected == expert_index).nonzero(as_tuple=False)
            if token_and_slot.numel() == 0:
                continue
            token_index = token_and_slot[:, 0]
            slot_index = token_and_slot[:, 1]
            expert_input = flat.index_select(0, token_index)
            expert_output = expert(expert_input)
            expert_gate = gates[token_index, slot_index].to(expert_output.dtype).unsqueeze(-1)
            # token_index is unique within one expert, and experts are accumulated serially.
            output = output.index_add(0, token_index, expert_output * expert_gate)

        counts = torch.bincount(selected.reshape(-1), minlength=self.num_experts)
        assignment_load = counts.to(router_probs.dtype) / max(1, selected.numel())
        mean_probability = router_probs.mean(dim=0)
        # Switch-style differentiable load-balancing objective. Hard assignment load is detached;
        # router probability retains gradients. Uniform routing has a baseline value of 1.
        aux_loss = self.num_experts * torch.sum(assignment_load.detach() * mean_probability)
        entropy = -(router_probs * router_probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
        self._last_aux_loss = aux_loss
        self._last_routing_record = {
            "tokens": flat.shape[0],
            "expert_counts": counts.detach(),
            "router_probability": mean_probability.detach(),
            "router_entropy": entropy.detach(),
            "load_balance_loss": aux_loss.detach(),
        }
        return output.reshape(shape)

    def routing_aux_loss(self) -> torch.Tensor | None:
        """Differentiable load-balancing loss from this expert bank's latest invocation."""

        return self._last_aux_loss

    def routing_record(self) -> dict[str, object] | None:
        """Detached routing telemetry from this expert bank's latest invocation."""

        return self._last_routing_record


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, kind: str = "attn"):
        super().__init__()
        self.kind = kind
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        # The mixer is either GQA attention or the gated short-conv; both share the
        # (x, cos, sin, cache) -> (out, new_cache) contract so dispatch is uniform.
        self.attn = GatedShortConv(cfg) if kind == "conv" else Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SparseSwiGLU(cfg) if cfg.sparse_ffn else SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None, use_cache: bool = False):
        mixer_input = self.attn_norm(x)
        if self.kind == "conv":
            a, new_cache = self.attn(mixer_input, cos, sin, cache, use_cache=use_cache)
        else:
            a, new_cache = self.attn(mixer_input, cos, sin, cache)
        x = x + a
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


class LocalAgentLM(nn.Module):
    """Tiny decoder with optional factorized embeddings + depth-recurrence + KV cache."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.assert_within_budget()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.in_proj = nn.Linear(cfg.embed_dim, cfg.d_model, bias=False) if cfg.factorized else None
        self.out_proj = nn.Linear(cfg.d_model, cfg.embed_dim, bias=False) if cfg.factorized else None
        self.blocks = nn.ModuleList(Block(cfg, kind) for kind in cfg.block_types())
        self.loop_embed = (
            nn.Parameter(torch.zeros(cfg.n_loops, cfg.d_model)) if cfg.n_loops > 1 else None
        )
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = (
            None if cfg.tie_embeddings else nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        )
        self.vision = VisualPatchEncoder(cfg) if cfg.vision_enabled else None
        self._rope = None
        self._last_routing_aux_losses: list[torch.Tensor] = []
        self._last_routing_records: list[dict[str, object]] = []
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Conv1d):  # depthwise short-conv (hybrid blocks only)
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _rope_slice(self, pos: int, T: int, device, dtype):
        if self._rope is None or self._rope[0].device != device or self._rope[0].dtype != dtype:
            self._rope = build_rope_cache(
                self.cfg.max_seq_len, self.cfg.head_dim, self.cfg.rope_theta, device, dtype
            )
        cos, sin = self._rope
        return cos[pos:pos + T], sin[pos:pos + T]

    def n_cache_slots(self) -> int:
        return self.cfg.n_loops * self.cfg.n_layers

    def forward_features(
        self,
        idx: torch.Tensor,
        pos: int = 0,
        caches=None,
        prefix_embeds: torch.Tensor | None = None,
    ):
        """Run the embedding and decoder backbone, returning post-final-norm features.

        This deliberately stops before the factorized output projection and LM head.  Action
        heads consume these ``d_model``-wide features directly, so browser runtimes can avoid
        materializing a ``vocab_size``-wide logits tensor when text generation is not needed.

        As with :meth:`forward`, passing ``caches`` enables the cached path and returns
        ``(features, new_caches)``.  Without caches, only ``features`` is returned.
        """
        return_cache = caches is not None
        if caches is None:
            caches = [None] * self.n_cache_slots()
        x = self.embed(idx)
        if self.in_proj is not None:
            x = self.in_proj(x)
        if prefix_embeds is not None:
            if return_cache:
                raise ValueError("visual prefixes are only supported for uncached prefill")
            if prefix_embeds.ndim != 3 or prefix_embeds.shape[0] != x.shape[0]:
                raise ValueError("prefix_embeds must have shape [batch, tokens, d_model]")
            if prefix_embeds.shape[2] != self.cfg.d_model:
                raise ValueError("prefix_embeds width must equal cfg.d_model")
            x = torch.cat([prefix_embeds.to(dtype=x.dtype, device=x.device), x], dim=1)
        if x.shape[1] > self.cfg.max_seq_len:
            raise ValueError("input sequence including visual prefix exceeds max_seq_len")
        cos, sin = self._rope_slice(pos, x.shape[1], x.device, x.dtype)
        new_caches = [None] * self.n_cache_slots()
        self._last_routing_aux_losses = []
        self._last_routing_records = []
        slot = 0
        for loop in range(self.cfg.n_loops):
            if self.loop_embed is not None:
                x = x + self.loop_embed[loop]
            for blk in self.blocks:
                x, nc = blk(x, cos, sin, caches[slot], use_cache=return_cache)
                if isinstance(blk.ffn, SparseSwiGLU):
                    aux_loss = blk.ffn.routing_aux_loss()
                    record = blk.ffn.routing_record()
                    if aux_loss is not None:
                        self._last_routing_aux_losses.append(aux_loss)
                    if record is not None:
                        self._last_routing_records.append(record)
                new_caches[slot] = nc
                slot += 1
        feats = self.norm(x)
        if return_cache:
            return feats, new_caches
        return feats

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Return screenshot tokens for an explicitly vision-enabled model."""

        if self.vision is None:
            raise RuntimeError("this checkpoint was built without vision_enabled=True")
        return self.vision(images)

    def forward_multimodal(
        self,
        idx: torch.Tensor,
        images: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_hidden: bool = False,
    ):
        """Run text generation conditioned on a screenshot prefix.

        ``targets`` and returned logits cover text tokens only; visual prefix positions are never
        scored as language-model targets.  KV-cache decode is intentionally separate until the
        visual prefill/export contract is verified.
        """

        visual = self.encode_images(images)
        feats = self.forward_features(idx, prefix_embeds=visual)
        text_feats = feats[:, visual.shape[1] :]
        h = self.out_proj(text_feats) if self.out_proj is not None else text_feats
        logits = F.linear(h, self.embed.weight) if self.lm_head is None else self.lm_head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        if return_hidden:
            return logits, loss, text_feats, visual
        return logits, loss

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                pos: int = 0, caches=None, return_hidden: bool = False):
        """If `caches` is provided (a list of n_cache_slots entries), runs the cached path and
        returns (logits, loss, new_caches). Otherwise returns (logits, loss). With
        `return_hidden`, also returns the post-final-norm (d_model) features for a probe/head."""
        return_cache = caches is not None
        features_out = self.forward_features(idx, pos=pos, caches=caches)
        if return_cache:
            feats, new_caches = features_out
        else:
            feats = features_out
        h = self.out_proj(feats) if self.out_proj is not None else feats
        logits = F.linear(h, self.embed.weight) if self.lm_head is None else self.lm_head(h)
        if return_hidden:
            return logits, feats
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        if return_cache:
            return logits, loss, new_caches
        return logits, loss

    def num_params(self) -> int:
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) not in seen:  # avoid double-counting tied weights
                seen.add(id(p))
                total += p.numel()
        return total

    def active_num_params(self) -> int:
        """Nominal parameters on one token's routed path (not checkpoint or peak-memory size)."""

        return self.cfg.estimate_active_params()

    def routing_aux_loss(self) -> torch.Tensor | None:
        """Differentiable mean router load-balancing loss from the most recent forward.

        Dense models return ``None``. Sparse models include every block invocation (and every
        recurrent loop invocation) in an unweighted mean.
        """

        if not self._last_routing_aux_losses:
            return None
        return torch.stack(self._last_routing_aux_losses).mean()

    def routing_diagnostics(self) -> dict[str, object]:
        """Return detached, JSON-safe aggregate routing telemetry for the latest forward."""

        if not self.cfg.sparse_ffn:
            return {
                "enabled": False,
                "total_parameters": self.num_params(),
                "active_parameters": self.active_num_params(),
            }

        records = self._last_routing_records
        counts = torch.zeros(self.cfg.ffn_num_experts, dtype=torch.long)
        probability_sum = torch.zeros(self.cfg.ffn_num_experts, dtype=torch.float64)
        entropy_sum = 0.0
        aux_sum = 0.0
        tokens = 0
        for record in records:
            record_tokens = int(record["tokens"])
            record_counts = record["expert_counts"]
            record_probability = record["router_probability"]
            record_entropy = record["router_entropy"]
            record_aux = record["load_balance_loss"]
            assert isinstance(record_counts, torch.Tensor)
            assert isinstance(record_probability, torch.Tensor)
            assert isinstance(record_entropy, torch.Tensor)
            assert isinstance(record_aux, torch.Tensor)
            counts += record_counts.to(device="cpu", dtype=torch.long)
            probability_sum += (
                record_probability.to(device="cpu", dtype=torch.float64) * record_tokens
            )
            entropy_sum += float(record_entropy.to(device="cpu")) * record_tokens
            aux_sum += float(record_aux.to(device="cpu"))
            tokens += record_tokens

        assignments = int(counts.sum())
        if assignments:
            expert_load_tensor = counts.to(torch.float64) / assignments
            expert_load = expert_load_tensor.tolist()
            mean_load = 1.0 / self.cfg.ffn_num_experts
            load_cv = float(
                expert_load_tensor.std(unbiased=False) / max(mean_load, torch.finfo(torch.float64).eps)
            )
        else:
            expert_load = [0.0] * self.cfg.ffn_num_experts
            load_cv = 0.0
        expert_token_fraction = (
            (counts.to(torch.float64) / tokens).tolist()
            if tokens
            else [0.0] * self.cfg.ffn_num_experts
        )
        router_probability = (
            (probability_sum / tokens).tolist()
            if tokens
            else [0.0] * self.cfg.ffn_num_experts
        )
        router_entropy = entropy_sum / tokens if tokens else 0.0
        dead_experts = [
            expert_index
            for expert_index, count in enumerate(counts.tolist())
            if count == 0
        ]
        return {
            "enabled": True,
            "num_experts": self.cfg.ffn_num_experts,
            "top_k": self.cfg.ffn_top_k,
            "invocations": len(records),
            "tokens": tokens,
            "assignments": assignments,
            "expert_counts": counts.tolist(),
            "expert_load": expert_load,
            "expert_token_fraction": expert_token_fraction,
            "active_experts": self.cfg.ffn_num_experts - len(dead_experts),
            "dead_experts": dead_experts,
            "router_probability": router_probability,
            "router_entropy": router_entropy,
            "router_entropy_normalized": (
                router_entropy / math.log(self.cfg.ffn_num_experts)
                if self.cfg.ffn_num_experts > 1
                else 0.0
            ),
            "load_cv": load_cv,
            "load_balance_loss": aux_sum / len(records) if records else 0.0,
            "total_parameters": self.num_params(),
            "active_parameters": self.active_num_params(),
        }
