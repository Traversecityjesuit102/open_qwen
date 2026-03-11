"""
Hybrid Multimodal Language Model — PyTorch Implementation
=========================================================

Architecture (from spec):
  • 64 hybrid transformer layers
      – Each layer = 3 × Gated DeltaNet block (linear attention + FFN)
                   + 1 × Gated Attention block (GQA + FFN)
  • Hidden dim          : 5 120
  • Vocabulary size     : 248 320
  • Native context      : 262 144 tokens
  • Extended context    : ~1 010 000 tokens via YaRN RoPE scaling
  • Multimodal inputs   : text tokens + ViT-style vision encoder
  • Output              : auto-regressive next-token logits
  • Multi-Token Prediction (MTP) head for accelerated inference

Gated DeltaNet (linear attention)
  QK heads : 16  |  head dim : 128  →  QK dim = 2 048
  V  heads : 48  |  head dim : 128  →  V  dim = 6 144
  Update rule: S_t = (1 - β_t) S_{t-1}  +  β_t (v_t − S_{t-1}ᵀ k_t) kᵀ_t
               o_t = S_t q_t

Gated Attention (grouped-query attention)
  Q  heads : 24  |  head dim : 256  →  Q  dim = 6 144
  KV heads :  4  |  head dim : 256  →  KV dim = 1 024

FFN (SwiGLU) hidden dim : 17 408  (both block types)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class VisionConfig:
    """Configuration for the ViT-style vision encoder."""

    image_size: int = 448
    """Input image resolution (height == width)."""

    patch_size: int = 14
    """Side length of each image patch (pixels)."""

    num_channels: int = 3
    """Number of input image channels."""

    hidden_size: int = 1152
    """Vision encoder hidden dimension."""

    num_layers: int = 27
    """Number of vision transformer layers."""

    num_heads: int = 16
    """Number of self-attention heads in the vision encoder."""

    mlp_ratio: float = 4.0
    """MLP expansion ratio inside the vision transformer."""

    dropout: float = 0.0
    """Dropout probability."""

    @property
    def num_patches(self) -> int:
        """Total number of patch tokens (excluding CLS) per image."""
        return (self.image_size // self.patch_size) ** 2


@dataclass
class ModelConfig:
    """Top-level model configuration."""

    # ── vocabulary / embedding ──────────────────────────────────────────────
    vocab_size: int = 248_320
    """Vocabulary size (text tokens + special multimodal tokens)."""

    hidden_size: int = 5_120
    """Model hidden dimension (d_model)."""

    # ── layers ──────────────────────────────────────────────────────────────
    num_layers: int = 64
    """Total number of hybrid transformer layers."""

    num_linear_blocks_per_layer: int = 3
    """Number of Gated DeltaNet blocks before each Gated Attention block."""

    # ── Gated DeltaNet (linear attention) ───────────────────────────────────
    delta_qk_heads: int = 16
    """QK head count for DeltaNet."""

    delta_v_heads: int = 48
    """Value head count for DeltaNet."""

    delta_head_dim: int = 128
    """Head dimension for DeltaNet (QK and V share the same head dim)."""

    # ── Gated Attention (GQA) ───────────────────────────────────────────────
    attn_q_heads: int = 24
    """Query head count for Gated Attention."""

    attn_kv_heads: int = 4
    """Key/value head count for Gated Attention (GQA)."""

    attn_head_dim: int = 256
    """Head dimension for Gated Attention."""

    # ── FFN ─────────────────────────────────────────────────────────────────
    ffn_hidden_dim: int = 17_408
    """Inner dimension of the SwiGLU feed-forward network."""

    # ── positional embeddings ───────────────────────────────────────────────
    max_seq_len: int = 262_144
    """Native context window length."""

    rope_base: float = 10_000.0
    """Base frequency for Rotary Position Embeddings."""

    rope_scaling_factor: float = 1.0
    """YaRN scale factor (set > 1 to extend context beyond max_seq_len)."""

    yarn_beta_fast: float = 32.0
    """YaRN β_fast parameter."""

    yarn_beta_slow: float = 1.0
    """YaRN β_slow parameter."""

    yarn_mscale: float = 0.1
    """YaRN magnitude scaling coefficient."""

    # ── Multi-Token Prediction ──────────────────────────────────────────────
    mtp_num_heads: int = 4
    """Number of future tokens predicted simultaneously."""

    # ── misc ────────────────────────────────────────────────────────────────
    dropout: float = 0.0
    """Dropout probability (0 = disabled)."""

    rms_norm_eps: float = 1e-6
    """Epsilon for RMSNorm."""

    pad_token_id: int = 0
    """Padding token id."""

    vision: VisionConfig = field(default_factory=VisionConfig)
    """Nested vision encoder configuration."""

    # ── derived properties ──────────────────────────────────────────────────
    @property
    def delta_qk_dim(self) -> int:
        """Total QK projection dimension for DeltaNet."""
        return self.delta_qk_heads * self.delta_head_dim  # 2 048

    @property
    def delta_v_dim(self) -> int:
        """Total V projection dimension for DeltaNet."""
        return self.delta_v_heads * self.delta_head_dim  # 6 144

    @property
    def attn_q_dim(self) -> int:
        """Total Q projection dimension for Gated Attention."""
        return self.attn_q_heads * self.attn_head_dim  # 6 144

    @property
    def attn_kv_dim(self) -> int:
        """Total KV projection dimension for Gated Attention."""
        return self.attn_kv_heads * self.attn_head_dim  # 1 024


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (no learnable bias)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by half for RoPE application."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embeddings to query and key tensors.

    Args:
        q:   Query  tensor of shape (..., seq_len, head_dim).
        k:   Key    tensor of shape (..., seq_len, head_dim).
        cos: Cosine table of shape (seq_len, head_dim).
        sin: Sine   table of shape (seq_len, head_dim).

    Returns:
        Rotated (q, k) tensors with the same shape as inputs.
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE + YaRN scaling)
# ---------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding with optional YaRN long-context scaling.

    YaRN (Peng et al., 2023) interpolates and extrapolates frequency
    components independently so that the model can generalise well beyond its
    training context window.

    Args:
        head_dim:        Dimension of each attention head.
        max_seq_len:     Maximum sequence length at training time.
        base:            Base frequency for the inverse-frequency formula.
        scaling_factor:  Multiplier applied to the effective sequence length
                         (> 1 enables YaRN context extension).
        beta_fast:       YaRN β_fast — highest interpolated dimension.
        beta_slow:       YaRN β_slow — lowest interpolated dimension.
        mscale:          Attention-magnitude rescaling coefficient (YaRN).
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 262_144,
        base: float = 10_000.0,
        scaling_factor: float = 1.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float = 0.1,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_factor = scaling_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale

        # Build (and register) the inverse-frequency buffer.
        inv_freq = self._build_inv_freq(
            head_dim, base, scaling_factor, max_seq_len, beta_fast, beta_slow
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute and cache cos/sin tables up to max_seq_len.
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_inv_freq(
        head_dim: int,
        base: float,
        scaling_factor: float,
        max_seq_len: int,
        beta_fast: float,
        beta_slow: float,
    ) -> torch.Tensor:
        """Construct the inverse frequency vector, applying YaRN when needed."""
        # Standard RoPE inverse frequencies.
        dim_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq_std = 1.0 / (base ** (dim_range / head_dim))

        if scaling_factor == 1.0:
            return inv_freq_std

        # YaRN: compute a per-dimension interpolation ratio (ramp).
        low = math.floor(
            head_dim
            * math.log(max_seq_len / (beta_fast * 2 * math.pi))
            / (2 * math.log(base))
        )
        high = math.ceil(
            head_dim
            * math.log(max_seq_len / (beta_slow * 2 * math.pi))
            / (2 * math.log(base))
        )
        low = max(low, 0)
        high = min(high, head_dim // 2 - 1)

        # Linear ramp from 0 (full interpolation) → 1 (no interpolation).
        ramp = torch.zeros(head_dim // 2, dtype=torch.float32)
        if high > low:
            idx = torch.arange(low, high + 1, dtype=torch.float32)
            ramp[low : high + 1] = (idx - low) / (high - low)
        ramp[high:] = 1.0

        # Blend: interpolated (1/scale) and extrapolated (1) components.
        inv_freq_interp = inv_freq_std / scaling_factor
        inv_freq_extrap = inv_freq_std
        inv_freq_yarn = (1 - ramp) * inv_freq_interp + ramp * inv_freq_extrap
        return inv_freq_yarn

    def _get_mscale(self) -> float:
        """Compute the YaRN magnitude-rescaling factor."""
        if self.scaling_factor == 1.0:
            return 1.0
        return 0.1 * math.log(self.scaling_factor) + 1.0

    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        """Populate cos/sin caches for positions 0 … seq_len-1."""
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)  # (seq, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq, head_dim)
        mscale = self._get_mscale()
        self._cos_cached = (emb.cos() * mscale).to(device)
        self._sin_cached = (emb.sin() * mscale).to(device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self, seq_len: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) tables for positions 0 … seq_len-1.

        Args:
            seq_len: Number of positions needed.
            device:  Target device.

        Returns:
            Tuple of tensors, each of shape (seq_len, head_dim).
        """
        if (
            self._cos_cached is None
            or self._cos_cached.shape[0] < seq_len
            or self._cos_cached.device != device
        ):
            self._build_cache(max(seq_len, self.max_seq_len), device)
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]  # type: ignore[index]


# ---------------------------------------------------------------------------
# Vision Encoder (ViT-style)
# ---------------------------------------------------------------------------


class PatchEmbedding(nn.Module):
    """Split an image into non-overlapping patches and embed them linearly.

    Args:
        config: Vision configuration.
    """

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.num_patches = config.num_patches

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Embed image patches.

        Args:
            pixel_values: Float tensor of shape (B, C, H, W).

        Returns:
            Patch tokens of shape (B, num_patches, hidden_size).
        """
        # (B, hidden, H//p, W//p) → (B, hidden, N) → (B, N, hidden)
        return self.proj(pixel_values).flatten(2).transpose(1, 2)


class VisionMLP(nn.Module):
    """Two-layer GELU MLP used inside ViT blocks."""

    def __init__(self, hidden_size: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        inner = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, inner)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(inner, hidden_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class VisionAttention(nn.Module):
    """Standard multi-head self-attention used inside ViT blocks."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attend over patch tokens.

        Args:
            x: Tensor of shape (B, N, hidden_size).

        Returns:
            Tensor of shape (B, N, hidden_size).
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (B, H, N, d)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class VisionBlock(nn.Module):
    """Single ViT transformer block: pre-norm attention + pre-norm MLP."""

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = VisionAttention(
            config.hidden_size, config.num_heads, config.dropout
        )
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.mlp = VisionMLP(config.hidden_size, config.mlp_ratio, config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """ViT-style vision encoder.

    Converts raw pixel values into a sequence of visual feature vectors that
    are projected into the language model's hidden dimension before being
    concatenated with text tokens.

    Args:
        vision_cfg: Vision encoder hyper-parameters.
        lm_hidden:  Hidden size of the language model (target projection dim).
    """

    def __init__(self, vision_cfg: VisionConfig, lm_hidden: int) -> None:
        super().__init__()
        H = vision_cfg.hidden_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, H))
        self.patch_embed = PatchEmbedding(vision_cfg)
        N = vision_cfg.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, N + 1, H))  # +1 for CLS
        self.pos_drop = nn.Dropout(vision_cfg.dropout)
        self.blocks = nn.ModuleList(
            [VisionBlock(vision_cfg) for _ in range(vision_cfg.num_layers)]
        )
        self.norm = nn.LayerNorm(H, eps=1e-6)
        self.proj = nn.Linear(H, lm_hidden, bias=False)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images to language-model-compatible feature tokens.

        Args:
            pixel_values: Float tensor (B, C, H, W), values in [0, 1] or
                          normalised with mean/std.

        Returns:
            Visual feature tokens of shape (B, num_patches + 1, lm_hidden).
        """
        B = pixel_values.shape[0]
        x = self.patch_embed(pixel_values)  # (B, N, H)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, H)
        x = torch.cat([cls, x], dim=1)  # (B, N+1, H)
        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return self.proj(x)  # (B, N+1, d_model)


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward block.

        FFN(x) = (Swish(x W_gate) ⊙ x W_up) W_down

    Args:
        hidden_size: Input / output dimension.
        inner_size:  Intermediate (expansion) dimension.
        dropout:     Dropout probability.
    """

    def __init__(self, hidden_size: int, inner_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_size, inner_size, bias=False)
        self.up = nn.Linear(hidden_size, inner_size, bias=False)
        self.down = nn.Linear(inner_size, hidden_size, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


# ---------------------------------------------------------------------------
# Gated DeltaNet (Linear Attention)
# ---------------------------------------------------------------------------


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet linear-attention block.

    Implements a gated variant of the DeltaNet recurrent linear attention.
    The recurrent state S (a key-value memory matrix) is updated at each step
    using the delta (error-correcting) rule:

        β_t  = σ(proj_beta(x_t))                 ∈ (0, 1)^{H_qk}
        q_t  = proj_q(x_t)                        shape: (H_qk, d_k)
        k_t  = norm(proj_k(x_t))                  shape: (H_qk, d_k)
        v_t  = proj_v(x_t)                        shape: (H_v,  d_k)
        g_t  = σ(proj_g(x_t))                     output gate ∈ (0,1)^{d_model}

        S_t  = (1 - β_t) ⊙ S_{t-1}
               + β_t ⊙ ( v_t − S_{t-1}ᵀ k_t ) kᵀ_t

        o_t  = g_t ⊙ concat( S_t q_{t,h} )_h    (across QK heads)

    During training the recurrence is parallelised over the sequence dimension
    via a causal chunk-wise formulation (simplified here to a sequential scan
    over the time axis for clarity).

    Args:
        config: Full model configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d = config.hidden_size
        H = config.delta_qk_heads  # 16
        Hv = config.delta_v_heads  # 48
        dk = config.delta_head_dim  # 128

        self.H = H
        self.Hv = Hv
        self.dk = dk
        self.d = d

        # Note: QK projections are *shared* across the two head groups.
        self.proj_q = nn.Linear(d, H * dk, bias=False)
        self.proj_k = nn.Linear(d, H * dk, bias=False)
        self.proj_v = nn.Linear(d, Hv * dk, bias=False)
        self.proj_beta = nn.Linear(d, H, bias=True)  # per QK head scalar
        self.proj_g = nn.Linear(d, d, bias=True)  # output gate
        self.proj_o = nn.Linear(Hv * dk, d, bias=False)  # mix V heads → d_model

        self.norm_k = RMSNorm(dk, eps=config.rms_norm_eps)
        self.norm_out = RMSNorm(d, eps=config.rms_norm_eps)
        self.drop = nn.Dropout(config.dropout)

    # ------------------------------------------------------------------
    # Core delta-rule recurrence (sequential; replace with chunk-parallel
    # kernel for production)
    # ------------------------------------------------------------------

    @staticmethod
    def _delta_recurrence(
        q: torch.Tensor,  # (B, T, H, dk)
        k: torch.Tensor,  # (B, T, H, dk)
        v: torch.Tensor,  # (B, T, Hv, dk)
        beta: torch.Tensor,  # (B, T, H)
    ) -> torch.Tensor:
        """Compute DeltaNet outputs via sequential recurrence.

        Returns:
            Tensor of shape (B, T, Hv, dk) — one output per V head.

        Note:
            Q heads (H=16) are broadcast over V heads (Hv=48) with a 1-to-3
            mapping (Hv // H == 3), matching the spec's asymmetric head counts.
        """
        B, T, H, dk = q.shape
        Hv = v.shape[2]
        group = Hv // H  # V heads per QK head = 3

        # Initialise per-head state matrices S: (B, H, dk, dk)
        S = torch.zeros(B, H, dk, dk, device=q.device, dtype=q.dtype)

        outputs = []
        for t in range(T):
            q_t = q[:, t]  # (B, H,  dk)
            k_t = k[:, t]  # (B, H,  dk)
            beta_t = beta[:, t]  # (B, H)
            v_t = v[:, t]  # (B, Hv, dk)

            # Retrieve current content for each QK head and broadcast to V heads.
            # v_hat_t = S_{t-1} k_t : (B, H, dk)
            v_hat = torch.einsum("bhij,bhj->bhi", S, k_t)  # (B, H, dk)

            # Stack V heads into groups of `group` per QK head.
            v_grouped = v_t.view(B, H, group, dk)  # (B, H, g, dk)
            v_hat_grouped = v_hat.unsqueeze(2).expand_as(v_grouped)  # (B, H, g, dk)

            # Error signal per V-group: δ = v - v_hat
            delta_grouped = v_grouped - v_hat_grouped  # (B, H, g, dk)

            # Update state: S_t = (1-β) S_{t-1} + β * mean_group(δ) outer k
            beta_t_e = beta_t.unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
            delta_mean = delta_grouped.mean(dim=2)  # (B, H, dk)
            S = (1 - beta_t_e) * S + beta_t.unsqueeze(-1).unsqueeze(-1) * torch.einsum(
                "bhi,bhj->bhij", delta_mean, k_t
            )  # outer product

            # Output: o_t = S q_t  → (B, H, dk)
            o_t_qk = torch.einsum("bhij,bhj->bhi", S, q_t)  # (B, H, dk)

            # Expand back to Hv by repeating along head dim.
            o_t = o_t_qk.unsqueeze(2).expand(B, H, group, dk)  # (B, H, g, dk)
            o_t = o_t.reshape(B, Hv, dk)  # (B, Hv, dk)
            outputs.append(o_t)

        return torch.stack(outputs, dim=1)  # (B, T, Hv, dk)

    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,  # (B, T, d_model)
        *,
        return_state: bool = False,
    ) -> torch.Tensor:
        """Forward pass of GatedDeltaNet.

        Args:
            x:            Input hidden states (B, T, d_model).
            return_state: If True, also return the final recurrent state
                          (not yet wired to KV-cache interface but kept for
                          future streaming / inference use).

        Returns:
            Output hidden states (B, T, d_model).
        """
        B, T, _ = x.shape

        q = self.proj_q(x).view(B, T, self.H, self.dk)  # (B, T, H,  dk)
        k = self.proj_k(x).view(B, T, self.H, self.dk)  # (B, T, H,  dk)
        v = self.proj_v(x).view(B, T, self.Hv, self.dk)  # (B, T, Hv, dk)

        # Normalise keys to unit sphere for numerical stability.
        k = self.norm_k(k)

        beta = torch.sigmoid(self.proj_beta(x))  # (B, T, H)
        g = torch.sigmoid(self.proj_g(x))  # (B, T, d)

        # Linear-attention recurrence.
        out = self._delta_recurrence(q, k, v, beta)  # (B, T, Hv, dk)
        out = out.reshape(B, T, self.Hv * self.dk)

        # Project and gate.
        out = self.proj_o(out)  # (B, T, d)
        out = self.norm_out(out) * g
        return self.drop(out)


# ---------------------------------------------------------------------------
# Gated Attention (GQA with causal mask)
# ---------------------------------------------------------------------------


class GatedAttention(nn.Module):
    """Grouped-Query Attention with an output sigmoid gate.

    Implements multi-head attention with:
      • Separate Q / KV head counts (GQA).
      • Rotary positional embeddings applied to Q and K.
      • Causal (left-to-right) attention mask.
      • Sigmoid output gate on the projected output.

    Args:
        config:   Full model configuration.
        rope:     Shared RotaryEmbedding module (passed in from the layer).
    """

    def __init__(self, config: ModelConfig, rope: RotaryEmbedding) -> None:
        super().__init__()
        d = config.hidden_size
        Hq = config.attn_q_heads  # 24
        Hkv = config.attn_kv_heads  # 4
        dh = config.attn_head_dim  # 256
        self.Hq = Hq
        self.Hkv = Hkv
        self.dh = dh
        self.rope = rope

        # Each KV head serves (Hq // Hkv) = 6 query heads.
        assert Hq % Hkv == 0, "Q heads must be divisible by KV heads for GQA."
        self.groups = Hq // Hkv

        self.proj_q = nn.Linear(d, Hq * dh, bias=False)
        self.proj_k = nn.Linear(d, Hkv * dh, bias=False)
        self.proj_v = nn.Linear(d, Hkv * dh, bias=False)
        self.proj_g = nn.Linear(d, d, bias=True)  # output gate
        self.proj_o = nn.Linear(Hq * dh, d, bias=False)

        self.norm_out = RMSNorm(d, eps=config.rms_norm_eps)
        self.drop = nn.Dropout(config.dropout)
        self.scale = dh**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Causally attend over the sequence.

        Args:
            x: Hidden states of shape (B, T, d_model).

        Returns:
            Output hidden states of shape (B, T, d_model).
        """
        B, T, _ = x.shape

        q = (
            self.proj_q(x).view(B, T, self.Hq, self.dh).transpose(1, 2)
        )  # (B, Hq,  T, dh)
        k = (
            self.proj_k(x).view(B, T, self.Hkv, self.dh).transpose(1, 2)
        )  # (B, Hkv, T, dh)
        v = (
            self.proj_v(x).view(B, T, self.Hkv, self.dh).transpose(1, 2)
        )  # (B, Hkv, T, dh)

        # Apply RoPE to Q and K.
        cos, sin = self.rope(T, x.device)
        # Unsqueeze for head dim: (1, 1, T, dh)
        cos_q = cos.unsqueeze(0).unsqueeze(0)
        sin_q = sin.unsqueeze(0).unsqueeze(0)
        q = q * cos_q + rotate_half(q) * sin_q
        k = k * cos_q + rotate_half(k) * sin_q

        # Expand KV heads to match Q heads (GQA).
        k = k.repeat_interleave(self.groups, dim=1)  # (B, Hq, T, dh)
        v = v.repeat_interleave(self.groups, dim=1)  # (B, Hq, T, dh)

        # Scaled dot-product attention with causal mask.
        # PyTorch ≥ 2.0: use flash-attention-compatible API.
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.drop.p if self.training else 0.0,
            is_causal=True,
            scale=self.scale,
        )  # (B, Hq, T, dh)

        attn_out = attn_out.transpose(1, 2).reshape(B, T, self.Hq * self.dh)
        g = torch.sigmoid(self.proj_g(x))  # (B, T, d)
        out = self.proj_o(attn_out)
        out = self.norm_out(out) * g
        return self.drop(out)


# ---------------------------------------------------------------------------
# Hybrid Layer  (3 × DeltaNet + 1 × Attention, each with FFN)
# ---------------------------------------------------------------------------


class HybridLayer(nn.Module):
    """One hybrid transformer layer as described in the spec.

    Layout per layer:
        for _ in range(3):
            x = x + GatedDeltaNet(RMSNorm(x))
            x = x + SwiGLUFFN(RMSNorm(x))

        x = x + GatedAttention(RMSNorm(x))
        x = x + SwiGLUFFN(RMSNorm(x))

    Args:
        config: Full model configuration.
        rope:   Shared RotaryEmbedding (passed through from the backbone).
    """

    def __init__(self, config: ModelConfig, rope: RotaryEmbedding) -> None:
        super().__init__()
        d = config.hidden_size
        ffn = config.ffn_hidden_dim
        dr = config.dropout

        # 3 × (DeltaNet block + FFN)
        self.delta_norms: nn.ModuleList = nn.ModuleList(
            [RMSNorm(d, config.rms_norm_eps) for _ in range(3)]
        )
        self.delta_blocks: nn.ModuleList = nn.ModuleList(
            [GatedDeltaNet(config) for _ in range(3)]
        )
        self.delta_ffn_norms: nn.ModuleList = nn.ModuleList(
            [RMSNorm(d, config.rms_norm_eps) for _ in range(3)]
        )
        self.delta_ffns: nn.ModuleList = nn.ModuleList(
            [SwiGLUFFN(d, ffn, dr) for _ in range(3)]
        )

        # 1 × (GQA block + FFN)
        self.attn_norm = RMSNorm(d, config.rms_norm_eps)
        self.attn = GatedAttention(config, rope)
        self.attn_ffn_norm = RMSNorm(d, config.rms_norm_eps)
        self.attn_ffn = SwiGLUFFN(d, ffn, dr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process one hybrid layer.

        Args:
            x: Hidden states (B, T, d_model).

        Returns:
            Updated hidden states (B, T, d_model).
        """
        # --- 3 × linear-attention sub-layers ---
        for norm, delta, ffn_norm, ffn in zip(
            self.delta_norms,
            self.delta_blocks,
            self.delta_ffn_norms,
            self.delta_ffns,
        ):
            x = x + delta(norm(x))
            x = x + ffn(ffn_norm(x))

        # --- 1 × quadratic-attention sub-layer ---
        x = x + self.attn(self.attn_norm(x))
        x = x + self.attn_ffn(self.attn_ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# Multi-Token Prediction head
# ---------------------------------------------------------------------------


class MultiTokenPredictionHead(nn.Module):
    """Predict the next ``num_heads`` tokens simultaneously.

    Each head after the first uses a small transformer block to refine the
    hidden state before predicting its target token.  This allows the model
    to generate multiple tokens per forward pass during speculative decoding
    or draft generation.

    Args:
        config: Full model configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d = config.hidden_size
        V = config.vocab_size
        n = config.mtp_num_heads

        self.num_heads = n

        # Each MTP step: project hidden state → same dim, then predict token.
        self.refiners = nn.ModuleList(
            [
                nn.Sequential(
                    RMSNorm(d, config.rms_norm_eps),
                    nn.Linear(d, d, bias=False),
                    nn.SiLU(),
                )
                for _ in range(n - 1)
            ]
        )
        # All heads share the same output vocabulary projection.
        self.lm_head = nn.Linear(d, V, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Generate logits for the next ``num_heads`` tokens.

        Args:
            hidden_states: Final backbone hidden states (B, T, d_model).

        Returns:
            Logits tensor of shape (B, T, num_heads, vocab_size).
        """
        logits_list: List[torch.Tensor] = []
        h = hidden_states

        # Head 0: direct prediction.
        logits_list.append(self.lm_head(h))  # (B, T, V)

        # Heads 1 … n-1: refine then predict.
        for refiner in self.refiners:
            h = h + refiner(h)
            logits_list.append(self.lm_head(h))

        return torch.stack(logits_list, dim=2)  # (B, T, n, V)


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------


class Qwen35(nn.Module):
    """Hybrid Multimodal Language Model.

    Full end-to-end model combining:
      • ViT-style vision encoder (optional — skip if no images).
      • Text token embeddings.
      • 64 hybrid transformer layers (3 × DeltaNet + 1 × GQA each).
      • LM output head (tied to input embeddings).
      • Multi-Token Prediction (MTP) auxiliary head.

    Args:
        config: Complete model configuration (see :class:`ModelConfig`).

    Example::

        cfg   = ModelConfig()
        model = HybridLanguageModel(cfg)

        # Text-only forward pass.
        input_ids  = torch.randint(0, cfg.vocab_size, (2, 64))
        logits, mtp = model(input_ids)
        # logits : (2, 64, vocab_size)
        # mtp    : (2, 64, mtp_num_heads, vocab_size)

        # With image input.
        images = torch.randn(2, 3, 448, 448)
        logits, mtp = model(input_ids, pixel_values=images)
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.hidden_size

        # ── Embedding ───────────────────────────────────────────────────────
        self.embed_tokens = nn.Embedding(
            config.vocab_size, d, padding_idx=config.pad_token_id
        )

        # ── Vision encoder (multimodal) ──────────────────────────────────────
        self.vision_encoder = VisionEncoder(config.vision, d)

        # ── Shared RoPE (used by all GQA layers) ────────────────────────────
        self.rope = RotaryEmbedding(
            head_dim=config.attn_head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_base,
            scaling_factor=config.rope_scaling_factor,
            beta_fast=config.yarn_beta_fast,
            beta_slow=config.yarn_beta_slow,
            mscale=config.yarn_mscale,
        )

        # ── 64 Hybrid layers ─────────────────────────────────────────────────
        self.layers = nn.ModuleList(
            [HybridLayer(config, self.rope) for _ in range(config.num_layers)]
        )

        # ── Final normalisation ──────────────────────────────────────────────
        self.norm = RMSNorm(d, config.rms_norm_eps)

        # ── LM head (weight-tied with token embeddings) ───────────────────
        self.lm_head = nn.Linear(d, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # tie

        # ── Multi-Token Prediction auxiliary head ────────────────────────────
        self.mtp_head = MultiTokenPredictionHead(config)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Initialise parameters with a scaled normal distribution."""
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count (trainable) parameters.

        Args:
            trainable_only: If True, count only parameters with
                            ``requires_grad=True``.

        Returns:
            Integer parameter count.
        """
        params = (
            p for p in self.parameters() if (not trainable_only or p.requires_grad)
        )
        return sum(p.numel() for p in params)

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the token embedding table."""
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        """Replace the token embedding table."""
        self.embed_tokens = new_embeddings
        self.lm_head.weight = new_embeddings.weight

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full model forward pass.

        Args:
            input_ids:      Long tensor of token ids (B, T).
            pixel_values:   Optional float tensor of images (B, C, H, W).
                            When provided, visual tokens are prepended to the
                            text token sequence.
            attention_mask: Optional boolean / float attention mask (B, T).
                            Currently used only for informational purposes;
                            the GQA block uses a built-in causal mask.

        Returns:
            A tuple ``(logits, mtp_logits)`` where:

            * ``logits``     — (B, T_out, vocab_size) next-token logits from
                              the primary LM head.
            * ``mtp_logits`` — (B, T_out, mtp_num_heads, vocab_size) logits
                              from the Multi-Token Prediction auxiliary head.

            ``T_out`` equals ``T`` when no images are provided, and
            ``T + num_visual_tokens`` when images are prepended.
        """
        # ── 1. Embed text tokens ─────────────────────────────────────────────
        x = self.embed_tokens(input_ids)  # (B, T, d)

        # ── 2. Prepend visual tokens (multimodal) ───────────────────────────
        if pixel_values is not None:
            visual_tokens = self.vision_encoder(pixel_values)  # (B, Nv, d)
            x = torch.cat([visual_tokens, x], dim=1)  # (B, Nv+T, d)

        # ── 3. Hybrid transformer backbone ───────────────────────────────────
        for layer in self.layers:
            x = layer(x)

        # ── 4. Final norm ─────────────────────────────────────────────────────
        x = self.norm(x)

        # ── 5. Primary LM head ────────────────────────────────────────────────
        logits = self.lm_head(x)  # (B, T_out, V)

        # ── 6. Multi-Token Prediction head ───────────────────────────────────
        mtp_logits = self.mtp_head(x)  # (B, T_out, n, V)

        return logits, mtp_logits


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------


def compute_lm_loss(
    logits: torch.Tensor,  # (B, T, V)
    input_ids: torch.Tensor,  # (B, T)
    pad_token_id: int = 0,
) -> torch.Tensor:
    """Standard next-token cross-entropy loss.

    Args:
        logits:       Raw (un-normalised) logits from the LM head.
        input_ids:    Token ids used as targets (shifted left by 1 internally).
        pad_token_id: Token id to ignore in the loss.

    Returns:
        Scalar mean loss.
    """
    # Shift: predict token t+1 from position t.
    shift_logits = logits[:, :-1].contiguous()  # (B, T-1, V)
    shift_labels = input_ids[:, 1:].contiguous()  # (B, T-1)
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=pad_token_id,
    )


def compute_mtp_loss(
    mtp_logits: torch.Tensor,  # (B, T, n, V)
    input_ids: torch.Tensor,  # (B, T)
    pad_token_id: int = 0,
) -> torch.Tensor:
    """Auxiliary Multi-Token Prediction loss.

    For head ``h`` (0-indexed), the target at position ``t`` is token
    ``t + h + 1`` (the ``h``-th future token).

    Args:
        mtp_logits:   MTP logits from :class:`MultiTokenPredictionHead`.
        input_ids:    Input token ids used to derive targets.
        pad_token_id: Ignored token id.

    Returns:
        Scalar mean loss averaged over all MTP heads.
    """
    B, T, n, V = mtp_logits.shape
    total_loss = mtp_logits.new_zeros(())

    for h in range(n):
        # Positions 0 … T-h-2 predict tokens h+1 … T-1.
        end = T - h - 1
        if end <= 0:
            break
        logits_h = mtp_logits[:, :end, h, :].contiguous()  # (B, end, V)
        labels_h = input_ids[:, h + 1 : h + 1 + end].contiguous()
        total_loss = total_loss + F.cross_entropy(
            logits_h.view(-1, V),
            labels_h.view(-1),
            ignore_index=pad_token_id,
        )

    return total_loss / n
