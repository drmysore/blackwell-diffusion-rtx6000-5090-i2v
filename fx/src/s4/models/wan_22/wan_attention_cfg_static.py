"""
WAN 2.2 Static Blocks for Classifier-Free Guidance
====================================================

Specialized static implementations for different guidance strategies.
At initialization, we allocate the exact block types needed:

1. CFGDualBlock: Processes conditional and unconditional in parallel
2. ConditionalOnlyBlock: Just conditional path (no guidance)
3. UnconditionalOnlyBlock: Just unconditional path (for testing)

This eliminates ALL runtime conditionals related to CFG.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math

from wan_attention_static import (
    StaticRMSNorm,
    StaticLayerNorm,
    StaticSelfAttention,
    StaticFeedForward,
    apply_rotary_emb_static,
)


# ================================================================================================
# Specialized Blocks for CFG
# ================================================================================================


class CFGDualSelfAttention(nn.Module):
    """
    Self-attention that processes conditional and unconditional paths in parallel.
    This is more efficient than running the same block twice.
    """

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        dim: int = 5120,
        num_heads: int = 40,
    ):
        super().__init__()

        # Double batch size internally for CFG
        self.cfg_batch = batch_size * 2
        self.seq_len = seq_len
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Shared weights for both paths
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
        self.norm_q = StaticRMSNorm(dim)
        self.norm_k = StaticRMSNorm(dim)
        self.to_out = nn.Linear(dim, dim, bias=True)

        self.scale = self.head_dim**-0.5

    def forward(
        self,
        hidden_cond: torch.Tensor,  # [batch_size, seq_len, dim] - conditional
        hidden_uncond: torch.Tensor,  # [batch_size, seq_len, dim] - unconditional
        cos_freqs: torch.Tensor,  # [1, seq_len, 1, head_dim]
        sin_freqs: torch.Tensor,  # [1, seq_len, 1, head_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process both conditional and unconditional paths.
        Returns both outputs separately for guidance weighting.
        """

        # Stack for parallel processing
        hidden_states = torch.cat([hidden_cond, hidden_uncond], dim=0)  # [2B, L, D]

        # QKV projection - shared weights
        qkv = self.to_qkv(hidden_states)  # [2B, L, 3D]

        # Reshape for multi-head attention
        qkv = qkv.reshape(self.cfg_batch, self.seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, 2B, H, L, HD]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Normalize Q and K
        q_flat = q.transpose(1, 2).reshape(self.cfg_batch, self.seq_len, self.dim)
        q_flat = self.norm_q(q_flat)
        q = q_flat.reshape(self.cfg_batch, self.seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        k_flat = k.transpose(1, 2).reshape(self.cfg_batch, self.seq_len, self.dim)
        k_flat = self.norm_k(k_flat)
        k = k_flat.reshape(self.cfg_batch, self.seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        # Apply RoPE
        q_rot = q.transpose(1, 2)
        k_rot = k.transpose(1, 2)
        q_rot = apply_rotary_emb_static(q_rot, cos_freqs, sin_freqs)
        k_rot = apply_rotary_emb_static(k_rot, cos_freqs, sin_freqs)
        q = q_rot.transpose(1, 2)
        k = k_rot.transpose(1, 2)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(self.cfg_batch, self.seq_len, self.dim)
        out = self.to_out(out)

        # Split back into conditional and unconditional
        out_cond, out_uncond = out.chunk(2, dim=0)

        return out_cond, out_uncond


class CFGDualCrossAttention(nn.Module):
    """
    Cross-attention for CFG with separate context for conditional/unconditional.
    """

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        context_len: int,
        dim: int = 5120,
        num_heads: int = 40,
    ):
        super().__init__()

        self.cfg_batch = batch_size * 2
        self.seq_len = seq_len
        self.context_len = context_len
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_kv = nn.Linear(dim, dim * 2, bias=True)
        self.norm_q = StaticRMSNorm(dim)
        self.norm_k = StaticRMSNorm(dim)
        self.to_out = nn.Linear(dim, dim, bias=True)

        self.scale = self.head_dim**-0.5

    def forward(
        self,
        hidden_cond: torch.Tensor,  # [batch_size, seq_len, dim]
        hidden_uncond: torch.Tensor,  # [batch_size, seq_len, dim]
        context_cond: torch.Tensor,  # [batch_size, context_len, dim]
        context_uncond: torch.Tensor,  # [batch_size, context_len, dim] - often zeros
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-attention with different context for each path.
        Unconditional context is often zeros or learned null embeddings.
        """

        # Stack hidden states
        hidden_states = torch.cat([hidden_cond, hidden_uncond], dim=0)

        # Stack contexts
        context = torch.cat([context_cond, context_uncond], dim=0)

        # Query projection
        q = self.to_q(hidden_states)
        q = self.norm_q(q)
        q = q.reshape(self.cfg_batch, self.seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)

        # Key-value projection
        kv = self.to_kv(context)
        kv = kv.reshape(self.cfg_batch, self.context_len, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # Normalize keys
        k_flat = k.transpose(1, 2).reshape(self.cfg_batch, self.context_len, self.dim)
        k_flat = self.norm_k(k_flat)
        k = k_flat.reshape(self.cfg_batch, self.context_len, self.num_heads, self.head_dim)
        k = k.transpose(1, 2)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(self.cfg_batch, self.seq_len, self.dim)
        out = self.to_out(out)

        # Split outputs
        out_cond, out_uncond = out.chunk(2, dim=0)

        return out_cond, out_uncond


class CFGTransformerBlock(nn.Module):
    """
    Transformer block optimized for CFG inference.
    Processes both paths efficiently with shared weights.
    """

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        context_len: int,
        dim: int = 5120,
        num_heads: int = 40,
        ffn_dim: int = 13824,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.cfg_batch = batch_size * 2
        self.dim = dim

        # Normalization layers
        self.norm1 = StaticLayerNorm(dim)
        self.norm2 = StaticLayerNorm(dim)
        self.norm3 = StaticLayerNorm(dim)

        # CFG-aware attention layers
        self.self_attn = CFGDualSelfAttention(batch_size, seq_len, dim, num_heads)
        self.cross_attn = CFGDualCrossAttention(batch_size, seq_len, context_len, dim, num_heads)

        # Shared FFN
        self.ffn = StaticFeedForward(dim, ffn_dim)

        # Modulation parameters
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / math.sqrt(dim))

    def forward(
        self,
        hidden_cond: torch.Tensor,
        hidden_uncond: torch.Tensor,
        context_cond: torch.Tensor,
        context_uncond: torch.Tensor,
        conditioning_cond: torch.Tensor,
        conditioning_uncond: torch.Tensor,
        cos_freqs: torch.Tensor,
        sin_freqs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process both CFG paths through the block.
        """

        # Modulation for conditional path
        mod_cond = self.scale_shift_table + conditioning_cond
        shift_sa_c, scale_sa_c, gate_sa_c, shift_ff_c, scale_ff_c, gate_ff_c = mod_cond.chunk(
            6, dim=1
        )

        # Modulation for unconditional path
        mod_uncond = self.scale_shift_table + conditioning_uncond
        shift_sa_u, scale_sa_u, gate_sa_u, shift_ff_u, scale_ff_u, gate_ff_u = mod_uncond.chunk(
            6, dim=1
        )

        # Self-attention with modulation
        normed_cond = self.norm1(hidden_cond)
        normed_uncond = self.norm1(hidden_uncond)

        modulated_cond = normed_cond * (1 + scale_sa_c) + shift_sa_c
        modulated_uncond = normed_uncond * (1 + scale_sa_u) + shift_sa_u

        sa_cond, sa_uncond = self.self_attn(modulated_cond, modulated_uncond, cos_freqs, sin_freqs)

        hidden_cond = hidden_cond + gate_sa_c * sa_cond
        hidden_uncond = hidden_uncond + gate_sa_u * sa_uncond

        # Cross-attention
        normed_cond = self.norm2(hidden_cond)
        normed_uncond = self.norm2(hidden_uncond)

        ca_cond, ca_uncond = self.cross_attn(
            normed_cond, normed_uncond, context_cond, context_uncond
        )

        hidden_cond = hidden_cond + ca_cond
        hidden_uncond = hidden_uncond + ca_uncond

        # FFN with modulation - process separately for different modulation
        normed_cond = self.norm3(hidden_cond)
        modulated_cond = normed_cond * (1 + scale_ff_c) + shift_ff_c
        ffn_cond = self.ffn(modulated_cond)
        hidden_cond = hidden_cond + gate_ff_c * ffn_cond

        normed_uncond = self.norm3(hidden_uncond)
        modulated_uncond = normed_uncond * (1 + scale_ff_u) + shift_ff_u
        ffn_uncond = self.ffn(modulated_uncond)
        hidden_uncond = hidden_uncond + gate_ff_u * ffn_uncond

        return hidden_cond, hidden_uncond


class ConditionalOnlyBlock(nn.Module):
    """
    Optimized block for when CFG is disabled (conditional only).
    This is just a standard static block but named clearly.
    """

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        context_len: int,
        dim: int = 5120,
        num_heads: int = 40,
        ffn_dim: int = 13824,
    ):
        super().__init__()

        # This is just the standard static block
        from wan_attention_static import StaticTransformerBlock

        self.block = StaticTransformerBlock(
            batch_size, seq_len, context_len, dim, num_heads, ffn_dim
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        conditioning: torch.Tensor,
        cos_freqs: torch.Tensor,
        sin_freqs: torch.Tensor,
    ) -> torch.Tensor:
        """Single path forward."""
        return self.block(hidden_states, context, conditioning, cos_freqs, sin_freqs)


class UnconditionalOnlyBlock(nn.Module):
    """
    Block for unconditional generation (no text/image context).
    Skips cross-attention entirely.
    """

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        dim: int = 5120,
        num_heads: int = 40,
        ffn_dim: int = 13824,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dim = dim

        # Layers
        self.norm1 = StaticLayerNorm(dim)
        self.norm3 = StaticLayerNorm(dim)

        self.self_attn = StaticSelfAttention(batch_size, seq_len, dim, num_heads)
        self.ffn = StaticFeedForward(dim, ffn_dim)

        # Modulation (only for self-attention and FFN, no cross-attention)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 4, dim) / math.sqrt(dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning: torch.Tensor,  # [B, 4, D] - less params without cross-attn
        cos_freqs: torch.Tensor,
        sin_freqs: torch.Tensor,
    ) -> torch.Tensor:
        """Forward without cross-attention."""

        # Modulation parameters (4 instead of 6)
        modulation = self.scale_shift_table + conditioning
        shift_sa, scale_sa, shift_ff, scale_ff = modulation.chunk(4, dim=1)

        # Self-attention with modulation (no gating for unconditional)
        normed = self.norm1(hidden_states)
        modulated = normed * (1 + scale_sa) + shift_sa
        attn_out = self.self_attn(modulated, cos_freqs, sin_freqs)
        hidden_states = hidden_states + attn_out  # No gating

        # No cross-attention

        # FFN with modulation
        normed = self.norm3(hidden_states)
        modulated = normed * (1 + scale_ff) + shift_ff
        ffn_out = self.ffn(modulated)
        hidden_states = hidden_states + ffn_out  # No gating

        return hidden_states


# ================================================================================================
# Complete Models with CFG Support
# ================================================================================================


class StaticCFGInferenceModel(nn.Module):
    """
    Complete static model optimized for classifier-free guidance.
    All blocks process both conditional and unconditional paths.
    """

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        context_len: int,
        num_layers: int = 48,
        dim: int = 5120,
        num_heads: int = 40,
        ffn_dim: int = 13824,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.context_len = context_len
        self.num_layers = num_layers
        self.dim = dim

        # Position embeddings (shared)
        from wan_attention_static import StaticRotaryEmbed

        self.rope = StaticRotaryEmbed(seq_len, dim // num_heads, device, dtype)

        # CFG-optimized blocks
        self.blocks = nn.ModuleList(
            [
                CFGTransformerBlock(batch_size, seq_len, context_len, dim, num_heads, ffn_dim)
                for _ in range(num_layers)
            ]
        )

        # Output norm (applied separately)
        self.norm_out = StaticLayerNorm(dim)

        # Output modulation
        self.output_scale_shift = nn.Parameter(torch.randn(1, 2, dim) / math.sqrt(dim))

    def forward(
        self,
        hidden_cond: torch.Tensor,
        hidden_uncond: torch.Tensor,
        context_cond: torch.Tensor,
        context_uncond: torch.Tensor,
        block_conditioning_cond: torch.Tensor,
        block_conditioning_uncond: torch.Tensor,
        output_conditioning_cond: torch.Tensor,
        output_conditioning_uncond: torch.Tensor,
        guidance_scale: float = 7.5,  # Fixed at init in practice
    ) -> torch.Tensor:
        """
        Forward pass with CFG.

        Returns the guided output directly.
        """

        # Get position embeddings
        cos_freqs, sin_freqs = self.rope()

        # Process through blocks
        for i, block in enumerate(self.blocks):
            hidden_cond, hidden_uncond = block(
                hidden_cond,
                hidden_uncond,
                context_cond,
                context_uncond,
                block_conditioning_cond[:, i],
                block_conditioning_uncond[:, i],
                cos_freqs,
                sin_freqs,
            )

        # Output processing for conditional
        mod_cond = self.output_scale_shift + output_conditioning_cond
        shift_c, scale_c = mod_cond.chunk(2, dim=1)
        hidden_cond = self.norm_out(hidden_cond)
        hidden_cond = hidden_cond * (1 + scale_c) + shift_c

        # Output processing for unconditional
        mod_uncond = self.output_scale_shift + output_conditioning_uncond
        shift_u, scale_u = mod_uncond.chunk(2, dim=1)
        hidden_uncond = self.norm_out(hidden_uncond)
        hidden_uncond = hidden_uncond * (1 + scale_u) + shift_u

        # Apply classifier-free guidance
        # Note: guidance_scale is FIXED at initialization for static graph
        guided = hidden_uncond + guidance_scale * (hidden_cond - hidden_uncond)

        return guided


class StaticConditionalOnlyModel(nn.Module):
    """
    Model for when CFG is disabled - just conditional path.
    More efficient than CFG model when guidance is not needed.
    """

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        context_len: int,
        num_layers: int = 48,
        dim: int = 5120,
        num_heads: int = 40,
        ffn_dim: int = 13824,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        # Just use the standard static model
        from wan_attention_static import StaticWANInferenceModel

        self.model = StaticWANInferenceModel(
            batch_size,
            seq_len,
            context_len,
            num_layers,
            dim,
            num_heads,
            ffn_dim,
            0.0,
            device,
            dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        block_conditioning: torch.Tensor,
        output_conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """Standard forward without CFG."""
        return self.model(hidden_states, context, block_conditioning, output_conditioning)


# ================================================================================================
# Factory for Creating CFG-Aware Models
# ================================================================================================


def create_cfg_static_model(
    batch_size: int,
    seq_len: int,
    context_len: int,
    use_cfg: bool,
    guidance_scale: float = 7.5,
    use_unconditional_only: bool = False,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> nn.Module:
    """
    Create appropriate static model based on CFG configuration.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        context_len: Context length
        use_cfg: Whether to use classifier-free guidance
        guidance_scale: Guidance scale (fixed at init)
        use_unconditional_only: Whether to use unconditional-only model
        device: Device
        dtype: Data type

    Returns:
        Appropriate static model
    """

    if use_unconditional_only:
        # Special case: unconditional generation
        print("Creating unconditional-only model (no context)")
        return StaticUnconditionalOnlyModel(batch_size, seq_len, device=device, dtype=dtype)
    elif use_cfg:
        # CFG-optimized model
        print(f"Creating CFG model with guidance_scale={guidance_scale}")
        return StaticCFGInferenceModel(batch_size, seq_len, context_len, device=device, dtype=dtype)
    else:
        # Standard conditional model
        print("Creating conditional-only model (no CFG)")
        return StaticConditionalOnlyModel(
            batch_size, seq_len, context_len, device=device, dtype=dtype
        )


class StaticUnconditionalOnlyModel(nn.Module):
    """Complete model for unconditional generation."""

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        num_layers: int = 48,
        dim: int = 5120,
        num_heads: int = 40,
        ffn_dim: int = 13824,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        from wan_attention_static import StaticRotaryEmbed

        self.rope = StaticRotaryEmbed(seq_len, dim // num_heads, device, dtype)

        self.blocks = nn.ModuleList(
            [
                UnconditionalOnlyBlock(batch_size, seq_len, dim, num_heads, ffn_dim)
                for _ in range(num_layers)
            ]
        )

        self.norm_out = StaticLayerNorm(dim)
        self.output_scale_shift = nn.Parameter(torch.randn(1, 2, dim) / math.sqrt(dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_conditioning: torch.Tensor,  # [B, num_layers, 4, D]
        output_conditioning: torch.Tensor,  # [B, 2, D]
    ) -> torch.Tensor:
        """Forward for unconditional generation."""

        cos_freqs, sin_freqs = self.rope()

        for i, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                block_conditioning[:, i],
                cos_freqs,
                sin_freqs,
            )

        # Output processing
        modulation = self.output_scale_shift + output_conditioning
        shift, scale = modulation.chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift

        return hidden_states
