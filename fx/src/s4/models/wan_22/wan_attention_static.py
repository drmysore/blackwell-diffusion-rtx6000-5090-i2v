"""
WAN 2.2 Static Inference Implementation
========================================

FULLY STATIC, BRANCH-FREE implementation for maximum inference performance.
Every shape is fixed at initialization. No conditionals in forward passes.

This module is designed for:
- torch.compile(fullgraph=True)
- CUDAGraph capture
- TensorRT conversion
- Custom kernel fusion
- NVFP4 quantization

ALL shapes and behaviors are determined at initialization time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math


# ================================================================================================
# Fixed Configuration Classes (No Runtime Flexibility)
# ================================================================================================


class StaticRotaryEmbed(nn.Module):
    """
    Static rotary position embeddings with fixed sequence length.
    All frequencies are pre-computed at initialization.
    """

    def __init__(
        self,
        seq_len: int,  # FIXED sequence length
        head_dim: int,  # FIXED head dimension
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.head_dim = head_dim

        # Pre-compute and register ALL frequencies as buffers
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim)
        )
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
        freqs_expanded = torch.cat([freqs, freqs], dim=-1)

        # Register as buffers (no gradients, moves with model)
        self.register_buffer("cos_cached", freqs_expanded.cos().unsqueeze(0).unsqueeze(2))
        self.register_buffer("sin_cached", freqs_expanded.sin().unsqueeze(0).unsqueeze(2))

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns pre-computed cos and sin frequencies.
        NO INPUTS - everything is pre-computed.
        """
        return self.cos_cached, self.sin_cached


def apply_rotary_emb_static(
    hidden_states: torch.Tensor,  # EXACTLY [B, L, H, HD]
    cos_freqs: torch.Tensor,  # EXACTLY [1, L, 1, HD]
    sin_freqs: torch.Tensor,  # EXACTLY [1, L, 1, HD]
) -> torch.Tensor:
    """
    Apply rotary embeddings - completely static version.
    No shape checks, no branches, pure computation.
    """
    # Split into pairs - no conditionals
    x_pairs = hidden_states.unflatten(-1, (-1, 2))
    x_even, x_odd = x_pairs[..., 0], x_pairs[..., 1]

    # Extract frequencies - fixed indexing
    cos_even = cos_freqs[..., 0::2]
    sin_odd = sin_freqs[..., 1::2]

    # Apply rotation - no allocation in forward pass
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x_even * cos_even - x_odd * sin_odd
    out[..., 1::2] = x_even * sin_odd + x_odd * cos_even

    return out


# ================================================================================================
# Static Normalization (No Optional Parameters)
# ================================================================================================


class StaticRMSNorm(nn.Module):
    """
    RMSNorm with fixed dimension and always-on scaling.
    No optional parameters, no conditionals.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Static forward - no branches."""
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


class StaticLayerNorm(nn.Module):
    """
    LayerNorm that always computes in FP32 internally.
    Fixed behavior, no conditionals.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Always compute in FP32, return in input dtype."""
        dtype = x.dtype
        x_fp32 = x.float()
        normed = self.norm(x_fp32)
        return normed.to(dtype)


# ================================================================================================
# Static Attention Modules (No Optional Parameters)
# ================================================================================================


class StaticSelfAttention(nn.Module):
    """
    Self-attention with completely fixed configuration.
    No optional parameters in forward, all shapes predetermined.
    """

    def __init__(
        self,
        batch_size: int,  # FIXED batch size
        seq_len: int,  # FIXED sequence length
        dim: int = 5120,
        num_heads: int = 40,
        dropout: float = 0.0,  # Fixed at init, not optional in forward
    ):
        super().__init__()

        # Store fixed dimensions
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Fused QKV projection
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)

        # Always-on normalization (no conditionals)
        self.norm_q = StaticRMSNorm(dim)
        self.norm_k = StaticRMSNorm(dim)

        # Output projection
        self.to_out = nn.Linear(dim, dim, bias=True)

        # Fixed dropout (even if 0.0)
        self.dropout = nn.Dropout(dropout)

        # Pre-allocate attention scale
        self.scale = self.head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,  # EXACTLY [batch_size, seq_len, dim]
        cos_freqs: torch.Tensor,  # EXACTLY [1, seq_len, 1, head_dim]
        sin_freqs: torch.Tensor,  # EXACTLY [1, seq_len, 1, head_dim]
    ) -> torch.Tensor:
        """
        Static forward pass - no shape flexibility.
        ALL inputs required, no optionals.
        """
        # Fused QKV projection
        qkv = self.to_qkv(hidden_states)  # [B, L, 3*D]

        # Split and reshape - fixed sizes
        qkv = qkv.reshape(self.batch_size, self.seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, HD]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply normalization - always on
        q = q.transpose(1, 2).reshape(self.batch_size, self.seq_len, self.dim)
        q = self.norm_q(q)
        q = q.reshape(self.batch_size, self.seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        k = k.transpose(1, 2).reshape(self.batch_size, self.seq_len, self.dim)
        k = self.norm_k(k)
        k = k.reshape(self.batch_size, self.seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE - always
        q = q.transpose(1, 2)  # [B, L, H, HD] for RoPE
        k = k.transpose(1, 2)
        q = apply_rotary_emb_static(q, cos_freqs, sin_freqs)
        k = apply_rotary_emb_static(k, cos_freqs, sin_freqs)
        q = q.transpose(1, 2)  # Back to [B, H, L, HD]
        k = k.transpose(1, 2)

        # Attention - no mask option, no dropout option
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)

        # Apply attention
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(self.batch_size, self.seq_len, self.dim)

        # Output projection and dropout (always applied, even if 0.0)
        out = self.to_out(out)
        out = self.dropout(out)

        return out


class StaticCrossAttention(nn.Module):
    """
    Cross-attention with fixed dimensions.
    Context length is fixed at initialization.
    """

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        context_len: int,  # FIXED context length
        dim: int = 5120,
        num_heads: int = 40,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Fixed dimensions
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.context_len = context_len
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Projections
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_kv = nn.Linear(dim, dim * 2, bias=True)

        # Always-on normalization
        self.norm_q = StaticRMSNorm(dim)
        self.norm_k = StaticRMSNorm(dim)

        # Output
        self.to_out = nn.Linear(dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout)

        self.scale = self.head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,  # EXACTLY [batch_size, seq_len, dim]
        context: torch.Tensor,  # EXACTLY [batch_size, context_len, dim]
    ) -> torch.Tensor:
        """
        Static cross-attention. No optional parameters.
        """
        # Query projection
        q = self.to_q(hidden_states)
        q = self.norm_q(q)
        q = q.reshape(self.batch_size, self.seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)

        # Key-value projection
        kv = self.to_kv(context)
        kv = kv.reshape(self.batch_size, self.context_len, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # Normalize keys
        k = k.transpose(1, 2).reshape(self.batch_size, self.context_len, self.dim)
        k = self.norm_k(k)
        k = k.reshape(self.batch_size, self.context_len, self.num_heads, self.head_dim)
        k = k.transpose(1, 2)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)

        # Apply attention
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(self.batch_size, self.seq_len, self.dim)

        # Output projection
        out = self.to_out(out)
        out = self.dropout(out)

        return out


# ================================================================================================
# Static Feed-Forward Network
# ================================================================================================


class StaticFeedForward(nn.Module):
    """
    FFN with fixed activation and dimensions.
    No dynamic behavior.
    """

    def __init__(
        self,
        dim: int = 5120,
        hidden_dim: int = 13824,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.linear1 = nn.Linear(dim, hidden_dim, bias=True)
        self.activation = nn.GELU(approximate="tanh")  # Fixed activation
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Static FFN forward."""
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# ================================================================================================
# Static Transformer Block
# ================================================================================================


class StaticTransformerBlock(nn.Module):
    """
    Complete transformer block with fixed dimensions and no conditionals.
    Modulation is always applied, dropout is always applied (even if 0.0).
    """

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        context_len: int,
        dim: int = 5120,
        num_heads: int = 40,
        ffn_dim: int = 13824,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Store dimensions
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.context_len = context_len
        self.dim = dim

        # Layer norms
        self.norm1 = StaticLayerNorm(dim)
        self.norm2 = StaticLayerNorm(dim)
        self.norm3 = StaticLayerNorm(dim)

        # Attention layers
        self.self_attn = StaticSelfAttention(batch_size, seq_len, dim, num_heads, dropout)

        self.cross_attn = StaticCrossAttention(
            batch_size, seq_len, context_len, dim, num_heads, dropout
        )

        # FFN
        self.ffn = StaticFeedForward(dim, ffn_dim, dropout)

        # Modulation parameters - always used
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / math.sqrt(dim))

    def forward(
        self,
        hidden_states: torch.Tensor,  # EXACTLY [batch_size, seq_len, dim]
        context: torch.Tensor,  # EXACTLY [batch_size, context_len, dim]
        conditioning: torch.Tensor,  # EXACTLY [batch_size, 6, dim]
        cos_freqs: torch.Tensor,  # EXACTLY [1, seq_len, 1, head_dim]
        sin_freqs: torch.Tensor,  # EXACTLY [1, seq_len, 1, head_dim]
    ) -> torch.Tensor:
        """
        Static forward pass. ALL parameters required.
        No conditionals, no optional behavior.
        """
        # Modulation is ALWAYS applied
        modulation = self.scale_shift_table + conditioning
        shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = modulation.chunk(6, dim=1)

        # Self-attention with modulation
        normed = self.norm1(hidden_states)
        modulated = normed * (1 + scale_sa) + shift_sa
        attn_out = self.self_attn(modulated, cos_freqs, sin_freqs)
        hidden_states = hidden_states + gate_sa * attn_out

        # Cross-attention (no modulation)
        normed = self.norm2(hidden_states)
        cross_out = self.cross_attn(normed, context)
        hidden_states = hidden_states + cross_out

        # FFN with modulation
        normed = self.norm3(hidden_states)
        modulated = normed * (1 + scale_ff) + shift_ff
        ffn_out = self.ffn(modulated)
        hidden_states = hidden_states + gate_ff * ffn_out

        return hidden_states


# ================================================================================================
# Static Model Configuration
# ================================================================================================


class StaticWANInferenceModel(nn.Module):
    """
    Complete static inference model with ALL dimensions fixed.
    This is what gets compiled/exported for production.
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
        dropout: float = 0.0,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        # Fixed dimensions
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.context_len = context_len
        self.num_layers = num_layers
        self.dim = dim
        self.head_dim = dim // num_heads

        # Pre-computed position embeddings
        self.rope = StaticRotaryEmbed(seq_len, self.head_dim, device, dtype)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList(
            [
                StaticTransformerBlock(
                    batch_size, seq_len, context_len, dim, num_heads, ffn_dim, dropout
                )
                for _ in range(num_layers)
            ]
        )

        # Output norm
        self.norm_out = StaticLayerNorm(dim)

        # Fixed modulation for output
        self.output_scale_shift = nn.Parameter(torch.randn(1, 2, dim) / math.sqrt(dim))

    def forward(
        self,
        hidden_states: torch.Tensor,  # EXACTLY [batch_size, seq_len, dim]
        context: torch.Tensor,  # EXACTLY [batch_size, context_len, dim]
        block_conditioning: torch.Tensor,  # EXACTLY [batch_size, num_layers, 6, dim]
        output_conditioning: torch.Tensor,  # EXACTLY [batch_size, 2, dim]
    ) -> torch.Tensor:
        """
        Completely static forward pass.
        ALL inputs required with exact shapes.
        """
        # Get pre-computed RoPE
        cos_freqs, sin_freqs = self.rope()

        # Process through blocks - no conditionals
        for i, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                context,
                block_conditioning[:, i],  # Select conditioning for this block
                cos_freqs,
                sin_freqs,
            )

        # Output normalization with modulation
        modulation = self.output_scale_shift + output_conditioning
        shift, scale = modulation.chunk(2, dim=1)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift

        return hidden_states


# ================================================================================================
# Factory Functions for Creating Static Models
# ================================================================================================


def create_static_inference_model(
    batch_size: int,
    frames: int,
    height: int,
    width: int,
    patch_size: tuple = (1, 2, 2),
    context_len: int = 1024,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> StaticWANInferenceModel:
    """
    Create a static inference model with fixed dimensions.

    Args:
        batch_size: Fixed batch size
        frames: Number of video frames
        height: Frame height in pixels
        width: Frame width in pixels
        patch_size: Patch dimensions (t, h, w)
        context_len: Total context length (text + image)
        device: Device to place model on
        dtype: Model dtype

    Returns:
        Static inference model with all dimensions fixed
    """
    # Calculate sequence length from video dimensions
    frames_patched = frames // patch_size[0]
    height_patched = height // patch_size[1]
    width_patched = width // patch_size[2]
    seq_len = frames_patched * height_patched * width_patched

    # Create model
    model = StaticWANInferenceModel(
        batch_size=batch_size,
        seq_len=seq_len,
        context_len=context_len,
        num_layers=48,
        dim=5120,
        num_heads=40,
        ffn_dim=13824,
        dropout=0.0,  # No dropout for inference
        device=device,
        dtype=dtype,
    )

    return model.to(device).to(dtype).eval()


def compile_static_model(
    model: StaticWANInferenceModel,
    mode: str = "max-autotune",
) -> callable:
    """
    Compile static model for maximum performance.

    Args:
        model: Static model to compile
        mode: Compilation mode

    Returns:
        Compiled model
    """
    import torch._dynamo

    # Reset cache for clean compilation
    torch._dynamo.reset()

    # Compile with full graph
    compiled = torch.compile(
        model,
        mode=mode,
        fullgraph=True,  # No graph breaks allowed
        dynamic=False,  # No dynamic shapes
    )

    return compiled


def create_cudagraph_model(
    model: StaticWANInferenceModel,
    example_inputs: tuple,
) -> callable:
    """
    Create CUDAGraph-captured model for minimum latency.

    Args:
        model: Static model to capture
        example_inputs: Example inputs for graph capture

    Returns:
        CUDAGraph-captured callable
    """
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model(*example_inputs)

    # Create graph
    graph = torch.cuda.CUDAGraph()

    # Capture
    with torch.cuda.graph(graph):
        static_output = model(*example_inputs)

    def graphed_forward(*inputs):
        # Copy inputs to static memory
        for static, new in zip(example_inputs, inputs):
            static.copy_(new)
        # Replay graph
        graph.replay()
        return static_output

    return graphed_forward
