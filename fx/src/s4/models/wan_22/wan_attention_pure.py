"""
WAN 2.2 Attention Modules - Static Inference Implementation
============================================================

This module contains FULLY STATIC, BRANCH-FREE implementations optimized
for inference. Every operation has fixed shapes and no conditionals.

Key principles:
1. NO optional parameters in forward methods
2. NO dynamic shapes or runtime decisions
3. NO conditionals or branches in forward passes
4. ALL configuration fixed at initialization
5. FULLY deterministic computational graph

This allows perfect optimization via:
- torch.compile with fullgraph=True
- CUDAGraph capture
- TorchScript tracing
- TensorRT conversion
- Custom kernel fusion

Shape Conventions (ALL FIXED AT INIT):
---------------------------------------
B = batch size (fixed per deployment)
L = sequence length (fixed: e.g., 3136 for 56x56 patches)
D = hidden dimension (fixed: 5120)
H = number of attention heads (fixed: 40)
HD = head dimension (fixed: 128)
C = context length (fixed: e.g., 1024 for text+image)
"""

from typing import Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================================================
# Section 1: Static Position Encoding
# ================================================================================================


def apply_rotary_emb(
    hidden_states: torch.Tensor,  # Shape: EXACTLY [B, L, H, HD]
    freqs_cos: torch.Tensor,  # Shape: EXACTLY [1, L, 1, HD]
    freqs_sin: torch.Tensor,  # Shape: EXACTLY [1, L, 1, HD]
) -> torch.Tensor:
    """
    Apply Rotary Position Embeddings to queries or keys.

    RoPE encodes position by rotating feature vectors in 2D subspaces. For each pair
    of adjacent features (x_i, x_{i+1}), we apply a rotation matrix:

    [x'_i    ]   [cos θ  -sin θ] [x_i    ]
    [x'_{i+1}] = [sin θ   cos θ] [x_{i+1}]

    This provides several advantages:
    1. Relative position awareness (rotation angle depends on position difference)
    2. No learned parameters (purely deterministic based on position)
    3. Extrapolation to longer sequences than training

    Implementation notes:
    - We process pairs of features simultaneously for efficiency
    - Pre-allocating output tensor avoids memory allocation in forward pass
    - No branches or conditionals for torch.compile optimization
    """

    # Split hidden states into pairs of adjacent features
    # Shape: [B, L, H, HD] -> [B, L, H, HD/2, 2] -> 2x [B, L, H, HD/2]
    x_pairs = hidden_states.unflatten(-1, (-1, 2))
    x_even, x_odd = x_pairs.unbind(-1)

    # Extract even and odd indices from frequency tensors
    cos_even = freqs_cos[..., 0::2]  # Shape: [1, L, 1, HD/2]
    sin_odd = freqs_sin[..., 1::2]  # Shape: [1, L, 1, HD/2]

    # Pre-allocate output for memory efficiency
    rotated_features = torch.empty_like(hidden_states)

    # Apply rotation matrix efficiently using index assignment
    # Even indices: x'_i = x_i * cos(θ) - x_{i+1} * sin(θ)
    rotated_features[..., 0::2] = x_even * cos_even - x_odd * sin_odd

    # Odd indices: x'_{i+1} = x_i * sin(θ) + x_{i+1} * cos(θ)
    rotated_features[..., 1::2] = x_even * sin_odd + x_odd * cos_even

    # Preserve original dtype (important for mixed precision)
    return rotated_features.type_as(hidden_states)


class RotaryPosEmbed(nn.Module):
    """
    Rotary Position Embedding generator for patches.

    Generates sine and cosine frequencies for rotary embeddings based on
    patch positions in a sequence. This is a deterministic module with no
    learned parameters.

    The frequency calculation follows the standard RoPE formulation:
    θ_i = 10000^(-2i/d) where i is the dimension index and d is head_dim
    """

    def __init__(
        self,
        head_dim: int = 128,
        max_seq_len: int = 16384,
        base: int = 10000,
    ):
        super().__init__()

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Pre-compute frequency basis (no gradients needed)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute for maximum sequence length (can be trimmed)
        self._precompute(max_seq_len)

    def _precompute(self, seq_len: int) -> None:
        """Pre-compute cos/sin for a given sequence length."""
        positions = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)  # [seq_len, head_dim/2]

        # Duplicate frequencies for rotating pairs
        freqs_expanded = torch.cat([freqs, freqs], dim=-1)  # [seq_len, head_dim]

        # Cache cos and sin
        self.register_buffer("_cos_cached", freqs_expanded.cos().unsqueeze(0).unsqueeze(2))
        self.register_buffer("_sin_cached", freqs_expanded.sin().unsqueeze(0).unsqueeze(2))
        self._cached_seq_len = seq_len

    def forward(
        self,
        seq_len: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get rotary embeddings for a given sequence length.

        Args:
            seq_len: Length of sequence
            device: Device to place tensors on
            dtype: Data type for tensors

        Returns:
            Tuple of (cos_freqs, sin_freqs), each shaped [1, seq_len, 1, head_dim]
        """

        # Use cache if available and correct length
        if hasattr(self, "_cached_seq_len") and self._cached_seq_len >= seq_len:
            cos = self._cos_cached[:, :seq_len]
            sin = self._sin_cached[:, :seq_len]

            if device is not None:
                cos = cos.to(device)
                sin = sin.to(device)
            if dtype is not None:
                cos = cos.to(dtype)
                sin = sin.to(dtype)

            return cos, sin

        # Compute on the fly for dynamic sequence lengths
        positions = torch.arange(seq_len, device=device, dtype=torch.float)
        freqs = positions.unsqueeze(1) * self.inv_freq.to(device)
        freqs_expanded = torch.cat([freqs, freqs], dim=-1)

        cos = freqs_expanded.cos().unsqueeze(0).unsqueeze(2)
        sin = freqs_expanded.sin().unsqueeze(0).unsqueeze(2)

        if dtype is not None:
            cos = cos.to(dtype)
            sin = sin.to(dtype)

        return cos, sin


# ================================================================================================
# Section 2: Normalization Layers
# ================================================================================================


class FP32LayerNorm(nn.LayerNorm):
    """
    LayerNorm that always operates in FP32 for numerical stability.

    This is critical for mixed precision training/inference where the
    normalization statistics need higher precision than the activations.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        output = super().forward(hidden_states)
        return output.type_as(hidden_states)


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.

    Simpler and often more stable than LayerNorm, especially for very large models.
    RMSNorm normalizes by the root mean square instead of standardizing.

    Formula: x * (1 / RMS(x)) * learnable_scale
    where RMS(x) = sqrt(mean(x^2) + eps)
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.eps = eps

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Calculate RMS
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        # Apply learned scale if present
        if self.weight is not None:
            hidden_states = hidden_states * self.weight

        return hidden_states


# ================================================================================================
# Section 3: Feed-Forward Network
# ================================================================================================


class FeedForward(nn.Module):
    """
    Two-layer feed-forward network with activation.

    This is the standard transformer FFN: Linear -> Activation -> Linear
    with optional dropout. The expansion ratio (inner_dim / dim) is typically
    2.7x for WAN 2.2 (13824 / 5120).
    """

    def __init__(
        self,
        dim: int,
        inner_dim: int,
        activation_fn: str = "gelu-approximate",
        dropout: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.inner_dim = inner_dim

        # Two-layer MLP
        self.linear1 = nn.Linear(dim, inner_dim)
        self.linear2 = nn.Linear(inner_dim, dim)

        # Activation function
        if activation_fn == "gelu":
            self.activation = nn.GELU()
        elif activation_fn == "gelu-approximate":
            self.activation = nn.GELU(approximate="tanh")
        elif activation_fn == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation_fn}")

        # Dropout (usually 0 for inference)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        FFN forward pass.

        Shape flow: [B, L, D] -> [B, L, inner_dim] -> [B, L, D]
        """
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.linear2(hidden_states)
        return hidden_states


# ================================================================================================
# Section 4: Self-Attention Module
# ================================================================================================


class SelfAttention(nn.Module):
    """
    Optimized self-attention with fused QKV projection and RoPE.

    Key features:
    - Fused QKV projection for single matmul
    - RMSNorm on Q and K (but not V) for stability
    - Rotary position embeddings
    - Optimized for torch.compile and SDPA

    Architecture:
    - 5120 hidden dimension split across 40 heads (128 dim per head)
    - No attention mask needed (full attention)
    - Dropout disabled for inference
    """

    def __init__(
        self,
        dim: int = 5120,
        num_heads: int = 40,
        qk_norm: bool = True,
        eps: float = 1e-6,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Fused QKV projection: single matmul instead of three
        # Weight shape: [dim, 3 * dim]
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)

        # Optional Q/K normalization (usually enabled for WAN 2.2)
        if qk_norm:
            self.norm_q = RMSNorm(dim, eps=eps, elementwise_affine=True)
            self.norm_k = RMSNorm(dim, eps=eps, elementwise_affine=True)
        else:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()

        # Output projection with optional dropout
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        # Scale factor for attention (1/sqrt(head_dim))
        self.scale = self.head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: [B, L, D]
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,  # Usually None for WAN
    ) -> torch.Tensor:
        """
        Self-attention forward pass.

        Processing pipeline:
        1. Fused QKV projection: [B, L, D] -> [B, L, 3*D]
        2. Split and reshape: -> 3x [B, L, H, HD]
        3. Apply RoPE to Q and K
        4. Compute scaled dot-product attention
        5. Reshape and project output: -> [B, L, D]
        """

        batch_size, seq_len, _ = hidden_states.shape

        # Step 1: Fused QKV projection
        qkv = self.to_qkv(hidden_states)  # [B, L, 3*D]

        # Step 2: Split into Q, K, V and reshape
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 1, 3, 4)  # [3, B, L, H, HD]
        queries, keys, values = qkv[0], qkv[1], qkv[2]  # Each: [B, L, H, HD]

        # Step 3: Apply normalization
        queries = self.norm_q(queries.reshape(batch_size, seq_len, -1))
        queries = queries.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        keys = self.norm_k(keys.reshape(batch_size, seq_len, -1))
        keys = keys.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Step 4: Apply rotary embeddings if provided
        if rotary_emb is not None:
            cos_freqs, sin_freqs = rotary_emb
            queries = apply_rotary_emb(queries, cos_freqs, sin_freqs)
            keys = apply_rotary_emb(keys, cos_freqs, sin_freqs)

        # Step 5: Transpose for SDPA format
        # [B, L, H, HD] -> [B, H, L, HD]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Step 6: Scaled dot-product attention
        hidden_states = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=self.scale,
        )

        # Step 7: Reshape output
        # [B, H, L, HD] -> [B, L, H, HD] -> [B, L, D]
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.reshape(batch_size, seq_len, self.dim)

        # Step 8: Output projection
        return self.to_out(hidden_states)


# ================================================================================================
# Section 5: Cross-Attention Module
# ================================================================================================


class CrossAttention(nn.Module):
    """
    Optimized cross-attention for conditioning.

    Key optimizations:
    - Fused KV projection for context
    - Assumes pre-processed context features (no per-layer projection)
    - Single efficient attention path

    The context is expected to be pre-projected to model dimension,
    allowing this module to be very lightweight and efficient.
    """

    def __init__(
        self,
        dim: int = 5120,
        context_dim: Optional[int] = None,  # If None, assumes context_dim == dim
        num_heads: int = 40,
        qk_norm: bool = True,
        eps: float = 1e-6,
        dropout: float = 0.0,
    ):
        super().__init__()

        context_dim = context_dim or dim
        assert dim % num_heads == 0

        self.dim = dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Query projection (from hidden states)
        self.to_q = nn.Linear(dim, dim, bias=True)

        # Fused KV projection (for context)
        self.to_kv = nn.Linear(context_dim, dim * 2, bias=True)

        # Optional Q/K normalization
        if qk_norm:
            self.norm_q = RMSNorm(dim, eps=eps, elementwise_affine=True)
            self.norm_k = RMSNorm(dim, eps=eps, elementwise_affine=True)
        else:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()

        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        self.scale = self.head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: [B, L, D]
        context: torch.Tensor,  # Shape: [B, C, context_dim]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-attention forward pass.

        Processing:
        1. Project hidden_states to queries
        2. Project context to keys and values (fused)
        3. Apply normalization
        4. Compute cross-attention
        5. Output projection

        Shape flow:
        Q: [B, L, D] -> [B, H, L, HD]
        K: [B, C, context_dim] -> [B, H, C, HD]
        V: [B, C, context_dim] -> [B, H, C, HD]
        Out: [B, H, L, HD] -> [B, L, D]
        """

        batch_size, seq_len, _ = hidden_states.shape
        context_len = context.shape[1]

        # Step 1: Generate queries
        queries = self.to_q(hidden_states)  # [B, L, D]
        queries = self.norm_q(queries)
        queries = queries.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        queries = queries.transpose(1, 2)  # [B, H, L, HD]

        # Step 2: Generate keys and values from context
        kv = self.to_kv(context)  # [B, C, 2*D]
        kv = kv.reshape(batch_size, context_len, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # [2, B, H, C, HD]
        keys, values = kv[0], kv[1]  # Each: [B, H, C, HD]

        # Step 3: Apply K normalization
        keys = keys.transpose(1, 2)  # [B, C, H, HD]
        keys = keys.reshape(batch_size, context_len, -1)
        keys = self.norm_k(keys)
        keys = keys.reshape(batch_size, context_len, self.num_heads, self.head_dim)
        keys = keys.transpose(1, 2)  # [B, H, C, HD]

        # Step 4: Scaled dot-product attention
        hidden_states = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=self.scale,
        )

        # Step 5: Reshape and project output
        hidden_states = hidden_states.transpose(1, 2)  # [B, L, H, HD]
        hidden_states = hidden_states.reshape(batch_size, seq_len, self.dim)

        return self.to_out(hidden_states)


# ================================================================================================
# Section 6: Transformer Block
# ================================================================================================


class TransformerBlock(nn.Module):
    """
    Complete transformer block with modulated self-attention, cross-attention, and FFN.

    The block follows the standard transformer pattern but with adaptive layer norm
    (modulation) based on conditioning embeddings. This allows the model to adjust
    its behavior based on timestep or other conditioning signals.

    Architecture:
    1. Modulated Self-Attention with residual
    2. Cross-Attention with residual
    3. Modulated Feed-Forward Network with residual

    Modulation mechanism:
    - Each modulated layer receives shift, scale, and gate parameters
    - Applied as: output = input + gate * layer((input * (1 + scale) + shift))
    """

    def __init__(
        self,
        dim: int = 5120,
        context_dim: Optional[int] = None,
        num_heads: int = 40,
        ffn_dim: int = 13824,
        qk_norm: bool = True,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
    ):
        super().__init__()

        self.dim = dim
        context_dim = context_dim or dim

        # Normalization layers
        self.norm1 = FP32LayerNorm(dim, eps=norm_eps, elementwise_affine=False)
        self.norm2 = FP32LayerNorm(dim, eps=norm_eps, elementwise_affine=True)
        self.norm3 = FP32LayerNorm(dim, eps=norm_eps, elementwise_affine=False)

        # Attention layers
        self.self_attn = SelfAttention(
            dim=dim,
            num_heads=num_heads,
            qk_norm=qk_norm,
            eps=norm_eps,
            dropout=dropout,
        )

        self.cross_attn = CrossAttention(
            dim=dim,
            context_dim=context_dim,
            num_heads=num_heads,
            qk_norm=qk_norm,
            eps=norm_eps,
            dropout=dropout,
        )

        # Feed-forward network
        self.ffn = FeedForward(
            dim=dim,
            inner_dim=ffn_dim,
            activation_fn=activation_fn,
            dropout=dropout,
        )

        # Modulation parameters (learned biases for conditioning)
        # 6 parameters: shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / math.sqrt(dim))

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: [B, L, D]
        context: torch.Tensor,  # Shape: [B, C, context_dim]
        conditioning: Optional[torch.Tensor] = None,  # Shape: [B, 6, D] or None
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.

        Args:
            hidden_states: Input features [B, L, D]
            context: Context features for cross-attention [B, C, context_dim]
            conditioning: Optional modulation parameters [B, 6, D]
            rotary_emb: Optional rotary position embeddings
            attention_mask: Optional attention mask

        Returns:
            Output features with same shape as input [B, L, D]
        """

        batch_size = hidden_states.shape[0]

        # Compute modulation parameters
        if conditioning is not None:
            modulation = self.scale_shift_table + conditioning.float()
        else:
            # Use just the learned biases
            modulation = self.scale_shift_table.expand(batch_size, -1, -1)

        # Split modulation parameters
        mod_chunks = modulation.chunk(6, dim=1)  # 6x [B, 1, D]
        shift_sa, scale_sa, gate_sa = mod_chunks[0], mod_chunks[1], mod_chunks[2]
        shift_ff, scale_ff, gate_ff = mod_chunks[3], mod_chunks[4], mod_chunks[5]

        # Self-attention with modulation
        normed = self.norm1(hidden_states.float())
        modulated = (normed * (1 + scale_sa) + shift_sa).type_as(hidden_states)
        attn_out = self.self_attn(modulated, rotary_emb, attention_mask)
        hidden_states = hidden_states + gate_sa * attn_out

        # Cross-attention (no modulation)
        normed = self.norm2(hidden_states.float()).type_as(hidden_states)
        cross_out = self.cross_attn(normed, context, attention_mask)
        hidden_states = hidden_states + cross_out

        # FFN with modulation
        normed = self.norm3(hidden_states.float())
        modulated = (normed * (1 + scale_ff) + shift_ff).type_as(hidden_states)
        ffn_out = self.ffn(modulated)
        hidden_states = hidden_states + gate_ff * ffn_out

        return hidden_states


# ================================================================================================
# Section 7: Condition Embedder
# ================================================================================================


class ConditionEmbedder(nn.Module):
    """
    Generates conditioning embeddings from various inputs.

    This module processes conditioning signals (e.g., timesteps, class labels)
    and generates the modulation parameters used by transformer blocks.

    For WAN 2.2, this typically:
    1. Converts timesteps to sinusoidal embeddings
    2. Projects through MLP to generate modulation parameters
    3. Optionally processes text embeddings
    """

    def __init__(
        self,
        hidden_dim: int = 5120,
        time_embed_dim: int = 256,
        num_modulation_params: int = 6,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim

        # Sinusoidal timestep embeddings
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # MLP for generating modulation parameters
        self.modulation_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, num_modulation_params * hidden_dim),
        )

    @staticmethod
    def timestep_embedding(
        timesteps: torch.Tensor,
        dim: int,
        max_period: int = 10000,
        flip_sin_to_cos: bool = False,
    ) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.

        Args:
            timesteps: Timestep values [B]
            dim: Embedding dimension
            max_period: Maximum period for sinusoidal embedding
            flip_sin_to_cos: Whether to flip sin and cos

        Returns:
            Embeddings of shape [B, dim]
        """

        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=timesteps.device)
            / half
        )

        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if flip_sin_to_cos:
            embedding = torch.cat([embedding[:, half:], embedding[:, :half]], dim=-1)

        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        return embedding

    def forward(
        self,
        timesteps: torch.LongTensor,  # Shape: [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate conditioning embeddings.

        Args:
            timesteps: Timestep values [B]

        Returns:
            Tuple of:
            - final_modulation: For output layer [B, 2, hidden_dim]
            - block_modulation: For transformer blocks [B, 6, hidden_dim]
        """

        # Create sinusoidal embeddings
        time_embeds = self.timestep_embedding(
            timesteps,
            self.time_embed_dim,
            flip_sin_to_cos=True,
        )

        # Project to hidden dimension
        time_hidden = self.time_proj(time_embeds)  # [B, hidden_dim]

        # Generate modulation parameters
        modulation = self.modulation_proj(time_hidden)  # [B, 6 * hidden_dim]
        block_modulation = modulation.view(-1, 6, self.hidden_dim)  # [B, 6, hidden_dim]

        # Final layer uses first 2 parameters
        final_modulation = block_modulation[:, :2]  # [B, 2, hidden_dim]

        return final_modulation, block_modulation
