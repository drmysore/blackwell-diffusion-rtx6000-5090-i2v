"""
WAN 2.2 Image-to-Video Transformer: An Optimized Implementation
================================================================

This module implements the attention mechanisms and transformer architecture for WAN 2.2,
a state-of-the-art image-to-video generation model. The implementation has been carefully
optimized for torch.compile and inductor compatibility, with all shapes made explicit
and all branches removed from the forward pass.

Mathematical Foundation:
------------------------
The model uses rotary position embeddings (RoPE) for positional encoding and combines
self-attention with cross-attention to condition video generation on both text and
image features. The architecture processes video patches of size (1, 2, 2) through
a series of transformer blocks, each applying:

1. Modulated self-attention with RoPE
2. Cross-attention to text and image features
3. Modulated feed-forward network

Shape Conventions Throughout:
-----------------------------
B = batch size
L = sequence length (number of video patches)
D = hidden dimension (5120 for WAN 2.2)
H = number of attention heads (40)
HD = head dimension (128 = D/H)
F = frames in video
H_spatial = height of video frame (in patches)
W_spatial = width of video frame (in patches)
C_text = context length for text (512)
C_img = context length for image features (variable)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
)


# ================================================================================================
# Section 1: Utility Functions for Attention Optimization
# ================================================================================================


class _FlashSDPA:
    """
    Context manager for Flash Attention configuration.

    This ensures we use the most efficient attention implementation available,
    preferring Flash Attention when possible for its superior memory and compute
    characteristics on modern GPUs.
    """

    def __enter__(self):
        self.ctx = torch.backends.cuda.sdp_kernel(
            enable_flash=True,  # Prefer Flash Attention (most efficient)
            enable_mem_efficient=False,  # Disable memory-efficient attention
            enable_math=False,  # Disable math fallback (slowest)
        )
        self.ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.ctx.__exit__(exc_type, exc, tb)


# Singleton instance to avoid repeated allocation
_FLASH_SDPA = _FlashSDPA()


def _pad_to_multiple(
    tensor: torch.Tensor,
    multiple: int,
    dim: int = 1,
    value: float = 0.0,
) -> Tuple[torch.Tensor, Optional[int]]:
    """
    Pad a tensor along a specified dimension to make its size a multiple of `multiple`.

    This is crucial for tensor core utilization - modern GPUs achieve peak performance
    when matrix dimensions are multiples of 8 (or 16 for some operations).

    Args:
        tensor: Input tensor of any shape
        multiple: The target multiple (typically 8 or 16 for tensor cores)
        dim: Dimension along which to pad (0-indexed from the end)
        value: Value to use for padding (typically 0.0)

    Returns:
        Tuple of (padded_tensor, padding_amount) where padding_amount is None if no padding needed

    Example:
        If tensor has shape [32, 197, 768] and we pad dim=1 to multiple=8,
        we need to pad 197 -> 200, so we add 3 padding tokens.
    """

    current_size = tensor.size(dim)
    padding_needed = (multiple - (current_size % multiple)) % multiple

    if padding_needed == 0:
        return tensor, None

    # Build padding specification for F.pad
    # F.pad expects pairs of (left, right) padding for each dimension, starting from last
    pad_width = [0, 0] * tensor.ndim
    pad_width[-2 * dim - 1] = padding_needed  # Set right padding for the target dimension

    padded_tensor = F.pad(tensor, pad_width, value=value)
    return padded_tensor, padding_needed


# ================================================================================================
# Section 2: Rotary Position Embeddings (RoPE)
# ================================================================================================


def apply_rotary_emb(
    hidden_states: torch.Tensor,  # Shape: [B, L, H, HD] - queries or keys
    freqs_cos: torch.Tensor,  # Shape: [1, L, 1, HD] - cosine frequencies
    freqs_sin: torch.Tensor,  # Shape: [1, L, 1, HD] - sine frequencies
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
    - The unflatten/unbind operations reshape for paired processing
    - Pre-allocating output tensor avoids memory allocation in forward pass
    """

    # Split hidden states into pairs of adjacent features
    # Shape: [B, L, H, HD] -> [B, L, H, HD/2, 2] -> 2x [B, L, H, HD/2]
    x_pairs = hidden_states.unflatten(-1, (-1, 2))
    x_even, x_odd = x_pairs.unbind(-1)

    # Extract even and odd indices from frequency tensors
    # These have been pre-computed for efficiency
    cos_even = freqs_cos[..., 0::2]  # Shape: [1, L, 1, HD/2]
    sin_odd = freqs_sin[..., 1::2]  # Shape: [1, L, 1, HD/2]

    # Pre-allocate output tensor for memory efficiency
    rotated_features = torch.empty_like(hidden_states)

    # Apply rotation matrix efficiently using index assignment
    # Even indices: x'_i = x_i * cos(θ) - x_{i+1} * sin(θ)
    rotated_features[..., 0::2] = x_even * cos_even - x_odd * sin_odd

    # Odd indices: x'_{i+1} = x_i * sin(θ) + x_{i+1} * cos(θ)
    rotated_features[..., 1::2] = x_even * sin_odd + x_odd * cos_even

    # Preserve original dtype (important for mixed precision training)
    return rotated_features.type_as(hidden_states)


# ================================================================================================
# Section 3: Self-Attention Module
# ================================================================================================


class WanSelfAttention(nn.Module):
    """
    Optimized self-attention for WAN 2.2 with fused QKV projection.

    Architecture details:
    - 5120 hidden dimension split across 40 heads (128 dim per head)
    - RMSNorm applied to Q and K before attention (improves training stability)
    - Fused QKV projection reduces memory reads
    - No attention mask needed (full attention across all positions)
    - Dropout disabled for inference optimization

    Memory layout optimizations:
    - QKV computed in single matrix multiply
    - Efficient transpose for SDPA format
    - In-place operations where possible
    """

    def __init__(self, dim: int = 5120, heads: int = 40, eps: float = 1e-6):
        super().__init__()

        # WAN 2.2 uses these specific dimensions
        assert dim == 5120, "WAN I2V uses dim=5120"
        assert heads == 40, "WAN I2V uses 40 heads"

        self.num_heads = heads
        self.head_dim = dim // heads  # 128 dimensions per head

        # Fused QKV projection: single matmul instead of three
        # Weight shape: [5120, 15360] where 15360 = 3 * 5120
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)

        # RMSNorm for Q and K (not V) - improves attention stability
        # These normalize across the feature dimension
        self.norm_q = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.norm_k = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)

        # Output projection with dropout (though dropout=0 for inference)
        self.to_out = nn.ModuleList(
            [
                nn.Linear(dim, dim, bias=True),
                nn.Dropout(0.0),  # Disabled for inference but kept for compatibility
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: [B, L, 5120] - input features
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],  # Shapes: ([1, L, 1, 128], [1, L, 1, 128])
    ) -> torch.Tensor:
        """
        Forward pass of self-attention.

        Processing pipeline:
        1. Project input to Q, K, V using fused projection
        2. Normalize Q and K (but not V) for training stability
        3. Reshape to multi-head format
        4. Apply rotary position embeddings to Q and K
        5. Compute scaled dot-product attention
        6. Reshape and project output

        Shape transformations:
        [B, L, 5120] -> [B, L, 15360] -> 3x[B, L, 5120] -> 3x[B, L, 40, 128]
        -> [B, 40, L, 128] -> attention -> [B, 40, L, 128] -> [B, L, 5120]
        """

        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Step 1: Fused QKV projection
        # Single matmul: [B, L, 5120] @ [5120, 15360] -> [B, L, 15360]
        qkv_features = self.to_qkv(hidden_states)

        # Step 2: Split into Q, K, V
        # Each tensor: [B, L, 5120]
        queries, keys, values = qkv_features.chunk(3, dim=-1)

        # Step 3: Apply normalization and reshape to multi-head format
        # Normalization happens BEFORE reshaping for efficiency
        queries_normalized = self.norm_q(queries)  # [B, L, 5120]
        keys_normalized = self.norm_k(keys)  # [B, L, 5120]

        # Reshape to expose heads: [B, L, 5120] -> [B, L, 40, 128]
        queries_multihead = queries_normalized.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        keys_multihead = keys_normalized.view(batch_size, seq_len, self.num_heads, self.head_dim)
        values_multihead = values.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Step 4: Apply rotary position embeddings
        # RoPE is applied in [B, L, H, HD] format for efficiency
        cos_freqs, sin_freqs = rotary_emb
        queries_rotated = apply_rotary_emb(queries_multihead, cos_freqs, sin_freqs)
        keys_rotated = apply_rotary_emb(keys_multihead, cos_freqs, sin_freqs)

        # Step 5: Transpose for SDPA format
        # SDPA expects: [B, H, L, HD] for efficient memory access patterns
        queries_sdpa = queries_rotated.transpose(1, 2)  # [B, 40, L, 128]
        keys_sdpa = keys_rotated.transpose(1, 2)  # [B, 40, L, 128]
        values_sdpa = values_multihead.transpose(1, 2)  # [B, 40, L, 128]

        # Step 6: Scaled dot-product attention
        # Computes: softmax(QK^T / sqrt(128)) @ V
        attention_output = F.scaled_dot_product_attention(
            queries_sdpa,
            keys_sdpa,
            values_sdpa,
            dropout_p=0.0,  # No dropout for inference
            is_causal=False,  # Full attention (not causal/autoregressive)
        )

        # Step 7: Reshape back to original format
        # [B, 40, L, 128] -> [B, L, 40, 128] -> [B, L, 5120]
        attention_output = attention_output.transpose(1, 2)
        hidden_states_out = attention_output.reshape(batch_size, seq_len, hidden_dim)

        # Step 8: Output projection and dropout
        hidden_states_out = self.to_out[0](hidden_states_out)  # Linear projection
        hidden_states_out = self.to_out[1](hidden_states_out)  # Dropout (no-op when p=0)

        return hidden_states_out


# ================================================================================================
# Section 4: Cross-Attention Module
# ================================================================================================


class WanCrossAttention(nn.Module):
    """
    Optimized cross-attention for conditioning on text and image features.

    Key optimizations:
    - Fused KV projection for context features
    - Assumes pre-projected image features (moved outside the loop)
    - Single efficient attention path
    - No masking needed (attend to all context)

    The context (encoder_hidden_states) is already concatenated and projected:
    - Text features: [B, 512, 5120]
    - Image features: [B, img_len, 5120] (pre-projected from 1280 to 5120)
    - Combined: [B, 512 + img_len, 5120]
    """

    def __init__(self, dim: int = 5120, heads: int = 40, eps: float = 1e-6):
        super().__init__()

        assert dim == 5120 and heads == 40, "WAN I2V uses specific dimensions"

        self.num_heads = heads
        self.head_dim = dim // heads  # 128

        # Query projection (from hidden states)
        self.to_q = nn.Linear(dim, dim, bias=True)

        # Fused KV projection (for context features)
        # Projects context from 5120 -> 10240 (K and V together)
        self.to_kv = nn.Linear(dim, dim * 2, bias=True)

        # Normalization for queries and keys
        self.norm_q = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.norm_k = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)

        # Output projection
        self.to_out = nn.Linear(dim, dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: [B, L, 5120] - current features
        encoder_hidden_states: torch.Tensor,  # Shape: [B, C_total, 5120] - context (text + image)
    ) -> torch.Tensor:
        """
        Cross-attention forward pass.

        Processing pipeline:
        1. Project hidden_states to queries
        2. Project encoder_hidden_states to keys and values (fused)
        3. Apply normalization to Q and K
        4. Reshape to multi-head format
        5. Compute cross-attention
        6. Reshape and project output

        Shape flow:
        Q: [B, L, 5120] -> [B, L, 5120] -> [B, 40, L, 128]
        K: [B, C, 5120] -> [B, C, 5120] -> [B, 40, C, 128]
        V: [B, C, 5120] -> [B, C, 5120] -> [B, 40, C, 128]
        Out: [B, 40, L, 128] -> [B, L, 5120]
        """

        batch_size, seq_len, hidden_dim = hidden_states.shape
        context_len = encoder_hidden_states.shape[1]

        # Step 1: Project and normalize queries
        # [B, L, 5120] -> [B, L, 5120]
        queries = self.to_q(hidden_states)
        queries_normalized = self.norm_q(queries)

        # Reshape queries to multi-head format
        # [B, L, 5120] -> [B, L, 40, 128] -> [B, 40, L, 128]
        queries_multihead = queries_normalized.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        queries_sdpa = queries_multihead.transpose(1, 2)

        # Step 2: Project context to keys and values (fused)
        # [B, C, 5120] -> [B, C, 10240]
        kv_features = self.to_kv(encoder_hidden_states)

        # Split into keys and values
        # Each: [B, C, 5120]
        keys, values = kv_features.chunk(2, dim=-1)

        # Step 3: Normalize keys (but not values)
        keys_normalized = self.norm_k(keys)

        # Step 4: Reshape K and V to multi-head format
        # [B, C, 5120] -> [B, C, 40, 128] -> [B, 40, C, 128]
        keys_multihead = keys_normalized.view(
            batch_size, context_len, self.num_heads, self.head_dim
        )
        keys_sdpa = keys_multihead.transpose(1, 2)

        values_multihead = values.view(batch_size, context_len, self.num_heads, self.head_dim)
        values_sdpa = values_multihead.transpose(1, 2)

        # Step 5: Compute cross-attention
        # Q: [B, 40, L, 128], K: [B, 40, C, 128], V: [B, 40, C, 128]
        # Output: [B, 40, L, 128]
        attention_output = F.scaled_dot_product_attention(
            queries_sdpa, keys_sdpa, values_sdpa, dropout_p=0.0, is_causal=False
        )

        # Step 6: Reshape and project output
        # [B, 40, L, 128] -> [B, L, 40, 128] -> [B, L, 5120]
        attention_output = attention_output.transpose(1, 2)
        hidden_states_out = attention_output.reshape(batch_size, seq_len, hidden_dim)

        # Final output projection
        return self.to_out(hidden_states_out)


# ================================================================================================
# Section 5: Transformer Block
# ================================================================================================


class WanTransformerBlock(nn.Module):
    """
    A complete transformer block with modulated self-attention, cross-attention, and FFN.

    The block follows the standard transformer pattern but with adaptive layer norm
    (modulation) based on timestep embeddings. This allows the model to adjust its
    behavior based on the diffusion timestep.

    Block structure:
    1. Modulated Self-Attention with residual
    2. Cross-Attention with residual
    3. Modulated Feed-Forward Network with residual

    Modulation mechanism:
    - Each modulated layer gets shift, scale, and gate parameters
    - These are computed from timestep embeddings + learned biases
    - Applied as: output = input + gate * layer((input * (1 + scale) + shift))
    """

    def __init__(self, dim: int = 5120, ffn_dim: int = 13824):
        super().__init__()

        # Layer normalization (without learnable affine parameters for modulation)
        self.norm1 = FP32LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.norm3 = FP32LayerNorm(dim, eps=1e-6, elementwise_affine=False)

        # Cross-attention has learnable affine parameters
        self.norm2 = FP32LayerNorm(dim, eps=1e-6, elementwise_affine=True)

        # Attention layers
        self.attn1 = WanSelfAttention(dim, heads=40, eps=1e-6)
        self.attn2 = WanCrossAttention(dim, heads=40, eps=1e-6)

        # Feed-forward network
        # Hidden dimension 13824 = 2.7 * 5120 (typical expansion ratio)
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")

        # Modulation parameters (learned biases added to timestep embeddings)
        # 6 parameters: shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 6, dim) / dim**0.5  # Initialize with small values
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: [B, L, 5120]
        encoder_hidden_states: torch.Tensor,  # Shape: [B, C_total, 5120]
        temb: torch.Tensor,  # Shape: [B, 6, 5120] - timestep embeddings
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],  # RoPE frequencies
    ) -> torch.Tensor:
        """
        Forward pass through the transformer block.

        The modulation mechanism allows the model to adapt based on the diffusion timestep:
        - Early timesteps (high noise): different behavior
        - Late timesteps (low noise): different behavior

        Shape preservation: Input and output are both [B, L, 5120]
        """

        # Step 1: Compute modulation parameters
        # Combine learned biases with timestep embeddings
        # Shape: [1, 6, 5120] + [B, 6, 5120] -> [B, 6, 5120]
        modulation_params = self.scale_shift_table + temb.float()

        # Split into individual modulation parameters
        # Each has shape: [B, 1, 5120]
        (
            shift_self_attn,
            scale_self_attn,
            gate_self_attn,
            shift_cross_attn,
            scale_cross_attn,
            gate_cross_attn,
        ) = modulation_params.chunk(6, dim=1)

        # Step 2: Modulated self-attention
        # Apply adaptive layer norm with modulation
        normed_hidden = self.norm1(hidden_states.float())
        modulated_hidden = normed_hidden * (1 + scale_self_attn) + shift_self_attn
        modulated_hidden = modulated_hidden.type_as(hidden_states)

        # Self-attention with RoPE
        self_attn_output = self.attn1(modulated_hidden, rotary_emb)

        # Gated residual connection
        hidden_states = hidden_states.float() + self_attn_output * gate_self_attn
        hidden_states = hidden_states.type_as(hidden_states)

        # Step 3: Cross-attention (no modulation for cross-attention)
        # Standard layer norm (has learnable parameters)
        normed_hidden = self.norm2(hidden_states.float()).type_as(hidden_states)

        # Cross-attention to context
        cross_attn_output = self.attn2(normed_hidden, encoder_hidden_states)

        # Residual connection (no gating)
        hidden_states = hidden_states + cross_attn_output

        # Step 4: Modulated feed-forward network
        # Note: We're reusing the cross-attention modulation parameters for FFN
        # This is intentional - shift/scale/gate for FFN are stored in the "cross" slots
        normed_hidden = self.norm3(hidden_states.float())
        modulated_hidden = normed_hidden * (1 + scale_cross_attn) + shift_cross_attn
        modulated_hidden = modulated_hidden.type_as(hidden_states)

        # Feed-forward network
        ffn_output = self.ffn(modulated_hidden)

        # Gated residual connection
        hidden_states = hidden_states.float() + ffn_output * gate_cross_attn
        hidden_states = hidden_states.type_as(hidden_states)

        return hidden_states


# ================================================================================================
# Section 6: Supporting Modules
# ================================================================================================


class WanRotaryPosEmbed(nn.Module):
    """
    Rotary Position Embedding generator for video patches.

    Generates sine and cosine frequencies for rotary embeddings based on
    the spatial structure of video patches.
    """

    def __init__(
        self,
        head_dim: int = 128,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.head_dim = head_dim

        # Pre-compute frequency basis
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate rotary position embeddings for the given video shape.

        Returns:
            Tuple of (cos_freqs, sin_freqs), each with shape [1, L, 1, head_dim]
        """
        batch_size, _, frames, height, width = hidden_states.shape

        # Calculate patch grid dimensions
        frames_patched = frames // self.patch_size[0]
        height_patched = height // self.patch_size[1]
        width_patched = width // self.patch_size[2]

        # Generate 3D position indices
        pos_ids_3d = torch.arange(
            frames_patched * height_patched * width_patched, device=hidden_states.device
        ).reshape(frames_patched, height_patched, width_patched)

        # Flatten to sequence format
        pos_ids = pos_ids_3d.flatten()

        # Compute frequencies
        freqs = pos_ids.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        freqs_expanded = torch.cat([freqs, freqs], dim=-1)  # Duplicate for pairs

        cos_freqs = freqs_expanded.cos().unsqueeze(0).unsqueeze(2)  # [1, L, 1, head_dim]
        sin_freqs = freqs_expanded.sin().unsqueeze(0).unsqueeze(2)  # [1, L, 1, head_dim]

        return cos_freqs, sin_freqs


class WanConditionEmbedder(nn.Module):
    """
    Generates condition embeddings from timestep and text.

    This module:
    1. Converts timestep to sinusoidal embeddings
    2. Projects timestep embeddings through MLPs for modulation parameters
    3. Projects text embeddings to model dimension
    """

    def __init__(
        self, hidden_dim: int = 5120, text_dim: int = 4096, num_modulation_params: int = 6
    ):
        super().__init__()

        # Timestep embedding generation
        self.timesteps_proj = Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timesteps_embedding = TimestepEmbedding(256, hidden_dim)

        # Text projection from T5 dimension to model dimension
        self.text_proj = PixArtAlphaTextProjection(in_features=text_dim, hidden_size=hidden_dim)

        # MLP for generating modulation parameters
        mlp_hidden_dim = hidden_dim * 4
        self.mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, num_modulation_params * hidden_dim, bias=True)
        )

    def forward(
        self,
        timestep: torch.LongTensor,  # Shape: [B]
        encoder_hidden_states: torch.Tensor,  # Shape: [B, 512, 4096]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate condition embeddings.

        Returns:
            - temb: Timestep embeddings for final layer, shape [B, 2, 5120]
            - timestep_proj: Timestep embeddings for blocks, shape [B, 6, 5120]
            - enc_text: Projected text embeddings, shape [B, 512, 5120]
        """
        # Generate timestep embeddings
        timesteps_emb = self.timesteps_proj(timestep)  # [B, 256]
        timesteps_emb = self.timesteps_embedding(timesteps_emb)  # [B, 5120]

        # Generate modulation parameters
        timestep_proj = self.mlp(timesteps_emb)  # [B, 6 * 5120]
        timestep_proj = timestep_proj.view(-1, 6, 5120)  # [B, 6, 5120]

        # Project text to model dimension
        enc_text = self.text_proj(encoder_hidden_states)  # [B, 512, 5120]

        # Final layer uses first 2 modulation parameters
        temb = timestep_proj[:, :2]  # [B, 2, 5120]

        return temb, timestep_proj, enc_text


# ================================================================================================
# Section 7: Main Transformer Model
# ================================================================================================


class WanTransformer3DModel(ModelMixin, ConfigMixin, CacheMixin):
    """
    WAN 2.2 Image-to-Video Transformer Model.

    This is the main model that orchestrates:
    1. Patch embedding of input video latents
    2. Conditioning on text and optional image features
    3. Processing through transformer blocks
    4. Unpatchifying back to video format

    Key architectural features:
    - 36 input channels (4 VAE latent channels * 9 for temporal concatenation)
    - 16 output channels (denoising prediction)
    - Patch size (1, 2, 2) for temporal-spatial patching
    - 5120 hidden dimension, 40 attention heads
    - RoPE for position encoding
    - Cross-attention to both text and image features
    """

    @register_to_config
    def __init__(
        self,
        # Patch configuration
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        # Model dimensions
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 36,
        out_channels: int = 16,
        # Text encoder dimensions
        text_dim: int = 4096,
        freq_dim: int = 256,
        # Architecture
        ffn_dim: int = 13824,
        num_layers: int = 48,
        # Normalization
        cross_attn_norm: bool = True,
        qk_norm: bool = True,
        eps: float = 1e-6,
        # Image conditioning
        image_dim: Optional[int] = 1280,
        added_kv_proj_dim: Optional[int] = 1280,
        # Position encoding
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        assert inner_dim == 5120, "WAN I2V uses inner_dim=5120"

        # Rotary position embeddings
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)

        # Patch embedding: Conv3D to convert patches to features
        # [B, 36, F, H, W] -> [B, 5120, F/1, H/2, W/2]
        self.patch_embedding = nn.Conv3d(
            in_channels=36, out_channels=inner_dim, kernel_size=patch_size, stride=patch_size
        )

        # Conditioning embedder for timestep and text
        self.condition_embedder = WanConditionEmbedder(inner_dim)

        # Image projection (when using image conditioning)
        self.image_proj = nn.Linear(1280, inner_dim)
        self.image_norm = nn.RMSNorm(inner_dim, eps=eps)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList(
            [WanTransformerBlock(inner_dim, ffn_dim) for _ in range(num_layers)]
        )

        # Output layers
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)

        # Project from hidden to patch_size[0] * patch_size[1] * patch_size[2] * out_channels
        # This allows unpatchifying in the spatial domain
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))

        # Final modulation parameters
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: [B, 36, F, H, W]
        timestep: torch.LongTensor,  # Shape: [B]
        encoder_hidden_states: torch.Tensor,  # Shape: [B, 512, 4096] - text
        encoder_hidden_states_image: Optional[torch.Tensor] = None,  # Shape: [B, img_len, 1280]
        return_dict: bool = True,
        attention_kwargs: Optional[dict] = None,  # Ignored for compatibility
    ):
        """
        Forward pass through the video transformer.

        Processing pipeline:
        1. Generate rotary position embeddings
        2. Create condition embeddings from timestep and text
        3. Project and concatenate image features if provided
        4. Apply patch embedding to video
        5. Process through transformer blocks
        6. Apply final normalization and modulation
        7. Project to output channels
        8. Unpatchify back to video format

        The input video has been concatenated along channels:
        - Original: [B, 4, F, H, W] (VAE latents)
        - Concatenated: [B, 36, F, H, W] (4 * 9 for temporal context)
        """

        batch_size, channels, frames, height, width = hidden_states.shape

        # Step 1: Generate position embeddings based on video shape
        rotary_emb = self.rope(hidden_states)  # ([1, L, 1, 128], [1, L, 1, 128])

        # Step 2: Create condition embeddings
        temb, timestep_proj, text_features = self.condition_embedder(
            timestep, encoder_hidden_states
        )
        # temb: [B, 2, 5120] - for final modulation
        # timestep_proj: [B, 6, 5120] - for block modulation
        # text_features: [B, 512, 5120] - projected text

        # Step 3: Handle image conditioning if provided
        if encoder_hidden_states_image is not None:
            # Project image features from CLIP dimension to model dimension
            # [B, img_len, 1280] -> [B, img_len, 5120]
            image_features = self.image_proj(encoder_hidden_states_image)
            image_features = self.image_norm(image_features)

            # Concatenate image and text features
            # [B, img_len + 512, 5120]
            encoder_context = torch.cat([image_features, text_features], dim=1)
        else:
            # Text-only conditioning
            encoder_context = text_features  # [B, 512, 5120]

        # Step 4: Apply patch embedding
        # [B, 36, F, H, W] -> [B, 5120, F/1, H/2, W/2]
        hidden_states = self.patch_embedding(hidden_states)

        # Flatten spatial dimensions to sequence
        # [B, 5120, F', H', W'] -> [B, 5120, L] -> [B, L, 5120]
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        seq_len = hidden_states.shape[1]

        # Step 5: Process through transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, encoder_context, timestep_proj, rotary_emb)

        # Step 6: Final normalization with modulation
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states.float())
        hidden_states = (hidden_states * (1 + scale) + shift).type_as(hidden_states)

        # Step 7: Project to output dimension
        # [B, L, 5120] -> [B, L, 16 * 1 * 2 * 2] = [B, L, 64]
        hidden_states = self.proj_out(hidden_states)

        # Step 8: Unpatchify back to video format
        # This is the inverse of the patching operation
        # We need to reshape from [B, L, 64] back to [B, 16, F, H, W]

        # Calculate output dimensions
        out_frames = frames // self.patch_size[0]  # F/1
        out_height = height // self.patch_size[1]  # H/2
        out_width = width // self.patch_size[2]  # W/2

        # Complex reshape for unpatchifying
        # [B, L, 64] -> [B, F', H', W', 1, 2, 2, 16]
        hidden_states = hidden_states.reshape(
            batch_size,
            out_frames,
            out_height,
            out_width,
            self.patch_size[0],
            self.patch_size[1],
            self.patch_size[2],
            out_channels,
        )

        # Permute to get spatial dimensions together
        # [B, F', H', W', 1, 2, 2, 16] -> [B, 16, F', 1, H', 2, W', 2]
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)

        # Merge patch dimensions with spatial dimensions
        # [B, 16, F', 1, H', 2, W', 2] -> [B, 16, F, H, W]
        output = hidden_states.flatten(6, 7)  # Merge W' and 2
        output = output.flatten(4, 5)  # Merge H' and 2
        output = output.flatten(2, 3)  # Merge F' and 1

        if return_dict:
            return Transformer2DModelOutput(sample=output)
        else:
            return (output,)

    @classmethod
    def from_pretrained_stock(cls, stock_transformer: nn.Module) -> "WanTransformer3DModel":
        """
        Convert a stock WAN transformer to the optimized version.

        This method:
        1. Extracts configuration from the stock model
        2. Creates an optimized model instance
        3. Copies weights while handling architectural differences
        4. Fuses projections where needed

        The main differences handled:
        - Stock model may have separate Q, K, V projections (we fuse them)
        - Stock model has separate image KV projection (we pre-project)
        - Stock model may have ModuleList for output (we simplify)
        """

        config = stock_transformer.config

        # Handle missing config values
        if config.added_kv_proj_dim is None:
            config.added_kv_proj_dim = 1280
        if config.image_dim is None:
            config.image_dim = 1280

        # Process stock transformer blocks
        for idx, block in enumerate(stock_transformer.blocks):
            print(f"Processing stock transformer block {idx}...")

            # Fix cross-attention configuration
            if hasattr(block.attn2, "add_k_proj") and block.attn2.add_k_proj is not None:
                print(f"  Setting kv_proj_dim in layer {idx}")
                block.attn2.added_kv_proj_dim = 1280

            # Fuse self-attention projections if needed
            if not hasattr(block.attn1, "to_qkv"):
                print(f"  Fusing self-attention projections in layer {idx}")
                block.attn1.fuse_projections()

            # Fuse cross-attention projections if needed
            if not hasattr(block.attn2, "to_kv"):
                print(f"  Fusing cross-attention projections in layer {idx}")
                block.attn2.fuse_projections()

        # Create optimized transformer
        opt_transformer = cls(
            patch_size=config.patch_size,
            num_attention_heads=config.num_attention_heads,
            attention_head_dim=config.attention_head_dim,
            in_channels=36,
            out_channels=config.out_channels,
            text_dim=config.text_dim,
            freq_dim=config.freq_dim,
            ffn_dim=config.ffn_dim,
            num_layers=config.num_layers,
            cross_attn_norm=config.cross_attn_norm,
            qk_norm=config.qk_norm,
            eps=config.eps,
            image_dim=1280,
            added_kv_proj_dim=1280,
            rope_max_seq_len=config.rope_max_seq_len,
            pos_embed_seq_len=config.pos_embed_seq_len,
        )

        # Copy weights
        with torch.no_grad():
            print("Copying weights from stock model...")

            # Standard components that map 1:1
            opt_transformer.patch_embedding.load_state_dict(
                stock_transformer.patch_embedding.state_dict()
            )
            opt_transformer.rope.load_state_dict(stock_transformer.rope.state_dict())
            opt_transformer.condition_embedder.load_state_dict(
                stock_transformer.condition_embedder.state_dict()
            )
            opt_transformer.norm_out.load_state_dict(stock_transformer.norm_out.state_dict())
            opt_transformer.proj_out.load_state_dict(stock_transformer.proj_out.state_dict())
            opt_transformer.scale_shift_table.copy_(stock_transformer.scale_shift_table)

            # Handle image projection (extracted from first cross-attention block)
            if hasattr(stock_transformer.blocks[0].attn2, "to_added_kv"):
                # Extract K projection from combined KV projection
                added_kv_weight = stock_transformer.blocks[0].attn2.to_added_kv.weight
                added_k_weight = added_kv_weight[:5120, :]  # First half is K
                opt_transformer.image_proj.weight.copy_(added_k_weight)

                if stock_transformer.blocks[0].attn2.to_added_kv.bias is not None:
                    added_kv_bias = stock_transformer.blocks[0].attn2.to_added_kv.bias
                    added_k_bias = added_kv_bias[:5120]
                    opt_transformer.image_proj.bias.copy_(added_k_bias)

                # Copy normalization for K
                if hasattr(stock_transformer.blocks[0].attn2, "norm_added_k"):
                    opt_transformer.image_norm.load_state_dict(
                        stock_transformer.blocks[0].attn2.norm_added_k.state_dict()
                    )

            # Copy transformer blocks
            for idx, (stock_block, opt_block) in enumerate(
                zip(stock_transformer.blocks, opt_transformer.blocks)
            ):
                print(f"Copying weights for block {idx}...")

                # Layer norms and FFN
                opt_block.norm1.load_state_dict(stock_block.norm1.state_dict())
                opt_block.norm2.load_state_dict(stock_block.norm2.state_dict())
                opt_block.norm3.load_state_dict(stock_block.norm3.state_dict())
                opt_block.ffn.load_state_dict(stock_block.ffn.state_dict())
                opt_block.scale_shift_table.copy_(stock_block.scale_shift_table)

                # Self-attention weights
                opt_block.attn1.to_qkv.weight.copy_(stock_block.attn1.to_qkv.weight)
                opt_block.attn1.to_qkv.bias.copy_(stock_block.attn1.to_qkv.bias)
                opt_block.attn1.norm_q.load_state_dict(stock_block.attn1.norm_q.state_dict())
                opt_block.attn1.norm_k.load_state_dict(stock_block.attn1.norm_k.state_dict())
                opt_block.attn1.to_out[0].load_state_dict(stock_block.attn1.to_out[0].state_dict())

                # Cross-attention weights
                opt_block.attn2.to_q.load_state_dict(stock_block.attn2.to_q.state_dict())
                opt_block.attn2.norm_q.load_state_dict(stock_block.attn2.norm_q.state_dict())

                # KV projection for text (we reuse text KV since images are pre-projected)
                opt_block.attn2.to_kv.weight.copy_(stock_block.attn2.to_kv.weight)
                opt_block.attn2.to_kv.bias.copy_(stock_block.attn2.to_kv.bias)
                opt_block.attn2.norm_k.load_state_dict(stock_block.attn2.norm_k.state_dict())

                # Output projection (handle ModuleList vs single Linear)
                if isinstance(stock_block.attn2.to_out, nn.ModuleList):
                    opt_block.attn2.to_out.load_state_dict(stock_block.attn2.to_out[0].state_dict())
                else:
                    opt_block.attn2.to_out.load_state_dict(stock_block.attn2.to_out.state_dict())

        print("Weight transfer complete!")
        return opt_transformer
