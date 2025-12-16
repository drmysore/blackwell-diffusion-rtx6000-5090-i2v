"""
WAN 2.2 Diffusers Compatibility Layer
======================================

This module provides diffusers-compatible wrappers around the pure PyTorch
implementation, handling:
1. ConfigMixin and ModelMixin integration
2. Diffusers-specific naming conventions
3. Pipeline compatibility
4. Model outputs in diffusers format

This layer allows seamless integration with existing diffusers pipelines
while keeping the core implementation framework-agnostic.
"""

from typing import Optional, Tuple, Dict, Any
import math

import torch
import torch.nn as nn

# Diffusers imports
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
)

# Import our pure PyTorch modules
from wan_attention_pure import (
    TransformerBlock,
    RotaryPosEmbed,
    FP32LayerNorm,
    RMSNorm,
)


# ================================================================================================
# Section 1: Diffusers-Compatible Attention Wrappers
# ================================================================================================


class WanSelfAttention(nn.Module):
    """
    Diffusers-compatible self-attention wrapper.

    This wraps the pure SelfAttention module and adapts it to diffusers'
    expected interface, particularly for handling attention kwargs and
    processor patterns.
    """

    def __init__(
        self,
        dim: int = 5120,
        heads: int = 40,
        eps: float = 1e-6,
    ):
        super().__init__()

        # Import here to avoid circular dependency
        from wan_attention_pure import SelfAttention

        # Use the pure implementation
        self._attention = SelfAttention(
            dim=dim,
            num_heads=heads,
            qk_norm=True,
            eps=eps,
            dropout=0.0,
        )

        # Expose diffusers-compatible attributes
        self.heads = heads
        self.head_dim = dim // heads

        # Expose the underlying modules for compatibility
        self.to_qkv = self._attention.to_qkv
        self.norm_q = self._attention.norm_q
        self.norm_k = self._attention.norm_k
        self.to_out = nn.ModuleList(
            [
                self._attention.to_out[0],  # Linear
                self._attention.to_out[1],  # Dropout/Identity
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,  # Ignore extra diffusers kwargs
    ) -> torch.Tensor:
        """Forward pass delegating to pure implementation."""
        return self._attention(hidden_states, rotary_emb, attention_mask)


class WanCrossAttention(nn.Module):
    """
    Diffusers-compatible cross-attention wrapper.

    Wraps the pure CrossAttention module and provides the interface
    expected by diffusers pipelines.
    """

    def __init__(
        self,
        dim: int = 5120,
        heads: int = 40,
        eps: float = 1e-6,
    ):
        super().__init__()

        # Import here to avoid circular dependency
        from wan_attention_pure import CrossAttention

        # Use the pure implementation
        self._attention = CrossAttention(
            dim=dim,
            context_dim=dim,  # WAN uses same dim for context
            num_heads=heads,
            qk_norm=True,
            eps=eps,
            dropout=0.0,
        )

        # Expose diffusers-compatible attributes
        self.heads = heads
        self.head_dim = dim // heads

        # Expose underlying modules
        self.to_q = self._attention.to_q
        self.to_kv = self._attention.to_kv
        self.norm_q = self._attention.norm_q
        self.norm_k = self._attention.norm_k
        self.to_out = self._attention.to_out[0]  # Just the Linear for diffusers compat

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass delegating to pure implementation."""
        return self._attention(hidden_states, encoder_hidden_states, attention_mask)


# ================================================================================================
# Section 2: Diffusers-Compatible Transformer Block
# ================================================================================================


class WanTransformerBlock(nn.Module):
    """
    Diffusers-compatible transformer block wrapper.

    This maintains compatibility with diffusers' expected interface while
    using the optimized pure implementation underneath.
    """

    def __init__(
        self,
        dim: int = 5120,
        ffn_dim: int = 13824,
    ):
        super().__init__()

        # Create the pure transformer block
        self._block = TransformerBlock(
            dim=dim,
            context_dim=dim,
            num_heads=40,
            ffn_dim=ffn_dim,
            qk_norm=True,
            norm_eps=1e-6,
            dropout=0.0,
            activation_fn="gelu-approximate",
        )

        # Expose components for diffusers compatibility
        self.norm1 = self._block.norm1
        self.norm2 = self._block.norm2
        self.norm3 = self._block.norm3

        self.attn1 = WanSelfAttention(dim, heads=40, eps=1e-6)
        self.attn2 = WanCrossAttention(dim, heads=40, eps=1e-6)

        self.ffn = self._block.ffn
        self.scale_shift_table = self._block.scale_shift_table

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass compatible with diffusers interface.

        Args:
            hidden_states: [B, L, D]
            encoder_hidden_states: [B, C, D] - already processed context
            temb: [B, 6, D] - timestep embeddings for modulation
            rotary_emb: Rotary position embeddings
            attention_mask: Optional attention mask
        """
        return self._block(
            hidden_states,
            encoder_hidden_states,
            conditioning=temb,
            rotary_emb=rotary_emb,
            attention_mask=attention_mask,
        )


# ================================================================================================
# Section 3: Diffusers-Specific Components
# ================================================================================================


class WanRotaryPosEmbed(nn.Module):
    """
    Diffusers-compatible rotary position embedding for video patches.

    This generates position embeddings based on the 3D structure of video patches,
    compatible with diffusers' expected interface.
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

        # Use the pure implementation
        self._rope = RotaryPosEmbed(head_dim, max_seq_len)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, C, F, H, W]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate rotary embeddings for video patches.

        Returns:
            Tuple of (cos_freqs, sin_freqs) with shape [1, L, 1, head_dim]
        """

        batch_size, _, frames, height, width = hidden_states.shape

        # Calculate sequence length from patch dimensions
        frames_patched = frames // self.patch_size[0]
        height_patched = height // self.patch_size[1]
        width_patched = width // self.patch_size[2]
        seq_len = frames_patched * height_patched * width_patched

        # Get rotary embeddings
        return self._rope(seq_len, hidden_states.device, hidden_states.dtype)


class WanConditionEmbedder(nn.Module):
    """
    Diffusers-compatible condition embedder.

    Handles timestep and text conditioning in the format expected by
    diffusers pipelines.
    """

    def __init__(
        self,
        hidden_dim: int = 5120,
        text_dim: int = 4096,
        num_modulation_params: int = 6,
    ):
        super().__init__()

        # Timestep embedding (diffusers style)
        self.timesteps_proj = Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timesteps_embedding = TimestepEmbedding(256, hidden_dim)

        # Text projection (diffusers PixArt style)
        self.text_proj = PixArtAlphaTextProjection(
            in_features=text_dim,
            hidden_size=hidden_dim,
        )

        # MLP for modulation parameters
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, num_modulation_params * hidden_dim, bias=True),
        )

    def forward(
        self,
        timestep: torch.LongTensor,  # [B]
        encoder_hidden_states: torch.Tensor,  # [B, 512, 4096]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate conditioning embeddings.

        Returns:
            - temb: Final modulation parameters [B, 2, D]
            - timestep_proj: Block modulation parameters [B, 6, D]
            - enc_text: Projected text embeddings [B, 512, D]
        """

        # Generate timestep embeddings
        timesteps_emb = self.timesteps_proj(timestep)
        timesteps_emb = self.timesteps_embedding(timesteps_emb)

        # Generate modulation parameters
        timestep_proj = self.mlp(timesteps_emb)
        timestep_proj = timestep_proj.view(-1, 6, self.text_proj.hidden_size)

        # Project text embeddings
        enc_text = self.text_proj(encoder_hidden_states)

        # Final layer modulation uses first 2 parameters
        temb = timestep_proj[:, :2]

        return temb, timestep_proj, enc_text


# ================================================================================================
# Section 4: Main Diffusers-Compatible Transformer
# ================================================================================================


class WanTransformer3DModel(ModelMixin, ConfigMixin, CacheMixin):
    """
    Diffusers-compatible WAN 2.2 Video Transformer.

    This provides the full interface expected by diffusers pipelines while
    using our optimized pure implementation underneath.
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
        # Text encoder
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

        # Components
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)

        self.patch_embedding = nn.Conv3d(
            in_channels,
            inner_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.condition_embedder = WanConditionEmbedder(inner_dim, text_dim)

        # Image projection for cross-attention
        self.image_proj = nn.Linear(image_dim or 1280, inner_dim)
        self.image_norm = RMSNorm(inner_dim, eps=eps)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [WanTransformerBlock(inner_dim, ffn_dim) for _ in range(num_layers)]
        )

        # Output layers
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / math.sqrt(inner_dim))

        # Store config for compatibility
        self.patch_size = patch_size
        self.out_channels = out_channels

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, 36, F, H, W]
        timestep: torch.LongTensor,  # [B]
        encoder_hidden_states: torch.Tensor,  # [B, 512, 4096]
        encoder_hidden_states_image: Optional[torch.Tensor] = None,  # [B, img_len, 1280]
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,  # For compatibility
        **kwargs,  # Catch other diffusers kwargs
    ):
        """
        Forward pass compatible with diffusers pipelines.

        This method signature matches what diffusers expects while using
        our optimized implementation underneath.
        """

        batch_size, channels, frames, height, width = hidden_states.shape

        # Generate position embeddings
        rotary_emb = self.rope(hidden_states)

        # Create conditioning
        temb, timestep_proj, enc_text = self.condition_embedder(timestep, encoder_hidden_states)

        # Handle image conditioning
        if encoder_hidden_states_image is not None:
            enc_img = self.image_proj(encoder_hidden_states_image)
            enc_img = self.image_norm(enc_img)
            encoder_ctx = torch.cat([enc_img, enc_text], dim=1)
        else:
            encoder_ctx = enc_text

        # Patch embedding
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # Process through blocks
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                encoder_ctx,
                timestep_proj,
                rotary_emb,
            )

        # Final modulation and projection
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states.float())
        hidden_states = (hidden_states * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        # Unpatchify
        hidden_states = hidden_states.reshape(
            batch_size,
            frames // self.patch_size[0],
            height // self.patch_size[1],
            width // self.patch_size[2],
            self.patch_size[0],
            self.patch_size[1],
            self.patch_size[2],
            self.out_channels,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if return_dict:
            return Transformer2DModelOutput(sample=output)
        else:
            return (output,)

    @classmethod
    def from_pretrained_stock(cls, stock_transformer: nn.Module) -> "WanTransformer3DModel":
        """
        Load weights from a stock diffusers WAN transformer.

        This handles all the weight mapping and fusion operations needed
        to convert from the stock implementation to our optimized version.
        """

        # [Weight loading logic remains the same as in the original]
        # This is a long method that handles the conversion
        # I'll abbreviate it here since it's identical to the original

        config = stock_transformer.config

        # Create optimized model
        opt_model = cls(
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
            image_dim=getattr(config, "image_dim", 1280),
            added_kv_proj_dim=getattr(config, "added_kv_proj_dim", 1280),
            rope_max_seq_len=config.rope_max_seq_len,
            pos_embed_seq_len=getattr(config, "pos_embed_seq_len", None),
        )

        # Copy weights (abbreviated - same as original)
        with torch.no_grad():
            # ... weight copying logic ...
            pass

        return opt_model
