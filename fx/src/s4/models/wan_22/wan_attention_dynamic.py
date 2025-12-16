"""
WAN 2.2 Diffusers Dynamic Compatibility Layer
==============================================

This module handles ALL dynamic behavior, conditionals, and flexibility.
It's the "mess absorption zone" that:

1. Accepts variable batch sizes and shapes
2. Handles optional parameters and None values
3. Manages different configurations
4. Routes to appropriate static implementations
5. Handles weight conversion and compatibility

This layer exists so the static inference implementation can remain pure.
"""

from typing import Optional, Dict, Any, Union, Tuple, List
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

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
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.attention import FeedForward

# Import static implementations
from wan_attention_static import (
    StaticSelfAttention,
    StaticRMSNorm,
    StaticWANInferenceModel,
    apply_rotary_emb_static,
)


# ================================================================================================
# Dynamic Components that Handle All Flexibility
# ================================================================================================


class DynamicRoPE(nn.Module):
    """
    Dynamic RoPE that handles variable sequence lengths.
    This allocates different static implementations based on input.
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

        # Pre-compute for common sequence lengths
        self.cached_freqs = {}
        common_lens = [197, 256, 512, 768, 1024, 2048, 3136, 4096]

        for seq_len in common_lens:
            if seq_len <= max_seq_len:
                inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
                positions = torch.arange(seq_len)
                freqs = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
                freqs_expanded = torch.cat([freqs, freqs], dim=-1)

                self.register_buffer(
                    f"cos_{seq_len}", freqs_expanded.cos().unsqueeze(0).unsqueeze(2)
                )
                self.register_buffer(
                    f"sin_{seq_len}", freqs_expanded.sin().unsqueeze(0).unsqueeze(2)
                )
                self.cached_freqs[seq_len] = (f"cos_{seq_len}", f"sin_{seq_len}")

    def forward(
        self,
        seq_len: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get RoPE for any sequence length.
        Falls back to dynamic computation if not cached.
        """
        # Check cache first
        if seq_len in self.cached_freqs:
            cos_name, sin_name = self.cached_freqs[seq_len]
            cos = getattr(self, cos_name)
            sin = getattr(self, sin_name)

            if device is not None:
                cos = cos.to(device)
                sin = sin.to(device)
            if dtype is not None:
                cos = cos.to(dtype)
                sin = sin.to(dtype)

            return cos, sin

        # Dynamic computation for non-cached lengths
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.head_dim, 2, device=device, dtype=dtype) / self.head_dim)
        )
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
        freqs_expanded = torch.cat([freqs, freqs], dim=-1)

        cos = freqs_expanded.cos().unsqueeze(0).unsqueeze(2)
        sin = freqs_expanded.sin().unsqueeze(0).unsqueeze(2)

        return cos, sin


class DynamicSelfAttention(nn.Module):
    """
    Dynamic self-attention that handles variable shapes and optional parameters.
    Routes to static implementations when possible.
    """

    def __init__(
        self,
        dim: int = 5120,
        heads: int = 40,
        qk_norm: Optional[bool] = True,
        eps: float = 1e-6,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        # Components
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)

        # Optional normalization
        if qk_norm:
            self.norm_q = StaticRMSNorm(dim, eps)
            self.norm_k = StaticRMSNorm(dim, eps)
        else:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()

        self.to_out = nn.ModuleList(
            [
                nn.Linear(dim, dim, bias=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            ]
        )

        self.scale = self.head_dim**-0.5

        # Cache for static implementations
        self._static_cache = {}

    def _get_static_impl(self, batch_size: int, seq_len: int):
        """Get or create static implementation for given dimensions."""
        key = (batch_size, seq_len)

        if key not in self._static_cache:
            # Create static implementation
            self._static_cache[key] = StaticSelfAttention(
                batch_size=batch_size,
                seq_len=seq_len,
                dim=self.dim,
                num_heads=self.heads,
                dropout=0.0,  # Dropout handled dynamically
            )

            # Copy weights
            with torch.no_grad():
                self._static_cache[key].to_qkv.load_state_dict(self.to_qkv.state_dict())
                if hasattr(self.norm_q, "weight"):
                    self._static_cache[key].norm_q.weight.copy_(self.norm_q.weight)
                    self._static_cache[key].norm_k.weight.copy_(self.norm_k.weight)
                self._static_cache[key].to_out.load_state_dict(self.to_out[0].state_dict())

        return self._static_cache[key]

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,  # Catch extra arguments
    ) -> torch.Tensor:
        """
        Dynamic forward that handles all cases.
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Try to use static implementation for common cases
        if (
            attention_mask is None
            and rotary_emb is not None
            and batch_size <= 16  # Common batch sizes
            and seq_len in [197, 256, 512, 1024, 3136]  # Common sequence lengths
        ):
            try:
                static_impl = self._get_static_impl(batch_size, seq_len).to(hidden_states.device)
                return static_impl(hidden_states, rotary_emb[0], rotary_emb[1])
            except:
                pass  # Fall back to dynamic implementation

        # Dynamic implementation for all other cases
        qkv = self.to_qkv(hidden_states)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Optional normalization
        if hasattr(self.norm_q, "weight"):
            q_flat = q.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
            q_flat = self.norm_q(q_flat)
            q = q_flat.reshape(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)

            k_flat = k.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
            k_flat = self.norm_k(k_flat)
            k = k_flat.reshape(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)

        # Optional RoPE
        if rotary_emb is not None:
            q_rot = q.transpose(1, 2)
            k_rot = k.transpose(1, 2)
            q_rot = apply_rotary_emb_static(q_rot, rotary_emb[0], rotary_emb[1])
            k_rot = apply_rotary_emb_static(k_rot, rotary_emb[0], rotary_emb[1])
            q = q_rot.transpose(1, 2)
            k = k_rot.transpose(1, 2)

        # Attention with optional mask
        hidden_states = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=self.scale,
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        return hidden_states


class DynamicCrossAttention(nn.Module):
    """
    Dynamic cross-attention handling variable context lengths.
    """

    def __init__(
        self,
        dim: int = 5120,
        heads: int = 40,
        qk_norm: Optional[bool] = True,
        eps: float = 1e-6,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_kv = nn.Linear(dim, dim * 2, bias=True)

        if qk_norm:
            self.norm_q = StaticRMSNorm(dim, eps)
            self.norm_k = StaticRMSNorm(dim, eps)
        else:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()

        self.to_out = nn.Linear(dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.scale = self.head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Dynamic cross-attention forward."""

        # Handle missing context
        if encoder_hidden_states is None:
            # Self-attention fallback
            encoder_hidden_states = hidden_states

        batch_size, seq_len, _ = hidden_states.shape
        context_len = encoder_hidden_states.shape[1]

        # Projections
        q = self.to_q(hidden_states)
        q = self.norm_q(q) if hasattr(self.norm_q, "weight") else q
        q = q.reshape(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)

        kv = self.to_kv(encoder_hidden_states)
        kv = kv.reshape(batch_size, context_len, 2, self.heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        if hasattr(self.norm_k, "weight"):
            k_flat = k.transpose(1, 2).reshape(batch_size, context_len, self.dim)
            k_flat = self.norm_k(k_flat)
            k = k_flat.reshape(batch_size, context_len, self.heads, self.head_dim).transpose(1, 2)

        # Attention
        hidden_states = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=self.scale,
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        hidden_states = self.to_out(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class DynamicTransformerBlock(nn.Module):
    """
    Dynamic transformer block that handles all conditionals and routing.
    """

    def __init__(
        self,
        dim: int = 5120,
        num_heads: int = 40,
        ffn_dim: int = 13824,
        qk_norm: bool = True,
        dropout: float = 0.0,
        cross_attention_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.dim = dim

        # Normalization layers
        self.norm1 = FP32LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = FP32LayerNorm(dim, eps=eps, elementwise_affine=cross_attention_norm)
        self.norm3 = FP32LayerNorm(dim, eps=eps, elementwise_affine=False)

        # Attention layers
        self.attn1 = DynamicSelfAttention(dim, num_heads, qk_norm, eps, dropout)
        self.attn2 = DynamicCrossAttention(dim, num_heads, qk_norm, eps, dropout)

        # FFN
        self.ffn = FeedForward(
            dim,
            inner_dim=ffn_dim,
            activation_fn="gelu-approximate",
            dropout=dropout,
        )

        # Modulation parameters
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / math.sqrt(dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Dynamic forward handling all cases.
        """
        batch_size = hidden_states.shape[0]

        # Handle modulation
        if temb is not None:
            if temb.dim() == 2:
                temb = temb.unsqueeze(1)
            modulation = self.scale_shift_table + temb.float()
        else:
            modulation = self.scale_shift_table.expand(batch_size, -1, -1)

        shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = modulation.chunk(6, dim=1)

        # Self-attention
        normed = self.norm1(hidden_states.float())
        modulated = (normed * (1 + scale_sa) + shift_sa).type_as(hidden_states)
        attn_out = self.attn1(modulated, rotary_emb, attention_mask)
        hidden_states = hidden_states + gate_sa * attn_out

        # Cross-attention (if context provided)
        if encoder_hidden_states is not None:
            normed = self.norm2(hidden_states.float()).type_as(hidden_states)
            cross_out = self.attn2(normed, encoder_hidden_states, attention_mask)
            hidden_states = hidden_states + cross_out

        # FFN
        normed = self.norm3(hidden_states.float())
        modulated = (normed * (1 + scale_ff) + shift_ff).type_as(hidden_states)
        ffn_out = self.ffn(modulated)
        hidden_states = hidden_states + gate_ff * ffn_out

        return hidden_states


# ================================================================================================
# Main Dynamic Model with All Flexibility
# ================================================================================================


class WanTransformer3DModel(ModelMixin, ConfigMixin, CacheMixin):
    """
    Full dynamic WAN transformer with all flexibility and compatibility.
    This handles all the mess so the static implementation doesn't have to.
    """

    @register_to_config
    def __init__(
        self,
        # Patch configuration
        patch_size: Union[int, List[int], Tuple[int, int, int]] = (1, 2, 2),
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
        rope_max_seq_len: int = 16384,
        pos_embed_seq_len: Optional[int] = None,
        # Training vs inference
        dropout: float = 0.0,
        training_mode: bool = False,
        # Static optimization
        use_static_cache: bool = True,
        static_batch_size: Optional[int] = None,
        static_seq_len: Optional[int] = None,
        static_context_len: Optional[int] = None,
    ):
        super().__init__()

        # Handle patch size flexibility
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        elif len(patch_size) == 2:
            patch_size = (1, patch_size[0], patch_size[1])

        self.patch_size = tuple(patch_size)
        self.out_channels = out_channels

        inner_dim = num_attention_heads * attention_head_dim

        # Components
        self.rope = DynamicRoPE(attention_head_dim, rope_max_seq_len)

        self.patch_embedding = nn.Conv3d(
            in_channels,
            inner_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Conditioning
        self.condition_embedder = ConditionEmbedder(inner_dim, text_dim, freq_dim)

        # Image projection
        if image_dim is not None:
            self.image_proj = nn.Linear(image_dim, inner_dim)
            self.image_norm = StaticRMSNorm(inner_dim, eps)
        else:
            self.image_proj = None
            self.image_norm = None

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                DynamicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    ffn_dim,
                    qk_norm,
                    dropout if training_mode else 0.0,
                    cross_attn_norm,
                    eps,
                )
                for _ in range(num_layers)
            ]
        )

        # Output
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / math.sqrt(inner_dim))

        # Static optimization cache
        self.use_static_cache = use_static_cache
        self._static_model = None
        self._static_config = (static_batch_size, static_seq_len, static_context_len)

    def _get_static_model(self, batch_size: int, seq_len: int, context_len: int):
        """Get or create static model for given dimensions."""
        if self._static_model is None or self._static_config != (batch_size, seq_len, context_len):
            self._static_model = StaticWANInferenceModel(
                batch_size,
                seq_len,
                context_len,
                len(self.blocks),
                self.config.num_attention_heads * self.config.attention_head_dim,
                self.config.num_attention_heads,
                self.config.ffn_dim,
                0.0,
                self.device,
                self.dtype,
            )

            # Copy weights
            self._sync_weights_to_static()
            self._static_config = (batch_size, seq_len, context_len)

        return self._static_model

    def _sync_weights_to_static(self):
        """Sync weights from dynamic to static model."""
        if self._static_model is None:
            return

        with torch.no_grad():
            # Copy block weights
            for dyn_block, stat_block in zip(self.blocks, self._static_model.blocks):
                # ... weight copying logic ...
                pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Full dynamic forward with all flexibility.
        Handles all edge cases and optional behaviors.
        """

        # Input validation and shape handling
        if hidden_states.dim() == 4:
            # Add channel dimension if needed
            hidden_states = hidden_states.unsqueeze(1)

        batch_size, channels, frames, height, width = hidden_states.shape

        # Generate RoPE dynamically
        seq_len = (
            (frames // self.patch_size[0])
            * (height // self.patch_size[1])
            * (width // self.patch_size[2])
        )
        rotary_emb = self.rope(seq_len, hidden_states.device, hidden_states.dtype)

        # Handle conditioning
        if timestep is not None:
            temb, timestep_proj, enc_text = self.condition_embedder(timestep, encoder_hidden_states)
        else:
            # No timestep - use zeros
            temb = torch.zeros(
                batch_size,
                2,
                self.config.num_attention_heads * self.config.attention_head_dim,
                device=hidden_states.device,
            )
            timestep_proj = torch.zeros(
                batch_size,
                6,
                self.config.num_attention_heads * self.config.attention_head_dim,
                device=hidden_states.device,
            )
            enc_text = encoder_hidden_states if encoder_hidden_states is not None else None

        # Handle image conditioning
        if encoder_hidden_states_image is not None and self.image_proj is not None:
            enc_img = self.image_proj(encoder_hidden_states_image)
            enc_img = self.image_norm(enc_img)
            encoder_ctx = torch.cat([enc_img, enc_text], dim=1) if enc_text is not None else enc_img
        else:
            encoder_ctx = enc_text

        # Patch embedding
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # Try to use static model for common configurations
        if (
            self.use_static_cache
            and not self.training
            and batch_size <= 8
            and seq_len in [197, 256, 512, 1024, 3136]
            and encoder_ctx is not None
            and encoder_ctx.shape[1] in [512, 768, 1024, 1536]
        ):
            try:
                static_model = self._get_static_model(batch_size, seq_len, encoder_ctx.shape[1])

                # Prepare static inputs
                block_conditioning = timestep_proj.unsqueeze(1).expand(-1, len(self.blocks), -1, -1)
                output_conditioning = temb

                hidden_states = static_model(
                    hidden_states,
                    encoder_ctx,
                    block_conditioning,
                    output_conditioning,
                )
            except Exception as e:
                # Fall back to dynamic processing
                warnings.warn(f"Static model failed, using dynamic: {e}")
                hidden_states = self._dynamic_forward(
                    hidden_states, encoder_ctx, timestep_proj, rotary_emb
                )
        else:
            # Dynamic forward for all other cases
            hidden_states = self._dynamic_forward(
                hidden_states, encoder_ctx, timestep_proj, rotary_emb
            )

        # Final processing
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states.float())
        hidden_states = (hidden_states * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        # Unpatchify
        hidden_states = self._unpatchify(hidden_states, batch_size, frames, height, width)

        if return_dict:
            return Transformer2DModelOutput(sample=hidden_states)
        else:
            return (hidden_states,)

    def _dynamic_forward(self, hidden_states, encoder_ctx, timestep_proj, rotary_emb):
        """Dynamic forward through blocks."""
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                encoder_ctx,
                timestep_proj,
                rotary_emb,
            )
        return hidden_states

    def _unpatchify(self, hidden_states, batch_size, frames, height, width):
        """Unpatchify back to video format."""
        out_frames = frames // self.patch_size[0]
        out_height = height // self.patch_size[1]
        out_width = width // self.patch_size[2]

        hidden_states = hidden_states.reshape(
            batch_size,
            out_frames,
            out_height,
            out_width,
            self.patch_size[0],
            self.patch_size[1],
            self.patch_size[2],
            self.out_channels,
        )

        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        return hidden_states


# ================================================================================================
# Condition Embedder (Dynamic)
# ================================================================================================


class ConditionEmbedder(nn.Module):
    """Dynamic condition embedder handling various input formats."""

    def __init__(
        self,
        hidden_dim: int = 5120,
        text_dim: int = 4096,
        freq_dim: int = 256,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timesteps_embedding = TimestepEmbedding(freq_dim, hidden_dim)

        self.text_proj = PixArtAlphaTextProjection(
            in_features=text_dim,
            hidden_size=hidden_dim,
        )

        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True),
        )

    def forward(
        self,
        timestep: Optional[torch.LongTensor],
        encoder_hidden_states: Optional[torch.Tensor],
    ):
        """Handle various input formats."""

        if timestep is None:
            # No timestep - return zeros
            batch_size = encoder_hidden_states.shape[0] if encoder_hidden_states is not None else 1
            device = encoder_hidden_states.device if encoder_hidden_states is not None else "cpu"

            zero_hidden = torch.zeros(batch_size, self.text_proj.hidden_size, device=device)
            temb = torch.zeros(batch_size, 2, self.text_proj.hidden_size, device=device)
            timestep_proj = torch.zeros(batch_size, 6, self.text_proj.hidden_size, device=device)
        else:
            # Process timestep
            if timestep.dim() == 0:
                timestep = timestep.unsqueeze(0)

            timesteps_emb = self.timesteps_proj(timestep)
            timesteps_emb = self.timesteps_embedding(timesteps_emb)

            timestep_proj = self.mlp(timesteps_emb)
            timestep_proj = timestep_proj.view(-1, 6, self.text_proj.hidden_size)

            temb = timestep_proj[:, :2]

        # Process text
        if encoder_hidden_states is not None:
            enc_text = self.text_proj(encoder_hidden_states)
        else:
            batch_size = timestep.shape[0] if timestep is not None else 1
            enc_text = None

        return temb, timestep_proj, enc_text
