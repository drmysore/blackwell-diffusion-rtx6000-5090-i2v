import math

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from torch import nn


import transformer_engine.pytorch as te
from transformer_engine.common import recipe
import transformer_engine

format = recipe.Format.HYBRID


class _FlashSDPA:
    def __enter__(self):
        self.ctx = torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_mem_efficient=False, enable_math=False
        )
        self.ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.ctx.__exit__(exc_type, exc, tb)


_FLASH_SDPA = _FlashSDPA()  # reuse object to avoid reallocation


def _pad_to_multiple(
    t: torch.Tensor,
    multiple: int,
    dim: int = 1,
    value: float = 0.0,
) -> torch.Tensor:
    """Pad tensor on the right along `dim` to a multiple of `multiple`."""
    size = t.size(dim)
    pad = (multiple - (size % multiple)) % multiple

    if pad == 0:
        return t, None

    pad_width = [0, 0] * t.ndim
    pad_width[-2 * dim - 1] = pad  # (left, right) pair for this dimension

    return F.pad(t, pad_width, value=value), pad


def apply_rotary_emb(
    hidden_states: torch.Tensor,  # [B, seq_len, heads, head_dim]
    freqs_cos: torch.Tensor,  # [1, seq_len, 1, head_dim]
    freqs_sin: torch.Tensor,  # [1, seq_len, 1, head_dim]
) -> torch.Tensor:
    """Apply rotary position embeddings - no branches"""
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]

    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos

    return out.type_as(hidden_states)


class WanSelfAttention(nn.Module):
    """Pure self-attention - no branches"""

    def __init__(self, dim: int = 5120, heads: int = 40, eps: float = 1e-6):
        super().__init__()
        assert dim == 5120, "WAN I2V uses dim=5120"
        assert heads == 40, "WAN I2V uses 40 heads"

        self.heads = heads
        self.head_dim = dim // heads  # 128

        # fused QKV for I2V
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
        self.norm_q = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.norm_k = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)

        self.to_out = nn.ModuleList(
            [
                nn.Linear(dim, dim, bias=True),
                nn.Dropout(0.0),
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, seq_len, 5120]
        rotary_emb: tuple[
            torch.Tensor, torch.Tensor
        ],  # ([1, seq_len, 1, 128], [1, seq_len, 1, 128])
    ) -> torch.Tensor:
        B, L, D = hidden_states.shape

        # QKV -> normalize -> reshape -> RoPE -> attention -> out
        qkv = self.to_qkv(hidden_states)  # [B, L, 15360]
        q, k, v = qkv.chunk(3, dim=-1)  # 3x [B, L, 5120]

        q = self.norm_q(q).view(B, L, 40, 128)  # [B, L, 40, 128]
        k = self.norm_k(k).view(B, L, 40, 128)  # [B, L, 40, 128]
        v = v.view(B, L, 40, 128)  # [B, L, 40, 128]

        q = apply_rotary_emb(q, *rotary_emb)
        k = apply_rotary_emb(k, *rotary_emb)

        # [B, 40, L, 128] for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        h = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        h = h.transpose(1, 2).reshape(B, L, 5120)  # [B, L, 5120]

        h = self.to_out[0](h)
        h = self.to_out[1](h)
        return h


class WanCrossAttention(nn.Module):
    """Pure cross-attention with composed weights - single efficient path"""

    def __init__(self, dim: int = 5120, heads: int = 40, eps: float = 1e-6):
        super().__init__()
        assert dim == 5120 and heads == 40
        self.heads = heads
        self.head_dim = dim // heads

        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_kv = nn.Linear(dim, dim * 2, bias=True)
        self.norm_q = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.norm_k = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)

        self.to_out = nn.Linear(dim, dim, bias=True)

    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        B, L, D = hidden_states.shape
        C = encoder_hidden_states.shape[1]

        q = self.norm_q(self.to_q(hidden_states))
        q = q.view(B, L, self.heads, self.head_dim).transpose(1, 2)

        kv = self.to_kv(encoder_hidden_states)  # [B, C, 10240]
        k, v = kv.chunk(2, dim=-1)
        k = self.norm_k(k).view(B, C, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, C, self.heads, self.head_dim).transpose(1, 2)

        h = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        h = h.transpose(1, 2).reshape(B, L, D)
        return self.to_out(h)


class WanTransformerBlock(nn.Module):
    """Transformer block - no branches"""

    def __init__(self, dim: int = 5120, ffn_dim: int = 13824):
        super().__init__()

        # Fixed architecture for WAN I2V
        self.norm1 = FP32LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.attn1 = WanSelfAttention(dim, heads=40, eps=1e-6)

        self.norm2 = FP32LayerNorm(
            dim, eps=1e-6, elementwise_affine=True
        )  # cross_attn_norm=True
        self.attn2 = WanCrossAttention(dim, heads=40, eps=1e-6)

        self.norm3 = FP32LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, seq_len, 5120]
        encoder_hidden_states: torch.Tensor,  # [B, img_len + 512, 5120] <- This is already processed!
        temb: torch.Tensor,  # [B, 6, 5120]
        rotary_emb: tuple[
            torch.Tensor, torch.Tensor
        ],  # ([1, seq_len, 1, 128], [1, seq_len, 1, 128])
    ) -> torch.Tensor:
        # Always: modulation -> self-attn -> cross-attn -> ffn
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)  # 6x [B, 1, 5120]

        # Self-attention with modulation
        h = self.norm1(hidden_states.float())
        h = (h * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        h = self.attn1(h, rotary_emb)
        hidden_states = (hidden_states.float() + h * gate_msa).type_as(hidden_states)

        # Cross-attention - FIX: use encoder_hidden_states which is already the processed encoder_ctx!
        h = self.norm2(hidden_states.float()).type_as(hidden_states)
        h = self.attn2(
            h, encoder_hidden_states
        )  # This should receive the processed 5120-dim embeddings

        hidden_states = hidden_states + h

        # Feed-forward with modulation
        h = self.norm3(hidden_states.float())
        h = (h * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
        h = self.ffn(h)
        hidden_states = (hidden_states.float() + h.float() * c_gate_msa).type_as(
            hidden_states
        )

        return hidden_states


class WanConditionEmbedder(nn.Module):
    """Handles all condition embedding - no branches in forward"""

    def __init__(self, dim: int = 5120):
        super().__init__()

        self.timesteps_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, dim * 6)
        self.text_embedder = PixArtAlphaTextProjection(4096, dim, act_fn="gelu_tanh")

    def forward(
        self,
        timestep: torch.Tensor,  # [B]
        encoder_hidden_states_text: torch.Tensor,  # [B, 512, 4096]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # timestep -> temb -> proj, text -> embed
        t_freq = self.timesteps_proj(timestep)  # [B, 256]
        temb = self.time_embedder(t_freq)  # [B, 5120]
        timestep_proj = self.time_proj(self.act_fn(temb))  # [B, 30720]
        timestep_proj = timestep_proj.unflatten(1, (6, -1))  # [B, 6, 5120]
        enc_text = self.text_embedder(encoder_hidden_states_text)  # [B, 512, 5120]

        return temb, timestep_proj, enc_text


class WanRotaryPosEmbed(nn.Module):
    """3D rotary position embeddings - no branches"""

    def __init__(
        self,
        attention_head_dim: int,
        patch_size: tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        # n.b. for 3D: time, height, width
        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim
        freqs_dtype = (
            torch.float32 if torch.backends.mps.is_available() else torch.float64
        )

        freqs_cos = []
        freqs_sin = []

        for dim in [t_dim, h_dim, w_dim]:
            freq_cos, freq_sin = get_1d_rotary_pos_embed(
                dim,
                max_seq_len,
                theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=freqs_dtype,
            )

            freqs_cos.append(freq_cos)
            freqs_sin.append(freq_sin)

        self.register_buffer("freqs_cos", torch.cat(freqs_cos, dim=1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(freqs_sin, dim=1), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with te.fp8_autocast(
            enabled=True,
            fp8_recipe=transformer_engine.common.recipe.DelayedScaling(),
        ):
            B, C, F, H, W = hidden_states.shape
            p_t, p_h, p_w = self.patch_size
            ppf, pph, ppw = F // p_t, H // p_h, W // p_w

            # split -> expand -> concat
            split_sizes = [
                self.attention_head_dim - 2 * (self.attention_head_dim // 3),
                self.attention_head_dim // 3,
                self.attention_head_dim // 3,
            ]

            freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
            freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

            # 3D positions
            freqs_cos = torch.cat(
                [
                    freqs_cos[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1),
                    freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1),
                    freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1),
                ],
                dim=-1,
            ).reshape(1, ppf * pph * ppw, 1, -1)

            freqs_sin = torch.cat(
                [
                    freqs_sin[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1),
                    freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1),
                    freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1),
                ],
                dim=-1,
            ).reshape(1, ppf * pph * ppw, 1, -1)

            return freqs_cos, freqs_sin


class WanTransformer3DModel(ModelMixin, ConfigMixin, CacheMixin):
    """WAN I2V Transformer - zero branches in forward"""

    @register_to_config
    def __init__(
        self,
        patch_size: tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: str | None = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: int | None = None,
        added_kv_proj_dim: int | None = 1280,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: int | None = None,
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        assert inner_dim == 5120, "WAN I2V uses inner_dim=5120"

        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)

        self.patch_embedding = nn.Conv3d(
            36, inner_dim, kernel_size=patch_size, stride=patch_size
        )

        self.condition_embedder = WanConditionEmbedder(inner_dim)

        self.image_proj = nn.Linear(1280, inner_dim)
        self.image_norm = nn.RMSNorm(inner_dim, eps=eps)

        self.blocks = nn.ModuleList(
            [WanTransformerBlock(inner_dim, ffn_dim) for _ in range(num_layers)]
        )

        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, 36, F, H, W]
        timestep: torch.LongTensor,  # [B]
        encoder_hidden_states: torch.Tensor,  # [B, 512, 4096] - text
        encoder_hidden_states_image: torch.Tensor
        | None = None,  # [B, img_len, 1280] - image features
        return_dict: bool = True,
        attention_kwargs: dict | None = None,  # Ignored
    ):
        B, C, F, H, W = hidden_states.shape

        # rope -> patch -> embed conditions -> concat -> blocks -> out
        rotary_emb = self.rope(hidden_states)  # ([1, L, 1, 128], [1, L, 1, 128])

        temb, timestep_proj, enc_text = self.condition_embedder(
            timestep, encoder_hidden_states
        )

        if encoder_hidden_states_image is not None:
            # Project and normalize image features once
            enc_img = self.image_proj(encoder_hidden_states_image)  # [B, img_len, 5120]
            enc_img = self.image_norm(
                enc_img
            )  # Apply same normalization as would be in cross-attn
            encoder_ctx = torch.cat(
                [enc_img, enc_text], dim=1
            )  # [B, img_len+512, 5120]
        else:
            encoder_ctx = enc_text  # [B, 512, 5120]

        # CRITICAL: Apply patch embedding!
        hidden_states = self.patch_embedding(hidden_states)  # [B, 5120, F', H', W']
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # [B, L, 5120]

        for block in self.blocks:
            hidden_states = block(hidden_states, encoder_ctx, timestep_proj, rotary_emb)

        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(
            2, dim=1
        )  # 2x [B, 1, 5120]
        hidden_states = self.norm_out(hidden_states.float())  # [B, L, 5120]
        hidden_states = (hidden_states * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)  # [B, L, 64]

        # Reshape back to spatial format
        hidden_states = hidden_states.reshape(B, F // 1, H // 2, W // 2, 1, 2, 2, -1)
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = (
            hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        )  # [B, 16, F, H, W]

        return Transformer2DModelOutput(sample=output) if return_dict else (output,)

    @classmethod
    def from_pretrained_stock(
        cls, stock_transformer: nn.Module
    ) -> "WanTransformer3DModel":
        """Load weights from a stock WanTransformer3DModel and fuse projections"""
        config = stock_transformer.config

        # Fix config
        if config.added_kv_proj_dim is None:
            config.added_kv_proj_dim = 1280
        if config.image_dim is None:
            config.image_dim = 1280

        # Fix attention modules and fuse projections
        for idx, block in enumerate(stock_transformer.blocks):
            print(f"fixing up stock transformer blocks in layer {idx}")
            # Fix added_kv_proj_dim for cross-attention
            if (
                hasattr(block.attn2, "add_k_proj")
                and block.attn2.add_k_proj is not None
            ):
                print(f"setting kv_proj_dim in layer {idx}")
                block.attn2.added_kv_proj_dim = 1280

            # Fuse projections if not already fused
            if not hasattr(block.attn1, "to_qkv"):
                print(f"fusing self-attention projections in layer {idx}...")
                block.attn1.fuse_projections()
                print(f"fused self-attention projections in layer {idx}")
            if not hasattr(block.attn2, "to_kv"):
                print(f"fusing cross-attention projections in layer {idx}...")
                block.attn2.fuse_projections()
                print(f"fused cross-attention projections in layer {idx}")

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

        # Copy weights - handle both fused and unfused cases
        with torch.no_grad():
            # Standard components

            print("loading patch embedding...")
            opt_transformer.patch_embedding.load_state_dict(
                stock_transformer.patch_embedding.state_dict()
            )
            print("loaded patch embedding.")

            print("loading rope encodings...")
            opt_transformer.rope.load_state_dict(stock_transformer.rope.state_dict())
            print("loaded rope encodings.")

            print("loading rope conditional embedder...")
            opt_transformer.condition_embedder.load_state_dict(
                stock_transformer.condition_embedder.state_dict()
            )
            print("loaded rope conditional embedder.")

            print("loading norm out...")
            opt_transformer.norm_out.load_state_dict(
                stock_transformer.norm_out.state_dict()
            )
            print("loaded norm out.")

            print("loading proj_out...")
            opt_transformer.proj_out.load_state_dict(
                stock_transformer.proj_out.state_dict()
            )
            print("loaded proj_out...")

            print("loading scale_shift_table...")
            opt_transformer.scale_shift_table.copy_(stock_transformer.scale_shift_table)
            print("loaded scale_shift_table...")

            if hasattr(stock_transformer.blocks[0].attn2, "to_added_kv"):
                # Extract just the K projection part (first half of the weight)
                added_kv_weight = stock_transformer.blocks[0].attn2.to_added_kv.weight
                added_k_weight = added_kv_weight[:5120, :]  # First 5120 rows for K
                opt_transformer.image_proj.weight.copy_(added_k_weight)

                added_kv_bias = stock_transformer.blocks[0].attn2.to_added_kv.bias
                added_k_bias = added_kv_bias[:5120]  # First 5120 elements for K
                opt_transformer.image_proj.bias.copy_(added_k_bias)

                # Copy normalization
                if hasattr(stock_transformer.blocks[0].attn2, "norm_added_k"):
                    opt_transformer.image_norm.load_state_dict(
                        stock_transformer.blocks[0].attn2.norm_added_k.state_dict()
                    )

            for idx, (stock_block, opt_block) in enumerate(
                zip(stock_transformer.blocks, opt_transformer.blocks, strict=False)
            ):
                print(f"handling layer {idx}...")

                # Copy norms and FFN
                opt_block.norm1.load_state_dict(stock_block.norm1.state_dict())
                opt_block.norm2.load_state_dict(stock_block.norm2.state_dict())
                opt_block.norm3.load_state_dict(stock_block.norm3.state_dict())
                opt_block.ffn.load_state_dict(stock_block.ffn.state_dict())
                opt_block.scale_shift_table.copy_(stock_block.scale_shift_table)

                # Self-attention (as before)
                opt_block.attn1.to_qkv.weight.copy_(stock_block.attn1.to_qkv.weight)
                opt_block.attn1.to_qkv.bias.copy_(stock_block.attn1.to_qkv.bias)
                opt_block.attn1.norm_q.load_state_dict(
                    stock_block.attn1.norm_q.state_dict()
                )
                opt_block.attn1.norm_k.load_state_dict(
                    stock_block.attn1.norm_k.state_dict()
                )
                opt_block.attn1.to_out[0].load_state_dict(
                    stock_block.attn1.to_out[0].state_dict()
                )

                # Cross-attention - COMPOSE the weights!
                # Q projection stays the same
                opt_block.attn2.to_q.load_state_dict(
                    stock_block.attn2.to_q.state_dict()
                )
                opt_block.attn2.norm_q.load_state_dict(
                    stock_block.attn2.norm_q.state_dict()
                )

                # For KV, we need to handle the fact that stock has separate text/image projections
                # but we want a unified one. Since the image features will be pre-projected to 5120,
                # we just use the text KV projection (which already expects 5120 input)
                opt_block.attn2.to_kv.weight.copy_(stock_block.attn2.to_kv.weight)
                opt_block.attn2.to_kv.bias.copy_(stock_block.attn2.to_kv.bias)
                opt_block.attn2.norm_k.load_state_dict(
                    stock_block.attn2.norm_k.state_dict()
                )

                # Output projection - handle ModuleList vs single Linear
                if isinstance(stock_block.attn2.to_out, nn.ModuleList):
                    opt_block.attn2.to_out.load_state_dict(
                        stock_block.attn2.to_out[0].state_dict()
                    )
                else:
                    opt_block.attn2.to_out.load_state_dict(
                        stock_block.attn2.to_out.state_dict()
                    )

                # Note: We're ignoring to_added_kv and norm_added_k from stock
                # The image features will be pre-projected and normalized in the main forward pass

        return opt_transformer
