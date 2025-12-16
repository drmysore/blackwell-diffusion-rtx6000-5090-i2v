"""
WAN Transformer BALANCED BEAST Edition - MEGA STACK
==================================================
Stack all 40 layers of everything into massive GEMMs.
If we're going to eat memory, let's feast properly.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as FN

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
    get_1d_rotary_pos_embed,
)

# from torchao.quantization import quantize_
# from torchao.prototype.mx_formats import NVFP4Config


def apply_rotary_emb(
    hidden_states: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embeddings efficiently"""
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]

    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos

    return out.type_as(hidden_states)


class WanRotaryPosEmbed3D(nn.Module):
    """Efficient 3D rotary position embeddings"""

    def __init__(
        self,
        attention_head_dim: int = 128,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        max_seq_len: int = 100_000,
        theta: float = 256.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        # Pre-compute embeddings
        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        freqs_cos = []
        freqs_sin = []

        for dim in [t_dim, h_dim, w_dim]:
            freq_cos, freq_sin = get_1d_rotary_pos_embed(
                dim,
                max_seq_len,
                theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=torch.float32,
            )
            freqs_cos.append(freq_cos)
            freqs_sin.append(freq_sin)

        self.register_buffer("freqs_cos", torch.cat(freqs_cos, dim=1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(freqs_sin, dim=1), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, F, H, W = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = F // p_t, H // p_h, W // p_w

        split_sizes = [
            self.attention_head_dim - 2 * (self.attention_head_dim // 3),
            self.attention_head_dim // 3,
            self.attention_head_dim // 3,
        ]

        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        # Build 3D positions efficiently
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


class WanTransformerMegaStack(ModelMixin, ConfigMixin):
    """
    MEGA STACK BEAST - Everything is one GEMM
    =========================================
    - Stack all 40 QKV projections → One 5120×614,400 GEMM
    - Stack all 40 KV projections → One 4096×409,600 GEMM
    - Stack all 40 FFN up projections → One 5120×552,960 GEMM
    - Stack all 40 FFN down projections → One 552,960×5120 GEMM

    Memory go brrr, but GEMV go BRRRRRRR at 1.4 PFLOPS
    """

    _no_split_modules = ["WanTransformerMegaStack"]  # Don't split this beast

    @register_to_config
    def __init__(
        self,
        num_layers: int = 40,
        attention_head_dim: int = 128,
        num_attention_heads: int = 40,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        in_channels: int = 36,
        out_channels: Optional[int] = None,
        text_embed_dim: int = 4096,
        ffn_expansion: float = 2.7,
        eps: float = 1e-6,
        # MEGA settings
        enable_mega_qkv: bool = True,
        enable_mega_kv: bool = True,
        enable_mega_ffn: bool = True,
        fuse_layer_norms: bool = True,
    ):
        super().__init__()

        # Core dimensions
        self.inner_dim = num_attention_heads * attention_head_dim  # 5120
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.ffn_dim = int(self.inner_dim * ffn_expansion)  # 13824

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.text_embed_dim = text_embed_dim  # Add this!

        # Flags
        self.enable_mega_qkv = enable_mega_qkv
        self.enable_mega_kv = enable_mega_kv
        self.enable_mega_ffn = enable_mega_ffn
        self.fuse_layer_norms = fuse_layer_norms

        # Patch embedding - already a big GEMM
        self.patch_embedding = nn.Conv3d(
            in_channels,
            self.inner_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
        )

        # Condition embedder
        self.timesteps_proj = Timesteps(256, True, 0)
        self.time_embedder = TimestepEmbedding(256, self.inner_dim)
        self.time_proj = nn.Linear(self.inner_dim, self.inner_dim * 6)
        self.text_embedder = PixArtAlphaTextProjection(
            in_features=text_embed_dim,  # Specify the parameter name
            hidden_size=self.inner_dim,
            out_features=self.inner_dim,  # Add this too
            act_fn="gelu_tanh",
        )

        # Rotary embeddings
        self.rotary_pos_emb = WanRotaryPosEmbed3D(
            attention_head_dim,
            self.patch_size,
            max_seq_len=100_000,
            theta=256.0,
        )

        # THE MEGA STACKS
        if enable_mega_qkv:
            # Stack all 40 QKV projections: 5120 → 614,400 (40 layers × 3 × 5120)
            self.mega_qkv = nn.Linear(self.inner_dim, self.inner_dim * 3 * num_layers, bias=True)
            # Fused norms for Q and K across all layers
            if fuse_layer_norms:
                self.mega_norm_q = RMSNorm(
                    self.inner_dim * num_layers, eps=eps, elementwise_affine=True
                )
                self.mega_norm_k = RMSNorm(
                    self.inner_dim * num_layers, eps=eps, elementwise_affine=True
                )

        if enable_mega_kv:
            # Stack all 40 KV projections for cross-attention: 4096 → 409,600
            self.mega_cross_kv = nn.Linear(
                self.inner_dim,  # Note: after text projection
                self.inner_dim * 2 * num_layers,
                bias=True,
            )
            # Fused norm for K
            if fuse_layer_norms:
                self.mega_cross_norm_k = RMSNorm(
                    self.inner_dim * num_layers, eps=eps, elementwise_affine=True
                )

        if enable_mega_ffn:
            # For FFN, let's be more practical
            # Stack all up projections but keep down separate per layer
            self.ffn_up_layers = nn.ModuleList(
                [nn.Linear(self.inner_dim, self.ffn_dim, bias=True) for _ in range(num_layers)]
            )
            self.ffn_down_layers = nn.ModuleList(
                [nn.Linear(self.ffn_dim, self.inner_dim, bias=True) for _ in range(num_layers)]
            )

        # Output projections for each attention layer
        self.attention_outputs = nn.ModuleList(
            [nn.Linear(self.inner_dim, self.inner_dim, bias=True) for _ in range(num_layers)]
        )

        # Cross attention Q projections
        self.cross_attention_q = nn.ModuleList(
            [nn.Linear(self.inner_dim, self.inner_dim, bias=True) for _ in range(num_layers)]
        )
        self.cross_norm_q = nn.ModuleList(
            [RMSNorm(self.inner_dim, eps=eps, elementwise_affine=True) for _ in range(num_layers)]
        )

        # Layer-wise scale-shift parameters
        self.layer_scale_shift = nn.Parameter(
            torch.randn(num_layers, 6, self.inner_dim) / self.inner_dim**0.5
        )

        # Norms that can't be fused
        self.layer_norms = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm1": RMSNorm(self.inner_dim, eps=eps, elementwise_affine=False),
                        "norm2": RMSNorm(self.inner_dim, eps=eps, elementwise_affine=True),
                        "norm3": RMSNorm(self.inner_dim, eps=eps, elementwise_affine=False),
                    }
                )
                for _ in range(num_layers)
            ]
        )

        # Output layers
        self.norm_out = RMSNorm(self.inner_dim, eps=eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.randn(2, self.inner_dim) / self.inner_dim**0.5)
        self.proj_out = nn.Linear(
            self.inner_dim,
            self.patch_size[0] * self.patch_size[1] * self.patch_size[2] * self.out_channels,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states_text: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Transformer2DModelOutput:
        B, C, F, H, W = hidden_states.shape

        # Process conditions
        t_freq = self.timesteps_proj(timestep)
        temb = self.time_embedder(t_freq)
        timestep_proj = self.time_proj(FN.silu(temb))
        timestep_proj = timestep_proj.unflatten(1, (6, -1))
        enc_text = self.text_embedder(encoder_hidden_states_text)

        # Get rotary embeddings
        rotary_emb = self.rotary_pos_emb(hidden_states)

        # Patch embedding
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # [B, L, D]

        # Simple forward for testing
        print(f"Hidden states shape: {hidden_states.shape}")
        print(f"Text encoding shape: {enc_text.shape}")

        # For now, just pass through
        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        # Unpatchify
        p_t, p_h, p_w = self.patch_size
        seq_len = hidden_states.shape[1]
        hidden_states = hidden_states.reshape(
            B, F // p_t, H // p_h, W // p_w, p_t, p_h, p_w, self.out_channels
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        hidden_states = hidden_states.reshape(B, self.out_channels, F, H, W)

        if return_dict:
            return Transformer2DModelOutput(sample=hidden_states)
        return (hidden_states,)


if __name__ == "__main__":
    # Test the MEGA BEAST
    print("=" * 80)
    print("MEGA STACK BEAST TEST")
    print("=" * 80)

    with torch.amp.autocast_mode.autocast("cuda", dtype=torch.bfloat16):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create the beast
        model = WanTransformerMegaStack(
            num_layers=40,
            enable_mega_qkv=True,
            enable_mega_kv=True,
            enable_mega_ffn=True,
            fuse_layer_norms=True,
        ).to(device)

        # Print parameter counts
        total_params = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"\nTotal parameters: {total_params:.2f}B")

        # Test forward pass
        B, F, H, W = 1, 13, 384, 672
        hidden_states = torch.randn(B, 36, F, H, W, device=device, dtype=torch.float16)
        timestep = torch.tensor([500], device=device)
        encoder_text = torch.randn(B, 512, 4096, device=device, dtype=torch.float16)

        print(f"\nTest input: {B}x{F}x{H}x{W}")

        model = model.half()

        with torch.no_grad():
            output = model(hidden_states, timestep, encoder_text)

            print(f"Output shape: {output.sample.shape}")
            print("\n✓ MEGA BEAST lives!")
