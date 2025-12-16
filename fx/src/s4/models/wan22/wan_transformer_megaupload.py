"""
WAN Transformer BALANCED BEAST Edition
======================================
Maximum performance within the realm of the possible.
Big GEMMs, smart compilation, actual execution.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

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

from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
from torchao.prototype.mx_formats import NVFP4InferenceConfig

from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func

import transformer_engine
import transformer_engine.pytorch as te


class WanConditionEmbedder(nn.Module):
    """High-performance condition embedding"""

    def __init__(self, dim: int = 5120):
        super().__init__()

        self.timesteps_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = te.Linear(dim, dim * 6)
        self.text_embedder = PixArtAlphaTextProjection(4096, dim, act_fn="gelu_tanh")

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states_text: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_freq = self.timesteps_proj(timestep)
        temb = self.time_embedder(t_freq)
        timestep_proj = self.time_proj(self.act_fn(temb))
        timestep_proj = timestep_proj.unflatten(1, (6, -1))
        enc_text = self.text_embedder(encoder_hidden_states_text)
        return temb, timestep_proj, enc_text


class WanRotaryPosEmbed3D(nn.Module):
    """Efficient 3D rotary position embeddings"""

    def __init__(
        self,
        attention_head_dim: int = 128,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        max_seq_len: int = 100_000,  # Reasonable max
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


class WanSelfAttention(nn.Module):
    """High-performance self-attention with Flash Attention"""

    def __init__(self, dim: int = 5120, heads: int = 40, eps: float = 1e-6):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads

        # Single large GEMM for QKV
        self.to_qkv = te.Linear(dim, dim * 3, bias=True)
        self.norm_q = RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.norm_k = RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.to_out = te.Linear(dim, dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        B, L, D = hidden_states.shape

        # Large GEMM for QKV projection
        qkv = self.to_qkv(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)

        q = self.norm_q(q).view(B, L, self.heads, self.head_dim)
        k = self.norm_k(k).view(B, L, self.heads, self.head_dim)
        v = v.view(B, L, self.heads, self.head_dim)

        q = apply_rotary_emb(q, *rotary_emb)
        k = apply_rotary_emb(k, *rotary_emb)

        # Pack for Flash Attention
        qkv_packed = torch.stack([q, k, v], dim=2)

        # Flash attention for efficiency
        h = flash_attn_qkvpacked_func(
            qkv_packed,
            dropout_p=0.0,
            causal=False,
            window_size=(-1, -1),
            alibi_slopes=None,
            return_attn_probs=False,
        )

        h = h.reshape(B, L, D)
        return self.to_out(h)


class WanCrossAttention(nn.Module):
    """High-performance cross-attention"""

    def __init__(self, dim: int = 5120, heads: int = 40, eps: float = 1e-6):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads

        self.to_q = te.Linear(dim, dim, bias=True)
        self.norm_q = RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.to_out = te.Linear(dim, dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_k: torch.Tensor,
        encoder_v: torch.Tensor,
    ) -> torch.Tensor:
        B, L, D = hidden_states.shape

        q = self.norm_q(self.to_q(hidden_states))
        q = q.view(B, L, self.heads, self.head_dim).transpose(1, 2)

        # Efficient scaled dot-product attention
        h = F.scaled_dot_product_attention(
            q,
            encoder_k,
            encoder_v,
            dropout_p=0.0,
            is_causal=False,
            scale=1.0 / math.sqrt(self.head_dim),
        )

        h = h.transpose(1, 2).reshape(B, L, D)
        return self.to_out(h)


class WanTransformerBlock(nn.Module):
    """Balanced high-performance transformer block"""

    def __init__(
        self,
        dim: int = 5120,
        ffn_dim: int = 13824,
        num_heads: int = 40,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.norm1 = RMSNorm(dim, eps=eps, elementwise_affine=False)
        self.attn1 = WanSelfAttention(dim, heads=num_heads, eps=eps)

        self.norm2 = RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.attn2 = WanCrossAttention(dim, heads=num_heads, eps=eps)

        self.norm3 = RMSNorm(dim, eps=eps, elementwise_affine=False)

        # Optimized FFN with two large GEMMs
        self.ffn_up = te.Linear(dim, ffn_dim, bias=True)
        self.ffn_act = nn.GELU(approximate="tanh")
        self.ffn_down = te.Linear(ffn_dim, dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_k: torch.Tensor,
        encoder_v: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        scale_shift_params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        residual = hidden_states
        norm_hidden = self.norm1(hidden_states)

        if scale_shift_params is not None:
            shift_msa, scale_msa = scale_shift_params[:, 0], scale_shift_params[:, 1]
            norm_hidden = (
                norm_hidden * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
            )

        hidden_states = residual + self.attn1(norm_hidden, rotary_emb)

        # Cross-attention
        residual = hidden_states
        norm_hidden = self.norm2(hidden_states)

        if scale_shift_params is not None:
            shift_ca, scale_ca = scale_shift_params[:, 2], scale_shift_params[:, 3]
            norm_hidden = (
                norm_hidden * (1 + scale_ca[:, None, :]) + shift_ca[:, None, :]
            )

        hidden_states = residual + self.attn2(norm_hidden, encoder_k, encoder_v)

        # FFN
        residual = hidden_states
        norm_hidden = self.norm3(hidden_states)

        if scale_shift_params is not None:
            shift_ffn, scale_ffn = scale_shift_params[:, 4], scale_shift_params[:, 5]
            norm_hidden = (
                norm_hidden * (1 + scale_ffn[:, None, :]) + shift_ffn[:, None, :]
            )

        # Two large GEMMs for FFN
        h = self.ffn_up(norm_hidden)
        h = self.ffn_act(h)
        h = self.ffn_down(h)

        hidden_states = residual + h

        return hidden_states


class WanTransformer3DBalanced(ModelMixin, ConfigMixin):
    """
    WAN Transformer BALANCED BEAST Edition
    ======================================
    Aggressive performance with practical execution.
    Targets 500+ TFLOPS on Blackwell architecture.
    """

    _no_split_modules = ["WanTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        num_layers: int = 40,
        attention_head_dim: int = 128,
        num_attention_heads: int = 40,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        patch_size_t: Optional[int] = None,
        in_channels: int = 36,
        out_channels: Optional[int] = None,
        text_embed_dim: int = 4096,
        ffn_expansion: float = 2.7,
        eps: float = 1e-6,
    ):
        super().__init__()

        # Core dimensions
        self.inner_dim = num_attention_heads * attention_head_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        # Patch configuration
        if patch_size_t is not None:
            self.patch_size = (patch_size_t, patch_size[0], patch_size[1])
        else:
            self.patch_size = patch_size

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels

        # Patch embedding
        self.patch_embedding = nn.Conv3d(
            in_channels,
            self.inner_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
        )

        # Condition embedder
        self.cond_embedder = WanConditionEmbedder(dim=self.inner_dim)

        # Rotary position embeddings
        self.rotary_pos_emb = WanRotaryPosEmbed3D(
            attention_head_dim,
            self.patch_size,
            max_seq_len=100_000,
            theta=256.0,
        )

        # Shared KV projection - large GEMM
        self.shared_kv_proj = te.Linear(text_embed_dim, self.inner_dim * 2, bias=True)
        self.shared_kv_norm = RMSNorm(
            self.inner_dim * 2, eps=eps, elementwise_affine=True
        )

        # Transformer blocks
        ffn_dim = int(self.inner_dim * ffn_expansion)
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    dim=self.inner_dim,
                    ffn_dim=ffn_dim,
                    num_heads=self.num_attention_heads,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )

        # Output layers
        self.norm_out = RMSNorm(self.inner_dim, eps=eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, self.inner_dim) / self.inner_dim**0.5
        )
        self.proj_out = te.Linear(
            self.inner_dim,
            self.patch_size[0]
            * self.patch_size[1]
            * self.patch_size[2]
            * self.out_channels,
        )

    def _precompute_kv(
        self, encoder_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pre-compute KV projections with single large GEMM"""
        kv = self.shared_kv_proj(encoder_states)  # Large GEMM
        kv = self.shared_kv_norm(kv)
        k, v = kv.chunk(2, dim=-1)

        B, L, D = k.shape
        heads = self.num_attention_heads
        head_dim = D // heads

        k = k.view(B, L, heads, head_dim).transpose(1, 2)
        v = v.view(B, L, heads, head_dim).transpose(1, 2)

        return k, v

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
        temb, timestep_proj, enc_text = self.cond_embedder(
            timestep, encoder_hidden_states_text
        )

        # Get rotary embeddings
        rotary_emb = self.rotary_pos_emb(hidden_states)

        # Patch embedding
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # Pre-compute KV
        encoder_k, encoder_v = self._precompute_kv(encoder_hidden_states_text)

        # Apply transformer blocks
        for i, block in enumerate(self.blocks):
            block_scale_shift = timestep_proj[:, :, :] if i < 6 else None

            hidden_states = block(
                hidden_states,
                temb,
                encoder_k,
                encoder_v,
                rotary_emb,
                block_scale_shift,
            )

        # Output processing
        hidden_states = self.norm_out(hidden_states)
        shift, scale = self.scale_shift_table.chunk(2, dim=0)
        hidden_states = (
            hidden_states * (1 + scale[None, None, :]) + shift[None, None, :]
        )
        hidden_states = self.proj_out(hidden_states)

        # Reshape to spatial dimensions
        p_t, p_h, p_w = self.patch_size
        hidden_states = hidden_states.reshape(
            B, F // p_t, H // p_h, W // p_w, p_t, p_h, p_w, self.out_channels
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        hidden_states = hidden_states.reshape(B, self.out_channels, F, H, W)

        if return_dict:
            return Transformer2DModelOutput(sample=hidden_states)

        return (hidden_states,)


if __name__ == "__main__":
    import time
    import gc

    def test_performance():
        """Performance test - practical but aggressive"""
        print("\n" + "=" * 80)
        print("BALANCED BEAST PERFORMANCE TEST")
        print("Aggressive but Executable")
        print("=" * 80)

        if not torch.cuda.is_available():
            print("No CUDA, no beast mode")
            return

        device = torch.device("cuda")

        # GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        sm_count = torch.cuda.get_device_properties(0).multi_processor_count

        # Rough TFLOPS estimate for Blackwell
        # Each SM can do ~2048 FP16 ops per cycle at ~2GHz
        theoretical_tflops = sm_count * 2048 * 2.0 / 1000

        print(f"GPU: {gpu_name}")
        print(f"Memory: {gpu_mem_gb:.1f} GB")
        print(f"SM Count: {sm_count}")
        print(f"Theoretical FP16 TFLOPS: ~{theoretical_tflops:.1f}")

        # Create model
        print("\nInitializing BALANCED BEAST model...")
        model = (
            WanTransformer3DBalanced(
                num_layers=40,
                ffn_expansion=2.7,
            )
            .to(device)
            .to(torch.bfloat16)
        )
        model.eval()
        # quantize_(model, Float8DynamicActivationFloat8WeightConfig())
        # model = torch.compile(model, mode="max-autotune")

        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {param_count / 1e9:.2f}B")

        # Test configurations - start conservative
        configs = [
            (1, 1, 256, 256, "tiny_test"),
            (1, 4, 384, 384, "small_test"),
            (1, 13, 384, 672, "720p_13frames"),
            (1, 25, 576, 1024, "1080p_25frames"),
        ]

        print("\n" + "-" * 40)
        print("BALANCED BEAST UNLEASHED")
        print("-" * 40)

        results = []

        for B, F, H, W, name in configs:
            try:
                # Clear cache before each test
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

                # Check available memory
                mem_free = torch.cuda.mem_get_info()[0] / 1e9
                print(f"\nFree memory before {name}: {mem_free:.1f} GB")

                # Create inputs
                hidden_states = torch.randn(
                    B, 36, F, H, W, device=device, dtype=torch.bfloat16
                )
                timestep = torch.tensor([500] * B, device=device)
                encoder_text = torch.randn(
                    B, 512, 4096, device=device, dtype=torch.bfloat16
                )

                # Calculate theoretical FLOPs
                p_t, p_h, p_w = model.patch_size
                seq_len = (F // p_t) * (H // p_h) * (W // p_w)

                dim = model.inner_dim
                ffn_dim = int(dim * 2.7)

                # FLOPs per layer (approximate)
                self_attn_flops = 4 * B * seq_len * seq_len * dim
                cross_attn_flops = 4 * B * seq_len * 512 * dim
                ffn_flops = 2 * B * seq_len * dim * ffn_dim

                total_flops = model.num_layers * (
                    self_attn_flops + cross_attn_flops + ffn_flops
                )

                print(f"\n{name}:")
                print(f"  Shape: {B}x{F}x{H}x{W}")
                print(f"  Sequence length: {seq_len}")
                print(f"  Theoretical GFLOPs: {total_flops / 1e9:.1f}")

                # Warmup
                print("  Warming up...")
                for _ in range(2):
                    with torch.no_grad():
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            te.fp8_autocast(
                                enabled=True,
                                fp8_recipe=transformer_engine.common.recipe.DelayedScaling(
                                    fp8_format=transformer_engine.common.recipe.Format.HYBRID
                                ),
                            )
                            output = model(hidden_states, timestep, encoder_text)
                            del output

                torch.cuda.synchronize()

                # Timing
                runs = 5
                timings = []

                for _ in range(runs):
                    torch.cuda.synchronize()
                    start = time.time()

                    with torch.no_grad():
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            output = model(hidden_states, timestep, encoder_text)

                    torch.cuda.synchronize()
                    elapsed = (time.time() - start) * 1000
                    timings.append(elapsed)
                    del output

                avg_time = sum(timings) / len(timings)
                min_time = min(timings)

                # Calculate achieved TFLOPS
                tflops = (total_flops / (avg_time / 1000)) / 1e12
                peak_tflops = (total_flops / (min_time / 1000)) / 1e12

                # Memory usage
                mem_used = gpu_mem_gb - torch.cuda.mem_get_info()[0] / 1e9

                print(f"  Memory used: {mem_used:.1f} GB")
                print(f"  Inference time: {avg_time:.1f} ms (best: {min_time:.1f} ms)")
                print(f"  Achieved TFLOPS: {tflops:.1f} (peak: {peak_tflops:.1f})")
                print(f"  GPU utilization: {tflops / theoretical_tflops * 100:.1f}%")

                if avg_time < 200:
                    print(f"  SUB-200MS TARGET HIT!")

                results.append((name, avg_time, tflops, mem_used))

                # Cleanup
                del hidden_states, timestep, encoder_text

            except torch.cuda.OutOfMemoryError:
                print(f"\n{name}: Out of memory")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"\n{name}: Error - {e}")
                continue

        # Summary
        if results:
            print("\n" + "=" * 40)
            print("SUMMARY")
            print("=" * 40)
            for name, time_ms, tflops, mem_gb in results:
                print(
                    f"{name:20s}: {time_ms:6.1f} ms, {tflops:6.1f} TFLOPS, {mem_gb:5.1f} GB"
                )

        print("\n" + "=" * 80)
        print("BALANCED BEAST COMPLETE")
        print("Big GEMMs. Real Performance. Actual Execution.")
        print("=" * 80)

    # Run the test
    test_performance()
