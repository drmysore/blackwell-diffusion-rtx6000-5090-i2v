#!/usr/bin/env python3
"""
WAN 2.2 Example Usage Script
============================

Demonstrates the three-layer architecture and various usage patterns.
"""

import torch
import torch.nn as nn
import time

# Import our modules
from wan_attention_pure import (
    TransformerBlock,
    SelfAttention,
    RotaryPosEmbed,
)

from wan_attention_diffusers import (
    WanTransformer3DModel,
)

from wan_serialization import (
    WANModelSerializer,
    QuantizationConfig,
    QuantizationType,
)


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print("=" * 60)


def example_pure_inference():
    """Example: Using pure PyTorch modules for inference."""
    print_header("Pure PyTorch Inference")

    # Create a transformer block
    block = TransformerBlock(
        dim=5120,
        num_heads=40,
        ffn_dim=13824,
    )

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    block = block.to(device).eval()
    print(f"Device: {device}")

    # Create dummy inputs
    batch_size = 1
    seq_len = 197  # Typical for 14x14 patches
    context_len = 512  # Text tokens

    hidden_states = torch.randn(batch_size, seq_len, 5120, device=device) * 0.02
    context = torch.randn(batch_size, context_len, 5120, device=device) * 0.02
    conditioning = torch.randn(batch_size, 6, 5120, device=device) * 0.02

    # Create RoPE
    rope = RotaryPosEmbed(128)
    rotary_emb = rope(seq_len, device=device, dtype=hidden_states.dtype)

    # Forward pass
    with torch.no_grad():
        output = block(hidden_states, context, conditioning, rotary_emb)

    print(f"Input shape:  {hidden_states.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output norm:  {output.norm().item():.4f}")

    # Compile for performance (if using PyTorch 2.0+)
    if hasattr(torch, "compile"):
        print("\nCompiling model...")
        compiled_block = torch.compile(block, mode="reduce-overhead")

        # Benchmark
        if device == "cuda":
            torch.cuda.synchronize()
            start = time.perf_counter()

            for _ in range(10):
                with torch.no_grad():
                    _ = compiled_block(hidden_states, context, conditioning, rotary_emb)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            print(f"Compiled: {elapsed / 10 * 1000:.2f} ms per forward pass")


def example_diffusers_compatibility():
    """Example: Using diffusers-compatible wrapper."""
    print_header("Diffusers Compatibility")

    # Create diffusers-compatible model
    model = WanTransformer3DModel(
        num_layers=2,  # Small for demo
        num_attention_heads=40,
        attention_head_dim=128,
        in_channels=36,
        out_channels=16,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # Create inputs in diffusers format
    batch_size = 1
    frames = 16
    height = 32  # Small for demo
    width = 32

    hidden_states = torch.randn(batch_size, 36, frames, height, width, device=device) * 0.02

    timestep = torch.tensor([500], device=device)

    encoder_hidden_states = torch.randn(batch_size, 512, 4096, device=device) * 0.02

    # Optional image conditioning
    encoder_hidden_states_image = torch.randn(batch_size, 256, 1280, device=device) * 0.02

    # Forward pass
    print(f"Input shape: {hidden_states.shape}")

    with torch.no_grad():
        output = model(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_hidden_states_image,
            return_dict=True,
        )

    print(f"Output shape: {output.sample.shape}")
    print(f"Output type: {type(output).__name__}")


def example_serialization():
    """Example: Model serialization and quantization setup."""
    print_header("Serialization and Quantization")

    # Create a small model for demo
    model = SelfAttention(dim=768, num_heads=12)

    # Create serializer
    serializer = WANModelSerializer()

    # Extract state dict
    state_dict = serializer.extract_state_dict(model)
    print(f"State dict keys: {list(state_dict.keys())}")

    # Show tensor metadata
    for name, tensor in list(state_dict.items())[:2]:
        metadata = serializer._create_metadata(name, tensor)
        print(f"\n{name}:")
        print(f"  Shape: {metadata.shape}")
        print(f"  Dtype: {metadata.dtype}")
        print(f"  Range: [{metadata.min_val:.4f}, {metadata.max_val:.4f}]")

    # Demonstrate quantization config
    print("\nQuantization configurations:")

    configs = {
        "FP16 (baseline)": QuantizationConfig(
            weight_quant=QuantizationType.NONE,
            weight_bits=16,
        ),
        "FP8 E4M3": QuantizationConfig(
            weight_quant=QuantizationType.FP8_E4M3,
            weight_bits=8,
        ),
        "NVFP4 (Blackwell)": QuantizationConfig(
            weight_quant=QuantizationType.NVFP4,
            weight_bits=4,
            kv_cache_quant=QuantizationType.NVFP4,
            kv_cache_bits=4,
        ),
    }

    for name, config in configs.items():
        total_bits = sum(t.numel() * config.weight_bits for t in state_dict.values())
        total_mb = total_bits / (8 * 1024 * 1024)
        print(f"  {name}: {total_mb:.2f} MB")


def example_optimization_comparison():
    """Example: Compare optimized vs standard implementations."""
    print_header("Optimization Comparison")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    device = "cuda"

    # Configuration
    dim = 5120
    num_heads = 40
    seq_len = 197
    batch_size = 1

    # Create standard attention (simplified for comparison)
    class StandardAttention(nn.Module):
        def __init__(self, dim, num_heads):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.scale = self.head_dim**-0.5

            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
            self.out_proj = nn.Linear(dim, dim)

        def forward(self, x):
            B, L, D = x.shape

            q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = torch.softmax(attn, dim=-1)

            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).reshape(B, L, D)
            return self.out_proj(out)

    # Create both versions
    standard_attn = StandardAttention(dim, num_heads).to(device).eval()
    optimized_attn = SelfAttention(dim, num_heads, qk_norm=False).to(device).eval()

    # Create input
    x = torch.randn(batch_size, seq_len, dim, device=device) * 0.02

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = standard_attn(x)
            _ = optimized_attn(x)

    # Benchmark standard
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(100):
        with torch.no_grad():
            _ = standard_attn(x)

    torch.cuda.synchronize()
    standard_time = (time.perf_counter() - start) / 100 * 1000

    # Benchmark optimized
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(100):
        with torch.no_grad():
            _ = optimized_attn(x)

    torch.cuda.synchronize()
    optimized_time = (time.perf_counter() - start) / 100 * 1000

    print(f"Standard attention:  {standard_time:.2f} ms")
    print(f"Optimized attention: {optimized_time:.2f} ms")
    print(f"Speedup: {standard_time / optimized_time:.2f}x")

    # Memory comparison
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        _ = standard_attn(x)
    standard_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        _ = optimized_attn(x)
    optimized_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

    print("\nMemory usage:")
    print(f"Standard:  {standard_memory:.2f} MB")
    print(f"Optimized: {optimized_memory:.2f} MB")
    print(f"Reduction: {(1 - optimized_memory / standard_memory) * 100:.1f}%")


def example_deployment_scenarios():
    """Example: Different deployment configurations."""
    print_header("Deployment Scenarios")

    def create_deployment_config(scenario: str) -> dict:
        """Create configuration for different deployment scenarios."""

        configs = {
            "edge": {
                "dim": 768,  # Smaller model
                "num_heads": 12,
                "ffn_dim": 2048,
                "dtype": torch.float32,
                "device": "cpu",
            },
            "consumer_gpu": {
                "dim": 5120,
                "num_heads": 40,
                "ffn_dim": 13824,
                "dtype": torch.float16,
                "device": "cuda",
            },
            "datacenter": {
                "dim": 5120,
                "num_heads": 40,
                "ffn_dim": 13824,
                "dtype": torch.bfloat16,
                "device": "cuda",
            },
            "blackwell": {
                "dim": 5120,
                "num_heads": 40,
                "ffn_dim": 13824,
                "dtype": torch.float16,  # Will be FP4 when available
                "device": "cuda",
            },
        }

        return configs.get(scenario, configs["consumer_gpu"])

    # Show different deployment configurations
    scenarios = ["edge", "consumer_gpu", "datacenter", "blackwell"]

    for scenario in scenarios:
        config = create_deployment_config(scenario)
        print(f"\n{scenario.upper()} Configuration:")
        print(f"  Dimensions: {config['dim']}")
        print(f"  Heads: {config['num_heads']}")
        print(f"  FFN: {config['ffn_dim']}")
        print(f"  Dtype: {config['dtype']}")
        print(f"  Device: {config['device']}")

        # Estimate memory
        params = (
            config["dim"] * config["dim"] * 3  # QKV
            + config["dim"] * config["dim"]  # Out
            + config["dim"] * config["ffn_dim"] * 2  # FFN
        )

        bytes_per_param = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
            # torch.float4: 0.5,  # Future
        }

        memory_mb = params * bytes_per_param[config["dtype"]] / (1024 * 1024)
        print(f"  Est. memory per block: {memory_mb:.1f} MB")


def main():
    """Run all examples."""
    print("WAN 2.2 Implementation Examples")
    print("=" * 60)

    # Run examples
    example_pure_inference()
    example_diffusers_compatibility()
    example_serialization()
    example_optimization_comparison()
    example_deployment_scenarios()

    print("\n" + "=" * 60)
    print("Examples completed successfully!")


if __name__ == "__main__":
    main()
