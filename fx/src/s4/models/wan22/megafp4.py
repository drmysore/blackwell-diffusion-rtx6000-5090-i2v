#!/usr/bin/env python3
"""
FP8 Format Comparison Test
Compares BF16, E4M3, and HYBRID formats for linear layers
"""

import torch
import torch.functional as F
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling
import time
import numpy as np


def benchmark_format(layer, input_tensor, recipe=None, warmup=5, runs=20):
    """Benchmark a single format"""
    # Warmup
    if recipe:
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            for _ in range(warmup):
                _ = layer(input_tensor)
    else:
        for _ in range(warmup):
            _ = layer(input_tensor)

    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        if recipe:
            with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
                output = layer(input_tensor)
        else:
            output = layer(input_tensor)

        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    return output, np.mean(times), np.std(times)


def compare_formats():
    """Compare different FP8 formats with BF16 baseline"""

    print("=" * 80)
    print("FP8 Format Comparison (BF16 vs E4M3 vs HYBRID)")
    print("=" * 80)

    # Create recipes for different formats
    e4m3_recipe = DelayedScaling(
        fp8_format=Format.E4M3,
        amax_history_len=16,
        amax_compute_algo="most_recent",
    )

    # HYBRID uses E5M2 for forward activations and E4M3 for everything else
    hybrid_recipe = DelayedScaling(
        fp8_format=Format.HYBRID,
        amax_history_len=16,
        amax_compute_algo="most_recent",
    )

    # Test configurations
    test_configs = [
        (4096, 4096, 4096, "Square 4K"),
        (8192, 5120, 5120, "8K × 5K × 5K"),
        (16384, 5120, 5120, "16K × 5K × 5K"),
        (8192, 5120, 15360, "8K × 5K × 15K (3-layer QKV)"),
        (65536, 5120, 5120, "64K × 5K × 5K (large batch)"),
    ]

    for m, n, k, desc in test_configs:
        print(f"\n{desc}: M={m}, N={n}, K={k}")
        print("-" * 60)

        # Create layer and input
        layer = te.Linear(n, k, bias=True).cuda().bfloat16()
        inp = torch.randn(m, n, device="cuda").bfloat16()

        # Benchmark BF16 (baseline)
        out_bf16, bf16_time, bf16_std = benchmark_format(layer, inp, recipe=None)

        # Benchmark E4M3
        out_e4m3, e4m3_time, e4m3_std = benchmark_format(layer, inp, recipe=e4m3_recipe)

        # Benchmark HYBRID
        out_hybrid, hybrid_time, hybrid_std = benchmark_format(
            layer, inp, recipe=hybrid_recipe
        )

        # Calculate TFLOPS
        flops = 2 * m * n * k / 1e12
        bf16_tflops = flops / (bf16_time / 1000)
        e4m3_tflops = flops / (e4m3_time / 1000)
        hybrid_tflops = flops / (hybrid_time / 1000)

        # Print performance results
        print(
            f"BF16:   {bf16_time:7.2f} ± {bf16_std:5.2f} ms, {bf16_tflops:7.1f} TFLOPS (baseline)"
        )
        print(
            f"E4M3:   {e4m3_time:7.2f} ± {e4m3_std:5.2f} ms, {e4m3_tflops:7.1f} TFLOPS ({e4m3_tflops / bf16_tflops:.2f}x)"
        )
        print(
            f"HYBRID: {hybrid_time:7.2f} ± {hybrid_std:5.2f} ms, {hybrid_tflops:7.1f} TFLOPS ({hybrid_tflops / bf16_tflops:.2f}x)"
        )

        # Accuracy comparison
        print("\nAccuracy vs BF16:")

        # E4M3 errors
        abs_error_e4m3 = torch.abs(out_bf16 - out_e4m3)
        rel_error_e4m3 = abs_error_e4m3 / (torch.abs(out_bf16) + 1e-8)
        rmse_e4m3 = torch.sqrt(torch.mean((out_bf16 - out_e4m3) ** 2))

        # HYBRID errors
        abs_error_hybrid = torch.abs(out_bf16 - out_hybrid)
        rel_error_hybrid = abs_error_hybrid / (torch.abs(out_bf16) + 1e-8)
        rmse_hybrid = torch.sqrt(torch.mean((out_bf16 - out_hybrid) ** 2))

        print(
            f"E4M3:   MAE={abs_error_e4m3.mean():.6f}, RMSE={rmse_e4m3:.6f}, "
            f"Rel={rel_error_e4m3.mean():.4f}"
        )
        print(
            f"HYBRID: MAE={abs_error_hybrid.mean():.6f}, RMSE={rmse_hybrid:.6f}, "
            f"Rel={rel_error_hybrid.mean():.4f}"
        )

        # Cleanup
        del layer, inp, out_bf16, out_e4m3, out_hybrid
        torch.cuda.empty_cache()


def transformer_layer_comparison():
    """Compare formats in a transformer-like setting"""

    print("\n" + "=" * 80)
    print("Transformer Layer Comparison")
    print("=" * 80)

    batch_size = 8
    seq_len = 2048
    hidden_dim = 5120
    num_heads = 40
    head_dim = hidden_dim // num_heads

    print(f"\nConfiguration:")
    print(f"  Batch: {batch_size}, Seq: {seq_len}, Hidden: {hidden_dim}")
    print(f"  Heads: {num_heads}, Head dim: {head_dim}")

    # Create transformer components
    qkv_proj = te.Linear(hidden_dim, hidden_dim * 3).cuda().bfloat16()
    o_proj = te.Linear(hidden_dim, hidden_dim).cuda().bfloat16()
    ffn1 = te.Linear(hidden_dim, hidden_dim * 4).cuda().bfloat16()
    ffn2 = te.Linear(hidden_dim * 4, hidden_dim).cuda().bfloat16()

    # Input
    x = torch.randn(batch_size * seq_len, hidden_dim, device="cuda").bfloat16()

    # Recipes
    e4m3_recipe = DelayedScaling(fp8_format=Format.E4M3)
    hybrid_recipe = DelayedScaling(fp8_format=Format.HYBRID)

    def transformer_forward(x, recipe=None):
        """Simple transformer forward pass"""
        if recipe:
            with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
                # Attention
                qkv = qkv_proj(x)
                q, k, v = qkv.chunk(3, dim=-1)

                # Reshape for attention
                bs_seq = x.shape[0]
                q = q.view(bs_seq, num_heads, head_dim)
                k = k.view(bs_seq, num_heads, head_dim)
                v = v.view(bs_seq, num_heads, head_dim)

                # Simple attention (no causal mask for benchmarking)
                scores = torch.bmm(q, k.transpose(-2, -1)) / (head_dim**0.5)
                attn = torch.softmax(scores, dim=-1)
                out = torch.bmm(attn, v)
                out = out.view(bs_seq, hidden_dim)

                # Output projection
                out = o_proj(out)

                # FFN
                out = ffn2(torch.relu(ffn1(x + out)))
                return x + out
        else:
            # BF16 version
            qkv = qkv_proj(x)
            q, k, v = qkv.chunk(3, dim=-1)

            bs_seq = x.shape[0]
            q = q.view(bs_seq, num_heads, head_dim)
            k = k.view(bs_seq, num_heads, head_dim)
            v = v.view(bs_seq, num_heads, head_dim)

            scores = torch.bmm(q, k.transpose(-2, -1)) / (head_dim**0.5)
            attn = torch.softmax(scores, dim=-1)
            out = torch.bmm(attn, v)
            out = out.view(bs_seq, hidden_dim)

            out = o_proj(out)
            out = ffn2(torch.relu(ffn1(x + out)))
            return x + out

    # Warmup
    for _ in range(3):
        _ = transformer_forward(x)
    torch.cuda.synchronize()

    # Benchmark
    formats = [("BF16", None), ("E4M3", e4m3_recipe), ("HYBRID", hybrid_recipe)]

    print("\nTransformer Layer Performance:")
    print("-" * 40)

    results = {}
    for name, recipe in formats:
        times = []
        for _ in range(10):
            torch.cuda.synchronize()
            start = time.perf_counter()

            output = transformer_forward(x, recipe)

            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

        avg_time = np.mean(times)
        std_time = np.std(times)
        results[name] = (output, avg_time, std_time)

        print(f"{name:6s}: {avg_time:7.2f} ± {std_time:5.2f} ms")

    # Speedups
    bf16_time = results["BF16"][1]
    print("\nSpeedups vs BF16:")
    for name in ["E4M3", "HYBRID"]:
        speedup = bf16_time / results[name][1]
        print(f"  {name}: {speedup:.2f}x")

    # Accuracy
    print("\nAccuracy vs BF16:")
    bf16_out = results["BF16"][0]
    for name in ["E4M3", "HYBRID"]:
        fp8_out = results[name][0]
        rmse = torch.sqrt(torch.mean((bf16_out - fp8_out) ** 2))
        rel_error = torch.mean(
            torch.abs(bf16_out - fp8_out) / (torch.abs(bf16_out) + 1e-8)
        )
        print(f"  {name}: RMSE={rmse:.6f}, Rel Error={rel_error:.4f}")


def scaling_comparison():
    """Compare different scaling strategies"""

    print("\n" + "=" * 80)
    print("Scaling Strategy Comparison")
    print("=" * 80)

    # Different scaling strategies
    recipes = {
        "Delayed (16)": DelayedScaling(
            fp8_format=Format.HYBRID,
            amax_history_len=16,
            amax_compute_algo="most_recent",
        ),
        "Delayed (32)": DelayedScaling(
            fp8_format=Format.HYBRID,
            amax_history_len=32,
            amax_compute_algo="max",
        ),
        "Delayed (8)": DelayedScaling(
            fp8_format=Format.HYBRID,
            amax_history_len=8,
            amax_compute_algo="most_recent",
        ),
    }

    # Test configuration
    m, n, k = 16384, 5120, 5120
    layer = te.Linear(n, k).cuda().bfloat16()

    print(f"\nMatrix size: {m} × {n} × {k}")
    print("\nRunning multiple iterations to test scaling behavior...")

    for name, recipe in recipes.items():
        print(f"\n{name}:")

        # Run multiple iterations to see scaling adaptation
        times = []
        errors = []

        # First run to establish baseline
        inp = torch.randn(m, n, device="cuda").bfloat16()
        with torch.no_grad():
            bf16_out = layer(inp)

        # Multiple iterations
        for i in range(20):
            # Vary input magnitude slightly
            scale = 1.0 + 0.1 * np.sin(i * 0.5)
            inp = torch.randn(m, n, device="cuda").bfloat16() * scale

            torch.cuda.synchronize()
            start = time.perf_counter()

            with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
                with torch.no_grad():
                    fp8_out = layer(inp)

            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

            # Compare with BF16
            with torch.no_grad():
                bf16_out = layer(inp)
            rel_error = torch.mean(
                torch.abs(fp8_out - bf16_out) / (torch.abs(bf16_out) + 1e-8)
            )
            errors.append(rel_error.item())

        # Statistics
        print(f"  Avg time: {np.mean(times):.2f} ± {np.std(times):.2f} ms")
        print(f"  Avg error: {np.mean(errors):.4f} ± {np.std(errors):.4f}")
        print(f"  Error range: [{np.min(errors):.4f}, {np.max(errors):.4f}]")


def batch_size_scaling():
    """Test how performance scales with batch size"""

    print("\n" + "=" * 80)
    print("Batch Size Scaling Analysis")
    print("=" * 80)

    n, k = 5120, 5120
    layer = te.Linear(n, k).cuda().bfloat16()

    # Recipes
    e4m3_recipe = DelayedScaling(fp8_format=Format.E4M3)
    hybrid_recipe = DelayedScaling(fp8_format=Format.HYBRID)

    batch_sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536]

    print("\nBatch Size Performance Scaling:")
    print("-" * 70)
    print(
        "Batch    BF16 (ms)    E4M3 (ms)    HYBRID (ms)    E4M3 Speedup    HYBRID Speedup"
    )
    print("-" * 70)

    for m in batch_sizes:
        inp = torch.randn(m, n, device="cuda").bfloat16()

        # Benchmark each format
        _, bf16_time, _ = benchmark_format(layer, inp, recipe=None, warmup=3, runs=10)
        _, e4m3_time, _ = benchmark_format(
            layer, inp, recipe=e4m3_recipe, warmup=3, runs=10
        )
        _, hybrid_time, _ = benchmark_format(
            layer, inp, recipe=hybrid_recipe, warmup=3, runs=10
        )

        # Calculate speedups
        e4m3_speedup = bf16_time / e4m3_time
        hybrid_speedup = bf16_time / hybrid_time

        print(
            f"{m:6d}    {bf16_time:8.2f}    {e4m3_time:8.2f}    {hybrid_time:9.2f}    "
            f"{e4m3_speedup:11.2f}x    {hybrid_speedup:13.2f}x"
        )

        # Clean up for next iteration
        del inp
        torch.cuda.empty_cache()

    print(
        "\nNote: Larger batch sizes typically show better FP8 speedups due to better GPU utilization"
    )


if __name__ == "__main__":
    # Set memory fraction
    torch.cuda.set_per_process_memory_fraction(0.95)

    # Device info
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Compute capability: {torch.cuda.get_device_capability()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Run comparisons
    compare_formats()
    transformer_layer_comparison()
    scaling_comparison()
    batch_size_scaling()
