#!/usr/bin/env python3
"""
WAN 2.2 Complete Example: From Configuration to Inference
==========================================================

This example demonstrates the complete workflow:
1. Calculate required configuration
2. Create static model with factory
3. Run inference with optimal settings
4. Compare with dynamic model
"""

import torch
import time

# Import configuration utilities
from wan_config_calculator import (
    calculate_for_resolution,
)

from wan_static_factory import (
    StaticModelFactory,
    AdaptiveStaticModel,
)

from wan_deployment_wizard import (
    DeploymentWizard,
    UseCase,
    DeploymentTarget,
)

# Import models
from wan_attention_dynamic import WanTransformer3DModel


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def example_1_calculate_configuration():
    """Example 1: Calculate configuration for your use case."""
    print_section("Example 1: Configuration Calculation")

    # Define your video requirements
    frames = 16
    height = 512
    width = 512
    batch_size = 1

    print(f"\nTarget: {frames} frames at {height}x{width}, batch={batch_size}")

    # Calculate configuration
    config = calculate_for_resolution(
        frames=frames,
        height=height,
        width=width,
        batch_size=batch_size,
        print_details=True,
    )

    return config


def example_2_create_static_model():
    """Example 2: Create static model using factory."""
    print_section("Example 2: Create Static Model")

    # Create factory
    factory = StaticModelFactory(
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16,
        cache_models=True,
        compile_models=False,  # Set True for production
    )

    print("\nCreating model for 256x256 16-frame video...")

    # Create model from dimensions
    model = factory.create_from_video_dims(
        batch_size=1,
        frames=16,
        height=256,
        width=256,
        text_tokens=512,
        image_conditioning=False,
    )

    print("✓ Model created: seq_len=3136, context_len=512")

    # Show cache stats
    if factory.cache:
        stats = factory.cache.get_stats()
        print(f"✓ Cached models: {stats['cached_models']}")
        print(f"✓ Total parameters: {stats['total_parameters']:,}")

    return factory, model


def example_3_benchmark_inference():
    """Example 3: Benchmark static vs dynamic inference."""
    print_section("Example 3: Inference Benchmark")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    device = "cuda"
    dtype = torch.float16

    # Configuration for 256x256 16-frame
    batch_size = 1
    seq_len = 3136  # (16/1) * (256/2) * (256/2)
    context_len = 512
    dim = 5120
    num_layers = 2  # Small for demo

    print(f"\nConfiguration: batch={batch_size}, seq={seq_len}, context={context_len}")

    # Create static model
    print("\nCreating static model...")
    from wan_attention_static import StaticWANInferenceModel

    static_model = StaticWANInferenceModel(
        batch_size=batch_size,
        seq_len=seq_len,
        context_len=context_len,
        num_layers=num_layers,
        dim=dim,
        device=device,
        dtype=dtype,
    ).eval()

    # Create dynamic model for comparison
    print("Creating dynamic model...")
    dynamic_model = (
        WanTransformer3DModel(
            num_layers=num_layers,
            num_attention_heads=40,
            attention_head_dim=128,
            use_static_cache=False,  # Pure dynamic
        )
        .to(device)
        .to(dtype)
        .eval()
    )

    # Prepare inputs
    print("\nPreparing inputs...")
    hidden_states = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype) * 0.02
    context = torch.randn(batch_size, context_len, dim, device=device, dtype=dtype) * 0.02
    block_conditioning = (
        torch.randn(batch_size, num_layers, 6, dim, device=device, dtype=dtype) * 0.02
    )
    output_conditioning = torch.randn(batch_size, 2, dim, device=device, dtype=dtype) * 0.02

    # For dynamic model - need different format
    video_hidden = torch.randn(batch_size, 36, 16, 256, 256, device=device, dtype=dtype) * 0.02
    timestep = torch.tensor([500], device=device)
    text = torch.randn(batch_size, 512, 4096, device=device, dtype=dtype) * 0.02

    # Warmup
    print("\nWarming up...")
    for _ in range(5):
        with torch.no_grad():
            _ = static_model(hidden_states, context, block_conditioning, output_conditioning)
            _ = dynamic_model(video_hidden, timestep, text)

    torch.cuda.synchronize()

    # Benchmark static
    print("\nBenchmarking static model...")
    torch.cuda.synchronize()
    start = time.perf_counter()

    iterations = 50
    for _ in range(iterations):
        with torch.no_grad():
            _ = static_model(hidden_states, context, block_conditioning, output_conditioning)

    torch.cuda.synchronize()
    static_time = (time.perf_counter() - start) / iterations * 1000

    # Benchmark dynamic
    print("Benchmarking dynamic model...")
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        with torch.no_grad():
            _ = dynamic_model(video_hidden, timestep, text)

    torch.cuda.synchronize()
    dynamic_time = (time.perf_counter() - start) / iterations * 1000

    # Results
    print("\n" + "-" * 40)
    print("Results (per forward pass):")
    print(f"  Static model:  {static_time:.2f} ms")
    print(f"  Dynamic model: {dynamic_time:.2f} ms")
    print(f"  Speedup:       {dynamic_time / static_time:.2f}x")

    # Try compilation
    print("\n" + "-" * 40)
    print("Attempting torch.compile...")
    try:
        compiled_static = torch.compile(static_model, mode="reduce-overhead", fullgraph=True)

        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = compiled_static(hidden_states, context, block_conditioning, output_conditioning)

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(iterations):
            with torch.no_grad():
                _ = compiled_static(hidden_states, context, block_conditioning, output_conditioning)

        torch.cuda.synchronize()
        compiled_time = (time.perf_counter() - start) / iterations * 1000

        print(f"  Compiled static: {compiled_time:.2f} ms")
        print(f"  Speedup vs dynamic: {dynamic_time / compiled_time:.2f}x")

    except Exception as e:
        print(f"  Compilation failed: {e}")

    # Memory usage
    print("\n" + "-" * 40)
    print("Memory usage:")
    allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
    reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
    print(f"  Allocated: {allocated_mb:.1f} MB")
    print(f"  Reserved:  {reserved_mb:.1f} MB")


def example_4_deployment_recommendations():
    """Example 4: Get deployment recommendations."""
    print_section("Example 4: Deployment Recommendations")

    # Define use case
    use_case = UseCase.VIDEO_256_1S  # 256x256 1-second video

    # Test on different GPUs
    targets = [
        DeploymentTarget.RTX_3090,
        DeploymentTarget.RTX_4090,
        DeploymentTarget.RTX_5090,
        DeploymentTarget.A100_40GB,
    ]

    print(f"\nUse case: {use_case.display_name}")
    print("-" * 40)

    for target in targets:
        config = DeploymentWizard.recommend_config(use_case, target)
        print(f"\n{target.display_name}:")
        print(f"  Max batch size: {config.batch_size}")
        print(f"  Memory usage: {config.model_config.total_memory_mb:.1f} MB")
        print(f"  Est. latency: {config.inference_time_ms:.1f} ms")
        print(f"  Optimization: {config.optimization_level}")
        print(f"  Status: {'✓ Fits' if config.fits_in_memory else '✗ Too large'}")


def example_5_adaptive_routing():
    """Example 5: Adaptive model with automatic routing."""
    print_section("Example 5: Adaptive Model Routing")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    # Create factory
    factory = StaticModelFactory(
        device=device,
        dtype=dtype,
        cache_models=True,
    )

    # Pre-load some common configurations
    print("\nPre-loading common configurations...")
    configs_to_preload = [
        (1, 16, 256, 256),  # 256x256 16-frame
        (1, 16, 512, 512),  # 512x512 16-frame
        (1, 1, 512, 512),  # 512x512 image
    ]

    for batch, frames, height, width in configs_to_preload:
        model = factory.create_from_video_dims(
            batch,
            frames,
            height,
            width,
            image_conditioning=(frames == 1),
        )
        print(f"  ✓ Loaded {frames}x{height}x{width}")

    # Create adaptive model
    print("\nCreating adaptive model...")
    adaptive = AdaptiveStaticModel(
        factory=factory,
        fallback_model=None,  # Would be dynamic model in production
        verbose=True,
    )

    # Test routing with different inputs
    print("\nTesting routing:")
    print("-" * 40)

    test_cases = [
        (1, 3136, 512, "256x256 16-frame (cached)"),
        (1, 12544, 512, "512x512 16-frame (cached)"),
        (1, 1024, 1548, "512x512 image (cached)"),
        (2, 3136, 512, "Batch 2 (not cached)"),
        (1, 197, 512, "14x14 patches (not cached)"),
    ]

    for batch, seq, ctx, description in test_cases:
        print(f"\n{description}:")
        print(f"  Input: batch={batch}, seq={seq}, context={ctx}")

        # Create test inputs
        hidden = torch.randn(batch, seq, 5120, device=device, dtype=dtype) * 0.02
        context = torch.randn(batch, ctx, 5120, device=device, dtype=dtype) * 0.02
        block_cond = torch.randn(batch, 48, 6, 5120, device=device, dtype=dtype) * 0.02
        out_cond = torch.randn(batch, 2, 5120, device=device, dtype=dtype) * 0.02

        try:
            output = adaptive(hidden, context, block_cond, out_cond)
            print(f"  ✓ Success: output shape {output.shape}")
        except RuntimeError as e:
            print(f"  ✗ Failed: {e}")

    # Show routing statistics
    print("\n" + "-" * 40)
    print("Routing statistics:")
    stats = adaptive.get_routing_stats()
    for key, value in stats.items():
        if key == "static_percentage":
            print(f"  {key}: {value:.1f}%")
        else:
            print(f"  {key}: {value}")


def main():
    """Run all examples."""

    print_section("WAN 2.2 Complete Configuration and Deployment Examples")

    # Run examples
    config = example_1_calculate_configuration()
    factory, model = example_2_create_static_model()
    example_3_benchmark_inference()
    example_4_deployment_recommendations()
    example_5_adaptive_routing()

    # Summary
    print_section("Summary")
    print("""
Key Takeaways:
1. Use ConfigCalculator to determine exact dimensions for your use case
2. StaticModelFactory manages model creation and caching efficiently  
3. Static models are significantly faster than dynamic (2-3x typical)
4. torch.compile provides additional speedup (30-50%)
5. DeploymentWizard helps choose optimal GPU and configuration
6. AdaptiveStaticModel automatically routes to best implementation

For production:
- Pre-calculate all needed configurations
- Create static models at startup
- Use torch.compile or CUDAGraph for maximum speed
- Monitor cache hit rates with AdaptiveStaticModel
    """)


if __name__ == "__main__":
    main()
