import time
import torch

import nvfp4_linear_cpp

# Try to import torchao
try:
    from torchao.dtypes.floatx import (
        FloatxTensorCoreAQTLayout,
        FloatxTensorCoreLayoutType,
    )
    from torchao.dtypes import to_affine_quantized_floatx

    TORCHAO_AVAILABLE = True
except ImportError:
    print("Warning: torchao not available, skipping torchao comparisons")
    TORCHAO_AVAILABLE = False


def round_up_to_multiple(n, multiple):
    """Round up n to the nearest multiple"""
    return ((n + multiple - 1) // multiple) * multiple


def test_mega_performance():
    print("\n" + "=" * 60)
    print("MEGA UPLOAD Blackwell FP4 Performance Comparison")
    print("=" * 60)

    # Your MEGA stack dimensions - ensure M is divisible by 128
    test_configs = [
        (8192, 5120, 5120, "8K×5K×5K"),
        (16384, 5120, 5120, "16K×5K×5K"),
        (65536, 5120, 5120, "64K×5K×5K"),
        (8192, 16384, 122880, "8K×16K×123K (MEGA QKV)"),
    ]

    for M, K, N, desc in test_configs:
        print(f"\n{desc}:")

        # Ensure M is divisible by 128
        M_padded = round_up_to_multiple(M, 128)

        # Create test tensors with controlled magnitude
        input_tensor = torch.randn(M_padded, K, dtype=torch.bfloat16, device="cuda") * 0.1
        weight = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.1
        bias = torch.zeros(N, dtype=torch.bfloat16, device="cuda")

        # Use conservative scales
        input_scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)
        weight_scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)

        # Warmup and benchmark CUTLASS FP4
        for _ in range(10):
            _ = nvfp4_linear_cpp.forward(input_tensor, weight, bias, input_scale, weight_scale)
        torch.cuda.synchronize()

        iterations = 50 if M > 32768 else 100
        start = time.time()
        for _ in range(iterations):
            output_fp4 = nvfp4_linear_cpp.forward(
                input_tensor, weight, bias, input_scale, weight_scale
            )
        torch.cuda.synchronize()
        elapsed_fp4 = (time.time() - start) / iterations

        # For cuBLAS, use original size
        input_original = input_tensor[:M, :] if M != M_padded else input_tensor

        # Warmup and benchmark cuBLAS
        for _ in range(10):
            _ = torch.matmul(input_original, weight.T) + bias
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(iterations):
            output_cublas = torch.matmul(input_original, weight.T) + bias
        torch.cuda.synchronize()
        elapsed_cublas = (time.time() - start) / iterations

        # Calculate performance (use original M for fair comparison)
        flops = 2 * M * N * K
        tflops_fp4 = flops / (elapsed_fp4 * 1e12)
        tflops_cublas = flops / (elapsed_cublas * 1e12)

        print(f"  CUTLASS FP4: {elapsed_fp4 * 1000:.2f} ms, {tflops_fp4:.1f} TFLOPS")
        print(f"  cuBLAS BF16: {elapsed_cublas * 1000:.2f} ms, {tflops_cublas:.1f} TFLOPS")
        print(f"  Speedup: {elapsed_cublas / elapsed_fp4:.2f}x")
        if M != M_padded:
            print(f"  Note: M padded from {M} to {M_padded} for FP4 kernel")


def debug_fp4_accuracy():
    """Debug function to understand FP4 accuracy issues"""
    print("\n" + "=" * 60)
    print("Debugging FP4 Accuracy")
    print("=" * 60)

    # Use M divisible by 128
    M, K, N = 128, 128, 128

    # Create simple test case
    input_tensor = torch.ones(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    weight = torch.ones(N, K, dtype=torch.bfloat16, device="cuda") * 0.1
    bias = torch.zeros(N, dtype=torch.bfloat16, device="cuda")

    # Expected output: 0.1 * 0.1 * 128 = 1.28
    expected = 0.1 * 0.1 * K

    print(f"\nSimple test: all ones * 0.1, expected output: {expected}")

    # Try different scales
    test_scales = [0.1, 0.5, 1.0, 2.0, 4.0]

    for scale_val in test_scales:
        input_scale = torch.tensor([scale_val], device="cuda", dtype=torch.float32)
        weight_scale = torch.tensor([scale_val], device="cuda", dtype=torch.float32)

        output_fp4 = nvfp4_linear_cpp.forward(input_tensor, weight, bias, input_scale, weight_scale)

        output_ref = torch.matmul(input_tensor, weight.T) + bias

        abs_diff = torch.abs(output_fp4 - output_ref).max().item()
        rel_diff = (
            (torch.abs(output_fp4 - output_ref) / (torch.abs(output_ref) + 1e-8)).max().item()
        )

        print(f"\nScale: {scale_val}")
        print(f"  Expected output: {output_ref[0, 0].item():.6f}")
        print(f"  FP4 output: {output_fp4[0, 0].item():.6f}")
        print(f"  Max abs diff: {abs_diff:.6f}")
        print(f"  Max rel diff: {rel_diff:.2e}")

    # Test with random values
    print("\n\nRandom value test:")
    input_rand = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    weight_rand = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.1

    input_scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    weight_scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)

    output_fp4_rand = nvfp4_linear_cpp.forward(
        input_rand, weight_rand, bias, input_scale, weight_scale
    )
    output_ref_rand = torch.matmul(input_rand, weight_rand.T) + bias

    print(f"Input range: [{input_rand.min().item():.4f}, {input_rand.max().item():.4f}]")
    print(f"Weight range: [{weight_rand.min().item():.4f}, {weight_rand.max().item():.4f}]")
    print(
        f"Output FP4 range: [{output_fp4_rand.min().item():.4f}, {output_fp4_rand.max().item():.4f}]"
    )
    print(
        f"Output ref range: [{output_ref_rand.min().item():.4f}, {output_ref_rand.max().item():.4f}]"
    )

    abs_diff = torch.abs(output_fp4_rand - output_ref_rand)
    rel_error = abs_diff / (torch.abs(output_ref_rand) + 1e-8)
    print(f"Max absolute error: {abs_diff.max().item():.6f}")
    print(f"Mean absolute error: {abs_diff.mean().item():.6f}")
    print(f"Max relative error: {rel_error.max().item():.2e}")
    print(f"Mean relative error: {rel_error.mean().item():.2e}")


def test_blackwell_fp4():
    print("Testing native CUTLASS Blackwell FP4 GEMM...")

    M, K, N = 8192, 5120, 5120

    # M is already divisible by 128, but let's verify
    assert M % 128 == 0, f"M ({M}) must be divisible by 128"

    # Use smaller values to avoid overflow
    input_tensor = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    weight = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.1
    bias = torch.zeros(N, dtype=torch.bfloat16, device="cuda")

    # Use conservative scales
    input_scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    weight_scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)

    # Warmup
    for _ in range(10):
        output_fp4 = nvfp4_linear_cpp.forward(input_tensor, weight, bias, input_scale, weight_scale)
        output_cublas = torch.matmul(input_tensor, weight.T) + bias

    torch.cuda.synchronize()

    iterations = 100

    # Benchmark FP4
    start = time.time()
    for _ in range(iterations):
        output_fp4 = nvfp4_linear_cpp.forward(input_tensor, weight, bias, input_scale, weight_scale)
    torch.cuda.synchronize()
    elapsed_fp4 = (time.time() - start) / iterations

    # Benchmark cuBLAS
    start = time.time()
    for _ in range(iterations):
        output_cublas = torch.matmul(input_tensor, weight.T) + bias
    torch.cuda.synchronize()
    elapsed_cublas = (time.time() - start) / iterations

    flops = 2 * M * N * K
    tflops_fp4 = flops / (elapsed_fp4 * 1e12)
    tflops_cublas = flops / (elapsed_cublas * 1e12)

    print(f"Problem size: {M}×{K}×{N}")
    print("\nFP4 Performance:")
    print(f"  Time: {elapsed_fp4 * 1000:.2f} ms")
    print(f"  Performance: {tflops_fp4:.1f} TFLOPS")
    print("\ncuBLAS BF16 Performance:")
    print(f"  Time: {elapsed_cublas * 1000:.2f} ms")
    print(f"  Performance: {tflops_cublas:.1f} TFLOPS")
    print(f"\nSpeedup: {elapsed_cublas / elapsed_fp4:.2f}x")

    # Check accuracy with proper alignment
    with torch.no_grad():
        # Use a smaller aligned sample for accuracy analysis
        sample_size = 1024  # Divisible by 128
        input_sample = input_tensor[:sample_size, :]

        output_fp4_sample = nvfp4_linear_cpp.forward(
            input_sample, weight, bias, input_scale, weight_scale
        )
        output_ref_sample = torch.matmul(input_sample, weight.T) + bias

        abs_diff = torch.abs(output_fp4_sample - output_ref_sample)
        rel_error = abs_diff / (torch.abs(output_ref_sample) + 1e-8)

        print(f"\nAccuracy Analysis (on {sample_size} sample rows):")
        print(f"  Input scale: {input_scale.item():.4f}")
        print(f"  Weight scale: {weight_scale.item():.4f}")
        print(f"  Input range: [{input_tensor.min().item():.4f}, {input_tensor.max().item():.4f}]")
        print(f"  Weight range: [{weight.min().item():.4f}, {weight.max().item():.4f}]")
        print(
            f"  Output FP4 range: [{output_fp4_sample.min().item():.4f}, {output_fp4_sample.max().item():.4f}]"
        )
        print(
            f"  Output ref range: [{output_ref_sample.min().item():.4f}, {output_ref_sample.max().item():.4f}]"
        )
        print(f"  Max absolute difference: {abs_diff.max().item():.6f}")
        print(f"  Mean absolute difference: {abs_diff.mean().item():.6f}")
        print(f"  Max relative error: {rel_error.max().item():.2e}")
        print(f"  Mean relative error: {rel_error.mean().item():.2e}")

        # Compute percentiles
        rel_error_flat = rel_error.flatten()
        print(
            f"  90th percentile relative error: {torch.quantile(rel_error_flat.float(), 0.90).item():.2e}"
        )
        print(
            f"  99th percentile relative error: {torch.quantile(rel_error_flat.float(), 0.99).item():.2e}"
        )


if __name__ == "__main__":
    test_blackwell_fp4()
    debug_fp4_accuracy()  # Add debug function
    test_mega_performance()
