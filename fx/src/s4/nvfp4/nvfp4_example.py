"""
S4 NVFP4 Example: Demonstrating FP4 quantization for Blackwell GPUs
"""

import torch
import nvfp4
import time


def main():
    print("=" * 60)
    print("S4 NVFP4 - Native FP4 Quantization for Blackwell")
    print("=" * 60)

    # Initialize NVFP4
    nvfp4.init()

    # Example 1: Basic FP4 GEMM
    print("\n1. Basic FP4 GEMM Example")
    print("-" * 40)

    M, N, K = 4096, 4096, 4096
    print(f"Matrix dimensions: {M}x{K} @ {N}x{K}")

    # Create random matrices
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    # Warm up
    for _ in range(5):
        _ = nvfp4.blackwell_fp4_gemm(A, B)
    torch.cuda.synchronize()

    # Time FP4 GEMM
    start = time.time()
    C_fp4 = nvfp4.blackwell_fp4_gemm(A, B)
    torch.cuda.synchronize()
    fp4_time = (time.time() - start) * 1000

    # Time BF16 GEMM for comparison
    start = time.time()
    C_bf16 = torch.matmul(A, B.T)
    torch.cuda.synchronize()
    bf16_time = (time.time() - start) * 1000

    print(f"FP4 GEMM time: {fp4_time:.2f} ms")
    print(f"BF16 GEMM time: {bf16_time:.2f} ms")
    print(f"Speedup: {bf16_time / fp4_time:.2f}x")

    # Calculate TFLOPS
    flops = 2 * M * N * K
    fp4_tflops = flops / (fp4_time * 1e9)
    bf16_tflops = flops / (bf16_time * 1e9)
    print(f"FP4 Performance: {fp4_tflops:.1f} TFLOPS")
    print(f"BF16 Performance: {bf16_tflops:.1f} TFLOPS")

    # Example 2: FP4 Tensor Quantization
    print("\n2. FP4 Tensor Quantization Example")
    print("-" * 40)

    # Create a weight matrix
    weight = torch.randn(2048, 2048, dtype=torch.bfloat16, device="cuda")
    print(f"Original weight shape: {weight.shape}")
    print(f"Original weight memory: {weight.numel() * 2 / 1024 / 1024:.2f} MB")

    # Quantize to FP4
    weight_fp4 = nvfp4.FP4Tensor.from_bfloat16(weight, nvfp4.QuantizationMode.block_1d)
    print(f"FP4 weight memory: {weight_fp4.memory_usage() / 1024 / 1024:.2f} MB")
    print(f"Memory reduction: {(1 - weight_fp4.memory_usage() / (weight.numel() * 2)) * 100:.1f}%")

    # Dequantize and check error
    weight_deq = weight_fp4.to_bfloat16()
    rel_error = torch.abs(weight_deq - weight) / (torch.abs(weight) + 1e-8)
    print(f"Quantization error - Mean: {rel_error.mean():.4f}, Max: {rel_error.max():.4f}")

    # Example 3: Different Quantization Modes
    print("\n3. Quantization Mode Comparison")
    print("-" * 40)

    test_tensor = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")

    for mode in [nvfp4.QuantizationMode.block_1d, nvfp4.QuantizationMode.per_tensor]:
        fp4_tensor = nvfp4.FP4Tensor.from_bfloat16(test_tensor, mode)
        deq_tensor = fp4_tensor.to_bfloat16()

        error = torch.abs(deq_tensor - test_tensor).mean()
        print(f"{mode.name}: Memory={fp4_tensor.memory_usage() / 1024:.1f}KB, Error={error:.6f}")

    # Example 4: End-to-end Linear Layer
    print("\n4. Quantized Linear Layer Example")
    print("-" * 40)

    batch_size = 32
    in_features = 4096
    out_features = 11008  # FFN dimension

    # Create input and weight
    x = torch.randn(batch_size, in_features, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(out_features, in_features, dtype=torch.bfloat16, device="cuda")

    # Quantize weight
    w_fp4 = nvfp4.FP4Tensor.from_bfloat16(w, nvfp4.QuantizationMode.block_1d)

    # Forward pass with FP4
    start = time.time()
    y_fp4 = nvfp4.gemm_fp4_quantized(x, w)
    torch.cuda.synchronize()
    fp4_time = (time.time() - start) * 1000

    # Reference BF16
    start = time.time()
    y_ref = torch.matmul(x, w.T)
    torch.cuda.synchronize()
    ref_time = (time.time() - start) * 1000

    print(f"Input shape: {x.shape}")
    print(f"Weight shape: {w.shape}")
    print(f"Output shape: {y_fp4.shape}")
    print(f"FP4 time: {fp4_time:.2f} ms")
    print(f"BF16 time: {ref_time:.2f} ms")
    print(f"Speedup: {ref_time / fp4_time:.2f}x")

    # Memory savings
    weight_bf16_mb = w.numel() * 2 / 1024 / 1024
    weight_fp4_mb = w_fp4.memory_usage() / 1024 / 1024
    print(f"Weight memory - BF16: {weight_bf16_mb:.1f} MB, FP4: {weight_fp4_mb:.1f} MB")
    print(f"Memory savings: {(1 - weight_fp4_mb / weight_bf16_mb) * 100:.1f}%")

    print("\n" + "=" * 60)
    print("S4 NVFP4 - Powering the next generation of AI inference")
    print("=" * 60)


if __name__ == "__main__":
    main()
