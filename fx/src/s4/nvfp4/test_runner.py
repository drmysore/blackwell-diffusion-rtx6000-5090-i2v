#!/usr/bin/env python3
"""
Quick test runner for S4 NVFP4
"""

import sys
import torch

# Add the bazel output directory to path
sys.path.insert(0, "bazel-bin/src/s4/nvfp4")

try:
    import nvfp4

    print("✓ Successfully imported nvfp4 module")
except ImportError as e:
    print(f"✗ Failed to import nvfp4: {e}")
    sys.exit(1)


def test_basic_functionality():
    """Test basic NVFP4 functionality"""
    print("\n=== Testing S4 NVFP4 ===")

    # Initialize
    try:
        nvfp4.init()
        print("✓ NVFP4 initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return False

    # Test basic GEMM
    print("\n--- Testing Basic GEMM ---")
    M, N, K = 1024, 1024, 1024  # Aligned dimensions

    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    try:
        # Run FP4 GEMM
        C = nvfp4.blackwell_fp4_gemm(A, B)
        print(f"✓ FP4 GEMM successful: output shape {C.shape}")

        # Check against reference
        C_ref = torch.matmul(A, B.T)
        error = torch.abs(C - C_ref).mean().item()
        print(f"  Mean absolute error vs BF16: {error:.6f}")

    except Exception as e:
        print(f"✗ FP4 GEMM failed: {e}")
        return False

    # Test quantization
    print("\n--- Testing Quantization ---")
    try:
        # Create FP4 tensor
        tensor_fp4 = nvfp4.FP4Tensor.from_bfloat16(A, nvfp4.QuantizationMode.block_1d)
        print("✓ Quantization successful")
        print(f"  Original memory: {A.numel() * 2 / 1024:.1f} KB")
        print(f"  FP4 memory: {tensor_fp4.memory_usage() / 1024:.1f} KB")
        print(f"  Compression ratio: {A.numel() * 2 / tensor_fp4.memory_usage():.1f}x")

        # Test dequantization
        A_deq = tensor_fp4.to_bfloat16()
        quant_error = torch.abs(A - A_deq).mean().item()
        print(f"  Quantization error: {quant_error:.6f}")

    except Exception as e:
        print(f"✗ Quantization failed: {e}")
        return False

    # Benchmark
    print("\n--- Performance Benchmark ---")
    try:
        avg_ms, tflops = nvfp4.benchmark(A, B, iterations=20)
        print("✓ Benchmark complete:")
        print(f"  Average latency: {avg_ms:.2f} ms")
        print(f"  Performance: {tflops:.1f} TFLOPS")

    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        return False

    return True


def test_alignment_requirements():
    """Test dimension alignment requirements"""
    print("\n--- Testing Alignment Requirements ---")

    # Test aligned check
    assert nvfp4.is_aligned(1024, 1024, 1024) == True
    assert nvfp4.is_aligned(1000, 1000, 1000) == False
    print("✓ Alignment checks working")

    # Test padding calculation
    M_pad, N_pad, K_pad = nvfp4.get_aligned_dims(1000, 1000, 1000)
    assert (M_pad, N_pad, K_pad) == (1024, 1024, 1024)
    print("✓ Padding calculation working")


def main():
    """Run all tests"""
    print("S4 NVFP4 Test Runner")
    print("=" * 50)

    # Check CUDA device
    if not torch.cuda.is_available():
        print("✗ No CUDA device available")
        return 1

    device_name = torch.cuda.get_device_name(0)
    compute_cap = torch.cuda.get_device_capability(0)
    print(f"Device: {device_name}")
    print(f"Compute capability: {compute_cap[0]}.{compute_cap[1]}")

    if compute_cap[0] < 8:  # For testing on older GPUs
        print("Warning: This GPU doesn't support FP4 natively")
        print("   The kernels may fall back to emulation mode")

    # Run tests
    success = test_basic_functionality()
    if success:
        test_alignment_requirements()
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
