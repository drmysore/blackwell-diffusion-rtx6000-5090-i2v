# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
# 
# Adapted from TransformerEngine/tests/pytorch/nvfp4/test_nvfp4_gemm_exact.py
# for S4 NVFP4 implementation

import pytest
import torch
import sys
import os

# Add the bazel output directory to path
sys.path.insert(0, "bazel-bin/src/s4/nvfp4")

try:
    import nvfp4
    HAS_NVFP4 = True
except ImportError:
    HAS_NVFP4 = False

def check_nvfp4_gemm_basic(
    x_dtype: torch.dtype,
    w_dtype: torch.dtype,
    out_dtype: torch.dtype,
    M: int,
    K: int,
    N: int,
    accumulate: bool = False,
):
    """Basic GEMM test adapted from TransformerEngine"""
    # Setup device and random seed
    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Input tensors: A is M x K, B is N x K (to be transposed for B^T)
    A = torch.randn((M, K), dtype=x_dtype, device=device) * 0.1  # Scale down for FP4
    B = torch.randn((N, K), dtype=w_dtype, device=device) * 0.1

    # Setup output tensor if accumulate is True
    if accumulate:
        C = torch.randn((M, N), dtype=out_dtype, device=device)
    else:
        C = None

    try:
        nvfp4.init()
        
        # Quantize A and B to FP4
        A_fp4 = nvfp4.FP4Tensor.from_bfloat16(A, nvfp4.QuantizationMode.block_1d)
        B_fp4 = nvfp4.FP4Tensor.from_bfloat16(B, nvfp4.QuantizationMode.block_1d)
        
        # Perform FP4 GEMM: C = A @ B^T
        if accumulate and C is not None:
            C_fp4 = nvfp4.fp4_gemm(A_fp4, B_fp4, alpha=1.0, beta=1.0, bias=C)
        else:
            C_fp4 = nvfp4.fp4_gemm(A_fp4, B_fp4, alpha=1.0, beta=0.0)
        
        # Reference computation in BF16
        C_ref = torch.matmul(A, B.T)
        if accumulate and C is not None:
            C_ref = C_ref + C
        
        # Check dimensions
        assert C_fp4.shape == C_ref.shape
        assert C_fp4.shape == (M, N)
        
        # Check accuracy (FP4 GEMM has limited precision)
        rel_error = torch.abs(C_fp4 - C_ref) / (torch.abs(C_ref) + 1e-8)
        mean_rel_error = rel_error.mean().item()
        
        # FP4 GEMM should have reasonable accuracy
        assert mean_rel_error < 0.4  # 40% relative error acceptable for FP4 GEMM
        
        print(f"✓ GEMM test passed: {M}x{K}x{N}, mean rel error: {mean_rel_error:.4f}")
        
    except Exception as e:
        print(f"✗ GEMM test failed: {e}")
        raise

@pytest.mark.skipif(not HAS_NVFP4, reason="NVFP4 module not available")
@pytest.mark.parametrize(
    "M, K, N",
    [
        # Small aligned cases
        (128, 128, 128),
        (256, 256, 256),
        # Medium cases
        (512, 512, 512),
        (1024, 1024, 1024),
        # Non-square cases
        (256, 512, 384),
        (384, 256, 512),
        # Larger cases
        (2048, 2048, 2048),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("w_dtype", [torch.bfloat16], ids=str) 
@pytest.mark.parametrize("out_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("accumulate", [False, True], ids=["no_bias", "with_bias"])
def test_nvfp4_gemm_basic(
    x_dtype: torch.dtype,
    w_dtype: torch.dtype,
    out_dtype: torch.dtype,
    M: int,
    K: int,
    N: int,
    accumulate: bool,
) -> None:
    """Test basic FP4 GEMM functionality"""
    check_nvfp4_gemm_basic(
        x_dtype=x_dtype,
        w_dtype=w_dtype,
        out_dtype=out_dtype,
        M=M,
        K=K,
        N=N,
        accumulate=accumulate,
    )

@pytest.mark.skipif(not HAS_NVFP4, reason="NVFP4 module not available")
def test_nvfp4_gemm_performance_benchmark():
    """GEMM performance benchmark adapted from TransformerEngine"""
    device = "cuda"
    
    # Test configurations typical of LLM inference
    test_configs = [
        (1024, 1024, 1024, "1K x 1K x 1K"),
        (2048, 2048, 2048, "2K x 2K x 2K"),
        (4096, 4096, 4096, "4K x 4K x 4K"),
        (8192, 5120, 5120, "8K x 5K x 5K (LLM-like)"),
    ]
    
    try:
        nvfp4.init()
        
        for M, K, N, desc in test_configs:
            print(f"\nBenchmarking: {desc}")
            
            A = torch.randn((M, K), dtype=torch.bfloat16, device=device) * 0.05
            B = torch.randn((N, K), dtype=torch.bfloat16, device=device) * 0.05
            
            # Quantize once
            A_fp4 = nvfp4.FP4Tensor.from_bfloat16(A, nvfp4.QuantizationMode.block_1d)
            B_fp4 = nvfp4.FP4Tensor.from_bfloat16(B, nvfp4.QuantizationMode.block_1d)
            
            # Warmup
            for _ in range(5):
                C_fp4 = nvfp4.fp4_gemm(A_fp4, B_fp4)
                C_ref = torch.matmul(A, B.T)
            
            torch.cuda.synchronize()
            
            # Benchmark FP4 GEMM
            iterations = 10 if M * K * N > 1e9 else 50
            
            import time
            start = time.time()
            for _ in range(iterations):
                C_fp4 = nvfp4.fp4_gemm(A_fp4, B_fp4)
            torch.cuda.synchronize()
            end = time.time()
            
            fp4_time_ms = (end - start) * 1000 / iterations
            
            # Benchmark reference BF16 GEMM
            start = time.time()
            for _ in range(iterations):
                C_ref = torch.matmul(A, B.T)
            torch.cuda.synchronize()
            end = time.time()
            
            ref_time_ms = (end - start) * 1000 / iterations
            
            # Calculate performance metrics
            flops = 2.0 * M * N * K
            fp4_tflops = flops / (fp4_time_ms * 1e-3) / 1e12
            ref_tflops = flops / (ref_time_ms * 1e-3) / 1e12
            speedup = ref_time_ms / fp4_time_ms
            
            print(f"  FP4 GEMM: {fp4_time_ms:.2f} ms, {fp4_tflops:.1f} TFLOPS")
            print(f"  BF16 GEMM: {ref_time_ms:.2f} ms, {ref_tflops:.1f} TFLOPS")
            print(f"  Speedup: {speedup:.2f}x")
            
            # Performance expectations
            if M >= 1024 and N >= 1024 and K >= 1024:
                assert fp4_tflops > 10.0  # Should achieve reasonable performance
            
            # Check accuracy
            rel_error = torch.abs(C_fp4 - C_ref) / (torch.abs(C_ref) + 1e-8)
            mean_rel_error = rel_error.mean().item()
            print(f"  Mean relative error: {mean_rel_error:.4f}")
            
            assert mean_rel_error < 0.5  # Reasonable accuracy
            
    except Exception as e:
        print(f"✗ GEMM benchmark failed: {e}")
        raise

@pytest.mark.skipif(not HAS_NVFP4, reason="NVFP4 module not available")
def test_nvfp4_gemm_numerical_stability():
    """Test GEMM with different value ranges"""
    device = "cuda"
    M, K, N = 256, 256, 256
    
    try:
        nvfp4.init()
        
        # Test different scales
        test_scales = [0.01, 0.1, 1.0, 2.0]
        
        for scale in test_scales:
            print(f"Testing scale: {scale}")
            
            A = torch.randn((M, K), dtype=torch.bfloat16, device=device) * scale
            B = torch.randn((N, K), dtype=torch.bfloat16, device=device) * scale
            
            A_fp4 = nvfp4.FP4Tensor.from_bfloat16(A, nvfp4.QuantizationMode.block_1d)
            B_fp4 = nvfp4.FP4Tensor.from_bfloat16(B, nvfp4.QuantizationMode.block_1d)
            
            C_fp4 = nvfp4.fp4_gemm(A_fp4, B_fp4)
            C_ref = torch.matmul(A, B.T)
            
            # Check for NaNs or infinities
            assert torch.isfinite(C_fp4).all(), f"FP4 GEMM produced non-finite values at scale {scale}"
            
            # Check bounds
            assert torch.abs(C_fp4).max() < 1000.0, f"FP4 GEMM produced extreme values at scale {scale}"
            
            # Check relative error
            if scale >= 0.1:  # For reasonable scales
                rel_error = torch.abs(C_fp4 - C_ref) / (torch.abs(C_ref) + 1e-8)
                mean_rel_error = rel_error.mean().item()
                assert mean_rel_error < 0.6, f"Poor accuracy at scale {scale}: {mean_rel_error}"
        
        print("✓ Numerical stability tests passed")
        
    except Exception as e:
        print(f"✗ Numerical stability test failed: {e}")
        raise

if __name__ == "__main__":
    print("S4 NVFP4 GEMM Tests (adapted from TransformerEngine)")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("✗ No CUDA device available")
        exit(1)
    
    if not HAS_NVFP4:
        print("✗ NVFP4 module not available")
        exit(1)
        
    device_name = torch.cuda.get_device_name(0)
    compute_cap = torch.cuda.get_device_capability(0)
    print(f"Device: {device_name}")
    print(f"Compute capability: {compute_cap[0]}.{compute_cap[1]}")
    
    # Run basic tests
    try:
        print("\n--- Basic GEMM Tests ---")
        check_nvfp4_gemm_basic(torch.bfloat16, torch.bfloat16, torch.bfloat16, 128, 128, 128, False)
        check_nvfp4_gemm_basic(torch.bfloat16, torch.bfloat16, torch.bfloat16, 256, 256, 256, True)
        check_nvfp4_gemm_basic(torch.bfloat16, torch.bfloat16, torch.bfloat16, 512, 512, 512, False)
        
        print("\n--- Numerical Stability Tests ---")
        test_nvfp4_gemm_numerical_stability()
        
        print("\n--- Performance Benchmark ---")
        test_nvfp4_gemm_performance_benchmark()
        
        print("\n✓ All GEMM tests passed!")
        
    except Exception as e:
        print(f"\n✗ GEMM tests failed: {e}")
        exit(1)