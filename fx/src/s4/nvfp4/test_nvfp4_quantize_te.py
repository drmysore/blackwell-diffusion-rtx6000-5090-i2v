# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
# 
# Adapted from TransformerEngine/tests/pytorch/nvfp4/test_nvfp4_quantize_exact.py
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

def unpack_fp4(x: torch.Tensor) -> torch.Tensor:
    """Unpack FP4 data from uint8 storage"""
    repeated = x.repeat_interleave(2, dim=1)
    repeated[:, 0::2] &= 0x0F
    repeated[:, 1::2] >>= 4
    return repeated

def check_quantization_nvfp4_basic(
    x_dtype: torch.dtype,
    M: int,
    N: int,
) -> None:
    """Basic quantization test adapted from TransformerEngine"""
    # Setup device and random seed
    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # Input tensor
    x = torch.randn((M, N), dtype=x_dtype, device=device)
    
    # Test our S4 NVFP4 implementation
    try:
        # Initialize NVFP4
        nvfp4.init()
        
        # Create FP4 tensor using block 1D quantization
        fp4_tensor = nvfp4.FP4Tensor.from_bfloat16(x, nvfp4.QuantizationMode.block_1d)
        
        # Check properties
        assert fp4_tensor.shape() == (M, N)
        assert fp4_tensor.numel() == M * N
        
        # Check memory usage (should be less than BF16)
        original_memory = x.numel() * 2  # BF16 = 2 bytes per element
        compressed_memory = fp4_tensor.memory_usage()
        assert compressed_memory < original_memory
        
        # Dequantize back
        x_deq = fp4_tensor.to_bfloat16()
        assert x_deq.shape == x.shape
        assert x_deq.dtype == x.dtype
        
        # Check quantization accuracy (FP4 has limited precision)
        rel_error = torch.abs(x_deq - x) / (torch.abs(x) + 1e-8)
        mean_rel_error = rel_error.mean().item()
        
        # FP4 should have reasonable accuracy
        assert mean_rel_error < 0.3  # 30% relative error acceptable for FP4
        
        print(f"✓ Basic quantization test passed: {M}x{N}, mean rel error: {mean_rel_error:.4f}")
        
    except Exception as e:
        print(f"✗ Basic quantization test failed: {e}")
        raise

@pytest.mark.skipif(not HAS_NVFP4, reason="NVFP4 module not available")
@pytest.mark.parametrize(
    "M, N",
    [
        # Small aligned cases
        (128, 128),
        (256, 256),
        # Medium cases
        (512, 512),
        (1024, 1024),
        # Non-square cases
        (256, 512),
        (512, 256),
        # Padding required cases
        (256, 272),
        (304, 304),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.bfloat16, torch.float32], ids=str)
def test_quantization_block_1d_basic(
    x_dtype: torch.dtype,
    M: int,
    N: int,
) -> None:
    """Test basic block 1D quantization functionality"""
    check_quantization_nvfp4_basic(x_dtype=x_dtype, M=M, N=N)

@pytest.mark.skipif(not HAS_NVFP4, reason="NVFP4 module not available")
@pytest.mark.parametrize(
    "M, N",
    [
        (128, 128),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("extrema_high", [False, True], ids=["zeros", "maxes"])
def test_nvfp4_quantization_extrema(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    extrema_high: bool,
):
    """Test quantization with extreme values (adapted from TE)"""
    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if extrema_high:
        x = torch.full((M, N), torch.finfo(x_dtype).max, dtype=x_dtype, device=device)
    else:
        x = torch.zeros((M, N), dtype=x_dtype, device=device)

    try:
        nvfp4.init()
        
        fp4_tensor = nvfp4.FP4Tensor.from_bfloat16(x, nvfp4.QuantizationMode.block_1d)
        x_deq = fp4_tensor.to_bfloat16()
        
        assert x_deq.shape == x.shape
        
        # For extreme values, check that results are finite
        assert torch.isfinite(x_deq).all()
        
        if extrema_high:
            # Large values should be clipped but still large
            assert x_deq.max() > 0.1 * torch.finfo(x_dtype).max
        else:
            # Zero values should remain close to zero
            assert torch.abs(x_deq).max() < 0.01
            
        print(f"✓ Extrema test passed: {'high' if extrema_high else 'zeros'}")
        
    except Exception as e:
        print(f"✗ Extrema test failed: {e}")
        raise

@pytest.mark.skipif(not HAS_NVFP4, reason="NVFP4 module not available")
def test_nvfp4_performance_benchmark():
    """Basic performance benchmark"""
    device = "cuda"
    
    # Test different sizes
    test_configs = [
        (1024, 1024, "1K x 1K"),
        (2048, 2048, "2K x 2K"),
        (4096, 4096, "4K x 4K"),
    ]
    
    try:
        nvfp4.init()
        
        for M, N, desc in test_configs:
            x = torch.randn((M, N), dtype=torch.bfloat16, device=device)
            
            # Warmup
            for _ in range(5):
                fp4_tensor = nvfp4.FP4Tensor.from_bfloat16(x, nvfp4.QuantizationMode.block_1d)
                x_deq = fp4_tensor.to_bfloat16()
            
            torch.cuda.synchronize()
            
            # Benchmark
            import time
            start = time.time()
            iterations = 50
            
            for _ in range(iterations):
                fp4_tensor = nvfp4.FP4Tensor.from_bfloat16(x, nvfp4.QuantizationMode.block_1d)
                x_deq = fp4_tensor.to_bfloat16()
                
            torch.cuda.synchronize()
            end = time.time()
            
            avg_time_ms = (end - start) * 1000 / iterations
            throughput_gb_s = (M * N * 2) / ((end - start) / iterations) / 1e9  # GB/s
            
            print(f"✓ {desc}: {avg_time_ms:.2f} ms/iter, {throughput_gb_s:.1f} GB/s")
            
            # Basic performance expectation
            assert avg_time_ms < 100.0  # Should complete in reasonable time
            
    except Exception as e:
        print(f"✗ Performance benchmark failed: {e}")
        raise

if __name__ == "__main__":
    print("S4 NVFP4 Quantization Tests (adapted from TransformerEngine)")
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
        print("\n--- Basic Quantization Tests ---")
        check_quantization_nvfp4_basic(torch.bfloat16, 128, 128)
        check_quantization_nvfp4_basic(torch.bfloat16, 256, 256)
        check_quantization_nvfp4_basic(torch.bfloat16, 512, 512)
        
        print("\n--- Extreme Value Tests ---")
        test_nvfp4_quantization_extrema(torch.bfloat16, 128, 128, False)
        test_nvfp4_quantization_extrema(torch.bfloat16, 128, 128, True)
        
        print("\n--- Performance Benchmark ---")
        test_nvfp4_performance_benchmark()
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Tests failed: {e}")
        exit(1)