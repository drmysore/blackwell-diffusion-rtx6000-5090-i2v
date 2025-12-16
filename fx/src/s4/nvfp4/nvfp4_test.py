import torch
import nvfp4


class TestNVFP4:
    """Test suite for S4 NVFP4 module"""

    def setup_method(self):
        """Initialize test environment"""
        nvfp4.init()
        self.device = torch.device("cuda")

    def test_basic_gemm(self):
        """Test basic FP4 GEMM functionality"""
        # Create aligned tensors
        M, N, K = 1024, 1024, 1024
        A = torch.randn(M, K, dtype=torch.bfloat16, device=self.device)
        B = torch.randn(N, K, dtype=torch.bfloat16, device=self.device)

        # Run FP4 GEMM
        C_fp4 = nvfp4.blackwell_fp4_gemm(A, B)

        # Compare with reference
        C_ref = torch.matmul(A, B.T)

        # Check output shape
        assert C_fp4.shape == C_ref.shape

        # Check numerical accuracy (relaxed for FP4)
        rel_error = torch.abs(C_fp4 - C_ref) / (torch.abs(C_ref) + 1e-8)
        assert rel_error.mean() < 0.1  # 10% average error acceptable for FP4

    def test_tensor_quantization(self):
        """Test FP4Tensor quantization and dequantization"""
        # Create test tensor
        shape = (512, 768)
        tensor = torch.randn(shape, dtype=torch.bfloat16, device=self.device)

        # Test different quantization modes
        for mode in [
            nvfp4.QuantizationMode.block_1d,
            nvfp4.QuantizationMode.per_tensor,
        ]:
            # Quantize
            fp4_tensor = nvfp4.FP4Tensor.from_bfloat16(tensor, mode)

            # Check properties
            assert fp4_tensor.shape() == shape
            assert fp4_tensor.numel() == tensor.numel()
            assert fp4_tensor.memory_usage() < tensor.numel() * 2  # Less than BF16

            # Dequantize
            tensor_deq = fp4_tensor.to_bfloat16()
            assert tensor_deq.shape == tensor.shape

    def test_performance(self):
        """Test performance meets S4 targets"""
        # Test at different scales
        test_configs = [
            (1024, 1024, 1024),
            (4096, 4096, 4096),
            (8192, 5120, 5120),  # MEGA-style dimensions
        ]

        for M, N, K in test_configs:
            A = torch.randn(M, K, dtype=torch.bfloat16, device=self.device)
            B = torch.randn(N, K, dtype=torch.bfloat16, device=self.device)

            # Benchmark
            avg_time_ms, tflops = nvfp4.benchmark(A, B, iterations=50)

            print(f"\n{M}x{K}x{N}: {avg_time_ms:.2f}ms, {tflops:.1f} TFLOPS")

            # Verify performance targets (relaxed for initial implementation)
            if M == 8192 and N == 5120 and K == 5120:
                assert tflops > 500  # Should achieve >500 TFLOPS on SM_120

    def test_alignment_requirements(self):
        """Test dimension alignment requirements"""
        # Test unaligned dimensions
        M, N, K = 1000, 1000, 1000  # Not divisible by 128

        assert not nvfp4.is_aligned(M, N, K)

        # Get aligned dimensions
        M_aligned, N_aligned, K_aligned = nvfp4.get_aligned_dims(M, N, K)
        assert M_aligned == 1024
        assert N_aligned == 1024
        assert K_aligned == 1024
        assert nvfp4.is_aligned(M_aligned, N_aligned, K_aligned)

    def test_quantization_accuracy(self):
        """Test quantization maintains acceptable accuracy"""
        # Create test data with known properties
        M, N, K = 512, 512, 512

        # Test with different value ranges
        for scale in [0.1, 1.0, 5.0]:
            A = torch.randn(M, K, dtype=torch.bfloat16, device=self.device) * scale
            B = torch.randn(N, K, dtype=torch.bfloat16, device=self.device) * scale

            # Quantize and compute
            A_fp4 = nvfp4.FP4Tensor.from_bfloat16(A)
            B_fp4 = nvfp4.FP4Tensor.from_bfloat16(B)
            C_fp4 = nvfp4.fp4_gemm(A_fp4, B_fp4)

            # Reference computation
            C_ref = torch.matmul(A, B.T)

            # Check relative error
            rel_error = torch.abs(C_fp4 - C_ref) / (torch.abs(C_ref) + 1e-8)

            # Error should be reasonable for FP4
            assert rel_error.mean() < 0.2  # 20% average error
            assert rel_error.max() < 1.0  # No catastrophic errors


if __name__ == "__main__":
    test = TestNVFP4()
    test.setup_method()
    test.test_basic_gemm()
    test.test_tensor_quantization()
    test.test_performance()
    test.test_alignment_requirements()
    test.test_quantization_accuracy()
    print("\nAll tests passed!")
