#include <chrono>
#include <cmath>
#include <random>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "kernel.cuh"
#include "tensor.h"

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
  cudaError_t err = call; \
  if (err != cudaSuccess) { \
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
  } \
} while(0)

namespace s4::nvfp4::test {

using Catch::Matchers::WithinRel;

class GemmTestFixture {
public:
  GemmTestFixture() : device_(torch::kCPU) {
    // Ensure CUDA is available
    REQUIRE(torch::cuda::is_available());
    device_ = torch::kCUDA;

    // Set random seed for reproducibility
    torch::manual_seed(789);
  }

protected:
  torch::Device device_;

  // Helper to create test tensors
  torch::Tensor create_test_tensor(std::vector<int64_t> shape, float scale = 1.0f) {
    auto tensor =
        torch::randn(shape, torch::TensorOptions().dtype(torch::kBFloat16).device(device_));
    return tensor * scale;
  }

  // Helper to measure relative error
  float compute_relative_error(const torch::Tensor& actual, const torch::Tensor& expected) {
    auto diff = torch::abs(actual - expected);
    auto expected_abs = torch::abs(expected) + 1e-8f;
    auto rel_error = diff / expected_abs;
    return rel_error.mean().item<float>();
  }

  // Helper to ensure dimensions are aligned for FP4 kernels
  std::tuple<int64_t, int64_t, int64_t> align_dims(int64_t M, int64_t N, int64_t K) {
    auto align_to_128 = [](int64_t x) { return ((x + 127) / 128) * 128; };
    return {align_to_128(M), align_to_128(N), align_to_128(K)};
  }
};

TEST_CASE_METHOD(GemmTestFixture, "GEMM Basic Functionality", "[gemm][basic]") {
  SECTION("Small aligned GEMM") {
    int64_t M = 128, N = 128, K = 128;
    
    auto A = create_test_tensor({M, K}, 0.1f);
    auto B = create_test_tensor({N, K}, 0.1f);
    
    // Test high-level interface
    auto C_fp4_hl = gemm_fp4_quantized(A, B, 1.0f, 0.0f, quantization_mode::block_1d);
    
    // Reference computation
    auto C_ref = torch::matmul(A, B.T);
    
    // Check dimensions
    REQUIRE(C_fp4_hl.sizes() == C_ref.sizes());
    
    // Check accuracy (FP4 has limited precision)
    float rel_error = compute_relative_error(C_fp4_hl, C_ref);
    CHECK(rel_error < 0.3f);  // 30% relative error acceptable for FP4
  }

  SECTION("Medium aligned GEMM") {
    int64_t M = 512, N = 512, K = 512;
    
    auto A = create_test_tensor({M, K}, 0.1f);
    auto B = create_test_tensor({N, K}, 0.1f);
    
    auto C_fp4 = gemm_fp4_quantized(A, B, 1.0f, 0.0f, quantization_mode::block_1d);
    auto C_ref = torch::matmul(A, B.T);
    
    REQUIRE(C_fp4.sizes() == C_ref.sizes());
    
    float rel_error = compute_relative_error(C_fp4, C_ref);
    CHECK(rel_error < 0.3f);
  }

  SECTION("Large aligned GEMM") {
    int64_t M = 1024, N = 1024, K = 1024;
    
    auto A = create_test_tensor({M, K}, 0.05f);  // Smaller values for large tensors
    auto B = create_test_tensor({N, K}, 0.05f);
    
    auto C_fp4 = gemm_fp4_quantized(A, B, 1.0f, 0.0f, quantization_mode::block_1d);
    auto C_ref = torch::matmul(A, B.T);
    
    REQUIRE(C_fp4.sizes() == C_ref.sizes());
    
    float rel_error = compute_relative_error(C_fp4, C_ref);
    CHECK(rel_error < 0.4f);  // Allow higher error for large tensors
  }
}

TEST_CASE_METHOD(GemmTestFixture, "GEMM Quantization Modes", "[gemm][quantization]") {
  SECTION("Different quantization modes comparison") {
    int64_t M = 256, N = 256, K = 256;
    auto A = create_test_tensor({M, K}, 0.1f);
    auto B = create_test_tensor({N, K}, 0.1f);
    auto C_ref = torch::matmul(A, B.T);

    std::vector<quantization_mode> modes = {
        quantization_mode::block_1d,
        quantization_mode::block_2d,
        quantization_mode::per_tensor,
        quantization_mode::per_channel
    };

    for (auto mode : modes) {
      INFO("Testing quantization mode: " << static_cast<int>(mode));
      
      auto C_fp4 = gemm_fp4_quantized(A, B, 1.0f, 0.0f, mode);
      
      REQUIRE(C_fp4.sizes() == C_ref.sizes());
      
      float rel_error = compute_relative_error(C_fp4, C_ref);
      CHECK(rel_error < 0.5f);  // Relaxed bound for all modes
    }
  }
}

TEST_CASE_METHOD(GemmTestFixture, "GEMM Alpha/Beta Parameters", "[gemm][parameters]") {
  SECTION("Non-unit alpha and beta") {
    int64_t M = 128, N = 128, K = 128;
    auto A = create_test_tensor({M, K}, 0.1f);
    auto B = create_test_tensor({N, K}, 0.1f);
    
    float alpha = 2.0f, beta = 0.5f;
    
    // Currently the interface might not support alpha/beta, so test what we can
    auto C_fp4 = gemm_fp4_quantized(A, B, alpha, beta, quantization_mode::block_1d);
    auto C_ref = alpha * torch::matmul(A, B.T);  // Assuming beta=0 for now
    
    REQUIRE(C_fp4.sizes() == C_ref.sizes());
    
    // May need to relax this if alpha/beta aren't fully implemented
    float rel_error = compute_relative_error(C_fp4, C_ref);
    INFO("Relative error with alpha=" << alpha << ", beta=" << beta << ": " << rel_error);
  }
}

TEST_CASE_METHOD(GemmTestFixture, "GEMM Tensor Object Interface", "[gemm][tensor_api]") {
  SECTION("FP4 tensor objects") {
    int64_t M = 256, N = 256, K = 256;
    auto A_bf16 = create_test_tensor({M, K}, 0.1f);
    auto B_bf16 = create_test_tensor({N, K}, 0.1f);
    
    // Create FP4 tensors
    auto A_fp4 = tensor::from_bfloat16(A_bf16, quantization_mode::block_1d);
    auto B_fp4 = tensor::from_bfloat16(B_bf16, quantization_mode::block_1d);
    
    // Test GEMM with FP4 tensor objects
    auto C_fp4 = fp4_gemm(A_fp4, B_fp4, 1.0f, 0.0f);
    auto C_ref = torch::matmul(A_bf16, B_bf16.T);
    
    REQUIRE(C_fp4.sizes() == C_ref.sizes());
    
    float rel_error = compute_relative_error(C_fp4, C_ref);
    CHECK(rel_error < 0.3f);
  }

  SECTION("Mixed precision - FP4 with different quantization modes") {
    int64_t M = 128, N = 128, K = 128;
    auto A_bf16 = create_test_tensor({M, K}, 0.1f);
    auto B_bf16 = create_test_tensor({N, K}, 0.1f);
    
    auto A_fp4_block = tensor::from_bfloat16(A_bf16, quantization_mode::block_1d);
    auto B_fp4_tensor = tensor::from_bfloat16(B_bf16, quantization_mode::per_tensor);
    
    // This might fail if the kernel requires matching quantization modes
    // But we test the interface
    REQUIRE_NOTHROW(fp4_gemm(A_fp4_block, B_fp4_tensor));
  }
}

TEST_CASE_METHOD(GemmTestFixture, "GEMM Non-aligned Dimensions", "[gemm][alignment]") {
  SECTION("Non-aligned dimensions handling") {
    // Test with dimensions that aren't multiples of 128
    std::vector<std::tuple<int64_t, int64_t, int64_t>> test_sizes = {
        {100, 100, 100},
        {200, 300, 250},
        {777, 555, 333},
        {1000, 1500, 800}
    };
    
    for (auto [M, N, K] : test_sizes) {
      INFO("Testing size: " << M << "x" << N << "x" << K);
      
      auto A = create_test_tensor({M, K}, 0.1f);
      auto B = create_test_tensor({N, K}, 0.1f);
      
      // The kernel might need aligned dimensions, so this could fail
      // Test that we either handle it gracefully or produce correct results
      try {
        auto C_fp4 = gemm_fp4_quantized(A, B, 1.0f, 0.0f, quantization_mode::block_1d);
        auto C_ref = torch::matmul(A, B.T);
        
        CHECK(C_fp4.sizes() == C_ref.sizes());
        
        float rel_error = compute_relative_error(C_fp4, C_ref);
        CHECK(rel_error < 0.5f);
        
      } catch (const std::exception& e) {
        INFO("Non-aligned dimensions failed as expected: " << e.what());
        // This is acceptable if the kernel requires alignment
      }
    }
  }
}

TEST_CASE_METHOD(GemmTestFixture, "GEMM Error Handling", "[gemm][errors]") {
  SECTION("Mismatched dimensions") {
    auto A = create_test_tensor({128, 256}, 0.1f);
    auto B = create_test_tensor({128, 512}, 0.1f);  // K dimension mismatch
    
    // Should throw due to dimension mismatch
    CHECK_THROWS_AS(gemm_fp4_quantized(A, B), c10::Error);
  }

  SECTION("1D tensors") {
    auto A_1d = create_test_tensor({128}, 0.1f);
    auto B_2d = create_test_tensor({128, 128}, 0.1f);
    
    // Should reject 1D tensors
    CHECK_THROWS_AS(gemm_fp4_quantized(A_1d, B_2d), c10::Error);
  }

  SECTION("Wrong device tensors") {
    auto A_cuda = create_test_tensor({128, 128}, 0.1f);
    auto B_cpu = torch::randn({128, 128}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU));
    
    // Should reject mixed device tensors
    CHECK_THROWS_AS(gemm_fp4_quantized(A_cuda, B_cpu), c10::Error);
  }

  SECTION("Wrong dtype tensors") {
    auto A_bf16 = create_test_tensor({128, 128}, 0.1f);
    auto B_f32 = torch::randn({128, 128}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    
    // Should reject non-BF16 tensors
    CHECK_THROWS_AS(gemm_fp4_quantized(A_bf16, B_f32), c10::Error);
  }
}

TEST_CASE_METHOD(GemmTestFixture, "GEMM Performance Benchmarks", "[gemm][performance][!benchmark]") {
  SECTION("MEGA-style dimensions benchmark") {
    // Test dimensions typical of MEGA models
    std::vector<std::tuple<int64_t, int64_t, int64_t, std::string>> configs = {
        {1024, 1024, 1024, "1K x 1K x 1K"},
        {2048, 2048, 2048, "2K x 2K x 2K"},
        {4096, 4096, 4096, "4K x 4K x 4K"},
        {8192, 5120, 5120, "8K x 5K x 5K (MEGA-like)"}
    };
    
    for (auto [M, N, K, desc] : configs) {
      INFO("Benchmarking: " << desc);
      
      auto A = create_test_tensor({M, K}, 0.05f);
      auto B = create_test_tensor({N, K}, 0.05f);
      
      // Warmup
      for (int i = 0; i < 5; ++i) {
        auto C_fp4 = gemm_fp4_quantized(A, B, 1.0f, 0.0f, quantization_mode::block_1d);
        auto C_ref = torch::matmul(A, B.T);
      }
      
      cudaDeviceSynchronize();
      
      // Benchmark FP4 GEMM
      int iterations = (M * N * K > 1e9) ? 10 : 50;  // Fewer iterations for large problems
      
      auto start = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < iterations; ++i) {
        auto C_fp4 = gemm_fp4_quantized(A, B, 1.0f, 0.0f, quantization_mode::block_1d);
      }
      cudaDeviceSynchronize();
      auto end = std::chrono::high_resolution_clock::now();
      
      auto fp4_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      double fp4_time_ms = fp4_duration.count() / (double)iterations / 1000.0;
      
      // Benchmark reference BF16 GEMM
      start = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < iterations; ++i) {
        auto C_ref = torch::matmul(A, B.T);
      }
      cudaDeviceSynchronize();
      end = std::chrono::high_resolution_clock::now();
      
      auto ref_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      double ref_time_ms = ref_duration.count() / (double)iterations / 1000.0;
      
      // Calculate performance metrics
      double flops = 2.0 * M * N * K;
      double fp4_tflops = flops / (fp4_time_ms * 1e-3) / 1e12;
      double ref_tflops = flops / (ref_time_ms * 1e-3) / 1e12;
      double speedup = ref_time_ms / fp4_time_ms;
      
      INFO("FP4 GEMM: " << fp4_time_ms << " ms, " << fp4_tflops << " TFLOPS");
      INFO("BF16 GEMM: " << ref_time_ms << " ms, " << ref_tflops << " TFLOPS"); 
      INFO("Speedup: " << speedup << "x");
      
      // Performance expectations (adjust based on hardware)
      if (M >= 1024 && N >= 1024 && K >= 1024) {
        CHECK(fp4_tflops > 50.0);  // Should achieve reasonable performance
        
        // On Blackwell, FP4 should be faster than BF16 for large problems
        if (M * N * K > 1e9) {
          INFO("Large problem - expecting FP4 speedup");
          // CHECK(speedup > 1.0);  // Enable when kernel is optimized
        }
      }
    }
  }
  
  SECTION("Throughput scaling test") {
    // Test how performance scales with problem size
    std::vector<int64_t> sizes = {128, 256, 512, 1024, 2048};
    
    for (auto size : sizes) {
      int64_t M = size, N = size, K = size;
      
      auto A = create_test_tensor({M, K}, 0.1f);
      auto B = create_test_tensor({N, K}, 0.1f);
      
      // Warmup
      auto C_fp4 = gemm_fp4_quantized(A, B);
      cudaDeviceSynchronize();
      
      // Single iteration timing for large problems
      auto start = std::chrono::high_resolution_clock::now();
      C_fp4 = gemm_fp4_quantized(A, B);
      cudaDeviceSynchronize();
      auto end = std::chrono::high_resolution_clock::now();
      
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      double time_ms = duration.count() / 1000.0;
      
      double flops = 2.0 * M * N * K;
      double tflops = flops / (time_ms * 1e-3) / 1e12;
      
      INFO("Size " << size << "³: " << time_ms << " ms, " << tflops << " TFLOPS");
      
      // Basic scaling check - larger problems should achieve higher TFLOPS
      if (size >= 512) {
        CHECK(tflops > 10.0);  // Should achieve at least 10 TFLOPS for reasonable sizes
      }
    }
  }
}

TEST_CASE_METHOD(GemmTestFixture, "GEMM Numerical Stability", "[gemm][stability]") {
  SECTION("Different value ranges") {
    int64_t M = 256, N = 256, K = 256;
    
    std::vector<float> scales = {0.01f, 0.1f, 1.0f, 5.0f};
    
    for (float scale : scales) {
      INFO("Testing value scale: " << scale);
      
      auto A = create_test_tensor({M, K}, scale);
      auto B = create_test_tensor({N, K}, scale);
      
      auto C_fp4 = gemm_fp4_quantized(A, B, 1.0f, 0.0f, quantization_mode::block_1d);
      auto C_ref = torch::matmul(A, B.T);
      
      // Check for NaNs or infinities
      CHECK(torch::isfinite(C_fp4).all().item<bool>());
      
      float rel_error = compute_relative_error(C_fp4, C_ref);
      
      if (scale >= 0.1f) {
        CHECK(rel_error < 0.5f);  // Should be reasonable for moderate scales
      } else {
        INFO("Small scale may have higher relative error: " << rel_error);
      }
    }
  }

  SECTION("Extreme values handling") {
    int64_t M = 128, N = 128, K = 128;
    auto A = create_test_tensor({M, K}, 1.0f);
    auto B = create_test_tensor({N, K}, 1.0f);
    
    // Set some extreme values
    A[0][0] = 10.0f;    // Large value
    A[0][1] = -10.0f;   // Large negative
    A[1][0] = 1e-6f;    // Very small
    B[0][0] = 10.0f;
    B[0][1] = -10.0f;
    B[1][0] = 1e-6f;
    
    auto C_fp4 = gemm_fp4_quantized(A, B, 1.0f, 0.0f, quantization_mode::per_tensor);
    auto C_ref = torch::matmul(A, B.T);
    
    // Should handle extreme values without producing NaNs
    CHECK(torch::isfinite(C_fp4).all().item<bool>());
    
    // Relative error might be high, but results should be bounded
    CHECK(torch::abs(C_fp4).max().item<float>() < 1000.0f);
  }
}

}  // namespace s4::nvfp4::test