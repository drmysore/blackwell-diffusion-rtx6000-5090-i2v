#include <cmath>
#include <random>
#include <vector>
#include <chrono>

#define CATCH_CONFIG_MAIN
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "quantization.cuh"

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
  cudaError_t err = call; \
  if (err != cudaSuccess) { \
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
  } \
} while(0)

namespace s4::nvfp4::test {

using Catch::Matchers::WithinRel;

class QuantizationTestFixture {
public:
  QuantizationTestFixture() : device_(torch::kCPU) {
    // Ensure CUDA is available
    REQUIRE(torch::cuda::is_available());
    device_ = torch::kCUDA;

    // Set random seed for reproducibility
    torch::manual_seed(42);
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
    auto expected_abs = torch::abs(expected) + 1e-8f;  // Add small epsilon
    auto rel_error = diff / expected_abs;
    return rel_error.mean().item<float>();
  }
};

TEST_CASE_METHOD(QuantizationTestFixture, "Block 1D Quantization", "[quantization][block_1d]") {
  SECTION("Small tensor quantization and dequantization") {
    auto input = create_test_tensor({64, 32});  // 2048 elements, 128 blocks
    auto flat_input = input.flatten();

    // Prepare output tensors
    auto num_elements = flat_input.numel();
    auto num_blocks = (num_elements + 15) / 16;  // FP4_BLOCK_SIZE = 16

    auto quantized = torch::empty({(num_elements + 1) / 2},
                                  torch::TensorOptions().dtype(torch::kUInt8).device(device_));
    auto scales =
        torch::empty({num_blocks}, torch::TensorOptions().dtype(torch::kUInt8).device(device_));

    // Test quantization
    REQUIRE_NOTHROW(launch_block_1d_quantization_cuda(reinterpret_cast<const at::BFloat16*>(flat_input.data_ptr()),
                                                      quantized.data_ptr<uint8_t>(),
                                                      scales.data_ptr<uint8_t>(), num_elements));

    // Test dequantization
    auto output = torch::zeros_like(flat_input);
    REQUIRE_NOTHROW(launch_block_1d_dequantization_cuda(
        quantized.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), reinterpret_cast<at::BFloat16*>(output.data_ptr()),
        num_elements));

    // Verify shape preservation
    REQUIRE(output.numel() == flat_input.numel());

    // Check quantization accuracy (FP4 has limited precision)
    float rel_error = compute_relative_error(output, flat_input);
    CHECK(rel_error < 0.3f);  // 30% relative error acceptable for FP4
  }

  SECTION("Large tensor stress test") {
    auto input = create_test_tensor({1024, 1024});  // 1M elements
    auto flat_input = input.flatten();
    auto num_elements = flat_input.numel();
    auto num_blocks = (num_elements + 15) / 16;

    auto quantized = torch::empty({(num_elements + 1) / 2},
                                  torch::TensorOptions().dtype(torch::kUInt8).device(device_));
    auto scales =
        torch::empty({num_blocks}, torch::TensorOptions().dtype(torch::kUInt8).device(device_));
    auto output = torch::zeros_like(flat_input);

    // Should handle large tensors without errors
    REQUIRE_NOTHROW(launch_block_1d_quantization_cuda(reinterpret_cast<const at::BFloat16*>(flat_input.data_ptr()),
                                                      quantized.data_ptr<uint8_t>(),
                                                      scales.data_ptr<uint8_t>(), num_elements));

    REQUIRE_NOTHROW(launch_block_1d_dequantization_cuda(
        quantized.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), reinterpret_cast<at::BFloat16*>(output.data_ptr()),
        num_elements));

    float rel_error = compute_relative_error(output, flat_input);
    CHECK(rel_error < 0.3f);
  }
}

TEST_CASE_METHOD(QuantizationTestFixture, "Per-tensor Quantization", "[quantization][per_tensor]") {
  SECTION("Basic per-tensor quantization") {
    auto input = create_test_tensor({256, 128});
    auto flat_input = input.flatten();
    auto num_elements = flat_input.numel();

    // Find global scale
    float global_max = torch::abs(input).max().item<float>();
    float scale = global_max / 6.0f;  // FP4_E2M1_MAX = 6.0f

    auto quantized = torch::empty({(num_elements + 1) / 2},
                                  torch::TensorOptions().dtype(torch::kUInt8).device(device_));
    auto output = torch::zeros_like(flat_input);

    // Test quantization
    REQUIRE_NOTHROW(launch_per_tensor_quantization_cuda(
        reinterpret_cast<const at::BFloat16*>(flat_input.data_ptr()), quantized.data_ptr<uint8_t>(), scale, num_elements));

    // Test dequantization
    REQUIRE_NOTHROW(launch_per_tensor_dequantization_cuda(
        quantized.data_ptr<uint8_t>(), scale, reinterpret_cast<at::BFloat16*>(output.data_ptr()), num_elements));

    float rel_error = compute_relative_error(output, flat_input);
    CHECK(rel_error < 0.4f);  // Per-tensor may have higher error
  }

  SECTION("Different value ranges") {
    std::vector<float> test_scales = {0.1f, 1.0f, 5.0f, 10.0f};

    for (float test_scale : test_scales) {
      INFO("Testing with scale: " << test_scale);

      auto input = create_test_tensor({128, 64}, test_scale);
      auto flat_input = input.flatten();
      auto num_elements = flat_input.numel();

      float global_max = torch::abs(input).max().item<float>();
      float quant_scale = global_max / 6.0f;

      auto quantized = torch::empty({(num_elements + 1) / 2},
                                    torch::TensorOptions().dtype(torch::kUInt8).device(device_));
      auto output = torch::zeros_like(flat_input);

      REQUIRE_NOTHROW(launch_per_tensor_quantization_cuda(reinterpret_cast<const at::BFloat16*>(flat_input.data_ptr()),
                                                          quantized.data_ptr<uint8_t>(),
                                                          quant_scale, num_elements));

      REQUIRE_NOTHROW(
          launch_per_tensor_dequantization_cuda(quantized.data_ptr<uint8_t>(), quant_scale,
                                                reinterpret_cast<at::BFloat16*>(output.data_ptr()), num_elements));

      float rel_error = compute_relative_error(output, flat_input);
      CHECK(rel_error < 0.5f);
    }
  }
}

TEST_CASE_METHOD(QuantizationTestFixture, "Block 2D Quantization", "[quantization][block_2d]") {
  SECTION("2D block quantization") {
    int64_t rows = 64, cols = 128;
    auto input = create_test_tensor({rows, cols});
    auto flat_input = input.reshape({rows, cols});

    auto blocks_per_row = (cols + 15) / 16;  // FP4_BLOCK_SIZE = 16
    auto blocks_per_col = (rows + 15) / 16;
    auto total_blocks = blocks_per_col * blocks_per_row;

    auto quantized = torch::empty({(input.numel() + 1) / 2},
                                  torch::TensorOptions().dtype(torch::kUInt8).device(device_));
    auto scales =
        torch::empty({total_blocks}, torch::TensorOptions().dtype(torch::kUInt8).device(device_));
    auto output = torch::zeros_like(flat_input);

    REQUIRE_NOTHROW(launch_block_2d_quantization_cuda(reinterpret_cast<const at::BFloat16*>(flat_input.data_ptr()),
                                                      quantized.data_ptr<uint8_t>(),
                                                      scales.data_ptr<uint8_t>(), rows, cols));

    REQUIRE_NOTHROW(launch_block_2d_dequantization_cuda(
        quantized.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), reinterpret_cast<at::BFloat16*>(output.data_ptr()),
        rows, cols));

    float rel_error = compute_relative_error(output, flat_input);
    CHECK(rel_error < 0.3f);
  }
}

TEST_CASE_METHOD(QuantizationTestFixture, "Per-channel Quantization",
                 "[quantization][per_channel]") {
  SECTION("Multi-channel quantization") {
    int64_t channels = 16, elements_per_channel = 64;
    auto input = create_test_tensor({channels, elements_per_channel});

    auto quantized = torch::empty({(input.numel() + 1) / 2},
                                  torch::TensorOptions().dtype(torch::kUInt8).device(device_));
    auto scales =
        torch::empty({channels}, torch::TensorOptions().dtype(torch::kUInt8).device(device_));
    auto output = torch::zeros_like(input);

    REQUIRE_NOTHROW(launch_per_channel_quantization_cuda(
        reinterpret_cast<const at::BFloat16*>(input.data_ptr()), quantized.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),
        channels, elements_per_channel));

    REQUIRE_NOTHROW(launch_per_channel_dequantization_cuda(
        quantized.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), reinterpret_cast<at::BFloat16*>(output.data_ptr()),
        channels, elements_per_channel));

    float rel_error = compute_relative_error(output, input);
    CHECK(rel_error < 0.3f);
  }
}

TEST_CASE_METHOD(QuantizationTestFixture, "Edge Cases", "[quantization][edge_cases]") {
  SECTION("Zero tensor") {
    auto input =
        torch::zeros({32, 32}, torch::TensorOptions().dtype(torch::kBFloat16).device(device_));
    auto flat_input = input.flatten();
    auto num_elements = flat_input.numel();
    auto num_blocks = (num_elements + 15) / 16;

    auto quantized = torch::empty({(num_elements + 1) / 2},
                                  torch::TensorOptions().dtype(torch::kUInt8).device(device_));
    auto scales =
        torch::empty({num_blocks}, torch::TensorOptions().dtype(torch::kUInt8).device(device_));
    auto output = torch::zeros_like(flat_input);

    // Should handle zero input gracefully
    REQUIRE_NOTHROW(launch_block_1d_quantization_cuda(reinterpret_cast<const at::BFloat16*>(flat_input.data_ptr()),
                                                      quantized.data_ptr<uint8_t>(),
                                                      scales.data_ptr<uint8_t>(), num_elements));

    REQUIRE_NOTHROW(launch_block_1d_dequantization_cuda(
        quantized.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), reinterpret_cast<at::BFloat16*>(output.data_ptr()),
        num_elements));

    // Output should be close to zero
    float max_abs = torch::abs(output).max().item<float>();
    CHECK(max_abs < 1e-6f);
  }

  SECTION("Single element") {
    auto input = create_test_tensor({1});
    auto flat_input = input.flatten();

    auto quantized = torch::empty({1}, torch::TensorOptions().dtype(torch::kUInt8).device(device_));
    auto scales = torch::empty({1}, torch::TensorOptions().dtype(torch::kUInt8).device(device_));
    auto output = torch::zeros_like(flat_input);

    // Should handle single element
    REQUIRE_NOTHROW(launch_block_1d_quantization_cuda(reinterpret_cast<const at::BFloat16*>(flat_input.data_ptr()),
                                                      quantized.data_ptr<uint8_t>(),
                                                      scales.data_ptr<uint8_t>(), 1));

    REQUIRE_NOTHROW(launch_block_1d_dequantization_cuda(quantized.data_ptr<uint8_t>(),
                                                        scales.data_ptr<uint8_t>(),
                                                        reinterpret_cast<at::BFloat16*>(output.data_ptr()), 1));
  }
}

TEST_CASE_METHOD(QuantizationTestFixture, "Memory Efficiency", "[quantization][memory]") {
  SECTION("Compression ratio verification") {
    auto input = create_test_tensor({1024, 512});
    auto num_elements = input.numel();

    // Original memory: BFloat16 = 2 bytes per element
    size_t original_memory = num_elements * 2;

    // Quantized memory: 4 bits per element + scales
    auto num_blocks = (num_elements + 15) / 16;
    size_t quantized_memory = (num_elements + 1) / 2 + num_blocks;  // data + scales

    INFO("Original memory: " << original_memory << " bytes");
    INFO("Quantized memory: " << quantized_memory << " bytes");
    INFO("Compression ratio: " << (float)original_memory / quantized_memory);

    // Should achieve significant compression
    CHECK(quantized_memory < original_memory / 2);  // At least 2x compression
  }
}

TEST_CASE_METHOD(QuantizationTestFixture, "Stream Support", "[quantization][streams]") {
  SECTION("Non-default CUDA streams") {
    auto input = create_test_tensor({256, 128});
    auto flat_input = input.flatten();
    auto num_elements = flat_input.numel();
    auto num_blocks = (num_elements + 15) / 16;

    // Create custom CUDA stream
    cudaStream_t custom_stream;
    CUDA_CHECK(cudaStreamCreate(&custom_stream));

    auto quantized = torch::empty({(num_elements + 1) / 2},
                                  torch::TensorOptions().dtype(torch::kUInt8).device(device_));
    auto scales = torch::empty({num_blocks}, torch::TensorOptions().dtype(torch::kUInt8).device(device_));
    auto output = torch::zeros_like(flat_input);

    // Test quantization with custom stream
    REQUIRE_NOTHROW(launch_block_1d_quantization_cuda(reinterpret_cast<const at::BFloat16*>(flat_input.data_ptr()),
                                                      quantized.data_ptr<uint8_t>(),
                                                      scales.data_ptr<uint8_t>(), num_elements, custom_stream));

    REQUIRE_NOTHROW(launch_block_1d_dequantization_cuda(quantized.data_ptr<uint8_t>(),
                                                        scales.data_ptr<uint8_t>(),
                                                        reinterpret_cast<at::BFloat16*>(output.data_ptr()), num_elements, custom_stream));

    // Synchronize custom stream
    CUDA_CHECK(cudaStreamSynchronize(custom_stream));

    // Verify results
    float rel_error = compute_relative_error(output, flat_input);
    CHECK(rel_error < 0.3f);

    CUDA_CHECK(cudaStreamDestroy(custom_stream));
  }
}

TEST_CASE_METHOD(QuantizationTestFixture, "Error Conditions", "[quantization][errors]") {
  SECTION("Null pointer handling") {
    auto input = create_test_tensor({64, 32});
    auto flat_input = input.flatten();
    auto num_elements = flat_input.numel();

    auto quantized = torch::empty({(num_elements + 1) / 2},
                                  torch::TensorOptions().dtype(torch::kUInt8).device(device_));
    auto scales = torch::empty({(num_elements + 15) / 16}, torch::TensorOptions().dtype(torch::kUInt8).device(device_));

    // Test with null input pointer (should handle gracefully or assert)
    // Note: These might cause segfaults, so we test the API exists but don't expect graceful handling
    REQUIRE_THROWS_AS(launch_block_1d_quantization_cuda(nullptr, quantized.data_ptr<uint8_t>(),
                                                        scales.data_ptr<uint8_t>(), num_elements), std::exception);
  }

  SECTION("Zero elements") {
    auto quantized = torch::empty({0}, torch::TensorOptions().dtype(torch::kUInt8).device(device_));
    auto scales = torch::empty({0}, torch::TensorOptions().dtype(torch::kUInt8).device(device_));
    auto output = torch::empty({0}, torch::TensorOptions().dtype(torch::kBFloat16).device(device_));

    // Should handle zero elements gracefully
    REQUIRE_NOTHROW(launch_block_1d_quantization_cuda(nullptr, quantized.data_ptr<uint8_t>(),
                                                      scales.data_ptr<uint8_t>(), 0));
    REQUIRE_NOTHROW(launch_block_1d_dequantization_cuda(quantized.data_ptr<uint8_t>(),
                                                        scales.data_ptr<uint8_t>(),
                                                        reinterpret_cast<at::BFloat16*>(output.data_ptr()), 0));
  }
}

TEST_CASE_METHOD(QuantizationTestFixture, "Performance Regression", "[quantization][performance][!benchmark]") {
  SECTION("Large tensor throughput") {
    // Test on progressively larger tensors
    std::vector<std::pair<int, int>> sizes = {{1024, 1024}, {2048, 2048}, {4096, 2048}, {8192, 2048}};

    for (auto [rows, cols] : sizes) {
      auto input = create_test_tensor({rows, cols});
      auto flat_input = input.flatten();
      auto num_elements = flat_input.numel();
      auto num_blocks = (num_elements + 15) / 16;

      auto quantized = torch::empty({(num_elements + 1) / 2},
                                    torch::TensorOptions().dtype(torch::kUInt8).device(device_));
      auto scales = torch::empty({num_blocks}, torch::TensorOptions().dtype(torch::kUInt8).device(device_));
      auto output = torch::zeros_like(flat_input);

      // Warmup
      for (int i = 0; i < 10; ++i) {
        launch_block_1d_quantization_cuda(reinterpret_cast<const at::BFloat16*>(flat_input.data_ptr()),
                                          quantized.data_ptr<uint8_t>(),
                                          scales.data_ptr<uint8_t>(), num_elements);
        launch_block_1d_dequantization_cuda(quantized.data_ptr<uint8_t>(),
                                            scales.data_ptr<uint8_t>(),
                                            reinterpret_cast<at::BFloat16*>(output.data_ptr()), num_elements);
      }

      cudaDeviceSynchronize();

      // Benchmark quantization
      auto start = std::chrono::high_resolution_clock::now();
      int iterations = 100;
      
      for (int i = 0; i < iterations; ++i) {
        launch_block_1d_quantization_cuda(reinterpret_cast<const at::BFloat16*>(flat_input.data_ptr()),
                                          quantized.data_ptr<uint8_t>(),
                                          scales.data_ptr<uint8_t>(), num_elements);
      }
      cudaDeviceSynchronize();
      
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      
      double avg_time_us = duration.count() / (double)iterations;
      double throughput_gb_s = (num_elements * 2) / (avg_time_us * 1e-6) / 1e9;  // GB/s

      INFO("Size: " << rows << "x" << cols);
      INFO("Quantization throughput: " << throughput_gb_s << " GB/s");
      
      // Basic performance expectation: should be > 100 GB/s for reasonable sizes
      if (num_elements > 1000000) {  // Only check for larger tensors
        CHECK(throughput_gb_s > 100.0);
      }
    }
  }
}

}  // namespace s4::nvfp4::test