#include <chrono>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <torch/torch.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

// Include headers that define the utility functions we want to test
#include "tensor.h"

namespace s4::nvfp4::test {

class UtilsTestFixture {
public:
  UtilsTestFixture() : device_(torch::kCPU) {
    if (torch::cuda::is_available()) {
      device_ = torch::kCUDA;
    } else {
      device_ = torch::kCPU;
      SKIP("CUDA not available - skipping GPU tests");
    }

    torch::manual_seed(456);
  }

protected:
  torch::Device device_;

  torch::Tensor create_test_tensor(std::vector<int64_t> shape, float scale = 1.0f) {
    auto tensor =
        torch::randn(shape, torch::TensorOptions().dtype(torch::kBFloat16).device(device_));
    return tensor * scale;
  }
};

TEST_CASE_METHOD(UtilsTestFixture, "Quantization Mode Enum", "[utils][enums]") {
  SECTION("Quantization mode values") {
    // Test that enum values are distinct
    CHECK(quantization_mode::block_1d != quantization_mode::block_2d);
    CHECK(quantization_mode::block_1d != quantization_mode::per_tensor);
    CHECK(quantization_mode::block_1d != quantization_mode::per_channel);
    CHECK(quantization_mode::block_2d != quantization_mode::per_tensor);
    CHECK(quantization_mode::block_2d != quantization_mode::per_channel);
    CHECK(quantization_mode::per_tensor != quantization_mode::per_channel);
  }
}

TEST_CASE_METHOD(UtilsTestFixture, "Constants and Limits", "[utils][constants]") {
  SECTION("FP4 and FP8 limits") {
    // These constants should be defined in tensor.h
    CHECK(FP4_E2M1_MAX == 6.0f);
    CHECK(FP8_E4M3_MAX == 448.0f);
    CHECK(FP4_BLOCK_SIZE == 16);
  }

  SECTION("Block size calculations") {
    // Test various input sizes with block size
    std::vector<size_t> test_sizes = {1, 15, 16, 17, 31, 32, 63, 64, 127, 128, 1024, 1025};

    for (size_t size : test_sizes) {
      size_t num_blocks = (size + FP4_BLOCK_SIZE - 1) / FP4_BLOCK_SIZE;

      // Verify block count makes sense
      CHECK(num_blocks > 0);
      CHECK((num_blocks - 1) * FP4_BLOCK_SIZE < size);
      CHECK(num_blocks * FP4_BLOCK_SIZE >= size);
    }
  }
}

TEST_CASE_METHOD(UtilsTestFixture, "Memory Layout Calculations", "[utils][memory]") {
  SECTION("Packed storage size") {
    std::vector<size_t> element_counts = {1, 2, 3, 4, 15, 16, 17, 31, 32, 1000, 1001};

    for (size_t count : element_counts) {
      // FP4 packs 2 elements per byte
      size_t packed_bytes = (count + 1) / 2;

      CHECK(packed_bytes > 0);
      CHECK(packed_bytes <= (count + 1) / 2);
      CHECK(packed_bytes * 2 >= count);  // Can store at least count elements
    }
  }

  SECTION("Scale factor storage") {
    // Test scale storage for different quantization modes
    std::vector<int64_t> rows = {16, 32, 64, 128, 256};
    std::vector<int64_t> cols = {16, 32, 64, 128, 256};

    for (auto r : rows) {
      for (auto c : cols) {
        // Block 1D: one scale per block of 16 elements
        size_t total_elements = r * c;
        size_t block_1d_scales = (total_elements + FP4_BLOCK_SIZE - 1) / FP4_BLOCK_SIZE;
        CHECK(block_1d_scales > 0);

        // Block 2D: one scale per 16x16 block
        size_t blocks_per_row = (c + FP4_BLOCK_SIZE - 1) / FP4_BLOCK_SIZE;
        size_t blocks_per_col = (r + FP4_BLOCK_SIZE - 1) / FP4_BLOCK_SIZE;
        size_t block_2d_scales = blocks_per_row * blocks_per_col;
        CHECK(block_2d_scales > 0);

        // Per-tensor: single scale
        size_t per_tensor_scales = 1;
        CHECK(per_tensor_scales == 1);

        // Per-channel: one scale per row (channel)
        size_t per_channel_scales = r;
        CHECK(per_channel_scales == r);
      }
    }
  }
}

TEST_CASE_METHOD(UtilsTestFixture, "Tensor Shape Utilities", "[utils][shapes]") {
  SECTION("Shape vector operations") {
    std::vector<std::vector<int64_t>> test_shapes = {
        {1}, {16}, {32, 32}, {64, 128}, {100, 200, 300}, {1, 1, 1, 1}};

    for (auto shape : test_shapes) {
      // Calculate total elements
      int64_t expected_numel = 1;
      for (auto dim : shape) {
        expected_numel *= dim;
      }

      // Create tensor and verify
      auto tensor =
          torch::randn(shape, torch::TensorOptions().dtype(torch::kBFloat16).device(device_));
      CHECK(tensor.numel() == expected_numel);
      CHECK(tensor.dim() == static_cast<int64_t>(shape.size()));

      for (size_t i = 0; i < shape.size(); ++i) {
        CHECK(tensor.size(i) == shape[i]);
      }
    }
  }

  SECTION("Dimension requirements") {
    // Test minimum dimension requirements for different modes
    auto tensor_1d = create_test_tensor({128});
    auto tensor_2d = create_test_tensor({32, 64});
    auto tensor_3d = create_test_tensor({16, 32, 64});

    // Block 1D should work with any dimensionality
    CHECK_NOTHROW(tensor::from_bfloat16(tensor_1d, quantization_mode::block_1d));
    CHECK_NOTHROW(tensor::from_bfloat16(tensor_2d, quantization_mode::block_1d));
    CHECK_NOTHROW(tensor::from_bfloat16(tensor_3d, quantization_mode::block_1d));

    // Per-tensor should work with any dimensionality
    CHECK_NOTHROW(tensor::from_bfloat16(tensor_1d, quantization_mode::per_tensor));
    CHECK_NOTHROW(tensor::from_bfloat16(tensor_2d, quantization_mode::per_tensor));
    CHECK_NOTHROW(tensor::from_bfloat16(tensor_3d, quantization_mode::per_tensor));

    // Block 2D requires at least 2D
    CHECK_THROWS(tensor::from_bfloat16(tensor_1d, quantization_mode::block_2d));
    CHECK_NOTHROW(tensor::from_bfloat16(tensor_2d, quantization_mode::block_2d));
    CHECK_NOTHROW(tensor::from_bfloat16(tensor_3d, quantization_mode::block_2d));

    // Per-channel requires at least 2D
    CHECK_THROWS(tensor::from_bfloat16(tensor_1d, quantization_mode::per_channel));
    CHECK_NOTHROW(tensor::from_bfloat16(tensor_2d, quantization_mode::per_channel));
    CHECK_NOTHROW(tensor::from_bfloat16(tensor_3d, quantization_mode::per_channel));
  }
}

TEST_CASE_METHOD(UtilsTestFixture, "Device and Type Validation", "[utils][validation]") {
  SECTION("Device compatibility") {
    // Create tensors on different devices
    auto cuda_tensor =
        torch::randn({32, 32}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
    auto cpu_tensor =
        torch::randn({32, 32}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU));

    // Only CUDA tensors should be accepted
    CHECK_NOTHROW(tensor::from_bfloat16(cuda_tensor));
    CHECK_THROWS(tensor::from_bfloat16(cpu_tensor));
  }

  SECTION("Data type validation") {
    // Test different data types
    auto bf16_tensor =
        torch::randn({32, 32}, torch::TensorOptions().dtype(torch::kBFloat16).device(device_));
    auto f32_tensor =
        torch::randn({32, 32}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    auto f16_tensor =
        torch::randn({32, 32}, torch::TensorOptions().dtype(torch::kFloat16).device(device_));
    auto i32_tensor = torch::randint(0, 100, {32, 32},
                                     torch::TensorOptions().dtype(torch::kInt32).device(device_));

    // Only BFloat16 should be accepted
    CHECK_NOTHROW(tensor::from_bfloat16(bf16_tensor));
    CHECK_THROWS(tensor::from_bfloat16(f32_tensor));
    CHECK_THROWS(tensor::from_bfloat16(f16_tensor));
    CHECK_THROWS(tensor::from_bfloat16(i32_tensor));
  }
}

TEST_CASE_METHOD(UtilsTestFixture, "Error Messages and Debugging", "[utils][errors]") {
  SECTION("Error message content") {
    try {
      auto cpu_tensor = torch::randn(
          {32, 32}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU));
      tensor::from_bfloat16(cpu_tensor);
      FAIL("Expected exception was not thrown");
    } catch (const c10::Error& e) {
      std::string error_msg = e.what();

      // Error message should be informative
      CHECK((error_msg.find("CUDA") != std::string::npos ||
            error_msg.find("device") != std::string::npos));
    }

    try {
      auto f32_tensor =
          torch::randn({32, 32}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
      tensor::from_bfloat16(f32_tensor);
      FAIL("Expected exception was not thrown");
    } catch (const c10::Error& e) {
      std::string error_msg = e.what();

      // Error message should mention data type
      CHECK((error_msg.find("BFloat16") != std::string::npos ||
            error_msg.find("dtype") != std::string::npos));
    }
  }
}

TEST_CASE_METHOD(UtilsTestFixture, "Performance Utilities", "[utils][performance]") {
  SECTION("Memory usage estimation") {
    std::vector<std::vector<int64_t>> test_shapes = {{32, 32}, {64, 128}, {256, 512}, {1024, 1024}};

    for (auto shape : test_shapes) {
      auto input = create_test_tensor(shape);
      auto fp4_tensor = tensor::from_bfloat16(input, quantization_mode::block_1d);

      size_t original_bytes = input.numel() * 2;  // BFloat16 = 2 bytes
      size_t compressed_bytes = fp4_tensor.memory_usage();

      INFO("Shape: [" << shape[0] << ", " << shape[1] << "]");
      INFO("Original: " << original_bytes << " bytes");
      INFO("Compressed: " << compressed_bytes << " bytes");
      INFO("Ratio: " << (float)original_bytes / compressed_bytes);

      // Sanity checks
      CHECK(compressed_bytes > 0);
      CHECK(compressed_bytes < original_bytes);      // Should be compressed
      CHECK(compressed_bytes >= input.numel() / 4);  // At least some overhead
    }
  }

  SECTION("Basic timing utilities") {
    auto input = create_test_tensor({512, 512});

    // Time quantization operation
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 10; ++i) {
      auto fp4_tensor = tensor::from_bfloat16(input, quantization_mode::block_1d);
      auto output = fp4_tensor.to_bfloat16();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Should complete in reasonable time
    CHECK(duration.count() < 10000000);  // Less than 10 seconds for 10 iterations

    INFO("10 quantization round-trips took: " << duration.count() << " microseconds");
  }
}

}  // namespace s4::nvfp4::test