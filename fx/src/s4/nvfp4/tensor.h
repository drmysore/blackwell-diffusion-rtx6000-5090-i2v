#pragma once

#include <memory>
#include <vector>

#include <cuda_fp8.h>
#include <torch/types.h>

namespace s4::nvfp4 {

constexpr float FP4_E2M1_MAX = 6.0f;
constexpr float FP8_E4M3_MAX = 448.0f;
constexpr std::size_t FP4_BLOCK_SIZE = 16;

enum class quantization_mode {
  block_1d,    // 1D blocks of 16 elements
  block_2d,    // 2D blocks of 16x16 elements
  per_tensor,  // Single scale for entire tensor
  per_channel  // Scale per output channel (for weights)
};

struct tensor {

  static auto from_bfloat16(const torch::Tensor input,
                            quantization_mode mode = quantization_mode::block_1d) -> tensor {

    TORCH_CHECK(input.dtype() == torch::kBFloat16, "[s4] [nvfp4] [tensor] Input must be BFloat16");
    TORCH_CHECK(input.is_cuda(), "[s4] [nvfp4] [tensor] Input must be on CUDA device");

    auto result = s4::nvfp4::tensor{};
    result.shape_ = input.sizes().vec();
    result.mode_ = mode;
    result.device_ = input.device();

    switch (mode) {
      case quantization_mode::block_1d: {
        result.quantize_block_1d(input);
      } break;
      case quantization_mode::block_2d: {
        result.quantize_block_2d(input);
      } break;
      case quantization_mode::per_tensor: {
        result.quantize_per_tensor(input);
      } break;
      case quantization_mode::per_channel: {
        result.quantize_per_channel(input);
      } break;
    }

    return result;
  }

  [[nodiscard]] auto to_bfloat16() const -> torch::Tensor {
    auto rv = torch::empty(shape_, torch::TensorOptions().dtype(torch::kBFloat16).device(device_));

    switch (mode_) {
      case quantization_mode::block_1d:
        dequantize_block_1d(rv);
        break;
      case quantization_mode::block_2d:
        dequantize_block_2d(rv);
        break;
      case quantization_mode::per_tensor:
        dequantize_per_tensor(rv);
        break;
      case quantization_mode::per_channel:
        dequantize_per_channel(rv);
        break;
    }

    return rv;
  }

  [[nodiscard]] auto quantized_data() const -> torch::Tensor {
    return quantized_data_;
  }

  [[nodiscard]] auto scale_factors() const -> torch::Tensor {
    return scale_factors_;
  }

  [[nodiscard]] auto shape() const -> torch::IntArrayRef {
    return shape_;
  }

  [[nodiscard]] auto numel() const -> int64_t {
    int64_t n = 1;
    for (auto dim : shape_)
      n *= dim;
    return n;
  }

  [[nodiscard]] auto memory_usage() const -> std::size_t {
    auto data_bytes = (numel() + 1) / 2;
    auto scale_bytes = scale_factors_.numel();
    return data_bytes + scale_bytes;
  }

private:
  torch::Tensor quantized_data_;
  torch::Tensor scale_factors_;
  std::vector<int64_t> shape_;
  quantization_mode mode_;
  torch::Device device_ = torch::kCPU;

  auto quantize_block_1d(const torch::Tensor& input) -> void;
  auto quantize_block_2d(const torch::Tensor& input) -> void;
  auto quantize_per_tensor(const torch::Tensor& input) -> void;
  auto quantize_per_channel(const torch::Tensor& input) -> void;

  auto dequantize_block_1d(torch::Tensor& output) const -> void;
  auto dequantize_block_2d(torch::Tensor& output) const -> void;
  auto dequantize_per_tensor(torch::Tensor& output) const -> void;
  auto dequantize_per_channel(torch::Tensor& output) const -> void;
};

// External kernel declaration - now uses pre-quantized data
torch::Tensor blackwell_fp4_gemm_prequantized(torch::Tensor A_fp4,     // Pre-quantized FP4 data
                                              torch::Tensor B_fp4,     // Pre-quantized FP4 data
                                              torch::Tensor A_scales,  // FP8 E4M3 scales
                                              torch::Tensor B_scales,  // FP8 E4M3 scales
                                              int64_t M, int64_t N, int64_t K, float alpha,
                                              float beta);

// High-level operations
[[nodiscard]] inline auto fp4_gemm(const tensor& A, const tensor& B, float alpha = 1.0f,
                                   float beta = 0.0f) -> torch::Tensor {

  auto A_shape = A.shape();
  auto B_shape = B.shape();

  TORCH_CHECK(A_shape.size() == 2 && B_shape.size() == 2,
              "[s4] [nvfp4] [gemm] Inputs must be 2D matrices");

  TORCH_CHECK(A_shape[1] == B_shape[1], "[s4] [nvfp4] [gemm] Inner dimensions must match");

  return blackwell_fp4_gemm_prequantized(A.quantized_data(), B.quantized_data(), A.scale_factors(),
                                         B.scale_factors(),
                                         A_shape[0],  // M
                                         B_shape[0],  // N
                                         A_shape[1],  // K
                                         alpha, beta);
}

[[nodiscard]] inline auto gemm_fp4_quantized(const torch::Tensor& A, const torch::Tensor& B,
                                             float alpha = 1.0f, float beta = 0.0f,
                                             quantization_mode mode = quantization_mode::block_1d)
    -> torch::Tensor {

  auto A_fp4 = tensor::from_bfloat16(A, mode);
  auto B_fp4 = tensor::from_bfloat16(B, mode);

  return fp4_gemm(A_fp4, B_fp4, alpha, beta);
}

}  // namespace s4::nvfp4
