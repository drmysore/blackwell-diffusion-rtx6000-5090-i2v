#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include <cuda_fp8.h>
#include <torch/extension.h>

#include "fxy/core/result.h"

namespace fxy::ops::quantization {

// Forward declaration
torch::Tensor blackwell_fp4_gemm(torch::Tensor A, torch::Tensor B, float alpha, float beta);

// Constants following NVIDIA's implementation
constexpr float FP4_E2M1_MAX = 6.0f;
constexpr float FP8_E4M3_MAX = 448.0f;
constexpr std::size_t NVFP4_BLOCK_SIZE = 16;

// FP4 E2M1 representable values for round-to-nearest
constexpr std::array<float, 8> FP4_E2M1_VALUES = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

class nvfp4_block_quantizer {
public:
  nvfp4_block_quantizer() = default;

  // Compute global encoding scale factor
  [[nodiscard]] auto compute_global_scale(float global_amax) const noexcept -> float {
    if (global_amax == 0.0f || std::isinf(global_amax)) {
      return 1.0f;
    }
    auto scale = FP8_E4M3_MAX * FP4_E2M1_MAX / global_amax;
    return std::min(scale, std::numeric_limits<float>::max());
  }

  // Round to nearest FP4 E2M1 value
  [[nodiscard]] auto round_to_nearest_fp4(float value) const noexcept -> float {
    value = std::clamp(value, -FP4_E2M1_MAX, FP4_E2M1_MAX);
    auto abs_val = std::abs(value);
    auto sign = std::copysign(1.0f, value);

    // Find nearest representable value
    float best_val = 0.0f;
    float min_diff = std::numeric_limits<float>::max();

    for (auto fp4_val : FP4_E2M1_VALUES) {
      auto diff = std::abs(abs_val - fp4_val);
      if (diff < min_diff) {
        min_diff = diff;
        best_val = fp4_val;
      }
    }

    return sign * best_val;
  }

  // Quantize tensor with 1D block quantization
  auto quantize_1d_blocks(const torch::Tensor& input, torch::Tensor& scale_factors)
      -> torch::Tensor {

    // Validate inputs
    TORCH_CHECK(input.dim() >= 2, "Input must be at least 2D");
    TORCH_CHECK(input.dtype() == torch::kBFloat16, "Input must be BFloat16");

    auto input_sizes = input.sizes();
    auto rows = input_sizes[0];
    auto cols = input_sizes[1];
    auto blocks_per_row = (cols + NVFP4_BLOCK_SIZE - 1) / NVFP4_BLOCK_SIZE;

    // Allocate scale factors
    scale_factors =
        torch::empty({rows, blocks_per_row},
                     torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));

    // Compute global amax
    auto global_amax = input.abs().max().item<float>();
    auto global_encode_scale = compute_global_scale(global_amax);
    auto global_decode_scale = 1.0f / global_encode_scale;

    // Create output tensor
    auto output = torch::empty_like(input);

    // Process each row
    for (std::int64_t row = 0; row < rows; ++row) {
      for (std::int64_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        auto col_start = block_idx * NVFP4_BLOCK_SIZE;
        auto col_end = std::min(col_start + NVFP4_BLOCK_SIZE, cols);

        // Find block maximum
        float block_amax = 0.0f;
        for (auto col = col_start; col < col_end; ++col) {
          auto val = input[row][col].item<float>();
          block_amax = std::max(block_amax, std::abs(val));
        }

        // Compute and store scale factor
        auto decode_scale = block_amax / FP4_E2M1_MAX;
        auto fp8_scale_val = decode_scale * global_encode_scale;
        fp8_scale_val = std::clamp(fp8_scale_val, -FP8_E4M3_MAX, FP8_E4M3_MAX);
        scale_factors[row][block_idx] = fp8_scale_val;

        // Compute encoding scale
        auto encode_scale = 1.0f / (fp8_scale_val * global_decode_scale);

        // Quantize block elements
        for (auto col = col_start; col < col_end; ++col) {
          auto val = input[row][col].item<float>();
          auto scaled_val = val * encode_scale;
          auto quantized_val = round_to_nearest_fp4(scaled_val);
          output[row][col] = quantized_val;
        }
      }
    }

    return output;
  }
};

// Main wrapper function following the style guide
auto nvfp4_gemm_wrapper(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                        torch::Tensor scale_input, torch::Tensor scale_weight, float alpha,
                        float beta) -> torch::Tensor {

  // Create quantizer
  auto quantizer = nvfp4_block_quantizer{};

  // Quantize inputs using 1D block quantization
  auto input_scale_factors = torch::Tensor{};
  auto input_quantized = quantizer.quantize_1d_blocks(input, input_scale_factors);

  auto weight_scale_factors = torch::Tensor{};
  auto weight_quantized = quantizer.quantize_1d_blocks(weight.t(), weight_scale_factors);

  // Call the FP4 kernel with properly quantized inputs
  auto result = blackwell_fp4_gemm(input_quantized, weight_quantized, alpha, 0.0f);

  // Dequantize output using scale factors
  // This is simplified - production would need proper block-wise dequantization
  auto input_global_scale = input_scale_factors.mean().item<float>();
  auto weight_global_scale = weight_scale_factors.mean().item<float>();
  result = result * input_global_scale * weight_global_scale;

  // Add bias if needed
  if (bias.defined() && bias.numel() > 0 && beta != 0.0f) {
    result = result + bias.unsqueeze(0) * beta;
  }

  return result;
}

auto nvfp4_linear_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                          torch::Tensor scale_input, torch::Tensor scale_weight) -> torch::Tensor {

  // Validate inputs following style guide
  TORCH_CHECK(input.dim() >= 2, "[fxy] [ops] [nvfp4] Input must be at least 2D");
  TORCH_CHECK(weight.dim() == 2, "[fxy] [ops] [nvfp4] Weight must be 2D");
  TORCH_CHECK(input.dtype() == torch::kBFloat16, "[fxy] [ops] [nvfp4] Input must be BFloat16");
  TORCH_CHECK(weight.dtype() == torch::kBFloat16, "[fxy] [ops] [nvfp4] Weight must be BFloat16");

  // Check M divisibility requirement
  auto batch_size = input.size(0);
  TORCH_CHECK(batch_size % 128 == 0,
              "[fxy] [ops] [nvfp4] Batch size must be divisible by 128, got {}", batch_size);

  // Reshape input to 2D if needed
  auto input_sizes = input.sizes();
  auto total_batch_size = 1;

  for (int idx = 0; idx < input.dim() - 1; idx++) {
    total_batch_size *= input_sizes[idx];
  }

  auto in_features = input_sizes[input.dim() - 1];
  auto out_features = weight.size(0);

  TORCH_CHECK(weight.size(1) == in_features,
              "[fxy] [ops] [nvfp4] Weight shape mismatch. Expected [{}, {}], got {}", out_features,
              in_features, weight.sizes());

  auto input_2d = input.reshape({total_batch_size, in_features});

  // Perform quantized GEMM
  auto output = nvfp4_gemm_wrapper(input_2d, weight, bias, scale_input, scale_weight, 1.0f, 1.0f);

  // Reshape output back to original batch dimensions
  std::vector<std::int64_t> output_sizes;
  for (int i = 0; i < input.dim() - 1; i++) {
    output_sizes.push_back(input_sizes[i]);
  }
  output_sizes.push_back(out_features);

  return output.reshape(output_sizes);
}

// Backward pass placeholder
std::vector<torch::Tensor> nvfp4_linear_backward(torch::Tensor grad_output, torch::Tensor input,
                                                 torch::Tensor weight, torch::Tensor scale_input,
                                                 torch::Tensor scale_weight,
                                                 torch::Tensor scale_grad_output,
                                                 bool needs_input_grad, bool needs_weight_grad,
                                                 bool needs_bias_grad) {

  // TODO: Implement backward pass with proper quantization
  torch::Tensor grad_input, grad_weight, grad_bias;
  return {grad_input, grad_weight, grad_bias};
}

PYBIND11_MODULE(nvfp4_linear_cpp, m) {
  m.def("forward", &nvfp4_linear_forward, "NVFP4 Linear forward with block quantization");
  m.def("backward", &nvfp4_linear_backward, "NVFP4 Linear backward");
}

}  // namespace fxy::ops::quantization
