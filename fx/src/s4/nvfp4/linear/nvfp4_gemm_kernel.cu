#pragma once

#include <type_traits>
#include <vector>

#include <torch/extension.h>

namespace s4::nvfp4 {

// Forward declarations for kernel functions
extern "C" {

torch::Tensor blackwell_fp4_gemm(torch::Tensor A,  // [M, K] in BF16
                                 torch::Tensor B,  // [N, K] in BF16
                                 float alpha, float beta);

}  // extern "C"

// Pure function for FP4 GEMM - stateless, just does the operation
template <typename AccumulatorType = float>
[[nodiscard]] auto nvfp4_gemm(const torch::Tensor& input_matrix,   // [M, K]
                              const torch::Tensor& weight_matrix,  // [N, K]
                              float alpha = 1.0f, float beta = 0.0f) -> torch::Tensor {

  // Validate tensor properties
  TORCH_CHECK(input_matrix.is_cuda() && weight_matrix.is_cuda(),
              "[fxy] [ops] [nvfp4] Tensors must be on CUDA device");
  TORCH_CHECK(input_matrix.dtype() == torch::kBFloat16 && weight_matrix.dtype() == torch::kBFloat16,
              "[fxy] [ops] [nvfp4] Tensors must be BFloat16");
  TORCH_CHECK(input_matrix.dim() == 2 && weight_matrix.dim() == 2,
              "[fxy] [ops] [nvfp4] Tensors must be 2D matrices");

  auto M = input_matrix.size(0);
  auto K = input_matrix.size(1);
  auto N = weight_matrix.size(0);

  TORCH_CHECK(weight_matrix.size(1) == K,
              "[fxy] [ops] [nvfp4] Inner dimensions must match: {} vs {}", K,
              weight_matrix.size(1));

  // Check alignment requirements
  auto M_aligned = (M % 128 == 0);
  auto K_aligned = (K % 128 == 0);
  auto N_aligned = (N % 128 == 0);

  if (M_aligned && K_aligned && N_aligned) {
    // Fast path - no padding needed
    return blackwell_fp4_gemm(input_matrix, weight_matrix, alpha, beta);
  }

  // Slow path - need padding
  auto M_padded = ((M + 127) / 128) * 128;
  auto K_padded = ((K + 127) / 128) * 128;
  auto N_padded = ((N + 127) / 128) * 128;

  auto input_padded = torch::zeros({M_padded, K_padded}, input_matrix.options());
  input_padded.slice(0, 0, M).slice(1, 0, K) = input_matrix;

  auto weight_padded = torch::zeros({N_padded, K_padded}, weight_matrix.options());
  weight_padded.slice(0, 0, N).slice(1, 0, K) = weight_matrix;

  auto output_padded = blackwell_fp4_gemm(input_padded, weight_padded, alpha, beta);

  return output_padded.slice(0, 0, M).slice(1, 0, N);
}

// Pure function for FP4 GEMV - when one dimension is 1
template <typename AccumulatorType = float>
[[nodiscard]] auto nvfp4_gemv(const torch::Tensor& input_vector,   // [M] or [M, 1]
                              const torch::Tensor& weight_matrix,  // [N, M]
                              float alpha = 1.0f) -> torch::Tensor {

  // Ensure input is 2D for GEMM
  auto input_2d = input_vector.dim() == 1 ? input_vector.unsqueeze(0)  // [1, M]
                                          : input_vector;              // [M, 1] -> need transpose

  if (input_2d.size(0) != 1) {
    input_2d = input_2d.t();  // Now [1, M]
  }

  // GEMV via GEMM: [1, M] @ [N, M].T = [1, N]
  auto output_2d = nvfp4_gemm<AccumulatorType>(input_2d, weight_matrix, alpha, 0.0f);

  // Return as vector
  return output_2d.squeeze(0);
}

// Batched GEMM for multiple matrices
template <typename AccumulatorType = float>
[[nodiscard]] auto nvfp4_batched_gemm(const torch::Tensor& input_batch,    // [B, M, K]
                                      const torch::Tensor& weight_matrix,  // [N, K]
                                      float alpha = 1.0f) -> torch::Tensor {

  TORCH_CHECK(input_batch.dim() == 3, "[fxy] [ops] [nvfp4] Input must be 3D for batched GEMM");

  auto batch_size = input_batch.size(0);
  auto M = input_batch.size(1);
  auto K = input_batch.size(2);

  // Reshape to 2D for single GEMM call
  auto input_2d = input_batch.reshape({batch_size * M, K});
  auto output_2d = nvfp4_gemm<AccumulatorType>(input_2d, weight_matrix, alpha, 0.0f);

  // Reshape back to batched
  return output_2d.reshape({batch_size, M, weight_matrix.size(0)});
}

// Linear layer forward - pure function
[[nodiscard]] auto nvfp4_linear_forward(
    const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias,
    const torch::Tensor& scale_input,   // Reserved for future use
    const torch::Tensor& scale_weight)  // Reserved for future use
    -> torch::Tensor {

  // Validate inputs
  TORCH_CHECK(input.dim() >= 2, "[fxy] [ops] [nvfp4] Input must be at least 2D, got {}D",
              input.dim());
  TORCH_CHECK(weight.dim() == 2, "[fxy] [ops] [nvfp4] Weight must be 2D, got {}D", weight.dim());

  // Get dimensions
  auto input_sizes = input.sizes();
  auto batch_dims = std::vector<int64_t>(input_sizes.begin(), input_sizes.end() - 1);
  auto in_features = input_sizes.back();
  auto out_features = weight.size(0);

  TORCH_CHECK(weight.size(1) == in_features, "[fxy] [ops] [nvfp4] Weight shape mismatch");

  // Flatten batch dimensions
  auto batch_size = 1;
  for (auto dim : batch_dims) {
    batch_size *= dim;
  }

  auto input_2d = input.reshape({batch_size, in_features});

  // Compute output = input @ weight.T
  auto output = nvfp4_gemm<float>(input_2d, weight, 1.0f, 0.0f);

  // Add bias if provided
  if (bias.defined() && bias.numel() > 0) {
    output = output + bias.unsqueeze(0);
  }

  // Reshape to original batch dims
  auto output_sizes = batch_dims;
  output_sizes.push_back(out_features);

  return output.reshape(output_sizes);
}

// Linear layer backward - pure function
[[nodiscard]] auto nvfp4_linear_backward(const torch::Tensor& grad_output,
                                         const torch::Tensor& input, const torch::Tensor& weight,
                                         const torch::Tensor& scale_input,        // Reserved
                                         const torch::Tensor& scale_weight,       // Reserved
                                         const torch::Tensor& scale_grad_output,  // Reserved
                                         bool needs_input_grad, bool needs_weight_grad,
                                         bool needs_bias_grad)
    -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> {

  // Get dimensions
  auto input_sizes = input.sizes();
  auto batch_dims = std::vector<int64_t>(input_sizes.begin(), input_sizes.end() - 1);
  auto batch_size = 1;
  for (auto dim : batch_dims) {
    batch_size *= dim;
  }
  auto in_features = input_sizes.back();
  auto out_features = weight.size(0);

  // Flatten to 2D
  auto input_2d = input.reshape({batch_size, in_features});
  auto grad_output_2d = grad_output.reshape({batch_size, out_features});

  torch::Tensor grad_input;
  torch::Tensor grad_weight;
  torch::Tensor grad_bias;

  if (needs_input_grad) {
    // grad_input = grad_output @ weight
    grad_input = nvfp4_gemm<float>(grad_output_2d, weight.t(), 1.0f, 0.0f);
    grad_input = grad_input.reshape(input_sizes);
  }

  if (needs_weight_grad) {
    // grad_weight = grad_output.T @ input
    grad_weight = nvfp4_gemm<float>(grad_output_2d.t(), input_2d, 1.0f, 0.0f).t();
  }

  if (needs_bias_grad) {
    // grad_bias = sum(grad_output, dim=0)
    grad_bias = grad_output_2d.sum(0);
  }

  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

// Utility functions for working with FP4 operations

// Check if dimensions are aligned for FP4 GEMM
[[nodiscard]] inline auto are_dimensions_aligned(int64_t M, int64_t N, int64_t K) noexcept -> bool {
  return (M % 128 == 0) && (N % 128 == 0) && (K % 128 == 0);
}

// Get padded dimensions for FP4 GEMM
[[nodiscard]] inline auto get_padded_dimensions(int64_t M, int64_t N, int64_t K) noexcept
    -> std::tuple<int64_t, int64_t, int64_t> {
  return {((M + 127) / 128) * 128, ((N + 127) / 128) * 128, ((K + 127) / 128) * 128};
}

// Estimate memory overhead from padding
[[nodiscard]] inline auto estimate_padding_overhead(int64_t M, int64_t N, int64_t K) noexcept
    -> float {
  auto [M_pad, N_pad, K_pad] = get_padded_dimensions(M, N, K);
  auto original_size = M * K + N * K + M * N;
  auto padded_size = M_pad * K_pad + N_pad * K_pad + M_pad * N_pad;
  return static_cast<float>(padded_size - original_size) / original_size;
}

}  // namespace s4::nvfp4

// Python bindings
PYBIND11_MODULE(nvfp4_linear_cpp, m) {
  m.doc() = "FP4 quantized operations for Blackwell GPUs";

  // Core operations
  m.def("gemm", &fxy::ops::quantization::nvfp4_gemm<float>, "FP4 quantized GEMM operation",
        py::arg("input_matrix"), py::arg("weight_matrix"), py::arg("alpha") = 1.0f,
        py::arg("beta") = 0.0f);

  m.def("gemv", &fxy::ops::quantization::nvfp4_gemv<float>, "FP4 quantized GEMV operation",
        py::arg("input_vector"), py::arg("weight_matrix"), py::arg("alpha") = 1.0f);

  m.def("batched_gemm", &fxy::ops::quantization::nvfp4_batched_gemm<float>,
        "FP4 quantized batched GEMM operation", py::arg("input_batch"), py::arg("weight_matrix"),
        py::arg("alpha") = 1.0f);

  // Linear layer operations (for compatibility)
  m.def("forward", &fxy::ops::quantization::nvfp4_linear_forward,
        "FP4 quantized linear forward pass", py::arg("input"), py::arg("weight"), py::arg("bias"),
        py::arg("scale_input"), py::arg("scale_weight"));

  m.def(
      "backward",
      [](const torch::Tensor& grad_output, const torch::Tensor& input, const torch::Tensor& weight,
         const torch::Tensor& scale_input, const torch::Tensor& scale_weight,
         const torch::Tensor& scale_grad_output, bool needs_input_grad, bool needs_weight_grad,
         bool needs_bias_grad) -> std::vector<torch::Tensor> {
        auto [grad_input, grad_weight, grad_bias] = fxy::ops::quantization::nvfp4_linear_backward(
            grad_output, input, weight, scale_input, scale_weight, scale_grad_output,
            needs_input_grad, needs_weight_grad, needs_bias_grad);
        return {grad_input, grad_weight, grad_bias};
      },
      "FP4 quantized linear backward pass", py::arg("grad_output"), py::arg("input"),
      py::arg("weight"), py::arg("scale_input"), py::arg("scale_weight"),
      py::arg("scale_grad_output"), py::arg("needs_input_grad"), py::arg("needs_weight_grad"),
      py::arg("needs_bias_grad"));

  // Utility functions
  m.def("are_dimensions_aligned", &fxy::ops::quantization::are_dimensions_aligned,
        "Check if dimensions are aligned for FP4 GEMM");

  m.def("get_padded_dimensions", &fxy::ops::quantization::get_padded_dimensions,
        "Get padded dimensions for FP4 GEMM");

  m.def("estimate_padding_overhead", &fxy::ops::quantization::estimate_padding_overhead,
        "Estimate memory overhead from padding");
}
