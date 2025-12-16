#include "tensor.h"

#include "quantization.cuh"

namespace s4::nvfp4 {

auto tensor::quantize_block_1d(const torch::Tensor& input) -> void {
  auto flat = input.flatten();
  auto num_elements = flat.numel();
  auto num_blocks = (num_elements + FP4_BLOCK_SIZE - 1) / FP4_BLOCK_SIZE;

  quantized_data_ = torch::empty({(num_elements + 1) / 2},
                                 torch::TensorOptions().dtype(torch::kUInt8).device(device_));

  scale_factors_ = torch::empty({static_cast<int64_t>(num_blocks)},
                                torch::TensorOptions().dtype(torch::kUInt8).device(device_));

  launch_block_1d_quantization_cuda(reinterpret_cast<const at::BFloat16*>(flat.data_ptr()),
                                    quantized_data_.data_ptr<uint8_t>(),
                                    scale_factors_.data_ptr<uint8_t>(), num_elements);
}

auto tensor::quantize_block_2d(const torch::Tensor& input) -> void {

  TORCH_CHECK(input.dim() >= 2,
              "[s4] [nvfp4] [tensor] Input must be at least 2D for block_2d quantization");

  auto rows = input.size(0);
  auto cols = input.numel() / rows;
  auto flat = input.reshape({rows, cols});

  auto blocks_per_row = (cols + FP4_BLOCK_SIZE - 1) / FP4_BLOCK_SIZE;
  auto blocks_per_col = (rows + FP4_BLOCK_SIZE - 1) / FP4_BLOCK_SIZE;

  quantized_data_ = torch::empty({(input.numel() + 1) / 2},
                                 torch::TensorOptions().dtype(torch::kUInt8).device(device_));

  scale_factors_ = torch::empty({static_cast<int64_t>(blocks_per_col * blocks_per_row)},
                                torch::TensorOptions().dtype(torch::kUInt8).device(device_));

  launch_block_2d_quantization_cuda(reinterpret_cast<const at::BFloat16*>(flat.data_ptr()),
                                    quantized_data_.data_ptr<uint8_t>(),
                                    scale_factors_.data_ptr<uint8_t>(), rows, cols);
}

auto tensor::quantize_per_tensor(const torch::Tensor& input) -> void {
  auto num_elements = input.numel();

  quantized_data_ = torch::empty({(num_elements + 1) / 2},
                                 torch::TensorOptions().dtype(torch::kUInt8).device(device_));

  scale_factors_ = torch::empty({1}, torch::TensorOptions().dtype(torch::kUInt8).device(device_));

  auto global_max = input.abs().max().item<float>();
  auto scale = global_max / FP4_E2M1_MAX;

  __nv_fp8_e4m3 scale_fp8 = __nv_fp8_e4m3(scale);
  scale_factors_[0] = *reinterpret_cast<uint8_t*>(&scale_fp8);

  launch_per_tensor_quantization_cuda(reinterpret_cast<const at::BFloat16*>(input.data_ptr()),
                                      quantized_data_.data_ptr<uint8_t>(), scale, num_elements);
}

auto tensor::quantize_per_channel(const torch::Tensor& input) -> void {
  TORCH_CHECK(input.dim() >= 2,
              "[s4] [nvfp4] [tensor] Input must be at least 2D for per-channel quantization");

  auto channels = input.size(0);
  auto elements_per_channel = input.numel() / channels;

  quantized_data_ = torch::empty({(input.numel() + 1) / 2},
                                 torch::TensorOptions().dtype(torch::kUInt8).device(device_));

  scale_factors_ =
      torch::empty({channels}, torch::TensorOptions().dtype(torch::kUInt8).device(device_));

  launch_per_channel_quantization_cuda(
      reinterpret_cast<const at::BFloat16*>(input.data_ptr()), quantized_data_.data_ptr<uint8_t>(),
      scale_factors_.data_ptr<uint8_t>(), channels, elements_per_channel);
}

auto tensor::dequantize_block_1d(torch::Tensor& output) const -> void {
  auto flat = output.flatten();

  launch_block_1d_dequantization_cuda(quantized_data_.data_ptr<uint8_t>(),
                                      scale_factors_.data_ptr<uint8_t>(),
                                      reinterpret_cast<at::BFloat16*>(flat.data_ptr()), flat.numel());
}

auto tensor::dequantize_block_2d(torch::Tensor& output) const -> void {
  auto rows = output.size(0);
  auto cols = output.numel() / rows;
  auto flat = output.reshape({rows, cols});

  launch_block_2d_dequantization_cuda(quantized_data_.data_ptr<uint8_t>(),
                                      scale_factors_.data_ptr<uint8_t>(),
                                      reinterpret_cast<at::BFloat16*>(flat.data_ptr()), rows, cols);
}

auto tensor::dequantize_per_tensor(torch::Tensor& output) const -> void {
  auto scale_fp8 = *reinterpret_cast<const __nv_fp8_e4m3*>(scale_factors_.data_ptr<uint8_t>());
  auto scale = static_cast<float>(scale_fp8);

  launch_per_tensor_dequantization_cuda(quantized_data_.data_ptr<uint8_t>(), scale,
                                        reinterpret_cast<at::BFloat16*>(output.data_ptr()), output.numel());
}

auto tensor::dequantize_per_channel(torch::Tensor& output) const -> void {
  auto channels = output.size(0);
  auto elements_per_channel = output.numel() / channels;

  launch_per_channel_dequantization_cuda(
      quantized_data_.data_ptr<uint8_t>(), scale_factors_.data_ptr<uint8_t>(),
      reinterpret_cast<at::BFloat16*>(output.data_ptr()), channels, elements_per_channel);
}

}  // namespace s4::nvfp4
