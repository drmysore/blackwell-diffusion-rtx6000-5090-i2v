#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Forward declare at::BFloat16 to avoid including torch headers in CUDA code
namespace at {
class BFloat16;
}

namespace s4::nvfp4 {

// Block 1D quantization - hardware accelerated
void launch_block_1d_quantization_cuda(const at::BFloat16* input, uint8_t* output, uint8_t* scales,
                                       int64_t num_elements, cudaStream_t stream = 0);

// Block 1D dequantization
void launch_block_1d_dequantization_cuda(const uint8_t* input, const uint8_t* scales,
                                         at::BFloat16* output, int64_t num_elements,
                                         cudaStream_t stream = 0);

// Per-tensor quantization
void launch_per_tensor_quantization_cuda(const at::BFloat16* input, uint8_t* output, float scale,
                                         int64_t num_elements, cudaStream_t stream = 0);

// Per-tensor dequantization
void launch_per_tensor_dequantization_cuda(const uint8_t* input, float scale, at::BFloat16* output,
                                           int64_t num_elements, cudaStream_t stream = 0);

// Block 2D quantization
void launch_block_2d_quantization_cuda(const at::BFloat16* input, uint8_t* output, uint8_t* scales,
                                       int64_t rows, int64_t cols, cudaStream_t stream = 0);

// Block 2D dequantization
void launch_block_2d_dequantization_cuda(const uint8_t* input, const uint8_t* scales,
                                         at::BFloat16* output, int64_t rows, int64_t cols,
                                         cudaStream_t stream = 0);

// Per-channel quantization
void launch_per_channel_quantization_cuda(const at::BFloat16* input, uint8_t* output,
                                          uint8_t* scales, int64_t channels,
                                          int64_t elements_per_channel, cudaStream_t stream = 0);

// Per-channel dequantization
void launch_per_channel_dequantization_cuda(const uint8_t* input, const uint8_t* scales,
                                            at::BFloat16* output, int64_t channels,
                                            int64_t elements_per_channel, cudaStream_t stream = 0);

// TransformerEngine-style functions (adapted)
void launch_block_1d_quantization_cuda_te(const at::BFloat16* input, uint8_t* output, 
                                        uint8_t* scales, int64_t num_elements,
                                        float global_amax, cudaStream_t stream = 0);

void launch_block_1d_dequantization_cuda_te(const uint8_t* input, const uint8_t* scales,
                                          at::BFloat16* output, int64_t num_elements,
                                          float global_amax, cudaStream_t stream = 0);

}  // namespace s4::nvfp4
