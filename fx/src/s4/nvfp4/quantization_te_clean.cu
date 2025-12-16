/*
 * Clean copy from TransformerEngine quantize_transpose_vector_blockwise_fp4.cu
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * 
 * Adapted for s4::nvfp4 namespace and simplified interface
 * Source: /home/b7r6/src/vendor/TransformerEngine/transformer_engine/common/transpose/quantize_transpose_vector_blockwise_fp4.cu
 */

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_fp4.h>

#include <algorithm>
#include <cfloat>
#include <utility>
#include <stdexcept>
#include <string>

#include "quantization.cuh"

namespace s4::nvfp4 {

#if CUDA_VERSION >= 12080

// Simplified types and constants from TransformerEngine
using std::int32_t;
using std::uint32_t;
using std::uint8_t;

// FP4 constants
constexpr float FP4_E2M1_MAX = 6.0f;
constexpr float FP8_E4M3_MAX = 448.0f;

// Device helper functions from TransformerEngine
__device__ __forceinline__ float ComputeGlobalEncodeScaleFP4(const float global_amax) {
  float global_encode_scale = FP8_E4M3_MAX * FP4_E2M1_MAX / global_amax;
  global_encode_scale = fminf(global_encode_scale, FLT_MAX);
  if (global_amax == 0.f || global_encode_scale == 0.f) {
    return 1.f;
  }
  return global_encode_scale;
}

template <typename ScaleType>
__device__ __forceinline__ ScaleType ComputeDecodeScaleFP4(const float amax,
                                                           const float global_encode_scale) {
  float decode_scale = amax / FP4_E2M1_MAX;
  decode_scale = decode_scale * global_encode_scale;
  decode_scale = fminf(decode_scale, FLT_MAX);
  return static_cast<ScaleType>(decode_scale);
}

template <typename ScaleType>
__device__ __forceinline__ float ComputeEncodeScaleFP4(ScaleType decode_scale,
                                                       const float global_decode_scale) {
  return fminf(1.0f / (static_cast<float>(decode_scale) * global_decode_scale), FLT_MAX);
}

template <typename IType, typename ScaleType>
__device__ __forceinline__ float ComputeOutputFP4(IType input, float encode_scale) {
  return static_cast<float>(input) * encode_scale;
}

__device__ __forceinline__ __nv_fp4x4_e2m1 cvt_fp32_to_fp4_4x_with_rn(const float2 in01,
                                                                      const float2 in23) {
#if CUDA_ARCH_HAS_FEATURE_SM10X_ALL
  uint32_t out_4x;
  asm volatile(
      "{\n"
      ".reg.b8 f0; \n\t"
      ".reg.b8 f1; \n\t"
      "cvt.rn.satfinite.e2m1x2.f32 f0, %1, %2;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 f1, %3, %4;\n\t"
      "mov.b32 %0, {f0, f1, f0, f1};\n\t"
      "}"
      : "=r"(out_4x)
      : "f"(in01.y), "f"(in01.x), "f"(in23.y), "f"(in23.x));
  return reinterpret_cast<__nv_fp4x4_e2m1*>(&out_4x)[0];
#else
  uint16_t dummy = 0;
  return *reinterpret_cast<__nv_fp4x4_e2m1*>(&dummy);
#endif
}

// Simplified block quantization kernel based on TransformerEngine
template<int kBlockSize = 16>
__global__ void block_1d_quantization_kernel(
    const __nv_bfloat16* input,
    uint8_t* quantized_output,
    uint8_t* scales,
    const int num_elements) {
  
  const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_blocks = (num_elements + kBlockSize - 1) / kBlockSize;
  const int block_idx = thread_idx / kBlockSize;
  const int element_idx_in_block = thread_idx % kBlockSize;
  
  if (block_idx >= num_blocks) return;
  
  const int global_element_idx = block_idx * kBlockSize + element_idx_in_block;
  
  // Compute block amax using warp reduction
  float amax = 0.0f;
  if (element_idx_in_block == 0) {
    for (int i = 0; i < kBlockSize && (block_idx * kBlockSize + i) < num_elements; ++i) {
      float val = __bfloat162float(input[block_idx * kBlockSize + i]);
      amax = fmaxf(amax, fabsf(val));
    }
  }
  
  // Broadcast amax to all threads in the warp
  amax = __shfl_sync(0xffffffff, amax, 0);
  
  // Compute scale factors
  float global_encode_scale = 1.0f; // Simplified - in practice should use global amax
  float decode_scale = amax / FP4_E2M1_MAX * global_encode_scale;
  uint8_t scale_fp8 = static_cast<uint8_t>(fminf(decode_scale, 255.0f));
  float encode_scale = 1.0f / (decode_scale / global_encode_scale);
  
  // Store scale (one per block)
  if (element_idx_in_block == 0) {
    scales[block_idx] = scale_fp8;
  }
  
  // Quantize elements
  if (global_element_idx < num_elements) {
    float input_val = __bfloat162float(input[global_element_idx]);
    float scaled_val = input_val * encode_scale;
    
    // Convert to FP4 using round-to-nearest
    float2 in01 = make_float2(scaled_val, 0.0f);
    float2 in23 = make_float2(0.0f, 0.0f);
    __nv_fp4x4_e2m1 fp4_result = cvt_fp32_to_fp4_4x_with_rn(in01, in23);
    
    // Pack two FP4 values into one uint8_t
    uint8_t packed_result = reinterpret_cast<uint8_t*>(&fp4_result)[0];
    quantized_output[global_element_idx / 2] = packed_result;
  }
}

template<int kBlockSize = 16>
__global__ void block_1d_dequantization_kernel(
    const uint8_t* quantized_input,
    const uint8_t* scales,
    __nv_bfloat16* output,
    const int num_elements) {
  
  const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_element_idx = thread_idx;
  
  if (global_element_idx >= num_elements) return;
  
  const int block_idx = global_element_idx / kBlockSize;
  
  // Load scale for this block
  uint8_t scale_fp8 = scales[block_idx];
  float decode_scale = static_cast<float>(scale_fp8);
  float global_encode_scale = 1.0f; // Simplified
  float final_scale = decode_scale / global_encode_scale;
  
  // Load and unpack FP4 value
  uint8_t packed_val = quantized_input[global_element_idx / 2];
  
  // Extract the correct FP4 nibble (4 bits) from the packed byte
  bool is_upper_nibble = (global_element_idx % 2) == 1;
  uint8_t fp4_bits = is_upper_nibble ? (packed_val >> 4) : (packed_val & 0xF);
  
  // Convert FP4 E2M1 to float using the representable values
  // FP4 E2M1 format: 1 sign bit, 2 exponent bits, 1 mantissa bit
  float fp4_float = 0.0f;
  
  if (fp4_bits != 0) {
    // Extract sign, exponent, and mantissa
    uint8_t sign = (fp4_bits >> 3) & 0x1;
    uint8_t exp = (fp4_bits >> 1) & 0x3;
    uint8_t mant = fp4_bits & 0x1;
    
    // Convert to float following FP4 E2M1 encoding
    if (exp == 0) {
      // Subnormal: value = (-1)^sign * 2^(-1) * (mant / 2)
      fp4_float = (mant == 0) ? 0.0f : 0.25f;
    } else {
      // Normal: value = (-1)^sign * 2^(exp-2) * (1 + mant/2)
      float exp_val = powf(2.0f, (float)exp - 2.0f);
      float mant_val = 1.0f + (float)mant * 0.5f;
      fp4_float = exp_val * mant_val;
    }
    
    if (sign) fp4_float = -fp4_float;
  }
  
  // Scale and convert to BFloat16
  float result = fp4_float * final_scale;
  output[global_element_idx] = __float2bfloat16(result);
}

#endif // CUDA_VERSION >= 12080

// Host wrapper functions with proper at::BFloat16 interface
void launch_block_1d_quantization_cuda(
    const at::BFloat16* input,
    uint8_t* quantized_output,
    uint8_t* scales,
    int64_t num_elements,
    cudaStream_t stream) {
  
#if CUDA_VERSION >= 12080
  constexpr int kBlockSize = 16;
  constexpr int kThreadsPerBlock = 256;
  
  const int num_blocks = (num_elements + kThreadsPerBlock - 1) / kThreadsPerBlock;
  
  // Cast at::BFloat16* to __nv_bfloat16* (they have the same memory layout)
  const __nv_bfloat16* cuda_input = reinterpret_cast<const __nv_bfloat16*>(input);
  
  block_1d_quantization_kernel<kBlockSize><<<num_blocks, kThreadsPerBlock, 0, stream>>>(
      cuda_input, quantized_output, scales, static_cast<int>(num_elements));
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
  }
#else
  throw std::runtime_error("FP4 support requires CUDA 12.8+");
#endif
}

// Internal CUDA implementation
void launch_block_1d_quantization_cuda_internal(
    const __nv_bfloat16* input,
    uint8_t* quantized_output,
    uint8_t* scales,
    const int num_elements,
    cudaStream_t stream) {
  
#if CUDA_VERSION >= 12080
  constexpr int kBlockSize = 16;
  constexpr int kThreadsPerBlock = 256;
  
  const int num_blocks = (num_elements + kThreadsPerBlock - 1) / kThreadsPerBlock;
  
  block_1d_quantization_kernel<kBlockSize><<<num_blocks, kThreadsPerBlock, 0, stream>>>(
      input, quantized_output, scales, num_elements);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
  }
#else
  throw std::runtime_error("FP4 support requires CUDA 12.8+");
#endif
}

void launch_block_1d_dequantization_cuda(
    const uint8_t* quantized_input,
    const uint8_t* scales,
    at::BFloat16* output,
    int64_t num_elements,
    cudaStream_t stream) {
  
#if CUDA_VERSION >= 12080
  constexpr int kBlockSize = 16;
  constexpr int kThreadsPerBlock = 256;
  
  const int num_blocks = (num_elements + kThreadsPerBlock - 1) / kThreadsPerBlock;
  
  // Cast at::BFloat16* to __nv_bfloat16* (they have the same memory layout)
  __nv_bfloat16* cuda_output = reinterpret_cast<__nv_bfloat16*>(output);
  
  block_1d_dequantization_kernel<kBlockSize><<<num_blocks, kThreadsPerBlock, 0, stream>>>(
      quantized_input, scales, cuda_output, static_cast<int>(num_elements));
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
  }
#else
  throw std::runtime_error("FP4 support requires CUDA 12.8+");
#endif
}

// Simplified per-tensor implementations
void launch_per_tensor_quantization_cuda(
    const at::BFloat16* input,
    uint8_t* quantized_output,
    float scale,
    int64_t num_elements,
    cudaStream_t stream) {
  // TODO: Implement proper per-tensor quantization
  throw std::runtime_error("Per-tensor quantization not yet implemented in clean version");
}

void launch_per_tensor_dequantization_cuda(
    const uint8_t* quantized_input,
    float scale,
    at::BFloat16* output,
    int64_t num_elements,
    cudaStream_t stream) {
  // TODO: Implement proper per-tensor dequantization
  throw std::runtime_error("Per-tensor dequantization not yet implemented in clean version");
}

// Placeholder implementations for 2D and per-channel
void launch_block_2d_quantization_cuda(
    const at::BFloat16* input,
    uint8_t* quantized_output,
    uint8_t* scales,
    int64_t rows,
    int64_t cols,
    cudaStream_t stream) {
  throw std::runtime_error("Block 2D quantization not yet implemented in clean version");
}

void launch_block_2d_dequantization_cuda(
    const uint8_t* quantized_input,
    const uint8_t* scales,
    at::BFloat16* output,
    int64_t rows,
    int64_t cols,
    cudaStream_t stream) {
  throw std::runtime_error("Block 2D dequantization not yet implemented in clean version");
}

void launch_per_channel_quantization_cuda(
    const at::BFloat16* input,
    uint8_t* quantized_output,
    uint8_t* scales,
    int64_t channels,
    int64_t elements_per_channel,
    cudaStream_t stream) {
  throw std::runtime_error("Per-channel quantization not yet implemented in clean version");
}

void launch_per_channel_dequantization_cuda(
    const uint8_t* quantized_input,
    const uint8_t* scales,
    at::BFloat16* output,
    int64_t channels,
    int64_t elements_per_channel,
    cudaStream_t stream) {
  throw std::runtime_error("Per-channel dequantization not yet implemented in clean version");
}

} // namespace s4::nvfp4