/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 * 
 * Adapted from TransformerEngine/transformer_engine/common/transpose/quantize_transpose_vector_blockwise_fp4.cu
 * for S4 NVFP4 implementation
 ************************************************************************/

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

#include <algorithm>
#include <cfloat>
#include <utility>

#include "quantization.cuh"

namespace s4::nvfp4 {

// Adapted from TransformerEngine TypeExtrema
template<typename T>
struct TypeExtrema;

template<>
struct TypeExtrema<__nv_fp4_e2m1> {
  static constexpr float max = 6.0f;
};

template<>
struct TypeExtrema<__nv_fp8_e4m3> {
  static constexpr float max = 448.0f;
};

template<>
struct TypeExtrema<float> {
  static constexpr float max = FLT_MAX;
};

// Adapted FP4 quantization functions from TransformerEngine
template <typename ScaleType>
__device__ __forceinline__ ScaleType ComputeDecodeScaleFP4(const float amax,
                                                           const float global_encode_scale) {
  float decode_scale = amax / TypeExtrema<__nv_fp4_e2m1>::max;
  decode_scale = decode_scale * global_encode_scale;
  decode_scale = fminf(decode_scale, TypeExtrema<float>::max);
  return static_cast<ScaleType>(decode_scale);
}

template <typename ScaleType>
__device__ __forceinline__ float ComputeEncodeScaleFP4(ScaleType decode_scale,
                                                       const float global_decode_scale) {
  return fminf(1.0f / (static_cast<float>(decode_scale) * global_decode_scale),
               TypeExtrema<float>::max);
}

__device__ __forceinline__ float ComputeGlobalEncodeScaleFP4(const float global_amax) {
  constexpr float fp8_max = TypeExtrema<__nv_fp8_e4m3>::max;
  constexpr float fp4_max = TypeExtrema<__nv_fp4_e2m1>::max;
  float global_encode_scale = fp8_max * fp4_max / global_amax;
  // If scale is infinity, return max value of float32
  global_encode_scale = fminf(global_encode_scale, TypeExtrema<float>::max);
  // If global amax is 0 or infinity, return 1
  if (global_amax == 0.f || global_encode_scale == 0.f) {
    return 1.f;
  }
  return global_encode_scale;
}

// Simplified FP4 conversion without stochastic rounding
__device__ __forceinline__ uint8_t convert_fp32_to_fp4_simple(float val) {
  // Clamp to FP4 range
  val = fminf(fmaxf(val, -TypeExtrema<__nv_fp4_e2m1>::max), TypeExtrema<__nv_fp4_e2m1>::max);
  
  // Simple linear quantization to 4 bits (0-15)
  // Map [-6, 6] to [0, 15]
  uint8_t quantized = static_cast<uint8_t>((val + 6.0f) * (15.0f / 12.0f)) & 0x0F;
  return quantized;
}

__device__ __forceinline__ float convert_fp4_to_fp32_simple(uint8_t fp4_val) {
  // Map [0, 15] back to [-6, 6]
  float val = (fp4_val * (12.0f / 15.0f)) - 6.0f;
  return val;
}

// Block 1D quantization kernel adapted from TransformerEngine
__global__ void block_1d_quantization_kernel_te(const __nv_bfloat16* input, uint8_t* output,
                                              uint8_t* scale_factors, int64_t num_elements,
                                              float global_amax) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_blocks = (num_elements + 15) / 16;  // 16-element blocks

  if (tid >= num_blocks) return;

  const int block_start = tid * 16;
  const int block_end = min(block_start + 16, (int)num_elements);

  // Find block maximum
  float amax = 0.0f;
  for (int i = block_start; i < block_end; i++) {
    float val = fabsf(__bfloat162float(input[i]));
    amax = fmaxf(amax, val);
  }

  // Compute scales using TE methodology
  float global_encode_scale = ComputeGlobalEncodeScaleFP4(global_amax);
  __nv_fp8_e4m3 scale_inv = ComputeDecodeScaleFP4<__nv_fp8_e4m3>(amax, global_encode_scale);
  float encode_scale = ComputeEncodeScaleFP4<__nv_fp8_e4m3>(scale_inv, 1.0f / global_encode_scale);

  // Store scale as FP8 E4M3
  scale_factors[tid] = *reinterpret_cast<uint8_t*>(&scale_inv);

  // Quantize elements in pairs
  for (int i = block_start; i < block_end; i += 2) {
    float val0 = 0.0f, val1 = 0.0f;

    if (i < num_elements) {
      val0 = __bfloat162float(input[i]) * encode_scale;
    }
    if (i + 1 < num_elements) {
      val1 = __bfloat162float(input[i + 1]) * encode_scale;
    }

    // Pack two FP4 values into one byte
    uint8_t fp4_0 = convert_fp32_to_fp4_simple(val0);
    uint8_t fp4_1 = convert_fp32_to_fp4_simple(val1);
    
    output[i / 2] = (fp4_1 << 4) | fp4_0;
  }
}

// Block 1D dequantization kernel adapted from TransformerEngine
__global__ void block_1d_dequantization_kernel_te(const uint8_t* input, const uint8_t* scales,
                                                __nv_bfloat16* output, int64_t num_elements,
                                                float global_amax) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int element_idx = tid * 2;

  if (element_idx >= num_elements) return;

  const int block_idx = element_idx / 16;
  
  // Get scale factor and compute decode scale
  __nv_fp8_e4m3 scale_inv = *reinterpret_cast<const __nv_fp8_e4m3*>(&scales[block_idx]);
  float global_encode_scale = ComputeGlobalEncodeScaleFP4(global_amax);
  float decode_scale = static_cast<float>(scale_inv) / global_encode_scale;

  // Get packed FP4 values
  uint8_t packed = input[tid];
  uint8_t fp4_0 = packed & 0x0F;
  uint8_t fp4_1 = (packed >> 4) & 0x0F;

  // Dequantize
  float val0 = convert_fp4_to_fp32_simple(fp4_0) * decode_scale;
  float val1 = convert_fp4_to_fp32_simple(fp4_1) * decode_scale;

  // Store
  if (element_idx < num_elements) {
    output[element_idx] = __float2bfloat16(val0);
  }
  if (element_idx + 1 < num_elements) {
    output[element_idx + 1] = __float2bfloat16(val1);
  }
}

// Launch functions with TE-style interface
void launch_block_1d_quantization_cuda_te(const at::BFloat16* input, uint8_t* output, 
                                        uint8_t* scales, int64_t num_elements,
                                        float global_amax, cudaStream_t stream) {
  const int threads = 256;
  const int blocks = ((num_elements + 15) / 16 + threads - 1) / threads;

  block_1d_quantization_kernel_te<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input), output, scales, num_elements, global_amax);
}

void launch_block_1d_dequantization_cuda_te(const uint8_t* input, const uint8_t* scales,
                                          at::BFloat16* output, int64_t num_elements,
                                          float global_amax, cudaStream_t stream) {
  const int threads = 256;
  const int blocks = ((num_elements + 1) / 2 + threads - 1) / threads;

  block_1d_dequantization_kernel_te<<<blocks, threads, 0, stream>>>(
      input, scales, reinterpret_cast<__nv_bfloat16*>(output), num_elements, global_amax);
}

}  // namespace s4::nvfp4