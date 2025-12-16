#include <cuda_fp8.h>

#include "quantization.cuh"

namespace s4::nvfp4 {

constexpr float FP4_E2M1_MAX = 6.0f;
constexpr int BLOCK_SIZE = 16;

// Kernel for block 1D quantization
__global__ void block_1d_quantization_kernel(const __nv_bfloat16* input, uint8_t* output,
                                             uint8_t* scale_factors, int64_t num_elements) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

  if (tid >= num_blocks)
    return;

  const int block_start = tid * BLOCK_SIZE;
  const int block_end = min(block_start + BLOCK_SIZE, (int)num_elements);

  // Find block maximum
  float max_abs = 0.0f;
  for (int i = block_start; i < block_end; i++) {
    float val = fabsf(__bfloat162float(input[i]));
    max_abs = fmaxf(max_abs, val);
  }

  // Compute scale
  float scale = max_abs / FP4_E2M1_MAX;
  if (scale == 0.0f)
    scale = 1.0f;
  float inv_scale = 1.0f / scale;

  // Store scale as FP8 E4M3
  __nv_fp8_e4m3 scale_fp8 = __nv_fp8_e4m3(scale);
  scale_factors[tid] = *reinterpret_cast<uint8_t*>(&scale_fp8);

  // Quantize elements
  for (int i = block_start; i < block_end; i += 2) {
    float val0 = 0.0f, val1 = 0.0f;

    if (i < num_elements) {
      val0 = __bfloat162float(input[i]) * inv_scale;
      val0 = fminf(fmaxf(val0, -FP4_E2M1_MAX), FP4_E2M1_MAX);
    }

    if (i + 1 < num_elements) {
      val1 = __bfloat162float(input[i + 1]) * inv_scale;
      val1 = fminf(fmaxf(val1, -FP4_E2M1_MAX), FP4_E2M1_MAX);
    }

    // Pack two FP4 values into one byte
    uint8_t fp4_0 = static_cast<uint8_t>((val0 + 6.0f) * (15.0f / 12.0f)) & 0x0F;
    uint8_t fp4_1 = static_cast<uint8_t>((val1 + 6.0f) * (15.0f / 12.0f)) & 0x0F;

    output[i / 2] = (fp4_1 << 4) | fp4_0;
  }
}

// Kernel for block 1D dequantization
__global__ void block_1d_dequantization_kernel(const uint8_t* input, const uint8_t* scale_factors,
                                               __nv_bfloat16* output, int64_t num_elements) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int element_idx = tid * 2;

  if (element_idx >= num_elements)
    return;

  const int block_idx = element_idx / BLOCK_SIZE;

  // Get scale factor
  __nv_fp8_e4m3 scale_fp8 = *reinterpret_cast<const __nv_fp8_e4m3*>(&scale_factors[block_idx]);
  float scale = static_cast<float>(scale_fp8);

  // Get packed FP4 values
  uint8_t packed = input[tid];
  uint8_t fp4_0 = packed & 0x0F;
  uint8_t fp4_1 = (packed >> 4) & 0x0F;

  // Dequantize
  float val0 = (fp4_0 * (12.0f / 15.0f) - 6.0f) * scale;
  float val1 = (fp4_1 * (12.0f / 15.0f) - 6.0f) * scale;

  // Store
  if (element_idx < num_elements) {
    output[element_idx] = __float2bfloat16(val0);
  }
  if (element_idx + 1 < num_elements) {
    output[element_idx + 1] = __float2bfloat16(val1);
  }
}

// Per-tensor quantization kernel
__global__ void per_tensor_quantization_kernel(const __nv_bfloat16* input, uint8_t* output,
                                               float inv_scale, int64_t num_elements) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int element_idx = tid * 2;

  if (element_idx >= num_elements)
    return;

  float val0 = 0.0f, val1 = 0.0f;

  if (element_idx < num_elements) {
    val0 = __bfloat162float(input[element_idx]) * inv_scale;
    val0 = fminf(fmaxf(val0, -FP4_E2M1_MAX), FP4_E2M1_MAX);
  }

  if (element_idx + 1 < num_elements) {
    val1 = __bfloat162float(input[element_idx + 1]) * inv_scale;
    val1 = fminf(fmaxf(val1, -FP4_E2M1_MAX), FP4_E2M1_MAX);
  }

  // Pack two FP4 values into one byte
  uint8_t fp4_0 = static_cast<uint8_t>((val0 + 6.0f) * (15.0f / 12.0f)) & 0x0F;
  uint8_t fp4_1 = static_cast<uint8_t>((val1 + 6.0f) * (15.0f / 12.0f)) & 0x0F;

  output[tid] = (fp4_1 << 4) | fp4_0;
}

// Per-tensor dequantization kernel
__global__ void per_tensor_dequantization_kernel(const uint8_t* input, float scale,
                                                 __nv_bfloat16* output, int64_t num_elements) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int element_idx = tid * 2;

  if (element_idx >= num_elements)
    return;

  // Get packed FP4 values
  uint8_t packed = input[tid];
  uint8_t fp4_0 = packed & 0x0F;
  uint8_t fp4_1 = (packed >> 4) & 0x0F;

  // Dequantize
  float val0 = (fp4_0 * (12.0f / 15.0f) - 6.0f) * scale;
  float val1 = (fp4_1 * (12.0f / 15.0f) - 6.0f) * scale;

  // Store
  if (element_idx < num_elements) {
    output[element_idx] = __float2bfloat16(val0);
  }
  if (element_idx + 1 < num_elements) {
    output[element_idx + 1] = __float2bfloat16(val1);
  }
}

// Launch functions
void launch_block_1d_quantization_cuda(const at::BFloat16* input, uint8_t* output, uint8_t* scales,
                                       int64_t num_elements, cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (num_elements / BLOCK_SIZE + threads - 1) / threads;

  block_1d_quantization_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input), output, scales, num_elements);
}

void launch_block_1d_dequantization_cuda(const uint8_t* input, const uint8_t* scales,
                                         at::BFloat16* output, int64_t num_elements,
                                         cudaStream_t stream) {
  const int threads = 256;
  const int blocks = ((num_elements + 1) / 2 + threads - 1) / threads;

  block_1d_dequantization_kernel<<<blocks, threads, 0, stream>>>(
      input, scales, reinterpret_cast<__nv_bfloat16*>(output), num_elements);
}

void launch_per_tensor_quantization_cuda(const at::BFloat16* input, uint8_t* output, float scale,
                                         int64_t num_elements, cudaStream_t stream) {
  float inv_scale = 1.0f / scale;
  const int threads = 256;
  const int blocks = ((num_elements + 1) / 2 + threads - 1) / threads;

  per_tensor_quantization_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input), output, inv_scale, num_elements);
}

void launch_per_tensor_dequantization_cuda(const uint8_t* input, float scale, at::BFloat16* output,
                                           int64_t num_elements, cudaStream_t stream) {
  const int threads = 256;
  const int blocks = ((num_elements + 1) / 2 + threads - 1) / threads;

  per_tensor_dequantization_kernel<<<blocks, threads, 0, stream>>>(
      input, scale, reinterpret_cast<__nv_bfloat16*>(output), num_elements);
}

// Block 2D quantization kernel
__global__ void block_2d_quantization_kernel(const __nv_bfloat16* input, uint8_t* output,
                                             uint8_t* scale_factors, int64_t rows, int64_t cols) {
  const int block_row = blockIdx.y;
  const int block_col = blockIdx.x;
  const int tid = threadIdx.x;

  const int blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int blocks_per_col = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;

  if (block_row >= blocks_per_col || block_col >= blocks_per_row)
    return;

  const int row_start = block_row * BLOCK_SIZE;
  const int row_end = min(row_start + BLOCK_SIZE, (int)rows);
  const int col_start = block_col * BLOCK_SIZE;
  const int col_end = min(col_start + BLOCK_SIZE, (int)cols);

  // Find block maximum (parallel reduction would be better for large blocks)
  float max_abs = 0.0f;
  for (int r = row_start; r < row_end; r++) {
    for (int c = col_start; c < col_end; c++) {
      float val = fabsf(__bfloat162float(input[r * cols + c]));
      max_abs = fmaxf(max_abs, val);
    }
  }

  // Compute scale
  float scale = max_abs / FP4_E2M1_MAX;
  if (scale == 0.0f)
    scale = 1.0f;
  float inv_scale = 1.0f / scale;

  // Store scale as FP8 E4M3
  if (tid == 0) {
    __nv_fp8_e4m3 scale_fp8 = __nv_fp8_e4m3(scale);
    scale_factors[block_row * blocks_per_row + block_col] = *reinterpret_cast<uint8_t*>(&scale_fp8);
  }

  __syncthreads();

  // Quantize elements in the block
  for (int r = row_start; r < row_end; r++) {
    for (int c = col_start; c < col_end; c += 2) {
      int idx = r * cols + c;
      float val0 = 0.0f, val1 = 0.0f;

      if (c < cols) {
        val0 = __bfloat162float(input[idx]) * inv_scale;
        val0 = fminf(fmaxf(val0, -FP4_E2M1_MAX), FP4_E2M1_MAX);
      }

      if (c + 1 < cols) {
        val1 = __bfloat162float(input[idx + 1]) * inv_scale;
        val1 = fminf(fmaxf(val1, -FP4_E2M1_MAX), FP4_E2M1_MAX);
      }

      uint8_t fp4_0 = static_cast<uint8_t>((val0 + 6.0f) * (15.0f / 12.0f)) & 0x0F;
      uint8_t fp4_1 = static_cast<uint8_t>((val1 + 6.0f) * (15.0f / 12.0f)) & 0x0F;

      output[idx / 2] = (fp4_1 << 4) | fp4_0;
    }
  }
}

void launch_block_2d_quantization_cuda(const at::BFloat16* input, uint8_t* output, uint8_t* scales,
                                       int64_t rows, int64_t cols, cudaStream_t stream) {
  dim3 blocks((cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 threads(32);  // One warp per block for now

  block_2d_quantization_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input), output, scales, rows, cols);
}

// Block 2D dequantization kernel (stub implementation)
__global__ void block_2d_dequantization_kernel(const uint8_t* input, const uint8_t* scales,
                                               __nv_bfloat16* output, int64_t rows, int64_t cols) {
  // Similar to block_1d_dequantization but with 2D blocks
  // For now, use 1D approach as fallback
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int element_idx = tid * 2;

  if (element_idx >= rows * cols) return;

  const int block_idx = element_idx / BLOCK_SIZE;
  
  // Get scale factor
  __nv_fp8_e4m3 scale_fp8 = *reinterpret_cast<const __nv_fp8_e4m3*>(&scales[block_idx]);
  float scale = static_cast<float>(scale_fp8);

  // Get packed FP4 values
  uint8_t packed = input[tid];
  uint8_t fp4_0 = packed & 0x0F;
  uint8_t fp4_1 = (packed >> 4) & 0x0F;

  // Dequantize
  float val0 = (fp4_0 * (12.0f / 15.0f) - 6.0f) * scale;
  float val1 = (fp4_1 * (12.0f / 15.0f) - 6.0f) * scale;

  // Store
  if (element_idx < rows * cols) {
    output[element_idx] = __float2bfloat16(val0);
  }
  if (element_idx + 1 < rows * cols) {
    output[element_idx + 1] = __float2bfloat16(val1);
  }
}

void launch_block_2d_dequantization_cuda(const uint8_t* input, const uint8_t* scales,
                                         at::BFloat16* output, int64_t rows, int64_t cols,
                                         cudaStream_t stream) {
  const int threads = 256;
  const int blocks = ((rows * cols + 1) / 2 + threads - 1) / threads;

  block_2d_dequantization_kernel<<<blocks, threads, 0, stream>>>(
      input, scales, reinterpret_cast<__nv_bfloat16*>(output), rows, cols);
}

// Per-channel quantization kernel (stub implementation)
__global__ void per_channel_quantization_kernel(const __nv_bfloat16* input, uint8_t* output,
                                               uint8_t* scale_factors, int64_t channels,
                                               int64_t elements_per_channel) {
  const int channel_idx = blockIdx.x;
  const int tid = threadIdx.x;

  if (channel_idx >= channels) return;

  const int channel_start = channel_idx * elements_per_channel;
  const int channel_end = channel_start + elements_per_channel;

  // Find channel maximum
  float max_abs = 0.0f;
  for (int i = channel_start + tid; i < channel_end; i += blockDim.x) {
    float val = fabsf(__bfloat162float(input[i]));
    max_abs = fmaxf(max_abs, val);
  }

  // Reduce within block
  __shared__ float sdata[256];
  sdata[tid] = max_abs;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    // Compute scale
    float scale = sdata[0] / FP4_E2M1_MAX;
    if (scale == 0.0f) scale = 1.0f;
    float inv_scale = 1.0f / scale;

    // Store scale as FP8 E4M3
    __nv_fp8_e4m3 scale_fp8 = __nv_fp8_e4m3(scale);
    scale_factors[channel_idx] = *reinterpret_cast<uint8_t*>(&scale_fp8);

    // Quantize elements in this channel
    for (int i = channel_start; i < channel_end; i += 2) {
      float val0 = 0.0f, val1 = 0.0f;

      if (i < channel_end) {
        val0 = __bfloat162float(input[i]) * inv_scale;
        val0 = fminf(fmaxf(val0, -FP4_E2M1_MAX), FP4_E2M1_MAX);
      }

      if (i + 1 < channel_end) {
        val1 = __bfloat162float(input[i + 1]) * inv_scale;
        val1 = fminf(fmaxf(val1, -FP4_E2M1_MAX), FP4_E2M1_MAX);
      }

      // Pack two FP4 values into one byte
      uint8_t fp4_0 = static_cast<uint8_t>((val0 + 6.0f) * (15.0f / 12.0f)) & 0x0F;
      uint8_t fp4_1 = static_cast<uint8_t>((val1 + 6.0f) * (15.0f / 12.0f)) & 0x0F;

      output[i / 2] = (fp4_1 << 4) | fp4_0;
    }
  }
}

void launch_per_channel_quantization_cuda(const at::BFloat16* input, uint8_t* output,
                                          uint8_t* scales, int64_t channels,
                                          int64_t elements_per_channel, cudaStream_t stream) {
  const int threads = 256;
  dim3 blocks(channels);

  per_channel_quantization_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input), output, scales, channels, elements_per_channel);
}

// Per-channel dequantization kernel (stub implementation)
__global__ void per_channel_dequantization_kernel(const uint8_t* input, const uint8_t* scales,
                                                  __nv_bfloat16* output, int64_t channels,
                                                  int64_t elements_per_channel) {
  const int channel_idx = blockIdx.x;
  const int tid = threadIdx.x;

  if (channel_idx >= channels) return;

  // Get scale factor for this channel
  __nv_fp8_e4m3 scale_fp8 = *reinterpret_cast<const __nv_fp8_e4m3*>(&scales[channel_idx]);
  float scale = static_cast<float>(scale_fp8);

  const int channel_start = channel_idx * elements_per_channel;
  const int channel_end = channel_start + elements_per_channel;

  // Dequantize elements in this channel
  for (int i = channel_start + tid * 2; i < channel_end; i += blockDim.x * 2) {
    if (i >= channel_end) break;

    // Get packed FP4 values
    uint8_t packed = input[i / 2];
    uint8_t fp4_0 = packed & 0x0F;
    uint8_t fp4_1 = (packed >> 4) & 0x0F;

    // Dequantize
    float val0 = (fp4_0 * (12.0f / 15.0f) - 6.0f) * scale;
    float val1 = (fp4_1 * (12.0f / 15.0f) - 6.0f) * scale;

    // Store
    if (i < channel_end) {
      output[i] = __float2bfloat16(val0);
    }
    if (i + 1 < channel_end) {
      output[i + 1] = __float2bfloat16(val1);
    }
  }
}

void launch_per_channel_dequantization_cuda(const uint8_t* input, const uint8_t* scales,
                                            at::BFloat16* output, int64_t channels,
                                            int64_t elements_per_channel, cudaStream_t stream) {
  const int threads = 256;
  dim3 blocks(channels);

  per_channel_dequantization_kernel<<<blocks, threads, 0, stream>>>(
      input, scales, reinterpret_cast<__nv_bfloat16*>(output), channels, elements_per_channel);
}

}  // namespace s4::nvfp4
