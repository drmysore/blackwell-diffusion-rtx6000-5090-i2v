#pragma once

#include <torch/types.h>

namespace s4::nvfp4 {

// Original kernel - takes BF16 inputs and does internal quantization
auto blackwell_fp4_gemm(torch::Tensor A,  // [M, K] in BF16
                        torch::Tensor B,  // [N, K] in BF16
                        float alpha, float beta) -> torch::Tensor;

// New kernel - takes pre-quantized FP4 data
auto blackwell_fp4_gemm_prequantized(
    torch::Tensor A_fp4,     // [M*K/2] packed FP4 data (2 values per byte)
    torch::Tensor B_fp4,     // [N*K/2] packed FP4 data
    torch::Tensor A_scales,  // FP8 E4M3 scale factors
    torch::Tensor B_scales,  // FP8 E4M3 scale factors
    int64_t M, int64_t N, int64_t K, float alpha, float beta) -> torch::Tensor;

}  // namespace s4::nvfp4
