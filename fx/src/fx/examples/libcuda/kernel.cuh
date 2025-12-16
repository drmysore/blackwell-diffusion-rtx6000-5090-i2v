#pragma once

#include <torch/torch.h>

void cutlass_gemm(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor* C);
