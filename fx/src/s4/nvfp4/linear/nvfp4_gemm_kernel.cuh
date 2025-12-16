#pragma once

#include <torch/extension.h>

auto blackwell_fp4_gemm(torch::Tensor A, torch::Tensor B, float alpha, float beta) -> torch::Tensor;
