#include <cuda_fp16.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <torch/all.h>

#include "src/fx/examples/libcuda/kernel.cuh"


auto cutlass_gemm(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor* C) -> void {
  TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
  TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");

  TORCH_CHECK(nullptr != C, "C must not be nullptr");
  TORCH_CHECK(C->device().is_cuda(), "C must be a CUDA tensor");

  TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
  TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
  TORCH_CHECK(C->is_contiguous(), "C must be contiguous");

  TORCH_CHECK(A.scalar_type() == torch::kFloat32, "unsupported type");

  using GEMM =
      cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float,
                                  cutlass::layout::RowMajor, float, cutlass::layout::RowMajor>;

  auto gemm = GEMM{};
  cutlass::gemm::GemmCoord problem_size(A.size(0), B.size(1), A.size(1));

  typename GEMM::Arguments args(
      problem_size, {A.data_ptr<float>(), A.size(1)}, {B.data_ptr<float>(), B.size(1)},
      {C->data_ptr<float>(), C->size(1)}, {C->data_ptr<float>(), C->size(1)}, {1.0f, 0.0f});

  const auto status = gemm(args);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM operation failed");
  }
}
