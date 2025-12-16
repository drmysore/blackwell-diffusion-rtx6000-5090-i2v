#include <chrono>

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include "src/fx/examples/libcuda/kernel.cuh"

auto benchmark = [](auto&& target_function, std::string_view name) {
  for (std::size_t idx = 0; idx < 16; ++idx) {
    target_function();
  }

  torch::cuda::synchronize();

  auto start = std::chrono::high_resolution_clock::now();
  const auto iterations = std::size_t{128};

  for (std::size_t idx = 0; idx < iterations; ++idx) {
    target_function();
  }

  torch::cuda::synchronize();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  spdlog::info("{}: {:.2f} μs/iteration", name, duration / static_cast<double>(iterations));
};

auto main(int argc, char* argv[]) -> int {

  if (!torch::cuda::is_available()) {
    spdlog::critical("CUDA is not available!");
    return 1;
  }

  spdlog::info("[fxy] [cuda] {} devices", torch::cuda::device_count());

  auto props = cudaDeviceProp{};
  cudaGetDeviceProperties(&props, 0);
  spdlog::info("[fxy] [cuda] [capability] {} {}.{}", props.name, props.major, props.minor);

  spdlog::info("[fxy] [cuda] cutlass gemm");
  const auto M = 8192, N = 8192, K = 8192;
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);

  auto A = torch::randn({M, K}, options);
  auto B = torch::randn({K, N}, options);
  auto C = torch::zeros({M, N}, options);

  cutlass_gemm(A, B, &C);
  auto C2 = torch::matmul(A, B);

  auto diff = torch::abs(C - C2);
  auto max_difference = torch::max(diff).item<float>();
  spdlog::info("[fxy] [cuda] maximum difference: {}", max_difference);

  benchmark([&]() { cutlass_gemm(A, B, &C); }, "cutlass");
  benchmark([&]() { torch::matmul(A, B); }, "torch/cublas");

  return 0;
}
