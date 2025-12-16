#include <chrono>
#include <iostream>
#include <numeric>

#include <spdlog/spdlog.h>
#include <torch/torch.h>

int main() {
  // spdlog
  spdlog::info("Starting PyTorch example with spdlog");

  // torch version
  spdlog::info("[fx] [torch] version: {}", TORCH_VERSION);

  const auto cuda_available = torch::cuda::is_available();
  spdlog::info("[fx] [torch] [cuda] available: {}", cuda_available);

  if (cuda_available) {
    const auto device_count = torch::cuda::device_count();
    spdlog::info("[fx] [torch] [cuda] devices: {}", device_count);
  }

  // random tensor
  const auto tensor = torch::randn({2, 3});
  spdlog::info("Created CPU tensor with shape: [{}, {}]", tensor.size(0), tensor.size(1));
  std::cout << "CPU Tensor:\n" << tensor << std::endl;

  // tensor operations
  const auto squared = tensor.pow(2);
  const auto sum = squared.sum();
  spdlog::info("Sum of squared elements: {}", sum.item<float>());

  // matrix operations
  const auto mat1 = torch::randn({3, 4});
  const auto mat2 = torch::randn({4, 5});
  const auto result = torch::matmul(mat1, mat2);
  spdlog::info("Matrix multiplication result shape: [{}, {}]", result.size(0), result.size(1));

  if (cuda_available) {
    spdlog::info("Moving tensors to CUDA...");

    const auto cuda_tensor = tensor.to(torch::kCUDA);
    spdlog::info("CUDA Tensor device: cuda:{}", cuda_tensor.get_device());

    // Benchmark CPU vs GPU
    constexpr auto size = 1000;
    const auto cpu_a = torch::randn({size, size});
    const auto cpu_b = torch::randn({size, size});

    const auto start_cpu = std::chrono::high_resolution_clock::now();
    [[maybe_unused]] const auto cpu_result = torch::matmul(cpu_a, cpu_b);
    const auto end_cpu = std::chrono::high_resolution_clock::now();

    const auto cuda_a = cpu_a.to(torch::kCUDA);
    const auto cuda_b = cpu_b.to(torch::kCUDA);

    // Warm-up
    [[maybe_unused]] const auto warmup = torch::matmul(cuda_a, cuda_b);
    torch::cuda::synchronize();

    const auto start_cuda = std::chrono::high_resolution_clock::now();
    [[maybe_unused]] const auto cuda_result = torch::matmul(cuda_a, cuda_b);
    torch::cuda::synchronize();
    const auto end_cuda = std::chrono::high_resolution_clock::now();

    const auto cpu_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu);
    const auto cuda_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end_cuda - start_cuda);

    spdlog::info("Matrix multiplication {}x{}", size, size);
    spdlog::info("  CPU time: {} μs", cpu_duration.count());
    spdlog::info("  CUDA time: {} μs", cuda_duration.count());
    spdlog::info("  Speedup: {:.2f}x",
                 static_cast<double>(cpu_duration.count()) / cuda_duration.count());
  } else {
    spdlog::warn("CUDA not available, skipping GPU tests");
  }

  // neural network example
  spdlog::info("Creating a simple neural network...");

  struct Net : torch::nn::Module {
    Net() {
      mlp_1 = register_module("mlp_1", torch::nn::Linear(784, 128));
      mlp_2 = register_module("mlp_2", torch::nn::Linear(128, 64));
      mlp_3 = register_module("mlp_3", torch::nn::Linear(64, 10));
    }

    auto forward(torch::Tensor x) -> torch::Tensor {
      x = torch::relu(mlp_1->forward(x));
      x = torch::relu(mlp_2->forward(x));
      x = mlp_3->forward(x);

      return torch::log_softmax(x, /*dim=*/1);
    }

    torch::nn::Linear mlp_1{nullptr}, mlp_2{nullptr}, mlp_3{nullptr};
  };

  auto net = std::make_shared<Net>();
  int64_t param_count = 0;

  for (const auto& p : net->parameters()) {
    param_count += p.numel();
  }

  spdlog::info("Network created with {} parameters", param_count);


  if (cuda_available) {
    net->to(torch::kCUDA);
    spdlog::info("Network moved to CUDA");
  }

  auto input = torch::randn({32, 784});
  if (cuda_available) {
    input = input.to(torch::kCUDA);
  }

  const auto output = net->forward(input);
  spdlog::info("[fx] [torch] forward pass complete. output shape: [{}, {}]", output.size(0),
               output.size(1));

  spdlog::info("[fx] [torch] example completed successfully...");

  return 0;
}
