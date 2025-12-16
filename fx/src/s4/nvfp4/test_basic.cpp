#define CATCH_CONFIG_MAIN
#include <torch/torch.h>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Basic torch test", "[basic]") {
  // Test that torch is working
  auto tensor = torch::ones({2, 2});
  REQUIRE(tensor.numel() == 4);
  
  if (torch::cuda::is_available()) {
    INFO("CUDA is available");
    auto cuda_tensor = torch::ones({2, 2}, torch::kCUDA);
    REQUIRE(cuda_tensor.device().is_cuda());
  } else {
    INFO("CUDA not available - tests will be limited");
  }
}