#include <chrono>
#include <cstdint>
#include <expected>
#include <format>
#include <print>
#include <ranges>
#include <span>
#include <string_view>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>

namespace fx::inference {

using clock_t = std::chrono::high_resolution_clock;
using duration_t = std::chrono::duration<double, std::milli>;

struct configuration {
  std::string_view model_path;

  std::int64_t batch_size{32};
  std::int64_t warmup_iterations{10};
  std::int64_t benchmark_iterations{1000};
  std::int64_t input_dim{10};
};

struct stats {
  double avg_latency_ms;
  double throughput_samples_per_sec;
  std::string device_name;
};

class model_runner {
public:
  [[nodiscard]] static auto create(std::string_view model_path) noexcept
      -> std::expected<fx::inference::model_runner, std::string> {

    try {
      torch::jit::script::Module module = torch::jit::load(std::string{model_path});
      module.eval();

      return fx::inference::model_runner{std::move(module)};
    } catch (const c10::Error& ex) {
      return std::unexpected{std::format("Failed to load model: {}", ex.what())};
    }
  }

  auto run_inference(torch::Tensor input) -> torch::Tensor {
    torch::NoGradGuard no_grad;
    return module_.forward({input}).toTensor();
  }

  auto benchmark(const fx::inference::configuration& config) -> fx::inference::stats {
    const auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    module_.to(device);

    auto input = torch::randn({config.batch_size, config.input_dim}).to(device);
    for ([[maybe_unused]] auto _ : std::views::iota(0, config.warmup_iterations)) {
      run_inference(input);
    }

    const auto start = clock_t::now();
    for ([[maybe_unused]] auto _ : std::views::iota(0, config.benchmark_iterations)) {
      [[maybe_unused]] auto output = run_inference(input);
    }

    const auto duration = duration_t{clock_t::now() - start};
    return {.avg_latency_ms = duration.count() / config.benchmark_iterations,
            .throughput_samples_per_sec =
                (config.batch_size * config.benchmark_iterations * 1000.0) / duration.count()};
  }

private:
  explicit model_runner(torch::jit::script::Module&& module) : module_{std::move(module)} {
  }

  torch::jit::script::Module module_;
};

[[nodiscard]] auto parse_args(std::span<char*> args)
    -> std::expected<fx::inference::configuration, std::string> {

  if (args.size() != 2) {
    return std::unexpected{std::format("usage: {} <model.pt>", args[0])};
  }

  return fx::inference::configuration{.model_path = args[1]};
}
}  // namespace fx::inference

auto main(int argc, char* argv[]) -> int {

  auto config_result = fx::inference::parse_args({argv, static_cast<std::size_t>(argc)});
  if (!config_result) {
    std::println(stderr, "Error: {}", config_result.error());
    return EXIT_FAILURE;
  }

  const auto config = *config_result;

  auto runner_result = fx::inference::model_runner::create(config.model_path);
  if (!runner_result) {
    std::println(stderr, "Error: {}", runner_result.error());
    return EXIT_FAILURE;
  }

  auto& runner = *runner_result;
  std::println("Loading model from: {}", config.model_path);
  std::println("Batch size: {}", config.batch_size);
  std::println("Benchmark iterations: {}", config.benchmark_iterations);

  const auto stats = runner.benchmark(config);
  std::println("Average latency: {:.3f} ms", stats.avg_latency_ms);
  std::println("Throughput: {:.1f} samples/sec", stats.throughput_samples_per_sec);

  return EXIT_SUCCESS;
}
