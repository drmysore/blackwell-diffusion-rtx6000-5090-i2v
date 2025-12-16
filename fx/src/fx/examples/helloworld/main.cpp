#include <print>

#include <spdlog/spdlog.h>

auto main(int argc, char* argv[]) -> int {
  std::println("hello, world...");
  spdlog::info("hello, `spdlog`...");

  return 0;
}
