#include "src/fx/examples/nvfp4/gemm_nvfp4_nvfp4_nvidia.cuh"

#include "helper.h"

#include <iostream>

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

auto main(int argc, const char* argv[]) -> int {

  auto props = ::cudaDeviceProp{};
  auto current_device_id = int{};
  CUDA_CHECK(::cudaGetDevice(&current_device_id));

  CUDA_CHECK(::cudaGetDeviceProperties(&props, current_device_id));
  if (!(props.major == 12 && (props.minor == 0 || props.minor == 1))) {
    spdlog::error("[fx] [nvidia] compute capability {}.{} not supported", props.major, props.minor);
    return 1;
  }

  spdlog::info("[fx] [nvidia] compute capability {}.{}", props.major, props.minor);

  auto options = fx::nvidia::examples::options{};
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  [[maybe_unused]] auto rv = launch(options);
  return 0;
}
