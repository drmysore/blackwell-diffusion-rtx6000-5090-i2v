#include <iostream>

#include <cutlass/cutlass.h>

#include "cutlass/util/command_line.h"

namespace fx::nvidia::examples {

struct options {

  bool help;

  int m, n, k;
  float alpha, beta;
  int iterations;

  options() : help(false), m(1024), n(1024), k(1024), alpha(1.f), beta(0.f), iterations(10) {
  }

  auto parse(int argc, char const** args) -> void {
    auto cmd = cutlass::CommandLine{argc, args};

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);
  }

  std::ostream& print_usage(std::ostream& out) const {

    out << "79b_blackwell_geforce_nvfp4_nvfp4_gemm\n\n"
        << "  Blackwell NVFP4 GEMM using a Warp Specialized kernel.\n\n"
        << "Options:\n\n"
        << "  --help                      If specified, displays this usage statement\n\n"
        << "  --m=<int>                   Sets the M extent of the GEMM\n"
        << "  --n=<int>                   Sets the N extent of the GEMM\n"
        << "  --k=<int>                   Sets the K extent of the GEMM\n"
        << "  --alpha=<f32>               Epilogue scalar alpha\n"
        << "  --beta=<f32>                Epilogue scalar beta\n\n"
        << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out << "\n\nExamples:\n\n"
        << "$ " << "./examples/79_blackwell_geforce_gemm/79b_blackwell_geforce_nvfp4_nvfp4_gemm"
        << " --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  auto gflops(double runtime_s) const -> double {
    // n.b. two flops per multiply-add
    std::uint64_t flop = uint64_t(2) * m * n * k;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

struct result {
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  result(double avg_runtime_ms = 0, double gflops = 0,
         cutlass::Status status = cutlass::Status::kSuccess, cudaError_t error = cudaSuccess)
      : avg_runtime_ms(avg_runtime_ms),
        gflops(gflops),
        status(status),
        error(error),
        passed(false) {
  }
};

auto launch(options options) -> fx::nvidia::examples::result;

}  // namespace fx::nvidia::examples
