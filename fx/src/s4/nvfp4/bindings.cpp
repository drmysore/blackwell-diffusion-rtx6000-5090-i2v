#include "tensor.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "kernel.cuh"
#include "quantization.cuh"

namespace py = pybind11;

PYBIND11_MODULE(nvfp4, m) {
  m.doc() = "S4 NVFP4 - Native FP4 quantization for Blackwell SM_120/SM_120a";

  // Module initialization
  m.def("init", []() {
    // Verify SM_120 support
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    if (prop.major < 12) {
      throw std::runtime_error("[s4] [nvfp4] NVFP4 requires SM_120 or higher. "
                               "Current device: SM_" +
                               std::to_string(prop.major) + std::to_string(prop.minor));
    }

    py::print("[s4] [nvfp4] Initialized on", prop.name, "with compute capability", prop.major, ".",
              prop.minor);
  });

  // Quantization modes enum
  py::enum_<s4::nvfp4::quantization_mode>(m, "QuantizationMode")
      .value("block_1d", s4::nvfp4::quantization_mode::block_1d,
             "1D block quantization (16 elements)")
      .value("block_2d", s4::nvfp4::quantization_mode::block_2d,
             "2D block quantization (16x16 elements)")
      .value("per_tensor", s4::nvfp4::quantization_mode::per_tensor,
             "Single scale for entire tensor")
      .value("per_channel", s4::nvfp4::quantization_mode::per_channel, "Scale per output channel");

  // FP4Tensor class
  py::class_<s4::nvfp4::tensor>(m, "FP4Tensor")
      .def_static("from_bfloat16", &s4::nvfp4::tensor::from_bfloat16,
                  "Create FP4 tensor from BFloat16 tensor with specified quantization mode",
                  py::arg("input"), py::arg("mode") = s4::nvfp4::quantization_mode::block_1d)
      .def("to_bfloat16", &s4::nvfp4::tensor::to_bfloat16, "Dequantize FP4 tensor back to BFloat16")
      .def("quantized_data", &s4::nvfp4::tensor::quantized_data,
           "Get raw quantized FP4 data (packed, 2 values per byte)")
      .def("scale_factors", &s4::nvfp4::tensor::scale_factors, "Get FP8 E4M3 scale factors")
      .def("shape", &s4::nvfp4::tensor::shape, "Get original tensor shape")
      .def("numel", &s4::nvfp4::tensor::numel, "Get number of elements")
      .def("memory_usage", &s4::nvfp4::tensor::memory_usage, "Get memory usage in bytes")
      .def("__repr__", [](const s4::nvfp4::tensor& t) {
        std::stringstream ss;
        ss << "FP4Tensor(shape=";
        auto shape = t.shape();
        ss << "[";
        for (int i = 0; i < shape.size(); ++i) {
          ss << shape[i];
          if (i < shape.size() - 1)
            ss << ", ";
        }
        ss << "], memory=" << t.memory_usage() << " bytes)";
        return ss.str();
      });

  // Low-level GEMM kernels
  m.def("blackwell_fp4_gemm", &s4::nvfp4::blackwell_fp4_gemm,
        "Blackwell FP4 GEMM with automatic quantization\n"
        "Inputs must be BFloat16 2D tensors with dimensions divisible by 128",
        py::arg("A"), py::arg("B"), py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f);

  // High-level operations
  m.def("fp4_gemm", &s4::nvfp4::fp4_gemm, "FP4 GEMM with pre-quantized tensor objects",
        py::arg("A"), py::arg("B"), py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f);

  m.def("gemm_fp4_quantized", &s4::nvfp4::gemm_fp4_quantized,
        "FP4 GEMM with automatic quantization from BFloat16 tensors", py::arg("A"), py::arg("B"),
        py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
        py::arg("mode") = s4::nvfp4::quantization_mode::block_1d);

  // Utility functions
  m.def(
      "is_aligned",
      [](int64_t M, int64_t N, int64_t K) {
        return (M % 128 == 0) && (N % 128 == 0) && (K % 128 == 0);
      },
      "Check if dimensions are aligned for FP4 GEMM (divisible by 128)");

  m.def(
      "get_aligned_dims",
      [](int64_t M, int64_t N, int64_t K) {
        return py::make_tuple(((M + 127) / 128) * 128, ((N + 127) / 128) * 128,
                              ((K + 127) / 128) * 128);
      },
      "Get padded dimensions aligned to 128");

  // Performance metrics
  m.def(
      "benchmark",
      [](const torch::Tensor& A, const torch::Tensor& B, int iterations = 100) {
        // Warmup
        for (int i = 0; i < 10; ++i) {
          s4::nvfp4::blackwell_fp4_gemm(A, B, 1.0f, 0.0f);
        }
        torch::cuda::synchronize();

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
          s4::nvfp4::blackwell_fp4_gemm(A, B, 1.0f, 0.0f);
        }
        torch::cuda::synchronize();
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        float avg_time_ms = duration.count() / 1000.0f / iterations;

        // Calculate TFLOPS
        int64_t M = A.size(0);
        int64_t K = A.size(1);
        int64_t N = B.size(0);
        double flops = 2.0 * M * N * K;
        double tflops = (flops / avg_time_ms) / 1e9;

        return py::make_tuple(avg_time_ms, tflops);
      },
      "Benchmark FP4 GEMM performance", py::arg("A"), py::arg("B"), py::arg("iterations") = 100);

  // Version info
  m.attr("__version__") = "1.0.0";
  m.attr("__cuda_arch__") = "sm_120";
}
