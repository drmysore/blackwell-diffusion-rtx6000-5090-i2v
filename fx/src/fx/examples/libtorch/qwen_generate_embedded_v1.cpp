#include <chrono>
#include <cmath>
#include <cstddef>
#include <print>

#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <spdlog/spdlog.h>
#include <torch/extension.h>
#include <torch/python.h>
#include <torch/torch.h>

namespace py = pybind11;

auto main(int argc, char* argv[]) -> int {
  std::println("// fx // crude python embed // qwen test...");

  ::setenv("PYTHONPATH", PYTHON_SITE_PACKAGES, 1);
  ::setenv("LD_LIBRARY_PATH", TORCH_LIB_PATH, 1);

  auto interpreter_guard = py::scoped_interpreter{};

  try {
    auto total_start = std::chrono::high_resolution_clock::now();

    spdlog::info("[fx] [torch] importing modules...");

    auto math = py::module_::import("math");
    auto torch = py::module_::import("torch");
    auto diffusers = py::module_::import("diffusers");
    auto nunchaku = py::module_::import("nunchaku");

    auto FlowMatchEulerDiscreteScheduler = diffusers.attr("FlowMatchEulerDiscreteScheduler");

    auto QwenImagePipeline = diffusers.attr("QwenImagePipeline");

    auto NunchakuQwenImageTransformer2DModel = nunchaku.attr("models")
                                                   .attr("transformers")
                                                   .attr("transformer_qwenimage")
                                                   .attr("NunchakuQwenImageTransformer2DModel");

    // Scheduler config
    auto scheduler_config = py::dict{};
    scheduler_config["base_image_seq_len"] = 256;
    scheduler_config["base_shift"] = std::log(3.0);
    scheduler_config["max_image_seq_len"] = 8192;
    scheduler_config["max_shift"] = std::log(3.0);
    scheduler_config["num_train_timesteps"] = 1000;
    scheduler_config["shift"] = 1.0;
    scheduler_config["use_dynamic_shifting"] = true;

    auto scheduler = FlowMatchEulerDiscreteScheduler.attr("from_config")(scheduler_config);

    std::string precision = nunchaku.attr("utils").attr("get_precision")().cast<std::string>();

    std::string model_path = std::format(
        "nunchaku-tech/nunchaku-qwen-image/svdq-{}_r32-qwen-image-lightningv1.0-4steps.safetensors",
        precision);

    spdlog::info("[fx] [torch] loading transformer: {}", model_path);

    auto transformer = NunchakuQwenImageTransformer2DModel.attr("from_pretrained")(model_path);
    auto pipe = QwenImagePipeline.attr("from_pretrained")(
        "Qwen/Qwen-Image", py::arg("transformer") = transformer, py::arg("scheduler") = scheduler,
        py::arg("torch_dtype") = torch.attr("bfloat16"));

    if (torch::cuda::is_available()) {
      pipe = pipe.attr("to")("cuda");
    }

    auto setup_end = std::chrono::high_resolution_clock::now();

    auto setup_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(setup_end - total_start).count();

    spdlog::info("[fx] [torch] setup time: {}ms", setup_ms);
    const auto prompt = std::string{"A cyberpunk cat wearing sunglasses in neon city"};
    const auto negative_prompt = std::string{"blurry, low quality"};

    for (std::size_t idx = 0; idx < 5; ++idx) {
      auto gen_start = std::chrono::high_resolution_clock::now();

      auto result = pipe(py::arg("prompt") = prompt, py::arg("negative_prompt") = negative_prompt,
                         py::arg("width") = 1024, py::arg("height") = 1024,
                         py::arg("num_inference_steps") = 4, py::arg("true_cfg_scale") = 1.0);

      torch::cuda::synchronize();

      auto gen_end = std::chrono::high_resolution_clock::now();
      auto gen_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(gen_end - gen_start).count();

      auto images = result.attr("images");
      auto image = images[py::int_(0)];

      std::string filename = std::format("/tmp/qwen-{}.png", idx);
      image.attr("save")(filename);

      spdlog::info("[fx] [torch] image {}: {}ms - saved {}", idx, gen_ms, filename);
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();

    spdlog::info("[fx] [torch] total runtime: {}ms", total_ms);
    spdlog::info("[fx] [torch] average_per_image: {}ms", (total_ms - setup_ms) / 5.0);

  } catch (const py::error_already_set& ex) {
    spdlog::critical("[fx] [pybind11] {}", ex.what());
    return 1;
  }

  return 0;
}
