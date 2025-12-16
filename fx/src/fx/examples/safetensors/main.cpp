#include <iostream>
#include <print>

#include <nlohmann/json.hpp>

#if !defined(SAFETENSORS_CPP_NO_IMPLEMENTATION)
#define SAFETENSORS_CPP_IMPLEMENTATION
#endif

#include "safetensors.hh"

#define USE_MMAP

using json = nlohmann::json;

auto to_string(safetensors::dtype dtype, const uint8_t* data) noexcept -> std::string {
  switch (dtype) {
    case safetensors::dtype::kBOOL: {
      return std::to_string(data[0] ? 1 : 0);
    } break;
    case safetensors::dtype::kUINT8: {
      return std::to_string(data[0]);
    } break;
    case safetensors::dtype::kINT8: {
      return std::to_string(*reinterpret_cast<const int8_t*>(data));
    } break;
    case safetensors::dtype::kUINT16: {
      return std::to_string(*reinterpret_cast<const uint16_t*>(data));
    } break;
    case safetensors::dtype::kINT16: {
      return std::to_string(*reinterpret_cast<const int16_t*>(data));
    } break;
    case safetensors::dtype::kUINT32: {
      return std::to_string(*reinterpret_cast<const uint32_t*>(data));
    } break;
    case safetensors::dtype::kINT32: {
      return std::to_string(*reinterpret_cast<const int32_t*>(data));
    } break;
    case safetensors::dtype::kUINT64: {
      return std::to_string(*reinterpret_cast<const uint64_t*>(data));
    } break;
    case safetensors::dtype::kINT64: {
      return std::to_string(*reinterpret_cast<const int64_t*>(data));
    } break;
    case safetensors::dtype::kFLOAT16: {
      return std::to_string(safetensors::fp16_to_float(*reinterpret_cast<const uint16_t*>(data)));
    } break;
    case safetensors::dtype::kBFLOAT16: {
      return std::to_string(
          safetensors::bfloat16_to_float(*reinterpret_cast<const uint16_t*>(data)));
    } break;
    case safetensors::dtype::kFLOAT32: {
      return std::to_string(*reinterpret_cast<const float*>(data));
    } break;
    case safetensors::dtype::kFLOAT64: {
      return std::to_string(*reinterpret_cast<const double*>(data));
    } break;
  }

  // n.b. we would probably fatal here...
  return std::string("???");
}

//
// print tensor in linearized 1D array
// In safetensors, data is not strided(tightly packed)
//
auto to_string_snipped(const safetensors::tensor_t& t, const std::uint8_t* databuffer,
                       std::size_t N = 8) {

  std::size_t nitems = safetensors::get_shape_size(t);
  std::size_t itembytes = safetensors::get_dtype_bytes(t.dtype);

  json arr = json::array();

  if ((N == 0) || ((N * 2) >= nitems)) {
    for (size_t idx = 0; idx < nitems; idx++) {
      arr.push_back(to_string(t.dtype, databuffer + t.data_offsets[0] + idx * itembytes));
    }
  } else {
    size_t head_end = (std::min)(N, nitems);
    size_t tail_start = (std::max)(nitems - N, head_end);

    for (size_t idx = 0; idx < head_end; idx++) {
      arr.push_back(to_string(t.dtype, databuffer + t.data_offsets[0] + idx * itembytes));
    }

    arr.push_back("...");

    for (size_t idx = tail_start; idx < nitems; idx++) {
      arr.push_back(to_string(t.dtype, databuffer + t.data_offsets[0] + idx * itembytes));
    }
  }

  return arr.dump();
}

auto main(int argc, char* argv[]) -> int {
  auto filename = std::string{"gen/model.safetensors"};

  auto st = safetensors::safetensors_t{};

  if (argc > 1) {
    filename = argv[1];
  }

  auto warn = std::string{};
  auto err = std::string{};

#if defined(USE_MMAP)
  bool ret = safetensors::mmap_from_file(filename, &st, &warn, &err);
#else
  bool ret = safetensors::load_from_file(filename, &st, &warn, &err);
#endif

  if (warn.size()) {
    std::print("WARN: {}", warn);
  }

  if (!ret) {
    std::println("Failed to load: {}", filename);
    std::println("  ERR: {}", err);
    return EXIT_FAILURE;
  }

  if (!safetensors::validate_data_offsets(st, err)) {
    std::println("Invalid data_offsets: {}", err);
    return EXIT_FAILURE;
  }

  const uint8_t* databuffer{nullptr};

  if (st.mmaped) {
    databuffer = st.databuffer_addr;
  } else {
    databuffer = st.storage.data();
  }

  for (std::size_t idx = 0; idx < st.tensors.size(); idx++) {
    std::string key = st.tensors.keys()[idx];

    auto tensor = safetensors::tensor_t{};
    st.tensors.at(idx, &tensor);

    std::println("{}: {}", key, safetensors::get_dtype_str(tensor.dtype));

    json shape_arr = json::array();
    for (std::size_t idx = 0; idx < tensor.shape.size(); idx++) {
      shape_arr.push_back(tensor.shape[idx]);
    }
    std::println("{}", shape_arr.dump());

    std::print("  data_offsets[{}, {}]", std::to_string(tensor.data_offsets[0]),
               std::to_string(tensor.data_offsets[1]));

    std::print(" {}", to_string_snipped(tensor, databuffer));
  }

  if (st.metadata.size()) {
    std::print("\n");
    std::println("__metadata__");
    for (std::size_t idx = 0; idx < st.metadata.size(); idx++) {
      std::string key = st.metadata.keys()[idx];

      auto value = std::string{};
      st.metadata.at(idx, &value);

      std::println("  {}: ", key, value);
    }
  }

  return EXIT_SUCCESS;
}
