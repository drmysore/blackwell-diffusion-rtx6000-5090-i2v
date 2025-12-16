#include "zpp_bits.h"

#include <cstddef>
#include <print>
#include <span>
#include <string>
#include <vector>

namespace fx::exmaple {
struct image {
  std::string prompt;
  std::size_t width_px;
  std::size_t height_px;
};
}  // namespace fx::exmaple

auto main(int argc, char* argv[]) -> int {
  auto image = fx::exmaple::image{
      .prompt = "a cyberpunk cat in future tokyo", .width_px = 1024, .height_px = 768};

  auto data = std::vector<std::byte>{};
  auto out = zpp::bits::out{data};

  if (auto result = out(image); failure(result)) {
    std::println("serialization failed: {}", static_cast<std::int64_t>(std::errc(result)));
    return 1;
  }

  std::println("serialized {} bytes", data.size());

  // Deserialize
  auto image_two = fx::exmaple::image{};
  auto in = zpp::bits::in{data};

  if (auto result = in(image_two); failure(result)) {
    std::println("deserialization failed: {}", static_cast<std::int64_t>(std::errc(result)));
    return 1;
  }

  if (image.prompt != image_two.prompt || image.width_px != image_two.width_px ||
      image.height_px != image_two.height_px) {

    std::println("roundtrip failed!");
    return 1;
  }

  std::println("success! {}: {}x{}", image_two.prompt, image_two.width_px, image_two.height_px);
  return 0;
}
