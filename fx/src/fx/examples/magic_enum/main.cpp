#include "magic_enum/magic_enum.hpp"

enum class test_enum { first, second, third };

auto main(int argc, char* argv[]) -> int {
  auto name = magic_enum::enum_name(test_enum::second);
  if (name != "second") {
    return 1;
  }

  auto value = magic_enum::enum_cast<test_enum>("third");
  if (!value || *value != test_enum::third) {
    return 2;
  }

  constexpr auto count = magic_enum::enum_count<test_enum>();
  if (count != 3) {
    return 3;
  }

  constexpr auto names = magic_enum::enum_names<test_enum>();
  if (names.size() != 3) {
    return 4;
  }

  return 0;
}
