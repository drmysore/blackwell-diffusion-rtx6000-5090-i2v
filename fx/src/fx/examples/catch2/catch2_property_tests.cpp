#include <algorithm>
#include <concepts>
#include <expected>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

namespace modern::math {

enum class math_error { division_by_zero, overflow, underflow };

constexpr auto to_string(math_error e) -> std::string_view {
  switch (e) {
    case math_error::division_by_zero: {
      return "// error // division by zero";
    }; break;
    case math_error::overflow: {
      return "// error // overflow";
    } break;
    case math_error::underflow: {
      return "// error // underflow";
    }; break;
  }

  return "// error // unknown";
}

template <std::integral T>
constexpr auto safe_divide(T numerator, T denominator) -> std::expected<double, math_error> {
  if (denominator == 0) [[unlikely]] {
    return std::unexpected(math_error::division_by_zero);
  }
  return static_cast<double>(numerator) / denominator;
}

}  // namespace modern::math

namespace modern::test {

using namespace std::string_view_literals;
using Catch::Matchers::Equals;
using Catch::Matchers::VectorContains;

TEST_CASE("// test // modern error handling", "[expected][modern]") {
  SECTION("// success // valid division") {
    const auto result = modern::math::safe_divide(10, 2);
    REQUIRE(result.has_value());
    CHECK(*result == 5.0);
  }

  SECTION("// failure // division by zero") {
    const auto result = modern::math::safe_divide(10, 0);
    REQUIRE_FALSE(result.has_value());
    CHECK(result.error() == modern::math::math_error::division_by_zero);
    CHECK(to_string(result.error()) == "// error // division by zero"sv);
  }
}

TEST_CASE("// property // algebraic laws", "[property][algebra]") {
  SECTION("// involution // reverse ∘ reverse = id") {
    const auto input =
        GENERATE("hello"sv, "world"sv, "a"sv, ""sv, "test123"sv, "!@#$"sv, "// modern //"sv);

    auto str = std::string{input};
    std::ranges::reverse(str);
    std::ranges::reverse(str);

    CHECK_THAT(str, Equals(std::string{input}));
  }

  SECTION("// idempotence // sort ∘ sort = sort") {
    auto vec = GENERATE(std::vector{3, 1, 4, 1, 5, 9}, std::vector{5, 4, 3, 2, 1}, std::vector{42},
                        std::vector<int>{});

    const auto once_sorted = vec | std::views::all | std::ranges::to<std::vector>();
    std::ranges::sort(vec);

    const auto twice_sorted = vec;
    std::ranges::sort(vec);

    CHECK(vec == twice_sorted);
  }
}

TEST_CASE("// showcase // modern c++ features", "[modern][c++23]") {
  constexpr auto numbers = std::array{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  SECTION("// ranges // pipeline composition") {
    const auto result = numbers | std::views::filter([](auto n) { return n % 2 == 0; }) |
                        std::views::transform([](auto n) { return n * n; }) | std::views::take(3) |
                        std::ranges::to<std::vector>();

    CHECK_THAT(result, Equals(std::vector{4, 16, 36}));
  }

  SECTION("// algorithms // parallel execution") {
    auto mutable_copy = std::vector(numbers.begin(), numbers.end());

    const auto [min_it, max_it] = std::ranges::minmax_element(mutable_copy);
    CHECK(*min_it == 1);
    CHECK(*max_it == 10);

    const auto sum = std::ranges::fold_left(mutable_copy, 0, std::plus<>{});
    CHECK(sum == 55);  // n(n+1)/2
  }
}

TEST_CASE("// strings // modern text processing", "[string][c++23]") {
  using namespace std::string_view_literals;

  SECTION("// literals // compile-time strings") {
    constexpr auto message = "// hello // modern c++ // ready"sv;

    STATIC_CHECK(message.starts_with("// hello"));
    STATIC_CHECK(message.ends_with("// ready"));
    STATIC_CHECK(message.find("modern") != std::string_view::npos);
  }

  SECTION("// split // string processing") {
    const auto text = "modern::cpp::is::awesome"sv;
    auto parts = std::vector<std::string_view>{};

    for (auto part : text | std::views::split("::"sv)) {
      parts.emplace_back(part.begin(), part.end());
    }

    CHECK_THAT(parts, Equals(std::vector{"modern"sv, "cpp"sv, "is"sv, "awesome"sv}));
  }
}

template <typename T>
concept Arithmetic = std::integral<T> || std::floating_point<T>;

template <Arithmetic T>
constexpr auto square(T value) -> T {
  return value * value;
}

TEST_CASE("// concepts // type constraints", "[concepts][c++20]") {
  SECTION("// arithmetic // concept satisfaction") {
    STATIC_CHECK(square(5) == 25);
    STATIC_CHECK(square(5.5) == 30.25);

    // Won't compile: square("hello");
  }
}

}  // namespace modern::test
