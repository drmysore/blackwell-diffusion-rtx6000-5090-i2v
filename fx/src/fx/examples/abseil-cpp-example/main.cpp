#include <algorithm>
#include <chrono>
#include <print>
#include <ranges>
#include <string_view>

#include <absl/container/flat_hash_map.h>
#include <absl/hash/hash.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>
#include <absl/strings/str_split.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <absl/types/span.h>
#include <spdlog/spdlog.h>

namespace modern {

struct score_entry {
  std::string_view name;
  int score;

  auto operator<=>(const score_entry&) const = default;
};

constexpr auto format_duration(absl::Duration d) -> std::string {
  return absl::StrFormat("// duration // %s // ready", absl::FormatDuration(d));
}

}  // namespace modern

auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int {
  std::println("// init // hypermodern demo...");
  spdlog::info("// spdlog // online");

  // string composition with style
  const auto greeting = absl::StrCat("// hello // ", "abseil", " // ready");
  std::println("{}", greeting);

  // formatting with intent
  const auto formatted = absl::StrFormat("// compute // answer = %d // done", 42);
  spdlog::info("{}", formatted);

  // modern containers + algorithms
  absl::flat_hash_map<std::string_view, int> scores{
      {"alice", 100}, {"bob", 95}, {"charlie", 87}, {"diana", 92}};

  // ranges pipeline
  auto entries =
      scores |
      std::views::transform([](const auto& p) { return modern::score_entry{p.first, p.second}; }) |
      std::ranges::to<std::vector>();

  std::ranges::sort(entries, std::greater{}, &modern::score_entry::score);

  std::println("// leaderboard // top performers:");
  for (const auto& [name, score] : entries | std::views::take(3)) {
    std::println("  // {} // score: {} // rank: top3", name, score);
  }

  // time with style - correct FormatTime usage
  const auto start = absl::Now();
  const auto time_str = absl::StrFormat("// timestamp // %s // ", absl::FormatTime(start));
  spdlog::info("{}", time_str);

  // string splitting/joining
  const auto components = absl::StrSplit("modern,cpp,abseil,style", ',');
  const auto rejoined = absl::StrJoin(components, " // ");
  std::println("// stack // {} // loaded", rejoined);

  // hash demonstration
  const auto hash_value = absl::Hash<std::string_view>{}("hypermodern");
  std::println("// hash // 0x{:016x} // computed", hash_value);

  // duration with custom formatting
  const auto pi_duration = absl::Seconds(3.14159);
  std::println("{}", modern::format_duration(pi_duration));

  // elapsed time
  const auto elapsed = absl::Now() - start;
  spdlog::info("// elapsed // {} // complete", absl::FormatDuration(elapsed));

  std::println("// exit // success // 0");
  return 0;
}
