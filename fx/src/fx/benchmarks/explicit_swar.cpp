#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <print>
#include <ranges>
#include <span>
#include <vector>

auto count_byte_scalar(const uint8_t* data, size_t len, uint8_t target) -> std::size_t {
  auto count = size_t{0};

  for (auto i = size_t{0}; i < len; ++i) {
    if (data[i] == target) {
      ++count;
    }
  }

  return count;
}

// SWAR
auto count_byte_swar(const uint8_t* data, size_t len, uint8_t target) -> size_t {
  auto count = size_t{0};

  // 8 bytes at a time
  auto i = size_t{0};
  for (; i + 8 <= len; i += 8) {
    auto chunk = *reinterpret_cast<const uint64_t*>(data + i);

    // each byte independently
    auto b0 = (uint8_t)(chunk >> 0) == target ? 1 : 0;
    auto b1 = (uint8_t)(chunk >> 8) == target ? 1 : 0;
    auto b2 = (uint8_t)(chunk >> 16) == target ? 1 : 0;
    auto b3 = (uint8_t)(chunk >> 24) == target ? 1 : 0;
    auto b4 = (uint8_t)(chunk >> 32) == target ? 1 : 0;
    auto b5 = (uint8_t)(chunk >> 40) == target ? 1 : 0;
    auto b6 = (uint8_t)(chunk >> 48) == target ? 1 : 0;
    auto b7 = (uint8_t)(chunk >> 56) == target ? 1 : 0;

    count += b0 + b1 + b2 + b3 + b4 + b5 + b6 + b7;
  }

  // remaining
  for (; i < len; ++i) {
    if (data[i] == target) {
      ++count;
    }
  }

  return count;
}

// range-based version using algorithms
auto count_byte_ranges(std::span<const uint8_t> data, uint8_t target) -> size_t {
  return std::ranges::count(data, target);
}

// chunked processing with ranges (cache-line aware)
auto count_byte_chunked(std::span<const uint8_t> data, uint8_t target) -> size_t {
  auto count = size_t{0};

  // chunks of 64 bytes (cache line)
  for (auto chunk : data | std::views::chunk(64)) {
    count += std::ranges::count(chunk, target);
  }

  return count;
}

auto main() -> int {
  // 1MB of random-ish data
  auto data = std::vector<uint8_t>(1024 * 1024);
  for (auto i = size_t{0}; i < data.size(); ++i) {
    data[i] = static_cast<uint8_t>(i * 31 + 17);  // Deterministic pattern
  }

  auto target = uint8_t{'A'};
  auto data_span = std::span{data};

  // verify
  auto scalar_result = count_byte_scalar(data.data(), data.size(), target);
  auto swar_result = count_byte_swar(data.data(), data.size(), target);
  auto ranges_result = count_byte_ranges(data_span, target);
  auto chunked_result = count_byte_chunked(data_span, target);

  if (scalar_result != swar_result || scalar_result != ranges_result ||
      scalar_result != chunked_result) {
    std::println("ERROR: Results don't match! scalar={} swar={} ranges={} chunked={}",
                 scalar_result, swar_result, ranges_result, chunked_result);
    return 1;
  }

  std::println("All methods found {} matches", scalar_result);

  auto iterations = 1000;

  auto start = std::chrono::steady_clock::now();
  for (auto i = 0; i < iterations; ++i) {
    auto result = count_byte_scalar(data.data(), data.size(), target);
    asm volatile("" : : "r"(result) : "memory");
  }
  auto scalar_time = std::chrono::steady_clock::now() - start;

  start = std::chrono::steady_clock::now();
  for (auto i = 0; i < iterations; ++i) {
    auto result = count_byte_swar(data.data(), data.size(), target);
    asm volatile("" : : "r"(result) : "memory");
  }
  auto swar_time = std::chrono::steady_clock::now() - start;

  start = std::chrono::steady_clock::now();
  for (auto i = 0; i < iterations; ++i) {
    auto result = count_byte_ranges(data_span, target);
    asm volatile("" : : "r"(result) : "memory");
  }
  auto ranges_time = std::chrono::steady_clock::now() - start;

  start = std::chrono::steady_clock::now();
  for (auto i = 0; i < iterations; ++i) {
    auto result = count_byte_chunked(data_span, target);
    asm volatile("" : : "r"(result) : "memory");
  }
  auto chunked_time = std::chrono::steady_clock::now() - start;

  auto scalar_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(scalar_time).count();
  auto swar_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(swar_time).count();
  auto ranges_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(ranges_time).count();
  auto chunked_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(chunked_time).count();

  std::println("Scalar:  {} ns total, {} ns per iteration", scalar_ns, scalar_ns / iterations);
  std::println("SWAR:    {} ns total, {} ns per iteration", swar_ns, swar_ns / iterations);
  std::println("Ranges:  {} ns total, {} ns per iteration", ranges_ns, ranges_ns / iterations);
  std::println("Chunked: {} ns total, {} ns per iteration", chunked_ns, chunked_ns / iterations);

  std::println("\nSpeedups vs scalar:");
  std::println("SWAR:    {:.2f}x", double(scalar_ns) / double(swar_ns));
  std::println("Ranges:  {:.2f}x", double(scalar_ns) / double(ranges_ns));
  std::println("Chunked: {:.2f}x", double(scalar_ns) / double(chunked_ns));

  return 0;
}
