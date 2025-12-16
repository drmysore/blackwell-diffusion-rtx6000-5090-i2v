// n.b. `bazel run //src/fx/examples/boost:boost_features_example -- --url https://www.google.com`

#include <chrono>
#include <expected>
#include <format>
#include <iostream>
#include <print>
#include <ranges>
#include <stop_token>

#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <boost/filesystem.hpp>
#include <boost/json.hpp>
#include <boost/program_options.hpp>
#include <boost/url.hpp>

namespace net = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
namespace fs = boost::filesystem;
namespace po = boost::program_options;

using tcp = net::ip::tcp;
using namespace std::chrono_literals;

template <typename T>
concept AsyncResult = requires(T t) {
  { t.get() } -> std::convertible_to<std::size_t>;
};

template <typename T>
using Result = std::expected<T, std::string>;

auto async_http_get(std::string_view host, std::string_view target, net::io_context& ioc)
    -> net::awaitable<Result<std::string>> {

  try {
    auto resolver = net::use_awaitable_t<>::as_default_on(tcp::resolver{ioc});
    auto stream = net::use_awaitable_t<>::as_default_on(beast::tcp_stream{ioc});

    // Resolve and connect
    auto const results = co_await resolver.async_resolve(host, "80");
    stream.expires_after(30s);
    co_await stream.async_connect(results);

    // Send request
    http::request<http::string_body> req{http::verb::get, target, 11};
    req.set(http::field::host, host);
    req.set(http::field::user_agent, "libmodern-cpp/1.0");

    co_await http::async_write(stream, req);

    // Receive response
    beast::flat_buffer buffer;
    http::response<http::string_body> res;
    co_await http::async_read(stream, buffer, res);

    // Graceful shutdown
    beast::error_code ec;
    stream.socket().shutdown(tcp::socket::shutdown_both, ec);

    co_return res.body();
  } catch (std::exception const& e) {
    co_return std::unexpected(std::format("HTTP error: {}", e.what()));
  }
}

auto process_directory(fs::path const& dir) -> Result<std::vector<fs::path>> {
  try {

    auto files = fs::recursive_directory_iterator{dir} |
                 std::views::filter([](auto const& entry) { return entry.is_regular_file(); }) |
                 std::views::transform([](auto const& entry) { return entry.path(); }) |
                 std::ranges::to<std::vector>();

    return files;
  } catch (fs::filesystem_error const& e) {
    return std::unexpected(std::format("Filesystem error: {}", e.what()));
  }
}

auto parse_json_config(std::string_view json_str) -> Result<boost::json::object> {
  try {
    auto parsed = boost::json::parse(json_str);

    if (auto* obj = parsed.if_object()) {
      return *obj;
    }

    return std::unexpected("JSON is not an object");
  } catch (std::exception const& e) {
    return std::unexpected(std::format("JSON parse error: {}", e.what()));
  }
}

// Entry point with structured bindings and modern option parsing
auto main(int argc, char* argv[]) -> int {
  try {
    po::options_description desc{"libmodern-cpp boost link check"};

    desc.add_options()("help,h", "Show help")(
        "directory,d", po::value<std::string>()->default_value("."), "Directory to scan")(
        "url,u", po::value<std::string>(), "URL to check")(
        "threads,t", po::value<int>()->default_value(4), "Thread count");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
      std::cout << desc << std::endl;
      return 0;
    }

    if (auto files = process_directory(vm["directory"].as<std::string>())) {
      std::println("Found {} files", files->size());

      for (auto const& file : *files | std::views::take(10)) {
        std::println("  {}", file.string());
      }
    } else {
      std::println(stderr, "Error: {}", files.error());
    }

    if (vm.count("url")) {
      net::io_context ioc{vm["threads"].as<int>()};

      auto url = boost::urls::parse_uri(vm["url"].as<std::string>());
      if (!url) {
        std::println(stderr, "Invalid URL");
        return 1;
      }

      net::co_spawn(
          ioc,
          [&]() -> net::awaitable<void> {
            auto result = co_await async_http_get(url->host(), url->path(), ioc);

            if (result) {
              std::println("{}", *result);
              std::println("Response length: {} bytes", result->size());
            } else {
              std::println(stderr, "HTTP failed: {}", result.error());
            }
          },
          net::detached);

      ioc.run();
    }

    constexpr auto sample_json = R"({"boost": "works", "modern": true, "cpp": 23})";
    if (auto config = parse_json_config(sample_json)) {
      std::println("JSON parsed successfully: {} keys", config->size());
    }

    std::println("✓ All boost libraries linked successfully!");
    return 0;
  } catch (std::exception const& e) {
    std::println(stderr, "Fatal error: {}", e.what());
    return 1;
  }
}
