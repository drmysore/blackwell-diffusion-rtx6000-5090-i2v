#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <print>
#include <thread>
#include <utility>

#include "hv/AsyncHttpClient.h"
#include "hv/HttpClient.h"
#include "hv/HttpMessage.h"
#include "hv/hloop.h"

auto main(int argc, char* argv[]) -> int {
  ::hloop_t* loop = hloop_new(0);  // n.b. leaks...

  auto http_client = std::make_shared<hv::AsyncHttpClient>();
  auto http_request = std::make_shared<HttpRequest>();
  http_request->url = "https://www.fal.ai";
  http_request->method = HTTP_GET;

  http_client->send(http_request, [](HttpResponsePtr http_response) {
    std::print("status: {}", std::to_underlying(http_response->status_code));
    std::print("body size: {} bytes", http_response->body.size());

    std::print("headers:");
    for (const auto& [key, value] : http_response->headers) {
      std::print("  '{}': '{}'", key, value);
    }

    std::exit(0);
  });

  ::hloop_run(loop);
  return 0;
}
