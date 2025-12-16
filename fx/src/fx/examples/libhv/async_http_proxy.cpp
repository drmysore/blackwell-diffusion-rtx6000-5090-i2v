#include <cstdint>
#include <format>
#include <optional>
#include <string>

#include <spdlog/spdlog.h>

#include <nlohmann/json.hpp>

#include "hv/AsyncHttpClient.h"
#include "hv/HttpMessage.h"
#include "hv/HttpServer.h"
#include "hv/HttpService.h"
#include "hv/requests.h"

auto forward_upstream_response(std::string host, hv::HttpResponseWriter* downstream_response_writer,
                               HttpResponse* upstream_http_response) -> void {

  if (!upstream_http_response) {
    spdlog::error("[proxy] {} connection failed", host);
    downstream_response_writer->WriteStatus(HTTP_STATUS_BAD_GATEWAY);
    downstream_response_writer->WriteHeader("Content-Type", "text/plain");
    downstream_response_writer->EndHeaders();
    downstream_response_writer->End("Bad Gateway");
    return;
  }

  auto status_code = static_cast<std::uint16_t>(upstream_http_response->status_code);

  spdlog::info("[proxy] {} responded: {} ({} bytes)", host, status_code,
               upstream_http_response->body.size());


  downstream_response_writer->WriteStatus(upstream_http_response->status_code);
  for (const auto& [key, value] : upstream_http_response->headers) {
    downstream_response_writer->WriteHeader(key.c_str(), value.c_str());
  }

  downstream_response_writer->EndHeaders();
  downstream_response_writer->End(upstream_http_response->body);
}

auto async_proxy_upstream(hv::HttpClient* http_client, HttpRequest* downstream_http_request,
                          hv::HttpResponseWriter* downstream_response_writer) noexcept -> void {

  auto host = downstream_http_request->GetHeader("Host");

  auto upstream_http_request = std::make_shared<HttpRequest>();
  upstream_http_request->method = std::move(downstream_http_request->method);
  upstream_http_request->headers = std::move(downstream_http_request->headers);
  upstream_http_request->body = std::move(downstream_http_request->body);

  upstream_http_request->url =
      std::format("https://{}{}", host, downstream_http_request->FullPath());

  upstream_http_request->headers["Connection"] = "keepalive";

  spdlog::info("[proxy] {} {} -> https://{}{}\n", downstream_http_request->Method(),
               downstream_http_request->Url(), host, downstream_http_request->FullPath());

  downstream_response_writer->Begin();
  http_client->sendAsync(upstream_http_request, [=, upstream_host = std::move(host)](
                                                    HttpResponsePtr upstream_http_response) {
    spdlog::info("[proxy] upstream response from {}", upstream_host);

    forward_upstream_response(std::move(upstream_host), downstream_response_writer,
                              upstream_http_response.get());
  });
};

auto main(int argc, char* argv[]) -> int {
  auto http_server = std::make_unique<hv::HttpServer>();
  auto http_service = std::make_unique<hv::HttpService>();
  auto http_client = std::make_shared<hv::HttpClient>();

  http_service->Any("*", [=](const HttpRequestPtr& request, const HttpResponseWriterPtr& writer) {
    async_proxy_upstream(http_client.get(), request.get(), writer.get());
  });

  http_server->registerHttpService(http_service.get());
  http_server->setPort(10000);

  spdlog::info("async proxy listening on :10000\n");
  spdlog::info("forwarding HTTP -> HTTPS based on Host header\n");
  http_server->run();

  return 0;
}
