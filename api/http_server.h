#pragma once

#include <atomic>
#include <functional>
#include <string>
#include "pipeline/inference.h"

using RequestHandler = std::function<std::string(const std::string& method, const std::string& path, const std::string& body)>;

// SSE stream callback: invoked repeatedly to emit tokens; return false to abort.
using StreamHandler = std::function<void(const std::string& body, std::function<bool(const std::string& chunk)> send_chunk)>;

class HttpServer {
public:
    HttpServer() = default;
    ~HttpServer();

    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    bool init(const std::string& listen_address, const int http_port, InferenceEngine* engine);

    // Start listening (blocking -- call from main thread or a dedicated thread).
    void run();

    // Signal graceful shutdown.
    void stop();

    const std::string& last_error() const { return last_error_; }

private:
    void handle_connection(int client_fd);
    bool parse_request(int client_fd, std::string& method, std::string& path, std::string& body);
    void send_response(int client_fd, int status, const std::string& content_type, const std::string& body);
    void send_sse_response(int client_fd, const std::string& body);
    bool send_all(int fd, const char* data, size_t len);

    // Route handlers
    std::string handle_chat_completions(const std::string& body);
    void handle_chat_completions_stream(int client_fd, const std::string& body);
    std::string handle_models();
    std::string handle_health();
    std::string handle_metrics();
    void serve_web_ui(int client_fd);

    // Lightweight JSON helpers (no external library)
    static std::string extract_json_string(const std::string& json, const std::string& key);
    static bool extract_json_bool(const std::string& json, const std::string& key, bool default_value);
    static int extract_json_int(const std::string& json, const std::string& key, int default_value);
    static float extract_json_float(const std::string& json, const std::string& key, float default_value);
    static std::string extract_last_user_content(const std::string& json);

    InferenceEngine* engine_ = nullptr;
    int server_fd_ = -1;
    int port_ = 8080;
    std::string listen_address_ = "0.0.0.0";
    std::atomic<bool> running_{false};
    std::string last_error_;
    bool initialized_ = false;
};
