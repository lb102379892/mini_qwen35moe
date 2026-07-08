#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <chrono>
#include <ctime>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include "pipeline/tokenizer.h"
#include "api/http_server.h"
#include "api/routes.h"

static constexpr const char* TAG = "http";
static constexpr int kMaxRequestBytes = 1 * 1024 * 1024;  // 1 MiB
static constexpr int kReadBufSize = 8192;
static constexpr int kListenBacklog = 64;

static constexpr int kMaxGenerateTokens = 4096;

static std::string debug_escape_token_piece(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (unsigned char c : text) {
        switch (c) {
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\x%02X", c);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
                break;
        }
    }
    if (out.size() > 48) {
        out.resize(48);
        out += "...";
    }
    return out;
}

HttpServer::~HttpServer() {
    stop();
    if (server_fd_ >= 0) {
        ::close(server_fd_);
        server_fd_ = -1;
    }
}

bool HttpServer::init(const std::string& listen_address, const int http_port, ChatEngine* engine) {
    if (initialized_) {
        last_error_ = "Server already initialized";
        return false;
    }

    engine_ = engine;
    port_ = http_port;
    listen_address_ = listen_address;

    // Create socket
    server_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        last_error_ = std::string("socket() failed: ") + std::strerror(errno);
        fprintf(stderr, "%s\n", last_error_.c_str());
        return false;
    }

    // Allow address reuse (avoid EADDRINUSE on fast restart)
    int opt = 1;
    if (::setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        last_error_ = std::string("setsockopt(SO_REUSEADDR) failed: ") + std::strerror(errno);
        fprintf(stderr, "%s\n", last_error_.c_str());
        ::close(server_fd_);
        server_fd_ = -1;
        return false;
    }

    // Bind
    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port_));
    if (::inet_pton(AF_INET, listen_address_.c_str(), &addr.sin_addr) != 1) {
        last_error_ = "Invalid listen address: " + listen_address_;
        fprintf(stderr, "%s\n", last_error_.c_str());
        ::close(server_fd_);
        server_fd_ = -1;
        return false;
    }

    if (::bind(server_fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        last_error_ = std::string("bind() failed: ") + std::strerror(errno);
        fprintf(stderr, "%s\n", last_error_.c_str());
        ::close(server_fd_);
        server_fd_ = -1;
        return false;
    }

    // Listen
    if (::listen(server_fd_, kListenBacklog) < 0) {
        last_error_ = std::string("listen() failed: ") + std::strerror(errno);
        fprintf(stderr, "%s\n", last_error_.c_str());
        ::close(server_fd_);
        server_fd_ = -1;
        return false;
    }

    initialized_ = true;
    printf("HTTP server initialized on %s:%d\n", listen_address_.c_str(), port_);
    return true;
}

void HttpServer::run() {
    if (!initialized_) {
        fprintf(stderr, "Cannot run: server not initialized\n");
        return;
    }

    running_.store(true, std::memory_order_release);
    printf("HTTP server accepting connections on %s:%d\n", listen_address_.c_str(), port_);

    // Use a timeout via SO_RCVTIMEO so accept() does not block forever
    // and we can check the running_ flag periodically.
    struct timeval tv{};
    tv.tv_sec = 1;
    tv.tv_usec = 0;
    ::setsockopt(server_fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    while (running_.load(std::memory_order_acquire)) {
        struct sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_fd = ::accept(server_fd_, reinterpret_cast<struct sockaddr*>(&client_addr), &client_len);
        if (client_fd < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
                continue;  // timeout or signal -- re-check running_
            }
            if (running_.load(std::memory_order_acquire)) {
                printf("accept() failed: %s\n", std::strerror(errno));
            }
            continue;
        }

        // Disable Nagle for low-latency SSE streaming
        int flag = 1;
        ::setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

        char client_ip[INET_ADDRSTRLEN];
        ::inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, sizeof(client_ip));
        printf("Connection from %s:%d\n", client_ip, ntohs(client_addr.sin_port));

        std::thread([this, client_fd]() {
            handle_connection(client_fd);
            ::close(client_fd);
        }).detach();
    }

    printf("HTTP server stopped\n");
}

void HttpServer::stop() {
    if (running_.load(std::memory_order_acquire)) {
        running_.store(false, std::memory_order_release);
        printf("HTTP server shutdown requested\n");
    }
}

// ---------------------------------------------------------------------------
// Connection handling
// ---------------------------------------------------------------------------
void HttpServer::handle_connection(int client_fd) {
    std::string method, path, body;
    if (!parse_request(client_fd, method, path, body)) {
        send_response(client_fd, 400, "application/json", R"({"error":{"message":"Bad request","type":"invalid_request"}})");
        return;
    }

    printf("%s %s (body %zu bytes)\n", method.c_str(), path.c_str(), body.size());

    // ---- CORS preflight ----
    if (method == "OPTIONS") {
        std::string headers =
            "HTTP/1.1 204 No Content\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
            "Access-Control-Max-Age: 86400\r\n"
            "Content-Length: 0\r\n"
            "Connection: close\r\n\r\n";
        ::send(client_fd, headers.data(), headers.size(), MSG_NOSIGNAL);
        return;
    }

    // ---- Routing ----
    if (method == "POST" && path == "/v1/chat/completions") {
        if (!engine_) {
            send_response(client_fd, 503, "application/json", R"({"error":{"message":"Engine not available","type":"server_error"}})");
            return;
        }
        // Detect streaming request
        bool stream = extract_json_bool(body, "stream", false);
        if (!stream) {
            std::string response = handle_chat_completions(body);
            send_response(client_fd, 200, "application/json", response);
        } else {
            send_response(client_fd, 400, "application/json",
                R"({"error":{"message":"stream=true is not supported by this server yet","type":"invalid_request"}})");
        }
    } else if (method == "GET" && path == "/v1/models") {
        std::string response = handle_models();
        send_response(client_fd, 200, "application/json", response);
    } else if (method == "GET" && path == "/health") {
        std::string response = handle_health();
        send_response(client_fd, 200, "application/json", response);
    } else if (method == "GET" && path == "/metrics") {
        std::string response = handle_metrics();
        send_response(client_fd, 200, "application/json", response);
    } else if (method == "GET" && (path == "/" || path == "/index.html")) {
        printf("serve_web_ui\n");
        serve_web_ui(client_fd);
    } else {
        send_response(client_fd, 404, "application/json", R"({"error":{"message":"Not found","type":"invalid_request"}})");
    }
}

// ---------------------------------------------------------------------------
// HTTP parsing
// ---------------------------------------------------------------------------
bool HttpServer::parse_request(int client_fd, std::string& method, std::string& path, std::string& body) {
    std::string raw;
    raw.reserve(kReadBufSize);
    char buf[kReadBufSize];

    // Read until we have the full headers (terminated by \r\n\r\n).
    size_t header_end = std::string::npos;
    while (header_end == std::string::npos) {
        ssize_t n = ::recv(client_fd, buf, sizeof(buf), 0);
        if (n <= 0) return false;
        raw.append(buf, static_cast<size_t>(n));
        header_end = raw.find("\r\n\r\n");
        if (raw.size() > static_cast<size_t>(kMaxRequestBytes)) return false;
    }

    // Parse request line: METHOD PATH HTTP/1.x
    size_t first_line_end = raw.find("\r\n");
    if (first_line_end == std::string::npos) return false;

    std::string request_line = raw.substr(0, first_line_end);
    size_t sp1 = request_line.find(' ');
    if (sp1 == std::string::npos) return false;
    size_t sp2 = request_line.find(' ', sp1 + 1);
    if (sp2 == std::string::npos) return false;

    method = request_line.substr(0, sp1);
    path = request_line.substr(sp1 + 1, sp2 - sp1 - 1);

    // Strip query string from path
    size_t q = path.find('?');
    if (q != std::string::npos) path = path.substr(0, q);

    // Extract Content-Length header (case-insensitive search)
    int content_length = 0;
    std::string headers_block = raw.substr(first_line_end + 2, header_end - first_line_end - 2);
    std::string headers_lower = headers_block;
    std::transform(headers_lower.begin(), headers_lower.end(), headers_lower.begin(), ::tolower);
    size_t cl_pos = headers_lower.find("content-length:");
    if (cl_pos != std::string::npos) {
        size_t val_start = cl_pos + 15;  // strlen("content-length:")
        size_t val_end = headers_lower.find("\r\n", val_start);
        if (val_end == std::string::npos) val_end = headers_lower.size();
        std::string cl_str = headers_block.substr(val_start, val_end - val_start);
        // Trim whitespace
        size_t first_digit = cl_str.find_first_not_of(' ');
        if (first_digit != std::string::npos) {
            content_length = std::atoi(cl_str.c_str() + first_digit);
        }
    }

    if (content_length < 0 || content_length > kMaxRequestBytes) return false;

    // Read remaining body bytes
    size_t body_start = header_end + 4;  // skip \r\n\r\n
    size_t body_bytes_read = raw.size() - body_start;
    size_t remaining = (content_length > 0) ? static_cast<size_t>(content_length) - body_bytes_read : 0;
    while (remaining > 0) {
        ssize_t n = ::recv(client_fd, buf, std::min(sizeof(buf), remaining), 0);
        if (n <= 0) return false;
        raw.append(buf, static_cast<size_t>(n));
        remaining -= static_cast<size_t>(n);
    }

    body = (content_length > 0) ? raw.substr(body_start, static_cast<size_t>(content_length)) : "";
    return true;
}

// ---------------------------------------------------------------------------
// HTTP responses
// ---------------------------------------------------------------------------
void HttpServer::serve_web_ui(int client_fd) {
    // Try loading web/index.html from the working directory.
    static std::string cached_html;
    static std::mutex cached_html_mutex;
    std::lock_guard<std::mutex> lock(cached_html_mutex);
    if (cached_html.empty()) {
        const char* paths[] = { "web/index.html", "/home/xc/code/my_llama-copy-update.cpp/web/index.html", nullptr };
        for (int i = 0; paths[i]; ++i) {
            FILE* f = std::fopen(paths[i], "rb");
            if (f) {
                std::fseek(f, 0, SEEK_END);
                long sz = std::ftell(f);
                std::fseek(f, 0, SEEK_SET);
                if (sz > 0 && sz < 1024 * 1024) {
                    cached_html.resize(static_cast<size_t>(sz));
                    std::fread(&cached_html[0], 1, static_cast<size_t>(sz), f);
                }
                std::fclose(f);
                if (!cached_html.empty()) {
                    printf("Loaded web UI from %s (%zu bytes)", paths[i], cached_html.size());
                    break;
                }
            }
        }
    }
    if (cached_html.empty()) {
        send_response(client_fd, 404, "text/plain", "Web UI not found. Place web/index.html in the working directory.");
        return;
    }
    send_response(client_fd, 200, "text/html; charset=utf-8", cached_html);
}

void HttpServer::send_response(int client_fd, int status, const std::string& content_type, const std::string& body) {
    const char* status_text = "OK";
    switch (status) {
        case 200: status_text = "OK"; break;
        case 204: status_text = "No Content"; break;
        case 400: status_text = "Bad Request"; break;
        case 404: status_text = "Not Found"; break;
        case 500: status_text = "Internal Server Error"; break;
        case 503: status_text = "Service Unavailable"; break;
        default:  status_text = "Unknown"; break;
    }

    std::ostringstream oss;
    oss << "HTTP/1.1 " << status << " " << status_text << "\r\n"
        << "Content-Type: " << content_type << "\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Access-Control-Allow-Origin: *\r\n"
        << "Connection: close\r\n"
        << "\r\n"
        << body;

    std::string response = oss.str();
    size_t total_sent = 0;
    while (total_sent < response.size()) {
        ssize_t n = ::send(client_fd, response.data() + total_sent, response.size() - total_sent, MSG_NOSIGNAL);
        if (n <= 0) {
            printf("send() failed: %s\n", std::strerror(errno));
            break;
        }
        total_sent += static_cast<size_t>(n);
    }
}

bool HttpServer::send_all(int fd, const char* data, size_t len) {
    size_t total_sent = 0;
    while (total_sent < len) {
        ssize_t n = ::send(fd, data + total_sent, len - total_sent, MSG_NOSIGNAL);
        if (n <= 0) {
            printf("send_all() failed: %s\n", std::strerror(errno));
            return false;
        }
        total_sent += static_cast<size_t>(n);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------
std::string HttpServer::handle_chat_completions(const std::string& body) {
    if (!engine_->chat_template_ready()) {
        return R"({"error":{"message":"Chat template not available","type":"server_error"}})";
    }

    const ChatTemplateApplyResult templated = engine_->build_chat_prompt(body);
    if (!templated.ok) {
        const std::string msg = templated.error.empty() ? "Invalid chat request" : templated.error;
        return make_error_response(msg, "invalid_request");
    }

    int max_tokens = extract_json_int(body, "max_tokens", kMaxGenerateTokens);
    if (max_tokens <= 0 || max_tokens > kMaxGenerateTokens) max_tokens = kMaxGenerateTokens;

    std::string generated_text;
    if (!engine_->run_complete(templated.prompt, max_tokens, generated_text)) {
        return R"({"error":{"message":"Generation failed","type":"server_error"}})";
    }

    const ChatTemplateParseResult parsed =
        engine_->parse_chat_response(generated_text, templated.enable_thinking);
    std::string escaped_content = json_escape(parsed.content);
    std::string escaped_reasoning = json_escape(parsed.reasoning_content);

    std::ostringstream oss;
    oss << "{"
        << "\"id\":\"chatcmpl-phantom\","
        << "\"object\":\"chat.completion\","
        << "\"model\":\"" << kModelId << "\","
        << "\"choices\":[{"
        << "\"index\":0,"
        << "\"message\":{\"role\":\"assistant\",\"content\":\"" << escaped_content << "\"";
    if (!parsed.reasoning_content.empty()) {
        oss << ",\"reasoning_content\":\"" << escaped_reasoning << "\"";
    }
    oss << "},"
        << "\"finish_reason\":\"stop\""
        << "}]"
        << "}";

    return oss.str();
}

std::string HttpServer::handle_models() {
    return std::string("{")
        + "\"object\":\"list\","
            "\"data\":[{"
            "\"id\":\"" + kModelId + "\","
            "\"object\":\"model\","
            "\"owned_by\":\"phantom\""
            "}]"
            "}";
}

std::string HttpServer::handle_health() {
    bool engine_ready = (engine_ != nullptr);
    std::ostringstream oss;
    oss << "{"
        << "\"status\":\"" << (engine_ready ? "ok" : "no_engine") << "\","
        << "\"version\":\"" << kServerVersion << "\""
        << "}";
    return oss.str();
}

std::string HttpServer::handle_metrics() {
    return R"({"tokens_per_second":0,"cache_hit_rate":0,"memory_used_gb":0,"uptime_seconds":0})";
}

// ---------------------------------------------------------------------------
// JSON helpers (minimal, no external library)
// ---------------------------------------------------------------------------

std::string HttpServer::extract_json_string(const std::string& json, const std::string& key) {
    // Find "key": "value" -- intentionally simple; handles the common case.
    std::string needle = "\"" + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return "";

    // Skip past the key, colon, and optional whitespace to the opening quote.
    pos += needle.size();
    pos = json.find(':', pos);
    if (pos == std::string::npos) return "";
    pos++;

    // Skip whitespace
    while (pos < json.size() && 
        (
            json[pos] == ' ' || 
            json[pos] == '\t' ||
            json[pos] == '\n' || 
            json[pos] == '\r'
        )
    )
        pos++;

    if (pos >= json.size() || json[pos] != '"') return "";
    pos++;  // skip opening quote

    // Collect characters until the unescaped closing quote.
    std::string value;
    while (pos < json.size()) {
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            char next = json[pos + 1];
            switch (next) {
                case '"':  value += '"';  break;
                case '\\': value += '\\'; break;
                case 'n':  value += '\n'; break;
                case 'r':  value += '\r'; break;
                case 't':  value += '\t'; break;
                case 'b':  value += '\b'; break;
                case 'f':  value += '\f'; break;
                default:   value += next; break;
            }
            pos += 2;
        } else if (json[pos] == '"') {
            break;
        } else {
            value += json[pos];
            pos++;
        }
    }

    return value;
}

bool HttpServer::extract_json_bool(const std::string& json, const std::string& key, bool default_value) {
    std::string needle = "\"" + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return default_value;

    pos += needle.size();
    pos = json.find(':', pos);
    if (pos == std::string::npos) return default_value;
    pos++;

    // Skip whitespace
    while (pos < json.size() && 
        (
            json[pos] == ' ' || 
            json[pos] == '\t' ||
            json[pos] == '\n' || 
            json[pos] == '\r'
        )
    )
        pos++;

    if (pos + 4 <= json.size() && json.substr(pos, 4) == "true") return true;
    if (pos + 5 <= json.size() && json.substr(pos, 5) == "false") return false;

    return default_value;
}

int HttpServer::extract_json_int(const std::string& json, const std::string& key, int default_value) {
    std::string needle = "\"" + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return default_value;

    pos += needle.size();
    pos = json.find(':', pos);
    if (pos == std::string::npos) return default_value;
    pos++;

    while (pos < json.size() && 
        (
            json[pos] == ' ' || 
            json[pos] == '\t' ||
            json[pos] == '\n' || 
            json[pos] == '\r'
        )
    )
        pos++;

    if (pos >= json.size()) return default_value;
    char* end = nullptr;
    long val = std::strtol(json.c_str() + pos, &end, 10);
    if (end == json.c_str() + pos) return default_value;
    if (val < INT_MIN || val > INT_MAX) return default_value;
    return static_cast<int>(val);
}

float HttpServer::extract_json_float(const std::string& json, const std::string& key, float default_value) {
    std::string needle = "\"" + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return default_value;

    pos += needle.size();
    pos = json.find(':', pos);
    if (pos == std::string::npos) return default_value;
    pos++;

    while (pos < json.size() && 
        (
            json[pos] == ' ' || 
            json[pos] == '\t' ||
            json[pos] == '\n' || 
            json[pos] == '\r'
        )
    )
        pos++;

    if (pos >= json.size()) return default_value;
    char* end = nullptr;
    float val = std::strtof(json.c_str() + pos, &end);
    if (end == json.c_str() + pos) return default_value;
    if (!std::isfinite(val)) return default_value;
    return val;
}

std::string HttpServer::extract_last_user_content(const std::string& json) {
    // Find the last "role":"user" entry and extract its "content" value.
    // Searches backwards for the last user message in the messages array.
    std::string result;
    size_t search_from = 0;

    while (true) {
        // Look for "role" : "user" patterns
        size_t role_pos = json.find("\"role\"", search_from);
        if (role_pos == std::string::npos) break;

        // Check if this role is "user"
        size_t colon = json.find(':', role_pos + 6);
        if (colon == std::string::npos) break;

        size_t quote = json.find('"', colon + 1);
        if (quote == std::string::npos) break;

        size_t end_quote = json.find('"', quote + 1);
        if (end_quote == std::string::npos) break;

        std::string role_value = json.substr(quote + 1, end_quote - quote - 1);

        if (role_value == "user") {
            // Find the "content" key near this role entry.
            // Search within a reasonable window after the role.
            size_t content_search_start = end_quote;
            size_t content_search_end = std::min(json.size(), content_search_start + 2048);
            std::string window = json.substr(content_search_start, content_search_end - content_search_start);
            std::string content = extract_json_string(window, "content");
            if (!content.empty()) {
                result = content;
            }
        }

        search_from = end_quote + 1;
    }

    return result;
}
