#pragma once

#include <string>

// API version prefix
static constexpr const char* kApiPrefix = "/v1";

// Model identifier reported in responses
static constexpr const char* kModelId = "qwen3.5-35b-a3b";

// Server version
static constexpr const char* kServerVersion = "0.1.0";

// Escape a string for safe embedding in a JSON value.
std::string json_escape(const std::string& s);

// Build an OpenAI-compatible error response body.
std::string make_error_response(const std::string& message, const std::string& type);
