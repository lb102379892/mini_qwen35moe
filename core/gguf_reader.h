#pragma once

#include <ggml.h>
#include <ggml-backend.h>
#include <gguf.h>
#include <string>
#include <vector>
#include <cstdio>
#include <utility>
#include <cstring>
#include <algorithm>
#include <cstdint>
#include <limits>

class GGUFReader {
public:
    GGUFReader();
    ~GGUFReader();

    // ===== 禁用拷贝 =====
    GGUFReader(const GGUFReader&) = delete;
    GGUFReader& operator=(const GGUFReader&) = delete;

    // ===== 允许移动 =====
    GGUFReader(GGUFReader&& other) noexcept;

    GGUFReader& operator=(GGUFReader&& other) noexcept;

    bool open(const std::string& model_path_);

public:
    // ============================================================
    // 查询接口
    // ============================================================
    struct ggml_context* ggml_ctx() const;
    struct gguf_context* gguf_ctx() const;
    int tensor_count() const;
    int kv_count() const;

private:
    void close();

public:
    std::string model_path_{""}; 
    struct gguf_context* gguf_ctx_ = nullptr;  
    struct ggml_context* ggml_ctx_  = nullptr;
};
