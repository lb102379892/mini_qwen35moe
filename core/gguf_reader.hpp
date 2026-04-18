// GGUF 文件读取器 — RAII 管理文件生命周期
#ifndef FUNASR_CORE_GGUF_READER_HPP
#define FUNASR_CORE_GGUF_READER_HPP

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
    GGUFReader() = default;

    ~GGUFReader() {
        close();
    }

    // ===== 禁用拷贝 =====
    GGUFReader(const GGUFReader&) = delete;
    GGUFReader& operator=(const GGUFReader&) = delete;

    // ===== 允许移动 =====
    GGUFReader(GGUFReader&& other) noexcept
        : gctx_(other.gctx_)
        , ctx_(other.ctx_)
        , gpu_buf_(other.gpu_buf_)
        , uploaded_backend_(other.uploaded_backend_)
        , path_(std::move(other.path_))
        , missing_(std::move(other.missing_))
    {
        other.gctx_ = nullptr;
        other.ctx_  = nullptr;
        other.gpu_buf_ = nullptr;
        other.uploaded_backend_ = nullptr;
    }

    GGUFReader& operator=(GGUFReader&& other) noexcept {
        if (this != &other) {
            close();
            gctx_    = other.gctx_;
            ctx_     = other.ctx_;
            gpu_buf_ = other.gpu_buf_;
            uploaded_backend_ = other.uploaded_backend_;
            path_    = std::move(other.path_);
            missing_ = std::move(other.missing_);
            other.gctx_ = nullptr;
            other.ctx_  = nullptr;
            other.gpu_buf_ = nullptr;
            other.uploaded_backend_ = nullptr;
        }
        return *this;
    }

    // ============================================================
    // 打开 GGUF 文件
    // 分配内存并读取 tensor 数据到 ggml_context（默认 no_alloc=false）
    // ============================================================
    bool open(const std::string& path) {
        return open_impl(path, /* no_alloc = */ false);
    }

    // ============================================================
    // 打开 GGUF 文件（no_alloc=true）
    // 仅加载 metadata + tensor 信息，不在 CPU 分配权重数据
    // ============================================================
    bool open_no_alloc(const std::string& path) {
        return open_impl(path, /* no_alloc = */ true);
    }

    // ============================================================
    // 打开 GGUF 文件（可选 no_alloc）
    // ============================================================
    bool open(const std::string& path, bool no_alloc) {
        return open_impl(path, no_alloc);
    }

private:
    /**
     * @brief 打开并初始化 GGUF 文件读取
     * 
     * 该方法负责打开指定路径的 GGUF 格式文件，并初始化相关上下文
     * 
     * @param path 文件路径
     * @param no_alloc 是否不分配内存（true 表示只读取元数据，不分配张量数据内存）
     * @return bool 是否成功打开文件
     */
    bool open_impl(const std::string& path, bool no_alloc) {
        // 如果已经打开了别的文件，先关闭
        if (gctx_) {
            close();
        }

        // 保存文件路径并清空缺失项列表
        path_ = path;
        missing_.clear();

        // 初始化 GGUF 打开参数
        // ctx 传递二级指针，gguf_init_from_file 会填充它
        struct gguf_init_params params = {
            /*.no_alloc =*/ no_alloc,  // 是否不分配内存
            /*.ctx      =*/ &ctx_,     // 输出参数：GGML 上下文
        };

        // 从文件初始化 GGUF 上下文
        gctx_ = gguf_init_from_file(path.c_str(), params);
        if (!gctx_) {
            printf("[GGUFReader] ERROR: failed to open '%s'\n", path.c_str());
            ctx_ = nullptr;
            return false;
        }

        // 检查 GGML 上下文是否成功创建
        if (!ctx_) {
            printf("[GGUFReader] ERROR: ggml_context is null after opening '%s'\n", path.c_str());
            gguf_free(gctx_);
            gctx_ = nullptr;
            return false;
        }

        // 打印文件打开成功信息
        printf("[GGUFReader] Opened '%s' (no_alloc=%s)\n", path.c_str(), no_alloc ? "true" : "false");
        printf("  Tensors: %d\n", tensor_count());  // 打印张量数量
        printf("  KV pairs: %d\n", kv_count());     // 打印键值对数量

        return true;
    }

public:
    // ============================================================
    // 关闭文件，释放所有资源
    // 调用后所有 tensor 指针失效
    // ============================================================
    void close() {
        if (gpu_buf_) {
            ggml_backend_buffer_free(gpu_buf_);
            gpu_buf_ = nullptr;
        }
        uploaded_backend_ = nullptr;
        // 注意释放顺序：
        //   gguf_free 释放 metadata（不影响 ggml_context）
        //   ggml_free 释放 tensor 数据
        if (gctx_) {
            gguf_free(gctx_);
            gctx_ = nullptr;
        }
        if (ctx_) {
            ggml_free(ctx_);
            ctx_ = nullptr;
        }
        missing_.clear();
    }

    // ============================================================
    // 将当前 context 中的 tensor 数据上传到指定 backend（如 CUDA）
    // ============================================================
    /**
     * @brief 将 GGUF 文件中的张量上传到指定的后端
     * 
     * 该方法负责将已加载的 GGUF 张量数据上传到指定的计算后端（如 GPU），
     * 以便后续的模型推理可以在该后端上执行。
     * 
     * @param backend 目标计算后端
     * @return bool 是否成功上传
     */
    bool upload_to_backend(ggml_backend_t backend) {
        // 检查上下文和后端是否有效
        if (!ctx_ || !gctx_ || !backend) return false;
        
        // 检查是否已经上传到其他后端
        if (gpu_buf_) {
            if (uploaded_backend_ != backend) {
                printf("[GGUFReader] ERROR: tensors already uploaded to backend %p, cannot upload to %p\n",
                    (void*)uploaded_backend_, (void*)backend);
                return false;
            }
            // 已经上传到相同的后端，直接返回成功
            return true;
        }

        // 为后端分配张量缓冲区
        gpu_buf_ = ggml_backend_alloc_ctx_tensors(ctx_, backend);
        if (!gpu_buf_) {
            printf("[GGUFReader] ERROR: failed to allocate backend tensor buffer\n");
            return false;
        }

        // 重新打开文件以读取张量数据
        FILE* f = std::fopen(path_.c_str(), "rb");
        if (!f) {
            printf("[GGUFReader] ERROR: failed to reopen '%s' for tensor upload\n", path_.c_str());
            ggml_backend_buffer_free(gpu_buf_);
            gpu_buf_ = nullptr;
            return false;
        }

        bool ok = true;
        // 创建临时缓冲区用于分块读取
        std::vector<uint8_t> staging(kUploadStagingBytes);
        // 获取数据偏移量和张量数量
        const size_t data_offset = gguf_get_data_offset(gctx_);
        const int n_tensors = gguf_get_n_tensors(gctx_);

        // 遍历所有张量并上传
        for (int i = 0; i < n_tensors && ok; ++i) {
            const char* name = gguf_get_tensor_name(gctx_, i);
            ggml_tensor* t = ggml_get_tensor(ctx_, name);
            if (!t) {
                continue; // 跳过不存在的张量
            }

            const size_t nbytes = ggml_nbytes(t);
            if (nbytes == 0) {
                continue; // 跳过空张量
            }

            // 计算张量在文件中的偏移量
            const size_t tensor_offset = gguf_get_tensor_offset(gctx_, i);
            const size_t file_offset = data_offset + tensor_offset;

            // 检查偏移量是否有效并定位文件指针
            if (file_offset > static_cast<size_t>(std::numeric_limits<off_t>::max()) ||
                fseeko(f, static_cast<off_t>(file_offset), SEEK_SET) != 0) {
                printf("[GGUFReader] ERROR: seek failed for tensor '%s'\n", name);
                ok = false;
                break;
            }

            // 分块读取并上传张量数据
            size_t done = 0;
            while (done < nbytes) {
                // 计算当前块大小
                const size_t chunk = std::min(staging.size(), nbytes - done);
                // 读取数据
                const size_t got = std::fread(staging.data(), 1, chunk, f);
                if (got != chunk) {
                    // 处理读取错误
                    if (std::ferror(f)) {
                        printf("[GGUFReader] ERROR: read error while uploading tensor '%s'\n", name);
                    } else {
                        printf("[GGUFReader] ERROR: unexpected EOF while uploading tensor '%s'\n", name);
                    }
                    ok = false;
                    break;
                }
                // 将数据上传到后端张量
                ggml_backend_tensor_set(t, staging.data(), done, chunk);
                done += chunk;
            }
        }

        // 关闭文件
        std::fclose(f);
        
        // 处理上传失败的情况
        if (!ok) {
            ggml_backend_buffer_free(gpu_buf_);
            gpu_buf_ = nullptr;
            uploaded_backend_ = nullptr;
            return false;
        }

        // 标记上传成功并记录后端
        uploaded_backend_ = backend;
        return true;
    }

    // ============================================================
    // 按名字获取 tensor
    // 返回 nullptr 表示未找到（不记录错误）
    // ============================================================
    struct ggml_tensor* get_tensor(const std::string& name) const {
        if (!ctx_) return nullptr;
        return ggml_get_tensor(ctx_, name.c_str());
    }

    // ============================================================
    // 按名字获取 tensor，必须存在
    // 未找到时记录到 missing 列表，返回 nullptr
    // 这样可以一次加载完后统一报告所有缺失的 tensor
    // ============================================================
    struct ggml_tensor* require_tensor(const std::string& name) {
        struct ggml_tensor* t = get_tensor(name);
        if (!t) {
            missing_.push_back(name);
        }
        return t;
    }

    // ============================================================
    // 查询接口
    // ============================================================
    bool is_open()                          const { return gctx_ != nullptr && ctx_ != nullptr; }
    struct gguf_context* gguf_ctx()         const { return gctx_; }
    struct ggml_context* ggml_ctx()         const { return ctx_; }
    const std::string& path()               const { return path_; }

    int tensor_count() const {
        return gctx_ ? gguf_get_n_tensors(gctx_) : 0;
    }

    int kv_count() const {
        return gctx_ ? gguf_get_n_kv(gctx_) : 0;
    }

    // ============================================================
    // 错误追踪
    // ============================================================
    const std::vector<std::string>& missing_tensors() const { return missing_; }
    bool has_errors()                                 const { return !missing_.empty(); }

    // 清除错误记录（在开始新一轮加载前调用）
    void clear_errors() { missing_.clear(); }

    // 打印所有缺失的 tensor
    void print_errors() const {
        if (missing_.empty()) {
            printf("[GGUFReader] No errors.\n");
            return;
        }
        printf("[GGUFReader] %zu missing tensor(s):\n", missing_.size());
        for (const auto& name : missing_) {
            printf("  - %s\n", name.c_str());
        }
    }

private:
    static constexpr size_t kUploadStagingBytes = 64u * 1024u * 1024u;
    struct gguf_context* gctx_ = nullptr;  // GGUF metadata (keys, tensor info)
    struct ggml_context* ctx_  = nullptr;  // GGML tensor data (weights in memory)
    ggml_backend_buffer_t gpu_buf_ = nullptr;
    ggml_backend_t uploaded_backend_ = nullptr;
    std::string path_;
    std::vector<std::string> missing_;     // require_tensor 失败时记录
};

#endif