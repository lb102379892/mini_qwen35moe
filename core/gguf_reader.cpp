#include "core/gguf_reader.h"

GGUFReader::GGUFReader() {
};

GGUFReader::~GGUFReader() {
    close();
}

GGUFReader::GGUFReader(GGUFReader&& other) noexcept
    :model_path_(std::move(other.model_path_)), gguf_ctx_(other.gguf_ctx_), ggml_ctx_(other.ggml_ctx_)
{
    other.gguf_ctx_ = nullptr;
    other.ggml_ctx_ = nullptr;
}

GGUFReader& GGUFReader::operator=(GGUFReader&& other) noexcept {
    if (this != &other) {
        close();
        model_path_ = std::move(other.model_path_);
        gguf_ctx_ = other.gguf_ctx_;
        ggml_ctx_ = other.ggml_ctx_;

        other.gguf_ctx_ = nullptr;
        other.ggml_ctx_ = nullptr;
    }
    return *this;
}

bool GGUFReader::open(const std::string& model_path) {
    if (gguf_ctx_) {
        close();
    }

    model_path_ = model_path;

    // 初始化 GGUF 打开参数
    // ctx 传递二级指针，gguf_init_from_file 会填充它
    struct gguf_init_params params = {
        /*.no_alloc =*/ true,       // 是否不分配内存
        /*.ctx      =*/ &ggml_ctx_, // 输出参数：GGML 上下文
    };

    // 从文件初始化 GGUF 上下文
    gguf_ctx_ = gguf_init_from_file(model_path.c_str(), params);
    if (!gguf_ctx_) {
        printf("[GGUFReader] ERROR: failed to open '%s'\n", model_path.c_str());
        gguf_ctx_ = nullptr;
        return false;
    }
    if (!ggml_ctx_) {
        printf("[GGUFReader] ERROR: ggml_context is null after opening '%s'\n", model_path.c_str());
        gguf_free(gguf_ctx_);
        gguf_ctx_ = nullptr;
        return false;
    }

    printf("[GGUFReader] Opened '%s' Tensors: %d, KV pairs: %d\n", model_path.c_str(), tensor_count(), kv_count());

    return true;
}

int GGUFReader::tensor_count() const {
    return gguf_ctx_ ? gguf_get_n_tensors(gguf_ctx_) : 0;
}

int GGUFReader::kv_count() const {
    return gguf_ctx_ ? gguf_get_n_kv(gguf_ctx_) : 0;
}

void GGUFReader::close() {
    if (gguf_ctx_) {
        gguf_free(gguf_ctx_);
        gguf_ctx_ = nullptr;
    }
    if (ggml_ctx_) {
        ggml_free(ggml_ctx_);
        ggml_ctx_ = nullptr;
    }
}

