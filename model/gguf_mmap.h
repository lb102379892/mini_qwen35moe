#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <ggml.h>
#include <ggml-backend.h>
#include <gguf.h>
#include "common.h"

#define GGUF_MAX_STRING_LENGTH  (1024*1024*1024)
#define GGUF_MAX_ARRAY_ELEMENTS (1024*1024*1024)

template <typename T>
struct template_to_gguf_attr;

template <>
struct template_to_gguf_attr<uint8_t> {
    static constexpr enum gguf_type value = GGUF_TYPE_UINT8;
    static constexpr size_t type_size = sizeof(uint8_t);
};

template <>
struct template_to_gguf_attr<int8_t> {
    static constexpr enum gguf_type value = GGUF_TYPE_INT8;
    static constexpr size_t type_size = sizeof(int8_t);
};

template <>
struct template_to_gguf_attr<uint16_t> {
    static constexpr enum gguf_type value = GGUF_TYPE_UINT16;
    static constexpr size_t type_size = sizeof(uint16_t);
};

template <>
struct template_to_gguf_attr<int16_t> {
    static constexpr enum gguf_type value = GGUF_TYPE_INT16;
    static constexpr size_t type_size = sizeof(int16_t);
};

template <>
struct template_to_gguf_attr<uint32_t> {
    static constexpr enum gguf_type value = GGUF_TYPE_UINT32;
    static constexpr size_t type_size = sizeof(uint32_t);
};

template <>
struct template_to_gguf_attr<int32_t> {
    static constexpr enum gguf_type value = GGUF_TYPE_INT32;
    static constexpr size_t type_size = sizeof(int32_t);
};

template <>
struct template_to_gguf_attr<float> {
    static constexpr enum gguf_type value = GGUF_TYPE_FLOAT32;
    static constexpr size_t type_size = sizeof(float);
};

template <>
struct template_to_gguf_attr<bool> {
    static constexpr enum gguf_type value = GGUF_TYPE_BOOL;
    static constexpr size_t type_size = sizeof(int8_t);
};

template <>
struct template_to_gguf_attr<std::string> {
    static constexpr enum gguf_type value = GGUF_TYPE_STRING;
    static constexpr size_t type_size = 0;
};

template <>
struct template_to_gguf_attr<uint64_t> {
    static constexpr enum gguf_type value = GGUF_TYPE_UINT64;
    static constexpr size_t type_size = sizeof(uint64_t);
};

template <>
struct template_to_gguf_attr<int64_t> {
    static constexpr enum gguf_type value = GGUF_TYPE_INT64;
    static constexpr size_t type_size = sizeof(int64_t);
};

template <>
struct template_to_gguf_attr<double> {
    static constexpr enum gguf_type value = GGUF_TYPE_FLOAT64;
    static constexpr size_t type_size = sizeof(double);
};

struct kv_info {
    std::string key = "";

    bool is_array = false;
    enum gguf_type type = GGUF_TYPE_COUNT;

    std::vector<int8_t> data;
    std::vector<std::string> data_string;

    template <typename T>
    kv_info(const std::string & key, const T value) 
    : key(key), is_array(false), type(template_to_gguf_attr<T>::value) {
        data.resize(sizeof(T));
        memcpy(data.data(), &value, sizeof(T));
    }

    template <typename T>
    kv_info(const std::string & key, const std::vector<T> & value)
    : key(key), is_array(true), type(template_to_gguf_attr<T>::value) {
        data.resize(value.size()*sizeof(T));
        for (size_t i = 0; i < value.size(); ++i) {
            const T tmp = value[i];
            memcpy(data.data() + i*sizeof(T), &tmp, sizeof(T));
        }
    }

    kv_info(const std::string & key, const std::string & value)
    : key(key), is_array(false), type(GGUF_TYPE_STRING) {
        data_string.push_back(value);
    }

    kv_info(const std::string & key, const std::vector<std::string> & value)
    : key(key), is_array(true), type(GGUF_TYPE_STRING) {
        data_string = value;
    }

    template <typename T>
    const T & get_val(const size_t i = 0) const {
        if (template_to_gguf_attr<T>::value != type) {
            printf("ERROR: type mismatch when getting value for key %s, expected type %d but actual type is %d\n", key.c_str(), template_to_gguf_attr<T>::value, type);
            throw std::runtime_error("type mismatch");
        }
        if constexpr (std::is_same<T, std::string>::value) {
            if (data_string.size() < i+1) {
                printf("ERROR: index out of bounds when getting string value for key %s, index: %lu, size: %lu\n", key.c_str(), i, data_string.size());
                throw std::runtime_error("data_string size mismatch");
            }
            return data_string[i];
        }
        const size_t type_size = template_to_gguf_attr<T>::type_size;
        if (data.size() % type_size != 0) {
            printf("ERROR: data size is not a multiple of type size for key %s, data size: %lu, type size: %lu\n", key.c_str(), data.size(), type_size);
            throw std::runtime_error("data size mismatch");
        }
        if (data.size() < (i+1)*type_size) {
            printf("ERROR: index out of bounds when getting value for key %s, index: %lu, size: %lu\n", key.c_str(), i, data.size() / type_size);
            throw std::runtime_error("data size mismatch");
        }
        return reinterpret_cast<const T *>(data.data())[i];
    }
};

struct tensor_info {
    // 张量名称
    std::string name = "";

    // 张量维度
    uint32_t n_dims = 0;

    // 张量形状
    int64_t dims[GGML_MAX_DIMS];

    // 张量类型
    enum ggml_type type = GGML_TYPE_COUNT;

    // 张量每个类型元素的字节大小
    size_t type_size = 0;

    // 张量块大小（如果是量化类型）
    int64_t blck_size = 0;

    // 张量总字节大小
    size_t byte_size = 0;

    // 张量数据在文件中的偏移量
    uint64_t offset = 0;

    // 张量所属层索引，-1 表示不属于任何层
    int layer_idx = -1;

    // 张量权重类型
    EN_WEIGHT_TYPE weight_type = EN_WEIGHT_TYPE_COUNT;
};

class GGUFLoader {
public:
    GGUFLoader();
    ~GGUFLoader();

    bool load(const std::string& path);
    void unload();
    size_t get_all_tensor_bytesize();
    size_t get_tensor_bytesize(const tensor_info& tensor);
    bool load_tensor_head_data(ggml_tensor* dst, ggml_backend_t upload_backend = nullptr);
    bool load_tensor_layer_data(ggml_tensor* dst, ggml_backend_t upload_backend = nullptr);
    void get_tensor_data(tensor_info* tensor, std::vector<uint8_t>& src_data);
    const uint8_t* get_tensor_file_ptr(const tensor_info& tensor) const;

    // Zero-copy mmap helpers for CPU backend.
    // Returns true when the mmap region is still open and usable.
    bool is_mmap_active() const;
    // Create a CPU backend buffer that wraps the entire mmap tensor-data region.
    // Returns nullptr on failure (alignment issue, mmap not active, etc.).
    ggml_backend_buffer_t create_cpu_mmap_buffer();
    // Bind a single tensor to an existing mmap-backed buffer without copying.
    // is_head=true  => look up in tensors_head_, false => tensors_layer_.
    // Returns false when any safety check fails (falls back to copy path).
    bool bind_tensor_to_mmap(ggml_backend_buffer_t buf, ggml_tensor* dst, bool is_head);

    int32_t get_i32_or(const char* key, int default_val);
    uint32_t get_u32_or(const char* key, uint32_t default_val);
    float get_f32_or(const char* key, float default_val);
    std::string get_str_or(const char* key, const char* default_val);
    bool get_bool_or(const char* key, bool default_val);
    std::vector<std::string> get_arrary_string_or(const char* key);
    std::vector<uint32_t> get_arrary_u32_or(const char* key);
    std::vector<int32_t> get_arrary_i32_or(const char* key);

private:
    bool load_model_file(const std::string& path);
    bool prase_model_file();

    template <typename T>
    bool read(T& dst) const {
        const size_t size = sizeof(dst);
        if (nbytes_remain_ < size) {
            printf("%s: failed to read value(size:%lu)\n", __func__, size);
            return false;
        }

        std::memcpy(&dst, data_ + pos_, size);
        pos_ += size;
        nbytes_remain_ -= size;
        return true;
    }

    template <typename T>
    bool read(std::vector<T> & dst, const size_t n) const {
        if (n > GGUF_MAX_ARRAY_ELEMENTS) {
            printf("%s: failed to read value(n:%lu)\n", __func__, n);
            return false;
        }
        if constexpr (std::is_same<T, std::string>::value) {
            // strings are prefixed with their length, so we need to account for that
            if (n > SIZE_MAX / sizeof(uint64_t)) {
                printf("%s: failed to read value(n:%lu)\n", __func__, n);
                return false;
            }
            if (nbytes_remain_ < n * sizeof(uint64_t)) {
                printf("%s: failed to read value(n:%lu)\n", __func__, n);
                return false;
            }
        } else {
            if (n > SIZE_MAX / sizeof(T)) {
                printf("%s: failed to read value(n:%lu)\n", __func__, n);
                return false;
            }
            if (nbytes_remain_ < n * sizeof(T)) {
                printf("%s: failed to read value(n:%lu)\n", __func__, n);
                return false;
            }
        }
        
        dst.resize(n);
        for (size_t i = 0; i < dst.size(); ++i) {
            if constexpr (std::is_same<T, bool>::value) {
                bool tmp;
                if (!read(tmp)) {
                    printf("%s: failed to read value(i:%lu)\n", __func__, i);
                    return false;
                }
                dst[i] = tmp;
            } else {
                if (!read(dst[i])) {
                    printf("%s: failed to read value(i:%lu)\n", __func__, i);
                    return false;
                }
            }
        }
        return true;
    }

    bool read(bool & dst) const;
    bool read(enum ggml_type & dst) const;
    bool read(enum gguf_type & dst) const;
    bool read(std::string & dst) const;
    bool read(void * dst, const size_t size) const;

    template<typename T>
    bool read_value(const std::string & key, const bool is_array, const size_t array_n) {
        bool rnt = false;
        if (is_array) {
            std::vector<T> value;
            rnt = read(value, array_n);
            if (false == rnt) {
                printf("%s: failed to read arrary value(key:%s, array_n:%lu)\n", __func__, key.c_str(), array_n);
                return false;
            }
            kv_.emplace_back(key, value);
        } else {
            T value;
            rnt = read(value);
            if (false == rnt) {
                printf("%s: failed to read value(key:%s)\n", __func__, key.c_str());
                return false;
            }
            kv_.emplace_back(key, value);
        }
        kv_index_map_[key] = kv_.size() - 1;
        return true;
    }

    void print_prase_info() const;

    int find_key(const char* key);

public:
    std::string magic_ = "";
    uint32_t version_ = 0;
    int64_t n_tensors_ = 0;
    int64_t n_kv_ = 0;

    size_t data_offset_ = 0;
    std::vector<struct kv_info> kv_;
    std::map<std::string, size_t> kv_index_map_;
    std::vector<struct tensor_info> tensors_layer_;
    std::map<std::string, size_t> tensor_layer_index_map_;
    std::map<int, std::set<size_t>, std::less<int>> tensor_layer_index_list_;
    std::map<std::string, struct tensor_info> tensors_head_;
    uint32_t alignment_idx_ = 0;

private:
    int fd_ = -1;
    void* mmap_base_ = nullptr;
    size_t mmap_size_ = 0;
    std::string last_error_ = "";

    uint8_t* data_ = nullptr;
    mutable size_t pos_ = 0;
    mutable uint64_t nbytes_remain_ = 0;
};
