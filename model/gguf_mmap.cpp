#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cerrno>
#include <cstring>
#include <algorithm>
#include <string>
#include <ggml.h>
#include <ggml-backend.h>
#include "gguf_mmap.h"

std::set<std::string> g_tensor_head_name_list = {
    WEIGHT_TOKEN_EMBD_NAME,
    WEIGHT_OUTPUT_NAME,
    WEIGHT_OUTPUT_NORM_NAME
};

GGUFLoader::GGUFLoader() {
}

GGUFLoader::~GGUFLoader() {
    unload();
}

bool GGUFLoader::load(const std::string& path) {
    bool rnt = load_model_file(path);
    if (false == rnt) {
        return false;
    }

    rnt = prase_model_file();
    if (false == rnt) {
        return false;
    }

    //print_prase_info();

    return true;
}

void GGUFLoader::unload() {
    if (mmap_base_ && mmap_base_ != MAP_FAILED) {
        munmap(mmap_base_, mmap_size_);
    }
    mmap_base_ = nullptr;
    mmap_size_ = 0;
    data_ = nullptr;
    nbytes_remain_ = 0;
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
}

int32_t GGUFLoader::get_i32_or(const char* key, int default_val) {
    int idx = find_key(key);
    if (idx < 0) 
        return default_val;

    return kv_[idx].get_val<int32_t>();
}

uint32_t GGUFLoader::get_u32_or(const char* key, uint32_t default_val) {
    int idx = find_key(key);
    if (idx < 0) 
        return default_val;
    return kv_[idx].get_val<uint32_t>();
}

float GGUFLoader::get_f32_or(const char* key, float default_val) {
    int idx = find_key(key);
    if (idx < 0) 
        return default_val;
    return kv_[idx].get_val<float>();
}

std::string GGUFLoader::get_str_or(const char* key, const char* default_val) {
    int idx = find_key(key);
    if (idx < 0) 
        return default_val;
    return kv_[idx].get_val<std::string>();
}

bool GGUFLoader::get_bool_or(const char* key, bool default_val) {
    int idx = find_key(key);
    if (idx < 0) 
        return default_val;
    return kv_[idx].get_val<bool>();
}

std::vector<std::string> GGUFLoader::get_arrary_string_or(const char* key) {
    std::vector<std::string> result;
    int idx = find_key(key);
    if (idx < 0) 
        return result;

    return kv_[idx].data_string;
}

std::vector<uint32_t> GGUFLoader::get_arrary_u32_or(const char* key) {
    std::vector<uint32_t> result;
    int idx = find_key(key);
    if (idx < 0) 
        return result;
    int n = kv_[idx].data.size()/sizeof(uint32_t);
    result.resize(n);
    const uint32_t * values = (const uint32_t *)kv_[idx].data.data();
    for (int i = 0; i < n; ++i) {
        result[i] = values[i];
    }

    return result;
}

std::vector<int32_t> GGUFLoader::get_arrary_i32_or(const char* key) {
    std::vector<int32_t> result;
    int idx = find_key(key);
    if (idx < 0) 
        return result;
    int n = kv_[idx].data.size()/sizeof(int32_t);
    result.resize(n);
    const int32_t * values = (const int32_t *)kv_[idx].data.data();
    for (int i = 0; i < n; ++i) {
        result[i] = values[i];
    }

    return result;
}

bool GGUFLoader::load_model_file(const std::string& path) {
    if (mmap_base_ || fd_ >= 0) {
        last_error_ = "Loader already has an open file; create a new instance";
        printf("%s\n", last_error_.c_str());
        return false;
    }

    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        last_error_ = "Failed to open file: " + path;
        printf("%s\n", last_error_.c_str());
        return false;
    }

    struct stat st;
    if (fstat(fd_, &st) != 0) {
        last_error_ = "Failed to stat file: " + path;
        printf("%s\n", last_error_.c_str());
        close(fd_);
        fd_ = -1;
        return false;
    }

    mmap_size_ = static_cast<size_t>(st.st_size);
    mmap_base_ = mmap(nullptr, mmap_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mmap_base_ == MAP_FAILED) {
        last_error_ = "mmap failed for " + path;
        printf("%s\n", last_error_.c_str());
        mmap_base_ = nullptr;
        close(fd_);
        fd_ = -1;
        return false;
    }

    if (madvise(mmap_base_, mmap_size_, MADV_RANDOM) != 0) {
        printf("madvise(MADV_RANDOM) failed: %s\n", strerror(errno));
    }
    nbytes_remain_ = mmap_size_;
    data_ = static_cast<uint8_t*>(mmap_base_);

    return true;
}

bool GGUFLoader::prase_model_file() {
    bool rnt = true;
    // "GGUF"文件标识
    {
        std::vector<char> magic;
        rnt = read(magic, 4);
        if (false == rnt) {
            printf("%s: failed to read magic\n", __func__);
            return false;
        }
        for (uint32_t i = 0; i < magic.size(); i++) {
            if (magic[i] != GGUF_MAGIC[i]) {
                char c0 = isprint(magic[0]) ? magic[0] : '?';
                char c1 = isprint(magic[1]) ? magic[1] : '?';
                char c2 = isprint(magic[2]) ? magic[2] : '?';
                char c3 = isprint(magic[3]) ? magic[3] : '?';
                printf("%s: invalid magic characters: '%c%c%c%c', expected 'GGUF'\n", __func__, c0, c1, c2, c3);
                return false;
            }
        }
        magic_ = std::string(magic.data(), magic.size());
    }

    // 文件头
    {
        rnt = read(version_);
        if (false == rnt) {
            printf("%s: failed to read version\n", __func__);
            return false;
        }

        rnt = read(n_tensors_);
        if (false == rnt) {
            printf("%s: failed to read tensor_count\n", __func__);
            return false;
        }

        rnt = read(n_kv_);
        if (false == rnt) {
            printf("%s: failed to read kv_count\n", __func__);
            return false;
        }
    }

    // KV键值对
    {
        for (int64_t i = 0; i < n_kv_; ++i) {
            std::string key;
            rnt = read(key);
            if (false == rnt) {
                printf("%s: failed to read key(i:%ld)\n", __func__, i);
                return false;
            }

            gguf_type type = gguf_type(-1);
            rnt = read(type);
            if (false == rnt) {
                printf("%s: failed to read type(i:%ld,key:%s)\n", __func__, i, key.c_str());
                return false;
            }

            bool is_array = false;
            uint64_t array_n = 1;
            if (type == GGUF_TYPE_ARRAY) {
                rnt = read(type);
                if (false == rnt) {
                    printf("%s: failed to read array type(i:%ld,key:%s)\n", __func__, i, key.c_str());
                    return false;
                }
                rnt = read(array_n);
                if (false == rnt) {
                    printf("%s: failed to read array size(i:%ld,type:%d,key:%s)\n", __func__, i, type, key.c_str());
                    return false;
                }
                is_array = true;
            }

            switch (type) {
                case GGUF_TYPE_UINT8:   
                    rnt = read_value<uint8_t>(key, is_array, array_n); 
                    break;
                case GGUF_TYPE_INT8:    
                    rnt = read_value<int8_t>(key, is_array, array_n); 
                    break;
                case GGUF_TYPE_UINT16:  
                    rnt = read_value<uint16_t>(key, is_array, array_n); 
                    break;
                case GGUF_TYPE_INT16:   
                    rnt = read_value<int16_t>(key, is_array, array_n); 
                    break;
                case GGUF_TYPE_UINT32:  
                    rnt = read_value<uint32_t>(key, is_array, array_n); 
                    break;
                case GGUF_TYPE_INT32:   
                    rnt = read_value<int32_t>(key, is_array, array_n); 
                    break;
                case GGUF_TYPE_FLOAT32: 
                    rnt = read_value<float>(key, is_array, array_n); 
                    break;
                case GGUF_TYPE_BOOL:    
                    rnt = read_value<bool>(key, is_array, array_n); 
                    break;
                case GGUF_TYPE_STRING:  
                    rnt = read_value<std::string>(key, is_array, array_n); 
                    break;
                case GGUF_TYPE_UINT64:  
                    rnt = read_value<uint64_t>(key, is_array, array_n); 
                    break;
                case GGUF_TYPE_INT64:   
                    rnt = read_value<int64_t>(key, is_array, array_n); 
                    break;
                case GGUF_TYPE_FLOAT64: 
                    rnt = read_value<double>(key, is_array, array_n); 
                    break;
                case GGUF_TYPE_ARRAY:
                default:
                    {
                        printf("%s: key '%s' has invalid GGUF type %d\n", __func__, key.c_str(), type);
                        rnt = false;
                    } break;
            }
            if (false == rnt) {
                printf("%s: failed to read kv(i:%ld,type:%d,key:%s,is_array:%d,array_n:%lu)\n", __func__, i, type, key.c_str(), is_array, array_n);
                return false;
            }
        }

        alignment_idx_ = get_u32_or(GGUF_KEY_GENERAL_ALIGNMENT, GGUF_DEFAULT_ALIGNMENT);
        if (alignment_idx_ == 0 || (alignment_idx_ & (alignment_idx_ - 1)) != 0) {
            printf("%s: alignment %u is not a power of 2\n", __func__, alignment_idx_);
            return false;
        }
    }

    // 读取张量信息
    {
        for (int64_t i = 0; i < n_tensors_; ++i) {
            struct tensor_info info;

            // 张量名称
            rnt = read(info.name);
            if (false == rnt) {
                printf("%s: failed to read tensor name(i:%ld)\n", __func__, i);
                return false;
            }

            // 张量维度
            rnt = read(info.n_dims);
            if (false == rnt) {
                printf("%s: failed to read tensor n_dims(i:%ld,name:%s)\n", __func__, i, info.name.c_str());
                return false;
            }
            if (info.n_dims > GGML_MAX_DIMS) {
                printf("%s: tensor n_dims is not invaild(i:%ld,name:%s,n_dims:%u)\n", __func__, i, info.name.c_str(), info.n_dims);
                return false;
            }

            // 张量形状
            for (uint32_t j = 0; j < info.n_dims; ++j) {
                info.dims[j] = 1;
                rnt = read(info.dims[j]);
                if (false == rnt) {
                    printf("%s: failed to read tensor dims(i:%ld,j:%u,name:%s)\n", __func__, i, j, info.name.c_str());
                    return false;
                }
                if (info.dims[j] < 0) {
                    printf("%s: dims is not invaild(i:%ld,j:%u,name:%s,dims:%ld)\n", __func__, i, j, info.name.c_str(), info.dims[j]);
                    return false;
                }
            }

            // 张量类型
            rnt = read(info.type);
            if (false == rnt) {
                printf("%s: failed to read tensor type(i:%ld,name:%s)\n", __func__, i, info.name.c_str());
                return false;
            }
            if (info.type < 0 || info.type >= GGML_TYPE_COUNT) {
                printf("%s: tensor type is not invaild(i:%ld,name:%s,type:%d)\n", __func__, i, info.name.c_str(), info.type);
                return false;
            }
            info.type_size = ggml_type_size(info.type);
            info.blck_size = ggml_blck_size(info.type);
            // 检查行大小是否能被块大小整除
            if (info.blck_size == 0 || info.dims[0] % info.blck_size != 0) {
                printf("%s: tensor %s of type %d (%s) has %ld elements per row, not a multiple of block size (%ld)\n",
                    __func__, info.name.c_str(), (int)info.type, ggml_type_name(info.type), info.dims[0], info.blck_size);
                return false;
            }

            // 张量总字节大小
            uint64_t n_elements = 1;
            for (uint32_t d = 0; d < info.n_dims; ++d) {
                n_elements *= info.dims[d];
            }
            info.byte_size = (n_elements / info.blck_size) * info.type_size;

            // 张量数据偏移量
            rnt = read(info.offset);
            if (false == rnt) {
                printf("%s: failed to read tensor offset(i:%ld,name:%s)\n", __func__, i, info.name.c_str());
                return false;
            }

            if (g_tensor_head_name_list.find(info.name) != g_tensor_head_name_list.end()) {
                info.layer_idx = -1;
                auto iter = g_tensor_head_names.find(info.name);
                if (iter != g_tensor_head_names.end()) {
                    info.weight_type = iter->second;
                }
                tensors_head_[info.name] = info;
            } else {
                info.layer_idx = extractNumber(info.name);
                if (info.layer_idx < 0) {
                    printf("%s: failed to extract layer index from tensor name '%s'\n", __func__, info.name.c_str());
                    return false;
                }
                auto aftername = getAfterNumber(info.name);
                auto iter = g_tensor_layer_names.find(aftername);
                if (iter != g_tensor_layer_names.end()) {
                    info.weight_type = iter->second;
                }
                tensors_layer_.push_back(info);
                auto index = tensors_layer_.size() - 1;
                tensor_layer_index_map_[info.name] = index;
                tensor_layer_index_list_[info.layer_idx].insert(index);
            }
            
        }
    }
    data_offset_ = pos_;
    data_offset_ = GGML_PAD(data_offset_, alignment_idx_);

    return true;
}

size_t GGUFLoader::get_all_tensor_bytesize() {
    size_t total = 0;
    for (const auto& info : tensors_layer_) {
        total += get_tensor_bytesize(info);
    }
    for (const auto& info : tensors_head_) {
        total += get_tensor_bytesize(info.second);
    }

    return total;
}

size_t GGUFLoader::get_tensor_bytesize(const tensor_info& tensor) {
    size_t total = 0;
    const size_t ggml_tensor_struct_size = sizeof(ggml_tensor);

    // 内存对齐要求（通常是 32 字节）,将数值 info.byte_size 向上取整到 ctx->alignment 的最小倍数，ctx->alignment的倍数
    size_t padded_size = GGML_PAD(tensor.byte_size, alignment_idx_);
    total += padded_size + ggml_tensor_struct_size;
    total = GGML_PAD(total, alignment_idx_);

    return total;
}

static void upload_tensor_from_file(
    ggml_tensor* dst,
    const void* src,
    size_t nbytes,
    ggml_backend_t upload_backend
) {
    if (upload_backend != nullptr) {
        ggml_backend_tensor_set_async(upload_backend, dst, src, 0, nbytes);
    } else {
        ggml_backend_tensor_set(dst, src, 0, nbytes);
    }
}

bool GGUFLoader::load_tensor_head_data(ggml_tensor* dst, ggml_backend_t upload_backend) {
    // find() is used (instead of operator[]) so the lookup is read-only and
    // safe to call concurrently from multiple threads on a fully-parsed loader
    // (operator[] could insert a default entry on a missing key).
    auto it = tensors_head_.find(dst->name);
    if (it == tensors_head_.end()) {
        return false;
    }
    const tensor_info* tensor_info = &it->second;
    const void* src = get_tensor_file_ptr(*tensor_info);
    if (!src) {
        return false;
    }
    upload_tensor_from_file(dst, src, tensor_info->byte_size, upload_backend);
    return true;
}

bool GGUFLoader::load_tensor_layer_data(ggml_tensor* dst, ggml_backend_t upload_backend) {
    auto idx_it = tensor_layer_index_map_.find(dst->name);
    if (idx_it == tensor_layer_index_map_.end()) {
        return false;
    }
    const auto index = idx_it->second;
    if (index >= tensors_layer_.size()) {
        return false;
    }
    const tensor_info* tensor_info = &tensors_layer_[index];
    const void* src = get_tensor_file_ptr(*tensor_info);
    if (!src) {
        return false;
    }
    upload_tensor_from_file(dst, src, tensor_info->byte_size, upload_backend);
    return true;
}

const uint8_t* GGUFLoader::get_tensor_file_ptr(const tensor_info& tensor) const {
    if (!data_) {
        return nullptr;
    }
    return data_ + data_offset_ + tensor.offset;
}

void GGUFLoader::get_tensor_data(tensor_info* tensor, std::vector<uint8_t>& src_data) {
    const uint8_t* src = get_tensor_file_ptr(*tensor);
    if (!src) {
        return;
    }
    std::memcpy(src_data.data(), src, tensor->byte_size);
}

bool GGUFLoader::is_mmap_active() const {
    return mmap_base_ != nullptr && mmap_base_ != MAP_FAILED;
}

ggml_backend_buffer_t GGUFLoader::create_cpu_mmap_buffer() {
    if (!is_mmap_active()) {
        return nullptr;
    }

    // Compute the extent of the tensor data section (offsets are relative to data_offset_).
    size_t last_byte = 0;
    for (const auto& t : tensors_layer_) {
        size_t end = t.offset + t.byte_size;
        if (end > last_byte) last_byte = end;
    }
    for (const auto& kv : tensors_head_) {
        size_t end = kv.second.offset + kv.second.byte_size;
        if (end > last_byte) last_byte = end;
    }
    if (last_byte == 0) {
        return nullptr;
    }

    uint8_t* base = data_ + data_offset_;
    // ggml_backend_cpu_buffer_from_ptr requires TENSOR_ALIGNMENT (32-byte) alignment.
    if ((uintptr_t)base % 32 != 0) {
        fprintf(stderr, "[mmap] data region not 32-byte aligned (addr=%p, data_offset=%zu) -- fallback to copy\n",
                (void*)base, data_offset_);
        return nullptr;
    }
    return ggml_backend_cpu_buffer_from_ptr(base, last_byte);
}

bool GGUFLoader::bind_tensor_to_mmap(ggml_backend_buffer_t buf, ggml_tensor* dst, bool is_head) {
    if (!buf || !dst) {
        return false;
    }

    const tensor_info* ti = nullptr;
    if (is_head) {
        auto it = tensors_head_.find(dst->name);
        if (it == tensors_head_.end()) {
            fprintf(stderr, "[mmap] head tensor '%s' not found\n", dst->name);
            return false;
        }
        ti = &it->second;
    } else {
        auto it = tensor_layer_index_map_.find(dst->name);
        if (it == tensor_layer_index_map_.end()) {
            fprintf(stderr, "[mmap] layer tensor '%s' not found\n", dst->name);
            return false;
        }
        ti = &tensors_layer_[it->second];
    }

    uint8_t* addr = data_ + data_offset_ + ti->offset;

    // Alignment safety check (TENSOR_ALIGNMENT = 32).
    if ((uintptr_t)addr % 32 != 0) {
        fprintf(stderr, "[mmap] tensor '%s' data addr %p not 32-byte aligned -- fallback to copy\n",
                dst->name, (void*)addr);
        return false;
    }

    // Bounds check: tensor must fit within the buffer.
    void* buf_base = ggml_backend_buffer_get_base(buf);
    size_t buf_size = ggml_backend_buffer_get_size(buf);
    if ((uintptr_t)addr < (uintptr_t)buf_base ||
        (uintptr_t)addr + ti->byte_size > (uintptr_t)buf_base + buf_size) {
        fprintf(stderr, "[mmap] tensor '%s' offset out of buffer bounds\n", dst->name);
        return false;
    }

    return ggml_backend_tensor_alloc(buf, dst, addr) == GGML_STATUS_SUCCESS;
}

bool GGUFLoader::read(bool & dst) const {
    int8_t tmp = -1;
    if (!read(tmp)) {
        return false;
    }
    dst = tmp != 0;
    return true;
}

bool GGUFLoader::read(enum ggml_type & dst) const {
    int32_t tmp = -1;
    if (!read(tmp)) {
        return false;
    }
    dst = ggml_type(tmp);
    return true;
}

bool GGUFLoader::read(enum gguf_type & dst) const {
    int32_t tmp = -1;
    if (!read(tmp)) {
        return false;
    }
    dst = gguf_type(tmp);
    return true;
}

bool GGUFLoader::read(std::string & dst) const {
    uint64_t size = 0;
    if (!read(size)) {
        return false;
    }
    if (size > GGUF_MAX_STRING_LENGTH) {
        printf("%s: string length %lu exceeds maximum %lu\n", __func__, size, (uint64_t)GGUF_MAX_STRING_LENGTH);
        return false;
    }
    if (size > nbytes_remain_) {
        printf("%s: string length %lu exceeds remaining file size %lu bytes\n", __func__, size, nbytes_remain_);
        return false;
    }
    dst.resize(static_cast<size_t>(size));
    std::memcpy(dst.data(), data_ + pos_, size);
    pos_ += size;
    nbytes_remain_ -= size;
    return true;
}

bool GGUFLoader::read(void * dst, const size_t size) const {
    if (size > nbytes_remain_) {
        return false;
    }
    std::memcpy(dst, data_ + pos_, size);
    pos_ += size;
    nbytes_remain_ -= size;
    return true;
}

void GGUFLoader::print_prase_info() const {
    printf("version: %u\n", version_);
    printf("n_tensors: %ld\n", n_tensors_);
    printf("n_kv: %ld\n", n_kv_);

    for (size_t i = 0; i < kv_.size(); ++i) {
        const auto& kv = kv_[i];
        printf("kv[%zu]: key='%s', type=%s, is_array=%d\n", i, kv.key.c_str(), gguf_type_name(kv.type), kv.is_array);
    }

    for (auto& info : tensors_head_) {
        const auto& t = info.second;
        printf("tensor: name='%s', n_dims=%u, dims=[", t.name.c_str(), t.n_dims);
        for (uint32_t d = 0; d < t.n_dims; ++d) {
            printf("%ld", t.dims[d]);
            if (d < t.n_dims - 1) {
                printf(", ");
            }
        }
        printf("], type=%d, type_size=%zu, blck_size=%ld, byte_size=%zu, data_offset=%lu, file_offset=%lu, layer_idx=%d, weight_type=%d\n",
            t.type, t.type_size, t.blck_size, t.byte_size, t.offset, data_offset_ + t.offset, t.layer_idx, t.weight_type);
    }
    for (size_t i = 0; i < tensors_layer_.size(); ++i) {
        const auto& t = tensors_layer_[i];
        printf("tensor[%zu]: name='%s', n_dims=%u, dims=[", i, t.name.c_str(), t.n_dims);
        for (uint32_t d = 0; d < t.n_dims; ++d) {
            printf("%ld", t.dims[d]);
            if (d < t.n_dims - 1) {
                printf(", ");
            }
        }
        printf("], type=%d, type_size=%zu, blck_size=%ld, byte_size=%zu, data_offset=%lu, file_offset=%lu, layer_idx=%d, weight_type=%d\n",
            t.type, t.type_size, t.blck_size, t.byte_size, t.offset, data_offset_ + t.offset, t.layer_idx, t.weight_type);
    }
}

int GGUFLoader::find_key(const char* key) {
    auto iter = kv_index_map_.find(key);
    if (iter != kv_index_map_.end()) {
        return iter->second;
    }
    return -1;
}
