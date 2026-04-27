#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <ggml.h>

class GGUFMmapTensorLoader {
public:
    GGUFMmapTensorLoader();
    ~GGUFMmapTensorLoader();

    bool load(const std::string& path);
    bool get_tensor_data(const size_t offset, const size_t size, std::vector<uint8_t>& tensor_data);
private:
    int fd_ = -1;
    void* mmap_base_ = nullptr;
    size_t mmap_size_ = 0;
    std::string last_error_ = "";
};
