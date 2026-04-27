#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cerrno>
#include <cstring>
#include <algorithm>
#include "gguf_mmap.h"

GGUFMmapTensorLoader::GGUFMmapTensorLoader() {
}

GGUFMmapTensorLoader::~GGUFMmapTensorLoader() {
    if (mmap_base_ && mmap_base_ != MAP_FAILED) {
        munmap(mmap_base_, mmap_size_);
    }
    if (fd_ >= 0) {
        close(fd_);
    }
}

bool GGUFMmapTensorLoader::load(const std::string& path) {
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

    return true;
}

bool GGUFMmapTensorLoader::get_tensor_data(const size_t offset, const size_t size, std::vector<uint8_t>& tensor_data) {
    tensor_data.resize(size);
    const auto* data = static_cast<const uint8_t*>(mmap_base_);
    memcpy(tensor_data.data(), data + offset, size);

    return true;
}

