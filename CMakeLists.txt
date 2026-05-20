cmake_minimum_required(VERSION 3.14)
project(Qwen35moe-GGML LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GPU support (optional)
option(QWEN35MOE_CUDA "Enable CUDA GPU support" OFF)

if(QWEN35MOE_CUDA)
    enable_language(CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE)
    add_compile_definitions(QWEN35MOE_USE_CUDA)
endif()

# ============================================================
# third_party: ggml
# ============================================================
add_subdirectory(third_party/ggml)

# ============================================================
# ggml_core (header-only interface)
# ============================================================
add_library(ggml_core INTERFACE)
target_include_directories(ggml_core INTERFACE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party/ggml/include
    ${CMAKE_SOURCE_DIR}/third_party              # miniaudio.h
)
target_link_libraries(ggml_core INTERFACE ggml)

# ============================================================
# test_qwen35moe — main CLI executable
# ============================================================
add_executable(test_qwen35moe
    api/routes.cpp
    api/http_server.cpp
    graph/deltanet_state.cpp
    graph/kv_cache_simple.cpp
    graph/graph.cpp
    graph/turboquant.cpp
    model/common.cpp
    model/gguf_mmap.cpp
    model/metadata.cpp
    model/weights.cpp
    model/model.cpp
    pipeline/tokenizer.cpp
    pipeline/sampling.cpp
    pipeline/chat.cpp
    main.cpp
)
target_link_libraries(test_qwen35moe PRIVATE ggml_core pthread dl m)

enable_testing()

add_executable(paged_kv_cache_test
    graph/kv_cache_simple.cpp
    graph/paged_kv_cache_test.cpp
)
target_link_libraries(paged_kv_cache_test PRIVATE ggml_core pthread dl m)
add_test(NAME paged_kv_cache_test COMMAND paged_kv_cache_test)

add_executable(context_limit_test
    pipeline/context_limit_test.cpp
)
target_link_libraries(context_limit_test PRIVATE ggml_core pthread dl m)
add_test(NAME context_limit_test COMMAND context_limit_test)
