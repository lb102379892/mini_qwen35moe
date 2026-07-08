cmake_minimum_required(VERSION 3.14)
project(Qwen35moe-GGML LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GPU support (optional)
option(QWEN35MOE_CUDA "Enable CUDA GPU support" OFF)
option(QWEN35MOE_PYTHON "Build pybind11 extension for Strata Runtime" OFF)

if(QWEN35MOE_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    add_subdirectory(third_party/pybind11)
endif()

if(QWEN35MOE_CUDA)
    enable_language(CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE)
    add_compile_definitions(QWEN35MOE_USE_CUDA)
else()
    set(GGML_CUDA OFF CACHE BOOL "" FORCE)
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
# Shared object sources used by both the main CLI and the bench harness.
# Note: api/* is HTTP-only and intentionally left out of the bench target.
# ============================================================
set(QWEN35MOE_JINJA_SOURCES
    third_party/jinja/caps.cpp
    third_party/jinja/lexer.cpp
    third_party/jinja/parser.cpp
    third_party/jinja/runtime.cpp
    third_party/jinja/strfmt.cpp
    third_party/jinja/unicode.cpp
    third_party/jinja/value.cpp
)

add_library(qwen35_chat_template STATIC
    pipeline/chat_template.cpp
    ${QWEN35MOE_JINJA_SOURCES}
)
target_include_directories(qwen35_chat_template PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/third_party/jinja
)
# Required when linking this static archive into the pybind11 shared module (_native).
set_target_properties(qwen35_chat_template PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(QWEN35MOE_CORE_SOURCES
    graph/deltanet_state.cpp
    graph/kv_cache_simple.cpp
    graph/graph.cpp
    graph/mtp.cpp
    graph/turboquant.cpp
    model/dev_defaults.cpp
    model/common.cpp
    model/gguf_mmap.cpp
    model/metadata.cpp
    model/weights.cpp
    model/model.cpp
    model/tensor_override.cpp
    pipeline/tokenizer.cpp
    pipeline/sampling.cpp
    pipeline/chat.cpp
)

# ============================================================
# test_qwen35moe — main CLI executable (HTTP server)
# ============================================================
add_executable(test_qwen35moe
    ${QWEN35MOE_CORE_SOURCES}
    api/routes.cpp
    api/http_server.cpp
    main.cpp
)
target_link_libraries(test_qwen35moe PRIVATE ggml_core qwen35_chat_template pthread dl m)

# ============================================================
# qwen35moe_bench — C++ benchmark harness for tests/perf
# Same CParam → ChatEngine path as test_qwen35moe, but without the HTTP
# layer; reports per-request prefill/decode tok/s and wall throughput.
# ============================================================
add_executable(qwen35moe_bench
    ${QWEN35MOE_CORE_SOURCES}
    tests/perf/bench_main.cpp
)
target_link_libraries(qwen35moe_bench PRIVATE ggml_core qwen35_chat_template pthread dl m)

enable_testing()

add_executable(paged_kv_cache_test
    graph/kv_cache_simple.cpp
    graph/paged_kv_cache_test.cpp
)
target_link_libraries(paged_kv_cache_test PRIVATE ggml_core pthread dl m)
add_test(NAME paged_kv_cache_test COMMAND paged_kv_cache_test)

if(QWEN35MOE_PYTHON)
    pybind11_add_module(_native
        bindings/pybind_module.cpp
        ${QWEN35MOE_CORE_SOURCES}
    )
    target_link_libraries(_native PRIVATE ggml_core qwen35_chat_template pthread dl m)
    target_include_directories(_native PRIVATE ${CMAKE_SOURCE_DIR})
    set_target_properties(_native PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY
        "${CMAKE_SOURCE_DIR}/../../src/strata_runtime/engines/ggml_moe"
    )
endif()
