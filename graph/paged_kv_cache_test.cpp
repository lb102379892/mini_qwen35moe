#include "graph/kv_cache_simple.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

static void require(bool cond, const char * msg) {
    if (!cond) {
        throw std::runtime_error(msg);
    }
}

int main() {
    simple_kv_cache non_paged(
        1, 8, 2,
        4, 4,
        GGML_TYPE_F16,
        GGML_TYPE_F16,
        nullptr,
        {},
        PagedKVConfig{false, 2, false}
    );
    require(!non_paged.paged_enabled(), "non-paged mode should remain disabled by default");
    require(non_paged.logical_to_physical(1, 3) == 11, "non-paged mapping should preserve legacy layout");

    simple_kv_cache cache(
        1, 8, 2,
        4, 4,
        GGML_TYPE_F16,
        GGML_TYPE_F16,
        nullptr,
        {},
        PagedKVConfig{true, 2, false}
    );

    require(cache.paged_enabled(), "paged mode should be enabled");
    require(cache.paged_total_blocks() == 8, "unexpected total block count");

    cache.set_pos(1, 0);
    require(cache.paged_used_blocks() == 1, "slot0 should allocate first block");
    require(cache.logical_to_physical(0, 0) == 0, "slot0 logical0 should map to physical0");

    cache.advance(1, 0);
    require(cache.paged_used_blocks() == 1, "slot0 token1 should still use first block");
    cache.advance(1, 0);
    require(cache.paged_used_blocks() == 2, "slot0 token2 should allocate second block");
    require(cache.logical_to_physical(0, 2) == 2, "slot0 logical2 should map to physical2");
    require(!cache.has_materialized_logical_pos(0, 6), "logical position should start unmapped");
    require(!cache.copy_pos(6, 0, 0), "copy_pos should fail gracefully when source mapping is missing");
    require(cache.ensure_materialized_logical_pos(0, 6), "ensure_materialized_logical_pos should allocate source mapping");
    require(cache.copy_pos(6, 1, 0), "copy_pos should succeed once source and destination mappings exist");

    cache.clear_slot(0);
    require(cache.paged_used_blocks() == 0, "clear_slot should release slot0 blocks");

    cache.set_pos(1, 1);
    require(cache.logical_to_physical(1, 0) == 0, "released block should be reusable by slot1");

    cache.set_pos(3, 1);
    std::vector<int32_t> indices;
    cache.fill_gather_indices({1}, 4, indices);
    require(indices.size() == 4, "gather index size mismatch");
    require(indices[0] == 0 && indices[1] == 1 && indices[2] == 2 && indices[3] == 3,
        "gather index mapping mismatch");

    // set_pos() must allocate blocks through the destination logical index so
    // batched decode gather at positions[slot] never throws.
    simple_kv_cache boundary_cache(
        1, 32, 1,
        4, 4,
        GGML_TYPE_F16,
        GGML_TYPE_F16,
        nullptr,
        {},
        PagedKVConfig{true, 16, false}
    );
    boundary_cache.set_pos(16, 0);
    std::vector<int32_t> boundary_indices;
    boundary_cache.fill_gather_indices({0}, 20, boundary_indices);
    require(boundary_indices.size() == 20, "boundary gather index size mismatch");
    require(boundary_cache.has_materialized_logical_pos(0, 16),
        "set_pos across block boundary should materialize logical position 16");
    require(static_cast<uint32_t>(boundary_indices[16]) == boundary_cache.logical_to_physical(0, 16),
        "gather index at block boundary mismatch");

    bool threw = false;
    try {
        cache.fill_gather_indices({2}, 1, indices);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    require(threw, "out-of-range slot index should throw in fill_gather_indices");

    threw = false;
    try {
        (void) cache.logical_to_physical(1, 4);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    require(threw, "out-of-range logical position should throw");

    simple_kv_cache write_cache(
        1, 8, 1,
        4, 4,
        GGML_TYPE_F16,
        GGML_TYPE_F16,
        nullptr,
        {},
        PagedKVConfig{true, 2, false}
    );

    std::vector<uint8_t> ggml_buf(1024 * 1024);
    ggml_init_params params {
        .mem_size   = ggml_buf.size(),
        .mem_buffer = ggml_buf.data(),
        .no_alloc   = true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    require(ctx != nullptr, "failed to create ggml test context");

    ggml_tensor* k_cur = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, 4, 1, 3);
    ggml_tensor* v_cur = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, 4, 1, 3);
    require(write_cache.cpy_k(ctx.get(), k_cur, 0, 0) != nullptr, "multi-token paged K write should build");
    require(write_cache.cpy_v(ctx.get(), v_cur, 0, 0) != nullptr, "multi-token paged V write should build");
    require(write_cache.paged_used_blocks() == 2, "three-token write should allocate two paged blocks");
    require(write_cache.has_materialized_logical_pos(0, 2), "pending multi-token write should materialize final logical position");

    ggml_tensor* k_full = write_cache.get_k(ctx.get(), 0, 3, 0);
    ggml_tensor* v_full = write_cache.get_v(ctx.get(), 0, 3, 0);
    require(k_full->ne[0] == 4 && k_full->ne[1] == 3, "multi-token paged K prefix view shape mismatch");
    require(v_full->ne[0] == 4 && v_full->ne[1] == 3, "multi-token paged V prefix view shape mismatch");

    write_cache.advance(3, 0);
    require(write_cache.logical_to_physical(0, 2) == 2, "multi-token paged write should keep logical positions contiguous");

    std::cout << "paged_kv_cache_test: OK" << std::endl;
    return 0;
}
