#include "graph/kv_cache_simple.h"

#include <iostream>
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

    std::cout << "paged_kv_cache_test: OK" << std::endl;
    return 0;
}
