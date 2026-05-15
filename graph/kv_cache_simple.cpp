#include "graph/kv_cache_simple.h"
#include <string>
#include <unordered_map>

namespace {
constexpr size_t KV_COPY_SCRATCH_BUFFER_SIZE = 1024 * 1024; // Stores temporary view metadata for per-layer KV copy views.
}

simple_kv_cache::simple_kv_cache(
    uint32_t n_layers, uint32_t n_ctx_max, uint32_t n_batch_max,
    uint32_t n_embd_k, uint32_t n_embd_v, ggml_type type_k,
    ggml_type type_v, ggml_backend_t backend,
    const std::vector<ggml_backend_t>& layer_backends
) : n_layers(n_layers), n_ctx_max(n_ctx_max), n_batch_max(n_batch_max),
    n_embd_k(n_embd_k), n_embd_v(n_embd_v), type_k(type_k),
    type_v(type_v), backend(backend), layer_backends_(layer_backends),
    scratch_ctx(nullptr, ggml_free) {
    
    positions.resize(n_batch_max, 0);
    init_cache();

    scratch_buffer_.resize(KV_COPY_SCRATCH_BUFFER_SIZE);
    struct ggml_init_params scratch_params = {
        .mem_size   = scratch_buffer_.size(),
        .mem_buffer = scratch_buffer_.data(),
        .no_alloc   = true,
    };
    scratch_ctx.reset(ggml_init(scratch_params));
    GGML_ASSERT(scratch_ctx);
}

void simple_kv_cache::init_cache() {
    // Create per-layer contexts and tensors (metadata only first)
    k_cache.resize(n_layers);
    v_cache.resize(n_layers);
    layer_ctxs_.clear();
    layer_bufs_.clear();
    layer_ctxs_.reserve(n_layers);
    layer_bufs_.reserve(n_layers);

    std::vector<size_t> backend_layer_counts;
    std::vector<std::string> backend_names;
    std::unordered_map<std::string, size_t> backend_index;

    const auto count_backend = [&](ggml_backend_t be) {
        const std::string name = be ? ggml_backend_name(be) : "CPU";
        const auto it = backend_index.find(name);
        if (it != backend_index.end()) {
            backend_layer_counts[it->second]++;
            return;
        }
        backend_names.push_back(name);
        backend_layer_counts.push_back(1);
        backend_index.emplace(name, backend_names.size() - 1);
    };

    for (uint32_t il = 0; il < n_layers; ++il) {
        const size_t ctx_size = (2 + 64) * ggml_tensor_overhead();
        struct ggml_init_params params = {
            .mem_size   = ctx_size,
            .mem_buffer = nullptr,
            .no_alloc   = true,
        };
        layer_ctxs_.emplace_back(ggml_init(params), ggml_free);
        GGML_ASSERT(layer_ctxs_.back());

        k_cache[il] = ggml_new_tensor_3d(layer_ctxs_.back().get(), type_k, n_embd_k, n_ctx_max, n_batch_max);
        v_cache[il] = ggml_new_tensor_3d(layer_ctxs_.back().get(), type_v, n_embd_v, n_ctx_max, n_batch_max);

        ggml_backend_t layer_backend = backend;
        if (il < layer_backends_.size() && layer_backends_[il] != nullptr) {
            layer_backend = layer_backends_[il];
        }
        const ggml_backend_buffer_type_t buffer_type = layer_backend
            ? ggml_backend_get_default_buffer_type(layer_backend)
            : ggml_backend_cpu_buffer_type();
        layer_bufs_.emplace_back(
            ggml_backend_alloc_ctx_tensors_from_buft(layer_ctxs_.back().get(), buffer_type),
            ggml_backend_buffer_free
        );
        GGML_ASSERT(layer_bufs_.back());

        count_backend(layer_backend);
    }

    double total_mb = 0.0;
    for (const auto& b : layer_bufs_) {
        total_mb += ggml_backend_buffer_get_size(b.get()) / (1024.0 * 1024.0);
    }
    std::printf("KV cache segmented allocation enabled (global reserve disabled for mixed decode)\n");
    std::printf("KV cache total size: %.2f MB\n", total_mb);
    for (size_t i = 0; i < backend_names.size(); ++i) {
        std::printf("KV cache layers on %s: %zu\n", backend_names[i].c_str(), backend_layer_counts[i]);
    }
}

size_t simple_kv_cache::memory_bytes() const {
    size_t total = 0;
    for (const auto& b : layer_bufs_) {
        total += ggml_backend_buffer_get_size(b.get());
    }
    return total;
}

ggml_tensor * simple_kv_cache::get_k(ggml_context * ctx_compute, int32_t il, uint32_t n_kv, uint32_t slot_idx) {
    return ggml_view_2d(ctx_compute, 
        k_cache[il],
        n_embd_k, 
        n_kv,
        k_cache[il]->nb[1],
        slot_idx * k_cache[il]->nb[2]
    );
}

ggml_tensor * simple_kv_cache::get_v(ggml_context * ctx_compute, int32_t il, uint32_t n_kv, uint32_t slot_idx) {
    return ggml_view_2d(ctx_compute, 
        v_cache[il],
        n_embd_v, 
        n_kv,
        v_cache[il]->nb[1],
        slot_idx * v_cache[il]->nb[2]
    );
}

ggml_tensor * simple_kv_cache::cpy_k(ggml_context * ctx_compute, ggml_tensor * k_cur, int32_t il, uint32_t slot_idx) {
    GGML_ASSERT(slot_idx < n_batch_max);
    GGML_ASSERT(il >= 0 && static_cast<uint32_t>(il) < n_layers);

    const uint32_t n_tokens = static_cast<uint32_t>(k_cur->ne[2]);

    GGML_ASSERT(positions[slot_idx] + n_tokens <= n_ctx_max);
    
    // Create view at current position: [n_embd_k, n_tokens]
    ggml_tensor * k_dst = ggml_view_2d(ctx_compute, k_cache[il],
        n_embd_k, n_tokens,
        k_cache[il]->nb[1],
        slot_idx * k_cache[il]->nb[2] + positions[slot_idx] * k_cache[il]->nb[1]
    );

    return ggml_cpy(ctx_compute, k_cur, k_dst);
}

ggml_tensor * simple_kv_cache::cpy_v(ggml_context * ctx_compute, ggml_tensor * v_cur, int32_t il, uint32_t slot_idx) {
    GGML_ASSERT(slot_idx < n_batch_max);
    GGML_ASSERT(il >= 0 && static_cast<uint32_t>(il) < n_layers);

    const uint32_t n_tokens = static_cast<uint32_t>(v_cur->ne[2]);

    GGML_ASSERT(positions[slot_idx] + n_tokens <= n_ctx_max);

    // Create view at current position: [n_embd_v, n_tokens]
    ggml_tensor * v_dst = ggml_view_2d(ctx_compute, v_cache[il],
        n_embd_v, n_tokens,
        v_cache[il]->nb[1],
        slot_idx * v_cache[il]->nb[2] + positions[slot_idx] * v_cache[il]->nb[1]
    );

    return ggml_cpy(ctx_compute, v_cur, v_dst);
}

void simple_kv_cache::advance(uint32_t n_tokens, uint32_t slot_idx) {
    positions[slot_idx] += n_tokens;
    GGML_ASSERT(positions[slot_idx] <= n_ctx_max);
}

void simple_kv_cache::clear_slot(uint32_t slot_idx) {
    positions[slot_idx] = 0;
}

void simple_kv_cache::clear_all() {
    std::fill(positions.begin(), positions.end(), 0);
}

void simple_kv_cache::set_pos(uint32_t p, uint32_t slot_idx) {
    GGML_ASSERT(p <= n_ctx_max);
    positions[slot_idx] = p;
}

void simple_kv_cache::copy_pos(uint32_t src_pos, uint32_t dst_pos, uint32_t slot_idx) {
    GGML_ASSERT(slot_idx < n_batch_max);
    GGML_ASSERT(src_pos < n_ctx_max);
    GGML_ASSERT(dst_pos < n_ctx_max);
    if (src_pos == dst_pos) {
        return;
    }

    GGML_ASSERT(scratch_ctx);
    ggml_reset(scratch_ctx.get());
    ggml_context* scratch = scratch_ctx.get();

    for (uint32_t il = 0; il < n_layers; ++il) {
        ggml_tensor* k_src = ggml_view_2d(scratch, k_cache[il], n_embd_k, 1,
            k_cache[il]->nb[1], slot_idx * k_cache[il]->nb[2] + src_pos * k_cache[il]->nb[1]);
        ggml_tensor* k_dst = ggml_view_2d(scratch, k_cache[il], n_embd_k, 1,
            k_cache[il]->nb[1], slot_idx * k_cache[il]->nb[2] + dst_pos * k_cache[il]->nb[1]);
        k_src->buffer = k_cache[il]->buffer;
        k_dst->buffer = k_cache[il]->buffer;
        ggml_backend_tensor_copy(k_src, k_dst);

        ggml_tensor* v_src = ggml_view_2d(scratch, v_cache[il], n_embd_v, 1,
            v_cache[il]->nb[1], slot_idx * v_cache[il]->nb[2] + src_pos * v_cache[il]->nb[1]);
        ggml_tensor* v_dst = ggml_view_2d(scratch, v_cache[il], n_embd_v, 1,
            v_cache[il]->nb[1], slot_idx * v_cache[il]->nb[2] + dst_pos * v_cache[il]->nb[1]);
        v_src->buffer = v_cache[il]->buffer;
        v_dst->buffer = v_cache[il]->buffer;
        ggml_backend_tensor_copy(v_src, v_dst);
    }
}

void simple_kv_cache::compact(uint32_t slot_idx, const std::vector<uint32_t>& retained_positions) {
    GGML_ASSERT(slot_idx < n_batch_max);

    const uint32_t n_retained = retained_positions.size();
    if (n_retained == 0) {
        positions[slot_idx] = 0;
        return;
    }

    // Read retained rows into a temp buffer, then write them back contiguously.
    // Each row = n_embd floats (we use element-size-agnostic byte copies via
    // ggml_backend_tensor_get/set which handle any type).
    const size_t k_row_bytes = n_embd_k * ggml_type_size(type_k);
    const size_t v_row_bytes = n_embd_v * ggml_type_size(type_v);
    const size_t max_row_bytes = std::max(k_row_bytes, v_row_bytes);

    std::vector<uint8_t> tmp(n_retained * max_row_bytes);

    for (uint32_t il = 0; il < n_layers; ++il) {
        const size_t slot_offset_k = slot_idx * k_cache[il]->nb[2];
        const size_t slot_offset_v = slot_idx * v_cache[il]->nb[2];

        // Compact K
        for (uint32_t i = 0; i < n_retained; ++i) {
            size_t src_off = slot_offset_k + retained_positions[i] * k_cache[il]->nb[1];
            ggml_backend_tensor_get(k_cache[il], tmp.data() + i * k_row_bytes, src_off, k_row_bytes);
        }
        for (uint32_t i = 0; i < n_retained; ++i) {
            size_t dst_off = slot_offset_k + i * k_cache[il]->nb[1];
            ggml_backend_tensor_set(k_cache[il], tmp.data() + i * k_row_bytes, dst_off, k_row_bytes);
        }

        // Compact V
        for (uint32_t i = 0; i < n_retained; ++i) {
            size_t src_off = slot_offset_v + retained_positions[i] * v_cache[il]->nb[1];
            ggml_backend_tensor_get(v_cache[il], tmp.data() + i * v_row_bytes, src_off, v_row_bytes);
        }
        for (uint32_t i = 0; i < n_retained; ++i) {
            size_t dst_off = slot_offset_v + i * v_cache[il]->nb[1];
            ggml_backend_tensor_set(v_cache[il], tmp.data() + i * v_row_bytes, dst_off, v_row_bytes);
        }
    }

    positions[slot_idx] = n_retained;
}

void simple_kv_cache::clone_slot(uint32_t src_slot, uint32_t dst_slot, uint32_t n_tokens) {
    GGML_ASSERT(src_slot < n_batch_max);
    GGML_ASSERT(dst_slot < n_batch_max);
    GGML_ASSERT(n_tokens <= n_ctx_max);

    GGML_ASSERT(scratch_ctx);
    ggml_reset(scratch_ctx.get());
    ggml_context* scratch = scratch_ctx.get();

    for (uint32_t il = 0; il < n_layers; ++il) {
        ggml_tensor * k_src = ggml_view_2d(scratch, k_cache[il], n_embd_k, n_tokens, k_cache[il]->nb[1], src_slot * k_cache[il]->nb[2]);
        ggml_tensor * k_dst = ggml_view_2d(scratch, k_cache[il], n_embd_k, n_tokens, k_cache[il]->nb[1], dst_slot * k_cache[il]->nb[2]);
        
        k_src->buffer = k_cache[il]->buffer;
        k_dst->buffer = k_cache[il]->buffer;

        ggml_tensor * v_src = ggml_view_2d(scratch, v_cache[il], n_embd_v, n_tokens, v_cache[il]->nb[1], src_slot * v_cache[il]->nb[2]);
        ggml_tensor * v_dst = ggml_view_2d(scratch, v_cache[il], n_embd_v, n_tokens, v_cache[il]->nb[1], dst_slot * v_cache[il]->nb[2]);

        v_src->buffer = v_cache[il]->buffer;
        v_dst->buffer = v_cache[il]->buffer;

        ggml_backend_tensor_copy(k_src, k_dst);
        ggml_backend_tensor_copy(v_src, v_dst);
    }

    positions[dst_slot] = n_tokens;
}

ggml_tensor* simple_kv_cache::gather_k(ggml_context* ctx_compute, ggml_cgraph* gf, int32_t il, ggml_tensor* indices, uint32_t n_active, uint32_t n_kv) {
    // 1. Reshape Cache to Flat [n_embd, n_ctx_max * n_batch_max]
    int32_t flat_size = n_ctx_max * n_batch_max;
    ggml_tensor* flat_cache = ggml_reshape_2d(ctx_compute, k_cache[il], n_embd_k, flat_size);

    // 2. Gather
    // src[ne0, ne1], indices[n] -> dst[ne0, n]
    ggml_tensor* gathered_flat = ggml_get_rows(ctx_compute, flat_cache, indices);

    // 3. Reshape to [n_embd, n_kv, n_active]
    ggml_tensor* dst = ggml_reshape_3d(ctx_compute, gathered_flat, n_embd_k, n_kv, n_active);
    
    return dst;
}

ggml_tensor* simple_kv_cache::gather_v(ggml_context* ctx_compute, ggml_cgraph* gf, int32_t il, ggml_tensor* indices, uint32_t n_active, uint32_t n_kv) {
    int32_t flat_size = n_ctx_max * n_batch_max;
    ggml_tensor* flat_cache = ggml_reshape_2d(ctx_compute, v_cache[il], n_embd_v, flat_size);

    ggml_tensor* gathered_flat = ggml_get_rows(ctx_compute, flat_cache, indices);

    ggml_tensor* dst = ggml_reshape_3d(ctx_compute, gathered_flat, n_embd_v, n_kv, n_active);
    
    return dst;
}
