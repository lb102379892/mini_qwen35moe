#include "graph/kv_cache_simple.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {
constexpr size_t KV_COPY_SCRATCH_BUFFER_SIZE = 1024 * 1024; // Stores temporary view metadata for per-layer KV copy views.
}

simple_kv_cache::simple_kv_cache(
    uint32_t n_layers, uint32_t n_ctx_max, uint32_t n_batch_max,
    uint32_t n_embd_k, uint32_t n_embd_v, ggml_type type_k,
    ggml_type type_v, ggml_backend_t backend,
    const std::vector<ggml_backend_t>& layer_backends,
    PagedKVConfig paged_config
) : n_layers(n_layers), n_ctx_max(n_ctx_max), n_batch_max(n_batch_max),
    n_embd_k(n_embd_k), n_embd_v(n_embd_v), type_k(type_k),
    type_v(type_v), backend(backend), layer_backends_(layer_backends),
    scratch_ctx(nullptr, ggml_free), paged_config_(paged_config) {
    
    positions.resize(n_batch_max, 0);
    paged_enabled_ = paged_config_.enabled;
    paged_block_size_ = std::max<uint32_t>(1u, paged_config_.block_size);
    if (paged_enabled_ && paged_block_size_ > n_ctx_max) {
        throw std::invalid_argument("simple_kv_cache: paged block size cannot exceed n_ctx_max");
    }
    total_token_capacity_ = n_ctx_max * n_batch_max;
    total_blocks_ = (total_token_capacity_ + paged_block_size_ - 1) / paged_block_size_;

    if (paged_enabled_) {
        block_owner_.assign(total_blocks_, -1);
        slot_block_tables_.assign(n_batch_max, {});
        slot_allocated_blocks_.assign(n_batch_max, 0);
    }

    init_cache();

    scratch_buffer_.resize(KV_COPY_SCRATCH_BUFFER_SIZE);
    struct ggml_init_params scratch_params = {
        .mem_size   = scratch_buffer_.size(),
        .mem_buffer = scratch_buffer_.data(),
        .no_alloc   = true,
    };
    scratch_ctx.reset(ggml_init(scratch_params));
    GGML_ASSERT(scratch_ctx);

    if (paged_enabled_) {
        if (paged_config_.diagnostics && (paged_block_size_ & (paged_block_size_ - 1)) != 0) {
            std::fprintf(stderr,
                "[paged-kv] warning: non-power-of-two block size (%u) may reduce paging efficiency\n",
                paged_block_size_);
        }
        maybe_log_paged_stats("init", 0);
    }
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

uint32_t simple_kv_cache::paged_used_blocks() const {
    if (!paged_enabled_) {
        return 0;
    }
    uint32_t used = 0;
    for (int32_t owner : block_owner_) {
        if (owner >= 0) {
            ++used;
        }
    }
    return used;
}

uint32_t simple_kv_cache::paged_free_blocks() const {
    if (!paged_enabled_) {
        return 0;
    }
    return total_blocks_ - paged_used_blocks();
}

void simple_kv_cache::maybe_log_paged_stats(const char* event, uint32_t slot_idx) const {
    if (!paged_enabled_ || !paged_config_.diagnostics) {
        return;
    }
    const uint32_t used = paged_used_blocks();
    const uint32_t free = total_blocks_ - used;
    const uint32_t slot_blocks = slot_idx < slot_block_tables_.size() ? static_cast<uint32_t>(slot_block_tables_[slot_idx].size()) : 0;
    std::fprintf(stderr,
        "[paged-kv] %s slot=%u total_blocks=%u used=%u free=%u slot_blocks=%u\n",
        event,
        slot_idx,
        total_blocks_,
        used,
        free,
        slot_blocks);
}

uint32_t simple_kv_cache::allocate_next_block_for_slot(uint32_t slot_idx) {
    std::vector<uint32_t>& table = slot_block_tables_[slot_idx];
    if (table.empty()) {
        for (uint32_t i = 0; i < total_blocks_; ++i) {
            if (block_owner_[i] < 0) {
                block_owner_[i] = static_cast<int32_t>(slot_idx);
                table.push_back(i);
                slot_allocated_blocks_[slot_idx]++;
                zero_block(i);
                maybe_log_paged_stats("block-table-resize", slot_idx);
                return i;
            }
        }
    } else {
        const uint32_t next = table.back() + 1;
        if (next < total_blocks_ && block_owner_[next] < 0) {
            block_owner_[next] = static_cast<int32_t>(slot_idx);
            table.push_back(next);
            slot_allocated_blocks_[slot_idx]++;
            zero_block(next);
            maybe_log_paged_stats("block-table-resize", slot_idx);
            return next;
        }
    }

    const uint32_t used = paged_used_blocks();
    const uint32_t free = total_blocks_ - used;
    std::fprintf(stderr,
        "[paged-kv] allocation-failure slot=%u total_blocks=%u used=%u free=%u reason=contiguous-run-unavailable\n",
        slot_idx, total_blocks_, used, free);
    throw std::runtime_error(
        "simple_kv_cache::allocate_next_block_for_slot: paged KV block allocation failed (contiguous block run unavailable)");
}

void simple_kv_cache::ensure_slot_has_block(uint32_t slot_idx, uint32_t logical_block) {
    if (!paged_enabled_) {
        return;
    }
    if (slot_idx >= n_batch_max) {
        throw std::runtime_error("simple_kv_cache::ensure_slot_has_block: slot_idx out of range");
    }
    while (slot_block_tables_[slot_idx].size() <= logical_block) {
        allocate_next_block_for_slot(slot_idx);
    }
}

void simple_kv_cache::ensure_slot_for_logical_range(uint32_t slot_idx, uint32_t logical_start, uint32_t n_tokens) {
    if (!paged_enabled_ || n_tokens == 0) {
        return;
    }
    if (slot_idx >= n_batch_max) {
        throw std::runtime_error("simple_kv_cache::ensure_slot_for_logical_range: slot_idx out of range");
    }
    if (logical_start >= n_ctx_max || logical_start + n_tokens > n_ctx_max) {
        throw std::runtime_error("simple_kv_cache::ensure_slot_for_logical_range: logical position exceeds n_ctx_max");
    }
    const uint32_t last_pos = logical_start + n_tokens - 1;
    const uint32_t last_block = last_pos / paged_block_size_;
    ensure_slot_has_block(slot_idx, last_block);
}

uint32_t simple_kv_cache::logical_to_physical_internal(uint32_t slot_idx, uint32_t logical_pos, bool allow_pending_pos) const {
    if (!paged_enabled_) {
        return slot_idx * n_ctx_max + logical_pos;
    }
    if (slot_idx >= n_batch_max) {
        throw std::runtime_error("simple_kv_cache::logical_to_physical: slot_idx out of range");
    }
    const uint32_t current_pos = positions[slot_idx];
    const bool has_mapping = has_materialized_logical_pos(slot_idx, logical_pos);
    const bool ok = allow_pending_pos
        ? (logical_pos <= current_pos || has_mapping)
        : (logical_pos < current_pos);
    if (!ok) {
        throw std::runtime_error("simple_kv_cache::logical_to_physical: logical position not available in block table");
    }
    const uint32_t logical_block = logical_pos / paged_block_size_;
    const uint32_t offset = logical_pos % paged_block_size_;
    if (logical_block >= slot_block_tables_[slot_idx].size()) {
        throw std::runtime_error("simple_kv_cache::logical_to_physical: invalid block-table access");
    }
    const uint32_t physical_block = slot_block_tables_[slot_idx][logical_block];
    const uint32_t physical_pos = physical_block * paged_block_size_ + offset;
    if (physical_pos >= total_token_capacity_) {
        throw std::runtime_error("simple_kv_cache::logical_to_physical: mapped physical position out of range");
    }
    return physical_pos;
}

uint32_t simple_kv_cache::logical_to_physical(uint32_t slot_idx, uint32_t logical_pos) const {
    return logical_to_physical_internal(slot_idx, logical_pos, false);
}

bool simple_kv_cache::slot_prefix_is_contiguous(uint32_t slot_idx, uint32_t n_tokens) const {
    if (!paged_enabled_ || n_tokens == 0) {
        return true;
    }
    if (slot_idx >= n_batch_max) {
        return false;
    }
    const uint32_t n_blocks = (n_tokens + paged_block_size_ - 1) / paged_block_size_;
    if (n_blocks == 0 || slot_block_tables_[slot_idx].size() < n_blocks) {
        return false;
    }
    const uint32_t first = slot_block_tables_[slot_idx][0];
    for (uint32_t i = 1; i < n_blocks; ++i) {
        if (slot_block_tables_[slot_idx][i] != first + i) {
            return false;
        }
    }
    return true;
}

void simple_kv_cache::fill_gather_indices(const std::vector<uint32_t>& slots, uint32_t n_kv, std::vector<int32_t>& out) const {
    if (n_kv > n_ctx_max) {
        throw std::runtime_error("simple_kv_cache::fill_gather_indices: n_kv exceeds n_ctx_max");
    }
    out.assign(slots.size() * n_kv, 0);
    for (size_t b = 0; b < slots.size(); ++b) {
        const uint32_t slot = slots[b];
        if (slot >= n_batch_max) {
            throw std::runtime_error("simple_kv_cache::fill_gather_indices: slot index out of range");
        }
        for (uint32_t j = 0; j < n_kv; ++j) {
            const bool allow_pending = true;
            uint32_t mapped = 0;
            if (!paged_enabled_) {
                mapped = slot * n_ctx_max + j;
            // j == positions[slot] is intentionally allowed for decode graphs:
            // the pending token is written to this logical position earlier in the
            // same graph before gather reads execute.
            } else if (slot < n_batch_max && !slot_block_tables_[slot].empty() && j <= positions[slot]) {
                mapped = logical_to_physical_internal(slot, j, allow_pending);
            }
            out[b * n_kv + j] = static_cast<int32_t>(mapped);
        }
    }
}

void simple_kv_cache::zero_block(uint32_t block_idx) {
    if (!paged_enabled_) {
        return;
    }
    const size_t base_token = static_cast<size_t>(block_idx) * paged_block_size_;
    if (base_token >= total_token_capacity_) {
        return;
    }
    const size_t token_count = std::min<size_t>(paged_block_size_, total_token_capacity_ - base_token);
    const size_t k_row_bytes = static_cast<size_t>(n_embd_k) * ggml_type_size(type_k);
    const size_t v_row_bytes = static_cast<size_t>(n_embd_v) * ggml_type_size(type_v);
    const size_t k_block_bytes = k_row_bytes * token_count;
    const size_t v_block_bytes = v_row_bytes * token_count;
    const size_t k_offset = base_token * k_row_bytes;
    const size_t v_offset = base_token * v_row_bytes;
    std::vector<uint8_t> k_zero(k_block_bytes, 0);
    std::vector<uint8_t> v_zero(v_block_bytes, 0);
    for (uint32_t il = 0; il < n_layers; ++il) {
        ggml_backend_tensor_set(k_cache[il], k_zero.data(), k_offset, k_block_bytes);
        ggml_backend_tensor_set(v_cache[il], v_zero.data(), v_offset, v_block_bytes);
    }
}

void simple_kv_cache::release_slot_blocks(uint32_t slot_idx) {
    if (!paged_enabled_ || slot_idx >= n_batch_max) {
        return;
    }
    for (uint32_t block : slot_block_tables_[slot_idx]) {
        if (block < block_owner_.size()) {
            block_owner_[block] = -1;
            zero_block(block);
        }
    }
    slot_block_tables_[slot_idx].clear();
    slot_allocated_blocks_[slot_idx] = 0;
    maybe_log_paged_stats("slot-release", slot_idx);
}

ggml_tensor * simple_kv_cache::get_k(ggml_context * ctx_compute, int32_t il, uint32_t n_kv, uint32_t slot_idx) {
    if (paged_enabled_) {
        if (slot_idx >= n_batch_max) {
            throw std::runtime_error("simple_kv_cache::get_k: slot_idx out of range");
        }
        if (n_kv > positions[slot_idx] + 1) {
            throw std::runtime_error("simple_kv_cache::get_k: n_kv exceeds logical cache position");
        }
        if (!slot_prefix_is_contiguous(slot_idx, n_kv)) {
            throw std::runtime_error("simple_kv_cache::get_k: non-contiguous paged prefix is not supported in phase-1 path");
        }
        const uint32_t first_block = n_kv == 0 ? 0 : slot_block_tables_[slot_idx][0];
        const size_t token_offset = static_cast<size_t>(first_block) * paged_block_size_ * k_cache[il]->nb[1];
        ggml_tensor* flat = ggml_reshape_2d(ctx_compute, k_cache[il], n_embd_k, total_token_capacity_);
        return ggml_view_2d(ctx_compute, flat, n_embd_k, n_kv, flat->nb[1], token_offset);
    }
    return ggml_view_2d(ctx_compute, 
        k_cache[il],
        n_embd_k, 
        n_kv,
        k_cache[il]->nb[1],
        slot_idx * k_cache[il]->nb[2]
    );
}

ggml_tensor * simple_kv_cache::get_v(ggml_context * ctx_compute, int32_t il, uint32_t n_kv, uint32_t slot_idx) {
    if (paged_enabled_) {
        if (slot_idx >= n_batch_max) {
            throw std::runtime_error("simple_kv_cache::get_v: slot_idx out of range");
        }
        if (n_kv > positions[slot_idx] + 1) {
            throw std::runtime_error("simple_kv_cache::get_v: n_kv exceeds logical cache position");
        }
        if (!slot_prefix_is_contiguous(slot_idx, n_kv)) {
            throw std::runtime_error("simple_kv_cache::get_v: non-contiguous paged prefix is not supported in phase-1 path");
        }
        const uint32_t first_block = n_kv == 0 ? 0 : slot_block_tables_[slot_idx][0];
        const size_t token_offset = static_cast<size_t>(first_block) * paged_block_size_ * v_cache[il]->nb[1];
        ggml_tensor* flat = ggml_reshape_2d(ctx_compute, v_cache[il], n_embd_v, total_token_capacity_);
        return ggml_view_2d(ctx_compute, flat, n_embd_v, n_kv, flat->nb[1], token_offset);
    }
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
    if (paged_enabled_) {
        if (n_tokens != 1) {
            throw std::runtime_error("simple_kv_cache::cpy_k: paged mode phase-1 only supports single-token writes");
        }
        ensure_slot_for_logical_range(slot_idx, positions[slot_idx], n_tokens);
        const uint32_t physical_pos = logical_to_physical_internal(slot_idx, positions[slot_idx], true);
        ggml_tensor* flat = ggml_reshape_2d(ctx_compute, k_cache[il], n_embd_k, total_token_capacity_);
        ggml_tensor* k_dst = ggml_view_2d(ctx_compute, flat,
            n_embd_k, n_tokens, flat->nb[1], static_cast<size_t>(physical_pos) * flat->nb[1]);
        return ggml_cpy(ctx_compute, k_cur, k_dst);
    }
    
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
    if (paged_enabled_) {
        if (n_tokens != 1) {
            throw std::runtime_error("simple_kv_cache::cpy_v: paged mode phase-1 only supports single-token writes");
        }
        ensure_slot_for_logical_range(slot_idx, positions[slot_idx], n_tokens);
        const uint32_t physical_pos = logical_to_physical_internal(slot_idx, positions[slot_idx], true);
        ggml_tensor* flat = ggml_reshape_2d(ctx_compute, v_cache[il], n_embd_v, total_token_capacity_);
        ggml_tensor* v_dst = ggml_view_2d(ctx_compute, flat,
            n_embd_v, n_tokens, flat->nb[1], static_cast<size_t>(physical_pos) * flat->nb[1]);
        return ggml_cpy(ctx_compute, v_cur, v_dst);
    }

    // Create view at current position: [n_embd_v, n_tokens]
    ggml_tensor * v_dst = ggml_view_2d(ctx_compute, v_cache[il],
        n_embd_v, n_tokens,
        v_cache[il]->nb[1],
        slot_idx * v_cache[il]->nb[2] + positions[slot_idx] * v_cache[il]->nb[1]
    );

    return ggml_cpy(ctx_compute, v_cur, v_dst);
}

void simple_kv_cache::advance(uint32_t n_tokens, uint32_t slot_idx) {
    if (paged_enabled_ && n_tokens > 0) {
        ensure_slot_for_logical_range(slot_idx, positions[slot_idx], n_tokens);
    }
    positions[slot_idx] += n_tokens;
    GGML_ASSERT(positions[slot_idx] <= n_ctx_max);
}

void simple_kv_cache::clear_slot(uint32_t slot_idx) {
    if (paged_enabled_) {
        release_slot_blocks(slot_idx);
    }
    positions[slot_idx] = 0;
}

void simple_kv_cache::clear_all() {
    if (paged_enabled_) {
        for (uint32_t slot = 0; slot < n_batch_max; ++slot) {
            release_slot_blocks(slot);
        }
    }
    std::fill(positions.begin(), positions.end(), 0);
}

void simple_kv_cache::set_pos(uint32_t p, uint32_t slot_idx) {
    GGML_ASSERT(p <= n_ctx_max);
    if (paged_enabled_) {
        if (p > positions[slot_idx]) {
            ensure_slot_for_logical_range(slot_idx, positions[slot_idx], p - positions[slot_idx]);
        } else if (p == 0) {
            release_slot_blocks(slot_idx);
        }
    }
    positions[slot_idx] = p;
}

bool simple_kv_cache::has_materialized_logical_pos(uint32_t slot_idx, uint32_t logical_pos) const {
    if (!paged_enabled_) {
        return slot_idx < n_batch_max && logical_pos < n_ctx_max;
    }
    if (slot_idx >= n_batch_max || logical_pos >= n_ctx_max) {
        return false;
    }
    const uint32_t logical_block = logical_pos / paged_block_size_;
    if (logical_block >= slot_block_tables_[slot_idx].size()) {
        return false;
    }
    const uint32_t physical_block = slot_block_tables_[slot_idx][logical_block];
    if (physical_block >= total_blocks_ || block_owner_[physical_block] != static_cast<int32_t>(slot_idx)) {
        return false;
    }
    const uint32_t offset = logical_pos % paged_block_size_;
    const uint32_t physical_pos = physical_block * paged_block_size_ + offset;
    return physical_pos < total_token_capacity_;
}

bool simple_kv_cache::ensure_materialized_logical_pos(uint32_t slot_idx, uint32_t logical_pos) {
    if (!paged_enabled_) {
        return slot_idx < n_batch_max && logical_pos < n_ctx_max;
    }
    if (slot_idx >= n_batch_max || logical_pos >= n_ctx_max) {
        return false;
    }
    try {
        ensure_slot_for_logical_range(slot_idx, logical_pos, 1);
    } catch (const std::runtime_error&) {
        return false;
    }
    return has_materialized_logical_pos(slot_idx, logical_pos);
}

bool simple_kv_cache::copy_pos(uint32_t src_pos, uint32_t dst_pos, uint32_t slot_idx) {
    GGML_ASSERT(slot_idx < n_batch_max);
    GGML_ASSERT(src_pos < n_ctx_max);
    GGML_ASSERT(dst_pos < n_ctx_max);
    if (src_pos == dst_pos) {
        return true;
    }

    GGML_ASSERT(scratch_ctx);
    ggml_reset(scratch_ctx.get());
    ggml_context* scratch = scratch_ctx.get();

    uint32_t src_physical = slot_idx * n_ctx_max + src_pos;
    uint32_t dst_physical = slot_idx * n_ctx_max + dst_pos;
    if (paged_enabled_) {
        if (!has_materialized_logical_pos(slot_idx, src_pos)) {
            std::fprintf(stderr,
                "[paged-kv] copy_pos skipped: source logical position not materialized slot=%u src=%u dst=%u\n",
                slot_idx, src_pos, dst_pos);
            return false;
        }
        if (!ensure_materialized_logical_pos(slot_idx, dst_pos)) {
            std::fprintf(stderr,
                "[paged-kv] copy_pos skipped: destination logical position not materialized slot=%u src=%u dst=%u\n",
                slot_idx, src_pos, dst_pos);
            return false;
        }
        try {
            src_physical = logical_to_physical_internal(slot_idx, src_pos, true);
            dst_physical = logical_to_physical_internal(slot_idx, dst_pos, true);
        } catch (const std::runtime_error&) {
            std::fprintf(stderr,
                "[paged-kv] copy_pos skipped: logical-to-physical mapping unavailable slot=%u src=%u dst=%u\n",
                slot_idx, src_pos, dst_pos);
            return false;
        }
    }

    for (uint32_t il = 0; il < n_layers; ++il) {
        ggml_tensor* k_flat = paged_enabled_ ? ggml_reshape_2d(scratch, k_cache[il], n_embd_k, total_token_capacity_) : k_cache[il];
        ggml_tensor* v_flat = paged_enabled_ ? ggml_reshape_2d(scratch, v_cache[il], n_embd_v, total_token_capacity_) : v_cache[il];
        ggml_tensor* k_src = ggml_view_2d(scratch, k_flat, n_embd_k, 1,
            k_flat->nb[1], static_cast<size_t>(src_physical) * k_flat->nb[1]);
        ggml_tensor* k_dst = ggml_view_2d(scratch, k_flat, n_embd_k, 1,
            k_flat->nb[1], static_cast<size_t>(dst_physical) * k_flat->nb[1]);
        k_src->buffer = k_cache[il]->buffer;
        k_dst->buffer = k_cache[il]->buffer;
        ggml_backend_tensor_copy(k_src, k_dst);

        ggml_tensor* v_src = ggml_view_2d(scratch, v_flat, n_embd_v, 1,
            v_flat->nb[1], static_cast<size_t>(src_physical) * v_flat->nb[1]);
        ggml_tensor* v_dst = ggml_view_2d(scratch, v_flat, n_embd_v, 1,
            v_flat->nb[1], static_cast<size_t>(dst_physical) * v_flat->nb[1]);
        v_src->buffer = v_cache[il]->buffer;
        v_dst->buffer = v_cache[il]->buffer;
        ggml_backend_tensor_copy(v_src, v_dst);
    }
    return true;
}

void simple_kv_cache::compact(uint32_t slot_idx, const std::vector<uint32_t>& retained_positions) {
    GGML_ASSERT(slot_idx < n_batch_max);
    if (paged_enabled_) {
        throw std::runtime_error("simple_kv_cache::compact: paged KV compact is not supported in phase-1");
    }

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

void simple_kv_cache::truncate_to_position(int pos, uint32_t slot_idx) {
    if (slot_idx >= n_batch_max) {
        throw std::runtime_error("simple_kv_cache::truncate_to_position: slot_idx out of range");
    }
    if (pos < 0 || static_cast<uint32_t>(pos) > n_ctx_max) {
        throw std::runtime_error("simple_kv_cache::truncate_to_position: pos out of range");
    }
    if (paged_enabled_ && static_cast<uint32_t>(pos) < positions[slot_idx]) {
        const uint32_t keep_blocks = (static_cast<uint32_t>(pos) + paged_block_size_ - 1) / paged_block_size_;
        auto& table = slot_block_tables_[slot_idx];
        while (table.size() > keep_blocks) {
            const uint32_t block = table.back();
            table.pop_back();
            block_owner_[block] = -1;
            zero_block(block);
        }
    }
    positions[slot_idx] = static_cast<uint32_t>(pos);
}

void simple_kv_cache::clone_slot(uint32_t src_slot, uint32_t dst_slot, uint32_t n_tokens) {
    GGML_ASSERT(src_slot < n_batch_max);
    GGML_ASSERT(dst_slot < n_batch_max);
    GGML_ASSERT(n_tokens <= n_ctx_max);
    if (paged_enabled_) {
        throw std::runtime_error("simple_kv_cache::clone_slot: paged KV clone is not supported in phase-1");
    }

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
    int32_t flat_size = static_cast<int32_t>(total_token_capacity_);
    ggml_tensor* flat_cache = ggml_reshape_2d(ctx_compute, k_cache[il], n_embd_k, flat_size);

    // 2. Gather
    // src[ne0, ne1], indices[n] -> dst[ne0, n]
    ggml_tensor* gathered_flat = ggml_get_rows(ctx_compute, flat_cache, indices);

    // 3. Reshape to [n_embd, n_kv, n_active]
    ggml_tensor* dst = ggml_reshape_3d(ctx_compute, gathered_flat, n_embd_k, n_kv, n_active);
    
    return dst;
}

ggml_tensor* simple_kv_cache::gather_v(ggml_context* ctx_compute, ggml_cgraph* gf, int32_t il, ggml_tensor* indices, uint32_t n_active, uint32_t n_kv) {
    int32_t flat_size = static_cast<int32_t>(total_token_capacity_);
    ggml_tensor* flat_cache = ggml_reshape_2d(ctx_compute, v_cache[il], n_embd_v, flat_size);

    ggml_tensor* gathered_flat = ggml_get_rows(ctx_compute, flat_cache, indices);

    ggml_tensor* dst = ggml_reshape_3d(ctx_compute, gathered_flat, n_embd_v, n_kv, n_active);
    
    return dst;
}
