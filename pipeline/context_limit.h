#pragma once

#include <algorithm>
#include <cstdint>

inline bool qwen35moe_can_decode_at_position(int pos, uint32_t context_len) {
    if (pos < 0) {
        return false;
    }
    return static_cast<uint32_t>(pos) < context_len;
}

inline uint32_t qwen35moe_decode_tokens_remaining(int pos, uint32_t context_len) {
    if (!qwen35moe_can_decode_at_position(pos, context_len)) {
        return 0;
    }
    return context_len - static_cast<uint32_t>(pos);
}

inline uint32_t qwen35moe_clamp_generation_tokens(int pos, uint32_t context_len, uint32_t requested_tokens) {
    return std::min(requested_tokens, qwen35moe_decode_tokens_remaining(pos, context_len));
}
