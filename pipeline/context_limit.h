#pragma once

#include <algorithm>
#include <cstdint>

inline bool qwen35moe_can_decode_at_position(uint32_t pos, uint32_t context_len) {
    return pos < context_len;
}

inline bool qwen35moe_can_decode_at_position(int pos, uint32_t context_len) {
    if (pos < 0) {
        return false;
    }
    return qwen35moe_can_decode_at_position(static_cast<uint32_t>(pos), context_len);
}

inline uint32_t qwen35moe_decode_tokens_remaining(uint32_t pos, uint32_t context_len) {
    if (!qwen35moe_can_decode_at_position(pos, context_len)) {
        return 0;
    }
    return context_len - pos;
}

inline uint32_t qwen35moe_decode_tokens_remaining(int pos, uint32_t context_len) {
    if (!qwen35moe_can_decode_at_position(pos, context_len)) {
        return 0;
    }
    return qwen35moe_decode_tokens_remaining(static_cast<uint32_t>(pos), context_len);
}

inline uint32_t qwen35moe_clamp_generation_tokens(uint32_t pos, uint32_t context_len, uint32_t requested_tokens) {
    return std::min(requested_tokens, qwen35moe_decode_tokens_remaining(pos, context_len));
}

inline uint32_t qwen35moe_clamp_generation_tokens(int pos, uint32_t context_len, uint32_t requested_tokens) {
    return std::min(requested_tokens, qwen35moe_decode_tokens_remaining(pos, context_len));
}
