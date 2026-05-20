#include <cassert>
#include "pipeline/context_limit.h"

int main() {
    const uint32_t ctx = 4096;

    // position == context_len - 1: one token is still allowed
    assert(qwen35moe_can_decode_at_position(4095, ctx));
    assert(qwen35moe_decode_tokens_remaining(4095, ctx) == 1);
    assert(qwen35moe_clamp_generation_tokens(4095, ctx, 64) == 1);

    // position == context_len: must stop gracefully
    assert(!qwen35moe_can_decode_at_position(4096, ctx));
    assert(qwen35moe_decode_tokens_remaining(4096, ctx) == 0);
    assert(qwen35moe_clamp_generation_tokens(4096, ctx, 64) == 0);

    // requests exceeding remaining context must be clamped
    assert(qwen35moe_clamp_generation_tokens(4094, ctx, 10) == 2);

    return 0;
}
