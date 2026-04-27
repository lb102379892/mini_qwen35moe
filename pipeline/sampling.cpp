#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <chrono>
#include <cstdio>
#include "sampling.h"

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

void Sampler::init(const SamplingConfig& config) {
    config_ = config;
    recent_tokens_.clear();
    recent_tokens_.reserve(256);

    // Seed the RNG. 0 means derive from clock.
    if (config_.seed != 0) {
        rng_state_ = config_.seed;
    } else {
        auto now = std::chrono::high_resolution_clock::now();
        rng_state_ = static_cast<uint64_t>(now.time_since_epoch().count());
    }
    // Ensure nonzero state for xorshift64
    if (rng_state_ == 0) rng_state_ = 1;

    initialized_ = true;
    fprintf(stderr, "Sampler initialized: temp=%.2f top_k=%d top_p=%.2f rep_pen=%.2f seed=%lu\n",
        config_.temperature, config_.top_k, config_.top_p, config_.repetition_penalty, static_cast<unsigned long>(config_.seed));
}

// ---------------------------------------------------------------------------
// Token tracking
// ---------------------------------------------------------------------------

void Sampler::add_token(int token_id) {
    recent_tokens_.push_back(token_id);
    // Keep a bounded window to avoid unbounded growth.
    static constexpr size_t kMaxRecentTokens = 512;
    if (recent_tokens_.size() > kMaxRecentTokens) {
        recent_tokens_.erase(recent_tokens_.begin(), recent_tokens_.begin() + static_cast<long>(recent_tokens_.size() - kMaxRecentTokens));
    }
}

void Sampler::reset() {
    recent_tokens_.clear();
}

// ---------------------------------------------------------------------------
// Repetition penalty
// ---------------------------------------------------------------------------

void Sampler::apply_repetition_penalty(float* logits, int vocab_size) {
    if (config_.repetition_penalty == 1.0f) return;
    if (recent_tokens_.empty()) return;

    for (int tid : recent_tokens_) {
        if (tid < 0 || tid >= vocab_size) continue;
        float& logit = logits[tid];
        if (logit > 0.0f) {
            logit /= config_.repetition_penalty;
        } else {
            logit *= config_.repetition_penalty;
        }
    }
}

// ---------------------------------------------------------------------------
// Temperature scaling
// ---------------------------------------------------------------------------

void Sampler::apply_temperature(float* logits, int vocab_size) {
    // temperature == 0 is handled by the caller (greedy path).
    if (config_.temperature == 1.0f) return;
    float inv_temp = 1.0f / config_.temperature;
    for (int i = 0; i < vocab_size; ++i) {
        logits[i] *= inv_temp;
    }
}

// ---------------------------------------------------------------------------
// Top-k filtering
// ---------------------------------------------------------------------------

int Sampler::apply_top_k(float* logits, int vocab_size) {
    if (config_.top_k <= 0 || config_.top_k >= vocab_size) return vocab_size;

    int k = config_.top_k;

    // Build index array.
    std::vector<int> indices(static_cast<size_t>(vocab_size));
    std::iota(indices.begin(), indices.end(), 0);

    // Partial sort to get top-k largest logits.
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [logits](int a, int b) { return logits[a] > logits[b]; });

    // Keep exactly the top-k tokens (by sorted index), suppress the rest.
    std::vector<bool> keep(static_cast<size_t>(vocab_size), false);
    for (int i = 0; i < k; ++i) {
        keep[static_cast<size_t>(indices[static_cast<size_t>(i)])] = true;
    }
    for (int i = 0; i < vocab_size; ++i) {
        if (!keep[static_cast<size_t>(i)]) {
            logits[i] = -std::numeric_limits<float>::infinity();
        }
    }
    return k;
}

// ---------------------------------------------------------------------------
// Top-p (nucleus) filtering -- operates on probabilities after softmax
// ---------------------------------------------------------------------------

int Sampler::apply_top_p(float* logits, int vocab_size, int k_size) {
    if (config_.top_p <= 0.0f) return 0;  // will be handled as greedy
    if (config_.top_p >= 1.0f) return k_size;

    // Build (index, probability) pairs for non-negative-infinity entries.
    struct TokenProb {
        int index;
        float prob;
    };
    std::vector<TokenProb> candidates;
    candidates.reserve(static_cast<size_t>(k_size));

    for (int i = 0; i < vocab_size; ++i) {
        if (logits[i] > 0.0f) {
            // After softmax, only tokens with non-zero probability mass are candidates.
            candidates.push_back({i, logits[i]});
        }
    }

    if (candidates.empty()) return 0;

    // Sort descending by probability.
    std::sort(candidates.begin(), candidates.end(),
              [](const TokenProb& a, const TokenProb& b) {
                  return a.prob > b.prob;
              });

    // Accumulate until we reach top_p.
    float cumulative = 0.0f;
    size_t cutoff = candidates.size();
    for (size_t i = 0; i < candidates.size(); ++i) {
        cumulative += candidates[i].prob;
        if (cumulative >= config_.top_p) {
            cutoff = i + 1;
            break;
        }
    }

    // Zero out everything not in the nucleus.
    // First, mark which tokens are kept.
    std::vector<bool> keep(static_cast<size_t>(vocab_size), false);
    for (size_t i = 0; i < cutoff; ++i) {
        keep[static_cast<size_t>(candidates[i].index)] = true;
    }
    for (int i = 0; i < vocab_size; ++i) {
        if (!keep[static_cast<size_t>(i)]) {
            logits[i] = 0.0f;
        }
    }

    return static_cast<int>(cutoff);
}

// ---------------------------------------------------------------------------
// Numerically stable softmax
// ---------------------------------------------------------------------------

void Sampler::softmax(float* logits, int size) {
    // Find max for numerical stability.
    float max_val = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < size; ++i) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    // Handle degenerate case: all -inf.
    if (std::isinf(max_val) && max_val < 0.0f) {
        for (int i = 0; i < size; ++i) logits[i] = 0.0f;
        return;
    }

    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        logits[i] = std::exp(logits[i] - max_val);
        sum += logits[i];
    }

    if (sum > 0.0f) {
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < size; ++i) {
            logits[i] *= inv_sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Categorical sampling with xorshift64 RNG
// ---------------------------------------------------------------------------

int Sampler::categorical_sample(const float* probs, int size) {
    // xorshift64
    rng_state_ ^= rng_state_ << 13;
    rng_state_ ^= rng_state_ >> 7;
    rng_state_ ^= rng_state_ << 17;

    // Convert to uniform [0, 1).
    double u = static_cast<double>(rng_state_) / static_cast<double>(UINT64_MAX);

    float cumulative = 0.0f;
    for (int i = 0; i < size; ++i) {
        cumulative += probs[i];
        if (static_cast<double>(cumulative) >= u) {
            return i;
        }
    }

    // Fallback: return last token with nonzero probability.
    for (int i = size - 1; i >= 0; --i) {
        if (probs[i] > 0.0f) return i;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Main sampling pipeline
// ---------------------------------------------------------------------------

int Sampler::sample(float* logits, int vocab_size) {
    if (!initialized_ || vocab_size <= 0) return 0;

    // --- Step 1: Repetition penalty (on raw logits) ---
    apply_repetition_penalty(logits, vocab_size);

    // --- Greedy shortcuts ---
    // temperature == 0, top_k == 1, or top_p == 0 all mean argmax.
    if (config_.temperature == 0.0f || config_.top_k == 1 ||
        config_.top_p == 0.0f) {
        int best = 0;
        for (int i = 1; i < vocab_size; ++i) {
            if (logits[i] > logits[best]) best = i;
        }
        return best;
    }

    // --- Step 2: Temperature scaling ---
    apply_temperature(logits, vocab_size);

    // --- Step 3: Top-k filtering ---
    int effective_size = apply_top_k(logits, vocab_size);

    // --- Step 4: Softmax to get probabilities ---
    softmax(logits, vocab_size);

    // --- Step 5: Top-p (nucleus) filtering ---
    int nucleus_size = apply_top_p(logits, vocab_size, effective_size);

    // If everything was zeroed out (degenerate), return token 0.
    if (nucleus_size == 0) {
        // Check if any probability remains.
        bool any_nonzero = false;
        for (int i = 0; i < vocab_size; ++i) {
            if (logits[i] > 0.0f) { any_nonzero = true; break; }
        }
        if (!any_nonzero) return 0;
    }

    // --- Step 6: Re-normalize after top-p filtering ---
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) sum += logits[i];
    if (sum > 0.0f && sum != 1.0f) {
        float inv = 1.0f / sum;
        for (int i = 0; i < vocab_size; ++i) logits[i] *= inv;
    }

    // --- Step 7: Sample from the distribution ---
    return categorical_sample(logits, vocab_size);
}
