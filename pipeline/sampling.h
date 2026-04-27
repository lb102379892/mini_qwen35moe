#pragma once

#include <cstdint>
#include <vector>

struct SamplingConfig {
    float temperature = 0.7f;
    int top_k = 50;
    float top_p = 0.9f;
    float repetition_penalty = 1.0f;
    uint64_t seed = 0;  // 0 = random
};

class Sampler {
public:
    Sampler() = default;
    void init(const SamplingConfig& config);
    void set_temperature(float temp) { config_.temperature = temp; }

    // Sample one token from logits [vocab_size]
    // Applies: repetition penalty -> temperature -> top-k -> top-p -> sample
    int sample(float* logits, int vocab_size);

    // Track recent tokens for repetition penalty
    void add_token(int token_id);
    void reset();

private:
    void apply_repetition_penalty(float* logits, int vocab_size);
    void apply_temperature(float* logits, int vocab_size);
    int apply_top_k(float* logits, int vocab_size);  // returns new effective size
    int apply_top_p(float* logits, int vocab_size, int k_size);
    int categorical_sample(const float* probs, int size);
    void softmax(float* logits, int size);

    SamplingConfig config_;
    std::vector<int> recent_tokens_;
    uint64_t rng_state_ = 0;
    bool initialized_ = false;
};
