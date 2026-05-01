#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <unordered_set>
#include <vector>
#include <string>

class Sampler {
public:
    virtual ~Sampler() = default;

    // Sample a token ID from logits.
    // last_tokens: recent token history for repetition penalty.
    // token_strs:  vocab[i] = decoded string of token i (for grammar constraints).
    virtual int sample(std::vector<float>& logits, const std::vector<int32_t>& last_tokens) = 0;

    // Attach a vocab pruning set (optional).
    void set_pruned_vocab(const std::unordered_set<int32_t>* pruned_vocab);

    void set_eos_token_id(int32_t id);

protected:
    void apply_vocab_pruning(std::vector<float>& logits);
    
protected:
    const std::unordered_set<int32_t>* pruned_vocab_ = nullptr;
    int32_t eos_token_id_ = -1;
};

class GreedySampler : public Sampler {
public:
    GreedySampler(float repetition_penalty = 1.2f, int   repetition_lookback = 32);

    int sample(std::vector<float>& logits, const std::vector<int32_t>& last_tokens) override;

private:
    void apply_repetition_penalty(std::vector<float>& logits, const std::vector<int32_t>& last_tokens);

private:
    float repetition_penalty_ = 0.0f;
    int repetition_lookback_ = 0;
};

class TemperatureSampler : public Sampler {
public:
    TemperatureSampler(float temperature = 0.7f, 
        float repetition_penalty = 1.1f, int repetition_lookback = 64, int top_k = 40, float top_p = 0.95f);

    int sample(std::vector<float>& logits, const std::vector<int32_t>& last_tokens) override;

private:
    void apply_repetition_penalty(std::vector<float>& logits, const std::vector<int32_t>& last_tokens);

private:
    float temperature_ = 0.0f;
    float repetition_penalty_ = 0.0f;
    int repetition_lookback_ = 0;
    int top_k_ = 0;
    float top_p_ = 0.0f;
    std::mt19937 gen_;
};
