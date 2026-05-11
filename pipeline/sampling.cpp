#include <algorithm>
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <numeric>
#include <vector>
#include <iostream>
#include "pipeline/sampling.h"

void Sampler::set_pruned_vocab(const std::unordered_set<int32_t>* pruned_vocab) {
    pruned_vocab_ = pruned_vocab;
}

void Sampler::set_eos_token_id(int32_t id) {
    eos_token_id_ = id;
}

void Sampler::apply_vocab_pruning(std::vector<float>& logits) {
    if (!pruned_vocab_) return;
    for (size_t i = 0; i < logits.size(); ++i) {
        if (pruned_vocab_->find(static_cast<int32_t>(i)) == pruned_vocab_->end())
            logits[i] = -std::numeric_limits<float>::infinity();
    }
}

// ---- GreedySampler ----
GreedySampler::GreedySampler(float repetition_penalty, int repetition_lookback)
: repetition_penalty_(repetition_penalty),repetition_lookback_(repetition_lookback) {
}

void GreedySampler::apply_repetition_penalty(std::vector<float>& logits, const std::vector<int32_t>& last_tokens) {
    if (repetition_penalty_ == 1.0f || last_tokens.empty()) return;
    const size_t lookback = std::min(static_cast<size_t>(repetition_lookback_), last_tokens.size());
    std::unordered_set<int32_t> recent(last_tokens.end() - lookback, last_tokens.end());
    for (int32_t id : recent) {
        if (id >= 0 && static_cast<size_t>(id) < logits.size()) {
            logits[id] = logits[id] < 0 ? logits[id] * repetition_penalty_ : logits[id] / repetition_penalty_;
        }
    }
}

int GreedySampler::sample(std::vector<float>& logits,const std::vector<int32_t>& last_tokens) {
    if (logits.empty()) 
        throw std::runtime_error("Cannot sample from empty logits");
    apply_vocab_pruning(logits);
    apply_repetition_penalty(logits, last_tokens);
    auto max_it = std::max_element(logits.begin(), logits.end());
    return static_cast<int>(std::distance(logits.begin(), max_it));
}

// ---- TemperatureSampler ----
TemperatureSampler::TemperatureSampler(float temperature, 
    float repetition_penalty, int repetition_lookback, int top_k, float top_p)
: temperature_(temperature), repetition_penalty_(repetition_penalty),
repetition_lookback_(repetition_lookback), top_k_(top_k), top_p_(top_p)
{
    std::random_device rd;
    gen_ = std::mt19937(rd());
}

void TemperatureSampler::apply_repetition_penalty(std::vector<float>& logits, const std::vector<int32_t>& last_tokens) {
    if (repetition_penalty_ == 1.0f || last_tokens.empty()) 
        return;
    const size_t lookback = std::min(static_cast<size_t>(repetition_lookback_), last_tokens.size());
    std::unordered_set<int32_t> recent(last_tokens.end() - lookback, last_tokens.end());
    for (int32_t id : recent) {
        if (id >= 0 && static_cast<size_t>(id) < logits.size()) {
            logits[id] = logits[id] < 0 ? logits[id] * repetition_penalty_ : logits[id] / repetition_penalty_;
        }
    }
}

int TemperatureSampler::sample(std::vector<float>& logits, const std::vector<int32_t>& last_tokens) {
    if (logits.empty()) 
        throw std::runtime_error("Cannot sample from empty logits");

    apply_vocab_pruning(logits);
    apply_repetition_penalty(logits, last_tokens);

    if (temperature_ <= 0.0f) {
        auto max_it = std::max_element(logits.begin(), logits.end());
        return static_cast<int>(std::distance(logits.begin(), max_it));
    }

    const size_t vocab_size = logits.size();
    for (size_t i = 0; i < vocab_size; ++i)
        logits[i] /= temperature_;

    std::vector<std::pair<int, float>> sorted;
    const size_t candidate_count = (top_k_ > 0 && top_k_ < static_cast<int>(vocab_size))
        ? static_cast<size_t>(top_k_)
        : vocab_size;
    sorted.reserve(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i)
        sorted.emplace_back(static_cast<int>(i), logits[i]);

    auto by_logit_desc = [](const auto& a, const auto& b) { return a.second > b.second; };
    if (candidate_count < vocab_size) {
        auto kth = sorted.begin() + static_cast<std::ptrdiff_t>(candidate_count);
        std::nth_element(sorted.begin(), kth, sorted.end(), by_logit_desc);
        sorted.resize(candidate_count);
    }
    std::sort(sorted.begin(), sorted.end(), by_logit_desc);

    if (top_p_ > 0.0f && top_p_ < 1.0f) {
        const float max_l = sorted[0].second;
        float sum_exp = 0.0f;
        for (const auto& p : sorted) sum_exp += expf(p.second - max_l);
        float cum = 0.0f;
        size_t cutoff = sorted.size();
        for (size_t i = 0; i < sorted.size(); ++i) {
            cum += expf(sorted[i].second - max_l) / sum_exp;
            if (cum >= top_p_) { 
                cutoff = i + 1; 
                break; 
            }
        }
        sorted.resize(cutoff);
    }

    const float max_l = sorted[0].second;
    float sum_exp = 0.0f;
    std::vector<float> probs;
    probs.reserve(sorted.size());
    for (const auto& p : sorted) {
        float e = expf(p.second - max_l);
        probs.push_back(e);
        sum_exp += e;
    }
    for (float& p : probs) p /= sum_exp;

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    return sorted[dist(gen_)].first;
}
