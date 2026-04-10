// pipeline/generator.hpp
// Text generation pipeline for Qwen3.5 MoE.
//
// Orchestrates: BPETokenizer + Recognizer + Qwen35moeInference
//
#ifndef FUNASR_PIPELINE_GENERATOR_HPP
#define FUNASR_PIPELINE_GENERATOR_HPP

#include "pipeline/recognizer.hpp"
#include "pipeline/tokenizer.hpp"
#include "pipeline/inference.hpp"
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstdio>

// Sampling / generation parameters (declared outside Generator to avoid a
// GCC restriction on nested-struct default-member-init in default args).
struct GeneratorParams {
    int   n_predict   = 128;    // max new tokens to generate
    float temperature = 1.0f;   // 0 → greedy argmax
    int   top_k       = 40;     // 0 → disabled
    float top_p       = 0.95f;  // 1.0 → disabled
    int   n_threads   = 4;
    int   max_ctx     = 2048;
    unsigned int seed = 42;     // RNG seed; use a time-based value for non-deterministic output
};

class Generator {
public:
    // Keep the nested alias for backward-compatible usage
    using Params = GeneratorParams;

    // ============================================================
    // init: load model and set up tokenizer + inference engine
    // ============================================================
    bool init(const std::string& model_path, const Params& p = Params()) {
        params_ = p;
        rng_.seed(p.seed);

        if (!recognizer_.init(model_path)) {
            printf("[Generator] ERROR: tokenizer failed to init\n");
            return false;
        }

        tokenizer_.load(recognizer_.config().tokenizer);
        if (!tokenizer_.is_loaded()) {
            printf("[Generator] ERROR: tokenizer failed to load\n");
            return false;
        }

        if (!inference_.init(*recognizer_.model(), p.max_ctx, p.n_threads)) {
            printf("[Generator] ERROR: inference init failed\n");
            return false;
        }
        return true;
    }

    // ============================================================
    // generate: tokenize prompt, run forward loop, return decoded text
    // ============================================================
    std::string generate(const std::string& prompt) {
        std::vector<int> ids = tokenizer_.encode(prompt);
        if (ids.empty()) {
            printf("[Generator] WARNING: prompt produced no tokens\n");
            return "";
        }
        printf("[Generator] Prompt: \"%s\"\n", prompt.c_str());
        printf("[Generator] Prompt tokens: %d\n", (int)ids.size());
        printf("[Generator] Generating (max %d tokens)...\n", params_.n_predict);

        // Prefill: run forward pass for each prompt token to populate KV cache
        std::vector<float> logits;
        for (int i = 0; i < (int)ids.size(); i++) {
            logits = inference_.forward(ids[i], i);
        }

        // Decode loop
        std::vector<int> generated;
        int eos = tokenizer_.eos_token_id();
        int pos = (int)ids.size();

        for (int step = 0; step < params_.n_predict; step++) {
            int next_id = sample(logits);
            if (next_id == eos) {
                printf("[EOS]\n");
                break;
            }
            // printf("\n[DBG] next_id=%d\n", next_id);
            // std::string s = tokenizer_.decode({ next_id });   // 你项目里怎么 decode 就怎么调用
            // printf("[DBG] token_str_len=%zu\n", s.size());
            // for (size_t i = 0; i < std::min<size_t>(s.size(), 16); ++i) {
            //     printf("%02X ", (unsigned char)s[i]);
            // }
            // printf("\n[DBG] token_str='%s'\n", s.c_str());

            generated.push_back(next_id);

            // Print decoded piece incrementally
            std::string piece = tokenizer_.decode({next_id});
            printf("[DBG_POS] step=%d pos=%d next_id=%d piece=%s\n", step, pos, next_id, piece.c_str());
            fflush(stdout);

            logits = inference_.forward(next_id, pos++);
        }
        printf("\n");

        return tokenizer_.decode(generated);
    }

    bool is_ready() const { return recognizer_.is_ready(); }

private:
    Params             params_;
    Recognizer         recognizer_;
    BPETokenizer       tokenizer_;
    Qwen35moeInference inference_;
    std::mt19937 rng_{42};

    // Sample one token from a logit vector using the configured strategy
    int sample(const std::vector<float>& logits) {
        if (logits.empty()) return 0;

        // Greedy
        if (params_.temperature <= 0.0f) {
            return (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
        }

        // Apply temperature → softmax probabilities
        int V = (int)logits.size();
        std::vector<float> probs(V);
        float max_l = *std::max_element(logits.begin(), logits.end());
        float sum   = 0.0f;
        for (int i = 0; i < V; i++) {
            probs[i] = std::exp((logits[i] - max_l) / params_.temperature);
            sum += probs[i];
        }
        for (auto& p : probs) p /= sum;

        // Top-k filtering
        if (params_.top_k > 0 && params_.top_k < V) {
            // Find the k-th largest probability threshold
            std::vector<float> sorted_probs(probs);
            std::partial_sort(sorted_probs.begin(),
                              sorted_probs.begin() + params_.top_k,
                              sorted_probs.end(),
                              std::greater<float>());
            float kth = sorted_probs[params_.top_k - 1];
            sum = 0.0f;
            for (auto& p : probs) {
                if (p < kth) p = 0.0f;
                sum += p;
            }
            if (sum > 0.0f) for (auto& p : probs) p /= sum;
        }

        // Top-p (nucleus) filtering
        if (params_.top_p < 1.0f) {
            // Sort indices by descending probability
            std::vector<int> idx(V);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(),
                      [&](int a, int b){ return probs[a] > probs[b]; });
            float cumsum = 0.0f;
            bool  cutoff = false;
            sum = 0.0f;
            for (int i : idx) {
                if (cutoff) { probs[i] = 0.0f; continue; }
                cumsum += probs[i];
                sum    += probs[i];
                if (cumsum >= params_.top_p) cutoff = true;
            }
            if (sum > 0.0f) for (auto& p : probs) p /= sum;
        }

        // Multinomial sample
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r   = dist(rng_);
        float cdf = 0.0f;
        for (int i = 0; i < V; i++) {
            cdf += probs[i];
            if (r < cdf) return i;
        }
        return V - 1;
    }
};

#endif // FUNASR_PIPELINE_GENERATOR_HPP
